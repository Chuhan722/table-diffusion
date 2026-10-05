#!/usr/bin/env python3
"""重放 NLTCS 前 232 轮并刻画 A/R 相对进度核的首次 logit 截断。"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Iterator

import numpy as np
import pandas as pd

from scripts import issue53_gap_weight_dual_ar_progress_clip_diagnostic_protocol as protocol
from scripts import run_issue53_gap_weight_dual_ar_progress_screen as source_collector
from table_diffevo import gap_l1_diffusion as gap
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import eval_condition
from table_diffevo.schema import load_schema


stage6d_runner = source_collector.stage6d_runner


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("正式短诊断报告不允许非有限 JSON 数值")
    return value


def _strict_json_text(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _assert_clean_descendant(root: Path) -> str:
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("短诊断要求包含未跟踪文件在内的干净工作树")
    current = _git_text(root, "rev-parse", "HEAD")
    if (
        _git_text(
            root,
            "merge-base",
            protocol.SOURCE_EXECUTION_COMMIT,
            current,
        )
        != protocol.SOURCE_EXECUTION_COMMIT
    ):
        raise RuntimeError("短诊断提交不是 v2 执行提交的后代")
    return current


def _source_gpu_preflight() -> dict[str, Any]:
    with source_collector._execution_runtime():
        return stage6d_runner._gpu_idle_audit(protocol.source.LOCAL_SHARD)


def _query_matches(frame: pd.DataFrame, query: dict[str, Any]) -> np.ndarray:
    matched = np.ones(len(frame), dtype=bool)
    for condition in query["conditions"]:
        matched &= np.asarray(eval_condition(frame, condition), dtype=bool)
    return matched


def _affected_queries(
    *,
    current: pd.DataFrame,
    donors: pd.DataFrame,
    mask: np.ndarray,
    row_index: int,
    attribute_index: int,
    attribute_names: tuple[str, ...],
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_reset = current.reset_index(drop=True)
    donor_reset = donors.reset_index(drop=True)
    row_values = {
        name: (
            donor_reset.iloc[row_index][name]
            if mask[row_index, index]
            else current_reset.iloc[row_index][name]
        )
        for index, name in enumerate(attribute_names)
    }
    attribute = attribute_names[attribute_index]
    row0 = dict(row_values)
    row1 = dict(row_values)
    row0[attribute] = current_reset.iloc[row_index][attribute]
    row1[attribute] = donor_reset.iloc[row_index][attribute]
    pair = pd.DataFrame([row0, row1], columns=list(attribute_names))
    records: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        if not any(
            condition.get("attribute") == attribute
            for condition in query["conditions"]
        ):
            continue
        truth = _query_matches(pair, query)
        if bool(truth[0]) == bool(truth[1]):
            continue
        records.append({
            "query_index": query_index,
            "query_id": query.get("id"),
            "query_order": len(query["conditions"]),
            "target_count": query.get("result"),
            "row_indicator_if_switch_0": bool(truth[0]),
            "row_indicator_if_switch_1": bool(truth[1]),
        })
    return records


class _ClipRecorder:
    def __init__(self) -> None:
        self.round_index = 0
        self.round_summaries: list[dict[str, Any]] = []
        self.clipped_steps: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._channel_values: list[tuple[float, float, float, float]] = []
        self._scan_vectors: dict[str, np.ndarray] | None = None

    def begin_round(self, context: dict[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("短诊断扫描上下文发生嵌套")
        self.round_index += 1
        context["round"] = self.round_index
        context["capture"] = self.round_index == protocol.DIAGNOSTIC_ROUNDS
        self._active = context
        self._channel_values = []
        self._scan_vectors = None

    def end_round(self) -> None:
        self._active = None
        self._channel_values = []
        self._scan_vectors = None

    def record_channel_values(
        self, values: tuple[float, float | None, float, float | None]
    ) -> None:
        if self._active is None or not self._active["capture"]:
            return
        if values[1] is None or values[3] is None:
            raise RuntimeError("第 232 轮缺少 R 通道值")
        self._channel_values.append(tuple(float(item) for item in values))

    def record_scan_vectors(self, keyword: dict[str, Any]) -> None:
        if self._active is None or not self._active["capture"]:
            return
        required = ("scores", "normalized_scores", "raw_logits", "probabilities")
        self._scan_vectors = {
            name: np.asarray(keyword[name], dtype=np.float64).copy()
            for name in required
        }

    def finish_round(self, diagnostics: dict[str, Any]) -> None:
        if self._active is None:
            raise RuntimeError("短诊断扫描缺少活动上下文")
        round_index = int(self._active["round"])
        k = int(diagnostics["active_switches_k"])
        microsteps = int(diagnostics["gibbs_microsteps"])
        summary = {
            "round": round_index,
            "active_switches_k": k,
            "gibbs_microsteps": microsteps,
            "expected_microsteps": 8 * k,
            "clip_hit_count": int(diagnostics["clip_hit_count"]),
            "nonfinite_condition_count": int(
                diagnostics["nonfinite_condition_count"]
            ),
            "exact_zero_or_one_probability_count": int(
                diagnostics["exact_zero_or_one_probability_count"]
            ),
            "reference_scale": float(diagnostics["reference_scale"]),
            "normalized_score_distribution": diagnostics[
                "normalized_score_distribution"
            ],
            "raw_logit_distribution": diagnostics["raw_logit_distribution"],
            "probability_distribution": diagnostics["probability_distribution"],
            "near_deterministic_count": int(
                diagnostics["near_deterministic_count"]
            ),
        }
        if microsteps != 8 * k:
            raise RuntimeError(f"第 {round_index} 轮不是严格 8*K")
        if summary["nonfinite_condition_count"] != 0:
            raise RuntimeError(f"第 {round_index} 轮出现非有限条件")
        if summary["exact_zero_or_one_probability_count"] != 0:
            raise RuntimeError(f"第 {round_index} 轮出现精确 0/1 概率")
        self.round_summaries.append(summary)
        if not self._active["capture"]:
            return
        if self._scan_vectors is None:
            raise RuntimeError("第 232 轮完整扫描向量没有被只读捕获")
        if len(self._channel_values) != microsteps:
            raise RuntimeError("第 232 轮通道值数量与微步数不一致")
        raw_logits = self._scan_vectors["raw_logits"]
        scores = self._scan_vectors["scores"]
        normalized = self._scan_vectors["normalized_scores"]
        probabilities = self._scan_vectors["probabilities"]
        if any(len(values) != microsteps for values in self._scan_vectors.values()):
            raise RuntimeError("第 232 轮扫描向量长度漂移")
        clip = float(self._active["logit_clip"])
        clipped_indices = np.flatnonzero(
            raw_logits != np.clip(raw_logits, -clip, clip)
        ).astype(np.int64)
        if len(clipped_indices) != summary["clip_hit_count"]:
            raise RuntimeError("第 232 轮逐向量截断数与汇总不一致")
        if len(clipped_indices) == 0:
            raise RuntimeError("第 232 轮没有复现已知 logit 截断")

        current = self._active["current"]
        donors = self._active["donors"]
        schema = self._active["schema"]
        queries = self._active["queries"]
        participate = np.asarray(self._active["participate"], dtype=bool)
        mask = np.asarray(self._active["initial_mask"], dtype=bool).copy()
        attribute_names = tuple(schema.attribute_names())
        current_values = current.reset_index(drop=True).loc[
            :, list(attribute_names)
        ].to_numpy()
        donor_values = donors.reset_index(drop=True).loc[
            :, list(attribute_names)
        ].to_numpy()
        active_coordinates = np.argwhere(
            participate[:, None] & (current_values != donor_values)
        ).astype(np.intp, copy=False)
        if len(active_coordinates) != k:
            raise RuntimeError("第 232 轮重建活跃坐标数漂移")
        replay_rng = np.random.default_rng()
        replay_rng.bit_generator.state = copy.deepcopy(self._active["rng_state"])
        clipped_set = set(map(int, clipped_indices))
        current_reset = current.reset_index(drop=True)
        donor_reset = donors.reset_index(drop=True)
        for step in range(microsteps):
            coordinate_index = int(replay_rng.integers(0, k))
            random_roll = float(replay_rng.random())
            row_index, attribute_index = map(
                int, active_coordinates[coordinate_index]
            )
            before = bool(mask[row_index, attribute_index])
            after = bool(random_roll < probabilities[step])
            if step in clipped_set:
                absolute0, relative0, absolute1, relative1 = (
                    self._channel_values[step]
                )
                e0 = max(absolute0, relative0)
                e1 = max(absolute1, relative1)
                score_from_channels = float(e0 - e1)
                if not math.isclose(
                    score_from_channels,
                    float(scores[step]),
                    rel_tol=8.0 * np.finfo(np.float64).eps,
                    abs_tol=0.0,
                ):
                    raise RuntimeError("截断微步的通道能量与 score 不一致")
                if not abs(float(normalized[step])) > 15.0:
                    raise RuntimeError("截断微步标准化分数没有严格超过 15")
                attribute = attribute_names[attribute_index]
                affected = _affected_queries(
                    current=current,
                    donors=donors,
                    mask=mask,
                    row_index=row_index,
                    attribute_index=attribute_index,
                    attribute_names=attribute_names,
                    queries=queries,
                )
                self.clipped_steps.append({
                    "step_zero_based": step,
                    "step_one_based": step + 1,
                    "row_index_zero_based": row_index,
                    "attribute_index_zero_based": attribute_index,
                    "attribute": attribute,
                    "current_cell_value": current_reset.iloc[row_index][attribute],
                    "donor_cell_value": donor_reset.iloc[row_index][attribute],
                    "switch_before": before,
                    "switch_after": after,
                    "random_roll": random_roll,
                    "score": float(scores[step]),
                    "normalized_score": float(normalized[step]),
                    "raw_logit": float(raw_logits[step]),
                    "effective_logit": float(np.clip(raw_logits[step], -clip, clip)),
                    "probability": float(probabilities[step]),
                    "absolute_channel_if_switch_0": absolute0,
                    "relative_channel_if_switch_0": relative0,
                    "dominant_channel_if_switch_0": gap._dominant_channel(
                        absolute0, relative0
                    ),
                    "absolute_channel_if_switch_1": absolute1,
                    "relative_channel_if_switch_1": relative1,
                    "dominant_channel_if_switch_1": gap._dominant_channel(
                        absolute1, relative1
                    ),
                    "energy_if_switch_0": e0,
                    "energy_if_switch_1": e1,
                    "affected_measured_queries": affected,
                })
            mask[row_index, attribute_index] = after


@contextlib.contextmanager
def _instrumentation(recorder: _ClipRecorder) -> Iterator[None]:
    original_scan = gap._evolve_step_gap_l1_global_cuda
    original_dominance = gap._record_channel_dominance
    original_build = gap._build_scan_diagnostics

    def scan_wrapper(*args: Any, **kwargs: Any):
        if len(args) < 6:
            raise RuntimeError("CUDA 缺口扫描调用签名漂移")
        context = {
            "current": args[0],
            "donors": args[1],
            "schema": args[2],
            "queries": args[3],
            "target": args[4],
            "current_counts": args[5],
            "participate": kwargs["participate"],
            "initial_mask": kwargs["initial_mask"],
            "rng_state": copy.deepcopy(kwargs["rng"].bit_generator.state),
            "logit_clip": kwargs["logit_clip"],
        }
        recorder.begin_round(context)
        try:
            result = original_scan(*args, **kwargs)
            recorder.finish_round(result[2])
            return result
        finally:
            recorder.end_round()

    def dominance_wrapper(counts: Any, values: Any) -> None:
        original_dominance(counts, values)
        recorder.record_channel_values(values)

    def build_wrapper(*args: Any, **kwargs: Any):
        recorder.record_scan_vectors(kwargs)
        return original_build(*args, **kwargs)

    gap._evolve_step_gap_l1_global_cuda = scan_wrapper
    gap._record_channel_dominance = dominance_wrapper
    gap._build_scan_diagnostics = build_wrapper
    try:
        yield
    finally:
        gap._build_scan_diagnostics = original_build
        gap._record_channel_dominance = original_dominance
        gap._evolve_step_gap_l1_global_cuda = original_scan


def build_plan() -> dict[str, Any]:
    root = _repo_root()
    observed = protocol.assert_frozen_protocol_identity(root)
    return {
        **protocol.build_plan(),
        "protocol_sha256": observed,
        "runner_sha256": protocol.file_sha256(Path(__file__)),
        "output": str(protocol.OUTPUT_DIR / protocol.REPORT),
        "ready_for_user_confirmed_short_replay": False,
    }


def preflight() -> dict[str, Any]:
    root = _repo_root()
    plan = build_plan()
    commit = _assert_clean_descendant(root)
    output = root / protocol.OUTPUT_DIR
    if output.exists():
        raise FileExistsError(f"短诊断输出已存在，不覆盖：{output}")
    gpu = _source_gpu_preflight()
    return {
        **plan,
        "mode": "read_only_preflight_no_generator_or_quality_access",
        "diagnostic_commit": commit,
        "worktree_clean_including_untracked": True,
        "output_absent": True,
        "gpu": gpu,
        "ready_for_user_confirmed_short_replay": True,
    }


def _run_diagnostic() -> dict[str, Any]:
    root = _repo_root()
    spec = protocol.source.DATASETS[protocol.DATASET]
    schema = load_schema(str(root / spec["schema"]))
    with source_collector._execution_runtime():
        queries, target, identity = stage6d_runner._query_target_identity_audit(
            root, protocol.DATASET
        )
    marginals = load_marginals(str(root / spec["marginals"]))

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("短诊断工作进程没有恰好一张可用显卡")
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    recorder = _ClipRecorder()
    monitor = stage6d_runner._GpuMonitor(
        protocol.source.EXPECTED_GPU["physical_index"]
    )
    monitor.start()
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    try:
        with _instrumentation(recorder):
            returned, diagnostics = stage6d_runner._call_generator_silently(
                "issue53-progress-clip-diagnostic-nltcs-232",
                target=target,
                queries=queries,
                schema=schema,
                marginals=marginals,
                **protocol.generator_params(),
            )
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
    finally:
        monitor.stop()
    if monitor.error is not None:
        raise RuntimeError(f"短诊断 GPU 监控失败：{monitor.error}")
    if "final_table" not in diagnostics:
        raise RuntimeError("短诊断生成器没有返回最后当前表")
    diagnostics.pop("final_table")
    del returned

    applied = len(diagnostics["accept_history"])
    if (
        applied != protocol.DIAGNOSTIC_ROUNDS
        or diagnostics["accept_history"] != [True] * applied
        or diagnostics["proposal_attempts_history"] != [1] * applied
        or diagnostics["accepted_attempt_history"] != [1] * applied
        or int(diagnostics["rounds_run"]) != applied
        or recorder.round_index != applied
        or len(recorder.round_summaries) != applied
    ):
        raise RuntimeError("短诊断没有形成 232 个唯一无条件状态转移")
    if diagnostics["initial_table_sha256"] != (
        protocol.EXPECTED_INITIAL_TABLE_SHA256
    ):
        raise RuntimeError("短诊断初始表身份与冻结值不一致")
    if diagnostics["primary_rng_post_initialization_state_sha256"] != (
        protocol.EXPECTED_PRIMARY_RNG_POST_INITIALIZATION_SHA256
    ):
        raise RuntimeError("短诊断初始化后主 RNG 身份与冻结值不一致")
    expected_reference = protocol.source.EXPECTED_INITIAL_CHANNEL_REFERENCES[
        protocol.DATASET
    ]
    observed_reference = diagnostics["gap_l1_channel_reference"]
    if (
        observed_reference.get("absolute_initial")
        != expected_reference["absolute_initial"]
        or observed_reference.get("relative_initial")
        != expected_reference["relative_initial"]
        or observed_reference.get("source")
        != "initial_current_before_round_1"
    ):
        raise RuntimeError("短诊断 A/R 初始参照身份漂移")
    early_clip_total = sum(
        row["clip_hit_count"]
        for row in recorder.round_summaries[: protocol.PREFAIL_ZERO_CLIP_ROUNDS]
    )
    final_round = recorder.round_summaries[-1]
    if early_clip_total != 0 or final_round["clip_hit_count"] <= 0:
        raise RuntimeError("短诊断没有复现第 232 轮首次截断边界")
    if len(recorder.clipped_steps) != final_round["clip_hit_count"]:
        raise RuntimeError("短诊断截断微步证据数量漂移")

    total_microsteps = sum(
        row["gibbs_microsteps"] for row in recorder.round_summaries
    )
    total_clip_hits = sum(
        row["clip_hit_count"] for row in recorder.round_summaries
    )
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "protocol": protocol.frozen_protocol_manifest(),
        "diagnostic_commit": _git_text(root, "rev-parse", "HEAD"),
        "runner_sha256": protocol.file_sha256(Path(__file__)),
        "source_identity": {
            "execution_commit": protocol.SOURCE_EXECUTION_COMMIT,
            "execution_protocol_sha256": (
                protocol.SOURCE_EXECUTION_PROTOCOL_SHA256
            ),
            "scientific_protocol_sha256": (
                protocol.SOURCE_SCIENTIFIC_PROTOCOL_SHA256
            ),
        },
        "task": {
            "dataset": protocol.DATASET,
            "arm": protocol.ARM,
            "seed": protocol.SEED,
            "requested_rounds": protocol.DIAGNOSTIC_ROUNDS,
            "applied_rounds": applied,
            "termination_reason": "candidate_budget",
        },
        "generation_identity": {
            **identity,
            "initial_table_sha256": diagnostics["initial_table_sha256"],
            "primary_rng_post_initialization_sha256": diagnostics[
                "primary_rng_post_initialization_state_sha256"
            ],
            "primary_rng_endpoint_sha256": diagnostics[
                "primary_rng_state_sha256"
            ],
            "gap_l1_channel_reference": observed_reference,
            "gap_l1_reference_scale": diagnostics["gap_l1_reference_scale"],
        },
        "clip_summary": {
            "first_clip_round": protocol.DIAGNOSTIC_ROUNDS,
            "rounds_1_through_231_clip_hit_count": early_clip_total,
            "round_232_clip_hit_count": final_round["clip_hit_count"],
            "round_232_microsteps": final_round["gibbs_microsteps"],
            "round_232_clip_rate": float(
                final_round["clip_hit_count"] / final_round["gibbs_microsteps"]
            ),
            "all_232_rounds_microsteps": total_microsteps,
            "all_232_rounds_clip_hit_count": total_clip_hits,
            "clipping_threshold_raw_logit_abs": 30.0,
            "clipping_threshold_normalized_score_abs": 15.0,
        },
        "round_232_diagnostics": final_round,
        "clipped_steps": recorder.clipped_steps,
        "execution": {
            "elapsed_sec": float(elapsed),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "gpu_samples": monitor.samples,
        },
        "diagnostic_valid": True,
        "raw_reference_data_accessed": False,
        "terminal_table_persisted": False,
        "checkpoint_answers_persisted": False,
        "quality_metrics_computed": False,
        "quality_interpretation_allowed": False,
        "unseen_queries_evaluated": False,
        "method_comparison_performed": False,
        "parameter_retuning_performed": False,
        "full_candidate_rerun_performed": False,
        "automatic_followup_authorized": False,
    }


def run(confirmed_protocol_sha256: str) -> Path:
    protocol.require_confirmation(confirmed_protocol_sha256)
    root = _repo_root()
    protocol.assert_frozen_protocol_identity(root)
    _assert_clean_descendant(root)
    destination = root / protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"短诊断输出已存在，不覆盖：{destination}")
    _source_gpu_preflight()
    report = _run_diagnostic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        report_path = temporary / protocol.REPORT
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_strict_json_text(report))
        os.replace(temporary, destination)
    except Exception as exc:
        raise RuntimeError(
            f"短诊断发布失败，临时目录已保留：{temporary}"
        ) from exc
    return destination / protocol.REPORT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("preflight")
    execute = commands.add_parser("run")
    execute.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    if args.command == "preflight":
        print(_strict_json_text(preflight()), end="")
        return
    path = run(args.confirm_protocol_sha)
    print(f"progress clip diagnostic -> {path}")
    print(f"diagnostic SHA-256 -> {protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
