#!/usr/bin/env python
"""Collect the frozen Issue #53 P=6 unseen artificial trajectories.

``plan`` is read-only.  ``collect`` requires the exact preregistered protocol
SHA-256 and a clean worktree, executes only the frozen 12-case primary matrix,
and writes raw per-case artifacts without applying the acceptance gates.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import issue53_p6_unseen_protocol as protocol
except ModuleNotFoundError as exc:  # direct ``python scripts/...py``
    if exc.name != "scripts":
        raise
    import issue53_p6_unseen_protocol as protocol

from table_diffevo.evolution import run_evolution
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema
from table_diffevo.stationarity import (
    StationarityTrace,
    ordered_query_identity_sha256,
    save_stationarity_trace,
    target_answer_identity_sha256,
)

COLLECTION_CONTRACT_VERSION = "issue53-p6-unseen-primary-collection-v1"
DEFAULT_OUTPUT_DIR = Path("outputs/issue53_p6_unseen_primary")
PROTOCOL_DOCUMENT = Path("docs/设计/Issue53_P6未见轨迹质量计算验收协议.md")
SOURCE_PATHS = (
    Path("scripts/issue53_p6_unseen_protocol.py"),
    Path("scripts/collect_issue53_p6_unseen.py"),
    Path("src/table_diffevo/evolution.py"),
    Path("src/table_diffevo/inner_early_stopping.py"),
    Path("src/table_diffevo/stationarity.py"),
    PROTOCOL_DOCUMENT,
)

_ONLINE_DIAGNOSTIC_KEYS = (
    "rounds_run",
    "termination_reason",
    "stopped_early",
    "fit_target_reached",
    "inner_complete",
    "output_table_identity",
    "output_squared_loss",
    "final_current_squared_loss",
    "final_current_normalized_l1",
    "normalized_l1_error",
    "normalized_l1_median",
    "normalized_l1_p90",
    "normalized_l1_max",
    "best_loss_diagnostic_only",
    "normalized_l1_at_best_squared_loss_diagnostic_only",
    "state_evaluation_count",
    "candidate_evaluation_count",
    "candidate_budget_exhausted",
    "initial_table_sha256",
    "primary_rng_post_initialization_state_sha256",
    "primary_rng_state_sha256",
    "current_state_metrics_history",
    "transition_clock_history",
    "accept_history",
    "proposal_attempts_history",
    "accepted_attempt_history",
    "inner_early_stopping",
    "elapsed_sec",
    "sec_per_round",
    "params",
)


@dataclass(frozen=True)
class ArtificialWorkload:
    name: str
    schema: Schema
    queries: list[dict[str, Any]]
    target: np.ndarray
    n_records: int
    family_identity_sha256: str
    query_identity_sha256: str
    target_identity_sha256: str


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    _strict_json_bytes(value)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def _write_frame_exclusive(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def materialize_family(family_name: str) -> ArtificialWorkload:
    """Build schema/queries/target without expanding the reference multiset."""

    families = {family["family"]: family for family in protocol.family_manifests()}
    if family_name not in families:
        raise ValueError(f"family 不在冻结集合中：{family_name!r}")
    family = families[family_name]
    protocol.recompute_family_arithmetic(family)
    schema = Schema(
        [
            AttributeBlock(
                name=attribute["name"],
                type=attribute["type"],
                description=attribute["name"],
                values=list(attribute["values"]),
            )
            for attribute in family["schema"]["attributes"]
        ]
    )
    queries = json.loads(_strict_json_bytes(family["ordered_queries"]))
    target = np.asarray(family["ordered_targets"], dtype=float)
    return ArtificialWorkload(
        name=family_name,
        schema=schema,
        queries=queries,
        target=target,
        n_records=int(family["n_records"]),
        family_identity_sha256=family["family_identity_sha256"],
        query_identity_sha256=ordered_query_identity_sha256(queries),
        target_identity_sha256=target_answer_identity_sha256(target),
    )


def _canonical_primary_case(case: dict[str, Any]) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in protocol.primary_case_matrix()}
    case_id = case.get("case_id") if isinstance(case, dict) else None
    if case_id not in by_id or case != by_id[case_id]:
        raise ValueError("case 必须与冻结 primary case 完全一致")
    return dict(by_id[case_id])


def _generator_kwargs(
    case: dict[str, Any],
    *,
    shadow: bool,
) -> dict[str, Any]:
    case = _canonical_primary_case(case)
    frozen = protocol.frozen_protocol_manifest()["generator"]
    _require(frozen["tol"] == "positive_infinity", "协议 tol 编码发生变化")
    return {
        "n_records": case["n_records"],
        "n_rounds": case["n_rounds"],
        "seed": case["seed"],
        "rho": case["rho"],
        "eta": frozen["eta"],
        "mu": frozen["mu"],
        "tol": float("inf"),
        "device": frozen["device"],
        "init_method": frozen["init_method"],
        "log_every": 0,
        "distance_mode": frozen["distance_mode"],
        "max_retries": frozen["max_retries"],
        "residual_directed_diffusion": frozen["residual_directed_diffusion"],
        "diffusion_direction_strength": frozen["diffusion_direction_strength"],
        "diffusion_direction_normalization": frozen[
            "diffusion_direction_normalization"
        ],
        "factorized_gibbs_sweeps": frozen["factorized_gibbs_sweeps"],
        "candidate_budget": case["candidate_budget"],
        "residual_self_cooling": frozen["residual_self_cooling"],
        "return_final_table": frozen["return_final_table"],
        "alpha_schedule_mode": frozen["alpha_schedule_mode"],
        "fixed_alpha": frozen["fixed_alpha"],
        "diffusion_direction_reference_scale": frozen[
            "diffusion_direction_reference_scale"
        ],
        "diffusion_direction_logit_clip": frozen["diffusion_direction_logit_clip"],
        "record_transition_clocks": frozen["record_transition_clocks"],
        "record_stationarity_trace": shadow,
        "stop_on_exact_residual": (
            False if shadow else frozen["stop_on_exact_residual"]
        ),
        "horizon_invariant": frozen["horizon_invariant"],
        "inner_early_stopping_patience_ticks": (
            None if shadow else case["patience_ticks"]
        ),
    }


def build_collection_plan(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    frozen_sha = protocol.assert_frozen_protocol_identity()
    cases = protocol.primary_case_matrix()
    online_round_cap = sum(case["n_rounds"] for case in cases)
    plan = {
        "contract_version": COLLECTION_CONTRACT_VERSION,
        "mode": "plan_only_no_generation_or_rng_instantiation",
        "protocol_sha256": frozen_sha,
        "output_dir": str(output_dir),
        "case_count": len(cases),
        "cases": cases,
        "online_round_cap": online_round_cap,
        "shadow_policy": "B_only",
        "maximum_generator_call_count": 2 * len(cases),
        "maximum_total_round_cap_if_all_B": 2 * online_round_cap,
        "requires_confirmed_protocol_sha256": True,
        "requires_clean_worktree_including_untracked": True,
        "overwrites_existing_output": False,
        "acceptance_evaluated_during_collection": False,
        "real_data_accessed": False,
        "privacy_budget_consumed": False,
        "generation_started": False,
    }
    _strict_json_bytes(plan)
    return plan


def build_execution_manifest(root: Path) -> dict[str, Any]:
    """Fail closed before any formal RNG construction or output write."""

    expected_parameters = {"output_dir", "confirmed_protocol_sha256"}
    if set(inspect.signature(run_primary_collection).parameters) != (
        expected_parameters
    ):
        raise RuntimeError("formal collector gained an unexpected override")
    frozen_sha = protocol.assert_frozen_protocol_identity()
    status = _git_text(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError("正式 P6 采集要求包含 untracked 在内的干净工作树")
    missing = [str(path) for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"正式 P6 采集源文件缺失：{missing}")
    manifest = {
        "contract_version": COLLECTION_CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": True,
        "protocol_sha256": frozen_sha,
        "protocol": protocol.frozen_protocol_manifest(),
        "source_sha256": {
            str(path): _sha256_file(root / path) for path in SOURCE_PATHS
        },
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": "numpy",
        },
        "execution_started": False,
        "formal_rng_instantiated": False,
        "acceptance_evaluated": False,
    }
    _strict_json_bytes(manifest)
    return manifest


def _run_evolution_quietly(*args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(StringIO()):
        return run_evolution(*args, **kwargs)


def _run_online(
    workload: ArtificialWorkload,
    case: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _run_evolution_quietly(
        workload.target,
        workload.queries,
        workload.schema,
        **_generator_kwargs(case, shadow=False),
    )


def _run_shadow(
    workload: ArtificialWorkload,
    case: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], StationarityTrace]:
    _, diagnostics = _run_evolution_quietly(
        workload.target,
        workload.queries,
        workload.schema,
        **_generator_kwargs(case, shadow=True),
    )
    trace = diagnostics.pop("stationarity_trace")
    final_table = diagnostics.pop("final_table")
    if not isinstance(trace, StationarityTrace):
        raise TypeError("shadow run 未返回 StationarityTrace")
    return final_table, diagnostics, trace


def _applied_rows(clock: dict[str, Any]) -> int:
    accepted_attempt = int(clock["accepted_attempt"])
    if accepted_attempt == 0:
        return 0
    attempts = clock["attempts"]
    if accepted_attempt < 0 or accepted_attempt > len(attempts):
        raise RuntimeError("transition clock 的 accepted_attempt 非法")
    return int(attempts[accepted_attempt - 1]["participating_rows"])


def _normalized_work_from_clocks(
    clocks: list[dict[str, Any]],
    n_records: int,
) -> float:
    return float(sum(_applied_rows(clock) for clock in clocks) / n_records)


def _audit_online_terminal(
    workload: ArtificialWorkload,
    case: dict[str, Any],
    output: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    reason = diagnostics["termination_reason"]
    expected_complete = {
        "fit_target_reached": True,
        "early_stopped": True,
        "resource_cap_reached": False,
    }
    _require(reason in expected_complete, "online run 返回未知 A/B/C 原因")
    _require(
        diagnostics["inner_complete"] is expected_complete[reason],
        "online inner_complete 与 A/B/C 原因不一致",
    )
    _require(
        diagnostics["output_table_identity"] == "terminal_current",
        "online output 不是 terminal current",
    )
    final_table = diagnostics["final_table"].reset_index(drop=True)
    output = output.reset_index(drop=True)
    _require(output.equals(final_table), "主输出与 diagnostics.final_table 不一致")
    terminal_sha = _frame_sha256(final_table)
    answers = np.asarray(evaluate_table(final_table, workload.queries), dtype=float)
    loss = float(compute_squared_loss(workload.target, answers))
    normalized_l1 = float(
        compute_normalized_l1(workload.target, answers, workload.n_records)
    )
    _require(
        loss == diagnostics["final_current_squared_loss"],
        "terminal table 的 loss 复算不一致",
    )
    _require(
        normalized_l1 == diagnostics["final_current_normalized_l1"],
        "terminal table 的 L1 复算不一致",
    )
    rounds_run = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    _require(rounds_run == len(clocks), "online rounds 与 transition clocks 不一致")
    stop_work = _normalized_work_from_clocks(clocks, workload.n_records)
    decision = diagnostics["inner_early_stopping"]["last_decision"]
    _require(decision is not None, "online run 缺少 A/B/C decision")
    _require(
        decision["termination_reason"] == reason,
        "online decision reason 与 diagnostics 不一致",
    )
    _require(
        decision["terminal_output_state_index"] == rounds_run,
        "online terminal state index 与 rounds 不一致",
    )
    _require(
        decision["normalized_work"] == stop_work,
        "online decision work 与 transition clocks 不一致",
    )
    _require(
        diagnostics["params"]["n_rounds"] == case["n_rounds"]
        and diagnostics["params"]["candidate_budget"] == case["candidate_budget"],
        "online C 参数与冻结 case 不一致",
    )
    return {
        "termination_reason": reason,
        "inner_complete": diagnostics["inner_complete"],
        "stop_state_index": rounds_run,
        "stop_normalized_work": stop_work,
        "candidate_evaluation_count": int(diagnostics["candidate_evaluation_count"]),
        "terminal_current_table_sha256": terminal_sha,
        "terminal_query_answers": [float(value) for value in answers],
        "terminal_current_squared_loss": loss,
        "terminal_current_normalized_l1": normalized_l1,
        "historical_best_loss_diagnostic_only": float(
            diagnostics["best_loss_diagnostic_only"]
        ),
    }


def _audit_b_shadow_prefix(
    workload: ArtificialWorkload,
    online_output: pd.DataFrame,
    online: dict[str, Any],
    shadow: dict[str, Any],
    trace: StationarityTrace,
) -> dict[str, Any]:
    stop_state = int(online["rounds_run"])
    _require(
        online["termination_reason"] == "early_stopped",
        "只有 B 才允许 shadow prefix audit",
    )
    _require(trace.state_count > stop_state, "shadow trace 未覆盖 B terminal")
    _require(
        trace.query_identity_sha256 == workload.query_identity_sha256,
        "shadow query identity 不一致",
    )
    _require(
        trace.target_identity_sha256 == workload.target_identity_sha256,
        "shadow target identity 不一致",
    )
    _require(
        online["current_state_metrics_history"]
        == shadow["current_state_metrics_history"][: stop_state + 1],
        "online/shadow current metrics prefix 不一致",
    )
    _require(
        online["transition_clock_history"]
        == shadow["transition_clock_history"][:stop_state],
        "online/shadow transition clock prefix 不一致",
    )
    for key in (
        "accept_history",
        "proposal_attempts_history",
        "accepted_attempt_history",
    ):
        _require(
            online[key] == shadow[key][:stop_state],
            f"online/shadow {key} prefix 不一致",
        )
    expected = trace.observations[stop_state]
    terminal_sha = _frame_sha256(online_output.reset_index(drop=True))
    _require(
        terminal_sha == expected["current_table_sha256"],
        "online terminal table 与 shadow 同状态身份不一致",
    )
    _require(
        online["primary_rng_state_sha256"] == expected["primary_rng_state_sha256"],
        "online/shadow RNG prefix 不一致",
    )
    _require(
        online["candidate_evaluation_count"]
        == expected["candidate_evaluation_count_cumulative"],
        "online/shadow candidate count prefix 不一致",
    )
    answers = np.asarray(evaluate_table(online_output, workload.queries), dtype=float)
    _require(
        np.array_equal(answers, trace.measured_query_answers[stop_state]),
        "online terminal query vector 与 shadow 不一致",
    )
    return {
        "current_metrics_prefix_equal": True,
        "transition_clocks_prefix_equal": True,
        "accept_history_prefix_equal": True,
        "proposal_attempts_prefix_equal": True,
        "accepted_attempt_prefix_equal": True,
        "terminal_table_identity_equal": True,
        "terminal_query_vector_equal": True,
        "primary_rng_prefix_equal": True,
        "candidate_evaluations_prefix_equal": True,
    }


def locate_b_shadow_checkpoints(
    trace: StationarityTrace,
    stop_state_index: int,
) -> list[dict[str, Any]]:
    """Locate the frozen +6/+12 work states without interpolation or imputation."""

    trace.validate()
    if (
        isinstance(stop_state_index, bool)
        or not isinstance(stop_state_index, int)
        or stop_state_index < 0
        or stop_state_index >= trace.state_count
    ):
        raise ValueError("stop_state_index 不在 shadow trace 中")
    cumulative_rows = []
    total = 0
    for observation in trace.observations:
        total += int(observation["applied_participating_row_count"])
        cumulative_rows.append(total)
    stop_work = cumulative_rows[stop_state_index] / trace.n_records
    checkpoints = []
    for offset in protocol.SHADOW_WORK_OFFSETS:
        target_work = stop_work + offset
        position = next(
            (
                state_index
                for state_index in range(stop_state_index + 1, trace.state_count)
                if cumulative_rows[state_index] / trace.n_records >= target_work
            ),
            None,
        )
        if position is None:
            checkpoints.append(
                {
                    "work_offset": offset,
                    "target_normalized_work": target_work,
                    "status": "right_censored_by_resource_guard",
                    "state_index": None,
                    "actual_normalized_work": None,
                    "actual_extra_work": None,
                    "extra_raw_rounds": None,
                    "extra_candidate_evaluations": None,
                    "current_table_sha256": None,
                    "current_query_answers": None,
                    "current_squared_loss": None,
                    "current_normalized_l1": None,
                }
            )
            continue
        observation = trace.observations[position]
        actual_work = cumulative_rows[position] / trace.n_records
        stop_candidates = trace.observations[stop_state_index][
            "candidate_evaluation_count_cumulative"
        ]
        checkpoints.append(
            {
                "work_offset": offset,
                "target_normalized_work": target_work,
                "status": "observed",
                "state_index": position,
                "actual_normalized_work": actual_work,
                "actual_extra_work": actual_work - stop_work,
                "extra_raw_rounds": position - stop_state_index,
                "extra_candidate_evaluations": (
                    observation["candidate_evaluation_count_cumulative"]
                    - stop_candidates
                ),
                "current_table_sha256": observation["current_table_sha256"],
                "current_query_answers": [
                    float(value) for value in trace.measured_query_answers[position]
                ],
                "current_squared_loss": observation["current_squared_loss"],
                "current_normalized_l1": observation["current_normalized_l1"],
            }
        )
    _strict_json_bytes(checkpoints)
    return checkpoints


def _serializable_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    record = {key: diagnostics[key] for key in _ONLINE_DIAGNOSTIC_KEYS}
    params = dict(record["params"])
    if not np.isposinf(params["tol"]):
        raise RuntimeError("diagnostics tol 不是冻结的正无穷")
    params["tol"] = "positive_infinity"
    record["params"] = params
    return json.loads(_strict_json_bytes(record))


def _collect_case(
    cases_dir: Path,
    case: dict[str, Any],
    workload: ArtificialWorkload,
    execution_manifest_sha256: str,
) -> Path:
    case = _canonical_primary_case(case)
    destination = cases_dir / case["case_id"]
    if destination.exists():
        raise FileExistsError(f"case 输出已存在，拒绝覆盖：{destination}")
    temporary = Path(
        tempfile.mkdtemp(dir=cases_dir, prefix=f".{case['case_id']}.partial-")
    )
    try:
        online_output, online = _run_online(workload, case)
        online_audit = _audit_online_terminal(
            workload,
            case,
            online_output,
            online,
        )
        final_table = online["final_table"].reset_index(drop=True)
        online_record = _serializable_diagnostics(online)
        terminal_table_path = temporary / "terminal_current.csv"
        online_diagnostics_path = temporary / "online_diagnostics.json"
        _write_frame_exclusive(terminal_table_path, final_table)
        _write_json_exclusive(online_diagnostics_path, online_record)

        shadow_manifest: dict[str, Any] = {
            "collected": False,
            "role": "B_only_read_only_continuation",
            "prefix_audit": None,
            "termination_reason": None,
            "rounds_run": None,
            "candidate_evaluation_count": None,
            "checkpoints": [],
            "files": {},
        }
        if online_audit["termination_reason"] == "early_stopped":
            shadow_final_table, shadow, trace = _run_shadow(workload, case)
            _require(
                shadow["rounds_run"] == case["n_rounds"],
                "shadow 未跑满冻结 raw-round C",
            )
            _require(
                shadow["candidate_evaluation_count"] == case["candidate_budget"],
                "shadow 未跑满冻结 candidate C",
            )
            _require(
                shadow["termination_reason"] == trace.termination_reason,
                "shadow diagnostics/trace 终止原因不一致",
            )
            _require(
                _frame_sha256(shadow_final_table)
                == trace.observations[-1]["current_table_sha256"],
                "shadow final table 与 trace 末状态不一致",
            )
            prefix_audit = _audit_b_shadow_prefix(
                workload,
                online_output,
                online,
                shadow,
                trace,
            )
            checkpoints = locate_b_shadow_checkpoints(
                trace,
                int(online["rounds_run"]),
            )
            trace_paths = save_stationarity_trace(
                trace,
                temporary / "shadow_trace",
            )
            shadow_summary_path = temporary / "shadow_summary.json"
            shadow_summary = {
                "termination_reason": shadow["termination_reason"],
                "rounds_run": shadow["rounds_run"],
                "candidate_evaluation_count": shadow["candidate_evaluation_count"],
                "final_current_squared_loss": shadow["final_current_squared_loss"],
                "final_current_normalized_l1": shadow["final_current_normalized_l1"],
                "final_current_table_sha256": _frame_sha256(shadow_final_table),
            }
            _write_json_exclusive(shadow_summary_path, shadow_summary)
            metadata_path = Path(trace_paths["metadata_path"])
            query_array_path = Path(trace_paths["query_array_path"])
            shadow_manifest = {
                "collected": True,
                "role": "B_only_read_only_continuation",
                "prefix_audit": prefix_audit,
                "termination_reason": shadow["termination_reason"],
                "rounds_run": shadow["rounds_run"],
                "candidate_evaluation_count": shadow["candidate_evaluation_count"],
                "checkpoints": checkpoints,
                "files": {
                    "trace_metadata": {
                        "path": str(metadata_path.relative_to(temporary)),
                        "sha256": _sha256_file(metadata_path),
                    },
                    "trace_query_array": {
                        "path": str(query_array_path.relative_to(temporary)),
                        "sha256": _sha256_file(query_array_path),
                    },
                    "summary": {
                        "path": shadow_summary_path.name,
                        "sha256": _sha256_file(shadow_summary_path),
                    },
                },
            }

        case_manifest = {
            "contract_version": COLLECTION_CONTRACT_VERSION,
            "protocol_sha256": protocol.assert_frozen_protocol_identity(),
            "execution_manifest_sha256": execution_manifest_sha256,
            "case": case,
            "family_identity_sha256": workload.family_identity_sha256,
            "query_identity_sha256": workload.query_identity_sha256,
            "target_identity_sha256": workload.target_identity_sha256,
            "reference_multiset_passed_to_generator": False,
            "online": {
                **online_audit,
                "files": {
                    "terminal_current_table": {
                        "path": terminal_table_path.name,
                        "sha256": _sha256_file(terminal_table_path),
                    },
                    "diagnostics": {
                        "path": online_diagnostics_path.name,
                        "sha256": _sha256_file(online_diagnostics_path),
                    },
                },
            },
            "shadow": shadow_manifest,
            "acceptance_evaluated": False,
            "partial_matrix_classification_emitted": False,
        }
        case_manifest_path = temporary / "case_manifest.json"
        _write_json_exclusive(case_manifest_path, case_manifest)
        temporary.replace(destination)
        return destination / case_manifest_path.name
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_primary_collection(
    output_dir: Path,
    confirmed_protocol_sha256: str,
) -> Path:
    """Execute exactly the frozen primary matrix after separate authorization."""

    expected_sha = protocol.assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected_sha:
        raise ValueError("必须显式确认完整冻结 P6 protocol SHA-256")
    root = _repo_root()
    execution_manifest = build_execution_manifest(root)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    execution_manifest_path = output_dir / "execution_manifest.json"
    _write_json_exclusive(execution_manifest_path, execution_manifest)
    execution_manifest_sha = _sha256_file(execution_manifest_path)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir()

    started = time.perf_counter()
    case_files: dict[str, dict[str, str]] = {}
    workloads: dict[str, ArtificialWorkload] = {}
    cases = protocol.primary_case_matrix()
    for position, case in enumerate(cases, start=1):
        if case["family"] not in workloads:
            workloads[case["family"]] = materialize_family(case["family"])
        workload = workloads[case["family"]]
        case_manifest_path = _collect_case(
            cases_dir,
            case,
            workload,
            execution_manifest_sha,
        )
        case_files[case["case_id"]] = {
            "path": str(case_manifest_path.relative_to(output_dir)),
            "sha256": _sha256_file(case_manifest_path),
        }
        print(
            f"[P6 collection {position}/{len(cases)}] {case['case_id']} collected",
            flush=True,
        )

    collection = {
        "contract_version": COLLECTION_CONTRACT_VERSION,
        "protocol_sha256": expected_sha,
        "execution_manifest": {
            "path": execution_manifest_path.name,
            "sha256": execution_manifest_sha,
        },
        "formal_primary_collection_complete": True,
        "case_count": len(case_files),
        "case_manifest_files": case_files,
        "collection_elapsed_sec": float(time.perf_counter() - started),
        "acceptance_evaluated": False,
        "partial_matrix_classification_emitted": False,
        "real_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    collection_path = output_dir / "collection_manifest.json"
    _write_json_exclusive(collection_path, collection)
    return collection_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(
            json.dumps(
                build_collection_plan(args.output_dir),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        return
    destination = run_primary_collection(
        args.output_dir,
        args.confirm_protocol_sha,
    )
    print(f"P6 raw collection -> {destination}", flush=True)


if __name__ == "__main__":
    main()
