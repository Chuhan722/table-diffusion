#!/usr/bin/env python3
"""采集 Issue #53 固定 α 响应曲线的两数据冻结矩阵。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__:
    from scripts import freeze_issue53_test_query_workload_ab as freeze_test
else:
    import freeze_issue53_test_query_workload_ab as freeze_test


PROTOCOL_VERSION = "issue53-fixed-alpha-calibration-collection-v1"
FROZEN_PROTOCOL_SHA256 = (
    "6a3716f11ed6a4233256b9d3a549fc45281bc464470cc82a6e64d66d0104b311"
)
PROTOCOL_DOC = Path("docs/设计/Issue53_固定alpha响应曲线结果前冻结协议.md")
PROTOCOL_DOC_SHA256 = (
    "dfd2739f7c9968ad9a9f7a094d8362407f31740fbc84ffce30cac851c439fd6d"
)
PROTOCOL_DOC_COMMIT = "a3724e0c97a1bc1a899e87d1f3727e21977d56ba"

OUTPUT_DIR = Path("outputs/issue53_fixed_alpha_calibration_v1")
COLLECTION_REPORT = "collection_report.json"
SEEDS = (323, 324, 325, 326, 327)
ALPHAS = (16.0, 12.0, 24.0)
DATASET_ORDER = ("test_300x10", "nltcs")
CASE_ORDER = tuple(
    (dataset, alpha) for dataset in DATASET_ORDER for alpha in ALPHAS
)

PATIENCE_TICKS = 6
RHO = 0.01
ROUND_CAP = 6000
CANDIDATE_BUDGET = 6000
EXECUTION_HOSTNAME = "linyao-system"

DATASETS: dict[str, dict[str, Any]] = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path(
            "configs/test_300x10/measured_50query_30_15_5.json"
        ),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "device": "numpy",
        "query_count": 50,
        "order_counts": {2: 30, 3: 15, 4: 5},
        "query_identity_sha256": (
            "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
        ),
        "target_vector_sha256": (
            "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
        ),
        "sha256": {
            "schema": (
                "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
            ),
            "queries": (
                "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
            ),
            "marginals": (
                "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
            ),
        },
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        "device": "cuda",
        "query_count": 1001,
        "order_counts": {2: 479, 3: 522},
        "query_identity_sha256": (
            "48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429"
        ),
        "target_vector_sha256": (
            "f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d"
        ),
        "sha256": {
            "schema": (
                "5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4"
            ),
            "queries": (
                "b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f"
            ),
            "marginals": (
                "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e"
            ),
        },
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: Any) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _git_text(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, float) and math.isinf(value):
            value = "positive_infinity"
        result[key] = value
    return result


def _alpha_label(alpha: float) -> str:
    if alpha not in ALPHAS:
        raise ValueError(f"alpha 不在冻结矩阵中：{alpha!r}")
    return f"alpha_{int(alpha)}"


def generator_params(seed: int, alpha: float) -> dict[str, Any]:
    """返回冻结矩阵中的一个精确生成配置。"""

    if seed not in SEEDS:
        raise ValueError(f"seed 不在冻结矩阵中：{seed!r}")
    if alpha not in ALPHAS:
        raise ValueError(f"alpha 不在冻结矩阵中：{alpha!r}")
    return {
        "n_rounds": ROUND_CAP,
        "seed": seed,
        "beta": 1.0,
        "h": 0.8,
        "rho": RHO,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "eval_method": "vectorized",
        "batch_size": 256,
        "log_every": 100,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": alpha,
        "alpha_max": alpha,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": alpha,
        "record_transition_clocks": True,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def _common_generator_params() -> dict[str, Any]:
    params = generator_params(SEEDS[0], ALPHAS[0]).copy()
    for key in ("seed", "alpha_min", "alpha_max", "fixed_alpha"):
        params.pop(key)
    return _jsonable_params(params)


def frozen_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "fixed_alpha_response_calibration_not_alpha_selection",
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
        "protocol_doc_commit": PROTOCOL_DOC_COMMIT,
        "datasets": {
            name: {
                "schema": str(spec["schema"]),
                "queries": str(spec["queries"]),
                "marginals": str(spec["marginals"]),
                "n_records": spec["n_records"],
                "device": spec["device"],
                "query_count": spec["query_count"],
                "order_counts": {
                    str(order): count
                    for order, count in spec["order_counts"].items()
                },
                "query_identity_sha256": spec["query_identity_sha256"],
                "target_vector_sha256": spec["target_vector_sha256"],
                "input_sha256": spec["sha256"],
            }
            for name, spec in DATASETS.items()
        },
        "dataset_order": list(DATASET_ORDER),
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "case_order_within_seed": [
            {"dataset": dataset, "alpha": alpha}
            for dataset, alpha in CASE_ORDER
        ],
        "trajectory_count": len(SEEDS) * len(CASE_ORDER),
        "common_generator": _common_generator_params(),
        "only_scientific_variable": "fixed_alpha",
        "natural_work": "cumulative_applied_participating_rows/n_records",
        "output_identity": "terminal_current",
        "initial_table_pairing": "same_sha_across_three_alphas_per_dataset_seed",
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
        "parameter_retuning_allowed": False,
        "fixed_alpha_selection_allowed": False,
        "adaptive_alpha_design_in_scope": False,
        "execution": {
            "server": EXECUTION_HOSTNAME,
            "one_visible_gpu_per_seed_shard": True,
            "maximum_parallel_seed_shards": 2,
            "cases_within_seed_shard_serial": True,
        },
    }


def protocol_sha256() -> str:
    return hashlib.sha256(_strict_json_bytes(frozen_protocol_manifest())).hexdigest()


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol 身份漂移：expected={FROZEN_PROTOCOL_SHA256}, "
            f"observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": frozen_protocol_manifest(),
        "shards": [
            {
                "shard_index": index,
                "seed": seed,
                "case_count": len(CASE_ORDER),
            }
            for index, seed in enumerate(SEEDS)
        ],
        "output_dir": str(OUTPUT_DIR),
        "scientific_overrides_allowed": False,
        "generation_started": False,
    }


def _target_vector_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(_strict_json_bytes(list(values))).hexdigest()


def _audit_dataset(root: Path, name: str) -> dict[str, Any]:
    if name not in DATASETS:
        raise ValueError(f"未知数据集：{name!r}")
    spec = DATASETS[name]
    observed_sha = {
        key: _sha256_file(root / spec[key])
        for key in ("schema", "queries", "marginals")
    }
    if observed_sha != spec["sha256"]:
        raise RuntimeError(f"{name} 输入 SHA 漂移")
    payload = _load_json(root / spec["queries"])
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != spec["query_count"]:
        raise RuntimeError(f"{name} query count 漂移")
    identity = freeze_test.query_set_identity(queries)
    if identity != spec["query_identity_sha256"]:
        raise RuntimeError(f"{name} query identity 漂移")
    order_counts = freeze_test._order_counts(queries)
    if order_counts != spec["order_counts"] or order_counts.get(1, 0) != 0:
        raise RuntimeError(f"{name} 查询阶数或 1-way 边界漂移")
    targets = []
    for index, query in enumerate(queries):
        value = query.get("result")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} result 非整数：index={index}")
        targets.append(value)
    if _target_vector_sha256(targets) != spec["target_vector_sha256"]:
        raise RuntimeError(f"{name} target vector 漂移")
    return {
        "dataset": name,
        "sha256": observed_sha,
        "queries": queries,
        "targets": targets,
        "query_count": len(queries),
        "query_identity_sha256": identity,
        "target_vector_sha256": spec["target_vector_sha256"],
        "order_counts": order_counts,
    }


def _audit_inputs(root: Path) -> dict[str, Any]:
    if _sha256_file(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("protocol 文档 SHA 漂移")
    return {
        name: {
            key: value
            for key, value in _audit_dataset(root, name).items()
            if key not in {"queries", "targets"}
        }
        for name in DATASET_ORDER
    }


def _load_runtime() -> SimpleNamespace:
    import numpy as np
    import pandas as pd
    import torch

    from table_diffevo.evolution import run_evolution
    from table_diffevo.marginals import load_marginals
    from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
    from table_diffevo.queries import evaluate_table, load_queries
    from table_diffevo.schema import load_schema

    return SimpleNamespace(
        np=np,
        pd=pd,
        torch=torch,
        run_evolution=run_evolution,
        load_marginals=load_marginals,
        compute_normalized_l1=compute_normalized_l1,
        compute_squared_loss=compute_squared_loss,
        evaluate_table=evaluate_table,
        load_queries=load_queries,
        load_schema=load_schema,
    )


def _environment(root: Path, runtime: SimpleNamespace) -> dict[str, Any]:
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式采集要求包含 untracked 在内的干净工作树")
    hostname = platform.node()
    if hostname != EXECUTION_HOSTNAME:
        raise RuntimeError(
            f"执行主机漂移：expected={EXECUTION_HOSTNAME}, observed={hostname}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise RuntimeError("每个 seed shard 必须且只能显式暴露一张 GPU")
    if not runtime.torch.cuda.is_available() or runtime.torch.cuda.device_count() != 1:
        raise RuntimeError("nltcs CUDA 路径要求进程内恰好一张可用 GPU")
    return {
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "worktree_clean_including_untracked": True,
        "hostname": hostname,
        "python": sys.version,
        "numpy": runtime.np.__version__,
        "pandas": runtime.pd.__version__,
        "torch": runtime.torch.__version__,
        "cuda_visible_devices": visible,
        "cuda_device_name": runtime.torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _applied_rows(clock: dict[str, Any]) -> int:
    accepted = int(clock["accepted_attempt"])
    if accepted == 0:
        return 0
    return int(clock["attempts"][accepted - 1]["participating_rows"])


def _series_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("集中度诊断历史为空")
    tail = values[-min(100, len(values)) :]
    return {
        "final": float(values[-1]),
        "tail_mean": float(statistics.fmean(tail)),
        "full_min": float(min(values)),
        "full_max": float(max(values)),
    }


def _concentration_summary(
    diagnostics: dict[str, Any], n_records: int
) -> dict[str, Any]:
    row_mean = diagnostics["row_max_prob_mean_history"]
    row_max = diagnostics["row_max_prob_max_history"]
    effective = diagnostics["effective_donors_mean_history"]
    top_share = diagnostics["donor_top_share_history"]
    rounds = int(diagnostics["rounds_run"])
    if not all(len(values) == rounds for values in (
        row_mean, row_max, effective, top_share
    )):
        raise RuntimeError("供体集中度诊断长度与运行轮数不一致")
    denominator = n_records - 1
    effective_fraction = [float(value / denominator) for value in effective]
    return {
        "row_max_prob_mean": _series_summary(row_mean),
        "row_max_prob_max": _series_summary(row_max),
        "effective_donors_mean": _series_summary(effective),
        "effective_donor_fraction": _series_summary(effective_fraction),
        "donor_top_share": _series_summary(top_share),
        "tail_window_rounds": min(100, rounds),
        "effective_donor_fraction_denominator": denominator,
    }


def _run_case(
    root: Path,
    shard_dir: Path,
    *,
    dataset: str,
    alpha: float,
    seed: int,
    protocol_sha: str,
    git_commit: str,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    audit = _audit_dataset(root, dataset)
    schema = runtime.load_schema(str(root / spec["schema"]))
    queries = runtime.load_queries(str(root / spec["queries"]))
    marginals = runtime.load_marginals(str(root / spec["marginals"]))
    target = runtime.np.asarray(audit["targets"], dtype=float)
    params = generator_params(seed, alpha)

    started = time.perf_counter()
    table, diagnostics = runtime.run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=spec["n_records"],
        marginals=marginals,
        device=spec["device"],
        init_method="marginal",
        **params,
    )
    elapsed = time.perf_counter() - started

    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{dataset}/alpha{alpha}/seed{seed} 不是终态当前表")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError("输出身份不是 terminal current")
    observed = diagnostics["params"]
    if (
        observed["alpha_schedule_mode"] != "fixed"
        or observed["fixed_alpha"] != alpha
        or observed["selection_scale_invariant"] is not True
        or observed["residual_geometry"] != "relative"
        or observed["residual_geometry_floor"] != 8.0
        or observed["factorized_gibbs_sweeps"] != 0
    ):
        raise RuntimeError(f"{dataset}/alpha{alpha}/seed{seed} 生成参数漂移")
    if diagnostics["alpha_history"] != [alpha] * diagnostics["rounds_run"]:
        raise RuntimeError(f"{dataset}/alpha{alpha}/seed{seed} alpha 历史漂移")

    reason = diagnostics["termination_reason"]
    if reason not in {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }:
        raise RuntimeError(f"{dataset}/alpha{alpha}/seed{seed} 未返回 A/B/C")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != rounds or diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError(f"{dataset}/alpha{alpha}/seed{seed} no-gate 审计失败")
    if diagnostics["candidate_evaluation_count"] != rounds:
        raise RuntimeError("候选评估数与轮数不一致")

    answers = runtime.np.asarray(runtime.evaluate_table(final_table, queries))
    loss = float(runtime.compute_squared_loss(target, answers))
    l1 = float(
        runtime.compute_normalized_l1(target, answers, spec["n_records"])
    )
    if (
        loss != diagnostics["final_current_squared_loss"]
        or l1 != diagnostics["final_current_normalized_l1"]
    ):
        raise RuntimeError("终态测量指标复算不一致")
    work = sum(_applied_rows(clock) for clock in clocks) / spec["n_records"]
    concentration = _concentration_summary(diagnostics, spec["n_records"])

    case_dir = shard_dir / dataset / _alpha_label(alpha)
    case_dir.mkdir(parents=True)
    table_path = case_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    result = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "git_commit": git_commit,
        "dataset": dataset,
        "alpha": alpha,
        "seed": seed,
        "device": spec["device"],
        "query_file": str(spec["queries"]),
        "query_count": spec["query_count"],
        "query_order_counts": {
            str(order): count for order, count in audit["order_counts"].items()
        },
        "query_identity_sha256": audit["query_identity_sha256"],
        "target_vector_sha256": audit["target_vector_sha256"],
        "termination_reason": reason,
        "inner_complete": bool(diagnostics["inner_complete"]),
        "output_table_identity": "terminal_current",
        "rounds_run": rounds,
        "candidate_evaluations": int(diagnostics["candidate_evaluation_count"]),
        "normalized_work_at_stop": float(work),
        "terminal_current_squared_loss": loss,
        "terminal_current_normalized_l1": l1,
        "best_loss_diagnostic_only": float(diagnostics["best_loss"]),
        "elapsed_sec": float(elapsed),
        "initial_table_sha256": diagnostics["initial_table_sha256"],
        "primary_rng_post_initialization_state_sha256": diagnostics[
            "primary_rng_post_initialization_state_sha256"
        ],
        "terminal_table_sha256": _frame_sha256(final_table),
        "donor_concentration": concentration,
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[seed={seed} {dataset}/alpha={alpha:g}] {reason} "
        f"rounds={rounds} work={work:.4f} L1={l1:.10f} "
        f"effective={concentration['effective_donor_fraction']['final']:.6f} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _assert_seed_pairing(
    rows: Sequence[dict[str, Any]], seed: int
) -> dict[str, dict[str, str]]:
    if len(rows) != len(CASE_ORDER):
        raise RuntimeError(f"seed {seed} 不是完整六臂")
    pairing = {}
    for dataset in DATASET_ORDER:
        selected = [row for row in rows if row["dataset"] == dataset]
        if len(selected) != len(ALPHAS):
            raise RuntimeError(f"seed {seed}/{dataset} 不是完整三 α")
        table_hashes = {row["initial_table_sha256"] for row in selected}
        rng_hashes = {
            row["primary_rng_post_initialization_state_sha256"]
            for row in selected
        }
        if len(table_hashes) != 1 or len(rng_hashes) != 1:
            raise RuntimeError(f"seed {seed}/{dataset} 初始状态未配对")
        pairing[dataset] = {
            "initial_table_sha256": next(iter(table_hashes)),
            "primary_rng_post_initialization_state_sha256": next(iter(rng_hashes)),
        }
    return pairing


def run_shard(confirmed_protocol_sha256: str, shard_index: int) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    if shard_index not in range(len(SEEDS)):
        raise ValueError(f"shard_index 非法：{shard_index!r}")

    root = _repo_root()
    seed = SEEDS[shard_index]
    destination = root / OUTPUT_DIR / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(f"seed shard 已存在，不覆盖：{destination}")
    runtime = _load_runtime()
    environment = _environment(root, runtime)
    input_audit = _audit_inputs(root)
    git_commit = environment["git_commit"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".seed_{seed}.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        results = [
            _run_case(
                root,
                temporary,
                dataset=dataset,
                alpha=alpha,
                seed=seed,
                protocol_sha=expected,
                git_commit=git_commit,
                runtime=runtime,
            )
            for dataset, alpha in CASE_ORDER
        ]
        pairing = _assert_seed_pairing(results, seed)
        manifest = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "shard_index": shard_index,
            "seed": seed,
            "git_commit": git_commit,
            "environment": environment,
            "input_audit": input_audit,
            "case_order": [
                {"dataset": dataset, "alpha": alpha}
                for dataset, alpha in CASE_ORDER
            ],
            "initial_state_pairing": pairing,
            "results": results,
        }
        (temporary / "shard_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "shard_manifest.json"


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = {dataset: {} for dataset in DATASET_ORDER}
    for dataset in DATASET_ORDER:
        for alpha in ALPHAS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["alpha"] == alpha
            ]
            if len(selected) != len(SEEDS):
                raise RuntimeError(f"{dataset}/alpha{alpha} case 数漂移")
            cells[dataset][_alpha_label(alpha)] = {
                "case_count": len(selected),
                "termination_counts": dict(
                    sorted(Counter(row["termination_reason"] for row in selected).items())
                ),
                "terminal_normalized_l1_mean": _mean(
                    [row["terminal_current_normalized_l1"] for row in selected]
                ),
                "terminal_normalized_l1_median": _median(
                    [row["terminal_current_normalized_l1"] for row in selected]
                ),
                "terminal_squared_loss_mean": _mean(
                    [row["terminal_current_squared_loss"] for row in selected]
                ),
                "rounds_mean": _mean([row["rounds_run"] for row in selected]),
                "normalized_work_mean": _mean(
                    [row["normalized_work_at_stop"] for row in selected]
                ),
                "elapsed_sec_mean": _mean([row["elapsed_sec"] for row in selected]),
                "row_max_prob_mean_final_mean": _mean(
                    [
                        row["donor_concentration"]["row_max_prob_mean"]["final"]
                        for row in selected
                    ]
                ),
                "effective_donor_fraction_final_mean": _mean(
                    [
                        row["donor_concentration"]["effective_donor_fraction"][
                            "final"
                        ]
                        for row in selected
                    ]
                ),
            }
    return {
        "cells": cells,
        "resource_cap_case_count": sum(
            row["termination_reason"] == "resource_cap_reached" for row in rows
        ),
        "normal_completion_case_count": sum(
            row["termination_reason"] in {"fit_target_reached", "early_stopped"}
            for row in rows
        ),
    }


def aggregate(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    root = _repo_root()
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("aggregate 要求包含 untracked 在内的干净工作树")
    destination = root / OUTPUT_DIR
    report_path = destination / COLLECTION_REPORT
    if report_path.exists():
        raise FileExistsError(f"采集报告已存在，不覆盖：{report_path}")

    rows = []
    commits = set()
    pairing = {}
    for index, seed in enumerate(SEEDS):
        shard_path = destination / f"seed_{seed}" / "shard_manifest.json"
        payload = _load_json(shard_path)
        if (
            payload.get("protocol_sha256") != expected
            or payload.get("shard_index") != index
            or payload.get("seed") != seed
        ):
            raise RuntimeError(f"seed {seed} shard 身份漂移")
        commits.add(payload["git_commit"])
        seed_rows = payload.get("results")
        if not isinstance(seed_rows, list):
            raise TypeError(f"seed {seed} results 缺失")
        pairing[str(seed)] = _assert_seed_pairing(seed_rows, seed)
        for row in seed_rows:
            table_path = (
                destination
                / f"seed_{seed}"
                / row["dataset"]
                / _alpha_label(row["alpha"])
                / "terminal_current.csv"
            )
            if _sha256_file(table_path) != row["terminal_table_sha256"]:
                raise RuntimeError(
                    f"seed {seed}/{row['dataset']}/alpha{row['alpha']} table SHA 漂移"
                )
            rows.append(row)

    identities = {(row["seed"], row["dataset"], row["alpha"]) for row in rows}
    expected_identities = {
        (seed, dataset, alpha)
        for seed in SEEDS
        for dataset, alpha in CASE_ORDER
    }
    if len(rows) != 30 or identities != expected_identities:
        raise RuntimeError("30-case 矩阵不完整或重复")
    if len(commits) != 1:
        raise RuntimeError(f"shard Git commit 不一致：{sorted(commits)}")

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    alpha_rank = {value: index for index, value in enumerate(ALPHAS)}
    report = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": expected,
        "protocol": frozen_protocol_manifest(),
        "execution_git_commit": next(iter(commits)),
        "case_count": len(rows),
        "raw_results": sorted(
            rows,
            key=lambda row: (
                row["seed"],
                dataset_rank[row["dataset"]],
                alpha_rank[row["alpha"]],
            ),
        ),
        "initial_state_pairing_by_seed": pairing,
        "summary": _summarize(rows),
        "claim_scope": "fixed_alpha_response_collection_before_offline_evaluation",
        "fixed_alpha_selection_allowed": False,
        "adaptive_alpha_design_in_scope": False,
        "parameter_retuning_performed": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".collection-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_path, report_path)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--confirm-protocol-sha", required=True)
    shard.add_argument(
        "--shard-index", required=True, type=int, choices=range(len(SEEDS))
    )
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    if args.command == "run-shard":
        path = run_shard(args.confirm_protocol_sha, args.shard_index)
        print(f"固定 alpha shard -> {path}")
        return
    path = aggregate(args.confirm_protocol_sha)
    print(f"固定 alpha collection -> {path}")
    print(f"collection SHA-256 -> {_sha256_file(path)}")


if __name__ == "__main__":
    main()
