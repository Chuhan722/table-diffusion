#!/usr/bin/env python3
"""Collect the frozen Issue #53 test query-workload A/B matrix."""

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
    from scripts import freeze_issue53_test_query_workload_ab as freeze
    from scripts import materialize_issue53_test_query_workload_b as materialize
else:
    import freeze_issue53_test_query_workload_ab as freeze
    import materialize_issue53_test_query_workload_b as materialize


PROTOCOL_VERSION = "issue53-test-query-workload-ab-collection-v1"
FROZEN_PROTOCOL_SHA256 = (
    "5b27cc3ddd5b39829a584f1cdc06b961ef50204840d957481444297023a18f0f"
)
PROTOCOL_DOC = freeze.PROTOCOL_DOC
PROTOCOL_DOC_SHA256 = materialize.PROTOCOL_DOC_SHA256
PROTOCOL_DOC_COMMIT = "5eda8263db8e292a9f9999ff748d9ee71a5a67f9"
ANSWER_INPUT_COMMIT = "35f2850b8191e8b7acac48493b15d2b865be39a9"

OUTPUT_DIR = Path("outputs/issue53_test_query_workload_ab_v1")
COLLECTION_REPORT = "collection_report.json"
DATASET = "test_300x10"
SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
MARGINALS_PATH = freeze.MARGINALS_PATH
N_RECORDS = freeze.N_RECORDS
DEVICE = "numpy"

INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "marginals": freeze.EXPECTED_INPUT_SHA256["marginals"],
    "identity_artifact": materialize.IDENTITY_ARTIFACT_SHA256,
    "workload_a": freeze.EXPECTED_INPUT_SHA256["workload_a"],
    "workload_b": (
        "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
    ),
    "protocol_doc": PROTOCOL_DOC_SHA256,
}
WORKLOADS = {
    "A": {
        "path": freeze.WORKLOAD_A_PATH,
        "sha256": INPUT_SHA256["workload_a"],
        "query_identity_sha256": freeze.EXPECTED_WORKLOAD_A_IDENTITY,
        "target_vector_sha256": (
            "7f6abd69b51cc85ed01a5b160e5c87ee01e892ed641a7631909e2fe892ab6f31"
        ),
        "order_counts": {1: 25, 2: 20, 3: 5},
    },
    "B": {
        "path": materialize.OUTPUT_PATH,
        "sha256": INPUT_SHA256["workload_b"],
        "query_identity_sha256": materialize.WORKLOAD_B_IDENTITY_SHA256,
        "target_vector_sha256": (
            "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
        ),
        "order_counts": {2: 30, 3: 15, 4: 5},
    },
}

SEEDS = (318, 319, 320, 321, 322)
GEOMETRIES = ("absolute", "sqrt_relative", "relative")
CASE_ORDER = tuple(
    (workload, geometry)
    for geometry in GEOMETRIES
    for workload in ("A", "B")
)

PATIENCE_TICKS = 6
RHO = 0.01
ROUND_CAP = 6000
CANDIDATE_BUDGET = 6000
EXECUTION_HOSTNAME = "linyao-system"


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


def generator_params(seed: int, geometry: str) -> dict[str, Any]:
    """Return one exact case from the frozen 30-case matrix."""

    if seed not in SEEDS:
        raise ValueError(f"seed 不在冻结矩阵中：{seed!r}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry 不在冻结矩阵中：{geometry!r}")
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
        "alpha_min": 16.0,
        "alpha_max": 16.0,
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
        "residual_geometry": geometry,
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 16.0,
        "record_transition_clocks": True,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def _common_generator_params() -> dict[str, Any]:
    params = generator_params(SEEDS[0], GEOMETRIES[0]).copy()
    params.pop("seed")
    params.pop("residual_geometry")
    return _jsonable_params(params)


def frozen_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "frozen_test_query_workload_ab_raw_collection",
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
        "protocol_doc_commit": PROTOCOL_DOC_COMMIT,
        "answer_input_commit": ANSWER_INPUT_COMMIT,
        "dataset": {
            "name": DATASET,
            "n_records": N_RECORDS,
            "device": DEVICE,
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
        },
        "workloads": {
            name: {
                "path": str(spec["path"]),
                "query_count": 50,
                "order_counts": {
                    str(order): count
                    for order, count in spec["order_counts"].items()
                },
                "query_identity_sha256": spec["query_identity_sha256"],
                "target_vector_sha256": spec["target_vector_sha256"],
                "input_sha256": spec["sha256"],
            }
            for name, spec in WORKLOADS.items()
        },
        "geometries": list(GEOMETRIES),
        "seeds": list(SEEDS),
        "case_order_within_seed": [
            {"workload": workload, "geometry": geometry}
            for workload, geometry in CASE_ORDER
        ],
        "trajectory_count": len(WORKLOADS) * len(GEOMETRIES) * len(SEEDS),
        "common_generator": _common_generator_params(),
        "workload_parameter": "measured_query_file_and_target_vector",
        "geometry_parameter": "residual_geometry",
        "four_way_path_contract": {
            "workload_b_four_way_query_count": 5,
            "queries_passed_untruncated_to_objective_and_direction": True,
            "terminal_measured_loss_recomputed_over_all_50_queries": True,
            "factorized_gibbs_sweeps": 0,
            "factorized_gibbs_use_compiled_workload": False,
            "factorized_gibbs_max_order_is_inert": True,
        },
        "natural_work": "cumulative_applied_participating_rows/n_records",
        "output_identity": "terminal_current",
        "initial_table_pairing": "same_sha_across_six_cases_within_seed",
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
        "parameter_retuning_allowed": False,
        "canonical_selection_allowed": False,
        "execution_concurrency": {
            "server": EXECUTION_HOSTNAME,
            "generator_device": "numpy",
            "cuda_visible_devices": "empty",
            "worker_count": 5,
            "parallel_unit": "seed_shard",
            "cases_within_shard_serial": True,
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
        "case_order_within_shard": [
            {"workload": workload, "geometry": geometry}
            for workload, geometry in CASE_ORDER
        ],
        "output_dir": str(OUTPUT_DIR),
        "scientific_overrides_allowed": False,
        "generation_started": False,
    }


def _target_vector_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(_strict_json_bytes(list(values))).hexdigest()


def _audit_workload(root: Path, name: str) -> dict[str, Any]:
    if name not in WORKLOADS:
        raise ValueError(f"未知 workload：{name!r}")
    spec = WORKLOADS[name]
    path = root / spec["path"]
    if _sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"workload {name} 文件 SHA 漂移")
    payload = _load_json(path)
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != 50:
        raise RuntimeError(f"workload {name} 必须包含 50 条查询")
    identity = freeze.query_set_identity(queries)
    if identity != spec["query_identity_sha256"]:
        raise RuntimeError(f"workload {name} query identity 漂移")
    order_counts = freeze._order_counts(queries)
    if order_counts != spec["order_counts"]:
        raise RuntimeError(f"workload {name} 阶数构成漂移")
    targets = []
    for index, query in enumerate(queries):
        result = query.get("result")
        if isinstance(result, bool) or not isinstance(result, int):
            raise TypeError(f"workload {name} result 非整数：index={index}")
        targets.append(result)
    target_identity = _target_vector_sha256(targets)
    if target_identity != spec["target_vector_sha256"]:
        raise RuntimeError(f"workload {name} target vector 漂移")
    return {
        "name": name,
        "path": str(spec["path"]),
        "queries": queries,
        "targets": targets,
        "query_count": len(queries),
        "query_identity_sha256": identity,
        "target_vector_sha256": target_identity,
        "order_counts": order_counts,
        "four_way_query_count": order_counts.get(4, 0),
    }


def _audit_inputs(root: Path) -> dict[str, Any]:
    paths = {
        "schema": SCHEMA_PATH,
        "marginals": MARGINALS_PATH,
        "identity_artifact": freeze.OUTPUT_PATH,
        "workload_a": WORKLOADS["A"]["path"],
        "workload_b": WORKLOADS["B"]["path"],
        "protocol_doc": PROTOCOL_DOC,
    }
    observed = {name: _sha256_file(root / path) for name, path in paths.items()}
    if observed != INPUT_SHA256:
        raise RuntimeError(
            f"collection 输入 SHA 漂移：expected={INPUT_SHA256}, observed={observed}"
        )
    workloads = {name: _audit_workload(root, name) for name in WORKLOADS}
    if workloads["B"]["four_way_query_count"] != 5:
        raise RuntimeError("workload B 的 5 条 4-way 未进入 generation input")
    return {
        "sha256": observed,
        "workloads": {
            name: {
                key: value
                for key, value in audit.items()
                if key not in {"queries", "targets"}
            }
            for name, audit in workloads.items()
        },
    }


def _load_runtime() -> SimpleNamespace:
    import numpy as np
    import pandas as pd

    from table_diffevo.evolution import run_evolution
    from table_diffevo.marginals import load_marginals
    from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
    from table_diffevo.queries import evaluate_table, load_queries
    from table_diffevo.schema import load_schema

    return SimpleNamespace(
        np=np,
        pd=pd,
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
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "":
        raise RuntimeError("本协议要求 CUDA_VISIBLE_DEVICES 为空")
    hostname = platform.node()
    if hostname != EXECUTION_HOSTNAME:
        raise RuntimeError(
            "正式采集服务器身份漂移："
            f"expected={EXECUTION_HOSTNAME}, observed={hostname}"
        )
    return {
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "worktree_clean_including_untracked": True,
        "hostname": hostname,
        "python": sys.version,
        "numpy": runtime.np.__version__,
        "pandas": runtime.pd.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": visible,
        "generator_device": DEVICE,
    }


def _applied_rows(clock: dict[str, Any]) -> int:
    accepted = int(clock["accepted_attempt"])
    if accepted == 0:
        return 0
    return int(clock["attempts"][accepted - 1]["participating_rows"])


def _frame_sha256(frame: Any) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _run_case(
    root: Path,
    shard_dir: Path,
    *,
    workload: str,
    geometry: str,
    seed: int,
    protocol_sha: str,
    git_commit: str,
    runtime: SimpleNamespace,
) -> dict[str, Any]:
    audit = _audit_workload(root, workload)
    spec = WORKLOADS[workload]
    schema = runtime.load_schema(str(root / SCHEMA_PATH))
    queries = runtime.load_queries(str(root / spec["path"]))
    marginals = runtime.load_marginals(str(root / MARGINALS_PATH))
    if freeze.query_set_identity(queries) != audit["query_identity_sha256"]:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} runtime query 漂移")
    target = runtime.np.asarray(audit["targets"], dtype=float)
    if len(queries) != len(target) or len(queries) != 50:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} query/target 不完整")

    params = generator_params(seed, geometry)
    started = time.perf_counter()
    table, diagnostics = runtime.run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=N_RECORDS,
        marginals=marginals,
        device=DEVICE,
        init_method="marginal",
        **params,
    )
    elapsed = time.perf_counter() - started

    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} 输出不是 terminal current")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} terminal identity 漂移")
    observed_params = diagnostics["params"]
    if observed_params["residual_geometry"] != geometry:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} geometry 漂移")
    expected_floor = None if geometry == "absolute" else 8.0
    if observed_params["residual_geometry_floor"] != expected_floor:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} geometry floor 漂移")
    if (
        observed_params["factorized_gibbs_sweeps"] != 0
        or observed_params["factorized_gibbs_use_compiled_workload"] is not False
        or diagnostics["factorized_gibbs_factor_count"] != 0
    ):
        raise RuntimeError("factorized Gibbs 必须关闭，max_order=3 才不会截断 4-way")

    reason = diagnostics["termination_reason"]
    if reason not in {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} 未返回 A/B/C")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != rounds or diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} no-gate clocks 漂移")
    if diagnostics["candidate_evaluation_count"] != rounds:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} candidate count 漂移")
    if diagnostics["state_evaluation_count"] != max(1, rounds):
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} state count 漂移")

    answers = runtime.np.asarray(runtime.evaluate_table(final_table, queries), dtype=float)
    if answers.shape != target.shape or answers.shape != (50,):
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} terminal answer shape 漂移")
    loss = float(runtime.compute_squared_loss(target, answers))
    l1 = float(runtime.compute_normalized_l1(target, answers, N_RECORDS))
    if loss != diagnostics["final_current_squared_loss"]:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} full-query loss 漂移")
    if l1 != diagnostics["final_current_normalized_l1"]:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} full-query L1 漂移")
    metrics_history = diagnostics["current_state_metrics_history"]
    if not metrics_history or metrics_history[-1]["current_squared_loss"] != loss:
        raise RuntimeError(f"{workload}/{geometry}/seed{seed} early-stop loss 漂移")
    work = sum(_applied_rows(clock) for clock in clocks) / N_RECORDS

    case_dir = shard_dir / workload / geometry
    case_dir.mkdir(parents=True)
    table_path = case_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    result = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "git_commit": git_commit,
        "dataset": DATASET,
        "workload": workload,
        "geometry": geometry,
        "seed": seed,
        "device": DEVICE,
        "query_file": str(spec["path"]),
        "query_count": len(queries),
        "query_order_counts": {
            str(order): count for order, count in audit["order_counts"].items()
        },
        "query_identity_sha256": audit["query_identity_sha256"],
        "target_vector_sha256": audit["target_vector_sha256"],
        "measured_four_way_query_count": audit["four_way_query_count"],
        "four_way_queries_in_full_objective_and_early_stop": (
            workload != "B" or audit["four_way_query_count"] == 5
        ),
        "factorized_gibbs_inactive": True,
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
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[seed={seed} {workload}/{geometry}] {reason} rounds={rounds} "
        f"work={work:.4f} L1={l1:.10f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _assert_seed_pairing(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, str]:
    if len(rows) != len(CASE_ORDER):
        raise RuntimeError(f"seed {seed} 不是完整六臂")
    table_hashes = {row["initial_table_sha256"] for row in rows}
    rng_hashes = {
        row["primary_rng_post_initialization_state_sha256"] for row in rows
    }
    if len(table_hashes) != 1 or len(rng_hashes) != 1:
        raise RuntimeError(f"seed {seed} 六臂初始状态未配对")
    return {
        "initial_table_sha256": next(iter(table_hashes)),
        "primary_rng_post_initialization_state_sha256": next(iter(rng_hashes)),
    }


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
        prefix=f".seed_{seed}.tmp-",
        dir=destination.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        results = [
            _run_case(
                root,
                temporary,
                workload=workload,
                geometry=geometry,
                seed=seed,
                protocol_sha=expected,
                git_commit=git_commit,
                runtime=runtime,
            )
            for workload, geometry in CASE_ORDER
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
                {"workload": workload, "geometry": geometry}
                for workload, geometry in CASE_ORDER
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
    cells = {}
    for workload in WORKLOADS:
        cells[workload] = {}
        for geometry in GEOMETRIES:
            cell_rows = [
                row
                for row in rows
                if row["workload"] == workload and row["geometry"] == geometry
            ]
            if len(cell_rows) != len(SEEDS):
                raise RuntimeError(f"{workload}/{geometry} case 数漂移")
            cells[workload][geometry] = {
                "case_count": len(cell_rows),
                "termination_counts": dict(
                    sorted(
                        Counter(
                            row["termination_reason"] for row in cell_rows
                        ).items()
                    )
                ),
                "terminal_normalized_l1_mean": _mean(
                    [row["terminal_current_normalized_l1"] for row in cell_rows]
                ),
                "terminal_normalized_l1_median": _median(
                    [row["terminal_current_normalized_l1"] for row in cell_rows]
                ),
                "terminal_squared_loss_mean": _mean(
                    [row["terminal_current_squared_loss"] for row in cell_rows]
                ),
                "rounds_mean": _mean([row["rounds_run"] for row in cell_rows]),
                "normalized_work_mean": _mean(
                    [row["normalized_work_at_stop"] for row in cell_rows]
                ),
                "elapsed_sec_mean": _mean(
                    [row["elapsed_sec"] for row in cell_rows]
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
        if payload.get("protocol_sha256") != expected:
            raise RuntimeError(f"seed {seed} protocol SHA 漂移")
        if payload.get("shard_index") != index or payload.get("seed") != seed:
            raise RuntimeError(f"seed {seed} shard 身份漂移")
        commits.add(payload["git_commit"])
        seed_rows = payload.get("results")
        if not isinstance(seed_rows, list):
            raise TypeError(f"seed {seed} results 缺失")
        pairing[str(seed)] = _assert_seed_pairing(seed_rows, seed)
        for row in seed_rows:
            if row["seed"] != seed or row["protocol_sha256"] != expected:
                raise RuntimeError(f"seed {seed} case 身份漂移")
            table_path = (
                destination
                / f"seed_{seed}"
                / row["workload"]
                / row["geometry"]
                / "terminal_current.csv"
            )
            if _sha256_file(table_path) != row["terminal_table_sha256"]:
                raise RuntimeError(
                    f"seed {seed}/{row['workload']}/{row['geometry']} table SHA 漂移"
                )
            rows.append(row)

    identities = {
        (row["seed"], row["workload"], row["geometry"]) for row in rows
    }
    expected_identities = {
        (seed, workload, geometry)
        for seed in SEEDS
        for workload, geometry in CASE_ORDER
    }
    if len(rows) != 30 or identities != expected_identities:
        raise RuntimeError("30-case 矩阵不完整或重复")
    if len(commits) != 1:
        raise RuntimeError(f"shard Git commit 不一致：{sorted(commits)}")

    workload_rank = {name: index for index, name in enumerate(WORKLOADS)}
    geometry_rank = {name: index for index, name in enumerate(GEOMETRIES)}
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
                geometry_rank[row["geometry"]],
                workload_rank[row["workload"]],
            ),
        ),
        "initial_state_pairing_by_seed": pairing,
        "summary": _summarize(rows),
        "claim_scope": "query_workload_ab_raw_collection_before_common_evaluation",
        "canonical_selection_allowed": False,
        "parameter_retuning_performed": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
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
        "--shard-index",
        required=True,
        type=int,
        choices=range(len(SEEDS)),
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
        print(f"query-workload A/B shard -> {path}")
        return
    path = aggregate(args.confirm_protocol_sha)
    print(f"query-workload A/B collection -> {path}")
    print(f"collection SHA-256 -> {_sha256_file(path)}")


if __name__ == "__main__":
    main()
