#!/usr/bin/env python3
"""Run the post-screen test-workload coverage diagnostic.

This diagnostic augments workload B with the 25 fixed one-way queries from
workload A.  It is intentionally separate from the main development screen:
the result is descriptive evidence about workload coverage, not a promotion
or formal multi-seed experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts import run_fitness_only_attribution as base
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_paired_fitness_only_attribution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.quality import (
    evaluate_quality_snapshot,
    query_error_metrics,
    query_fingerprint,
    validate_query_partition,
)
from table_diffevo.queries import evaluate_table, load_data, load_queries
from table_diffevo.schema import load_schema


PROTOCOL_VERSION = "fitness-only-test-coverage-diagnostic-v1"
OUTPUT_DIR = Path(
    "outputs/fitness_only_test_coverage_diagnostic_seed9908_v1"
)
FROZEN_PROTOCOL_SHA256 = (
    "73cdb95e1d5c098ce41dd132343a250a564a6bb48e26a653242dd78183b2925c"
)
SEED = 9908
N_ROUNDS = 3000
ARMS = ("residual", "equal")

SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
WORKLOAD_A_PATH = Path("configs/test_300x10/measured_50query.json")
WORKLOAD_B_PATH = Path(
    "configs/test_300x10/measured_50query_30_15_5.json"
)
HELDOUT_PATH = Path("configs/test_300x10/heldout_issue53_v1.json")
REFERENCE_PATH = Path("data/test_300x10/test_300x10.csv")
N_RECORDS = 300

INPUT_SHA256 = {
    "schema": (
        "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
    ),
    "marginals": (
        "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
    ),
    "workload_a": (
        "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88"
    ),
    "workload_b": (
        "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
    ),
    "heldout": (
        "300bffea1f3d9105ad8f1840d50a900616115659065efec35b3c02f7a38cc1e0"
    ),
    "reference": (
        "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
    ),
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象: {path}")
    return value


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "post_screen_workload_coverage_diagnostic_only",
        "seed": SEED,
        "n_rounds": N_ROUNDS,
        "arms": list(ARMS),
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "generation_input": {
            "composition": "workload_B_50_plus_workload_A_one_way_25",
            "combined_query_count": 75,
            "workload_b_query_count": 50,
            "one_way_query_count": 25,
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "workload_a": str(WORKLOAD_A_PATH),
            "workload_b": str(WORKLOAD_B_PATH),
            "input_sha256": {
                key: INPUT_SHA256[key]
                for key in ("schema", "marginals", "workload_a", "workload_b")
            },
        },
        "shared_generation_config": {
            "init_method": "marginal",
            "eval_method": "vectorized",
            "rho": 0.01,
            "eta": 0.5,
            "mu": 0.01,
            "distance_mode": "geometric",
            "selection_scale_invariant": True,
            "selection_scale_invariant_min_spread": 1e-3,
            "alpha_schedule_mode": "fixed",
            "fixed_alpha": 16.0,
            "lambda_param": 0.5,
            "delta": 0.05,
            "winsorize_quantiles": [0.01, 0.99],
            "residual_geometry": "relative",
            "residual_geometry_floor": 8.0,
            "exclude_self": True,
            "tol": "positive_infinity",
            "max_retries": 0,
            "residual_directed_diffusion": False,
            "factorized_gibbs_sweeps": 0,
            "gap_l1_sweeps": 0,
            "candidate_budget": None,
            "residual_self_cooling": None,
            "rho_anneal_end": None,
            "stop_on_exact_residual": False,
            "inner_early_stopping_patience_ticks": None,
            "horizon_invariant": True,
        },
        "pairing": {
            "same_initial_table": True,
            "same_random_addresses": True,
            "only_treatment_difference": "row_fitness",
        },
        "phase_boundary": {
            "heldout_and_reference_loaded_after_generation": True,
            "quality_used_online": False,
            "parameter_retuning_allowed": False,
            "output_overwrite_allowed": False,
        },
        "interpretation": {
            "diagnostic_only": True,
            "formal_claim_allowed": False,
            "question": "does adding fixed one-way coverage remove test-B degradation?",
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "coverage diagnostic 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": _protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "trajectory_count": 2,
        "generation_started": False,
    }


def _query_payload(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(path)
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"queries 缺失: {path}")
    return queries


def _audit_generation_inputs(root: Path) -> dict[str, Any]:
    observed = {
        "schema": _sha256_file(root / SCHEMA_PATH),
        "marginals": _sha256_file(root / MARGINALS_PATH),
        "workload_a": _sha256_file(root / WORKLOAD_A_PATH),
        "workload_b": _sha256_file(root / WORKLOAD_B_PATH),
    }
    expected = {key: INPUT_SHA256[key] for key in observed}
    if observed != expected:
        raise RuntimeError(f"generation 输入 SHA 漂移: expected={expected}, observed={observed}")
    workload_a = _query_payload(root / WORKLOAD_A_PATH)
    workload_b = _query_payload(root / WORKLOAD_B_PATH)
    one_way = [
        query for query in workload_a
        if len(query.get("conditions", [])) == 1
    ]
    if len(workload_b) != 50 or len(one_way) != 25:
        raise RuntimeError("B 或 one-way 查询数量漂移")
    combined = list(workload_b) + list(one_way)
    fingerprints = [query_fingerprint(query) for query in combined]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("B+one-way 组合包含重复语义查询")
    targets = []
    for query in combined:
        result = query.get("result")
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError("生成 target 必须是数值")
        targets.append(float(result))
    return {
        "workload_b_queries": workload_b,
        "one_way_queries": one_way,
        "queries": combined,
        "targets": targets,
        "workload_b_count": len(workload_b),
        "one_way_count": len(one_way),
        "combined_count": len(combined),
        "combined_query_identity_sha256": _sha256_bytes(
            "\n".join(fingerprints).encode("ascii")
        ),
        "input_sha256": observed,
    }


def _config() -> FitnessOnlyConfig:
    return FitnessOnlyConfig(
        n_rounds=N_ROUNDS,
        seed=SEED,
        device="numpy",
        eval_method="vectorized",
        batch_size=256,
        init_method="marginal",
        log_every=100,
        rho=0.01,
        eta=0.5,
        mu=0.01,
        lambda_param=0.5,
        fixed_alpha=16.0,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        selection_scale_invariant_min_spread=1e-3,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
        exclude_self=True,
        record_transition_clocks=False,
    )


def _run_generation(
    root: Path,
    staging: Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    schema = load_schema(str(root / SCHEMA_PATH))
    marginals = load_marginals(str(root / MARGINALS_PATH))
    started = time.perf_counter()
    runs, pairing = run_paired_fitness_only_attribution(
        np.asarray(audit["targets"], dtype=float),
        list(audit["queries"]),
        schema,
        N_RECORDS,
        config=_config(),
        marginals=marginals,
    )
    elapsed = time.perf_counter() - started
    generation_dir = staging / "generation"
    generation_dir.mkdir(parents=True, exist_ok=False)
    base._write_json(generation_dir / "pairing.json", pairing)
    arm_results = {}
    for arm in ARMS:
        table, diagnostics = runs[arm]
        table = table.reset_index(drop=True)
        table.to_csv(generation_dir / f"{arm}_terminal_current.csv", index=False)
        base._write_json(
            generation_dir / f"{arm}_diagnostics.json",
            diagnostics,
        )
        arm_results[arm] = {
            "terminal_table_sha256": base._frame_sha256(table),
            "rounds_run": int(diagnostics["rounds_run"]),
            "candidate_evaluation_count": int(
                diagnostics["candidate_evaluation_count"]
            ),
            "termination_reason": diagnostics["termination_reason"],
            "output_table_identity": diagnostics["output_table_identity"],
            "final_current_squared_loss": float(
                diagnostics["final_current_squared_loss"]
            ),
            "final_current_normalized_l1": float(
                diagnostics["final_current_normalized_l1"]
            ),
            "best_loss_diagnostic_only": float(
                diagnostics["best_loss_diagnostic_only"]
            ),
            "elapsed_sec": float(diagnostics["elapsed_sec"]),
        }
    return {
        "dataset": "test_300x10",
        "seed": SEED,
        "elapsed_sec_wall": float(elapsed),
        "pairing": pairing,
        "arms": arm_results,
    }


def _load_heldout_and_reference(root: Path) -> tuple[list[dict[str, Any]], list[float], pd.DataFrame]:
    if _sha256_file(root / HELDOUT_PATH) != INPUT_SHA256["heldout"]:
        raise RuntimeError("heldout SHA 漂移")
    if _sha256_file(root / REFERENCE_PATH) != INPUT_SHA256["reference"]:
        raise RuntimeError("reference SHA 漂移")
    payload = _load_json_object(root / HELDOUT_PATH)
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != 1024:
        raise RuntimeError("heldout 查询数量漂移")
    if [len(q.get("conditions", [])) for q in queries].count(3) != 512:
        raise RuntimeError("heldout 3-way 数量漂移")
    if [len(q.get("conditions", [])) for q in queries].count(4) != 512:
        raise RuntimeError("heldout 4-way 数量漂移")
    targets = [float(query["result"]) for query in queries]
    return queries, targets, load_data(str(root / REFERENCE_PATH))


def _evaluate_arm(
    root: Path,
    staging: Path,
    arm: str,
    audit: Mapping[str, Any],
    generation_result: Mapping[str, Any],
    heldout_queries: Sequence[dict[str, Any]],
    heldout_targets: Sequence[float],
    reference: pd.DataFrame,
) -> dict[str, Any]:
    path = staging / "generation" / f"{arm}_terminal_current.csv"
    table = pd.read_csv(path)
    schema = load_schema(str(root / SCHEMA_PATH))
    if base._frame_sha256(table) != generation_result["terminal_table_sha256"]:
        raise RuntimeError(f"{arm} terminal table SHA 漂移")
    validate_query_partition(list(audit["queries"]), heldout_queries)
    validate_query_partition(
        list(audit["workload_b_queries"]),
        list(audit["one_way_queries"]),
    )
    # The one-way subset is deliberately part of generation input; this call
    # documents its separate safety report without introducing a new channel.
    answers = np.asarray(evaluate_table(table, list(audit["queries"])), dtype=float)
    heldout_answers = np.asarray(evaluate_table(table, list(heldout_queries)), dtype=float)
    one_way_count = int(audit["workload_b_count"])
    workload_b_answers = answers[:one_way_count]
    one_way_answers = answers[one_way_count:]
    quality = evaluate_quality_snapshot(
        table,
        schema,
        list(audit["queries"]),
        np.asarray(audit["targets"], dtype=float),
        list(heldout_queries),
        heldout_targets,
        reference_table=reference,
    )
    quality["workload_b_measured"] = query_error_metrics(
        np.asarray(audit["targets"][:one_way_count], dtype=float),
        workload_b_answers,
        N_RECORDS,
    )
    quality["one_way_safety"] = query_error_metrics(
        np.asarray(audit["targets"][one_way_count:], dtype=float),
        one_way_answers,
        N_RECORDS,
    )
    quality["evaluation_inputs"] = {
        "generation_query_count": int(audit["combined_count"]),
        "workload_b_query_count": int(audit["workload_b_count"]),
        "one_way_query_count": int(audit["one_way_count"]),
        "heldout_sha256": INPUT_SHA256["heldout"],
        "reference_sha256": INPUT_SHA256["reference"],
        "reference_loaded_after_generation": True,
    }
    quality_dir = staging / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    base._write_json(quality_dir / f"{arm}.json", quality)
    return quality


def _comparison(quality: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    residual = quality["residual"]
    equal = quality["equal"]
    paths = {
        "combined_measured": ("measured", "normalized_l1_mean"),
        "workload_b_measured": ("workload_b_measured", "normalized_l1_mean"),
        "one_way_safety": ("one_way_safety", "normalized_l1_mean"),
        "heldout_3way": ("heldout", "3way", "normalized_l1_mean"),
        "heldout_4way": ("heldout", "4way", "normalized_l1_mean"),
    }
    deltas = {}
    for name, path in paths.items():
        left: Any = residual
        right: Any = equal
        for key in path:
            left = left[key]
            right = right[key]
        deltas[name] = {
            "residual_minus_equal": float(left - right),
            "residual": float(left),
            "equal": float(right),
        }
    return {
        "delta_residual_minus_equal": deltas,
        "interpretation": "diagnostic_only_lower_is_better_no_promotion_gate",
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("coverage diagnostic protocol SHA-256 确认值不一致")
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = _audit_generation_inputs(root)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        generation = _run_generation(root, staging, audit)
        heldout_queries, heldout_targets, reference = _load_heldout_and_reference(root)
        quality: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            quality[arm] = _evaluate_arm(
                root,
                staging,
                arm,
                audit,
                generation["arms"][arm],
                heldout_queries,
                heldout_targets,
                reference,
            )
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "generation_inputs": {
                key: value
                for key, value in audit.items()
                if key not in {"queries", "targets", "workload_b_queries", "one_way_queries"}
            },
            "generation": generation,
            "quality": quality,
            "paired_quality_comparison": _comparison(quality),
            "summary": {
                "diagnostic_only": True,
                "formal_claim_allowed": False,
                "all_terminal_current": True,
                "all_fixed_rounds": all(
                    generation["arms"][arm]["rounds_run"] == N_ROUNDS
                    for arm in ARMS
                ),
                "all_unconditional": True,
                "parameter_retuning_performed": False,
                "heldout_and_reference_loaded_after_generation": True,
            },
            "completed_at_unix": time.time(),
        }
        base._write_json(staging / "report.json", report)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "report.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    print(f"coverage diagnostic report -> {run(args.confirm_protocol_sha)}")


if __name__ == "__main__":
    main()
