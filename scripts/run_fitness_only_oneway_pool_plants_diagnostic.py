#!/usr/bin/env python3
"""One-way-in-pool diagnostic on plants under the frozen v3 configuration.

PGM baseline calibration exposed a 70x one-way gap on plants: the one-way
marginals are only used at initialisation, the fitness never sees them, and
evolution sacrifices them.  This diagnostic adds every one-way cell to the
evolution query pool as an ordinary query (fitness and residual guidance see
them; zero mechanism change, zero new hyper-parameters) and reruns the paired
residual/equal arms under the byte-identical v3 frozen configuration.
Evaluation grouping is unchanged (measured/heldout/one-way safety), so all
numbers remain comparable to the frozen plants diagnostic report and the PGM
baseline report, both pinned by SHA-256.  Diagnostic only: single development
seed, no gate, no formal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from table_diffevo.queries import evaluate_table, load_data
from table_diffevo.schema import load_schema


PROTOCOL_VERSION = "fitness-only-oneway-pool-plants-diagnostic-v1"
OUTPUT_DIR = Path("outputs/fitness_only_oneway_pool_plants_seed9908_v1")
FROZEN_PROTOCOL_SHA256 = (
    "11b96b3ce84a7715de6a2c44f6d40e41188fb46ede2462aed5a69511f7746457"
)
SEED = 9908
N_ROUNDS = 6000
RHO_BASE = 0.01
RHO_ANNEAL_START_ROUND = 900
RHO_ANNEAL_ROUNDS = 600
RHO_ANNEAL_END = 0.001
ARMS = ("residual", "equal")

DATASET = "plants"
N_RECORDS = 17412
EXPECTED_QUERY_COUNT = 980
EXPECTED_ONE_WAY_COUNT = 138  # 69 binary attributes x 2 values
EXPECTED_POOL_COUNT = EXPECTED_QUERY_COUNT + EXPECTED_ONE_WAY_COUNT
SCHEMA_PATH = Path("configs/plants/schema.yaml")
MARGINALS_PATH = Path("configs/plants/init_marginals.json")
MEASURED_PATH = Path("configs/plants/measured_1000query.json")
HELDOUT_PATH = Path("configs/plants/heldout_issue53_v1.json")
REFERENCE_PATH = Path("data/plants/plants.csv")
FROZEN_REPORT_PATH = Path(
    "outputs/fitness_only_plants_diagnostic_seed9908_v1/report.json"
)
PGM_REPORT_PATH = Path("outputs/baseline_pgm_plants_v1/report.json")

INPUT_SHA256 = {
    "schema": (
        "7eaf404bfe6d3a8824a5d29cebacac704a0fae5fc5cb2227a275a35a989e0a9b"
    ),
    "marginals": (
        "4e302b18e1e6e34871a2651378ec357b6fd35edcccfc42385a0b773b1d701535"
    ),
    "measured": (
        "f93c2d9717e2f4f87536e703019f5b48230657db227656d1af62a8640babf649"
    ),
    "heldout": (
        "d65401761bade19ad40d9588eaa57609de0bdbc3c26835a5343ab8cc332eca85"
    ),
    "reference": (
        "c7b5cf1e2230df3facf8d4b5d4a077d747a4f84e2dd47ce8779335f553599532"
    ),
    "frozen_report": (
        "26a8934270e9042d05e90d0a77ba1859aebd34a48ca1082291dbee9330d95565"
    ),
    "pgm_report": (
        "fbe0c88d793fde6ed3e6b919933b7bedf2d239cb6ea606f3865eadad9f026105"
    ),
}

FLOOR_SEGMENT_ROUNDS = 500  # observation-only segment means over the floor.


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
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 顶层必须是对象")
    return payload


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "one_way_in_pool_hypothesis_diagnostic_only",
        "hypothesis": (
            "adding every one-way cell to the evolution query pool as an"
            " ordinary query lets the residual mechanism hold first-order"
            " marginals automatically (silent when held, self-repairing when"
            " violated), improving one-way and held-out quality on plants"
            " under the unchanged v3 budget"
        ),
        "seed": SEED,
        "n_rounds": N_ROUNDS,
        "arms": list(ARMS),
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "measured": str(MEASURED_PATH),
            "expected_measured_count": EXPECTED_QUERY_COUNT,
            "expected_one_way_count": EXPECTED_ONE_WAY_COUNT,
            "expected_pool_count": EXPECTED_POOL_COUNT,
            "workload_composition": "double_460_plus_triple_520_v2",
            "one_way_role": (
                "in_evolution_pool_as_ordinary_queries_and_in_safety_metric"
            ),
            "one_way_pool_construction": (
                "init_marginals_attributes_order_each_value_one_cell_query"
                "_target_frozen_count"
            ),
            "input_sha256": {
                key: INPUT_SHA256[key]
                for key in ("schema", "marginals", "measured")
            },
        },
        "shared_generation_config": {
            "init_method": "marginal",
            "eval_method": "vectorized",
            "device": "cuda",
            "rho": RHO_BASE,
            "rho_anneal_start_round": RHO_ANNEAL_START_ROUND,
            "rho_anneal_rounds": RHO_ANNEAL_ROUNDS,
            "rho_anneal_end": RHO_ANNEAL_END,
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
            "eta_anneal": None,
            "mw_query_weights": None,
            "configuration_source": (
                "v3_frozen_configuration_verbatim_no_retuning"
            ),
        },
        "pairing": {
            "same_initial_table": True,
            "same_random_addresses": True,
            "only_treatment_difference": "row_fitness",
        },
        "evaluation_grouping": {
            "measured_metric_uses_only_original_measured_queries": True,
            "one_way_safety_same_construction_as_frozen_reports": True,
            "heldout_frozen_untouched": True,
            "one_way_now_in_evolution_pool": True,
        },
        "comparison_reference": {
            "frozen_report": str(FROZEN_REPORT_PATH),
            "frozen_report_sha256": INPUT_SHA256["frozen_report"],
            "pgm_report": str(PGM_REPORT_PATH),
            "pgm_report_sha256": INPUT_SHA256["pgm_report"],
            "rerun_frozen_arms": False,
            "equal_arm_identity_expectation": (
                "equal fitness is constant zero, so the equal trajectory"
                " should be byte-identical to the frozen equal arm; recorded"
                " as equal_table_identity_match, observational only"
            ),
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
            "question": (
                "with one-way cells in the evolution pool under the unchanged"
                " v3 budget, how much of the plants one-way and held-out gap"
                " to the PGM baseline is recovered, and is the measured"
                " workload crowded out?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "oneway-pool plants diagnostic 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": _protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "trajectory_count": len(ARMS),
        "generation_started": False,
    }


def _one_way_pool_queries(
    root: Path,
) -> tuple[list[dict[str, Any]], list[float]]:
    marginals = load_marginals(str(root / MARGINALS_PATH))
    queries: list[dict[str, Any]] = []
    targets: list[float] = []
    index = 0
    for attribute, attribute_spec in marginals["attributes"].items():
        values = attribute_spec.get("values")
        counts = attribute_spec.get("counts")
        if not isinstance(values, list) or not isinstance(counts, list):
            raise RuntimeError("marginal one-way 数据格式错误")
        if len(values) != len(counts):
            raise RuntimeError("marginal values/counts 数量不一致")
        if abs(sum(float(c) for c in counts) - N_RECORDS) > 1e-9:
            raise RuntimeError(f"{attribute} counts 总和不等于 N")
        for value, count in zip(values, counts):
            count = float(count)
            if not math.isfinite(count) or count < 0:
                raise ValueError(f"{attribute}={value} 计数非法")
            index += 1
            queries.append({
                "id": f"OW{index:04d}",
                "type": "single",
                "expression": f"{attribute} == {value}",
                "conditions": [{
                    "attribute": attribute,
                    "operator": "==",
                    "value": value,
                }],
                "result": count,
            })
            targets.append(count)
    if len(queries) != EXPECTED_ONE_WAY_COUNT:
        raise RuntimeError(f"one-way 池数量错误: {len(queries)}")
    return queries, targets


def _audit_generation_inputs(root: Path) -> dict[str, Any]:
    observed = {
        "schema": _sha256_file(root / SCHEMA_PATH),
        "marginals": _sha256_file(root / MARGINALS_PATH),
        "measured": _sha256_file(root / MEASURED_PATH),
    }
    expected = {key: INPUT_SHA256[key] for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"generation 输入 SHA 漂移: expected={expected}, "
            f"observed={observed}"
        )
    payload = _load_json_object(root / MEASURED_PATH)
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != EXPECTED_QUERY_COUNT:
        raise RuntimeError("plants measured 查询数量漂移")
    if int(payload.get("record_count", -1)) != N_RECORDS:
        raise RuntimeError("plants measured record_count 漂移")
    measured_fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(measured_fingerprints)) != len(measured_fingerprints):
        raise RuntimeError("plants measured 查询存在重复语义")
    orders = [len(query.get("conditions", [])) for query in queries]
    if min(orders) < 2:
        raise RuntimeError("plants measured 不应包含 one-way 查询")
    targets = []
    for index, query in enumerate(queries):
        result = query.get("result")
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, float))
            or not math.isfinite(float(result))
            or result < 0
        ):
            raise ValueError(f"plants measured target 非法: index={index}")
        targets.append(float(result))

    one_way_queries, one_way_targets = _one_way_pool_queries(root)
    one_way_fingerprints = [
        query_fingerprint(query) for query in one_way_queries
    ]
    if len(set(one_way_fingerprints)) != len(one_way_fingerprints):
        raise RuntimeError("one-way 池存在重复语义")
    if set(measured_fingerprints) & set(one_way_fingerprints):
        raise RuntimeError("one-way 池与 measured 指纹相交")
    if any(
        len(query["conditions"]) != 1 for query in one_way_queries
    ):
        raise RuntimeError("one-way 池必须全部为单条件查询")

    pool_queries = list(queries) + list(one_way_queries)
    pool_targets = list(targets) + list(one_way_targets)
    if len(pool_queries) != EXPECTED_POOL_COUNT:
        raise RuntimeError(f"演化池数量错误: {len(pool_queries)}")
    pool_fingerprints = measured_fingerprints + one_way_fingerprints
    return {
        "queries": queries,
        "targets": targets,
        "query_count": len(queries),
        "one_way_queries": one_way_queries,
        "one_way_targets": one_way_targets,
        "one_way_count": len(one_way_queries),
        "pool_queries": pool_queries,
        "pool_targets": pool_targets,
        "pool_query_count": len(pool_queries),
        "query_identity_sha256": _sha256_bytes(
            "\n".join(measured_fingerprints).encode("ascii")
        ),
        "target_vector_sha256": _sha256_bytes(_strict_json_bytes(targets)),
        "pool_identity_sha256": _sha256_bytes(
            "\n".join(pool_fingerprints).encode("ascii")
        ),
        "pool_target_vector_sha256": _sha256_bytes(
            _strict_json_bytes(pool_targets)
        ),
        "input_sha256": observed,
    }


def _config() -> FitnessOnlyConfig:
    return FitnessOnlyConfig(
        n_rounds=N_ROUNDS,
        seed=SEED,
        device="cuda",
        eval_method="vectorized",
        batch_size=256,
        init_method="marginal",
        log_every=100,
        rho=RHO_BASE,
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
        rho_anneal_start_round=RHO_ANNEAL_START_ROUND,
        rho_anneal_rounds=RHO_ANNEAL_ROUNDS,
        rho_anneal_end=RHO_ANNEAL_END,
    )


def _expected_rho_schedule(n_rounds: int) -> list[float]:
    values = []
    for t in range(n_rounds):
        progress = min(
            1.0,
            max(
                0.0,
                (t - RHO_ANNEAL_START_ROUND) / RHO_ANNEAL_ROUNDS,
            ),
        )
        values.append(RHO_BASE * (RHO_ANNEAL_END / RHO_BASE) ** progress)
    return values


def _audit_rho_schedule(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    observed = [float(x) for x in diagnostics["rho_schedule_history"]]
    expected = _expected_rho_schedule(N_ROUNDS)
    if len(observed) != len(expected):
        raise RuntimeError("rho schedule 长度漂移")
    max_abs = max(
        abs(a - b) for a, b in zip(observed, expected)
    )
    if max_abs > 1e-12:
        raise RuntimeError(f"rho schedule 数值漂移: max_abs={max_abs}")
    return {
        "length": len(observed),
        "max_abs_deviation": max_abs,
        "verdict": "pass",
    }


def _floor_morphology(loss_history: Sequence[float]) -> dict[str, Any]:
    floor_start = RHO_ANNEAL_START_ROUND + RHO_ANNEAL_ROUNDS
    floor = [float(x) for x in loss_history[floor_start:]]
    segments = []
    for begin in range(0, len(floor), FLOOR_SEGMENT_ROUNDS):
        chunk = floor[begin:begin + FLOOR_SEGMENT_ROUNDS]
        if chunk:
            segments.append(float(np.mean(chunk)))
    best_index = int(np.argmin(loss_history))
    return {
        "floor_start_round": floor_start,
        "segment_rounds": FLOOR_SEGMENT_ROUNDS,
        "segment_means": segments,
        "best_round_index": best_index,
        "last_round_index": len(loss_history) - 1,
    }


def _run_generation(
    root: Path,
    staging: Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    schema = load_schema(str(root / SCHEMA_PATH))
    marginals = load_marginals(str(root / MARGINALS_PATH))
    started = time.perf_counter()
    runs, pairing = run_paired_fitness_only_attribution(
        np.asarray(audit["pool_targets"], dtype=float),
        list(audit["pool_queries"]),
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
        table.to_csv(
            generation_dir / f"{arm}_terminal_current.csv",
            index=False,
        )
        base._write_json(
            generation_dir / f"{arm}_diagnostics.json",
            diagnostics,
        )
        best_loss = float(diagnostics["best_loss_diagnostic_only"])
        final_loss = float(diagnostics["final_current_squared_loss"])
        arm_results[arm] = {
            "terminal_table_sha256": base._frame_sha256(table),
            "rounds_run": int(diagnostics["rounds_run"]),
            "candidate_evaluation_count": int(
                diagnostics["candidate_evaluation_count"]
            ),
            "termination_reason": diagnostics["termination_reason"],
            "output_table_identity": diagnostics["output_table_identity"],
            "final_current_squared_loss": final_loss,
            "final_current_normalized_l1": float(
                diagnostics["final_current_normalized_l1"]
            ),
            "best_loss_diagnostic_only": best_loss,
            "drift_ratio_final_over_best_observation_only": (
                final_loss / best_loss if best_loss > 0 else None
            ),
            "rho_schedule_audit": _audit_rho_schedule(diagnostics),
            "floor_morphology_observation_only": _floor_morphology(
                diagnostics["loss_history"]
            ),
            "elapsed_sec": float(diagnostics["elapsed_sec"]),
        }
    return {
        "dataset": DATASET,
        "seed": SEED,
        "device": "cuda",
        "n_records": N_RECORDS,
        "measured_query_count": int(audit["query_count"]),
        "one_way_count": int(audit["one_way_count"]),
        "pool_query_count": int(audit["pool_query_count"]),
        "elapsed_sec_wall": float(elapsed),
        "pairing": pairing,
        "arms": arm_results,
    }


def _load_heldout_and_reference(
    root: Path,
) -> tuple[list[dict[str, Any]], list[float], pd.DataFrame]:
    if _sha256_file(root / HELDOUT_PATH) != INPUT_SHA256["heldout"]:
        raise RuntimeError("heldout SHA 漂移")
    if _sha256_file(root / REFERENCE_PATH) != INPUT_SHA256["reference"]:
        raise RuntimeError("reference SHA 漂移")
    payload = _load_json_object(root / HELDOUT_PATH)
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != 1024:
        raise RuntimeError("heldout 查询数量漂移")
    orders = [len(query.get("conditions", [])) for query in queries]
    if orders.count(3) != 512 or orders.count(4) != 512:
        raise RuntimeError("heldout 3/4-way 分组漂移")
    targets = []
    for query in queries:
        result = query.get("result")
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError("heldout target 非法")
        targets.append(float(result))
    return queries, targets, load_data(str(root / REFERENCE_PATH))


def _one_way_safety_queries(
    root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    marginals = load_marginals(str(root / MARGINALS_PATH))
    queries = []
    targets = []
    for attribute, attribute_spec in marginals["attributes"].items():
        values = attribute_spec.get("values")
        counts = attribute_spec.get("counts")
        if not isinstance(values, list) or not isinstance(counts, list):
            raise RuntimeError("plants marginal one-way 数据格式错误")
        if len(values) != len(counts):
            raise RuntimeError("plants marginal values/counts 数量不一致")
        for value, count in zip(values, counts):
            queries.append({
                "conditions": [{
                    "attribute": attribute,
                    "operator": "==",
                    "value": value,
                }],
            })
            targets.append(float(count))
    if len(queries) != EXPECTED_ONE_WAY_COUNT:
        raise RuntimeError(
            f"plants one-way safety 数量错误: {len(queries)}"
        )
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("plants one-way safety 存在重复语义")
    return queries, np.asarray(targets, dtype=float)


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
    if len(table) != N_RECORDS:
        raise RuntimeError(f"{arm} terminal row count 漂移")

    measured_queries = list(audit["queries"])
    measured_targets = np.asarray(audit["targets"], dtype=float)
    validate_query_partition(measured_queries, list(heldout_queries))
    safety_queries, safety_targets = _one_way_safety_queries(root)
    validate_query_partition(measured_queries, safety_queries)
    validate_query_partition(list(heldout_queries), safety_queries)

    measured_answers = np.asarray(
        evaluate_table(table, measured_queries), dtype=float
    )
    heldout_answers = np.asarray(
        evaluate_table(table, list(heldout_queries)), dtype=float
    )
    safety_answers = np.asarray(
        evaluate_table(table, safety_queries), dtype=float
    )
    quality = evaluate_quality_snapshot(
        table,
        schema,
        measured_queries,
        measured_targets,
        list(heldout_queries),
        list(heldout_targets),
        reference_table=reference,
    )
    quality["measured_by_order_and_target_bucket"] = (
        base._grouped_error_metrics(
            measured_queries,
            measured_targets,
            measured_answers,
            N_RECORDS,
        )
    )
    quality["heldout_by_order_and_target_bucket"] = (
        base._grouped_error_metrics(
            list(heldout_queries),
            np.asarray(heldout_targets, dtype=float),
            heldout_answers,
            N_RECORDS,
        )
    )
    safety_metrics = query_error_metrics(
        safety_targets,
        safety_answers,
        N_RECORDS,
    )
    abs_errors = np.abs(safety_targets - safety_answers)
    safety_metrics.update({
        "absolute_error_mean": float(np.mean(abs_errors)),
        "absolute_error_median": float(np.median(abs_errors)),
        "absolute_error_p90": float(np.percentile(abs_errors, 90)),
        "absolute_error_max": float(np.max(abs_errors)),
    })
    quality["one_way_safety"] = safety_metrics
    quality["one_way_safety_by_target_bucket"] = (
        base._grouped_error_metrics(
            safety_queries,
            safety_targets,
            safety_answers,
            N_RECORDS,
        )
    )
    quality["evaluation_inputs"] = {
        "measured_sha256": INPUT_SHA256["measured"],
        "heldout_sha256": INPUT_SHA256["heldout"],
        "reference_sha256": INPUT_SHA256["reference"],
        "one_way_safety_source": "marginal_values_and_counts",
        "one_way_safety_query_count": EXPECTED_ONE_WAY_COUNT,
        "one_way_now_in_evolution_pool": True,
        "measured_metric_uses_only_original_measured_queries": True,
        "reference_loaded_after_generation": True,
    }
    quality_dir = staging / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    base._write_json(quality_dir / f"{arm}.json", quality)
    return quality


_METRIC_PATHS = {
    "measured": ("measured",),
    "heldout_3way": ("heldout", "3way"),
    "heldout_4way": ("heldout", "4way"),
    "heldout_combined": ("heldout", "combined"),
    "one_way_safety": (
        "one_way_safety_by_target_bucket", "by_order", "1way"
    ),
}
_STATS = ("mean", "median", "p90", "max")


def _metric_snapshot(quality: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, path in _METRIC_PATHS.items():
        node: Any = quality
        for key in path:
            node = node[key]
        stats = {}
        for stat in _STATS:
            value = float(node[f"normalized_l1_{stat}"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} {stat} 非法: {value}")
            stats[stat] = value
        snapshot[name] = stats
    return snapshot


def _load_frozen_references(root: Path) -> dict[str, Any]:
    frozen_path = root / FROZEN_REPORT_PATH
    pgm_path = root / PGM_REPORT_PATH
    if _sha256_file(frozen_path) != INPUT_SHA256["frozen_report"]:
        raise RuntimeError("冻结对照报告 SHA 漂移")
    if _sha256_file(pgm_path) != INPUT_SHA256["pgm_report"]:
        raise RuntimeError("PGM 对照报告 SHA 漂移")
    frozen = _load_json_object(frozen_path)
    pgm = _load_json_object(pgm_path)
    arms = {
        arm: _metric_snapshot(frozen["quality"][arm]) for arm in ARMS
    }
    return {
        "frozen_report_sha256": INPUT_SHA256["frozen_report"],
        "pgm_report_sha256": INPUT_SHA256["pgm_report"],
        "frozen_arms": arms,
        "frozen_equal_terminal_table_sha256": (
            frozen["generation"]["arms"]["equal"]["terminal_table_sha256"]
        ),
        "pgm": _metric_snapshot(pgm["quality"]["pgm"]),
    }


def _comparison(
    quality: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    new_snapshots = {arm: _metric_snapshot(quality[arm]) for arm in ARMS}
    frozen_arms = references["frozen_arms"]
    pgm = references["pgm"]

    def deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
        return {
            name: {
                "delta_mean": float(
                    left[name]["mean"] - right[name]["mean"]
                ),
                "delta_median": float(
                    left[name]["median"] - right[name]["median"]
                ),
                "left_mean": float(left[name]["mean"]),
                "right_mean": float(right[name]["mean"]),
            }
            for name in _METRIC_PATHS
        }

    equal_match = (
        generation["arms"]["equal"]["terminal_table_sha256"]
        == references["frozen_equal_terminal_table_sha256"]
    )
    return {
        "new_snapshots": new_snapshots,
        "internal_residual_minus_equal": deltas(
            new_snapshots["residual"], new_snapshots["equal"]
        ),
        "new_residual_minus_frozen_residual": deltas(
            new_snapshots["residual"], frozen_arms["residual"]
        ),
        "new_equal_minus_frozen_equal": deltas(
            new_snapshots["equal"], frozen_arms["equal"]
        ),
        "new_residual_minus_pgm": deltas(new_snapshots["residual"], pgm),
        "equal_table_identity_match": bool(equal_match),
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "oneway-pool plants diagnostic protocol SHA-256 确认值不一致"
        )
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
        heldout_queries, heldout_targets, reference = (
            _load_heldout_and_reference(root)
        )
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
        references = _load_frozen_references(root)
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "generation_inputs": {
                key: value
                for key, value in audit.items()
                if key not in {
                    "queries",
                    "targets",
                    "one_way_queries",
                    "one_way_targets",
                    "pool_queries",
                    "pool_targets",
                }
            },
            "generation": generation,
            "quality": quality,
            "reference_report": references,
            "comparison": _comparison(quality, references, generation),
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
    print(
        "oneway-pool plants diagnostic report -> "
        f"{run(args.confirm_protocol_sha)}"
    )


if __name__ == "__main__":
    main()
