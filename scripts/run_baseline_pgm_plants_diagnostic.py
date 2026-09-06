#!/usr/bin/env python3
"""Run the Private-PGM baseline on the plants measured workload.

Same construction as scripts/run_baseline_pgm_nltcs_diagnostic.py, applied
to the plants medium-scale benchmark: identical published measurements (980
exact cell queries + 69 one-way init marginals) are handed to Private-PGM's
mirror descent, the fitted model is sampled deterministically to 17412 rows,
and the table is scored with the exact evaluation caliber of the frozen
plants diagnostic.  Our residual/equal numbers are quoted from the frozen
plants report (SHA-pinned), never re-run.

Adds a junction-tree feasibility precheck (memory cap) before inference:
if the hypothetical model size exceeds the cap, the run records
``baseline_infeasible`` and does not attempt estimation.

Diagnostic only: single configuration, zero tuning on either side, no gate,
no formal claim.  See docs/设计/PGM基线plants生成对比结果前协议.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
jax.config.update("jax_platforms", "cpu")

from mbi import Domain, LinearMeasurement, estimation  # noqa: E402
from mbi.junction_tree import (  # noqa: E402
    hypothetical_model_size,
    make_junction_tree,
    maximal_cliques,
)

from scripts import run_fitness_only_attribution as base  # noqa: E402
from table_diffevo.marginals import load_marginals  # noqa: E402
from table_diffevo.quality import (  # noqa: E402
    evaluate_quality_snapshot,
    query_error_metrics,
    query_fingerprint,
    validate_query_partition,
)
from table_diffevo.queries import evaluate_table, load_data  # noqa: E402
from table_diffevo.schema import load_schema  # noqa: E402


PROTOCOL_VERSION = "baseline-pgm-plants-v1"
OUTPUT_DIR = Path("outputs/baseline_pgm_plants_v1")
FROZEN_PROTOCOL_SHA256 = (
    "06dbbb90f15b6a492066f2b5ffd531ae837cbaf613f28385d79f247c642b271e"
)

DATASET = "plants"
N_RECORDS = 17412
MD_ITERS = 1000
MEASUREMENT_STDDEV = 1.0
SAMPLING_METHOD = "round"
MBI_SOURCE_COMMIT = "07635f9a0f1150237a1da1083a29b38aa4ff8645"
FEASIBILITY_CAP_MB = 4096.0

EXPECTED_CELL_QUERY_COUNT = 980
EXPECTED_CLIQUE_COUNT = 375
EXPECTED_ONE_WAY_MARGINAL_COUNT = 69
EXPECTED_ONE_WAY_CELL_COUNT = 138
EXPECTED_HELDOUT_COUNT = 1024

SCHEMA_PATH = Path("configs/plants/schema.yaml")
MARGINALS_PATH = Path("configs/plants/init_marginals.json")
MEASURED_PATH = Path("configs/plants/measured_1000query.json")
HELDOUT_PATH = Path("configs/plants/heldout_issue53_v1.json")
REFERENCE_PATH = Path("data/plants/plants.csv")
REFERENCE_REPORT_PATH = Path(
    "outputs/fitness_only_plants_diagnostic_seed9908_v1/report.json"
)

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
    "reference_report": (
        "26a8934270e9042d05e90d0a77ba1859aebd34a48ca1082291dbee9330d95565"
    ),
}

COMPARISON_ARMS = ("residual", "equal")


@dataclasses.dataclass(frozen=True, eq=False)
class CellSubsetQuery:
    """Linear query selecting a fixed subset of marginal datavector cells.

    Rows of the implied 0/1 selection matrix are distinct unit vectors, so
    the squared operator norm is exactly 1.
    """

    indices: tuple[int, ...]

    def __call__(self, factor: Any) -> Any:
        return factor.datavector()[np.asarray(self.indices)]

    def op_norm_sq(self) -> float:
        return 1.0

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


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
    return _sha256_bytes(path.read_bytes())


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "external_baseline_calibration_diagnostic_only",
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "baseline": {
            "package": "private-pgm (mbi)",
            "source_commit": MBI_SOURCE_COMMIT,
            "estimator": "estimation.MirrorDescent",
            "iters": MD_ITERS,
            "stepsize": "library_auto",
            "known_total": N_RECORDS,
            "measurement_stddev_policy": (
                "uniform_1.0_exact_answers_no_noise"
            ),
            "precision": "float64",
            "device": "cpu",
            "sampling": {
                "rows": N_RECORDS,
                "method": SAMPLING_METHOD,
                "deterministic": True,
            },
            "feasibility_precheck": {
                "method": "junction_tree.hypothetical_model_size",
                "cap_mb": FEASIBILITY_CAP_MB,
                "on_exceed": (
                    "record_baseline_infeasible_do_not_attempt_estimation"
                ),
            },
            "tuning_allowed": False,
        },
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "measured": str(MEASURED_PATH),
            "expected_cell_query_count": EXPECTED_CELL_QUERY_COUNT,
            "expected_clique_count": EXPECTED_CLIQUE_COUNT,
            "expected_one_way_marginal_count": (
                EXPECTED_ONE_WAY_MARGINAL_COUNT
            ),
            "expected_one_way_cell_count": EXPECTED_ONE_WAY_CELL_COUNT,
            "information_parity": (
                "cell_queries_as_fitness_plus_one_way_init_marginals_"
                "matches_frozen_plants_diagnostic_inputs"
            ),
            "input_sha256": {
                key: INPUT_SHA256[key]
                for key in ("schema", "marginals", "measured")
            },
        },
        "evaluation": {
            "heldout": str(HELDOUT_PATH),
            "reference": str(REFERENCE_PATH),
            "same_code_path_as_frozen_plants_diagnostic": True,
            "input_sha256": {
                key: INPUT_SHA256[key]
                for key in ("heldout", "reference")
            },
        },
        "comparison_reference": {
            "path": str(REFERENCE_REPORT_PATH),
            "sha256": INPUT_SHA256["reference_report"],
            "arms": list(COMPARISON_ARMS),
            "quality_layout": "top_level_by_arm",
            "rerun_our_arms": False,
        },
        "phase_boundary": {
            "heldout_and_reference_loaded_after_generation": True,
            "quality_used_online": False,
            "parameter_retuning_allowed": False,
            "output_overwrite_allowed": False,
        },
        "failure_policy": {
            "junction_tree_over_cap": (
                "record_baseline_infeasible_no_quality_interpretation"
            ),
            "divergence_or_nan": (
                "record_baseline_infeasible_no_quality_interpretation"
            ),
            "any_input_sha_drift": "fail_closed_abort",
            "table_shape_or_domain_drift": "fail_closed_abort",
        },
        "interpretation": {
            "diagnostic_only": True,
            "formal_claim_allowed": False,
            "question": (
                "given the identical published plants measurements, where"
                " does Private-PGM land on the exact frozen plants"
                " diagnostic caliber (measured / held-out 3&4-way /"
                " one-way safety), especially on the held-out means where"
                " the residual arm trailed the blind arm?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "pgm plants baseline 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": _protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "generation_started": False,
    }


def _schema_domain(root: Path) -> tuple[Domain, list[str], dict[str, int]]:
    schema = load_schema(str(root / SCHEMA_PATH))
    names: list[str] = []
    sizes: list[int] = []
    for attribute in schema.attributes:
        values = list(attribute.values)
        cardinality = len(values)
        if values != list(range(cardinality)):
            raise RuntimeError(
                "schema 类别必须恰为 0..k-1 以保证 PGM 编码一致："
                f"{attribute.name} -> {values}"
            )
        names.append(attribute.name)
        sizes.append(cardinality)
    domain = Domain(names, sizes)
    cards = dict(zip(names, sizes))
    return domain, names, cards


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
    if (
        not isinstance(queries, list)
        or len(queries) != EXPECTED_CELL_QUERY_COUNT
    ):
        raise RuntimeError("plants measured 查询数量漂移")
    if int(payload.get("record_count", -1)) != N_RECORDS:
        raise RuntimeError("plants measured record_count 漂移")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
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
    return {
        "queries": queries,
        "targets": targets,
        "query_count": len(queries),
        "query_identity_sha256": _sha256_bytes(
            "\n".join(fingerprints).encode("ascii")
        ),
        "target_vector_sha256": _sha256_bytes(_strict_json_bytes(targets)),
        "input_sha256": observed,
    }


def _build_cell_measurements(
    root: Path,
    audit: Mapping[str, Any],
    names: Sequence[str],
    cards: Mapping[str, int],
) -> tuple[list[LinearMeasurement], list[dict[str, Any]]]:
    order_index = {name: position for position, name in enumerate(names)}
    grouped: dict[tuple[str, ...], list[tuple[int, float]]] = {}
    for query, target in zip(audit["queries"], audit["targets"]):
        conditions = sorted(
            query["conditions"],
            key=lambda c: order_index[c["attribute"]],
        )
        clique = tuple(c["attribute"] for c in conditions)
        if len(set(clique)) != len(clique):
            raise RuntimeError(f"查询 clique 属性重复: {clique}")
        values = []
        for condition in conditions:
            if condition.get("operator", "==") != "==":
                raise RuntimeError("仅支持等值条件转 PGM 测量")
            values.append(int(condition["value"]))
        dims = [cards[a] for a in clique]
        flat = int(np.ravel_multi_index(tuple(values), tuple(dims)))
        grouped.setdefault(clique, []).append((flat, float(target)))

    measurements: list[LinearMeasurement] = []
    measurement_audit: list[dict[str, Any]] = []
    for clique in sorted(grouped):
        cells = sorted(grouped[clique])
        indices = tuple(flat for flat, _ in cells)
        if len(set(indices)) != len(indices):
            raise RuntimeError(f"clique {clique} 存在重复 cell")
        y = np.asarray([target for _, target in cells], dtype=np.float64)
        measurements.append(
            LinearMeasurement(
                y,
                clique,
                stddev=MEASUREMENT_STDDEV,
                query=CellSubsetQuery(indices),
            )
        )
        measurement_audit.append({
            "clique": list(clique),
            "cell_count": len(indices),
            "kind": "cell_subset",
        })
    if len(measurements) != EXPECTED_CLIQUE_COUNT:
        raise RuntimeError(
            f"clique 数漂移: {len(measurements)} != {EXPECTED_CLIQUE_COUNT}"
        )
    total_cells = sum(item["cell_count"] for item in measurement_audit)
    if total_cells != EXPECTED_CELL_QUERY_COUNT:
        raise RuntimeError("cell 测量值总数漂移")
    return measurements, measurement_audit


def _build_one_way_measurements(
    root: Path,
    names: Sequence[str],
    cards: Mapping[str, int],
) -> tuple[list[LinearMeasurement], list[dict[str, Any]], dict[str, Any]]:
    marginals = load_marginals(str(root / MARGINALS_PATH))
    attributes = marginals["attributes"]
    if set(attributes) != set(names):
        raise RuntimeError("init_marginals 属性集合与 schema 不一致")
    measurements: list[LinearMeasurement] = []
    measurement_audit: list[dict[str, Any]] = []
    safety_queries: list[dict[str, Any]] = []
    safety_targets: list[float] = []
    for attribute in names:
        spec = attributes[attribute]
        values = spec.get("values")
        counts = spec.get("counts")
        if not isinstance(values, list) or not isinstance(counts, list):
            raise RuntimeError("init_marginals one-way 数据格式错误")
        if len(values) != len(counts):
            raise RuntimeError("init_marginals values/counts 数量不一致")
        if list(values) != list(range(cards[attribute])):
            raise RuntimeError(
                f"init_marginals 值域必须恰为 0..k-1: {attribute}"
            )
        y = np.asarray(counts, dtype=np.float64)
        if not np.all(np.isfinite(y)) or np.any(y < 0):
            raise RuntimeError(f"init_marginals counts 非法: {attribute}")
        measurements.append(
            LinearMeasurement(y, (attribute,), stddev=MEASUREMENT_STDDEV)
        )
        measurement_audit.append({
            "clique": [attribute],
            "cell_count": len(counts),
            "kind": "full_one_way_marginal",
        })
        for value, count in zip(values, counts):
            safety_queries.append({
                "conditions": [{
                    "attribute": attribute,
                    "operator": "==",
                    "value": value,
                }],
            })
            safety_targets.append(float(count))
    if len(measurements) != EXPECTED_ONE_WAY_MARGINAL_COUNT:
        raise RuntimeError("one-way 边缘数量漂移")
    if len(safety_queries) != EXPECTED_ONE_WAY_CELL_COUNT:
        raise RuntimeError("one-way cell 数量漂移")
    fingerprints = [query_fingerprint(query) for query in safety_queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("one-way safety 存在重复语义")
    safety = {
        "queries": safety_queries,
        "targets": np.asarray(safety_targets, dtype=float),
    }
    return measurements, measurement_audit, safety


def _feasibility_precheck(
    domain: Domain,
    measurements: Sequence[LinearMeasurement],
) -> dict[str, Any]:
    cliques = [tuple(measurement.clique) for measurement in measurements]
    size_mb = float(hypothetical_model_size(domain, cliques))
    tree, _ = make_junction_tree(domain, cliques)
    tree_cliques = maximal_cliques(tree)
    biggest = max(tree_cliques, key=lambda clique: domain.size(clique))
    result = {
        "model_size_mb": size_mb,
        "cap_mb": FEASIBILITY_CAP_MB,
        "maximal_clique_count": len(tree_cliques),
        "max_clique_attribute_count": len(biggest),
        "max_clique_cells": int(domain.size(biggest)),
        "feasible": size_mb <= FEASIBILITY_CAP_MB,
    }
    return result


def _run_pgm(
    domain: Domain,
    measurements: Sequence[LinearMeasurement],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    model = estimation.MirrorDescent().estimate(
        domain,
        list(measurements),
        known_total=float(N_RECORDS),
        iters=MD_ITERS,
    )
    estimate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dataset = model.synthetic_data(rows=N_RECORDS, method=SAMPLING_METHOD)
    sample_seconds = time.perf_counter() - started
    table = pd.DataFrame(dataset.to_dict())
    timing = {
        "estimate_seconds": estimate_seconds,
        "sample_seconds": sample_seconds,
        "iters": MD_ITERS,
    }
    return table, timing


def _audit_pgm_table(
    table: pd.DataFrame,
    names: Sequence[str],
    cards: Mapping[str, int],
) -> pd.DataFrame:
    if set(table.columns) != set(names):
        raise RuntimeError("PGM 合成表列集合漂移")
    table = table[list(names)].astype(int)
    if len(table) != N_RECORDS:
        raise RuntimeError(f"PGM 合成表行数漂移: {len(table)}")
    for name in names:
        column = table[name].to_numpy()
        if column.min() < 0 or column.max() >= cards[name]:
            raise RuntimeError(f"PGM 合成表取值越界: {name}")
    return table


def _load_heldout_and_reference(
    root: Path,
) -> tuple[list[dict[str, Any]], list[float], pd.DataFrame]:
    if _sha256_file(root / HELDOUT_PATH) != INPUT_SHA256["heldout"]:
        raise RuntimeError("heldout SHA 漂移")
    if _sha256_file(root / REFERENCE_PATH) != INPUT_SHA256["reference"]:
        raise RuntimeError("reference SHA 漂移")
    payload = _load_json_object(root / HELDOUT_PATH)
    queries = payload.get("queries")
    if (
        not isinstance(queries, list)
        or len(queries) != EXPECTED_HELDOUT_COUNT
    ):
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


def _evaluate_pgm_table(
    root: Path,
    staging: Path,
    table: pd.DataFrame,
    audit: Mapping[str, Any],
    safety: Mapping[str, Any],
    heldout_queries: Sequence[dict[str, Any]],
    heldout_targets: Sequence[float],
    reference: pd.DataFrame,
) -> dict[str, Any]:
    schema = load_schema(str(root / SCHEMA_PATH))
    measured_queries = list(audit["queries"])
    measured_targets = np.asarray(audit["targets"], dtype=float)
    safety_queries = list(safety["queries"])
    safety_targets = np.asarray(safety["targets"], dtype=float)
    validate_query_partition(measured_queries, list(heldout_queries))
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
        "one_way_safety_query_count": EXPECTED_ONE_WAY_CELL_COUNT,
        "reference_loaded_after_generation": True,
    }
    quality_dir = staging / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    base._write_json(quality_dir / "pgm.json", quality)
    return quality


_COMPARISON_PATHS: dict[str, tuple[str, ...]] = {
    "measured": ("measured",),
    "heldout_3way": ("heldout", "3way"),
    "heldout_4way": ("heldout", "4way"),
    "heldout_combined": ("heldout", "combined"),
}


def _arm_snapshot(arm_quality: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, path in _COMPARISON_PATHS.items():
        node: Any = arm_quality
        for key in path:
            node = node[key]
        snapshot[name] = {
            statistic: float(node[f"normalized_l1_{statistic}"])
            for statistic in ("mean", "median", "p90", "max")
        }
    one_way = arm_quality["one_way_safety_by_target_bucket"]["by_order"]
    node = one_way["1way"]
    snapshot["one_way_safety"] = {
        statistic: float(node[f"normalized_l1_{statistic}"])
        for statistic in ("mean", "median", "p90", "max")
    }
    return snapshot


def _load_reference_report(root: Path) -> dict[str, Any]:
    path = root / REFERENCE_REPORT_PATH
    observed = _sha256_file(path)
    if observed != INPUT_SHA256["reference_report"]:
        raise RuntimeError(
            "plants 对照报告 SHA 漂移："
            f"expected={INPUT_SHA256['reference_report']}, "
            f"observed={observed}"
        )
    report = _load_json_object(path)
    quality = report["quality"]
    arms = {}
    for arm in COMPARISON_ARMS:
        arms[arm] = _arm_snapshot(quality[arm])
    return {
        "path": str(REFERENCE_REPORT_PATH),
        "sha256": observed,
        "arms": arms,
    }


def _comparison(
    pgm_quality: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    pgm = _arm_snapshot(pgm_quality)
    arms = reference["arms"]
    deltas: dict[str, Any] = {}
    for name in pgm:
        entry: dict[str, Any] = {"pgm": pgm[name]}
        for arm in COMPARISON_ARMS:
            entry[arm] = arms[arm][name]
            entry[f"pgm_minus_{arm}"] = float(
                pgm[name]["mean"] - arms[arm][name]["mean"]
            )
        deltas[name] = entry
    return {
        "normalized_l1_mean_deltas": deltas,
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("pgm plants baseline protocol SHA-256 确认值不一致")
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = _audit_generation_inputs(root)
    domain, names, cards = _schema_domain(root)
    cell_measurements, cell_audit = _build_cell_measurements(
        root, audit, names, cards
    )
    one_way_measurements, one_way_audit, safety = (
        _build_one_way_measurements(root, names, cards)
    )
    measurements = cell_measurements + one_way_measurements
    feasibility = _feasibility_precheck(domain, measurements)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        report: dict[str, Any] = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "generation_inputs": {
                "query_count": int(audit["query_count"]),
                "query_identity_sha256": audit["query_identity_sha256"],
                "target_vector_sha256": audit["target_vector_sha256"],
                "input_sha256": dict(audit["input_sha256"]),
            },
            "measurement_audit": {
                "cell_measurements": cell_audit,
                "one_way_measurements": one_way_audit,
                "measurement_count": len(measurements),
                "measured_value_count": (
                    EXPECTED_CELL_QUERY_COUNT + EXPECTED_ONE_WAY_CELL_COUNT
                ),
            },
            "feasibility": feasibility,
        }
        if not feasibility["feasible"]:
            report["verdict"] = "baseline_infeasible"
            report["completed_at_unix"] = time.time()
            base._write_json(staging / "report.json", report)
            staging.rename(destination)
            return destination
        table, timing = _run_pgm(domain, measurements)
        table = _audit_pgm_table(table, names, cards)
        generation_dir = staging / "generation"
        generation_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(generation_dir / "pgm_table.csv", index=False)
        table_sha256 = base._frame_sha256(table)
        heldout_queries, heldout_targets, reference = (
            _load_heldout_and_reference(root)
        )
        quality = _evaluate_pgm_table(
            root,
            staging,
            table,
            audit,
            safety,
            heldout_queries,
            heldout_targets,
            reference,
        )
        reference_report = _load_reference_report(root)
        report.update({
            "runtime": {
                "mbi_version": _mbi_version(),
                "mbi_source_commit": MBI_SOURCE_COMMIT,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "device": "cpu",
                "timing": timing,
            },
            "generation": {
                "table_sha256": table_sha256,
                "row_count": int(len(table)),
            },
            "quality": {"pgm": quality},
            "reference_report": reference_report,
            "comparison": _comparison(quality, reference_report),
        })
        report["completed_at_unix"] = time.time()
        base._write_json(staging / "report.json", report)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _mbi_version() -> str:
    try:
        import importlib.metadata as metadata

        for name in ("mbi", "private-pgm"):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Private-PGM baseline on the plants measured workload",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the frozen read-only plan")
    runner = subparsers.add_parser("run", help="execute the baseline")
    runner.add_argument(
        "--confirm-protocol-sha256",
        required=True,
        help="must equal the frozen protocol SHA-256",
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    destination = run(arguments.confirm_protocol_sha256)
    print(f"完成：{destination}")


if __name__ == "__main__":
    main()
