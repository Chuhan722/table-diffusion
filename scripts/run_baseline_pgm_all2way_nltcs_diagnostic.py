#!/usr/bin/env python3
"""Private-PGM baseline refit on the nltcs all-2way full marginal family.

Closes the information-asymmetry caveat of the original PGM baseline
(``baseline_pgm_nltcs_v1``): that run consumed the 1001-query legacy exam
while our all-2way-era engine runs consume the full 2-way family.  This
protocol hands Private-PGM the frozen all-2way exam (120 pairs x 4 cells
= 480 exact answers) plus the same 16 one-way init marginals, so both
methods eat the same published statistics.

Junction-tree feasibility precheck first (the 120-pair complete graph on
16 binary attributes collapses to one 2^16 clique = 0.5 MB, trivially
under the 4096 MB cap -- unlike plants, where the same construction needs
2^69 cells and is structurally infeasible; see the plants feasibility
receipt protocol).

Evaluation reuses the byte-identical caliber (same heldout 1024, same
one-way safety, same grouped metrics).  Comparison deltas are restricted
to the groups whose query sets are identical across reports (heldout
3/4-way, one-way safety); the in-pool/measured sets differ by design
(480-family here vs the 1001 legacy exam elsewhere) and are reported as
anchors, not deltas.  The engine-on-pure-all2way-pool nltcs run is queued
as the apples-to-apples counterpart on the family itself.

Diagnostic only: no noise, no tuning, no formal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import jax

from mbi import LinearMeasurement  # noqa: E402
from mbi.junction_tree import (  # noqa: E402
    hypothetical_model_size,
    make_junction_tree,
    maximal_cliques,
)

from scripts import run_baseline_pgm_nltcs_diagnostic as pgm980
from scripts import run_fitness_only_attribution as base
from table_diffevo.quality import query_fingerprint


PROTOCOL_VERSION = "baseline-pgm-all2way-nltcs-v1"
OUTPUT_DIR = Path("outputs/baseline_pgm_all2way_nltcs_v1")
FROZEN_PROTOCOL_SHA256 = (
    "a80a73d2d55e60c5ccb251101c805719c95f52410950194a0a15dc5522781f5d"
)

DATASET = "nltcs"
N_RECORDS = 16181
MD_ITERS = pgm980.MD_ITERS
MEASUREMENT_STDDEV = pgm980.MEASUREMENT_STDDEV
SAMPLING_METHOD = pgm980.SAMPLING_METHOD
MBI_SOURCE_COMMIT = pgm980.MBI_SOURCE_COMMIT
FEASIBILITY_CAP_MB = 4096.0

EXPECTED_CELL_QUERY_COUNT = 480
EXPECTED_CLIQUE_COUNT = 120
EXPECTED_CELLS_PER_CLIQUE = 4
EXPECTED_ONE_WAY_MARGINAL_COUNT = pgm980.EXPECTED_ONE_WAY_MARGINAL_COUNT
EXPECTED_ONE_WAY_CELL_COUNT = pgm980.EXPECTED_ONE_WAY_CELL_COUNT

SCHEMA_PATH = pgm980.SCHEMA_PATH
MARGINALS_PATH = pgm980.MARGINALS_PATH
ALL2WAY_PATH = Path("configs/nltcs/all2way_issue53_v1.json")
HELDOUT_PATH = pgm980.HELDOUT_PATH
REFERENCE_PATH = pgm980.REFERENCE_PATH

ALL2WAY_SHA256 = (
    "5821fa4e8dc11e499c468ef618843656f51052ae91a3a2e6e6b338cde9c0552d"
)

POLISH_REPORT_PATH = Path(
    "outputs/fitness_only_polish_budget_nltcs_seed9908_v1/report.json"
)
POLISH_REPORT_SHA256 = (
    "919b418059c621470d197ce983278443432b43b974c004bdaaa4558ec3de497d"
)
OLD_PGM_REPORT_PATH = Path("outputs/baseline_pgm_nltcs_v1/report.json")
OLD_PGM_REPORT_SHA256 = (
    "bd349bc3dcaf4751cd34d3fdeb2bcd485858be0594bd0e5c149fba9d9e59b3b1"
)

COMPARABLE_GROUPS = (
    "heldout_3way",
    "heldout_4way",
    "heldout_combined",
    "one_way_safety",
)


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
        "purpose": (
            "pgm_information_parity_refit_on_all2way_family_"
            "diagnostic_only"
        ),
        "motivation": (
            "the original nltcs PGM baseline consumed the 1001-query"
            " legacy exam while the all-2way-era engine runs consume the"
            " full 2-way family; refitting PGM on the identical frozen"
            " family closes the information-asymmetry caveat so the"
            " multiples become fair"
        ),
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
            "tuning_allowed": False,
            "settings_source": (
                "byte-identical to the baseline-pgm-nltcs-v1 protocol"
                " (imported constants); the only change is the measured"
                " workload swap to the all-2way family"
            ),
        },
        "feasibility_precheck": {
            "method": "junction_tree.hypothetical_model_size",
            "cap_mb": FEASIBILITY_CAP_MB,
            "expectation": (
                "the 120-pair complete graph over 16 binary attributes"
                " collapses to a single 2^16 clique (~0.5 MB), far under"
                " the cap; any precheck failure means input drift and"
                " aborts fail-closed"
            ),
            "plants_contrast": (
                "the same all-2way construction on plants (69 attributes)"
                " requires a 2^69-cell clique and is structurally"
                " infeasible; recorded by the plants feasibility receipt"
                " protocol, not attempted"
            ),
        },
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "measured": str(ALL2WAY_PATH),
            "expected_cell_query_count": EXPECTED_CELL_QUERY_COUNT,
            "expected_clique_count": EXPECTED_CLIQUE_COUNT,
            "expected_cells_per_clique": EXPECTED_CELLS_PER_CLIQUE,
            "expected_one_way_marginal_count": (
                EXPECTED_ONE_WAY_MARGINAL_COUNT
            ),
            "expected_one_way_cell_count": EXPECTED_ONE_WAY_CELL_COUNT,
            "information_parity": (
                "all2way_family_cells_plus_one_way_init_marginals_matches"
                "_the_engine_pure_all2way_pool_diet_(queued_run)"
            ),
            "input_sha256": {
                "schema": pgm980.INPUT_SHA256["schema"],
                "marginals": pgm980.INPUT_SHA256["marginals"],
                "measured": ALL2WAY_SHA256,
            },
        },
        "evaluation": {
            "heldout": str(HELDOUT_PATH),
            "reference": str(REFERENCE_PATH),
            "same_code_path_as_engine_reports": True,
            "input_sha256": {
                "heldout": pgm980.INPUT_SHA256["heldout"],
                "reference": pgm980.INPUT_SHA256["reference"],
            },
        },
        "comparison_reference": {
            "engine_polish_budget_report": str(POLISH_REPORT_PATH),
            "engine_polish_budget_report_sha256": POLISH_REPORT_SHA256,
            "old_pgm_980_report": str(OLD_PGM_REPORT_PATH),
            "old_pgm_980_report_sha256": OLD_PGM_REPORT_SHA256,
            "comparable_groups": list(COMPARABLE_GROUPS),
            "measured_sets_note": (
                "the measured/in-pool sets differ by design (480-family"
                " here vs the 1001 legacy exam in both references), so"
                " measured errors are reported as per-report anchors and"
                " never as deltas; the engine-on-pure-all2way-pool nltcs"
                " run is queued as the apples-to-apples counterpart"
            ),
            "rerun_referenced_arms": False,
        },
        "phase_boundary": {
            "heldout_and_reference_loaded_after_generation": True,
            "quality_used_online": False,
            "parameter_retuning_allowed": False,
            "output_overwrite_allowed": False,
        },
        "failure_policy": {
            "junction_tree_over_cap": "fail_closed_abort_input_drift",
            "estimation_divergence": (
                "record_baseline_infeasible_no_quality_interpretation"
            ),
            "any_input_sha_drift": "fail_closed_abort",
            "table_shape_or_domain_drift": "fail_closed_abort",
        },
        "interpretation": {
            "diagnostic_only": True,
            "formal_claim_allowed": False,
            "question": (
                "given the identical all-2way family and one-way"
                " marginals, where does Private-PGM land on the exact"
                " held-out / one-way caliber, and how does that move the"
                " context multiples quoted against the engine?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "pgm all2way baseline 协议身份漂移："
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


def _audit_generation_inputs(root: Path) -> dict[str, Any]:
    observed = {
        "schema": _sha256_file(root / SCHEMA_PATH),
        "marginals": _sha256_file(root / MARGINALS_PATH),
        "measured": _sha256_file(root / ALL2WAY_PATH),
    }
    expected = {
        "schema": pgm980.INPUT_SHA256["schema"],
        "marginals": pgm980.INPUT_SHA256["marginals"],
        "measured": ALL2WAY_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"generation 输入 SHA 漂移: expected={expected}, "
            f"observed={observed}"
        )
    payload = _load_json_object(root / ALL2WAY_PATH)
    queries = payload.get("queries")
    if (
        not isinstance(queries, list)
        or len(queries) != EXPECTED_CELL_QUERY_COUNT
    ):
        raise RuntimeError("nltcs all2way 查询数量漂移")
    if int(payload.get("record_count", -1)) != N_RECORDS:
        raise RuntimeError("nltcs all2way record_count 漂移")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("nltcs all2way 查询存在重复语义")
    group_cells: dict[str, int] = {}
    group_sums: dict[str, float] = {}
    targets = []
    for index, query in enumerate(queries):
        if query.get("type") != "double" or len(
            query.get("conditions", [])
        ) != 2:
            raise RuntimeError(f"all2way 第 {index} 条不是 double")
        result = query.get("result")
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, float))
            or not math.isfinite(float(result))
            or result < 0
        ):
            raise ValueError(f"all2way target 非法: index={index}")
        targets.append(float(result))
        group = str(query.get("group"))
        group_cells[group] = group_cells.get(group, 0) + 1
        group_sums[group] = group_sums.get(group, 0.0) + float(result)
    if len(group_cells) != EXPECTED_CLIQUE_COUNT:
        raise RuntimeError("all2way 属性对分组数量漂移")
    if any(
        count != EXPECTED_CELLS_PER_CLIQUE
        for count in group_cells.values()
    ):
        raise RuntimeError("all2way 存在不完整 2x2 列联表分组")
    if any(
        int(total) != N_RECORDS for total in group_sums.values()
    ):
        raise RuntimeError("all2way 分组四格计数之和不等于 N")
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
        if len(indices) != EXPECTED_CELLS_PER_CLIQUE or len(
            set(indices)
        ) != len(indices):
            raise RuntimeError(f"clique {clique} 不是完整 2x2 列联表")
        y = np.asarray([target for _, target in cells], dtype=np.float64)
        measurements.append(
            LinearMeasurement(
                y,
                clique,
                stddev=MEASUREMENT_STDDEV,
                query=pgm980.CellSubsetQuery(indices),
            )
        )
        measurement_audit.append({
            "clique": list(clique),
            "cell_count": len(indices),
            "kind": "full_two_way_marginal_as_cell_subset",
        })
    if len(measurements) != EXPECTED_CLIQUE_COUNT:
        raise RuntimeError(
            f"clique 数漂移: {len(measurements)} != {EXPECTED_CLIQUE_COUNT}"
        )
    return measurements, measurement_audit


def _feasibility_precheck(
    domain: Any,
    measurements: Sequence[LinearMeasurement],
) -> dict[str, Any]:
    cliques = [tuple(measurement.clique) for measurement in measurements]
    size_mb = float(hypothetical_model_size(domain, cliques))
    tree, _ = make_junction_tree(domain, cliques)
    tree_cliques = maximal_cliques(tree)
    biggest = max(tree_cliques, key=lambda clique: domain.size(clique))
    return {
        "model_size_mb": size_mb,
        "cap_mb": FEASIBILITY_CAP_MB,
        "maximal_clique_count": len(tree_cliques),
        "max_clique_attribute_count": len(biggest),
        "max_clique_cells": int(domain.size(biggest)),
        "feasible": size_mb <= FEASIBILITY_CAP_MB,
    }


def _reference_snapshot(
    root: Path,
    path: Path,
    expected_sha: str,
    quality_path: Sequence[str],
    label: str,
) -> dict[str, Any]:
    observed = _sha256_file(root / path)
    if observed != expected_sha:
        raise RuntimeError(
            f"{label} 报告 SHA 漂移: expected={expected_sha}, "
            f"observed={observed}"
        )
    report = _load_json_object(root / path)
    node: Any = report
    for key in quality_path:
        node = node[key]
    return {
        "path": str(path),
        "sha256": observed,
        "snapshot": pgm980._arm_snapshot(node),
    }


def _load_references(root: Path) -> dict[str, Any]:
    return {
        "engine_polish_budget_residual": _reference_snapshot(
            root,
            POLISH_REPORT_PATH,
            POLISH_REPORT_SHA256,
            ("quality", "residual"),
            "engine polish",
        ),
        "old_pgm_980": _reference_snapshot(
            root,
            OLD_PGM_REPORT_PATH,
            OLD_PGM_REPORT_SHA256,
            ("quality", "pgm"),
            "old pgm",
        ),
    }


def _comparison(
    pgm_quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    pgm = pgm980._arm_snapshot(pgm_quality)
    deltas: dict[str, Any] = {}
    for name in COMPARABLE_GROUPS:
        entry: dict[str, Any] = {"pgm_all2way": pgm[name]}
        for ref_name, reference in references.items():
            snapshot = reference["snapshot"][name]
            entry[ref_name] = snapshot
            entry[f"pgm_all2way_minus_{ref_name}"] = float(
                pgm[name]["mean"] - snapshot["mean"]
            )
        deltas[name] = entry
    return {
        "normalized_l1_mean_deltas_comparable_groups_only": deltas,
        "measured_anchors_not_compared": {
            "pgm_all2way_on_480_family": pgm["measured"],
            "engine_polish_on_1001_exam": references[
                "engine_polish_budget_residual"
            ]["snapshot"]["measured"],
            "old_pgm_on_1001_exam": references["old_pgm_980"][
                "snapshot"
            ]["measured"],
            "note": (
                "different measured sets by design; engine pure-all2way-"
                "pool run queued for the family-level apples-to-apples"
            ),
        },
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "pgm all2way baseline protocol SHA-256 确认值不一致"
        )
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = _audit_generation_inputs(root)
    domain, names, cards = pgm980._schema_domain(root)
    cell_measurements, cell_audit = _build_cell_measurements(
        audit, names, cards
    )
    one_way_measurements, one_way_audit, safety = (
        pgm980._build_one_way_measurements(root, names, cards)
    )
    measurements = cell_measurements + one_way_measurements
    precheck = _feasibility_precheck(domain, measurements)
    if not precheck["feasible"]:
        raise RuntimeError(
            f"连接树预检超上限（输入漂移，fail-closed）: {precheck}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        table, timing = pgm980._run_pgm(domain, measurements)
        table = pgm980._audit_pgm_table(table, names, cards)
        generation_dir = staging / "generation"
        generation_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(generation_dir / "pgm_table.csv", index=False)
        table_sha256 = base._frame_sha256(table)
        heldout_queries, heldout_targets, reference = (
            pgm980._load_heldout_and_reference(root)
        )
        quality = pgm980._evaluate_pgm_table(
            root,
            staging,
            table,
            audit,
            safety,
            heldout_queries,
            heldout_targets,
            reference,
        )
        # 复用的评估器把 measured_sha256 写成旧考卷常数；本协议的
        # measured=all2way 考卷，修正后整体重写同一 quality json。
        quality["evaluation_inputs"]["measured_sha256"] = ALL2WAY_SHA256
        quality["evaluation_inputs"]["measured_workload"] = str(
            ALL2WAY_PATH
        )
        base._write_json(staging / "quality" / "pgm.json", quality)
        references = _load_references(root)
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "generation_inputs": {
                "query_count": int(audit["query_count"]),
                "query_identity_sha256": audit["query_identity_sha256"],
                "target_vector_sha256": audit["target_vector_sha256"],
                "input_sha256": dict(audit["input_sha256"]),
            },
            "feasibility_precheck": precheck,
            "measurement_audit": {
                "cell_measurements": cell_audit,
                "one_way_measurements": one_way_audit,
                "measurement_count": len(measurements),
                "measured_value_count": (
                    EXPECTED_CELL_QUERY_COUNT + EXPECTED_ONE_WAY_CELL_COUNT
                ),
            },
            "runtime": {
                "mbi_version": pgm980._mbi_version(),
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
            "references": references,
            "comparison": _comparison(quality, references),
            "completed_at_unix": time.time(),
        }
        base._write_json(staging / "report.json", report)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Private-PGM baseline refit on the nltcs all-2way family"
        ),
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
