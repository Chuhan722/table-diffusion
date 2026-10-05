#!/usr/bin/env python3
"""All-2way full-workload pool diagnostic on plants (kappa=1, residual arm).

Disease-3 forensics isolated the plants held-out gap to coverage: the 980
measured pool pins only 460/9384 = 4.9% of the 2-way cell family, so the
unmeasured structure drifts to low-entropy spurious correlations (the equal
blind arm shows the drift is proposal-kernel-native), while nltcs is immune
because its exam covers 479/480 = 99.8% of its 2-way family.  Survey of the
field standard (AIM, VLDB'22; Private-GSD, ICML'23) shows formal workloads
are full k-way marginal families -- the sparse 980 exam is a nonstandard,
harsher setting.  This diagnostic aligns the evolution pool with the GSD
one-shot workload shape: every 2-way cell (9384, true answers, noise-free
inner limit) plus every one-way cell (138), with the polish-density budget
kappa=1 (cap 27000) and gate-free inner A/B/C early stopping unchanged.

Evaluation keeps three rulers: (1) the original 980 exam as a continuity
group -- its 460 doubles are now in-pool while its 520 triples fall OUT of
the pool and become the sharpest generalization probes; (2) the frozen
heldout 3/4-way exam, untouched; (3) the new all-2way in-pool workload
error, the GSD-style number that anchors future noisy one-shot runs.  The
PGM reference was fit on the 980 pool, so PGM deltas carry an information
asymmetry and are context only; a PGM refit on the all-2way workload is
queued separately.  Diagnostic only: single development seed, no gate, no
formal claim.
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
from scripts import (
    run_fitness_only_oneway_pool_plants_diagnostic as oneway,
)
from scripts import (
    run_fitness_only_polish_budget_plants_diagnostic as polish,
)
from table_diffevo.fitness_only import run_fitness_only_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.quality import (
    query_error_metrics,
    query_fingerprint,
    validate_query_partition,
)
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import load_schema


PROTOCOL_VERSION = "fitness-only-all2way-pool-plants-diagnostic-v1"
OUTPUT_DIR = Path("outputs/fitness_only_all2way_pool_plants_seed9908_v1")
FROZEN_PROTOCOL_SHA256 = (
    "16f03797098cb621b471a5e74ea7759f9e183e71e9e0c25f100d8765a9dd3f03"
)
SEED = 9908
DATASET = "plants"
N_RECORDS = 17412
N_ATTRIBUTES = 69

# 预算与时刻表逐位沿用 polish-budget plants 协议（kappa=1 统一配方）；
# 打磨密度只看列数 M，与池子大小无关，因此常数不变、直接导入复用。
N_ROUNDS = polish.N_ROUNDS
RHO_BASE = polish.RHO_BASE
RHO_ANNEAL_END = polish.RHO_ANNEAL_END
RHO_ANNEAL_START_ROUND = polish.RHO_ANNEAL_START_ROUND
RHO_ANNEAL_ROUNDS = polish.RHO_ANNEAL_ROUNDS
EARLY_STOPPING_PATIENCE_TICKS = polish.EARLY_STOPPING_PATIENCE_TICKS

ARM = "residual"

ALL2WAY_PATH = Path("configs/plants/all2way_issue53_v1.json")
ALL2WAY_SHA256 = (
    "9dc37994e912ce52c5859a122150144cad487c06d71de6bb0f8bd8a4a584409a"
)
EXPECTED_PAIR_COUNT = N_ATTRIBUTES * (N_ATTRIBUTES - 1) // 2  # 2346
EXPECTED_ALL2WAY_COUNT = EXPECTED_PAIR_COUNT * 4  # 9384
EXPECTED_POOL_COUNT = (
    EXPECTED_ALL2WAY_COUNT + oneway.EXPECTED_ONE_WAY_COUNT
)  # 9522

REFERENCE_SHA256 = {
    "polish_budget_report": (
        "fef4fb5e999722775f9e2013a312009a5db5d24c847ec7fbccd9cdcaad54fd29"
    ),
    "oneway_pool_report": (
        "30fe3cd233bb72b51d8218769bdcb651f4376b32ebb122e14fb99c64133a72ec"
    ),
    "frozen_report": (
        "26a8934270e9042d05e90d0a77ba1859aebd34a48ca1082291dbee9330d95565"
    ),
    "pgm_report": (
        "fbe0c88d793fde6ed3e6b919933b7bedf2d239cb6ea606f3865eadad9f026105"
    ),
}
POLISH_BUDGET_REPORT_PATH = Path(
    "outputs/fitness_only_polish_budget_plants_seed9908_v1/report.json"
)
ONEWAY_POOL_REPORT_PATH = polish.ONEWAY_POOL_REPORT_PATH
FROZEN_REPORT_PATH = polish.FROZEN_REPORT_PATH
PGM_REPORT_PATH = polish.PGM_REPORT_PATH

SCHEMA_PATH = oneway.SCHEMA_PATH
MARGINALS_PATH = oneway.MARGINALS_PATH


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
        "purpose": "all2way_full_workload_pool_hypothesis_diagnostic_only",
        "hypothesis": (
            "the plants held-out gap (disease 3) is a coverage artifact of"
            " the sparse 980 exam pinning only 4.9% of the 2-way cell"
            " family; aligning the evolution pool with the field-standard"
            " full all-2way workload (AIM/GSD form, GSD one-shot shape,"
            " noise-free inner limit) closes the held-out 3/4-way gap"
            " substantially, following the nltcs precedent where 99.8%"
            " 2-way coverage makes the engine match or beat PGM out of"
            " pool, with no mechanism change and no new hyper-parameters"
        ),
        "seed": SEED,
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "n_attributes": N_ATTRIBUTES,
        "arms": [ARM],
        "workload_alignment": {
            "field_standard_source": (
                "AIM (VLDB'22): workloads are full 3-way / 2-way marginal"
                " families (fixed-seed variants aside), candidates from"
                " the workload downward closure; Private-GSD (ICML'23):"
                " all 2-way marginals measured once (one-shot Gaussian)"
                " then fitted by the zeroth-order generator; both papers"
                " evaluate on the same workload they measure"
            ),
            "adopted_form": (
                "GSD one-shot: pool = full all-2way cell family plus all"
                " one-way cells; the noise-free inner diagnostic is the"
                " zero-noise limit of that form (targets are exact"
                " counts); the future formal run only swaps the target"
                " column for Gaussian-noised measurements"
            ),
            "three_way_track_deferred": (
                "all-3way workloads require the adaptive select-measure"
                " shell (budget split across epochs); deferred per the"
                " outer-shell decision"
            ),
        },
        "polish_density_principle": {
            "source_protocol": polish.PROTOCOL_VERSION,
            "note": (
                "budget constants imported verbatim from the polish-budget"
                " plants protocol: kappa depends only on the attribute"
                " count M=69, not on the pool size"
            ),
            "target_kappa": 1.0,
            "polish_target_per_row": polish.POLISH_TARGET_PER_ROW,
            "solved_minimal_rounds": polish.SOLVED_MINIMAL_ROUNDS,
            "cap_rounding_unit": polish.CAP_ROUNDING_UNIT,
        },
        "budget": {
            "n_rounds": N_ROUNDS,
            "rho": RHO_BASE,
            "rho_anneal_start_round": RHO_ANNEAL_START_ROUND,
            "rho_anneal_rounds": RHO_ANNEAL_ROUNDS,
            "rho_anneal_end": RHO_ANNEAL_END,
        },
        "early_stopping": {
            "mechanism": "inner_early_stopping_a_b_c",
            "patience_ticks": EARLY_STOPPING_PATIENCE_TICKS,
            "patience_source": "engine DEFAULT_PATIENCE_TICKS, no tuning",
            "stop_on_exact_residual": True,
            "termination_labels": [
                "fit_target_reached",
                "early_stopped",
                "resource_cap_reached",
            ],
        },
        "equal_arm_cut": {
            "rerun": False,
            "justification": (
                "equal_table_identity_match=true in both the plants and"
                " nltcs one-way-in-pool reports: the blind arm trajectory"
                " is independent of the query pool and budget treatment;"
                " its reference numbers are reused from the pinned"
                " one-way-in-pool plants report"
            ),
        },
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "all2way": str(ALL2WAY_PATH),
            "all2way_sha256": ALL2WAY_SHA256,
            "expected_pair_count": EXPECTED_PAIR_COUNT,
            "expected_all2way_count": EXPECTED_ALL2WAY_COUNT,
            "expected_one_way_count": oneway.EXPECTED_ONE_WAY_COUNT,
            "expected_pool_count": EXPECTED_POOL_COUNT,
            "pool_construction": (
                "full all-2way cell family (every attribute pair, complete"
                " 2x2 contingency group, exact counts) plus every one-way"
                " cell as ordinary queries; the original 980 measured"
                " workload no longer feeds the pool -- its 460 doubles are"
                " subsumed by the family (fingerprint-and-target audited)"
                " and its 520 triples fall out of the pool"
            ),
            "input_sha256": {
                key: oneway.INPUT_SHA256[key]
                for key in ("schema", "marginals", "measured")
            },
        },
        "shared_generation_config": {
            "configuration_source": (
                "byte-identical to the polish-budget plants protocol"
                " config (imported function): v3 frozen configuration plus"
                " the predeclared kappa=1 budget and early-stopping"
                " amendments; the only change in this protocol is the"
                " query pool"
            ),
            "changed_relative_to_polish_budget": ["query_pool"],
        },
        "evaluation_grouping": {
            "measured_group_is_original_980_exam_continuity": True,
            "measured_980_doubles_now_in_pool": 460,
            "measured_980_triples_now_out_of_pool": 520,
            "all2way_pool_group_added": True,
            "heldout_frozen_untouched": True,
            "one_way_safety_same_construction_as_frozen_reports": True,
        },
        "pgm_information_asymmetry_caveat": (
            "the pinned PGM baseline was fit on the 980 exam; this run"
            " consumes the full all-2way family of true answers, so PGM"
            " deltas are context only and carry no fairness claim; a PGM"
            " refit on the all-2way workload is queued as separate work"
        ),
        "comparison_reference": {
            "polish_budget_report": str(POLISH_BUDGET_REPORT_PATH),
            "polish_budget_report_sha256": (
                REFERENCE_SHA256["polish_budget_report"]
            ),
            "oneway_pool_report": str(ONEWAY_POOL_REPORT_PATH),
            "oneway_pool_report_sha256": (
                REFERENCE_SHA256["oneway_pool_report"]
            ),
            "frozen_report": str(FROZEN_REPORT_PATH),
            "frozen_report_sha256": REFERENCE_SHA256["frozen_report"],
            "pgm_report": str(PGM_REPORT_PATH),
            "pgm_report_sha256": REFERENCE_SHA256["pgm_report"],
            "primary_contrast": (
                "new residual (all-2way pool) minus polish-budget residual"
                " (980+one-way pool): isolates the pool effect with"
                " budget, schedule, seed and engine held byte-fixed"
            ),
        },
        "observation_points": {
            "O1_termination_and_audit": (
                "termination reason within the amended contract label set;"
                " rho schedule prefix audit passes; report realized"
                " rounds, realized polish sum and early-stopping summary"
            ),
            "O2_heldout_generalization_primary": (
                "polish-budget residual heldout was 3way 0.032006 (2.10x"
                " PGM 0.015229) and 4way 0.024527 (1.91x PGM 0.012866);"
                " hypothesis expects substantial closure toward the nltcs"
                " precedent (<= ~1x PGM); record the new means and ratios"
            ),
            "O3_out_of_pool_triples_probe": (
                "the 980 exam's 520 triple cells were in-pool at"
                " 0.001646 (by-order 3way bucket, polish report) and are"
                " now out of pool; if coverage closes the generalization"
                " gap they should stay near heldout-3way level rather"
                " than the old fitted level; record the bucket"
            ),
            "O4_one_way_safety_held": (
                "one-way safety stays at or below the one-way-in-pool"
                " healing level 0.004230; record against the polish"
                " residual level 0.002007"
            ),
            "O5_all2way_workload_error_anchor": (
                "first measurement of the GSD-style full-workload in-pool"
                " error (9384 cells); no prior reference exists; this"
                " number anchors the future noisy one-shot formal runs"
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
                "with the evolution pool aligned to the field-standard"
                " all-2way workload under the unchanged kappa=1 budget,"
                " how much of the plants held-out generalization gap"
                " closes, and do the out-of-pool triple probes confirm"
                " coverage as the disease-3 mechanism?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "all2way-pool plants diagnostic 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "polish_budget_identity": polish.assert_polish_budget_identity(),
        "protocol": _protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "trajectory_count": 1,
        "generation_started": False,
    }


def _audit_generation_inputs(root: Path) -> dict[str, Any]:
    """在 one-way 协议审计之上，加载并审计 all-2way 考卷，重建演化池。

    保留 audit["queries"]/["targets"] = 原 980 考卷（评估延续组），
    只替换 pool_* 为 all-2way + one-way。
    """
    audit = dict(oneway._audit_generation_inputs(root))

    observed_sha = _sha256_file(root / ALL2WAY_PATH)
    if observed_sha != ALL2WAY_SHA256:
        raise RuntimeError(
            f"all2way 考卷 SHA 漂移: expected={ALL2WAY_SHA256}, "
            f"observed={observed_sha}"
        )
    payload = _load_json_object(root / ALL2WAY_PATH)
    queries = payload.get("queries")
    if not isinstance(queries, list) or (
        len(queries) != EXPECTED_ALL2WAY_COUNT
    ):
        raise RuntimeError("all2way 查询数量漂移")
    if int(payload.get("record_count", -1)) != N_RECORDS:
        raise RuntimeError("all2way record_count 漂移")

    targets: list[float] = []
    group_sums: dict[str, float] = {}
    group_cells: dict[str, int] = {}
    for index, query in enumerate(queries):
        if len(query.get("conditions", [])) != 2:
            raise RuntimeError(f"all2way 第 {index} 条不是二维查询")
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
        group_sums[group] = group_sums.get(group, 0.0) + float(result)
        group_cells[group] = group_cells.get(group, 0) + 1
    if len(group_sums) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"all2way 属性对数量漂移: {len(group_sums)}"
        )
    for group, cell_count in group_cells.items():
        if cell_count != 4:
            raise RuntimeError(f"{group} 不是完整 2x2 成组")
        if abs(group_sums[group] - N_RECORDS) > 1e-9:
            raise RuntimeError(f"{group} 四格计数之和不等于 N")

    all2way_fingerprints = [
        query_fingerprint(query) for query in queries
    ]
    if len(set(all2way_fingerprints)) != len(all2way_fingerprints):
        raise RuntimeError("all2way 考卷存在重复语义")

    by_fingerprint = dict(zip(all2way_fingerprints, targets))
    subsumed = 0
    for query, target in zip(audit["queries"], audit["targets"]):
        if len(query.get("conditions", [])) != 2:
            continue
        fingerprint = query_fingerprint(query)
        if fingerprint not in by_fingerprint:
            raise RuntimeError(
                f"原池 double 不在全家族中: {query.get('id')}"
            )
        if by_fingerprint[fingerprint] != float(target):
            raise RuntimeError(
                f"原池 double 目标值不一致: {query.get('id')}"
            )
        subsumed += 1
    if subsumed != 460:
        raise RuntimeError(f"原池 double 对账数量错误: {subsumed}")

    one_way_fingerprints = [
        query_fingerprint(query) for query in audit["one_way_queries"]
    ]
    if set(all2way_fingerprints) & set(one_way_fingerprints):
        raise RuntimeError("all2way 与 one-way 池指纹相交")

    pool_queries = list(queries) + list(audit["one_way_queries"])
    pool_targets = list(targets) + list(audit["one_way_targets"])
    if len(pool_queries) != EXPECTED_POOL_COUNT:
        raise RuntimeError(f"演化池数量错误: {len(pool_queries)}")
    pool_fingerprints = all2way_fingerprints + one_way_fingerprints

    audit.update({
        "all2way_queries": queries,
        "all2way_targets": targets,
        "all2way_count": len(queries),
        "all2way_sha256": observed_sha,
        "all2way_identity_sha256": _sha256_bytes(
            "\n".join(all2way_fingerprints).encode("ascii")
        ),
        "all2way_target_vector_sha256": _sha256_bytes(
            _strict_json_bytes(targets)
        ),
        "measured_doubles_subsumed_by_all2way": subsumed,
        "pool_queries": pool_queries,
        "pool_targets": pool_targets,
        "pool_query_count": len(pool_queries),
        "pool_identity_sha256": _sha256_bytes(
            "\n".join(pool_fingerprints).encode("ascii")
        ),
        "pool_target_vector_sha256": _sha256_bytes(
            _strict_json_bytes(pool_targets)
        ),
    })
    return audit


def _run_generation(
    root: Path,
    staging: Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    schema = load_schema(str(root / SCHEMA_PATH))
    marginals = load_marginals(str(root / MARGINALS_PATH))
    started = time.perf_counter()
    table, diagnostics = run_fitness_only_evolution(
        np.asarray(audit["pool_targets"], dtype=float),
        list(audit["pool_queries"]),
        schema,
        N_RECORDS,
        config=polish._config(),
        fitness_mode=ARM,
        marginals=marginals,
    )
    elapsed = time.perf_counter() - started
    generation_dir = staging / "generation"
    generation_dir.mkdir(parents=True, exist_ok=False)
    table = table.reset_index(drop=True)
    table.to_csv(
        generation_dir / f"{ARM}_terminal_current.csv",
        index=False,
    )
    base._write_json(
        generation_dir / f"{ARM}_diagnostics.json",
        diagnostics,
    )
    best_loss = float(diagnostics["best_loss_diagnostic_only"])
    final_loss = float(diagnostics["final_current_squared_loss"])
    arm_result = {
        "terminal_table_sha256": base._frame_sha256(table),
        "rounds_run": int(diagnostics["rounds_run"]),
        "budget_rounds": N_ROUNDS,
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
        "rho_schedule_audit": polish._audit_rho_schedule(diagnostics),
        "early_stopping": polish._early_stopping_summary(diagnostics),
        "floor_morphology_observation_only": polish._floor_morphology(
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
        "all2way_count": int(audit["all2way_count"]),
        "one_way_count": int(audit["one_way_count"]),
        "pool_query_count": int(audit["pool_query_count"]),
        "elapsed_sec_wall": float(elapsed),
        "arms": {ARM: arm_result},
    }


def _evaluate_arm_with_all2way(
    root: Path,
    staging: Path,
    audit: Mapping[str, Any],
    generation_result: Mapping[str, Any],
    heldout_queries: Sequence[dict[str, Any]],
    heldout_targets: Sequence[float],
    reference: pd.DataFrame,
) -> dict[str, Any]:
    all2way_queries = list(audit["all2way_queries"])
    all2way_targets = np.asarray(audit["all2way_targets"], dtype=float)
    validate_query_partition(all2way_queries, list(heldout_queries))

    quality = oneway._evaluate_arm(
        root,
        staging,
        ARM,
        audit,
        generation_result,
        heldout_queries,
        heldout_targets,
        reference,
    )

    path = staging / "generation" / f"{ARM}_terminal_current.csv"
    table = pd.read_csv(path)
    if base._frame_sha256(table) != generation_result[
        "terminal_table_sha256"
    ]:
        raise RuntimeError("all2way 评估阶段 terminal table SHA 漂移")
    answers = np.asarray(
        evaluate_table(table, all2way_queries), dtype=float
    )
    metrics = query_error_metrics(all2way_targets, answers, N_RECORDS)
    abs_errors = np.abs(all2way_targets - answers)
    metrics.update({
        "absolute_error_mean": float(np.mean(abs_errors)),
        "absolute_error_median": float(np.median(abs_errors)),
        "absolute_error_p90": float(np.percentile(abs_errors, 90)),
        "absolute_error_max": float(np.max(abs_errors)),
        "query_count": len(all2way_queries),
    })
    quality["all2way_pool"] = metrics
    quality["all2way_pool_by_target_bucket"] = (
        base._grouped_error_metrics(
            all2way_queries,
            all2way_targets,
            answers,
            N_RECORDS,
        )
    )
    quality["evaluation_inputs"].update({
        "all2way_sha256": ALL2WAY_SHA256,
        "all2way_query_count": len(all2way_queries),
        "measured_group_is_original_980_exam_continuity": True,
        "measured_980_triples_now_out_of_pool": True,
    })
    quality_dir = staging / "quality"
    base._write_json(quality_dir / f"{ARM}.json", quality)
    return quality


def _load_reference_reports(root: Path) -> dict[str, Any]:
    paths = {
        "polish_budget_report": root / POLISH_BUDGET_REPORT_PATH,
        "oneway_pool_report": root / ONEWAY_POOL_REPORT_PATH,
        "frozen_report": root / FROZEN_REPORT_PATH,
        "pgm_report": root / PGM_REPORT_PATH,
    }
    for key, path in paths.items():
        observed = _sha256_file(path)
        if observed != REFERENCE_SHA256[key]:
            raise RuntimeError(
                f"{key} SHA 漂移: expected={REFERENCE_SHA256[key]}, "
                f"observed={observed}"
            )
    polish_budget = _load_json_object(paths["polish_budget_report"])
    oneway_pool = _load_json_object(paths["oneway_pool_report"])
    frozen = _load_json_object(paths["frozen_report"])
    pgm = _load_json_object(paths["pgm_report"])
    return {
        "reference_sha256": dict(REFERENCE_SHA256),
        "polish_budget_residual": oneway._metric_snapshot(
            polish_budget["quality"]["residual"]
        ),
        "polish_budget_measured_by_order": {
            order: {
                "normalized_l1_mean": float(
                    bucket["normalized_l1_mean"]
                ),
            }
            for order, bucket in polish_budget["quality"]["residual"][
                "measured_by_order_and_target_bucket"
            ]["by_order"].items()
        },
        "oneway_pool_residual": oneway._metric_snapshot(
            oneway_pool["quality"]["residual"]
        ),
        "oneway_pool_equal_context_only": oneway._metric_snapshot(
            oneway_pool["quality"]["equal"]
        ),
        "frozen_residual": oneway._metric_snapshot(
            frozen["quality"]["residual"]
        ),
        "pgm": oneway._metric_snapshot(pgm["quality"]["pgm"]),
    }


def _comparison(
    quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    new_snapshot = oneway._metric_snapshot(quality)

    def deltas(right: Mapping[str, Any]) -> dict:
        return {
            name: {
                "delta_mean": float(
                    new_snapshot[name]["mean"] - right[name]["mean"]
                ),
                "delta_median": float(
                    new_snapshot[name]["median"] - right[name]["median"]
                ),
                "left_mean": float(new_snapshot[name]["mean"]),
                "right_mean": float(right[name]["mean"]),
            }
            for name in new_snapshot
        }

    return {
        "new_residual_snapshot": new_snapshot,
        "new_residual_minus_polish_budget_residual": deltas(
            references["polish_budget_residual"]
        ),
        "new_residual_minus_oneway_pool_residual": deltas(
            references["oneway_pool_residual"]
        ),
        "new_residual_minus_frozen_residual": deltas(
            references["frozen_residual"]
        ),
        "new_residual_minus_pgm_context_only": deltas(
            references["pgm"]
        ),
        "out_of_pool_triples_probe": {
            "new_measured_3way_bucket_mean": float(
                quality["measured_by_order_and_target_bucket"][
                    "by_order"
                ]["3way"]["normalized_l1_mean"]
            ),
            "polish_in_pool_3way_bucket_mean": float(
                references["polish_budget_measured_by_order"]["3way"][
                    "normalized_l1_mean"
                ]
            ),
        },
        "all2way_pool_anchor": {
            "normalized_l1_mean": float(
                quality["all2way_pool"]["normalized_l1_mean"]
            ),
            "normalized_l1_median": float(
                quality["all2way_pool"]["normalized_l1_median"]
            ),
            "normalized_l1_max": float(
                quality["all2way_pool"]["normalized_l1_max"]
            ),
        },
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate;"
            " pgm_deltas_context_only_information_asymmetry"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "all2way-pool plants diagnostic protocol SHA-256 确认值不一致"
        )
    budget_identity = polish.assert_polish_budget_identity()
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
            oneway._load_heldout_and_reference(root)
        )
        quality = {
            ARM: _evaluate_arm_with_all2way(
                root,
                staging,
                audit,
                generation["arms"][ARM],
                heldout_queries,
                heldout_targets,
                reference,
            ),
        }
        references = _load_reference_reports(root)
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "polish_budget_identity": budget_identity,
            "generation_inputs": {
                key: value
                for key, value in audit.items()
                if key not in {
                    "queries",
                    "targets",
                    "one_way_queries",
                    "one_way_targets",
                    "all2way_queries",
                    "all2way_targets",
                    "pool_queries",
                    "pool_targets",
                }
            },
            "generation": generation,
            "quality": quality,
            "reference_report": references,
            "comparison": _comparison(quality[ARM], references),
            "summary": {
                "diagnostic_only": True,
                "formal_claim_allowed": False,
                "all_terminal_current": True,
                "budget_rounds": N_ROUNDS,
                "rounds_run": generation["arms"][ARM]["rounds_run"],
                "early_stopping_enabled": True,
                "termination_reason": (
                    generation["arms"][ARM]["termination_reason"]
                ),
                "equal_arm_rerun": False,
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
        "all2way-pool plants diagnostic report -> "
        f"{run(args.confirm_protocol_sha)}"
    )


if __name__ == "__main__":
    main()
