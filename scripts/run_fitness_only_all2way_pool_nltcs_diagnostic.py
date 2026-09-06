#!/usr/bin/env python3
"""All-2way full-workload pool diagnostic on nltcs (kappa=1, residual arm).

The plants all-2way run confirmed coverage as the disease-3 mechanism; this
run completes the two-dataset picture and, more importantly, sets up the
decisive fair contrast against the PGM all-2way refit.  The pinned nltcs
polish run evolved on the 1001-query exam whose 522 triples leak 3-way
information into the pool, and the pinned PGM-980 baseline consumed the same
mixed-order exam -- so neither side of the old comparison was a pure 2-way
diet.  The PGM refit on the frozen 480-cell all-2way family (report pinned
here) showed that on a pure 2-way diet PGM's higher-order heldout degrades
to 0.001431 (from 0.000762 with in-exam triples): higher-order information,
not cell count, drives extrapolation quality.

This diagnostic gives the engine the byte-identical diet: the evolution pool
becomes every 2-way cell (480, true answers, noise-free inner limit) plus
every one-way cell (32) -- exactly the measurements the PGM refit consumed
-- with the polish-density budget kappa=1 (cap 7000) and gate-free inner
A/B/C early stopping unchanged.  Question: with both fitters on the same
pure 2-way diet, does zeroth-order guided evolution extrapolate the frozen
3/4-way heldout better than maximum-entropy estimation?

Evaluation keeps three rulers: (1) the original 1001 exam as a continuity
group -- its 479 doubles are now in-pool while its 522 triples fall OUT of
the pool and become the sharpest generalization probes; (2) the frozen
heldout 3/4-way exam, untouched; (3) the all-2way in-pool workload error,
directly comparable to the PGM refit's measured-family error on the same
480 cells.  Diagnostic only: single development seed, no gate, no formal
claim.
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
    run_fitness_only_oneway_pool_nltcs_diagnostic as oneway,
)
from scripts import (
    run_fitness_only_polish_budget_nltcs_diagnostic as polish,
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


PROTOCOL_VERSION = "fitness-only-all2way-pool-nltcs-diagnostic-v1"
OUTPUT_DIR = Path("outputs/fitness_only_all2way_pool_nltcs_seed9908_v1")
FROZEN_PROTOCOL_SHA256 = (
    "e11ea5b218ce8264b6f93011379c6ed98f4b1b8efd6032a4afe8194ed8407eab"
)
SEED = 9908
DATASET = "nltcs"
N_RECORDS = 16181
N_ATTRIBUTES = 16

# 预算与时刻表逐位沿用 polish-budget nltcs 协议（kappa=1 统一配方）；
# 打磨密度只看列数 M，与池子大小无关，因此常数不变、直接导入复用。
N_ROUNDS = polish.N_ROUNDS
RHO_BASE = polish.RHO_BASE
RHO_ANNEAL_END = polish.RHO_ANNEAL_END
RHO_ANNEAL_START_ROUND = polish.RHO_ANNEAL_START_ROUND
RHO_ANNEAL_ROUNDS = polish.RHO_ANNEAL_ROUNDS
EARLY_STOPPING_PATIENCE_TICKS = polish.EARLY_STOPPING_PATIENCE_TICKS

ARM = "residual"

ALL2WAY_PATH = Path("configs/nltcs/all2way_issue53_v1.json")
ALL2WAY_SHA256 = (
    "5821fa4e8dc11e499c468ef618843656f51052ae91a3a2e6e6b338cde9c0552d"
)
EXPECTED_PAIR_COUNT = N_ATTRIBUTES * (N_ATTRIBUTES - 1) // 2  # 120
EXPECTED_ALL2WAY_COUNT = EXPECTED_PAIR_COUNT * 4  # 480
EXPECTED_POOL_COUNT = (
    EXPECTED_ALL2WAY_COUNT + oneway.EXPECTED_ONE_WAY_COUNT
)  # 512
EXPECTED_SUBSUMED_DOUBLES = 479  # 原 1001 考卷中的二维题，全部被家族收编
EXPECTED_OUT_OF_POOL_TRIPLES = 522  # 原考卷三维题，本局全部出池

REFERENCE_SHA256 = {
    "polish_budget_report": (
        "919b418059c621470d197ce983278443432b43b974c004bdaaa4558ec3de497d"
    ),
    "oneway_pool_report": (
        "0d3fa1469cdf73bb0da0a0cc60fd7f2c6f462c91e617b8e59d84a8dd8e3447ce"
    ),
    "frozen_report": (
        "0a613bb4bdb695f53469d5b7bb132725eb4a3df66f9404be707f81280d14dd48"
    ),
    "pgm_report": (
        "bd349bc3dcaf4751cd34d3fdeb2bcd485858be0594bd0e5c149fba9d9e59b3b1"
    ),
    "pgm_all2way_report": (
        "b7760bab264a4a88e8ced44d10a139d59ac4be59506c51a6b8ab8341112227bc"
    ),
}
POLISH_BUDGET_REPORT_PATH = Path(
    "outputs/fitness_only_polish_budget_nltcs_seed9908_v1/report.json"
)
ONEWAY_POOL_REPORT_PATH = polish.ONEWAY_POOL_REPORT_PATH
FROZEN_REPORT_PATH = polish.FROZEN_REPORT_PATH
PGM_REPORT_PATH = polish.PGM_REPORT_PATH
PGM_ALL2WAY_REPORT_PATH = Path(
    "outputs/baseline_pgm_all2way_nltcs_v1/report.json"
)

FAIR_COMPARABLE_GROUPS = (
    "heldout_3way",
    "heldout_4way",
    "heldout_combined",
    "one_way_safety",
)

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
        "purpose": (
            "all2way_full_workload_pool_fair_pure_2way_diet_contrast"
            "_diagnostic_only"
        ),
        "hypothesis": (
            "with the evolution pool restricted to the pure all-2way cell"
            " family plus one-way cells (byte-identical diet to the PGM"
            " all-2way refit: same 480 frozen cells, same one-way"
            " marginals, true answers, noise-free inner limit), the"
            " zeroth-order guided engine extrapolates the frozen 3/4-way"
            " heldout at least as well as maximum-entropy estimation"
            " (PGM refit heldout_combined 0.001431), because residual"
            " guidance moves mass along observed-cell directions rather"
            " than flattening unobserved structure; the polish-run"
            " advantage from 522 in-pool triples (heldout_combined"
            " 0.000575) is expected to shrink -- how much survives is the"
            " open observation"
        ),
        "seed": SEED,
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "n_attributes": N_ATTRIBUTES,
        "arms": [ARM],
        "workload_alignment": {
            "field_standard_source": (
                "AIM (VLDB'22): workloads are full k-way marginal"
                " families; Private-GSD (ICML'23): all 2-way marginals"
                " measured once (one-shot Gaussian) then fitted by the"
                " zeroth-order generator; both papers evaluate on the"
                " same workload they measure"
            ),
            "adopted_form": (
                "GSD one-shot: pool = full all-2way cell family plus all"
                " one-way cells; the noise-free inner diagnostic is the"
                " zero-noise limit of that form (targets are exact"
                " counts); the future formal run only swaps the target"
                " column for Gaussian-noised measurements"
            ),
            "fair_contrast_design": (
                "the PGM all-2way refit consumed exactly this diet (480"
                " family cells grouped into 120 cliques plus all one-way"
                " marginals, stddev=1.0 exact answers); this run gives"
                " the engine the same information set, making the frozen"
                " heldout 3/4-way exam a fair extrapolation duel:"
                " zeroth-order guided evolution vs maximum-entropy"
                " estimation, no information asymmetry in either"
                " direction"
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
                " nltcs protocol: kappa depends only on the attribute"
                " count M=16, not on the pool size"
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
                " one-way-in-pool nltcs report"
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
                " cell as ordinary queries; the original 1001 measured"
                " workload no longer feeds the pool -- its 479 doubles are"
                " subsumed by the family (fingerprint-and-target audited)"
                " and its 522 triples fall out of the pool"
            ),
            "input_sha256": {
                key: oneway.INPUT_SHA256[key]
                for key in ("schema", "marginals", "measured")
            },
        },
        "shared_generation_config": {
            "configuration_source": (
                "byte-identical to the polish-budget nltcs protocol"
                " config (imported function): v3 frozen configuration plus"
                " the predeclared kappa=1 budget and early-stopping"
                " amendments; the only change in this protocol is the"
                " query pool"
            ),
            "changed_relative_to_polish_budget": ["query_pool"],
        },
        "evaluation_grouping": {
            "measured_group_is_original_1001_exam_continuity": True,
            "measured_1001_doubles_now_in_pool": (
                EXPECTED_SUBSUMED_DOUBLES
            ),
            "measured_1001_triples_now_out_of_pool": (
                EXPECTED_OUT_OF_POOL_TRIPLES
            ),
            "all2way_pool_group_added": True,
            "heldout_frozen_untouched": True,
            "one_way_safety_same_construction_as_frozen_reports": True,
        },
        "pgm_contrast_roles": {
            "pgm_980_context_only": (
                "the pinned PGM-980 baseline was fit on the mixed-order"
                " 1001 exam (522 triples leak higher-order information);"
                " its deltas remain context only"
            ),
            "pgm_all2way_fair": (
                "the pinned PGM all-2way refit consumed the identical"
                " pure 2-way diet as this run; its heldout and one-way"
                " deltas are the fair extrapolation duel, and its"
                " measured-family error on the same 480 cells is the"
                " family-level in-pool fair contrast"
            ),
        },
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
            "pgm_all2way_report": str(PGM_ALL2WAY_REPORT_PATH),
            "pgm_all2way_report_sha256": (
                REFERENCE_SHA256["pgm_all2way_report"]
            ),
            "primary_contrast": (
                "new residual (all-2way pool) minus PGM all-2way refit:"
                " identical pure 2-way diet on both sides, frozen heldout"
                " 3/4-way exam as the fair extrapolation duel; secondary:"
                " minus polish-budget residual (980+one-way pool) to"
                " price the loss of 522 in-pool triples with budget,"
                " schedule, seed and engine held byte-fixed"
            ),
        },
        "observation_points": {
            "O1_termination_and_audit": (
                "termination reason within the amended contract label set;"
                " rho schedule prefix audit passes; report realized"
                " rounds, realized polish sum and early-stopping summary"
            ),
            "O2_fair_duel_primary": (
                "PGM all-2way refit on the identical diet scored heldout"
                " 3way 0.001368 / 4way 0.001494 / combined 0.001431 and"
                " one-way 0.000185; record the engine's means and the"
                " deltas; engine at or below PGM refit supports the"
                " zeroth-order extrapolation claim"
            ),
            "O3_price_of_pure_diet": (
                "polish-budget residual (522 triples in-pool) scored"
                " heldout 3way 0.000433 / 4way 0.000716 / combined"
                " 0.000575; the pure 2-way diet is expected to score"
                " worse than that; record how much of the gap toward the"
                " PGM refit level (0.001431) opens up"
            ),
            "O4_out_of_pool_triples_probe": (
                "the 1001 exam's 522 triple cells were in-pool at"
                " 0.000189 (by-order 3way bucket, polish report) and are"
                " now out of pool; they should land near the heldout-3way"
                " level rather than the old fitted level; record the"
                " bucket"
            ),
            "O5_one_way_safety_held": (
                "one-way cells stay in-pool; safety should hold at or"
                " below the polish residual level 0.000070; PGM refit"
                " scored 0.000185 on the same construction"
            ),
            "O6_all2way_workload_error_anchor": (
                "family-level in-pool error on the 480 cells, directly"
                " comparable to the PGM refit measured-family error"
                " 0.000357 (same frozen cells, same diet); also anchors"
                " future noisy one-shot formal runs"
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
                "with both fitters restricted to the identical pure"
                " all-2way diet under the unchanged kappa=1 budget, does"
                " zeroth-order guided evolution extrapolate the frozen"
                " 3/4-way heldout better than maximum-entropy estimation,"
                " and what is the price of losing the 522 in-pool"
                " triples relative to the polish run?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "all2way-pool nltcs diagnostic 协议身份漂移："
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

    保留 audit["queries"]/["targets"] = 原 1001 考卷（评估延续组），
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
    triples = 0
    for query, target in zip(audit["queries"], audit["targets"]):
        if len(query.get("conditions", [])) == 3:
            triples += 1
            continue
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
    if subsumed != EXPECTED_SUBSUMED_DOUBLES:
        raise RuntimeError(f"原池 double 对账数量错误: {subsumed}")
    if triples != EXPECTED_OUT_OF_POOL_TRIPLES:
        raise RuntimeError(f"原池 triple 对账数量错误: {triples}")

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
        "measured_triples_out_of_pool": triples,
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
        "measured_group_is_original_1001_exam_continuity": True,
        "measured_1001_triples_now_out_of_pool": True,
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
        "pgm_all2way_report": root / PGM_ALL2WAY_REPORT_PATH,
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
    pgm_all2way = _load_json_object(paths["pgm_all2way_report"])
    pgm_all2way_quality = pgm_all2way["quality"]["pgm"]
    observed_family_sha = pgm_all2way_quality["evaluation_inputs"][
        "measured_sha256"
    ]
    if observed_family_sha != ALL2WAY_SHA256:
        raise RuntimeError(
            "PGM all2way 重拟合报告的 measured 家族 SHA 与本局考卷不一致"
        )
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
            frozen["quality"]["nltcs"]["residual"]
        ),
        "pgm": oneway._metric_snapshot(pgm["quality"]["pgm"]),
        "pgm_all2way": oneway._metric_snapshot(pgm_all2way_quality),
        "pgm_all2way_measured_family": {
            stat: float(
                pgm_all2way_quality["measured"][f"normalized_l1_{stat}"]
            )
            for stat in ("mean", "median", "p90", "max")
        },
    }


def _comparison(
    quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    new_snapshot = oneway._metric_snapshot(quality)

    def deltas(
        right: Mapping[str, Any],
        names: Sequence[str] | None = None,
    ) -> dict:
        selected = list(names) if names is not None else list(new_snapshot)
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
            for name in selected
        }

    return {
        "new_residual_snapshot": new_snapshot,
        "new_residual_minus_pgm_all2way_fair_pure_2way_diet": deltas(
            references["pgm_all2way"],
            FAIR_COMPARABLE_GROUPS,
        ),
        "family_level_in_pool_fair_contrast": {
            "engine_all2way_pool": {
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
            "pgm_all2way_measured_family": dict(
                references["pgm_all2way_measured_family"]
            ),
            "note": (
                "both fitters consumed the identical frozen 480-cell"
                " family; same cells, same exact answers"
            ),
        },
        "new_residual_minus_polish_budget_residual": deltas(
            references["polish_budget_residual"]
        ),
        "new_residual_minus_oneway_pool_residual": deltas(
            references["oneway_pool_residual"]
        ),
        "new_residual_minus_frozen_residual": deltas(
            references["frozen_residual"]
        ),
        "new_residual_minus_pgm_980_context_only": deltas(
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
            " pgm_all2way_deltas_fair_identical_diet;"
            " pgm_980_deltas_context_only_information_asymmetry"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "all2way-pool nltcs diagnostic protocol SHA-256 确认值不一致"
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
        "all2way-pool nltcs diagnostic report -> "
        f"{run(args.confirm_protocol_sha)}"
    )


if __name__ == "__main__":
    main()
