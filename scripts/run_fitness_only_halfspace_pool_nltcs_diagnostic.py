#!/usr/bin/env python3
"""Halfspace (non-differentiable) capability diagnostic on nltcs.

Twin of the plants halfspace-pool diagnostic: appends the 40 measured
halfspace queries (frozen two-tier exam, A row-sum k=16-full / B general
k=8 sparse) to the polish-budget nltcs pool (1033 -> 1073) under the
byte-identical kappa=1 nltcs budget (cap 7000), single residual arm, seed
9908.  The 40 held-out halfspace queries never enter the pool.

Same preregistered questions as the plants twin (design doc
docs/设计/半空间不可微查询能力线设计稿.md): H1 optimizability against the
bare-guess floor (the polish-budget nltcs terminal table, which never saw
a halfspace query, evaluated read-only), H2 held-out generalization, H3
no-regression on the ordinary groups against the pinned polish-budget
nltcs report (same pool minus the 40 halfspaces, everything else
byte-fixed), H4 A/B difficulty tiers.

Pool-basis note (disclosed): the nltcs basis is the polish-budget pool,
whose 1001-query exam keeps 522 triples IN pool (unlike the plants twin,
whose basis is the pure all-2way family).  The pure all-2way nltcs pool
run is queued as separate work for cross-dataset protocol uniformity;
within this line the same-pool contrast (polish report) is internally
consistent.  Boundary note: halfspace exam selection reads the source
projection spectrum and is NOT result-blind -- diagnostic-only capability
exam.  Diagnostic only: single development seed, no gate, no formal
claim, PGM absent by design (it cannot eat halfspaces).
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
from typing import Any, Iterable, Mapping, Sequence

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


PROTOCOL_VERSION = "fitness-only-halfspace-pool-nltcs-diagnostic-v1"
OUTPUT_DIR = Path("outputs/fitness_only_halfspace_pool_nltcs_seed9908_v1")
FROZEN_PROTOCOL_SHA256 = (
    "29a9fa80152a657f174e621c3c5828f3a33db2b274530ab55b6f6234004f6c08"
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

HALFSPACE_PATH = Path("configs/nltcs/halfspace_issue53_v1.json")
HALFSPACE_SHA256 = (
    "40159977210411b571cefb6aebc11eebcb46bb416be0b87bbbf1f651342f7fee"
)
EXPECTED_HALFSPACE_TOTAL = 80
EXPECTED_HALFSPACE_MEASURED = 40
EXPECTED_HALFSPACE_HELDOUT = 40
EXPECTED_TIER_COUNTS = {
    # tier -> {role -> count}；A 档 16 道全谱交替，B 档 64 道对半
    "rowsum": {"measured": 8, "heldout": 8},
    "general": {"measured": 32, "heldout": 32},
}
EXPECTED_POOL_COUNT = (
    oneway.EXPECTED_POOL_COUNT + EXPECTED_HALFSPACE_MEASURED
)  # 1073

# 同池对照（唯一差异 = 40 道半空间进池）+ 裸猜地板表：polish nltcs 收官跑。
POLISH_REPORT_PATH = Path(
    "outputs/fitness_only_polish_budget_nltcs_seed9908_v1/report.json"
)
POLISH_REPORT_SHA256 = (
    "919b418059c621470d197ce983278443432b43b974c004bdaaa4558ec3de497d"
)
BARE_GUESS_TABLE_PATH = Path(
    "outputs/fitness_only_polish_budget_nltcs_seed9908_v1/generation/"
    "residual_terminal_current.csv"
)
BARE_GUESS_TABLE_SHA256 = (
    "2ec0de960b2a1e6ef02f1991763c24e3690215e3381f9fe13c83b46c8a644683"
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 顶层必须是对象")
    return payload


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": (
            "halfspace_non_differentiable_capability_diagnostic_only"
        ),
        "hypothesis": (
            "the zeroth-order residual-guided engine optimizes hard-"
            "threshold halfspace counting queries exactly as it optimizes"
            " conjunctive marginals -- fitness only needs query masks, not"
            " gradients -- so in-pool halfspace errors drop far below the"
            " bare-guess floor (H1) without harming the ordinary groups"
            " (H3), while held-out halfspaces (H2) and the A/B difficulty"
            " tiers (H4) locate the capability boundary; nltcs replicates"
            " the plants twin on a second dataset"
        ),
        "seed": SEED,
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "n_attributes": N_ATTRIBUTES,
        "arms": [ARM],
        "design_doc": "docs/设计/半空间不可微查询能力线设计稿.md",
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
            "geometry_note": (
                "the nltcs floor segment (7000-1750=5250 rounds) is"
                " shorter than the 6-tick floor window (~6000 rounds), so"
                " early stopping is geometrically unlikely before the cap;"
                " resource_cap_reached is the expected label, as in the"
                " polish-budget nltcs run"
            ),
        },
        "equal_arm_cut": {
            "rerun": False,
            "justification": (
                "equal_table_identity_match=true in both one-way-in-pool"
                " reports: the blind arm trajectory is independent of the"
                " query pool and budget treatment"
            ),
        },
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "halfspace_exam": str(HALFSPACE_PATH),
            "halfspace_exam_sha256": HALFSPACE_SHA256,
            "expected_halfspace_total": EXPECTED_HALFSPACE_TOTAL,
            "expected_halfspace_measured": EXPECTED_HALFSPACE_MEASURED,
            "expected_halfspace_heldout": EXPECTED_HALFSPACE_HELDOUT,
            "expected_tier_counts": EXPECTED_TIER_COUNTS,
            "expected_pool_count": EXPECTED_POOL_COUNT,
            "pool_construction": (
                "the polish-budget nltcs pool (1001-query exam plus every"
                " one-way cell, audited by the imported one-way protocol"
                " audit) plus the 40 measured halfspace queries from the"
                " frozen two-tier exam; the 40 held-out halfspace queries"
                " never enter the pool"
            ),
            "pool_audit_source": oneway.PROTOCOL_VERSION,
            "input_sha256": {
                key: oneway.INPUT_SHA256[key]
                for key in ("schema", "marginals", "measured")
            },
        },
        "pool_basis_caveat": (
            "the nltcs basis pool keeps the original exam's 522 triples"
            " in-pool (polish-budget basis), unlike the plants twin whose"
            " basis is the pure all-2way family; the pure all-2way nltcs"
            " pool run is queued as separate work for cross-dataset"
            " protocol uniformity"
        ),
        "engine_path_note": (
            "halfspace queries evaluate on the exact legacy fallback path"
            " (vectorized whitelist excludes type=halfspace by design);"
            " correctness over speed, per the engine extension contract"
        ),
        "shared_generation_config": {
            "configuration_source": (
                "byte-identical to the polish-budget nltcs protocol"
                " config (imported function); the only change relative to"
                " that protocol is the 40 appended halfspace pool members"
            ),
            "changed_relative_to_polish_budget_protocol": [
                "query_pool_plus_40_measured_halfspaces"
            ],
        },
        "evaluation_grouping": {
            "ordinary_groups_from_oneway_protocol": (
                "measured (1001-query in-pool exam), heldout 3/4-way,"
                " one_way_safety -- evaluator imported and extended,"
                " definitions byte-identical"
            ),
            "halfspace_measured_group_added": True,
            "halfspace_heldout_group_added": True,
            "halfspace_tier_buckets": ["rowsum", "general"],
            "bare_guess_baseline": (
                "the pinned polish-budget nltcs terminal table (never saw"
                " any halfspace query) evaluated read-only on both"
                " halfspace groups: the free no-guidance floor for H1/H2;"
                " no old arm is rerun"
            ),
        },
        "comparison_reference": {
            "polish_budget_report": str(POLISH_REPORT_PATH),
            "polish_budget_report_sha256": POLISH_REPORT_SHA256,
            "bare_guess_table": str(BARE_GUESS_TABLE_PATH),
            "bare_guess_table_sha256": BARE_GUESS_TABLE_SHA256,
            "primary_contrast": (
                "new residual (polish pool + 40 halfspaces) minus polish-"
                "budget residual (same pool without them): isolates the"
                " halfspace-pool effect with budget, schedule, seed and"
                " engine held byte-fixed"
            ),
            "pgm_absent_by_design": (
                "PGM cannot consume halfspace queries (discrete marginal"
                " measurements only); this line makes no PGM comparison"
            ),
        },
        "observation_points": {
            "H1_optimizability": (
                "halfspace_measured normalized_l1_mean against the bare-"
                "guess floor on the same 40 queries; the hypothesis"
                " expects a large drop; report per-tier means too"
            ),
            "H2_generalization": (
                "halfspace_heldout (40 never-pooled queries) against the"
                " bare-guess floor on the same queries; record whether"
                " guidance transfers or overfits"
            ),
            "H3_no_regression": (
                "measured / heldout_3way / heldout_4way / one_way_safety"
                " against the pinned polish-budget nltcs report"
                " (reference: measured 0.000177, heldout3 0.000433,"
                " heldout4 0.000716, one_way 0.000070); drifts should"
                " stay within small-number noise"
            ),
            "H4_difficulty_tiers": (
                "A row-sum vs B general buckets inside both halfspace"
                " groups; on nltcs the A tier row-sum spans the full"
                " 16-attribute spectrum while B uses k=8 sparse"
                " directions; locate the capability boundary if any"
            ),
        },
        "boundary_note": (
            "halfspace exam selection reads the source projection spectrum"
            " (non-degenerate thetas, quantile thresholds) and is NOT"
            " result-blind; diagnostic-only capability exam; the formal"
            " track must adopt a public-information selection rule"
        ),
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
                "does the halfspace capability replicate on a second"
                " dataset: in-pool optimization (H1), held-out transfer"
                " (H2), zero cost to ordinary groups (H3), and the A/B"
                " difficulty boundary (H4)?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "halfspace-pool nltcs diagnostic 协议身份漂移："
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


def _load_halfspace_exam(root: Path) -> dict[str, Any]:
    observed_sha = _sha256_file(root / HALFSPACE_PATH)
    if observed_sha != HALFSPACE_SHA256:
        raise RuntimeError(
            f"halfspace 考卷 SHA 漂移: expected={HALFSPACE_SHA256}, "
            f"observed={observed_sha}"
        )
    payload = _load_json_object(root / HALFSPACE_PATH)
    queries = payload.get("queries")
    if not isinstance(queries, list) or (
        len(queries) != EXPECTED_HALFSPACE_TOTAL
    ):
        raise RuntimeError("halfspace 考卷数量漂移")
    if int(payload.get("record_count", -1)) != N_RECORDS:
        raise RuntimeError("halfspace record_count 漂移")

    tier_role_counts: dict[str, dict[str, int]] = {}
    for index, query in enumerate(queries):
        if query.get("type") != "halfspace":
            raise RuntimeError(f"halfspace 第 {index} 条 type 非法")
        tier = str(query.get("tier"))
        role = str(query.get("role"))
        if tier not in EXPECTED_TIER_COUNTS or role not in (
            "measured",
            "heldout",
        ):
            raise RuntimeError(f"halfspace 第 {index} 条 tier/role 非法")
        recomputed = query_fingerprint(query)
        if recomputed != query.get("fingerprint_sha256"):
            raise RuntimeError(
                f"halfspace 第 {index} 条指纹与冻结值不一致"
            )
        result = query.get("result")
        if (
            isinstance(result, bool)
            or not isinstance(result, int)
            or not 0 < result < N_RECORDS
        ):
            raise RuntimeError(
                f"halfspace 第 {index} 条 result 退化或非法"
            )
        tier_role_counts.setdefault(tier, {}).setdefault(role, 0)
        tier_role_counts[tier][role] += 1
    if tier_role_counts != EXPECTED_TIER_COUNTS:
        raise RuntimeError(
            f"halfspace 档位×角色数量漂移: {tier_role_counts}"
        )

    measured = [q for q in queries if q["role"] == "measured"]
    heldout = [q for q in queries if q["role"] == "heldout"]
    partition = validate_query_partition(measured, heldout)
    construction = payload.get("construction", {})
    for key in (
        "measured_query_identity_sha256",
        "heldout_query_identity_sha256",
    ):
        if partition[key] != construction.get(key):
            raise RuntimeError(f"halfspace {key} 与冻结元数据不一致")

    return {
        "halfspace_sha256": observed_sha,
        "measured_queries": measured,
        "measured_targets": [float(q["result"]) for q in measured],
        "heldout_queries": heldout,
        "heldout_targets": [float(q["result"]) for q in heldout],
        "measured_identity_sha256": partition[
            "measured_query_identity_sha256"
        ],
        "heldout_identity_sha256": partition[
            "heldout_query_identity_sha256"
        ],
    }


def _audit_generation_inputs(root: Path) -> dict[str, Any]:
    """在 one-way 协议审计之上，加载半空间考卷并把 measured 档追加进池。"""
    audit = dict(oneway._audit_generation_inputs(root))
    exam = _load_halfspace_exam(root)

    pool_fingerprints = [
        query_fingerprint(query) for query in audit["pool_queries"]
    ]
    measured_fingerprints = [
        query["fingerprint_sha256"] for query in exam["measured_queries"]
    ]
    heldout_fingerprints = [
        query["fingerprint_sha256"] for query in exam["heldout_queries"]
    ]
    if set(pool_fingerprints) & set(
        measured_fingerprints + heldout_fingerprints
    ):
        raise RuntimeError("halfspace 指纹与既有池指纹相交")

    pool_queries = list(audit["pool_queries"]) + list(
        exam["measured_queries"]
    )
    pool_targets = list(audit["pool_targets"]) + list(
        exam["measured_targets"]
    )
    if len(pool_queries) != EXPECTED_POOL_COUNT:
        raise RuntimeError(f"演化池数量错误: {len(pool_queries)}")

    audit.update({
        "halfspace_sha256": exam["halfspace_sha256"],
        "halfspace_measured_queries": exam["measured_queries"],
        "halfspace_measured_targets": exam["measured_targets"],
        "halfspace_heldout_queries": exam["heldout_queries"],
        "halfspace_heldout_targets": exam["heldout_targets"],
        "halfspace_measured_count": len(exam["measured_queries"]),
        "halfspace_heldout_count": len(exam["heldout_queries"]),
        "halfspace_measured_identity_sha256": exam[
            "measured_identity_sha256"
        ],
        "halfspace_heldout_identity_sha256": exam[
            "heldout_identity_sha256"
        ],
        "pool_queries": pool_queries,
        "pool_targets": pool_targets,
        "pool_query_count": len(pool_queries),
        "pool_identity_sha256": _sha256_bytes(
            "\n".join(
                pool_fingerprints + measured_fingerprints
            ).encode("ascii")
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
        "one_way_count": int(audit["one_way_count"]),
        "halfspace_measured_count": int(audit["halfspace_measured_count"]),
        "pool_query_count": int(audit["pool_query_count"]),
        "elapsed_sec_wall": float(elapsed),
        "arms": {ARM: arm_result},
    }


def _halfspace_group_metrics(
    queries: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    answers: Sequence[float],
) -> dict[str, Any]:
    target_values = np.asarray(targets, dtype=float)
    answer_values = np.asarray(answers, dtype=float)

    def metrics_for(indices: Iterable[int]) -> dict[str, Any]:
        selected = np.asarray(list(indices), dtype=int)
        result = query_error_metrics(
            target_values[selected],
            answer_values[selected],
            N_RECORDS,
        )
        abs_errors = np.abs(
            target_values[selected] - answer_values[selected]
        )
        result.update({
            "absolute_error_mean": float(np.mean(abs_errors)),
            "absolute_error_median": float(np.median(abs_errors)),
            "absolute_error_p90": float(np.percentile(abs_errors, 90)),
            "absolute_error_max": float(np.max(abs_errors)),
            "query_count": int(len(selected)),
        })
        return result

    tiers: dict[str, list[int]] = {}
    for index, query in enumerate(queries):
        tiers.setdefault(str(query["tier"]), []).append(index)
    return {
        "overall": metrics_for(range(len(queries))),
        "by_tier": {
            tier: metrics_for(indices)
            for tier, indices in sorted(tiers.items())
        },
    }


def _evaluate_arm_with_halfspace(
    root: Path,
    staging: Path,
    audit: Mapping[str, Any],
    generation_result: Mapping[str, Any],
    heldout_queries: Sequence[dict[str, Any]],
    heldout_targets: Sequence[float],
    reference: pd.DataFrame,
) -> dict[str, Any]:
    # 普通组（1001 在池考卷 / heldout 3-4way / one-way）逐字复用 one-way
    # 协议评估器；它会写 quality/<arm>.json，半空间组补齐后重写同一文件
    # （_write_json 为整体覆盖写）。
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

    measured_queries = list(audit["halfspace_measured_queries"])
    measured_targets = np.asarray(
        audit["halfspace_measured_targets"], dtype=float
    )
    heldout_hs_queries = list(audit["halfspace_heldout_queries"])
    heldout_hs_targets = np.asarray(
        audit["halfspace_heldout_targets"], dtype=float
    )
    validate_query_partition(measured_queries, heldout_hs_queries)

    path = staging / "generation" / f"{ARM}_terminal_current.csv"
    table = pd.read_csv(path)
    if base._frame_sha256(table) != generation_result[
        "terminal_table_sha256"
    ]:
        raise RuntimeError("halfspace 评估阶段 terminal table SHA 漂移")

    quality["halfspace_measured"] = _halfspace_group_metrics(
        measured_queries,
        measured_targets,
        np.asarray(evaluate_table(table, measured_queries), dtype=float),
    )
    quality["halfspace_heldout"] = _halfspace_group_metrics(
        heldout_hs_queries,
        heldout_hs_targets,
        np.asarray(
            evaluate_table(table, heldout_hs_queries), dtype=float
        ),
    )

    # 裸猜地板（免费、离线、不重跑旧臂）：polish nltcs 收官终表从未见过
    # 任何半空间题，只读评价两组，作为 H1/H2 的无引导参照。
    bare_path = root / BARE_GUESS_TABLE_PATH
    observed_bare_sha = _sha256_file(bare_path)
    if observed_bare_sha != BARE_GUESS_TABLE_SHA256:
        raise RuntimeError(
            f"裸猜表 SHA 漂移: expected={BARE_GUESS_TABLE_SHA256}, "
            f"observed={observed_bare_sha}"
        )
    bare_table = pd.read_csv(bare_path)
    if len(bare_table) != N_RECORDS:
        raise RuntimeError("裸猜表行数漂移")
    quality["halfspace_bare_guess_baseline"] = {
        "table_source": str(BARE_GUESS_TABLE_PATH),
        "table_sha256": observed_bare_sha,
        "table_never_saw_halfspace_queries": True,
        "measured": _halfspace_group_metrics(
            measured_queries,
            measured_targets,
            np.asarray(
                evaluate_table(bare_table, measured_queries), dtype=float
            ),
        ),
        "heldout": _halfspace_group_metrics(
            heldout_hs_queries,
            heldout_hs_targets,
            np.asarray(
                evaluate_table(bare_table, heldout_hs_queries),
                dtype=float,
            ),
        ),
    }

    quality["evaluation_inputs"].update({
        "halfspace_sha256": HALFSPACE_SHA256,
        "halfspace_measured_count": len(measured_queries),
        "halfspace_heldout_count": len(heldout_hs_queries),
        "bare_guess_table_sha256": observed_bare_sha,
    })
    base._write_json(staging / "quality" / f"{ARM}.json", quality)
    return quality


def _load_reference_reports(root: Path) -> dict[str, Any]:
    path = root / POLISH_REPORT_PATH
    observed = _sha256_file(path)
    if observed != POLISH_REPORT_SHA256:
        raise RuntimeError(
            f"polish 报告 SHA 漂移: expected={POLISH_REPORT_SHA256}, "
            f"observed={observed}"
        )
    report = _load_json_object(path)
    residual_quality = report["quality"][ARM]
    if report["generation"]["arms"][ARM][
        "terminal_table_sha256"
    ] != BARE_GUESS_TABLE_SHA256:
        raise RuntimeError("polish 报告终表 SHA 与裸猜表钉死值不一致")
    return {
        "reference_sha256": {"polish_budget_report": POLISH_REPORT_SHA256},
        "polish_budget_residual": oneway._metric_snapshot(
            residual_quality
        ),
    }


def _comparison(
    quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    new_snapshot = oneway._metric_snapshot(quality)
    right = references["polish_budget_residual"]
    h3_deltas = {
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

    def group_view(group: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "overall_normalized_l1_mean": float(
                group["overall"]["normalized_l1_mean"]
            ),
            "by_tier_normalized_l1_mean": {
                tier: float(bucket["normalized_l1_mean"])
                for tier, bucket in group["by_tier"].items()
            },
        }

    bare = quality["halfspace_bare_guess_baseline"]

    def h_block(
        new_group: Mapping[str, Any], bare_group: Mapping[str, Any]
    ) -> dict[str, Any]:
        new_mean = float(new_group["overall"]["normalized_l1_mean"])
        bare_mean = float(bare_group["overall"]["normalized_l1_mean"])
        return {
            "new": group_view(new_group),
            "bare_guess_floor": group_view(bare_group),
            "improvement_ratio_bare_over_new_observation_only": (
                bare_mean / new_mean if new_mean > 0 else None
            ),
        }

    return {
        "new_residual_snapshot": new_snapshot,
        "H1_optimizability_measured": h_block(
            quality["halfspace_measured"], bare["measured"]
        ),
        "H2_generalization_heldout": h_block(
            quality["halfspace_heldout"], bare["heldout"]
        ),
        "H3_no_regression_vs_polish_report": h3_deltas,
        "H4_difficulty_tiers": {
            "measured_by_tier": group_view(
                quality["halfspace_measured"]
            )["by_tier_normalized_l1_mean"],
            "heldout_by_tier": group_view(
                quality["halfspace_heldout"]
            )["by_tier_normalized_l1_mean"],
            "bare_guess_measured_by_tier": group_view(
                bare["measured"]
            )["by_tier_normalized_l1_mean"],
            "bare_guess_heldout_by_tier": group_view(
                bare["heldout"]
            )["by_tier_normalized_l1_mean"],
        },
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate;"
            " pgm_absent_by_design_it_cannot_eat_halfspaces"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "halfspace-pool nltcs diagnostic protocol SHA-256 确认值不一致"
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
            ARM: _evaluate_arm_with_halfspace(
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
                    "halfspace_measured_queries",
                    "halfspace_measured_targets",
                    "halfspace_heldout_queries",
                    "halfspace_heldout_targets",
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
        "halfspace-pool nltcs diagnostic report -> "
        f"{run(args.confirm_protocol_sha)}"
    )


if __name__ == "__main__":
    main()
