"""第 6B-1 阶段采集/评价/独立审计接线的人工输入测试。"""

import ast
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import audit_issue53_stage6b1_arithmetic as independent
from scripts import audit_issue53_stage6b1_structure as structural
from scripts import calibrate_issue53_stage6b1_gap_l1 as calibrator
from scripts import collect_issue53_stage6b1_screen as collector
from scripts import evaluate_issue53_stage6b1_screen as evaluator
from scripts import issue53_stage6b1_common as common
from scripts import issue53_stage6b1_protocol as protocol
from table_diffevo.gap_l1_diffusion import (
    evolve_step_gap_l1_global,
    isolated_gap_l1_scores,
    stable_nonzero_rms,
)
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _schema():
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in ("a", "b")
    ])


def _queries():
    return [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "b", "operator": "==", "value": 1}]},
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
    ]


def _independent_context():
    current = pd.DataFrame({"a": [0, 0, 1], "b": [0, 1, 0]})
    queries = _queries()
    q = evaluate_table(current, queries)
    return independent._SourceContext(
        dataset="artificial",
        seed=1,
        group="work_q50",
        mode="smoke",
        current=current,
        schema=_schema(),
        queries=queries,
        target=np.array([1.0, 1.0, 1.0]),
        source_target=np.array([1, 1, 1], dtype=np.int64),
        runtime_n=3,
        source_n=3,
        q=q,
        residual=np.zeros(3),
        fitness=np.zeros(3),
        probabilities=None,
        device="numpy",
        trajectory={},
        source_proposal_state={},
    )


def test_independent_microstep_engine_matches_production_bit_for_bit():
    context = _independent_context()
    donors = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 1]})
    participate = np.array([True, True, False])
    initial = np.array([
        [False, True],
        [True, False],
        [False, False],
    ])
    scale = 0.08
    seed = 8675309

    production_table, production_mask, production_diagnostics = (
        evolve_step_gap_l1_global(
            context.current,
            donors,
            context.schema,
            context.queries,
            context.target,
            context.q,
            participate=participate,
            initial_mask=initial,
            reference_scale=scale,
            rng=np.random.default_rng(seed),
        )
    )
    audited_table, audited_mask, audited_diagnostics = (
        independent._independent_gap_replay(
            context, donors, participate, initial, scale, seed
        )
    )

    pd.testing.assert_frame_equal(production_table, audited_table)
    np.testing.assert_array_equal(production_mask, audited_mask)
    assert (
        production_diagnostics["microstep_trace_sha256"]
        == audited_diagnostics["trace_sha256"]
    )
    assert (
        production_diagnostics["gibbs_microsteps"]
        == audited_diagnostics["microsteps"]
    )
    assert production_diagnostics["final_query_counts"] == (
        audited_diagnostics["final_query_counts"].tolist()
    )


def test_independent_calibration_matches_production_scores_and_rms():
    context = _independent_context()
    donors = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 1]})
    production = isolated_gap_l1_scores(
        context.current,
        donors,
        context.schema,
        context.queries,
        context.target,
        context.q,
        exact_target_numerators=(
            context.source_target * context.runtime_n
        ),
        exact_target_denominator=context.source_n,
    )
    production_rms, production_distribution = stable_nonzero_rms(
        production["scores"]
    )
    coordinates, scores, rms, distribution = (
        independent._independent_isolated_scores(context, donors)
    )

    np.testing.assert_array_equal(production["coordinates"], coordinates)
    np.testing.assert_array_equal(production["scores"], scores)
    assert production_rms == rms
    assert production_distribution == distribution


def test_exact_error_common_scale_matches_direct_fraction_arithmetic():
    source_target = [0, 3, 10, 7]
    runtime_n = 4
    source_n = 6
    counts = [1, 1, 8, 3]
    system = evaluator.build_exact_error_system(
        source_target, runtime_n, source_n
    )
    actual = system.fraction(system.units(counts))
    expected = sum(
        Fraction(
            abs(target * runtime_n - count * source_n),
            max(target * runtime_n, 8 * source_n),
        )
        for target, count in zip(source_target, counts)
    ) / len(counts)
    assert actual == expected


def test_frozen_comparison_uses_strict_four_of_five_and_exact_ratio():
    seeds = protocol.FORMAL_SEEDS
    baseline = {seed: Fraction(10, 1) for seed in seeds}
    new = {
        seeds[0]: Fraction(9, 1),
        seeds[1]: Fraction(9, 1),
        seeds[2]: Fraction(9, 1),
        seeds[3]: Fraction(9, 1),
        seeds[4]: Fraction(10, 1),
    }
    result = evaluator._comparison(new, baseline, relation="strict_lower")
    independent_result = independent._comparison(
        new, baseline, "strict_lower"
    )
    assert result == independent_result
    assert result["passes"] is True
    assert result["passing_seed_count"] == 4

    tied = dict(new)
    tied[seeds[3]] = Fraction(10, 1)
    assert evaluator._comparison(
        tied, baseline, relation="strict_lower"
    )["passes"] is False


def test_calibration_manifest_validation_rejects_fallback_free_scale_drift():
    rows = []
    for dataset in protocol.DATASET_ORDER:
        rows.append({
            "dataset": dataset,
            "source_seed": protocol.SMOKE_SEED,
            "status": "already_exact",
            "current_error_exactly_zero": True,
            "reference_scale": None,
        })
    value = {
        "calibration_format": calibrator.CALIBRATION_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "calibrations": rows,
    }
    value["calibration_scientific_sha256"] = protocol.canonical_sha256(
        calibrator.scientific_payload(value)
    )
    calibrator.validate_calibration_manifest(value, mode="smoke")

    rows[0]["reference_scale"] = 1e-12
    value["calibration_scientific_sha256"] = protocol.canonical_sha256(
        calibrator.scientific_payload(value)
    )
    try:
        calibrator.validate_calibration_manifest(value, mode="smoke")
    except RuntimeError as error:
        assert "already_exact" in str(error)
    else:
        raise AssertionError("already_exact 不得临时填极小参考尺度")


def test_independent_auditor_has_no_stage6b1_collector_or_evaluator_import():
    path = REPOSITORY_ROOT / "scripts/audit_issue53_stage6b1_arithmetic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name.endswith(suffix)
        for name in imported
        for suffix in (
            "collect_issue53_stage6b1_screen",
            "issue53_stage6b1_common",
            "audit_issue53_stage6b1_structure",
            "evaluate_issue53_stage6b1_screen",
        )
    )


def test_result_artifact_validators_keep_smoke_result_blind(monkeypatch):
    monkeypatch.setattr(
        protocol, "expected_pair_ids", lambda mode: ("artificial_pair",)
    )
    arm_record = {"retained_unconditionally": True}
    gap_record = {
        "retained_unconditionally": True,
        "kernel_diagnostics": {
            "gibbs_microsteps": 16,
            "active_switches_k": 2,
        },
    }
    collection = {
        "collection_format": collector.COLLECTION_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "pairs": [{
            "pair_id": "artificial_pair",
            "retained_unconditionally": True,
            "address_status": "generated_unconditionally",
            "arms": {
                protocol.ARM_INDEPENDENT: arm_record,
                protocol.ARM_FACTOR: arm_record,
                protocol.ARM_GAP_L1: gap_record,
            },
        }],
    }
    collection["collection_scientific_sha256"] = (
        protocol.canonical_sha256(collector.scientific_payload(collection))
    )
    # 产物写盘时会使用 sort_keys=True；校验器必须接受真实落盘后
    # 的键顺序，同时仍严格要求三个冻结组恰好完整覆盖。
    persisted_collection = json.loads(json.dumps(collection, sort_keys=True))
    collector.validate_collection(persisted_collection, mode="smoke")

    structural_report = {
        "structural_audit_format": structural.AUDIT_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "audit_passed": True,
    }
    structural_report["structural_audit_scientific_sha256"] = (
        protocol.canonical_sha256(
            structural.scientific_payload(structural_report)
        )
    )
    structural.validate_structural_audit(structural_report, mode="smoke")

    evaluation = {
        "evaluation_format": evaluator.EVALUATION_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "formal_result_valid": False,
        "mechanism_evidence_emitted": False,
        "final_screen_classification": None,
        "dataset_evaluations": None,
    }
    evaluation["evaluation_scientific_sha256"] = (
        protocol.canonical_sha256(evaluator.scientific_payload(evaluation))
    )
    evaluator.validate_evaluation(evaluation, mode="smoke")

    arithmetic = {
        "arithmetic_audit_format": independent.AUDIT_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "formal_result_valid": False,
        "mechanism_evidence_validated": False,
        "independent_dataset_evaluations": None,
        "independent_final_screen_classification": None,
        "audit_boundary": independent.AUDIT_BOUNDARY,
        "audit_passed": True,
    }
    arithmetic["arithmetic_audit_scientific_sha256"] = (
        independent._canonical_sha256(
            independent._scientific_payload(arithmetic)
        )
    )
    independent.validate_arithmetic_audit(arithmetic, mode="smoke")
