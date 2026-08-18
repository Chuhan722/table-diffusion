"""固定 α 响应曲线采集与评价协议测试。"""

from copy import deepcopy

import pytest

from scripts import evaluate_issue53_fixed_alpha_calibration as evaluator
from scripts import run_issue53_fixed_alpha_calibration as runner


def test_collection_plan_binds_updated_queries_and_no_adaptive_design():
    plan = runner.build_plan()

    assert plan["protocol_sha256"] == runner.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    assert plan["scientific_overrides_allowed"] is False
    protocol = plan["protocol"]
    assert protocol["alphas"] == [16.0, 12.0, 24.0]
    assert protocol["seeds"] == [323, 324, 325, 326, 327]
    assert protocol["trajectory_count"] == 30
    assert protocol["fixed_alpha_selection_allowed"] is False
    assert protocol["adaptive_alpha_design_in_scope"] is False
    test = protocol["datasets"]["test_300x10"]
    assert test["queries"].endswith("measured_50query_30_15_5.json")
    assert test["order_counts"] == {"2": 30, "3": 15, "4": 5}
    assert "1" not in test["order_counts"]


def test_protocol_document_and_manifest_identities_are_frozen():
    root = runner._repo_root()

    assert runner.protocol_sha256() == runner.FROZEN_PROTOCOL_SHA256
    assert runner._sha256_file(root / runner.PROTOCOL_DOC) == (
        runner.PROTOCOL_DOC_SHA256
    )
    text = (root / runner.PROTOCOL_DOC).read_text(encoding="utf-8")
    assert "不选择跨数据统一固定 α" in text
    assert "不预先规定后续自适应 α" in text
    assert "最后最多 100 轮的均值" in text
    assert "measured_50query_30_15_5.json" in text
    assert "measured_50query.json" not in text


def test_only_alpha_changes_between_fixed_response_arms():
    baseline = runner.generator_params(323, 16.0)
    for alpha in (12.0, 24.0):
        candidate = runner.generator_params(323, alpha)
        for key in ("alpha_min", "alpha_max", "fixed_alpha"):
            assert candidate.pop(key) == alpha
            baseline_value = baseline.pop(key)
            assert baseline_value == 16.0
        assert candidate == baseline
        baseline = runner.generator_params(323, 16.0)

    params = runner.generator_params(323, 16.0)
    assert params["residual_geometry"] == "relative"
    assert params["residual_geometry_floor"] == 8.0
    assert params["selection_scale_invariant"] is True
    assert params["inner_early_stopping_patience_ticks"] == 6
    assert params["factorized_gibbs_sweeps"] == 0
    with pytest.raises(ValueError, match="seed"):
        runner.generator_params(322, 16.0)
    with pytest.raises(ValueError, match="alpha"):
        runner.generator_params(323, 20.0)


def test_frozen_public_inputs_have_expected_query_orders():
    root = runner._repo_root()
    audit = runner._audit_inputs(root)

    assert audit["test_300x10"]["order_counts"] == {2: 30, 3: 15, 4: 5}
    assert audit["nltcs"]["order_counts"] == {2: 479, 3: 522}
    assert audit["test_300x10"]["query_count"] == 50
    assert audit["nltcs"]["query_count"] == 1001


def test_concentration_summary_uses_tail_and_normalized_effective_donors():
    diagnostics = {
        "rounds_run": 3,
        "row_max_prob_mean_history": [0.1, 0.2, 0.3],
        "row_max_prob_max_history": [0.4, 0.5, 0.6],
        "effective_donors_mean_history": [9.0, 8.0, 7.0],
        "donor_top_share_history": [0.2, 0.3, 0.4],
    }

    result = runner._concentration_summary(diagnostics, n_records=10)

    assert result["row_max_prob_mean"]["final"] == 0.3
    assert result["row_max_prob_mean"]["tail_mean"] == pytest.approx(0.2)
    assert result["effective_donor_fraction"]["final"] == pytest.approx(7 / 9)
    assert result["effective_donor_fraction"]["tail_mean"] == pytest.approx(8 / 9)
    assert result["tail_window_rounds"] == 3


def test_evaluation_plan_prevents_fixed_or_adaptive_selection():
    plan = evaluator.build_plan()

    assert plan["collection_protocol_sha256"] == runner.FROZEN_PROTOCOL_SHA256
    assert plan["baseline_alpha"] == 16.0
    assert plan["probe_alphas"] == [12.0, 24.0]
    assert plan["fixed_alpha_selection_allowed"] is False
    assert plan["adaptive_alpha_design_allowed"] is False
    assert plan["new_generation_allowed"] is False
    assert plan["concentration_monotonic_metric"].startswith("tail_mean")


def test_measured_l1_audit_allows_only_float_operation_order_noise():
    collected = 0.002666666666666667
    evaluated = 0.0026666666666666666

    assert evaluator._measured_l1_matches_collection(evaluated, collected)
    assert not evaluator._measured_l1_matches_collection(
        evaluated + 1e-12, collected
    )


def _metric_block(value):
    return {"normalized_l1_mean": value}


def _synthetic_case(dataset, alpha, seed):
    measured = {16.0: 1.0, 12.0: 0.9, 24.0: 1.1}[alpha]
    effective = {12.0: 0.20, 16.0: 0.10, 24.0: 0.05}[alpha]
    row_max = {12.0: 0.10, 16.0: 0.20, 24.0: 0.40}[alpha]
    offline_groups = {
        "one_way_safety": _metric_block(1.0),
    }
    metrics = {
        "measured": {
            "overall": {
                "normalized_l1_mean": measured,
                "absolute_count_error_mean": measured,
            }
        },
        "offline_query_groups": offline_groups,
        "validity": {"valid_row_rate": 1.0},
        "diversity": {
            "unique_row_rate": 0.8,
            "effective_unique_row_ratio": 0.7,
        },
    }
    if dataset == "test_300x10":
        for name in evaluator.TEST_GROUP_ORDER[1:]:
            offline_groups[name] = _metric_block(1.0)
    else:
        offline_groups["unmeasured_3way"] = _metric_block(1.0)
        offline_groups["all_4way"] = _metric_block(1.0)
        metrics["binned_joint"] = {"tvd": 1.0}
    return {
        "dataset": dataset,
        "alpha": alpha,
        "seed": seed,
        "termination_reason": "early_stopped",
        "rounds_run": 10.0,
        "normalized_work_at_stop": 1.0,
        "donor_concentration": {
            "effective_donor_fraction": {"tail_mean": effective},
            "row_max_prob_mean": {"tail_mean": row_max},
        },
        "metrics": metrics,
    }


def _synthetic_cases():
    return [
        _synthetic_case(dataset, alpha, seed)
        for dataset in runner.DATASET_ORDER
        for alpha in runner.ALPHAS
        for seed in runner.SEEDS
    ]


def test_probe_classification_is_response_only():
    cases = _synthetic_cases()

    supported = evaluator._classify_probe(
        cases, "test_300x10", 12.0, all_normal=True
    )
    unsupported = evaluator._classify_probe(
        cases, "test_300x10", 24.0, all_normal=True
    )

    assert supported["classification"] == "supported_fixed_response_point"
    assert supported["measured"]["paired_wins"] == 5
    assert unsupported["classification"] == "no_stable_measured_gain"

    compute_tradeoff = deepcopy(cases)
    for case in compute_tradeoff:
        if case["dataset"] == "test_300x10" and case["alpha"] == 12.0:
            case["normalized_work_at_stop"] = 1.1
    result = evaluator._classify_probe(
        compute_tradeoff, "test_300x10", 12.0, all_normal=True
    )
    assert result["classification"] == "quality_supported_with_compute_tradeoff"

    quality_risk = deepcopy(cases)
    for case in quality_risk:
        if case["dataset"] == "test_300x10" and case["alpha"] == 12.0:
            case["metrics"]["offline_query_groups"]["one_way_safety"][
                "normalized_l1_mean"
            ] = 1.1
    result = evaluator._classify_probe(
        quality_risk, "test_300x10", 12.0, all_normal=True
    )
    assert result["classification"] == (
        "measured_gain_with_quality_or_diversity_risk"
    )


def test_concentration_response_and_cross_dataset_direction():
    cases = _synthetic_cases()

    for dataset in runner.DATASET_ORDER:
        response = evaluator._concentration_response(
            cases, dataset, all_normal=True
        )
        assert response["classification"] == "concentration_response_monotonic"
        assert all(item["paired_wins"] == 5 for item in response["checks"].values())

    classification = evaluator._frozen_classification(cases)
    assert classification["cross_dataset_response"] == (
        "shared_fixed_response_direction"
    )
    assert classification["fixed_alpha_selected"] is None
    assert classification["adaptive_alpha_design"] is None


def test_nltcs_and_test_offline_query_identities_are_rebuilt_result_blind():
    root = runner._repo_root()

    test_groups, test_audit = evaluator._freeze_test_groups(root)
    nltcs_groups, nltcs_audit = evaluator._freeze_nltcs_groups(root)

    assert {name: len(value) for name, value in test_groups.items()} == (
        evaluator.TEST_GROUP_COUNTS
    )
    assert test_audit["identity_sha256"] == evaluator.TEST_GROUP_IDENTITIES
    assert {name: len(value) for name, value in nltcs_groups.items()} == (
        evaluator.NLTCS_GROUP_COUNTS
    )
    assert nltcs_audit["identity_sha256"] == evaluator.NLTCS_GROUP_IDENTITIES
    assert all("result" not in query for query in test_groups["common_unseen_2way"])
