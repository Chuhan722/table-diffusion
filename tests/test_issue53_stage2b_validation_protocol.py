"""Issue #53 Stage 2B detector 封存验证协议测试。"""

import inspect

import pytest

from scripts import issue53_stage2b_validation_protocol as protocol


def _summaries(status="stationary_qualified", redrift=False):
    rows = []
    for cell in protocol.expected_validation_cells():
        rows.append({
            **cell,
            "status": status,
            "candidate_round_index": (
                2000 if status == "stationary_qualified" else None
            ),
            "persistent_redrift_detected": redrift,
        })
    return rows


def test_protocol_freezes_exact_config_sources_and_hard_gates():
    manifest = protocol.frozen_validation_protocol_manifest()

    assert manifest["freeze_status"] == (
        "frozen_before_validation_seed_access"
    )
    assert manifest["scope"] == {
        "datasets": ["test_300x10", "nltcs"],
        "kernels": ["independent", "factorized_gibbs"],
        "validation_seeds": [220, 221, 222, 223, 224],
        "expected_trajectory_count": 20,
        "one_common_config": True,
        "per_dataset_query_kernel_exception": False,
    }
    config = manifest["detector"]["config"]
    assert config == protocol.FROZEN_DETECTOR_CONFIG.to_dict()
    assert config["window_size"] == 400
    assert config["stall_patience_checks"] == 4
    assert manifest["detector"][
        "required_consecutive_moving_stability_checks"
    ] == 2
    assert manifest["detector"]["persistent_redrift_checks"] == 4
    assert manifest["acceptance"] == {
        "all_20_trajectories_stationary_qualified": True,
        "stalled_trajectory_count_must_equal": 0,
        "persistent_redrift_trajectory_count_must_equal": 0,
        "cell_specific_exception_allowed": False,
        "threshold_retuning_after_validation_access_allowed": False,
        "validation_failure_action": (
            "reject_frozen_config_retire_seeds_and_redesign"
        ),
    }
    assert manifest["execution_safety"][
        "this_protocol_entry_reads_validation_data"
    ] is False


def test_plan_has_exactly_20_cells_and_cannot_start_execution():
    plan = protocol.build_validation_plan()

    assert plan["trajectory_count"] == 20
    assert plan["round_budget_per_trajectory"] == 8000
    assert plan["total_round_budget"] == 160000
    assert {row["seed"] for row in plan["cells"]} == {
        220, 221, 222, 223, 224
    }
    assert plan["validation_seed_accessed"] is False
    assert plan["generation_started"] is False
    assert plan["execution_authorized_by_this_command"] is False
    assert list(inspect.signature(protocol.build_validation_plan).parameters) == []


def test_all_twenty_qualified_without_redrift_passes():
    result = protocol.evaluate_validation_summaries(_summaries())

    assert result["classification"] == (
        "supports_frozen_detector_on_validation"
    )
    assert result["acceptance_gates"][
        "all_20_trajectories_stationary_qualified"
    ] is True
    assert result["retuning_on_these_validation_seeds_allowed"] is False


@pytest.mark.parametrize("failure", ["horizon", "stall", "redrift"])
def test_any_hard_gate_failure_rejects_common_config(failure):
    rows = _summaries()
    if failure == "horizon":
        rows[0]["status"] = "horizon_reached"
        rows[0]["candidate_round_index"] = None
    elif failure == "stall":
        rows[0]["status"] = "stalled"
        rows[0]["candidate_round_index"] = None
    else:
        rows[0]["persistent_redrift_detected"] = True

    result = protocol.evaluate_validation_summaries(rows)

    assert result["classification"] == (
        "does_not_support_frozen_detector_on_validation"
    )
    assert result["retuning_on_these_validation_seeds_allowed"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_validation_summary_scope_fails_closed(mutation):
    rows = _summaries()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[0]["unknown"] = True

    with pytest.raises(ValueError):
        protocol.evaluate_validation_summaries(rows)


def test_unknown_full_budget_status_fails_closed():
    rows = _summaries()
    rows[0]["status"] = "running"
    rows[0]["candidate_round_index"] = None

    with pytest.raises(ValueError, match="全预算状态"):
        protocol.evaluate_validation_summaries(rows)


@pytest.mark.parametrize("round_index", [True, 1200, 2100, 8400])
def test_qualified_candidate_round_must_follow_frozen_grid(round_index):
    rows = _summaries()
    rows[0]["candidate_round_index"] = round_index

    with pytest.raises(ValueError, match="候选停止轮次"):
        protocol.evaluate_validation_summaries(rows)
