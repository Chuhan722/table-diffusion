"""Contract tests for the Issue #53 V2 artificial validation entry."""

import inspect

import numpy as np
import pytest

from scripts import validate_issue53_v2_effective_evidence as validator


def test_plan_freezes_one_matrix_without_real_data_or_scientific_knobs() -> None:
    plan = validator.build_plan()
    protocol = plan["protocol"]

    assert plan["mode"] == "plan_only_no_artificial_draws"
    assert plan["family_count"] == 4
    assert plan["trajectory_count"] == 8000
    assert plan["evidence_evaluation_count"] == 72000
    assert plan["real_data_accessed"] is False
    assert plan["execution_started"] is False
    assert protocol == validator.frozen_protocol()
    assert protocol["scope"]["reads_project_dataset"] is False
    assert protocol["scope"]["reads_saved_real_trajectory"] is False
    assert protocol["scope"]["runs_generator"] is False
    assert protocol["scope"]["uses_gpu"] is False
    assert protocol["scope"]["consumes_privacy_budget"] is False
    assert protocol["trajectory"]["observation_prefix_lengths"] == list(
        validator.OBSERVATION_LENGTHS
    )
    assert set(
        inspect.signature(validator.run_artificial_protocol).parameters
    ) == {"output_dir"}


def test_frozen_family_theory_is_exact() -> None:
    observed = {
        family.name: (
            family.theoretical_long_run_variance,
            family.theoretical_raw_ess_ratio,
        )
        for family in validator.FAMILIES
    }

    assert observed == {
        "iid": (1.0, 1.0),
        "ar1_phi_0p5": (3.0, 1.0 / 3.0),
        "ar1_phi_0p8": (9.000000000000002, 1.0 / 9.000000000000002),
        "ar1_phi_m0p5": (1.0 / 3.0, 3.0),
    }


def test_family_generation_is_deterministic_stationary_and_seed_separated() -> None:
    family = validator.FAMILIES[1]
    first = validator.generate_family_trajectories(
        family,
        repeat_count=2,
        maximum_length=8,
    )
    second = validator.generate_family_trajectories(
        family,
        repeat_count=2,
        maximum_length=8,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 8)
    assert not np.array_equal(first[0], first[1])

    seed = np.random.SeedSequence([
        *validator.SEED_NAMESPACE,
        family.code,
        0,
    ])
    rng = np.random.Generator(np.random.PCG64(seed))
    innovations = rng.standard_normal(8)
    expected = innovations.copy()
    scale = np.sqrt(1.0 - family.phi**2)
    for position in range(1, len(expected)):
        expected[position] = (
            family.phi * expected[position - 1]
            + scale * innovations[position]
        )
    np.testing.assert_array_equal(first[0], expected)


@pytest.mark.parametrize(
    ("repeat_count", "maximum_length"),
    [
        (True, 8),
        (0, 8),
        (1.5, 8),
        (2, True),
        (2, 1),
        (2, 8.5),
    ],
)
def test_family_generation_rejects_non_protocol_shapes(
    repeat_count,
    maximum_length,
) -> None:
    with pytest.raises(ValueError):
        validator.generate_family_trajectories(
            validator.FAMILIES[0],
            repeat_count=repeat_count,
            maximum_length=maximum_length,
        )


def _length_decisions(default: bool = True):
    return [
        {
            "length": length,
            "positive_history_acceptance_pass": default,
        }
        for length in validator.OBSERVATION_LENGTHS
    ]


def test_minimum_history_requires_the_candidate_and_every_later_length() -> None:
    decisions = _length_decisions()
    assert validator.select_minimum_history(decisions) == 16

    decisions[0]["positive_history_acceptance_pass"] = False
    assert validator.select_minimum_history(decisions) == 32

    decisions = _length_decisions()
    decisions[4]["positive_history_acceptance_pass"] = False
    assert validator.select_minimum_history(decisions) == 512

    decisions = _length_decisions(default=False)
    assert validator.select_minimum_history(decisions) is None

    with pytest.raises(ValueError, match="frozen ordered grid"):
        validator.select_minimum_history(list(reversed(_length_decisions())))


def _passing_cell_summaries():
    ess_medians = {
        "iid": 1.0,
        "ar1_phi_0p5": 1.0 / 3.0,
        "ar1_phi_0p8": 1.0 / 9.0,
    }
    rows = []
    for length in validator.OBSERVATION_LENGTHS:
        for family in validator.FAMILIES:
            row = {
                "family": family.name,
                "length": length,
                "formal_ess_ratio_median": None,
                "individual_acceptance_pass": None,
                "negative_raw_ess_pass": None,
            }
            if family.role == "positive_history_selection":
                row["formal_ess_ratio_median"] = ess_medians[family.name]
                row["individual_acceptance_pass"] = True
            else:
                row["negative_raw_ess_pass"] = True
            rows.append(row)
    return rows


def test_length_decisions_keep_ordering_and_negative_control_separate() -> None:
    cells = _passing_cell_summaries()
    decisions = validator.build_length_decisions(cells)

    assert all(
        decision["positive_history_acceptance_pass"]
        for decision in decisions
    )
    assert all(decision["negative_control_pass"] for decision in decisions)

    for cell in cells:
        if cell["family"] == "ar1_phi_0p8" and cell["length"] == 16:
            cell["formal_ess_ratio_median"] = 0.75
    decisions = validator.build_length_decisions(cells)
    assert decisions[0]["positive_ess_ordering_pass"] is False
    assert decisions[0]["positive_history_acceptance_pass"] is False
    assert decisions[0]["negative_control_pass"] is True


def test_fixed_boundary_checks_pass_without_a_history_floor() -> None:
    result = validator.run_fixed_boundary_checks()

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["deferred_until_history_selection"] == [
        "less_than_selected_history_returns_insufficient_history"
    ]


def test_execution_manifest_rejects_dirty_tree_before_hashing_sources(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    def fake_git_text(_root, *arguments):
        calls.append(arguments)
        return " M tracked.py"

    monkeypatch.setattr(validator, "_git_text", fake_git_text)

    with pytest.raises(RuntimeError, match="clean worktree"):
        validator.build_execution_manifest(tmp_path)
    assert calls == [("status", "--porcelain", "--untracked-files=all")]
