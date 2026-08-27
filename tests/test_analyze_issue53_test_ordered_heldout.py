from __future__ import annotations

import pandas as pd
import pytest

from scripts import analyze_issue53_test_ordered_heldout as diagnostic


def _condition(attribute: str, value: int) -> dict:
    return {"attribute": attribute, "operator": "==", "value": value}


def _query(*conditions: dict) -> dict:
    return {"conditions": list(conditions)}


def test_unmeasured_2way_freeze_uses_all_cells_but_no_answers():
    marginals = {
        "n_records": 8,
        "attributes": {
            attribute: {
                "type": "categorical",
                "values": [0, 1],
                "counts": [4, 4],
            }
            for attribute in ("a", "b", "c")
        },
    }
    measured = [_query(_condition("a", 0), _condition("b", 0))]

    first = diagnostic.freeze_unmeasured_2way(marginals, measured)
    second = diagnostic.freeze_unmeasured_2way(marginals, measured)

    assert first == second
    assert first["all_public_2way_count"] == 12
    assert first["exact_measured_overlap_count"] == 1
    assert first["selected_count"] == 11
    assert first["partition"]["overlap_count"] == 0
    assert first["selection_used_reference_answers"] is False
    assert first["selection_used_terminal_errors"] is False
    assert all("result" not in query for query in first["queries"])


def test_formal_group_identities_freeze_without_reference_loader(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("identity freeze must not load raw reference")

    monkeypatch.setattr(diagnostic, "load_data", forbidden)

    groups, audit = diagnostic.freeze_query_groups(diagnostic.base._repo_root())

    assert {name: len(queries) for name, queries in groups.items()} == (
        diagnostic.EXPECTED_GROUP_COUNTS
    )
    assert audit["identity_frozen_before_reference_load"] is True
    unseen = audit["unmeasured_2way"]
    assert unseen["all_public_2way_count"] == 548
    assert unseen["exact_measured_overlap_count"] == 17
    assert unseen["selected_count"] == 531
    assert unseen["selection_used_reference_answers"] is False
    assert audit["heldout_3way_4way"][
        "result_blind_identity_rebuild_equal"
    ] is True


def _summary_frame() -> pd.DataFrame:
    rows = []
    errors = {
        0: {
            310: {"absolute": 2, "sqrt_relative": 1, "relative": 0},
            311: {"absolute": 1, "sqrt_relative": 1, "relative": 2},
            312: {"absolute": 0, "sqrt_relative": 1, "relative": 1},
        },
        1: {
            310: {"absolute": 4, "sqrt_relative": 3, "relative": 2},
            311: {"absolute": 3, "sqrt_relative": 3, "relative": 4},
            312: {"absolute": 2, "sqrt_relative": 1, "relative": 0},
        },
    }
    for query_index, seeds in errors.items():
        for seed, arms in seeds.items():
            for arm, error in arms.items():
                rows.append(
                    {
                        "query_group": "g",
                        "query_group_index": query_index,
                        "seed": seed,
                        "arm": arm,
                        "abs_error": error,
                    }
                )
    return pd.DataFrame(rows)


def test_group_summary_keeps_seed_pairing_and_has_no_cross_group_total():
    summary = diagnostic.summarize_group(_summary_frame())

    assert summary["query_count"] == 2
    assert summary["query_seed_count"] == 6
    assert set(summary) == {"query_count", "query_seed_count", "arms", "pairwise"}
    assert summary["arms"]["absolute"]["mean_abs_error_count"] == 2.0
    assert summary["arms"]["absolute"]["mean_abs_error_normalized"] == (
        pytest.approx(2 / 300)
    )
    pair = summary["pairwise"]["relative_minus_absolute"]
    assert pair["query_seed_candidate_better_count"] == 3
    assert pair["query_seed_tie_count"] == 0
    assert pair["query_seed_candidate_worse_count"] == 3
    assert pair["paired_seed_candidate_better_count"] == 2
    assert pair["paired_seed_candidate_worse_count"] == 1


def _group_report(delta_by_pair: dict[str, float]) -> dict:
    return {
        "pairwise": {
            key: {"mean_abs_error_delta_count": value}
            for key, value in delta_by_pair.items()
        }
    }


def test_directional_interpretation_uses_frozen_three_unseen_groups():
    pair_keys = [
        f"{candidate}_minus_{baseline}"
        for candidate, baseline in diagnostic.PAIRWISE_COMPARISONS
    ]
    reports = {
        group: _group_report({key: 0.0 for key in pair_keys})
        for group in diagnostic.GROUP_ORDER
    }
    key = "relative_minus_absolute"
    reports["measured_1way"]["pairwise"][key][
        "mean_abs_error_delta_count"
    ] = 1.0
    for group in diagnostic.UNMEASURED_GROUPS:
        reports[group]["pairwise"][key]["mean_abs_error_delta_count"] = -0.5

    result = diagnostic._directional_interpretation(reports)

    assert result[key]["classification"] == "supports_measured_1way_dominance"
    reports["heldout_4way_512"]["pairwise"][key][
        "mean_abs_error_delta_count"
    ] = 0.5
    result = diagnostic._directional_interpretation(reports)
    assert result[key]["classification"] == "mixed_no_universal_winner"


def test_plan_and_cli_forbid_cross_group_total_and_scientific_overrides():
    plan = diagnostic.build_plan()
    assert plan["cross_group_aggregate_allowed"] is False
    assert plan["new_generation_performed"] is False
    assert plan["raw_reference_data_access_planned_after_identity_freeze"] is True
    assert plan["canonical_selection_allowed"] is False
    assert plan["scientific_overrides_allowed"] is False

    parser = diagnostic._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--confirm-source-report-sha",
            diagnostic.SOURCE_REPORT_SHA256,
        ]
    )
    assert vars(args) == {
        "command": "run",
        "confirm_source_report_sha": diagnostic.SOURCE_REPORT_SHA256,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--confirm-source-report-sha",
                diagnostic.SOURCE_REPORT_SHA256,
                "--per-order-limit",
                "10",
            ]
        )


def test_wrong_source_sha_fails_before_environment(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong SHA must fail before reading environment")

    monkeypatch.setattr(diagnostic.base, "_repo_root", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        diagnostic.run_analysis("0" * 64)
