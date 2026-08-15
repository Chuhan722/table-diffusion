import json

import numpy as np
import pytest

from scripts import audit_issue49_stage_a as auditor
from scripts import probe_factorized_gibbs_mixing as probe
from scripts import run_issue49_stage_a as runner


def test_probability_diagnostics_record_normalization_and_sign():
    accumulator = probe._empty_probability_accumulator()
    probe._accumulate_probability_diagnostics(
        accumulator, np.asarray([0.25, 0.75])
    )
    probe._accumulate_probability_diagnostics(
        accumulator, np.asarray([-0.1, 1.1])
    )

    result = probe._finalize_probability_diagnostics(accumulator)

    assert result == {
        "distribution_count": 2,
        "all_finite": True,
        "all_nonnegative": False,
        "probability_sum_max_error": 0.0,
        "minimum_probability": -0.1,
        "maximum_probability": 1.1,
    }


def _classification_aggregates(protocol):
    groups = runner.REQUIRED_MIXING_GROUPS
    aggregates = {}
    for group in groups:
        kernels = {}
        recovery = {}
        for temperature in protocol["evaluation_temperatures"]:
            for sweeps in protocol["candidate_sweeps"]:
                config = probe._gibbs_name(temperature, sweeps)
                if temperature == 4.0 and sweeps == 8:
                    tvd, restored = 0.051, 0.81
                else:
                    tvd, restored = 0.04, 0.85
                kernels[config] = {
                    "participating_active_rows": 10,
                    "tvd_to_joint": tvd,
                }
                recovery[config] = restored
        aggregates[group] = {
            "kernel_summary": kernels,
            "expected_direction_gap_recovery": recovery,
        }
    return aggregates


def test_classification_chooses_first_sufficient_sweep_across_all_families():
    protocol = runner._protocol("smoke")
    aggregates = _classification_aggregates(protocol)
    a0 = {
        "temperatures": {
            runner._tau_key(temperature): {
                "status": "eligible_for_mixing",
                "eligible_for_mixing": True,
            }
            for temperature in protocol["evaluation_temperatures"]
        }
    }
    correctness = {
        runner._tau_key(temperature): {
            "all_correctness_gates_passed": True
        }
        for temperature in protocol["evaluation_temperatures"]
    }

    selection = runner._classify(
        aggregates, a0, correctness, protocol
    )

    assert selection["temperatures"]["tau_4"][
        "minimal_sufficient_sweeps"
    ] == 16
    assert selection["temperatures"]["tau_8"][
        "minimal_sufficient_sweeps"
    ] == 8


def test_a0_failure_only_excludes_its_own_temperature():
    protocol = runner._protocol("smoke")
    aggregates = _classification_aggregates(protocol)
    a0 = {
        "temperatures": {
            runner._tau_key(temperature): {
                "status": (
                    "out_of_numerical_domain"
                    if temperature == 8.0 else "eligible_for_mixing"
                ),
                "eligible_for_mixing": temperature != 8.0,
            }
            for temperature in protocol["evaluation_temperatures"]
        }
    }
    correctness = {
        runner._tau_key(temperature): {
            "all_correctness_gates_passed": True
        }
        for temperature in protocol["evaluation_temperatures"]
    }

    selection = runner._classify(
        aggregates, a0, correctness, protocol
    )

    assert selection["temperatures"]["tau_8"] == {
        "status": "out_of_numerical_domain",
        "minimal_sufficient_sweeps": None,
        "candidates": [],
    }
    assert selection["temperatures"]["tau_7"][
        "minimal_sufficient_sweeps"
    ] == 8


def test_smoke_pipeline_is_reloadable_auditable_and_non_overwriting(tmp_path):
    output_dir = tmp_path / "smoke"
    report_path, library_path, report = runner.run_stage_a(
        "smoke", output_dir
    )

    assert report["status"] == "complete"
    assert report["interpretation"] == "pipeline_smoke_only_not_evidence"
    assert report["formal_result_valid"] is False
    assert report["state_library"]["state_count"] == 11
    assert report["protocol"]["stage_t_seeds"] == [99]
    assert report["stage_t"]["aggregates"]["trajectory_count"] == 5
    eligible = report["a0"]["classification"]["eligible_temperatures"]
    assert eligible
    assert len(eligible) < len(report["protocol"]["evaluation_temperatures"])
    assert set(report["a1"]["aggregates"]["global"][
        "production_sampler_diagnostics"
    ]) == {runner._tau_key(temperature) for temperature in eligible}
    assert all(
        row["all_exact_tape_replays_match"]
        and row["mismatch_count"] == 0
        and row["microsteps"] > 0
        for row in report["a1"]["aggregates"]["global"][
            "production_sampler_diagnostics"
        ].values()
    )
    assert set(runner.REQUIRED_STATE_FAMILIES).issubset(
        report["a1"]["aggregates"]
    )
    assert report["a1"]["aggregates"]["global"][
        "probability_diagnostics"
    ]["distribution_count"] > 0
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "protocol"
    ] == runner._protocol("smoke")
    assert json.loads(library_path.read_text(encoding="utf-8"))[
        "state_count"
    ] == 11

    audit_path, audit = auditor.audit_stage_a(
        report_path, library_path, output_dir / "stage_a_audit.json"
    )
    assert audit_path.exists()
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    auditor._assert_same(
        audit["selection"], report["a1"]["selection"], "selection"
    )

    with pytest.raises(FileExistsError, match="尚未启动任何轨迹"):
        runner.run_stage_a("smoke", output_dir)

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["a1"]["selection"]["temperatures"]["tau_4"][
        "status"
    ] = "tampered"
    tampered_path = tmp_path / "tampered_report.json"
    tampered_path.write_text(
        json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="report.a1.selection"):
        auditor.audit_stage_a(
            tampered_path,
            library_path,
            tmp_path / "tampered_audit.json",
        )
