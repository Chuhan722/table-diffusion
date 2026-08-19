"""Frozen protocol and kernel wiring tests for Issue #53 Stage 4."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import audit_issue53_stage4_mixing as auditor
from scripts import build_issue53_stage4_state_library as builder
from scripts import issue53_stage4_protocol as protocol
from scripts import probe_factorized_gibbs_mixing as probe
from scripts import run_issue53_stage4_mixing as runner
from table_diffevo.objective import compute_loss
from table_diffevo.schema import AttributeBlock, Schema
from table_diffevo.vectorized_eval import evaluate_vectorized


def _write_payload(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def real_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("issue53-stage4-smoke")
    library_path, library = builder.build_state_library(
        "smoke", root / "state_library.json"
    )
    report_path, report = runner.run_stage4_mixing(
        "smoke", library_path, root / "mixing_report.json"
    )
    audit_path, audit = auditor.audit_stage4_mixing(
        report_path,
        library_path,
        root / "mixing_audit.json",
    )
    return {
        "root": root,
        "library_path": library_path,
        "library": library,
        "report_path": report_path,
        "report": report,
        "audit_path": audit_path,
        "audit": audit,
    }


def test_protocol_freezes_shared_tau_sweeps_seeds_and_dataset_specific_limits():
    development = protocol.stage4_protocol("development")
    qualification = protocol.stage4_protocol("qualification")
    smoke = protocol.stage4_protocol("smoke")

    assert development["seeds"] == [323, 324, 325, 326, 327]
    assert qualification["seeds"] == [333, 334, 335, 336, 337]
    assert qualification["candidate_sweeps"] == [8, 16, 32]
    assert qualification["source_temperature"] == 2.0
    assert qualification["evaluation_temperature"] == 2.0
    assert qualification["fixed_alpha"] == 16.0
    assert qualification["selection_scale_invariant"] is True
    assert qualification["residual_geometry"] == "relative"
    assert qualification["residual_geometry_floor"] == 8.0
    assert qualification["datasets"]["test_300x10"][
        "max_factor_order"
    ] == 4
    assert qualification["datasets"]["nltcs"]["max_factor_order"] == 3
    assert qualification["datasets"]["test_300x10"][
        "proposals_per_state"
    ] == 200
    assert qualification["datasets"]["nltcs"][
        "proposals_per_state"
    ] == 4
    assert smoke["pipeline_only_runtime_override"] is True
    assert smoke["formal_result_valid"] is False
    assert smoke["datasets"]["test_300x10"]["proposals_per_state"] == 8
    assert smoke["datasets"]["nltcs"]["proposals_per_state"] == 8
    assert set(qualification["allowed_results"]) == {
        "qualified_random_scan_s8",
        "qualified_random_scan_s16",
        "qualified_random_scan_s32",
        "unqualified_at_s32",
        "invalid_or_incomplete",
    }


def test_qualification_mode_requires_exact_current_protocol_confirmation():
    expected = protocol.protocol_sha256("qualification")
    with pytest.raises(PermissionError, match="尚未获显式授权"):
        protocol.require_qualification_confirmation("qualification", None)
    with pytest.raises(PermissionError, match="尚未获显式授权"):
        protocol.require_qualification_confirmation(
            "qualification", "0" * 64
        )
    protocol.require_qualification_confirmation("qualification", expected)
    protocol.require_qualification_confirmation("development", None)
    protocol.require_qualification_confirmation("smoke", None)


def _snapshot(state_index, work):
    return {
        "state_index": state_index,
        "normalized_work": work,
        "phase": "initial" if state_index == 0 else "post_round",
        "termination_reason": (
            "resource_cap_reached" if state_index == 6 else "in_progress"
        ),
    }


def test_milestone_selection_is_distinct_nearest_and_lexicographic():
    snapshots = [
        _snapshot(0, 0.0),
        _snapshot(1, 0.9),
        _snapshot(2, 1.1),
        _snapshot(3, 2.0),
        _snapshot(4, 3.0),
        _snapshot(5, 3.1),
        _snapshot(6, 4.0),
    ]

    selected = builder._select_milestones(snapshots)

    assert [item["state_group"] for item in selected] == list(
        protocol.STATE_GROUPS
    )
    assert [item["source_snapshot_index"] for item in selected] == [
        0,
        1,
        3,
        4,
        6,
    ]
    assert auditor._milestone_indices(snapshots) == (0, 1, 3, 4, 6)
    with pytest.raises(RuntimeError, match="不足五个"):
        builder._select_milestones(snapshots[:4])


def _attempt(sweeps, *, valid=True, passed=False):
    return {"sweeps": sweeps, "valid": valid, "passed": passed}


def test_shared_sweep_stop_rule_selects_only_first_two_dataset_pass():
    assert runner._result_from_attempts([
        _attempt(8, passed=True)
    ]) == ("qualified_random_scan_s8", 8)
    assert runner._result_from_attempts([
        _attempt(8),
        _attempt(16, passed=True),
    ]) == ("qualified_random_scan_s16", 16)
    exhausted = [_attempt(8), _attempt(16), _attempt(32)]
    assert runner._result_from_attempts(exhausted) == (
        "unqualified_at_s32",
        None,
    )
    invalid = copy.deepcopy(exhausted)
    invalid[0]["valid"] = False
    assert runner._result_from_attempts(invalid[:1]) == (
        "invalid_or_incomplete",
        None,
    )
    with pytest.raises(RuntimeError, match="invalid 后"):
        runner._result_from_attempts(invalid)
    with pytest.raises(RuntimeError, match="错误提前停止"):
        runner._result_from_attempts([_attempt(8)])
    with pytest.raises(RuntimeError, match="首次双数据通过后"):
        runner._result_from_attempts([
            _attempt(8, passed=True),
            _attempt(16, passed=True),
        ])


def _four_way_problem():
    schema = Schema([
        AttributeBlock(
            name=f"x{index}",
            type="categorical",
            description=f"x{index}",
            values=[0, 1],
        )
        for index in range(4)
    ])
    records = [
        {f"x{index}": (row >> index) & 1 for index in range(4)}
        for row in range(16)
    ]
    state = pd.DataFrame(records, columns=schema.attribute_names())
    queries = [{
        "conditions": [
            {"attribute": f"x{index}", "operator": "==", "value": 1}
            for index in range(order)
        ]
    } for order in (1, 2, 3, 4)]
    target = np.asarray([7.0, 5.0, 4.0, 3.0])
    return state, target, queries, schema


def test_generalized_probe_supports_four_way_and_shared_candidate_tape(
    monkeypatch,
):
    state, target, queries, schema = _four_way_problem()
    q, _, _ = evaluate_vectorized(
        state,
        queries,
        schema,
        target=target,
        n_records=len(state),
        want_fitness=False,
        device="numpy",
        verbose=False,
    )
    controls = {
        "source_seed": 53,
        "state_round": 0,
        "state_sha256": probe._frame_sha256(state),
        "current_loss": float(compute_loss(target, q)),
        "probe_alpha": 16.0,
        "direction_reference_scale": 1.0,
    }
    monkeypatch.setattr(
        probe,
        "sample_donors",
        lambda probabilities, rng, device: np.arange(len(state)) ^ 15,
    )
    observed_scale_invariant = []
    original_sampling = probe.compute_sampling_probs

    def record_sampling(*args, **kwargs):
        observed_scale_invariant.append(kwargs["scale_invariant"])
        return original_sampling(*args, **kwargs)

    monkeypatch.setattr(probe, "compute_sampling_probs", record_sampling)

    def run(sweeps):
        return probe._probe_state(
            state,
            target,
            queries,
            schema,
            seed=53,
            state_index=0,
            state_rounds=0,
            temperatures=[2.0],
            sweeps=[0, sweeps],
            proposals=2,
            device="numpy",
            max_active_attributes=4,
            external_snapshot_controls=controls,
            n_records=len(state),
            rho=1.0,
            eta=0.5,
            max_factor_order=4,
            selection_scale_invariant=True,
            selection_scale_invariant_min_spread=1e-3,
            residual_geometry="relative",
            residual_geometry_floor=8.0,
        )

    at_eight = run(8)
    at_sixteen = run(16)

    assert observed_scale_invariant == [True, True]
    assert at_eight["factor_diagnostics"][
        "maximum_active_factor_order"
    ] == 4
    assert at_eight["factor_diagnostics"]["exact_energy_max_error"] <= 1e-10
    assert at_eight["kernel_summary_by_active_width"][
        "active_width_1_4"
    ][probe._gibbs_name(2.0, 8)]["participating_active_rows"] == 32
    assert at_eight["shared_condition_identity"] == at_sixteen[
        "shared_condition_identity"
    ]
    assert auditor._condition_digest(
        at_eight["shared_condition_identity"]["proposal_sha256"]
    ) == at_eight["shared_condition_identity"]["scientific_sha256"]


def test_real_two_dataset_smoke_is_complete_audited_and_nonformal(real_smoke):
    library = real_smoke["library"]
    report = real_smoke["report"]
    audit = real_smoke["audit"]

    assert library["status"] == "complete"
    assert library["formal_result_valid"] is False
    assert library["manifest"]["state_count"] == 10
    assert report["formal_result_valid"] is False
    assert report["result"] == "qualified_random_scan_s8"
    assert report["attempted_sweeps"] == [8]
    assert list(report["attempts"][0]["datasets"]) == [
        "test_300x10",
        "nltcs",
    ]
    assert all(
        item["valid"] and item["passed"]
        for item in report["attempts"][0]["datasets"].values()
    )
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert audit["execution_scientific_sha256"] == report[
        "execution_scientific_sha256"
    ]


def test_independent_audit_rejects_condition_metric_and_state_tampering(
    real_smoke,
):
    root = real_smoke["root"]

    condition = copy.deepcopy(real_smoke["report"])
    identity = condition["attempts"][0]["datasets"]["test_300x10"][
        "state_results"
    ][0]["probe"]["shared_condition_identity"]
    identity["proposal_sha256"][0] = "0" * 64
    condition_path = _write_payload(
        root / "tampered_condition.json", condition
    )
    with pytest.raises(RuntimeError, match="shared condition"):
        auditor.audit_stage4_mixing(
            condition_path,
            real_smoke["library_path"],
            root / "tampered_condition_audit.json",
        )

    metric = copy.deepcopy(real_smoke["report"])
    metric["attempts"][0]["datasets"]["nltcs"]["mixing"]["global"][
        "tvd_to_joint"
    ] += 0.01
    metric_path = _write_payload(root / "tampered_metric.json", metric)
    with pytest.raises(RuntimeError, match="mixing"):
        auditor.audit_stage4_mixing(
            metric_path,
            real_smoke["library_path"],
            root / "tampered_metric_audit.json",
        )

    library = copy.deepcopy(real_smoke["library"])
    first = library["states"][0]["snapshot"]["table_records"][0]
    attribute = next(iter(first))
    first[attribute] = 1 - int(first[attribute])
    library_path = _write_payload(
        root / "tampered_library.json", library
    )
    with pytest.raises(RuntimeError, match="table hash"):
        auditor.audit_stage4_mixing(
            real_smoke["report_path"],
            library_path,
            root / "tampered_library_audit.json",
        )


def test_artifacts_are_no_overwrite_and_formal_entry_is_fail_closed(
    real_smoke,
):
    with pytest.raises(FileExistsError, match="不覆盖"):
        builder.build_state_library("smoke", real_smoke["library_path"])
    with pytest.raises(FileExistsError, match="不覆盖"):
        runner.run_stage4_mixing(
            "smoke",
            real_smoke["library_path"],
            real_smoke["report_path"],
        )
    with pytest.raises(FileExistsError, match="不覆盖"):
        auditor.audit_stage4_mixing(
            real_smoke["report_path"],
            real_smoke["library_path"],
            real_smoke["audit_path"],
        )
    with pytest.raises(PermissionError, match="尚未获显式授权"):
        builder.build_state_library(
            "qualification", real_smoke["root"] / "forbidden_library.json"
        )
    with pytest.raises(PermissionError, match="尚未获显式授权"):
        runner.run_stage4_mixing(
            "qualification",
            real_smoke["library_path"],
            real_smoke["root"] / "forbidden_report.json",
        )
