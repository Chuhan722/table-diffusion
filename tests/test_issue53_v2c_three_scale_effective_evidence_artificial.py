"""Contract tests for the frozen Issue #53 V2c artificial experiment."""

import ast
import copy
from dataclasses import replace
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_issue53_v2c_three_scale_effective_evidence as auditor
from scripts import validate_issue53_v2c_three_scale_effective_evidence as runner
from table_diffevo.adaptive_effective_evidence_v2c import (
    compute_v2c_adaptive_checkpoint_evidence,
)


TEST_SEED_NAMESPACE = (999, 53, 2, 3)


@pytest.fixture(scope="module")
def tiny_independent_matrix():
    """A non-formal replay: special namespace and only three repeats."""

    runner_matrix = runner.collect_matrix(
        families=runner.FAMILIES,
        repeat_count=3,
        maximum_length=runner.MAX_TRAJECTORY_LENGTH,
        checkpoints=runner.CHECKPOINTS,
        seed_namespace=TEST_SEED_NAMESPACE,
    )
    audit_matrix = auditor.collect_matrix_independent(
        families=auditor.FAMILIES,
        repeat_count=3,
        maximum_length=auditor.MAX_TRAJECTORY_LENGTH,
        checkpoints=auditor.CHECKPOINTS,
        seed_namespace=TEST_SEED_NAMESPACE,
    )
    return runner_matrix, audit_matrix


def test_plan_is_exact_and_does_not_instantiate_rng(monkeypatch) -> None:
    def forbidden_seed(*args, **kwargs):
        raise AssertionError("plan must not instantiate SeedSequence")

    monkeypatch.setattr(runner.np.random, "SeedSequence", forbidden_seed)
    plan = runner.build_plan()

    assert plan["mode"] == "plan_only_no_artificial_draws"
    assert plan["family_count"] == 5
    assert plan["trajectory_count"] == 10_000
    assert plan["checkpoint_classification_count"] == 150_000
    assert plan["scale_evaluation_count"] == 450_000
    assert plan["maximum_artificial_scalar_count"] == 20_480_000
    assert plan["real_data_accessed"] is False
    assert plan["generation_started"] is False
    assert plan["execution_started"] is False


def test_runner_and_auditor_reconstruct_the_same_frozen_protocol() -> None:
    runner_protocol = runner.frozen_protocol()
    audit_protocol = auditor.expected_protocol()

    assert runner_protocol == audit_protocol
    assert runner._sha256_json(runner_protocol) == auditor._sha256_json(
        audit_protocol
    )
    theory = {
        family.name: (
            family.theoretical_long_run_variance,
            family.theoretical_raw_ess_ratio,
        )
        for family in runner.FAMILIES
    }
    assert theory["iid"] == pytest.approx((1.0, 1.0))
    assert theory["ar1_phi_0p5"] == pytest.approx((3.0, 1.0 / 3.0))
    assert theory["ar1_phi_0p8"] == pytest.approx((9.0, 1.0 / 9.0))
    assert theory["ar1_phi_m0p5"] == pytest.approx((1.0 / 3.0, 3.0))
    assert theory["ar1_phi_0p95"] == pytest.approx((39.0, 1.0 / 39.0))
    assert runner_protocol["randomness"]["seed_sequence_prefix"] == [
        53,
        2,
        3,
    ]
    assert runner_protocol["provenance"][
        "v2b_negative_result_scientific_sha256"
    ] == runner.V2B_NEGATIVE_SCIENTIFIC_SHA256
    assert runner_protocol["adaptive_rule"]["earliest_ready_round"] == 384
    assert runner_protocol["adaptive_rule"][
        "current_state_is_revocable"
    ] is True


def test_formal_entry_has_no_scientific_override() -> None:
    assert set(inspect.signature(runner.run_artificial_protocol).parameters) == {
        "output_dir"
    }
    parsed = runner._build_parser().parse_args([
        "run",
        "--output-dir",
        "new-output",
    ])
    assert vars(parsed) == {"command": "run", "output_dir": "new-output"}

    with pytest.raises(SystemExit):
        runner._build_parser().parse_args([
            "run",
            "--output-dir",
            "new-output",
            "--repeat-count",
            "1",
        ])


def test_auditor_entry_has_only_artifact_paths() -> None:
    parsed = auditor._build_parser().parse_args([
        "--report",
        "three_scale_evidence_report.json",
        "--output",
        "independent_audit.json",
    ])
    assert vars(parsed) == {
        "report": "three_scale_evidence_report.json",
        "output": "independent_audit.json",
    }
    with pytest.raises(SystemExit):
        auditor._build_parser().parse_args([
            "--report",
            "report.json",
            "--output",
            "audit.json",
            "--repeat-count",
            "1",
        ])


def test_independent_generator_matches_on_nonformal_seed_and_is_separated() -> None:
    runner_values = runner.generate_artificial_trajectory(
        runner.FAMILIES[2],
        repeat_index=7,
        maximum_length=runner.MAX_TRAJECTORY_LENGTH,
        seed_namespace=TEST_SEED_NAMESPACE,
    )
    audit_values = auditor.generate_artificial_trajectory_independent(
        auditor.FAMILIES[2],
        repeat_index=7,
        maximum_length=auditor.MAX_TRAJECTORY_LENGTH,
        seed_namespace=TEST_SEED_NAMESPACE,
    )
    next_repeat = runner.generate_artificial_trajectory(
        runner.FAMILIES[2],
        repeat_index=8,
        maximum_length=runner.MAX_TRAJECTORY_LENGTH,
        seed_namespace=TEST_SEED_NAMESPACE,
    )

    assert np.array_equal(runner_values, audit_values)
    assert not np.array_equal(runner_values, next_repeat)
    assert runner_values.shape == (2048,)
    assert np.all(np.isfinite(runner_values))


def test_test_generation_cannot_use_formal_namespace(monkeypatch) -> None:
    original_seed_sequence = np.random.SeedSequence
    observed = []

    def guarded_seed_sequence(entropy):
        normalized = tuple(int(value) for value in entropy)
        if normalized[:3] == runner.SEED_NAMESPACE:
            raise AssertionError("tests must not instantiate the formal seed")
        observed.append(normalized)
        return original_seed_sequence(entropy)

    monkeypatch.setattr(
        runner.np.random,
        "SeedSequence",
        guarded_seed_sequence,
    )
    runner.generate_artificial_trajectory(
        runner.FAMILIES[0],
        repeat_index=0,
        maximum_length=16,
        seed_namespace=TEST_SEED_NAMESPACE,
    )
    auditor.generate_artificial_trajectory_independent(
        auditor.FAMILIES[0],
        repeat_index=0,
        maximum_length=16,
        seed_namespace=TEST_SEED_NAMESPACE,
    )

    assert len(observed) == 2
    assert all(seed[:4] == TEST_SEED_NAMESPACE for seed in observed)


@pytest.mark.parametrize(
    "repeat_index,maximum_length,seed_namespace",
    [
        (True, 2048, TEST_SEED_NAMESPACE),
        (-1, 2048, TEST_SEED_NAMESPACE),
        (0, 1, TEST_SEED_NAMESPACE),
        (0, 2048, (999, True)),
        (0, 2048, ()),
    ],
)
def test_generator_rejects_invalid_identity(
    repeat_index,
    maximum_length,
    seed_namespace,
) -> None:
    with pytest.raises(ValueError):
        runner.generate_artificial_trajectory(
            runner.FAMILIES[0],
            repeat_index=repeat_index,
            maximum_length=maximum_length,
            seed_namespace=seed_namespace,
        )
    with pytest.raises(ValueError):
        auditor.generate_artificial_trajectory_independent(
            auditor.FAMILIES[0],
            repeat_index=repeat_index,
            maximum_length=maximum_length,
            seed_namespace=seed_namespace,
        )


def test_runner_and_independent_boundary_suites_both_pass() -> None:
    runner_checks = runner.run_fixed_boundary_checks()
    audit_checks = auditor.run_fixed_boundary_checks_independent()

    assert runner_checks["passed"] is True
    assert audit_checks["passed"] is True
    assert runner_checks == audit_checks
    assert len(runner_checks["checks"]) == 22


def test_tiny_nonformal_matrix_matches_independent_replay_exactly(
    tiny_independent_matrix,
) -> None:
    runner_matrix, audit_matrix = tiny_independent_matrix

    assert len(runner_matrix[0]) == 15
    assert len(runner_matrix[1]) == 75
    assert len(runner_matrix[2]) == 5
    assert runner_matrix == audit_matrix
    assert all(
        "post_first_incompatibility_trajectory_count" in summary
        and "cap_three_scale_compatible_count" in summary
        and "cap_adaptive_numerically_estimable_count" in summary
        for summary in runner_matrix[2]
    )

    runner_acceptance = runner.build_acceptance_gates(
        *runner_matrix,
        runner.run_fixed_boundary_checks(),
    )
    audit_acceptance = auditor.build_acceptance_gates_independent(
        *audit_matrix,
        auditor.run_fixed_boundary_checks_independent(),
    )
    assert runner_acceptance == audit_acceptance
    assert isinstance(
        runner_acceptance["cost_gates"]["main_pooled_resource_mean"],
        bool,
    )
    assert isinstance(
        runner_acceptance["cost_gates"][
            "main_pooled_resource_mean_value"
        ],
        float,
    )


def test_checkpoint_contract_checks_disagreement_outputs_too() -> None:
    indices = np.arange(1, 257, dtype=np.int64)
    values = np.linspace(0.0, 1.0, 256)
    result = compute_v2c_adaptive_checkpoint_evidence(indices, values)
    assert result.b1_numerically_estimable is True
    assert result.b2_numerically_estimable is True
    assert result.b3_numerically_estimable is True
    valid_row = runner.checkpoint_record(
        runner.FAMILIES[0],
        0,
        values,
        result,
    )
    broken = replace(
        result,
        official_long_run_variance=(
            result.official_long_run_variance + 1.0
        ),
    )
    broken_row = runner.checkpoint_record(
        runner.FAMILIES[0],
        0,
        values,
        broken,
    )

    assert valid_row["contract_violation"] is False
    assert broken_row["contract_violation"] is True


def test_checkpoint_contract_checks_consecutive_state_too() -> None:
    indices = np.arange(1, 385, dtype=np.int64)
    values = np.sin(0.5 * np.arange(384, dtype=np.float64))
    result = compute_v2c_adaptive_checkpoint_evidence(
        indices,
        values,
        previous_three_scale_compatible=False,
    )
    assert result.three_scale_compatible is True
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "awaiting_consecutive_multiscale_evidence"

    valid_row = runner.checkpoint_record(
        runner.FAMILIES[0],
        0,
        values,
        result,
    )
    broken = replace(
        result,
        adaptive_numerically_estimable=True,
        reason=None,
    )
    broken_row = runner.checkpoint_record(
        runner.FAMILIES[0],
        0,
        values,
        broken,
    )

    assert valid_row["contract_violation"] is False
    assert broken_row["contract_violation"] is True


def test_acceptance_rejects_duplicate_identity_and_summary_tamper(
    tiny_independent_matrix,
) -> None:
    runner_matrix, _ = tiny_independent_matrix
    trajectories, checkpoints, families = copy.deepcopy(runner_matrix)
    trajectories[1] = copy.deepcopy(trajectories[0])
    with pytest.raises(ValueError, match="trajectory rows"):
        runner.build_acceptance_gates(
            trajectories,
            checkpoints,
            families,
            runner.run_fixed_boundary_checks(),
        )

    trajectories, checkpoints, families = copy.deepcopy(runner_matrix)
    families[0]["resource_round_count_mean"] += 1.0
    with pytest.raises(ValueError, match="family summary"):
        runner.build_acceptance_gates(
            trajectories,
            checkpoints,
            families,
            runner.run_fixed_boundary_checks(),
        )


def test_runner_rejects_dirty_tree_before_hashing_or_draws(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_git_text",
        lambda root, *arguments: "?? dirty" if arguments[0] == "status" else "",
    )

    def forbidden_hash(path):
        raise AssertionError("dirty-tree rejection must precede source hashing")

    monkeypatch.setattr(runner, "_sha256_file", forbidden_hash)
    with pytest.raises(RuntimeError, match="clean worktree"):
        runner.build_execution_manifest(tmp_path)


def test_manifest_binds_every_source_and_archived_v2b_result(
    monkeypatch,
) -> None:
    root = Path(runner.__file__).resolve().parents[1]
    original_git_text = runner._git_text

    def clean_status(manifest_root, *arguments):
        if arguments[0] == "status":
            return ""
        return original_git_text(manifest_root, *arguments)

    monkeypatch.setattr(runner, "_git_text", clean_status)
    manifest = runner.build_execution_manifest(root)

    assert set(manifest["source_sha256"]) == {
        str(path) for path in runner.SOURCE_PATHS
    }
    assert manifest["upstream_v2b_negative_result"] == {
        "report_path": str(runner.V2B_NEGATIVE_REPORT),
        "report_sha256": runner._sha256_file(
            root / runner.V2B_NEGATIVE_REPORT
        ),
        "scientific_result_sha256": (
            runner.V2B_NEGATIVE_SCIENTIFIC_SHA256
        ),
        "status": "candidate_failed",
        "audit_path": str(runner.V2B_NEGATIVE_AUDIT),
        "audit_sha256": runner._sha256_file(
            root / runner.V2B_NEGATIVE_AUDIT
        ),
        "audit_passed": True,
    }


def test_runner_rejects_wrong_archived_v2b_provenance(monkeypatch) -> None:
    root = Path(runner.__file__).resolve().parents[1]
    monkeypatch.setattr(
        runner,
        "_git_text",
        lambda manifest_root, *arguments: (
            "" if arguments[0] == "status" else "test-commit"
        ),
    )

    def broken_upstream(path):
        if path == root / runner.V2B_NEGATIVE_REPORT:
            return {
                "status": "candidate_failed",
                "scientific_result_sha256": "0" * 64,
            }
        return {
            "passed": True,
            "recorded_scientific_result_sha256": (
                runner.V2B_NEGATIVE_SCIENTIFIC_SHA256
            ),
        }

    monkeypatch.setattr(runner, "_load_json_strict", broken_upstream)
    with pytest.raises(RuntimeError, match="V2b negative-result provenance"):
        runner.build_execution_manifest(root)


def test_auditor_source_has_no_runner_or_project_core_imports() -> None:
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert not any(
        name.startswith(
            "scripts.validate_issue53_v2c_three_scale_effective_evidence"
        )
        for name in imported
    )
    assert "table_diffevo.effective_evidence" not in imported
    assert "table_diffevo.adaptive_effective_evidence" not in imported
    assert "table_diffevo.adaptive_effective_evidence_v2c" not in imported


def test_strict_json_loader_rejects_duplicate_and_nonfinite_values(
    tmp_path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        auditor._load_json_strict(duplicate)
    with pytest.raises(ValueError, match="non-finite JSON"):
        auditor._load_json_strict(nonfinite)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._load_json_strict(duplicate)
    with pytest.raises(ValueError, match="non-finite JSON"):
        runner._load_json_strict(nonfinite)


def test_special_seed_end_to_end_report_passes_independent_audit(
    monkeypatch,
    tmp_path,
) -> None:
    """Exercise formal entry plumbing without using its seed or matrix size."""

    monkeypatch.setattr(runner, "REPEAT_COUNT", 2)
    monkeypatch.setattr(auditor, "REPEAT_COUNT", 2)
    monkeypatch.setattr(runner, "SEED_NAMESPACE", TEST_SEED_NAMESPACE)
    monkeypatch.setattr(auditor, "SEED_NAMESPACE", TEST_SEED_NAMESPACE)
    root = Path(runner.__file__).resolve().parents[1]
    git_commit = runner._git_text(root, "rev-parse", "HEAD")

    def test_manifest(manifest_root):
        protocol = runner.frozen_protocol()
        return {
            "contract_version": runner.ARTIFICIAL_PROTOCOL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "git_worktree_clean_including_untracked": True,
            "protocol_sha256": runner._sha256_json(protocol),
            "source_sha256": {
                str(path): runner._sha256_file(manifest_root / path)
                for path in runner.SOURCE_PATHS
            },
            "upstream_v2b_negative_result": {
                "report_path": str(runner.V2B_NEGATIVE_REPORT),
                "report_sha256": runner._sha256_file(
                    manifest_root / runner.V2B_NEGATIVE_REPORT
                ),
                "scientific_result_sha256": (
                    runner.V2B_NEGATIVE_SCIENTIFIC_SHA256
                ),
                "status": "candidate_failed",
                "audit_path": str(runner.V2B_NEGATIVE_AUDIT),
                "audit_sha256": runner._sha256_file(
                    manifest_root / runner.V2B_NEGATIVE_AUDIT
                ),
                "audit_passed": True,
            },
            "environment": {"test_only": True},
            "protocol": protocol,
        }

    monkeypatch.setattr(runner, "build_execution_manifest", test_manifest)
    _, report_path, report = runner.run_artificial_protocol(
        tmp_path / "special-seed-run"
    )
    audit_path, audit = auditor.audit_artificial_report(
        report_path,
        report_path.parent / "independent_audit.json",
    )

    assert report["execution"]["trajectory_count"] == 10
    assert report["execution"]["scale_evaluation_count"] == 450
    assert report["status"] == "candidate_failed"
    assert audit["passed"] is True
    assert audit["recorded_scientific_result_sha256"] == audit[
        "recomputed_scientific_result_sha256"
    ]
    assert report["manifest_path"] == "protocol_manifest.json"
    assert audit["report_path"] == report_path.name
    assert audit["manifest_path"] == "protocol_manifest.json"
    assert audit["independence"] == {
        "imports_formal_runner": False,
        "imports_project_v2_core": False,
        "imports_project_v2b_core": False,
        "imports_project_v2c_core": False,
        "regenerates_artificial_trajectories": True,
        "recomputes_obm_directly": True,
    }
    assert audit_path.exists()
    with pytest.raises(FileExistsError):
        auditor.audit_artificial_report(report_path, audit_path)

    tampered = copy.deepcopy(report)
    tampered["scientific_payload"]["trajectory_first_ready_rows"][0][
        "resource_round_count"
    ] += 1
    tampered["scientific_result_sha256"] = runner._sha256_json(
        tampered["scientific_payload"]
    )
    tampered_path = report_path.parent / "tampered_report.json"
    tampered_path.write_text(
        json.dumps(tampered, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _, tampered_audit = auditor.audit_artificial_report(
        tampered_path,
        report_path.parent / "tampered_audit.json",
    )
    assert tampered_audit["passed"] is False
    assert tampered_audit["checks"]["scientific_payload_exact"] is False
    assert any(
        path.startswith("$.trajectory_first_ready_rows[0]")
        for path in tampered_audit["first_mismatch_paths"]
    )

    absolute_manifest = copy.deepcopy(report)
    absolute_manifest["manifest_path"] = str(
        (report_path.parent / "protocol_manifest.json").resolve()
    )
    absolute_path = report_path.parent / "absolute_manifest_report.json"
    absolute_path.write_text(
        json.dumps(absolute_manifest, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="portable sibling filename"):
        auditor.audit_artificial_report(
            absolute_path,
            report_path.parent / "absolute_manifest_audit.json",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="portable sibling artifact"):
        auditor.audit_artificial_report(
            report_path,
            outside / "audit.json",
        )


def test_mismatch_paths_identifies_nested_scientific_tamper() -> None:
    actual = {"rows": [{"n": 256, "mcse": 0.1}]}
    expected = copy.deepcopy(actual)
    expected["rows"][0]["mcse"] = 0.2

    assert auditor._mismatch_paths(actual, expected) == [
        "$.rows[0].mcse"
    ]
