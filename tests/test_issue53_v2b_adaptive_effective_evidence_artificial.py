"""Contract tests for the frozen Issue #53 V2b artificial experiment."""

import ast
import copy
from dataclasses import replace
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_issue53_v2b_adaptive_effective_evidence as auditor
from scripts import validate_issue53_v2b_adaptive_effective_evidence as runner
from table_diffevo.adaptive_effective_evidence import (
    compute_v2b_adaptive_checkpoint_evidence,
)


TEST_SEED_NAMESPACE = (999, 53, 2, 2)


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
    assert plan["scale_evaluation_count"] == 300_000
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
    assert len(runner_checks["checks"]) == 16


def test_tiny_nonformal_matrix_matches_independent_replay_exactly(
    tiny_independent_matrix,
) -> None:
    runner_matrix, audit_matrix = tiny_independent_matrix

    assert len(runner_matrix[0]) == 15
    assert len(runner_matrix[1]) == 75
    assert len(runner_matrix[2]) == 5
    assert runner_matrix == audit_matrix

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
    result = compute_v2b_adaptive_checkpoint_evidence(indices, values)
    assert result.short_numerically_estimable is True
    assert result.long_numerically_estimable is True
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
            "scripts.validate_issue53_v2b_adaptive_effective_evidence"
        )
        for name in imported
    )
    assert "table_diffevo.effective_evidence" not in imported
    assert "table_diffevo.adaptive_effective_evidence" not in imported


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


def test_manifest_path_is_portable_and_legacy_absolute_metadata_is_supported(
    tmp_path,
) -> None:
    report_path = tmp_path / "adaptive_evidence_report.json"
    manifest_path = tmp_path / "protocol_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    expected = manifest_path.resolve()
    assert (
        auditor._resolve_sibling_manifest(
            report_path,
            "protocol_manifest.json",
        )
        == expected
    )
    assert (
        auditor._resolve_sibling_manifest(
            report_path,
            "/legacy/author/output/protocol_manifest.json",
        )
        == expected
    )
    with pytest.raises(ValueError, match="portable sibling"):
        auditor._resolve_sibling_manifest(report_path, "../protocol_manifest.json")
    with pytest.raises(ValueError, match="must name"):
        auditor._resolve_sibling_manifest(
            report_path,
            "/legacy/author/output/other.json",
        )


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
                for path in (
                    runner.PROTOCOL_DOCUMENT,
                    runner.DESIGN_DOCUMENT,
                    runner.V2_CORE_MODULE,
                    runner.V2B_CORE_MODULE,
                    runner.RUNNER_MODULE,
                    runner.AUDITOR_MODULE,
                    runner.V2_CORE_TEST_MODULE,
                    runner.V2B_CORE_TEST_MODULE,
                    runner.RUNNER_TEST_MODULE,
                )
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
    assert report["status"] == "candidate_failed"
    assert report["manifest_path"] == "protocol_manifest.json"
    assert audit["passed"] is True
    assert audit["report_path"] == report_path.name
    assert audit["manifest_path"] == "protocol_manifest.json"
    assert audit["recorded_scientific_result_sha256"] == audit[
        "recomputed_scientific_result_sha256"
    ]
    assert audit_path.exists()
    with pytest.raises(FileExistsError):
        auditor.audit_artificial_report(report_path, audit_path)


def test_mismatch_paths_identifies_nested_scientific_tamper() -> None:
    actual = {"rows": [{"n": 256, "mcse": 0.1}]}
    expected = copy.deepcopy(actual)
    expected["rows"][0]["mcse"] = 0.2

    assert auditor._mismatch_paths(actual, expected) == [
        "$.rows[0].mcse"
    ]
