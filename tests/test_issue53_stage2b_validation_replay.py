"""Issue #53 Stage 2B 冻结 V1 validation 正式回放测试。"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import issue53_stage2b_validation_protocol as protocol
from scripts import replay_issue53_stage2b_validation as validation_replay
from table_diffevo.stationarity import StationarityDetectorConfig
from tests.test_stationarity import _make_trace


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_run_manifest(dataset: str, seed: int, kernel: str):
    pair_name = f"{dataset}-{seed}"
    return {
        "dataset": dataset,
        "seed": seed,
        "kernel": kernel,
        "validation_protocol_sha256": (
            protocol.validation_protocol_sha256()
        ),
        "query_identity_sha256": "1" * 64,
        "target_identity_sha256": "2" * 64,
        "s0_preflight": {
            "direction_reference_scale": float(seed),
            "primary_rng_post_initialization_state_sha256": _identity(
                f"rng-{pair_name}"
            ),
        },
        "run_summary": {
            "initial_table_sha256": _identity(f"table-{pair_name}"),
            "clip_audit": {
                "direction": {
                    "evaluated_count": 10,
                    "clipped_count": 0,
                },
                "gibbs_conditional": {
                    "evaluated_count": 5,
                    "clipped_count": 0,
                },
            },
        },
        "environment": {
            "git_commit": (
                validation_replay.REFERENCE_VALIDATION_COLLECTION_GIT_COMMIT
            ),
            "git_worktree_clean_including_untracked": True,
            "python": "3.11",
            "gpu": {
                "device_name": "fake GPU",
                "torch_visible_device_count": 1,
            },
        },
    }


def _fake_collection(tmp_path, monkeypatch):
    input_dir = tmp_path / "validation"
    manifests_by_dir = {}
    run_hashes = {}
    for row in protocol.expected_validation_cells():
        dataset = row["dataset"]
        seed = int(row["seed"])
        kernel = row["kernel"]
        run_dir = input_dir / dataset / f"seed_{seed}" / kernel
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {"dataset": dataset, "seed": seed, "kernel": kernel},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        cell_name = f"{dataset}/seed_{seed}/{kernel}"
        run_hashes[cell_name] = _sha256_file(manifest_path)
        manifests_by_dir[run_dir] = _fake_run_manifest(
            dataset, seed, kernel
        )

    collection_manifest = {
        "contract_version": (
            validation_replay.collection.
            VALIDATION_COLLECTION_CONTRACT_VERSION
        ),
        "validation_protocol_sha256": (
            protocol.validation_protocol_sha256()
        ),
        "formal_validation_collection_complete": True,
        "trajectory_count": 20,
        "rounds_per_trajectory": protocol.VALIDATION_ROUND_BUDGET,
        "total_round_count": 20 * protocol.VALIDATION_ROUND_BUDGET,
        "detector_replay_performed": False,
        "partial_validation_classification_read": False,
        "run_manifest_sha256": run_hashes,
        "collection_elapsed_sec": 1.0,
    }
    collection_path = input_dir / "collection_manifest.json"
    collection_path.write_text(
        json.dumps(collection_manifest, sort_keys=True),
        encoding="utf-8",
    )

    calls = []

    def fake_audit(run_dir, *, expected_git_commit=None):
        calls.append(run_dir)
        assert expected_git_commit == (
            validation_replay.REFERENCE_VALIDATION_COLLECTION_GIT_COMMIT
        )
        return manifests_by_dir[run_dir]

    monkeypatch.setattr(
        validation_replay.collection,
        "audit_validation_run",
        fake_audit,
    )
    return {
        "input_dir": input_dir,
        "collection_path": collection_path,
        "collection_manifest": collection_manifest,
        "manifests_by_dir": manifests_by_dir,
        "calls": calls,
    }


def _small_config() -> StationarityDetectorConfig:
    return StationarityDetectorConfig(
        window_size=2,
        query_mean_shift_tolerance=0.1,
        query_p95_shift_tolerance=0.1,
        l1_mean_shift_tolerance=0.1,
        l1_p90_minus_p10_shift_tolerance=0.1,
        unique_row_rate_tolerance=0.1,
        normalized_row_entropy_tolerance=0.1,
        minimum_active_round_rate=0.5,
        minimum_mean_changed_row_fraction=0.5,
        stall_patience_checks=2,
    )


def _fake_replay_inputs():
    rows = []
    for cell in protocol.expected_validation_cells():
        dataset = cell["dataset"]
        seed = int(cell["seed"])
        kernel = cell["kernel"]
        rows.append(validation_replay.AuditedValidationInput(
            dataset=dataset,
            kernel=kernel,
            seed=seed,
            run_dir=Path("unused") / dataset / f"seed_{seed}" / kernel,
            manifest=_fake_run_manifest(dataset, seed, kernel),
            manifest_sha256=_identity(f"{dataset}-{seed}-{kernel}"),
        ))
    return rows


def _patch_small_protocol(monkeypatch, *, rounds):
    config = _small_config()
    monkeypatch.setattr(protocol, "FROZEN_DETECTOR_CONFIG", config)
    monkeypatch.setattr(protocol, "VALIDATION_ROUND_BUDGET", rounds)
    monkeypatch.setattr(
        validation_replay.calibration,
        "WINDOW_SIZE",
        config.window_size,
    )
    return config


def test_plan_reads_no_validation_and_exposes_no_threshold_override():
    plan = validation_replay.build_plan(Path("input"), Path("output"))

    assert plan["mode"] == "plan_only_no_validation_trace_read"
    assert plan["expected_trajectory_count"] == 20
    assert plan["expected_full_check_count_per_trajectory"] == 18
    assert plan["requires_confirmed_protocol_sha256"] is True
    assert plan["requires_confirmed_collection_manifest_sha256"] is True
    assert plan["threshold_override_parameters_present"] is False
    assert plan["validation_traces_read"] is False
    assert plan["classification_output_present"] is False
    assert set(
        inspect.signature(validation_replay.generate_report).parameters
    ) == {
        "input_dir",
        "output_dir",
        "confirmed_protocol_sha256",
        "confirmed_collection_manifest_sha256",
    }


def test_wrong_protocol_confirmation_fails_before_environment_or_input(
    tmp_path, monkeypatch
):
    touched = False

    def forbidden_environment():
        nonlocal touched
        touched = True
        raise AssertionError("protocol guard must run first")

    monkeypatch.setattr(
        validation_replay,
        "analysis_environment_manifest",
        forbidden_environment,
    )
    with pytest.raises(ValueError, match="显式确认"):
        validation_replay.generate_report(
            tmp_path / "input",
            tmp_path / "output",
            "wrong",
            "a" * 64,
        )
    assert touched is False
    assert not (tmp_path / "output").exists()


def test_formal_report_rejects_dirty_tree_before_reading_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation_replay,
        "analysis_environment_manifest",
        lambda: {"git_worktree_clean_including_untracked": False},
    )
    touched = False

    def forbidden_build(*_args):
        nonlocal touched
        touched = True
        raise AssertionError("dirty-tree guard must run first")

    monkeypatch.setattr(validation_replay, "build_report", forbidden_build)
    with pytest.raises(RuntimeError, match="干净工作树"):
        validation_replay.generate_report(
            tmp_path / "input",
            tmp_path / "output",
            protocol.validation_protocol_sha256(),
            "a" * 64,
        )
    assert touched is False


def test_complete_collection_is_reaudited_and_bound_to_confirmed_sha(
    tmp_path, monkeypatch
):
    fixture = _fake_collection(tmp_path, monkeypatch)
    confirmed_sha = _sha256_file(fixture["collection_path"])

    inputs, descriptor = validation_replay.audit_validation_collection(
        fixture["input_dir"], confirmed_sha
    )

    assert len(inputs) == 20
    assert len(fixture["calls"]) == 20
    assert descriptor["collection_manifest_sha256"] == confirmed_sha
    assert descriptor["trajectory_count"] == 20
    assert descriptor["total_round_count"] == 160000
    assert descriptor["paired_s0_s0_rng_binding_count"] == 10
    assert len(descriptor["execution_environment_groups"]) == 1


def test_collection_hash_confirmation_fails_before_per_run_audit(
    tmp_path, monkeypatch
):
    fixture = _fake_collection(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="显式确认值"):
        validation_replay.audit_validation_collection(
            fixture["input_dir"], "f" * 64
        )
    assert fixture["calls"] == []


def test_collection_rejects_run_manifest_tampering(
    tmp_path, monkeypatch
):
    fixture = _fake_collection(tmp_path, monkeypatch)
    confirmed_sha = _sha256_file(fixture["collection_path"])
    manifest_path = next(
        fixture["input_dir"].rglob("run_manifest.json")
    )
    manifest_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="哈希与 collection"):
        validation_replay.audit_validation_collection(
            fixture["input_dir"], confirmed_sha
        )


@pytest.mark.parametrize(
    "field",
    ["detector_replay_performed", "partial_validation_classification_read"],
)
def test_collection_rejects_preexisting_partial_classification(
    tmp_path, monkeypatch, field
):
    fixture = _fake_collection(tmp_path, monkeypatch)
    fixture["collection_manifest"][field] = True
    fixture["collection_path"].write_text(
        json.dumps(fixture["collection_manifest"], sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="封存状态"):
        validation_replay.audit_validation_collection(
            fixture["input_dir"],
            _sha256_file(fixture["collection_path"]),
        )
    assert fixture["calls"] == []


def test_collection_rejects_cross_kernel_pair_binding_mismatch(
    tmp_path, monkeypatch
):
    fixture = _fake_collection(tmp_path, monkeypatch)
    mismatched = next(
        manifest
        for manifest in fixture["manifests_by_dir"].values()
        if manifest["kernel"] == "factorized_gibbs"
    )
    mismatched["s0_preflight"]["direction_reference_scale"] += 1.0

    with pytest.raises(RuntimeError, match="共享 s0/S0"):
        validation_replay.audit_validation_collection(
            fixture["input_dir"],
            _sha256_file(fixture["collection_path"]),
        )


def test_full_replay_passes_all_twenty_with_real_detector_formula(
    monkeypatch,
):
    _patch_small_protocol(monkeypatch, rounds=8)
    trace = _make_trace([[2.0, 1.0]] * 8, moving=True)
    monkeypatch.setattr(
        validation_replay,
        "load_stationarity_trace",
        lambda _path: trace,
    )

    trajectories, checks, result = (
        validation_replay.replay_full_validation(_fake_replay_inputs())
    )

    assert len(trajectories) == 20
    assert len(checks) == 40
    assert {row["status"] for row in trajectories} == {
        "stationary_qualified"
    }
    assert {row["candidate_round_index"] for row in trajectories} == {8}
    assert result["classification"] == (
        "supports_frozen_detector_on_validation"
    )


def test_stall_round_is_not_misreported_as_candidate_round(monkeypatch):
    _patch_small_protocol(monkeypatch, rounds=8)
    trace = _make_trace([[2.0, 1.0]] * 8, moving=False)
    monkeypatch.setattr(
        validation_replay,
        "load_stationarity_trace",
        lambda _path: trace,
    )

    trajectories, _checks, result = (
        validation_replay.replay_full_validation(_fake_replay_inputs())
    )

    assert {row["status"] for row in trajectories} == {"stalled"}
    assert {row["candidate_round_index"] for row in trajectories} == {None}
    assert {row["stall_round_index"] for row in trajectories} == {8}
    assert result["classification"] == (
        "does_not_support_frozen_detector_on_validation"
    )
    assert result["acceptance_gates"]["stalled_trajectory_count"] == 20


def test_four_post_candidate_instabilities_fail_redrift_gate(monkeypatch):
    _patch_small_protocol(monkeypatch, rounds=16)
    query_vectors = (
        [[2.0, 1.0]] * 8
        + [[2.5, 1.0]] * 2
        + [[3.0, 1.0]] * 2
        + [[3.5, 1.0]] * 2
        + [[4.0, 1.0]] * 2
    )
    trace = _make_trace(query_vectors, moving=True)
    monkeypatch.setattr(
        validation_replay,
        "load_stationarity_trace",
        lambda _path: trace,
    )

    trajectories, _checks, result = (
        validation_replay.replay_full_validation(_fake_replay_inputs())
    )

    assert {row["candidate_round_index"] for row in trajectories} == {8}
    assert {
        row["maximum_post_candidate_unstable_streak"]
        for row in trajectories
    } == {4}
    assert all(row["persistent_redrift_detected"] for row in trajectories)
    assert result["classification"] == (
        "does_not_support_frozen_detector_on_validation"
    )


def test_report_publish_is_atomic_hashed_and_non_overwriting(
    tmp_path, monkeypatch
):
    environment = {
        "analysis_git_commit": "a" * 40,
        "git_worktree_clean_including_untracked": True,
    }
    monkeypatch.setattr(
        validation_replay,
        "analysis_environment_manifest",
        lambda: environment,
    )
    report = {
        "source_audit": {"trajectory_count": 20},
        "validation_replay": {
            "result": {
                "classification": (
                    "supports_frozen_detector_on_validation"
                )
            }
        },
    }
    monkeypatch.setattr(
        validation_replay,
        "build_report",
        lambda *_args: (
            report,
            pd.DataFrame([{"status": "stationary_qualified"}]),
            pd.DataFrame([{"round_index": 1600}]),
        ),
    )
    output_dir = tmp_path / "report"
    collection_sha = "b" * 64

    destination = validation_replay.generate_report(
        tmp_path / "input",
        output_dir,
        protocol.validation_protocol_sha256(),
        collection_sha,
    )
    manifest = json.loads(
        (destination / "report_manifest.json").read_text(encoding="utf-8")
    )

    assert destination == output_dir
    assert manifest["formal_frozen_v1_validation_report"] is True
    assert manifest["collection_manifest_sha256"] == collection_sha
    assert set(manifest["artifacts"]) == {
        "validation_report.json",
        "trajectory_results.csv",
        "full_replay_checks.csv",
    }
    for relative, identity in manifest["artifacts"].items():
        assert _sha256_file(output_dir / relative) == identity["sha256"]
    assert not list(tmp_path.glob(".report.partial-*"))

    with pytest.raises(FileExistsError):
        validation_replay.generate_report(
            tmp_path / "input",
            output_dir,
            protocol.validation_protocol_sha256(),
            collection_sha,
        )
