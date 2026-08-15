"""Issue #53 Stage 2B 封存验证采集入口测试。"""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import collect_issue53_stage2b_validation as validation
from tests.test_stationarity import _make_trace


def test_plan_has_no_generation_or_validation_access(tmp_path):
    plan = validation.build_collection_plan(tmp_path / "output")

    assert plan["trajectory_count"] == 20
    assert plan["round_budget_per_trajectory"] == 8000
    assert plan["total_round_budget"] == 160000
    assert plan["requires_exactly_one_visible_cuda_gpu"] is True
    assert plan["detector_replay_during_collection"] is False
    assert plan["validation_seed_accessed"] is False
    assert plan["generation_started"] is False
    assert set(inspect.signature(validation.build_collection_plan).parameters) == {
        "output_dir"
    }


def test_wrong_protocol_sha_fails_before_environment_or_output(
    tmp_path, monkeypatch
):
    touched = False

    def forbidden_environment():
        nonlocal touched
        touched = True
        raise AssertionError("SHA guard must run first")

    monkeypatch.setattr(
        validation, "formal_environment_manifest", forbidden_environment
    )

    with pytest.raises(ValueError, match="显式确认"):
        validation.run_frozen_validation_collection(
            tmp_path / "output", "wrong"
        )
    assert touched is False
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("visible", [None, "", "0,1"])
def test_formal_environment_requires_one_explicit_visible_gpu(
    visible, monkeypatch
):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    with pytest.raises(RuntimeError, match="一张卡"):
        validation._single_visible_gpu_manifest()


def test_validation_run_artifacts_are_atomic_and_role_strict(
    tmp_path, monkeypatch
):
    trace = _make_trace([[2.0, 1.0]] * 3, moving=True)
    workload = SimpleNamespace(
        name="test_300x10",
        input_sha256={"schema": "a", "queries": "b", "marginals": "c"},
        query_identity_sha256=trace.query_identity_sha256,
        target_identity_sha256=trace.target_identity_sha256,
    )
    diagnostics = {
        "params": {
            "seed": 220,
            "factorized_gibbs_sweeps": 0,
            "factorized_gibbs_max_order": 3,
            "factorized_gibbs_logit_clip": 30.0,
            "factorized_gibbs_use_compiled_workload": False,
        },
        "reference_process_contract": {"version": "test"},
        "rounds_run": 2,
        "termination_reason": "max_rounds",
        "candidate_evaluation_count": 2,
        "initial_table_sha256": "1" * 64,
        "primary_rng_state_sha256": "2" * 64,
        "factorized_gibbs_rng_state_sha256": None,
        "direction_logit_evaluated_count_history": [3, 3],
        "direction_logit_clipped_count_history": [0, 0],
        "factorized_gibbs_conditional_logit_evaluated_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
    }
    environment = {
        "git_commit": "a" * 40,
        "git_worktree_clean_including_untracked": True,
        "gpu": {"torch_visible_device_count": 1},
    }
    monkeypatch.setattr(validation.protocol, "VALIDATION_ROUND_BUDGET", 2)
    monkeypatch.setattr(
        validation.protocol,
        "validation_protocol_sha256",
        lambda: "b" * 64,
    )

    destination = validation.save_validation_run(
        tmp_path,
        workload=workload,
        kernel="independent",
        seed=220,
        preflight={"direction_reference_scale": 1.0},
        diagnostics=diagnostics,
        trace=trace,
        final_table=pd.DataFrame({"x": [0, 1]}),
        elapsed_sec=0.1,
        environment=environment,
    )
    manifest = json.loads(
        (destination / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["formal_heldout_validation"] is True
    assert manifest["validation_protocol_sha256"] == "b" * 64
    assert manifest["maximum_round_budget"] == 2
    assert not list(destination.parent.glob(".independent.partial-*"))
    with pytest.raises(FileExistsError):
        validation.save_validation_run(
            tmp_path,
            workload=workload,
            kernel="independent",
            seed=220,
            preflight={"direction_reference_scale": 1.0},
            diagnostics=diagnostics,
            trace=trace,
            final_table=pd.DataFrame({"x": [0, 1]}),
            elapsed_sec=0.1,
            environment=environment,
        )
