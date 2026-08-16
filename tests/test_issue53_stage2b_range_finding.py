"""Issue #53 Stage 2B 初始量程协议与防误运行测试。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import collect_issue53_stage2b_range_finding as collector
from table_diffevo.stationarity import load_stationarity_trace
from tests.test_stationarity import _make_trace


def test_protocol_freezes_the_three_agreed_decisions():
    protocol = collector.frozen_protocol_manifest()

    assert protocol["scope"] == {
        "datasets": ["test_300x10", "nltcs"],
        "kernels": ["independent", "factorized_gibbs"],
        "one_common_detector_required": True,
        "per_cell_detector_tuning": False,
    }
    assert protocol["seed_split"]["development"] == [200, 201, 202]
    assert protocol["seed_split"]["validation"] == [
        220, 221, 222, 223, 224
    ]
    assert protocol["seed_split"]["validation_status"] == (
        "sealed_until_detector_config_frozen"
    )
    assert protocol["datasets"]["test_300x10"]["query_count"] == 50
    assert protocol["datasets"]["nltcs"]["query_count"] == 1001

    generator = protocol["generator"]
    assert generator["fixed_alpha"] == 16.0
    assert generator["rho"] == 0.01
    assert generator["eta"] == 0.5
    assert generator["mu"] == 0.01
    assert generator["diffusion_direction_strength_tau"] == 2.0
    assert generator["diffusion_direction_logit_clip"] == 30.0
    assert generator["no_gate"] is True
    assert generator["online_stationarity_stop"] is False
    assert generator["s0"]["shared_within_dataset_seed_across_kernels"]
    assert not generator["s0"]["preflight_states_in_stationarity_trace"]
    assert protocol["detector"]["validation_may_run"] is False
    assert protocol["collection"]["development_round_budget"] == 8000
    assert protocol["collection"]["budget_status"] == "frozen"
    json.dumps(protocol, ensure_ascii=False, allow_nan=False)


def test_only_algorithmic_kernel_difference_is_sweep_count():
    independent = collector.KERNELS["independent"]
    factorized = collector.KERNELS["factorized_gibbs"]

    differing = {
        key
        for key in independent
        if independent[key] != factorized[key]
    }
    assert differing == {
        "factorized_gibbs_sweeps",
        "factorized_gibbs_use_compiled_workload",
    }
    assert differing - {
        "factorized_gibbs_use_compiled_workload"
    } == {"factorized_gibbs_sweeps"}
    assert independent["factorized_gibbs_sweeps"] == 0
    assert factorized["factorized_gibbs_sweeps"] == 8
    assert independent["factorized_gibbs_logit_clip"] == 30.0
    assert factorized["factorized_gibbs_logit_clip"] == 30.0
    assert independent["factorized_gibbs_use_compiled_workload"] is False
    assert factorized["factorized_gibbs_use_compiled_workload"] is True
    assert collector.frozen_protocol_manifest()["generator"][
        "factor_builder"
    ]["compiled_batch_role"] == (
        "output_equivalent_performance_implementation"
    )


def test_plan_has_twelve_development_cells_and_no_validation_seed():
    plan = collector.build_execution_plan(
        list(collector.DATASETS),
        list(collector.KERNELS),
        list(collector.DEVELOPMENT_SEEDS),
    )

    assert plan["trajectory_count"] == 12
    assert plan["development_execution_locked"] is False
    assert plan["development_round_budget"] == 8000
    assert plan["validation_seeds_touched"] is False
    assert {cell["seed"] for cell in plan["cells"]} == {200, 201, 202}


@pytest.mark.parametrize(
    "seeds",
    [
        [220],
        [224],
        [0],
        [200, 200],
        [True],
        [],
    ],
)
def test_development_seed_guard_rejects_nondevelopment_seed(seeds):
    with pytest.raises(ValueError):
        collector.validate_development_seeds(seeds)


@pytest.mark.parametrize("reserved", [200, 202, 220, 224])
def test_smoke_cannot_consume_any_reserved_seed(reserved):
    with pytest.raises(ValueError, match="未保留"):
        collector.validate_smoke_request(
            ["test_300x10"], [reserved], 2
        )


def test_development_rejects_nonfrozen_budget_before_any_trajectory(
    tmp_path, monkeypatch
):
    touched = False

    def forbidden_environment():
        nonlocal touched
        touched = True
        raise AssertionError("预算校验前不应读取运行环境")

    monkeypatch.setattr(
        collector,
        "environment_manifest",
        forbidden_environment,
    )
    with pytest.raises(ValueError, match="冻结 development 预算"):
        collector.run_collection(
            mode="development",
            datasets=["test_300x10"],
            kernels=["independent"],
            seeds=[200],
            rounds=10,
            output_dir=tmp_path,
        )
    assert touched is False


@pytest.mark.parametrize("name", ["test_300x10", "nltcs"])
def test_public_workload_matches_pinned_files(name):
    workload = collector.load_public_workload(name)
    specification = collector.DATASETS[name]

    assert workload.n_records == specification["n_records"]
    assert len(workload.queries) == specification["query_count"]
    assert workload.target.shape == (specification["query_count"],)
    assert workload.input_sha256 == specification["sha256"]
    assert len(workload.query_identity_sha256) == 64
    assert len(workload.target_identity_sha256) == 64


def test_smoke_reuses_one_s0_for_both_kernels(tmp_path, monkeypatch):
    workload = SimpleNamespace(name="test_300x10")
    derived = []
    collected = []

    monkeypatch.setattr(
        collector,
        "environment_manifest",
        lambda: {"git_worktree_clean_including_untracked": False},
    )
    monkeypatch.setattr(
        collector,
        "load_public_workload",
        lambda name: workload,
    )

    def fake_derive(received, seed):
        derived.append((received, seed))
        return 7.25, {"direction_reference_scale": 7.25}

    monkeypatch.setattr(collector, "derive_workload_seed_s0", fake_derive)

    def fake_collect(received, kernel, seed, rounds, s0, preflight):
        collected.append((kernel, seed, rounds, s0, preflight))
        return object(), object(), object()

    monkeypatch.setattr(collector, "collect_one_trajectory", fake_collect)
    monkeypatch.setattr(
        collector,
        "save_collected_run",
        lambda output_dir, **kwargs: Path(output_dir) / kwargs["kernel"],
    )

    outputs = collector.run_collection(
        mode="smoke",
        datasets=["test_300x10"],
        kernels=["independent", "factorized_gibbs"],
        seeds=[999],
        rounds=2,
        output_dir=tmp_path,
    )

    assert derived == [(workload, 999)]
    assert [row[0] for row in collected] == [
        "independent", "factorized_gibbs"
    ]
    assert {row[3] for row in collected} == {7.25}
    assert collected[0][4] is collected[1][4]
    assert outputs == [tmp_path / "independent", tmp_path / "factorized_gibbs"]


def test_clip_audit_keeps_direction_and_gibbs_denominators_separate():
    audit = collector._clip_audit({
        "direction_logit_evaluated_count_history": [10, 20],
        "direction_logit_clipped_count_history": [1, 2],
        "factorized_gibbs_conditional_logit_evaluated_count": 40,
        "factorized_gibbs_conditional_logit_clipped_count": 8,
    })

    assert audit["direction"] == {
        "evaluated_count": 30,
        "clipped_count": 3,
        "clipped_rate": 0.1,
    }
    assert audit["gibbs_conditional"] == {
        "evaluated_count": 40,
        "clipped_count": 8,
        "clipped_rate": 0.2,
    }


def test_run_artifacts_are_atomic_strict_and_never_overwritten(tmp_path):
    trace = _make_trace([[2.0, 1.0]] * 3, moving=True)
    workload = SimpleNamespace(
        name="test_300x10",
        input_sha256={"schema": "a", "queries": "b", "marginals": "c"},
        query_identity_sha256=trace.query_identity_sha256,
        target_identity_sha256=trace.target_identity_sha256,
    )
    diagnostics = {
        "params": {"seed": 999},
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
        "git_worktree_clean_including_untracked": False,
    }
    kwargs = dict(
        mode="smoke",
        workload=workload,
        kernel="independent",
        seed=999,
        rounds=2,
        preflight={"direction_reference_scale": 1.0},
        diagnostics=diagnostics,
        trace=trace,
        final_table=pd.DataFrame({"x": [0, 1]}),
        elapsed_sec=0.1,
        environment=environment,
    )

    destination = collector.save_collected_run(tmp_path, **kwargs)
    manifest_path = destination / "run_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["formal_development_calibration"] is False
    assert manifest["run_summary"]["clip_audit"]["direction"][
        "clipped_count"
    ] == 0
    assert load_stationarity_trace(destination / "trace").observations == (
        trace.observations
    )
    assert not list(destination.parent.glob(".independent.partial-*"))

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        collector.save_collected_run(tmp_path, **kwargs)
