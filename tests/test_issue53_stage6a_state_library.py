from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_issue53_stage6a_state_library as builder
from scripts import issue53_stage6a_protocol as protocol


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_runtime_inputs():
    input_audit = {}
    runtime_inputs = {}
    runtime_targets = {}
    for dataset_name in protocol.DATASET_ORDER:
        dataset = protocol.DATASETS[dataset_name]
        query_count = int(dataset["query_count"])
        queries = [
            {
                "conditions": {
                    f"x_{index % 3}": int(index % 2),
                },
                "result": 2,
            }
            for index in range(query_count)
        ]
        target = [2.0] * query_count
        input_audit[dataset_name] = {
            "dataset": dataset_name,
            "sha256": {
                key: value
                for key, value in dataset["input_sha256"].items()
            },
            "query_count": query_count,
            "query_identity_sha256": dataset[
                "query_identity_sha256"
            ],
            "target_vector_sha256": dataset[
                "target_vector_sha256"
            ],
            "order_counts": dataset["query_order_counts"],
        }
        runtime_inputs[dataset_name] = {
            **input_audit[dataset_name],
            "queries": queries,
            "targets": [2] * query_count,
        }
        runtime_targets[dataset_name] = {
            "runtime_n_records": protocol.SMOKE_N_RECORDS,
            "target_values": target,
            "target_vector_sha256": protocol.canonical_sha256(target),
            "source_target_vector_sha256": dataset[
                "target_vector_sha256"
            ],
        }
    return input_audit, runtime_inputs, runtime_targets


def _fake_run_evolution(
    target,
    queries,
    schema,
    *,
    n_records,
    marginals,
    **params,
):
    del schema, marginals
    query_count = len(queries)
    assert np.asarray(target).shape == (query_count,)
    frame = pd.DataFrame({
        "a": np.arange(n_records, dtype=np.int64) % 2,
        "b": np.arange(n_records, dtype=np.int64) % 3,
    })
    frame_hash = builder.fixed_inputs._frame_sha256(frame)
    scale = 0.25
    works = (0.0, 8.0, 12.0, 20.0, 28.0, 32.0, 40.0)
    answers = np.zeros(query_count, dtype=float)
    residual_signal = builder.compute_residual(
        np.asarray(target, dtype=float),
        answers,
        n_records,
        geometry="relative",
        geometry_floor=8.0,
    )
    squared_loss = builder.compute_loss(target, answers)
    normalized_l1 = float(
        np.mean(np.abs(np.asarray(target) - answers)) / n_records
    )
    snapshots = []
    for index, work in enumerate(works):
        terminal = index == len(works) - 1
        snapshots.append({
            "snapshot_format": "natural_work_current_v1",
            "state_index": index,
            "round": index,
            "phase": "initial" if index == 0 else "post_round",
            "completed_work_ticks": int(work),
            "cumulative_participating_rows": int(
                work * n_records
            ),
            "normalized_work": work,
            "work_tick_completed": index > 0,
            "termination_reason": (
                "early_stopped" if terminal else "in_progress"
            ),
            "current_squared_loss": squared_loss,
            "current_normalized_l1": normalized_l1,
            "current_query_answers": answers.tolist(),
            "current_residual_signal": residual_signal.tolist(),
            "current_table_sha256": frame_hash,
            "table_columns": list(frame.columns),
            "table_records": frame.to_dict(orient="records"),
            "primary_rng_state_sha256": _sha(
                f"{query_count}/snapshot/{index}"
            ),
            "factorized_gibbs_rng_state_sha256": None,
            "candidate_evaluation_count_cumulative": index,
            "direction_reference_scale": scale,
        })

    rounds = len(works) - 1
    diagnostic_params = {
        "seed": params["seed"],
        "n_records": n_records,
        "tol": params["tol"],
        "max_retries": params["max_retries"],
        "rho": params["rho"],
        "eta": params["eta"],
        "mu": params["mu"],
        "fixed_alpha": params["fixed_alpha"],
        "factorized_gibbs_sweeps": params[
            "factorized_gibbs_sweeps"
        ],
        "selection_scale_invariant": params[
            "selection_scale_invariant"
        ],
        "residual_geometry": params["residual_geometry"],
        "residual_geometry_floor": params[
            "residual_geometry_floor"
        ],
        "residual_directed_diffusion": params[
            "residual_directed_diffusion"
        ],
        "diffusion_direction_strength": params[
            "diffusion_direction_strength"
        ],
        "diffusion_direction_normalization": params[
            "diffusion_direction_normalization"
        ],
        "rho_anneal_end": params["rho_anneal_end"],
        "residual_self_cooling": params["residual_self_cooling"],
        "record_natural_work_snapshots": params[
            "record_natural_work_snapshots"
        ],
        "record_stationarity_trace": params[
            "record_stationarity_trace"
        ],
        "inner_early_stopping_patience_ticks": params[
            "inner_early_stopping_patience_ticks"
        ],
    }
    diagnostics = {
        "output_table_identity": "terminal_current",
        "termination_reason": "early_stopped",
        "rounds_run": rounds,
        "accept_history": [True] * rounds,
        "proposal_attempts_history": [1] * rounds,
        "accepted_attempt_history": [1] * rounds,
        "candidate_evaluation_count": rounds,
        "transition_clock_count": rounds,
        "direction_reference_scale": scale,
        "params": diagnostic_params,
        "final_table": frame.copy(deep=True),
        "natural_work_snapshots": snapshots,
        "initial_table_sha256": frame_hash,
        "primary_rng_post_initialization_state_sha256": _sha(
            f"{query_count}/initial-rng"
        ),
        "primary_rng_state_sha256": _sha(
            f"{query_count}/terminal-rng"
        ),
        "final_current_squared_loss": squared_loss,
        "final_current_normalized_l1": normalized_l1,
    }
    return frame.copy(deep=True), diagnostics


@pytest.fixture
def mocked_collection(monkeypatch):
    input_audit, runtime_inputs, runtime_targets = (
        _fake_runtime_inputs()
    )
    calls = []

    def audit_inputs(root, mode):
        del root
        assert mode == "smoke"
        return (
            copy.deepcopy(input_audit),
            copy.deepcopy(runtime_inputs),
            copy.deepcopy(runtime_targets),
        )

    def load_queries(path):
        dataset_name = (
            "nltcs" if "nltcs" in str(path) else "test_300x10"
        )
        return copy.deepcopy(runtime_inputs[dataset_name]["queries"])

    def tracked_run(*args, **kwargs):
        calls.append({
            "target_count": len(args[0]),
            "seed": kwargs["seed"],
            "n_records": kwargs["n_records"],
        })
        return _fake_run_evolution(*args, **kwargs)

    monkeypatch.setattr(builder, "_audit_runtime_inputs", audit_inputs)
    monkeypatch.setattr(builder, "load_schema", lambda path: object())
    monkeypatch.setattr(builder, "load_queries", load_queries)
    monkeypatch.setattr(builder, "load_marginals", lambda path: object())
    monkeypatch.setattr(builder, "run_evolution", tracked_run)
    monkeypatch.setattr(
        builder,
        "_git_identity",
        lambda root=builder.REPOSITORY_ROOT: {
            "commit": "f" * 40,
            "worktree_clean_including_untracked": True,
            "status": [],
        },
    )
    return {
        "calls": calls,
        "input_audit": input_audit,
        "runtime_inputs": runtime_inputs,
        "runtime_targets": runtime_targets,
    }


def _snapshots_for_selection(works, *, terminal_reason="early_stopped"):
    return [
        {
            "state_index": index,
            "round": index,
            "phase": "initial" if index == 0 else "post_round",
            "normalized_work": float(work),
            "termination_reason": (
                terminal_reason
                if index == len(works) - 1
                else "in_progress"
            ),
        }
        for index, work in enumerate(works)
    ]


def test_plan_is_read_only_and_exposes_exact_matrix(monkeypatch):
    monkeypatch.setattr(
        builder,
        "run_evolution",
        lambda *args, **kwargs: pytest.fail("plan 不得调用 generator"),
    )
    plan = builder.build_plan("formal")
    assert plan["mode"] == "plan_only_no_input_read_no_generation"
    assert plan["seed_order"] == list(range(348, 353))
    assert plan["trajectory_count"] == 10
    assert plan["state_count"] == 50
    assert [row["state_count"] for row in plan["seed_shards"]] == [
        10
    ] * 5
    assert plan["generation_started"] is False
    assert plan["formal_confirmation_consumed"] is False
    monkeypatch.setattr(protocol, "protocol_sha256", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="protocol manifest"):
        builder.build_plan("formal")


def test_milestone_selection_is_global_deterministic_and_earlier_tied():
    snapshots = _snapshots_for_selection(
        [0, 8, 12, 20, 28, 32, 40]
    )
    selected = builder._select_milestones(snapshots)
    assert [row["state_group"] for row in selected] == list(
        protocol.STATE_GROUPS
    )
    assert [row["source_snapshot_index"] for row in selected] == [
        0,
        1,
        3,
        4,
        6,
    ]
    assert [row["target_normalized_work"] for row in selected] == [
        0,
        10,
        20,
        30,
        40,
    ]


@pytest.mark.parametrize(
    ("snapshots", "message"),
    [
        (
            _snapshots_for_selection([0, 1, 2, 3]),
            "不足五个",
        ),
        (
            _snapshots_for_selection([0, 2, 1, 3, 4]),
            "顺序",
        ),
        (
            _snapshots_for_selection(
                [0, 1, 2, 3, 4],
                terminal_reason="in_progress",
            ),
            "terminal",
        ),
        (
            _snapshots_for_selection([0, 0, 0, 0, 0]),
            "正有限",
        ),
    ],
)
def test_milestone_selection_fails_closed(snapshots, message):
    with pytest.raises(RuntimeError, match=message):
        builder._select_milestones(snapshots)


def test_formal_protocol_guard_precedes_any_generator_call(
    tmp_path, monkeypatch
):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(builder, "run_evolution", forbidden)
    with pytest.raises(PermissionError, match="单独显式授权"):
        builder.build_state_library(
            "formal", tmp_path / "forbidden.json"
        )
    assert called is False
    assert not (tmp_path / "forbidden.json").exists()


def test_formal_execution_commit_and_clean_tree_are_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        builder,
        "_git_identity",
        lambda root=builder.REPOSITORY_ROOT: {
            "commit": "a" * 40,
            "worktree_clean_including_untracked": True,
            "status": [],
        },
    )
    with pytest.raises(PermissionError, match="execution commit"):
        builder.build_state_library(
            "formal",
            tmp_path / "missing_commit.json",
            confirmed_protocol_sha256=(
                protocol.FROZEN_PROTOCOL_SHA256
            ),
        )
    with pytest.raises(PermissionError, match="execution commit"):
        builder.build_state_library(
            "formal",
            tmp_path / "wrong_commit.json",
            confirmed_protocol_sha256=(
                protocol.FROZEN_PROTOCOL_SHA256
            ),
            confirmed_execution_commit="b" * 40,
        )

    monkeypatch.setattr(
        builder,
        "_git_identity",
        lambda root=builder.REPOSITORY_ROOT: {
            "commit": "a" * 40,
            "worktree_clean_including_untracked": False,
            "status": ["?? unexpected"],
        },
    )
    with pytest.raises(RuntimeError, match="clean worktree"):
        builder.build_state_library(
            "formal",
            tmp_path / "dirty.json",
            confirmed_protocol_sha256=(
                protocol.FROZEN_PROTOCOL_SHA256
            ),
            confirmed_execution_commit="a" * 40,
        )


def test_seed_shard_selection_is_exact():
    assert builder._select_seeds("formal", None) == (
        protocol.FORMAL_SEEDS,
        "full",
        builder.STATE_LIBRARY_FORMAT,
    )
    assert builder._select_seeds("formal", [350]) == (
        (350,),
        "seed_shard",
        builder.STATE_LIBRARY_SHARD_FORMAT,
    )
    with pytest.raises(ValueError, match="恰好包含一个"):
        builder._select_seeds("formal", [])
    with pytest.raises(ValueError, match="恰好包含一个"):
        builder._select_seeds("formal", [348, 349])
    with pytest.raises(ValueError, match="恰好包含一个"):
        builder._select_seeds("formal", [347])
    with pytest.raises(ValueError, match="恰好包含一个"):
        builder._select_seeds("smoke", [348])


def test_mocked_smoke_collection_materializes_two_trajectories_ten_states(
    tmp_path, mocked_collection
):
    output, library = builder.build_state_library(
        "smoke", tmp_path / "state_library.json"
    )
    assert output.exists()
    assert library["status"] == "complete"
    assert library["mode"] == "smoke"
    assert library["artifact_scope"] == "full"
    assert library["selected_seeds"] == [9906]
    assert library["formal_result_valid"] is False
    assert library["manifest"]["state_count"] == 10
    assert len(library["trajectories"]) == 2
    assert len(library["states"]) == 10
    assert [row["state_group"] for row in library["states"][:5]] == list(
        protocol.STATE_GROUPS
    )
    assert [
        row["source_snapshot_index"] for row in library["states"][:5]
    ] == [0, 1, 3, 4, 6]
    assert all(
        set(row["current_count_residual"]) == {2.0}
        for row in library["states"]
    )
    assert library["source_boundary"] == {
        "kernel": "independent_s0",
        "terminal_current": True,
        "post_proposal_gate": False,
        "reference_table_read": False,
        "offline_metric_read": False,
        "proposal_probe_run": False,
    }
    assert mocked_collection["calls"] == [
        {
            "target_count": 50,
            "seed": 9906,
            "n_records": 128,
        },
        {
            "target_count": 1001,
            "seed": 9906,
            "n_records": 128,
        },
    ]
    loaded = builder._strict_load_json(output)
    assert loaded == library
    builder._validate_library_structure(
        loaded,
        mode="smoke",
        artifact_scope="full",
        selected_seeds=(9906,),
    )


def test_collection_is_no_overwrite_before_second_generator_call(
    tmp_path, mocked_collection
):
    output = tmp_path / "state_library.json"
    builder.build_state_library("smoke", output)
    assert len(mocked_collection["calls"]) == 2
    with pytest.raises(FileExistsError, match="不覆盖"):
        builder.build_state_library("smoke", output)
    assert len(mocked_collection["calls"]) == 2


def test_seed_shard_aggregation_preserves_scientific_identity(
    tmp_path, mocked_collection
):
    direct_path, direct = builder.build_state_library(
        "smoke", tmp_path / "direct.json"
    )
    shard_path, shard = builder.build_state_library(
        "smoke",
        tmp_path / "seed_9906.json",
        selected_seeds=(9906,),
    )
    aggregate_path, aggregate = (
        builder.aggregate_state_library_shards(
            "smoke",
            [shard_path],
            tmp_path / "aggregate.json",
        )
    )
    assert direct_path.exists()
    assert aggregate_path.exists()
    assert shard["artifact_scope"] == "seed_shard"
    assert aggregate["artifact_scope"] == "full"
    assert aggregate["states"] == direct["states"]
    assert [
        builder._trajectory_scientific_payload(row)
        for row in aggregate["trajectories"]
    ] == [
        builder._trajectory_scientific_payload(row)
        for row in direct["trajectories"]
    ]
    assert (
        aggregate["state_library_scientific_sha256"]
        == direct["state_library_scientific_sha256"]
    )
    assert list(
        aggregate["manifest"]["source_seed_shard_sha256"]
    ) == ["9906"]
    assert aggregate["manifest"]["source_seed_shard_sha256"][
        "9906"
    ] == protocol.file_sha256(shard_path)

    with pytest.raises(RuntimeError, match="重复"):
        builder.aggregate_state_library_shards(
            "smoke",
            [shard_path, shard_path],
            tmp_path / "duplicate.json",
        )
    with pytest.raises(RuntimeError, match="恰好覆盖"):
        builder.aggregate_state_library_shards(
            "smoke", [], tmp_path / "missing.json"
        )


def test_library_validation_rejects_table_residual_and_science_tampering(
    tmp_path, mocked_collection
):
    _, library = builder.build_state_library(
        "smoke", tmp_path / "state_library.json"
    )

    table = copy.deepcopy(library)
    table["states"][0]["snapshot"]["table_records"][0]["a"] = 9
    with pytest.raises(RuntimeError, match="snapshot"):
        builder._validate_library_structure(
            table,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(9906,),
        )

    residual = copy.deepcopy(library)
    residual["states"][0]["current_count_residual"][0] += 1
    with pytest.raises(RuntimeError, match="count residual"):
        builder._validate_library_structure(
            residual,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(9906,),
        )

    science = copy.deepcopy(library)
    science["state_library_scientific_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="scientific SHA"):
        builder._validate_library_structure(
            science,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(9906,),
        )


def test_library_validation_rejects_input_param_and_milestone_tampering(
    tmp_path, mocked_collection
):
    _, library = builder.build_state_library(
        "smoke", tmp_path / "state_library.json"
    )

    inputs = copy.deepcopy(library)
    inputs["input_audit"]["nltcs"]["sha256"]["queries"] = "0" * 64
    with pytest.raises(RuntimeError, match="input audit"):
        builder._validate_library_structure(
            inputs,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(9906,),
        )

    params = copy.deepcopy(library)
    params["trajectories"][0]["source_generator_params"]["rho"] = 0.5
    with pytest.raises(RuntimeError, match="trajectory artifact"):
        builder._validate_library_structure(
            params,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(9906,),
        )

    milestone = copy.deepcopy(library)
    milestone["states"][1]["source_snapshot_index"] = 2
    with pytest.raises(RuntimeError, match="milestone selection"):
        builder._validate_library_structure(
            milestone,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(9906,),
        )


def test_scientific_identity_excludes_environment_and_elapsed(
    tmp_path, mocked_collection
):
    _, library = builder.build_state_library(
        "smoke", tmp_path / "state_library.json"
    )
    changed = copy.deepcopy(library)
    changed["environment"] = {"different": True}
    changed["elapsed_sec_diagnostic_only"] = 999.0
    for row in changed["trajectories"]:
        row["elapsed_sec_diagnostic_only"] = 123.0
    assert protocol.canonical_sha256(
        builder.scientific_payload(changed)
    ) == library["state_library_scientific_sha256"]


def test_exclusive_writer_and_strict_loader_reject_overwrite_and_nan(
    tmp_path,
):
    output = builder._exclusive_write_json(
        tmp_path / "value.json", {"value": 1}
    )
    assert builder._strict_load_json(output) == {"value": 1}
    with pytest.raises(FileExistsError, match="不覆盖"):
        builder._exclusive_write_json(output, {"value": 2})

    nonstandard = tmp_path / "nan.json"
    nonstandard.write_text('{"value": NaN}\\n', encoding="utf-8")
    with pytest.raises(ValueError, match="非标准"):
        builder._strict_load_json(nonstandard)


def test_actual_input_audit_is_measured_only_and_bound():
    input_audit, runtime_inputs, runtime_targets = (
        builder._audit_runtime_inputs(
            builder.REPOSITORY_ROOT, "formal"
        )
    )
    assert list(input_audit) == list(protocol.DATASET_ORDER)
    assert list(runtime_targets) == list(protocol.DATASET_ORDER)
    for dataset_name in protocol.DATASET_ORDER:
        assert "queries" not in input_audit[dataset_name]
        assert "targets" not in input_audit[dataset_name]
        assert len(runtime_inputs[dataset_name]["queries"]) == (
            protocol.DATASETS[dataset_name]["query_count"]
        )
        target = runtime_targets[dataset_name]["target_values"]
        assert all(float(value).is_integer() for value in target)
        assert runtime_targets[dataset_name][
            "target_vector_sha256"
        ] == protocol.canonical_sha256(target)
