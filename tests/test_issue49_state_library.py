"""Issue #49 无门控 current-state 状态库的小规模测试。"""

import contextlib
import io
import json

import numpy as np
import pytest

import scripts.build_issue49_unfiltered_state_library as builder
import scripts.compare_factorized_gibbs_unfiltered as trajectory
import scripts.probe_factorized_gibbs_mixing as probe
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


def _inputs():
    schema = load_schema(trajectory.SCHEMA_PATH)
    queries = load_queries(trajectory.QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(trajectory.MARGINALS_PATH)
    return target, queries, schema, marginals


def test_short_tau4_tau8_state_library_is_aligned_and_probe_readable():
    target, queries, schema, marginals = _inputs()
    with contextlib.redirect_stdout(io.StringIO()):
        library = builder.build_state_library(
            target,
            queries,
            schema,
            marginals,
            seeds=[8],
            rounds=12,
            snapshot_rounds=[0, 6, 12],
            device="numpy",
        )

    assert library["state_library_format"] == (
        "issue49_unfiltered_state_library_v1"
    )
    assert library["source_temperatures"] == [4.0, 8.0]
    assert library["source_sweeps"] == 0
    assert library["snapshot_rounds"] == [0, 6, 12]
    assert library["state_count"] == library["expected_state_count"] == 5
    assert library["all_gates_passed"] is True
    assert all(library["gates"].values())

    states = library["states"]
    assert [state["state_family"] for state in states] == [
        "initial",
        "mid_source_tau_4",
        "mid_source_tau_8",
        "late_source_tau_4",
        "late_source_tau_8",
    ]
    assert states[0]["source_temperature"] is None
    assert states[0]["shared_source_temperatures"] == [4.0, 8.0]

    seed_row = library["seed_rows"][0]
    assert seed_row["all_gates_passed"] is True
    assert all(seed_row["gates"].values())
    tau4 = seed_row["source_trajectories"]["tau_4"]
    tau8 = seed_row["source_trajectories"]["tau_8"]
    assert tau4["initial_state_sha256"] == tau8[
        "initial_state_sha256"
    ]
    assert tau4["direction_reference_scale"] == tau8[
        "direction_reference_scale"
    ]
    assert tau4["primary_rng_endpoint_sha256"] == tau8[
        "primary_rng_endpoint_sha256"
    ]

    for entry in states:
        snapshot = entry["snapshot"]
        restored, controls = probe._restore_current_snapshot(
            snapshot,
            target,
            queries,
            schema,
            device="numpy",
        )
        assert probe._frame_sha256(restored) == snapshot["state_sha256"]
        assert controls["source_seed"] == 8
        assert controls["state_round"] == entry["state_round"]

    json.dumps(library, ensure_ascii=False, allow_nan=False)


def test_state_library_rejects_misaligned_direction_scale(monkeypatch):
    target, queries, schema, marginals = _inputs()
    original_run_one = builder.trajectory._run_one

    def misaligned_run_one(*args, **kwargs):
        run = original_run_one(*args, **kwargs)
        if kwargs["temperature"] == 8.0:
            run["direction_reference_scale"] *= 2.0
        return run

    monkeypatch.setattr(
        builder.trajectory, "_run_one", misaligned_run_one
    )
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(
            RuntimeError, match="direction_reference_scale_aligned"
        ):
            builder.build_state_library(
                target,
                queries,
                schema,
                marginals,
                seeds=[8],
                rounds=2,
                snapshot_rounds=[0, 1, 2],
                device="numpy",
            )
