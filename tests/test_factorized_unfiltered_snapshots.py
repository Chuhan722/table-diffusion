"""因子 Gibbs 无门控研究轨迹的 current-state 快照测试。"""

import json

import numpy as np
import pandas as pd
import pytest

import scripts.compare_factorized_gibbs_unfiltered as experiment
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


def _inputs():
    schema = load_schema(experiment.SCHEMA_PATH)
    queries = load_queries(experiment.QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(experiment.MARGINALS_PATH)
    return target, queries, schema, marginals


def test_current_snapshots_do_not_change_unfiltered_trajectory():
    target, queries, schema, marginals = _inputs()
    common = {
        "seed": 8,
        "rounds": 12,
        "temperature": 4.0,
        "sweeps": 0,
        "device": "numpy",
        "record_state_hashes": True,
    }

    plain = experiment._run_one(
        target, queries, schema, marginals, **common
    )
    snapped = experiment._run_one(
        target,
        queries,
        schema,
        marginals,
        snapshot_rounds=[0, 7, 12],
        **common,
    )

    assert "state_snapshots" not in plain
    for key in (
        "final_csv_sha256",
        "final_loss",
        "best_loss_diagnostic_only",
        "primary_rng_state_sha256",
        "direction_reference_scale",
        "state_sha256_history",
        "loss_history",
        "gain_history",
        "changed_cells_history",
    ):
        assert snapped[key] == plain[key]

    snapshots = snapped["state_snapshots"]
    assert snapped["snapshot_rounds"] == [0, 7, 12]
    assert [row["state_round"] for row in snapshots] == [0, 7, 12]
    assert all(row["state_kind"] == "current" for row in snapshots)
    assert snapshots[0]["state_sha256"] == snapped["initial_csv_sha256"]
    assert snapshots[1]["state_sha256"] == snapped[
        "state_sha256_history"
    ][6]
    assert snapshots[2]["state_sha256"] == snapped[
        "state_sha256_history"
    ][11]
    assert snapshots[2]["state_sha256"] == snapped["final_csv_sha256"]
    assert snapshots[2]["current_loss"] == snapped["final_loss"]
    assert snapshots[2]["primary_rng_state_sha256"] == snapped[
        "primary_rng_state_sha256"
    ]
    assert snapshots[1]["current_loss"] > snapshots[1][
        "best_loss_so_far_diagnostic_only"
    ]

    assert snapshots[0]["donor_alpha"] == 2.0
    assert 2.0 < snapshots[1]["donor_alpha"] < 10.0
    assert snapshots[2]["donor_alpha"] == 10.0
    assert snapped["direction_reference_scale"] > 0.0
    assert {
        row["direction_reference_scale"] for row in snapshots
    } == {snapped["direction_reference_scale"]}
    assert {
        row["direction_reference_scale_round"] for row in snapshots
    } == {0}

    for snapshot in snapshots:
        reconstructed = pd.DataFrame(
            snapshot["table_records"],
            columns=snapshot["table_columns"],
        )
        assert experiment._frame_sha256(reconstructed) == snapshot[
            "state_sha256"
        ]
    json.dumps(snapped, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize(
    "snapshot_rounds,message",
    [
        ([0, 0], "不得重复"),
        ([-1], "0..12"),
        ([13], "0..12"),
        ([1.5], "只包含整数"),
    ],
)
def test_snapshot_rounds_are_validated(snapshot_rounds, message):
    with pytest.raises(ValueError, match=message):
        experiment._normalize_snapshot_rounds(snapshot_rounds, 12)
