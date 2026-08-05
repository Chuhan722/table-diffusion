"""共同状态曲率阶段交互脚本的统计、结构与输出保护测试。"""

import json

import numpy as np
import pandas as pd
import pytest

import scripts.diagnose_curvature_state_interaction as interaction
from table_diffevo.schema import AttributeBlock, Schema


def _summary(values):
    return interaction._seed_summary(values)


@pytest.mark.parametrize(
    "values,expected",
    [
        (
            np.linspace(1.0, 2.0, 10),
            "curvature_advantage_strengthens_late",
        ),
        (
            np.linspace(-2.0, -1.0, 10),
            "curvature_advantage_weakens_late",
        ),
        (
            [-1.0, 1.0] * 5,
            "state_interaction_inconclusive",
        ),
        (
            [10.0] * 7 + [-0.1] * 3,
            "state_interaction_inconclusive",
        ),
    ],
)
def test_classification_applies_interval_and_seed_direction_rule(
    values, expected
):
    assert interaction._classify_stage_interaction(
        _summary(values)
    ) == expected


def test_classification_requires_preregistered_seed_count():
    with pytest.raises(ValueError, match="固定数量"):
        interaction._classify_stage_interaction(_summary([1.0] * 9))


@pytest.mark.parametrize(
    "values,confidence",
    [([], 0.95), ([1.0, np.nan], 0.95), ([1.0], 0.0), ([1.0], 1.0)],
)
def test_mean_t_interval_rejects_invalid_inputs(values, confidence):
    with pytest.raises(ValueError):
        interaction._mean_t_interval(values, confidence=confidence)


def test_mean_t_interval_handles_singleton_and_constant_values():
    assert interaction._mean_t_interval([2.5]) == [2.5, 2.5]
    assert interaction._mean_t_interval([3.0] * 10) == [3.0, 3.0]


def test_frozen_probe_constants_are_pinned(monkeypatch):
    assert interaction._frozen_probe_constants_match() is True
    monkeypatch.setattr(interaction.frozen, "RHO", 0.02)
    assert interaction._frozen_probe_constants_match() is False


def _paired_metric(difference):
    return {
        "baseline": {"mean": 1.0},
        "candidate": {"mean": 1.0 + difference},
        "difference": {"mean": float(difference)},
    }


def _fake_state(seed, rounds, difference):
    state_hash = f"initial-{seed}" if rounds == 0 else f"late-{seed}"
    metrics = {
        metric: _paired_metric(difference)
        for metric in interaction.PRIMARY_METRICS
    }
    gates = {
        name: True for name in interaction.BOOLEAN_STATE_GATES
    }
    gates.update({
        name: 0.0 for name in interaction.MAX_ERROR_STATE_GATES
    })
    gates["conditional_logit_abs_max"] = 1.0
    loss = 50.0 if rounds == 0 else 12.5
    raw_rows = [{"proposal_index": index} for index in range(2)]
    return {
        "seed": seed,
        "state_rounds": rounds,
        "state_sha256": state_hash,
        "state_loss": loss,
        "direction_reference_scale": 0.25,
        "n_proposals": 2,
        "elapsed_sec": 0.1,
        "paired": metrics,
        "gates": gates,
        "baseline_rows": list(raw_rows),
        "candidate_rows": list(raw_rows),
        "pair_rows": list(raw_rows),
        "state_generation": {
            "elapsed_sec": 0.2,
            "initial_table_sha256": f"initial-{seed}",
            "direction_reference_scale": 0.25,
            "method": (
                "marginal_initialization"
                if rounds == 0 else "standard_closed_loop_best"
            ),
            "rounds": rounds,
            "rounds_run": rounds,
            "stopped_early": False,
            "best_loss": None if rounds == 0 else loss,
        },
    }


def _fake_states(seeds=(30, 31)):
    return [
        _fake_state(seed, rounds, 1.0 if rounds == 0 else 3.0)
        for seed in seeds
        for rounds in interaction.FORMAL_STATE_ROUNDS
    ]


def _tiny_schema_queries():
    schema = Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in ("a", "b", "c")
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
            {"attribute": "c", "operator": "==", "value": 1},
        ]},
    ]
    return schema, queries


def test_seed_stage_rows_compute_seed_level_difference_in_differences():
    rows = interaction._build_seed_stage_rows(
        _fake_states(), [30, 31], n_queries=50
    )

    assert [row["seed"] for row in rows] == [30, 31]
    for row in rows:
        assert row["stages"]["0"]["residual_rms"] == pytest.approx(
            np.sqrt(2.0)
        )
        assert row["stages"]["500"]["residual_rms"] == pytest.approx(
            np.sqrt(0.5)
        )
        assert (
            row["interactions_late_minus_initial"]["net_gain"] == 2.0
        )


def test_real_probe_rows_feed_stage_summary_and_all_gates(monkeypatch):
    schema, queries = _tiny_schema_queries()
    state = pd.DataFrame({
        "a": [0, 0, 1, 1],
        "b": [0, 1, 0, 1],
        "c": [0, 1, 1, 0],
    })
    target = np.asarray([2.5, 2.5, 1.5])
    monkeypatch.setattr(interaction.frozen, "N_RECORDS", len(state))
    monkeypatch.setattr(interaction.frozen, "RHO", 1.0)
    states = []
    for state_index, rounds in enumerate(interaction.FORMAL_STATE_ROUNDS):
        result = interaction.frozen._probe_state(
            state,
            target,
            queries,
            schema,
            seed=30,
            state_index=state_index,
            state_rounds=rounds,
            proposals=2,
            temperature=2.0,
            sweeps=2,
            max_factor_order=3,
            device="numpy",
            fixed_reference_scale=0.25,
        )
        result["state_generation"] = {
            "elapsed_sec": 0.1,
            "initial_table_sha256": interaction.frozen._frame_sha256(state),
            "direction_reference_scale": 0.25,
            "method": (
                "marginal_initialization"
                if rounds == 0 else "standard_closed_loop_best"
            ),
            "rounds": rounds,
            "rounds_run": rounds,
            "stopped_early": False,
            "best_loss": None if rounds == 0 else result["state_loss"],
        }
        states.append(result)

    seed_rows = interaction._build_seed_stage_rows(states, [30], 3)
    stage = interaction._stage_aggregates(states, [30], 3)
    gates = interaction._aggregate_gates(states, [30], proposals=2)

    assert len(seed_rows) == 1
    assert stage["0"]["n_paired_proposals"] == 2
    assert stage["500"]["n_paired_proposals"] == 2
    assert "query_delta_l2_squared" in stage["500"][
        "paired_proposal_metrics"
    ]
    assert interaction._diagnostic_gate_passed(gates) is True


def test_state_index_rejects_duplicate_or_incomplete_state_sets():
    states = _fake_states(seeds=(30,))
    with pytest.raises(ValueError, match="重复"):
        interaction._state_index(states + [states[0]], [30])
    with pytest.raises(ValueError, match="不完整"):
        interaction._state_index(states[:1], [30])


def test_aggregate_gates_checks_metadata_and_error_thresholds():
    states = _fake_states()
    gates = interaction._aggregate_gates(states, [30, 31], proposals=2)

    assert gates["state_initialization_aligned"] is True
    assert gates["direction_reference_scale_shared"] is True
    assert interaction._diagnostic_gate_passed(gates) is True

    states[1]["gates"]["gain_identity_max_error"] = 2e-10
    failed = interaction._aggregate_gates(states, [30, 31], proposals=2)
    assert interaction._diagnostic_gate_passed(failed) is False


def test_aggregate_gates_detects_state_initialization_mismatch():
    states = _fake_states(seeds=(30,))
    states[1]["state_generation"]["initial_table_sha256"] = "wrong"

    gates = interaction._aggregate_gates(states, [30], proposals=2)

    assert gates["state_initialization_aligned"] is False
    assert interaction._diagnostic_gate_passed(gates) is False


def test_atomic_json_writer_refuses_overwrite_and_cleans_failed_temp(tmp_path):
    output = tmp_path / "result.json"
    interaction._write_json_atomic(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    with pytest.raises(FileExistsError):
        interaction._write_json_atomic(output, {"value": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        interaction._write_json_atomic(invalid, {"value": np.nan})
    assert not invalid.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_main_rejects_duplicate_seeds(monkeypatch):
    monkeypatch.setattr(
        interaction.frozen.sys,
        "argv",
        [
            "diagnose_curvature_state_interaction.py",
            "--seeds",
            "30",
            "30",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        interaction.main()
