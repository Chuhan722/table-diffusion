"""因子 Gibbs mixing 探针读取无门控 current 快照的测试。"""

import json

import numpy as np
import pytest

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


def test_raw_conditional_logit_diagnostics_detect_clip_hits():
    masks = np.asarray([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ], dtype=bool)
    directions = np.asarray([0.0, 1.0, -1.0, 3.0])

    raw_logits = probe._raw_conditional_logits(
        masks, directions, eta=0.5, strength=10.0
    )
    np.testing.assert_array_equal(
        raw_logits, np.asarray([10.0, 40.0, -10.0, 20.0])
    )

    accumulator = probe._empty_logit_accumulator()
    probe._accumulate_logit_diagnostics(
        accumulator, raw_logits, logit_clip=30.0
    )
    result = probe._finalize_logit_diagnostics(
        accumulator, logit_clip=30.0
    )

    assert result["condition_count"] == 4
    assert result["raw_logit_min"] == -10.0
    assert result["raw_logit_max"] == 40.0
    assert result["raw_logit_abs_max"] == 40.0
    assert result["logit_clip"] == 30.0
    assert result["clip_hit_count"] == 1
    assert result["clip_hit_rate"] == 0.25
    assert result["raw_logit_strictly_inside_clip"] is False

    effective = np.asarray([10.0, 30.0, -10.0, 20.0])
    expected_probabilities = 1.0 / (1.0 + np.exp(-effective))
    expected_entropies = -(
        expected_probabilities * np.log(expected_probabilities)
        + (1.0 - expected_probabilities)
        * np.log1p(-expected_probabilities)
    )
    assert result["conditional_probability_min"] == pytest.approx(
        expected_probabilities.min()
    )
    assert result["conditional_probability_max"] == pytest.approx(
        expected_probabilities.max()
    )
    assert result[
        "minimum_binary_outcome_probability"
    ] == pytest.approx(np.minimum(
        expected_probabilities, 1.0 - expected_probabilities
    ).min())
    assert result["uniform_condition_entropy_mean"] == pytest.approx(
        expected_entropies.mean()
    )
    assert result["uniform_condition_entropy_maximum"] == pytest.approx(
        np.log(2.0)
    )
    assert result["all_conditionals_bidirectional"] is True


def test_probe_reads_verified_current_snapshot_without_recalibration(
    tmp_path, monkeypatch
):
    target, queries, schema, marginals = _inputs()
    run = trajectory._run_one(
        target,
        queries,
        schema,
        marginals,
        seed=8,
        rounds=12,
        temperature=4.0,
        sweeps=0,
        device="numpy",
        snapshot_rounds=[7],
    )
    snapshot = run["state_snapshots"][0]
    snapshot_path = tmp_path / "current_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    state, controls = probe._load_current_snapshot(
        snapshot_path,
        target,
        queries,
        schema,
        device="numpy",
    )
    assert controls["source_seed"] == 8
    assert controls["state_round"] == 7
    assert controls["source_sweeps"] == 0
    assert controls["state_sha256"] == snapshot["state_sha256"]
    assert controls["current_loss"] == snapshot["current_loss"]

    bad_hash = dict(snapshot, state_sha256="0" * 64)
    with pytest.raises(ValueError, match="哈希核验失败"):
        probe._restore_current_snapshot(
            bad_hash, target, queries, schema, device="numpy"
        )
    bad_loss = dict(snapshot, current_loss=snapshot["current_loss"] + 0.5)
    with pytest.raises(ValueError, match="loss 重新计算不一致"):
        probe._restore_current_snapshot(
            bad_loss, target, queries, schema, device="numpy"
        )

    observed_alpha = []
    original_sampling_probs = probe.compute_sampling_probs

    def record_sampling_alpha(*args, **kwargs):
        observed_alpha.append(kwargs["alpha"])
        return original_sampling_probs(*args, **kwargs)

    def reject_scale_recalibration(*args, **kwargs):
        raise AssertionError("外部快照路径不应重新估计 s0")

    monkeypatch.setattr(probe, "compute_sampling_probs", record_sampling_alpha)
    monkeypatch.setattr(
        probe, "direction_rms_scale", reject_scale_recalibration
    )
    result = probe._probe_state(
        state,
        target,
        queries,
        schema,
        seed=controls["source_seed"],
        state_index=0,
        state_rounds=controls["state_round"],
        temperatures=[4.0],
        sweeps=[0],
        proposals=2,
        device="numpy",
        max_active_attributes=12,
        external_snapshot_controls=controls,
    )

    assert observed_alpha == [snapshot["donor_alpha"]]
    assert result["probe_alpha"] == snapshot["donor_alpha"]
    assert result["direction_reference_scale"] == snapshot[
        "direction_reference_scale"
    ]
    assert result["reference_scale_proposal_index"] is None
    assert result["state_sha256"] == snapshot["state_sha256"]
    assert result["state_loss"] == snapshot["current_loss"]
    assert result["external_snapshot_controls"] == controls
    logit = result["conditional_logit_diagnostics"]["tau_4"]
    assert logit["condition_count"] > 0
    assert np.isfinite(logit["raw_logit_abs_max"])
    assert 0 <= logit["clip_hit_count"] <= logit["condition_count"]
    assert 0.0 < logit["conditional_probability_min"] < 1.0
    assert 0.0 < logit["conditional_probability_max"] < 1.0
    assert 0.0 < logit["minimum_binary_outcome_probability"] <= 0.5
    assert 0.0 <= logit["uniform_condition_entropy_mean"] <= np.log(2.0)
    assert logit["all_conditionals_bidirectional"] is True


def test_probe_default_path_keeps_legacy_scale_and_alpha_rules(monkeypatch):
    target, queries, schema, marginals = _inputs()
    state = probe._make_baseline_state(
        target, queries, schema, marginals, 0, 0, "numpy"
    )
    original_scale = probe.direction_rms_scale
    scale_calls = []

    def record_scale(values):
        scale_calls.append(1)
        return original_scale(values)

    monkeypatch.setattr(probe, "direction_rms_scale", record_scale)
    result = probe._probe_state(
        state,
        target,
        queries,
        schema,
        seed=0,
        state_index=0,
        state_rounds=0,
        temperatures=[1.0],
        sweeps=[0],
        proposals=2,
        device="numpy",
        max_active_attributes=12,
    )

    assert result["probe_alpha"] == 2.0
    assert result["direction_reference_scale"] > 0.0
    assert result["reference_scale_proposal_index"] == 0
    assert scale_calls == [1]
    assert "external_snapshot_controls" not in result
