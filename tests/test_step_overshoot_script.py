"""联合扩散整代步幅冻结脚本的配对与判断规则测试。"""

import numpy as np
import pandas as pd
import pytest

import scripts.probe_factorized_step_overshoot as probe
from table_diffevo.schema import AttributeBlock, Schema


def _schema_queries():
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


def test_copy_masks_recovers_donor_edits():
    current = pd.DataFrame({"a": [0, 1], "b": [0, 0]})
    donors = pd.DataFrame({"a": [1, 0], "b": [1, 0]})
    proposal = pd.DataFrame({"a": [1, 1], "b": [0, 0]})

    differs, mask = probe._copy_masks(
        current,
        proposal,
        donors,
        np.asarray([True, False]),
        ["a", "b"],
    )

    np.testing.assert_array_equal(
        differs,
        [[True, True], [True, False]],
    )
    np.testing.assert_array_equal(
        mask,
        [[True, False], [False, False]],
    )


@pytest.mark.parametrize(
    "proposal,participate,message",
    [
        (
            pd.DataFrame({"a": [2], "b": [0]}),
            np.asarray([True]),
            "非 donor",
        ),
        (
            pd.DataFrame({"a": [1], "b": [0]}),
            np.asarray([False]),
            "未参与",
        ),
    ],
)
def test_copy_masks_rejects_unexplained_edits(
    proposal, participate, message
):
    current = pd.DataFrame({"a": [0], "b": [0]})
    donors = pd.DataFrame({"a": [1], "b": [1]})
    with pytest.raises(RuntimeError, match=message):
        probe._copy_masks(
            current, proposal, donors, participate, ["a", "b"]
        )


def test_replayed_initial_mask_matches_zero_sweep_update(monkeypatch):
    schema, queries = _schema_queries()
    current = pd.DataFrame({
        "a": [0, 0, 0, 1, 1, 1],
        "b": [0, 1, 0, 1, 0, 1],
        "c": [0, 0, 1, 1, 1, 0],
    })
    donors = 1 - current
    scores = np.zeros((len(current), 3), dtype=float)
    differs = np.ones_like(scores, dtype=bool)
    update_seed = 137
    monkeypatch.setattr(probe, "N_RECORDS", len(current))
    monkeypatch.setattr(probe, "RHO", 0.5)

    update_rng = np.random.default_rng(update_seed)
    proposal, _ = probe.evolve_step_factorized_gibbs(
        current,
        donors,
        schema,
        queries,
        np.zeros(len(queries), dtype=float),
        rho=probe.RHO,
        eta=probe.ETA,
        mu=0.0,
        copy_direction_scores=scores,
        copy_direction_strength=0.0,
        n_sweeps=0,
        rng=update_rng,
        max_factor_order=3,
        gibbs_logit_clip=probe.DEFAULT_LOGIT_CLIP,
    )
    participate, initial_mask, endpoint = (
        probe._replay_independent_initial_mask(
            update_seed,
            differs,
            scores,
            0.0,
        )
    )
    _, applied_mask = probe._copy_masks(
        current,
        proposal,
        donors,
        participate,
        schema.attribute_names(),
    )

    assert np.any(participate)
    assert np.any(~participate)
    np.testing.assert_array_equal(
        applied_mask,
        initial_mask & participate[:, None],
    )
    assert endpoint == probe._rng_state_sha256(update_rng)


def _comparison(mean):
    return {"difference": {"mean": float(mean)}}


def _source_inputs(self_difference, cross_difference):
    total = self_difference + cross_difference
    paired = {
        "quadratic_penalty": _comparison(total),
        "self_penalty": _comparison(self_difference),
        "cross_penalty": _comparison(cross_difference),
    }
    return paired, [paired.copy() for _ in range(6)]


@pytest.mark.parametrize(
    "self_difference,cross_difference,expected",
    [
        (
            3.0,
            0.5,
            "supports_per_row_or_fixed_cardinality_normalization",
        ),
        (
            0.5,
            3.0,
            "supports_cross_row_time_step_normalization",
        ),
        (1.0, 1.0, "mixed_or_inconclusive_source"),
        (-1.0, 0.5, "mixed_or_inconclusive_source"),
    ],
)
def test_choose_source_applies_preregistered_rule(
    self_difference, cross_difference, expected
):
    paired, states = _source_inputs(self_difference, cross_difference)
    result = probe._choose_source(paired, states)
    assert result["decision"] == expected


def test_probe_state_pairs_primary_rng_and_step_identities(monkeypatch):
    schema, queries = _schema_queries()
    state = pd.DataFrame({
        "a": [0, 0, 1, 1],
        "b": [0, 1, 0, 1],
        "c": [0, 1, 1, 0],
    })
    target = np.asarray([2.5, 2.5, 1.5])
    monkeypatch.setattr(probe, "N_RECORDS", len(state))
    monkeypatch.setattr(probe, "RHO", 1.0)

    result = probe._probe_state(
        state,
        target,
        queries,
        schema,
        seed=3,
        state_index=0,
        state_rounds=0,
        proposals=3,
        temperature=2.0,
        sweeps=2,
        max_factor_order=3,
        device="numpy",
        fixed_reference_scale=0.25,
    )

    assert result["n_proposals"] == 3
    assert result["gates"]["primary_rng_aligned"] is True
    assert result["gates"]["donor_aligned"] is True
    assert result["gates"]["participation_aligned"] is True
    assert result["gates"]["initial_mask_aligned"] is True
    assert result["gates"]["initial_mask_replay_rng_aligned"] is True
    assert result["gates"][
        "baseline_applied_initial_mask_aligned"
    ] is True
    assert result["gates"]["row_delta_sum_max_error"] == 0.0
    assert result["gates"]["quadratic_identity_max_error"] == 0.0
    assert result["gates"]["gain_identity_max_error"] == 0.0
    assert (
        result["direction_reference_scale_source"]
        == "standard_closed_loop_initial_rms"
    )
    assert all(
        row["mask"]["participating_rows"] == len(state)
        for row in result["pair_rows"]
    )
    assert all(
        row["gibbs_microsteps"] > 0
        for row in result["candidate_rows"]
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_probe_state_rejects_invalid_fixed_scale(scale, monkeypatch):
    schema, queries = _schema_queries()
    state = pd.DataFrame({
        "a": [0, 1],
        "b": [0, 1],
        "c": [0, 1],
    })
    monkeypatch.setattr(probe, "N_RECORDS", len(state))
    with pytest.raises(ValueError, match="fixed_reference_scale"):
        probe._probe_state(
            state,
            np.asarray([1.0, 1.0, 1.0]),
            queries,
            schema,
            seed=0,
            state_index=0,
            state_rounds=0,
            proposals=1,
            temperature=2.0,
            sweeps=2,
            max_factor_order=3,
            device="numpy",
            fixed_reference_scale=scale,
        )
