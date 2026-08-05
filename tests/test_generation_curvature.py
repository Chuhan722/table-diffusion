"""整代曲率感知 Gibbs 扩散的纯函数与退化回归测试。"""

import json

import numpy as np
import pandas as pd
import pytest

from table_diffevo.factorized_diffusion import (
    build_sparse_mask_energy,
    evolve_step_factorized_gibbs,
)
from table_diffevo.generation_curvature import (
    build_sparse_query_delta,
    conditional_generation_copy_probability,
    conditional_generation_energy_difference,
    conditional_query_delta_difference,
    evaluate_sparse_query_delta,
    evolve_step_generation_curvature_gibbs,
    generation_curvature_energy,
)
from table_diffevo.joint_diffusion import enumerate_copy_masks
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


def _schema():
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in ("a", "b", "c")
    ])


def _queries():
    return [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
            {"attribute": "c", "operator": "==", "value": 1},
        ]},
    ]


def _row_models(residual):
    current = pd.DataFrame({"a": [0], "b": [0], "c": [0]})
    donor = pd.DataFrame({"a": [1], "b": [1], "c": [1]})
    return (
        build_sparse_mask_energy(
            current, donor, _schema(), _queries(), np.asarray(residual)
        ),
        build_sparse_query_delta(
            current, donor, _schema(), _queries()
        ),
    )


def _rng_state(rng):
    return json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )


class TestSparseQueryDelta:
    def test_manual_one_two_three_way_query_deltas(self):
        _, model = _row_models([1.0, 1.0, 1.0, 1.0])
        masks = enumerate_copy_masks(3)
        actual = np.asarray([
            evaluate_sparse_query_delta(model, mask) for mask in masks
        ])
        a = masks[:, 0].astype(np.int16)
        b = masks[:, 1].astype(np.int16)
        c = masks[:, 2].astype(np.int16)
        expected = np.column_stack([a, b, a * b, a * b * c])

        np.testing.assert_array_equal(actual, expected)
        assert model.max_active_query_order == 3
        assert {factor.query_index for factor in model.factors} == {
            0, 1, 2, 3
        }

    def test_negative_delta_when_recipient_satisfies_query(self):
        current = pd.DataFrame({"a": [1], "b": [0], "c": [0]})
        donor = pd.DataFrame({"a": [0], "b": [0], "c": [0]})
        model = build_sparse_query_delta(
            current, donor, _schema(), _queries()
        )

        np.testing.assert_array_equal(
            evaluate_sparse_query_delta(model, np.asarray([False])),
            [0, 0, 0, 0],
        )
        np.testing.assert_array_equal(
            evaluate_sparse_query_delta(model, np.asarray([True])),
            [-1, 0, 0, 0],
        )

    def test_conditional_delta_uses_current_other_bits(self):
        _, model = _row_models([1.0, 1.0, 1.0, 1.0])
        mask = np.asarray([True, False, True])

        np.testing.assert_array_equal(
            conditional_query_delta_difference(model, mask, 1),
            [0, 1, 1, 1],
        )
        mask = np.asarray([False, True, True])
        np.testing.assert_array_equal(
            conditional_query_delta_difference(model, mask, 0),
            [1, 0, 1, 1],
        )

    def test_repeated_numeric_conditions_share_one_active_bit(self):
        schema = Schema([
            AttributeBlock(
                name="age",
                type="numeric",
                description="age",
                range=[0, 100],
            )
        ])
        queries = [{"conditions": [
            {"attribute": "age", "operator": ">=", "value": 18},
            {
                "attribute": "age",
                "operator": "between",
                "lower": 18,
                "upper": 24,
            },
        ]}]
        model = build_sparse_query_delta(
            pd.DataFrame({"age": [17]}),
            pd.DataFrame({"age": [20]}),
            schema,
            queries,
            max_factor_order=1,
        )

        assert model.n_active_attributes == 1
        assert len(model.factors) == 1
        np.testing.assert_array_equal(model.factors[0].values, [0, 1])


class TestGenerationEnergy:
    def test_gamma_one_is_exact_loss_gain_divided_by_n(self):
        current_q = np.asarray([3.0, 1.0, 2.0, 0.0])
        target = np.asarray([5.0, 0.0, 4.0, 1.0])
        delta = np.asarray([1.0, -1.0, 0.0, 1.0])
        n_records = 7
        residual = (target - current_q) / n_records

        energy = generation_curvature_energy(
            residual, delta, n_records, curvature_weight=1.0
        )
        gain = (
            compute_loss(target, current_q)
            - compute_loss(target, current_q + delta)
        )
        assert n_records * energy == pytest.approx(gain, abs=1e-12)

    def test_conditional_difference_matches_direct_two_state_energy(self):
        residual = np.asarray([0.2, -0.1, 0.3, 0.4])
        linear_model, query_model = _row_models(residual)
        other_rows_delta = np.asarray([1.0, -1.0, 0.0, 1.0])
        n_records = 9

        for mask in enumerate_copy_masks(3):
            for variable in range(3):
                current_row_delta = evaluate_sparse_query_delta(
                    query_model, mask
                ).astype(float)
                total = other_rows_delta + current_row_delta
                result = conditional_generation_energy_difference(
                    linear_model,
                    query_model,
                    mask,
                    variable,
                    total,
                    n_records,
                    curvature_weight=1.0,
                )
                lower = mask.copy()
                upper = mask.copy()
                lower[variable] = False
                upper[variable] = True
                lower_total = other_rows_delta + evaluate_sparse_query_delta(
                    query_model, lower
                )
                upper_total = other_rows_delta + evaluate_sparse_query_delta(
                    query_model, upper
                )
                expected = generation_curvature_energy(
                    residual, upper_total, n_records, 1.0
                ) - generation_curvature_energy(
                    residual, lower_total, n_records, 1.0
                )

                assert result["energy_difference"] == pytest.approx(
                    expected, abs=1e-12
                )
                assert result["linear_difference"] == pytest.approx(
                    np.dot(
                        residual, result["query_delta_difference"]
                    ),
                    abs=1e-12,
                )

    def test_conditional_probability_matches_enumerated_target(self):
        residual = np.asarray([0.3, -0.2, 0.5, 0.7])
        linear_model, query_model = _row_models(residual)
        eta = 0.35
        strength = 1.7
        n_records = 5
        mask = np.asarray([True, False, True])
        variable = 1
        lower = mask.copy()
        upper = mask.copy()
        lower[variable] = False
        upper[variable] = True
        lower_delta = evaluate_sparse_query_delta(query_model, lower)
        upper_delta = evaluate_sparse_query_delta(query_model, upper)
        lower_energy = generation_curvature_energy(
            residual, lower_delta, n_records, 1.0
        )
        upper_energy = generation_curvature_energy(
            residual, upper_delta, n_records, 1.0
        )
        lower_weight = (1.0 - eta) * np.exp(strength * lower_energy)
        upper_weight = eta * np.exp(strength * upper_energy)
        expected = upper_weight / (lower_weight + upper_weight)

        result = conditional_generation_copy_probability(
            linear_model,
            query_model,
            mask,
            variable,
            evaluate_sparse_query_delta(query_model, mask),
            n_records,
            1.0,
            eta,
            strength,
        )
        assert result["probability"] == pytest.approx(expected, abs=1e-12)
        assert 0.0 < result["probability"] < 1.0

    def test_zero_strength_returns_reference_conditional(self):
        linear_model, query_model = _row_models([0.3, -0.2, 0.5, 0.7])
        eta = 0.27
        result = conditional_generation_copy_probability(
            linear_model,
            query_model,
            np.asarray([True, False, True]),
            1,
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            5,
            1.0,
            eta,
            0.0,
        )
        assert result["probability"] == eta

    def test_default_logit_guard_preserves_bidirectional_support(self):
        linear_model, query_model = _row_models([1.0, 1.0, 1.0, 1.0])
        result = conditional_generation_copy_probability(
            linear_model,
            query_model,
            np.asarray([False, False, False]),
            0,
            np.zeros(4),
            5,
            1.0,
            0.5,
            1e308,
        )

        assert result["logit"] == 30.0
        assert result["logit_clipped"] is True
        assert 0.0 < result["probability"] < 1.0


@pytest.mark.parametrize(
    "rho,mu,strength,sweeps",
    [
        (0.0, 0.0, 0.0, 0),
        (0.75, 0.2, 1.7, 0),
        (0.75, 0.2, 0.0, 3),
        (0.75, 0.0, 1.7, 3),
        (0.75, 0.0, 1000.0, 3),
    ],
)
def test_gamma_zero_exactly_matches_existing_factorized_step(
    rho, mu, strength, sweeps
):
    current = pd.DataFrame({
        "a": [0, 0, 1, 1, 0, 1],
        "b": [0, 1, 0, 1, 1, 0],
        "c": [0, 1, 1, 0, 1, 0],
    })
    donors = current.iloc[[3, 2, 5, 4, 1, 0]].reset_index(drop=True)
    residual = np.asarray([0.2, -0.1, 0.3, 0.4])
    direction_scores = np.asarray([
        [0.2, -0.3, 0.1],
        [-0.4, 0.5, 0.6],
        [0.1, 0.2, -0.2],
        [0.7, -0.1, 0.3],
        [-0.2, 0.4, -0.5],
        [0.3, 0.2, 0.1],
    ])
    old_rng = np.random.default_rng(413)
    new_rng = np.random.default_rng(413)
    old_gibbs = np.random.default_rng(991) if sweeps else None
    new_gibbs = np.random.default_rng(991) if sweeps else None

    old, old_diagnostics = evolve_step_factorized_gibbs(
        current,
        donors,
        _schema(),
        _queries(),
        residual,
        rho=rho,
        eta=0.4,
        mu=mu,
        copy_direction_scores=direction_scores,
        copy_direction_strength=strength,
        n_sweeps=sweeps,
        rng=old_rng,
        gibbs_rng=old_gibbs,
        max_factor_order=3,
    )
    new, new_diagnostics = evolve_step_generation_curvature_gibbs(
        current,
        donors,
        _schema(),
        _queries(),
        residual,
        rho=rho,
        eta=0.4,
        mu=mu,
        copy_direction_scores=direction_scores,
        copy_direction_strength=strength,
        n_sweeps=sweeps,
        curvature_weight=0.0,
        rng=new_rng,
        gibbs_rng=new_gibbs,
        max_factor_order=3,
    )

    pd.testing.assert_frame_equal(new, old)
    assert _rng_state(new_rng) == _rng_state(old_rng)
    if sweeps:
        assert _rng_state(new_gibbs) == _rng_state(old_gibbs)
    for key in (
        "participating_rows",
        "active_gibbs_rows",
        "active_blocks",
        "factor_count",
        "factor_table_entries",
        "gibbs_microsteps",
    ):
        assert new_diagnostics[key] == old_diagnostics[key]
    if sweeps:
        assert new_diagnostics["initial_copy_mask_sha256"]
        assert new_diagnostics["raw_initial_copy_mask_sha256"]
        assert new_diagnostics["final_copy_mask_sha256"]
        assert (
            new_diagnostics[
                "gamma_zero_reference_probability_max_error"
            ] == 0.0
        )
    else:
        for key in (
            "initial_query_delta",
            "final_query_delta",
            "initial_linear_energy",
            "final_linear_energy",
            "initial_quadratic_energy",
            "final_quadratic_energy",
            "initial_generation_energy",
            "final_generation_energy",
        ):
            assert new_diagnostics[key] is None
    assert new_diagnostics["linear_query_consistency_max_error"] <= 1e-15


def test_full_curvature_step_reports_exact_copy_proposal_gain():
    current = pd.DataFrame({
        "a": [0, 0, 1, 1],
        "b": [0, 1, 0, 1],
        "c": [0, 1, 1, 0],
    })
    donors = current.iloc[[3, 2, 1, 0]].reset_index(drop=True)
    current_q = evaluate_table(current, _queries())
    target = np.asarray([3.0, 1.0, 2.0, 1.0])
    residual = (target - current_q) / len(current)
    proposal, diagnostics = evolve_step_generation_curvature_gibbs(
        current,
        donors,
        _schema(),
        _queries(),
        residual,
        rho=1.0,
        eta=0.5,
        mu=0.0,
        copy_direction_scores=np.zeros((len(current), 3)),
        copy_direction_strength=2.0,
        n_sweeps=3,
        curvature_weight=1.0,
        rng=np.random.default_rng(17),
        gibbs_rng=np.random.default_rng(23),
    )
    proposal_q = evaluate_table(proposal, _queries())
    delta = proposal_q - current_q
    gain = compute_loss(target, current_q) - compute_loss(target, proposal_q)

    np.testing.assert_array_equal(diagnostics["final_query_delta"], delta)
    assert len(current) * diagnostics["final_generation_energy"] == pytest.approx(
        gain, abs=1e-12
    )
    assert diagnostics["conditional_probability_count"] > 0
    assert diagnostics["all_conditionals_bidirectional"] is True
    assert 0.0 < diagnostics["conditional_probability_min"]
    assert diagnostics["conditional_probability_max"] < 1.0


@pytest.mark.parametrize("rho,same_donor", [(0.0, False), (1.0, True)])
def test_zero_active_gibbs_work_does_not_consume_extra_rng(rho, same_donor):
    current = pd.DataFrame({
        "a": [0, 1],
        "b": [0, 1],
        "c": [0, 1],
    })
    donors = (
        current.copy()
        if same_donor
        else current.iloc[[1, 0]].reset_index(drop=True)
    )
    gibbs_rng = np.random.default_rng(29)
    initial_gibbs_state = _rng_state(gibbs_rng)

    proposal, diagnostics = evolve_step_generation_curvature_gibbs(
        current,
        donors,
        _schema(),
        _queries(),
        np.zeros(4),
        rho=rho,
        eta=0.5,
        mu=0.0,
        copy_direction_scores=np.zeros((2, 3)),
        copy_direction_strength=1.0,
        n_sweeps=3,
        curvature_weight=1.0,
        rng=np.random.default_rng(17),
        gibbs_rng=gibbs_rng,
    )

    pd.testing.assert_frame_equal(proposal, current)
    assert diagnostics["active_gibbs_rows"] == 0
    assert diagnostics["gibbs_microsteps"] == 0
    assert diagnostics["conditional_probability_count"] == 0
    assert diagnostics["all_conditionals_bidirectional"] is True
    assert _rng_state(gibbs_rng) == initial_gibbs_state


def test_query_delta_rejects_active_query_above_order_guard():
    with pytest.raises(ValueError, match="活跃因子阶数"):
        build_sparse_query_delta(
            pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
            pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
            _schema(),
            [_queries()[-1]],
            max_factor_order=2,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"curvature_weight": -1.0}, "curvature_weight"),
        ({"curvature_weight": np.nan}, "curvature_weight"),
        ({"eta": 0.0, "n_sweeps": 1}, "eta"),
        ({"max_factor_order": 9}, "绝对护栏"),
        ({"gibbs_logit_clip": 0.0}, "gibbs_logit_clip"),
        ({"gibbs_logit_clip": np.inf}, "gibbs_logit_clip"),
    ],
)
def test_curvature_step_rejects_invalid_inputs(kwargs, match):
    current = pd.DataFrame({"a": [0], "b": [0], "c": [0]})
    donors = pd.DataFrame({"a": [1], "b": [1], "c": [1]})
    parameters = {
        "rho": 1.0,
        "eta": 0.5,
        "mu": 0.0,
        "copy_direction_scores": np.zeros((1, 3)),
        "copy_direction_strength": 1.0,
        "n_sweeps": 1,
        "curvature_weight": 1.0,
        "rng": np.random.default_rng(1),
        "gibbs_rng": np.random.default_rng(2),
        "max_factor_order": 3,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=match):
        evolve_step_generation_curvature_gibbs(
            current,
            donors,
            _schema(),
            _queries(),
            np.zeros(4),
            **parameters,
        )
