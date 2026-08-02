"""低阶联合能量与随机扫描 Gibbs 扩散测试。"""

import numpy as np
import pandas as pd
import pytest

from table_diffevo.factorized_diffusion import (
    build_sparse_mask_energy,
    conditional_copy_probability,
    conditional_energy_difference,
    evolve_step_factorized_gibbs,
    evaluate_sparse_mask_energies,
    evaluate_sparse_mask_energy,
    propagate_random_scan_distribution,
    random_scan_gibbs_mask,
    sparse_single_directions,
)
from table_diffevo.generator import init_synthetic_table
from table_diffevo.joint_diffusion import (
    additive_mask_directions,
    baseline_mask_log_probabilities,
    compute_joint_mask_landscapes,
    enumerate_copy_masks,
    gibbs_mask_log_probabilities,
)
from table_diffevo.queries import load_queries
from table_diffevo.schema import AttributeBlock, Schema, load_schema
from table_diffevo.update import evolve_step


def _three_bit_schema():
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in ("a", "b", "c")
    ])


def _three_bit_queries():
    return [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "b", "operator": "==", "value": 1}]},
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
                {"attribute": "c", "operator": "==", "value": 1},
            ]
        },
    ]


def _three_bit_model():
    return build_sparse_mask_energy(
        pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
        pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
        _three_bit_schema(),
        _three_bit_queries(),
        np.array([1.0, 2.0, 4.0, 8.0]),
    )


def _target_and_independent_distributions(model, eta=0.5, strength=0.7):
    masks = enumerate_copy_masks(model.n_active_attributes)
    reference = baseline_mask_log_probabilities(masks, eta)
    exact = evaluate_sparse_mask_energies(model, masks)
    additive = additive_mask_directions(
        masks, sparse_single_directions(model)
    )
    target = np.exp(gibbs_mask_log_probabilities(
        reference, exact, strength
    ))
    independent = np.exp(gibbs_mask_log_probabilities(
        reference, additive, strength
    ))
    return masks, target, independent


class TestSparseMaskEnergy:
    def test_manual_single_pair_and_triple_energy(self):
        model = _three_bit_model()
        masks = enumerate_copy_masks(3)

        actual = evaluate_sparse_mask_energies(model, masks)
        a = masks[:, 0].astype(float)
        b = masks[:, 1].astype(float)
        c = masks[:, 2].astype(float)
        expected = a + 2.0 * b + 4.0 * a * b + 8.0 * a * b * c

        np.testing.assert_allclose(actual, expected)
        assert model.max_active_query_order == 3
        assert model.n_active_queries == 4
        assert {factor.scope for factor in model.factors} == {
            (0,), (1,), (0, 1), (0, 1, 2)
        }
        np.testing.assert_allclose(
            sparse_single_directions(model), [1.0, 2.0, 0.0]
        )

    def test_conditional_difference_uses_all_touching_factors(self):
        model = _three_bit_model()
        mask = np.array([1, 1, 1], dtype=bool)

        assert conditional_energy_difference(model, mask, 0) == 13.0
        assert conditional_energy_difference(model, mask, 1) == 14.0
        assert conditional_energy_difference(model, mask, 2) == 8.0

    def test_repeated_conditions_on_one_attribute_form_one_bit_factor(self):
        schema = Schema([
            AttributeBlock(
                name="age", type="numeric", description="age", range=[0, 100]
            )
        ])
        query = {
            "conditions": [
                {"attribute": "age", "operator": ">=", "value": 18},
                {"attribute": "age", "operator": "between", "lower": 18, "upper": 24},
            ]
        }
        model = build_sparse_mask_energy(
            pd.DataFrame({"age": [17]}),
            pd.DataFrame({"age": [20]}),
            schema,
            [query],
            np.array([3.0]),
            max_factor_order=1,
        )

        assert model.max_active_query_order == 1
        assert len(model.factors) == 1
        np.testing.assert_allclose(
            evaluate_sparse_mask_energies(
                model, enumerate_copy_masks(1)
            ),
            [0.0, 3.0],
        )

    def test_same_donor_has_empty_zero_energy_model(self):
        row = pd.DataFrame({"a": [0], "b": [1], "c": [0]})
        model = build_sparse_mask_energy(
            row,
            row.copy(),
            _three_bit_schema(),
            _three_bit_queries(),
            np.ones(4),
        )

        assert model.n_active_attributes == 0
        assert model.factors == ()
        assert evaluate_sparse_mask_energy(model, np.array([])) == 0.0
        np.testing.assert_array_equal(
            evaluate_sparse_mask_energies(
                model, np.zeros((1, 0), dtype=bool)
            ),
            [0.0],
        )

    def test_real_workload_matches_full_hybrid_oracle_for_all_masks(self):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        rng = np.random.default_rng(20260802)
        current = init_synthetic_table(8, schema, rng)
        donors = current.iloc[rng.permutation(len(current))].reset_index(drop=True)
        residual = rng.normal(size=len(queries))

        exact_landscapes = compute_joint_mask_landscapes(
            current,
            donors,
            np.arange(4),
            schema,
            queries,
            residual,
            max_active_attributes=12,
        )
        for landscape in exact_landscapes:
            row_index = landscape.row_index
            model = build_sparse_mask_energy(
                current.iloc[[row_index]],
                donors.iloc[[row_index]],
                schema,
                queries,
                residual,
            )
            assert tuple(model.active_attribute_indices) == tuple(
                landscape.active_attribute_indices
            )
            np.testing.assert_allclose(
                evaluate_sparse_mask_energies(model, landscape.masks),
                landscape.directions,
                atol=1e-12,
            )

    def test_rejects_query_above_factor_order(self):
        with pytest.raises(ValueError, match="活跃因子阶数"):
            build_sparse_mask_energy(
                pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
                pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
                _three_bit_schema(),
                [_three_bit_queries()[-1]],
                np.ones(1),
                max_factor_order=2,
            )

    @pytest.mark.parametrize(
        "recipient,donor,residual,kwargs,match",
        [
            (
                pd.DataFrame({"a": [0, 1], "b": [0, 1], "c": [0, 1]}),
                pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
                np.ones(4),
                {},
                "恰好包含一行",
            ),
            (
                pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
                pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
                np.ones(3),
                {},
                "residual",
            ),
            (
                pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
                pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
                np.ones(4),
                {"max_factor_order": 9},
                "绝对护栏",
            ),
        ],
    )
    def test_builder_validation(
        self, recipient, donor, residual, kwargs, match
    ):
        with pytest.raises(ValueError, match=match):
            build_sparse_mask_energy(
                recipient,
                donor,
                _three_bit_schema(),
                _three_bit_queries(),
                residual,
                **kwargs,
            )


class TestRandomScanGibbs:
    def test_conditional_probability_matches_joint_distribution(self):
        model = _three_bit_model()
        eta = 0.3
        strength = 0.4
        masks = enumerate_copy_masks(3)
        reference = baseline_mask_log_probabilities(masks, eta)
        target = np.exp(gibbs_mask_log_probabilities(
            reference,
            evaluate_sparse_mask_energies(model, masks),
            strength,
        ))
        lower_index = 0b110
        upper_index = 0b111
        expected = target[upper_index] / (
            target[lower_index] + target[upper_index]
        )

        actual = conditional_copy_probability(
            model,
            masks[lower_index],
            variable=0,
            eta=eta,
            strength=strength,
        )

        assert actual == pytest.approx(expected)

    def test_exact_joint_distribution_is_stationary(self):
        model = _three_bit_model()
        _, target, _ = _target_and_independent_distributions(model)

        propagated = propagate_random_scan_distribution(
            model,
            target,
            eta=0.5,
            strength=0.7,
            n_steps=25,
        )

        np.testing.assert_allclose(propagated, target, atol=1e-14)

    def test_distribution_converges_monotonically_from_independent_kernel(self):
        model = _three_bit_model()
        _, target, independent = _target_and_independent_distributions(model)
        total_variations = []
        for steps in (0, 1, 3, 6, 12, 24, 48):
            propagated = propagate_random_scan_distribution(
                model,
                independent,
                eta=0.5,
                strength=0.7,
                n_steps=steps,
            )
            total_variations.append(
                0.5 * float(np.abs(propagated - target).sum())
            )

        assert np.all(np.diff(total_variations) <= 1e-14)
        assert total_variations[-1] < 1e-5

    def test_zero_steps_preserve_distribution_and_rng(self):
        model = _three_bit_model()
        _, _, independent = _target_and_independent_distributions(model)
        propagated = propagate_random_scan_distribution(
            model,
            independent,
            eta=0.5,
            strength=0.7,
            n_steps=0,
        )
        np.testing.assert_array_equal(propagated, independent)

        first_rng = np.random.default_rng(17)
        second_rng = np.random.default_rng(17)
        initial = np.array([1, 0, 1])
        result = random_scan_gibbs_mask(
            model,
            initial,
            eta=0.5,
            strength=0.7,
            n_steps=0,
            rng=first_rng,
        )
        np.testing.assert_array_equal(result, initial.astype(bool))
        np.testing.assert_array_equal(
            first_rng.random(10), second_rng.random(10)
        )

    def test_zero_strength_keeps_historical_kernel_stationary(self):
        model = _three_bit_model()
        masks = enumerate_copy_masks(3)
        reference = np.exp(baseline_mask_log_probabilities(masks, 0.3))

        propagated = propagate_random_scan_distribution(
            model,
            reference,
            eta=0.3,
            strength=0.0,
            n_steps=30,
        )

        np.testing.assert_allclose(propagated, reference, atol=1e-14)

    def test_sampler_matches_exact_random_scan_propagation(self):
        model = _three_bit_model()
        masks, _, independent = _target_and_independent_distributions(
            model, strength=0.4
        )
        steps = 6
        expected = propagate_random_scan_distribution(
            model,
            independent,
            eta=0.5,
            strength=0.4,
            n_steps=steps,
        )
        rng = np.random.default_rng(314159)
        counts = np.zeros(len(masks), dtype=int)
        powers = 1 << np.arange(3)
        for _ in range(30_000):
            initial_index = int(rng.choice(len(masks), p=independent))
            sampled = random_scan_gibbs_mask(
                model,
                masks[initial_index],
                eta=0.5,
                strength=0.4,
                n_steps=steps,
                rng=rng,
            )
            counts[int(sampled.astype(int) @ powers)] += 1

        empirical = counts / counts.sum()
        np.testing.assert_allclose(empirical, expected, atol=0.012)

    def test_finite_temperature_propagation_keeps_full_support(self):
        model = _three_bit_model()
        _, _, independent = _target_and_independent_distributions(model)
        propagated = propagate_random_scan_distribution(
            model,
            independent,
            eta=0.5,
            strength=0.7,
            n_steps=20,
        )

        assert np.all(propagated > 0.0)
        assert propagated.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "function,kwargs,match",
        [
            (
                random_scan_gibbs_mask,
                {"n_steps": -1, "rng": np.random.default_rng(0)},
                "n_steps",
            ),
            (
                random_scan_gibbs_mask,
                {"n_steps": 1, "rng": None},
                "rng",
            ),
            (
                random_scan_gibbs_mask,
                {
                    "n_steps": 1,
                    "rng": np.random.default_rng(0),
                    "logit_clip": 0.0,
                },
                "logit_clip",
            ),
        ],
    )
    def test_sampler_validation(self, function, kwargs, match):
        with pytest.raises(ValueError, match=match):
            function(
                _three_bit_model(),
                np.zeros(3),
                eta=0.5,
                strength=0.7,
                **kwargs,
            )


class TestFactorizedGibbsEvolutionStep:
    @staticmethod
    def _tables():
        current = pd.DataFrame({
            "a": [0, 0, 1, 1],
            "b": [0, 1, 0, 1],
            "c": [1, 0, 1, 0],
        })
        donors = current.iloc[[3, 2, 1, 0]].reset_index(drop=True)
        return current, donors

    def test_zero_sweeps_matches_existing_directional_step_and_rng(self):
        current, donors = self._tables()
        scores = np.random.default_rng(8).normal(size=(4, 3))
        existing_rng = np.random.default_rng(2026)
        factorized_rng = np.random.default_rng(2026)

        expected = evolve_step(
            current,
            donors,
            _three_bit_schema(),
            rho=0.75,
            eta=0.4,
            mu=0.6,
            rng=existing_rng,
            copy_direction_scores=scores,
            copy_direction_strength=0.7,
        )
        actual, diagnostics = evolve_step_factorized_gibbs(
            current,
            donors,
            _three_bit_schema(),
            _three_bit_queries(),
            np.ones(4),
            rho=0.75,
            eta=0.4,
            mu=0.6,
            rng=factorized_rng,
            copy_direction_scores=scores,
            copy_direction_strength=0.7,
            n_sweeps=0,
        )

        pd.testing.assert_frame_equal(actual, expected)
        np.testing.assert_array_equal(
            factorized_rng.random(20), existing_rng.random(20)
        )
        assert diagnostics["factor_count"] == 0
        assert diagnostics["gibbs_microsteps"] == 0

    def test_extra_sweeps_do_not_shift_primary_random_stream(self):
        current, donors = self._tables()
        scores = np.random.default_rng(9).normal(size=(4, 3))
        zero_rng = np.random.default_rng(77)
        candidate_rng = np.random.default_rng(77)

        evolve_step_factorized_gibbs(
            current,
            donors,
            _three_bit_schema(),
            _three_bit_queries(),
            np.ones(4),
            rho=1.0,
            eta=0.5,
            mu=0.5,
            rng=zero_rng,
            copy_direction_scores=scores,
            copy_direction_strength=0.5,
            n_sweeps=0,
        )
        _, diagnostics = evolve_step_factorized_gibbs(
            current,
            donors,
            _three_bit_schema(),
            _three_bit_queries(),
            np.ones(4),
            rho=1.0,
            eta=0.5,
            mu=0.5,
            rng=candidate_rng,
            gibbs_rng=np.random.default_rng(78),
            copy_direction_scores=scores,
            copy_direction_strength=0.5,
            n_sweeps=3,
        )

        np.testing.assert_array_equal(
            candidate_rng.random(20), zero_rng.random(20)
        )
        assert diagnostics["active_gibbs_rows"] == 4
        assert diagnostics["gibbs_microsteps"] > 0

    def test_same_primary_and_gibbs_seeds_are_reproducible(self):
        current, donors = self._tables()
        kwargs = dict(
            current=current,
            donors=donors,
            schema=_three_bit_schema(),
            queries=_three_bit_queries(),
            residual=np.ones(4),
            rho=1.0,
            eta=0.5,
            mu=0.0,
            copy_direction_scores=np.ones((4, 3)),
            copy_direction_strength=0.3,
            n_sweeps=4,
        )
        first, first_diagnostics = evolve_step_factorized_gibbs(
            **kwargs,
            rng=np.random.default_rng(10),
            gibbs_rng=np.random.default_rng(11),
        )
        second, second_diagnostics = evolve_step_factorized_gibbs(
            **kwargs,
            rng=np.random.default_rng(10),
            gibbs_rng=np.random.default_rng(11),
        )

        pd.testing.assert_frame_equal(first, second)
        assert first_diagnostics.keys() == second_diagnostics.keys()
        for key in first_diagnostics:
            if not key.endswith("elapsed_sec"):
                assert first_diagnostics[key] == second_diagnostics[key]

    @pytest.mark.parametrize("eta", [0.0, 1.0])
    def test_nonzero_sweeps_require_open_eta(self, eta):
        current, donors = self._tables()
        with pytest.raises(ValueError, match="eta"):
            evolve_step_factorized_gibbs(
                current,
                donors,
                _three_bit_schema(),
                _three_bit_queries(),
                np.ones(4),
                eta=eta,
                copy_direction_scores=np.zeros((4, 3)),
                copy_direction_strength=0.0,
                n_sweeps=1,
                rng=np.random.default_rng(0),
                gibbs_rng=np.random.default_rng(1),
            )
