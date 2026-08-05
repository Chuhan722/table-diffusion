"""低阶联合能量与随机扫描 Gibbs 扩散测试。"""

import numpy as np
import pandas as pd
import pytest

import table_diffevo.factorized_diffusion as factorized_diffusion
from table_diffevo.factorized_diffusion import (
    build_sparse_mask_energies_batch,
    build_sparse_mask_energy,
    compile_mask_workload,
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


def _one_bit_model(residual):
    schema = Schema([
        AttributeBlock(
            name="a",
            type="categorical",
            description="a",
            values=[0, 1],
        )
    ])
    return build_sparse_mask_energy(
        pd.DataFrame({"a": [0]}),
        pd.DataFrame({"a": [1]}),
        schema,
        [{
            "conditions": [{
                "attribute": "a", "operator": "==", "value": 1
            }]
        }],
        np.array([residual]),
        max_factor_order=1,
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


def _assert_sparse_models_equal(actual, expected):
    np.testing.assert_array_equal(
        actual.active_attribute_indices,
        expected.active_attribute_indices,
    )
    assert actual.active_attributes == expected.active_attributes
    assert actual.factors_by_variable == expected.factors_by_variable
    assert actual.n_queries == expected.n_queries
    assert actual.n_active_queries == expected.n_active_queries
    assert actual.max_active_query_order == expected.max_active_query_order
    assert len(actual.factors) == len(expected.factors)
    for actual_factor, expected_factor in zip(
        actual.factors, expected.factors
    ):
        assert actual_factor.scope == expected_factor.scope
        np.testing.assert_array_equal(
            actual_factor.values, expected_factor.values
        )


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

    @pytest.mark.parametrize("weighted", [False, True])
    def test_real_workload_matches_full_hybrid_oracle_for_all_masks(
        self, weighted
    ):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        rng = np.random.default_rng(20260802)
        current = init_synthetic_table(8, schema, rng)
        donors = current.iloc[rng.permutation(len(current))].reset_index(drop=True)
        residual = rng.normal(size=len(queries))
        weights = rng.uniform(0.1, 2.0, size=len(queries)) if weighted else None

        exact_landscapes = compute_joint_mask_landscapes(
            current,
            donors,
            np.arange(4),
            schema,
            queries,
            residual,
            weights=weights,
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
                weights=weights,
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

    @pytest.mark.parametrize(
        "queries,match",
        [
            ([None], "必须是字典"),
            ([{"conditions": [None]}], "必须是字典"),
        ],
    )
    def test_rejects_malformed_query_structure(self, queries, match):
        with pytest.raises(ValueError, match=match):
            build_sparse_mask_energy(
                pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
                pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
                _three_bit_schema(),
                queries,
                np.ones(len(queries)),
            )

    def test_rejects_weighted_residual_overflow(self):
        with pytest.raises(ValueError, match="float64"):
            build_sparse_mask_energy(
                pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
                pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
                _three_bit_schema(),
                [_three_bit_queries()[0]],
                np.array([np.finfo(float).max]),
                weights=np.array([2.0]),
            )

    def test_rejects_energy_overflow_across_distinct_factors(self):
        schema = Schema([
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("a", "b")
        ])
        queries = [
            {"conditions": [{
                "attribute": name, "operator": "==", "value": 1
            }]}
            for name in ("a", "b")
        ]
        model = build_sparse_mask_energy(
            pd.DataFrame({"a": [0], "b": [0]}),
            pd.DataFrame({"a": [1], "b": [1]}),
            schema,
            queries,
            np.full(2, np.finfo(float).max),
            max_factor_order=1,
        )

        with pytest.raises(ValueError, match="稀疏 mask 能量"):
            evaluate_sparse_mask_energy(model, np.ones(2))


class TestCompiledMaskWorkload:
    def test_compilation_deduplicates_conditions_and_freezes_templates(self):
        compiled = compile_mask_workload(
            _three_bit_schema(), _three_bit_queries()
        )

        assert compiled.attribute_names == ("a", "b", "c")
        assert compiled.n_queries == 4
        assert compiled.n_unique_conditions == 3
        assert compiled.max_factor_order == 3
        assert all(not masks.flags.writeable for masks in compiled._local_masks)
        with pytest.raises(ValueError, match="read-only"):
            compiled._local_masks[1][0, 0] = True

    @pytest.mark.parametrize("weighted", [False, True])
    def test_real_workload_batch_is_exactly_equal_to_rowwise_across_rounds(
        self, weighted
    ):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries(
            "configs/test_300x10/measured_50query.json"
        )
        compiled = compile_mask_workload(schema, queries)
        rng = np.random.default_rng(20260803)
        current = init_synthetic_table(12, schema, rng)
        donors = current.iloc[
            rng.permutation(len(current))
        ].reset_index(drop=True)
        weights = (
            rng.uniform(0.1, 2.0, size=len(queries))
            if weighted else None
        )

        assert compiled.n_unique_conditions == 35
        for _ in range(3):
            residual = rng.normal(size=len(queries))
            batched = build_sparse_mask_energies_batch(
                current,
                donors,
                schema,
                compiled,
                residual,
                weights=weights,
            )
            assert len(batched) == len(current)
            for row_index, actual in enumerate(batched):
                expected = build_sparse_mask_energy(
                    current.iloc[[row_index]],
                    donors.iloc[[row_index]],
                    schema,
                    queries,
                    residual,
                    weights=weights,
                )
                _assert_sparse_models_equal(actual, expected)
                masks = enumerate_copy_masks(
                    actual.n_active_attributes,
                    max_active_attributes=12,
                )
                np.testing.assert_array_equal(
                    evaluate_sparse_mask_energies(actual, masks),
                    evaluate_sparse_mask_energies(expected, masks),
                )

    def test_compiled_operands_do_not_alias_source_queries(self):
        queries = _three_bit_queries()
        compiled = compile_mask_workload(_three_bit_schema(), queries)
        queries[0]["conditions"][0]["value"] = 0

        model = build_sparse_mask_energies_batch(
            pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
            pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
            _three_bit_schema(),
            compiled,
            np.array([1.0, 2.0, 4.0, 8.0]),
        )[0]

        np.testing.assert_array_equal(
            evaluate_sparse_mask_energies(
                model, enumerate_copy_masks(3)
            ),
            evaluate_sparse_mask_energies(
                _three_bit_model(), enumerate_copy_masks(3)
            ),
        )

    def test_empty_batch_does_not_evaluate_conditions(self, monkeypatch):
        compiled = compile_mask_workload(
            _three_bit_schema(), _three_bit_queries()
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("空批次不应评价查询条件")

        monkeypatch.setattr(
            factorized_diffusion,
            "_evaluate_compiled_condition",
            fail_if_called,
        )
        empty = pd.DataFrame(columns=["a", "b", "c"])
        models = build_sparse_mask_energies_batch(
            empty,
            empty.copy(),
            _three_bit_schema(),
            compiled,
            np.ones(4),
        )

        assert models == ()

    @pytest.mark.parametrize(
        "queries,match",
        [
            (
                [{"conditions": [{
                    "attribute": "a", "operator": "!=", "value": 0
                }]}],
                "不支持的操作符",
            ),
            (
                [{"conditions": [{
                    "attribute": "a", "operator": "=="
                }]}],
                "缺少 value",
            ),
            (
                [{"conditions": [{
                    "attribute": "a", "operator": "between", "lower": 0
                }]}],
                "缺少",
            ),
        ],
    )
    def test_compile_rejects_invalid_conditions(self, queries, match):
        with pytest.raises(ValueError, match=match):
            compile_mask_workload(_three_bit_schema(), queries)

    def test_batch_rejects_row_count_and_schema_mismatch(self):
        compiled = compile_mask_workload(
            _three_bit_schema(), _three_bit_queries()
        )
        one = pd.DataFrame({"a": [0], "b": [0], "c": [0]})
        two = pd.concat([one, one], ignore_index=True)
        with pytest.raises(ValueError, match="行数"):
            build_sparse_mask_energies_batch(
                one,
                two,
                _three_bit_schema(),
                compiled,
                np.ones(4),
            )

        reordered = Schema([
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("b", "a", "c")
        ])
        with pytest.raises(ValueError, match="schema"):
            build_sparse_mask_energies_batch(
                one,
                one.copy(),
                reordered,
                compiled,
                np.ones(4),
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

    def test_random_scan_distribution_is_equivariant_to_schema_order(self):
        base_model = _three_bit_model()
        permuted_names = ("c", "a", "b")
        permuted_schema = Schema([
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in permuted_names
        ])
        permuted_model = build_sparse_mask_energy(
            pd.DataFrame({name: [0] for name in permuted_names}),
            pd.DataFrame({name: [1] for name in permuted_names}),
            permuted_schema,
            _three_bit_queries(),
            np.array([1.0, 2.0, 4.0, 8.0]),
        )

        base_masks, _, base_initial = _target_and_independent_distributions(
            base_model
        )
        permuted_masks, _, permuted_initial = (
            _target_and_independent_distributions(permuted_model)
        )
        base_result = propagate_random_scan_distribution(
            base_model, base_initial, eta=0.5, strength=0.7, n_steps=7
        )
        permuted_result = propagate_random_scan_distribution(
            permuted_model,
            permuted_initial,
            eta=0.5,
            strength=0.7,
            n_steps=7,
        )
        base_probabilities = {
            tuple(bool(value) for value in mask): probability
            for mask, probability in zip(base_masks, base_result)
        }
        for mask, probability in zip(permuted_masks, permuted_result):
            assignment = dict(zip(permuted_model.active_attributes, mask))
            base_assignment = tuple(
                bool(assignment[attr])
                for attr in base_model.active_attributes
            )
            assert probability == pytest.approx(
                base_probabilities[base_assignment], rel=0.0, abs=1e-14
            )

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

    @pytest.mark.parametrize("residual", [1.0e3, -1.0e3])
    def test_default_logit_guard_keeps_extreme_finite_support(self, residual):
        model = _one_bit_model(residual)

        probability = conditional_copy_probability(
            model,
            np.array([0]),
            variable=0,
            eta=0.5,
            strength=1.0,
        )
        propagated = propagate_random_scan_distribution(
            model,
            np.array([0.5, 0.5]),
            eta=0.5,
            strength=1.0,
            n_steps=1,
        )

        assert 0.0 < probability < 1.0
        assert np.all(propagated > 0.0)
        np.testing.assert_allclose(propagated, [1.0 - probability, probability])

    def test_neutral_condition_preserves_extreme_baseline_exactly(self):
        eta = np.nextafter(1.0, 0.0)
        model = build_sparse_mask_energy(
            pd.DataFrame({"a": [0], "b": [0], "c": [0]}),
            pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
            _three_bit_schema(),
            [],
            np.zeros(0),
        )

        probability = conditional_copy_probability(
            model,
            np.zeros(3),
            variable=0,
            eta=eta,
            strength=np.finfo(float).max,
        )

        assert probability == eta

    def test_unclipped_overflow_is_rejected_explicitly(self):
        model = _one_bit_model(np.finfo(float).max)

        guarded = conditional_copy_probability(
            model,
            np.array([0]),
            variable=0,
            eta=0.5,
            strength=2.0,
        )
        assert 0.0 < guarded < 1.0

        with pytest.raises(ValueError, match="float64"):
            conditional_copy_probability(
                model,
                np.array([0]),
                variable=0,
                eta=0.5,
                strength=2.0,
                logit_clip=None,
            )

    def test_rejects_probability_sum_overflow(self):
        with pytest.raises(ValueError, match="总和为正"):
            propagate_random_scan_distribution(
                _three_bit_model(),
                np.full(8, np.finfo(float).max),
                eta=0.5,
                strength=0.7,
                n_steps=1,
            )

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

    @pytest.mark.parametrize("eta", [0.0, 0.4, 1.0])
    def test_zero_sweeps_matches_existing_directional_step_and_rng(self, eta):
        current, donors = self._tables()
        scores = np.random.default_rng(8).normal(size=(4, 3))
        existing_rng = np.random.default_rng(2026)
        factorized_rng = np.random.default_rng(2026)
        unused_gibbs_rng = np.random.default_rng(2027)
        reference_gibbs_rng = np.random.default_rng(2027)

        expected = evolve_step(
            current,
            donors,
            _three_bit_schema(),
            rho=0.75,
            eta=eta,
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
            eta=eta,
            mu=0.6,
            rng=factorized_rng,
            gibbs_rng=unused_gibbs_rng,
            copy_direction_scores=scores,
            copy_direction_strength=0.7,
            n_sweeps=0,
        )

        pd.testing.assert_frame_equal(actual, expected)
        np.testing.assert_array_equal(
            factorized_rng.random(20), existing_rng.random(20)
        )
        np.testing.assert_array_equal(
            unused_gibbs_rng.random(20), reference_gibbs_rng.random(20)
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

    def test_compiled_builder_matches_rowwise_update_and_rng_across_rounds(
        self,
    ):
        schema = _three_bit_schema()
        queries = _three_bit_queries()
        compiled = compile_mask_workload(schema, queries)
        legacy_state, _ = self._tables()
        compiled_state = legacy_state.copy()
        control_rng = np.random.default_rng(20260803)
        legacy_rng = np.random.default_rng(301)
        compiled_rng = np.random.default_rng(301)
        legacy_gibbs_rng = np.random.default_rng(302)
        compiled_gibbs_rng = np.random.default_rng(302)

        for _ in range(20):
            permutation = control_rng.permutation(len(legacy_state))
            legacy_donors = legacy_state.iloc[
                permutation
            ].reset_index(drop=True)
            compiled_donors = compiled_state.iloc[
                permutation
            ].reset_index(drop=True)
            residual = control_rng.normal(size=len(queries))
            weights = control_rng.uniform(0.2, 1.7, size=len(queries))
            scores = control_rng.normal(size=(len(legacy_state), 3))

            legacy_state, legacy_diagnostics = (
                evolve_step_factorized_gibbs(
                    legacy_state,
                    legacy_donors,
                    schema,
                    queries,
                    residual,
                    weights=weights,
                    rho=0.75,
                    eta=0.4,
                    mu=0.25,
                    copy_direction_scores=scores,
                    copy_direction_strength=0.6,
                    n_sweeps=4,
                    rng=legacy_rng,
                    gibbs_rng=legacy_gibbs_rng,
                )
            )
            compiled_state, compiled_diagnostics = (
                evolve_step_factorized_gibbs(
                    compiled_state,
                    compiled_donors,
                    schema,
                    queries,
                    residual,
                    weights=weights,
                    rho=0.75,
                    eta=0.4,
                    mu=0.25,
                    copy_direction_scores=scores,
                    copy_direction_strength=0.6,
                    n_sweeps=4,
                    rng=compiled_rng,
                    gibbs_rng=compiled_gibbs_rng,
                    compiled_workload=compiled,
                )
            )

            pd.testing.assert_frame_equal(compiled_state, legacy_state)
            for key in (
                "participating_rows",
                "active_gibbs_rows",
                "active_blocks",
                "factor_count",
                "factor_table_entries",
                "gibbs_microsteps",
                "factor_model_builds",
            ):
                assert compiled_diagnostics[key] == legacy_diagnostics[key]
            assert legacy_diagnostics["factor_builder"] == "legacy_rowwise"
            assert compiled_diagnostics["factor_builder"] == "compiled_batch"
            assert legacy_diagnostics["condition_evaluation_batches"] == 0
            assert compiled_diagnostics[
                "condition_evaluation_batches"
            ] in (0, 1)
            assert compiled_diagnostics["compiled_unique_conditions"] == 3

        np.testing.assert_array_equal(
            compiled_rng.random(20), legacy_rng.random(20)
        )
        np.testing.assert_array_equal(
            compiled_gibbs_rng.random(20), legacy_gibbs_rng.random(20)
        )

    @pytest.mark.parametrize("same_donors", [False, True])
    def test_compiled_empty_active_set_skips_condition_evaluation_and_rng(
        self, monkeypatch, same_donors
    ):
        current, donors = self._tables()
        if same_donors:
            donors = current.copy()
            rho = 1.0
        else:
            rho = 0.0
        compiled = compile_mask_workload(
            _three_bit_schema(), _three_bit_queries()
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("空活跃集不应评价条件")

        monkeypatch.setattr(
            factorized_diffusion,
            "_evaluate_compiled_condition",
            fail_if_called,
        )
        gibbs_rng = np.random.default_rng(401)
        reference_gibbs_rng = np.random.default_rng(401)
        _, diagnostics = evolve_step_factorized_gibbs(
            current,
            donors,
            _three_bit_schema(),
            _three_bit_queries(),
            np.ones(4),
            rho=rho,
            eta=0.5,
            mu=0.0,
            copy_direction_scores=np.zeros((4, 3)),
            copy_direction_strength=0.0,
            n_sweeps=2,
            rng=np.random.default_rng(400),
            gibbs_rng=gibbs_rng,
            compiled_workload=compiled,
        )

        assert diagnostics["condition_evaluation_batches"] == 0
        assert diagnostics["factor_model_builds"] == 0
        np.testing.assert_array_equal(
            gibbs_rng.random(20), reference_gibbs_rng.random(20)
        )

    def test_zero_sweeps_ignore_invalid_compiled_workload(self):
        current, donors = self._tables()
        actual, diagnostics = evolve_step_factorized_gibbs(
            current,
            donors,
            _three_bit_schema(),
            [None],
            "unused",
            rho=1.0,
            eta=0.0,
            mu=0.0,
            copy_direction_scores=np.zeros((4, 3)),
            copy_direction_strength=0.0,
            n_sweeps=0,
            rng=np.random.default_rng(501),
            compiled_workload=object(),
        )
        expected = evolve_step(
            current,
            donors,
            _three_bit_schema(),
            rho=1.0,
            eta=0.0,
            mu=0.0,
            copy_direction_scores=np.zeros((4, 3)),
            copy_direction_strength=0.0,
            rng=np.random.default_rng(501),
        )

        pd.testing.assert_frame_equal(actual, expected)
        assert diagnostics["factor_builder"] == "not_used"
        assert diagnostics["compiled_unique_conditions"] == 0

    def test_nonzero_sweeps_reject_mismatched_compiled_workload_before_rng(self):
        current, donors = self._tables()
        schema = _three_bit_schema()
        queries = _three_bit_queries()
        compiled = compile_mask_workload(schema, queries)
        changed_queries = _three_bit_queries()
        changed_queries[0]["conditions"][0]["value"] = 0
        primary_rng = np.random.default_rng(601)
        reference_primary_rng = np.random.default_rng(601)
        gibbs_rng = np.random.default_rng(602)
        reference_gibbs_rng = np.random.default_rng(602)

        with pytest.raises(ValueError, match="不匹配"):
            evolve_step_factorized_gibbs(
                current,
                donors,
                schema,
                changed_queries,
                np.ones(4),
                rho=1.0,
                eta=0.5,
                mu=0.0,
                copy_direction_scores=np.zeros((4, 3)),
                copy_direction_strength=0.0,
                n_sweeps=1,
                rng=primary_rng,
                gibbs_rng=gibbs_rng,
                compiled_workload=compiled,
            )

        np.testing.assert_array_equal(
            primary_rng.random(20), reference_primary_rng.random(20)
        )
        np.testing.assert_array_equal(
            gibbs_rng.random(20), reference_gibbs_rng.random(20)
        )

    def test_update_does_not_mutate_inputs(self):
        current, donors = self._tables()
        current_before = current.copy(deep=True)
        donors_before = donors.copy(deep=True)
        residual = np.ones(4)
        residual_before = residual.copy()
        scores = np.ones((4, 3))
        scores_before = scores.copy()

        evolve_step_factorized_gibbs(
            current,
            donors,
            _three_bit_schema(),
            _three_bit_queries(),
            residual,
            rho=1.0,
            eta=0.5,
            mu=0.0,
            copy_direction_scores=scores,
            copy_direction_strength=0.3,
            n_sweeps=2,
            rng=np.random.default_rng(4),
            gibbs_rng=np.random.default_rng(5),
        )

        pd.testing.assert_frame_equal(current, current_before)
        pd.testing.assert_frame_equal(donors, donors_before)
        np.testing.assert_array_equal(residual, residual_before)
        np.testing.assert_array_equal(scores, scores_before)

    def test_no_active_rows_do_not_consume_gibbs_rng(self):
        current, _ = self._tables()
        gibbs_rng = np.random.default_rng(91)
        reference_rng = np.random.default_rng(91)

        _, diagnostics = evolve_step_factorized_gibbs(
            current,
            current.copy(),
            _three_bit_schema(),
            _three_bit_queries(),
            np.ones(4),
            rho=1.0,
            eta=0.5,
            mu=0.0,
            copy_direction_scores=np.zeros((4, 3)),
            copy_direction_strength=0.0,
            n_sweeps=2,
            rng=np.random.default_rng(90),
            gibbs_rng=gibbs_rng,
        )

        assert diagnostics["active_gibbs_rows"] == 0
        np.testing.assert_array_equal(
            gibbs_rng.random(20), reference_rng.random(20)
        )

    def test_nonzero_sweeps_validate_residual_without_participants(self):
        current, donors = self._tables()
        with pytest.raises(ValueError, match="residual"):
            evolve_step_factorized_gibbs(
                current,
                donors,
                _three_bit_schema(),
                _three_bit_queries(),
                np.ones(3),
                rho=0.0,
                eta=0.5,
                mu=0.0,
                copy_direction_scores=np.zeros((4, 3)),
                copy_direction_strength=0.0,
                n_sweeps=1,
                rng=np.random.default_rng(0),
                gibbs_rng=np.random.default_rng(1),
            )

    def test_nonzero_sweeps_validate_gibbs_logit_clip(self):
        current, donors = self._tables()
        with pytest.raises(ValueError, match="logit_clip"):
            evolve_step_factorized_gibbs(
                current,
                donors,
                _three_bit_schema(),
                _three_bit_queries(),
                np.ones(4),
                eta=0.5,
                copy_direction_scores=np.zeros((4, 3)),
                copy_direction_strength=0.0,
                n_sweeps=1,
                rng=np.random.default_rng(0),
                gibbs_rng=np.random.default_rng(1),
                gibbs_logit_clip=0.0,
            )

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
