"""残差驱动扩散方向与连续转移核测试。"""

import numpy as np
import pandas as pd
import pytest

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
    tilted_copy_probabilities,
)
from table_diffevo.evolution import run_evolution
from table_diffevo.generator import init_synthetic_table
from table_diffevo.queries import eval_query_mask, load_queries
from table_diffevo.schema import AttributeBlock, Schema, load_schema
from table_diffevo.update import evolve_step
from table_diffevo.vectorized_eval import evaluate_directional_potential
import table_diffevo.evolution as evolution_module


def _devices():
    devices = ["numpy"]
    try:
        import torch
    except ImportError:
        return devices
    devices.append("cpu")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _binary_schema():
    return Schema([
        AttributeBlock(
            name="a", type="categorical", description="a", values=[0, 1]
        ),
        AttributeBlock(
            name="b", type="categorical", description="b", values=[0, 1]
        ),
    ])


def _binary_queries():
    return [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "b", "operator": "==", "value": 1}]},
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
    ]


class TestDirectionalPotential:
    def test_matches_manual_query_contributions(self):
        schema = _binary_schema()
        queries = _binary_queries()
        rows = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
        residual = np.array([2.0, -1.0, 3.0])

        potential = evaluate_directional_potential(
            rows, queries, schema, residual, device="numpy"
        )

        np.testing.assert_allclose(potential, [0.0, -1.0, 2.0, 4.0])

    @pytest.mark.parametrize("device", _devices())
    def test_device_paths_match_manual_result(self, device):
        rows = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
        potential = evaluate_directional_potential(
            rows,
            _binary_queries(),
            _binary_schema(),
            np.array([2.0, -1.0, 3.0]),
            device=device,
            batch_size=2,
        )
        np.testing.assert_allclose(
            potential, [0.0, -1.0, 2.0, 4.0], atol=1e-6
        )

    def test_weights_are_applied(self):
        schema = _binary_schema()
        queries = _binary_queries()
        rows = pd.DataFrame({"a": [1], "b": [1]})
        residual = np.array([1.0, 2.0, 3.0])
        weights = np.array([2.0, 0.5, 4.0])

        potential = evaluate_directional_potential(
            rows, queries, schema, residual, weights=weights
        )

        assert potential[0] == pytest.approx(15.0)

    @pytest.mark.parametrize("device", _devices())
    def test_four_condition_query_uses_correct_fallback(self, device):
        rows = pd.DataFrame({"a": [0, 1, 1], "b": [1, 0, 1]})
        query = {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        }

        potential = evaluate_directional_potential(
            rows,
            [query],
            _binary_schema(),
            np.array([2.5]),
            device=device,
            verbose=False,
        )

        np.testing.assert_array_equal(potential, [0.0, 0.0, 2.5])

    @pytest.mark.parametrize(
        "bad_residual",
        [np.array([1.0]), np.array([1.0, np.nan, 0.0]), np.array(["x"] * 3)],
    )
    def test_rejects_invalid_residual(self, bad_residual):
        with pytest.raises(ValueError, match="residual"):
            evaluate_directional_potential(
                pd.DataFrame({"a": [0], "b": [0]}),
                _binary_queries(),
                _binary_schema(),
                bad_residual,
            )


class TestCopyDirectionScores:
    def test_scores_actual_recipient_conditioned_block_edit(self):
        schema = _binary_schema()
        queries = _binary_queries()
        current = pd.DataFrame({"a": [0, 1, 0], "b": [1, 1, 0]})
        donors = pd.DataFrame({"a": [1, 0, 1], "b": [1, 1, 0]})
        residual = np.array([0.5, 0.0, 1.0])

        scores = compute_copy_direction_scores(
            current, donors, schema, queries, residual
        )

        # 01->11 同时打开 a 与 a&b；11->01 同时关闭；00->10 只打开 a。
        np.testing.assert_allclose(scores[:, 0], [1.5, -1.5, 0.5])
        np.testing.assert_array_equal(scores[:, 1], np.zeros(3))

    def test_same_donor_block_has_zero_direction(self):
        current = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
        scores = compute_copy_direction_scores(
            current,
            current.copy(),
            _binary_schema(),
            _binary_queries(),
            np.ones(3),
        )
        np.testing.assert_array_equal(scores, np.zeros((2, 2)))

    @pytest.mark.parametrize("device", _devices())
    def test_matches_bruteforce_full_query_delta(self, device):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries(
            "configs/test_300x10/measured_50query.json"
        )
        rng = np.random.default_rng(20260801)
        current = init_synthetic_table(24, schema, rng)
        donors = current.iloc[rng.permutation(len(current))].reset_index(
            drop=True
        )
        residual = rng.normal(size=len(queries))
        weights = rng.uniform(0.1, 2.0, size=len(queries))

        actual = compute_copy_direction_scores(
            current,
            donors,
            schema,
            queries,
            residual,
            weights=weights,
            batch_size=7,
            device=device,
        )

        base_contributions = np.column_stack([
            eval_query_mask(current, query) for query in queries
        ]).astype(float)
        expected = np.zeros_like(actual)
        weighted_residual = weights * residual
        for attr_idx, attr in enumerate(schema.attribute_names()):
            candidates = current.copy()
            candidates[attr] = donors[attr].to_numpy()
            candidate_contributions = np.column_stack([
                eval_query_mask(candidates, query) for query in queries
            ]).astype(float)
            expected[:, attr_idx] = (
                candidate_contributions - base_contributions
            ) @ weighted_residual

        np.testing.assert_allclose(actual, expected, atol=1e-5)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"batch_size": 0}, "batch_size"),
            ({"batch_size": True}, "batch_size"),
            ({"device": "tpu"}, "device"),
        ],
    )
    def test_computation_controls_are_validated_even_without_edits(
        self, kwargs, match
    ):
        current = pd.DataFrame({"a": [0], "b": [0]})
        with pytest.raises(ValueError, match=match):
            compute_copy_direction_scores(
                current,
                current.copy(),
                _binary_schema(),
                _binary_queries(),
                np.ones(3),
                **kwargs,
            )


class TestContinuousCopyKernel:
    def test_rms_scale_is_stable_for_empty_zero_and_extreme_values(self):
        assert direction_rms_scale(np.array([])) == 0.0
        assert direction_rms_scale(np.zeros((2, 3))) == 0.0
        assert direction_rms_scale(np.array([3.0, 4.0])) == pytest.approx(
            np.sqrt(12.5)
        )
        extreme = direction_rms_scale(np.array([1.0e300, -1.0e300]))
        assert np.isfinite(extreme)
        assert extreme == pytest.approx(1.0e300)

    @pytest.mark.parametrize(
        "scores",
        [np.array([np.nan]), np.array([np.inf]), np.array(["bad"])],
    )
    def test_rms_scale_rejects_invalid_values(self, scores):
        with pytest.raises(ValueError, match="direction_scores"):
            direction_rms_scale(scores)

    def test_rms_normalized_temperature_is_scale_invariant(self):
        scores = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
        scaled = 37.0 * scores
        temperature = 0.8

        original = tilted_copy_probabilities(
            0.5,
            scores,
            temperature / direction_rms_scale(scores),
        )
        rescaled = tilted_copy_probabilities(
            0.5,
            scaled,
            temperature / direction_rms_scale(scaled),
        )

        np.testing.assert_allclose(original, rescaled)

    def test_positive_is_higher_negative_is_lower_and_neutral_is_exact(self):
        probs = tilted_copy_probabilities(
            0.5, np.array([-2.0, 0.0, 2.0]), strength=1.0
        )
        assert 0.0 < probs[0] < 0.5
        assert probs[1] == 0.5
        assert 0.5 < probs[2] < 1.0
        assert probs[0] == pytest.approx(1.0 - probs[2])

    def test_finite_extreme_scores_keep_reverse_support(self):
        probs = tilted_copy_probabilities(
            0.5, np.array([-1.0e100, 1.0e100]), strength=1.0e100
        )
        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)

    def test_eta_endpoints_and_strength_zero_are_exact(self):
        scores = np.array([-3.0, 0.0, 4.0])
        np.testing.assert_array_equal(
            tilted_copy_probabilities(0.0, scores, 2.0), np.zeros(3)
        )
        np.testing.assert_array_equal(
            tilted_copy_probabilities(1.0, scores, 2.0), np.ones(3)
        )
        np.testing.assert_array_equal(
            tilted_copy_probabilities(0.3, scores, 0.0), np.full(3, 0.3)
        )

    @pytest.mark.parametrize(
        "eta,scores,strength,match",
        [
            (-0.1, np.array([0.0]), 1.0, "eta"),
            (np.nan, np.array([0.0]), 1.0, "eta"),
            (True, np.array([0.0]), 1.0, "eta"),
            (0.5, np.array([np.nan]), 1.0, "direction_scores"),
            (0.5, np.array(["bad"]), 1.0, "direction_scores"),
            (0.5, np.array([0.0]), -1.0, "strength"),
            (0.5, np.array([0.0]), True, "strength"),
        ],
    )
    def test_rejects_invalid_kernel_inputs(
        self, eta, scores, strength, match
    ):
        with pytest.raises(ValueError, match=match):
            tilted_copy_probabilities(eta, scores, strength)

    def test_negative_direction_is_not_gated_out(self):
        n = 1000
        schema = Schema([
            AttributeBlock(
                name="a", type="categorical", description="a", values=[0, 1]
            )
        ])
        current = pd.DataFrame({"a": np.zeros(n, dtype=int)})
        donors = pd.DataFrame({"a": np.ones(n, dtype=int)})
        negative_scores = -np.ones((n, 1))
        positive_scores = np.ones((n, 1))

        negative = evolve_step(
            current,
            donors,
            schema,
            rho=1.0,
            eta=0.5,
            mu=0.0,
            rng=np.random.default_rng(9),
            copy_direction_scores=negative_scores,
            copy_direction_strength=1.0,
        )
        positive = evolve_step(
            current,
            donors,
            schema,
            rho=1.0,
            eta=0.5,
            mu=0.0,
            rng=np.random.default_rng(9),
            copy_direction_scores=positive_scores,
            copy_direction_strength=1.0,
        )

        negative_copies = int(negative["a"].sum())
        positive_copies = int(positive["a"].sum())
        assert 0 < negative_copies < positive_copies < n

    def test_strength_zero_preserves_table_and_rng_stream(self):
        n = 100
        schema = _binary_schema()
        current = pd.DataFrame({
            "a": np.arange(n) % 2,
            "b": (np.arange(n) // 2) % 2,
        })
        donors = current.iloc[::-1].reset_index(drop=True)
        scores = np.random.default_rng(17).normal(size=(n, 2))
        rng_baseline = np.random.default_rng(23)
        rng_endpoint = np.random.default_rng(23)

        baseline = evolve_step(
            current, donors, schema, rho=0.7, eta=0.3, mu=0.2,
            rng=rng_baseline,
        )
        endpoint = evolve_step(
            current, donors, schema, rho=0.7, eta=0.3, mu=0.2,
            rng=rng_endpoint,
            copy_direction_scores=scores,
            copy_direction_strength=0.0,
        )

        pd.testing.assert_frame_equal(baseline, endpoint)
        np.testing.assert_array_equal(
            rng_baseline.random(20), rng_endpoint.random(20)
        )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"copy_direction_strength": 1.0}, "必须提供"),
            (
                {"copy_direction_scores": np.zeros((2, 1))},
                "shape",
            ),
            (
                {
                    "copy_direction_scores": np.array(
                        [[0.0, np.nan], [0.0, 0.0]]
                    )
                },
                "有限",
            ),
            (
                {
                    "copy_direction_scores": np.zeros((2, 2)),
                    "copy_direction_strength": -1.0,
                },
                "非负有限",
            ),
        ],
    )
    def test_validation(self, kwargs, match):
        current = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
        with pytest.raises(ValueError, match=match):
            evolve_step(
                current, current.copy(), _binary_schema(), **kwargs
            )


class TestEvolutionIntegration:
    @pytest.mark.parametrize("normalization", ["none", "initial_rms"])
    def test_enabled_strength_zero_matches_disabled_path(self, normalization):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([query["result"] for query in queries])
        common = dict(
            n_records=100,
            n_rounds=20,
            seed=37,
            device="numpy",
            log_every=100,
        )

        baseline, baseline_diag = run_evolution(
            target, queries, schema, **common
        )
        endpoint, endpoint_diag = run_evolution(
            target,
            queries,
            schema,
            residual_directed_diffusion=True,
            diffusion_direction_strength=0.0,
            diffusion_direction_normalization=normalization,
            **common,
        )

        pd.testing.assert_frame_equal(baseline, endpoint)
        for key in (
            "loss_history",
            "accept_history",
            "donor_fitness_history",
            "donor_distance_history",
            "donor_self_rate_history",
            "alpha_history",
            "proposal_attempts_history",
            "accepted_attempt_history",
            "accepted_rho_history",
            "raw_proposal_gain_history",
            "raw_proposal_linear_gain_history",
            "raw_proposal_quadratic_penalty_history",
        ):
            assert endpoint_diag[key] == baseline_diag[key]

    def test_initial_rms_scale_is_fixed_after_first_nonzero_round(self):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([query["result"] for query in queries])

        _, diagnostics = run_evolution(
            target,
            queries,
            schema,
            n_records=100,
            n_rounds=8,
            seed=51,
            residual_directed_diffusion=True,
            diffusion_direction_strength=0.7,
            diffusion_direction_normalization="initial_rms",
            device="numpy",
            log_every=100,
        )

        scale = diagnostics["direction_reference_scale"]
        assert scale is not None
        assert scale > 0.0
        present_scales = [
            value
            for value in diagnostics["direction_reference_scale_history"]
            if value is not None
        ]
        assert present_scales
        np.testing.assert_array_equal(present_scales, scale)
        effective = [
            value
            for value in diagnostics["effective_direction_strength_history"]
            if value is not None
        ]
        np.testing.assert_allclose(effective, 0.7 / scale)

    def test_raw_proposal_gain_decomposition_is_exact(self):
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([query["result"] for query in queries])

        _, diagnostics = run_evolution(
            target,
            queries,
            schema,
            n_records=80,
            n_rounds=8,
            seed=4,
            device="numpy",
            log_every=100,
        )

        for gains, linear, quadratic in zip(
            diagnostics["raw_proposal_gain_history"],
            diagnostics["raw_proposal_linear_gain_history"],
            diagnostics["raw_proposal_quadratic_penalty_history"],
        ):
            np.testing.assert_allclose(
                gains, np.asarray(linear) - np.asarray(quadratic)
            )

    def test_disabled_path_does_not_compute_directions(self, monkeypatch):
        monkeypatch.setattr(
            evolution_module,
            "compute_copy_direction_scores",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("默认关闭路径不应计算局部方向")
            ),
        )
        schema = _binary_schema()
        queries = _binary_queries()
        run_evolution(
            np.array([1, 1, 1]),
            queries,
            schema,
            n_records=6,
            n_rounds=2,
            seed=1,
            device="numpy",
            log_every=100,
        )

    def test_retry_reuses_same_direction_matrix(self, monkeypatch):
        schema = Schema([
            AttributeBlock(
                name="a", type="categorical", description="a", values=[0, 1]
            )
        ])
        queries = [
            {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]}
        ]
        initial = pd.DataFrame({"a": [1, 0, 0, 0]})
        direction = np.array([[1.0], [-1.0], [0.0], [0.5]])
        direction_calls = 0
        seen_direction_ids = []
        proposal_calls = 0

        monkeypatch.setattr(
            evolution_module,
            "init_synthetic_table",
            lambda *args, **kwargs: initial.copy(),
        )

        def fake_direction(*args, **kwargs):
            nonlocal direction_calls
            direction_calls += 1
            return direction

        def fake_evolve(*args, **kwargs):
            nonlocal proposal_calls
            proposal_calls += 1
            seen_direction_ids.append(id(kwargs["copy_direction_scores"]))
            if proposal_calls == 1:
                return pd.DataFrame({"a": [1, 1, 1, 1]})
            return pd.DataFrame({"a": [0, 0, 0, 0]})

        monkeypatch.setattr(
            evolution_module, "compute_copy_direction_scores", fake_direction
        )
        monkeypatch.setattr(evolution_module, "evolve_step", fake_evolve)

        _, diagnostics = run_evolution(
            np.array([0]),
            queries,
            schema,
            n_records=4,
            n_rounds=1,
            seed=0,
            max_retries=1,
            residual_directed_diffusion=True,
            diffusion_direction_strength=2.0,
            device="numpy",
            log_every=100,
        )

        assert diagnostics["accepted_attempt_history"] == [2]
        assert direction_calls == 1
        assert seen_direction_ids == [id(direction), id(direction)]
        assert diagnostics["direction_evaluation_count"] == 1

    @pytest.mark.parametrize(
        "value",
        [-1.0, np.nan, np.inf, True, "1"],
    )
    def test_direction_strength_validation(self, value):
        with pytest.raises(ValueError, match="diffusion_direction_strength"):
            run_evolution(
                np.array([1]),
                [{
                    "conditions": [
                        {"attribute": "a", "operator": "==", "value": 1}
                    ]
                }],
                Schema([
                    AttributeBlock(
                        name="a", type="categorical", description="a",
                        values=[0, 1],
                    )
                ]),
                n_records=2,
                n_rounds=0,
                diffusion_direction_strength=value,
            )

    def test_direction_mode_requires_boolean(self):
        with pytest.raises(ValueError, match="residual_directed_diffusion"):
            run_evolution(
                np.array([1]),
                [{
                    "conditions": [
                        {"attribute": "a", "operator": "==", "value": 1}
                    ]
                }],
                Schema([
                    AttributeBlock(
                        name="a", type="categorical", description="a",
                        values=[0, 1],
                    )
                ]),
                n_records=2,
                n_rounds=0,
                residual_directed_diffusion="yes",
            )

    def test_direction_normalization_validation(self):
        with pytest.raises(
            ValueError, match="diffusion_direction_normalization"
        ):
            run_evolution(
                np.array([1]),
                [{
                    "conditions": [
                        {"attribute": "a", "operator": "==", "value": 1}
                    ]
                }],
                Schema([
                    AttributeBlock(
                        name="a", type="categorical", description="a",
                        values=[0, 1],
                    )
                ]),
                n_records=2,
                n_rounds=0,
                diffusion_direction_normalization="per_round",
            )
