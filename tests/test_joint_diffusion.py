"""联合属性块 Gibbs 扩散 oracle 的单元与控制测试。"""

import numpy as np
import pandas as pd
import pytest

from table_diffevo.directional_diffusion import tilted_copy_probabilities
from table_diffevo.joint_diffusion import (
    additive_mask_directions,
    baseline_mask_log_probabilities,
    categorical_entropy,
    categorical_kl,
    compute_joint_mask_landscapes,
    enumerate_copy_masks,
    gibbs_mask_log_probabilities,
    independent_mask_log_probabilities,
    mask_distribution_diagnostics,
    match_gibbs_strength_for_kl,
    sample_mask_index,
)
from table_diffevo.schema import AttributeBlock, Schema


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


class TestMaskEnumeration:
    def test_binary_order_and_empty_mask(self):
        masks = enumerate_copy_masks(3)

        np.testing.assert_array_equal(
            masks[:5],
            [
                [False, False, False],
                [True, False, False],
                [False, True, False],
                [True, True, False],
                [False, False, True],
            ],
        )
        assert masks.shape == (8, 3)
        assert enumerate_copy_masks(0).shape == (1, 0)

    @pytest.mark.parametrize(
        "n_active,maximum,match",
        [
            (3, 2, "护栏"),
            (-1, 4, "非负整数"),
            (True, 4, "非负整数"),
            (1, 21, "绝对护栏"),
        ],
    )
    def test_enumeration_guard_and_validation(self, n_active, maximum, match):
        with pytest.raises(ValueError, match=match):
            enumerate_copy_masks(
                n_active, max_active_attributes=maximum
            )


class TestJointGibbsKernel:
    def test_additive_energy_exactly_recovers_independent_logistic_kernel(self):
        eta = 0.3
        strength = 0.7
        single = np.array([-1.0, 0.2, 2.0])
        masks = enumerate_copy_masks(len(single))
        reference = baseline_mask_log_probabilities(masks, eta)
        additive = additive_mask_directions(masks, single)

        joint = gibbs_mask_log_probabilities(
            reference, additive, strength
        )
        copy_probabilities = tilted_copy_probabilities(
            eta, single, strength
        )
        independent = independent_mask_log_probabilities(
            masks, copy_probabilities
        )

        np.testing.assert_allclose(joint, independent, atol=1e-14)
        np.testing.assert_allclose(np.exp(joint).sum(), 1.0)

    def test_interaction_tilt_changes_only_the_joint_distribution(self):
        masks = enumerate_copy_masks(2)
        reference = baseline_mask_log_probabilities(masks, 0.5)
        additive = additive_mask_directions(masks, np.array([0.2, 0.2]))
        exact = additive + 2.0 * (masks[:, 0] & masks[:, 1])

        independent = gibbs_mask_log_probabilities(
            reference, additive, 1.0
        )
        joint = gibbs_mask_log_probabilities(reference, exact, 1.0)

        assert np.exp(joint[-1]) > np.exp(independent[-1])
        assert not np.allclose(joint, independent)
        assert np.all(np.isfinite(joint))
        assert np.all(np.exp(joint) > 0.0)

    def test_empty_mask_distribution_is_a_point_mass(self):
        masks = enumerate_copy_masks(0)
        reference = baseline_mask_log_probabilities(masks, 0.4)
        independent = independent_mask_log_probabilities(
            masks, np.array([])
        )
        tilted = gibbs_mask_log_probabilities(
            reference, np.array([0.0]), 100.0
        )

        np.testing.assert_array_equal(reference, [0.0])
        np.testing.assert_array_equal(independent, [0.0])
        np.testing.assert_array_equal(tilted, [0.0])
        assert categorical_kl(tilted, reference) == 0.0
        assert categorical_entropy(tilted) == 0.0

    def test_distribution_diagnostics_keep_reverse_support(self):
        masks = enumerate_copy_masks(2)
        reference = baseline_mask_log_probabilities(masks, 0.5)
        directions = np.array([0.0, -2.0, 1.0, 3.0])
        tilted = gibbs_mask_log_probabilities(
            reference, directions, 1.5
        )

        diagnostics = mask_distribution_diagnostics(
            tilted, reference, directions
        )

        assert diagnostics["kl_to_baseline"] > 0.0
        assert 0.0 < diagnostics["entropy"] < np.log(4.0)
        assert diagnostics["negative_direction_mass"] > 0.0
        assert diagnostics["positive_direction_mass"] > 0.0
        assert sum(
            diagnostics[key]
            for key in (
                "negative_direction_mass",
                "neutral_direction_mass",
                "positive_direction_mass",
            )
        ) == pytest.approx(1.0)

    def test_shared_gumbels_give_deterministic_paired_samples(self):
        masks = enumerate_copy_masks(2)
        reference = baseline_mask_log_probabilities(masks, 0.5)
        candidate = gibbs_mask_log_probabilities(
            reference, np.array([0.0, 0.2, -0.1, 1.0]), 2.0
        )
        gumbels = np.array([-0.5, 0.1, 0.2, 0.0])

        first = sample_mask_index(candidate, gumbels)
        second = sample_mask_index(candidate.copy(), gumbels.copy())

        assert first == second
        assert 0 <= first < len(masks)

    def test_kl_matching_reaches_independent_kernel_budget(self):
        masks = enumerate_copy_masks(3)
        reference = baseline_mask_log_probabilities(masks, 0.5)
        additive = additive_mask_directions(
            masks, np.array([-0.7, 0.3, 1.1])
        )
        independent = gibbs_mask_log_probabilities(
            reference, additive, 1.8
        )
        target_kl = categorical_kl(independent, reference)
        exact = additive + 0.8 * (
            masks[:, 0] & masks[:, 1]
        ) - 0.4 * (masks[:, 1] & masks[:, 2])

        matched_strength = match_gibbs_strength_for_kl(
            reference, exact, target_kl
        )
        matched = gibbs_mask_log_probabilities(
            reference, exact, matched_strength
        )

        assert matched_strength > 0.0
        assert categorical_kl(matched, reference) == pytest.approx(
            target_kl, rel=1e-9, abs=1e-11
        )

    def test_kl_matching_rejects_unreachable_target(self):
        reference = np.log(np.full(4, 0.25))
        directions = np.array([0.0, 0.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="正温极限"):
            match_gibbs_strength_for_kl(
                reference, directions, target_kl=np.log(4.0) + 0.1
            )
        with pytest.raises(ValueError, match="常数方向"):
            match_gibbs_strength_for_kl(
                reference, np.zeros(4), target_kl=0.1
            )

    @pytest.mark.parametrize(
        "function,args,match",
        [
            (
                baseline_mask_log_probabilities,
                (np.array([[0], [0]]), 0.5),
                "重复",
            ),
            (
                baseline_mask_log_probabilities,
                (enumerate_copy_masks(1), 0.0),
                "eta",
            ),
            (
                independent_mask_log_probabilities,
                (enumerate_copy_masks(2), np.array([0.5])),
                "属性数",
            ),
            (
                gibbs_mask_log_probabilities,
                (np.log(np.full(2, 0.5)), np.array([0.0, np.nan]), 1.0),
                "directions",
            ),
            (
                sample_mask_index,
                (np.log(np.full(2, 0.5)), np.array([0.0, np.inf])),
                "gumbels",
            ),
        ],
    )
    def test_invalid_inputs_are_rejected(self, function, args, match):
        with pytest.raises(ValueError, match=match):
            function(*args)


class TestJointMaskLandscapes:
    @pytest.mark.parametrize("device", _devices())
    def test_exact_hybrid_directions_expose_nonadditive_interaction(
        self, device
    ):
        current = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
        donors = pd.DataFrame({"a": [1, 0], "b": [1, 0]})

        landscapes = compute_joint_mask_landscapes(
            current,
            donors,
            [0, 1],
            _binary_schema(),
            _binary_queries(),
            np.array([1.0, 2.0, 4.0]),
            device=device,
        )

        assert [item.row_index for item in landscapes] == [0, 1]
        assert landscapes[0].active_attributes == ("a", "b")
        np.testing.assert_array_equal(
            landscapes[0].masks, enumerate_copy_masks(2)
        )
        np.testing.assert_allclose(
            landscapes[0].directions, [0.0, 1.0, 2.0, 7.0], atol=1e-6
        )
        np.testing.assert_allclose(
            landscapes[1].directions,
            [0.0, -5.0, -6.0, -7.0],
            atol=1e-6,
        )
        additive = additive_mask_directions(
            landscapes[0].masks,
            landscapes[0].directions[[1, 2]],
        )
        np.testing.assert_allclose(
            landscapes[0].directions - additive,
            [0.0, 0.0, 0.0, 4.0],
            atol=1e-6,
        )

    def test_same_row_has_one_empty_mask_and_zero_direction(self):
        current = pd.DataFrame({"a": [0], "b": [1]})
        landscapes = compute_joint_mask_landscapes(
            current,
            current.copy(),
            np.array([0]),
            _binary_schema(),
            _binary_queries(),
            np.ones(3),
        )

        assert landscapes[0].active_attributes == ()
        assert landscapes[0].masks.shape == (1, 0)
        np.testing.assert_array_equal(landscapes[0].directions, [0.0])

    def test_empty_row_selection_returns_empty_list(self):
        current = pd.DataFrame({"a": [0], "b": [1]})
        landscapes = compute_joint_mask_landscapes(
            current,
            current.copy(),
            [],
            _binary_schema(),
            _binary_queries(),
            np.ones(3),
        )

        assert landscapes == []

    def test_exact_enumeration_guard_is_enforced(self):
        current = pd.DataFrame({"a": [0], "b": [0]})
        donors = pd.DataFrame({"a": [1], "b": [1]})

        with pytest.raises(ValueError, match="护栏"):
            compute_joint_mask_landscapes(
                current,
                donors,
                [0],
                _binary_schema(),
                _binary_queries(),
                np.ones(3),
                max_active_attributes=1,
            )

    @pytest.mark.parametrize(
        "row_indices,match",
        [([1], "越界"), ([0, 0], "不得重复"), ([0.0], "整数")],
    )
    def test_row_index_validation(self, row_indices, match):
        current = pd.DataFrame({"a": [0], "b": [0]})
        with pytest.raises(ValueError, match=match):
            compute_joint_mask_landscapes(
                current,
                current.copy(),
                row_indices,
                _binary_schema(),
                _binary_queries(),
                np.ones(3),
            )
