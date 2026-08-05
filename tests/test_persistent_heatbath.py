"""持久化 workload 能量热浴扩散的语义、增量与精确 oracle 测试。"""

import itertools

import numpy as np
import pandas as pd
import pytest

from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.persistent_heatbath import (
    HeatbathConditional,
    PersistentHeatbathState,
    apply_heatbath_choice,
    build_persistent_heatbath_conditional,
    heatbath_probabilities,
    initial_gain_rms_scale,
    initialize_persistent_heatbath_state,
    legal_attribute_values,
    persistent_heatbath_step,
    sample_heatbath_index,
    verify_persistent_heatbath_state,
)
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import AttributeBlock, Schema, load_schema


ORACLE_BETAS = (0.0, 0.7, 1.3)
ORACLE_STATES = tuple(itertools.product((0, 1), repeat=4))
ORACLE_INDEX = {state: index for index, state in enumerate(ORACLE_STATES)}


def _binary_schema():
    return Schema([
        AttributeBlock(
            name="a", type="categorical", description="", values=[0, 1]
        ),
        AttributeBlock(
            name="b", type="categorical", description="", values=[0, 1]
        ),
    ])


def _binary_queries():
    return [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
    ]


def _binary_table(bits):
    return pd.DataFrame({
        "a": [bits[0], bits[2]],
        "b": [bits[1], bits[3]],
    })


def _binary_state(bits=(0, 0, 1, 0), target=None):
    if target is None:
        target = np.asarray([1, 1, 1])
    return initialize_persistent_heatbath_state(
        _binary_table(bits), _binary_schema(), _binary_queries(), target
    )


def _conditional(state, coordinate, beta=0.7, target=None):
    if target is None:
        target = np.asarray([1, 1, 1])
    row, attribute = divmod(coordinate, 2)
    return build_persistent_heatbath_conditional(
        state,
        _binary_schema(),
        _binary_queries(),
        target,
        row_index=row,
        attribute_index=attribute,
        inverse_temperature=beta,
    )


def _full_candidate(table, coordinate, value):
    result = table.copy(deep=True)
    row, attribute = divmod(coordinate, 2)
    result.iat[row, attribute] = value
    return result


def _transition_matrix(beta):
    schema = _binary_schema()
    queries = _binary_queries()
    target = np.asarray([1, 1, 1])
    matrix = np.zeros((16, 16), dtype=float)
    losses = np.zeros(16, dtype=float)
    conditionals = {}
    for state_index, bits in enumerate(ORACLE_STATES):
        state = initialize_persistent_heatbath_state(
            _binary_table(bits), schema, queries, target
        )
        losses[state_index] = state.loss
        for coordinate in range(4):
            row, attribute = divmod(coordinate, 2)
            conditional = build_persistent_heatbath_conditional(
                state,
                schema,
                queries,
                target,
                row_index=row,
                attribute_index=attribute,
                inverse_temperature=beta,
            )
            conditionals[(state_index, coordinate)] = conditional
            for choice, value in enumerate(conditional.values):
                next_bits = list(bits)
                next_bits[coordinate] = value
                matrix[state_index, ORACLE_INDEX[tuple(next_bits)]] += (
                    conditional.probabilities[choice] / 4.0
                )
    weights = np.exp(-beta * losses / 2.0)
    stationary = weights / weights.sum()
    return matrix, stationary, losses, conditionals


def test_legal_values_cover_categorical_and_integer_numeric_domains():
    categorical = AttributeBlock(
        name="kind", type="categorical", description="", values=["x", "y"]
    )
    numeric = AttributeBlock(
        name="age", type="numeric", description="", range=[-1, 2]
    )

    assert legal_attribute_values(categorical) == ("x", "y")
    assert legal_attribute_values(numeric) == (-1, 0, 1, 2)


@pytest.mark.parametrize(
    "block,match",
    [
        (
            AttributeBlock(
                name="x", type="categorical", description="", values=[]
            ),
            "不能为空",
        ),
        (
            AttributeBlock(
                name="x", type="categorical", description="", values=[1, 1]
            ),
            "不能重复",
        ),
        (
            AttributeBlock(
                name="x", type="categorical", description="", values=[np.nan]
            ),
            "NaN",
        ),
        (
            AttributeBlock(
                name="x", type="numeric", description="", range=[0.5, 2]
            ),
            "整数端点",
        ),
        (
            AttributeBlock(
                name="x", type="numeric", description="", range=[2, 1]
            ),
            "下界",
        ),
    ],
)
def test_legal_values_reject_invalid_domains(block, match):
    with pytest.raises(ValueError, match=match):
        legal_attribute_values(block)


def test_initialization_normalizes_columns_index_and_builds_exact_state():
    table = _binary_table((0, 0, 1, 0)).assign(extra=9)
    table.index = [8, 9]
    state = initialize_persistent_heatbath_state(
        table,
        _binary_schema(),
        _binary_queries(),
        np.asarray([1, 1, 1]),
    )

    assert list(state.table.columns) == ["a", "b"]
    assert state.table.index.equals(pd.RangeIndex(2))
    assert state.query_answers.tolist() == [1, 0, 0]
    assert state.loss == 1.0
    assert verify_persistent_heatbath_state(
        state,
        _binary_schema(),
        _binary_queries(),
        np.asarray([1, 1, 1]),
    )["loss_abs_error"] == 0.0


@pytest.mark.parametrize("coordinate", range(4))
def test_candidate_deltas_gains_and_losses_match_full_table_recomputation(
    coordinate,
):
    state = _binary_state()
    conditional = _conditional(state, coordinate)
    queries = _binary_queries()
    target = np.asarray([1, 1, 1])

    for choice, value in enumerate(conditional.values):
        candidate = _full_candidate(state.table, coordinate, value)
        answers = evaluate_table(candidate, queries)
        loss = compute_loss(target, answers)
        np.testing.assert_array_equal(
            conditional.query_deltas[choice],
            answers - state.query_answers,
        )
        assert conditional.candidate_losses[choice] == pytest.approx(
            loss, abs=1e-12
        )
        assert conditional.gains[choice] == pytest.approx(
            state.loss - loss, abs=1e-12
        )


def test_hand_computed_conjunctive_delta_reaches_target():
    state = _binary_state()
    conditional = _conditional(state, coordinate=3)

    np.testing.assert_array_equal(
        conditional.query_deltas,
        np.asarray([[0, 0, 0], [0, 1, 1]], dtype=np.int8),
    )
    np.testing.assert_allclose(conditional.candidate_losses, [1.0, 0.0])
    np.testing.assert_allclose(conditional.gains, [0.0, 1.0])


def test_beta_zero_is_exact_uniform_and_conditional_arrays_are_read_only():
    conditional = _conditional(_binary_state(), coordinate=3, beta=0.0)

    np.testing.assert_array_equal(conditional.probabilities, [0.5, 0.5])
    np.testing.assert_array_equal(conditional.scaled_log_weights, [0.0, 0.0])
    assert conditional.expected_loss == conditional.reference_expected_loss
    for array in (
        conditional.query_deltas,
        conditional.candidate_losses,
        conditional.gains,
        conditional.scaled_log_weights,
        conditional.probabilities,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = 7


def test_inactive_attribute_stays_uniform_at_positive_temperature():
    schema = _binary_schema()
    queries = _binary_queries()[:1]
    target = np.asarray([1])
    state = initialize_persistent_heatbath_state(
        _binary_table((0, 0, 1, 0)), schema, queries, target
    )
    conditional = build_persistent_heatbath_conditional(
        state,
        schema,
        queries,
        target,
        row_index=0,
        attribute_index=1,
        inverse_temperature=100.0,
    )

    np.testing.assert_array_equal(conditional.candidate_losses, [0.0, 0.0])
    np.testing.assert_array_equal(conditional.probabilities, [0.5, 0.5])


def test_single_value_domain_has_self_loop_probability_one():
    probabilities, centered = heatbath_probabilities(
        np.asarray([3.0]), n_records=2, inverse_temperature=4.0
    )

    np.testing.assert_array_equal(probabilities, [1.0])
    np.testing.assert_array_equal(centered, [0.0])


def test_finite_temperature_preserves_and_can_sample_uphill_move():
    state = _binary_state()
    conditional = _conditional(state, coordinate=0, beta=1.3)
    uphill = int(np.argmax(conditional.candidate_losses))

    assert conditional.candidate_losses[uphill] > state.loss
    assert conditional.probabilities[uphill] > 0.0
    gumbels = np.zeros(2)
    gumbels[uphill] = 100.0
    choice = sample_heatbath_index(conditional, gumbels=gumbels)
    diagnostics = apply_heatbath_choice(state, conditional, choice)

    assert choice == uphill
    assert diagnostics["gain"] < 0.0
    assert state.loss > diagnostics["loss_before"]


def test_gumbel_sampling_is_reproducible_and_validated():
    conditional = _conditional(_binary_state(), coordinate=3)
    rng_a = np.random.default_rng(91)
    rng_b = np.random.default_rng(91)

    assert sample_heatbath_index(conditional, rng=rng_a) == (
        sample_heatbath_index(conditional, rng=rng_b)
    )
    expected = int(np.argmax(
        np.log(conditional.probabilities) + np.asarray([0.2, -0.3])
    ))
    assert sample_heatbath_index(
        conditional, gumbels=np.asarray([0.2, -0.3])
    ) == expected
    with pytest.raises(ValueError, match="gumbels"):
        sample_heatbath_index(conditional, gumbels=np.asarray([0.0]))
    with pytest.raises(ValueError, match="rng"):
        sample_heatbath_index(conditional)


def test_apply_rejects_stale_row_query_answers_and_loss():
    for mutation in ("row", "answers", "loss"):
        state = _binary_state()
        conditional = _conditional(state, coordinate=3)
        if mutation == "row":
            state.table.iat[1, 0] = 0
        elif mutation == "answers":
            state.query_answers = state.query_answers.copy()
            state.query_answers[0] = 0
        else:
            state.loss = 2.0
        with pytest.raises(ValueError, match="源状态"):
            apply_heatbath_choice(state, conditional, 1)


def test_step_uses_fixed_coordinate_and_external_gumbels_without_acceptance():
    state = _binary_state()
    rng = np.random.default_rng(12)
    diagnostics = persistent_heatbath_step(
        state,
        _binary_schema(),
        _binary_queries(),
        np.asarray([1, 1, 1]),
        inverse_temperature=0.7,
        rng=rng,
        coordinate_index=3,
        gumbels=np.asarray([-10.0, 10.0]),
    )

    assert diagnostics["coordinate_index"] == 3
    assert diagnostics["changed"] is True
    assert state.table.iloc[1].tolist() == [1, 1]
    assert state.query_answers.tolist() == [1, 1, 1]
    assert state.loss == 0.0


def test_initial_gain_rms_scale_matches_independent_enumeration():
    state = _binary_state()
    result = initial_gain_rms_scale(
        state,
        _binary_schema(),
        _binary_queries(),
        np.asarray([1, 1, 1]),
    )
    gains = []
    for coordinate in range(4):
        conditional = _conditional(state, coordinate, beta=0.0)
        gains.extend((conditional.gains / 2.0).tolist())
    nonzero = np.asarray([value for value in gains if value != 0.0])

    assert result["nonzero_gain_count"] == len(nonzero)
    assert result["candidate_state_evaluations"] == 8
    assert result["scale"] == pytest.approx(
        np.sqrt(np.mean(nonzero ** 2)), abs=1e-12
    )


@pytest.mark.parametrize("beta", ORACLE_BETAS)
def test_complete_oracle_transition_matrix_is_reversible_and_irreducible(beta):
    matrix, stationary, _, conditionals = _transition_matrix(beta)

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    assert np.all(np.diag(matrix) > 0.0)
    for source, source_bits in enumerate(ORACLE_STATES):
        for target_index, target_bits in enumerate(ORACLE_STATES):
            distance = sum(a != b for a, b in zip(source_bits, target_bits))
            if distance <= 1:
                assert matrix[source, target_index] > 0.0
            else:
                assert matrix[source, target_index] == 0.0
    flow = stationary[:, None] * matrix
    np.testing.assert_allclose(flow, flow.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(stationary @ matrix, stationary, atol=1e-12)

    reached = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for target_index in np.flatnonzero(matrix[source] > 0.0):
            target_index = int(target_index)
            if target_index not in reached:
                reached.add(target_index)
                frontier.append(target_index)
    assert len(reached) == 16
    assert len(conditionals) == 64


@pytest.mark.parametrize("beta", ORACLE_BETAS)
def test_complete_oracle_conditionals_and_incremental_identities(beta):
    _, stationary, _, conditionals = _transition_matrix(beta)
    schema = _binary_schema()
    queries = _binary_queries()
    target = np.asarray([1, 1, 1])

    for state_index, bits in enumerate(ORACLE_STATES):
        state = initialize_persistent_heatbath_state(
            _binary_table(bits), schema, queries, target
        )
        for coordinate in range(4):
            conditional = conditionals[(state_index, coordinate)]
            alternative_indices = []
            for value in conditional.values:
                alternative = list(bits)
                alternative[coordinate] = value
                alternative_indices.append(ORACLE_INDEX[tuple(alternative)])
            expected = stationary[alternative_indices]
            expected = expected / expected.sum()
            np.testing.assert_allclose(
                conditional.probabilities, expected, atol=1e-12, rtol=0
            )
            for choice, value in enumerate(conditional.values):
                candidate = _full_candidate(state.table, coordinate, value)
                answers = evaluate_table(candidate, queries)
                loss = compute_loss(target, answers)
                np.testing.assert_array_equal(
                    state.query_answers + conditional.query_deltas[choice],
                    answers,
                )
                assert conditional.candidate_losses[choice] == pytest.approx(
                    loss, abs=1e-12
                )
                assert conditional.gains[choice] == pytest.approx(
                    state.loss - loss, abs=1e-12
                )


def test_conditional_expected_loss_monotonicity_and_derivative_identity():
    for bits in ORACLE_STATES:
        state = _binary_state(bits)
        for coordinate in range(4):
            expected = [
                _conditional(state, coordinate, beta).expected_loss
                for beta in ORACLE_BETAS
            ]
            assert expected[0] + 1e-12 >= expected[1]
            assert expected[1] + 1e-12 >= expected[2]

            center = _conditional(state, coordinate, 0.7)
            epsilon = 1e-5
            upper = _conditional(state, coordinate, 0.7 + epsilon)
            lower = _conditional(state, coordinate, 0.7 - epsilon)
            numerical = (
                upper.expected_loss - lower.expected_loss
            ) / (2.0 * epsilon)
            mean = center.expected_loss
            variance = float(np.dot(
                center.probabilities,
                (center.candidate_losses - mean) ** 2,
            ))
            analytic = -variance / 2.0
            assert numerical == pytest.approx(analytic, abs=1e-9, rel=1e-7)


@pytest.mark.parametrize(
    "losses,n,beta,match",
    [
        ([0.0, 1.0], 0, 1.0, "n_records"),
        ([0.0, -1.0], 1, 1.0, "非负"),
        ([0.0, np.nan], 1, 1.0, "有限"),
        ([0.0, 1.0], 1, -1.0, "inverse_temperature"),
        ([0.0, 1000.0], 1, 1.0, "下溢"),
    ],
)
def test_probability_validation_and_explicit_underflow_failure(
    losses, n, beta, match
):
    with pytest.raises(ValueError, match=match):
        heatbath_probabilities(np.asarray(losses), n, beta)


def test_invalid_inputs_fail_before_coordinate_rng_consumption():
    state = _binary_state()
    rng = np.random.default_rng(77)
    before = repr(rng.bit_generator.state)

    with pytest.raises(ValueError, match="inverse_temperature"):
        persistent_heatbath_step(
            state,
            _binary_schema(),
            _binary_queries(),
            np.asarray([1, 1, 1]),
            inverse_temperature=np.nan,
            rng=rng,
        )
    assert repr(rng.bit_generator.state) == before


def test_initialization_and_verification_reject_illegal_or_corrupt_state():
    illegal = _binary_table((0, 0, 1, 0))
    illegal.iat[0, 0] = 3
    with pytest.raises(ValueError, match="非法值"):
        initialize_persistent_heatbath_state(
            illegal,
            _binary_schema(),
            _binary_queries(),
            np.asarray([1, 1, 1]),
        )
    with pytest.raises(ValueError, match="至少包含一行"):
        initialize_persistent_heatbath_state(
            pd.DataFrame(columns=["a", "b"]),
            _binary_schema(),
            _binary_queries(),
            np.asarray([1, 1, 1]),
        )

    state = _binary_state()
    state.table.iat[0, 0] = 1
    with pytest.raises(RuntimeError, match="整表复核失败"):
        verify_persistent_heatbath_state(
            state,
            _binary_schema(),
            _binary_queries(),
            np.asarray([1, 1, 1]),
        )


def test_actual_configuration_short_cpu_smoke_keeps_exact_incremental_state():
    schema = load_schema("configs/test_300x10/schema.yaml")
    queries = load_queries("configs/test_300x10/measured_50query.json")
    marginals = load_marginals("configs/test_300x10/init_marginals.json")
    target = np.asarray([query["result"] for query in queries])
    table = init_synthetic_table(
        300, schema, np.random.default_rng(5), marginals=marginals
    )
    state = initialize_persistent_heatbath_state(
        table, schema, queries, target
    )
    rng = np.random.default_rng(105)

    for _ in range(20):
        persistent_heatbath_step(
            state,
            schema,
            queries,
            target,
            inverse_temperature=0.5,
            rng=rng,
        )
        audit = verify_persistent_heatbath_state(
            state, schema, queries, target
        )
        assert audit["query_answer_max_abs_error"] == 0
        assert audit["loss_abs_error"] <= 1e-12


def test_copy_does_not_share_mutable_state():
    state = _binary_state()
    copied = state.copy()
    copied.table.iat[0, 0] = 1
    copied.query_answers[0] = 2
    copied.loss = 9.0

    assert state.table.iat[0, 0] == 0
    assert state.query_answers.tolist() == [1, 0, 0]
    assert state.loss == 1.0


def test_conditional_type_guard_is_explicit():
    with pytest.raises(ValueError, match="conditional"):
        sample_heatbath_index("not-a-conditional", rng=np.random.default_rng(0))
    assert HeatbathConditional is not PersistentHeatbathState
