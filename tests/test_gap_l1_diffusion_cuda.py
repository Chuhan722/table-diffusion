"""缺口核 CUDA（显卡）双精度后端的人工一致性测试。"""

import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from table_diffevo import gap_l1_diffusion as gap
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema
from table_diffevo.vectorized_eval import evaluate_conditions_vectorized


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA（显卡运行时）不可用"
)


@pytest.fixture(autouse=True)
def _deterministic_cuda():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _binary_schema(*names: str) -> Schema:
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in names
    ])


def _equals(attribute: str, value=1) -> dict:
    return {"attribute": attribute, "operator": "==", "value": value}


def _case():
    schema = _binary_schema("a", "b", "c")
    queries = [
        {"conditions": [_equals("a")]},
        {"conditions": [_equals("b")]},
        {"conditions": [_equals("a"), _equals("b")]},
        {"conditions": [_equals("b"), _equals("c")]},
        {"conditions": [_equals("a"), _equals("b"), _equals("c")]},
    ]
    current = pd.DataFrame({
        "a": [0, 0, 1, 0, 1, 0],
        "b": [0, 1, 0, 0, 1, 1],
        "c": [0, 0, 0, 1, 0, 1],
    })
    donors = pd.DataFrame({
        "a": [1, 1, 0, 1, 1, 1],
        "b": [1, 0, 1, 1, 0, 0],
        "c": [1, 1, 1, 0, 1, 0],
    })
    counts = evaluate_table(current, queries)
    targets = np.array([3.5, 2.5, 2.0, 1.5, 1.0])
    participate = np.array([True, True, False, True, True, True])
    initial_mask = np.zeros((len(current), 3), dtype=bool)
    initial_mask[0, 1] = True
    initial_mask[3, 2] = True
    return (
        schema,
        queries,
        current,
        donors,
        counts,
        targets,
        participate,
        initial_mask,
    )


def test_cuda_condition_and_isolated_scores_match_cpu():
    (
        schema,
        queries,
        current,
        donors,
        counts,
        targets,
        participate,
        initial_mask,
    ) = _case()
    kwargs = dict(
        participate=participate,
        mask=initial_mask,
        row_index=0,
        attribute_index=0,
        reference_scale=0.02,
    )
    cpu = gap.evaluate_gap_l1_condition(
        current, donors, schema, queries, targets, counts, **kwargs
    )
    cuda = gap.evaluate_gap_l1_condition(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        device="cuda",
        **kwargs,
    )
    for key in (
        "e0",
        "e1",
        "score",
        "normalized_score",
        "raw_logit",
        "logit",
        "probability",
    ):
        assert abs(cpu[key] - cuda[key]) <= 1e-12
    assert cpu["clipped"] == cuda["clipped"]
    assert cuda["backend"] == "torch_cuda_float64"

    cpu_isolated = gap.isolated_gap_l1_scores(
        current, donors, schema, queries, targets, counts
    )
    cuda_isolated = gap.isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        device="cuda",
    )
    np.testing.assert_array_equal(
        cpu_isolated["coordinates"], cuda_isolated["coordinates"]
    )
    np.testing.assert_allclose(
        cpu_isolated["scores"], cuda_isolated["scores"], atol=1e-12, rtol=0
    )
    assert abs(
        cpu_isolated["current_error"] - cuda_isolated["current_error"]
    ) <= 1e-12


def test_cuda_exact_rational_zero_is_excluded_from_scale():
    schema = _binary_schema("a")
    queries = [{"conditions": [_equals("a")]} for _ in range(3)]
    current = pd.DataFrame({"a": [1] * 10 + [0] * 15})
    donors = current.copy()
    donors.at[10, "a"] = 1
    result = gap.isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        np.array([8.0, 12.0, 24.0]),
        np.array([10, 10, 10]),
        exact_target_numerators=np.array([8, 12, 24]) * len(current),
        exact_target_denominator=len(current),
        device="cuda",
    )
    scale, diagnostics = gap.stable_nonzero_rms(result["scores"])
    np.testing.assert_array_equal(result["coordinates"], [[10, 0]])
    assert result["scores"][0] == 0.0
    assert scale == 0.0
    assert diagnostics["zero_count"] == 1


def test_cuda_random_scan_matches_cpu_at_every_microstep(monkeypatch):
    (
        schema,
        queries,
        current,
        donors,
        counts,
        targets,
        participate,
        initial_mask,
    ) = _case()
    original_update_trace = gap._update_trace
    captured = []

    def capture(digest, **values):
        captured.append(dict(values))
        return original_update_trace(digest, **values)

    monkeypatch.setattr(gap, "_update_trace", capture)
    kwargs = dict(
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        n_sweeps=8,
    )
    cpu_rng = np.random.default_rng(20260826)
    cuda_rng = np.random.default_rng(20260826)
    cpu_result = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        rng=cpu_rng,
        **kwargs,
    )
    cpu_trace = captured.copy()
    captured.clear()
    cuda_result = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        rng=cuda_rng,
        device="cuda",
        **kwargs,
    )
    cuda_trace = captured.copy()

    assert len(cpu_trace) == len(cuda_trace) == (
        8 * cpu_result[2]["active_switches_k"]
    )
    for cpu_step, cuda_step in zip(cpu_trace, cuda_trace):
        for key in (
            "step",
            "row_index",
            "attribute_index",
            "before",
            "after",
            "clipped",
        ):
            assert cpu_step[key] == cuda_step[key]
        assert cpu_step["random_roll"] == cuda_step["random_roll"]
        for key in (
            "e0",
            "e1",
            "score",
            "normalized_score",
            "raw_logit",
            "probability",
        ):
            assert abs(cpu_step[key] - cuda_step[key]) <= 1e-12
    pd.testing.assert_frame_equal(cpu_result[0], cuda_result[0])
    np.testing.assert_array_equal(cpu_result[1], cuda_result[1])
    assert (
        cpu_result[2]["final_query_counts"]
        == cuda_result[2]["final_query_counts"]
    )
    assert cpu_rng.bit_generator.state == cuda_rng.bit_generator.state
    assert cuda_result[2]["backend"] == "torch_cuda_float64"


def test_cuda_k_zero_consumes_no_rng():
    schema = _binary_schema("a")
    queries = [{"conditions": [_equals("a")]}]
    current = pd.DataFrame({"a": [0, 1]})
    rng = np.random.default_rng(77)
    before = copy.deepcopy(rng.bit_generator.state)
    table, mask, diagnostics = gap.evolve_step_gap_l1_global(
        current,
        current.copy(),
        schema,
        queries,
        np.array([1]),
        np.array([1]),
        participate=np.array([True, True]),
        initial_mask=np.zeros((2, 1), dtype=bool),
        reference_scale=0.1,
        rng=rng,
        device="cuda",
    )
    pd.testing.assert_frame_equal(table, current)
    assert not np.any(mask)
    assert diagnostics["gibbs_microsteps"] == 0
    assert rng.bit_generator.state == before


def test_independent_condition_evaluator_matches_numpy():
    schema = Schema([
        AttributeBlock(
            name="category",
            type="categorical",
            description="category",
            values=["x", "y"],
        ),
        AttributeBlock(
            name="value",
            type="numeric",
            description="value",
            range=[0, 10],
        ),
    ])
    frame = pd.DataFrame({
        "category": ["x", "y", "x", "y"],
        "value": [0, 3, 7, 10],
    })
    conditions = [
        _equals("category", "x"),
        {"attribute": "category", "operator": "==", "value": "missing"},
        {"attribute": "value", "operator": ">=", "value": 7},
        {
            "attribute": "value",
            "operator": "between",
            "lower": 2,
            "upper": 8,
        },
    ]
    expected = evaluate_conditions_vectorized(
        frame, conditions, schema, device="numpy"
    )
    actual = evaluate_conditions_vectorized(
        frame,
        conditions,
        schema,
        device="cuda",
        float64=True,
    )
    np.testing.assert_array_equal(expected, actual)
