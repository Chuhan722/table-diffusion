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


@pytest.mark.parametrize(
    "weighting_kwargs",
    [
        {},
        {
            "weighting": gap.GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
            "max_weight_ratio": 8.0,
        },
        {
            "weighting": gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
        },
        {
            "weighting": gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX,
        },
    ],
    ids=("legacy", "bounded-r8", "sqrt-target", "dual-ar-max"),
)
def test_cuda_condition_and_isolated_scores_match_cpu(weighting_kwargs):
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
        **weighting_kwargs,
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
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        **weighting_kwargs,
    )
    cuda_isolated = gap.isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        device="cuda",
        **weighting_kwargs,
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


@pytest.mark.parametrize(
    "targets",
    [np.array([0.0, 2.0]), np.array([0.0, 0.0])],
    ids=("mixed-zero-positive", "all-zero"),
)
def test_dual_cuda_zero_target_boundaries_match_cpu(targets):
    schema = _binary_schema("a")
    queries = [
        {"conditions": [_equals("a", 1)]},
        {"conditions": [_equals("a", 0)]},
    ]
    current = pd.DataFrame({"a": [0, 1, 0]})
    donors = pd.DataFrame({"a": [1, 0, 1]})
    counts = evaluate_table(current, queries)
    kwargs = {
        "weighting": gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX,
    }

    cpu = gap.isolated_gap_l1_scores(
        current, donors, schema, queries, targets, counts, **kwargs
    )
    cuda = gap.isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        device="cuda",
        **kwargs,
    )

    np.testing.assert_array_equal(cpu["coordinates"], cuda["coordinates"])
    np.testing.assert_allclose(cpu["scores"], cuda["scores"], rtol=0, atol=1e-12)
    assert cpu["current_error"] == pytest.approx(
        gap.normalized_gap_l1_error(
            counts,
            targets,
            n_records=len(current),
            weighting=gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX,
        )
    )
    assert cuda["current_error"] == pytest.approx(
        cpu["current_error"], abs=1e-12
    )


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


def test_cuda_bounded_exact_rational_zero_is_excluded_from_scale():
    schema = _binary_schema("a")
    queries = [{"conditions": [_equals("a")]} for _ in range(3)]
    current = pd.DataFrame({"a": [1] * 10 + [0] * 60})
    donors = current.copy()
    donors.at[10, "a"] = 1
    targets = np.array([2.0, 14.0, 14.0])
    result = gap.isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        targets,
        np.array([10, 10, 10]),
        exact_target_numerators=targets.astype(np.int64),
        exact_target_denominator=1,
        weighting=gap.GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
        max_weight_ratio=8.0,
        device="cuda",
    )
    scale, diagnostics = gap.stable_nonzero_rms(result["scores"])

    np.testing.assert_array_equal(result["coordinates"], [[10, 0]])
    assert result["scores"][0] == 0.0
    assert scale == 0.0
    assert diagnostics["zero_count"] == 1


@pytest.mark.parametrize(
    "weighting_kwargs",
    [
        {},
        {
            "weighting": gap.GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
            "max_weight_ratio": 8.0,
        },
        {
            "weighting": gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
        },
        {
            "weighting": gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX,
        },
    ],
    ids=("legacy", "bounded-r8", "sqrt-target", "dual-ar-max"),
)
def test_cuda_random_scan_matches_cpu_at_every_microstep(
    monkeypatch, weighting_kwargs
):
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
        **weighting_kwargs,
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


def test_cuda_scan_keeps_coordinate_tape_on_host(monkeypatch):
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
    original_evolve_cuda = gap._evolve_step_gap_l1_global_cuda
    original_prepare = gap._prepare_cuda_plan
    original_as_tensor = torch.as_tensor
    state = {"inside_scan": False, "plan_ready": False}
    coordinate_transfers = []

    def observe_evolve_cuda(*args, **kwargs):
        state["inside_scan"] = True
        try:
            return original_evolve_cuda(*args, **kwargs)
        finally:
            state["inside_scan"] = False
            state["plan_ready"] = False

    def observe_prepare(*args, **kwargs):
        plan = original_prepare(*args, **kwargs)
        if state["inside_scan"]:
            state["plan_ready"] = True
        return plan

    def observe_as_tensor(data, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if (
            state["plan_ready"]
            and isinstance(data, np.ndarray)
            and data.ndim == 2
            and data.shape[1] == 2
            and dtype == torch.long
        ):
            coordinate_transfers.append(data.shape)
        return original_as_tensor(data, *args, **kwargs)

    monkeypatch.setattr(
        gap, "_evolve_step_gap_l1_global_cuda", observe_evolve_cuda
    )
    monkeypatch.setattr(gap, "_prepare_cuda_plan", observe_prepare)
    monkeypatch.setattr(torch, "as_tensor", observe_as_tensor)
    gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        rng=np.random.default_rng(20260826),
        n_sweeps=8,
        device="cuda",
    )

    assert coordinate_transfers == []


def test_cuda_condition_pair_reuses_old_error_term_reduction(monkeypatch):
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
    plan = gap._prepare_cuda_plan(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate,
        initial_mask,
        floor=8.0,
        compiled_workload=None,
    )
    original_sum = torch.Tensor.sum
    reduction_widths = []

    def observe_sum(tensor, *args, **kwargs):
        if kwargs.get("dtype") == torch.float64:
            reduction_widths.append(tensor.numel())
        return original_sum(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", observe_sum)
    *_, old_term_sum = gap._condition_pair_cuda(plan, 0, 0)

    assert reduction_widths == [3, 3, 3]
    assert old_term_sum is not None


def test_cuda_triton_condition_matches_eager():
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
    plan = gap._prepare_cuda_plan(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate,
        initial_mask,
        floor=8.0,
        compiled_workload=None,
    )
    from table_diffevo import _gap_l1_triton as triton_gap

    width = max(
        int(indices.numel()) for indices in plan.query_indices_by_attribute
    )
    failures0 = torch.empty(width, dtype=torch.int32, device=plan.device)
    failures1 = torch.empty_like(failures0)
    indicators0 = torch.empty(width, dtype=torch.bool, device=plan.device)
    indicators1 = torch.empty_like(indicators0)
    counts0 = torch.empty(width, dtype=torch.int64, device=plan.device)
    counts1 = torch.empty_like(counts0)
    terms0 = torch.empty(width, dtype=torch.float64, device=plan.device)
    terms1 = torch.empty_like(terms0)
    old_terms = torch.empty_like(terms0)
    original_state = (
        plan.failure_counts.clone(),
        plan.row_indicators.clone(),
        plan.plan_counts.clone(),
        plan.error_terms.clone(),
        plan.mask.clone(),
    )

    for row_raw, attribute_raw in plan.active_coordinates:
        row = int(row_raw)
        attribute = int(attribute_raw)
        local_row = int(plan.row_lookup[row])
        query_indices = plan.query_indices_by_attribute[attribute]
        query_width = int(query_indices.numel())
        eager = gap._condition_pair_cuda(
            plan,
            row,
            attribute,
            local_row=local_row,
        )
        triton_gap.launch_condition(
            plan.current_attribute_failures[attribute][local_row],
            plan.donor_attribute_failures[attribute][local_row],
            plan.failure_counts[local_row],
            plan.row_indicators[local_row],
            plan.plan_counts,
            plan.target,
            plan.denominators,
            plan.error_terms,
            query_indices,
            plan.mask[row],
            attribute,
            failures0,
            failures1,
            indicators0,
            indicators1,
            counts0,
            counts1,
            terms0,
            terms1,
            old_terms,
            width=query_width,
        )
        expected_counts0 = (
            plan.plan_counts[query_indices]
            + eager[4].to(torch.int64)
            - plan.row_indicators[local_row, query_indices].to(torch.int64)
        )
        expected_counts1 = (
            plan.plan_counts[query_indices]
            + eager[5].to(torch.int64)
            - plan.row_indicators[local_row, query_indices].to(torch.int64)
        )
        expected_terms0 = (
            torch.abs(
                plan.target[query_indices]
                - expected_counts0.to(torch.float64)
            )
            / plan.denominators[query_indices]
        )
        expected_terms1 = (
            torch.abs(
                plan.target[query_indices]
                - expected_counts1.to(torch.float64)
            )
            / plan.denominators[query_indices]
        )
        old_term_sum = old_terms[:query_width].sum(dtype=torch.float64)
        shadow_e0 = (
            plan.error_sum
            - old_term_sum
            + terms0[:query_width].sum(dtype=torch.float64)
        ) / plan.compiled.n_queries
        shadow_e1 = (
            plan.error_sum
            - old_term_sum
            + terms1[:query_width].sum(dtype=torch.float64)
        ) / plan.compiled.n_queries

        assert torch.equal(failures0[:query_width], eager[2])
        assert torch.equal(failures1[:query_width], eager[3])
        assert torch.equal(indicators0[:query_width], eager[4])
        assert torch.equal(indicators1[:query_width], eager[5])
        assert torch.equal(counts0[:query_width], expected_counts0)
        assert torch.equal(counts1[:query_width], expected_counts1)
        assert torch.equal(terms0[:query_width], expected_terms0)
        assert torch.equal(terms1[:query_width], expected_terms1)
        assert torch.equal(old_terms[:query_width], plan.error_terms[query_indices])
        assert torch.equal(old_term_sum, eager[6])
        assert torch.equal(shadow_e0, eager[0])
        assert torch.equal(shadow_e1, eager[1])

    for actual, expected in zip(
        (
            plan.failure_counts,
            plan.row_indicators,
            plan.plan_counts,
            plan.error_terms,
            plan.mask,
        ),
        original_state,
    ):
        assert torch.equal(actual, expected)


def test_cuda_triton_scalar_fusion_matches_eager(monkeypatch):
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
    from table_diffevo import _gap_l1_triton as triton_gap

    original_combine = triton_gap.launch_combine
    original_prepare = triton_gap.launch_prepare
    original_commit = triton_gap.launch_commit
    state = {}
    comparisons = []

    def observe_combine(*args, **kwargs):
        error_sum = args[0]
        old_term_sum = args[1]
        expected_e0 = (
            error_sum - old_term_sum + args[2]
        ) / kwargs["n_queries"]
        expected_e1 = (
            error_sum - old_term_sum + args[3]
        ) / kwargs["n_queries"]
        step = args[-1]
        result = original_combine(*args, **kwargs)
        state["error_sum"] = error_sum
        state["old_term_sum"] = old_term_sum
        comparisons.append((
            torch.equal(args[4][step], expected_e0),
            torch.equal(args[5][step], expected_e1),
        ))
        return result

    def observe_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        state["candidate_term_sum"] = args[-6]
        state["candidate_terms"] = args[-4]
        state["width"] = kwargs["width"]
        historical_candidate_sum = state["candidate_terms"][
            :state["width"]
        ].sum(dtype=state["error_sum"].dtype)
        state["candidate_error_sum"] = (
            gap._exact_cuda_candidate_error_sum(
                state["error_sum"],
                state["old_term_sum"],
                state["candidate_term_sum"],
            )
        )
        state["unchanged_error_sum"] = state["error_sum"].clone()
        comparisons[-1] += (
            torch.equal(
                state["candidate_term_sum"],
                historical_candidate_sum,
            ),
        )
        return result

    def observe_commit(*args, **kwargs):
        assert args[11] is state["candidate_term_sum"]
        step = args[-1]
        changed = bool(
            (args[16][step] != args[15][step]).item()
        )
        expected_error_sum = (
            state["candidate_error_sum"]
            if changed
            else state["unchanged_error_sum"]
        )
        result = original_commit(*args, **kwargs)
        comparisons[-1] += (
            torch.equal(args[12], expected_error_sum),
        )
        return result

    monkeypatch.setattr(triton_gap, "launch_combine", observe_combine)
    monkeypatch.setattr(triton_gap, "launch_prepare", observe_prepare)
    monkeypatch.setattr(triton_gap, "launch_commit", observe_commit)
    _, _, diagnostics = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        rng=np.random.default_rng(20260827),
        n_sweeps=8,
        device="cuda",
    )

    assert len(comparisons) == diagnostics["gibbs_microsteps"]
    assert set(comparisons) == {(True, True, True, True)}


def test_cuda_scan_fuses_scalars_after_three_condition_reductions(
    monkeypatch,
):
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
    original_term_sums = gap._cuda_condition_term_sums
    original_sum = torch.Tensor.sum
    state = {"inside_reduction": False, "reduction_count": 0}
    reductions_per_microstep = []

    def observe_term_sums(*args, **kwargs):
        assert state["inside_reduction"] is False
        state["inside_reduction"] = True
        state["reduction_count"] = 0
        try:
            return original_term_sums(*args, **kwargs)
        finally:
            reductions_per_microstep.append(state["reduction_count"])
            state["inside_reduction"] = False

    def observe_sum(tensor, *args, **kwargs):
        if state["inside_reduction"] and kwargs.get("dtype") == torch.float64:
            state["reduction_count"] += 1
        return original_sum(tensor, *args, **kwargs)

    def forbidden_eager_scalar_path(*_args, **_kwargs):
        raise AssertionError("非空查询扫描不得回到eager标量结合")

    monkeypatch.setattr(gap, "_cuda_condition_term_sums", observe_term_sums)
    monkeypatch.setattr(
        gap,
        "_exact_cuda_condition_error_pair",
        forbidden_eager_scalar_path,
    )
    monkeypatch.setattr(
        gap,
        "_exact_cuda_candidate_error_sum",
        forbidden_eager_scalar_path,
    )
    monkeypatch.setattr(torch.Tensor, "sum", observe_sum)
    _, _, diagnostics = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        rng=np.random.default_rng(20260826),
        n_sweeps=8,
        device="cuda",
    )

    assert state["inside_reduction"] is False
    assert len(reductions_per_microstep) == diagnostics["gibbs_microsteps"]
    assert set(reductions_per_microstep) == {3}


def test_cuda_scan_launches_four_triton_kernels_per_microstep(monkeypatch):
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
    from table_diffevo import _gap_l1_triton as triton_gap

    original_condition = triton_gap.launch_condition
    original_combine = triton_gap.launch_combine
    original_prepare = triton_gap.launch_prepare
    original_commit = triton_gap.launch_commit
    calls = {"condition": 0, "combine": 0, "prepare": 0, "commit": 0}

    def observe_condition(*args, **kwargs):
        calls["condition"] += 1
        return original_condition(*args, **kwargs)

    def observe_combine(*args, **kwargs):
        calls["combine"] += 1
        return original_combine(*args, **kwargs)

    def observe_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def observe_commit(*args, **kwargs):
        calls["commit"] += 1
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(triton_gap, "launch_condition", observe_condition)
    monkeypatch.setattr(triton_gap, "launch_combine", observe_combine)
    monkeypatch.setattr(triton_gap, "launch_prepare", observe_prepare)
    monkeypatch.setattr(triton_gap, "launch_commit", observe_commit)
    _, _, diagnostics = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        rng=np.random.default_rng(20260827),
        n_sweeps=8,
        device="cuda",
    )

    assert calls == {
        "condition": diagnostics["gibbs_microsteps"],
        "combine": diagnostics["gibbs_microsteps"],
        "prepare": diagnostics["gibbs_microsteps"],
        "commit": diagnostics["gibbs_microsteps"],
    }


def test_cuda_triton_microstep_trajectory_is_frozen():
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
    rng = np.random.default_rng(20260827)
    _, mask, diagnostics = gap.evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        rng=rng,
        n_sweeps=8,
        device="cuda",
    )

    assert diagnostics["microstep_trace_sha256"] == (
        "5fa1c82c7d8e374ac06636c68832e2580871497a316f22958abd9ee67955b278"
    )
    assert diagnostics["final_query_counts"] == [4, 2, 1, 2, 1]
    np.testing.assert_array_equal(
        mask,
        np.asarray([
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=bool),
    )


def test_cuda_triton_normalizes_column_major_initial_mask():
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
    column_major_mask = np.asfortranarray(initial_mask)
    assert column_major_mask.flags.f_contiguous
    assert not column_major_mask.flags.c_contiguous

    results = [
        gap.evolve_step_gap_l1_global(
            current,
            donors,
            schema,
            queries,
            targets,
            counts,
            participate=participate,
            initial_mask=mask,
            reference_scale=0.02,
            rng=np.random.default_rng(20260826),
            n_sweeps=8,
            device="cuda",
        )
        for mask in (initial_mask, column_major_mask)
    ]

    pd.testing.assert_frame_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][2]["microstep_trace_sha256"] == (
        results[1][2]["microstep_trace_sha256"]
    )
    assert results[0][2]["final_query_counts"] == (
        results[1][2]["final_query_counts"]
    )


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


@pytest.mark.parametrize(
    "weighting_kwargs",
    [
        {},
        {
            "weighting": gap.GAP_L1_WEIGHTING_BOUNDED_RELATIVE,
            "max_weight_ratio": 8.0,
        },
        {
            "weighting": gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE,
        },
        {
            "weighting": gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX,
        },
    ],
    ids=("legacy", "bounded-r8", "sqrt-target", "dual-ar-max"),
)
def test_cuda_batched_different_addresses_match_single_results(
    weighting_kwargs,
):
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
    donor_tables = (
        donors,
        current.iloc[[1, 2, 3, 4, 5, 0]].reset_index(drop=True),
        current.copy(deep=True),
    )
    participates = (
        participate,
        np.array([True, False, True, True, False, True]),
        np.ones(len(current), dtype=bool),
    )
    initial_masks = [initial_mask]
    for index in range(1, len(donor_tables)):
        active = participates[index][:, None] & (
            current.to_numpy() != donor_tables[index].to_numpy()
        )
        initial_masks.append(active & (
            np.random.default_rng(700 + index).random(active.shape) < 0.5
        ))
    seeds = (20260841, 20260842, 20260843)
    reference_rngs = [np.random.default_rng(seed) for seed in seeds]
    batch_rngs = [np.random.default_rng(seed) for seed in seeds]
    compiled = gap.compile_gap_l1_workload(schema, queries)
    references = [
        gap.evolve_step_gap_l1_global(
            current,
            donor_tables[index],
            schema,
            queries,
            targets,
            counts,
            participate=participates[index],
            initial_mask=initial_masks[index],
            reference_scale=0.02,
            rng=reference_rngs[index],
            n_sweeps=8,
            compiled_workload=compiled,
            device="cuda",
            **weighting_kwargs,
        )
        for index in range(len(donor_tables))
    ]
    results = gap.evolve_step_gap_l1_global_batched(
        current,
        donor_tables,
        schema,
        queries,
        targets,
        counts,
        participates=participates,
        initial_masks=initial_masks,
        reference_scale=0.02,
        rngs=batch_rngs,
        n_sweeps=8,
        compiled_workload=compiled,
        device="cuda",
        **weighting_kwargs,
    )

    assert len(results) == len(references) == 3
    assert results[2][2]["active_switches_k"] == 0
    for index, (result, reference) in enumerate(zip(results, references)):
        pd.testing.assert_frame_equal(result[0], reference[0])
        np.testing.assert_array_equal(result[1], reference[1])
        assert result[2]["final_query_counts"] == (
            reference[2]["final_query_counts"]
        )
        if weighting_kwargs.get("weighting") == (
            gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_MAX
        ):
            # 哈希包含每个微步的 E0/E1、score、概率、随机数与开关。
            assert result[2]["microstep_trace_sha256"] == (
                reference[2]["microstep_trace_sha256"]
            )
        assert result[2]["no_gate"] is True
        assert result[2]["backend"] == "torch_cuda_float64_batched"
        assert result[2]["gibbs_microsteps"] == (
            8 * result[2]["active_switches_k"]
        )
        batch = result[2]["batch_execution"]
        assert batch["format"] == gap.BATCH_EXECUTION_FORMAT
        assert batch["batch_size"] == 3
        assert batch["batch_index"] == index
        assert batch["strict_internal_order_preserved"] is True
        assert batch["timing_scope"] == "shared_batch_not_additive"
        assert (
            batch_rngs[index].bit_generator.state
            == reference_rngs[index].bit_generator.state
        )


def test_cuda_batched_rejects_invalid_batch_before_preparation():
    with pytest.raises(ValueError, match="同长度非空序列"):
        gap.evolve_step_gap_l1_global_batched(
            None,
            (),
            None,
            [],
            None,
            None,
            participates=(),
            initial_masks=(),
            reference_scale=1.0,
            rngs=(),
        )
    with pytest.raises(ValueError, match="np.random.Generator"):
        gap.evolve_step_gap_l1_global_batched(
            None,
            (None,),
            None,
            [],
            None,
            None,
            participates=(None,),
            initial_masks=(None,),
            reference_scale=1.0,
            rngs=(7,),
        )
    with pytest.raises(ValueError, match="只支持 'cuda'"):
        gap.evolve_step_gap_l1_global_batched(
            None,
            (),
            None,
            [],
            None,
            None,
            participates=(),
            initial_masks=(),
            reference_scale=1.0,
            rngs=(),
            device="numpy",
        )


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
