"""GPU 批量核测试，跨后端全指标对拍与逐位可复现，无卡环境自动跳过。"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("无可用 GPU", allow_module_level=True)
    _ = cp.zeros(2).sum()
except Exception:
    pytest.skip("GPU 运行时不可用", allow_module_level=True)

from test_batchkernel import _scene  # noqa: E402

from resevo.batchkernel import build_batch_kernel, evolve_batch, make_batch_provider  # noqa: E402
from resevo.batchmenu import CodeBook, build_query_structure, generate_batch_menu  # noqa: E402
from resevo.dataset import query_field_sets, target_from_specs  # noqa: E402
from resevo.editspace import EditBudget  # noqa: E402
from resevo.gpukernel import GpuBatchContext  # noqa: E402
from resevo.grouping import group_state_ids  # noqa: E402


def _setup(seed: int, num_rows: int = 36):
    registry, ids, weights, rng = _scene(seed, num_rows)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=3, donor_copies=3, joint_edits=2, explore_edits=2),
        [(0, 1), (2, 3)],
    )
    target = target_from_specs(registry.specs)
    workload = registry.build_workload(target, weights)
    return registry, workload, grouped, menu, qs, codes, target, weights


@pytest.mark.parametrize("seed", range(6))
def test_gpu_matches_cpu(seed):
    """同输入下 GPU 与 CPU 批量核全部指标与概率在容差内一致。"""
    _, workload, grouped, menu, qs, codes, target, weights = _setup(seed)
    ref = build_batch_kernel(workload, grouped, menu, qs, codes)
    ctx = GpuBatchContext(qs, target, weights)
    got = ctx.build(grouped, menu, codes)
    assert got.status == ref.status
    np.testing.assert_array_equal(got.offsets, ref.offsets)
    np.testing.assert_allclose(got.old_loss, ref.old_loss, rtol=1e-12)
    # beta 根经布伦特法对跨后端 1e-16 级评估差有放大，容差放到 1e-7
    np.testing.assert_allclose(got.beta, ref.beta, rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose(got.direction_gain, ref.direction_gain, rtol=1e-9)
    np.testing.assert_allclose(got.interaction, ref.interaction, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.step, ref.step, rtol=1e-7)
    np.testing.assert_allclose(got.expected_loss, ref.expected_loss, rtol=1e-7)
    np.testing.assert_allclose(got.max_gain_sum, ref.max_gain_sum, rtol=1e-9)
    np.testing.assert_allclose(
        got.probabilities, ref.probabilities, rtol=1e-7, atol=1e-11
    )


@pytest.mark.parametrize("hint", [None, 7.3])
def test_gpu_beta_hint_consistency(hint):
    """热启动括根在 GPU 后端同样只影响路径不影响根。"""
    _, workload, grouped, menu, qs, codes, target, weights = _setup(2)
    ctx = GpuBatchContext(qs, target, weights)
    base = ctx.build(grouped, menu, codes)
    seed_hint = base.beta if hint is None else base.beta * hint
    again = ctx.build(grouped, menu, codes, beta_hint=seed_hint)
    np.testing.assert_allclose(again.beta, base.beta, rtol=1e-7)


def test_gpu_bitwise_deterministic():
    """同输入同卡两遍构核，概率向量与标量逐位相同。"""
    _, workload, grouped, menu, qs, codes, target, weights = _setup(4)
    ctx = GpuBatchContext(qs, target, weights)
    a = ctx.build(grouped, menu, codes)
    b = ctx.build(grouped, menu, codes)
    assert a.beta == b.beta
    assert a.direction_gain == b.direction_gain
    assert a.interaction == b.interaction
    assert a.step == b.step
    np.testing.assert_array_equal(a.probabilities, b.probabilities)


def test_gpu_frozen_on_zero_residual():
    """目标取自当前表的真实答案时残差为零，GPU 后端同样判冻结。"""
    registry, ids, weights, rng = _scene(0, num_rows=30)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    menu = generate_batch_menu(
        codes[grouped.unique_ids], grouped.counts, codebook.domain_sizes,
        rng, EditBudget(max_edit_fields=2, donor_copies=2, joint_edits=0, explore_edits=0),
    )
    probe = registry.build_workload(np.zeros(len(weights)), weights)
    target = probe.features[ids].sum(axis=0)
    ctx = GpuBatchContext(qs, target, weights)
    got = ctx.build(grouped, menu, codes)
    ref = build_batch_kernel(
        registry.build_workload(target, weights), grouped, menu, qs, codes
    )
    assert ref.status == "no_positive_direction"
    assert got.status == "no_positive_direction"
    np.testing.assert_array_equal(got.probabilities, ref.probabilities)


def test_evolve_batch_gpu_descends():
    """端到端 GPU 后端把小场景损失打下去，首轮损失与 CPU 相同。"""
    registry, ids, weights, _ = _scene(1, num_rows=30)
    fs = [f for f in query_field_sets(registry.specs, registry.schema) if len(f) >= 2]
    target = target_from_specs(registry.specs)
    provider = make_batch_provider(
        registry, target, weights, np.random.default_rng(5), joint_field_sets=fs
    )
    out = evolve_batch(
        ids, 60, np.random.default_rng(9), provider, registry,
        max_frozen_retries=5, backend="gpu",
    )
    losses = [r.old_loss for r in out.records]
    assert losses[0] > 0
    assert min(losses) < 0.2 * losses[0]

    registry2, ids2, weights2, _ = _scene(1, num_rows=30)
    fs2 = [f for f in query_field_sets(registry2.specs, registry2.schema) if len(f) >= 2]
    provider2 = make_batch_provider(
        registry2, target_from_specs(registry2.specs), weights2,
        np.random.default_rng(5), joint_field_sets=fs2,
    )
    out2 = evolve_batch(
        ids2, 1, np.random.default_rng(9), provider2, registry2, max_frozen_retries=5
    )
    np.testing.assert_allclose(losses[0], out2.records[0].old_loss, rtol=1e-12)


def test_gpu_loss_of_matches_table_loss():
    """卡上整表损失与 CPU 的 table_loss 一致，整数数据逐位相同。"""
    from resevo.state import table_loss

    registry, ids, weights, rng = _scene(5, num_rows=40)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    codes = codebook.sync()
    target = target_from_specs(registry.specs)
    ctx = GpuBatchContext(qs, target, weights)
    got = ctx.loss_of(codes, ids)
    ref = table_loss(registry.build_workload(target, weights), ids)
    np.testing.assert_allclose(got, ref, rtol=1e-12)
