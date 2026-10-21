"""方向存在性只由可靠局部增益决定，不能由无关旧损失决定。"""
from fractions import Fraction

import numpy as np
import pytest

from resevo.batchkernel import build_batch_kernel
from resevo.batchmenu import BatchMenu, CodeBook, build_query_structure, menu_to_supports
from resevo.dataset import QuerySpec, StateRegistry, TableSchema, target_from_specs
from resevo.engine import build_kernel
from resevo.grouping import build_group_kernel, group_state_ids
from resevo.tilt import calibrate_beta, calibrate_beta_flat

BACKENDS = ("rows", "groups", "batch", "gpu")


def _gpu_context(qs, target, weights):
    cp = pytest.importorskip("cupy")
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if not devices:
        pytest.skip("No CUDA device")
    from resevo.gpukernel import GpuBatchContext

    return GpuBatchContext(qs, target, weights)


def _inputs(targets, current=(0,), inert=None, weight=1.0, halfspace=()):
    schema = TableSchema(("v",), (("0", "1"),))
    conditions = ({"attribute": "v", "operator": "==", "value": "1"},)
    hs = ({"operator": "halfspace", "scores": {"v": {"0": 0, "1": 1}}, "threshold": 1},)
    specs = [
        QuerySpec(f"q{i}", hs if i in halfspace else conditions, float(target))
        for i, target in enumerate(targets)
    ]
    if inert is not None:
        kind, residual = inert
        condition = (
            {"attribute": "v", "operator": "==", "value": "missing"} if kind == "zero"
            else {"attribute": "v", "operator": ">=", "value": "0"}
        )
        specs.append(QuerySpec("inert", (condition,), residual + (len(current) if kind == "one" else 0)))
    registry = StateRegistry(schema, specs)
    registry.register_table([("0",), ("1",)])
    ids = registry.register_table([(str(value),) for value in current])
    book = CodeBook(registry)
    qs = build_query_structure(registry, book)
    codes = book.sync()
    grouped = group_state_ids(ids)
    groups = grouped.num_groups
    fields = np.full((groups, 3), -1, dtype=np.int32)
    values = fields.copy()
    fields[:, 0] = 0
    values[:, 0] = 1 - codes[grouped.unique_ids, 0]
    menu = BatchMenu(np.arange(groups + 1, dtype=np.int64),
                     np.arange(groups, dtype=np.int64), fields, values, np.ones(groups))
    supports = menu_to_supports(menu, grouped.unique_ids, registry)
    workload = registry.build_workload(target_from_specs(specs), np.full(len(specs), weight))
    return workload, ids, grouped, supports, menu, qs, codes


def _build(backend, targets, current=(0,), inert=None, weight=1.0, halfspace=(), **kwargs):
    workload, ids, grouped, supports, menu, qs, codes = _inputs(targets, current, inert, weight, halfspace)
    if backend == "rows":
        return build_kernel(workload, ids, **kwargs)
    if backend == "groups":
        return build_group_kernel(workload, grouped, supports, **kwargs)
    if backend == "batch":
        return build_batch_kernel(workload, grouped, menu, qs, codes, **kwargs)
    return _gpu_context(qs, workload.target, workload.weights).build(grouped, menu, codes, **kwargs)


def _probabilities(result):
    if hasattr(result, "blocks"):
        return np.concatenate([block.probabilities for block in result.blocks])
    return result.probabilities


def _exact_count_gain(targets, before, after):
    total = Fraction(0)
    for value in targets:
        y = Fraction.from_float(float(value))
        total += ((y - before) ** 2 - (y - after) ** 2) / 2
    return float(total)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ("zero", "one"))
@pytest.mark.parametrize("offset", (1e6, 1e12))
@pytest.mark.parametrize("rows", (1, 3))
def test_inert_query_cannot_hide_positive_direction(backend, kind, offset, rows):
    current = (0,) * rows
    reference = _build(backend, [0.500001], current=current)
    shifted = _build(backend, [0.500001], current=current, inert=(kind, offset))
    assert reference.status == shifted.status == "ok"
    assert shifted.max_gain_sum == pytest.approx(reference.max_gain_sum, rel=1e-9)
    np.testing.assert_allclose(_probabilities(shifted), _probabilities(reference), atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", (1e-100, 1.0, 1e100))
def test_loss_unit_scaling_preserves_direction(backend, scale):
    base = _build(backend, [0.500001], inert=("zero", 1e6))
    scaled = _build(backend, [0.500001], inert=("zero", 1e6), weight=scale)
    assert scaled.status == base.status == "ok"
    np.testing.assert_allclose(_probabilities(scaled), _probabilities(base), atol=1e-10, rtol=1e-9)
    assert scaled.max_gain_sum / scale == pytest.approx(base.max_gain_sum, rel=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("middle", (1.25, np.nextafter(1.5, -np.inf), 1.5, np.nextafter(1.5, np.inf)))
def test_cancellation_is_rechecked_not_merely_enabled(backend, middle):
    targets = [1e16, middle, -1e16]
    true_gain = _exact_count_gain(targets, 0, 1)
    result = _build(backend, targets)
    if true_gain > 0:
        assert result.status == "ok"
        assert result.max_gain_sum == pytest.approx(true_gain, rel=1e-12, abs=0)
        assert _probabilities(result)[1] == pytest.approx(1.0, abs=1e-12)
    else:
        assert result.status == "no_positive_direction"
        assert result.max_gain_sum == 0
        np.testing.assert_array_equal(_probabilities(result), [1.0, 0.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_recheck_uses_target_and_answers_not_rounded_residual(backend):
    targets = [1e16, 2.500001, -1e16]
    assert _exact_count_gain(targets, 1, 0) < 0
    assert _exact_count_gain(targets, 1, 2) < 0
    result = _build(backend, targets, current=(0, 1))
    assert result.status == "no_positive_direction"
    assert result.max_gain_sum == 0.0
    assert result.expected_changed_rows == 0.0


@pytest.mark.parametrize("backend", ("batch", "gpu"))
@pytest.mark.parametrize("halfspace", ((0, 1, 2), (1,), (0, 2)))
@pytest.mark.parametrize("current,middle", (((0,), 1.25), ((0, 1), 2.500001)))
def test_halfspace_and_mixed_sparse_cancellation(backend, halfspace, current, middle):
    targets = [1e16, middle, -1e16]
    before = sum(current)
    for value in current:
        assert _exact_count_gain(targets, before, before + 1 - 2 * value) < 0
    kwargs = {"entry_budget": 1} if backend == "gpu" else {}
    result = _build(backend, targets, current=current, halfspace=halfspace, **kwargs)
    assert result.status == "no_positive_direction"
    assert result.max_gain_sum == result.expected_changed_rows == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_true_frozen_state_is_preserved(backend):
    result = _build(backend, [0.0], inert=("zero", 1e6))
    assert result.status == "no_positive_direction"
    assert result.step == result.direction_gain == result.max_gain_sum == 0.0


@pytest.mark.parametrize("flat", (False, True))
@pytest.mark.parametrize("gain", (1e-100, 1e-16, 1e-6, 1.0))
def test_calibration_does_not_use_old_loss_as_scale(flat, gain):
    gains = np.array([0.0, gain])
    refs = np.array([0.9, 0.1])

    def calibrate(loss):
        if flat:
            return calibrate_beta_flat(gains, refs, np.array([0, 2]), loss)
        return calibrate_beta([gains], [refs], loss)

    a, b = calibrate(1.0), calibrate(1e30)
    assert not a.frozen and not b.frozen
    assert a.beta == b.beta
    np.testing.assert_array_equal(a.probabilities, b.probabilities)
    assert a.beta * gain == pytest.approx(np.log(9.0), rel=1e-9)


@pytest.mark.parametrize("flat", (False, True))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_gain_is_not_frozen(flat, bad):
    gains, refs = np.array([0.0, bad]), np.array([0.9, 0.1])
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        if flat:
            calibrate_beta_flat(gains, refs, np.array([0, 2]), 1.0)
        else:
            calibrate_beta([gains], [refs], 1.0)


@pytest.mark.parametrize("flat", (False, True))
def test_unrepresentable_positive_requirement_is_not_frozen(flat):
    gains, refs = np.array([0.0, np.nextafter(0.0, 1.0)]), np.array([0.9, 0.1])
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        if flat:
            calibrate_beta_flat(gains, refs, np.array([0, 2]), 1.0)
        else:
            calibrate_beta([gains], [refs], 1.0)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf"), np.nextafter(0.0, 1.0)))
def test_gpu_calibration_does_not_report_indeterminate_as_frozen(bad):
    workload, _, _, _, _, qs, _ = _inputs([0.0])
    ctx = _gpu_context(qs, workload.target, workload.weights)
    cp = ctx.cp
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        ctx._calibrate_flat(
            cp.asarray([0.0, bad]), cp.asarray([0.9, 0.1]),
            cp.asarray([0, 2]), cp.asarray([0, 0]), cp.asarray([1.0]),
            1.0, 0.5, 1e-12, None,
        )


def test_local_calibration_tolerance_remains_checked(monkeypatch):
    import resevo.tilt as tilt

    monkeypatch.setattr(tilt, "brentq", lambda *args, **kwargs: 0.0)
    with pytest.raises(FloatingPointError, match="校准误差"):
        tilt.calibrate_beta_flat(
            np.array([0.0, 1.0]), np.array([0.4, 0.6]), np.array([0, 2]),
            1e30, alpha=0.8,
        )


def test_exact_gain_can_resolve_sign_but_reports_underflow():
    from resevo.gain import exact_local_gain

    delta = np.ones(3)
    assert exact_local_gain(delta, np.array([1e16, 1.25, -1e16]), np.ones(3)) == -0.25
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        exact_local_gain(np.array([1.0]), np.array([1.0]), np.array([np.nextafter(0.0, 1.0)]))
