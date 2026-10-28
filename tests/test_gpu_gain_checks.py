"""固定整数性缓存及单次同步的局部复核检查，不牺牲空掩码上的错误检测。"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("无可用 GPU", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError as exc:
    pytest.skip(f"CUDA runtime unavailable: {exc}", allow_module_level=True)

from resevo.gain import gain_recheck_mask
from resevo.gpukernel import GpuBatchContext
from test_local_gain_freeze import _inputs


def _context(integral=False):
    workload, _, _, _, _, qs, _ = _inputs([1.0 if integral else 1.25])
    return GpuBatchContext(qs, workload.target, workload.weights)


@pytest.mark.parametrize("integral", (False, True))
@pytest.mark.parametrize("tolerance", (0.0, 1e-12, 1e-6))
@pytest.mark.parametrize("seed", range(4))
def test_fused_mask_matches_cpu_at_rounding_boundaries(integral, tolerance, seed):
    rng = np.random.default_rng(seed)
    magnitudes = np.concatenate([
        np.logspace(-320, 100, 256),
        [0.0, np.nextafter(2.0**50, 0), 2.0**50, np.nextafter(2.0**50, np.inf)],
    ])
    terms = rng.integers(0, 1024, magnitudes.size, dtype=np.int64)
    operations = 8.0 * (terms.astype(np.float64) + 1.0)
    ku = operations * (np.finfo(np.float64).eps / 2)
    gamma = ku / (1 - ku)
    threshold = np.maximum(
        np.maximum(tolerance, gamma / (1 - gamma)) * magnitudes,
        operations * np.nextafter(0.0, 1.0),
    )
    gains = np.concatenate([np.nextafter(threshold, 0), threshold, np.nextafter(threshold, np.inf)])
    gains[::2] *= -1
    magnitudes, terms = np.tile(magnitudes, 3), np.tile(terms, 3)
    expected = gain_recheck_mask(gains, magnitudes, terms, tolerance, integral=integral)
    actual, count = _context(integral)._gain_recheck_mask(
        cp.asarray(gains), cp.asarray(magnitudes), cp.asarray(terms), tolerance,
    )
    np.testing.assert_array_equal(cp.asnumpy(actual), expected)
    assert count == np.count_nonzero(expected)


@pytest.mark.parametrize("target,weight", ((1.0, 1.0), (1.25, 1.0), (1.0, 0.25)))
def test_integrality_is_cached_from_owned_static_inputs(monkeypatch, target, weight):
    import resevo.gpukernel as gpu

    _, _, _, _, _, qs, _ = _inputs([target])
    calls = []
    original = gpu.integral_values

    def checked(*values):
        calls.append(True)
        return original(*values)

    monkeypatch.setattr(gpu, "integral_values", checked)
    targets, weights = np.array([target]), np.array([weight])
    ctx = GpuBatchContext(qs, targets, weights)
    assert calls == [True]
    targets[0], weights[0] = 3.5, 2.5
    np.testing.assert_array_equal(cp.asnumpy(ctx.target), [target])
    np.testing.assert_array_equal(cp.asnumpy(ctx.weights), [weight])
    assert not ctx._target_host.flags.writeable and not ctx._weights_host.flags.writeable
    assert ctx._integral_gain_inputs is (target == 1.0 and weight == 1.0)
    for _ in range(3):
        ctx._gain_recheck_mask(cp.asarray([0.0]), cp.asarray([0.0]), cp.asarray([0]), 1e-12)
    assert calls == [True]


@pytest.mark.parametrize("selected", (False, True))
def test_check_status_and_selection_count_use_one_host_transfer(monkeypatch, selected):
    ctx = _context()
    arrays = (cp.asarray([0.0 if selected else 1.0]), cp.asarray([2.0]), cp.asarray([3]))
    transfers = []
    original = cp.asnumpy

    def transfer(array, *args, **kwargs):
        transfers.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", transfer)
    mask, count = ctx._gain_recheck_mask(*arrays, 1e-12)
    assert transfers == [(2,)]
    assert count == int(selected)
    np.testing.assert_array_equal(original(mask), [selected])


@pytest.mark.parametrize("gain,magnitude,terms", [
    (float("nan"), 1.0, 1), (float("inf"), 1.0, 1),
    (1.0, float("inf"), 1), (1.0, -1.0, 1), (1.0, 1.0, 2**60),
])
def test_invalid_inputs_are_not_hidden_by_an_empty_selection(gain, magnitude, terms):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        _context()._gain_recheck_mask(
            cp.asarray([gain]), cp.asarray([magnitude]), cp.asarray([terms]), 1e-12,
        )


def test_integer_shortcut_and_mixed_range_keep_cpu_error_semantics():
    ctx = _context(True)
    gains, terms = np.array([1.0, 1.0]), np.array([2**60, 1], dtype=np.int64)
    for magnitudes in (np.ones(2), np.array([1.0, 2.0**51])):
        if magnitudes.max() <= 2.0**50:
            expected = gain_recheck_mask(gains, magnitudes, terms, 1e-12, integral=True)
            actual, count = ctx._gain_recheck_mask(
                cp.asarray(gains), cp.asarray(magnitudes), cp.asarray(terms), 1e-12,
            )
            np.testing.assert_array_equal(cp.asnumpy(actual), expected)
            assert count == 0
        else:
            with pytest.raises(FloatingPointError, match="舍入界"):
                gain_recheck_mask(gains, magnitudes, terms, 1e-12, integral=True)
            with pytest.raises(FloatingPointError, match="舍入界"):
                ctx._gain_recheck_mask(cp.asarray(gains), cp.asarray(magnitudes), cp.asarray(terms), 1e-12)


def test_empty_paths_still_validate_tolerance():
    ctx = _context()
    arrays = (cp.empty(0), cp.empty(0), cp.empty(0, dtype=cp.int64))
    mask, count = ctx._gain_recheck_mask(*arrays, 1e-12)
    assert mask.size == count == 0
    with pytest.raises(ValueError, match="numerical_tol"):
        ctx._gain_recheck_mask(*arrays, float("nan"))
