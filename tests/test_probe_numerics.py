"""数值不可用的只读探针不能中断有效主核，也不能伪装成零下降。"""
from dataclasses import asdict
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from resevo.batchkernel import BatchRoundPlan, build_batch_kernel, evolve_batch, make_batch_provider
from resevo.dataset import target_from_specs
from resevo.editspace import tuples_from_ids
from resevo.engine import NumericalEvolutionError
from resevo.failure import write_numerical_failure
from resevo.grouping import group_state_ids
from test_batchkernel import _scene
from test_kernel_postconditions import _clock_kernel, _clock_scene
from test_local_gain_freeze import _gpu_context

ROOT = Path(__file__).resolve().parents[1]


def _case(kind):
    if kind == "range":
        registry, ids, workload, grouped, supports, book, qs, menu = _clock_scene(0.001, 1.0)
        workload = registry.build_workload(workload.target, workload.weights * 1e-320)
        alpha = 0.01
    else:
        weight, mobility = (200.0, 1e-320) if kind == "partial" else (4.0, 1e-300)
        registry, ids, workload, grouped, supports, book, qs, menu = _clock_scene(weight, mobility)
        alpha = 0.5
    return registry, ids, workload, grouped, book, qs, menu, alpha


def _build(backend, kind, grid):
    _, _, workload, grouped, book, qs, menu, alpha = _case(kind)
    if backend == "gpu":
        return _gpu_context(qs, workload.target, workload.weights).build(
            grouped, menu, book.sync(), alpha=alpha, probe_grid=grid,
        )
    return build_batch_kernel(workload, grouped, menu, qs, book.sync(), alpha=alpha, probe_grid=grid)


def _prepare(backend, kind):
    registry, ids, workload, _, book, qs, menu, alpha = _case(kind)
    if backend == "gpu":
        _gpu_context(qs, workload.target, workload.weights)
    rng = np.random.default_rng(7)

    def run(interval, rounds=1, fail_after=None):
        def provider(current, round_index, frozen_streak=0):
            if round_index == fail_after:
                raise FloatingPointError("numerical_indeterminate: after diagnostic")
            return BatchRoundPlan("batch", workload, group_state_ids(current), menu)

        provider.codebook, provider.structure = book, [qs]
        return evolve_batch(
            ids, rounds, rng, provider, registry, alpha=alpha,
            backend="gpu" if backend == "gpu" else "cpu", probe_interval=interval,
        )

    return registry, rng, run


def _records(result):
    return [{key: value for key, value in asdict(record).items() if key != "elapsed"}
            for record in result.records]


def _assert_main_equal(plain, probed):
    np.testing.assert_array_equal(probed.probabilities, plain.probabilities)
    for name in ("beta", "direction_gain", "interaction", "step", "expected_loss",
                 "analytic_upper_bound", "expected_changed_rows", "max_gain_sum", "required_gain", "status"):
        assert getattr(probed, name) == getattr(plain, name)


@pytest.mark.parametrize("backend", ("batch", "gpu"))
@pytest.mark.parametrize("grid", (8, 16))
def test_real_partial_grid_preserves_valid_kernel(backend, grid, recwarn):
    plain = _build(backend, "partial", 0)
    assert plain.step == 2.0
    np.testing.assert_array_equal(plain.probabilities, [0.0, 1.0, 0.0])
    probed = _build(backend, "partial", grid)
    _assert_main_equal(plain, probed)
    assert probed.probe is not None and np.isfinite(probed.probe).all()
    diagnostics = probed.probe_diagnostics
    assert diagnostics.status == "partial" and diagnostics.valid_points > 0
    assert diagnostics.unavailable_points
    assert all(point.reason.startswith("numerical_indeterminate:") for point in diagnostics.unavailable_points)
    assert any("beta 网格诊断" in str(w.message) for w in recwarn)


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_real_unrepresentable_grid_range_is_explicitly_unavailable(backend, recwarn):
    plain = _build(backend, "range", 0)
    assert plain.status == "ok" and np.isfinite(plain.probabilities).all()
    probed = _build(backend, "range", 16)
    _assert_main_equal(plain, probed)
    assert probed.probe is None
    assert probed.probe_diagnostics.status == "unavailable"
    assert probed.probe_diagnostics.valid_points == 0
    assert probed.probe_diagnostics.unavailable_points
    assert any("beta 网格诊断" in str(w.message) for w in recwarn)


@pytest.mark.parametrize("backend", ("batch", "gpu"))
@pytest.mark.parametrize("kind,status", (("partial", "partial"), ("range", "unavailable")))
def test_real_probe_failure_preserves_evolution_and_rng(backend, kind, status, recwarn):
    reference_registry, reference_rng, reference_run = _prepare(backend, kind)
    reference = reference_run(0)
    registry, rng, run = _prepare(backend, kind)
    result = run(1)
    assert tuples_from_ids(registry, result.state_ids) == tuples_from_ids(reference_registry, reference.state_ids)
    assert _records(result) == _records(reference)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert len(result.probes) == 1 and result.probes[0].diagnostics.status == status
    if status == "unavailable":
        assert result.probes[0].beta_best is None and result.probes[0].j_best is None


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_all_positive_points_unavailable_does_not_fabricate_best(backend, monkeypatch, recwarn):
    from resevo import probe

    def unavailable(*args, **kwargs):
        raise FloatingPointError("numerical_indeterminate: diagnostic-only fixture")

    monkeypatch.setattr(probe, "analytic_step", unavailable)
    plain = _build(backend, "normal", 0)
    probed = _build(backend, "normal", 8)
    _assert_main_equal(plain, probed)
    assert probed.probe is None
    assert probed.probe_diagnostics.status == "unavailable"
    assert probed.probe_diagnostics.valid_points == 0
    assert len(probed.probe_diagnostics.unavailable_points) > 1


@pytest.mark.parametrize("backend", ("batch", "gpu"))
@pytest.mark.parametrize("error", (
    FloatingPointError("unrelated"), ValueError("numerical_indeterminate: wrong type"), RuntimeError("device"),
))
def test_unrelated_diagnostic_errors_are_not_suppressed(backend, error, monkeypatch):
    from resevo import probe

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(probe, "analytic_step", fail)
    with pytest.raises(type(error)) as caught:
        _build(backend, "normal", 8)
    assert caught.value is error


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_actual_main_failure_still_aborts_with_probe_enabled(backend, recwarn):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        _clock_kernel(backend, weight=200.0, mobility=1e-320, alpha=2e-323, probe_grid=16)
    assert not any("beta 网格诊断" in str(w.message) for w in recwarn)


@pytest.mark.parametrize("backend", ("batch", "gpu"))
@pytest.mark.parametrize("seed", (2, 4))
def test_normal_probe_switch_preserves_both_rngs(backend, seed):
    observed = []
    for interval in (0, 1):
        registry, ids, weights, _ = _scene(seed, num_rows=30)
        target = target_from_specs(registry.specs)
        if backend == "gpu":
            registry.set_lazy_features(True)
        menu_rng, sample_rng = np.random.default_rng(5), np.random.default_rng(9)
        provider = make_batch_provider(
            registry, target, weights, menu_rng, pairing=False, defer_workload=backend == "gpu",
        )
        if backend == "gpu":
            from test_local_gain_freeze import _inputs
            workload, _, _, _, _, qs, _ = _inputs([1.0])
            _gpu_context(qs, workload.target, workload.weights)
        result = evolve_batch(
            ids, 20, sample_rng, provider, registry, max_frozen_retries=3,
            backend="gpu" if backend == "gpu" else "cpu", probe_interval=interval,
        )
        if interval:
            assert result.probes and all(p.diagnostics.status == "ok" for p in result.probes)
        observed.append((
            tuples_from_ids(registry, result.state_ids), _records(result),
            menu_rng.bit_generator.state, sample_rng.bit_generator.state,
        ))
    assert observed[0] == observed[1]


def _plants():
    spec = importlib.util.spec_from_file_location("run_plants_probe_test", ROOT / "scripts/run_plants.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind,status", (("normal", "ok"), ("partial", "partial"), ("range", "unavailable")))
def test_curve_records_unavailable_points_without_changing_normal_columns(tmp_path, kind, status, recwarn):
    _, _, run = _prepare("batch", kind)
    result = run(1)
    output = tmp_path / "curve.csv"
    _plants().write_curve(output, result, 0.0)
    with (tmp_path / "curve_probe.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames
    old_columns = ["round", "beta_used", "J_used", "beta_best", "J_best"]
    assert columns[:5] == old_columns and len(rows) == 1
    if status == "ok":
        assert columns == old_columns
    else:
        assert rows[0]["probe_status"] == status
        failures = json.loads(rows[0]["probe_unavailable_points"])
        assert failures and all(set(item) == {"beta", "reason"} for item in failures)
        if status == "unavailable":
            assert rows[0]["beta_best"] == rows[0]["J_best"] == ""


@pytest.mark.parametrize("kind", ("partial", "range"))
def test_later_main_failure_keeps_probe_diagnostics(tmp_path, kind, recwarn):
    registry, _, run = _prepare("batch", kind)
    with pytest.raises(NumericalEvolutionError, match="after diagnostic") as caught:
        run(1, rounds=2, fail_after=1)
    assert len(caught.value.partial_result.records) == len(caught.value.partial_result.probes) == 1
    folder = write_numerical_failure(registry, caught.value, tmp_path / "table.csv")
    with (folder / "probes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = "partial" if kind == "partial" else "unavailable"
    assert rows[0]["probe_status"] == expected
    assert json.loads(rows[0]["probe_unavailable_points"])
    if expected == "unavailable":
        assert rows[0]["beta_best"] == rows[0]["j_best"] == ""
