"""核出口的有限性与表示合同；不改变采样或 Issue83 的支持语义。"""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from resevo import batchkernel, engine, grouping
from resevo.batchkernel import BatchRoundPlan, build_batch_kernel, evolve_batch
from resevo.batchmenu import BatchMenu, CodeBook, build_query_structure
from resevo.candidates import BlockSupport
from resevo.dataset import QuerySpec, StateRegistry, TableSchema
from resevo.engine import NumericalEvolutionError, build_kernel, evolve
from resevo.grouping import RoundPlan, build_group_kernel, evolve_grouped, group_state_ids
from resevo.stepsize import (
    analytic_step, apply_row_budget, expected_improvement, final_probabilities, kernel_statistics,
)
from test_local_gain_freeze import BACKENDS, _build, _gpu_context

TINY = np.nextafter(0.0, 1.0)
ROOT = Path(__file__).resolve().parents[1]


def _clock_scene(weight=4.0, mobility=1e-300):
    schema = TableSchema(("v",), (("0", "1", "2"),))
    target, weights = np.array([1.0, 0.0]), np.array([1.0, weight])
    specs = [
        QuerySpec(f"q{i}", ({"attribute": "v", "operator": "==", "value": str(i + 1)},), value)
        for i, value in enumerate(target)
    ]
    registry = StateRegistry(schema, specs)
    ids = registry.register_table([("0",)])
    dest = registry.register_table([("1",), ("2",)])
    supports = [BlockSupport((0,), tuple((int(i),) for i in dest), (mobility, 1.0))]
    workload = registry.build_workload(target, weights)
    grouped = group_state_ids(ids)
    book = CodeBook(registry)
    qs = build_query_structure(registry, book)
    menu = BatchMenu(
        np.array([0, 2]), np.array([0, 0]),
        np.array([[0, -1, -1], [0, -1, -1]], dtype=np.int32),
        np.array([[1, -1, -1], [2, -1, -1]], dtype=np.int32),
        np.array([mobility, 1.0]),
    )
    return registry, ids, workload, grouped, supports, book, qs, menu


def _clock_kernel(backend, weight=4.0, mobility=1e-300, **kwargs):
    _, ids, workload, grouped, supports, book, qs, menu = _clock_scene(weight, mobility)
    if backend == "rows":
        return build_kernel(workload, ids, supports, **kwargs)
    if backend == "groups":
        return build_group_kernel(workload, grouped, supports, **kwargs)
    if backend == "batch":
        return build_batch_kernel(workload, grouped, menu, qs, book.sync(), **kwargs)
    return _gpu_context(qs, workload.target, workload.weights).build(grouped, menu, book.sync(), **kwargs)


def _probabilities(result):
    return result.blocks[0].probabilities if hasattr(result, "blocks") else result.probabilities


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs", ({"damping": TINY}, {"max_expected_rows": TINY}))
def test_positive_direction_cannot_return_zero_step(backend, kwargs):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        _build(backend, [1.25], current=(0, 0, 0, 0), **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_true_frozen_state_still_accepts_positive_tiny_controls(backend):
    result = _build(
        backend, [0.0], current=(0, 0, 0, 0), damping=TINY, max_expected_rows=TINY,
    )
    assert result.status == "no_positive_direction"
    assert result.step == result.expected_changed_rows == result.max_gain_sum == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_huge_finite_clock_has_finite_correct_ledger(backend):
    result = _clock_kernel(backend, alpha=1e-310)
    probabilities = _probabilities(result)
    assert result.status == "ok" and result.step > 1e154 and result.interaction == 0.0
    assert np.isfinite(probabilities).all() and probabilities.sum() == pytest.approx(1.0)
    exact_loss = float(probabilities @ np.array([0.5, 0.0, 2.5]))
    assert result.expected_loss == pytest.approx(exact_loss, rel=1e-12, abs=1e-14)
    assert np.isfinite(result.analytic_upper_bound)
    assert result.expected_changed_rows == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_long_clock_probe_uses_finite_same_formula(backend):
    result = _clock_kernel(backend, alpha=1e-310, probe_grid=8)
    assert result.probe is not None and np.isfinite(result.probe).all()
    assert result.probe[1] >= 0.0
    assert np.isfinite(result.expected_loss)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonfinite_clock_fails_before_kernel_or_sampler(backend):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        _clock_kernel(backend, weight=200.0, mobility=1e-320, alpha=2e-323)


@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_row_budget_can_bound_an_unrepresentable_feasibility_limit(backend):
    result = _clock_kernel(
        backend, weight=200.0, mobility=1e-320, alpha=2e-323, max_expected_rows=1e-100,
    )
    assert result.status == "ok" and 0 < result.step < np.inf
    assert np.isfinite(_probabilities(result)).all()
    assert np.isfinite(result.expected_loss)
    assert result.expected_changed_rows == pytest.approx(1e-100, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("rates", ([0.0, np.nan], [0.0, np.inf], [0.0, -0.1], [0.0, 1.1]))
@pytest.mark.parametrize("flat", (False, True))
def test_invalid_probability_construction_is_explicit(rates, flat):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        final_probabilities(np.array(rates), 1.0, np.array([0, 2]) if flat else None)


def test_probability_rounding_and_inactive_segments_preserve_old_behavior():
    rates = np.array([0.0, np.nextafter(1.0, np.inf), 0.0])
    result, moving = final_probabilities(rates, 1.0, np.array([0, 2, 3]))
    np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])
    assert moving
    result, moving = final_probabilities(np.zeros(1), 1.0)
    np.testing.assert_array_equal(result, [1.0])
    assert not moving
    with pytest.raises(FloatingPointError, match="非保持质量"):
        kernel_statistics(1.0, 1.0, 0.0, 1.0, 1.0, moving)


@pytest.mark.parametrize("flat", (False, True))
def test_ordinary_probabilities_keep_legacy_arithmetic_bitwise(flat):
    rng = np.random.default_rng(8201)
    offsets = np.array([0, 3, 4, 9]) if flat else np.array([0, 9])
    for _ in range(100):
        rates = rng.uniform(0.0, 1.0, int(offsets[-1]))
        rates[offsets[:-1]] = 0.0
        mass = np.add.reduceat(rates, offsets[:-1])
        step = float(0.8 / mass.max())
        expected = step * rates
        if flat:
            stay = 1.0 - np.add.reduceat(expected, offsets[:-1])
            expected[offsets[:-1]] = np.maximum(stay, 0.0)
        else:
            expected[0] = 1.0 - float(expected[1:].sum())
        actual, moving = final_probabilities(rates, step, offsets if flat else None)
        np.testing.assert_array_equal(actual, expected)
        assert moving


def test_stable_ledger_avoids_large_product_overflow():
    loss, upper, rows = kernel_statistics(1.2e308, 1.2e308, 4e307, 1.5, 0.5, True)
    assert loss == pytest.approx(3e307, rel=1e-14)
    assert upper == pytest.approx(3e307, rel=1e-14)
    assert rows == 0.75
    assert expected_improvement(1e240, 1e-250, 0.0) == pytest.approx(1e-10, rel=1e-14)


@pytest.mark.parametrize("b,D,C", [(np.nan, 1.0, 0.0), (np.inf, 1.0, 0.0), (0.5, 0.0, 0.0), (0.5, -1.0, 0.0)])
def test_invalid_active_moments_are_numerical_errors(b, D, C):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        analytic_step(b, D, C)


def test_budget_postcondition_and_missing_unit_rows():
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        apply_row_budget(1.0, 4.0, TINY)
    with pytest.raises(ValueError, match="unit_expected_rows"):
        analytic_step(0.5, 1.0, 0.0, max_expected_rows=0.3)
    assert apply_row_budget(0.0, 0.0, 0.3) == 0.0
    assert apply_row_budget(float("inf"), 1.0, 0.3) == 0.3


@pytest.mark.parametrize("backend", BACKENDS)
def test_actual_clock_failure_preserves_context_without_sampling(monkeypatch, backend):
    registry, ids, workload, grouped, supports, book, qs, menu = _clock_scene(200.0, 1e-320)
    rng = np.random.default_rng(7)
    before = copy.deepcopy(rng.bit_generator.state)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid final kernel reached sampling")

    def group_provider(*args):
        return RoundPlan("grouped", workload, supports, grouped)

    def batch_provider(*args):
        return BatchRoundPlan("batch", workload, grouped, menu)

    batch_provider.codebook, batch_provider.structure = book, [qs]
    with pytest.raises(NumericalEvolutionError) as error:
        if backend == "rows":
            monkeypatch.setattr(engine, "sample_next", forbidden)
            evolve(workload, ids, 3, rng, supports=supports, alpha=2e-323)
        elif backend == "groups":
            monkeypatch.setattr(grouping, "sample_group_next", forbidden)
            evolve_grouped(ids, 3, rng, group_provider, alpha=2e-323)
        else:
            if backend == "gpu":
                _gpu_context(qs, workload.target, workload.weights)
            monkeypatch.setattr(batchkernel, "sample_batch_next", forbidden)
            evolve_batch(
                ids, 3, rng, batch_provider, registry, alpha=2e-323,
                backend="gpu" if backend == "gpu" else "cpu",
            )
    np.testing.assert_array_equal(error.value.partial_result.state_ids, ids)
    assert error.value.partial_result.records == []
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_public_cli_zero_step_is_failure_not_normal_output(tmp_path, backend):
    if backend == "gpu":
        _, _, workload, _, _, _, qs, _ = _clock_scene()
        _gpu_context(qs, workload.target, workload.weights)
    output = tmp_path / "table.csv"
    result = subprocess.run([
        sys.executable, "-m", "resevo", "--schema", str(ROOT / "configs/resevo/smoke_schema.json"),
        "--queries", str(ROOT / "configs/resevo/smoke_queries.json"), "--rows", "64",
        "--rounds", "3", "--seed", "7", "--init", "uniform",
        "--max-expected-rows", "5e-324", "--backend", backend, "--output", str(output),
    ], env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "numerical_indeterminate" in result.stderr
    assert not output.exists() and result.stdout == ""
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed" and manifest["completed_rounds"] == 0
    schema = json.loads((ROOT / "configs/resevo/smoke_schema.json").read_text())
    initial_rng = np.random.default_rng(7)
    initial = [[domain[int(initial_rng.integers(len(domain)))] for domain in schema["domains"]] for _ in range(64)]
    with (manifests[0].parent / "current.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [schema["fields"], *initial]
