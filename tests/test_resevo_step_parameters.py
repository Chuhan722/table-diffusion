"""步长输入合同：非法值先拒绝，合法值保持原有核与抽样语义。"""
from __future__ import annotations

import copy
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from resevo.batchkernel import build_batch_kernel, evolve_batch
from resevo.batchmenu import BatchMenu, CodeBook, build_query_structure, menu_to_supports
from resevo.dataset import QuerySpec, StateRegistry, TableSchema
from resevo.engine import build_kernel, evolve
from resevo.grouping import build_group_kernel, evolve_grouped, group_state_ids
from resevo.stepsize import analytic_step, apply_row_budget

BACKENDS = ("rows", "groups", "batch", "gpu")
BAD_PARAMETERS = [
    pytest.param({"damping": value}, "damping", id=f"damping-{label}")
    for label, value in (
        ("zero", 0.0), ("negative", -1.0), ("over-one", 1.25), ("uphill", 3.0),
        ("nan", float("nan")), ("inf", float("inf")), ("negative-inf", float("-inf")),
    )
] + [
    pytest.param({"max_expected_rows": value}, "max_expected_rows", id=f"budget-{label}")
    for label, value in (
        ("zero", 0.0), ("negative", -1.0), ("nan", float("nan")),
        ("inf", float("inf")), ("negative-inf", float("-inf")),
    )
]


def _inputs(frozen=False):
    schema = TableSchema(("a",), (("0", "1"),))
    target = np.array([0.0 if frozen else 1.0])
    specs = [QuerySpec("a1", ({"attribute": "a", "operator": "==", "value": "1"},), target[0])]
    registry = StateRegistry(schema, specs)
    ids = registry.register_table([("0",)] * 3)
    registry.register(("1",))
    grouped = group_state_ids(ids)
    book = CodeBook(registry)
    codes = book.sync()
    qs = build_query_structure(registry, book)
    menu = BatchMenu(
        np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64),
        np.array([[0, -1, -1]], dtype=np.int32),
        np.array([[1, -1, -1]], dtype=np.int32), np.ones(1),
    )
    supports = menu_to_supports(menu, grouped.unique_ids, registry)
    workload = registry.build_workload(target, np.ones(1))
    return workload, ids, grouped, supports, menu, qs, codes


def _build(backend, frozen=False, **kwargs):
    workload, ids, grouped, supports, menu, qs, codes = _inputs(frozen)
    if backend == "rows":
        return build_kernel(workload, ids, **kwargs)
    if backend == "groups":
        return build_group_kernel(workload, grouped, supports, **kwargs)
    if backend == "batch":
        return build_batch_kernel(workload, grouped, menu, qs, codes, **kwargs)
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if count == 0:
        pytest.skip("No CUDA device")
    from resevo.gpukernel import GpuBatchContext

    ctx = GpuBatchContext(qs, workload.target, workload.weights)
    return ctx.build(grouped, menu, codes, **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("frozen", (False, True), ids=("active", "frozen"))
@pytest.mark.parametrize("kwargs,parameter", BAD_PARAMETERS)
def test_build_rejects_invalid_step_parameters(backend, frozen, kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        _build(backend, frozen, **kwargs)


def _evolve_without_provider_calls(backend, num_rounds, rng, **kwargs):
    def forbidden_provider(*_):
        pytest.fail("Invalid parameters must not invoke the candidate provider")

    ids = np.zeros(3, dtype=np.int64)
    if backend == "rows":
        return evolve(None, ids, num_rounds, rng, support_provider=forbidden_provider, **kwargs)
    if backend == "groups":
        return evolve_grouped(ids, num_rounds, rng, forbidden_provider, **kwargs)
    return evolve_batch(ids, num_rounds, rng, forbidden_provider, None,
                        backend="gpu" if backend == "gpu" else "cpu", **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs,parameter", BAD_PARAMETERS)
def test_evolve_rejects_before_provider_or_rng(backend, kwargs, parameter):
    rng = np.random.default_rng(7701)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match=parameter):
        _evolve_without_provider_calls(backend, 1, rng, **kwargs)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_round_behavior_is_preserved(backend):
    rng = np.random.default_rng(7702)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="轮数"):
        _evolve_without_provider_calls(backend, 0, rng, damping=1.0)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("damping,budget", [(1.0, None), (0.5, None), (1.0, 0.3), (0.5, 0.3)])
def test_valid_parameters_preserve_three_row_kernel(backend, damping, budget):
    result = _build(backend, damping=damping, max_expected_rows=budget)
    if backend in ("rows", "groups"):
        p = float(result.blocks[0].probabilities[1])
    else:
        p = float(result.probabilities[1])
    expected_step = damping * (0.5 if budget is None else min(0.5, budget / 1.5))
    assert result.step == pytest.approx(expected_step, abs=1e-12)
    assert p == pytest.approx(0.5 * expected_step, abs=1e-12)
    exact_loss = sum(math.comb(3, k) * p**k * (1-p)**(3-k) * (1-k)**2 / 2 for k in range(4))
    assert result.expected_loss == pytest.approx(exact_loss, abs=1e-12)
    assert result.expected_loss <= result.old_loss
    assert result.expected_changed_rows == pytest.approx(3*p, abs=1e-12)
    if budget is not None:
        assert result.expected_changed_rows <= damping * budget + 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_valid_parameters_preserve_frozen_kernel(backend):
    result = _build(backend, frozen=True, damping=0.5, max_expected_rows=0.3)
    assert result.status == "no_positive_direction"
    assert result.step == result.expected_changed_rows == result.expected_loss == 0.0


@pytest.mark.parametrize("value", [0.0, -1.0, 1.25, 3.0, float("nan"), float("inf"), float("-inf")])
def test_analytic_step_rejects_invalid_damping(value):
    with pytest.raises(ValueError, match="damping"):
        analytic_step(0.5, 0.75, 0.75, damping=value)


@pytest.mark.parametrize("value", [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("unit_rows", [0.0, 1.5])
def test_row_budget_rejects_invalid_cap_even_without_motion(value, unit_rows):
    with pytest.raises(ValueError, match="max_expected_rows"):
        apply_row_budget(0.5, unit_rows, value)


@pytest.mark.parametrize("value", ["0", "-1", "1.25", "3", "nan", "inf", "-inf"])
def test_benchmark_cli_rejects_damping_before_reading_data(value):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_plants.py"
    result = subprocess.run(
        [sys.executable, str(script), "--batched", "--data", "nonexistent_parameter_test",
         f"--damping={value}", "--rounds", "1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "damping" in result.stderr and "(0,1]" in result.stderr
    assert "Traceback" not in result.stderr
