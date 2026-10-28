"""保持项回算的有限精度合同：整体退化报错，部分受限明示，正常调用不变。"""
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from resevo.batchkernel import BatchRoundPlan, build_batch_kernel, evolve_batch, sample_batch_next
from resevo.batchmenu import BatchMenu, CodeBook, build_query_structure
from resevo.candidates import BlockSupport
from resevo.dataset import QuerySpec, StateRegistry, TableSchema
from resevo.engine import NumericalEvolutionError, build_kernel, evolve
from resevo.failure import save_numerical_failure
from resevo.grouping import RoundPlan, build_group_kernel, evolve_grouped, group_state_ids
from resevo.probe import ProbeDiagnostics, ProbeNumericalWarning
from resevo.sampling import (
    SAMPLING_CONTRACT, SamplingPrecision, SamplingPrecisionWarning, flat_sampling_mass,
    resolve_sampling_mass, sampling_summary,
)
from test_kernel_postconditions import _clock_kernel
from test_local_gain_freeze import BACKENDS, _build, _gpu_context
from test_probe_numerics import _plants

ROOT = Path(__file__).resolve().parents[1]
TINY_MASS = 2.0**-53 / 3


def _partial_scene(weight=100.0):
    schema = TableSchema(("a", "b", "tag"), (("0", "1"),) * 3)
    target, weights = np.array([1.0, 0.0]), np.array([1.0, weight])
    specs = [
        QuerySpec(name, ({"attribute": name, "operator": "==", "value": "1"},), value)
        for name, value in zip(("a", "b"), target)
    ]
    registry = StateRegistry(schema, specs)
    ids = registry.register_table([("0", "0", "0"), ("0", "0", "1")])
    dest = registry.register_table([("1", "0", "0"), ("0", "1", "1")])
    supports = [BlockSupport((i,), ((int(dest[i]),),)) for i in range(2)]
    workload = registry.build_workload(target, weights)
    grouped = group_state_ids(ids)
    book = CodeBook(registry)
    qs = build_query_structure(registry, book)
    menu = BatchMenu(
        np.array([0, 1, 2]), np.array([0, 1]),
        np.array([[0, -1, -1], [1, -1, -1]], dtype=np.int32),
        np.array([[1, -1, -1], [1, -1, -1]], dtype=np.int32), np.ones(2),
    )
    return registry, ids, workload, grouped, supports, book, qs, menu


def _partial_kernel(backend, scene, **kwargs):
    _, ids, wl, grouped, supports, book, qs, menu = scene
    if backend == "rows":
        return build_kernel(wl, ids, supports, **kwargs)
    if backend == "groups":
        return build_group_kernel(wl, grouped, supports, **kwargs)
    if backend == "batch":
        return build_batch_kernel(wl, grouped, menu, qs, book.sync(), **kwargs)
    return _gpu_context(qs, wl.target, wl.weights).build(grouped, menu, book.sync(), **kwargs)


def _partial_run(backend, *, fail_after_first=False, round_limit=None, **kwargs):
    registry, ids, wl, grouped, supports, book, qs, menu = _partial_scene()
    if backend == "gpu":
        _gpu_context(qs, wl.target, wl.weights)

    def provide(current, k, streak):
        if fail_after_first and k == 1:
            raise FloatingPointError("numerical_indeterminate: after precision warning")
        if backend == "rows":
            return wl, supports
        current_groups = group_state_ids(current)
        by_group = []
        fields = np.full((current_groups.num_groups, 3), -1, dtype=np.int32)
        values = fields.copy()
        for g, state in enumerate(current_groups.unique_ids):
            tag = registry.state_tuple(int(state))[2]
            j = 0 if tag == "0" else 1
            destination = ("1", "0", "0") if j == 0 else ("0", "1", "1")
            dest = int(registry.register_table([destination])[0])
            by_group.append(BlockSupport((g,), ((dest,),)))
            fields[g, 0], values[g, 0] = j, 1
        if backend == "groups":
            return RoundPlan("grouped", wl, by_group, current_groups)
        current_menu = BatchMenu(
            np.arange(current_groups.num_groups + 1), np.arange(current_groups.num_groups),
            fields, values, np.ones(current_groups.num_groups),
        )
        return BatchRoundPlan("batch", wl, current_groups, current_menu)

    provide.codebook, provide.structure = book, [qs]
    rng = np.random.default_rng(7)
    rounds = round_limit if round_limit is not None else (2 if fail_after_first else 1)
    if backend == "rows":
        run = lambda: evolve(None, ids, rounds, rng, support_provider=provide, **kwargs)
    elif backend == "groups":
        run = lambda: evolve_grouped(ids, rounds, rng, provide, **kwargs)
    else:
        run = lambda: evolve_batch(ids, rounds, rng, provide, registry, backend="gpu" if backend == "gpu" else "cpu", **kwargs)
    return registry, ids, rng, run


@pytest.mark.parametrize("backend", BACKENDS)
def test_whole_kernel_resolution_loss_is_numerical_failure(backend):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:.*全部非保持质量"):
        _build(backend, [np.nextafter(0.5, np.inf)], current=(0, 0, 0, 0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_old_tiny_bounded_clock_now_reports_unsupported_sampling(backend):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:.*采样表示"):
        _clock_kernel(
            backend, weight=200.0, mobility=1e-320, alpha=2e-323, max_expected_rows=1e-100,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_true_frozen_kernel_is_not_precision_failure(backend):
    result = _build(backend, [0.0], current=(0, 0, 0, 0))
    assert result.status == "no_positive_direction"
    assert result.sampling_precision is None


def test_resolver_preserves_normal_parameters_bitwise():
    stay = np.array([0.0, 0.25, 0.8, 1.0])
    mass = np.array([1.0, 0.75, 0.2, 0.0])
    move, precision = resolve_sampling_mass(stay, mass)
    np.testing.assert_array_equal(move, np.minimum(1.0, np.maximum(0.0, 1.0 - stay)))
    assert precision is None


def test_partial_resolution_loss_is_reported_without_probability_floor():
    with pytest.warns(SamplingPrecisionWarning):
        move, precision = resolve_sampling_mass(np.array([0.0, 1.0]), np.array([1.0, TINY_MASS]))
    np.testing.assert_array_equal(move, [1.0, TINY_MASS])
    assert precision.limited_segments == 1
    assert precision.max_limited_mass == TINY_MASS
    assert precision.round_index is None


def test_flat_resolution_is_recomputed_without_mutating_kernel():
    p = np.array([0.0, 1.0, 1.0, TINY_MASS])
    original = p.copy()
    offsets = np.array([0, 2, 4])
    move, _ = flat_sampling_mass(p, offsets, emit_warning=False)
    np.testing.assert_array_equal(move, [1.0, TINY_MASS])
    np.testing.assert_array_equal(p, original)
    p[-1] *= 2
    move, _ = flat_sampling_mass(p, offsets, emit_warning=False)
    assert move[-1] == 2 * TINY_MASS


@pytest.mark.parametrize("stay,mass", [([np.nan], [0.0]), ([1.0], [np.inf]), ([0.5], [0.4]), ([1.0], [-0.1])])
def test_invalid_sampling_parameters_do_not_succeed(stay, mass):
    with pytest.raises(FloatingPointError, match="numerical_indeterminate"):
        resolve_sampling_mass(np.array(stay), np.array(mass))


@pytest.mark.parametrize("backend", BACKENDS)
def test_partial_kernel_retains_progress_and_exposes_diagnostic(backend):
    scene = _partial_scene()
    with pytest.warns(SamplingPrecisionWarning):
        result = _partial_kernel(backend, scene)
    assert result.status == "ok" and result.expected_changed_rows == pytest.approx(1.0)
    assert result.sampling_precision.limited_segments == 1
    assert 0 < result.sampling_precision.max_limited_mass < TINY_MASS


@pytest.mark.parametrize("backend", ("batch", "gpu"))
@pytest.mark.parametrize("weight", (1.0, 100.0))
def test_batch_sampler_gets_positive_mass_only_when_complement_collapsed(backend, weight):
    scene = _partial_scene(weight)
    if weight == 100.0:
        with pytest.warns(SamplingPrecisionWarning):
            result = _partial_kernel(backend, scene)
    else:
        result = _partial_kernel(backend, scene)
    registry, ids, *_ = scene

    class RecordingRng:
        def __init__(self):
            self.rng = np.random.default_rng(7)
            self.p = None

        def binomial(self, n, p):
            self.p = p.copy()
            return self.rng.binomial(n, p)

        def multinomial(self, n, p):
            return self.rng.multinomial(n, p)

    rng = RecordingRng()
    sample_batch_next(result, ids, rng, registry)
    old = np.minimum(1.0, np.maximum(0.0, 1.0 - result.probabilities[result.offsets[:-1]]))
    old_rng = np.random.default_rng(7)
    old_movers = old_rng.binomial(result.grouped.counts, old)
    for g in np.flatnonzero(old_movers):
        lo, hi = int(result.offsets[g]), int(result.offsets[g + 1])
        segment = result.probabilities[lo + 1:hi]
        old_rng.multinomial(int(old_movers[g]), segment / segment.sum())
    if weight == 100.0:
        assert old[1] == 0.0 and rng.p[1] == result.probabilities[-1] > 0.0
        assert rng.p[0] == old[0]
        assert rng.rng.bit_generator.state != old_rng.bit_generator.state
    else:
        np.testing.assert_array_equal(rng.p, old)
        assert rng.rng.bit_generator.state == old_rng.bit_generator.state


@pytest.mark.parametrize("backend", BACKENDS)
def test_precision_records_survive_return_and_numerical_checkpoint(backend, tmp_path):
    _, _, _, run = _partial_run(backend)
    with pytest.warns(SamplingPrecisionWarning):
        result = run()
    assert result.sampling_precision[0].round_index == 0
    assert result.sampling_precision[0].limited_segments == 1
    registry, initial, rng, run = _partial_run(backend, fail_after_first=True)
    with pytest.warns(SamplingPrecisionWarning), pytest.raises(NumericalEvolutionError) as caught:
        with save_numerical_failure(registry, tmp_path / "table.csv"):
            run()
    partial = caught.value.partial_result
    assert len(partial.records) == 1
    assert partial.sampling_precision[0].round_index == 0
    assert not np.array_equal(partial.state_ids, initial)
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1
    meta = json.loads(manifests[0].read_text())
    assert meta["sampling_contract"] == SAMPLING_CONTRACT
    assert "sampling_precision.csv" in meta["files_sha256"]


def test_precision_records_survive_early_return_and_curve(tmp_path):
    _, _, _, run = _partial_run("batch", stop_floor=1.0)
    with pytest.warns(SamplingPrecisionWarning):
        result = run()
    assert result.stop_reason == "floor_reached"
    assert sampling_summary(result.sampling_precision)["affected_rounds"] == 1
    spec = importlib.util.spec_from_file_location("sampling_curve", ROOT / "scripts/run_plants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "curve.csv"
    module.write_curve(output, result, result.records[-1].old_loss)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["sampling_limited_segments"] == "1"
    assert rows[-1]["sampling_limited_segments"] == "1"
    assert float(rows[-1]["sampling_max_limited_mass"]) > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_precision_records_survive_later_true_freeze(backend):
    _, _, _, run = _partial_run(backend, round_limit=3)
    with pytest.warns(SamplingPrecisionWarning):
        result = run()
    assert result.stop_reason == "no_positive_direction"
    assert [record.round_index for record in result.sampling_precision] == [0]


@pytest.mark.parametrize("options,reason", [
    ({"group_index": np.array([0, 0]), "group_floors": np.array([1.0]), "stop_floor_groups": True},
     "group_floor_reached"),
    ({"stop_threshold": 1e-8, "stop_lag": 1, "max_frozen_retries": 3}, "loss_plateau"),
])
def test_precision_records_survive_other_batch_stop_paths(options, reason):
    _, _, _, run = _partial_run("batch", round_limit=4, **options)
    with pytest.warns(SamplingPrecisionWarning):
        result = run()
    assert result.stop_reason == reason
    assert [record.round_index for record in result.sampling_precision] == [0]


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_public_cli_reports_whole_resolution_loss_before_sampling(tmp_path, backend):
    if backend == "gpu":
        scene = _partial_scene()
        _gpu_context(scene[6], scene[2].target, scene[2].weights)
    seed = next(
        s for s in range(256)
        if all(np.random.default_rng(s).integers(2, size=4) == 0)
    )
    schema, queries = tmp_path / "schema.json", tmp_path / "queries.json"
    schema.write_text(json.dumps({"fields": ["v"], "domains": [["0", "1"]]}))
    queries.write_text(json.dumps({
        "record_count": 4, "result_unit": "records",
        "queries": [{"id": "v", "conditions": [{"attribute": "v", "operator": "==", "value": "1"}],
                     "result": float(np.nextafter(0.5, np.inf))}],
    }))
    output = tmp_path / "table.csv"
    command = [
        sys.executable, "-m", "resevo", "--schema", str(schema), "--queries", str(queries),
        "--rows", "4", "--rounds", "3", "--init", "uniform", "--seed", str(seed),
        "--backend", backend, "--output", str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                            text=True, capture_output=True)
    assert result.returncode != 0 and "numerical_indeterminate" in result.stderr
    assert result.stdout == "" and not output.exists()
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["completed_rounds"] == 0


def test_public_cli_declares_finite_precision_contract(tmp_path):
    output = tmp_path / "table.csv"
    result = subprocess.run([
        sys.executable, "-m", "resevo", "--schema", str(ROOT / "configs/resevo/smoke_schema.json"),
        "--queries", str(ROOT / "configs/resevo/smoke_queries.json"),
        "--rows", "64", "--rounds", "1", "--seed", "7", "--output", str(output),
    ], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["sampling_contract"] == SAMPLING_CONTRACT
    assert info["sampling_precision"]["contract"] == SAMPLING_CONTRACT
    assert info["sampling_precision"]["affected_rounds"] == 0
    assert output.exists()


@pytest.fixture(params=("ok", "unavailable"))
def probe_state(request, monkeypatch):
    if request.param == "unavailable":
        from resevo import probe

        def unavailable(*args, **kwargs):
            raise FloatingPointError("numerical_indeterminate: combined diagnostic fixture")

        monkeypatch.setattr(probe, "analytic_step", unavailable)
    return request.param


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_kernel_preserves_both_diagnostic_types(backend, probe_state, recwarn):
    scene = _partial_scene()
    plain = _partial_kernel(backend, scene)
    probed = _partial_kernel(backend, scene, probe_grid=16)
    np.testing.assert_array_equal(probed.probabilities, plain.probabilities)
    assert probed.step == plain.step and probed.expected_loss == plain.expected_loss
    assert isinstance(probed.sampling_precision, SamplingPrecision)
    assert probed.sampling_precision == plain.sampling_precision
    assert isinstance(probed.probe_diagnostics, ProbeDiagnostics)
    assert probed.probe_diagnostics.status == probe_state
    assert any(issubclass(w.category, SamplingPrecisionWarning) for w in recwarn)
    if probe_state == "unavailable":
        assert probed.probe is None
        assert any(issubclass(w.category, ProbeNumericalWarning) for w in recwarn)
    else:
        assert probed.probe is not None


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_combined_diagnostics_do_not_change_table_records_or_rng(backend, probe_state, recwarn):
    _, _, reference_rng, reference_run = _partial_run(backend)
    reference = reference_run()
    _, _, rng, run = _partial_run(backend, probe_interval=1)
    result = run()
    np.testing.assert_array_equal(result.state_ids, reference.state_ids)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert result.sampling_precision == reference.sampling_precision
    assert len(result.probes) == 1 and result.probes[0].diagnostics.status == probe_state
    for before, after in zip(reference.records, result.records):
        a, b = asdict(before), asdict(after)
        del a["elapsed"], b["elapsed"]
        assert a == b
    assert len(result.records) == len(reference.records)


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_curve_and_later_failure_keep_both_diagnostics(tmp_path, backend, probe_state, recwarn):
    _, _, _, run = _partial_run(backend, probe_interval=1)
    result = run()
    _plants().write_curve(tmp_path / "curve.csv", result, 0.0)
    with (tmp_path / "curve.csv").open(newline="") as stream:
        curve = list(csv.DictReader(stream))
    assert curve[0]["sampling_limited_segments"] == curve[-1]["sampling_limited_segments"] == "1"
    with (tmp_path / "curve_probe.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        probes = list(reader)
        columns = reader.fieldnames
    if probe_state == "unavailable":
        assert probes[0]["probe_status"] == "unavailable"
        assert probes[0]["beta_best"] == probes[0]["J_best"] == ""
        assert json.loads(probes[0]["probe_unavailable_points"])
    else:
        assert columns == ["round", "beta_used", "J_used", "beta_best", "J_best"]

    registry, initial, _, run = _partial_run(backend, probe_interval=1, fail_after_first=True)
    with pytest.raises(NumericalEvolutionError, match="after precision warning") as caught:
        with save_numerical_failure(registry, tmp_path / "table.csv"):
            run()
    partial = caught.value.partial_result
    assert len(partial.records) == len(partial.probes) == len(partial.sampling_precision) == 1
    assert not np.array_equal(partial.state_ids, initial)
    assert partial.probes[0].diagnostics.status == probe_state
    assert partial.sampling_precision[0].limited_segments == 1
    assert not (tmp_path / "table.csv").exists()
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1
    metadata = json.loads(manifests[0].read_text())
    assert metadata["sampling_contract"] == SAMPLING_CONTRACT
    assert {"probes.csv", "sampling_precision.csv"} <= metadata["files_sha256"].keys()
    for name, digest in metadata["files_sha256"].items():
        assert hashlib.sha256((manifests[0].parent / name).read_bytes()).hexdigest() == digest
    with (manifests[0].parent / "sampling_precision.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["round_index"] == "0" and saved[0]["limited_segments"] == "1"
    with (manifests[0].parent / "probes.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    if probe_state == "unavailable":
        assert saved[0]["probe_status"] == "unavailable"
        assert saved[0]["beta_best"] == saved[0]["j_best"] == ""
        assert json.loads(saved[0]["probe_unavailable_points"])


@pytest.mark.parametrize("backend", ("batch", "gpu"))
def test_whole_sampling_failure_is_not_downgraded_to_probe_warning(backend, monkeypatch, recwarn):
    from resevo import probe

    def unavailable(*args, **kwargs):
        raise FloatingPointError("numerical_indeterminate: diagnostic fixture")

    monkeypatch.setattr(probe, "analytic_step", unavailable)
    with pytest.raises(FloatingPointError, match="全部非保持质量"):
        _build(backend, [np.nextafter(0.5, np.inf)], current=(0, 0, 0, 0), probe_grid=16)
    assert not any(issubclass(w.category, ProbeNumericalWarning) for w in recwarn)
