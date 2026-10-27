"""数值错误保留失败前的全表/曲线，仍抛错，禁止把诊断快照当正常产物。"""
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

from resevo import batchkernel, engine, grouping
from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import target_from_specs
from resevo.editspace import make_edit_provider, tuples_from_ids
from resevo.engine import EvolveResult, NumericalEvolutionError, ProbeRecord, RoundRecord, evolve
from resevo.failure import save_numerical_failure, write_numerical_failure
from resevo.grouping import evolve_grouped, make_grouped_provider
from test_batchkernel import _scene

ROOT = Path(__file__).resolve().parents[1]


def _prepare(backend, frozen=False):
    registry, ids, weights, _ = _scene(3, num_rows=12)
    if backend == "gpu":
        pytest.importorskip("cupy")
        from test_local_gain_freeze import _gpu_context, _inputs
        wl, _, _, _, _, qs, _ = _inputs([1.0])
        _gpu_context(qs, wl.target, wl.weights)
        registry.set_lazy_features(True)
    target = (
        registry.answers_for_rows(tuples_from_ids(registry, ids))
        if frozen else target_from_specs(registry.specs)
    )
    menu_rng, sample_rng = np.random.default_rng(5), np.random.default_rng(9)
    if backend == "rows":
        provider = make_edit_provider(registry, target, weights, menu_rng)

        def run(rounds, supplied=None):
            return evolve(
                None, ids, rounds, sample_rng, support_provider=provider if supplied is None else supplied,
                max_frozen_retries=100,
            )
    elif backend == "groups":
        provider = make_grouped_provider(registry, target, weights, menu_rng, pairing=False)

        def run(rounds, supplied=None):
            return evolve_grouped(
                ids, rounds, sample_rng, provider if supplied is None else supplied, max_frozen_retries=100,
            )
    else:
        provider = make_batch_provider(
            registry, target, weights, menu_rng, pairing=False, defer_workload=backend == "gpu",
        )

        def run(rounds, supplied=None):
            return evolve_batch(
                ids, rounds, sample_rng, provider if supplied is None else supplied, registry,
                backend="gpu" if backend == "gpu" else "cpu", max_frozen_retries=100,
            )
    return registry, ids, sample_rng, run, provider


@pytest.mark.parametrize("backend", ("rows", "groups", "batch", "gpu"))
@pytest.mark.parametrize("frozen", (False, True))
def test_failure_checkpoint_is_last_completed_prefix(monkeypatch, backend, frozen):
    reference_registry, _, reference_rng, reference_run, _ = _prepare(backend, frozen)
    reference = reference_run(2)
    registry, original_ids, rng, run, _ = _prepare(backend, frozen)
    if backend == "rows":
        owner, name = engine, "build_kernel"
    elif backend == "groups":
        owner, name = grouping, "build_group_kernel"
    elif backend == "batch":
        owner, name = batchkernel, "build_batch_kernel"
    else:
        from resevo.gpukernel import GpuBatchContext
        owner, name = GpuBatchContext, "build"
    original = getattr(owner, name)
    calls = 0

    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise FloatingPointError("numerical_indeterminate: injected")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail_third)
    with pytest.raises(NumericalEvolutionError) as error:
        run(5)
    partial = error.value.partial_result
    assert partial.stop_reason == "numerical_indeterminate" and len(partial.records) == 2
    assert tuples_from_ids(registry, partial.state_ids) == tuples_from_ids(reference_registry, reference.state_ids)
    assert [r.old_loss for r in partial.records] == [r.old_loss for r in reference.records]
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert not np.shares_memory(partial.state_ids, original_ids)
    if frozen:
        assert all(r.status == "no_positive_direction" for r in partial.records)
    else:
        assert tuples_from_ids(registry, partial.state_ids) != tuples_from_ids(registry, original_ids)


@pytest.mark.parametrize("backend", ("rows", "groups", "batch"))
def test_provider_numerical_error_also_carries_current_state(backend):
    _, ids, _, run, _ = _prepare(backend)

    def broken(*args):
        raise FloatingPointError("numerical_indeterminate: provider")

    with pytest.raises(NumericalEvolutionError) as error:
        run(5, broken)
    np.testing.assert_array_equal(error.value.partial_result.state_ids, ids)
    assert error.value.partial_result.records == []


@pytest.mark.parametrize("mode", ("rescue", "tree"))
def test_subset_failure_snapshot_uses_global_not_small_registry_ids(monkeypatch, mode):
    from dataclasses import replace

    registry, ids, weights, _ = _scene(7, num_rows=30)
    provider = make_batch_provider(
        registry, target_from_specs(registry.specs), weights, np.random.default_rng(11),
        pairing=True, defer_workload=True, pair_rescue_rows=8, rescue_after=1,
    )
    plan = replace(provider(ids, 0, 1), mode=mode)
    assert len(plan.rescue_ids) == 8

    def broken(*args, **kwargs):
        raise FloatingPointError("numerical_indeterminate: subset")

    monkeypatch.setattr(batchkernel, "build_kernel", broken)
    with pytest.raises(NumericalEvolutionError) as error:
        evolve_batch(ids, 1, np.random.default_rng(9), lambda *args: plan, registry)
    np.testing.assert_array_equal(error.value.partial_result.state_ids, ids)
    assert len(error.value.partial_result.state_ids) == 30


@pytest.mark.parametrize("failure", (FloatingPointError("unrelated"), ValueError("numerical_indeterminate: other")))
def test_unrelated_errors_are_not_reclassified(failure):
    _, _, _, run, _ = _prepare("batch")

    def broken(*args):
        raise failure

    with pytest.raises(type(failure)) as error:
        run(2, broken)
    assert error.value is failure


def _failure():
    registry, ids, _, _ = _scene(4, num_rows=4)
    result = EvolveResult(
        ids, [RoundRecord(0, 2.0, 1.0, 0.5, 0.25, 0.5, 1.8125, "ok")],
        "numerical_indeterminate", [ProbeRecord(0, 1.0, 0.2, 1.1, 0.25)],
        [(0.25, 0.5)], [(0, np.array([0.5, 1.5]))],
    )
    return registry, NumericalEvolutionError("numerical_indeterminate: fixture", result)


def test_failure_files_are_distinct_complete_and_not_normal_outputs(tmp_path):
    registry, error = _failure()
    output = tmp_path / "table.csv"
    output.write_text("previous output")
    a = write_numerical_failure(registry, error, output, stage="stage2")
    b = write_numerical_failure(registry, error, output, stage="stage2")
    assert a != b and output.read_text() == "previous output"
    manifest = json.loads((a / "failure.json").read_text())
    assert manifest["status"] == "failed" and manifest["stop_reason"] == "numerical_indeterminate"
    assert manifest["stage"] == "stage2" and manifest["completed_rounds"] == 1
    assert manifest["normal_output_published"] is False and manifest["resumable"] is False
    for filename, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((a / filename).read_bytes()).hexdigest() == digest
    with (a / "current.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert list(map(tuple, rows[1:])) == tuples_from_ids(registry, error.partial_result.state_ids)
    with (a / "records.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1 and records[0]["status"] == "ok"
    assert set(manifest["files_sha256"]) == {
        "current.csv", "records.csv", "probes.csv", "temperatures.csv", "group_losses.csv",
    }


def test_checkpoint_write_failure_remains_an_error_without_complete_manifest(tmp_path, monkeypatch, capsys):
    registry, error = _failure()
    original = Path.open

    def blocked(path, *args, **kwargs):
        if path.name == "records.csv":
            raise PermissionError("injected write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", blocked)
    with pytest.raises(PermissionError) as raised:
        with save_numerical_failure(registry, tmp_path / "table.csv"):
            raise error
    assert raised.value.__context__ is error
    assert not (tmp_path / "table.csv").exists()
    assert not list(tmp_path.glob("*.failed-*/failure.json"))
    assert "已保存" not in capsys.readouterr().err


def _tiny_inputs(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"fields": ["v"], "domains": [["0", "1"]]}))
    payload = {
        "record_count": 2, "result_unit": "records", "query_count": 1,
        "queries": [{
            "id": "tiny", "conditions": [{"attribute": "v", "operator": "==", "value": "1"}],
            "result": 0.500001,
        }],
    }
    queries = tmp_path / "measured_tinyquery.json"
    queries.write_text(json.dumps(payload))
    return schema, queries


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_public_cli_real_underflow_saves_checkpoint_and_exits_nonzero(tmp_path, backend):
    if backend == "gpu":
        _prepare("gpu")
    schema, queries = _tiny_inputs(tmp_path)
    output = tmp_path / "table.csv"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([
        sys.executable, "-m", "resevo", "--schema", str(schema), "--queries", str(queries),
        "--rows", "2", "--init", "uniform", "--seed", "11", "--alpha", "5e-324",
        "--rounds", "3", "--backend", backend, "--output", str(output),
    ], env=env, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "numerical_indeterminate" in result.stderr
    assert not output.exists() and result.stdout == ""
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["rows"] == 2 and manifest["completed_rounds"] == 0
    with (manifests[0].parent / "current.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [["v"], ["0"], ["0"]]


@pytest.mark.parametrize("mode", ([], ["--grouped"], ["--batched"]))
def test_legacy_driver_real_underflow_also_saves_failure(tmp_path, monkeypatch, mode):
    spec = importlib.util.spec_from_file_location("run_plants_failure", ROOT / "scripts" / "run_plants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "data" / "toy"
    directory.mkdir(parents=True)
    _tiny_inputs(directory)
    (directory / "toy.csv").write_text("v\n0\n1\n")
    output = tmp_path / "table.csv"
    monkeypatch.setattr(module, "DATA_ROOT", directory.parent)
    monkeypatch.setattr(sys, "argv", [
        "run_plants.py", "--data", "toy", "--uniform-init", "--init-seed", "11",
        "--alpha", "5e-324", "--rounds", "3", "--save-table", str(output), *mode,
    ])
    with pytest.raises(NumericalEvolutionError):
        module.main()
    assert not output.exists()
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1 and json.loads(manifests[0].read_text())["stage"] == "stage1"
