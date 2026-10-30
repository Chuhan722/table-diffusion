"""只读记录工具不反馈给生成；输入/输出合同和未观察状态明确。"""
import csv
import importlib.util
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pytest

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.cli import load_public_schema
from resevo.dataset import StateRegistry, load_queries, target_from_specs
from resevo.editspace import tuples_from_ids

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("quadratic_recorder_test", ROOT / "scripts/record_quadratic_monitor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, learning_sigma=10.0):
    schema = {"fields": [f"x{j}" for j in range(4)], "domains": [["0", "1"]] * 4}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    rows = [(0, 0, 0, 0)] * 8 + [(1, 1, 1, 1)] * 8 + list(product(range(2), repeat=4))
    queries = []
    for a, b in combinations(range(4), 2):
        for va, vb in product(range(2), repeat=2):
            queries.append({
                "id": f"Q{len(queries):03d}",
                "conditions": [
                    {"attribute": f"x{a}", "operator": "==", "value": str(va)},
                    {"attribute": f"x{b}", "operator": "==", "value": str(vb)},
                ],
                "result": float(sum(row[a] == va and row[b] == vb for row in rows)),
                "sigma": learning_sigma,
            })
    payload = {"record_count": len(rows), "result_unit": "records", "queries": queries}
    learning = tmp_path / "learning.json"
    learning.write_text(json.dumps(payload))
    validation = json.loads(json.dumps(payload))
    rng = np.random.default_rng(99)
    for q in validation["queries"]:
        q["result"] += float(rng.normal(0, 2))
        q["sigma"] = 2.0
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation))
    return schema_path, learning, validation_path


def _argv(paths, output, mode, backend="cpu", rounds=8):
    schema, learning, validation = paths
    argv = [
        "--schema", str(schema), "--learning", str(learning), "--rows", "32",
        "--output", str(output), "--monitor", mode, "--backend", backend,
        "--rounds", str(rounds), "--interval", "1", "--seed", "7",
    ]
    if mode != "off":
        argv += ["--validation", str(validation)]
    if mode in ("l2", "both"):
        argv += ["--error-probability", "0.05"]
    return argv


def _without_elapsed(records):
    return [{key: value for key, value in row.items() if key != "elapsed"} for row in records]


def _gpu_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_recording_is_read_only_against_bare_generation(tmp_path, backend):
    if backend == "gpu":
        _gpu_or_skip()
    paths = _inputs(tmp_path)
    module = _script()
    off, on = tmp_path / "off", tmp_path / "on"
    module.main(_argv(paths, off, "off", backend))
    module.main(_argv(paths, on, "both", backend))
    base = json.loads((off / "summary.json").read_text())
    observed = json.loads((on / "summary.json").read_text())
    assert (off / "terminal.csv").read_bytes() == (on / "terminal.csv").read_bytes()
    assert (off / "initial.csv").read_bytes() == (on / "initial.csv").read_bytes()
    for key in ("state_ids", "rng", "stop_reason", "rounds", "sampling_precision"):
        assert observed[key] == base[key]
    assert _without_elapsed(observed["generation_records"]) == _without_elapsed(base["generation_records"])
    assert observed["record_only"] and observed["reference_round"] == 0
    assert observed["monitor_status"] == "observed"
    events = [json.loads(line) for line in (on / "observations.jsonl").read_text().splitlines()]
    assert events[0]["at_reference"] and events[0]["l2"] is None
    assert observed["l2_looks"] == len(events) - 1
    assert [row["l2"]["look"] for row in events[1:]] == list(range(1, len(events)))

    schema = load_public_schema(paths[0])
    specs = load_queries(str(paths[1]), expected_record_count=32)
    target, weights = target_from_specs(specs), np.ones(len(specs))
    init_rng = np.random.default_rng(7)
    rows = [
        tuple(domain[int(init_rng.integers(len(domain)))] for domain in schema.domains)
        for _ in range(32)
    ]
    registry = StateRegistry(schema, specs)
    registry.set_lazy_features(backend == "gpu")
    ids = registry.register_table(rows)
    menu_rng, select_rng, sample_rng = (
        np.random.default_rng(seed) for seed in (100007, 200007, 300007)
    )
    provider = make_batch_provider(
        registry, target, weights, menu_rng, pairing=False,
        defer_workload=backend == "gpu", select_rng=select_rng,
    )
    bare = evolve_batch(ids, 8, sample_rng, provider, registry, max_frozen_retries=6, backend=backend)
    assert observed["state_ids"] == bare.state_ids.tolist()
    assert observed["rng"] == {
        "initial": init_rng.bit_generator.state, "menu": menu_rng.bit_generator.state,
        "selection": select_rng.bit_generator.state, "sample": sample_rng.bit_generator.state,
    }
    assert _without_elapsed(observed["generation_records"]) == _without_elapsed(module._records(bare))
    with (on / "terminal.csv").open(newline="") as stream:
        table = list(csv.reader(stream))
    assert table[0] == list(schema.fields)
    assert [tuple(row) for row in table[1:]] == tuples_from_ids(registry, bare.state_ids)


@pytest.mark.parametrize("mode", ("l1", "l2"))
def test_each_route_can_be_recorded_independently(tmp_path, mode):
    paths = _inputs(tmp_path)
    output = tmp_path / mode
    _script().main(_argv(paths, output, mode))
    events = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert any(row[mode] is not None for row in events[1:])
    absent = "l2" if mode == "l1" else "l1"
    assert all(row[absent] is None for row in events)


def test_missing_reference_is_not_reported_as_no_alarm_evidence(tmp_path):
    paths = _inputs(tmp_path, learning_sigma=0.0)
    output = tmp_path / "no_reference"
    _script().main(_argv(paths, output, "both", rounds=1))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["monitor_status"] == "reference_not_reached"
    assert summary["reference_round"] is None and summary["l2_looks"] == 0
    assert not (output / "reference.csv").exists()


def test_reference_only_status_when_no_comparison_was_possible(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "reference_only"
    _script().main(_argv(paths, output, "both", rounds=1))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["monitor_status"] == "reference_only" and summary["l2_looks"] == 0


@pytest.mark.parametrize("mutation", ("rows", "identity", "missing_sigma", "negative_sigma", "nonfinite_target"))
def test_invalid_validation_rejected_before_output(tmp_path, mutation):
    paths = _inputs(tmp_path)
    payload = json.loads(paths[2].read_text())
    if mutation == "rows":
        payload["record_count"] = 31
    elif mutation == "identity":
        payload["queries"][0]["id"] = "wrong"
    elif mutation == "missing_sigma":
        del payload["queries"][0]["sigma"]
    elif mutation == "negative_sigma":
        payload["queries"][0]["sigma"] = -1
    else:
        payload["queries"][0]["result"] = float("nan")
    paths[2].write_text(json.dumps(payload))
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        _script().main(_argv(paths, output, "both"))
    assert exc.value.code != 0 and not output.exists()


def test_output_directory_is_not_reused(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "marker").write_text("retain")
    with pytest.raises(SystemExit):
        _script().main(_argv(paths, output, "both"))
    assert sorted(p.name for p in output.iterdir()) == ["marker"]
    assert (output / "marker").read_text() == "retain"


def test_off_mode_does_not_read_validation(tmp_path):
    paths = _inputs(tmp_path)
    paths[2].unlink()
    output = tmp_path / "off"
    _script().main(_argv(paths, output, "off"))
    assert json.loads((output / "summary.json").read_text())["monitor_status"] == "off"


def test_alarm_is_diagnostic_and_does_not_replace_terminal_table(tmp_path):
    paths = _inputs(tmp_path)
    module = _script()
    off, on = tmp_path / "off", tmp_path / "alarms"
    module.main(_argv(paths, off, "off"))
    schema = load_public_schema(paths[0])
    specs = load_queries(str(paths[1]), expected_record_count=32)
    with (off / "initial.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    answers = StateRegistry(schema, specs).answers_for_rows([tuple(row) for row in rows])
    validation = json.loads(paths[2].read_text())
    for query, answer in zip(validation["queries"], answers):
        query["result"], query["sigma"] = float(answer), 0.0
    paths[2].write_text(json.dumps(validation))
    module.main(_argv(paths, on, "both"))
    summary = json.loads((on / "summary.json").read_text())
    assert summary["first_l2_alarm"] is not None
    assert (on / "l2_first_alarm.csv").exists()
    assert (on / "terminal.csv").read_bytes() == (off / "terminal.csv").read_bytes()
    assert summary["rounds"] == json.loads((off / "summary.json").read_text())["rounds"]


def test_l1_finite_large_mean_does_not_overflow(tmp_path):
    paths = _inputs(tmp_path)
    validation = json.loads(paths[2].read_text())
    for query in validation["queries"]:
        query["result"] = 1e308
    paths[2].write_text(json.dumps(validation))
    output = tmp_path / "large_l1"
    _script().main(_argv(paths, output, "l1"))
    events = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert all(row["l1"]["mean_absolute_error"] == 1e308 for row in events)


def test_numerical_monitor_failure_is_not_a_completed_output(tmp_path):
    paths = _inputs(tmp_path)
    validation = json.loads(paths[2].read_text())
    for query in validation["queries"]:
        query["sigma"] = 1e308
    paths[2].write_text(json.dumps(validation))
    output = tmp_path / "numeric_failure"
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:"):
        _script().main(_argv(paths, output, "l2"))
    assert not (output / "terminal.csv").exists() and not (output / "summary.json").exists()
    failures = list(output.glob("*.failed-*"))
    assert len(failures) == 1
    assert json.loads((failures[0] / "failure.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("extra", [
    ["--rows", "0"], ["--rounds", "0"], ["--interval", "0"], ["--seed", "-1"],
    ["--error-probability", "0"], ["--l1-runs", "0"], ["--l1-z", "nan"],
])
def test_invalid_parameters_are_explicit(tmp_path, extra):
    paths = _inputs(tmp_path)
    output = tmp_path / "invalid_parameters"
    with pytest.raises(SystemExit):
        _script().main(_argv(paths, output, "both") + extra)
    assert not output.exists()
