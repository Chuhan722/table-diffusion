"""公开入口硬后处理：默认关闭、第一段隔离、真实交表与失败边界。"""
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
from itertools import product
import json

import numpy as np
import pytest

from resevo import cli, dataset


def _inputs(tmp_path, *, queries=None, rows=32):
    schema = {"fields": ["a", "b", "c"], "domains": [["0", "1"]] * 3}
    specs = []
    for pair in ((0, 1), (1, 2)):
        for values in product(range(2), repeat=2):
            specs.append({
                "id": f"q{len(specs)}",
                "conditions": [
                    {"attribute": schema["fields"][j], "operator": "==", "value": str(v)}
                    for j, v in zip(pair, values)
                ],
                "result": 8.25,
            })
    schema_path, query_path = tmp_path / "schema.json", tmp_path / "queries.json"
    schema_path.write_text(json.dumps(schema))
    query_path.write_text(json.dumps({
        "result_unit": "records", "record_count": rows,
        "queries": specs if queries is None else queries,
    }))
    return [
        "--schema", str(schema_path), "--queries", str(query_path),
        "--rows", str(rows), "--rounds", "40", "--seed", "11", "--init", "uniform",
    ]


def _rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["a", "b", "c"]
    return rows[1:]


def _margin(rows, fields):
    return Counter(tuple(row[j] for j in fields) for row in rows)


def _summary(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_absent_and_explicit_zero_leave_table_output_and_rng_untouched(tmp_path, monkeypatch, capsys):
    args = _inputs(tmp_path)
    original_rng = np.random.default_rng
    created = []

    def tracked(seed=None):
        rng = original_rng(seed)
        created.append((seed, rng))
        return rng

    def forbidden(*_, **__):
        pytest.fail("A disabled postprocessor was invoked")

    monkeypatch.setattr(cli.np.random, "default_rng", tracked)
    monkeypatch.setattr(cli, "hard_fiber_sample", forbidden)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.main(args + ["--output", str(a)])
    first = _summary(capsys)
    first_rng = [(seed, deepcopy(rng.bit_generator.state)) for seed, rng in created]
    created.clear()
    cli.main(args + ["--output", str(b), "--hard-fiber-steps", "0"])
    second = _summary(capsys)
    second_rng = [(seed, deepcopy(rng.bit_generator.state)) for seed, rng in created]
    assert a.read_bytes() == b.read_bytes() and first_rng == second_rng
    assert [seed for seed, _ in first_rng] == [11, 100011, 200011, 300011]
    first.pop("output")
    second.pop("output")
    assert first == second and "hard_fiber" not in second


def test_hard_rng_is_separate_and_first_stage_trajectory_unchanged(tmp_path, monkeypatch, capsys):
    args = _inputs(tmp_path)
    original_rng, original_evolve = np.random.default_rng, cli.evolve_batch
    created, captured = [], []

    def tracked(seed=None):
        rng = original_rng(seed)
        created.append((seed, rng))
        return rng

    def evolve(*args, **kwargs):
        result = original_evolve(*args, **kwargs)
        records = []
        for record in result.records:
            row = asdict(record)
            row.pop("elapsed")
            records.append(row)
        captured.append({
            "ids": result.state_ids.tolist(), "records": records, "reason": result.stop_reason,
            "sampling": [asdict(item) for item in result.sampling_precision or []],
        })
        return result

    monkeypatch.setattr(cli.np.random, "default_rng", tracked)
    monkeypatch.setattr(cli, "evolve_batch", evolve)
    snapshots = []
    for name, extra in (
        ("off", []),
        ("on", ["--hard-fiber-steps", "200", "--hard-fiber-seed", "7301"]),
        ("other", ["--hard-fiber-steps", "200", "--hard-fiber-seed", "7302"]),
    ):
        created.clear()
        cli.main(args + ["--output", str(tmp_path / f"{name}.csv"), *extra])
        info = _summary(capsys)
        snapshots.append([(seed, deepcopy(rng.bit_generator.state)) for seed, rng in created[:4]])
        assert len(created) == (4 if name == "off" else 5)
        if name != "off":
            assert info["hard_fiber"]["seed"] == int(extra[-1])
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert captured[0] == captured[1] == captured[2]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_public_eq_inputs_preserve_counts_after_cpu_or_gpu_generation(tmp_path, monkeypatch, capsys, backend):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        if count < 1:
            pytest.skip("No CUDA device")
    args = _inputs(tmp_path) + ["--backend", backend]

    def forbidden(*_, **__):
        pytest.fail("The public generator read a raw reference table")

    monkeypatch.setattr(dataset, "load_table", forbidden)
    initial, final = tmp_path / "base.csv", tmp_path / "hard.csv"
    cli.main(args + ["--output", str(initial)])
    baseline = _summary(capsys)
    cli.main(args + ["--output", str(final), "--hard-fiber-steps", "1000"])
    result = _summary(capsys)
    before, after = _rows(initial), _rows(final)
    assert len(before) == len(after) == 32
    for fields in ((0,), (1,), (2,), (0, 1), (1, 2)):
        assert _margin(before, fields) == _margin(after, fields)
    assert all(len(row) == 3 and set(row) <= {"0", "1"} for row in after)
    assert result["backend"] == backend and result["stop_reason"] == baseline["stop_reason"]
    assert result["rounds"] == baseline["rounds"] and result["sampling_precision"] == baseline["sampling_precision"]
    hard = result["hard_fiber"]
    assert hard["backend"] == "cpu" and hard["seed"] == 400011 and hard["executed_steps"] == 1000
    assert hard["stop_reason"] == "step_limit" and hard["invariants_verified"] and not hard["mixing_certified"]
    assert all(count != 8.25 for count in _margin(after, (0, 1)).values())


def test_same_seed_repeats_actual_hard_output(tmp_path, capsys):
    args = _inputs(tmp_path) + ["--hard-fiber-steps", "500", "--hard-fiber-seed", "7301"]
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.main(args + ["--output", str(a)])
    first = _summary(capsys)
    cli.main(args + ["--output", str(b)])
    second = _summary(capsys)
    assert a.read_bytes() == b.read_bytes() and first["hard_fiber"] == second["hard_fiber"]


@pytest.mark.parametrize("extra", [
    ["--hard-fiber-steps", "-1"],
    ["--hard-fiber-steps", "nan"],
    ["--hard-fiber-steps", "1.5"],
    ["--hard-fiber-seed", "0"],
    ["--hard-fiber-steps", "1", "--hard-fiber-seed", "-1"],
])
def test_invalid_options_fail_before_inputs_or_generation(tmp_path, monkeypatch, extra):
    args = _inputs(tmp_path)

    def forbidden(*_, **__):
        pytest.fail("Invalid hard options reached input loading or generation")

    monkeypatch.setattr(cli, "load_public_schema", forbidden)
    monkeypatch.setattr(cli, "evolve_batch", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(args + ["--output", str(tmp_path / "out.csv"), *extra])
    assert exc.value.code == 2 and not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize("condition", [
    {"attribute": "a", "operator": "between", "lower": 0, "upper": 1},
    {"attribute": "a", "operator": ">=", "value": 1},
    {"operator": "halfspace", "scores": {"a": {"0": 0, "1": 1}}, "threshold": 1},
])
def test_non_equality_hard_requests_fail_before_rng_or_generation(tmp_path, monkeypatch, condition):
    args = _inputs(tmp_path, queries=[{"id": "q", "conditions": [condition], "result": 10}])

    def forbidden(*_, **__):
        pytest.fail("Unsupported hard queries reached random initialization or generation")

    monkeypatch.setattr(cli.np.random, "default_rng", forbidden)
    monkeypatch.setattr(cli, "evolve_batch", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(args + ["--output", str(tmp_path / "out.csv"), "--hard-fiber-steps", "20"])
    assert exc.value.code == 2 and not (tmp_path / "out.csv").exists()


def test_query_prefix_does_not_remove_a_hard_constraint(tmp_path, capsys):
    args = _inputs(tmp_path, queries=[{
        "id": "P2_user_query",
        "conditions": [{"attribute": "a", "operator": "==", "value": "0"},
                       {"attribute": "b", "operator": "==", "value": "0"}],
        "result": 8.25,
    }])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.main(args + ["--output", str(a)])
    _summary(capsys)
    cli.main(args + ["--output", str(b), "--hard-fiber-steps", "500"])
    info = _summary(capsys)
    assert [0, 1] in info["hard_fiber"]["preserved_sets"]
    assert _margin(_rows(a), (0, 1)) == _margin(_rows(b), (0, 1))


def test_constant_mixed_with_equalities_does_not_create_empty_marginal(tmp_path, capsys):
    args = _inputs(tmp_path, queries=[
        {"id": "constant", "conditions": [], "result": 32},
        {"id": "cell", "conditions": [
            {"attribute": field, "operator": "==", "value": "0"} for field in ("a", "b", "c")
        ], "result": 4.25},
    ])
    output = tmp_path / "out.csv"
    cli.main(args + ["--output", str(output), "--hard-fiber-steps", "20"])
    info = _summary(capsys)
    assert info["hard_fiber"]["preserved_sets"] == [[0], [0, 1, 2], [1], [2]]
    assert len(_rows(output)) == 32


def test_constant_only_hard_request_is_explicitly_rejected_before_generation(tmp_path, monkeypatch):
    args = _inputs(tmp_path, queries=[{"id": "constant", "conditions": [], "result": 32}])

    def forbidden(*_, **__):
        pytest.fail("Constant-only input reached the existing unsupported first-stage path")

    monkeypatch.setattr(cli.np.random, "default_rng", forbidden)
    monkeypatch.setattr(cli, "evolve_batch", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(args + ["--output", str(tmp_path / "out.csv"), "--hard-fiber-steps", "20"])
    assert exc.value.code == 2 and not (tmp_path / "out.csv").exists()


def test_one_row_is_explicitly_reported(tmp_path, capsys):
    args = _inputs(tmp_path, rows=1, queries=[{
        "id": "q", "conditions": [{"attribute": "a", "operator": "==", "value": "0"}], "result": 0.25,
    }])
    output = tmp_path / "out.csv"
    cli.main(args + ["--output", str(output), "--hard-fiber-steps", "20"])
    info = _summary(capsys)
    assert info["hard_fiber"]["stop_reason"] == "no_row_pairs"
    assert info["hard_fiber"]["executed_steps"] == 0 and len(_rows(output)) == 1


def test_postprocessing_error_does_not_publish_stage_one_as_success(tmp_path, monkeypatch, capsys):
    args = _inputs(tmp_path)
    output = tmp_path / "not_created" / "out.csv"

    def failure(*_, **__):
        raise RuntimeError("injected postprocessing failure")

    monkeypatch.setattr(cli, "hard_fiber_sample", failure)
    with pytest.raises(RuntimeError, match="injected"):
        cli.main(args + ["--output", str(output), "--hard-fiber-steps", "5"])
    assert not output.exists() and not output.parent.exists()
    assert '"output":' not in capsys.readouterr().out


def test_existing_output_remains_read_only_with_hard_option(tmp_path):
    args = _inputs(tmp_path)
    output = tmp_path / "existing.csv"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit) as exc:
        cli.main(args + ["--output", str(output), "--hard-fiber-steps", "5"])
    assert exc.value.code == 2 and output.read_bytes() == b"keep"
