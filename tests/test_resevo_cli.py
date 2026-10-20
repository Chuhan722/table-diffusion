"""主线入口只读显式公开输入，不从原表推断 schema 或统计量。"""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from resevo import cli, dataset

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs" / "resevo" / "smoke_schema.json"
QUERIES = ROOT / "configs" / "resevo" / "smoke_queries.json"


def _args(output):
    return [
        "--schema", str(SCHEMA), "--queries", str(QUERIES), "--rows", "64",
        "--output", str(output), "--rounds", "50", "--seed", "7",
    ]


def test_public_inputs_are_sufficient(tmp_path, monkeypatch, capsys):
    def forbidden(*_, **__):
        pytest.fail("The generator must not read a raw reference table")

    monkeypatch.setattr(dataset, "load_table", forbidden)
    output = tmp_path / "nested" / "table.csv"
    cli.main(_args(output))
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["rows"] == 64
    assert result["stop_reason"] in ("round_limit", "no_positive_direction")
    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["a", "b"] and len(rows) == 65
    assert all(len(row) == 2 and set(row) <= {"0", "1"} for row in rows[1:])
    assert Counter(map(tuple, rows[1:])) == {("0", "0"): 24, ("0", "1"): 8, ("1", "0"): 8, ("1", "1"): 24}


def test_same_seed_same_output(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.main(_args(a))
    cli.main(_args(b))
    assert a.read_bytes() == b.read_bytes()


def test_existing_output_is_not_overwritten(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("keep me")
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(path))
    assert exc.value.code == 2
    assert path.read_text() == "keep me"


@pytest.mark.parametrize("option,value", [
    ("--damping", "3"), ("--damping", "nan"), ("--max-expected-rows", "-1"),
    ("--max-expected-rows", "inf"), ("--rows", "0"), ("--rounds", "0"),
    ("--alpha", "nan"), ("--stay", "inf"), ("--work-below", "nan"),
])
def test_invalid_arguments_fail_before_input_or_generation(tmp_path, monkeypatch, option, value):
    def forbidden(*_, **__):
        pytest.fail("Input loading or generation happened before argument validation")

    monkeypatch.setattr(cli, "load_public_schema", forbidden)
    monkeypatch.setattr(cli, "evolve_batch", forbidden)
    output = tmp_path / "table.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(output) + [f"{option}={value}"])
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("payload", [
    [], {"fields": [], "domains": []},
    {"fields": ["a", "a"], "domains": [["0"], ["1"]]},
    {"fields": ["a"], "domains": []},
    {"fields": ["a"], "domains": [[]]},
    {"fields": ["a"], "domains": [["0", "0"]]},
    {"fields": ["a"], "domains": [[0, 1]]},
])
def test_invalid_schema_is_explicitly_rejected(tmp_path, payload):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        cli.load_public_schema(path)


@pytest.mark.parametrize("count", [None, True, 64.0, 63])
def test_query_row_count_must_match_public_count(tmp_path, count):
    payload = json.loads(QUERIES.read_text())
    payload["record_count"] = count
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="record_count"):
        dataset.load_queries(str(path), expected_record_count=64)
    assert len(dataset.load_queries(str(path))) == 4


def test_nonfinite_targets_do_not_generate(tmp_path, monkeypatch):
    payload = json.loads(QUERIES.read_text())
    payload["queries"][0]["result"] = float("nan")
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload))
    output = tmp_path / "out.csv"

    def forbidden(*_, **__):
        pytest.fail("Nonfinite targets reached generation")

    monkeypatch.setattr(cli, "evolve_batch", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(output) + ["--queries", str(path)])
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("payload", [[], None, "queries", 64, True])
def test_non_object_query_file_is_rejected(tmp_path, payload):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="JSON"):
        dataset.load_queries(str(path), expected_record_count=64)


def _partial_queries(tmp_path):
    payload = json.loads(QUERIES.read_text())
    payload["queries"] = payload["queries"][:1]
    payload["query_count"] = 1
    path = tmp_path / "partial_queries.json"
    path.write_text(json.dumps(payload))
    return path


def test_incomplete_marginals_do_not_silently_fallback(tmp_path, capsys):
    queries = _partial_queries(tmp_path)
    output = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(output) + ["--queries", str(queries)])
    assert exc.value.code == 2
    assert "--init uniform" in capsys.readouterr().err
    assert not output.exists()


def test_explicit_uniform_initialization_supports_partial_queries(tmp_path, monkeypatch):
    queries = _partial_queries(tmp_path)
    output = tmp_path / "out.csv"

    def forbidden(*_, **__):
        pytest.fail("Explicit uniform initialization must not infer marginals")

    monkeypatch.setattr(cli, "derive_first_order", forbidden)
    cli.main(_args(output) + ["--queries", str(queries), "--init", "uniform"])
    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 65
