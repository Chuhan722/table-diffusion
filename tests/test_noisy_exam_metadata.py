"""制卷元数据须在预算计算、随机生成和新建产物之前符合公开入口合同。"""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_noisy_exam_artifacts import REFERENCE, _run, _setup
from resevo.dataset import load_queries

MISSING = object()
BAD_METADATA = [
    pytest.param("record_count", MISSING, id="count-missing"),
    pytest.param("record_count", None, id="count-null"),
    pytest.param("record_count", True, id="count-true"),
    pytest.param("record_count", False, id="count-false"),
    pytest.param("record_count", 0, id="count-zero"),
    pytest.param("record_count", -1, id="count-negative"),
    pytest.param("record_count", 8.0, id="count-integral-float"),
    pytest.param("record_count", 8.5, id="count-fractional-float"),
    pytest.param("record_count", "8", id="count-string"),
    pytest.param("record_count", float("nan"), id="count-nan"),
    pytest.param("record_count", float("inf"), id="count-inf"),
    pytest.param("record_count", -float("inf"), id="count-negative-inf"),
    pytest.param("result_unit", MISSING, id="unit-missing"),
    pytest.param("result_unit", None, id="unit-null"),
    pytest.param("result_unit", "frequency", id="unit-frequency"),
    pytest.param("result_unit", "Records", id="unit-case"),
    pytest.param("result_unit", "", id="unit-empty"),
    pytest.param("result_unit", True, id="unit-boolean"),
    pytest.param("result_unit", ["records"], id="unit-list"),
]


def _set(payload, field, value):
    if value is MISSING:
        payload.pop(field)
    else:
        payload[field] = value


def _forbidden(*args, **kwargs):
    pytest.fail("Invalid metadata reached budget computation, random generation or output creation")


@pytest.mark.parametrize("field,value", BAD_METADATA)
def test_invalid_metadata_fails_before_computation_and_output(tmp_path, monkeypatch, capsys, field, value):
    module, source, query, payload = _setup(tmp_path)
    _set(payload, field, value)
    query.write_text(json.dumps(payload))
    before = query.read_bytes()
    destination = tmp_path / "data/toy_eps10_s7"
    monkeypatch.setattr(module, "cdp_rho", _forbidden)
    monkeypatch.setattr(module.np.random, "default_rng", _forbidden)
    monkeypatch.setattr(Path, "mkdir", _forbidden)
    with pytest.raises(SystemExit, match=field):
        _run(module, monkeypatch)
    assert query.read_bytes() == before and not destination.exists()
    assert not (source / "toy.csv").exists()
    assert "已写出" not in capsys.readouterr().out


@pytest.mark.parametrize("payload", [[], None, "exam", 8, True])
def test_non_object_metadata_is_an_explicit_input_error(tmp_path, monkeypatch, payload):
    module, _, query, _ = _setup(tmp_path)
    query.write_text(json.dumps(payload))
    before = query.read_bytes()
    monkeypatch.setattr(module, "cdp_rho", _forbidden)
    monkeypatch.setattr(module.np.random, "default_rng", _forbidden)
    with pytest.raises(SystemExit, match="JSON"):
        _run(module, monkeypatch)
    assert query.read_bytes() == before and not (tmp_path / "data/toy_eps10_s7").exists()


@pytest.mark.parametrize("field,value", [("record_count", 8.0), ("result_unit", "frequency")])
@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_invalid_metadata_preserves_existing_artifacts(tmp_path, monkeypatch, field, value, kind):
    module, source, query, payload = _setup(tmp_path)
    _set(payload, field, value)
    query.write_text(json.dumps(payload))
    reference = source / "toy.csv"
    reference.write_bytes(REFERENCE)
    destination = tmp_path / "data/toy_eps10_s7"
    protected, link = [], None
    if kind == "directory":
        destination.mkdir()
        protected = [destination / query.name, destination / "old_reference.csv", destination / "README.md"]
    elif kind == "file":
        protected = [destination]
    else:
        target = tmp_path / "old_artifacts"
        if kind == "symlink":
            target.mkdir()
            protected = [target / query.name, target / "old_reference.csv"]
        destination.symlink_to(target, target_is_directory=True)
        link = os.readlink(destination)
    for path in protected:
        path.write_bytes(b"old artifact must remain unchanged\n")
    before = {path: path.read_bytes() for path in [query, reference, *protected]}
    names = sorted(path.name for path in destination.iterdir()) if destination.is_dir() else None
    monkeypatch.setattr(module, "cdp_rho", _forbidden)
    monkeypatch.setattr(module.np.random, "default_rng", _forbidden)
    with pytest.raises(SystemExit, match=field):
        _run(module, monkeypatch)
    assert all(path.read_bytes() == content for path, content in before.items())
    if link is not None:
        assert destination.is_symlink() and os.readlink(destination) == link
    if names is not None:
        assert sorted(path.name for path in destination.iterdir()) == names
    if kind == "dangling":
        assert not destination.exists()


def test_normal_integer_metadata_matches_public_loader_and_rng_endpoint(tmp_path, monkeypatch):
    module, source, query, payload = _setup(tmp_path)
    constructor = np.random.default_rng
    generators = []

    def tracked(seed):
        generator = constructor(seed)
        generators.append(generator)
        return generator

    monkeypatch.setattr(module.np.random, "default_rng", tracked)
    destination = _run(module, monkeypatch, seed=17, reserve=0.01)
    produced = json.loads((destination / query.name).read_text())
    assert type(produced["record_count"]) is int and produced["record_count"] == 8
    assert produced["result_unit"] == "records" and len(load_queries(str(destination / query.name), expected_record_count=8)) == 4
    assert not (source / "toy.csv").exists() and len(generators) == 1
    rho = module.cdp_rho(10.0, 1e-5) - 0.01
    sigma = float(np.sqrt((np.sqrt(2.0) / 8) ** 2 / (2 * rho)))
    reference = constructor(17)
    expected = np.clip(np.full(4, 0.25) + reference.normal(0.0, sigma, size=4), 0.0, 1.0) * 8
    assert deepcopy(generators[0].bit_generator.state) == reference.bit_generator.state
    np.testing.assert_array_equal([row["result"] for row in produced["queries"]], expected)
    assert payload["record_count"] == 8
