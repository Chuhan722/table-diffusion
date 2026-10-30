"""噪声考卷不读取/复制原表，也不复用可能含旧私有文件的目录。"""
import builtins
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = b"a,b\n0,0\n0,0\n0,1\n0,1\n1,0\n1,0\n1,1\n1,1\n"


def _setup(tmp_path):
    source = tmp_path / "data/toy"
    source.mkdir(parents=True)
    payload = {
        "dataset": "toy", "record_count": 8, "query_count": 4, "result_unit": "records",
        "description": "Handcrafted public-count fixture",
        "queries": [
            {"id": f"q{a}{b}", "conditions": [
                {"attribute": "a", "operator": "==", "value": str(a)},
                {"attribute": "b", "operator": "==", "value": str(b)},
            ], "result": 2.0}
            for a in range(2) for b in range(2)
        ],
    }
    query = source / "measured_4query.json"
    query.write_text(json.dumps(payload), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("noisy_artifact_test", ROOT / "scripts/make_noisy_exam.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = tmp_path
    return module, source, query, payload


def _run(module, monkeypatch, seed=7, reserve=0.0, tag=""):
    monkeypatch.setattr(sys, "argv", [
        "make_noisy_exam.py", "--data", "toy", "--epsilon", "10",
        "--delta", "1e-5", "--noise-seed", str(seed), "--reserve-rho", str(reserve), "--tag", tag,
    ])
    module.main()
    return module.REPO / f"data/toy_eps10{tag}_s{seed}"


@pytest.mark.parametrize("seed", (7, 17, 29))
@pytest.mark.parametrize("reserve", (0.0, 0.01))
def test_query_only_input_preserves_noise_arithmetic(tmp_path, monkeypatch, seed, reserve):
    module, source, query, payload = _setup(tmp_path)
    original = query.read_bytes()
    destination = _run(module, monkeypatch, seed, reserve)
    assert not (source / "toy.csv").exists()
    assert sorted(path.name for path in destination.iterdir()) == ["README.md", "measured_4query.json"]
    actual = json.loads((destination / query.name).read_text())
    rho = module.cdp_rho(10.0, 1e-5) - reserve
    sigma = float(np.sqrt((np.sqrt(2.0) / 8) ** 2 / (2 * rho)))
    expected = np.clip(np.full(4, 0.25) + np.random.default_rng(seed).normal(0.0, sigma, size=4), 0.0, 1.0) * 8
    np.testing.assert_array_equal([row["result"] for row in actual["queries"]], expected)
    assert actual["record_count"] == payload["record_count"]
    assert actual["result_unit"] == "records"
    assert query.read_bytes() == original
    text = (destination / "README.md").read_text()
    assert "原表" in text and "不能作为 DP 发布包" in text
    assert "md5" not in text


def test_existing_reference_is_neither_opened_nor_changed(tmp_path, monkeypatch):
    module, source, query, _ = _setup(tmp_path)
    reference = source / "toy.csv"
    reference.write_bytes(REFERENCE)
    before = query.read_bytes()
    builtin_open, path_open = builtins.open, Path.open

    def forbid(file):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == reference.resolve():
            pytest.fail("Noisy-exam generation accessed the reference table")

    def guarded_open(file, *args, **kwargs):
        forbid(file)
        return builtin_open(file, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        forbid(path)
        return path_open(path, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", guarded_open)
        guarded.setattr(Path, "open", guarded_path_open)
        destination = _run(module, guarded)
    assert reference.read_bytes() == REFERENCE and query.read_bytes() == before
    assert not list(destination.glob("*.csv"))


def test_new_tag_preserves_existing_artifacts_and_noise(tmp_path, monkeypatch):
    module, _, query, _ = _setup(tmp_path)
    old = _run(module, monkeypatch)
    (old / "old_reference.csv").write_bytes(REFERENCE)
    before = {path.name: path.read_bytes() for path in old.iterdir()}
    new = _run(module, monkeypatch, tag="_fresh")
    assert {path.name: path.read_bytes() for path in old.iterdir()} == before
    assert (new / query.name).read_bytes() == before[query.name]
    assert sorted(path.name for path in new.iterdir()) == ["README.md", query.name]


@pytest.mark.parametrize("kind", ("empty_dir", "mixed_dir", "file", "symlink_dir", "broken_symlink"))
def test_existing_output_is_rejected_without_modification(tmp_path, monkeypatch, kind, capsys):
    module, source, query, _ = _setup(tmp_path)
    (source / "toy.csv").write_bytes(REFERENCE)
    original_query = query.read_bytes()
    destination = tmp_path / "data/toy_eps10_s7"
    protected = []
    link_target = None
    if kind in ("empty_dir", "mixed_dir"):
        destination.mkdir()
        if kind == "mixed_dir":
            protected = [destination / "old_reference.csv", destination / query.name]
    elif kind == "file":
        protected = [destination]
    else:
        target = tmp_path / "private_reference"
        if kind == "symlink_dir":
            target.mkdir()
            protected = [target / "old_reference.csv"]
        destination.symlink_to(target, target_is_directory=True)
        link_target = os.readlink(destination)
    for path in protected:
        path.write_bytes(b"retain previous contents\n")
    before = {path: path.read_bytes() for path in protected}
    before_names = sorted(path.name for path in destination.iterdir()) if destination.is_dir() else None
    with pytest.raises(SystemExit, match="已有输出"):
        _run(module, monkeypatch)
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert (source / "toy.csv").read_bytes() == REFERENCE
    assert query.read_bytes() == original_query
    if link_target is not None:
        assert destination.is_symlink() and os.readlink(destination) == link_target
    if before_names is not None:
        assert sorted(path.name for path in destination.iterdir()) == before_names
    if kind == "broken_symlink":
        assert not destination.exists()
    assert "已写出" not in capsys.readouterr().out


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_query_only_artifact_runs_public_generation_pipeline(tmp_path, monkeypatch, backend):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        try:
            devices = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        if not devices:
            pytest.skip("No CUDA device")
    module, source, original_query, _ = _setup(tmp_path)
    destination = _run(module, monkeypatch)
    original_query.unlink()
    assert not list(source.iterdir())
    schema = tmp_path / "public_schema.json"
    schema.write_text(json.dumps({"fields": ["a", "b"], "domains": [["0", "1"], ["0", "1"]]}))
    output = tmp_path / "generated.csv"
    command = [
        sys.executable, "-m", "resevo", "--schema", str(schema),
        "--queries", str(destination / "measured_4query.json"), "--rows", "8",
        "--rounds", "3", "--init", "uniform", "--seed", "11",
        "--backend", backend, "--output", str(output),
    ]
    result = subprocess.run(
        command, cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["rows"] == 8 and info["backend"] == backend
    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["a", "b"] and len(rows) == 9
    assert all(len(row) == 2 and set(row) <= {"0", "1"} for row in rows[1:])
    assert not list(destination.glob("*.csv"))
