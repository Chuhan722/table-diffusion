"""因子构造性能脚本的小规模端到端门禁。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _load_benchmark_module(monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts/benchmark_factorized_workload.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location(
        "benchmark_factorized_workload_for_test", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_compares_same_algorithm_in_isolated_workers(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_factorized_workload.py",
            "--rounds",
            "2",
            "--seeds",
            "0",
            "--temperature",
            "2",
            "--sweeps",
            "2",
            "--device",
            "numpy",
            "--output",
            str(output),
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    legacy = result["runs"]["legacy_rowwise"][0]
    compiled = result["runs"]["compiled_batch"][0]
    assert result["equivalence"]["passed"] is True
    assert result["equivalence"]["checked_state_hashes"] == 2
    assert result["execution_order"] == [
        {"seed": 0, "builder": "legacy_rowwise"},
        {"seed": 0, "builder": "compiled_batch"},
    ]
    assert legacy["state_sha256_history"] == compiled[
        "state_sha256_history"
    ]
    assert legacy["final_csv_sha256"] == compiled["final_csv_sha256"]
    assert legacy["primary_rng_state_sha256"] == compiled[
        "primary_rng_state_sha256"
    ]
    assert legacy["gibbs_rng_state_sha256"] == compiled[
        "gibbs_rng_state_sha256"
    ]
    assert legacy["factor_model_builds"] == compiled[
        "factor_model_builds"
    ]
    assert legacy["condition_evaluation_batches"] == 0
    assert compiled["condition_evaluation_batches"] > 0
    assert "逐步等价门禁：True" in completed.stdout


def test_benchmark_preflights_all_worker_output_collisions(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark.json"
    worker_dir = tmp_path / "benchmark_workers"
    worker_dir.mkdir()
    existing_worker = worker_dir / "seed0_compiled_batch.json"
    existing_worker.write_text("保留已有结果", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_factorized_workload.py",
            "--rounds",
            "2",
            "--seeds",
            "0",
            "--temperature",
            "2",
            "--sweeps",
            "2",
            "--device",
            "numpy",
            "--output",
            str(output),
        ],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "尚未启动任何 worker" in completed.stderr
    assert not (worker_dir / "seed0_legacy_rowwise.json").exists()
    assert existing_worker.read_text(encoding="utf-8") == "保留已有结果"
    assert not output.exists()


def test_atomic_json_write_preserves_target_on_serialization_error(
    tmp_path, monkeypatch
):
    benchmark = _load_benchmark_module(monkeypatch)
    output = tmp_path / "result.json"
    output.write_text("原结果", encoding="utf-8")

    with pytest.raises(TypeError):
        benchmark._write_json_atomically(
            output,
            {"unsupported": object()},
            overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == "原结果"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []
