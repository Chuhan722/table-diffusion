"""因子构造性能脚本的小规模端到端门禁。"""

import json
import os
from pathlib import Path
import subprocess
import sys


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
