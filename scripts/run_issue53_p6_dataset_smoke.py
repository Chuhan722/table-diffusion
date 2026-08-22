#!/usr/bin/env python
"""Fixed two-dataset P=6 end-to-end smoke for Issue #53."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

PROTOCOL_VERSION = "issue53-p6-dataset-smoke-v1"
FROZEN_PROTOCOL_SHA256 = (
    "3b593ce71c8b4bd147b836dd03986d4e64d27bb782a57d0a9ac5759baf805c17"
)
OUTPUT_DIR = Path("outputs/issue53_p6_dataset_smoke_seed200")

SEED = 200
PATIENCE_TICKS = 6
RHO = 0.01
ROUND_CAP = 6000
CANDIDATE_BUDGET = 6000
EXPECTED_NORMALIZED_WORK_CAP = 60.0
BASELINE_ROUNDS = 2000
BASELINE_PATH = Path("docs/实验结果/formal_residual_geometry_5seed_2000round.json")
BASELINE_SHA256 = "51aff5414eb15c9cfdda496dc1549c6fba7216043159bd377be429fb11443f64"

DATASETS: dict[str, dict[str, Any]] = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path("configs/test_300x10/measured_50query.json"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "query_count": 50,
        "device": "numpy",
        "sha256": {
            "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
            "queries": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
            "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
        },
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        "query_count": 1001,
        "device": "cuda",
        "sha256": {
            "schema": "5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4",
            "queries": "b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f",
            "marginals": "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e",
        },
    },
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_text(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def generator_params() -> dict[str, Any]:
    """Return the current relative-f8 main arm plus P=6 stopping."""

    return {
        "n_rounds": ROUND_CAP,
        "seed": SEED,
        "beta": 1.0,
        "h": 0.8,
        "rho": RHO,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "eval_method": "vectorized",
        "batch_size": 256,
        "log_every": 100,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": 16.0,
        "alpha_max": 16.0,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 16.0,
        "record_transition_clocks": True,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def _json_generator_params() -> dict[str, Any]:
    result = {}
    for key, value in generator_params().items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, float) and math.isinf(value):
            value = "positive_infinity"
        result[key] = value
    return result


def frozen_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "pr_archive_real_dataset_smoke_not_tuning",
        "datasets": [
            {
                "name": name,
                "n_records": spec["n_records"],
                "query_count": spec["query_count"],
                "device": spec["device"],
                "input_sha256": spec["sha256"],
            }
            for name, spec in DATASETS.items()
        ],
        "generator": _json_generator_params(),
        "natural_work": "cumulative_applied_participating_rows/n_records",
        "expected_normalized_work_cap": EXPECTED_NORMALIZED_WORK_CAP,
        "output_identity": "terminal_current",
        "baseline": {
            "path": str(BASELINE_PATH),
            "sha256": BASELINE_SHA256,
            "seed": SEED,
            "arm": "relative_f8",
            "rounds": BASELINE_ROUNDS,
            "role": "descriptive_only_not_acceptance_gate",
        },
        "parameter_retuning_allowed": False,
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }


def protocol_sha256() -> str:
    return hashlib.sha256(_strict_json_bytes(frozen_protocol_manifest())).hexdigest()


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol 身份漂移：expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": frozen_protocol_manifest(),
        "dataset_run_order": list(DATASETS),
        "trajectory_count": 2,
        "output_dir": str(OUTPUT_DIR),
        "parameter_overrides_allowed": False,
        "generation_started": False,
    }


def _audit_inputs(root: Path) -> dict[str, dict[str, str]]:
    observed = {}
    for name, spec in DATASETS.items():
        observed[name] = {}
        for key in ("schema", "queries", "marginals"):
            path = root / spec[key]
            actual = _sha256_file(path)
            if actual != spec["sha256"][key]:
                raise RuntimeError(f"{name}.{key} SHA-256 漂移")
            observed[name][key] = actual
    return observed


def _load_baselines(root: Path) -> dict[str, dict[str, Any]]:
    path = root / BASELINE_PATH
    if _sha256_file(path) != BASELINE_SHA256:
        raise RuntimeError("baseline artifact 身份漂移")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for name in DATASETS:
        matches = [
            row
            for row in payload["datasets"][name]["runs"]
            if row["seed"] == SEED and row["arm"] == "relative_f8"
        ]
        if len(matches) != 1 or matches[0]["rounds_run"] != BASELINE_ROUNDS:
            raise RuntimeError(f"{name} baseline 身份不唯一或未跑满")
        row = matches[0]
        rows[name] = {
            "rounds_run": int(row["rounds_run"]),
            "final_loss": float(row["final_loss"]),
            "final_normalized_l1": float(row["final_table_measured_l1"]),
            "final_table_sha256": row["final_table_sha256"],
        }
    return rows


def _applied_rows(clock: dict[str, Any]) -> int:
    accepted = int(clock["accepted_attempt"])
    if accepted == 0:
        return 0
    return int(clock["attempts"][accepted - 1]["participating_rows"])


def _run_one(root: Path, output: Path, name: str, baseline: dict[str, Any]) -> dict:
    spec = DATASETS[name]
    schema = load_schema(str(root / spec["schema"]))
    queries = load_queries(str(root / spec["queries"]))
    marginals = load_marginals(str(root / spec["marginals"]))
    target = np.asarray([query["result"] for query in queries], dtype=float)
    if len(queries) != spec["query_count"]:
        raise RuntimeError(f"{name} query_count 漂移")

    start = time.perf_counter()
    table, diagnostics = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=spec["n_records"],
        marginals=marginals,
        device=spec["device"],
        init_method="marginal",
        **generator_params(),
    )
    elapsed = time.perf_counter() - start
    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{name} 主输出不是 terminal current")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError(f"{name} terminal identity 不一致")
    reason = diagnostics["termination_reason"]
    if reason not in {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }:
        raise RuntimeError(f"{name} 未返回 A/B/C 原因")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != rounds or diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError(f"{name} no-gate clocks 不一致")
    if diagnostics["candidate_evaluation_count"] != rounds:
        raise RuntimeError(f"{name} candidate count 不一致")
    if diagnostics["state_evaluation_count"] != max(1, rounds):
        raise RuntimeError(f"{name} state evaluation count 不一致")

    answers = np.asarray(evaluate_table(final_table, queries), dtype=float)
    loss = float(compute_squared_loss(target, answers))
    l1 = float(compute_normalized_l1(target, answers, spec["n_records"]))
    if loss != diagnostics["final_current_squared_loss"]:
        raise RuntimeError(f"{name} terminal loss 复算不一致")
    if l1 != diagnostics["final_current_normalized_l1"]:
        raise RuntimeError(f"{name} terminal L1 复算不一致")
    work = sum(_applied_rows(clock) for clock in clocks) / spec["n_records"]

    dataset_dir = output / name
    dataset_dir.mkdir()
    table_path = dataset_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    result = {
        "dataset": name,
        "seed": SEED,
        "device": spec["device"],
        "termination_reason": reason,
        "inner_complete": bool(diagnostics["inner_complete"]),
        "output_table_identity": "terminal_current",
        "rounds_run": rounds,
        "candidate_evaluations": int(diagnostics["candidate_evaluation_count"]),
        "normalized_work_at_stop": float(work),
        "terminal_current_squared_loss": loss,
        "terminal_current_normalized_l1": l1,
        "best_loss_diagnostic_only": float(diagnostics["best_loss"]),
        "elapsed_sec": float(elapsed),
        "terminal_table_sha256": _frame_sha256(final_table),
        "historical_2000_round_baseline": baseline,
        "descriptive_comparison": {
            "terminal_minus_baseline_normalized_l1": (
                l1 - baseline["final_normalized_l1"]
            ),
            "terminal_minus_baseline_squared_loss": loss - baseline["final_loss"],
            "raw_rounds_saved_vs_2000": BASELINE_ROUNDS - rounds,
            "raw_round_saving_fraction_vs_2000": (
                (BASELINE_ROUNDS - rounds) / BASELINE_ROUNDS
            ),
        },
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    (dataset_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{name}] {reason} rounds={rounds} work={work:.4f} "
        f"L1={l1:.8f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _environment(root: Path) -> dict[str, Any]:
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式 smoke 要求包含 untracked 在内的干净工作树")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len([item for item in visible.split(",") if item]) != 1:
        raise RuntimeError("必须用 CUDA_VISIBLE_DEVICES 暴露恰好一张 GPU")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("没有恰好一张可用 CUDA GPU")
    return {
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "worktree_clean_including_untracked": True,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "visible_gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": visible,
        "platform": platform.platform(),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    environment = _environment(root)
    input_sha256 = _audit_inputs(root)
    baselines = _load_baselines(root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        results = [
            _run_one(root, temporary, name, baselines[name]) for name in DATASETS
        ]
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": frozen_protocol_manifest(),
            "environment": environment,
            "input_sha256": input_sha256,
            "baseline_sha256": BASELINE_SHA256,
            "results": results,
            "summary": {
                "dataset_count": 2,
                "all_inner_complete": all(row["inner_complete"] for row in results),
                "all_terminal_current": True,
                "resource_cap_case_count": sum(
                    row["termination_reason"] == "resource_cap_reached"
                    for row in results
                ),
                "quality_comparison_role": "descriptive_only",
                "parameter_retuning_performed": False,
                "raw_reference_data_accessed": False,
                "privacy_budget_consumed": False,
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "report.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    print(f"P6 dataset smoke -> {run(args.confirm_protocol_sha)}")


if __name__ == "__main__":
    main()
