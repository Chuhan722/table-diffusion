"""Collect the frozen Issue #53 test residual fresh-seed matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import run_issue53_p6_dataset_smoke as base
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

PROTOCOL_VERSION = "issue53-test-residual-confirmation-v1"
FROZEN_PROTOCOL_SHA256 = (
    "9708f994c6c479b8e08c75cc662d0f79ec3ab5ec39cd9322e2ba5e8b7b30373b"
)
PROTOCOL_DOC = Path("docs/设计/Issue53_test残差几何fresh-seed确认协议.md")
PROTOCOL_DOC_SHA256 = (
    "4abe06d07f2eb59e880f8e2a16ff40e803c33a0e989487d732a1121b5e8bb785"
)
PROTOCOL_DOC_COMMIT = "abf676e93b07837ced96ac4a311a5b401364770d"
OUTPUT_DIR = Path("outputs/issue53_test_residual_geometry_confirmation_v1")
COLLECTION_REPORT = "collection_report.json"

SEEDS = (313, 314, 315, 316, 317)
ARMS = ("absolute", "sqrt_relative", "relative")
DATASET = "test_300x10"
SPEC = base.DATASETS[DATASET]


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, float) and math.isinf(value):
            value = "positive_infinity"
        result[key] = value
    return result


def generator_params(seed: int, arm: str) -> dict[str, Any]:
    """Return one exact case from the frozen 15-case matrix."""

    if seed not in SEEDS:
        raise ValueError(f"seed 不在冻结矩阵中：{seed!r}")
    if arm not in ARMS:
        raise ValueError(f"arm 不在冻结矩阵中：{arm!r}")
    params = base.generator_params()
    params["seed"] = seed
    params["residual_geometry"] = arm
    return params


def _common_generator_params() -> dict[str, Any]:
    params = generator_params(SEEDS[0], ARMS[0]).copy()
    params.pop("seed")
    params.pop("residual_geometry")
    return _jsonable_params(params)


def frozen_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "fresh_seed_test_residual_confirmation_raw_collection",
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
        "protocol_doc_commit": PROTOCOL_DOC_COMMIT,
        "dataset": {
            "name": DATASET,
            "n_records": SPEC["n_records"],
            "query_count": SPEC["query_count"],
            "device": SPEC["device"],
            "input_sha256": SPEC["sha256"],
        },
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "trajectory_count": len(ARMS) * len(SEEDS),
        "common_generator": _common_generator_params(),
        "arm_parameter": "residual_geometry",
        "natural_work": "cumulative_applied_participating_rows/n_records",
        "output_identity": "terminal_current",
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
        "parameter_retuning_allowed": False,
        "canonical_selection_allowed": False,
        "execution_concurrency": {
            "server": "root@10.8.176.53:6006",
            "generator_device": "numpy",
            "cuda_visible_devices": "empty",
            "worker_count": 5,
            "parallel_unit": "seed_shard",
            "arms_within_shard_serial": True,
        },
    }


def protocol_sha256() -> str:
    return hashlib.sha256(
        base._strict_json_bytes(frozen_protocol_manifest())
    ).hexdigest()


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol 身份漂移：expected={FROZEN_PROTOCOL_SHA256}, "
            f"observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": frozen_protocol_manifest(),
        "shards": [
            {"shard_index": index, "seed": seed, "case_count": len(ARMS)}
            for index, seed in enumerate(SEEDS)
        ],
        "case_order_within_shard": list(ARMS),
        "output_dir": str(OUTPUT_DIR),
        "scientific_overrides_allowed": False,
        "generation_started": False,
    }


def _audit_inputs(root: Path) -> dict[str, str]:
    observed = {}
    for key in ("schema", "queries", "marginals"):
        actual = base._sha256_file(root / SPEC[key])
        expected = SPEC["sha256"][key]
        if actual != expected:
            raise RuntimeError(f"{DATASET}.{key} SHA-256 漂移")
        observed[key] = actual
    protocol_actual = base._sha256_file(root / PROTOCOL_DOC)
    if protocol_actual != PROTOCOL_DOC_SHA256:
        raise RuntimeError("protocol doc SHA-256 漂移")
    observed["protocol_doc"] = protocol_actual
    return observed


def _environment(root: Path) -> dict[str, Any]:
    if base._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式采集要求包含 untracked 在内的干净工作树")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "":
        raise RuntimeError("本协议要求 CUDA_VISIBLE_DEVICES 为空")
    return {
        "git_commit": base._git_text(root, "rev-parse", "HEAD"),
        "worktree_clean_including_untracked": True,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": visible,
        "generator_device": "numpy",
    }


def _load_dataset(root: Path) -> tuple[Any, list, Any, np.ndarray]:
    schema = load_schema(str(root / SPEC["schema"]))
    queries = load_queries(str(root / SPEC["queries"]))
    marginals = load_marginals(str(root / SPEC["marginals"]))
    if len(queries) != SPEC["query_count"]:
        raise RuntimeError("measured query count 漂移")
    target = np.asarray([query["result"] for query in queries], dtype=float)
    return schema, queries, marginals, target


def _run_case(
    root: Path,
    shard_dir: Path,
    *,
    arm: str,
    seed: int,
    protocol_sha: str,
    git_commit: str,
) -> dict[str, Any]:
    schema, queries, marginals, target = _load_dataset(root)
    started = time.perf_counter()
    table, diagnostics = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=SPEC["n_records"],
        marginals=marginals,
        device=SPEC["device"],
        init_method="marginal",
        **generator_params(seed, arm),
    )
    elapsed = time.perf_counter() - started

    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{arm}/seed{seed} 主输出不是 terminal current")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError(f"{arm}/seed{seed} terminal identity 漂移")
    if diagnostics["params"]["residual_geometry"] != arm:
        raise RuntimeError(f"{arm}/seed{seed} residual geometry 漂移")
    expected_floor = None if arm == "absolute" else 8.0
    if diagnostics["params"]["residual_geometry_floor"] != expected_floor:
        raise RuntimeError(f"{arm}/seed{seed} residual floor 漂移")

    reason = diagnostics["termination_reason"]
    if reason not in {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }:
        raise RuntimeError(f"{arm}/seed{seed} 未返回 A/B/C")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != rounds or diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError(f"{arm}/seed{seed} no-gate clocks 漂移")
    if diagnostics["candidate_evaluation_count"] != rounds:
        raise RuntimeError(f"{arm}/seed{seed} candidate count 漂移")
    if diagnostics["state_evaluation_count"] != max(1, rounds):
        raise RuntimeError(f"{arm}/seed{seed} state evaluation count 漂移")

    answers = np.asarray(evaluate_table(final_table, queries), dtype=float)
    loss = float(compute_squared_loss(target, answers))
    l1 = float(compute_normalized_l1(target, answers, SPEC["n_records"]))
    if loss != diagnostics["final_current_squared_loss"]:
        raise RuntimeError(f"{arm}/seed{seed} terminal loss 复算漂移")
    if l1 != diagnostics["final_current_normalized_l1"]:
        raise RuntimeError(f"{arm}/seed{seed} terminal L1 复算漂移")
    work = sum(base._applied_rows(clock) for clock in clocks) / SPEC["n_records"]

    case_dir = shard_dir / DATASET / arm
    case_dir.mkdir(parents=True)
    table_path = case_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    result = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "git_commit": git_commit,
        "dataset": DATASET,
        "arm": arm,
        "seed": seed,
        "device": SPEC["device"],
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
        "terminal_table_sha256": base._frame_sha256(final_table),
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[seed={seed} {arm}] {reason} rounds={rounds} "
        f"work={work:.4f} L1={l1:.10f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def run_shard(confirmed_protocol_sha256: str, shard_index: int) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    if shard_index not in range(len(SEEDS)):
        raise ValueError(f"shard_index 非法：{shard_index!r}")

    root = base._repo_root()
    seed = SEEDS[shard_index]
    destination = root / OUTPUT_DIR / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(f"seed shard 已存在，不覆盖：{destination}")
    environment = _environment(root)
    input_sha256 = _audit_inputs(root)
    git_commit = environment["git_commit"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".seed_{seed}.tmp-",
        dir=destination.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        results = [
            _run_case(
                root,
                temporary,
                arm=arm,
                seed=seed,
                protocol_sha=expected,
                git_commit=git_commit,
            )
            for arm in ARMS
        ]
        manifest = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "shard_index": shard_index,
            "seed": seed,
            "git_commit": git_commit,
            "environment": environment,
            "input_sha256": input_sha256,
            "results": results,
        }
        (temporary / "shard_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "shard_manifest.json"


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) != len(SEEDS):
            raise RuntimeError(f"{arm} case 数不是 {len(SEEDS)}")
        arms[arm] = {
            "case_count": len(arm_rows),
            "termination_counts": dict(
                sorted(Counter(row["termination_reason"] for row in arm_rows).items())
            ),
            "terminal_normalized_l1_mean": _mean(
                [row["terminal_current_normalized_l1"] for row in arm_rows]
            ),
            "terminal_normalized_l1_median": _median(
                [row["terminal_current_normalized_l1"] for row in arm_rows]
            ),
            "terminal_squared_loss_mean": _mean(
                [row["terminal_current_squared_loss"] for row in arm_rows]
            ),
            "terminal_squared_loss_median": _median(
                [row["terminal_current_squared_loss"] for row in arm_rows]
            ),
            "rounds_mean": _mean([row["rounds_run"] for row in arm_rows]),
            "rounds_median": _median([row["rounds_run"] for row in arm_rows]),
            "normalized_work_mean": _mean(
                [row["normalized_work_at_stop"] for row in arm_rows]
            ),
            "normalized_work_median": _median(
                [row["normalized_work_at_stop"] for row in arm_rows]
            ),
            "elapsed_sec_mean": _mean([row["elapsed_sec"] for row in arm_rows]),
        }
    return {
        "arms": arms,
        "resource_cap_case_count": sum(
            row["termination_reason"] == "resource_cap_reached" for row in rows
        ),
        "normal_completion_case_count": sum(
            row["termination_reason"] in {"fit_target_reached", "early_stopped"}
            for row in rows
        ),
    }


def aggregate(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError("必须显式确认完整 protocol SHA-256")
    root = base._repo_root()
    if base._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("aggregate 要求包含 untracked 在内的干净工作树")
    destination = root / OUTPUT_DIR
    report_path = destination / COLLECTION_REPORT
    if report_path.exists():
        raise FileExistsError(f"采集报告已存在，不覆盖：{report_path}")

    rows = []
    commits = set()
    for index, seed in enumerate(SEEDS):
        shard_path = destination / f"seed_{seed}" / "shard_manifest.json"
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        if payload["protocol_sha256"] != expected:
            raise RuntimeError(f"seed {seed} protocol SHA 漂移")
        if payload["shard_index"] != index or payload["seed"] != seed:
            raise RuntimeError(f"seed {seed} shard 身份漂移")
        commits.add(payload["git_commit"])
        for row in payload["results"]:
            if row["seed"] != seed or row["protocol_sha256"] != expected:
                raise RuntimeError(f"seed {seed} case 身份漂移")
            table_path = (
                destination
                / f"seed_{seed}"
                / DATASET
                / row["arm"]
                / "terminal_current.csv"
            )
            if base._sha256_file(table_path) != row["terminal_table_sha256"]:
                raise RuntimeError(f"seed {seed}/{row['arm']} table SHA 漂移")
            rows.append(row)

    identities = {(row["seed"], row["arm"]) for row in rows}
    expected_identities = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if len(rows) != 15 or identities != expected_identities:
        raise RuntimeError("15-case 矩阵不完整或重复")
    if len(commits) != 1:
        raise RuntimeError(f"shard Git commit 不一致：{sorted(commits)}")

    report = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": expected,
        "protocol": frozen_protocol_manifest(),
        "execution_git_commit": next(iter(commits)),
        "case_count": len(rows),
        "raw_results": sorted(rows, key=lambda row: (row["seed"], row["arm"])),
        "summary": _summarize(rows),
        "claim_scope": "fresh_seed_test_raw_collection_before_heldout_evaluation",
        "canonical_selection_allowed": False,
        "parameter_retuning_performed": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".collection-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_path, report_path)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--confirm-protocol-sha", required=True)
    shard.add_argument(
        "--shard-index",
        required=True,
        type=int,
        choices=range(len(SEEDS)),
    )
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    if args.command == "run-shard":
        path = run_shard(args.confirm_protocol_sha, args.shard_index)
        print(f"confirmation shard -> {path}")
        return
    path = aggregate(args.confirm_protocol_sha)
    print(f"confirmation collection -> {path}")
    print(f"collection SHA-256 -> {base._sha256_file(path)}")


if __name__ == "__main__":
    main()
