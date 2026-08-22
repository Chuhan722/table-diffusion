"""Frozen two-dataset, three-arm P=6 residual-geometry comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts import run_issue53_p6_dataset_smoke as base
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1, compute_squared_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

PROTOCOL_VERSION = "issue53-sqrt-residual-earlystop-comparison-v1"
FROZEN_PROTOCOL_SHA256 = (
    "7e7b5e08f9d934031257cbd98b6a857f7ba1dcb4cf1f97077d48f781a4e2585f"
)
OUTPUT_DIR = Path("outputs/issue53_sqrt_residual_earlystop_comparison_v1")
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_平方根残差P6早停两数据三臂对比协议.md"
)

SEEDS = (310, 311, 312)
ARMS = ("absolute", "sqrt_relative", "relative")
DATASETS = base.DATASETS


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, float) and math.isinf(value):
            value = "positive_infinity"
        result[key] = value
    return result


def generator_params(seed: int, arm: str) -> dict[str, Any]:
    """Return the frozen P=6 configuration for one paired matrix case."""

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
        "purpose": (
            "development_descriptive_paired_p6_residual_geometry_comparison"
        ),
        "protocol_doc": str(PROTOCOL_DOC),
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
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "trajectory_count": len(DATASETS) * len(ARMS) * len(SEEDS),
        "common_generator": _common_generator_params(),
        "arm_parameter": "residual_geometry",
        "sqrt_relative_formula": (
            "sign(raw)*magnitude/sqrt(max(target,8))/n_records"
        ),
        "natural_work": "cumulative_applied_participating_rows/n_records",
        "output_identity": "terminal_current",
        "online_l1_used": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
        "parameter_retuning_allowed": False,
        "canonical_selection_allowed": False,
        "execution_concurrency": {
            "server": "root@10.8.176.53:6006",
            "visible_gpu_count": 1,
            "worker_count": 1,
            "seed_shards_serial": True,
        },
        "aggregation": {
            "primary_descriptive_metric": "terminal_current_normalized_l1",
            "per_dataset_only": True,
            "report_raw_cases": True,
            "report_mean_and_median": True,
            "report_paired_seed_wins": True,
            "quality_compute_scalarization": False,
        },
        "known_evidence_seen_before_protocol": [
            "pr59_absolute_relative_fixed_2000_round_results",
            "pr63_relative_seed200_p6_dataset_smoke",
        ],
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
            {
                "shard_index": index,
                "seed": seed,
                "case_count": len(DATASETS) * len(ARMS),
            }
            for index, seed in enumerate(SEEDS)
        ],
        "case_order_within_shard": [
            {"dataset": dataset, "arm": arm}
            for dataset in DATASETS
            for arm in ARMS
        ],
        "output_dir": str(OUTPUT_DIR),
        "scientific_overrides_allowed": False,
        "generation_started": False,
    }


def _load_dataset(root: Path, name: str) -> tuple[Any, list, Any, np.ndarray]:
    spec = DATASETS[name]
    schema = load_schema(str(root / spec["schema"]))
    queries = load_queries(str(root / spec["queries"]))
    marginals = load_marginals(str(root / spec["marginals"]))
    target = np.asarray([query["result"] for query in queries], dtype=float)
    if len(queries) != spec["query_count"]:
        raise RuntimeError(f"{name} query_count 漂移")
    return schema, queries, marginals, target


def _run_case(
    root: Path,
    shard_dir: Path,
    *,
    name: str,
    arm: str,
    seed: int,
    protocol_sha: str,
    git_commit: str,
) -> dict[str, Any]:
    spec = DATASETS[name]
    schema, queries, marginals, target = _load_dataset(root, name)

    started = time.perf_counter()
    table, diagnostics = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=spec["n_records"],
        marginals=marginals,
        device=spec["device"],
        init_method="marginal",
        **generator_params(seed, arm),
    )
    elapsed = time.perf_counter() - started

    final_table = diagnostics.pop("final_table").reset_index(drop=True)
    if not table.reset_index(drop=True).equals(final_table):
        raise RuntimeError(f"{name}/{arm}/seed{seed} 主输出不是 terminal current")
    if diagnostics["output_table_identity"] != "terminal_current":
        raise RuntimeError(f"{name}/{arm}/seed{seed} terminal identity 不一致")
    if diagnostics["params"]["residual_geometry"] != arm:
        raise RuntimeError(f"{name}/{arm}/seed{seed} residual arm 身份漂移")
    expected_floor = None if arm == "absolute" else 8.0
    if diagnostics["params"]["residual_geometry_floor"] != expected_floor:
        raise RuntimeError(f"{name}/{arm}/seed{seed} residual floor 身份漂移")

    reason = diagnostics["termination_reason"]
    if reason not in {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }:
        raise RuntimeError(f"{name}/{arm}/seed{seed} 未返回 A/B/C 原因")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    if len(clocks) != rounds or diagnostics["accept_history"] != [True] * rounds:
        raise RuntimeError(f"{name}/{arm}/seed{seed} no-gate clocks 不一致")
    if diagnostics["candidate_evaluation_count"] != rounds:
        raise RuntimeError(f"{name}/{arm}/seed{seed} candidate count 不一致")
    if diagnostics["state_evaluation_count"] != max(1, rounds):
        raise RuntimeError(f"{name}/{arm}/seed{seed} state evaluation count 不一致")

    answers = np.asarray(evaluate_table(final_table, queries), dtype=float)
    loss = float(compute_squared_loss(target, answers))
    l1 = float(compute_normalized_l1(target, answers, spec["n_records"]))
    if loss != diagnostics["final_current_squared_loss"]:
        raise RuntimeError(f"{name}/{arm}/seed{seed} terminal loss 复算不一致")
    if l1 != diagnostics["final_current_normalized_l1"]:
        raise RuntimeError(f"{name}/{arm}/seed{seed} terminal L1 复算不一致")
    work = (
        sum(base._applied_rows(clock) for clock in clocks) / spec["n_records"]
    )

    case_dir = shard_dir / name / arm
    case_dir.mkdir(parents=True)
    table_path = case_dir / "terminal_current.csv"
    final_table.to_csv(table_path, index=False)
    result = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "git_commit": git_commit,
        "dataset": name,
        "arm": arm,
        "seed": seed,
        "device": spec["device"],
        "termination_reason": reason,
        "inner_complete": bool(diagnostics["inner_complete"]),
        "output_table_identity": "terminal_current",
        "rounds_run": rounds,
        "candidate_evaluations": int(
            diagnostics["candidate_evaluation_count"]
        ),
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
        f"[seed={seed} {name} {arm}] {reason} rounds={rounds} "
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
    environment = base._environment(root)
    input_sha256 = base._audit_inputs(root)
    git_commit = environment["git_commit"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".seed_{seed}.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        results = [
            _run_case(
                root,
                temporary,
                name=name,
                arm=arm,
                seed=seed,
                protocol_sha=expected,
                git_commit=git_commit,
            )
            for name in DATASETS
            for arm in ARMS
        ]
        shard_manifest = {
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
            json.dumps(
                shard_manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return destination / "shard_manifest.json"


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) != len(SEEDS):
            raise RuntimeError(f"{arm} case 数不是 {len(SEEDS)}")
        arm_summary[arm] = {
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
            "elapsed_sec_mean": _mean(
                [row["elapsed_sec"] for row in arm_rows]
            ),
        }

    paired_winners = []
    win_counts: Counter[str] = Counter()
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if len(seed_rows) != len(ARMS):
            raise RuntimeError(f"seed {seed} 配对 case 不完整")
        minimum = min(
            row["terminal_current_normalized_l1"] for row in seed_rows
        )
        winners = sorted(
            row["arm"]
            for row in seed_rows
            if row["terminal_current_normalized_l1"] == minimum
        )
        paired_winners.append(
            {
                "seed": seed,
                "minimum_terminal_normalized_l1": minimum,
                "winner_arms": winners,
            }
        )
        for winner in winners:
            win_counts[winner] += 1

    minimum_mean = min(
        item["terminal_normalized_l1_mean"] for item in arm_summary.values()
    )
    mean_winners = sorted(
        arm
        for arm, item in arm_summary.items()
        if item["terminal_normalized_l1_mean"] == minimum_mean
    )
    return {
        "arms": arm_summary,
        "paired_seed_l1_winners": paired_winners,
        "paired_seed_l1_win_counts": dict(sorted(win_counts.items())),
        "lowest_mean_terminal_l1_arms": mean_winners,
        "lowest_mean_terminal_l1": minimum_mean,
        "resource_cap_case_count": sum(
            row["termination_reason"] == "resource_cap_reached" for row in rows
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
    report_path = destination / "report.json"
    if report_path.exists():
        raise FileExistsError(f"总报告已存在，不覆盖：{report_path}")

    rows: list[dict[str, Any]] = []
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
                / row["dataset"]
                / row["arm"]
                / "terminal_current.csv"
            )
            if base._sha256_file(table_path) != row["terminal_table_sha256"]:
                raise RuntimeError(
                    f"{row['dataset']}/{row['arm']}/seed{seed} table SHA 漂移"
                )
            rows.append(row)

    expected_cases = len(DATASETS) * len(ARMS) * len(SEEDS)
    identities = {
        (row["dataset"], row["arm"], row["seed"]) for row in rows
    }
    expected_identities = {
        (dataset, arm, seed)
        for dataset in DATASETS
        for arm in ARMS
        for seed in SEEDS
    }
    if len(rows) != expected_cases or identities != expected_identities:
        raise RuntimeError("18-case 矩阵不完整或有重复")
    if len(commits) != 1:
        raise RuntimeError(f"shard Git commit 不一致：{sorted(commits)}")

    report = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": expected,
        "protocol": frozen_protocol_manifest(),
        "execution_git_commit": next(iter(commits)),
        "case_count": len(rows),
        "raw_results": sorted(
            rows, key=lambda row: (row["dataset"], row["seed"], row["arm"])
        ),
        "datasets": {
            dataset: _summarize_dataset(
                [row for row in rows if row["dataset"] == dataset]
            )
            for dataset in DATASETS
        },
        "claim_scope": (
            "paired_three_seed_development_terminal_measured_quality_and_work_only"
        ),
        "canonical_selection_allowed": False,
        "parameter_retuning_performed": False,
        "raw_reference_data_accessed": False,
        "privacy_budget_consumed": False,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    os.replace(temporary_path, report_path)
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    shard_parser = subparsers.add_parser("run-shard")
    shard_parser.add_argument("--confirm-protocol-sha", required=True)
    shard_parser.add_argument(
        "--shard-index", required=True, type=int, choices=range(len(SEEDS))
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
        print(f"residual comparison shard -> {path}")
        return
    print(f"residual comparison report -> {aggregate(args.confirm_protocol_sha)}")


if __name__ == "__main__":
    main()
