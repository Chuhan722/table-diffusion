"""Run Issue #52 Stage T independent tau=1..5 long-horizon frontier."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue52_parallel_trajectories as parallel
    from scripts import issue52_protocol as frozen_protocol
    from scripts import run_issue49_stage_a as common
else:
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue52_parallel_trajectories as parallel
    import issue52_protocol as frozen_protocol
    import run_issue49_stage_a as common

from table_diffevo.experiment_parallel import (
    MAX_EXPERIMENT_WORKERS,
    scientific_sha256,
    validate_max_workers,
)


REPORT_FORMAT = "issue52_stage_t_report_v1"
EXPECTED_INPUT_SHA256 = frozen_protocol.EXPECTED_INPUT_SHA256


def _protocol(mode):
    return frozen_protocol.stage_t_protocol(mode)


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _config_id(temperature):
    return f"independent_{_tau_key(temperature)}"


def _valid_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_implementation_constants(protocol):
    checkpoints = protocol["trend_checkpoints"]
    snapshots = protocol["snapshot_rounds"]
    seeds = protocol["stage_t_seeds"]
    state_seeds = protocol["state_library_seeds"]
    gates = {
        "trajectory_rho_matches": trajectory.RHO == protocol["rho"],
        "trajectory_eta_matches": trajectory.ETA == protocol["eta"],
        "trajectory_mu_matches": (
            trajectory.MU == protocol["trajectory_mu"]
        ),
        "trajectory_logit_clip_matches": (
            trajectory.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
        "numpy_cpu_only": protocol["device"] == "numpy",
        "independent_only": protocol["source_sweeps"] == 0,
        "worker_cap_matches": (
            protocol["max_workers_allowed"]
            == MAX_EXPERIMENT_WORKERS
            == frozen_protocol.MAX_WORKERS
        ),
        "input_hash_constants_match_issue49": (
            EXPECTED_INPUT_SHA256 == common.EXPECTED_INPUT_SHA256
        ),
        "seeds_unique": len(seeds) == len(set(seeds)),
        "state_seeds_are_subset": set(state_seeds).issubset(seeds),
        "temperatures_exact": protocol["source_temperatures"] == [
            1.0, 2.0, 3.0, 4.0, 5.0
        ],
        "checkpoints_strict_and_end_at_horizon": (
            checkpoints == sorted(set(checkpoints))
            and checkpoints[-1] == protocol["rounds"]
            and checkpoints[0] > 0
        ),
        "snapshots_strict_and_cover_horizon": (
            snapshots == sorted(set(snapshots))
            and snapshots[0] == 0
            and snapshots[-1] == protocol["rounds"]
        ),
        "late_window_matches_final_checkpoint_block": (
            protocol["late_window_size"]
            == checkpoints[-1] - checkpoints[-2]
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Issue #52 Stage T 协议与实现不一致：{failed}")
    return gates


def _load_inputs():
    target, queries, schema, marginals, hashes = common._load_inputs()
    if hashes != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"Issue #52 输入哈希不一致：{hashes}")
    return target, queries, schema, marginals, hashes


def _build_tasks(protocol):
    state_seeds = set(protocol["state_library_seeds"])
    snapshots = tuple(protocol["snapshot_rounds"])
    return [
        parallel.TrajectoryTask(
            config_id=_config_id(temperature),
            kernel="independent",
            seed=seed,
            rounds=protocol["rounds"],
            temperature=temperature,
            sweeps=0,
            record_state_hashes=False,
            snapshot_rounds=(snapshots if seed in state_seeds else None),
        )
        for seed in protocol["stage_t_seeds"]
        for temperature in protocol["source_temperatures"]
    ]


def _numeric_summary(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise RuntimeError("汇总输入必须是非空有限数值")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _trend_for_run(run, protocol):
    history = np.asarray(
        run["current_loss_after_round_history"], dtype=float
    )
    if len(history) != protocol["rounds"] or not np.all(np.isfinite(history)):
        raise RuntimeError("current-loss history 不完整或包含非有限值")
    windows = {}
    previous = 0
    for checkpoint in protocol["trend_checkpoints"]:
        values = history[previous:checkpoint]
        if len(values) != checkpoint - previous:
            raise RuntimeError("趋势检查点窗口不完整")
        windows[str(checkpoint)] = {
            "start_round": previous + 1,
            "end_round": checkpoint,
            "round_count": len(values),
            "current_loss_mean": float(values.mean()),
            "current_loss_median": float(np.median(values)),
            "current_loss_final": float(values[-1]),
        }
        previous = checkpoint

    final_checkpoint = str(protocol["trend_checkpoints"][-1])
    previous_checkpoint = str(protocol["trend_checkpoints"][-2])
    final_mean = windows[final_checkpoint]["current_loss_mean"]
    previous_mean = windows[previous_checkpoint]["current_loss_mean"]
    relative_change = (
        (final_mean - previous_mean) / abs(previous_mean)
        if previous_mean != 0.0 else 0.0
    )
    clearly_descending = (
        relative_change
        <= -protocol["clear_descent_relative_threshold"]
    )
    return {
        "checkpoint_windows": windows,
        "late_window_start_round": (
            protocol["rounds"] - protocol["late_window_size"] + 1
        ),
        "late_window_end_round": protocol["rounds"],
        "late_window_current_loss_mean": final_mean,
        "last_two_windows_relative_change": float(relative_change),
        "clearly_descending_at_horizon": bool(clearly_descending),
        "horizon_interpretation": (
            "horizon_limited_still_descending"
            if clearly_descending
            else "not_clearly_descending_no_equilibrium_claim"
        ),
    }


def _snapshot_is_valid(run, protocol, should_exist):
    if not should_exist:
        return bool(
            "snapshot_rounds" not in run and "state_snapshots" not in run
        )
    if run.get("snapshot_rounds") != protocol["snapshot_rounds"]:
        return False
    snapshots = run.get("state_snapshots", [])
    if [row.get("state_round") for row in snapshots] != protocol[
        "snapshot_rounds"
    ]:
        return False
    history = run.get("current_loss_after_round_history", [])
    for snapshot in snapshots:
        state_round = snapshot.get("state_round")
        expected_loss = (
            run.get("initial_loss")
            if state_round == 0 else history[state_round - 1]
        )
        try:
            frame = pd.DataFrame(
                snapshot["table_records"],
                columns=snapshot["table_columns"],
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not (
            snapshot.get("snapshot_format")
            == trajectory.CURRENT_SNAPSHOT_FORMAT
            and snapshot.get("source_seed") == run.get("seed")
            and snapshot.get("source_rounds") == protocol["rounds"]
            and snapshot.get("state_kind") == "current"
            and snapshot.get("source_temperature")
            == run.get("temperature")
            and snapshot.get("source_sweeps") == 0
            and len(frame) == trajectory.N_RECORDS
            and _valid_sha256(snapshot.get("state_sha256"))
            and trajectory._frame_sha256(frame)
            == snapshot.get("state_sha256")
            and math.isclose(
                float(snapshot.get("current_loss")),
                float(expected_loss),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def _trajectory_identity_gates(rows, tasks, protocol):
    expected_ids = [task.task_id for task in tasks]
    expected_pairs = [
        (task.seed, float(task.temperature)) for task in tasks
    ]
    actual_pairs = [
        (row["run"].get("seed"), row["run"].get("temperature"))
        for row in rows
    ]
    state_seeds = set(protocol["state_library_seeds"])
    by_seed = {
        seed: [row["run"] for row in rows if row["run"]["seed"] == seed]
        for seed in protocol["stage_t_seeds"]
    }
    row_gates = []
    for row in rows:
        run = row["run"]
        diagnostic = run.get("independent_direction_diagnostics", {})
        row_gates.append(bool(
            row.get("kernel") == "independent"
            and row.get("config_id") == _config_id(run["temperature"])
            and run.get("name") == "independent"
            and run.get("sweeps") == 0
            and run.get("factor_builder") == "not_used"
            and run.get("rounds_run") == protocol["rounds"]
            and len(run.get("current_loss_after_round_history", []))
            == protocol["rounds"]
            and math.isclose(
                run["final_loss"],
                run["current_loss_after_round_history"][-1],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and row.get("trend") == _trend_for_run(run, protocol)
            and diagnostic.get("condition_count", 0) > 0
            and diagnostic.get("all_finite") is True
            and diagnostic.get("all_conditionals_bidirectional") is True
            and len(diagnostic.get("clip_hit_conditions", []))
            == diagnostic.get("clip_hit_count")
            and _valid_sha256(run.get("initial_csv_sha256"))
            and _valid_sha256(run.get("final_csv_sha256"))
            and _valid_sha256(run.get("primary_rng_state_sha256"))
            and _snapshot_is_valid(
                run, protocol, run["seed"] in state_seeds
            )
        ))
    gates = {
        "task_grid_complete_and_ordered": (
            [row.get("task_id") for row in rows] == expected_ids
            and actual_pairs == expected_pairs
            and len(rows) == len(expected_ids)
        ),
        "all_row_identity_gates_passed": all(row_gates),
        "initial_state_aligned_within_seed": all(
            len({run["initial_csv_sha256"] for run in seed_rows}) == 1
            and len({run["initial_loss"] for run in seed_rows}) == 1
            for seed_rows in by_seed.values()
        ),
        "primary_rng_endpoint_aligned_within_seed": all(
            len({run["primary_rng_state_sha256"] for run in seed_rows}) == 1
            for seed_rows in by_seed.values()
        ),
        "direction_reference_scale_aligned_within_seed": all(
            len({run["direction_reference_scale"] for run in seed_rows}) == 1
            and all(
                run["direction_reference_scale"] is not None
                and np.isfinite(run["direction_reference_scale"])
                and run["direction_reference_scale"] > 0.0
                and run["direction_reference_scale_round"] == 0
                for run in seed_rows
            )
            for seed_rows in by_seed.values()
        ),
        "round_zero_snapshots_aligned_within_seed": all(
            len({
                run["state_snapshots"][0]["state_sha256"]
                for run in by_seed[seed]
            }) == 1
            for seed in state_seeds
        ),
    }
    gates["all_identity_gates_passed"] = all(gates.values())
    if not gates["all_identity_gates_passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Issue #52 Stage T 身份门禁失败：{failed}")
    return gates


def _aggregate_stage_t(rows, protocol):
    by_temperature = {}
    for temperature in protocol["source_temperatures"]:
        selected = [
            row for row in rows
            if row["run"]["temperature"] == temperature
        ]
        runs = [row["run"] for row in selected]
        checkpoint_windows = {}
        for checkpoint in protocol["trend_checkpoints"]:
            key = str(checkpoint)
            checkpoint_windows[key] = {
                "start_round": selected[0]["trend"][
                    "checkpoint_windows"
                ][key]["start_round"],
                "end_round": checkpoint,
                "current_loss_mean": _numeric_summary([
                    row["trend"]["checkpoint_windows"][key][
                        "current_loss_mean"
                    ]
                    for row in selected
                ]),
                "current_loss_final": _numeric_summary([
                    row["trend"]["checkpoint_windows"][key][
                        "current_loss_final"
                    ]
                    for row in selected
                ]),
            }
        by_temperature[_tau_key(temperature)] = {
            "temperature": temperature,
            "trajectory_count": len(selected),
            "seeds": [run["seed"] for run in runs],
            "all_rounds_complete": all(
                run["rounds_run"] == protocol["rounds"] for run in runs
            ),
            "checkpoint_windows": checkpoint_windows,
            "late_window_current_loss": _numeric_summary([
                row["trend"]["late_window_current_loss_mean"]
                for row in selected
            ]),
            "final_current_loss": _numeric_summary([
                run["final_loss"] for run in runs
            ]),
            "current_loss_auc": _numeric_summary([
                run["current_loss_auc"] for run in runs
            ]),
            "positive_gain_rate": _numeric_summary([
                run["positive_gain_rate"] for run in runs
            ]),
            "negative_gain_rate": _numeric_summary([
                run["negative_gain_rate"] for run in runs
            ]),
            "mean_positive_gain": _numeric_summary([
                run["mean_positive_gain"] for run in runs
            ]),
            "mean_negative_gain": _numeric_summary([
                run["mean_negative_gain"] for run in runs
            ]),
            "mean_changed_cells": _numeric_summary([
                run["mean_changed_cells"] for run in runs
            ]),
            "last_two_windows_relative_change": _numeric_summary([
                row["trend"]["last_two_windows_relative_change"]
                for row in selected
            ]),
            "clearly_descending_seed_count": sum(
                row["trend"]["clearly_descending_at_horizon"]
                for row in selected
            ),
            "independent_direction_diagnostics": (
                common._aggregate_stage_t_logit(runs)
            ),
        }
    ranking = sorted(
        (
            {
                "config_id": _config_id(temperature),
                "temperature": temperature,
                "late_window_current_loss_mean": by_temperature[
                    _tau_key(temperature)
                ]["late_window_current_loss"]["mean"],
            }
            for temperature in protocol["source_temperatures"]
        ),
        key=lambda row: (
            row["late_window_current_loss_mean"], row["temperature"]
        ),
    )
    return {
        "trajectory_count": len(rows),
        "by_temperature": by_temperature,
        "fixed_3000_round_ranking_diagnostic_only": (
            ranking if protocol["mode"] == "formal" else []
        ),
        "no_candidate_selected_in_stage_t": True,
    }


def _state_source_manifest(rows, protocol):
    state_seeds = set(protocol["state_library_seeds"])
    selected = [row["run"] for row in rows if row["run"]["seed"] in state_seeds]
    raw_snapshot_count = sum(len(run["state_snapshots"]) for run in selected)
    expected_unique = len(state_seeds) * (
        1
        + len(protocol["source_temperatures"])
        * (len(protocol["snapshot_rounds"]) - 1)
    )
    return {
        "status": "source_snapshots_complete_not_yet_materialized",
        "state_library_seeds": protocol["state_library_seeds"],
        "source_temperatures": protocol["source_temperatures"],
        "snapshot_rounds": protocol["snapshot_rounds"],
        "raw_snapshot_count_in_trajectories": raw_snapshot_count,
        "expected_unique_current_states_after_round0_dedup": expected_unique,
        "round0_dedup_rule": "one shared initial state per seed",
    }


def run_stage_t(mode, output_dir, *, max_workers):
    protocol = _protocol(mode)
    workers = validate_max_workers(max_workers)
    implementation_gates = _validate_implementation_constants(protocol)
    output_directory = Path(output_dir)
    report_path = output_directory / "stage_t_report.json"
    if report_path.exists():
        raise FileExistsError(
            f"输出已存在，尚未启动任何 Stage T 轨迹：{report_path}"
        )
    git = common._git_identity()
    if mode == "formal" and not git["worktree_clean"]:
        raise RuntimeError("正式 Issue #52 Stage T 要求工作树干净（含未跟踪文件）")
    target, queries, schema, marginals, input_hashes = _load_inputs()
    protocol_sha256 = common._canonical_sha256({
        "protocol": protocol,
        "input_sha256": input_hashes,
        "git_commit": git["commit"],
    })
    tasks = _build_tasks(protocol)

    started = time.perf_counter()
    rows = parallel.run_trajectory_tasks(
        target,
        queries,
        schema,
        marginals,
        tasks,
        max_workers=workers,
    )
    for row in rows:
        row["trend"] = _trend_for_run(row["run"], protocol)
    identity_gates = _trajectory_identity_gates(rows, tasks, protocol)
    aggregates = _aggregate_stage_t(rows, protocol)
    state_manifest = _state_source_manifest(rows, protocol)
    formal_identity_gates = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _protocol("formal"),
        "worktree_clean_at_start": git["worktree_clean"],
        "input_hashes_match": input_hashes == EXPECTED_INPUT_SHA256,
        "worker_count_within_frozen_cap": 1 <= workers <= 8,
    }
    formal_result_valid = bool(
        all(formal_identity_gates.values())
        and identity_gates["all_identity_gates_passed"]
    )
    report = {
        "report_format": REPORT_FORMAT,
        "status": "complete",
        "experiment": "issue52_independent_low_temperature_long_horizon_stage_t",
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "interpretation": (
            "formal_preregistered_stage_t"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "input_sha256": input_hashes,
        "git": git,
        "command_argv": list(sys.argv),
        "environment": trajectory._environment(protocol["device"]),
        "implementation_gates": implementation_gates,
        "execution": {
            "requested_max_workers": workers,
            "effective_max_workers": min(workers, len(tasks)),
            "worker_count_is_nonscientific": True,
            "task_count": len(tasks),
            "task_order": [task.task_id for task in tasks],
            "trajectory_scientific_sha256": scientific_sha256(rows),
        },
        "stage_t": {
            "identity_gates": identity_gates,
            "aggregates": aggregates,
            "trajectories": rows,
        },
        "state_library_source_manifest": state_manifest,
        "formal_identity_gates": formal_identity_gates,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    common._write_json_atomic(report_path, report)
    return report_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), default="smoke"
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/issue52_low_temperature_long_horizon/stage_t_smoke"
        ),
    )
    args = parser.parse_args()
    if args.mode == "formal" and args.output_dir.endswith("stage_t_smoke"):
        parser.error("正式模式必须显式提供非 smoke 输出目录")
    try:
        validate_max_workers(args.max_workers)
    except ValueError as exc:
        parser.error(str(exc))

    output, report = run_stage_t(
        args.mode,
        args.output_dir,
        max_workers=args.max_workers,
    )
    print("\n===== Issue #52 Stage T =====")
    print(f"mode={report['mode']}")
    print(f"workers={report['execution']['requested_max_workers']}")
    print(f"trajectories={report['execution']['task_count']}")
    print(
        "identity_gates="
        f"{report['stage_t']['identity_gates']['all_identity_gates_passed']}"
    )
    print(f"formal_result_valid={report['formal_result_valid']}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
