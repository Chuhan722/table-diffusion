"""Independent JSON audit for the frozen Issue #52 Stage T report."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue52_protocol as frozen_protocol
    from scripts import run_issue49_stage_a as common
else:
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue52_protocol as frozen_protocol
    import run_issue49_stage_a as common

from table_diffevo.experiment_parallel import (
    MAX_EXPERIMENT_WORKERS,
    scientific_sha256,
)


AUDIT_FORMAT = "issue52_stage_t_audit_v1"
REPORT_FORMAT = "issue52_stage_t_report_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _assert_same(actual, expected, path):
    if actual != expected:
        raise RuntimeError(f"审计重算不一致：{path}")


def _valid_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _config_id(temperature):
    return f"independent_{_tau_key(temperature)}"


def _actual_input_hashes():
    paths = {
        "schema": REPOSITORY_ROOT / trajectory.SCHEMA_PATH,
        "queries": REPOSITORY_ROOT / trajectory.QUERY_PATH,
        "marginals": REPOSITORY_ROOT / trajectory.MARGINALS_PATH,
    }
    return {name: common._sha256_file(path) for name, path in paths.items()}


def _implementation_gates(protocol):
    checkpoints = protocol["trend_checkpoints"]
    snapshots = protocol["snapshot_rounds"]
    seeds = protocol["stage_t_seeds"]
    state_seeds = protocol["state_library_seeds"]
    return {
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
            frozen_protocol.EXPECTED_INPUT_SHA256
            == common.EXPECTED_INPUT_SHA256
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


def _expected_tasks(protocol):
    return [
        {
            "task_id": f"seed_{seed}__{_config_id(temperature)}",
            "config_id": _config_id(temperature),
            "kernel": "independent",
            "seed": seed,
            "temperature": float(temperature),
            "sweeps": 0,
        }
        for seed in protocol["stage_t_seeds"]
        for temperature in protocol["source_temperatures"]
    ]


def _recompute_trend(run, protocol):
    history = np.asarray(
        run.get("current_loss_after_round_history", []), dtype=float
    )
    if len(history) != protocol["rounds"] or not np.all(np.isfinite(history)):
        raise RuntimeError("审计发现 current-loss history 不完整或非有限")
    windows = {}
    previous = 0
    for checkpoint in protocol["trend_checkpoints"]:
        values = history[previous:checkpoint]
        if len(values) != checkpoint - previous:
            raise RuntimeError("审计发现趋势窗口不完整")
        windows[str(checkpoint)] = {
            "start_round": previous + 1,
            "end_round": checkpoint,
            "round_count": len(values),
            "current_loss_mean": float(values.mean()),
            "current_loss_median": float(np.median(values)),
            "current_loss_final": float(values[-1]),
        }
        previous = checkpoint
    final_key = str(protocol["trend_checkpoints"][-1])
    previous_key = str(protocol["trend_checkpoints"][-2])
    final_mean = windows[final_key]["current_loss_mean"]
    previous_mean = windows[previous_key]["current_loss_mean"]
    relative_change = (
        (final_mean - previous_mean) / abs(previous_mean)
        if previous_mean != 0.0 else 0.0
    )
    descending = (
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
        "clearly_descending_at_horizon": bool(descending),
        "horizon_interpretation": (
            "horizon_limited_still_descending"
            if descending
            else "not_clearly_descending_no_equilibrium_claim"
        ),
    }


def _snapshot_valid(
    run, protocol, should_exist, *, target, queries, schema
):
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
        counts, _, _ = trajectory.evaluate_vectorized(
            frame,
            queries,
            schema,
            target=target,
            n_records=trajectory.N_RECORDS,
            batch_size=256,
            device="numpy",
            want_fitness=False,
            verbose=False,
        )
        recomputed_loss = float(trajectory.compute_loss(target, counts))
        expected_alpha_round = min(state_round, protocol["rounds"] - 1)
        expected_alpha = trajectory._donor_alpha(
            expected_alpha_round, protocol["rounds"]
        )
        if not (
            snapshot.get("snapshot_format")
            == trajectory.CURRENT_SNAPSHOT_FORMAT
            and snapshot.get("source_seed") == run.get("seed")
            and snapshot.get("source_rounds") == protocol["rounds"]
            and snapshot.get("state_kind") == "current"
            and snapshot.get("source_temperature")
            == run.get("temperature")
            and snapshot.get("source_sweeps") == 0
            and snapshot.get("donor_alpha") == expected_alpha
            and snapshot.get("direction_reference_scale")
            == run.get("direction_reference_scale")
            and snapshot.get("direction_reference_scale_round")
            == run.get("direction_reference_scale_round")
            and len(frame) == trajectory.N_RECORDS
            and list(frame.columns) == schema.attribute_names()
            and _valid_sha256(snapshot.get("state_sha256"))
            and trajectory._frame_sha256(frame)
            == snapshot.get("state_sha256")
            and _valid_sha256(snapshot.get("primary_rng_state_sha256"))
            and snapshot.get("gibbs_rng_state_sha256") is None
            and math.isclose(
                float(snapshot.get("current_loss")),
                float(expected_loss),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                recomputed_loss,
                float(snapshot.get("current_loss")),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and (
                state_round != 0
                or snapshot.get("state_sha256")
                == run.get("initial_csv_sha256")
            )
            and (
                state_round != protocol["rounds"]
                or (
                    snapshot.get("state_sha256")
                    == run.get("final_csv_sha256")
                    and snapshot.get("primary_rng_state_sha256")
                    == run.get("primary_rng_state_sha256")
                )
            )
        ):
            return False
    return True


def _identity_gates(rows, protocol, *, target, queries, schema):
    expected = _expected_tasks(protocol)
    state_seeds = set(protocol["state_library_seeds"])
    by_seed = {
        seed: [row["run"] for row in rows if row["run"].get("seed") == seed]
        for seed in protocol["stage_t_seeds"]
    }
    row_gates = []
    for row, task in zip(rows, expected):
        run = row.get("run", {})
        diagnostic = run.get("independent_direction_diagnostics", {})
        trend = _recompute_trend(run, protocol)
        _assert_same(row.get("trend"), trend, f"trajectory.{task['task_id']}.trend")
        history = run.get("current_loss_after_round_history", [])
        row_gates.append(bool(
            row.get("task_id") == task["task_id"]
            and row.get("config_id") == task["config_id"]
            and row.get("kernel") == "independent"
            and run.get("seed") == task["seed"]
            and run.get("temperature") == task["temperature"]
            and run.get("sweeps") == 0
            and run.get("name") == "independent"
            and run.get("factor_builder") == "not_used"
            and run.get("rounds_run") == protocol["rounds"]
            and len(history) == protocol["rounds"]
            and math.isclose(
                float(run.get("final_loss")),
                float(history[-1]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and diagnostic.get("condition_count", 0) > 0
            and diagnostic.get("all_finite") is True
            and diagnostic.get("all_conditionals_bidirectional") is True
            and len(diagnostic.get("clip_hit_conditions", []))
            == diagnostic.get("clip_hit_count")
            and _valid_sha256(run.get("initial_csv_sha256"))
            and _valid_sha256(run.get("final_csv_sha256"))
            and _valid_sha256(run.get("primary_rng_state_sha256"))
            and run.get("gibbs_rng_state_sha256") is None
            and _snapshot_valid(
                run,
                protocol,
                run.get("seed") in state_seeds,
                target=target,
                queries=queries,
                schema=schema,
            )
        ))
    gates = {
        "task_grid_complete_and_ordered": bool(
            len(rows) == len(expected)
            and [row.get("task_id") for row in rows]
            == [task["task_id"] for task in expected]
            and [
                (row.get("run", {}).get("seed"),
                 row.get("run", {}).get("temperature"))
                for row in rows
            ] == [
                (task["seed"], task["temperature"]) for task in expected
            ]
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
    return gates


def _summary(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise RuntimeError("审计汇总遇到空值或非有限数值")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _logit_summary(runs, protocol):
    diagnostics = [run["independent_direction_diagnostics"] for run in runs]
    count = int(sum(row["condition_count"] for row in diagnostics))
    hits = int(sum(row["clip_hit_count"] for row in diagnostics))
    nonempty = [row for row in diagnostics if row["condition_count"]]
    if not count or not nonempty:
        raise RuntimeError("审计发现独立方向诊断为空")
    maximum_run = max(
        runs,
        key=lambda run: run["independent_direction_diagnostics"][
            "raw_logit_abs_max"
        ],
    )
    maximum = maximum_run["independent_direction_diagnostics"]
    hit_conditions = [
        {
            "seed": run["seed"],
            "temperature": run["temperature"],
            **condition,
        }
        for run in runs
        for condition in run["independent_direction_diagnostics"][
            "clip_hit_conditions"
        ]
    ]
    negative_count = int(sum(
        row["negative_direction_count"] for row in diagnostics
    ))
    positive_count = int(sum(
        row["positive_direction_count"] for row in diagnostics
    ))
    return {
        "condition_count": count,
        "raw_logit_min": float(min(row["raw_logit_min"] for row in nonempty)),
        "raw_logit_max": float(max(row["raw_logit_max"] for row in nonempty)),
        "raw_logit_abs_max": float(maximum["raw_logit_abs_max"]),
        "raw_logit_abs_max_condition": {
            "seed": maximum_run["seed"],
            "temperature": maximum_run["temperature"],
            **maximum["raw_logit_abs_max_condition"],
        },
        "logit_clip": float(protocol["logit_clip"]),
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count),
        "clip_hit_conditions": hit_conditions,
        "raw_logit_strictly_inside_clip": all(
            row["raw_logit_strictly_inside_clip"] for row in diagnostics
        ),
        "conditional_probability_min": float(min(
            row["conditional_probability_min"] for row in nonempty
        )),
        "conditional_probability_max": float(max(
            row["conditional_probability_max"] for row in nonempty
        )),
        "minimum_binary_outcome_probability": float(min(
            row["minimum_binary_outcome_probability"] for row in nonempty
        )),
        "conditional_entropy_mean": float(sum(
            row["conditional_entropy_mean"] * row["condition_count"]
            for row in nonempty
        ) / count),
        "negative_direction_count": negative_count,
        "negative_direction_copy_probability": (
            float(sum(
                row["negative_direction_copy_probability"]
                * row["negative_direction_count"]
                for row in diagnostics if row["negative_direction_count"]
            ) / negative_count) if negative_count else None
        ),
        "positive_direction_count": positive_count,
        "positive_direction_copy_probability": (
            float(sum(
                row["positive_direction_copy_probability"]
                * row["positive_direction_count"]
                for row in diagnostics if row["positive_direction_count"]
            ) / positive_count) if positive_count else None
        ),
        "all_finite": all(row["all_finite"] for row in diagnostics),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"] for row in diagnostics
        ),
    }


def _aggregates(rows, protocol):
    by_temperature = {}
    for temperature in protocol["source_temperatures"]:
        selected = [
            row for row in rows if row["run"]["temperature"] == temperature
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
                "current_loss_mean": _summary([
                    row["trend"]["checkpoint_windows"][key][
                        "current_loss_mean"
                    ] for row in selected
                ]),
                "current_loss_final": _summary([
                    row["trend"]["checkpoint_windows"][key][
                        "current_loss_final"
                    ] for row in selected
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
            "late_window_current_loss": _summary([
                row["trend"]["late_window_current_loss_mean"]
                for row in selected
            ]),
            "final_current_loss": _summary([
                run["final_loss"] for run in runs
            ]),
            "current_loss_auc": _summary([
                run["current_loss_auc"] for run in runs
            ]),
            "positive_gain_rate": _summary([
                run["positive_gain_rate"] for run in runs
            ]),
            "negative_gain_rate": _summary([
                run["negative_gain_rate"] for run in runs
            ]),
            "mean_positive_gain": _summary([
                run["mean_positive_gain"] for run in runs
            ]),
            "mean_negative_gain": _summary([
                run["mean_negative_gain"] for run in runs
            ]),
            "mean_changed_cells": _summary([
                run["mean_changed_cells"] for run in runs
            ]),
            "last_two_windows_relative_change": _summary([
                row["trend"]["last_two_windows_relative_change"]
                for row in selected
            ]),
            "clearly_descending_seed_count": sum(
                row["trend"]["clearly_descending_at_horizon"]
                for row in selected
            ),
            "independent_direction_diagnostics": _logit_summary(
                runs, protocol
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


def _state_manifest(rows, protocol):
    state_seeds = set(protocol["state_library_seeds"])
    selected = [row["run"] for row in rows if row["run"]["seed"] in state_seeds]
    return {
        "status": "source_snapshots_complete_not_yet_materialized",
        "state_library_seeds": protocol["state_library_seeds"],
        "source_temperatures": protocol["source_temperatures"],
        "snapshot_rounds": protocol["snapshot_rounds"],
        "raw_snapshot_count_in_trajectories": sum(
            len(run["state_snapshots"]) for run in selected
        ),
        "expected_unique_current_states_after_round0_dedup": (
            len(state_seeds) * (
                1 + len(protocol["source_temperatures"])
                * (len(protocol["snapshot_rounds"]) - 1)
            )
        ),
        "round0_dedup_rule": "one shared initial state per seed",
    }


def audit_stage_t(report_path, audit_path):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    audit_file = Path(audit_path)
    if audit_file.exists():
        raise FileExistsError(f"审计输出已存在，不覆盖：{audit_file}")
    report = common._load_json_strict(report_file)
    if report.get("report_format") != REPORT_FORMAT:
        raise RuntimeError("未知 Issue #52 Stage T report_format")
    mode = report.get("mode")
    protocol = frozen_protocol.stage_t_protocol(mode)
    _assert_same(report.get("protocol"), protocol, "report.protocol")
    actual_inputs = _actual_input_hashes()
    _assert_same(
        report.get("input_sha256"), actual_inputs, "report.input_sha256"
    )
    expected_protocol_sha256 = common._canonical_sha256({
        "protocol": protocol,
        "input_sha256": actual_inputs,
        "git_commit": report.get("git", {}).get("commit"),
    })
    _assert_same(
        report.get("protocol_sha256"),
        expected_protocol_sha256,
        "report.protocol_sha256",
    )
    implementation = _implementation_gates(protocol)
    if not all(implementation.values()):
        failed = [name for name, passed in implementation.items() if not passed]
        raise RuntimeError(f"审计实现常量门禁失败：{failed}")
    _assert_same(
        report.get("implementation_gates"),
        implementation,
        "report.implementation_gates",
    )
    target, queries, schema, _, loaded_hashes = common._load_inputs()
    _assert_same(loaded_hashes, actual_inputs, "audit.loaded_input_sha256")

    rows = report.get("stage_t", {}).get("trajectories", [])
    identity = _identity_gates(
        rows,
        protocol,
        target=target,
        queries=queries,
        schema=schema,
    )
    if not identity["all_identity_gates_passed"]:
        failed = [name for name, passed in identity.items() if not passed]
        raise RuntimeError(f"审计轨迹身份门禁失败：{failed}")
    _assert_same(
        report.get("stage_t", {}).get("identity_gates"),
        identity,
        "report.stage_t.identity_gates",
    )
    aggregates = _aggregates(rows, protocol)
    _assert_same(
        report.get("stage_t", {}).get("aggregates"),
        aggregates,
        "report.stage_t.aggregates",
    )
    manifest = _state_manifest(rows, protocol)
    _assert_same(
        report.get("state_library_source_manifest"),
        manifest,
        "report.state_library_source_manifest",
    )

    expected_tasks = _expected_tasks(protocol)
    execution = report.get("execution", {})
    workers = execution.get("requested_max_workers")
    execution_checks = {
        "worker_count_valid": bool(
            isinstance(workers, int)
            and not isinstance(workers, bool)
            and 1 <= workers <= protocol["max_workers_allowed"]
        ),
        "effective_worker_count_exact": (
            execution.get("effective_max_workers")
            == min(workers, len(expected_tasks))
            if isinstance(workers, int) and not isinstance(workers, bool)
            else False
        ),
        "worker_count_marked_nonscientific": (
            execution.get("worker_count_is_nonscientific") is True
        ),
        "task_count_exact": execution.get("task_count") == len(expected_tasks),
        "task_order_exact": execution.get("task_order") == [
            task["task_id"] for task in expected_tasks
        ],
        "trajectory_scientific_sha256_exact": (
            execution.get("trajectory_scientific_sha256")
            == scientific_sha256(rows)
        ),
    }
    if not all(execution_checks.values()):
        failed = [name for name, passed in execution_checks.items() if not passed]
        raise RuntimeError(f"审计执行身份门禁失败：{failed}")

    formal_gates = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol
        == frozen_protocol.stage_t_protocol("formal"),
        "worktree_clean_at_start": report.get("git", {}).get(
            "worktree_clean"
        ) is True,
        "input_hashes_match": (
            actual_inputs == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "worker_count_within_frozen_cap": execution_checks[
            "worker_count_valid"
        ],
    }
    _assert_same(
        report.get("formal_identity_gates"),
        formal_gates,
        "report.formal_identity_gates",
    )
    formal_result_valid = bool(
        all(formal_gates.values())
        and identity["all_identity_gates_passed"]
    )
    _assert_same(
        report.get("formal_result_valid"),
        formal_result_valid,
        "report.formal_result_valid",
    )
    current_git = common._git_identity()
    checks = {
        "report_status_complete": report.get("status") == "complete",
        "interpretation_exact": report.get("interpretation") == (
            "formal_preregistered_stage_t"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "input_hashes_match_files": (
            actual_inputs == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "report_commit_matches_current_checkout": (
            report.get("git", {}).get("commit") == current_git["commit"]
        ),
        "current_worktree_clean_when_formal": (
            mode != "formal" or current_git["worktree_clean"]
        ),
        "implementation_gates_passed": all(implementation.values()),
        "trajectory_identity_gates_passed": identity[
            "all_identity_gates_passed"
        ],
        "execution_gates_passed": all(execution_checks.values()),
        "formal_flag_exact": (
            report.get("formal_result_valid") == formal_result_valid
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Issue #52 Stage T 独立审计失败：{failed}")

    audit = {
        "audit_format": AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "formal_result_valid": formal_result_valid,
        "report_path": str(report_file),
        "report_sha256": common._sha256_file(report_file),
        "protocol_sha256": expected_protocol_sha256,
        "input_sha256": actual_inputs,
        "git": current_git,
        "checks": checks,
        "execution_checks": execution_checks,
        "recomputed": {
            "trajectory_identity_gates": identity,
            "trajectory_scientific_sha256": scientific_sha256(rows),
            "aggregates": aggregates,
            "state_library_source_manifest": manifest,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    common._write_json_atomic(audit_file, audit)
    return audit_file, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_stage_t(args.report, args.output)
    print("\n===== Issue #52 Stage T Audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"report_sha256={audit['report_sha256']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
