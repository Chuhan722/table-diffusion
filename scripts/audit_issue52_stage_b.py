"""Independent JSON audit for the frozen Issue #52 Stage B report."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from table_diffevo.experiment_parallel import (
    MAX_EXPERIMENT_WORKERS,
    scientific_sha256,
    validate_max_workers,
)

if __package__:
    from scripts import audit_issue52_stage_t as stage_t_auditor
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue52_protocol as frozen_protocol
    from scripts import run_issue49_stage_a as common
else:
    import audit_issue52_stage_t as stage_t_auditor
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue52_protocol as frozen_protocol
    import run_issue49_stage_a as common


AUDIT_FORMAT = "issue52_stage_b_audit_v1"
REPORT_FORMAT = "issue52_stage_b_report_v1"


def _assert_same(actual, expected, path):
    if actual != expected:
        raise RuntimeError(f"Stage B 审计重算不一致：{path}")


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _independent_config_id(temperature):
    return f"independent_{_tau_key(temperature)}"


def _factor_config_id(temperature, sweeps):
    return f"factor_{_tau_key(temperature)}_sweeps_{sweeps}"


def _factor_config(temperature, sweeps):
    return {
        "config_id": _factor_config_id(temperature, sweeps),
        "kernel": "factor",
        "temperature": float(temperature),
        "sweeps": int(sweeps),
    }


def _valid_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _implementation_gates(protocol):
    stage_t = frozen_protocol.stage_t_protocol(protocol["mode"])
    return {
        "trajectory_rho_matches": trajectory.RHO == protocol["rho"],
        "trajectory_eta_matches": trajectory.ETA == protocol["eta"],
        "trajectory_mu_matches": trajectory.MU == protocol["trajectory_mu"],
        "trajectory_logit_clip_matches": (
            trajectory.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
        "numpy_cpu_only": protocol["device"] == "numpy",
        "compiled_factor_builder_frozen": (
            protocol["factor_builder"] == "compiled_batch"
        ),
        "worker_cap_matches": (
            protocol["max_workers_allowed"]
            == MAX_EXPERIMENT_WORKERS
            == frozen_protocol.MAX_WORKERS
        ),
        "stage_t_horizon_exact": all((
            protocol["stage_b_seeds"] == stage_t["stage_t_seeds"],
            protocol["rounds"] == stage_t["rounds"],
            protocol["trend_checkpoints"] == stage_t["trend_checkpoints"],
            protocol["late_window_size"] == stage_t["late_window_size"],
            protocol["primary_metric"] == stage_t["primary_metric"],
        )),
        "ranking_tie_breakers_exact": (
            protocol["ranking_tie_breakers"] == [
                "late_window_current_loss_mean",
                "fewer_sweeps",
                "lower_temperature",
            ]
        ),
        "snapshots_disabled": protocol["snapshots_disabled"] is True,
        "input_hashes_frozen": (
            protocol["input_sha256"]
            == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
    }


def _load_existing_json(path, label):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} 不存在：{resolved}")
    return resolved, common._load_json_strict(resolved)


def _validate_stage_t_upstream(report, protocol):
    upstream = report.get("upstream", {})
    report_file, stage_t_report = _load_existing_json(
        upstream.get("stage_t_report_path"), "Stage T report"
    )
    audit_file, audit = _load_existing_json(
        upstream.get("stage_t_audit_path"), "Stage T audit"
    )
    mode = protocol["mode"]
    expected_protocol = frozen_protocol.stage_t_protocol(mode)
    report_sha = common._sha256_file(report_file)
    audit_sha = common._sha256_file(audit_file)
    expected_artifacts = protocol["expected_stage_t_artifact_sha256"]
    checks = {
        "report_format_exact": (
            stage_t_report.get("report_format")
            == protocol["stage_t_report_format"]
        ),
        "audit_format_exact": (
            audit.get("audit_format") == protocol["stage_t_audit_format"]
        ),
        "report_complete": stage_t_report.get("status") == "complete",
        "audit_complete_and_passed": (
            audit.get("status") == "complete" and audit.get("passed") is True
        ),
        "mode_exact": stage_t_report.get("mode") == mode,
        "protocol_exact": stage_t_report.get("protocol") == expected_protocol,
        "input_hashes_exact": (
            stage_t_report.get("input_sha256") == protocol["input_sha256"]
            and audit.get("input_sha256") == protocol["input_sha256"]
        ),
        "audit_report_path_exact": (
            Path(audit.get("report_path", "")).resolve() == report_file
        ),
        "audit_report_sha_exact": audit.get("report_sha256") == report_sha,
        "audit_protocol_sha_exact": (
            audit.get("protocol_sha256")
            == stage_t_report.get("protocol_sha256")
        ),
        "audited_aggregates_exact": (
            audit.get("recomputed", {}).get("aggregates")
            == stage_t_report.get("stage_t", {}).get("aggregates")
        ),
        "audited_trajectory_sha_exact": (
            audit.get("recomputed", {}).get(
                "trajectory_scientific_sha256"
            )
            == stage_t_report.get("execution", {}).get(
                "trajectory_scientific_sha256"
            )
        ),
        "formal_flags_when_required": bool(
            mode != "formal"
            or (
                stage_t_report.get("formal_result_valid") is True
                and audit.get("formal_result_valid") is True
            )
        ),
        "formal_artifact_sha_when_required": bool(
            mode != "formal"
            or (
                report_sha == expected_artifacts["report"]
                and audit_sha == expected_artifacts["audit"]
            )
        ),
    }
    _assert_same(
        upstream.get("stage_t_report_sha256"),
        report_sha,
        "upstream.stage_t_report_sha256",
    )
    _assert_same(
        upstream.get("stage_t_audit_sha256"),
        audit_sha,
        "upstream.stage_t_audit_sha256",
    )
    _assert_same(
        upstream.get("stage_t_protocol_sha256"),
        stage_t_report.get("protocol_sha256"),
        "upstream.stage_t_protocol_sha256",
    )
    _assert_same(
        upstream.get("stage_t_checks"), checks, "upstream.stage_t_checks"
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage T 上游独立复审失败：{failed}")
    return stage_t_report, {
        "report_sha256": report_sha,
        "audit_sha256": audit_sha,
        "checks": checks,
    }


def _validate_stage_a_upstream(report, protocol):
    upstream = report.get("upstream", {})
    report_file = Path(upstream.get("stage_a_report_path", "")).resolve()
    if not report_file.is_file():
        raise FileNotFoundError(f"Stage A report 不存在：{report_file}")
    audit_file, audit = _load_existing_json(
        upstream.get("stage_a_audit_path"), "Stage A audit"
    )
    mode = protocol["mode"]
    report_sha = common._sha256_file(report_file)
    audit_sha = common._sha256_file(audit_file)
    expected_artifacts = protocol["expected_stage_a_artifact_sha256"]
    selection = audit.get("selection")
    checks = {
        "audit_format_exact": (
            audit.get("audit_format") == protocol["stage_a_audit_format"]
        ),
        "audit_complete_and_passed": (
            audit.get("status") == "complete" and audit.get("passed") is True
        ),
        "audit_report_path_exact": (
            Path(audit.get("report_path", "")).resolve() == report_file
        ),
        "audit_report_sha_exact": audit.get("report_sha256") == report_sha,
        "selection_present": isinstance(selection, dict),
        "formal_flags_when_required": bool(
            mode != "formal" or audit.get("formal_result_valid") is True
        ),
        "formal_artifact_sha_when_required": bool(
            mode != "formal"
            or (
                report_sha == expected_artifacts["report"]
                and audit_sha == expected_artifacts["audit"]
            )
        ),
        "formal_protocol_sha_when_required": bool(
            mode != "formal"
            or audit.get("protocol_sha256")
            == protocol["expected_stage_a_protocol_sha256"]
        ),
        "formal_scientific_sha_when_required": bool(
            mode != "formal"
            or audit.get("execution_scientific_sha256")
            == protocol["expected_stage_a_scientific_sha256"]
        ),
        "formal_selection_when_required": bool(
            mode != "formal"
            or selection == protocol["expected_formal_stage_a_selection"]
        ),
    }
    if mode == "smoke":
        stage_a_report = common._load_json_strict(report_file)
        checks.update({
            "smoke_report_format_exact": (
                stage_a_report.get("report_format")
                == protocol["stage_a_report_format"]
            ),
            "smoke_report_complete": (
                stage_a_report.get("status") == "complete"
            ),
            "smoke_mode_exact": stage_a_report.get("mode") == "smoke",
            "smoke_protocol_exact": (
                stage_a_report.get("protocol")
                == frozen_protocol.stage_a_mixing_protocol("smoke")
            ),
            "smoke_input_hashes_exact": (
                stage_a_report.get("input_sha256")
                == protocol["input_sha256"]
            ),
            "smoke_selection_matches_audit": (
                stage_a_report.get("selection") == selection
            ),
            "smoke_protocol_sha_matches_audit": (
                stage_a_report.get("protocol_sha256")
                == audit.get("protocol_sha256")
            ),
            "smoke_scientific_sha_matches_audit": (
                stage_a_report.get("execution_scientific_sha256")
                == audit.get("execution_scientific_sha256")
            ),
        })
    _assert_same(
        upstream.get("stage_a_report_sha256"),
        report_sha,
        "upstream.stage_a_report_sha256",
    )
    _assert_same(
        upstream.get("stage_a_audit_sha256"),
        audit_sha,
        "upstream.stage_a_audit_sha256",
    )
    _assert_same(
        upstream.get("stage_a_protocol_sha256"),
        audit.get("protocol_sha256"),
        "upstream.stage_a_protocol_sha256",
    )
    _assert_same(
        upstream.get("stage_a_execution_scientific_sha256"),
        audit.get("execution_scientific_sha256"),
        "upstream.stage_a_execution_scientific_sha256",
    )
    _assert_same(
        upstream.get("stage_a_selection"),
        selection,
        "upstream.stage_a_selection",
    )
    _assert_same(
        upstream.get("stage_a_checks"), checks, "upstream.stage_a_checks"
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage A 上游独立复审失败：{failed}")
    return selection, {
        "report_sha256": report_sha,
        "audit_sha256": audit_sha,
        "protocol_sha256": audit.get("protocol_sha256"),
        "execution_scientific_sha256": audit.get(
            "execution_scientific_sha256"
        ),
        "checks": checks,
    }


def _factor_configs_from_selection(selection, protocol):
    temperatures = frozen_protocol.STAGE_A_EVALUATION_TEMPERATURES
    keys = [_tau_key(temperature) for temperature in temperatures]
    mapping = selection.get("minimal_sufficient_sweeps", {})
    if list(mapping) != keys:
        raise RuntimeError("审计发现 Stage A tau 身份缺失、重复或乱序")
    qualified = [
        float(temperature) for temperature in temperatures
        if mapping[_tau_key(temperature)] is not None
    ]
    unqualified = [
        float(temperature) for temperature in temperatures
        if mapping[_tau_key(temperature)] is None
    ]
    if (
        selection.get("qualified_temperatures") != qualified
        or selection.get("unqualified_temperatures") != unqualified
    ):
        raise RuntimeError("审计发现 Stage A 资格列表与 sweeps 冲突")
    allowed = set(frozen_protocol.STAGE_A_CANDIDATE_SWEEPS)
    configs = []
    for temperature in temperatures:
        sweeps = mapping[_tau_key(temperature)]
        if sweeps is None:
            continue
        if (
            isinstance(sweeps, bool)
            or not isinstance(sweeps, int)
            or sweeps not in allowed
        ):
            raise RuntimeError("审计发现非法最小充分 sweeps")
        configs.append(_factor_config(temperature, sweeps))
    if (
        protocol["mode"] == "formal"
        and selection != protocol["expected_formal_stage_a_selection"]
    ):
        raise RuntimeError("审计发现正式 Stage A selection 偏离冻结结果")
    return configs


def _stage_t_i_star(stage_t_report, protocol):
    aggregates = stage_t_report["stage_t"]["aggregates"]
    by_temperature = aggregates["by_temperature"]
    ranking = []
    for temperature in frozen_protocol.INDEPENDENT_TEMPERATURES:
        aggregate = by_temperature.get(_tau_key(temperature), {})
        if (
            aggregate.get("temperature") != float(temperature)
            or aggregate.get("seeds") != protocol["stage_b_seeds"]
            or aggregate.get("all_rounds_complete") is not True
        ):
            raise RuntimeError("审计发现 Stage T independent 聚合身份不完整")
        ranking.append({
            "config_id": _independent_config_id(temperature),
            "temperature": float(temperature),
            "late_window_current_loss_mean": aggregate[
                "late_window_current_loss"
            ]["mean"],
        })
    ranking.sort(key=lambda row: (
        row["late_window_current_loss_mean"], row["temperature"]
    ))
    if (
        protocol["mode"] == "formal"
        and aggregates["fixed_3000_round_ranking_diagnostic_only"] != ranking
    ):
        raise RuntimeError("审计发现 Stage T 正式排序与聚合不一致")
    i_star = ranking[0]["config_id"]
    if (
        protocol["mode"] == "formal"
        and i_star != protocol["expected_formal_i_star"]
    ):
        raise RuntimeError("审计发现正式 I* 偏离冻结 Stage T 结果")
    return i_star, ranking


def _expected_tasks(configs, protocol):
    return [
        {
            "task_id": f"seed_{seed}__{config['config_id']}",
            "config_id": config["config_id"],
            "kernel": "factor",
            "seed": seed,
            "temperature": config["temperature"],
            "sweeps": config["sweeps"],
        }
        for seed in protocol["stage_b_seeds"]
        for config in configs
    ]


def _diagnostic_valid(diagnostic, *, require_conditions):
    count = diagnostic.get("condition_count")
    hits = diagnostic.get("clip_hit_count")
    return bool(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= int(require_conditions)
        and isinstance(hits, int)
        and not isinstance(hits, bool)
        and hits >= 0
        and len(diagnostic.get("clip_hit_conditions", [])) == hits
        and diagnostic.get("all_finite") is True
        and diagnostic.get("all_conditionals_bidirectional") is True
        and diagnostic.get("raw_logit_strictly_inside_clip")
        == (diagnostic.get("raw_logit_abs_max", 0.0) < trajectory.GIBBS_LOGIT_CLIP)
    )


def _stage_t_rows_by_seed_and_temperature(stage_t_report, protocol):
    rows = stage_t_report["stage_t"]["trajectories"]
    expected = [
        (seed, float(temperature))
        for seed in protocol["stage_b_seeds"]
        for temperature in frozen_protocol.INDEPENDENT_TEMPERATURES
    ]
    actual = [
        (row.get("run", {}).get("seed"), row.get("run", {}).get("temperature"))
        for row in rows
    ]
    if actual != expected:
        raise RuntimeError("审计发现 Stage T 轨迹网格不完整或乱序")
    return {
        (row["run"]["seed"], row["run"]["temperature"]): row
        for row in rows
    }


def _identity_gates(rows, configs, protocol, stage_t_report):
    tasks = _expected_tasks(configs, protocol)
    baseline = _stage_t_rows_by_seed_and_temperature(stage_t_report, protocol)
    row_gates = []
    for row, task in zip(rows, tasks):
        run = row.get("run", {})
        same_tau = baseline.get((task["seed"], task["temperature"]), {}).get(
            "run", {}
        )
        history = run.get("current_loss_after_round_history", [])
        trend = stage_t_auditor._recompute_trend(run, protocol)
        _assert_same(row.get("trend"), trend, f"trajectory.{task['task_id']}.trend")
        independent_diagnostic = run.get(
            "independent_direction_diagnostics", {}
        )
        factor_diagnostic = run.get(
            "factor_conditional_logit_diagnostics", {}
        )
        row_gates.append(bool(
            row.get("task_id") == task["task_id"]
            and row.get("config_id") == task["config_id"]
            and row.get("kernel") == "factor"
            and run.get("seed") == task["seed"]
            and run.get("temperature") == task["temperature"]
            and run.get("sweeps") == task["sweeps"]
            and run.get("name") == f"gibbs_{task['sweeps']}_sweeps"
            and run.get("factor_builder") == protocol["factor_builder"]
            and run.get("rounds_run") == protocol["rounds"]
            and len(history) == protocol["rounds"]
            and len(run.get("gain_history", [])) == protocol["rounds"]
            and len(run.get("changed_cells_history", []))
            == protocol["rounds"]
            and len(run.get("loss_history", [])) == protocol["rounds"]
            and math.isclose(
                float(run.get("final_loss")),
                float(history[-1]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and _diagnostic_valid(
                independent_diagnostic, require_conditions=True
            )
            and _diagnostic_valid(factor_diagnostic, require_conditions=True)
            and factor_diagnostic.get("condition_count")
            == run.get("gibbs_microsteps")
            and run.get("gibbs_microsteps", 0) > 0
            and run.get("compiled_unique_conditions", 0) > 0
            and _valid_sha256(run.get("initial_csv_sha256"))
            and _valid_sha256(run.get("final_csv_sha256"))
            and _valid_sha256(run.get("primary_rng_state_sha256"))
            and _valid_sha256(run.get("gibbs_rng_state_sha256"))
            and run.get("state_sha256_history") == []
            and "snapshot_rounds" not in run
            and "state_snapshots" not in run
            and run.get("initial_csv_sha256")
            == same_tau.get("initial_csv_sha256")
            and run.get("initial_loss") == same_tau.get("initial_loss")
            and run.get("primary_rng_state_sha256")
            == same_tau.get("primary_rng_state_sha256")
            and run.get("direction_reference_scale")
            == same_tau.get("direction_reference_scale")
            and run.get("direction_reference_scale_round")
            == same_tau.get("direction_reference_scale_round") == 0
        ))
    by_seed = {
        seed: [
            row["run"] for row in rows if row["run"].get("seed") == seed
        ]
        for seed in protocol["stage_b_seeds"]
    }
    has_configs = bool(configs)
    gates = {
        "factor_task_grid_complete_and_ordered": bool(
            len(rows) == len(tasks)
            and [row.get("task_id") for row in rows]
            == [task["task_id"] for task in tasks]
        ),
        "all_factor_row_identities_valid": all(row_gates),
        "initial_state_aligned_within_seed": bool(
            not has_configs or all(
                len(seed_rows) == len(configs)
                and len({run["initial_csv_sha256"] for run in seed_rows}) == 1
                and len({run["initial_loss"] for run in seed_rows}) == 1
                for seed_rows in by_seed.values()
            )
        ),
        "primary_rng_endpoint_aligned_within_seed": bool(
            not has_configs or all(
                len({run["primary_rng_state_sha256"] for run in seed_rows})
                == 1
                for seed_rows in by_seed.values()
            )
        ),
        "direction_reference_scale_aligned_within_seed": bool(
            not has_configs or all(
                len({run["direction_reference_scale"] for run in seed_rows})
                == 1
                and all(
                    run["direction_reference_scale"] is not None
                    and np.isfinite(run["direction_reference_scale"])
                    and run["direction_reference_scale"] > 0.0
                    and run["direction_reference_scale_round"] == 0
                    for run in seed_rows
                )
                for seed_rows in by_seed.values()
            )
        ),
        "all_numeric_values_finite": common._all_numeric_finite(rows),
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


def _aggregate_logit(runs, field, protocol):
    diagnostics = [run[field] for run in runs]
    count = int(sum(row["condition_count"] for row in diagnostics))
    hits = int(sum(row["clip_hit_count"] for row in diagnostics))
    nonempty = [
        (run, row) for run, row in zip(runs, diagnostics)
        if row["condition_count"]
    ]
    if not count or not nonempty:
        raise RuntimeError("审计发现 Stage B 条件 logit 诊断为空")
    maximum_run, maximum = max(
        nonempty, key=lambda item: item[1]["raw_logit_abs_max"]
    )
    hit_conditions = [
        {
            "seed": run["seed"],
            "temperature": run["temperature"],
            **condition,
        }
        for run, diagnostic in zip(runs, diagnostics)
        for condition in diagnostic["clip_hit_conditions"]
    ]
    result = {
        "condition_count": count,
        "raw_logit_min": float(min(row["raw_logit_min"] for _, row in nonempty)),
        "raw_logit_max": float(max(row["raw_logit_max"] for _, row in nonempty)),
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
            row["conditional_probability_min"] for _, row in nonempty
        )),
        "conditional_probability_max": float(max(
            row["conditional_probability_max"] for _, row in nonempty
        )),
        "minimum_binary_outcome_probability": float(min(
            row["minimum_binary_outcome_probability"]
            for _, row in nonempty
        )),
        "conditional_entropy_mean": float(sum(
            row["conditional_entropy_mean"] * row["condition_count"]
            for row in diagnostics
        ) / count),
        "all_finite": all(row["all_finite"] for row in diagnostics),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"] for row in diagnostics
        ),
    }
    if all("negative_direction_count" in row for row in diagnostics):
        negative_count = int(sum(
            row["negative_direction_count"] for row in diagnostics
        ))
        positive_count = int(sum(
            row["positive_direction_count"] for row in diagnostics
        ))
        result.update({
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
        })
    return result


def _aggregate_config(config, wrappers, protocol):
    wrappers = sorted(wrappers, key=lambda row: row["run"]["seed"])
    runs = [row["run"] for row in wrappers]
    checkpoint_windows = {}
    for checkpoint in protocol["trend_checkpoints"]:
        key = str(checkpoint)
        checkpoint_windows[key] = {
            "start_round": wrappers[0]["trend"]["checkpoint_windows"][key][
                "start_round"
            ],
            "end_round": checkpoint,
            "current_loss_mean": _summary([
                row["trend"]["checkpoint_windows"][key]["current_loss_mean"]
                for row in wrappers
            ]),
            "current_loss_final": _summary([
                row["trend"]["checkpoint_windows"][key]["current_loss_final"]
                for row in wrappers
            ]),
        }
    independent_logit = _aggregate_logit(
        runs, "independent_direction_diagnostics", protocol
    )
    factor_logit = _aggregate_logit(
        runs, "factor_conditional_logit_diagnostics", protocol
    )
    return {
        "config": dict(config),
        "trajectory_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "all_rounds_complete": all(
            run["rounds_run"] == protocol["rounds"] for run in runs
        ),
        "checkpoint_windows": checkpoint_windows,
        "late_window_current_loss": _summary([
            row["trend"]["late_window_current_loss_mean"]
            for row in wrappers
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
        "gibbs_microsteps": _summary([
            run["gibbs_microsteps"] for run in runs
        ]),
        "last_two_windows_relative_change": _summary([
            row["trend"]["last_two_windows_relative_change"]
            for row in wrappers
        ]),
        "clearly_descending_seed_count": sum(
            row["trend"]["clearly_descending_at_horizon"]
            for row in wrappers
        ),
        "independent_direction_diagnostics": independent_logit,
        "factor_conditional_logit_diagnostics": factor_logit,
        "total_clip_hit_count": int(
            independent_logit["clip_hit_count"]
            + factor_logit["clip_hit_count"]
        ),
        "all_conditionals_finite_and_bidirectional": bool(
            independent_logit["all_finite"]
            and independent_logit["all_conditionals_bidirectional"]
            and factor_logit["all_finite"]
            and factor_logit["all_conditionals_bidirectional"]
        ),
    }


def _aggregates(rows, configs, protocol):
    return {
        config["config_id"]: _aggregate_config(
            config,
            [row for row in rows if row["config_id"] == config["config_id"]],
            protocol,
        )
        for config in configs
    }


def _metric_value(wrapper, metric):
    if metric == "late_window_current_loss":
        return wrapper["trend"]["late_window_current_loss_mean"]
    if metric == "final_current_loss":
        return wrapper["run"]["final_loss"]
    if metric == "current_loss_auc":
        return wrapper["run"]["current_loss_auc"]
    raise ValueError(f"未知 Stage B 配对指标：{metric}")


def _paired_metric(candidate, baseline, seeds, metric):
    candidate_map = {row["run"]["seed"]: row for row in candidate}
    baseline_map = {row["run"]["seed"]: row for row in baseline}
    if set(candidate_map) != set(seeds) or set(baseline_map) != set(seeds):
        raise RuntimeError("审计发现 Stage B 配对 seed 不完整")
    candidate_values = np.asarray([
        _metric_value(candidate_map[seed], metric) for seed in seeds
    ], dtype=float)
    baseline_values = np.asarray([
        _metric_value(baseline_map[seed], metric) for seed in seeds
    ], dtype=float)
    differences = candidate_values - baseline_values
    return {
        "candidate": _summary(candidate_values),
        "baseline": _summary(baseline_values),
        "candidate_minus_baseline": _summary(differences),
        "by_seed": [
            {
                "seed": seed,
                "candidate": float(candidate_value),
                "baseline": float(baseline_value),
                "difference": float(difference),
            }
            for seed, candidate_value, baseline_value, difference in zip(
                seeds, candidate_values, baseline_values, differences
            )
        ],
        "wins": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "losses": int(np.sum(differences > 0.0)),
    }


def _comparison(candidate, baseline, candidate_id, baseline_id, seeds):
    return {
        "candidate_config_id": candidate_id,
        "baseline_config_id": baseline_id,
        "seeds": list(seeds),
        "metrics": {
            metric: _paired_metric(candidate, baseline, seeds, metric)
            for metric in (
                "late_window_current_loss",
                "final_current_loss",
                "current_loss_auc",
            )
        },
    }


def _comparisons(rows, configs, stage_t_report, i_star, protocol):
    stage_t_by_id = {
        _independent_config_id(temperature): [
            row for row in stage_t_report["stage_t"]["trajectories"]
            if row["run"]["temperature"] == float(temperature)
        ]
        for temperature in frozen_protocol.INDEPENDENT_TEMPERATURES
    }
    results = {}
    for config in configs:
        config_id = config["config_id"]
        candidate = [row for row in rows if row["config_id"] == config_id]
        same_tau_id = _independent_config_id(config["temperature"])
        results[config_id] = {
            "same_temperature_independent": _comparison(
                candidate,
                stage_t_by_id[same_tau_id],
                config_id,
                same_tau_id,
                protocol["stage_b_seeds"],
            ),
            "i_star": _comparison(
                candidate,
                stage_t_by_id[i_star],
                config_id,
                i_star,
                protocol["stage_b_seeds"],
            ),
        }
    return results


def _selection(configs, aggregates, comparisons, i_star, protocol):
    if not configs:
        return {
            "i_star": i_star,
            "factor_ranking": [],
            "g_star": None,
            "g_star_config": None,
            "candidate_gates": {
                "eligible_factor_exists": False,
                "point_estimate_better_than_same_tau_independent": False,
                "point_estimate_better_than_i_star": False,
            },
            "stage_c_candidate": None,
            "stage_c_allowed": False,
            "status": "no_eligible_factor",
            "no_reselection": True,
        }
    ranking = sorted(
        ({
            "config_id": config["config_id"],
            "temperature": config["temperature"],
            "sweeps": config["sweeps"],
            "late_window_current_loss_mean": aggregates[
                config["config_id"]
            ]["late_window_current_loss"]["mean"],
        } for config in configs),
        key=lambda row: (
            row["late_window_current_loss_mean"],
            row["sweeps"],
            row["temperature"],
        ),
    )
    g_star = ranking[0]["config_id"]
    by_id = {config["config_id"]: config for config in configs}
    same_tau = comparisons[g_star]["same_temperature_independent"][
        "metrics"
    ]["late_window_current_loss"]
    vs_i_star = comparisons[g_star]["i_star"]["metrics"][
        "late_window_current_loss"
    ]
    gates = {
        "eligible_factor_exists": True,
        "point_estimate_better_than_same_tau_independent": (
            same_tau["candidate"]["mean"] < same_tau["baseline"]["mean"]
        ),
        "point_estimate_better_than_i_star": (
            vs_i_star["candidate"]["mean"]
            < vs_i_star["baseline"]["mean"]
        ),
    }
    allowed = all(gates.values())
    return {
        "i_star": i_star,
        "factor_ranking": ranking,
        "g_star": g_star,
        "g_star_config": dict(by_id[g_star]),
        "candidate_gates": gates,
        "stage_c_candidate": g_star if allowed else None,
        "stage_c_allowed": allowed,
        "status": (
            "factor_candidate_selected" if allowed
            else "no_factor_candidate"
        ),
        "no_reselection": protocol["no_reselection"],
    }


def audit_stage_b(report_path, output_path):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(f"审计输出已存在，不覆盖：{output_file}")
    report = common._load_json_strict(report_file)
    if report.get("report_format") != REPORT_FORMAT:
        raise RuntimeError("未知 Issue #52 Stage B report_format")
    if report.get("status") != "complete":
        raise RuntimeError("Issue #52 Stage B 报告未完整结束")
    mode = report.get("mode")
    protocol = frozen_protocol.stage_b_protocol(mode)
    _assert_same(report.get("protocol"), protocol, "report.protocol")
    _assert_same(
        report.get("experiment"),
        "issue52_stage_b_factor_long_horizon_comparison",
        "report.experiment",
    )
    _assert_same(
        report.get("interpretation"),
        (
            "formal_preregistered_stage_b"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "report.interpretation",
    )
    implementation = _implementation_gates(protocol)
    if not all(implementation.values()):
        failed = [name for name, passed in implementation.items() if not passed]
        raise RuntimeError(f"Stage B 审计实现常量门禁失败：{failed}")
    _assert_same(
        report.get("implementation_gates"),
        implementation,
        "report.implementation_gates",
    )
    stage_t_report, stage_t_identity = _validate_stage_t_upstream(
        report, protocol
    )
    stage_a_selection, stage_a_identity = _validate_stage_a_upstream(
        report, protocol
    )
    configs = _factor_configs_from_selection(stage_a_selection, protocol)
    _assert_same(
        report.get("factor_configurations"),
        configs,
        "report.factor_configurations",
    )
    i_star, independent_ranking = _stage_t_i_star(stage_t_report, protocol)
    _assert_same(
        report.get("independent_reference"),
        {"i_star": i_star, "ranking": independent_ranking},
        "report.independent_reference",
    )
    _, _, _, _, input_hashes = common._load_inputs()
    _assert_same(report.get("input_sha256"), input_hashes, "report.input")
    upstream_hashes = {
        "stage_t_report": stage_t_identity["report_sha256"],
        "stage_t_audit": stage_t_identity["audit_sha256"],
        "stage_a_report": stage_a_identity["report_sha256"],
        "stage_a_audit": stage_a_identity["audit_sha256"],
    }
    expected_protocol_sha = common._canonical_sha256({
        "protocol": protocol,
        "input_sha256": input_hashes,
        "upstream_sha256": upstream_hashes,
        "git_commit": report.get("git", {}).get("commit"),
    })
    _assert_same(
        report.get("protocol_sha256"),
        expected_protocol_sha,
        "report.protocol_sha256",
    )
    rows = report.get("stage_b", {}).get("factor_trajectories", [])
    identity = _identity_gates(rows, configs, protocol, stage_t_report)
    if not identity["all_identity_gates_passed"]:
        failed = [name for name, passed in identity.items() if not passed]
        raise RuntimeError(f"Stage B 审计轨迹身份门禁失败：{failed}")
    _assert_same(
        report.get("stage_b", {}).get("identity_gates"),
        identity,
        "report.stage_b.identity_gates",
    )
    aggregates = _aggregates(rows, configs, protocol)
    _assert_same(
        report.get("stage_b", {}).get("factor_aggregates"),
        aggregates,
        "report.stage_b.factor_aggregates",
    )
    comparisons = _comparisons(
        rows, configs, stage_t_report, i_star, protocol
    )
    _assert_same(
        report.get("stage_b", {}).get("comparisons"),
        comparisons,
        "report.stage_b.comparisons",
    )
    selection = _selection(
        configs, aggregates, comparisons, i_star, protocol
    )
    _assert_same(
        report.get("stage_b", {}).get("selection"),
        selection,
        "report.stage_b.selection",
    )
    tasks = _expected_tasks(configs, protocol)
    execution = report.get("execution", {})
    workers = execution.get("requested_max_workers")
    try:
        validated_workers = validate_max_workers(workers)
    except ValueError:
        validated_workers = None
    execution_checks = {
        "worker_count_valid": validated_workers is not None,
        "effective_worker_count_exact": (
            execution.get("effective_max_workers")
            == (min(validated_workers, len(tasks)) if tasks else 0)
            if validated_workers is not None else False
        ),
        "worker_count_marked_nonscientific": (
            execution.get("worker_count_is_nonscientific") is True
        ),
        "task_count_exact": execution.get("task_count") == len(tasks),
        "task_order_exact": execution.get("task_order") == [
            task["task_id"] for task in tasks
        ],
        "trajectory_scientific_sha256_exact": (
            execution.get("trajectory_scientific_sha256")
            == scientific_sha256(rows)
        ),
    }
    if not all(execution_checks.values()):
        failed = [name for name, passed in execution_checks.items() if not passed]
        raise RuntimeError(f"Stage B 审计执行身份门禁失败：{failed}")
    formal_identity = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": (
            protocol == frozen_protocol.stage_b_protocol("formal")
        ),
        "worktree_clean_at_start": report.get("git", {}).get(
            "worktree_clean"
        ) is True,
        "input_hashes_exact": input_hashes == frozen_protocol.EXPECTED_INPUT_SHA256,
        "stage_t_artifacts_exact": bool(
            mode == "formal"
            and upstream_hashes["stage_t_report"]
            == frozen_protocol.FORMAL_STAGE_T_ARTIFACT_SHA256["report"]
            and upstream_hashes["stage_t_audit"]
            == frozen_protocol.FORMAL_STAGE_T_ARTIFACT_SHA256["audit"]
        ),
        "stage_a_artifacts_exact": bool(
            mode == "formal"
            and upstream_hashes["stage_a_report"]
            == frozen_protocol.FORMAL_STAGE_A_MIXING_ARTIFACT_SHA256["report"]
            and upstream_hashes["stage_a_audit"]
            == frozen_protocol.FORMAL_STAGE_A_MIXING_ARTIFACT_SHA256["audit"]
        ),
        "stage_a_selection_exact": bool(
            mode == "formal"
            and stage_a_selection
            == frozen_protocol.FORMAL_STAGE_A_MIXING_SELECTION
        ),
        "i_star_exact": bool(
            mode == "formal" and i_star == protocol["expected_formal_i_star"]
        ),
        "worker_count_within_frozen_cap": validated_workers is not None,
    }
    _assert_same(
        report.get("formal_identity_gates"),
        formal_identity,
        "report.formal_identity_gates",
    )
    formal_result_valid = bool(
        all(formal_identity.values())
        and identity["all_identity_gates_passed"]
    )
    _assert_same(
        report.get("formal_result_valid"),
        formal_result_valid,
        "report.formal_result_valid",
    )
    current_git = common._git_identity()
    checks = {
        "report_status_complete": True,
        "frozen_protocol_exact": True,
        "input_hashes_match_files": (
            input_hashes == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "report_commit_matches_current_checkout": (
            report.get("git", {}).get("commit") == current_git["commit"]
        ),
        "current_worktree_clean_when_formal": (
            mode != "formal" or current_git["worktree_clean"]
        ),
        "implementation_gates_passed": all(implementation.values()),
        "upstream_stage_t_bound_and_rechecked": all(
            stage_t_identity["checks"].values()
        ),
        "upstream_stage_a_bound_and_rechecked": all(
            stage_a_identity["checks"].values()
        ),
        "factor_configuration_selection_recomputed": True,
        "trajectory_identity_gates_passed": identity[
            "all_identity_gates_passed"
        ],
        "aggregates_and_pairing_recomputed": True,
        "i_star_g_star_and_candidate_gate_recomputed": True,
        "execution_gates_passed": all(execution_checks.values()),
        "formal_flag_exact": (
            report.get("formal_result_valid") == formal_result_valid
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Issue #52 Stage B 独立审计失败：{failed}")
    audit = {
        "audit_format": AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "formal_result_valid": formal_result_valid,
        "report_path": str(report_file),
        "report_sha256": common._sha256_file(report_file),
        "protocol_sha256": expected_protocol_sha,
        "input_sha256": input_hashes,
        "upstream_sha256": upstream_hashes,
        "git": current_git,
        "checks": checks,
        "execution_checks": execution_checks,
        "recomputed": {
            "factor_configurations": configs,
            "independent_reference": {
                "i_star": i_star,
                "ranking": independent_ranking,
            },
            "trajectory_identity_gates": identity,
            "trajectory_scientific_sha256": scientific_sha256(rows),
            "factor_aggregates": aggregates,
            "comparisons": comparisons,
            "selection": selection,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    common._write_json_atomic(output_file, audit)
    return output_file, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_stage_b(args.report, args.output)
    print("\n===== Issue #52 Stage B Audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"report_sha256={audit['report_sha256']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
