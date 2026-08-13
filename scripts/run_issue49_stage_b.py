"""运行 Issue #49 协议 v2 的 Stage B 长期无门控对照与候选冻结。"""

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

if __package__:
    from scripts import audit_issue49_stage_a as stage_a_auditor
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue49_stage_t_a_protocol as frozen_protocol
    from scripts import probe_factorized_gibbs_mixing as probe
    from scripts import run_issue49_stage_a as stage_a
else:
    import audit_issue49_stage_a as stage_a_auditor
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue49_stage_t_a_protocol as frozen_protocol
    import probe_factorized_gibbs_mixing as probe
    import run_issue49_stage_a as stage_a


REPORT_FORMAT = "issue49_stage_b_report_v1"
UPSTREAM_REPORT_FORMAT = "issue49_stage_t_a_report_v2"
UPSTREAM_AUDIT_FORMAT = "issue49_stage_t_a_audit_v2"
SELF_REVIEW_FAMILIES = ("initial", "mid", "late")
SELF_REVIEW_GROUPS = ("global", *SELF_REVIEW_FAMILIES)


def _config_id(kind, temperature, sweeps):
    tau = f"{temperature:g}".replace(".", "p")
    if kind == "independent":
        return f"independent_tau_{tau}"
    return f"factor_tau_{tau}_sweeps_{sweeps}"


def _config(kind, temperature, sweeps):
    return {
        "config_id": _config_id(kind, temperature, sweeps),
        "kernel": kind,
        "temperature": float(temperature),
        "sweeps": int(sweeps),
    }


def _protocol(mode):
    return frozen_protocol.stage_b_protocol(mode)


def _validate_implementation_constants(protocol):
    gates = {
        "trajectory_rho_matches": trajectory.RHO == protocol["rho"],
        "trajectory_eta_matches": trajectory.ETA == protocol["eta"],
        "trajectory_mu_matches": (
            trajectory.MU == protocol["trajectory_mu"]
        ),
        "trajectory_logit_clip_matches": (
            trajectory.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
        "probe_rho_matches": probe.RHO == protocol["rho"],
        "probe_eta_matches": probe.ETA == protocol["eta"],
        "probe_mutation_disabled": (
            protocol["self_review_probe_mu"] == 0.0
        ),
        "probe_logit_clip_matches": (
            probe.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
        "probe_max_active_attributes_matches": (
            protocol["self_review_max_active_attributes"] == 12
        ),
        "factor_builder_is_frozen_legacy_path": (
            protocol["factor_builder"] == "legacy_rowwise"
        ),
        "self_review_groups_match": (
            protocol["self_review_required_groups"]
            == list(SELF_REVIEW_GROUPS)
        ),
        "ranking_tie_breakers_match": (
            protocol["ranking_tie_breakers"] == [
                "late_window_current_loss_mean",
                "late_window_current_loss_median",
                "current_loss_auc_mean",
                "final_current_loss_mean",
                "fewer_sweeps",
                "lower_temperature",
            ]
        ),
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Stage B 冻结协议与实现不一致：{failed}")
    return gates


def _load_upstream(report_path, audit_path, mode, current_git):
    report_file = Path(report_path).resolve()
    audit_file = Path(audit_path).resolve()
    report = stage_a._load_json_strict(report_file)
    audit = stage_a._load_json_strict(audit_file)
    expected_protocol = frozen_protocol.protocol(mode)
    checks = {
        "report_format": report.get("report_format") == UPSTREAM_REPORT_FORMAT,
        "audit_format": audit.get("audit_format") == UPSTREAM_AUDIT_FORMAT,
        "report_complete": report.get("status") == "complete",
        "audit_complete_and_passed": (
            audit.get("status") == "complete" and audit.get("passed") is True
        ),
        "mode_matches": (
            report.get("mode") == mode and audit.get("mode") == mode
        ),
        "protocol_exact": report.get("protocol") == expected_protocol,
        "input_hashes_exact": (
            report.get("input_sha256")
            == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "audit_report_path_matches": (
            Path(audit.get("report_path", "")).resolve() == report_file
        ),
        "audit_report_hash_matches": (
            audit.get("report_sha256") == stage_a._sha256_file(report_file)
        ),
        "protocol_hash_matches": (
            audit.get("protocol_sha256") == report.get("protocol_sha256")
        ),
        "selection_matches_audit": (
            audit.get("selection") == report.get("a1", {}).get("selection")
        ),
        "same_git_commit": (
            report.get("git", {}).get("commit") == current_git["commit"]
        ),
    }
    library_path = Path(
        report.get("state_library", {}).get("path", "")
    ).resolve()
    checks.update({
        "state_library_path_matches_audit": (
            Path(audit.get("state_library_path", "")).resolve()
            == library_path
        ),
        "state_library_exists": library_path.is_file(),
        "state_library_hash_matches": (
            library_path.is_file()
            and stage_a._sha256_file(library_path)
            == report.get("state_library", {}).get("sha256")
            == audit.get("state_library_sha256")
        ),
        "formal_upstream_valid_when_required": (
            mode != "formal"
            or (
                report.get("formal_result_valid") is True
                and audit.get("formal_result_valid") is True
            )
        ),
        "stage_t_a_semantic_gates_passed": (
            report.get("stage_t", {}).get("identity_gates", {}).get(
                "all_identity_gates_passed"
            ) is True
            and report.get("common_semantic_gates", {}).get(
                "all_common_semantic_gates_passed"
            ) is True
            and all(
                result.get("all_correctness_gates_passed") is True
                for result in report.get("a1", {}).get(
                    "correctness_gates", {}
                ).values()
                if result.get("applicable")
            )
        ),
    })
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Stage T/A 上游身份或审计无效：{failed}")
    with tempfile.TemporaryDirectory(
        prefix="issue49-stage-b-upstream-audit-"
    ) as temporary:
        _, fresh_audit = stage_a_auditor.audit_stage_a(
            report_file,
            library_path,
            Path(temporary) / "stage_t_a_audit.json",
        )
    checks["fresh_stage_t_a_reaudit_passed"] = bool(
        fresh_audit["passed"] is True
        and fresh_audit["report_sha256"] == stage_a._sha256_file(report_file)
        and fresh_audit["state_library_sha256"]
        == stage_a._sha256_file(library_path)
        and fresh_audit["protocol_sha256"] == report["protocol_sha256"]
        and fresh_audit["formal_result_valid"]
        == audit["formal_result_valid"]
        and fresh_audit["selection"] == audit["selection"]
    )
    if not checks["fresh_stage_t_a_reaudit_passed"]:
        raise RuntimeError("Stage T/A 上游独立复审失败")
    return report_file, audit_file, report, audit, checks


def _factor_configs_from_stage_a(stage_a_report):
    protocol = stage_a_report["protocol"]
    a0 = stage_a_report["a0"]["classification"]["temperatures"]
    selection = stage_a_report["a1"]["selection"]["temperatures"]
    configs = []
    for temperature in protocol["evaluation_temperatures"]:
        key = stage_a._tau_key(temperature)
        result = selection[key]
        sweep = result["minimal_sufficient_sweeps"]
        if sweep is None:
            if result["status"] == "sufficient_within_grid":
                raise RuntimeError(f"{key} A1 状态与 sweeps 冲突")
            continue
        candidates = result["candidates"]
        passed = [row["sweeps"] for row in candidates if row["passed"]]
        if (
            not a0[key]["eligible_for_mixing"]
            or result["status"] != "sufficient_within_grid"
            or sweep not in protocol["candidate_sweeps"]
            or not passed
            or sweep != passed[0]
        ):
            raise RuntimeError(f"{key} A1 最小充分 sweeps 身份无效")
        configs.append(_config("factor", temperature, sweep))
    return configs


def _require_stage_b_allowed(stage_a_report):
    """在任何长期轨迹前执行 Stage A 冻结停止规则。"""
    protocol = stage_a_report["protocol"]
    classification = stage_a_report["a0"]["classification"]
    temperatures = classification["temperatures"]
    recomputed = [
        temperature
        for temperature in protocol["evaluation_temperatures"]
        if temperatures[stage_a._tau_key(temperature)][
            "eligible_for_mixing"
        ]
    ]
    if classification.get("eligible_temperatures") != recomputed:
        raise RuntimeError("Stage A 的 A0 合格温度身份不一致")
    if not recomputed:
        raise RuntimeError(
            "A0 全部 tau 不合格；按冻结停止规则不得运行 Stage B"
        )
    return recomputed


def _all_configs(protocol, stage_a_report):
    _require_stage_b_allowed(stage_a_report)
    independent = [
        _config("independent", temperature, 0)
        for temperature in protocol["independent_temperatures"]
    ]
    return independent + _factor_configs_from_stage_a(stage_a_report)


def _run_trajectories(target, queries, schema, marginals, protocol, configs):
    rows = []
    for seed in protocol["stage_b_seeds"]:
        for config in configs:
            rows.append({
                "config_id": config["config_id"],
                "kernel": config["kernel"],
                "run": trajectory._run_one(
                    target,
                    queries,
                    schema,
                    marginals,
                    seed=seed,
                    rounds=protocol["rounds"],
                    temperature=config["temperature"],
                    sweeps=config["sweeps"],
                    device=protocol["device"],
                    factor_builder=protocol["factor_builder"],
                    snapshot_rounds=(
                        protocol["snapshot_rounds"]
                        if config["kernel"] == "factor" else None
                    ),
                ),
            })
    return rows


def _validate_logit_diagnostic(diagnostic, *, expect_conditions):
    count = diagnostic.get("condition_count")
    hits = diagnostic.get("clip_hit_count")
    return bool(
        isinstance(count, int)
        and isinstance(hits, int)
        and count >= (1 if expect_conditions else 0)
        and hits >= 0
        and len(diagnostic.get("clip_hit_conditions", [])) == hits
        and diagnostic.get("logit_clip") == 30.0
        and diagnostic.get("raw_logit_strictly_inside_clip")
        == (count == 0 or diagnostic.get("raw_logit_abs_max") < 30.0)
        and diagnostic.get("all_finite") is True
        and diagnostic.get("all_conditionals_bidirectional") is True
    )


def _trajectory_identity_gates(rows, configs, protocol):
    specs = {row["config_id"]: row for row in configs}
    expected = {
        (seed, config_id)
        for seed in protocol["stage_b_seeds"]
        for config_id in specs
    }
    actual = {
        (row["run"].get("seed"), row.get("config_id")) for row in rows
    }
    by_seed = {
        seed: [row["run"] for row in rows if row["run"]["seed"] == seed]
        for seed in protocol["stage_b_seeds"]
    }
    row_gates = []
    for wrapper in rows:
        run = wrapper["run"]
        spec = specs.get(wrapper["config_id"])
        is_factor = spec is not None and spec["kernel"] == "factor"
        history = np.asarray(
            run.get("current_loss_after_round_history", []), dtype=float
        )
        late = history[-protocol["late_window_size"]:]
        factor_logit = run.get("factor_conditional_logit_diagnostics", {})
        row_gates.append(bool(
            spec is not None
            and wrapper.get("kernel") == spec["kernel"]
            and run.get("temperature") == spec["temperature"]
            and run.get("sweeps") == spec["sweeps"]
            and run.get("name") == (
                f"gibbs_{spec['sweeps']}_sweeps"
                if is_factor else "independent"
            )
            and run.get("factor_builder") == (
                protocol["factor_builder"] if is_factor else "not_used"
            )
            and run.get("rounds_run") == protocol["rounds"]
            and len(history) == protocol["rounds"]
            and np.all(np.isfinite(history))
            and len(late) == min(
                protocol["late_window_size"], protocol["rounds"]
            )
            and math.isclose(
                float(late.mean()),
                run.get("late_window_current_loss_mean"),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and _validate_logit_diagnostic(
                run.get("independent_direction_diagnostics", {}),
                expect_conditions=True,
            )
            and _validate_logit_diagnostic(
                factor_logit, expect_conditions=is_factor
            )
            and factor_logit.get("condition_count")
            == run.get("gibbs_microsteps")
            and (
                run.get("snapshot_rounds") == protocol["snapshot_rounds"]
                and [
                    snapshot["state_round"]
                    for snapshot in run.get("state_snapshots", [])
                ] == protocol["snapshot_rounds"]
                if is_factor else (
                    "snapshot_rounds" not in run
                    and "state_snapshots" not in run
                )
            )
        ))
    gates = {
        "trajectory_grid_complete": len(rows) == len(expected) and actual == expected,
        "all_row_identities_valid": all(row_gates),
        "all_initial_states_aligned_within_seed": all(
            len({row["initial_csv_sha256"] for row in seed_rows}) == 1
            and len({row["initial_loss"] for row in seed_rows}) == 1
            for seed_rows in by_seed.values()
        ),
        "all_primary_rng_endpoints_aligned_within_seed": all(
            len({row["primary_rng_state_sha256"] for row in seed_rows}) == 1
            for seed_rows in by_seed.values()
        ),
        "all_direction_scales_aligned_within_seed": all(
            len({row["direction_reference_scale"] for row in seed_rows}) == 1
            and all(
                row["direction_reference_scale"] is not None
                and np.isfinite(row["direction_reference_scale"])
                and row["direction_reference_scale"] > 0.0
                and row["direction_reference_scale_round"] == 0
                for row in seed_rows
            )
            for seed_rows in by_seed.values()
        ),
        "all_numeric_values_finite": stage_a._all_numeric_finite(rows),
    }
    gates["all_trajectory_identity_gates_passed"] = all(gates.values())
    if not gates["all_trajectory_identity_gates_passed"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Stage B 轨迹身份门禁失败：{failed}")
    return gates


def _aggregate_trajectory_logit(rows, field):
    values = [row[field] for row in rows]
    count = int(sum(value["condition_count"] for value in values))
    hits = int(sum(value["clip_hit_count"] for value in values))
    hit_conditions = [
        {"seed": run["seed"], **condition}
        for run, value in zip(rows, values)
        for condition in value["clip_hit_conditions"]
    ]
    nonempty = [
        (run, value) for run, value in zip(rows, values)
        if value["condition_count"]
    ]
    if nonempty:
        maximum_run, maximum = max(
            nonempty, key=lambda item: item[1]["raw_logit_abs_max"]
        )
        maximum_context = {
            "seed": maximum_run["seed"],
            **maximum["raw_logit_abs_max_condition"],
        }
        entropy = float(sum(
            value["conditional_entropy_mean"] * value["condition_count"]
            for _, value in nonempty
        ) / count)
    else:
        maximum = None
        maximum_context = None
        entropy = None
    result = {
        "condition_count": count,
        "raw_logit_min": (
            float(min(value["raw_logit_min"] for _, value in nonempty))
            if nonempty else None
        ),
        "raw_logit_max": (
            float(max(value["raw_logit_max"] for _, value in nonempty))
            if nonempty else None
        ),
        "raw_logit_abs_max": (
            float(maximum["raw_logit_abs_max"]) if maximum else 0.0
        ),
        "raw_logit_abs_max_condition": maximum_context,
        "logit_clip": 30.0,
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "clip_hit_conditions": hit_conditions,
        "raw_logit_strictly_inside_clip": bool(
            count == 0 or maximum["raw_logit_abs_max"] < 30.0
        ),
        "conditional_probability_min": (
            float(min(
                value["conditional_probability_min"]
                for _, value in nonempty
            )) if nonempty else None
        ),
        "conditional_probability_max": (
            float(max(
                value["conditional_probability_max"]
                for _, value in nonempty
            )) if nonempty else None
        ),
        "minimum_binary_outcome_probability": (
            float(min(
                value["minimum_binary_outcome_probability"]
                for _, value in nonempty
            )) if nonempty else None
        ),
        "conditional_entropy_mean": entropy,
        "all_finite": all(value["all_finite"] for value in values),
        "all_conditionals_bidirectional": all(
            value["all_conditionals_bidirectional"] for value in values
        ),
    }
    negative_count = int(sum(
        value.get("negative_direction_count", 0) for value in values
    ))
    positive_count = int(sum(
        value.get("positive_direction_count", 0) for value in values
    ))
    if any("negative_direction_count" in value for value in values):
        result.update({
            "negative_direction_count": negative_count,
            "negative_direction_copy_probability": (
                float(sum(
                    value["negative_direction_copy_probability"]
                    * value["negative_direction_count"]
                    for value in values
                    if value["negative_direction_count"]
                ) / negative_count) if negative_count else None
            ),
            "positive_direction_count": positive_count,
            "positive_direction_copy_probability": (
                float(sum(
                    value["positive_direction_copy_probability"]
                    * value["positive_direction_count"]
                    for value in values
                    if value["positive_direction_count"]
                ) / positive_count) if positive_count else None
            ),
        })
    return result


def _aggregate_config(spec, wrappers, protocol):
    runs = sorted((row["run"] for row in wrappers), key=lambda row: row["seed"])
    metrics = {
        "late_window_current_loss": "late_window_current_loss_mean",
        "final_current_loss": "final_loss",
        "current_loss_auc": "current_loss_auc",
        "positive_gain_rate": "positive_gain_rate",
        "negative_gain_rate": "negative_gain_rate",
        "mean_positive_gain": "mean_positive_gain",
        "mean_negative_gain": "mean_negative_gain",
        "mean_changed_cells": "mean_changed_cells",
        "final_unique_states": "final_unique_states",
        "elapsed_sec": "elapsed_sec",
        "factor_pipeline_elapsed_sec": "factor_pipeline_elapsed_sec",
        "gibbs_sample_elapsed_sec": "gibbs_sample_elapsed_sec",
        "gibbs_microsteps": "gibbs_microsteps",
    }
    independent_logit = _aggregate_trajectory_logit(
        runs, "independent_direction_diagnostics"
    )
    factor_logit = _aggregate_trajectory_logit(
        runs, "factor_conditional_logit_diagnostics"
    )
    return {
        "config": dict(spec),
        "trajectory_count": len(runs),
        "seeds": [row["seed"] for row in runs],
        "all_rounds_complete": all(
            row["rounds_run"] == protocol["rounds"] for row in runs
        ),
        "late_window_rounds": [
            max(1, protocol["rounds"] - protocol["late_window_size"] + 1),
            protocol["rounds"],
        ],
        "metrics": {
            name: stage_a._numeric_summary([row[key] for row in runs])
            for name, key in metrics.items()
        },
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
        "snapshot_count": int(sum(
            len(row.get("state_snapshots", [])) for row in runs
        )),
    }


def _aggregate_all(rows, configs, protocol):
    return {
        spec["config_id"]: _aggregate_config(
            spec,
            [row for row in rows if row["config_id"] == spec["config_id"]],
            protocol,
        )
        for spec in configs
    }


def _paired_factor(rows, factor_config, protocol):
    factor = {
        row["run"]["seed"]: row["run"] for row in rows
        if row["config_id"] == factor_config["config_id"]
    }
    baseline_id = _config_id(
        "independent", factor_config["temperature"], 0
    )
    baseline = {
        row["run"]["seed"]: row["run"] for row in rows
        if row["config_id"] == baseline_id
    }
    seeds = protocol["stage_b_seeds"]
    if set(factor) != set(seeds) or set(baseline) != set(seeds):
        raise RuntimeError("factor 与同 tau independent 的 seed 不完整")
    differences = np.asarray([
        factor[seed]["late_window_current_loss_mean"]
        - baseline[seed]["late_window_current_loss_mean"]
        for seed in seeds
    ], dtype=float)
    return {
        "candidate_config_id": factor_config["config_id"],
        "baseline_config_id": baseline_id,
        "metric": "late_window_current_loss_mean",
        "candidate_minus_baseline_by_seed": [
            {"seed": seed, "difference": float(value)}
            for seed, value in zip(seeds, differences)
        ],
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "wins": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "losses": int(np.sum(differences > 0.0)),
    }


def _rank_key(config_id, configs_by_id, aggregates):
    spec = configs_by_id[config_id]
    metrics = aggregates[config_id]["metrics"]
    return (
        metrics["late_window_current_loss"]["mean"],
        metrics["late_window_current_loss"]["median"],
        metrics["current_loss_auc"]["mean"],
        metrics["final_current_loss"]["mean"],
        spec["sweeps"],
        spec["temperature"],
    )


def _preliminary_selection(
    configs, aggregates, comparisons, stage_a_report, protocol
):
    by_id = {row["config_id"]: row for row in configs}
    a0 = stage_a_report["a0"]["classification"]["temperatures"]
    independent = {}
    for temperature in protocol["independent_temperatures"]:
        config_id = _config_id("independent", temperature, 0)
        aggregate = aggregates[config_id]
        upstream = a0[stage_a._tau_key(temperature)]
        gates = {
            "stage_t_a0_zero_clip_and_eligible": bool(
                upstream["eligible_for_mixing"]
            ),
            "all_stage_b_seeds_retained": (
                aggregate["seeds"] == protocol["stage_b_seeds"]
            ),
            "all_stage_b_rounds_complete": aggregate["all_rounds_complete"],
            "stage_b_zero_clip_hits": aggregate["total_clip_hit_count"] == 0,
            "stage_b_conditionals_finite_and_bidirectional": aggregate[
                "all_conditionals_finite_and_bidirectional"
            ],
        }
        independent[config_id] = {
            "eligible": all(gates.values()),
            "gates": gates,
            "rank_key": list(_rank_key(config_id, by_id, aggregates)),
        }
    eligible_independent = [
        config_id for config_id, result in independent.items()
        if result["eligible"]
    ]
    i_star = (
        min(
            eligible_independent,
            key=lambda item: _rank_key(item, by_id, aggregates),
        ) if eligible_independent else None
    )

    factor = {}
    for spec in configs:
        if spec["kernel"] != "factor":
            continue
        config_id = spec["config_id"]
        aggregate = aggregates[config_id]
        paired = comparisons[config_id]
        gates = {
            "a0_a1_passed": True,
            "all_stage_b_seeds_retained": (
                aggregate["seeds"] == protocol["stage_b_seeds"]
            ),
            "all_stage_b_rounds_complete": aggregate["all_rounds_complete"],
            "stage_b_zero_clip_hits": aggregate["total_clip_hit_count"] == 0,
            "stage_b_conditionals_finite_and_bidirectional": aggregate[
                "all_conditionals_finite_and_bidirectional"
            ],
            "paired_mean_improves": paired["mean_difference"] < 0.0,
            "paired_median_improves": paired["median_difference"] < 0.0,
            "paired_wins_sufficient": (
                paired["wins"] >= protocol["minimum_paired_wins"]
            ),
        }
        factor[config_id] = {
            "eligible_for_g0": all(gates.values()),
            "gates": gates,
            "paired_same_temperature": paired,
            "rank_key": list(_rank_key(config_id, by_id, aggregates)),
        }
    eligible_factor = [
        config_id for config_id, result in factor.items()
        if result["eligible_for_g0"]
    ]
    g0 = (
        min(
            eligible_factor,
            key=lambda item: _rank_key(item, by_id, aggregates),
        ) if eligible_factor else None
    )
    return {
        "ranking_rule": (
            "late-window mean, late-window median, AUC mean, final-current "
            "mean, fewer sweeps, lower tau"
        ),
        "independent": independent,
        "i_star": i_star,
        "factor": factor,
        "g0": g0,
    }


def _self_review_entries(g0_rows, g0, protocol):
    entries = []
    rounds = protocol["snapshot_rounds"]
    families = dict(zip(rounds, SELF_REVIEW_FAMILIES))
    for wrapper in sorted(g0_rows, key=lambda row: row["run"]["seed"]):
        run = wrapper["run"]
        for snapshot in run["state_snapshots"]:
            state_round = snapshot["state_round"]
            family = families.get(state_round)
            if family is None:
                raise RuntimeError("G0 包含协议外自身状态快照")
            entries.append({
                "state_id": (
                    f"{g0}_seed_{run['seed']}_{family}_round_{state_round}"
                ),
                "seed": run["seed"],
                "state_round": state_round,
                "state_family": family,
                "source_temperature": run["temperature"],
                "snapshot": snapshot,
            })
    expected = len(protocol["stage_b_seeds"]) * len(SELF_REVIEW_FAMILIES)
    if len(entries) != expected or len({row["state_id"] for row in entries}) != expected:
        raise RuntimeError("G0 自身状态数量或身份不完整")
    return entries


def _self_review(g0, rows, configs, target, queries, schema, protocol):
    if g0 is None:
        return {
            "applicable": False,
            "status": "not_run_no_preliminary_factor_champion",
            "passed": None,
        }
    spec = next(row for row in configs if row["config_id"] == g0)
    g0_rows = [row for row in rows if row["config_id"] == g0]
    entries = _self_review_entries(g0_rows, g0, protocol)
    library = {"states": entries}
    probe_protocol = frozen_protocol.protocol(protocol["mode"])
    probe_protocol["proposals_per_state"] = protocol[
        "self_review_proposals_per_state"
    ]
    temperature = spec["temperature"]
    sweep = spec["sweeps"]
    a0_protocol = {**probe_protocol, "sweeps": [0]}
    a0_rows = stage_a._probe_library(
        library,
        target,
        queries,
        schema,
        a0_protocol,
        temperatures=[temperature],
        sweeps=[0],
    )
    a1_protocol = {**probe_protocol, "sweeps": [0, sweep]}
    a1_rows = stage_a._probe_library(
        library,
        target,
        queries,
        schema,
        a1_protocol,
        temperatures=[temperature],
        sweeps=[0, sweep],
    )
    groups = {"global": list(a1_rows)}
    groups.update({
        family: [row for row in a1_rows if row["state_family"] == family]
        for family in SELF_REVIEW_FAMILIES
    })
    aggregates = {
        name: stage_a._aggregate_group(
            name, values, a1_protocol, [temperature]
        )
        for name, values in groups.items()
    }
    factor = stage_a._aggregate_factor_diagnostics(
        a1_rows, [temperature]
    )
    probabilities = stage_a._aggregate_probability_diagnostics(
        a1_rows, temperature
    )
    production = stage_a._aggregate_production_sampler(
        a1_rows, temperature
    )
    tau_key = stage_a._tau_key(temperature)
    a0_logit = stage_a._aggregate_logit(a0_rows, temperature)
    a0_by_state = {row["state_id"]: row for row in a0_rows}
    a1_by_state = {row["state_id"]: row for row in a1_rows}
    baseline_name = probe._gibbs_name(temperature, 0)
    candidate_name = probe._gibbs_name(temperature, sweep)
    a0_gates = {
        "state_count_complete": len(entries) == len(a0_rows) == len(a1_rows),
        "proposal_counts_complete": all(
            row["probe"]["n_proposals"]
            == protocol["self_review_proposals_per_state"]
            for row in (*a0_rows, *a1_rows)
        ),
        "all_families_present": all(groups[family] for family in SELF_REVIEW_FAMILIES),
        "all_families_have_active_rows": all(
            aggregates[group]["kernel_summary"][baseline_name][
                "participating_active_rows"
            ] > 0
            for group in SELF_REVIEW_GROUPS
        ),
        "factor_raw_logits_strictly_inside_clip": (
            a0_logit["condition_count"] > 0
            and a0_logit["clip_hit_count"] == 0
            and a0_logit["raw_logit_strictly_inside_clip"]
        ),
        "factor_conditionals_finite_and_bidirectional": (
            a0_logit["all_conditionals_bidirectional"]
            and a0_logit["minimum_binary_outcome_probability"] is not None
            and a0_logit["minimum_binary_outcome_probability"] > 0.0
        ),
    }
    a1_gates = {
        "a0_replay_state_identity_exact": (
            list(a0_by_state) == list(a1_by_state)
            and all(
                a0_by_state[state_id]["state_sha256"]
                == a1_by_state[state_id]["state_sha256"]
                for state_id in a0_by_state
            )
        ),
        "a0_replay_factor_logits_exact": all(
            a0_by_state[state_id]["probe"][
                "conditional_logit_diagnostics"
            ][tau_key]
            == a1_by_state[state_id]["probe"][
                "conditional_logit_diagnostics"
            ][tau_key]
            for state_id in a0_by_state
        ),
        "exact_energy_error_within_tolerance": (
            factor["exact_energy_max_error"] <= protocol["energy_tolerance"]
        ),
        "one_hot_error_within_tolerance": (
            factor["one_hot_direction_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "tvd_monotonic_within_tolerance": (
            factor["tvd_snapshot_increase_max_by_temperature"][tau_key]
            <= protocol["tvd_monotonic_tolerance"]
        ),
        "probability_distributions_complete": probabilities[
            "distribution_count"
        ] > 0,
        "probability_values_finite": probabilities["all_finite"],
        "probability_values_nonnegative": probabilities["all_nonnegative"],
        "probability_sums_within_tolerance": (
            probabilities["probability_sum_max_error"]
            <= protocol["probability_sum_tolerance"]
        ),
        "production_sampler_replay_complete": (
            production["comparison_count"] > 0
            and production["microsteps"] > 0
        ),
        "production_sampler_matches_exact_tape_replay": (
            production["all_exact_tape_replays_match"]
            and production["mismatch_count"] == 0
        ),
        "all_numeric_values_finite": stage_a._all_numeric_finite({
            "a0_state_results": a0_rows,
            "a1_state_results": a1_rows,
            "aggregates": aggregates,
        }),
    }
    mixing_groups = {}
    for group in SELF_REVIEW_GROUPS:
        kernel = aggregates[group]["kernel_summary"][candidate_name]
        recovery = aggregates[group][
            "expected_direction_gap_recovery"
        ][candidate_name]
        passed = bool(
            kernel["participating_active_rows"] > 0
            and kernel["tvd_to_joint"]
            <= protocol["self_review_tvd_threshold"]
            and recovery is not None
            and recovery >= protocol["self_review_recovery_threshold"]
        )
        mixing_groups[group] = {
            "participating_active_rows": kernel[
                "participating_active_rows"
            ],
            "tvd_to_joint": kernel["tvd_to_joint"],
            "expected_direction_gap_recovery": recovery,
            "passed": passed,
        }
    passed = bool(
        all(a0_gates.values())
        and all(a1_gates.values())
        and all(value["passed"] for value in mixing_groups.values())
    )
    return {
        "applicable": True,
        "status": "passed" if passed else "failed_no_factor_fallback",
        "passed": passed,
        "g0": g0,
        "temperature": temperature,
        "sweeps": sweep,
        "required_groups": protocol["self_review_required_groups"],
        "state_count": len(entries),
        "state_entries": entries,
        "a0_state_results": a0_rows,
        "a1_state_results": a1_rows,
        "aggregates": aggregates,
        "a0_logit_diagnostics": a0_logit,
        "a0_gates": a0_gates,
        "a1_gates": a1_gates,
        "mixing_groups": mixing_groups,
        "production_sampler_diagnostics": production,
    }


def _final_selection(preliminary, self_review, configs, aggregates):
    by_id = {row["config_id"]: row for row in configs}
    i_star = preliminary["i_star"]
    g0 = preliminary["g0"]
    g_star = g0 if self_review.get("passed") is True else None
    if i_star is None:
        unique = None
        status = "no_certified_baseline_no_eligible_independent"
    elif g_star is None:
        unique = i_star
        status = "independent_candidate_frozen"
    else:
        unique = min(
            (i_star, g_star),
            key=lambda item: _rank_key(item, by_id, aggregates),
        )
        status = (
            "factor_candidate_frozen"
            if unique == g_star else "independent_candidate_frozen"
        )
    return {
        **preliminary,
        "g_star": g_star,
        "self_state_review": self_review,
        "unique_candidate": unique,
        "unique_candidate_config": (
            dict(by_id[unique]) if unique is not None else None
        ),
        "status": status,
        "no_runner_up_after_g0_review_failure": True,
    }


def run_stage_b(mode, stage_a_report_path, stage_a_audit_path, output_dir):
    protocol = _protocol(mode)
    implementation_gates = _validate_implementation_constants(protocol)
    output_directory = Path(output_dir)
    report_path = output_directory / "stage_b_report.json"
    if report_path.exists():
        raise FileExistsError(
            f"输出已存在，尚未启动任何 Stage B 轨迹：{report_path}"
        )
    git = stage_a._git_identity()
    if mode == "formal" and not git["worktree_clean"]:
        raise RuntimeError("正式 Stage B 要求 tracked 工作树干净")
    (
        upstream_report_file,
        upstream_audit_file,
        upstream_report,
        upstream_audit,
        upstream_checks,
    ) = _load_upstream(
        stage_a_report_path, stage_a_audit_path, mode, git
    )
    target, queries, schema, marginals, input_hashes = stage_a._load_inputs()
    configs = _all_configs(protocol, upstream_report)
    upstream_hashes = {
        "stage_t_a_report": stage_a._sha256_file(upstream_report_file),
        "stage_t_a_audit": stage_a._sha256_file(upstream_audit_file),
    }
    protocol_sha256 = stage_a._canonical_sha256({
        "protocol": protocol,
        "upstream_sha256": upstream_hashes,
        "input_sha256": input_hashes,
        "git_commit": git["commit"],
    })

    started = time.perf_counter()
    rows = _run_trajectories(
        target, queries, schema, marginals, protocol, configs
    )
    trajectory_gates = _trajectory_identity_gates(rows, configs, protocol)
    aggregates = _aggregate_all(rows, configs, protocol)
    comparisons = {
        spec["config_id"]: _paired_factor(rows, spec, protocol)
        for spec in configs if spec["kernel"] == "factor"
    }
    preliminary = _preliminary_selection(
        configs, aggregates, comparisons, upstream_report, protocol
    )
    self_review = _self_review(
        preliminary["g0"],
        rows,
        configs,
        target,
        queries,
        schema,
        protocol,
    )
    selection = _final_selection(
        preliminary, self_review, configs, aggregates
    )
    formal_identity_gates = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _protocol("formal"),
        "worktree_clean": git["worktree_clean"],
        "same_commit_as_stage_t_a": (
            git["commit"] == upstream_report["git"]["commit"]
        ),
        "upstream_formal_result_valid": (
            upstream_report["formal_result_valid"] is True
            and upstream_audit["formal_result_valid"] is True
        ),
        "input_hashes_match": (
            input_hashes == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
    }
    formal_result_valid = bool(
        all(formal_identity_gates.values())
        and trajectory_gates["all_trajectory_identity_gates_passed"]
    )
    report = {
        "report_format": REPORT_FORMAT,
        "status": "complete",
        "experiment": "issue49_stage_b_long_run_candidate_freeze",
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "interpretation": (
            "formal_preregistered_stage_b"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "input_sha256": input_hashes,
        "upstream": {
            "stage_t_a_report_path": str(upstream_report_file),
            "stage_t_a_report_sha256": upstream_hashes["stage_t_a_report"],
            "stage_t_a_audit_path": str(upstream_audit_file),
            "stage_t_a_audit_sha256": upstream_hashes["stage_t_a_audit"],
            "stage_t_a_protocol_sha256": upstream_report[
                "protocol_sha256"
            ],
            "checks": upstream_checks,
        },
        "git": git,
        "command_argv": list(sys.argv),
        "environment": trajectory._environment(protocol["device"]),
        "implementation_gates": implementation_gates,
        "configurations": configs,
        "trajectory_identity_gates": trajectory_gates,
        "trajectories": rows,
        "aggregates": aggregates,
        "same_temperature_factor_comparisons": comparisons,
        "selection": selection,
        "formal_identity_gates": formal_identity_gates,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    stage_a._write_json_atomic(report_path, report)
    return report_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), default="smoke"
    )
    parser.add_argument("--stage-a-report", required=True)
    parser.add_argument("--stage-a-audit", required=True)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/issue49_high_temperature_factor_gibbs/"
            "stage_b_smoke"
        ),
    )
    args = parser.parse_args()
    if args.mode == "formal" and args.output_dir.endswith("stage_b_smoke"):
        parser.error("正式模式必须显式提供非 smoke 输出目录")
    output, report = run_stage_b(
        args.mode,
        args.stage_a_report,
        args.stage_a_audit,
        args.output_dir,
    )
    print("\n===== Issue #49 Stage B =====")
    print(f"mode={report['mode']}")
    print(f"i_star={report['selection']['i_star']}")
    print(f"g0={report['selection']['g0']}")
    print(f"g_star={report['selection']['g_star']}")
    print(f"unique_candidate={report['selection']['unique_candidate']}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
