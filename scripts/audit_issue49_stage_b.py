"""独立复算并审计 Issue #49 协议 v2 的 Stage B 输出。"""

import argparse
import math
from pathlib import Path
import tempfile
import time

import numpy as np

if __package__:
    from scripts import audit_issue49_stage_a as stage_a_auditor
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import probe_factorized_gibbs_mixing as probe
else:
    import audit_issue49_stage_a as stage_a_auditor
    import compare_factorized_gibbs_unfiltered as trajectory
    import probe_factorized_gibbs_mixing as probe


AUDIT_FORMAT = "issue49_stage_b_audit_v1"
REPORT_FORMAT = "issue49_stage_b_report_v1"
UPSTREAM_REPORT_FORMAT = "issue49_stage_t_a_report_v2"
UPSTREAM_AUDIT_FORMAT = "issue49_stage_t_a_audit_v2"
TEMPERATURES = [4.0, 5.0, 6.0, 7.0, 8.0]
SELF_REVIEW_FAMILIES = ("initial", "mid", "late")
SELF_REVIEW_GROUPS = ("global", *SELF_REVIEW_FAMILIES)
EXPECTED_INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "queries": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
}


def _expected_protocol(mode):
    if mode == "formal":
        seeds, rounds, snapshots, proposals = (
            list(range(100, 110)), 1000, [0, 500, 1000], 200
        )
    elif mode == "smoke":
        seeds, rounds, snapshots, proposals = [99], 12, [0, 6, 12], 2
    else:
        raise RuntimeError(f"未知 Stage B mode：{mode!r}")
    return {
        "protocol_version": 2,
        "mode": mode,
        "dataset": "test_300x10",
        "stage_b_seeds": seeds,
        "rounds": rounds,
        "snapshot_rounds": snapshots,
        "late_window_size": 250,
        "independent_temperatures": list(TEMPERATURES),
        "factor_builder": "legacy_rowwise",
        "rho": 0.01,
        "eta": 0.5,
        "trajectory_mu": 0.01,
        "max_factor_order": 3,
        "logit_clip": 30.0,
        "device": "numpy",
        "minimum_paired_win_fraction": 0.60,
        "minimum_paired_wins": int(math.ceil(0.60 * len(seeds))),
        "ranking_tie_breakers": [
            "late_window_current_loss_mean",
            "late_window_current_loss_median",
            "current_loss_auc_mean",
            "final_current_loss_mean",
            "fewer_sweeps",
            "lower_temperature",
        ],
        "self_review_required_groups": list(SELF_REVIEW_GROUPS),
        "self_review_proposals_per_state": proposals,
        "self_review_probe_mu": 0.0,
        "self_review_max_active_attributes": 12,
        "self_review_tvd_threshold": 0.05,
        "self_review_recovery_threshold": 0.80,
        "energy_tolerance": 1e-10,
        "tvd_monotonic_tolerance": 1e-12,
        "probability_sum_tolerance": 1e-12,
    }


def _config_id(kind, temperature, sweeps):
    tau = f"{temperature:g}".replace(".", "p")
    return (
        f"independent_tau_{tau}"
        if kind == "independent"
        else f"factor_tau_{tau}_sweeps_{sweeps}"
    )


def _config(kind, temperature, sweeps):
    return {
        "config_id": _config_id(kind, temperature, sweeps),
        "kernel": kind,
        "temperature": float(temperature),
        "sweeps": int(sweeps),
    }


def _factor_configs(upstream):
    a0 = upstream["a0"]["classification"]["temperatures"]
    selection = upstream["a1"]["selection"]["temperatures"]
    candidates = upstream["protocol"]["candidate_sweeps"]
    configs = []
    for temperature in TEMPERATURES:
        key = stage_a_auditor._tau_key(temperature)
        result = selection[key]
        sweep = result["minimal_sufficient_sweeps"]
        if sweep is None:
            if result["status"] == "sufficient_within_grid":
                raise RuntimeError(f"上游 {key} 状态与 sweeps 冲突")
            continue
        passed = [row["sweeps"] for row in result["candidates"] if row["passed"]]
        if (
            not a0[key]["eligible_for_mixing"]
            or result["status"] != "sufficient_within_grid"
            or sweep not in candidates
            or not passed
            or sweep != passed[0]
        ):
            raise RuntimeError(f"上游 {key} 最小充分 sweeps 无效")
        configs.append(_config("factor", temperature, sweep))
    return configs


def _require_stage_b_allowed(upstream):
    """独立核对 A0 全失败时禁止进入 Stage B。"""
    classification = upstream["a0"]["classification"]
    temperatures = classification["temperatures"]
    recomputed = [
        temperature
        for temperature in upstream["protocol"][
            "evaluation_temperatures"
        ]
        if temperatures[stage_a_auditor._tau_key(temperature)][
            "eligible_for_mixing"
        ]
    ]
    if classification.get("eligible_temperatures") != recomputed:
        raise RuntimeError("Stage A 的 A0 合格温度身份不一致")
    if not recomputed:
        raise RuntimeError(
            "A0 全部 tau 不合格；按冻结停止规则不得存在 Stage B 报告"
        )
    return recomputed


def _expected_configs(upstream):
    _require_stage_b_allowed(upstream)
    return [
        *(_config("independent", temperature, 0) for temperature in TEMPERATURES),
        *_factor_configs(upstream),
    ]


def _validate_upstream(report):
    upstream = report["upstream"]
    upstream_report_file = Path(
        upstream["stage_t_a_report_path"]
    ).resolve()
    upstream_audit_file = Path(
        upstream["stage_t_a_audit_path"]
    ).resolve()
    stage_a_report = stage_a_auditor._load_json_strict(upstream_report_file)
    stage_a_audit = stage_a_auditor._load_json_strict(upstream_audit_file)
    if (
        stage_a_report.get("report_format") != UPSTREAM_REPORT_FORMAT
        or stage_a_audit.get("audit_format") != UPSTREAM_AUDIT_FORMAT
        or stage_a_audit.get("passed") is not True
        or Path(stage_a_audit.get("report_path", "")).resolve()
        != upstream_report_file
        or stage_a_auditor._sha256_file(upstream_report_file)
        != upstream["stage_t_a_report_sha256"]
        != stage_a_audit.get("report_sha256")
        or stage_a_auditor._sha256_file(upstream_audit_file)
        != upstream["stage_t_a_audit_sha256"]
        or stage_a_report.get("protocol_sha256")
        != upstream["stage_t_a_protocol_sha256"]
        != stage_a_audit.get("protocol_sha256")
        or stage_a_report["a1"]["selection"]
        != stage_a_audit.get("selection")
        or stage_a_report.get("mode") != report.get("mode")
        or stage_a_audit.get("mode") != report.get("mode")
        or stage_a_report.get("input_sha256") != EXPECTED_INPUT_SHA256
        or stage_a_report.get("git", {}).get("commit")
        != report.get("git", {}).get("commit")
    ):
        raise RuntimeError("Stage B 上游 Stage T/A 报告或审计绑定无效")
    library = Path(stage_a_report["state_library"]["path"]).resolve()
    if (
        not library.is_file()
        or Path(stage_a_audit["state_library_path"]).resolve() != library
        or stage_a_auditor._sha256_file(library)
        != stage_a_report["state_library"]["sha256"]
        != stage_a_audit["state_library_sha256"]
    ):
        raise RuntimeError("Stage B 上游状态库绑定无效")
    expected_checks = {
        "report_format": True,
        "audit_format": True,
        "report_complete": True,
        "audit_complete_and_passed": True,
        "mode_matches": True,
        "protocol_exact": True,
        "input_hashes_exact": True,
        "audit_report_path_matches": True,
        "audit_report_hash_matches": True,
        "protocol_hash_matches": True,
        "selection_matches_audit": True,
        "same_git_commit": True,
        "state_library_path_matches_audit": True,
        "state_library_exists": True,
        "state_library_hash_matches": True,
        "formal_upstream_valid_when_required": True,
        "stage_t_a_semantic_gates_passed": True,
        "fresh_stage_t_a_reaudit_passed": True,
    }
    stage_a_auditor._assert_same(
        upstream["checks"], expected_checks, "report.upstream.checks"
    )
    with tempfile.TemporaryDirectory(
        prefix="issue49-stage-b-independent-upstream-audit-"
    ) as temporary:
        _, fresh_audit = stage_a_auditor.audit_stage_a(
            upstream_report_file,
            library,
            Path(temporary) / "stage_t_a_audit.json",
        )
    if (
        fresh_audit["passed"] is not True
        or fresh_audit["report_sha256"]
        != stage_a_auditor._sha256_file(upstream_report_file)
        or fresh_audit["state_library_sha256"]
        != stage_a_auditor._sha256_file(library)
        or fresh_audit["formal_result_valid"]
        != stage_a_audit["formal_result_valid"]
        or fresh_audit["selection"] != stage_a_audit["selection"]
    ):
        raise RuntimeError("Stage B auditor 独立复审 Stage T/A 失败")
    return stage_a_report, stage_a_audit


def _valid_logit(value, expect_conditions):
    count = value.get("condition_count")
    hits = value.get("clip_hit_count")
    return bool(
        isinstance(count, int)
        and isinstance(hits, int)
        and count >= (1 if expect_conditions else 0)
        and hits >= 0
        and len(value.get("clip_hit_conditions", [])) == hits
        and value.get("logit_clip") == 30.0
        and value.get("raw_logit_strictly_inside_clip")
        == (count == 0 or value.get("raw_logit_abs_max") < 30.0)
        and value.get("all_finite") is True
        and value.get("all_conditionals_bidirectional") is True
    )


def _trajectory_gates(rows, configs, protocol):
    specs = {row["config_id"]: row for row in configs}
    expected = {
        (seed, config_id)
        for seed in protocol["stage_b_seeds"]
        for config_id in specs
    }
    actual = {(row["run"].get("seed"), row.get("config_id")) for row in rows}
    row_results = []
    for wrapper in rows:
        run = wrapper["run"]
        spec = specs.get(wrapper.get("config_id"))
        factor = spec is not None and spec["kernel"] == "factor"
        history = np.asarray(
            run.get("current_loss_after_round_history", []), dtype=float
        )
        late = history[-protocol["late_window_size"]:]
        factor_logit = run.get("factor_conditional_logit_diagnostics", {})
        row_results.append(bool(
            spec is not None
            and wrapper.get("kernel") == spec["kernel"]
            and run.get("temperature") == spec["temperature"]
            and run.get("sweeps") == spec["sweeps"]
            and run.get("name") == (
                f"gibbs_{spec['sweeps']}_sweeps" if factor else "independent"
            )
            and run.get("factor_builder") == (
                protocol["factor_builder"] if factor else "not_used"
            )
            and run.get("rounds_run") == protocol["rounds"]
            and len(history) == protocol["rounds"]
            and np.all(np.isfinite(history))
            and len(late) == min(protocol["late_window_size"], protocol["rounds"])
            and math.isclose(
                float(late.mean()),
                run.get("late_window_current_loss_mean"),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and _valid_logit(
                run.get("independent_direction_diagnostics", {}), True
            )
            and _valid_logit(factor_logit, factor)
            and factor_logit.get("condition_count") == run.get("gibbs_microsteps")
            and (
                run.get("snapshot_rounds") == protocol["snapshot_rounds"]
                and [item["state_round"] for item in run.get("state_snapshots", [])]
                == protocol["snapshot_rounds"]
                if factor else (
                    "snapshot_rounds" not in run and "state_snapshots" not in run
                )
            )
        ))
    by_seed = {
        seed: [row["run"] for row in rows if row["run"]["seed"] == seed]
        for seed in protocol["stage_b_seeds"]
    }
    gates = {
        "trajectory_grid_complete": len(rows) == len(expected) and actual == expected,
        "all_row_identities_valid": all(row_results),
        "all_initial_states_aligned_within_seed": all(
            len({row["initial_csv_sha256"] for row in values}) == 1
            and len({row["initial_loss"] for row in values}) == 1
            for values in by_seed.values()
        ),
        "all_primary_rng_endpoints_aligned_within_seed": all(
            len({row["primary_rng_state_sha256"] for row in values}) == 1
            for values in by_seed.values()
        ),
        "all_direction_scales_aligned_within_seed": all(
            len({row["direction_reference_scale"] for row in values}) == 1
            and all(
                row["direction_reference_scale"] is not None
                and math.isfinite(row["direction_reference_scale"])
                and row["direction_reference_scale"] > 0.0
                and row["direction_reference_scale_round"] == 0
                for row in values
            )
            for values in by_seed.values()
        ),
        "all_numeric_values_finite": stage_a_auditor._all_numeric_finite(rows),
    }
    gates["all_trajectory_identity_gates_passed"] = all(gates.values())
    return gates


def _aggregate_logit(rows, field):
    values = [row[field] for row in rows]
    count = int(sum(value["condition_count"] for value in values))
    hits = int(sum(value["clip_hit_count"] for value in values))
    nonempty = [
        (run, value) for run, value in zip(rows, values)
        if value["condition_count"]
    ]
    if nonempty:
        maximum_run, maximum = max(
            nonempty, key=lambda item: item[1]["raw_logit_abs_max"]
        )
    else:
        maximum_run = maximum = None
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
        "raw_logit_abs_max_condition": (
            {"seed": maximum_run["seed"], **maximum["raw_logit_abs_max_condition"]}
            if maximum else None
        ),
        "logit_clip": 30.0,
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "clip_hit_conditions": [
            {"seed": run["seed"], **condition}
            for run, value in zip(rows, values)
            for condition in value["clip_hit_conditions"]
        ],
        "raw_logit_strictly_inside_clip": bool(
            count == 0 or maximum["raw_logit_abs_max"] < 30.0
        ),
        "conditional_probability_min": (
            float(min(value["conditional_probability_min"] for _, value in nonempty))
            if nonempty else None
        ),
        "conditional_probability_max": (
            float(max(value["conditional_probability_max"] for _, value in nonempty))
            if nonempty else None
        ),
        "minimum_binary_outcome_probability": (
            float(min(
                value["minimum_binary_outcome_probability"]
                for _, value in nonempty
            )) if nonempty else None
        ),
        "conditional_entropy_mean": (
            float(sum(
                value["conditional_entropy_mean"] * value["condition_count"]
                for _, value in nonempty
            ) / count) if nonempty else None
        ),
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


def _numeric(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


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
    independent = _aggregate_logit(runs, "independent_direction_diagnostics")
    factor = _aggregate_logit(runs, "factor_conditional_logit_diagnostics")
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
            name: _numeric([row[key] for row in runs])
            for name, key in metrics.items()
        },
        "independent_direction_diagnostics": independent,
        "factor_conditional_logit_diagnostics": factor,
        "total_clip_hit_count": int(
            independent["clip_hit_count"] + factor["clip_hit_count"]
        ),
        "all_conditionals_finite_and_bidirectional": bool(
            independent["all_finite"]
            and independent["all_conditionals_bidirectional"]
            and factor["all_finite"]
            and factor["all_conditionals_bidirectional"]
        ),
        "snapshot_count": int(sum(
            len(row.get("state_snapshots", [])) for row in runs
        )),
    }


def _aggregates(rows, configs, protocol):
    return {
        spec["config_id"]: _aggregate_config(
            spec,
            [row for row in rows if row["config_id"] == spec["config_id"]],
            protocol,
        )
        for spec in configs
    }


def _paired(rows, spec, protocol):
    candidate = {
        row["run"]["seed"]: row["run"] for row in rows
        if row["config_id"] == spec["config_id"]
    }
    baseline_id = _config_id("independent", spec["temperature"], 0)
    baseline = {
        row["run"]["seed"]: row["run"] for row in rows
        if row["config_id"] == baseline_id
    }
    seeds = protocol["stage_b_seeds"]
    if set(candidate) != set(seeds) or set(baseline) != set(seeds):
        raise RuntimeError("Stage B 配对 seed 不完整")
    values = np.asarray([
        candidate[seed]["late_window_current_loss_mean"]
        - baseline[seed]["late_window_current_loss_mean"]
        for seed in seeds
    ])
    return {
        "candidate_config_id": spec["config_id"],
        "baseline_config_id": baseline_id,
        "metric": "late_window_current_loss_mean",
        "candidate_minus_baseline_by_seed": [
            {"seed": seed, "difference": float(value)}
            for seed, value in zip(seeds, values)
        ],
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "wins": int(np.sum(values < 0.0)),
        "ties": int(np.sum(values == 0.0)),
        "losses": int(np.sum(values > 0.0)),
    }


def _rank(config_id, specs, aggregates):
    spec = specs[config_id]
    metrics = aggregates[config_id]["metrics"]
    return (
        metrics["late_window_current_loss"]["mean"],
        metrics["late_window_current_loss"]["median"],
        metrics["current_loss_auc"]["mean"],
        metrics["final_current_loss"]["mean"],
        spec["sweeps"],
        spec["temperature"],
    )


def _preliminary(configs, aggregates, comparisons, upstream, protocol):
    specs = {row["config_id"]: row for row in configs}
    a0 = upstream["a0"]["classification"]["temperatures"]
    independent = {}
    for temperature in TEMPERATURES:
        config_id = _config_id("independent", temperature, 0)
        aggregate = aggregates[config_id]
        gates = {
            "stage_t_a0_zero_clip_and_eligible": bool(
                a0[stage_a_auditor._tau_key(temperature)]["eligible_for_mixing"]
            ),
            "all_stage_b_seeds_retained": aggregate["seeds"] == protocol["stage_b_seeds"],
            "all_stage_b_rounds_complete": aggregate["all_rounds_complete"],
            "stage_b_zero_clip_hits": aggregate["total_clip_hit_count"] == 0,
            "stage_b_conditionals_finite_and_bidirectional": aggregate[
                "all_conditionals_finite_and_bidirectional"
            ],
        }
        independent[config_id] = {
            "eligible": all(gates.values()),
            "gates": gates,
            "rank_key": list(_rank(config_id, specs, aggregates)),
        }
    eligible_i = [key for key, value in independent.items() if value["eligible"]]
    i_star = min(eligible_i, key=lambda key: _rank(key, specs, aggregates)) if eligible_i else None
    factor = {}
    for spec in configs:
        if spec["kernel"] != "factor":
            continue
        config_id = spec["config_id"]
        aggregate = aggregates[config_id]
        paired = comparisons[config_id]
        gates = {
            "a0_a1_passed": True,
            "all_stage_b_seeds_retained": aggregate["seeds"] == protocol["stage_b_seeds"],
            "all_stage_b_rounds_complete": aggregate["all_rounds_complete"],
            "stage_b_zero_clip_hits": aggregate["total_clip_hit_count"] == 0,
            "stage_b_conditionals_finite_and_bidirectional": aggregate[
                "all_conditionals_finite_and_bidirectional"
            ],
            "paired_mean_improves": paired["mean_difference"] < 0.0,
            "paired_median_improves": paired["median_difference"] < 0.0,
            "paired_wins_sufficient": paired["wins"] >= protocol["minimum_paired_wins"],
        }
        factor[config_id] = {
            "eligible_for_g0": all(gates.values()),
            "gates": gates,
            "paired_same_temperature": paired,
            "rank_key": list(_rank(config_id, specs, aggregates)),
        }
    eligible_g = [key for key, value in factor.items() if value["eligible_for_g0"]]
    g0 = min(eligible_g, key=lambda key: _rank(key, specs, aggregates)) if eligible_g else None
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


def _expected_self_entries(rows, g0, protocol):
    families = dict(zip(protocol["snapshot_rounds"], SELF_REVIEW_FAMILIES))
    entries = []
    for wrapper in sorted(
        (row for row in rows if row["config_id"] == g0),
        key=lambda row: row["run"]["seed"],
    ):
        run = wrapper["run"]
        for snapshot in run["state_snapshots"]:
            state_round = snapshot["state_round"]
            family = families.get(state_round)
            if family is None:
                raise RuntimeError("G0 自身快照轮次越界")
            entries.append({
                "state_id": f"{g0}_seed_{run['seed']}_{family}_round_{state_round}",
                "seed": run["seed"],
                "state_round": state_round,
                "state_family": family,
                "source_temperature": run["temperature"],
                "snapshot": snapshot,
            })
    return entries


def _self_review(report_review, g0, rows, configs, protocol, target, queries, schema):
    if g0 is None:
        expected = {
            "applicable": False,
            "status": "not_run_no_preliminary_factor_champion",
            "passed": None,
        }
        stage_a_auditor._assert_same(report_review, expected, "selection.self_state_review")
        return expected
    spec = next(row for row in configs if row["config_id"] == g0)
    entries = _expected_self_entries(rows, g0, protocol)
    stage_a_auditor._assert_same(
        report_review["state_entries"], entries, "self_review.state_entries"
    )
    if len(entries) != len(protocol["stage_b_seeds"]) * 3:
        raise RuntimeError("G0 自身状态数量不完整")
    for entry in entries:
        _, controls = probe._restore_current_snapshot(
            entry["snapshot"], target, queries, schema, device=protocol["device"]
        )
        if (
            controls["source_seed"] != entry["seed"]
            or controls["state_round"] != entry["state_round"]
            or controls["source_temperature"] != spec["temperature"]
            or controls["source_sweeps"] != spec["sweeps"]
        ):
            raise RuntimeError(f"G0 自身状态身份无效：{entry['state_id']}")
    library = {"states": entries}
    a0_rows = report_review["a0_state_results"]
    a1_rows = report_review["a1_state_results"]
    proposals = protocol["self_review_proposals_per_state"]
    stage_a_auditor._validate_state_results(
        a0_rows, library, [spec["temperature"]], [0], proposals
    )
    stage_a_auditor._validate_state_results(
        a1_rows, library, [spec["temperature"]], [0, spec["sweeps"]], proposals
    )
    aggregation_protocol = {"sweeps": [0, spec["sweeps"]]}
    groups = {"global": list(a1_rows)}
    groups.update({
        family: [row for row in a1_rows if row["state_family"] == family]
        for family in SELF_REVIEW_FAMILIES
    })
    aggregates = {
        name: stage_a_auditor._aggregate_group(
            name, values, aggregation_protocol, [spec["temperature"]]
        )
        for name, values in groups.items()
    }
    temperature = spec["temperature"]
    tau_key = stage_a_auditor._tau_key(temperature)
    factor = stage_a_auditor._aggregate_factor(a1_rows, [temperature])
    probabilities = stage_a_auditor._aggregate_probability(a1_rows, temperature)
    production = stage_a_auditor._aggregate_production(a1_rows, temperature)
    a0_logit = stage_a_auditor._aggregate_logit(a0_rows, temperature)
    a0_by_state = {row["state_id"]: row for row in a0_rows}
    a1_by_state = {row["state_id"]: row for row in a1_rows}
    baseline_name = stage_a_auditor._gibbs_name(temperature, 0)
    candidate_name = stage_a_auditor._gibbs_name(temperature, spec["sweeps"])
    a0_gates = {
        "state_count_complete": len(entries) == len(a0_rows) == len(a1_rows),
        "proposal_counts_complete": all(
            row["probe"]["n_proposals"] == proposals
            for row in (*a0_rows, *a1_rows)
        ),
        "all_families_present": all(groups[family] for family in SELF_REVIEW_FAMILIES),
        "all_families_have_active_rows": all(
            aggregates[group]["kernel_summary"][baseline_name]["participating_active_rows"] > 0
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
                a0_by_state[key]["state_sha256"]
                == a1_by_state[key]["state_sha256"]
                for key in a0_by_state
            )
        ),
        "a0_replay_factor_logits_exact": all(
            a0_by_state[key]["probe"]["conditional_logit_diagnostics"][tau_key]
            == a1_by_state[key]["probe"]["conditional_logit_diagnostics"][tau_key]
            for key in a0_by_state
        ),
        "exact_energy_error_within_tolerance": (
            factor["exact_energy_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "one_hot_error_within_tolerance": (
            factor["one_hot_direction_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "tvd_monotonic_within_tolerance": (
            factor["tvd_snapshot_increase_max_by_temperature"][tau_key]
            <= protocol["tvd_monotonic_tolerance"]
        ),
        "probability_distributions_complete": probabilities["distribution_count"] > 0,
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
        "all_numeric_values_finite": (
            stage_a_auditor._all_numeric_finite({
                "a0_state_results": a0_rows,
                "a1_state_results": a1_rows,
                "aggregates": aggregates,
            })
        ),
    }
    mixing = {}
    for group in SELF_REVIEW_GROUPS:
        kernel = aggregates[group]["kernel_summary"][candidate_name]
        recovery = aggregates[group]["expected_direction_gap_recovery"][candidate_name]
        passed = bool(
            kernel["participating_active_rows"] > 0
            and kernel["tvd_to_joint"] <= protocol["self_review_tvd_threshold"]
            and recovery is not None
            and recovery >= protocol["self_review_recovery_threshold"]
        )
        mixing[group] = {
            "participating_active_rows": kernel["participating_active_rows"],
            "tvd_to_joint": kernel["tvd_to_joint"],
            "expected_direction_gap_recovery": recovery,
            "passed": passed,
        }
    passed = bool(
        all(a0_gates.values())
        and all(a1_gates.values())
        and all(value["passed"] for value in mixing.values())
    )
    expected = {
        "applicable": True,
        "status": "passed" if passed else "failed_no_factor_fallback",
        "passed": passed,
        "g0": g0,
        "temperature": temperature,
        "sweeps": spec["sweeps"],
        "required_groups": protocol["self_review_required_groups"],
        "state_count": len(entries),
        "state_entries": entries,
        "a0_state_results": a0_rows,
        "a1_state_results": a1_rows,
        "aggregates": aggregates,
        "a0_logit_diagnostics": a0_logit,
        "a0_gates": a0_gates,
        "a1_gates": a1_gates,
        "mixing_groups": mixing,
        "production_sampler_diagnostics": production,
    }
    stage_a_auditor._assert_same(
        report_review, expected, "selection.self_state_review"
    )
    return expected


def _final(preliminary, review, configs, aggregates):
    specs = {row["config_id"]: row for row in configs}
    i_star = preliminary["i_star"]
    g0 = preliminary["g0"]
    g_star = g0 if review.get("passed") is True else None
    if i_star is None:
        unique = None
        status = "no_certified_baseline_no_eligible_independent"
    elif g_star is None:
        unique = i_star
        status = "independent_candidate_frozen"
    else:
        unique = min(
            (i_star, g_star),
            key=lambda key: _rank(key, specs, aggregates),
        )
        status = (
            "factor_candidate_frozen"
            if unique == g_star else "independent_candidate_frozen"
        )
    return {
        **preliminary,
        "g_star": g_star,
        "self_state_review": review,
        "unique_candidate": unique,
        "unique_candidate_config": dict(specs[unique]) if unique else None,
        "status": status,
        "no_runner_up_after_g0_review_failure": True,
    }


def audit_stage_b(report_path, output_path):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    report = stage_a_auditor._load_json_strict(report_file)
    if (
        report.get("report_format") != REPORT_FORMAT
        or report.get("status") != "complete"
    ):
        raise RuntimeError("Stage B 报告格式或完成状态无效")
    mode = report.get("mode")
    protocol = _expected_protocol(mode)
    stage_a_auditor._assert_same(
        report.get("protocol"), protocol, "report.protocol"
    )
    stage_a_auditor._assert_same(
        report.get("experiment"),
        "issue49_stage_b_long_run_candidate_freeze",
        "report.experiment",
    )
    stage_a_auditor._assert_same(
        report.get("interpretation"),
        (
            "formal_preregistered_stage_b"
            if mode == "formal"
            else "pipeline_smoke_only_not_evidence"
        ),
        "report.interpretation",
    )
    implementation = {
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
    if not all(implementation.values()):
        raise RuntimeError("Stage B auditor 发现实现常量漂移")
    stage_a_auditor._assert_same(
        report["implementation_gates"],
        implementation,
        "report.implementation_gates",
    )
    upstream_report, upstream_audit = _validate_upstream(report)
    target, queries, schema, hashes = stage_a_auditor._load_frozen_inputs()
    stage_a_auditor._assert_same(
        report["input_sha256"], hashes, "report.input_sha256"
    )
    configs = _expected_configs(upstream_report)
    stage_a_auditor._assert_same(
        report["configurations"], configs, "report.configurations"
    )
    expected_protocol_sha = stage_a_auditor._canonical_sha256({
        "protocol": protocol,
        "upstream_sha256": {
            "stage_t_a_report": report["upstream"]["stage_t_a_report_sha256"],
            "stage_t_a_audit": report["upstream"]["stage_t_a_audit_sha256"],
        },
        "input_sha256": hashes,
        "git_commit": report["git"]["commit"],
    })
    stage_a_auditor._assert_same(
        report["protocol_sha256"],
        expected_protocol_sha,
        "report.protocol_sha256",
    )

    rows = report["trajectories"]
    gates = _trajectory_gates(rows, configs, protocol)
    if not gates["all_trajectory_identity_gates_passed"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Stage B 轨迹门禁失败：{failed}")
    stage_a_auditor._assert_same(
        report["trajectory_identity_gates"],
        gates,
        "report.trajectory_identity_gates",
    )
    aggregates = _aggregates(rows, configs, protocol)
    stage_a_auditor._assert_same(
        report["aggregates"], aggregates, "report.aggregates"
    )
    comparisons = {
        spec["config_id"]: _paired(rows, spec, protocol)
        for spec in configs if spec["kernel"] == "factor"
    }
    stage_a_auditor._assert_same(
        report["same_temperature_factor_comparisons"], comparisons,
        "report.same_temperature_factor_comparisons",
    )
    preliminary = _preliminary(
        configs, aggregates, comparisons, upstream_report, protocol
    )
    stage_a_auditor._assert_same(
        {key: report["selection"][key] for key in preliminary},
        preliminary,
        "report.selection.preliminary",
    )
    review = _self_review(
        report["selection"]["self_state_review"],
        preliminary["g0"], rows, configs, protocol, target, queries, schema,
    )
    selection = _final(preliminary, review, configs, aggregates)
    stage_a_auditor._assert_same(report["selection"], selection, "report.selection")
    formal_identity = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _expected_protocol("formal"),
        "worktree_clean": report["git"]["worktree_clean"],
        "same_commit_as_stage_t_a": (
            report["git"]["commit"]
            == upstream_report["git"]["commit"]
        ),
        "upstream_formal_result_valid": (
            upstream_report["formal_result_valid"] is True
            and upstream_audit["formal_result_valid"] is True
        ),
        "input_hashes_match": hashes == EXPECTED_INPUT_SHA256,
    }
    stage_a_auditor._assert_same(
        report["formal_identity_gates"],
        formal_identity,
        "report.formal_identity_gates",
    )
    formal_result_valid = bool(
        all(formal_identity.values()) and gates["all_trajectory_identity_gates_passed"]
    )
    stage_a_auditor._assert_same(
        report["formal_result_valid"],
        formal_result_valid,
        "report.formal_result_valid",
    )
    audit = {
        "audit_format": AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "report_path": str(report_file),
        "report_sha256": stage_a_auditor._sha256_file(report_file),
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "protocol_sha256": expected_protocol_sha,
        "upstream_stage_t_a_report_sha256": report["upstream"]["stage_t_a_report_sha256"],
        "upstream_stage_t_a_audit_sha256": report["upstream"]["stage_t_a_audit_sha256"],
        "checks": {
            "frozen_protocol_exact": True,
            "upstream_report_audit_and_library_bound": True,
            "trajectory_grid_and_rng_identity_recomputed": True,
            "raw_logit_and_clip_aggregates_recomputed": True,
            "same_temperature_pairing_recomputed": True,
            "i_star_and_g0_recomputed": True,
            "g0_self_state_review_recomputed": True,
            "unique_candidate_recomputed": True,
        },
        "selection": selection,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    stage_a_auditor._write_json_atomic(output_path, audit)
    return Path(output_path), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_stage_b(args.report, args.output)
    print("\n===== Issue #49 Stage B audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"unique_candidate={audit['selection']['unique_candidate']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
