"""独立复算并审计 Issue #49 的最终确认输出。"""

import argparse
import math
from pathlib import Path
import tempfile
import time

import numpy as np

if __package__:
    from scripts import audit_issue49_stage_a as stage_a_auditor
    from scripts import audit_issue49_stage_b as stage_b_auditor
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import probe_factorized_gibbs_mixing as probe
else:
    import audit_issue49_stage_a as stage_a_auditor
    import audit_issue49_stage_b as stage_b_auditor
    import compare_factorized_gibbs_unfiltered as trajectory
    import probe_factorized_gibbs_mixing as probe


AUDIT_FORMAT = "issue49_confirmation_audit_v1"
REPORT_FORMAT = "issue49_confirmation_report_v1"
UPSTREAM_REPORT_FORMAT = "issue49_stage_b_report_v1"
UPSTREAM_AUDIT_FORMAT = "issue49_stage_b_audit_v1"
TEMPERATURES = [4.0, 5.0, 6.0, 7.0, 8.0]
SELF_REVIEW_GROUPS = ("global", "initial", "mid", "late")
EXPECTED_INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "queries": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
}


def _expected_protocol(mode):
    if mode == "formal":
        seeds, rounds, snapshots, proposals = (
            list(range(110, 120)), 1000, [0, 500, 1000], 200
        )
    elif mode == "smoke":
        seeds, rounds, snapshots, proposals = [99], 12, [0, 6, 12], 2
    else:
        raise RuntimeError(f"未知最终确认 mode：{mode!r}")
    return {
        "protocol_version": 2,
        "mode": mode,
        "dataset": "test_300x10",
        "confirmation_seeds": seeds,
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
        "paired_confidence_level": 0.95,
        "paired_t_critical_95_df9": 2.2621571627409915,
        "formal_paired_sample_size": 10,
        "incumbent_independent_temperature": 8.0,
        "self_review_required_groups": list(SELF_REVIEW_GROUPS),
        "self_review_proposals_per_state": proposals,
        "self_review_probe_mu": 0.0,
        "self_review_max_active_attributes": 12,
        "self_review_tvd_threshold": 0.05,
        "self_review_recovery_threshold": 0.80,
        "energy_tolerance": 1e-10,
        "tvd_monotonic_tolerance": 1e-12,
        "probability_sum_tolerance": 1e-12,
        "frozen_candidate_only": True,
        "runner_up_forbidden": True,
    }


def _execution_protocol(protocol):
    return {
        **protocol,
        "stage_b_seeds": list(protocol["confirmation_seeds"]),
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


def _validate_upstream(report):
    upstream = report["upstream"]
    report_file = Path(upstream["stage_b_report_path"]).resolve()
    audit_file = Path(upstream["stage_b_audit_path"]).resolve()
    stage_b_report = stage_a_auditor._load_json_strict(report_file)
    stage_b_audit = stage_a_auditor._load_json_strict(audit_file)
    mode = report["mode"]
    selection = stage_b_report.get("selection", {})
    candidate = selection.get("unique_candidate")
    candidate_config = selection.get("unique_candidate_config")
    if (
        stage_b_report.get("report_format") != UPSTREAM_REPORT_FORMAT
        or stage_b_audit.get("audit_format") != UPSTREAM_AUDIT_FORMAT
        or stage_b_report.get("status") != "complete"
        or stage_b_audit.get("status") != "complete"
        or stage_b_audit.get("passed") is not True
        or stage_b_report.get("mode") != mode
        or stage_b_audit.get("mode") != mode
        or stage_b_report.get("protocol")
        != stage_b_auditor._expected_protocol(mode)
        or stage_b_report.get("input_sha256") != EXPECTED_INPUT_SHA256
        or Path(stage_b_audit.get("report_path", "")).resolve()
        != report_file
        or not (
            stage_a_auditor._sha256_file(report_file)
            == upstream["stage_b_report_sha256"]
            == stage_b_audit.get("report_sha256")
        )
        or stage_a_auditor._sha256_file(audit_file)
        != upstream["stage_b_audit_sha256"]
        or not (
            stage_b_report.get("protocol_sha256")
            == upstream["stage_b_protocol_sha256"]
            == stage_b_audit.get("protocol_sha256")
        )
        or stage_b_audit.get("selection") != selection
        or stage_b_report.get("git", {}).get("commit")
        != report.get("git", {}).get("commit")
        or not isinstance(candidate, str)
        or not isinstance(candidate_config, dict)
        or candidate_config.get("config_id") != candidate
        or candidate_config not in stage_b_report.get("configurations", [])
        or selection.get("no_runner_up_after_g0_review_failure") is not True
        or (
            mode == "formal"
            and (
                stage_b_report.get("formal_result_valid") is not True
                or stage_b_audit.get("formal_result_valid") is not True
            )
        )
    ):
        raise RuntimeError("最终确认的 Stage B 上游绑定无效")
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
        "formal_upstream_valid_when_required": True,
        "unique_candidate_exists": True,
        "unique_candidate_config_bound": True,
        "runner_up_forbidden_upstream": True,
        "fresh_stage_b_reaudit_passed": True,
    }
    stage_a_auditor._assert_same(
        upstream["checks"], expected_checks, "report.upstream.checks"
    )
    with tempfile.TemporaryDirectory(
        prefix="issue49-confirmation-independent-upstream-audit-"
    ) as temporary:
        _, fresh_audit = stage_b_auditor.audit_stage_b(
            report_file,
            Path(temporary) / "stage_b_audit.json",
        )
    if (
        fresh_audit["passed"] is not True
        or fresh_audit["report_sha256"]
        != stage_a_auditor._sha256_file(report_file)
        or fresh_audit["protocol_sha256"]
        != stage_b_report["protocol_sha256"]
        or fresh_audit["formal_result_valid"]
        != stage_b_audit["formal_result_valid"]
        or fresh_audit["selection"] != stage_b_audit["selection"]
    ):
        raise RuntimeError("最终确认 auditor 独立复审 Stage B 失败")
    return stage_b_report, stage_b_audit


def _expected_configs(stage_b_report, protocol):
    selection = stage_b_report["selection"]
    candidate_id = selection["unique_candidate"]
    candidate = selection["unique_candidate_config"]
    expected_candidate = _config(
        candidate.get("kernel"),
        candidate.get("temperature"),
        candidate.get("sweeps"),
    )
    if (
        candidate != expected_candidate
        or candidate["temperature"] not in TEMPERATURES
    ):
        raise RuntimeError("最终确认冻结候选配置身份无效")
    configs = [_config("independent", tau, 0) for tau in TEMPERATURES]
    if candidate["kernel"] == "independent":
        if (
            candidate["sweeps"] != 0
            or selection.get("i_star") != candidate_id
            or candidate not in configs
        ):
            raise RuntimeError("最终确认 independent 候选身份无效")
    elif candidate["kernel"] == "factor":
        if (
            candidate["sweeps"] <= 0
            or selection.get("g_star") != candidate_id
        ):
            raise RuntimeError("最终确认 factor 候选身份无效")
        configs.append(dict(candidate))
    else:
        raise RuntimeError("最终确认候选核类型无效")
    return configs, dict(candidate)


def _stage_b_arm_was_certified(config, selection):
    config_id = config["config_id"]
    if config["kernel"] == "independent":
        return bool(
            selection.get("independent", {}).get(config_id, {}).get(
                "eligible"
            ) is True
        )
    return bool(
        selection.get("g_star") == config_id
        and selection.get("self_state_review", {}).get("passed") is True
    )


def _arm_eligibility(configs, aggregates, selection, protocol):
    results = {}
    for config in configs:
        config_id = config["config_id"]
        aggregate = aggregates[config_id]
        gates = {
            "stage_b_arm_was_certified": _stage_b_arm_was_certified(
                config, selection
            ),
            "all_confirmation_seeds_retained": (
                aggregate["seeds"] == protocol["confirmation_seeds"]
            ),
            "all_confirmation_rounds_complete": aggregate[
                "all_rounds_complete"
            ],
            "confirmation_zero_clip_hits": (
                aggregate["total_clip_hit_count"] == 0
            ),
            "confirmation_conditionals_finite_and_bidirectional": aggregate[
                "all_conditionals_finite_and_bidirectional"
            ],
        }
        results[config_id] = {
            "eligible": all(gates.values()),
            "gates": gates,
            "late_window_current_loss_mean": aggregate["metrics"][
                "late_window_current_loss"
            ]["mean"],
        }
    return results


def _comparison_plan(candidate, selection, protocol):
    plan = {}

    def add(baseline_id, role):
        if baseline_id is None:
            raise RuntimeError("最终确认缺少冻结的必需对照")
        plan.setdefault(baseline_id, []).append(role)

    if candidate["kernel"] == "factor":
        add(
            _config_id("independent", candidate["temperature"], 0),
            "same_temperature_independent",
        )
        add(selection.get("i_star"), "stage_b_i_star")
    elif candidate["temperature"] != protocol[
        "incumbent_independent_temperature"
    ]:
        add(
            _config_id(
                "independent",
                protocol["incumbent_independent_temperature"],
                0,
            ),
            "old_grid_incumbent_independent_tau_8",
        )
    return [
        {"baseline_config_id": baseline_id, "roles": roles}
        for baseline_id, roles in plan.items()
    ]


def _paired(rows, candidate_id, baseline_id, roles, protocol):
    candidate = {
        row["run"]["seed"]: row["run"]
        for row in rows if row["config_id"] == candidate_id
    }
    baseline = {
        row["run"]["seed"]: row["run"]
        for row in rows if row["config_id"] == baseline_id
    }
    seeds = protocol["confirmation_seeds"]
    if set(candidate) != set(seeds) or set(baseline) != set(seeds):
        raise RuntimeError("最终确认 audit 发现配对 seeds 不完整")
    differences = np.asarray([
        candidate[seed]["late_window_current_loss_mean"]
        - baseline[seed]["late_window_current_loss_mean"]
        for seed in seeds
    ], dtype=float)
    if not np.all(np.isfinite(differences)):
        raise RuntimeError("最终确认 audit 发现非有限配对差值")
    mean = float(differences.mean())
    median = float(np.median(differences))
    sample_size = len(differences)
    ci_available = sample_size == protocol["formal_paired_sample_size"]
    if ci_available:
        standard_deviation = float(differences.std(ddof=1))
        standard_error = float(standard_deviation / math.sqrt(sample_size))
        half_width = float(
            protocol["paired_t_critical_95_df9"] * standard_error
        )
        confidence_interval = [mean - half_width, mean + half_width]
    else:
        standard_deviation = (
            float(differences.std(ddof=1)) if sample_size > 1 else None
        )
        standard_error = (
            float(standard_deviation / math.sqrt(sample_size))
            if standard_deviation is not None else None
        )
        confidence_interval = None
    wins = int(np.sum(differences < 0.0))
    gates = {
        "all_paired_seeds_retained": sample_size == len(seeds),
        "paired_mean_improves": mean < 0.0,
        "paired_median_improves": median < 0.0,
        "paired_wins_sufficient": wins >= protocol["minimum_paired_wins"],
        "formal_sample_size_for_t_interval": ci_available,
        "paired_t_interval_upper_below_zero": bool(
            confidence_interval is not None
            and confidence_interval[1] < 0.0
        ),
    }
    return {
        "candidate_config_id": candidate_id,
        "baseline_config_id": baseline_id,
        "roles": list(roles),
        "metric": "late_window_current_loss_mean",
        "candidate_minus_baseline_by_seed": [
            {"seed": seed, "difference": float(value)}
            for seed, value in zip(seeds, differences)
        ],
        "sample_size": sample_size,
        "mean_difference": mean,
        "median_difference": median,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence_level": protocol["paired_confidence_level"],
        "t_critical": (
            protocol["paired_t_critical_95_df9"]
            if ci_available else None
        ),
        "confidence_interval": confidence_interval,
        "wins": wins,
        "ties": int(np.sum(differences == 0.0)),
        "losses": int(np.sum(differences > 0.0)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _comparisons(rows, candidate, selection, protocol):
    return {
        item["baseline_config_id"]: _paired(
            rows,
            candidate["config_id"],
            item["baseline_config_id"],
            item["roles"],
            protocol,
        )
        for item in _comparison_plan(candidate, selection, protocol)
    }


def _candidate_self_review(
    report_review,
    candidate,
    rows,
    configs,
    protocol,
    target,
    queries,
    schema,
):
    if candidate["kernel"] == "independent":
        expected = {
            "applicable": False,
            "status": "not_applicable_independent_candidate",
            "passed": None,
            "candidate_config_id": candidate["config_id"],
        }
        stage_a_auditor._assert_same(
            report_review, expected, "report.candidate_self_state_review"
        )
        return expected
    stage_b_shape = dict(report_review)
    stage_b_shape["g0"] = stage_b_shape.pop("candidate_config_id")
    if stage_b_shape.get("status") == "failed_confirmation_no_reselection":
        stage_b_shape["status"] = "failed_no_factor_fallback"
    recomputed = stage_b_auditor._self_review(
        stage_b_shape,
        candidate["config_id"],
        rows,
        configs,
        _execution_protocol(protocol),
        target,
        queries,
        schema,
    )
    expected = dict(recomputed)
    expected["candidate_config_id"] = expected.pop("g0")
    if expected["passed"] is False:
        expected["status"] = "failed_confirmation_no_reselection"
    stage_a_auditor._assert_same(
        report_review, expected, "report.candidate_self_state_review"
    )
    return expected


def _decision(
    mode,
    candidate,
    aggregates,
    eligibility,
    comparisons,
    self_review,
    trajectory_gates,
    formal_result_valid,
):
    candidate_id = candidate["config_id"]
    eligible_ids = [
        config_id for config_id, result in eligibility.items()
        if result["eligible"]
    ]
    if eligible_ids:
        minimum = min(
            aggregates[config_id]["metrics"]["late_window_current_loss"][
                "mean"
            ]
            for config_id in eligible_ids
        )
        lowest_ids = [
            config_id for config_id in eligible_ids
            if aggregates[config_id]["metrics"][
                "late_window_current_loss"
            ]["mean"] == minimum
        ]
    else:
        minimum = None
        lowest_ids = []
    criteria = {
        "formal_result_identity_valid": bool(
            mode != "formal" or formal_result_valid
        ),
        "frozen_candidate_unchanged": True,
        "candidate_confirmation_arm_eligible": bool(
            eligibility.get(candidate_id, {}).get("eligible") is True
        ),
        "candidate_lowest_eligible_late_window_mean": (
            candidate_id in lowest_ids
        ),
        "trajectory_identity_probability_and_clip_gates_passed": bool(
            trajectory_gates["all_trajectory_identity_gates_passed"]
        ),
        "factor_self_state_review_passed_or_not_applicable": bool(
            candidate["kernel"] == "independent"
            or self_review.get("passed") is True
        ),
        "all_required_paired_comparisons_passed": all(
            value["passed"] for value in comparisons.values()
        ),
        "all_confirmation_seeds_and_rounds_retained": bool(
            eligibility.get(candidate_id, {}).get("gates", {}).get(
                "all_confirmation_seeds_retained"
            ) is True
            and eligibility.get(candidate_id, {}).get("gates", {}).get(
                "all_confirmation_rounds_complete"
            ) is True
        ),
        "no_runner_up_reselection_or_retuning": True,
    }
    formal_confirmation_assessed = bool(
        mode == "formal" and formal_result_valid
    )
    if formal_confirmation_assessed:
        confirmed = all(criteria.values())
        status = (
            "confirmed"
            if confirmed else "not_confirmed_no_reselection"
        )
    elif mode == "formal":
        confirmed = None
        status = "invalid_formal_identity_not_assessed"
    else:
        confirmed = None
        status = "smoke_only_not_evidence"
    return {
        "frozen_candidate": candidate_id,
        "frozen_candidate_config": dict(candidate),
        "status": status,
        "confirmed": confirmed,
        "formal_confirmation_assessed": formal_confirmation_assessed,
        "eligible_confirmation_arms": eligible_ids,
        "lowest_eligible_late_window_mean": minimum,
        "lowest_eligible_config_ids": lowest_ids,
        "required_comparison_baselines": list(comparisons),
        "criteria": criteria,
        "no_runner_up_after_failure": True,
        "no_parameter_reselection_on_confirmation_seeds": True,
    }


def audit_confirmation(report_path, output_path):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    report = stage_a_auditor._load_json_strict(report_file)
    if (
        report.get("report_format") != REPORT_FORMAT
        or report.get("status") != "complete"
    ):
        raise RuntimeError("最终确认报告格式或完成状态无效")
    mode = report.get("mode")
    protocol = _expected_protocol(mode)
    stage_a_auditor._assert_same(
        report.get("protocol"), protocol, "report.protocol"
    )
    stage_a_auditor._assert_same(
        report.get("experiment"),
        "issue49_frozen_candidate_final_confirmation",
        "report.experiment",
    )
    stage_a_auditor._assert_same(
        report.get("interpretation"),
        (
            "formal_preregistered_final_confirmation"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
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
        "probe_mutation_disabled": protocol["self_review_probe_mu"] == 0.0,
        "probe_logit_clip_matches": (
            probe.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
        "factor_builder_is_frozen_legacy_path": (
            protocol["factor_builder"] == "legacy_rowwise"
        ),
        "self_review_groups_match": (
            protocol["self_review_required_groups"]
            == list(SELF_REVIEW_GROUPS)
        ),
        "formal_paired_sample_size_is_ten": (
            protocol["formal_paired_sample_size"] == 10
        ),
        "paired_t_critical_is_df9_95_percent": math.isclose(
            protocol["paired_t_critical_95_df9"],
            2.2621571627409915,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "frozen_candidate_only": protocol["frozen_candidate_only"] is True,
        "runner_up_forbidden": protocol["runner_up_forbidden"] is True,
    }
    if not all(implementation.values()):
        raise RuntimeError("最终确认 auditor 发现实现常量漂移")
    stage_a_auditor._assert_same(
        report["implementation_gates"],
        implementation,
        "report.implementation_gates",
    )
    stage_b_report, stage_b_audit = _validate_upstream(report)
    target, queries, schema, hashes = stage_a_auditor._load_frozen_inputs()
    stage_a_auditor._assert_same(
        report["input_sha256"], hashes, "report.input_sha256"
    )
    configs, candidate = _expected_configs(stage_b_report, protocol)
    stage_a_auditor._assert_same(
        report["frozen_candidate"], candidate, "report.frozen_candidate"
    )
    stage_a_auditor._assert_same(
        report["configurations"], configs, "report.configurations"
    )
    expected_protocol_sha = stage_a_auditor._canonical_sha256({
        "protocol": protocol,
        "frozen_candidate": candidate,
        "upstream_sha256": {
            "stage_b_report": report["upstream"][
                "stage_b_report_sha256"
            ],
            "stage_b_audit": report["upstream"][
                "stage_b_audit_sha256"
            ],
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
    execution = _execution_protocol(protocol)
    trajectory_gates = stage_b_auditor._trajectory_gates(
        rows, configs, execution
    )
    if not trajectory_gates["all_trajectory_identity_gates_passed"]:
        failed = [
            name for name, value in trajectory_gates.items() if not value
        ]
        raise RuntimeError(f"最终确认轨迹门禁失败：{failed}")
    stage_a_auditor._assert_same(
        report["trajectory_identity_gates"],
        trajectory_gates,
        "report.trajectory_identity_gates",
    )
    aggregates = stage_b_auditor._aggregates(rows, configs, execution)
    stage_a_auditor._assert_same(
        report["aggregates"], aggregates, "report.aggregates"
    )
    eligibility = _arm_eligibility(
        configs, aggregates, stage_b_report["selection"], protocol
    )
    stage_a_auditor._assert_same(
        report["arm_eligibility"], eligibility, "report.arm_eligibility"
    )
    comparisons = _comparisons(
        rows, candidate, stage_b_report["selection"], protocol
    )
    stage_a_auditor._assert_same(
        report["paired_comparisons"],
        comparisons,
        "report.paired_comparisons",
    )
    self_review = _candidate_self_review(
        report["candidate_self_state_review"],
        candidate,
        rows,
        configs,
        protocol,
        target,
        queries,
        schema,
    )
    formal_identity = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _expected_protocol("formal"),
        "worktree_clean": report["git"]["worktree_clean"],
        "same_commit_as_stage_b": (
            report["git"]["commit"]
            == stage_b_report["git"]["commit"]
        ),
        "upstream_formal_result_valid": (
            stage_b_report["formal_result_valid"] is True
            and stage_b_audit["formal_result_valid"] is True
        ),
        "input_hashes_match": hashes == EXPECTED_INPUT_SHA256,
        "candidate_frozen_from_stage_b": (
            candidate
            == stage_b_report["selection"]["unique_candidate_config"]
        ),
    }
    stage_a_auditor._assert_same(
        report["formal_identity_gates"],
        formal_identity,
        "report.formal_identity_gates",
    )
    formal_result_valid = bool(
        all(formal_identity.values())
        and trajectory_gates["all_trajectory_identity_gates_passed"]
    )
    decision = _decision(
        mode,
        candidate,
        aggregates,
        eligibility,
        comparisons,
        self_review,
        trajectory_gates,
        formal_result_valid,
    )
    stage_a_auditor._assert_same(
        report["decision"], decision, "report.decision"
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
        "upstream_stage_b_report_sha256": report["upstream"][
            "stage_b_report_sha256"
        ],
        "upstream_stage_b_audit_sha256": report["upstream"][
            "stage_b_audit_sha256"
        ],
        "checks": {
            "frozen_protocol_exact": True,
            "stage_b_report_and_audit_bound_and_reaudited": True,
            "frozen_candidate_identity_recomputed": True,
            "five_independent_plus_only_frozen_factor_grid_recomputed": True,
            "trajectory_grid_rng_and_clip_identity_recomputed": True,
            "raw_aggregates_and_eligible_arms_recomputed": True,
            "paired_t_intervals_recomputed": True,
            "candidate_self_state_review_recomputed": True,
            "confirmation_decision_recomputed_without_reselection": True,
        },
        "decision": decision,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    stage_a_auditor._write_json_atomic(output_path, audit)
    return Path(output_path), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_confirmation(args.report, args.output)
    print("\n===== Issue #49 final confirmation audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"status={audit['decision']['status']}")
    print(f"confirmed={audit['decision']['confirmed']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
