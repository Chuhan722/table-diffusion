"""运行 Issue #49 唯一候选的新 seeds 最终确认。"""

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

if __package__:
    from scripts import audit_issue49_stage_b as stage_b_auditor
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue49_stage_t_a_protocol as frozen_protocol
    from scripts import probe_factorized_gibbs_mixing as probe
    from scripts import run_issue49_stage_a as stage_a
    from scripts import run_issue49_stage_b as stage_b
else:
    import audit_issue49_stage_b as stage_b_auditor
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue49_stage_t_a_protocol as frozen_protocol
    import probe_factorized_gibbs_mixing as probe
    import run_issue49_stage_a as stage_a
    import run_issue49_stage_b as stage_b


REPORT_FORMAT = "issue49_confirmation_report_v1"
UPSTREAM_REPORT_FORMAT = "issue49_stage_b_report_v1"
UPSTREAM_AUDIT_FORMAT = "issue49_stage_b_audit_v1"
SELF_REVIEW_GROUPS = ("global", "initial", "mid", "late")


def _protocol(mode):
    return frozen_protocol.confirmation_protocol(mode)


def _execution_protocol(protocol):
    """把确认 seed 名称映射到已验证的通用轨迹/复查实现。"""
    return {
        **protocol,
        "stage_b_seeds": list(protocol["confirmation_seeds"]),
    }


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
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"最终确认冻结协议与实现不一致：{failed}")
    return gates


def _load_upstream(report_path, audit_path, mode, current_git):
    report_file = Path(report_path).resolve()
    audit_file = Path(audit_path).resolve()
    report = stage_a._load_json_strict(report_file)
    audit = stage_a._load_json_strict(audit_file)
    expected_protocol = frozen_protocol.stage_b_protocol(mode)
    selection = report.get("selection", {})
    candidate = selection.get("unique_candidate")
    candidate_config = selection.get("unique_candidate_config")
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
        "selection_matches_audit": audit.get("selection") == selection,
        "same_git_commit": (
            report.get("git", {}).get("commit") == current_git["commit"]
        ),
        "formal_upstream_valid_when_required": (
            mode != "formal"
            or (
                report.get("formal_result_valid") is True
                and audit.get("formal_result_valid") is True
            )
        ),
        "unique_candidate_exists": (
            isinstance(candidate, str) and bool(candidate)
        ),
        "unique_candidate_config_bound": (
            isinstance(candidate_config, dict)
            and candidate_config.get("config_id") == candidate
            and candidate_config in report.get("configurations", [])
        ),
        "runner_up_forbidden_upstream": (
            selection.get("no_runner_up_after_g0_review_failure") is True
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Stage B 上游身份、审计或候选无效：{failed}")
    with tempfile.TemporaryDirectory(
        prefix="issue49-confirmation-upstream-audit-"
    ) as temporary:
        _, fresh_audit = stage_b_auditor.audit_stage_b(
            report_file,
            Path(temporary) / "stage_b_audit.json",
        )
    checks["fresh_stage_b_reaudit_passed"] = bool(
        fresh_audit["passed"] is True
        and fresh_audit["report_sha256"] == stage_a._sha256_file(report_file)
        and fresh_audit["protocol_sha256"] == report["protocol_sha256"]
        and fresh_audit["formal_result_valid"]
        == audit["formal_result_valid"]
        and fresh_audit["selection"] == audit["selection"]
    )
    if not checks["fresh_stage_b_reaudit_passed"]:
        raise RuntimeError("Stage B 上游独立复审失败")
    return report_file, audit_file, report, audit, checks


def _confirmation_configs(protocol, stage_b_report):
    selection = stage_b_report.get("selection", {})
    candidate_id = selection.get("unique_candidate")
    candidate = selection.get("unique_candidate_config")
    if not isinstance(candidate_id, str) or not isinstance(candidate, dict):
        raise RuntimeError("Stage B 未冻结唯一候选，不能运行最终确认")
    if candidate.get("config_id") != candidate_id:
        raise RuntimeError("Stage B 唯一候选 ID 与配置不一致")
    expected_candidate = stage_b._config(
        candidate.get("kernel"),
        candidate.get("temperature"),
        candidate.get("sweeps"),
    )
    if candidate != expected_candidate:
        raise RuntimeError("Stage B 唯一候选配置身份无效")
    if candidate["temperature"] not in protocol["independent_temperatures"]:
        raise RuntimeError("Stage B 唯一候选温度不在冻结网格内")

    configs = [
        stage_b._config("independent", temperature, 0)
        for temperature in protocol["independent_temperatures"]
    ]
    if candidate["kernel"] == "independent":
        if (
            candidate["sweeps"] != 0
            or selection.get("i_star") != candidate_id
            or candidate not in configs
        ):
            raise RuntimeError("冻结的 independent 候选身份无效")
    elif candidate["kernel"] == "factor":
        if (
            candidate["sweeps"] <= 0
            or selection.get("g_star") != candidate_id
        ):
            raise RuntimeError("冻结的 factor 候选身份无效")
        configs.append(dict(candidate))
    else:
        raise RuntimeError("Stage B 唯一候选核类型无效")
    return configs, dict(candidate)


def _run_trajectories(target, queries, schema, marginals, protocol, configs):
    return stage_b._run_trajectories(
        target,
        queries,
        schema,
        marginals,
        _execution_protocol(protocol),
        configs,
    )


def _trajectory_identity_gates(rows, configs, protocol):
    return stage_b._trajectory_identity_gates(
        rows, configs, _execution_protocol(protocol)
    )


def _aggregate_all(rows, configs, protocol):
    return stage_b._aggregate_all(
        rows, configs, _execution_protocol(protocol)
    )


def _candidate_self_review(
    candidate, rows, configs, target, queries, schema, protocol
):
    if candidate["kernel"] == "independent":
        return {
            "applicable": False,
            "status": "not_applicable_independent_candidate",
            "passed": None,
            "candidate_config_id": candidate["config_id"],
        }
    review = stage_b._self_review(
        candidate["config_id"],
        rows,
        configs,
        target,
        queries,
        schema,
        _execution_protocol(protocol),
    )
    result = dict(review)
    result["candidate_config_id"] = result.pop("g0")
    if result["passed"] is False:
        result["status"] = "failed_confirmation_no_reselection"
    return result


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


def _arm_eligibility(configs, aggregates, stage_b_selection, protocol):
    results = {}
    for config in configs:
        config_id = config["config_id"]
        aggregate = aggregates[config_id]
        gates = {
            "stage_b_arm_was_certified": _stage_b_arm_was_certified(
                config, stage_b_selection
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


def _comparison_plan(candidate, stage_b_selection, protocol):
    plan = {}

    def add(baseline_id, role):
        if baseline_id is None:
            raise RuntimeError("冻结候选缺少必需的 Stage B 对照")
        plan.setdefault(baseline_id, []).append(role)

    if candidate["kernel"] == "factor":
        add(
            stage_b._config_id(
                "independent", candidate["temperature"], 0
            ),
            "same_temperature_independent",
        )
        add(stage_b_selection.get("i_star"), "stage_b_i_star")
    elif candidate["temperature"] != protocol[
        "incumbent_independent_temperature"
    ]:
        add(
            stage_b._config_id(
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


def _paired_comparison(rows, candidate_id, baseline_id, roles, protocol):
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
        raise RuntimeError("最终确认候选与对照的配对 seeds 不完整")
    differences = np.asarray([
        candidate[seed]["late_window_current_loss_mean"]
        - baseline[seed]["late_window_current_loss_mean"]
        for seed in seeds
    ], dtype=float)
    if not np.all(np.isfinite(differences)):
        raise RuntimeError("最终确认配对差值包含非有限值")
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


def _all_comparisons(rows, candidate, stage_b_selection, protocol):
    return {
        item["baseline_config_id"]: _paired_comparison(
            rows,
            candidate["config_id"],
            item["baseline_config_id"],
            item["roles"],
            protocol,
        )
        for item in _comparison_plan(candidate, stage_b_selection, protocol)
    }


def _confirmation_decision(
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


def run_confirmation(
    mode, stage_b_report_path, stage_b_audit_path, output_dir
):
    protocol = _protocol(mode)
    implementation_gates = _validate_implementation_constants(protocol)
    output_directory = Path(output_dir)
    report_path = output_directory / "confirmation_report.json"
    if report_path.exists():
        raise FileExistsError(
            f"输出已存在，尚未启动任何确认轨迹：{report_path}"
        )
    git = stage_a._git_identity()
    if mode == "formal" and not git["worktree_clean"]:
        raise RuntimeError("正式最终确认要求 tracked 工作树干净")
    (
        upstream_report_file,
        upstream_audit_file,
        upstream_report,
        upstream_audit,
        upstream_checks,
    ) = _load_upstream(
        stage_b_report_path, stage_b_audit_path, mode, git
    )
    target, queries, schema, marginals, input_hashes = stage_a._load_inputs()
    configs, candidate = _confirmation_configs(protocol, upstream_report)
    upstream_hashes = {
        "stage_b_report": stage_a._sha256_file(upstream_report_file),
        "stage_b_audit": stage_a._sha256_file(upstream_audit_file),
    }
    protocol_sha256 = stage_a._canonical_sha256({
        "protocol": protocol,
        "frozen_candidate": candidate,
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
    eligibility = _arm_eligibility(
        configs, aggregates, upstream_report["selection"], protocol
    )
    comparisons = _all_comparisons(
        rows, candidate, upstream_report["selection"], protocol
    )
    self_review = _candidate_self_review(
        candidate,
        rows,
        configs,
        target,
        queries,
        schema,
        protocol,
    )
    formal_identity_gates = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _protocol("formal"),
        "worktree_clean": git["worktree_clean"],
        "same_commit_as_stage_b": (
            git["commit"] == upstream_report["git"]["commit"]
        ),
        "upstream_formal_result_valid": (
            upstream_report["formal_result_valid"] is True
            and upstream_audit["formal_result_valid"] is True
        ),
        "input_hashes_match": (
            input_hashes == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "candidate_frozen_from_stage_b": (
            candidate == upstream_report["selection"][
                "unique_candidate_config"
            ]
        ),
    }
    formal_result_valid = bool(
        all(formal_identity_gates.values())
        and trajectory_gates["all_trajectory_identity_gates_passed"]
    )
    decision = _confirmation_decision(
        mode,
        candidate,
        aggregates,
        eligibility,
        comparisons,
        self_review,
        trajectory_gates,
        formal_result_valid,
    )
    report = {
        "report_format": REPORT_FORMAT,
        "status": "complete",
        "experiment": "issue49_frozen_candidate_final_confirmation",
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "interpretation": (
            "formal_preregistered_final_confirmation"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "input_sha256": input_hashes,
        "upstream": {
            "stage_b_report_path": str(upstream_report_file),
            "stage_b_report_sha256": upstream_hashes["stage_b_report"],
            "stage_b_audit_path": str(upstream_audit_file),
            "stage_b_audit_sha256": upstream_hashes["stage_b_audit"],
            "stage_b_protocol_sha256": upstream_report[
                "protocol_sha256"
            ],
            "checks": upstream_checks,
        },
        "git": git,
        "command_argv": list(sys.argv),
        "environment": trajectory._environment(protocol["device"]),
        "implementation_gates": implementation_gates,
        "frozen_candidate": candidate,
        "configurations": configs,
        "trajectory_identity_gates": trajectory_gates,
        "trajectories": rows,
        "aggregates": aggregates,
        "arm_eligibility": eligibility,
        "paired_comparisons": comparisons,
        "candidate_self_state_review": self_review,
        "decision": decision,
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
    parser.add_argument("--stage-b-report", required=True)
    parser.add_argument("--stage-b-audit", required=True)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/issue49_high_temperature_factor_gibbs/"
            "confirmation_smoke"
        ),
    )
    args = parser.parse_args()
    if args.mode == "formal" and args.output_dir.endswith(
        "confirmation_smoke"
    ):
        parser.error("正式模式必须显式提供非 smoke 输出目录")
    output, report = run_confirmation(
        args.mode,
        args.stage_b_report,
        args.stage_b_audit,
        args.output_dir,
    )
    print("\n===== Issue #49 final confirmation =====")
    print(f"mode={report['mode']}")
    print(f"frozen_candidate={report['frozen_candidate']['config_id']}")
    print(f"status={report['decision']['status']}")
    print(f"confirmed={report['decision']['confirmed']}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
