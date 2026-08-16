"""Independently audit Issue #52 Stage A factor-mixing qualification."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np

from table_diffevo.experiment_parallel import (
    scientific_sha256,
    validate_max_workers,
)

if __package__:
    from scripts import audit_issue49_stage_a as legacy_auditor
    from scripts import issue52_protocol as frozen_protocol
    from scripts import probe_factorized_gibbs_mixing as probe
    from scripts import run_issue49_stage_a as common
else:
    import audit_issue49_stage_a as legacy_auditor
    import issue52_protocol as frozen_protocol
    import probe_factorized_gibbs_mixing as probe
    import run_issue49_stage_a as common


AUDIT_FORMAT = "issue52_stage_a_mixing_audit_v1"
REPORT_FORMAT = "issue52_stage_a_mixing_report_v1"
LIBRARY_FORMAT = "issue52_stage_a_state_library_v1"
LIBRARY_AUDIT_FORMAT = "issue52_stage_a_state_library_audit_v1"


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _assert_same(actual, expected, path):
    if isinstance(expected, bool) or isinstance(actual, bool):
        if actual is not expected:
            raise RuntimeError(f"{path} 不一致：{actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(
        actual, (int, float)
    ):
        if not (
            math.isfinite(float(actual))
            and math.isfinite(float(expected))
            and math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise RuntimeError(f"{path} 不一致：{actual!r} != {expected!r}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"{path} 字段不一致")
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{path} 列表长度不一致")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _assert_same(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise RuntimeError(f"{path} 不一致：{actual!r} != {expected!r}")


def _expected_state_ids(protocol):
    state_ids = []
    for seed in protocol["state_library_seeds"]:
        state_ids.append(f"seed_{seed}_initial_round_0")
        for state_round in protocol["source_snapshot_rounds"][1:]:
            for temperature in protocol["source_temperatures"]:
                state_ids.append(
                    f"seed_{seed}_round_{state_round}_source_"
                    f"{_tau_key(temperature)}"
                )
    return state_ids


def _implementation_gates(protocol):
    return {
        "probe_rho_exact": probe.RHO == protocol["rho"],
        "probe_eta_exact": probe.ETA == protocol["eta"],
        "probe_mu_disabled": protocol["probe_mu"] == 0.0,
        "probe_logit_clip_exact": (
            probe.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
        "probe_record_count_exact": probe.N_RECORDS == 300,
        "factor_order_exact": protocol["max_factor_order"] == 3,
        "device_is_numpy": protocol["device"] == "numpy",
    }


def _library_binding(library_file, audit_file, protocol):
    library = common._load_json_strict(library_file)
    state_audit = common._load_json_strict(audit_file)
    library_sha = common._sha256_file(library_file)
    audit_sha = common._sha256_file(audit_file)
    expected_artifacts = protocol[
        "expected_state_library_artifact_sha256"
    ]
    expected_library_protocol = (
        frozen_protocol.stage_a_state_library_protocol(protocol["mode"])
    )
    state_ids = [state.get("state_id") for state in library.get("states", [])]
    expected_ids = _expected_state_ids(protocol)
    group_counts = {
        group: sum(
            state.get("state_family") == group
            for state in library.get("states", [])
        )
        for group in protocol["required_state_groups"]
    }
    gates = {
        "library_format_exact": (
            library.get("state_library_format") == LIBRARY_FORMAT
        ),
        "library_status_complete": library.get("status") == "complete",
        "library_mode_exact": (
            library.get("mode") == protocol["state_library_mode"]
        ),
        "library_protocol_exact": (
            library.get("protocol") == expected_library_protocol
        ),
        "library_protocol_sha_exact": (
            protocol["expected_state_library_protocol_sha256"] is None
            or library.get("protocol_sha256")
            == protocol["expected_state_library_protocol_sha256"]
        ),
        "library_scientific_sha_exact": (
            protocol["expected_state_library_scientific_sha256"] is None
            or library.get("state_library_scientific_sha256")
            == protocol["expected_state_library_scientific_sha256"]
        ),
        "library_file_sha_exact": (
            expected_artifacts is None
            or library_sha == expected_artifacts["library"]
        ),
        "library_gates_passed": bool(
            library.get("gates") and all(library["gates"].values())
        ),
        "library_state_count_exact": (
            len(state_ids) == protocol["expected_state_count"]
        ),
        "library_state_ids_unique": len(state_ids) == len(set(state_ids)),
        "library_state_order_exact": state_ids == expected_ids,
        "library_manifest_order_exact": (
            library.get("manifest", {}).get("state_ids_in_fixed_order")
            == expected_ids
        ),
        "library_groups_exact": (
            group_counts
            == {
                group: protocol["states_per_group"]
                for group in protocol["required_state_groups"]
            }
        ),
        "library_source_inputs_exact": (
            library.get("source_identity", {}).get("input_sha256")
            == protocol["input_sha256"]
        ),
        "audit_format_exact": (
            state_audit.get("audit_format") == LIBRARY_AUDIT_FORMAT
        ),
        "audit_status_complete": state_audit.get("status") == "complete",
        "audit_passed": state_audit.get("passed") is True,
        "audit_library_path_exact": (
            Path(state_audit.get("library_path", "")).resolve()
            == library_file
        ),
        "audit_library_sha_exact": (
            state_audit.get("library_sha256") == library_sha
        ),
        "audit_scientific_sha_exact": (
            state_audit.get("state_library_scientific_sha256")
            == library.get("state_library_scientific_sha256")
        ),
        "audit_protocol_sha_exact": (
            state_audit.get("protocol_sha256")
            == library.get("protocol_sha256")
        ),
        "audit_file_sha_exact": (
            expected_artifacts is None
            or audit_sha == expected_artifacts["audit"]
        ),
        "audit_manifest_exact": (
            state_audit.get("recomputed", {}).get("manifest")
            == library.get("manifest")
        ),
        "formal_flags_exact": (
            protocol["mode"] != "formal"
            or (
                library.get("formal_result_valid") is True
                and state_audit.get("formal_result_valid") is True
            )
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"独立审计发现状态库绑定失败：{failed}")
    identity = {
        "library_path": str(library_file),
        "library_sha256": library_sha,
        "audit_path": str(audit_file),
        "audit_sha256": audit_sha,
        "state_library_scientific_sha256": library[
            "state_library_scientific_sha256"
        ],
        "binding_gates": gates,
    }
    return library, identity


def _validate_and_recompute_rows(
    rows, library, protocol, temperature, sweeps
):
    entries = {entry["state_id"]: entry for entry in library["states"]}
    if [row.get("state_id") for row in rows] != list(entries):
        raise RuntimeError("逐状态结果缺失、重复或乱序")
    configs = probe._config_names([temperature], [0, sweeps])
    tau_key = _tau_key(temperature)
    raw_rows = copy.deepcopy(rows)
    recomputed_rows = copy.deepcopy(rows)
    for row in recomputed_rows:
        entry = entries[row["state_id"]]
        if (
            any(
                row.get(key) != entry.get(key)
                for key in (
                    "seed",
                    "state_round",
                    "state_family",
                    "source_temperature",
                )
            )
            or row.get("state_sha256")
            != entry["snapshot"]["state_sha256"]
        ):
            raise RuntimeError(f"逐状态 wrapper 身份错误：{row['state_id']}")
        result = row.get("probe", {})
        if (
            result.get("n_proposals") != protocol["proposals_per_state"]
            or result.get("state_sha256") != row["state_sha256"]
            or list(result.get("proposal_rows", {})) != configs
            or list(result.get("kernel_summary", {})) != configs
            or list(result.get("proposal_summary", {})) != configs
            or list(result.get("conditional_logit_diagnostics", {}))
            != [tau_key]
        ):
            raise RuntimeError(f"逐状态探针配置错误：{row['state_id']}")
        controls = result.get("external_snapshot_controls", {})
        snapshot = entry["snapshot"]
        if (
            controls.get("source_seed") != entry["seed"]
            or controls.get("state_round") != entry["state_round"]
            or controls.get("state_sha256") != snapshot["state_sha256"]
            or result.get("state_loss") != snapshot["current_loss"]
            or result.get("probe_alpha") != snapshot["donor_alpha"]
            or result.get("direction_reference_scale")
            != snapshot["direction_reference_scale"]
            or result.get("reference_scale_proposal_index") is not None
        ):
            raise RuntimeError(f"冻结状态控制被改变：{row['state_id']}")
        for config in configs:
            proposal_rows = result["proposal_rows"][config]
            if len(proposal_rows) != protocol["proposals_per_state"]:
                raise RuntimeError(f"proposal 数量错误：{row['state_id']}")
            kernel = legacy_auditor._kernel_from_proposal_rows(
                proposal_rows
            )
            proposal_summary = legacy_auditor._proposal_summary(
                proposal_rows
            )
            _assert_same(
                result["kernel_summary"][config],
                kernel,
                f"{row['state_id']}.kernel.{config}",
            )
            _assert_same(
                result["proposal_summary"][config],
                proposal_summary,
                f"{row['state_id']}.proposal.{config}",
            )
            result["kernel_summary"][config] = kernel
            result["proposal_summary"][config] = proposal_summary
        logit = result["conditional_logit_diagnostics"][tau_key]
        if (
            len(logit.get("clip_hit_conditions", []))
            != logit.get("clip_hit_count")
            or logit.get("logit_clip") != protocol["logit_clip"]
            or logit.get("raw_logit_strictly_inside_clip")
            != (logit.get("raw_logit_abs_max") < protocol["logit_clip"])
        ):
            raise RuntimeError(f"条件 logit 诊断不一致：{row['state_id']}")
    return raw_rows, recomputed_rows


def _shared_condition_payload(rows, temperature):
    baseline = probe._gibbs_name(temperature, 0)
    joint = probe._joint_name(temperature)
    tau_key = _tau_key(temperature)
    payload = []
    for row in rows:
        result = row["probe"]
        factor = result["factor_diagnostics"]
        payload.append({
            "state_id": row["state_id"],
            "state_sha256": row["state_sha256"],
            "direction_reference_scale": result[
                "direction_reference_scale"
            ],
            "external_snapshot_controls": result[
                "external_snapshot_controls"
            ],
            "factor_static": {
                "active_rows": factor["active_rows"],
                "exact_energy_max_error": factor[
                    "exact_energy_max_error"
                ],
                "one_hot_direction_max_error": factor[
                    "one_hot_direction_max_error"
                ],
                "mean_factor_count": factor["mean_factor_count"],
                "mean_factor_table_entries": factor[
                    "mean_factor_table_entries"
                ],
                "maximum_active_factor_order": factor[
                    "maximum_active_factor_order"
                ],
            },
            "conditional_logits": result[
                "conditional_logit_diagnostics"
            ][tau_key],
            "baseline_kernel": result["kernel_summary"][baseline],
            "baseline_proposals": result["proposal_rows"][baseline],
            "joint_kernel": result["kernel_summary"][joint],
            "joint_proposals": result["proposal_rows"][joint],
        })
    return payload


def _aggregate_group(rows, name, temperature, sweeps):
    baseline = probe._gibbs_name(temperature, 0)
    candidate = probe._gibbs_name(temperature, sweeps)
    joint = probe._joint_name(temperature)
    kernels = {
        config: legacy_auditor._weighted_kernel(rows, config)
        for config in (baseline, candidate, joint)
    }
    initial_gap = kernels[baseline]["absolute_expected_direction_gap"]
    remaining = kernels[candidate]["absolute_expected_direction_gap"]
    recovery = (
        1.0 - remaining / initial_gap
        if initial_gap is not None and initial_gap > 0.0
        else (1.0 if initial_gap == 0.0 else None)
    )
    return {
        "group": name,
        "state_count": len(rows),
        "state_ids": [row["state_id"] for row in rows],
        "kernel_summary": kernels,
        "expected_direction_gap_recovery": recovery,
    }


def _recompute_attempt(
    actual,
    library,
    protocol,
    temperature,
    sweeps,
    expected_shared_sha,
):
    raw_rows, recomputed_rows = _validate_and_recompute_rows(
        actual.get("state_results", []),
        library,
        protocol,
        temperature,
        sweeps,
    )
    tau_key = _tau_key(temperature)
    groups = {
        group: _aggregate_group(
            [
                row for row in recomputed_rows
                if row["state_family"] == group
            ],
            group,
            temperature,
            sweeps,
        )
        for group in protocol["required_state_groups"]
    }
    global_group = _aggregate_group(
        recomputed_rows, "global", temperature, sweeps
    )
    factor = legacy_auditor._aggregate_factor(
        recomputed_rows, [temperature]
    )
    probabilities = legacy_auditor._aggregate_probability(
        recomputed_rows, temperature
    )
    logit = legacy_auditor._aggregate_logit(
        recomputed_rows, temperature
    )
    logit["all_finite"] = legacy_auditor._all_numeric_finite(logit)
    production = legacy_auditor._aggregate_production(
        recomputed_rows, temperature
    )
    shared_sha = scientific_sha256(
        _shared_condition_payload(raw_rows, temperature)
    )
    correctness = {
        "state_count_and_order_exact": (
            len(raw_rows) == protocol["expected_state_count"]
        ),
        "all_required_groups_nonempty": all(
            group["state_count"] == protocol["states_per_group"]
            for group in groups.values()
        ),
        "exact_energy_within_tolerance": (
            factor["exact_energy_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "one_hot_within_tolerance": (
            factor["one_hot_direction_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "tvd_monotonic_within_tolerance": (
            factor["tvd_snapshot_increase_max_by_temperature"][tau_key]
            <= protocol["tvd_monotonic_tolerance"]
        ),
        "probability_distributions_present": (
            probabilities["distribution_count"] > 0
        ),
        "probabilities_finite": probabilities["all_finite"],
        "probabilities_nonnegative": probabilities["all_nonnegative"],
        "probability_sums_within_tolerance": (
            probabilities["probability_sum_max_error"]
            <= protocol["probability_sum_tolerance"]
        ),
        "production_replay_present": (
            production["comparison_count"] > 0
            and production["microsteps"] > 0
        ),
        "production_matches_exact_tape_replay": (
            production["all_exact_tape_replays_match"]
            and production["mismatch_count"] == 0
        ),
        "shared_conditions_match_prior_attempts": (
            expected_shared_sha is None or shared_sha == expected_shared_sha
        ),
        "all_numeric_values_finite": common._all_numeric_finite(raw_rows),
    }
    correctness_passed = all(correctness.values())
    candidate = probe._gibbs_name(temperature, sweeps)
    group_checks = {}
    for group, aggregate in groups.items():
        kernel = aggregate["kernel_summary"][candidate]
        recovery = aggregate["expected_direction_gap_recovery"]
        group_checks[group] = {
            "participating_active_rows": kernel[
                "participating_active_rows"
            ],
            "tvd_to_joint": kernel["tvd_to_joint"],
            "expected_direction_gap_recovery": recovery,
            "passed": bool(
                kernel["participating_active_rows"] > 0
                and kernel["tvd_to_joint"] <= protocol["tvd_threshold"]
                and recovery is not None
                and recovery >= protocol["recovery_threshold"]
            ),
        }
    qualification = {
        "correctness_passed": correctness_passed,
        "conditional_logits_present": logit["condition_count"] > 0,
        "conditional_logits_finite": logit["all_finite"],
        "zero_clip_hits": (
            logit["clip_hit_count"] == 0
            and logit["raw_logit_strictly_inside_clip"]
        ),
        "conditionals_bidirectional": bool(
            logit["all_conditionals_bidirectional"]
            and logit["minimum_binary_outcome_probability"] is not None
            and logit["minimum_binary_outcome_probability"] > 0.0
        ),
        "all_required_groups_pass": all(
            group["passed"] for group in group_checks.values()
        ),
    }
    expected = {
        "temperature": float(temperature),
        "sweeps": int(sweeps),
        "state_results": raw_rows,
        "state_results_scientific_sha256": scientific_sha256(raw_rows),
        "shared_condition_scientific_sha256": shared_sha,
        "factor_diagnostics": factor,
        "probability_diagnostics": probabilities,
        "conditional_logit_diagnostics": logit,
        "production_sampler_diagnostics": production,
        "global_diagnostic": global_group,
        "required_group_checks": group_checks,
        "correctness_gates": correctness,
        "all_correctness_gates_passed": correctness_passed,
        "qualification_gates": qualification,
        "passed": all(qualification.values()),
        "elapsed_sec": actual.get("elapsed_sec"),
    }
    if (
        not isinstance(expected["elapsed_sec"], (int, float))
        or not np.isfinite(expected["elapsed_sec"])
        or expected["elapsed_sec"] < 0.0
    ):
        raise RuntimeError("attempt elapsed_sec 无效")
    _assert_same(actual, expected, f"tau={temperature:g}.sweeps={sweeps}")
    return expected, shared_sha


def _sequence_result(temperature, attempts, protocol):
    attempted = [attempt["sweeps"] for attempt in attempts]
    expected_prefix = protocol["candidate_sweeps"][:len(attempted)]
    if not attempts or attempted != expected_prefix:
        raise RuntimeError(f"tau={temperature:g} sweeps 缺失、乱序或跳级")
    passing = [index for index, attempt in enumerate(attempts) if attempt["passed"]]
    if passing:
        first = passing[0]
        if first != len(attempts) - 1:
            raise RuntimeError(f"tau={temperature:g} 首次通过后仍运行了后续 sweeps")
        return {
            "temperature": float(temperature),
            "status": "qualified",
            "minimal_sufficient_sweeps": attempts[first]["sweeps"],
            "attempted_sweeps": attempted,
            "attempts": attempts,
        }
    if attempted != protocol["candidate_sweeps"]:
        raise RuntimeError(f"tau={temperature:g} 未通过却错误提前停止")
    return {
        "temperature": float(temperature),
        "status": "unqualified_at_sweeps_cap",
        "minimal_sufficient_sweeps": None,
        "attempted_sweeps": attempted,
        "attempts": attempts,
    }


def audit_stage_a_mixing(
    report_path,
    state_library_path,
    state_library_audit_path,
    output_path,
):
    started = time.perf_counter()
    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(f"审计输出已存在，不覆盖：{output_file}")
    report_file = Path(report_path).resolve()
    library_file = Path(state_library_path).resolve()
    state_audit_file = Path(state_library_audit_path).resolve()
    report = common._load_json_strict(report_file)
    if report.get("report_format") != REPORT_FORMAT:
        raise RuntimeError("Stage A mixing 报告格式不匹配")
    if report.get("status") != "complete":
        raise RuntimeError("Stage A mixing 报告未完整结束")
    mode = report.get("mode")
    protocol = frozen_protocol.stage_a_mixing_protocol(mode)
    _assert_same(report.get("protocol"), protocol, "report.protocol")
    _assert_same(
        report.get("experiment"),
        "issue52_stage_a_factor_mixing_qualification",
        "report.experiment",
    )
    _assert_same(
        report.get("interpretation"),
        (
            "formal_preregistered_stage_a_mixing"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "report.interpretation",
    )
    implementation = _implementation_gates(protocol)
    if not all(implementation.values()):
        raise RuntimeError("独立审计发现实现常量偏离冻结协议")
    _assert_same(
        report.get("implementation_gates"),
        implementation,
        "report.implementation_gates",
    )
    library, library_identity = _library_binding(
        library_file, state_audit_file, protocol
    )
    _assert_same(
        report.get("state_library"),
        library_identity,
        "report.state_library",
    )
    _, _, _, _, input_hashes = common._load_inputs()
    _assert_same(report.get("input_sha256"), input_hashes, "report.input")
    expected_protocol_sha = common._canonical_sha256({
        "protocol": protocol,
        "input_sha256": input_hashes,
        "git_commit": report["git"]["commit"],
        "state_library_scientific_sha256": library_identity[
            "state_library_scientific_sha256"
        ],
    })
    _assert_same(
        report.get("protocol_sha256"),
        expected_protocol_sha,
        "report.protocol_sha256",
    )
    workers = validate_max_workers(report.get("runtime", {}).get("max_workers"))
    _assert_same(
        report.get("runtime"),
        {
            "max_workers": workers,
            "worker_count_is_nonscientific": True,
        },
        "report.runtime",
    )

    actual_temperatures = report.get("temperatures", {})
    expected_keys = [
        _tau_key(temperature)
        for temperature in protocol["evaluation_temperatures"]
    ]
    if list(actual_temperatures) != expected_keys:
        raise RuntimeError("evaluation tau 缺失、重复或乱序")
    recomputed_temperatures = {}
    for temperature in protocol["evaluation_temperatures"]:
        key = _tau_key(temperature)
        actual_result = actual_temperatures[key]
        actual_attempts = actual_result.get("attempts", [])
        attempted_sweeps = [
            attempt.get("sweeps") for attempt in actual_attempts
        ]
        if any(
            not isinstance(sweeps, int)
            or isinstance(sweeps, bool)
            or sweeps > protocol["sweeps_hard_cap"]
            for sweeps in attempted_sweeps
        ):
            raise RuntimeError(f"tau={temperature:g} 包含非法 sweeps")
        recomputed_attempts = []
        shared_sha = None
        for actual_attempt, sweeps in zip(
            actual_attempts, attempted_sweeps
        ):
            expected_attempt, current_sha = _recompute_attempt(
                actual_attempt,
                library,
                protocol,
                temperature,
                sweeps,
                shared_sha,
            )
            if shared_sha is None:
                shared_sha = current_sha
            recomputed_attempts.append(expected_attempt)
        recomputed = _sequence_result(
            temperature, recomputed_attempts, protocol
        )
        _assert_same(actual_result, recomputed, f"report.temperatures.{key}")
        recomputed_temperatures[key] = recomputed

    execution_gates = {
        "temperature_grid_exact": True,
        "attempt_sequences_exact": True,
        "no_sweeps_above_hard_cap": True,
        "all_executed_attempts_correct": all(
            attempt["all_correctness_gates_passed"]
            for result in recomputed_temperatures.values()
            for attempt in result["attempts"]
        ),
        "shared_conditions_exact_within_tau": all(
            len({
                attempt["shared_condition_scientific_sha256"]
                for attempt in result["attempts"]
            }) == 1
            for result in recomputed_temperatures.values()
        ),
    }
    _assert_same(
        report.get("execution_gates"),
        execution_gates,
        "report.execution_gates",
    )
    selection = {
        "minimal_sufficient_sweeps": {
            key: result["minimal_sufficient_sweeps"]
            for key, result in recomputed_temperatures.items()
        },
        "qualified_temperatures": [
            result["temperature"]
            for result in recomputed_temperatures.values()
            if result["status"] == "qualified"
        ],
        "unqualified_temperatures": [
            result["temperature"]
            for result in recomputed_temperatures.values()
            if result["status"] != "qualified"
        ],
    }
    _assert_same(report.get("selection"), selection, "report.selection")
    formal_identity = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": (
            protocol == frozen_protocol.stage_a_mixing_protocol("formal")
        ),
        "worktree_clean": report["git"]["worktree_clean"],
        "input_hashes_exact": (
            input_hashes == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "frozen_library_artifacts_exact": bool(
            mode == "formal"
            and all(library_identity["binding_gates"].values())
        ),
    }
    _assert_same(
        report.get("formal_identity_gates"),
        formal_identity,
        "report.formal_identity_gates",
    )
    formal_result_valid = bool(
        all(formal_identity.values()) and all(execution_gates.values())
    )
    _assert_same(
        report.get("formal_result_valid"),
        formal_result_valid,
        "report.formal_result_valid",
    )
    execution_sha = scientific_sha256({
        "protocol_sha256": expected_protocol_sha,
        "library": library_identity,
        # The scientific hash commits to the report's exact serialized
        # floating-point payload.  Every scientific value in that payload has
        # already been independently recomputed above (within the frozen
        # numerical tolerance); hashing the original representation avoids a
        # false mismatch from harmless changes in summation order.
        "temperatures": actual_temperatures,
        "execution_gates": execution_gates,
    })
    _assert_same(
        report.get("execution_scientific_sha256"),
        execution_sha,
        "report.execution_scientific_sha256",
    )
    audit = {
        "audit_format": AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "formal_result_valid": formal_result_valid,
        "report_path": str(report_file),
        "report_sha256": common._sha256_file(report_file),
        "state_library_path": str(library_file),
        "state_library_sha256": library_identity["library_sha256"],
        "state_library_audit_path": str(state_audit_file),
        "state_library_audit_sha256": library_identity["audit_sha256"],
        "protocol_sha256": expected_protocol_sha,
        "execution_scientific_sha256": execution_sha,
        "checks": {
            "frozen_protocol_exact": True,
            "state_library_and_audit_bound": True,
            "state_order_and_groups_recomputed": True,
            "per_proposal_summaries_recomputed": True,
            "per_group_tvd_and_recovery_recomputed": True,
            "shared_condition_tape_recomputed": True,
            "production_exact_replay_rechecked": True,
            "incremental_stop_rule_recomputed": True,
            "selection_recomputed": True,
        },
        "selection": selection,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    common._write_json_atomic(output_file, audit)
    return output_file, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--state-library-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_stage_a_mixing(
        args.report,
        args.state_library,
        args.state_library_audit,
        args.output,
    )
    print("\n===== Issue #52 Stage A mixing audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
