"""Run Issue #52 Stage A factor-Gibbs mixing qualification.

The runner consumes an already audited Stage A state library.  It never
regenerates Stage T trajectories or states.  For each evaluation temperature
it executes the frozen sweep sequence 8 -> 16 -> 32 and stops at the first
candidate that passes every required state group.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from table_diffevo.experiment_parallel import (
    run_ordered_process_tasks,
    scientific_sha256,
    validate_max_workers,
)

if __package__:
    from scripts import issue52_protocol as frozen_protocol
    from scripts import probe_factorized_gibbs_mixing as probe
    from scripts import run_issue49_stage_a as common
else:
    import issue52_protocol as frozen_protocol
    import probe_factorized_gibbs_mixing as probe
    import run_issue49_stage_a as common


REPORT_FORMAT = "issue52_stage_a_mixing_report_v1"
LIBRARY_FORMAT = "issue52_stage_a_state_library_v1"
LIBRARY_AUDIT_FORMAT = "issue52_stage_a_state_library_audit_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_METRICS = (
    "tvd_to_joint",
    "kl_to_joint",
    "kl_to_reference",
    "entropy",
    "joint_entropy",
    "expected_direction",
    "joint_expected_direction",
    "absolute_expected_direction_gap",
    "negative_mass",
    "joint_negative_mass",
)


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


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _expected_state_ids(protocol):
    state_ids = []
    temperatures = protocol["source_temperatures"]
    for seed in protocol["state_library_seeds"]:
        state_ids.append(f"seed_{seed}_initial_round_0")
        for state_round in protocol["source_snapshot_rounds"][1:]:
            for temperature in temperatures:
                state_ids.append(
                    f"seed_{seed}_round_{state_round}_source_"
                    f"{_tau_key(temperature)}"
                )
    return state_ids


def _validate_implementation_constants(protocol):
    gates = {
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
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Stage A 冻结协议与实现常量不一致：{failed}")
    return gates


def _validate_state_library(library_path, audit_path, protocol):
    library_file = Path(library_path).resolve()
    audit_file = Path(audit_path).resolve()
    library = common._load_json_strict(library_file)
    audit = common._load_json_strict(audit_file)
    library_sha256 = common._sha256_file(library_file)
    audit_sha256 = common._sha256_file(audit_file)
    expected_artifacts = protocol[
        "expected_state_library_artifact_sha256"
    ]
    expected_library_protocol = (
        frozen_protocol.stage_a_state_library_protocol(protocol["mode"])
    )
    expected_state_ids = _expected_state_ids(protocol)
    states = library.get("states", [])
    state_ids = [state.get("state_id") for state in states]
    group_counts = {
        group: sum(state.get("state_family") == group for state in states)
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
            or library_sha256 == expected_artifacts["library"]
        ),
        "library_gates_passed": bool(
            library.get("gates")
            and all(library["gates"].values())
        ),
        "library_state_count_exact": (
            len(states) == protocol["expected_state_count"]
        ),
        "library_state_ids_unique": len(state_ids) == len(set(state_ids)),
        "library_state_order_exact": state_ids == expected_state_ids,
        "library_manifest_order_exact": (
            library.get("manifest", {}).get("state_ids_in_fixed_order")
            == expected_state_ids
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
            audit.get("audit_format") == LIBRARY_AUDIT_FORMAT
        ),
        "audit_status_complete": audit.get("status") == "complete",
        "audit_passed": audit.get("passed") is True,
        "audit_library_path_exact": (
            Path(audit.get("library_path", "")).resolve() == library_file
        ),
        "audit_library_sha_exact": (
            audit.get("library_sha256") == library_sha256
        ),
        "audit_scientific_sha_exact": (
            audit.get("state_library_scientific_sha256")
            == library.get("state_library_scientific_sha256")
        ),
        "audit_protocol_sha_exact": (
            audit.get("protocol_sha256") == library.get("protocol_sha256")
        ),
        "audit_file_sha_exact": (
            expected_artifacts is None
            or audit_sha256 == expected_artifacts["audit"]
        ),
        "audit_manifest_exact": (
            audit.get("recomputed", {}).get("manifest")
            == library.get("manifest")
        ),
        "formal_flags_exact": (
            protocol["mode"] != "formal"
            or (
                library.get("formal_result_valid") is True
                and audit.get("formal_result_valid") is True
            )
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Stage A 状态库绑定失败：{failed}")
    return library, audit, {
        "library_path": str(library_file),
        "library_sha256": library_sha256,
        "audit_path": str(audit_file),
        "audit_sha256": audit_sha256,
        "state_library_scientific_sha256": library[
            "state_library_scientific_sha256"
        ],
        "binding_gates": gates,
    }


@dataclass(frozen=True)
class _ProbeTask:
    state_index: int
    entry: dict
    temperature: float
    sweeps: int
    proposals: int

    @property
    def task_id(self):
        return (
            f"{self.entry['state_id']}__tau_{self.temperature:g}"
            f"__sweeps_{self.sweeps}"
        )


def _execute_probe_task(task, *, target, queries, schema, protocol):
    entry = task.entry
    snapshot = entry["snapshot"]
    state, controls = probe._restore_current_snapshot(
        snapshot,
        target,
        queries,
        schema,
        device=protocol["device"],
    )
    if (
        controls["source_seed"] != entry["seed"]
        or controls["state_round"] != entry["state_round"]
        or controls["state_sha256"] != snapshot["state_sha256"]
    ):
        raise RuntimeError(f"状态 wrapper 与快照不一致：{entry['state_id']}")
    result = probe._probe_state(
        state,
        target,
        queries,
        schema,
        seed=entry["seed"],
        state_index=task.state_index,
        state_rounds=entry["state_round"],
        temperatures=[task.temperature],
        sweeps=[0, task.sweeps],
        proposals=task.proposals,
        device=protocol["device"],
        max_active_attributes=protocol["max_active_attributes"],
        external_snapshot_controls=controls,
    )
    if (
        result["state_sha256"] != snapshot["state_sha256"]
        or result["state_loss"] != snapshot["current_loss"]
        or result["probe_alpha"] != snapshot["donor_alpha"]
        or result["direction_reference_scale"]
        != snapshot["direction_reference_scale"]
        or result["reference_scale_proposal_index"] is not None
    ):
        raise RuntimeError(f"探针改变了冻结状态控制：{entry['state_id']}")
    return {
        "task_id": task.task_id,
        "state_id": entry["state_id"],
        "seed": entry["seed"],
        "state_round": entry["state_round"],
        "state_family": entry["state_family"],
        "source_temperature": entry["source_temperature"],
        "state_sha256": snapshot["state_sha256"],
        "probe": result,
    }


def _run_raw_attempt(
    library,
    target,
    queries,
    schema,
    protocol,
    *,
    temperature,
    sweeps,
    max_workers,
):
    tasks = [
        _ProbeTask(
            state_index=index,
            entry=entry,
            temperature=float(temperature),
            sweeps=int(sweeps),
            proposals=protocol["proposals_per_state"],
        )
        for index, entry in enumerate(library["states"])
    ]
    worker = partial(
        _execute_probe_task,
        target=target,
        queries=queries,
        schema=schema,
        protocol=protocol,
    )
    rows = run_ordered_process_tasks(
        worker, tasks, max_workers=max_workers
    )
    if [row.get("task_id") for row in rows] != [
        task.task_id for task in tasks
    ]:
        raise RuntimeError("Stage A 并行返回顺序或任务身份错误")
    return rows


def _kernel_from_proposal_rows(rows):
    total = int(sum(
        row["kernel"]["participating_active_rows"] for row in rows
    ))
    result = {
        "participating_active_rows": total,
        "active_blocks": int(sum(
            row["kernel"]["active_blocks"] for row in rows
        )),
    }
    for metric in KERNEL_METRICS:
        result[metric] = (
            float(sum(
                row["kernel"][metric]
                * row["kernel"]["participating_active_rows"]
                for row in rows
            ) / total)
            if total else 0.0
        )
    return result


def _validate_state_rows(rows, library, protocol, temperature, sweeps):
    entries = {entry["state_id"]: entry for entry in library["states"]}
    if [row.get("state_id") for row in rows] != list(entries):
        raise RuntimeError("Stage A 逐状态顺序与状态库不一致")
    configs = probe._config_names([temperature], [0, sweeps])
    tau_key = _tau_key(temperature)
    for row in rows:
        entry = entries[row["state_id"]]
        if (
            any(
                row[key] != entry[key]
                for key in (
                    "seed",
                    "state_round",
                    "state_family",
                    "source_temperature",
                )
            )
            or row["state_sha256"] != entry["snapshot"]["state_sha256"]
        ):
            raise RuntimeError(f"逐状态身份错误：{row['state_id']}")
        result = row["probe"]
        if (
            result["n_proposals"] != protocol["proposals_per_state"]
            or result["state_sha256"] != row["state_sha256"]
            or list(result["proposal_rows"]) != configs
            or list(result["kernel_summary"]) != configs
            or list(result["proposal_summary"]) != configs
            or list(result["conditional_logit_diagnostics"])
            != [tau_key]
        ):
            raise RuntimeError(f"逐状态探针配置错误：{row['state_id']}")
        for config in configs:
            proposal_rows = result["proposal_rows"][config]
            if len(proposal_rows) != protocol["proposals_per_state"]:
                raise RuntimeError(f"proposal 数量错误：{row['state_id']}")
            _assert_same(
                result["kernel_summary"][config],
                _kernel_from_proposal_rows(proposal_rows),
                f"{row['state_id']}.kernel.{config}",
            )
            _assert_same(
                result["proposal_summary"][config],
                probe._summarize_proposals(proposal_rows),
                f"{row['state_id']}.proposal.{config}",
            )
        logit = result["conditional_logit_diagnostics"][tau_key]
        if (
            len(logit["clip_hit_conditions"])
            != logit["clip_hit_count"]
            or logit["logit_clip"] != protocol["logit_clip"]
            or logit["raw_logit_strictly_inside_clip"]
            != (logit["raw_logit_abs_max"] < protocol["logit_clip"])
        ):
            raise RuntimeError(f"logit 诊断不一致：{row['state_id']}")


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
        config: common._weighted_kernel(rows, config)
        for config in (baseline, candidate, joint)
    }
    initial_gap = kernels[baseline]["absolute_expected_direction_gap"]
    remaining_gap = kernels[candidate][
        "absolute_expected_direction_gap"
    ]
    recovery = (
        1.0 - remaining_gap / initial_gap
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


def _summarize_attempt(
    rows,
    library,
    protocol,
    temperature,
    sweeps,
    *,
    expected_shared_condition_sha256,
    elapsed_sec,
):
    _validate_state_rows(rows, library, protocol, temperature, sweeps)
    tau_key = _tau_key(temperature)
    groups = {
        group: _aggregate_group(
            [row for row in rows if row["state_family"] == group],
            group,
            temperature,
            sweeps,
        )
        for group in protocol["required_state_groups"]
    }
    global_group = _aggregate_group(
        rows, "global", temperature, sweeps
    )
    factor = common._aggregate_factor_diagnostics(rows, [temperature])
    probabilities = common._aggregate_probability_diagnostics(
        rows, temperature
    )
    logit = common._aggregate_logit(rows, temperature)
    logit["all_finite"] = common._all_numeric_finite(logit)
    production = common._aggregate_production_sampler(rows, temperature)
    shared_condition_sha256 = scientific_sha256(
        _shared_condition_payload(rows, temperature)
    )
    correctness_gates = {
        "state_count_and_order_exact": (
            len(rows) == protocol["expected_state_count"]
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
            expected_shared_condition_sha256 is None
            or shared_condition_sha256
            == expected_shared_condition_sha256
        ),
        "all_numeric_values_finite": common._all_numeric_finite(rows),
    }
    correctness_passed = all(correctness_gates.values())
    candidate = probe._gibbs_name(temperature, sweeps)
    group_checks = {}
    for group, aggregate in groups.items():
        kernel = aggregate["kernel_summary"][candidate]
        recovery = aggregate["expected_direction_gap_recovery"]
        passed = bool(
            kernel["participating_active_rows"] > 0
            and kernel["tvd_to_joint"] <= protocol["tvd_threshold"]
            and recovery is not None
            and recovery >= protocol["recovery_threshold"]
        )
        group_checks[group] = {
            "participating_active_rows": kernel[
                "participating_active_rows"
            ],
            "tvd_to_joint": kernel["tvd_to_joint"],
            "expected_direction_gap_recovery": recovery,
            "passed": passed,
        }
    qualification_gates = {
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
    return {
        "temperature": float(temperature),
        "sweeps": int(sweeps),
        "state_results": rows,
        "state_results_scientific_sha256": scientific_sha256(rows),
        "shared_condition_scientific_sha256": shared_condition_sha256,
        "factor_diagnostics": factor,
        "probability_diagnostics": probabilities,
        "conditional_logit_diagnostics": logit,
        "production_sampler_diagnostics": production,
        "global_diagnostic": global_group,
        "required_group_checks": group_checks,
        "correctness_gates": correctness_gates,
        "all_correctness_gates_passed": correctness_passed,
        "qualification_gates": qualification_gates,
        "passed": all(qualification_gates.values()),
        "elapsed_sec": float(elapsed_sec),
    }


def _attempt_sequence_valid(result, protocol):
    attempts = result["attempts"]
    attempted = [attempt["sweeps"] for attempt in attempts]
    expected_prefix = protocol["candidate_sweeps"][:len(attempted)]
    if attempted != expected_prefix or not attempts:
        return False
    passing = [index for index, attempt in enumerate(attempts) if attempt["passed"]]
    if passing:
        first = passing[0]
        return (
            first == len(attempts) - 1
            and result["status"] == "qualified"
            and result["minimal_sufficient_sweeps"]
            == attempts[first]["sweeps"]
        )
    return (
        attempted == protocol["candidate_sweeps"]
        and result["status"] == "unqualified_at_sweeps_cap"
        and result["minimal_sufficient_sweeps"] is None
    )


def run_stage_a_mixing(
    mode,
    state_library_path,
    state_library_audit_path,
    output_dir,
    *,
    max_workers=8,
):
    protocol = frozen_protocol.stage_a_mixing_protocol(mode)
    workers = validate_max_workers(max_workers)
    output_directory = Path(output_dir)
    report_path = output_directory / "stage_a_mixing_report.json"
    if report_path.exists():
        raise FileExistsError(
            f"输出已存在，尚未启动任何混合任务：{report_path}"
        )
    implementation_gates = _validate_implementation_constants(protocol)
    library, _, library_identity = _validate_state_library(
        state_library_path, state_library_audit_path, protocol
    )
    target, queries, schema, _, input_hashes = common._load_inputs()
    if input_hashes != protocol["input_sha256"]:
        raise RuntimeError("Stage A 输入哈希与冻结协议不一致")
    git = common._git_identity()
    if mode == "formal" and not git["worktree_clean"]:
        raise RuntimeError("正式 Stage A 要求 tracked 工作树干净")
    protocol_sha256 = common._canonical_sha256({
        "protocol": protocol,
        "input_sha256": input_hashes,
        "git_commit": git["commit"],
        "state_library_scientific_sha256": library_identity[
            "state_library_scientific_sha256"
        ],
    })

    started = time.perf_counter()
    temperature_results = {}
    for temperature in protocol["evaluation_temperatures"]:
        tau_key = _tau_key(temperature)
        attempts = []
        expected_shared_sha = None
        minimal = None
        for sweeps in protocol["candidate_sweeps"]:
            attempt_started = time.perf_counter()
            rows = _run_raw_attempt(
                library,
                target,
                queries,
                schema,
                protocol,
                temperature=temperature,
                sweeps=sweeps,
                max_workers=workers,
            )
            attempt = _summarize_attempt(
                rows,
                library,
                protocol,
                temperature,
                sweeps,
                expected_shared_condition_sha256=expected_shared_sha,
                elapsed_sec=time.perf_counter() - attempt_started,
            )
            if expected_shared_sha is None:
                expected_shared_sha = attempt[
                    "shared_condition_scientific_sha256"
                ]
            attempts.append(attempt)
            if attempt["passed"]:
                minimal = sweeps
                break
        temperature_results[tau_key] = {
            "temperature": float(temperature),
            "status": (
                "qualified" if minimal is not None
                else "unqualified_at_sweeps_cap"
            ),
            "minimal_sufficient_sweeps": minimal,
            "attempted_sweeps": [row["sweeps"] for row in attempts],
            "attempts": attempts,
        }

    execution_gates = {
        "temperature_grid_exact": (
            list(temperature_results)
            == [_tau_key(tau) for tau in protocol["evaluation_temperatures"]]
        ),
        "attempt_sequences_exact": all(
            _attempt_sequence_valid(result, protocol)
            for result in temperature_results.values()
        ),
        "no_sweeps_above_hard_cap": all(
            attempt["sweeps"] <= protocol["sweeps_hard_cap"]
            for result in temperature_results.values()
            for attempt in result["attempts"]
        ),
        "all_executed_attempts_correct": all(
            attempt["all_correctness_gates_passed"]
            for result in temperature_results.values()
            for attempt in result["attempts"]
        ),
        "shared_conditions_exact_within_tau": all(
            len({
                attempt["shared_condition_scientific_sha256"]
                for attempt in result["attempts"]
            }) == 1
            for result in temperature_results.values()
        ),
    }
    formal_identity_gates = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": (
            protocol == frozen_protocol.stage_a_mixing_protocol("formal")
        ),
        "worktree_clean": git["worktree_clean"],
        "input_hashes_exact": input_hashes == frozen_protocol.EXPECTED_INPUT_SHA256,
        "frozen_library_artifacts_exact": bool(
            mode == "formal"
            and all(library_identity["binding_gates"].values())
        ),
    }
    formal_result_valid = bool(
        all(formal_identity_gates.values())
        and all(execution_gates.values())
    )
    execution_scientific_sha256 = scientific_sha256({
        "protocol_sha256": protocol_sha256,
        "library": library_identity,
        "temperatures": temperature_results,
        "execution_gates": execution_gates,
    })
    report = {
        "report_format": REPORT_FORMAT,
        "status": "complete",
        "experiment": "issue52_stage_a_factor_mixing_qualification",
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "interpretation": (
            "formal_preregistered_stage_a_mixing"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "input_sha256": input_hashes,
        "git": git,
        "command_argv": list(sys.argv),
        "runtime": {
            "max_workers": workers,
            "worker_count_is_nonscientific": True,
        },
        "implementation_gates": implementation_gates,
        "state_library": library_identity,
        "temperatures": temperature_results,
        "selection": {
            "minimal_sufficient_sweeps": {
                key: result["minimal_sufficient_sweeps"]
                for key, result in temperature_results.items()
            },
            "qualified_temperatures": [
                result["temperature"]
                for result in temperature_results.values()
                if result["status"] == "qualified"
            ],
            "unqualified_temperatures": [
                result["temperature"]
                for result in temperature_results.values()
                if result["status"] != "qualified"
            ],
        },
        "execution_gates": execution_gates,
        "formal_identity_gates": formal_identity_gates,
        "execution_scientific_sha256": execution_scientific_sha256,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    common._write_json_atomic(report_path, report)
    return report_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["smoke", "formal"], default="smoke"
    )
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--state-library-audit", required=True)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/issue52_low_temperature_long_horizon/"
            "stage_a_mixing_smoke"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if (
        args.mode == "formal"
        and args.output_dir.endswith("stage_a_mixing_smoke")
    ):
        parser.error("正式模式必须显式提供非 smoke 输出目录")
    report_path, report = run_stage_a_mixing(
        args.mode,
        args.state_library,
        args.state_library_audit,
        args.output_dir,
        max_workers=args.max_workers,
    )
    print("\n===== Issue #52 Stage A mixing =====")
    print(f"mode={report['mode']}")
    for key, result in report["temperatures"].items():
        print(
            f"{key}: status={result['status']} "
            f"minimal_sweeps={result['minimal_sufficient_sweeps']}"
        )
    print(f"formal_result_valid={report['formal_result_valid']}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
