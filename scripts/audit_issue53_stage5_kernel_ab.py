#!/usr/bin/env python3
"""Independent terminal-table audit for Issue #53 Stage 5."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import evaluate_issue53_fixed_alpha_calibration as fixed_evaluation
    from scripts import issue53_stage5_protocol as protocol
    from scripts import run_issue53_stage5_kernel_ab as collection
else:
    import evaluate_issue53_fixed_alpha_calibration as fixed_evaluation
    import issue53_stage5_protocol as protocol
    import run_issue53_stage5_kernel_ab as collection


AUDIT_VERSION = "issue53-stage5-kernel-ab-independent-audit-v1"
AUDIT_REPORT = protocol.AUDIT_REPORT


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generator_access",
        "collection_protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "collection_report": str(
            protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
        ),
        "evaluation_report": str(
            protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
        ),
        "audit_report": str(protocol.OUTPUT_DIR / AUDIT_REPORT),
        "recompute_from_terminal_csv": True,
        "recompute_all_40_case_metrics": True,
        "recompute_pair_differences_gates_and_classification": True,
        "trust_evaluator_derived_values": False,
        "new_generation_allowed": False,
        "generation_started": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _require_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"必须显式确认完整小写 {name} SHA-256")


def _safe_artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact path 必须是相对字符串")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise RuntimeError(f"artifact path 越界：{relative!r}")
    return candidate


def _audit_result_artifacts_independently(
    shard_root: Path, row: dict[str, Any]
) -> dict[str, Any]:
    table_path = _safe_artifact_path(shard_root, row.get("terminal_table_path"))
    diagnostics_path = _safe_artifact_path(
        shard_root, row.get("diagnostics_path")
    )
    result_path = _safe_artifact_path(shard_root, row.get("result_path"))
    observed = {
        "terminal_table_sha256": protocol.file_sha256(table_path),
        "diagnostics_sha256": protocol.file_sha256(diagnostics_path),
        "result_artifact_sha256": protocol.file_sha256(result_path),
    }
    if any(observed[key] != row.get(key) for key in observed):
        raise RuntimeError(
            f"artifact SHA 漂移：seed={row.get('seed')} "
            f"{row.get('dataset')}/{row.get('arm')}"
        )
    result_payload = _load_json(result_path)
    manifest_payload = {
        key: value for key, value in row.items() if key != "result_artifact_sha256"
    }
    if result_payload != manifest_payload:
        raise RuntimeError("result.json 与 collection manifest 内容不一致")
    diagnostics = _load_json(diagnostics_path)
    if (
        diagnostics.get("output_table_identity") != "terminal_current"
        or diagnostics.get("initial_table_sha256")
        != row.get("initial_table_sha256")
        or diagnostics.get("primary_rng_post_initialization_state_sha256")
        != row.get("primary_rng_post_initialization_state_sha256")
        or diagnostics.get("primary_rng_state_sha256")
        != row.get("primary_rng_state_sha256")
        or [
            clock["primary_rng_state_sha256"]
            for clock in diagnostics["transition_clock_history"]
        ]
        != row.get("primary_rng_state_sha256_by_round")
        or diagnostics.get("factorized_gibbs_initial_rng_state_sha256")
        != row.get("factorized_gibbs_initial_rng_state_sha256")
        or diagnostics.get("factorized_gibbs_rng_state_sha256")
        != row.get("factorized_gibbs_rng_state_sha256")
        or diagnostics.get("params") != row.get("generator_params")
    ):
        raise RuntimeError("diagnostics 与 result identity 漂移")
    rounds = int(diagnostics["rounds_run"])
    clocks = diagnostics["transition_clock_history"]
    terminal_internal_sha = (
        diagnostics["initial_table_sha256"]
        if rounds == 0
        else clocks[-1]["post_current_table_sha256"]
    )
    decision = diagnostics["inner_early_stopping"]["last_decision"]
    if (
        len(clocks) != rounds
        or len(diagnostics["current_state_metrics_history"]) != rounds + 1
        or terminal_internal_sha != row["terminal_table_sha256"]
        or int(decision["state_index"])
        != int(row["terminal_behavior"]["terminal_state_index"])
        or float(decision["normalized_work"])
        != float(row["terminal_behavior"]["normalized_work_at_stop"])
        or diagnostics["termination_reason"] != row["termination_reason"]
    ):
        raise RuntimeError("diagnostics terminal-current identity 漂移")
    cost = row.get("cost")
    if not isinstance(cost, dict) or any(
        int(diagnostics[key]) != int(cost[result_key])
        for key, result_key in (
            ("rounds_run", "raw_rounds"),
            ("candidate_evaluation_count", "candidate_evaluation_count"),
            ("state_evaluation_count", "state_evaluation_count"),
            ("factorized_gibbs_microsteps", "factorized_gibbs_microsteps"),
            (
                "factorized_gibbs_conditional_logit_evaluated_count",
                "factorized_gibbs_conditional_logit_evaluated_count",
            ),
            (
                "factorized_gibbs_conditional_logit_clipped_count",
                "factorized_gibbs_conditional_logit_clipped_count",
            ),
        )
    ):
        raise RuntimeError("diagnostics cost counters 与 result 漂移")
    return {
        "table_path": table_path,
        "diagnostics": diagnostics,
        "artifact_sha256": observed,
    }


def _expected_arm_params(dataset: str, arm: str) -> dict[str, Any]:
    return {
        "n_records": protocol.DATASETS[dataset]["n_records"],
        "n_rounds": protocol.ROUND_CAP,
        "beta": 1.0,
        "h": 0.8,
        "rho": 0.01,
        "eta": 0.5,
        "mu": 0.01,
        "tol": "positive_infinity",
        "device": protocol.DATASETS[dataset]["device"],
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "marginal",
        "distance_mode": "geometric",
        "p": None,
        "lambda": 0.5,
        "alpha_min": None,
        "alpha_max": None,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": protocol.FIXED_ALPHA,
        "delta": 0.05,
        "winsorize_quantiles": [0.01, 0.99],
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "factorized_gibbs_sweeps": (
            0 if arm == protocol.ARM_INDEPENDENT else protocol.FACTOR_SWEEPS
        ),
        "factorized_gibbs_max_order": protocol.DATASETS[dataset][
            "max_factor_order"
        ],
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": arm == protocol.ARM_FACTOR,
        "diffusion_direction_strength": protocol.TAU,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_reference_scale": None,
        "diffusion_direction_logit_clip": 30.0,
        "candidate_budget": protocol.CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "self_cooling_monotone": None,
        "self_cooling_stop_ratio": None,
        "rho_anneal_end": None,
        "rho_anneal_rounds": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "inner_early_stopping_patience_ticks": protocol.PATIENCE_TICKS,
        "record_transition_clocks": True,
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "horizon_invariant": False,
    }


def _audit_case_generator_identity(row: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    params = diagnostics["params"]
    expected = {
        **_expected_arm_params(row["dataset"], row["arm"]),
        "seed": row["seed"],
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise RuntimeError("独立审计发现 kernel/common generator 参数漂移")
    if (
        row["termination_reason"]
        not in {*protocol.NORMAL_REASONS, protocol.RESOURCE_CAP_REASON}
        or not math.isfinite(float(row["direction_reference_scale"]))
        or float(row["direction_reference_scale"]) <= 0.0
        or float(diagnostics["direction_reference_scale"])
        != float(row["direction_reference_scale"])
        or sum(diagnostics["direction_logit_clipped_count_history"])
        != row["direction_logit_clipped_count"]
    ):
        raise RuntimeError("独立审计发现 termination/direction identity 漂移")
    rounds = int(diagnostics["rounds_run"])
    if (
        diagnostics["accept_history"] != [True] * rounds
        or diagnostics["proposal_attempts_history"] != [1] * rounds
        or diagnostics["accepted_attempt_history"] != [1] * rounds
        or diagnostics["candidate_evaluation_count"] != rounds
        or len(diagnostics["transition_clock_history"]) != rounds
        or diagnostics["alpha_history"] != [protocol.FIXED_ALPHA] * rounds
    ):
        raise RuntimeError("独立审计发现 no-gate/one-candidate 身份漂移")
    microsteps = int(diagnostics["factorized_gibbs_microsteps"])
    conditional = int(
        diagnostics["factorized_gibbs_conditional_logit_evaluated_count"]
    )
    active_blocks = int(diagnostics["factorized_gibbs_active_blocks"])
    if row["arm"] == protocol.ARM_INDEPENDENT:
        if microsteps or conditional or active_blocks:
            raise RuntimeError("independent 臂出现 Gibbs 成本")
    elif (
        microsteps != protocol.FACTOR_SWEEPS * active_blocks
        or conditional != microsteps
    ):
        raise RuntimeError("factor microsteps 与 active-block 身份漂移")


def _audit_pair(
    rows: Sequence[dict[str, Any]],
    seed: int,
    dataset: str,
    *,
    mode: str = "formal",
) -> dict[str, Any]:
    selected = [row for row in rows if row["seed"] == seed and row["dataset"] == dataset]
    if len(selected) != 2 or {row["arm"] for row in selected} != set(protocol.ARMS):
        raise RuntimeError(f"{dataset}/seed{seed} pair 不完整")
    initial = {row["initial_table_sha256"] for row in selected}
    primary = {
        row["primary_rng_post_initialization_state_sha256"] for row in selected
    }
    scales = {float(row["direction_reference_scale"]) for row in selected}
    if (
        len(initial) != 1
        or len(primary) != 1
        or len(scales) != 1
        or not all(math.isfinite(value) and value > 0.0 for value in scales)
    ):
        raise RuntimeError(f"{dataset}/seed{seed} S0/RNG/direction scale 配对失败")
    traces = [row["primary_rng_state_sha256_by_round"] for row in selected]
    shared_rounds = min(len(trace) for trace in traces)
    if traces[0][:shared_rounds] != traces[1][:shared_rounds]:
        raise RuntimeError(f"{dataset}/seed{seed} primary RNG prefix 配对失败")
    shared_prefix = traces[0][:shared_rounds]
    order = sorted(selected, key=lambda row: row["execution_order_index"])
    expected_arms = protocol.arm_order_for_seed(seed, mode=mode)
    if tuple(row["arm"] for row in order) != expected_arms:
        raise RuntimeError(f"{dataset}/seed{seed} 反平衡执行顺序漂移")
    if (
        len({row["hostname"] for row in selected}) != 1
        or len({row["cuda_device_name"] for row in selected}) != 1
    ):
        raise RuntimeError(f"{dataset}/seed{seed} host/device 未配对")
    return {
        "initial_table_sha256": next(iter(initial)),
        "primary_rng_post_initialization_state_sha256": next(iter(primary)),
        "direction_reference_scale": next(iter(scales)),
        "primary_rng_common_prefix_round_count": shared_rounds,
        "primary_rng_common_prefix_sha256": _canonical_sha256(shared_prefix),
        "hostname": selected[0]["hostname"],
        "cuda_device_name": selected[0]["cuda_device_name"],
    }


def _audit_shard_manifests(
    root: Path,
    report: dict[str, Any],
    indexed: dict[tuple[int, str, str], dict[str, Any]],
    pairing: dict[str, Any],
) -> dict[str, str]:
    observed_sha = {}
    for shard_index, seed in enumerate(protocol.FORMAL_SEEDS):
        path = (
            root
            / protocol.OUTPUT_DIR
            / f"seed_{seed}"
            / "shard_manifest.json"
        )
        digest = protocol.file_sha256(path)
        if digest != report["shard_manifest_sha256_by_seed"].get(str(seed)):
            raise RuntimeError(f"seed {seed} shard manifest SHA 漂移")
        payload = _load_json(path)
        order = protocol.case_order_for_seed(seed)
        expected_rows = [indexed[(seed, dataset, arm)] for dataset, arm in order]
        if (
            payload.get("contract_version") != protocol.PROTOCOL_VERSION
            or payload.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
            or payload.get("formal_result_valid") is not True
            or payload.get("shard_index") != shard_index
            or payload.get("seed") != seed
            or payload.get("execution_git_commit")
            != report["execution_git_commit"]
            or payload.get("case_count") != 4
            or payload.get("case_order")
            != [
                {"dataset": dataset, "arm": arm}
                for dataset, arm in order
            ]
            or payload.get("results") != expected_rows
            or payload.get("initial_state_pairing") != pairing[str(seed)]
            or payload.get("environment")
            != report["environment_by_seed"].get(str(seed))
            or payload.get("smoke_gate") != report.get("smoke_gate")
            or payload.get("raw_reference_data_accessed") is not False
            or payload.get("partial_matrix_comparison_emitted") is not False
            or payload.get("offline_classification_emitted") is not False
        ):
            raise RuntimeError(f"seed {seed} shard manifest 内容漂移")
        input_audit = payload.get("input_audit")
        if not isinstance(input_audit, dict):
            raise RuntimeError(f"seed {seed} input audit 缺失")
        for dataset in protocol.DATASET_ORDER:
            spec = protocol.DATASETS[dataset]
            cell = input_audit.get(dataset, {})
            if (
                cell.get("sha256") != spec["sha256"]
                or cell.get("query_count") != spec["query_count"]
                or cell.get("query_identity_sha256")
                != spec["query_identity_sha256"]
                or cell.get("target_vector_sha256")
                != spec["target_vector_sha256"]
                or cell.get("order_counts")
                != {str(key): value for key, value in spec["order_counts"].items()}
            ):
                raise RuntimeError(f"seed {seed}/{dataset} input audit 漂移")
        observed_sha[str(seed)] = digest
    return observed_sha


def _smoke_non_timing_identity(
    row: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    keys = (
        "current_state_metrics_history",
        "transition_clock_history",
        "loss_history",
        "accept_history",
        "rho_schedule_history",
        "alpha_history",
        "direction_reference_scale_history",
        "direction_logit_evaluated_count_history",
        "direction_logit_clipped_count_history",
        "factorized_gibbs_active_rows",
        "factorized_gibbs_active_blocks",
        "factorized_gibbs_factor_count",
        "factorized_gibbs_factor_table_entries",
        "factorized_gibbs_microsteps",
        "factorized_gibbs_conditional_logit_evaluated_count",
        "factorized_gibbs_conditional_logit_clipped_count",
    )
    return {
        "terminal_table_sha256": row["terminal_table_sha256"],
        "termination_reason": row["termination_reason"],
        "terminal_behavior": row["terminal_behavior"],
        "measured": row["measured"],
        "initial_table_sha256": row["initial_table_sha256"],
        "primary_rng_post_initialization_state_sha256": row[
            "primary_rng_post_initialization_state_sha256"
        ],
        "primary_rng_state_sha256": row["primary_rng_state_sha256"],
        "factorized_gibbs_initial_rng_state_sha256": row[
            "factorized_gibbs_initial_rng_state_sha256"
        ],
        "factorized_gibbs_rng_state_sha256": row[
            "factorized_gibbs_rng_state_sha256"
        ],
        "direction_reference_scale": row["direction_reference_scale"],
        **{key: diagnostics[key] for key in keys},
    }


def _canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(protocol._strict_json_bytes(value)).hexdigest()


def _audit_smoke_independently(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    gate = report.get("smoke_gate")
    if not isinstance(gate, dict):
        raise RuntimeError("collection smoke gate 缺失")
    smoke_path = _safe_artifact_path(root, gate["smoke_report_path"])
    if protocol.file_sha256(smoke_path) != gate.get("smoke_report_sha256"):
        raise RuntimeError("smoke report SHA 漂移")
    smoke = _load_json(smoke_path)
    if (
        smoke.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or smoke.get("execution_git_commit") != report["execution_git_commit"]
        or smoke.get("formal_result_valid") is not False
        or smoke.get("seed") != protocol.SMOKE_SEED
        or smoke.get("all_compiled_equivalence_passed") is not True
    ):
        raise RuntimeError("smoke gate 身份漂移")
    official = smoke.get("official_results")
    replays = smoke.get("rowwise_equivalence_replays")
    if not isinstance(official, list) or len(official) != 4:
        raise RuntimeError("smoke official cases 不完整")
    if not isinstance(replays, list) or len(replays) != 2:
        raise RuntimeError("smoke rowwise replays 不完整")
    smoke_root = smoke_path.parent
    official_audits = {
        (row["dataset"], row["arm"]): _audit_result_artifacts_independently(
            smoke_root, row
        )
        for row in official
    }
    replay_audits = {
        row["dataset"]: _audit_result_artifacts_independently(smoke_root, row)
        for row in replays
    }
    expected_official_order = protocol.case_order_for_seed(
        protocol.SMOKE_SEED, mode="smoke"
    )
    if (
        tuple((row["dataset"], row["arm"]) for row in official)
        != expected_official_order
        or [row["execution_order_index"] for row in official] != [0, 1, 2, 3]
        or [row["execution_order_index"] for row in replays] != [4, 5]
    ):
        raise RuntimeError("smoke official/replay 执行顺序漂移")
    smoke_pairing = {
        dataset: _audit_pair(
            official,
            protocol.SMOKE_SEED,
            dataset,
            mode="smoke",
        )
        for dataset in protocol.DATASET_ORDER
    }
    if smoke_pairing != smoke.get("initial_state_pairing"):
        raise RuntimeError("smoke pairing manifest 漂移")
    identities = {}
    for dataset in protocol.DATASET_ORDER:
        compiled = next(
            row
            for row in official
            if row["dataset"] == dataset and row["arm"] == protocol.ARM_FACTOR
        )
        rowwise = next(row for row in replays if row["dataset"] == dataset)
        compiled_audit = official_audits[(dataset, protocol.ARM_FACTOR)]
        rowwise_audit = replay_audits[dataset]
        if (
            compiled["generator_params"][
                "factorized_gibbs_use_compiled_workload"
            ]
            is not True
            or rowwise["generator_params"][
                "factorized_gibbs_use_compiled_workload"
            ]
            is not False
        ):
            raise RuntimeError("smoke compiled/rowwise replay 参数身份漂移")
        compiled_sha = _canonical_sha256(
            _smoke_non_timing_identity(
                compiled, compiled_audit["diagnostics"]
            )
        )
        rowwise_sha = _canonical_sha256(
            _smoke_non_timing_identity(rowwise, rowwise_audit["diagnostics"])
        )
        if compiled_sha != rowwise_sha:
            raise RuntimeError("独立审计发现 compiled/rowwise 非计时结果不等价")
        identities[dataset] = compiled_sha
    return {
        "smoke_report_sha256": gate["smoke_report_sha256"],
        "execution_git_commit": report["execution_git_commit"],
        "compiled_equivalence_non_timing_sha256_by_dataset": identities,
        "pass": True,
    }


def _audit_collection_independently(
    root: Path, confirmed_sha: str
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]], dict[str, Any]]:
    path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("collection report SHA 与显式确认值不一致")
    report = _load_json(path)
    if (
        report.get("contract_version") != protocol.PROTOCOL_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("formal_result_valid") is not True
        or report.get("case_count") != 40
        or report.get("paired_dataset_seed_count") != 20
        or report.get("raw_reference_data_accessed") is not False
        or report.get("offline_classification_emitted") is not False
    ):
        raise RuntimeError("collection 顶层身份漂移")
    execution_commit = report.get("execution_git_commit")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if (
        not isinstance(execution_commit, str)
        or collection._git_text(root, "merge-base", execution_commit, current_commit)
        != execution_commit
    ):
        raise RuntimeError("collection execution commit ancestry 漂移")
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 40:
        raise RuntimeError("collection 40 cases 不完整")
    indexed = {}
    artifact_audit = {}
    for row in rows:
        key = (row.get("seed"), row.get("dataset"), row.get("arm"))
        if (
            key in indexed
            or key[0] not in protocol.FORMAL_SEEDS
            or key[1] not in protocol.DATASET_ORDER
            or key[2] not in protocol.ARMS
            or row.get("git_commit") != execution_commit
            or row.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
            or row.get("formal_result_valid") is not True
            or row.get("raw_reference_data_accessed") is not False
        ):
            raise RuntimeError(f"collection case identity 漂移：{key}")
        shard_root = root / protocol.OUTPUT_DIR / f"seed_{key[0]}"
        audited = _audit_result_artifacts_independently(shard_root, row)
        _audit_case_generator_identity(row, audited["diagnostics"])
        indexed[key] = row
        artifact_audit["/".join(map(str, key))] = audited["artifact_sha256"]
    expected = {
        (seed, dataset, arm)
        for seed in protocol.FORMAL_SEEDS
        for dataset in protocol.DATASET_ORDER
        for arm in protocol.ARMS
    }
    if set(indexed) != expected:
        raise RuntimeError("collection identities 不完整")
    pair_audit = {
        str(seed): {
            dataset: _audit_pair(rows, seed, dataset)
            for dataset in protocol.DATASET_ORDER
        }
        for seed in protocol.FORMAL_SEEDS
    }
    if pair_audit != report.get("initial_state_pairing_by_seed"):
        raise RuntimeError("collection 顶层 pairing manifest 漂移")
    shard_sha = _audit_shard_manifests(
        root, report, indexed, pair_audit
    )
    smoke = _audit_smoke_independently(root, report)
    return report, indexed, {
        "artifact_sha256_by_case": artifact_audit,
        "shard_manifest_sha256_by_seed": shard_sha,
        "pairing_by_seed": pair_audit,
        "smoke_gate": smoke,
        "all_40_cases_present": True,
        "all_20_pairs_present": True,
        "pass": True,
    }


def _runtime_inputs(root: Path, runtime: Any) -> dict[str, Any]:
    if protocol.file_sha256(root / protocol.PROTOCOL_DOC) != (
        protocol.PROTOCOL_DOC_SHA256
    ):
        raise RuntimeError("Stage 5 protocol document SHA 漂移")
    result = {}
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
        observed_sha = {
            key: protocol.file_sha256(root / spec[key])
            for key in ("schema", "queries", "marginals")
        }
        if observed_sha != spec["sha256"]:
            raise RuntimeError(f"{dataset} measured input SHA 漂移")
        payload = _load_json(root / spec["queries"])
        result[dataset] = {
            "schema": runtime.load_schema(str(root / spec["schema"])),
            "queries": runtime.load_queries(str(root / spec["queries"])),
            "targets": [query["result"] for query in payload["queries"]],
        }
    return result


def _exact_count_sum(runtime: Any, target: Any, answers: Any) -> int:
    target_array = runtime.np.asarray(target)
    answer_array = runtime.np.asarray(answers)
    if (
        target_array.shape != answer_array.shape
        or not runtime.np.array_equal(target_array, runtime.np.rint(target_array))
        or not runtime.np.array_equal(answer_array, runtime.np.rint(answer_array))
    ):
        raise RuntimeError("独立 measured count identity 失败")
    return int(runtime.np.abs(target_array - answer_array).sum())


def _attach_exact_count_sums(
    runtime: Any,
    metrics: dict[str, Any],
    table: Any,
    queries: Sequence[dict[str, Any]],
    targets: Sequence[int],
) -> int:
    answers = runtime.np.asarray(runtime.evaluate_table(table, list(queries)))
    target = runtime.np.asarray(targets)
    total = _exact_count_sum(runtime, target, answers)
    metrics["measured"]["overall"]["absolute_count_error_sum"] = total
    for order in sorted({len(query["conditions"]) for query in queries}):
        indices = [
            index
            for index, query in enumerate(queries)
            if len(query["conditions"]) == order
        ]
        metrics["measured"]["by_order"][str(order)][
            "absolute_count_error_sum"
        ] = _exact_count_sum(runtime, target[indices], answers[indices])
    return total


def _recompute_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Freeze query identities before opening reference data.
    test_groups, test_identity = fixed_evaluation._freeze_test_groups(root)
    nltcs_groups, nltcs_identity = fixed_evaluation._freeze_nltcs_groups(root)
    runtime = fixed_evaluation._load_runtime()
    inputs = _runtime_inputs(root, runtime)
    references, reference_sha = fixed_evaluation._load_references(root, runtime)
    test_targets = {
        name: runtime.evaluate_table(references["test_300x10"], queries)
        for name, queries in test_groups.items()
    }
    nltcs_marginals = _load_json(root / protocol.DATASETS["nltcs"]["marginals"])
    nltcs_domains = runtime.offline._discretization_domains(nltcs_marginals)
    measured_triples = runtime.offline._measured_cell_keys(
        inputs["nltcs"]["queries"], nltcs_marginals, order=3
    )
    cases = []
    for seed in protocol.FORMAL_SEEDS:
        for dataset in protocol.DATASET_ORDER:
            for arm in protocol.ARMS:
                source = indexed[(seed, dataset, arm)]
                table_path = (
                    root
                    / protocol.OUTPUT_DIR
                    / f"seed_{seed}"
                    / source["terminal_table_path"]
                )
                table = runtime.pd.read_csv(table_path)
                current = inputs[dataset]
                if dataset == "test_300x10":
                    metrics = fixed_evaluation._evaluate_test_case(
                        runtime,
                        table,
                        current["queries"],
                        current["targets"],
                        test_groups,
                        test_targets,
                        current["schema"],
                        references[dataset],
                    )
                else:
                    metrics = fixed_evaluation._evaluate_nltcs_case(
                        runtime,
                        table,
                        current["queries"],
                        current["targets"],
                        nltcs_groups["one_way_safety"],
                        current["schema"],
                        nltcs_marginals,
                        nltcs_domains,
                        measured_triples,
                        references[dataset],
                    )
                count_sum = _attach_exact_count_sums(
                    runtime,
                    metrics,
                    table,
                    current["queries"],
                    current["targets"],
                )
                if (
                    count_sum != source["measured"]["absolute_count_error_sum"]
                    or metrics["measured"]["overall"][
                        "squared_loss_diagnostic_only"
                    ]
                    != source["measured"]["squared_loss"]
                    or not fixed_evaluation._measured_l1_matches_collection(
                        metrics["measured"]["overall"]["normalized_l1_mean"],
                        source["measured"]["normalized_l1_mean"],
                    )
                ):
                    raise RuntimeError("collection terminal measured 主值复算失败")
                cases.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "pair_execution_position": (
                            int(source["execution_order_index"]) % 2 + 1
                        ),
                        "termination_reason": source["termination_reason"],
                        "terminal_behavior": source["terminal_behavior"],
                        "cost": source["cost"],
                        "direction_logit_clipped_count": source[
                            "direction_logit_clipped_count"
                        ],
                        "terminal_table_sha256": source[
                            "terminal_table_sha256"
                        ],
                        "metrics": metrics,
                    }
                )
                print(
                    f"[independent audit {dataset}/{arm}/seed={seed}] complete",
                    flush=True,
                )
    return cases, {
        "query_identity_frozen_before_reference_load": True,
        "query_identity_audit": {
            "test_300x10": test_identity,
            "nltcs": nltcs_identity,
        },
        "reference_sha256": reference_sha,
    }


def _value(case: dict[str, Any], path: str) -> float:
    value: Any = case
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"audit metric {path} 不是数值")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"audit metric {path} 不是有限数")
    return normalized


def _arms(cases: Sequence[dict[str, Any]], dataset: str, path: str) -> dict[str, Any]:
    indexed = {
        (case["seed"], case["arm"]): _value(case, path)
        for case in cases
        if case["dataset"] == dataset
    }
    if len(indexed) != 20:
        raise RuntimeError(f"{dataset}/{path} pair values 不完整")
    factor = {
        str(seed): indexed[(seed, protocol.ARM_FACTOR)]
        for seed in protocol.FORMAL_SEEDS
    }
    independent = {
        str(seed): indexed[(seed, protocol.ARM_INDEPENDENT)]
        for seed in protocol.FORMAL_SEEDS
    }
    differences = {
        str(seed): factor[str(seed)] - independent[str(seed)]
        for seed in protocol.FORMAL_SEEDS
    }
    values = list(differences.values())
    return {
        "path": path,
        "factor_values_by_seed": factor,
        "independent_values_by_seed": independent,
        "factor_minus_independent_by_seed": differences,
        "factor_mean": float(statistics.fmean(factor.values())),
        "independent_mean": float(statistics.fmean(independent.values())),
        "factor_strictly_lower_count": sum(value < 0 for value in values),
        "tie_count": sum(value == 0 for value in values),
        "factor_strictly_higher_count": sum(value > 0 for value in values),
    }


def _lower_audit(cases: Sequence[dict[str, Any]], dataset: str, path: str) -> dict[str, Any]:
    result = _arms(cases, dataset, path)
    factor = result["factor_mean"]
    independent = result["independent_mean"]
    result["ratio"] = factor / independent if independent != 0.0 else None
    result["pass"] = bool(
        factor == 0.0
        if independent == 0.0
        else factor / independent <= protocol.LOWER_RISK_RATIO_MAX
    )
    return result


def _higher_audit(cases: Sequence[dict[str, Any]], dataset: str, path: str) -> dict[str, Any]:
    result = _arms(cases, dataset, path)
    factor = result["factor_mean"]
    independent = result["independent_mean"]
    result["ratio"] = factor / independent if independent != 0.0 else None
    result["ratio_evidence_available"] = independent != 0.0
    result["pass"] = bool(
        factor >= protocol.HIGHER_QUALITY_RATIO_MIN * independent
    )
    return result


def _execution(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        (seed, dataset, arm)
        for seed in protocol.FORMAL_SEEDS
        for dataset in protocol.DATASET_ORDER
        for arm in protocol.ARMS
    }
    identities = {
        (case["seed"], case["dataset"], case["arm"]) for case in cases
    }
    structural = len(cases) == 40 and identities == expected
    resource = sum(
        case["termination_reason"] == protocol.RESOURCE_CAP_REASON
        for case in cases
    )
    direction_clip = sum(
        int(case["direction_logit_clipped_count"]) for case in cases
    )
    factor_clip = sum(
        int(case["cost"]["factorized_gibbs_conditional_logit_clipped_count"])
        for case in cases
        if case["arm"] == protocol.ARM_FACTOR
    )
    if not structural:
        status = "execution_invalid"
    elif resource:
        status = "inconclusive_resource_cap"
    elif direction_clip or factor_clip:
        status = "outside_stage4_qualified_unclipped_regime"
    else:
        status = "qualified"
    return {
        "status": status,
        "case_count": len(cases),
        "normal_case_count": sum(
            case["termination_reason"] in protocol.NORMAL_REASONS for case in cases
        ),
        "resource_cap_case_count": resource,
        "direction_logit_clipped_count": direction_clip,
        "factor_conditional_logit_clipped_count": factor_clip,
    }


def _dataset_decision(
    cases: Sequence[dict[str, Any]], dataset: str, execution_status: str
) -> dict[str, Any]:
    measured = _arms(
        cases, dataset, "metrics.measured.overall.absolute_count_error_sum"
    )
    factor_sum = int(sum(measured["factor_values_by_seed"].values()))
    independent_sum = int(
        sum(measured["independent_values_by_seed"].values())
    )
    stable_factor = (
        factor_sum < independent_sum
        and measured["factor_strictly_lower_count"] >= protocol.STABLE_WIN_MINIMUM
    )
    stable_independent = (
        independent_sum < factor_sum
        and measured["factor_strictly_higher_count"]
        >= protocol.STABLE_WIN_MINIMUM
    )
    offline_paths = (
        {
            name: f"metrics.offline_query_groups.{name}.normalized_l1_mean"
            for name in protocol.TEST_GROUP_ORDER
        }
        if dataset == "test_300x10"
        else {
            "one_way_safety": (
                "metrics.offline_query_groups.one_way_safety.normalized_l1_mean"
            ),
            "unmeasured_3way": (
                "metrics.offline_query_groups.unmeasured_3way.normalized_l1_mean"
            ),
            "all_4way": (
                "metrics.offline_query_groups.all_4way.normalized_l1_mean"
            ),
            "binned_joint_tvd": "metrics.binned_joint.tvd",
        }
    )
    offline = {
        name: _lower_audit(cases, dataset, path)
        for name, path in offline_paths.items()
    }
    higher_paths = {
        "synthetic_mass_in_reference_support": (
            "metrics.reference_support.synthetic_mass_in_reference_support"
        ),
        "reference_mass_covered": (
            "metrics.reference_support.reference_mass_covered"
        ),
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
        "attribute_effective_support_ratio_mean": (
            "metrics.diversity.attribute_effective_support_ratio_mean"
        ),
        "attribute_effective_support_ratio_min": (
            "metrics.diversity.attribute_effective_support_ratio_min"
        ),
    }
    higher = {
        name: _higher_audit(cases, dataset, path)
        for name, path in higher_paths.items()
    }
    validity = {
        str(case["seed"]): _value(case, "metrics.validity.valid_row_rate")
        for case in cases
        if case["dataset"] == dataset and case["arm"] == protocol.ARM_FACTOR
    }
    validity_pass = len(validity) == 10 and all(value == 1.0 for value in validity.values())
    quality_pass = (
        all(item["pass"] for item in offline.values())
        and all(item["pass"] for item in higher.values())
        and validity_pass
    )
    compute = {
        "normalized_work": _lower_audit(
            cases, dataset, "terminal_behavior.normalized_work_at_stop"
        ),
        "candidate_evaluation_count": _lower_audit(
            cases, dataset, "cost.candidate_evaluation_count"
        ),
    }
    compute_pass = all(item["pass"] for item in compute.values())
    if execution_status != "qualified":
        classification = execution_status
    elif not stable_factor:
        classification = "no_stable_factor_gain"
    elif not quality_pass:
        classification = "factor_measured_gain_with_quality_or_diversity_risk"
    elif not compute_pass:
        classification = "factor_quality_supported_with_outer_compute_tradeoff"
    else:
        classification = "factor_quality_supported_outer_efficient"
    return {
        "classification": classification,
        "measured": {
            **measured,
            "factor_aggregate_count_error_sum": factor_sum,
            "independent_aggregate_count_error_sum": independent_sum,
            "stable_factor_measured_gain": stable_factor,
            "stable_independent_measured_advantage": stable_independent,
            "mixed_no_stable_kernel_winner": (
                not stable_factor and not stable_independent
            ),
        },
        "offline_lower_is_better": offline,
        "support_and_diversity_higher_is_better": higher,
        "factor_valid_row_rate_by_seed": validity,
        "validity_pass": validity_pass,
        "quality_pass": quality_pass,
        "outer_compute": compute,
        "outer_compute_pass": compute_pass,
    }


def _cross(decisions: dict[str, dict[str, Any]], execution_status: str) -> str:
    if execution_status != "qualified":
        return "inconclusive_or_invalid_stage5"
    values = [item["classification"] for item in decisions.values()]
    efficient = "factor_quality_supported_outer_efficient"
    tradeoff = "factor_quality_supported_with_outer_compute_tradeoff"
    supported = {efficient, tradeoff}
    count = sum(value in supported for value in values)
    if all(value == efficient for value in values):
        return "shared_factor_support_at_tau2"
    if count == 2:
        return "shared_factor_quality_support_with_compute_tradeoff"
    if count == 1:
        return "dataset_dependent_kernel_response"
    return "no_shared_factor_support"


def _independent_classification(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    execution = _execution(cases)
    datasets = {
        dataset: _dataset_decision(cases, dataset, execution["status"])
        for dataset in protocol.DATASET_ORDER
    }
    return {
        "execution": execution,
        "datasets": datasets,
        "cross_dataset_response": _cross(datasets, execution["status"]),
    }


def _assert_comparison_matches(evaluated: dict[str, Any], audited: dict[str, Any]) -> None:
    expected = {
        "candidate_mean": audited["factor_mean"],
        "baseline_mean": audited["independent_mean"],
        "candidate_over_baseline": audited["ratio"]
        if "ratio" in audited
        else (
            audited["factor_mean"] / audited["independent_mean"]
            if audited["independent_mean"] != 0.0
            else None
        ),
        "paired_wins": audited["factor_strictly_lower_count"]
        if evaluated.get("lower_is_better") is True
        else audited["factor_strictly_higher_count"],
        "paired_ties": audited["tie_count"],
        "paired_losses": audited["factor_strictly_higher_count"]
        if evaluated.get("lower_is_better") is True
        else audited["factor_strictly_lower_count"],
        "factor_values_by_seed": audited["factor_values_by_seed"],
        "independent_values_by_seed": audited["independent_values_by_seed"],
        "factor_minus_independent_by_seed": audited[
            "factor_minus_independent_by_seed"
        ],
    }
    if any(evaluated.get(key) != value for key, value in expected.items()):
        raise RuntimeError("evaluator paired comparison 与独立复算不一致")


def _audit_evaluator_classification(
    evaluation: dict[str, Any], audit: dict[str, Any]
) -> None:
    frozen = evaluation.get("frozen_classification")
    if not isinstance(frozen, dict):
        raise RuntimeError("evaluation frozen classification 缺失")
    observed_execution = frozen.get("execution_qualification", {})
    for key in (
        "status",
        "case_count",
        "normal_case_count",
        "resource_cap_case_count",
        "direction_logit_clipped_count",
        "factor_conditional_logit_clipped_count",
    ):
        if observed_execution.get(key) != audit["execution"].get(key):
            raise RuntimeError(f"evaluator execution gate 漂移：{key}")
    by_dataset = frozen.get("by_dataset")
    if not isinstance(by_dataset, dict):
        raise RuntimeError("evaluation dataset classification 缺失")
    for dataset in protocol.DATASET_ORDER:
        evaluated = by_dataset[dataset]
        audited = audit["datasets"][dataset]
        if evaluated.get("classification") != audited["classification"]:
            raise RuntimeError(f"{dataset} evaluator classification 漂移")
        measured = evaluated["measured_stability"]
        _assert_comparison_matches(measured, audited["measured"])
        for key, audit_key in (
            (
                "factor_10_seed_aggregate_count_error_sum",
                "factor_aggregate_count_error_sum",
            ),
            (
                "independent_10_seed_aggregate_count_error_sum",
                "independent_aggregate_count_error_sum",
            ),
            ("stable_factor_measured_gain", "stable_factor_measured_gain"),
            (
                "stable_independent_measured_advantage",
                "stable_independent_measured_advantage",
            ),
            (
                "mixed_no_stable_kernel_winner",
                "mixed_no_stable_kernel_winner",
            ),
        ):
            if measured.get(key) != audited["measured"].get(audit_key):
                raise RuntimeError(f"{dataset} measured stability gate 漂移")
        quality = evaluated["quality_gates"]
        for name, comparison in quality["offline_safety"].items():
            independent = audited["offline_lower_is_better"][name]
            _assert_comparison_matches(comparison, independent)
            if comparison.get("pass") != independent["pass"]:
                raise RuntimeError(f"{dataset}/{name} offline gate 漂移")
        merged_higher = {
            **quality["reference_support"],
            **quality["diversity"],
        }
        for name, comparison in merged_higher.items():
            independent = audited[
                "support_and_diversity_higher_is_better"
            ][name]
            _assert_comparison_matches(comparison, independent)
            if comparison.get("pass") != independent["pass"]:
                raise RuntimeError(f"{dataset}/{name} higher gate 漂移")
        if (
            quality.get("factor_valid_row_rate_by_seed")
            != audited["factor_valid_row_rate_by_seed"]
            or quality.get("validity_pass") != audited["validity_pass"]
            or quality.get("all_quality_gates_pass") != audited["quality_pass"]
        ):
            raise RuntimeError(f"{dataset} validity/quality aggregate 漂移")
        compute = evaluated["outer_compute_gate"]
        for name, comparison in compute["metrics"].items():
            independent = audited["outer_compute"][name]
            _assert_comparison_matches(comparison, independent)
            if comparison.get("pass") != independent["pass"]:
                raise RuntimeError(f"{dataset}/{name} compute gate 漂移")
        if compute.get("pass") != audited["outer_compute_pass"]:
            raise RuntimeError(f"{dataset} compute aggregate 漂移")
    if frozen.get("cross_dataset_response") != audit["cross_dataset_response"]:
        raise RuntimeError("evaluator cross-dataset classification 漂移")


def _arm_summary_independent(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str, path: str
) -> dict[str, Any]:
    selected = sorted(
        [
            case
            for case in cases
            if case["dataset"] == dataset and case["arm"] == arm
        ],
        key=lambda case: case["seed"],
    )
    if len(selected) != 10:
        raise RuntimeError("summary arm cell 缺少十种子")
    values = [_value(case, path) for case in selected]
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": min(values),
        "maximum": max(values),
        "values_by_seed": {
            str(case["seed"]): value for case, value in zip(selected, values)
        },
    }


def _evaluation_summary_independent(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    paths = {
        "measured_count_error_sum": (
            "metrics.measured.overall.absolute_count_error_sum"
        ),
        "measured_normalized_l1": (
            "metrics.measured.overall.normalized_l1_mean"
        ),
        "measured_squared_loss": (
            "metrics.measured.overall.squared_loss_diagnostic_only"
        ),
        "normalized_work": "terminal_behavior.normalized_work_at_stop",
        "candidate_evaluations": "cost.candidate_evaluation_count",
        "rounds": "cost.raw_rounds",
        "case_wall_clock_sec": "cost.case_wall_clock_elapsed_sec",
        "generator_elapsed_sec": "cost.generator_elapsed_sec",
        "gibbs_microsteps": "cost.factorized_gibbs_microsteps",
        "conditional_logit_evaluations": (
            "cost.factorized_gibbs_conditional_logit_evaluated_count"
        ),
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
        "synthetic_mass_in_reference_support": (
            "metrics.reference_support.synthetic_mass_in_reference_support"
        ),
        "reference_mass_covered": (
            "metrics.reference_support.reference_mass_covered"
        ),
    }
    return {
        dataset: {
            arm: {
                name: _arm_summary_independent(
                    cases, dataset, arm, path
                )
                for name, path in paths.items()
            }
            for arm in protocol.ARMS
        }
        for dataset in protocol.DATASET_ORDER
    }


def _wall_clock_pair_independent(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    path = "cost.case_wall_clock_elapsed_sec"
    values = _arms(cases, dataset, path)
    differences = [
        values["factor_minus_independent_by_seed"][str(seed)]
        for seed in protocol.FORMAL_SEEDS
    ]
    mean_difference = float(statistics.fmean(differences))
    half_width = (
        2.2621571628540993
        * statistics.stdev(differences)
        / math.sqrt(len(differences))
    )
    factor_mean = values["factor_mean"]
    independent_mean = values["independent_mean"]
    return {
        "metric": path,
        "candidate_arm": protocol.ARM_FACTOR,
        "baseline_arm": protocol.ARM_INDEPENDENT,
        "lower_is_better": True,
        "candidate_mean": factor_mean,
        "baseline_mean": independent_mean,
        "candidate_over_baseline": (
            factor_mean / independent_mean if independent_mean != 0.0 else None
        ),
        "mean_paired_difference": mean_difference,
        "paired_wins": values["factor_strictly_lower_count"],
        "paired_ties": values["tie_count"],
        "paired_losses": values["factor_strictly_higher_count"],
        "factor_values_by_seed": values["factor_values_by_seed"],
        "independent_values_by_seed": values["independent_values_by_seed"],
        "factor_minus_independent_by_seed": values[
            "factor_minus_independent_by_seed"
        ],
        "paired_difference_95pct_t_interval_diagnostic_only": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
    }


def _wall_clock_independent(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    path = "cost.case_wall_clock_elapsed_sec"
    result = {}
    for dataset in protocol.DATASET_ORDER:
        strata = {}
        for arm in protocol.ARMS:
            selected = [
                case
                for case in cases
                if case["dataset"] == dataset and case["arm"] == arm
            ]
            first = [
                _value(case, path)
                for case in selected
                if case["pair_execution_position"] == 1
            ]
            second = [
                _value(case, path)
                for case in selected
                if case["pair_execution_position"] == 2
            ]
            strata[arm] = {
                "executed_first_count": len(first),
                "executed_first_mean_sec": float(statistics.fmean(first)),
                "executed_second_count": len(second),
                "executed_second_mean_sec": float(statistics.fmean(second)),
            }
        result[dataset] = {
            "paired": _wall_clock_pair_independent(cases, dataset),
            "execution_order_strata": strata,
            "hard_gate_applied": False,
        }
    return result


def _audit_evaluation_report(
    root: Path,
    confirmed_sha: str,
    collection_sha: str,
    collection_execution_commit: str,
    recomputed_cases: list[dict[str, Any]],
    identities: dict[str, Any],
    independent_classification: dict[str, Any],
) -> dict[str, Any]:
    path = root / protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("evaluation report SHA 与显式确认值不一致")
    evaluation = _load_json(path)
    evaluation_commit = evaluation.get("evaluation_git_commit")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if (
        evaluation.get("contract_version")
        != "issue53-stage5-kernel-ab-evaluation-v1"
        or evaluation.get("collection_protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or evaluation.get("collection_report_sha256") != collection_sha
        or evaluation.get("collection_execution_git_commit")
        != collection_execution_commit
        or evaluation.get("case_count") != 40
        or evaluation.get("new_generation_performed_by_evaluator") is not False
        or evaluation.get("raw_reference_data_accessed") is not True
        or evaluation.get("parameter_retuning_performed") is not False
        or evaluation.get("cross_dataset_or_cross_group_weighted_score_present")
        is not False
        or not isinstance(evaluation_commit, str)
        or collection._git_text(
            root, "merge-base", evaluation_commit, current_commit
        )
        != evaluation_commit
    ):
        raise RuntimeError("evaluation report 顶层身份漂移")
    if (
        evaluation.get("query_identity_audit")
        != identities["query_identity_audit"]
        or evaluation.get("reference_sha256") != identities["reference_sha256"]
        or evaluation.get("query_identity_frozen_before_reference_load") is not True
    ):
        raise RuntimeError("evaluation query/reference identity 漂移")
    if evaluation.get("cases") != recomputed_cases:
        raise RuntimeError("evaluation 40-case metrics 与终表独立复算不一致")
    if evaluation.get("summary") != _evaluation_summary_independent(
        recomputed_cases
    ):
        raise RuntimeError("evaluation summary 与独立复算不一致")
    if evaluation.get("wall_clock_diagnostics") != _wall_clock_independent(
        recomputed_cases
    ):
        raise RuntimeError("evaluation wall-clock diagnostics 与独立复算不一致")
    _audit_evaluator_classification(evaluation, independent_classification)
    return {
        "evaluation_report_sha256": confirmed_sha,
        "all_40_case_metrics_exactly_recomputed": True,
        "all_paired_differences_recomputed": True,
        "all_frozen_gates_recomputed": True,
        "classification_exactly_reproduced": True,
        "pass": True,
    }


def audit(
    confirmed_collection_report_sha256: str,
    confirmed_evaluation_report_sha256: str,
) -> Path:
    _require_sha(confirmed_collection_report_sha256, name="collection report")
    _require_sha(confirmed_evaluation_report_sha256, name="evaluation report")
    root = collection._repo_root()
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("独立 audit 要求包含 untracked 在内的干净工作树")
    output = root / protocol.OUTPUT_DIR / AUDIT_REPORT
    if output.exists():
        raise FileExistsError(f"independent audit 已存在，不覆盖：{output}")
    collection_report, indexed, collection_audit = (
        _audit_collection_independently(
            root, confirmed_collection_report_sha256
        )
    )
    recomputed_cases, identities = _recompute_cases(root, indexed)
    independent_classification = _independent_classification(recomputed_cases)
    evaluation_audit = _audit_evaluation_report(
        root,
        confirmed_evaluation_report_sha256,
        confirmed_collection_report_sha256,
        collection_report["execution_git_commit"],
        recomputed_cases,
        identities,
        independent_classification,
    )
    report = {
        **build_plan(),
        "mode": "independent_recompute_from_terminal_tables",
        "audit_git_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_execution_git_commit": collection_report[
            "execution_git_commit"
        ],
        "collection_report_sha256": confirmed_collection_report_sha256,
        "evaluation_report_sha256": confirmed_evaluation_report_sha256,
        "collection_audit": collection_audit,
        "offline_identity_audit": identities,
        "recomputed_case_count": len(recomputed_cases),
        "recomputed_cases": recomputed_cases,
        "independent_classification": independent_classification,
        "evaluation_audit": evaluation_audit,
        "overall_pass": True,
        "new_generation_performed_by_auditor": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "parameter_retuning_performed": False,
    }
    collection._atomic_write_json(output, report)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--confirm-collection-sha", required=True)
    audit_parser.add_argument("--confirm-evaluation-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = audit(args.confirm_collection_sha, args.confirm_evaluation_sha)
    print(f"Stage 5 independent audit -> {path}")
    print(f"audit SHA-256 -> {protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
