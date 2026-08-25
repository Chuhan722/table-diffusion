#!/usr/bin/env python3
"""第 6B-1 阶段结果盲结构、身份和随机重放审计。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from table_diffevo.factorized_diffusion import (
    compile_mask_workload,
    evolve_step_factorized_gibbs,
)
from table_diffevo.gap_l1_diffusion import (
    compile_gap_l1_workload,
    evolve_step_gap_l1_global,
)
from table_diffevo.queries import evaluate_table

if __package__:
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import calibrate_issue53_stage6b1_gap_l1 as calibrator
    from scripts import collect_issue53_stage6b1_screen as collector
    from scripts import issue53_stage6b1_common as common
    from scripts import issue53_stage6b1_protocol as protocol
else:
    import build_issue53_stage6a_state_library as state_builder
    import calibrate_issue53_stage6b1_gap_l1 as calibrator
    import collect_issue53_stage6b1_screen as collector
    import issue53_stage6b1_common as common
    import issue53_stage6b1_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FORMAT = "issue53_stage6b1_structural_replay_audit_v1"


def _strip_diagnostic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_diagnostic(item)
            for key, item in value.items()
            if not key.endswith("_diagnostic_only")
            and not key.endswith("_elapsed_sec")
            and key != "environment"
        }
    if isinstance(value, list):
        return [_strip_diagnostic(item) for item in value]
    return value


def scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return _strip_diagnostic({
        key: item
        for key, item in value.items()
        if key not in {
            "structural_audit_scientific_sha256",
            "artifact_paths_diagnostic_only",
            "environment",
            "elapsed_sec_diagnostic_only",
        }
    })


def _validate_execution(
    mode: str,
    confirmed_execution_commit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    git = state_builder._git_identity(REPOSITORY_ROOT)
    if not git["worktree_clean_including_untracked"]:
        raise RuntimeError("结构审计要求包含未跟踪文件在内的干净工作树")
    if confirmed_execution_commit != git["commit"]:
        raise PermissionError("必须精确确认当前干净实现提交")
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=mode == "formal",
    )
    return git, environment


def _mutation_specs(events: Any) -> list[tuple[Any, ...]]:
    return [
        (
            int(event["row_index"]),
            int(event["attribute_index"]),
            str(event["attribute"]),
            event["sampled_value"],
        )
        for event in events
    ]


def _assert_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError("结构审计表重建不一致") from error


def _reconstruct_arm(
    context: common.StateContext,
    replay: common.PairReplay,
    record: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    copy_table, full_table = common.stage6a_collector.reconstruct_pair_tables(
        context.current,
        record["copy_edits"],
        record["mutation_events"],
    )
    if (
        common.frame_sha256(copy_table)
        != record["table_sha256"]["copy_only"]
        or common.frame_sha256(full_table)
        != record["table_sha256"]["full"]
    ):
        raise RuntimeError("稀疏编辑重建表哈希失败")
    for edit in record["copy_edits"]:
        row = int(edit["row_index"])
        if int(edit["donor_index"]) != int(replay.donor_indices[row]):
            raise RuntimeError("稀疏复制日志供体编号不一致")
        for cell in edit["cells"]:
            if cell["after"] != replay.donors.at[row, cell["attribute"]]:
                raise RuntimeError("稀疏复制日志不是供体值")
    copy_counts = evaluate_table(
        copy_table, context.runtime.queries
    ).astype(np.int64)
    full_counts = evaluate_table(
        full_table, context.runtime.queries
    ).astype(np.int64)
    if (
        not np.array_equal(
            copy_counts,
            common.reconstruct_query_counts(
                context.query_counts, record["query_delta"]["copy_only"]
            ),
        )
        or not np.array_equal(
            full_counts,
            common.reconstruct_query_counts(
                context.query_counts, record["query_delta"]["full"]
            ),
        )
    ):
        raise RuntimeError("稀疏查询改变量与完整复算不一致")
    mask = common.final_mask_from_copy_table(
        context.current,
        replay.donors,
        copy_table,
        context.runtime.schema.attribute_names(),
    )
    return copy_table, full_table, mask


def _audit_generated_pair(
    context: common.StateContext,
    pair: Mapping[str, Any],
    reference_scale: float,
    *,
    mode: str,
    factor_compiled: Any,
    gap_compiled: Any,
) -> dict[str, int]:
    proposal_index = int(pair["proposal_index"])
    replay = common.replay_pair(context, proposal_index, mode=mode)
    shared = pair["shared_replay"]
    if (
        shared["current_table_sha256"] != common.frame_sha256(context.current)
        or shared["donor_address_uint64"]
        != replay.source_pair["rng"]["donor_address_uint64"]
        or shared["update_address_uint64"]
        != replay.source_pair["rng"]["update_address_uint64"]
        or shared["donor_indices_sha256"]
        != common.array_sha256(replay.donor_indices)
        or shared["direction_scores_sha256"]
        != common.array_sha256(replay.direction_scores)
        or shared["participate_sha256"]
        != common.array_sha256(replay.common_update["participate"])
        or shared["initial_copy_mask_sha256"]
        != common.array_sha256(replay.common_update["copy_masks"])
        or float(shared["gap_l1_reference_scale"]) != reference_scale
    ):
        raise RuntimeError("共同来源、参与或初始开关身份失败")
    arms = pair["arms"]
    reconstructed = {
        arm: _reconstruct_arm(context, replay, arms[arm])
        for arm in protocol.ARMS
    }
    specs = [
        _mutation_specs(arms[arm]["mutation_events"])
        for arm in protocol.ARMS
    ]
    if not specs[0] == specs[1] == specs[2]:
        raise RuntimeError("三组没有应用同一突变设值随机量")
    if shared["mutation_specs_sha256"] != protocol.canonical_sha256([
        {
            "row_index": row,
            "attribute_index": attribute_index,
            "attribute": attribute,
            "sampled_value": sampled,
        }
        for row, attribute_index, attribute, sampled in specs[0]
    ]):
        raise RuntimeError("公共突变设值哈希失败")

    independent_copy, independent_full, independent_mask = reconstructed[
        protocol.ARM_INDEPENDENT
    ]
    _assert_frame_equal(
        independent_copy, replay.common_update["copy_table"]
    )
    _assert_frame_equal(
        independent_full, replay.common_update["full_table"]
    )
    if not np.array_equal(
        independent_mask, replay.common_update["copy_masks"]
    ):
        raise RuntimeError("独立组初始开关重建失败")

    update_seed = int(replay.source_pair["rng"]["update_address_uint64"])
    factor_seed = protocol.gibbs_address_seed(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        protocol.ARM_FACTOR,
        mode=mode,
    )
    factor_rng = np.random.default_rng(factor_seed)
    factor_main_rng = np.random.default_rng(update_seed)
    factor_copy, factor_diagnostics = evolve_step_factorized_gibbs(
        context.current,
        replay.donors,
        context.runtime.schema,
        context.runtime.queries,
        context.residual_signal,
        rho=protocol.RHO,
        eta=protocol.ETA,
        mu=0.0,
        copy_direction_scores=replay.direction_scores,
        copy_direction_strength=(
            protocol.B_STRENGTH
            / float(context.trajectory["direction_reference_scale"])
        ),
        n_sweeps=protocol.GIBBS_SWEEPS,
        rng=factor_main_rng,
        gibbs_rng=factor_rng,
        max_factor_order=protocol.DATASETS[
            context.runtime.dataset
        ]["max_factor_order"],
        gibbs_logit_clip=protocol.LOGIT_CLIP,
        compiled_workload=factor_compiled,
        direction_logit_clip=protocol.LOGIT_CLIP,
    )
    _assert_frame_equal(
        factor_copy, reconstructed[protocol.ARM_FACTOR][0]
    )
    recorded_factor = arms[protocol.ARM_FACTOR]
    if (
        recorded_factor["gibbs_rng"]["address_uint64"] != factor_seed
        or recorded_factor["gibbs_rng"]["initial_state_sha256"]
        != common.rng_state_sha256(np.random.default_rng(factor_seed))
        or recorded_factor["gibbs_rng"]["endpoint_state_sha256"]
        != common.rng_state_sha256(factor_rng)
        or common.rng_state_sha256(factor_main_rng)
        != replay.common_update["pre_mutation_rng_sha256"]
        or factor_diagnostics["gibbs_microsteps"]
        != recorded_factor["kernel_diagnostics"]["gibbs_microsteps"]
        or factor_diagnostics["conditional_logit_clipped_count"]
        != recorded_factor["kernel_diagnostics"][
            "conditional_logit_clipped_count"
        ]
    ):
        raise RuntimeError("factor 吉布斯随机重放/诊断失败")

    gap_seed = protocol.gibbs_address_seed(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        protocol.ARM_GAP_L1,
        mode=mode,
    )
    gap_rng = np.random.default_rng(gap_seed)
    gap_copy, gap_mask, gap_diagnostics = evolve_step_gap_l1_global(
        context.current,
        replay.donors,
        context.runtime.schema,
        context.runtime.queries,
        context.runtime.runtime_target,
        context.query_counts,
        participate=replay.common_update["participate"],
        initial_mask=replay.common_update["copy_masks"],
        reference_scale=reference_scale,
        rng=gap_rng,
        n_sweeps=protocol.GIBBS_SWEEPS,
        eta=protocol.ETA,
        strength=protocol.GAP_L1_STRENGTH,
        floor=protocol.GAP_L1_FLOOR,
        logit_clip=protocol.LOGIT_CLIP,
        compiled_workload=gap_compiled,
        verify_full_recount=True,
    )
    recorded_gap = arms[protocol.ARM_GAP_L1]
    _assert_frame_equal(gap_copy, reconstructed[protocol.ARM_GAP_L1][0])
    if (
        not np.array_equal(gap_mask, reconstructed[protocol.ARM_GAP_L1][2])
        or recorded_gap["gibbs_rng"]["address_uint64"] != gap_seed
        or recorded_gap["gibbs_rng"]["initial_state_sha256"]
        != common.rng_state_sha256(np.random.default_rng(gap_seed))
        or recorded_gap["gibbs_rng"]["endpoint_state_sha256"]
        != common.rng_state_sha256(gap_rng)
        or gap_diagnostics["microstep_trace_sha256"]
        != recorded_gap["kernel_diagnostics"]["microstep_trace_sha256"]
        or gap_diagnostics["probability_bins"]
        != recorded_gap["kernel_diagnostics"]["probability_bins"]
        or gap_diagnostics["gibbs_microsteps"]
        != protocol.GIBBS_SWEEPS * gap_diagnostics["active_switches_k"]
    ):
        raise RuntimeError("新核逐微步随机重放或 8*K 身份失败")
    return {
        "gap_microsteps": int(gap_diagnostics["gibbs_microsteps"]),
        "gap_clip_hits": int(gap_diagnostics["clip_hit_count"]),
        "gap_nonfinite_conditions": int(
            gap_diagnostics["nonfinite_condition_count"]
        ),
        "gap_exact_one_way_probabilities": int(
            gap_diagnostics["exact_zero_or_one_probability_count"]
        ),
        "factor_clip_hits": int(
            factor_diagnostics["conditional_logit_clipped_count"]
        ),
    }


def _audit_exact_pair(
    context: common.StateContext,
    pair: Mapping[str, Any],
) -> None:
    if (
        pair.get("address_status")
        != "already_exact_deterministic_no_op"
        or pair.get("shared_replay", {}).get("stopped_before_donor") is not True
    ):
        raise RuntimeError("精确命中地址没有在供体前确定性停止")
    scientific_arms = [
        collector._strip_diagnostic(pair["arms"][arm])
        for arm in protocol.ARMS
    ]
    if not scientific_arms[0] == scientific_arms[1] == scientific_arms[2]:
        raise RuntimeError("精确命中地址三组不一致")
    current_hash = common.frame_sha256(context.current)
    for arm in protocol.ARMS:
        record = pair["arms"][arm]
        if (
            record["table_sha256"]
            != {"copy_only": current_hash, "full": current_hash}
            or record["copy_edits"]
            or record["mutation_events"]
            or record["query_delta"] != {"copy_only": [], "full": []}
        ):
            raise RuntimeError("精确命中 no-op 稀疏记录无效")


def validate_structural_audit(value: Mapping[str, Any], *, mode: str) -> None:
    if (
        value.get("structural_audit_format") != AUDIT_FORMAT
        or value.get("status") != "complete"
        or value.get("mode") != mode
        or value.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or value.get("structural_audit_scientific_sha256")
        != protocol.canonical_sha256(scientific_payload(value))
        or value.get("audit_passed") is not True
    ):
        raise RuntimeError("第 6B-1 阶段结构审计结构/身份失败")


def audit_collection(
    mode: str,
    calibration_path: str | Path,
    collection_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None,
    confirmed_execution_commit: str | None,
) -> tuple[Path, dict[str, Any]]:
    protocol.require_run_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"结构审计输出已存在，不覆盖：{output}")
    git, environment = _validate_execution(
        mode, confirmed_execution_commit
    )
    calibration_resolved = Path(calibration_path).resolve()
    collection_resolved = Path(collection_path).resolve()
    calibration_sha = protocol.file_sha256(calibration_resolved)
    collection_sha = protocol.file_sha256(collection_resolved)
    calibration = state_builder._strict_load_json(calibration_resolved)
    collection = state_builder._strict_load_json(collection_resolved)
    calibrator.validate_calibration_manifest(calibration, mode=mode)
    collector.validate_collection(collection, mode=mode)
    if (
        calibration.get("git", {}).get("commit") != git["commit"]
        or collection.get("git", {}).get("commit") != git["commit"]
        or collection["calibration_artifact"]["file_sha256"]
        != calibration_sha
    ):
        raise RuntimeError("定尺/采集/审计实现提交或产物链身份失败")
    scale_index = {
        (row["dataset"], int(row["source_seed"])): row
        for row in calibration["calibrations"]
    }
    pair_index = {row["pair_id"]: row for row in collection["pairs"]}
    bundle = common.load_source_bundle(mode, REPOSITORY_ROOT)
    started = time.perf_counter()
    totals = {
        "pair_count": 0,
        "generated_pair_count": 0,
        "already_exact_pair_count": 0,
        "gap_microsteps": 0,
        "gap_clip_hits": 0,
        "gap_nonfinite_conditions": 0,
        "gap_exact_one_way_probabilities": 0,
        "factor_clip_hits": 0,
    }
    state_audits = []
    for dataset in protocol.DATASET_ORDER:
        runtime = common.load_dataset_runtime(bundle, dataset, REPOSITORY_ROOT)
        factor_compiled = compile_mask_workload(
            runtime.schema,
            runtime.queries,
            max_factor_order=protocol.DATASETS[dataset]["max_factor_order"],
        )
        gap_compiled = compile_gap_l1_workload(runtime.schema, runtime.queries)
        for seed in protocol.mode_seeds(mode):
            scale_row = scale_index[(dataset, seed)]
            for group in protocol.STATE_GROUPS:
                context = common.build_state_context(bundle, runtime, seed, group)
                state_counts = {key: 0 for key in totals if key != "pair_count"}
                for proposal_index in range(protocol.proposals_per_state(
                    dataset, mode=mode
                )):
                    pair = pair_index[protocol.pair_id(
                        dataset, seed, group, proposal_index, mode=mode
                    )]
                    totals["pair_count"] += 1
                    if pair["address_status"] == "already_exact_deterministic_no_op":
                        _audit_exact_pair(context, pair)
                        totals["already_exact_pair_count"] += 1
                        state_counts["already_exact_pair_count"] += 1
                    else:
                        if scale_row["status"] != "calibrated":
                            raise RuntimeError("生成地址缺少正固定参考尺度")
                        observed = _audit_generated_pair(
                            context,
                            pair,
                            float(scale_row["reference_scale"]),
                            mode=mode,
                            factor_compiled=factor_compiled,
                            gap_compiled=gap_compiled,
                        )
                        totals["generated_pair_count"] += 1
                        state_counts["generated_pair_count"] += 1
                        for key, amount in observed.items():
                            totals[key] += amount
                            state_counts[key] += amount
                state_audits.append({
                    "state_id": context.state["state_id"],
                    "checks_passed": True,
                    **state_counts,
                })
                print(
                    f"[Stage6B1 structural audit {mode} "
                    f"{context.state['state_id']}] passed",
                    flush=True,
                )
    checks = {
        "complete_pair_manifest": (
            totals["pair_count"] == len(protocol.expected_pair_ids(mode))
        ),
        "source_donor_direction_participation_initial_mask_replayed": True,
        "common_mutation_specs_replayed": True,
        "three_arms_reconstructed_from_sparse_logs": True,
        "incremental_gap_counts_equal_full_recount": True,
        "gap_microsteps_equal_8k": True,
        "no_gate_fields_or_missing_results": True,
        "gap_all_conditions_finite": totals["gap_nonfinite_conditions"] == 0,
        "gap_all_conditionals_bidirectional": (
            totals["gap_exact_one_way_probabilities"] == 0
        ),
        "gap_logit_clip_hits_zero": totals["gap_clip_hits"] == 0,
    }
    audit_passed = all(checks.values())
    failure_label = None
    if not audit_passed:
        failure_label = (
            "outside_frozen_soft_scale"
            if not all(checks[key] for key in (
                "gap_all_conditions_finite",
                "gap_all_conditionals_bidirectional",
                "gap_logit_clip_hits_zero",
            ))
            else "execution_invalid"
        )
    if protocol.file_sha256(calibration_resolved) != calibration_sha or (
        protocol.file_sha256(collection_resolved) != collection_sha
    ):
        raise RuntimeError("结构审计输入在运行期间改变")
    value = {
        "structural_audit_format": AUDIT_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "artifact_role": (
            "formal_result_blind_structural_audit"
            if mode == "formal" else "pipeline_smoke_only"
        ),
        "mechanism_evidence_emitted": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": state_builder._json_safe(environment),
        "artifact_identity": {
            "calibration_file_sha256": calibration_sha,
            "collection_file_sha256": collection_sha,
            "calibration_scientific_sha256": calibration[
                "calibration_scientific_sha256"
            ],
            "collection_scientific_sha256": collection[
                "collection_scientific_sha256"
            ],
        },
        "checks": checks,
        "totals": totals,
        "state_audits": state_audits,
        "audit_passed": audit_passed,
        "execution_failure_label": failure_label,
        "artifact_paths_diagnostic_only": {
            "calibration": str(calibration_resolved),
            "collection": str(collection_resolved),
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    value["structural_audit_scientific_sha256"] = protocol.canonical_sha256(
        scientific_payload(value)
    )
    if not audit_passed:
        raise RuntimeError(
            f"结构审计未通过，不发布审计产物：{failure_label}"
        )
    validate_structural_audit(value, mode=mode)
    published = state_builder._exclusive_write_json(output, value)
    return published, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--collection", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--confirmed-protocol-sha256", required=True)
    run.add_argument("--confirmed-execution-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        plan = protocol.build_plan(args.mode)
        plan.update({
            "audit_kind": "result_blind_structure_and_random_replay",
            "source_read_started": False,
            "audit_started": False,
        })
        print(json.dumps(
            plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ))
        return
    audit_collection(
        args.mode,
        args.calibration,
        args.collection,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
