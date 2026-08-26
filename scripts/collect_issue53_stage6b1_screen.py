#!/usr/bin/env python3
"""结果盲采集第 6B-1 阶段三组固定状态原始记录。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from table_diffevo.factorized_diffusion import (
    compile_mask_workload,
    evolve_step_factorized_gibbs,
)
from table_diffevo.gap_l1_diffusion import (
    compile_gap_l1_workload,
    evolve_step_gap_l1_global,
    evolve_step_gap_l1_global_batched,
)

if __package__:
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import calibrate_issue53_stage6b1_gap_l1 as calibrator
    from scripts import issue53_stage6b1_common as common
    from scripts.issue53_stage6b1_protocol_loader import load_protocol
else:
    import build_issue53_stage6a_state_library as state_builder
    import calibrate_issue53_stage6b1_gap_l1 as calibrator
    import issue53_stage6b1_common as common
    from issue53_stage6b1_protocol_loader import load_protocol

protocol = load_protocol()


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_FORMAT = "issue53_stage6b1_gap_l1_screen_collection_v1"


def _strip_diagnostic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_diagnostic(item)
            for key, item in value.items()
            if not (
                key.endswith("_diagnostic_only")
                or key.endswith("_elapsed_sec")
                or key == "environment"
            )
        }
    if isinstance(value, list):
        return [_strip_diagnostic(item) for item in value]
    return value


def scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {
            "collection_scientific_sha256",
            "artifact_paths_diagnostic_only",
            "environment",
            "elapsed_sec_diagnostic_only",
        }
    }
    return _strip_diagnostic(payload)


def _validate_execution(
    mode: str,
    confirmed_execution_commit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    git = state_builder._git_identity(REPOSITORY_ROOT)
    if not git["worktree_clean_including_untracked"]:
        raise RuntimeError("筛查采集要求包含未跟踪文件在内的干净工作树")
    if confirmed_execution_commit != git["commit"]:
        raise PermissionError("必须精确确认当前干净实现提交")
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=mode == "formal",
    )
    if hasattr(protocol, "validate_runtime_environment"):
        environment.update(protocol.validate_runtime_environment(mode))
    return git, environment


def _load_calibration(
    path: str | Path,
    *,
    mode: str,
    git_commit: str,
) -> tuple[Path, str, dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    resolved = Path(path).resolve()
    digest = protocol.file_sha256(resolved)
    value = state_builder._strict_load_json(resolved)
    calibrator.validate_calibration_manifest(value, mode=mode)
    if (
        value.get("execution_eligible") is not True
        or value.get("git", {}).get("commit") != git_commit
    ):
        raise RuntimeError("参考尺度清单无执行资格或实现提交不一致")
    index = {
        (row["dataset"], int(row["source_seed"])): row
        for row in value["calibrations"]
    }
    return resolved, digest, value, index


def _mutation_specs_sha256(events: Any) -> str:
    specs = [
        {
            "row_index": int(event["row_index"]),
            "attribute_index": int(event["attribute_index"]),
            "attribute": event["attribute"],
            "sampled_value": event["sampled_value"],
        }
        for event in events
    ]
    return protocol.canonical_sha256(specs)


def _no_op_arm(context: common.StateContext) -> dict[str, Any]:
    current = context.current
    q = context.query_counts
    error = common.gap_error_float(context.runtime, q)
    old = common.exact_quadratic_diagnostics(context, current, current)
    geometry = common.query_geometry(context.runtime, q, q)
    return {
        "retained_unconditionally": True,
        "copy_edits": [],
        "mutation_events": [],
        "table_sha256": {
            "copy_only": common.frame_sha256(current),
            "full": common.frame_sha256(current),
        },
        "query_delta": {"copy_only": [], "full": []},
        "gap_l1_float_reading_only": {
            "current": error,
            "copy_only": {
                "after": error,
                "gain": 0.0,
                "positive_gain": 0.0,
                "negative_harm": 0.0,
            },
            "full": {
                "after": error,
                "gain": 0.0,
                "positive_gain": 0.0,
                "negative_harm": 0.0,
            },
        },
        "query_geometry": {"copy_only": geometry, "full": geometry},
        "old_squared_geometry": old,
        "work": {
            "participating_rows": 0,
            "active_switches": 0,
            "final_on_switches": 0,
            "changed_rows_copy_only": 0,
            "changed_cells_copy_only": 0,
            "changed_rows_full": 0,
            "mutation_rows": 0,
            "mutation_changed_cells": 0,
        },
        "kernel_diagnostics": {
            "kernel": "deterministic_no_op_already_exact",
            "no_gate": True,
            "gibbs_microsteps": 0,
        },
    }


@dataclass
class _PreparedPair:
    started: float
    proposal_index: int
    replay: common.PairReplay
    mutation_specs: Any
    independent: dict[str, Any]
    factor: dict[str, Any]
    gap_seed: int
    gap_rng: np.random.Generator
    gap_initial_rng_sha: str


def _expected_gap_backend() -> str | None:
    return getattr(
        protocol,
        "COLLECTION_GAP_BACKEND",
        getattr(protocol, "NEW_KERNEL_BACKEND", None),
    )


def _prepare_pair(
    context: common.StateContext,
    proposal_index: int,
    *,
    mode: str,
    factor_compiled: Any,
) -> _PreparedPair:
    started = time.perf_counter()
    replay = common.replay_pair(context, proposal_index, mode=mode)
    source = replay.source_pair
    common_update = replay.common_update
    attributes = context.runtime.schema.attribute_names()
    update_seed = int(source["rng"]["update_address_uint64"])
    mutation_specs = common_update["mutation_events"]

    independent_mask = np.asarray(
        common_update["copy_masks"], dtype=bool
    )
    independent = common.arm_raw_metrics(
        context,
        replay,
        common_update["copy_table"],
        independent_mask,
        mutation_specs,
    )
    independent["kernel_diagnostics"] = {
        "kernel": "independent_b_s0",
        "no_gate": True,
        "gibbs_microsteps": 0,
        "initial_copy_probabilities_sha256": common.array_sha256(
            common_update["copy_probabilities"]
        ),
    }

    factor_seed = protocol.gibbs_address_seed(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        protocol.ARM_FACTOR,
        mode=mode,
    )
    factor_rng = np.random.default_rng(factor_seed)
    factor_initial_rng_sha = common.rng_state_sha256(factor_rng)
    factor_main_rng = np.random.default_rng(update_seed)
    factor_table, factor_diagnostics = evolve_step_factorized_gibbs(
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
    if (
        common.rng_state_sha256(factor_main_rng)
        != common_update["pre_mutation_rng_sha256"]
    ):
        raise RuntimeError("factor 组错位共同主随机流")
    factor_mask = common.final_mask_from_copy_table(
        context.current, replay.donors, factor_table, attributes
    )
    factor = common.arm_raw_metrics(
        context,
        replay,
        factor_table,
        factor_mask,
        mutation_specs,
    )
    factor["kernel_diagnostics"] = factor_diagnostics
    factor["gibbs_rng"] = {
        "address_uint64": int(factor_seed),
        "initial_state_sha256": factor_initial_rng_sha,
        "endpoint_state_sha256": common.rng_state_sha256(factor_rng),
    }

    gap_seed = protocol.gibbs_address_seed(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        protocol.ARM_GAP_L1,
        mode=mode,
    )
    gap_rng = np.random.default_rng(gap_seed)
    gap_initial_rng_sha = common.rng_state_sha256(gap_rng)

    return _PreparedPair(
        started=started,
        proposal_index=int(proposal_index),
        replay=replay,
        mutation_specs=mutation_specs,
        independent=independent,
        factor=factor,
        gap_seed=int(gap_seed),
        gap_rng=gap_rng,
        gap_initial_rng_sha=gap_initial_rng_sha,
    )


def _finish_pair(
    context: common.StateContext,
    prepared: _PreparedPair,
    reference_scale: float,
    gap_result: tuple[pd.DataFrame, np.ndarray, dict[str, Any]],
) -> dict[str, Any]:
    gap_table, gap_mask, gap_diagnostics = gap_result
    replay = prepared.replay
    source = replay.source_pair
    common_update = replay.common_update
    update_seed = int(source["rng"]["update_address_uint64"])
    gap = common.arm_raw_metrics(
        context,
        replay,
        gap_table,
        gap_mask,
        prepared.mutation_specs,
        query_device=getattr(protocol, "GAP_L1_DEVICE", "numpy"),
        copy_query_counts=(
            np.asarray(
                gap_diagnostics["final_query_counts"], dtype=np.int64
            )
            if getattr(protocol, "GAP_L1_DEVICE", "numpy") == "cuda"
            else None
        ),
    )
    gap["kernel_diagnostics"] = gap_diagnostics
    gap["gibbs_rng"] = {
        "address_uint64": prepared.gap_seed,
        "initial_state_sha256": prepared.gap_initial_rng_sha,
        "endpoint_state_sha256": common.rng_state_sha256(prepared.gap_rng),
    }

    return state_builder._json_safe({
        "pair_id": source["pair_id"],
        "dataset": context.runtime.dataset,
        "source_seed": int(context.state["seed"]),
        "state_group": context.state["state_group"],
        "proposal_index": prepared.proposal_index,
        "address_status": "generated_unconditionally",
        "retained_unconditionally": True,
        "shared_replay": {
            "current_table_sha256": common.frame_sha256(context.current),
            "donor_address_uint64": int(source["rng"]["donor_address_uint64"]),
            "update_address_uint64": update_seed,
            "donor_indices_sha256": common.array_sha256(
                replay.donor_indices
            ),
            "direction_scores_sha256": common.array_sha256(
                replay.direction_scores
            ),
            "participate_sha256": common.array_sha256(
                common_update["participate"]
            ),
            "initial_copy_mask_sha256": common.array_sha256(
                common_update["copy_masks"]
            ),
            "mutation_specs_sha256": _mutation_specs_sha256(
                prepared.mutation_specs
            ),
            "main_rng_initial_state_sha256": common_update[
                "initial_rng_sha256"
            ],
            "main_rng_pre_mutation_state_sha256": common_update[
                "pre_mutation_rng_sha256"
            ],
            "main_rng_endpoint_state_sha256": common_update[
                "endpoint_rng_sha256"
            ],
            "runtime_device_for_source_replay": context.runtime_device,
            "new_kernel_device": _expected_gap_backend(),
            "gap_l1_reference_scale": float(reference_scale),
        },
        "arms": {
            protocol.ARM_INDEPENDENT: prepared.independent,
            protocol.ARM_FACTOR: prepared.factor,
            protocol.ARM_GAP_L1: gap,
        },
        "elapsed_sec_diagnostic_only": (
            time.perf_counter() - prepared.started
        ),
    })


def _collect_pair(
    context: common.StateContext,
    proposal_index: int,
    reference_scale: float,
    *,
    mode: str,
    factor_compiled: Any,
    gap_compiled: Any,
) -> dict[str, Any]:
    prepared = _prepare_pair(
        context,
        proposal_index,
        mode=mode,
        factor_compiled=factor_compiled,
    )
    gap_result = evolve_step_gap_l1_global(
        context.current,
        prepared.replay.donors,
        context.runtime.schema,
        context.runtime.queries,
        context.runtime.runtime_target,
        context.query_counts,
        participate=prepared.replay.common_update["participate"],
        initial_mask=prepared.replay.common_update["copy_masks"],
        reference_scale=reference_scale,
        rng=prepared.gap_rng,
        n_sweeps=protocol.GIBBS_SWEEPS,
        eta=protocol.ETA,
        strength=protocol.GAP_L1_STRENGTH,
        floor=protocol.GAP_L1_FLOOR,
        logit_clip=protocol.LOGIT_CLIP,
        compiled_workload=gap_compiled,
        verify_full_recount=True,
        device=getattr(protocol, "GAP_L1_DEVICE", "numpy"),
    )
    return _finish_pair(context, prepared, reference_scale, gap_result)


def _collect_state_batched(
    context: common.StateContext,
    reference_scale: float,
    *,
    mode: str,
    factor_compiled: Any,
    gap_compiled: Any,
) -> list[dict[str, Any]]:
    proposal_count = protocol.proposals_per_state(
        context.runtime.dataset, mode=mode
    )
    expected = protocol.address_batch_size(mode)
    if proposal_count != expected:
        raise RuntimeError("冻结状态地址数与批量协议不一致")
    prepared = [
        _prepare_pair(
            context,
            proposal_index,
            mode=mode,
            factor_compiled=factor_compiled,
        )
        for proposal_index in range(proposal_count)
    ]
    gap_results = evolve_step_gap_l1_global_batched(
        context.current,
        [item.replay.donors for item in prepared],
        context.runtime.schema,
        context.runtime.queries,
        context.runtime.runtime_target,
        context.query_counts,
        participates=[
            item.replay.common_update["participate"] for item in prepared
        ],
        initial_masks=[
            item.replay.common_update["copy_masks"] for item in prepared
        ],
        reference_scale=reference_scale,
        rngs=[item.gap_rng for item in prepared],
        n_sweeps=protocol.GIBBS_SWEEPS,
        eta=protocol.ETA,
        strength=protocol.GAP_L1_STRENGTH,
        floor=protocol.GAP_L1_FLOOR,
        logit_clip=protocol.LOGIT_CLIP,
        compiled_workload=gap_compiled,
        verify_full_recount=True,
        device=getattr(protocol, "GAP_L1_DEVICE", "numpy"),
    )
    if len(gap_results) != proposal_count:
        raise RuntimeError("生产批量缺口核返回地址数不一致")
    return [
        _finish_pair(context, item, reference_scale, gap_result)
        for item, gap_result in zip(prepared, gap_results)
    ]


def _collect_exact_state(
    context: common.StateContext,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    no_op = _no_op_arm(context)
    for proposal_index in range(protocol.proposals_per_state(
        context.runtime.dataset, mode=mode
    )):
        rows.append({
            "pair_id": protocol.pair_id(
                context.runtime.dataset,
                int(context.state["seed"]),
                context.state["state_group"],
                proposal_index,
                mode=mode,
            ),
            "dataset": context.runtime.dataset,
            "source_seed": int(context.state["seed"]),
            "state_group": context.state["state_group"],
            "proposal_index": int(proposal_index),
            "address_status": "already_exact_deterministic_no_op",
            "retained_unconditionally": True,
            "shared_replay": {
                "current_table_sha256": common.frame_sha256(context.current),
                "donor_address_uint64": None,
                "update_address_uint64": None,
                "gap_l1_reference_scale": None,
                "stopped_before_donor": True,
            },
            "arms": {
                arm: state_builder._json_safe(no_op)
                for arm in protocol.ARMS
            },
            "elapsed_sec_diagnostic_only": 0.0,
        })
    return rows


def validate_collection(value: Mapping[str, Any], *, mode: str) -> None:
    expected_ids = protocol.expected_pair_ids(mode)
    pairs = value.get("pairs")
    observed_ids = (
        tuple(row.get("pair_id") for row in pairs)
        if isinstance(pairs, list) else ()
    )
    if (
        value.get("collection_format") != COLLECTION_FORMAT
        or value.get("status") != "complete"
        or value.get("mode") != mode
        or value.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or observed_ids != expected_ids
        or len(set(observed_ids)) != len(observed_ids)
        or value.get("collection_scientific_sha256")
        != protocol.canonical_sha256(scientific_payload(value))
    ):
        raise RuntimeError("第 6B-1 阶段采集集合结构/身份失败")
    forbidden = {
        "accepted", "rejected", "retry", "rollback", "winner",
        "selected_proposal", "final_screen_classification",
    }

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            overlap = forbidden.intersection(item)
            if overlap:
                raise RuntimeError(f"采集器出现门控/结论字段：{sorted(overlap)}")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    # 协议 manifest 会以否定布尔值显式列出这些词；这里只检查采集结果区，
    # 防止结果记录自身出现候选门控或筛查结论。
    walk(pairs)
    for pair in pairs:
        arms = pair.get("arms", {})
        if (
            pair.get("retained_unconditionally") is not True
            or not isinstance(arms, dict)
            or set(arms) != set(protocol.ARMS)
        ):
            raise RuntimeError("逐地址三组覆盖或无条件保留身份失败")
        for arm in protocol.ARMS:
            if arms[arm].get("retained_unconditionally") is not True:
                raise RuntimeError("组结果没有无条件保留")
        gap = arms[protocol.ARM_GAP_L1]["kernel_diagnostics"]
        if pair["address_status"] == "generated_unconditionally" and (
            gap.get("gibbs_microsteps")
            != protocol.GIBBS_SWEEPS * gap.get("active_switches_k", -1)
        ):
            raise RuntimeError("新核微步数不等于 8*K")
        expected_backend = _expected_gap_backend()
        if (
            expected_backend is not None
            and pair["address_status"] == "generated_unconditionally"
            and (
                pair.get("shared_replay", {}).get("new_kernel_device")
                != expected_backend
                or gap.get("backend") != expected_backend
            )
        ):
            raise RuntimeError("新核采集记录没有绑定冻结 CUDA 后端")
        if (
            getattr(protocol, "BATCHED_GAP_EXECUTION", False)
            and pair["address_status"] == "generated_unconditionally"
        ):
            batch = gap.get("batch_execution", {})
            if (
                batch.get("format")
                != protocol.PRODUCTION_BATCH_EXECUTION_FORMAT
                or batch.get("batch_size")
                != protocol.address_batch_size(mode)
                or batch.get("batch_index") != pair["proposal_index"]
                or batch.get("strict_internal_order_preserved") is not True
            ):
                raise RuntimeError("新核采集记录没有绑定冻结批次身份")


def collect_screen(
    mode: str,
    calibration_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None,
    confirmed_execution_commit: str | None,
) -> tuple[Path, dict[str, Any]]:
    protocol.require_run_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if hasattr(protocol, "assert_stage_output_path"):
        protocol.assert_stage_output_path(
            REPOSITORY_ROOT, mode, "collection", output
        )
    if output.exists():
        raise FileExistsError(f"采集输出已存在，不覆盖：{output}")
    git, environment = _validate_execution(
        mode, confirmed_execution_commit
    )
    calibration_resolved, calibration_sha, calibration, scale_index = (
        _load_calibration(
            calibration_path, mode=mode, git_commit=git["commit"]
        )
    )
    bundle = common.load_source_bundle(mode, REPOSITORY_ROOT)
    started = time.perf_counter()
    pairs = []
    state_manifest = []
    for dataset in protocol.DATASET_ORDER:
        runtime = common.load_dataset_runtime(
            bundle, dataset, REPOSITORY_ROOT
        )
        factor_compiled = compile_mask_workload(
            runtime.schema,
            runtime.queries,
            max_factor_order=protocol.DATASETS[dataset]["max_factor_order"],
        )
        gap_compiled = compile_gap_l1_workload(
            runtime.schema, runtime.queries
        )
        for seed in protocol.mode_seeds(mode):
            calibration_row = scale_index[(dataset, seed)]
            for group in protocol.STATE_GROUPS:
                context = common.build_state_context(
                    bundle, runtime, seed, group
                )
                target_numerators = (
                    runtime.source_target * runtime.runtime_n_records
                )
                exact = bool(np.all(
                    target_numerators
                    - context.query_counts * runtime.source_n_records
                    == 0
                ))
                state_started = time.perf_counter()
                if exact:
                    state_pairs = _collect_exact_state(context, mode=mode)
                else:
                    if calibration_row["status"] != "calibrated":
                        raise RuntimeError(
                            "非精确状态没有正固定参考尺度，拒绝采集"
                        )
                    reference_scale = float(
                        calibration_row["reference_scale"]
                    )
                    if getattr(protocol, "BATCHED_GAP_EXECUTION", False):
                        state_pairs = _collect_state_batched(
                            context,
                            reference_scale,
                            mode=mode,
                            factor_compiled=factor_compiled,
                            gap_compiled=gap_compiled,
                        )
                    else:
                        state_pairs = [
                            _collect_pair(
                                context,
                                proposal_index,
                                reference_scale,
                                mode=mode,
                                factor_compiled=factor_compiled,
                                gap_compiled=gap_compiled,
                            )
                            for proposal_index in range(
                                protocol.proposals_per_state(
                                    dataset, mode=mode
                                )
                            )
                        ]
                pairs.extend(state_pairs)
                state_manifest.append({
                    "state_id": context.state["state_id"],
                    "dataset": dataset,
                    "source_seed": int(seed),
                    "state_group": group,
                    "current_table_sha256": common.frame_sha256(
                        context.current
                    ),
                    "current_query_counts": context.query_counts.tolist(),
                    "current_query_answers_sha256": common.array_sha256(
                        context.query_counts
                    ),
                    "already_exact": exact,
                    "pair_count": len(state_pairs),
                    "gap_l1_batch_execution": (
                        {
                            "logical_batch_size": len(state_pairs),
                            "production_kernel_called": not exact,
                            "format": (
                                protocol.PRODUCTION_BATCH_EXECUTION_FORMAT
                                if not exact else None
                            ),
                            "address_order": "proposal_index_ascending",
                        }
                        if getattr(protocol, "BATCHED_GAP_EXECUTION", False)
                        else None
                    ),
                    "elapsed_sec_diagnostic_only": (
                        time.perf_counter() - state_started
                    ),
                })
                print(
                    f"[Stage6B1 collect {mode} {context.state['state_id']}] "
                    f"pairs={len(state_pairs)} exact={exact}",
                    flush=True,
                )
    source_hashes = protocol.assert_source_artifact_identities(
        REPOSITORY_ROOT, mode
    )
    if protocol.file_sha256(calibration_resolved) != calibration_sha:
        raise RuntimeError("参考尺度清单在采集期间改变")
    if state_builder._git_identity(REPOSITORY_ROOT) != git:
        raise RuntimeError("采集期间 git 身份改变")
    value = {
        "collection_format": COLLECTION_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "artifact_role": (
            "formal_fixed_state_development_screen_raw_collection"
            if mode == "formal" else "pipeline_smoke_only"
        ),
        "mechanism_evidence_emitted": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": state_builder._json_safe(environment),
        "source_artifact_sha256": source_hashes,
        "calibration_artifact": {
            "file_sha256": calibration_sha,
            "scientific_sha256": calibration[
                "calibration_scientific_sha256"
            ],
        },
        "no_gate_contract": dict(protocol.NO_GATE_CONTRACT),
        "state_manifest": state_manifest,
        "pairs": pairs,
        "manifest": {
            "pair_count": len(pairs),
            "arm_count": len(protocol.ARMS),
            "measurement_leg_count": len(protocol.MEASUREMENT_LEGS),
            "arm_leg_record_count": (
                len(pairs) * len(protocol.ARMS) * len(protocol.MEASUREMENT_LEGS)
            ),
            "pair_ids_in_fixed_order": list(protocol.expected_pair_ids(mode)),
        },
        "artifact_paths_diagnostic_only": {
            "calibration": str(calibration_resolved),
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    value["collection_scientific_sha256"] = protocol.canonical_sha256(
        scientific_payload(value)
    )
    validate_collection(value, mode=mode)
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
    run.add_argument("--output", required=True)
    run.add_argument("--confirmed-protocol-sha256", required=True)
    run.add_argument("--confirmed-execution-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        print(json.dumps(
            protocol.build_plan(args.mode),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    collect_screen(
        args.mode,
        args.calibration,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
