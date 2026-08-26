#!/usr/bin/env python3
"""构建第 6B-1 阶段只读固定参考尺度清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from table_diffevo.gap_l1_diffusion import (
    compile_gap_l1_workload,
    isolated_gap_l1_scores,
    stable_nonzero_rms,
)

if __package__:
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import issue53_stage6b1_common as common
    from scripts.issue53_stage6b1_protocol_loader import load_protocol
else:
    import build_issue53_stage6a_state_library as state_builder
    import issue53_stage6b1_common as common
    from issue53_stage6b1_protocol_loader import load_protocol

protocol = load_protocol()


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FORMAT = "issue53_stage6b1_gap_l1_calibration_v1"


def scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {
            "elapsed_sec_diagnostic_only",
            "calibration_scientific_sha256",
            "artifact_paths_diagnostic_only",
        }
    }


def _validate_execution(
    mode: str,
    confirmed_execution_commit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    git = state_builder._git_identity(REPOSITORY_ROOT)
    if not git["worktree_clean_including_untracked"]:
        raise RuntimeError("参考尺度运行要求包含未跟踪文件在内的干净工作树")
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


def _calibrate_one(
    bundle: common.SourceBundle,
    runtime: common.DatasetRuntime,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    context = common.build_state_context(bundle, runtime, seed, "initial")
    target_numerators = runtime.source_target * runtime.runtime_n_records
    exact_residual = (
        target_numerators
        - context.query_counts * runtime.source_n_records
    )
    current_exact = bool(np.all(exact_residual == 0))
    base = {
        "dataset": runtime.dataset,
        "source_seed": int(seed),
        "state_id": context.state["state_id"],
        "current_table_sha256": common.frame_sha256(context.current),
        "current_query_answers_sha256": common.array_sha256(
            context.query_counts
        ),
        "runtime_device_for_donor_replay": context.runtime_device,
        "new_kernel_device": getattr(
            protocol, "NEW_KERNEL_BACKEND", "numpy_float64_cpu"
        ),
        "search_order": "ascending_frozen_proposal_index",
        "current_error_exactly_zero": current_exact,
    }
    if current_exact:
        return {
            **base,
            "status": "already_exact",
            "calibration_proposal_index": None,
            "reference_scale": None,
            "distribution": None,
            "searched_proposal_count": 0,
            "elapsed_sec_diagnostic_only": time.perf_counter() - started,
        }

    compiled = compile_gap_l1_workload(runtime.schema, runtime.queries)
    searched = []
    proposal_count = protocol.proposals_per_state(
        runtime.dataset, mode=bundle.mode
    )
    for proposal_index in range(proposal_count):
        pair, donor_indices, donors, donor_rng = common.replay_donors(
            context, proposal_index, mode=bundle.mode
        )
        isolated_started = time.perf_counter()
        isolated = isolated_gap_l1_scores(
            context.current,
            donors,
            runtime.schema,
            runtime.queries,
            runtime.runtime_target,
            context.query_counts,
            floor=protocol.GAP_L1_FLOOR,
            compiled_workload=compiled,
            exact_target_numerators=(
                runtime.source_target * runtime.runtime_n_records
            ),
            exact_target_denominator=runtime.source_n_records,
            device=getattr(protocol, "GAP_L1_DEVICE", "numpy"),
        )
        scale, distribution = stable_nonzero_rms(isolated["scores"])
        item = {
            "proposal_index": int(proposal_index),
            "pair_id": pair["pair_id"],
            "donor_address_uint64": int(
                pair["rng"]["donor_address_uint64"]
            ),
            "donor_indices_sha256": common.array_sha256(donor_indices),
            "donor_rng": donor_rng,
            "isolated_coordinates_sha256": common.array_sha256(
                isolated["coordinates"]
            ),
            "isolated_scores_sha256": common.array_sha256(
                isolated["scores"]
            ),
            "distribution": distribution,
            "reference_scale": float(scale),
            "elapsed_sec_diagnostic_only": (
                time.perf_counter() - isolated_started
            ),
        }
        searched.append(item)
        if scale > 0.0:
            return {
                **base,
                "status": "calibrated",
                "calibration_proposal_index": int(proposal_index),
                "reference_scale": float(scale),
                "distribution": distribution,
                "selected_address": item,
                "searched_addresses": searched,
                "searched_proposal_count": len(searched),
                "elapsed_sec_diagnostic_only": time.perf_counter() - started,
            }
    return {
        **base,
        "status": "calibration_unsupported",
        "calibration_proposal_index": None,
        "reference_scale": None,
        "distribution": None,
        "selected_address": None,
        "searched_addresses": searched,
        "searched_proposal_count": len(searched),
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }


def validate_calibration_manifest(
    value: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    expected_keys = [
        (dataset, seed)
        for dataset in protocol.DATASET_ORDER
        for seed in protocol.mode_seeds(mode)
    ]
    rows = value.get("calibrations")
    observed_keys = (
        [(row.get("dataset"), row.get("source_seed")) for row in rows]
        if isinstance(rows, list)
        else []
    )
    if (
        value.get("calibration_format") != CALIBRATION_FORMAT
        or value.get("status") != "complete"
        or value.get("mode") != mode
        or value.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or observed_keys != expected_keys
        or value.get("calibration_scientific_sha256")
        != protocol.canonical_sha256(scientific_payload(value))
    ):
        raise RuntimeError("第 6B-1 阶段参考尺度清单结构/身份失败")
    for row in rows:
        expected_backend = getattr(protocol, "NEW_KERNEL_BACKEND", None)
        if expected_backend is not None and (
            row.get("new_kernel_device") != expected_backend
        ):
            raise RuntimeError("参考尺度清单没有绑定冻结 CUDA 后端")
        status = row.get("status")
        if status == "calibrated":
            scale = row.get("reference_scale")
            distribution = row.get("distribution", {})
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not np.isfinite(scale)
                or scale <= 0.0
                or distribution.get("rms") != scale
                or distribution.get("nonzero_count", 0) <= 0
                or row.get("calibration_proposal_index") is None
            ):
                raise RuntimeError("正参考尺度记录无效")
        elif status == "already_exact":
            if (
                row.get("current_error_exactly_zero") is not True
                or row.get("reference_scale") is not None
            ):
                raise RuntimeError("already_exact 记录无效")
        elif status == "calibration_unsupported":
            if row.get("reference_scale") is not None:
                raise RuntimeError("unsupported 不得携带替代参考尺度")
        else:
            raise RuntimeError("未知参考尺度状态")


def build_calibration_manifest(
    mode: str,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None,
    confirmed_execution_commit: str | None,
) -> tuple[Path, dict[str, Any]]:
    protocol.require_run_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"参考尺度输出已存在，不覆盖：{output}")
    git, environment = _validate_execution(
        mode, confirmed_execution_commit
    )
    source_hashes = protocol.assert_source_artifact_identities(
        REPOSITORY_ROOT, mode
    )
    bundle = common.load_source_bundle(mode, REPOSITORY_ROOT)
    started = time.perf_counter()
    rows = []
    for dataset in protocol.DATASET_ORDER:
        runtime = common.load_dataset_runtime(
            bundle, dataset, REPOSITORY_ROOT
        )
        for seed in protocol.mode_seeds(mode):
            row = _calibrate_one(bundle, runtime, seed)
            rows.append(state_builder._json_safe(row))
            print(
                f"[Stage6B1 calibration {mode} {dataset} seed={seed}] "
                f"status={row['status']} R={row['reference_scale']}",
                flush=True,
            )
    eligible = all(
        row["status"] in {"calibrated", "already_exact"} for row in rows
    )
    value = {
        "calibration_format": CALIBRATION_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "artifact_role": (
            "formal_fixed_scale_manifest"
            if mode == "formal" else "pipeline_smoke_only"
        ),
        "mechanism_evidence_emitted": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": state_builder._json_safe(environment),
        "source_artifact_sha256": source_hashes,
        "calibrations": rows,
        "execution_eligible": eligible,
        "execution_failure_label": (
            None if eligible else "calibration_unsupported"
        ),
        "artifact_paths_diagnostic_only": {
            name: str(REPOSITORY_ROOT / binding["path"])
            for name, binding in protocol.SOURCE_ARTIFACTS[mode].items()
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    value["calibration_scientific_sha256"] = protocol.canonical_sha256(
        scientific_payload(value)
    )
    validate_calibration_manifest(value, mode=mode)
    published = state_builder._exclusive_write_json(output, value)
    return published, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--confirmed-protocol-sha256", required=True)
    run.add_argument("--confirmed-execution-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        plan = protocol.build_plan(args.mode)
        plan.update({
            "calibration_count": (
                len(protocol.DATASET_ORDER)
                * len(protocol.mode_seeds(args.mode))
            ),
            "source_read_started": False,
            "calibration_started": False,
        })
        print(json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    build_calibration_manifest(
        args.mode,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
