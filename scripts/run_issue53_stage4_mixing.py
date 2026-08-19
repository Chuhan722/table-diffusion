#!/usr/bin/env python3
"""Run the frozen Issue #53 Stage 4 shared-sweep Gibbs qualification."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from table_diffevo.objective import compute_loss, compute_residual
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized

if __package__:
    from scripts import build_issue53_stage4_state_library as library_builder
    from scripts import issue53_stage4_protocol as frozen
    from scripts import probe_factorized_gibbs_mixing as probe
else:
    import build_issue53_stage4_state_library as library_builder
    import issue53_stage4_protocol as frozen
    import probe_factorized_gibbs_mixing as probe


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


def _load_json_strict(path: str | Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON 包含非标准数值：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("JSON 根必须是 object")
    return value


def _git_identity() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _expected_state_ids(protocol: dict) -> list[str]:
    return [
        f"{dataset}__seed_{seed}__{group}"
        for dataset in protocol["dataset_order"]
        for seed in protocol["seeds"]
        for group in protocol["state_groups"]
    ]


def _validate_library(
    mode: str,
    library_path: str | Path,
    confirmed_protocol_sha256: str | None,
) -> tuple[Path, dict, dict]:
    frozen.require_qualification_confirmation(
        mode, confirmed_protocol_sha256
    )
    protocol = frozen.stage4_protocol(mode)
    path = Path(library_path).resolve()
    library = _load_json_strict(path)
    state_ids = [state.get("state_id") for state in library.get("states", [])]
    raw_manifest = library.get("manifest")
    manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
    shard_provenance = manifest.get("source_seed_shard_sha256")
    expected_shard_keys = [str(seed) for seed in protocol["seeds"]]
    gates = {
        "format": library.get("state_library_format")
        == frozen.STATE_LIBRARY_FORMAT,
        "status": library.get("status") == "complete",
        "mode": library.get("mode") == mode,
        "artifact_scope": library.get("artifact_scope") == "full",
        "selected_seeds": library.get("selected_seeds")
        == protocol["seeds"],
        "manifest_type": isinstance(raw_manifest, dict),
        "protocol": library.get("protocol") == protocol,
        "protocol_sha256": library.get("protocol_sha256")
        == frozen.protocol_sha256(mode),
        "state_ids": state_ids == _expected_state_ids(protocol),
        "state_ids_unique": len(state_ids) == len(set(state_ids)),
        "manifest_state_count": manifest.get("state_count")
        == len(state_ids),
        "manifest_state_ids": manifest.get("state_ids_in_fixed_order")
        == state_ids,
        "manifest_dataset_order": manifest.get("dataset_order")
        == protocol["dataset_order"],
        "manifest_seed_order": manifest.get("seed_order")
        == protocol["seeds"],
        "manifest_state_group_order": manifest.get("state_group_order")
        == protocol["state_groups"],
        "shard_provenance": isinstance(shard_provenance, dict)
        and (
            shard_provenance == {}
            or (
                list(shard_provenance) == expected_shard_keys
                and all(
                    isinstance(value, str)
                    and len(value) == 64
                    and set(value) <= set("0123456789abcdef")
                    for value in shard_provenance.values()
                )
            )
        ),
        "scientific_sha256": library.get(
            "state_library_scientific_sha256"
        )
        == frozen.canonical_sha256(
            library_builder._scientific_payload(library)
        ),
        "formal_flag": mode != "qualification"
        or library.get("formal_result_valid") is True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Stage 4 状态库绑定失败：{failed}")
    return path, library, gates


def _restore_state(
    entry: dict,
    dataset: dict,
    target: np.ndarray,
    queries: list[dict],
    schema,
) -> tuple[pd.DataFrame, dict]:
    snapshot = entry.get("snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError(f"{entry.get('state_id')} 快照缺失")
    required = {
        "snapshot_format",
        "state_index",
        "round",
        "phase",
        "completed_work_ticks",
        "cumulative_participating_rows",
        "normalized_work",
        "termination_reason",
        "current_squared_loss",
        "current_normalized_l1",
        "current_query_answers",
        "current_residual_signal",
        "current_table_sha256",
        "table_columns",
        "table_records",
        "primary_rng_state_sha256",
        "factorized_gibbs_rng_state_sha256",
        "candidate_evaluation_count_cumulative",
        "direction_reference_scale",
    }
    if required - set(snapshot):
        raise RuntimeError(f"{entry['state_id']} 快照字段不完整")
    columns = schema.attribute_names()
    frame = pd.DataFrame(snapshot["table_records"], columns=columns)
    if (
        snapshot["snapshot_format"] != "natural_work_current_v1"
        or snapshot["table_columns"] != columns
        or len(frame) != dataset["runtime_n_records"]
        or probe._frame_sha256(frame) != snapshot["current_table_sha256"]
    ):
        raise RuntimeError(f"{entry['state_id']} 表身份失败")
    q, residual, _ = evaluate_vectorized(
        frame,
        queries,
        schema,
        target=target,
        n_records=dataset["runtime_n_records"],
        batch_size=256,
        device=dataset["runtime_device"],
        want_fitness=True,
        verbose=False,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
    )
    recorded_q = np.asarray(snapshot["current_query_answers"], dtype=float)
    recorded_residual = np.asarray(
        snapshot["current_residual_signal"], dtype=float
    )
    loss = float(compute_loss(target, q))
    s0 = float(snapshot["direction_reference_scale"])
    if (
        not np.array_equal(np.asarray(q, dtype=float), recorded_q)
        or not np.allclose(
            np.asarray(residual, dtype=float),
            recorded_residual,
            rtol=0.0,
            atol=1e-15,
        )
        or not math.isclose(
            loss,
            float(snapshot["current_squared_loss"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isfinite(s0)
        or s0 <= 0.0
    ):
        raise RuntimeError(f"{entry['state_id']} q/residual/loss/s0 复算失败")
    controls = {
        "snapshot_format": snapshot["snapshot_format"],
        "source_seed": int(entry["seed"]),
        "source_rounds": int(snapshot["round"]),
        "source_temperature": frozen.TAU,
        "source_sweeps": 0,
        "state_round": int(snapshot["round"]),
        "state_sha256": snapshot["current_table_sha256"],
        "current_loss": loss,
        "probe_alpha": frozen.FIXED_ALPHA,
        "direction_reference_scale": s0,
        "direction_reference_scale_round": 0,
        "primary_rng_state_sha256": snapshot[
            "primary_rng_state_sha256"
        ],
        "gibbs_rng_state_sha256": snapshot[
            "factorized_gibbs_rng_state_sha256"
        ],
    }
    return frame, controls


def _compact_probe(result: dict) -> dict:
    compact = copy.deepcopy(result)
    compact.pop("proposal_rows", None)
    for comparison in compact.get("paired", {}).values():
        comparison.pop("values", None)
    return compact


def _run_state(
    entry: dict,
    state_index: int,
    dataset: dict,
    target: np.ndarray,
    queries: list[dict],
    schema,
    *,
    sweeps: int,
) -> dict:
    frame, controls = _restore_state(
        entry, dataset, target, queries, schema
    )
    result = probe._probe_state(
        frame,
        target,
        queries,
        schema,
        seed=entry["seed"],
        state_index=state_index,
        state_rounds=entry["snapshot"]["round"],
        temperatures=[frozen.TAU],
        sweeps=[0, sweeps],
        proposals=dataset["proposals_per_state"],
        device=dataset["runtime_device"],
        max_active_attributes=dataset["max_active_attributes"],
        external_snapshot_controls=controls,
        n_records=dataset["runtime_n_records"],
        rho=frozen.RHO,
        eta=frozen.ETA,
        max_factor_order=dataset["max_factor_order"],
        selection_scale_invariant=True,
        selection_scale_invariant_min_spread=1e-3,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
    )
    if (
        result["state_sha256"] != entry["snapshot"][
            "current_table_sha256"
        ]
        or result["state_loss"] != controls["current_loss"]
        or result["probe_alpha"] != frozen.FIXED_ALPHA
        or result["direction_reference_scale"]
        != controls["direction_reference_scale"]
        or result["reference_scale_proposal_index"] is not None
    ):
        raise RuntimeError(f"{entry['state_id']} probe 控制漂移")
    return {
        "state_id": entry["state_id"],
        "dataset": entry["dataset"],
        "seed": entry["seed"],
        "state_group": entry["state_group"],
        "state_sha256": entry["snapshot"]["current_table_sha256"],
        "probe": _compact_probe(result),
    }


def _weighted_kernel(
    state_rows: list[dict],
    config_name: str,
    *,
    width_group: str | None = None,
) -> dict:
    weighted = {metric: 0.0 for metric in KERNEL_METRICS}
    rows = 0
    active_blocks = 0
    for state in state_rows:
        source = state["probe"]["kernel_summary"]
        if width_group is not None:
            source = state["probe"]["kernel_summary_by_active_width"][
                width_group
            ]
        kernel = source[config_name]
        count = int(kernel["participating_active_rows"])
        rows += count
        active_blocks += int(kernel["active_blocks"])
        for metric in KERNEL_METRICS:
            weighted[metric] += float(kernel[metric]) * count
    result = {
        "participating_active_rows": rows,
        "active_blocks": active_blocks,
    }
    for metric in KERNEL_METRICS:
        result[metric] = weighted[metric] / rows if rows else 0.0
    return result


def _mixing_group(
    state_rows: list[dict],
    sweeps: int,
    *,
    width_group: str | None = None,
    require_nonempty: bool,
) -> dict:
    baseline_name = probe._gibbs_name(frozen.TAU, 0)
    candidate_name = probe._gibbs_name(frozen.TAU, sweeps)
    baseline = _weighted_kernel(
        state_rows, baseline_name, width_group=width_group
    )
    candidate = _weighted_kernel(
        state_rows, candidate_name, width_group=width_group
    )
    initial_gap = baseline["absolute_expected_direction_gap"]
    remaining_gap = candidate["absolute_expected_direction_gap"]
    recovery = (
        1.0 - remaining_gap / initial_gap if initial_gap > 0.0 else 1.0
    )
    nonempty = candidate["participating_active_rows"] > 0
    gated = require_nonempty or nonempty
    tvd_pass = candidate["tvd_to_joint"] <= frozen.TVD_THRESHOLD
    recovery_pass = recovery >= frozen.RECOVERY_THRESHOLD
    passed = bool(
        (nonempty or not require_nonempty)
        and (not gated or (tvd_pass and recovery_pass))
    )
    return {
        "nonempty": nonempty,
        "gated": gated,
        "participating_active_rows": candidate[
            "participating_active_rows"
        ],
        "active_attribute_updates_per_sweep": candidate["active_blocks"],
        "tvd_to_joint": candidate["tvd_to_joint"],
        "initial_expected_direction_gap": initial_gap,
        "remaining_expected_direction_gap": remaining_gap,
        "gap_recovery": recovery,
        "tvd_pass": tvd_pass,
        "gap_recovery_pass": recovery_pass,
        "passed": passed,
    }


def _dataset_attempt(
    dataset_name: str,
    state_rows: list[dict],
    sweeps: int,
    protocol: dict,
    previous_tvd: dict[str, float],
    expected_conditions: dict[str, dict],
) -> tuple[dict, dict[str, float], dict[str, dict]]:
    global_group = _mixing_group(
        state_rows, sweeps, require_nonempty=True
    )
    stage_groups = {
        group: _mixing_group(
            [row for row in state_rows if row["state_group"] == group],
            sweeps,
            require_nonempty=True,
        )
        for group in protocol["required_stage_groups"]
    }
    width_groups = {
        group: _mixing_group(
            state_rows,
            sweeps,
            width_group=group,
            require_nonempty=False,
        )
        for group in protocol["active_width_groups"]
    }

    tau_key = "tau_2"
    probability = [
        row["probe"]["probability_diagnostics_by_temperature"][tau_key]
        for row in state_rows
    ]
    logits = [
        row["probe"]["conditional_logit_diagnostics"][tau_key]
        for row in state_rows
    ]
    samplers = [
        row["probe"]["production_sampler_diagnostics"][tau_key]
        for row in state_rows
    ]
    factors = [row["probe"]["factor_diagnostics"] for row in state_rows]
    probability_gate = all(
        item["distribution_count"] == 0
        or (
            item["all_finite"]
            and item["all_nonnegative"]
            and item["probability_sum_max_error"]
            <= frozen.PROBABILITY_SUM_TOLERANCE
        )
        for item in probability
    )
    clip_hits = sum(int(item["clip_hit_count"]) for item in logits)
    sampler_comparisons = sum(
        int(item["comparison_count"]) for item in samplers
    )
    sampler_mismatches = sum(int(item["mismatch_count"]) for item in samplers)
    energy_max_error = max(
        (float(item["exact_energy_max_error"]) for item in factors),
        default=math.inf,
    )

    condition_rows = []
    next_expected_conditions = dict(expected_conditions)
    conditions_equal = True
    for row in state_rows:
        identity = row["probe"]["shared_condition_identity"]
        state_id = row["state_id"]
        condition_rows.append({
            "state_id": state_id,
            "proposal_sha256": identity["proposal_sha256"],
            "scientific_sha256": identity["scientific_sha256"],
        })
        if state_id in expected_conditions:
            conditions_equal &= identity == expected_conditions[state_id]
        else:
            next_expected_conditions[state_id] = identity
    condition_sha = frozen.canonical_sha256(condition_rows)

    current_tvd = {"global": global_group["tvd_to_joint"]}
    current_tvd.update({
        f"stage:{group}": value["tvd_to_joint"]
        for group, value in stage_groups.items()
    })
    current_tvd.update({
        f"width:{group}": value["tvd_to_joint"]
        for group, value in width_groups.items()
        if value["nonempty"]
    })
    monotonic = all(
        key not in previous_tvd
        or value
        <= previous_tvd[key] + frozen.TVD_MONOTONIC_TOLERANCE
        for key, value in current_tvd.items()
    )
    within_probe_monotonic = all(
        float(item["tvd_snapshot_increase_max"])
        <= frozen.TVD_MONOTONIC_TOLERANCE
        for item in factors
    )

    validity_gates = {
        "all_required_stage_groups_nonempty": all(
            value["nonempty"] for value in stage_groups.values()
        ),
        "probabilities_valid": probability_gate,
        "exact_factor_energy": energy_max_error
        <= frozen.ENERGY_TOLERANCE,
        "production_tape_replay": sampler_comparisons > 0
        and sampler_mismatches == 0,
        "shared_conditions_exact": conditions_equal,
        "tvd_monotonic_across_candidates": monotonic,
        "tvd_monotonic_within_probe": within_probe_monotonic,
        "sweeps_within_hard_cap": sweeps <= frozen.CANDIDATE_SWEEPS[-1],
    }
    qualification_gates = {
        "global_mixing": global_group["passed"],
        "all_stage_groups_mixing": all(
            value["passed"] for value in stage_groups.values()
        ),
        "all_nonempty_width_groups_mixing": all(
            value["passed"] for value in width_groups.values()
        ),
        "zero_conditional_clip_hits": clip_hits == 0,
    }
    valid = all(validity_gates.values())
    passed = valid and all(qualification_gates.values())

    total_active_rows = sum(
        int(item["active_rows"]) for item in factors
    )
    total_factor_count = sum(
        int(item["total_factor_count"]) for item in factors
    )
    total_factor_entries = sum(
        int(item["total_factor_table_entries"]) for item in factors
    )
    total_microsteps = sum(int(item["microsteps"]) for item in samplers)
    result = {
        "dataset": dataset_name,
        "sweeps": sweeps,
        "valid": valid,
        "passed": passed,
        "validity_gates": validity_gates,
        "qualification_gates": qualification_gates,
        "mixing": {
            "global": global_group,
            "stage_groups": stage_groups,
            "active_width_groups": width_groups,
        },
        "numerical_diagnostics": {
            "conditional_clip_hit_count": clip_hits,
            "probability_sum_max_error": max(
                (
                    float(item["probability_sum_max_error"])
                    for item in probability
                ),
                default=math.inf,
            ),
            "exact_energy_max_error": energy_max_error,
            "production_sampler_comparison_count": sampler_comparisons,
            "production_sampler_mismatch_count": sampler_mismatches,
        },
        "shared_condition_rows": condition_rows,
        "shared_condition_scientific_sha256": condition_sha,
        "cost": {
            "participating_active_rows": total_active_rows,
            "active_attribute_updates_per_sweep": global_group[
                "active_attribute_updates_per_sweep"
            ],
            "gibbs_microsteps": total_microsteps,
            "factor_count": total_factor_count,
            "factor_table_entries": total_factor_entries,
            "factor_build_elapsed_sec_diagnostic_only": sum(
                float(item["factor_build_elapsed_sec"]) for item in factors
            ),
            "exact_propagation_elapsed_sec_diagnostic_only": sum(
                float(item["exact_finite_state_propagation_elapsed_sec"])
                for item in factors
            ),
            "production_sample_elapsed_sec_diagnostic_only": sum(
                float(item["production_sampler_elapsed_sec"])
                for item in samplers
            ),
        },
        "state_results": state_rows,
    }
    return result, current_tvd, next_expected_conditions


def _result_from_attempts(attempts: list[dict]) -> tuple[str, int | None]:
    if not attempts:
        return "invalid_or_incomplete", None
    expected = list(frozen.CANDIDATE_SWEEPS[: len(attempts)])
    observed = [attempt.get("sweeps") for attempt in attempts]
    if observed != expected:
        raise RuntimeError("候选 sweep 缺失、乱序或跳级")
    invalid = [
        item for item in attempts if item.get("valid") is not True
    ]
    if invalid:
        if len(invalid) != 1 or attempts[-1] is not invalid[0]:
            raise RuntimeError("结构性 invalid 后仍执行了更高 sweep")
        return "invalid_or_incomplete", None
    first_pass = next(
        (item for item in attempts if item.get("passed") is True), None
    )
    if first_pass is not None:
        if attempts[-1] is not first_pass:
            raise RuntimeError("首次双数据通过后仍执行了更高 sweep")
        sweep = int(first_pass["sweeps"])
        return f"qualified_random_scan_s{sweep}", sweep
    if len(attempts) != len(frozen.CANDIDATE_SWEEPS):
        raise RuntimeError("未通过时错误提前停止")
    return "unqualified_at_s32", None


def _without_diagnostics(value):
    if isinstance(value, dict):
        return {
            key: _without_diagnostics(item)
            for key, item in value.items()
            if "elapsed_sec" not in key
        }
    if isinstance(value, list):
        return [_without_diagnostics(item) for item in value]
    return value


def run_stage4_mixing(
    mode: str,
    state_library_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None = None,
) -> tuple[Path, dict]:
    library_path, library, library_gates = _validate_library(
        mode, state_library_path, confirmed_protocol_sha256
    )
    protocol = frozen.stage4_protocol(mode)
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise FileExistsError(f"Stage 4 报告已存在，不覆盖：{output_file}")
    git = _git_identity()
    if mode == "qualification" and git["dirty"]:
        raise RuntimeError("qualification mixing 必须从 clean worktree 运行")
    if (
        mode == "qualification"
        and library.get("git", {}).get("commit") != git["commit"]
    ):
        raise RuntimeError(
            "qualification mixing 与状态库必须来自同一个 clean commit"
        )

    dataset_runtime = {}
    for dataset_name in protocol["dataset_order"]:
        dataset = protocol["datasets"][dataset_name]
        schema = load_schema(str(REPOSITORY_ROOT / dataset["schema"]))
        queries = load_queries(str(REPOSITORY_ROOT / dataset["queries"]))
        raw_target = np.asarray(
            [query["result"] for query in queries], dtype=float
        )
        target = library_builder._runtime_target(raw_target, dataset)
        dataset_runtime[dataset_name] = (dataset, target, queries, schema)

    attempts = []
    previous_tvd = {name: {} for name in protocol["dataset_order"]}
    expected_conditions = {name: {} for name in protocol["dataset_order"]}
    started = time.perf_counter()
    all_states = library["states"]
    indexed_states = list(enumerate(all_states))
    for sweeps in protocol["candidate_sweeps"]:
        dataset_results = {}
        for dataset_name in protocol["dataset_order"]:
            dataset, target, queries, schema = dataset_runtime[dataset_name]
            selected = [
                (index, entry)
                for index, entry in indexed_states
                if entry["dataset"] == dataset_name
            ]
            state_rows = []
            for state_index, entry in selected:
                state_rows.append(_run_state(
                    entry,
                    state_index,
                    dataset,
                    target,
                    queries,
                    schema,
                    sweeps=sweeps,
                ))
            dataset_result, next_tvd, next_conditions = _dataset_attempt(
                dataset_name,
                state_rows,
                sweeps,
                protocol,
                previous_tvd[dataset_name],
                expected_conditions[dataset_name],
            )
            previous_tvd[dataset_name] = next_tvd
            expected_conditions[dataset_name] = next_conditions
            dataset_results[dataset_name] = dataset_result
            print(
                f"[Stage4 {mode}] sweeps={sweeps} {dataset_name}: "
                f"valid={dataset_result['valid']} "
                f"passed={dataset_result['passed']} "
                f"TVD={dataset_result['mixing']['global']['tvd_to_joint']:.6f} "
                f"recovery={dataset_result['mixing']['global']['gap_recovery']:.2%}",
                flush=True,
            )
        attempt = {
            "sweeps": int(sweeps),
            "valid": all(
                item["valid"] for item in dataset_results.values()
            ),
            "passed": all(
                item["passed"] for item in dataset_results.values()
            ),
            "datasets": dataset_results,
        }
        attempts.append(attempt)
        if not attempt["valid"] or attempt["passed"]:
            break

    result, selected_sweeps = _result_from_attempts(attempts)
    report = {
        "report_format": frozen.REPORT_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": bool(
            mode == "qualification"
            and result != "invalid_or_incomplete"
            and not git["dirty"]
        ),
        "protocol": protocol,
        "protocol_sha256": frozen.protocol_sha256(mode),
        "git": git,
        "state_library_binding": {
            "path": str(library_path),
            "file_sha256": frozen.file_sha256(library_path),
            "scientific_sha256": library[
                "state_library_scientific_sha256"
            ],
            "binding_gates": library_gates,
        },
        "attempted_sweeps": [item["sweeps"] for item in attempts],
        "attempts": attempts,
        "result": result,
        "selected_minimal_sufficient_sweeps": selected_sweeps,
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    scientific_payload = {
        key: report[key]
        for key in (
            "report_format",
            "mode",
            "protocol_sha256",
            "state_library_binding",
            "attempted_sweeps",
            "attempts",
            "result",
            "selected_minimal_sufficient_sweeps",
        )
    }
    report["execution_scientific_sha256"] = frozen.canonical_sha256(
        _without_diagnostics(scientific_payload)
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(f"Stage 4 结果：{result}；报告：{output_file}", flush=True)
    return output_file, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=tuple(frozen.MODE_CONFIG), default="smoke"
    )
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirmed-protocol-sha256")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_stage4_mixing(
        args.mode,
        args.state_library,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
    )


if __name__ == "__main__":
    main()
