#!/usr/bin/env python3
"""Materialize Issue #53 Stage 4 states from the frozen independent baseline."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema

if __package__:
    from scripts import issue53_stage4_protocol as frozen
    from scripts import run_issue53_fixed_alpha_calibration as baseline
else:
    import issue53_stage4_protocol as frozen
    import run_issue53_fixed_alpha_calibration as baseline


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MILESTONE_FRACTIONS = {
    "initial": 0.0,
    "work_q25": 0.25,
    "work_q50": 0.50,
    "work_q75": 0.75,
    "terminal": 1.0,
}


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


def _runtime_target(raw_target: np.ndarray, dataset: dict) -> np.ndarray:
    runtime_n = dataset["runtime_n_records"]
    source_n = dataset["n_records"]
    if runtime_n == source_n:
        return np.asarray(raw_target, dtype=float)
    return np.asarray(raw_target, dtype=float) * (runtime_n / source_n)


def _generator_parameters(protocol: dict, dataset: dict, seed: int) -> dict:
    return {
        "n_rounds": protocol["resource_cap_rounds"],
        "seed": seed,
        "beta": 1.0,
        "h": 0.8,
        "rho": protocol["rho"],
        "eta": protocol["eta"],
        "mu": protocol["trajectory_mu"],
        "tol": float("inf"),
        "device": dataset["runtime_device"],
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "marginal",
        "log_every": protocol["resource_cap_rounds"] + 1,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": protocol["fixed_alpha"],
        "alpha_max": protocol["fixed_alpha"],
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": protocol["max_retries"],
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": protocol["direction_strength"],
        "diffusion_direction_normalization": protocol[
            "direction_normalization"
        ],
        "diffusion_direction_logit_clip": protocol[
            "direction_logit_clip"
        ],
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": dataset["max_factor_order"],
        "factorized_gibbs_logit_clip": protocol["gibbs_logit_clip"],
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": protocol["candidate_budget"],
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": protocol[
            "selection_scale_invariant"
        ],
        "selection_scale_invariant_min_spread": protocol[
            "selection_scale_invariant_min_spread"
        ],
        "residual_geometry": protocol["residual_geometry"],
        "residual_geometry_floor": protocol["residual_geometry_floor"],
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": protocol["fixed_alpha"],
        "record_transition_clocks": True,
        "record_natural_work_snapshots": True,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": protocol["patience_ticks"],
    }


def _select_milestones(snapshots: list[dict]) -> list[dict]:
    if len(snapshots) < len(frozen.STATE_GROUPS):
        raise RuntimeError(
            "来源轨迹不足五个互异 natural-work 状态，不能物化 Stage 4"
        )
    state_indices = [item["state_index"] for item in snapshots]
    works = [float(item["normalized_work"]) for item in snapshots]
    if (
        state_indices != sorted(set(state_indices))
        or works != sorted(works)
        or snapshots[0]["phase"] != "initial"
        or snapshots[-1]["termination_reason"] == "in_progress"
    ):
        raise RuntimeError("natural-work 快照顺序或 terminal 身份无效")
    terminal_work = works[-1]
    if not math.isfinite(terminal_work) or terminal_work <= 0.0:
        raise RuntimeError("terminal natural work 必须为正有限值")

    targets = [terminal_work * fraction for fraction in (0.25, 0.5, 0.75)]
    choices = range(1, len(snapshots) - 1)
    candidates = []
    for indices in itertools.combinations(choices, 3):
        error = sum(abs(works[index] - target) for index, target in zip(
            indices, targets
        ))
        candidates.append((error, indices))
    if not candidates:
        raise RuntimeError("没有三个互异的 interior natural-work 状态")
    _, interior = min(candidates, key=lambda item: (item[0], item[1]))
    selected_indices = (0, *interior, len(snapshots) - 1)

    result = []
    for group, index in zip(frozen.STATE_GROUPS, selected_indices):
        fraction = MILESTONE_FRACTIONS[group]
        target_work = terminal_work * fraction
        snapshot = snapshots[index]
        result.append({
            "state_group": group,
            "target_fraction": fraction,
            "target_normalized_work": target_work,
            "selection_absolute_work_error": abs(
                float(snapshot["normalized_work"]) - target_work
            ),
            "source_snapshot_index": index,
            "snapshot": snapshot,
        })
    return result


def _validate_run(
    protocol: dict,
    dataset_name: str,
    dataset: dict,
    seed: int,
    output: pd.DataFrame,
    diagnostics: dict,
) -> None:
    expected_reasons = {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }
    params = diagnostics["params"]
    gates = {
        "terminal_current": diagnostics["output_table_identity"]
        == "terminal_current",
        "termination_reason": diagnostics["termination_reason"]
        in expected_reasons,
        "acceptance_disabled": diagnostics["accept_history"]
        == [True] * diagnostics["rounds_run"],
        "candidate_count": diagnostics["candidate_evaluation_count"]
        == diagnostics["rounds_run"],
        "seed": params["seed"] == seed,
        "n_records": params["n_records"] == dataset["runtime_n_records"],
        "alpha": params["fixed_alpha"] == protocol["fixed_alpha"],
        "scale_invariant": params["selection_scale_invariant"] is True,
        "relative_residual": params["residual_geometry"] == "relative",
        "relative_floor": params["residual_geometry_floor"] == 8.0,
        "independent_kernel": params["factorized_gibbs_sweeps"] == 0,
        "snapshot_recording": params[
            "record_natural_work_snapshots"
        ] is True,
        "terminal_hash": baseline._frame_sha256(output)
        == diagnostics["natural_work_snapshots"][-1][
            "current_table_sha256"
        ],
        "s0_frozen": isinstance(diagnostics["direction_reference_scale"], float)
        and diagnostics["direction_reference_scale"] > 0.0,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(
            f"{dataset_name}/seed{seed} 独立基线身份失败：{failed}"
        )


def _scientific_payload(library: dict) -> dict:
    return {
        "protocol_sha256": library["protocol_sha256"],
        "input_audit": library["input_audit"],
        "trajectories": [
            {
                key: trajectory[key]
                for key in (
                    "dataset",
                    "seed",
                    "runtime_n_records",
                    "runtime_target_sha256",
                    "initial_table_sha256",
                    "terminal_table_sha256",
                    "rounds_run",
                    "candidate_evaluations",
                    "termination_reason",
                    "terminal_normalized_work",
                    "terminal_squared_loss",
                    "terminal_normalized_l1",
                    "direction_reference_scale",
                    "primary_rng_endpoint_sha256",
                    "natural_work_snapshot_manifest",
                    "selected_state_ids",
                )
            }
            for trajectory in library["trajectories"]
        ],
        "states": [
            {
                "state_id": state["state_id"],
                "dataset": state["dataset"],
                "seed": state["seed"],
                "state_group": state["state_group"],
                "target_fraction": state["target_fraction"],
                "target_normalized_work": state[
                    "target_normalized_work"
                ],
                "selection_absolute_work_error": state[
                    "selection_absolute_work_error"
                ],
                "source_snapshot_index": state["source_snapshot_index"],
                "state_index": state["snapshot"]["state_index"],
                "round": state["snapshot"]["round"],
                "normalized_work": state["snapshot"]["normalized_work"],
                "completed_work_ticks": state["snapshot"][
                    "completed_work_ticks"
                ],
                "current_squared_loss": state["snapshot"][
                    "current_squared_loss"
                ],
                "current_normalized_l1": state["snapshot"][
                    "current_normalized_l1"
                ],
                "current_query_answers": state["snapshot"][
                    "current_query_answers"
                ],
                "current_residual_signal": state["snapshot"][
                    "current_residual_signal"
                ],
                "current_table_sha256": state["snapshot"][
                    "current_table_sha256"
                ],
                "direction_reference_scale": state["snapshot"][
                    "direction_reference_scale"
                ],
                "primary_rng_state_sha256": state["snapshot"][
                    "primary_rng_state_sha256"
                ],
            }
            for state in library["states"]
        ],
    }


def build_state_library(
    mode: str,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None = None,
) -> tuple[Path, dict]:
    frozen.require_qualification_confirmation(
        mode, confirmed_protocol_sha256
    )
    protocol = frozen.stage4_protocol(mode)
    protocol_sha = frozen.protocol_sha256(mode)
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise FileExistsError(f"状态库输出已存在，不覆盖：{output_file}")
    git = _git_identity()
    if mode == "qualification" and git["dirty"]:
        raise RuntimeError("qualification 状态库必须从 clean worktree 生成")

    input_audit = {}
    runtime_inputs = {}
    for dataset_name in protocol["dataset_order"]:
        observed = baseline._audit_dataset(REPOSITORY_ROOT, dataset_name)
        expected = protocol["datasets"][dataset_name]
        if (
            observed["sha256"] != expected["input_sha256"]
            or observed["query_identity_sha256"]
            != expected["query_identity_sha256"]
            or observed["target_vector_sha256"]
            != expected["target_vector_sha256"]
            or {str(k): v for k, v in observed["order_counts"].items()}
            != expected["query_order_counts"]
        ):
            raise RuntimeError(f"{dataset_name} 冻结输入身份漂移")
        input_audit[dataset_name] = {
            key: value
            for key, value in observed.items()
            if key not in {"queries", "targets"}
        }
        runtime_inputs[dataset_name] = observed

    trajectories = []
    states = []
    started = time.perf_counter()
    for dataset_name in protocol["dataset_order"]:
        dataset = protocol["datasets"][dataset_name]
        observed = runtime_inputs[dataset_name]
        schema = load_schema(str(REPOSITORY_ROOT / dataset["schema"]))
        queries = load_queries(str(REPOSITORY_ROOT / dataset["queries"]))
        marginals = load_marginals(
            str(REPOSITORY_ROOT / dataset["marginals"])
        )
        target = _runtime_target(
            np.asarray(observed["targets"], dtype=float), dataset
        )
        runtime_target_sha = frozen.canonical_sha256(target.tolist())
        for seed in protocol["seeds"]:
            params = _generator_parameters(protocol, dataset, seed)
            trajectory_started = time.perf_counter()
            output, diagnostics = run_evolution(
                target,
                queries,
                schema,
                n_records=dataset["runtime_n_records"],
                marginals=marginals,
                **params,
            )
            elapsed = time.perf_counter() - trajectory_started
            final_table = diagnostics.pop("final_table")
            snapshots = diagnostics.pop("natural_work_snapshots")
            pd.testing.assert_frame_equal(
                output.reset_index(drop=True),
                final_table.reset_index(drop=True),
            )
            _validate_run(
                protocol,
                dataset_name,
                dataset,
                seed,
                final_table,
                {**diagnostics, "natural_work_snapshots": snapshots},
            )
            selected = _select_milestones(snapshots)
            selected_state_ids = []
            for item in selected:
                group = item["state_group"]
                state_id = f"{dataset_name}__seed_{seed}__{group}"
                selected_state_ids.append(state_id)
                states.append({
                    "state_id": state_id,
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "state_group": group,
                    "target_fraction": item["target_fraction"],
                    "target_normalized_work": item[
                        "target_normalized_work"
                    ],
                    "selection_absolute_work_error": item[
                        "selection_absolute_work_error"
                    ],
                    "source_snapshot_index": item[
                        "source_snapshot_index"
                    ],
                    "snapshot": item["snapshot"],
                })
            terminal = snapshots[-1]
            trajectory = {
                "dataset": dataset_name,
                "seed": int(seed),
                "runtime_n_records": dataset["runtime_n_records"],
                "runtime_target_sha256": runtime_target_sha,
                "initial_table_sha256": diagnostics["initial_table_sha256"],
                "terminal_table_sha256": baseline._frame_sha256(final_table),
                "rounds_run": int(diagnostics["rounds_run"]),
                "candidate_evaluations": int(
                    diagnostics["candidate_evaluation_count"]
                ),
                "termination_reason": diagnostics["termination_reason"],
                "terminal_normalized_work": float(
                    terminal["normalized_work"]
                ),
                "terminal_squared_loss": float(
                    diagnostics["final_current_squared_loss"]
                ),
                "terminal_normalized_l1": float(
                    diagnostics["final_current_normalized_l1"]
                ),
                "direction_reference_scale": float(
                    diagnostics["direction_reference_scale"]
                ),
                "primary_rng_endpoint_sha256": diagnostics[
                    "primary_rng_state_sha256"
                ],
                "recorded_natural_work_state_count": len(snapshots),
                "natural_work_snapshot_manifest": [
                    {
                        "source_snapshot_index": index,
                        "state_index": snapshot["state_index"],
                        "round": snapshot["round"],
                        "phase": snapshot["phase"],
                        "completed_work_ticks": snapshot[
                            "completed_work_ticks"
                        ],
                        "cumulative_participating_rows": snapshot[
                            "cumulative_participating_rows"
                        ],
                        "normalized_work": snapshot["normalized_work"],
                        "work_tick_completed": snapshot[
                            "work_tick_completed"
                        ],
                        "termination_reason": snapshot[
                            "termination_reason"
                        ],
                        "current_squared_loss": snapshot[
                            "current_squared_loss"
                        ],
                        "current_normalized_l1": snapshot[
                            "current_normalized_l1"
                        ],
                        "current_table_sha256": snapshot[
                            "current_table_sha256"
                        ],
                        "primary_rng_state_sha256": snapshot[
                            "primary_rng_state_sha256"
                        ],
                        "direction_reference_scale": snapshot[
                            "direction_reference_scale"
                        ],
                    }
                    for index, snapshot in enumerate(snapshots)
                ],
                "selected_state_ids": selected_state_ids,
                "elapsed_sec_diagnostic_only": elapsed,
            }
            trajectories.append(trajectory)
            print(
                f"[{mode} {dataset_name} seed={seed}] "
                f"{trajectory['termination_reason']} "
                f"rounds={trajectory['rounds_run']} "
                f"work={trajectory['terminal_normalized_work']:.4f} "
                f"states={len(snapshots)} elapsed={elapsed:.2f}s",
                flush=True,
            )

    expected_count = (
        len(protocol["dataset_order"])
        * protocol["expected_states_per_dataset"]
    )
    state_ids = [state["state_id"] for state in states]
    if len(states) != expected_count or len(state_ids) != len(set(state_ids)):
        raise RuntimeError("Stage 4 状态库数量或唯一性失败")
    library = {
        "state_library_format": frozen.STATE_LIBRARY_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": bool(
            protocol["formal_result_valid"] and not git["dirty"]
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "git": git,
        "input_audit": input_audit,
        "trajectories": trajectories,
        "states": states,
        "manifest": {
            "state_count": len(states),
            "state_ids_in_fixed_order": state_ids,
            "dataset_order": protocol["dataset_order"],
            "seed_order": protocol["seeds"],
            "state_group_order": protocol["state_groups"],
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    library["state_library_scientific_sha256"] = frozen.canonical_sha256(
        _scientific_payload(library)
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            library,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(f"Stage 4 状态库：{output_file}", flush=True)
    return output_file, library


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=tuple(frozen.MODE_CONFIG),
        default="smoke",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirmed-protocol-sha256")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    build_state_library(
        args.mode,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
    )


if __name__ == "__main__":
    main()
