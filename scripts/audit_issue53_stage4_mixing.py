#!/usr/bin/env python3
"""Independent audit of Issue #53 Stage 4 state and mixing artifacts.

This module intentionally does not import the Stage 4 builder or runner.
It reconstructs state identities, milestone selection, aggregation, gates and
the global 8 -> 16 -> 32 stopping decision directly from public inputs and the
two artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from table_diffevo.objective import compute_loss, compute_residual
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized

if __package__:
    from scripts import freeze_issue53_test_query_workload_ab as query_identity
    from scripts import issue53_stage4_protocol as frozen
else:
    import freeze_issue53_test_query_workload_ab as query_identity
    import issue53_stage4_protocol as frozen


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


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_same(actual, expected, path: str) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if actual is not expected:
            raise RuntimeError(f"审计不一致：{path}")
        return
    if isinstance(expected, (int, float)) and isinstance(
        actual, (int, float)
    ):
        if not (
            math.isfinite(float(actual))
            and math.isfinite(float(expected))
            and math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"审计不一致：{path}: {actual!r} != {expected!r}"
            )
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"审计字段不一致：{path}")
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"审计列表长度不一致：{path}")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _assert_same(
                actual_item, expected_item, f"{path}[{index}]"
            )
        return
    if actual != expected:
        raise RuntimeError(
            f"审计不一致：{path}: {actual!r} != {expected!r}"
        )


def _runtime_target(raw_target: np.ndarray, dataset: dict) -> np.ndarray:
    return np.asarray(raw_target, dtype=float) * (
        dataset["runtime_n_records"] / dataset["n_records"]
    )


def _input_audit(protocol: dict) -> tuple[dict, dict]:
    audits = {}
    runtime = {}
    for dataset_name in protocol["dataset_order"]:
        dataset = protocol["datasets"][dataset_name]
        paths = {
            key: REPOSITORY_ROOT / dataset[key]
            for key in ("schema", "queries", "marginals")
        }
        hashes = {
            key: frozen.file_sha256(path) for key, path in paths.items()
        }
        if hashes != dataset["input_sha256"]:
            raise RuntimeError(f"{dataset_name} 输入文件 SHA 漂移")
        with paths["queries"].open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise RuntimeError(f"{dataset_name} query payload 无效")
        identity = query_identity.query_set_identity(raw_queries)
        orders = query_identity._order_counts(raw_queries)
        targets = []
        for index, query in enumerate(raw_queries):
            value = query.get("result")
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(
                    f"{dataset_name} target 非整数：{index}"
                )
            targets.append(value)
        target_sha = frozen.canonical_sha256(targets)
        expected_orders = {
            int(key): value
            for key, value in dataset["query_order_counts"].items()
        }
        if (
            len(raw_queries) != dataset["query_count"]
            or identity != dataset["query_identity_sha256"]
            or orders != expected_orders
            or target_sha != dataset["target_vector_sha256"]
        ):
            raise RuntimeError(f"{dataset_name} query/target 身份漂移")
        audits[dataset_name] = {
            "dataset": dataset_name,
            "sha256": hashes,
            "query_count": len(raw_queries),
            "query_identity_sha256": identity,
            "target_vector_sha256": target_sha,
            "order_counts": {
                str(key): value for key, value in orders.items()
            },
        }
        schema = load_schema(str(paths["schema"]))
        queries = load_queries(str(paths["queries"]))
        target = _runtime_target(np.asarray(targets, dtype=float), dataset)
        runtime[dataset_name] = (dataset, target, queries, schema)
    return audits, runtime


def _library_scientific_payload(library: dict) -> dict:
    return {
        "state_library_format": library["state_library_format"],
        "mode": library["mode"],
        "artifact_scope": library["artifact_scope"],
        "selected_seeds": library["selected_seeds"],
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


def _milestone_indices(manifest: list[dict]) -> tuple[int, ...]:
    if len(manifest) < 5:
        raise RuntimeError("轨迹没有五个互异 natural-work 状态")
    works = [float(item["normalized_work"]) for item in manifest]
    indices = [int(item["state_index"]) for item in manifest]
    if works != sorted(works) or indices != sorted(set(indices)):
        raise RuntimeError("natural-work manifest 顺序无效")
    terminal_work = works[-1]
    targets = [terminal_work * value for value in (0.25, 0.5, 0.75)]
    candidates = []
    for selected in itertools.combinations(range(1, len(manifest) - 1), 3):
        error = sum(abs(works[index] - target) for index, target in zip(
            selected, targets
        ))
        candidates.append((error, selected))
    if not candidates:
        raise RuntimeError("natural-work manifest 无三个 interior 状态")
    interior = min(candidates, key=lambda item: (item[0], item[1]))[1]
    return 0, *interior, len(manifest) - 1


def _audit_state_library(
    path: Path,
    library: dict,
    protocol: dict,
    input_audit: dict,
    runtime: dict,
) -> dict:
    expected_ids = [
        f"{dataset}__seed_{seed}__{group}"
        for dataset in protocol["dataset_order"]
        for seed in protocol["seeds"]
        for group in protocol["state_groups"]
    ]
    states = library.get("states", [])
    state_ids = [item.get("state_id") for item in states]
    raw_manifest = library.get("manifest")
    manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
    shard_provenance = manifest.get("source_seed_shard_sha256")
    expected_shard_keys = [str(seed) for seed in protocol["seeds"]]
    gates = {
        "format": library.get("state_library_format")
        == frozen.STATE_LIBRARY_FORMAT,
        "status": library.get("status") == "complete",
        "mode": library.get("mode") == protocol["mode"],
        "artifact_scope": library.get("artifact_scope") == "full",
        "selected_seeds": library.get("selected_seeds")
        == protocol["seeds"],
        "manifest_type": isinstance(raw_manifest, dict),
        "protocol": library.get("protocol") == protocol,
        "protocol_sha256": library.get("protocol_sha256")
        == frozen.protocol_sha256(protocol["mode"]),
        "input_audit": library.get("input_audit") == input_audit,
        "state_order": state_ids == expected_ids,
        "state_unique": len(state_ids) == len(set(state_ids)),
        "manifest_state_count": manifest.get("state_count")
        == len(expected_ids),
        "manifest_state_ids": manifest.get("state_ids_in_fixed_order")
        == expected_ids,
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
        == frozen.canonical_sha256(_library_scientific_payload(library)),
        "formal_flag": protocol["mode"] != "qualification"
        or library.get("formal_result_valid") is True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"状态库外层审计失败：{failed}")

    trajectory_index = {}
    for trajectory in library.get("trajectories", []):
        key = (trajectory.get("dataset"), trajectory.get("seed"))
        if key in trajectory_index:
            raise RuntimeError(f"状态库来源轨迹重复：{key}")
        trajectory_index[key] = trajectory
    expected_trajectories = [
        (dataset, seed)
        for dataset in protocol["dataset_order"]
        for seed in protocol["seeds"]
    ]
    if list(trajectory_index) != expected_trajectories:
        raise RuntimeError("状态库来源轨迹缺失、重复或乱序")

    state_index = {item["state_id"]: item for item in states}
    recomputed_states = 0
    for dataset_name, seed in expected_trajectories:
        trajectory = trajectory_index[(dataset_name, seed)]
        dataset, target, queries, schema = runtime[dataset_name]
        manifest = trajectory.get("natural_work_snapshot_manifest")
        if not isinstance(manifest, list):
            raise RuntimeError("来源轨迹缺少 natural-work manifest")
        chosen = _milestone_indices(manifest)
        selected_ids = [
            f"{dataset_name}__seed_{seed}__{group}"
            for group in protocol["state_groups"]
        ]
        if trajectory.get("selected_state_ids") != selected_ids:
            raise RuntimeError("来源轨迹 selected state IDs 漂移")
        if (
            trajectory.get("runtime_n_records")
            != dataset["runtime_n_records"]
            or trajectory.get("runtime_target_sha256")
            != frozen.canonical_sha256(target.tolist())
            or trajectory.get("recorded_natural_work_state_count")
            != len(manifest)
            or trajectory.get("initial_table_sha256")
            != manifest[0]["current_table_sha256"]
            or trajectory.get("terminal_table_sha256")
            != manifest[-1]["current_table_sha256"]
            or trajectory.get("terminal_normalized_work")
            != manifest[-1]["normalized_work"]
            or trajectory.get("termination_reason")
            != manifest[-1]["termination_reason"]
            or trajectory.get("primary_rng_endpoint_sha256")
            != manifest[-1]["primary_rng_state_sha256"]
        ):
            raise RuntimeError("来源轨迹 endpoint/manifest 身份漂移")

        terminal_work = float(manifest[-1]["normalized_work"])
        for group, selected_index, state_id in zip(
            protocol["state_groups"], chosen, selected_ids
        ):
            entry = state_index[state_id]
            snapshot = entry["snapshot"]
            selected_manifest = manifest[selected_index]
            fraction = {
                "initial": 0.0,
                "work_q25": 0.25,
                "work_q50": 0.5,
                "work_q75": 0.75,
                "terminal": 1.0,
            }[group]
            target_work = terminal_work * fraction
            if (
                entry["dataset"] != dataset_name
                or entry["seed"] != seed
                or entry["state_group"] != group
                or entry["source_snapshot_index"] != selected_index
                or entry["target_fraction"] != fraction
                or entry["target_normalized_work"] != target_work
                or entry["selection_absolute_work_error"]
                != abs(float(snapshot["normalized_work"]) - target_work)
            ):
                raise RuntimeError(f"{state_id} milestone 选择漂移")
            for key in (
                "state_index",
                "round",
                "phase",
                "completed_work_ticks",
                "cumulative_participating_rows",
                "normalized_work",
                "work_tick_completed",
                "termination_reason",
                "current_squared_loss",
                "current_normalized_l1",
                "current_table_sha256",
                "primary_rng_state_sha256",
                "direction_reference_scale",
            ):
                if snapshot[key] != selected_manifest[key]:
                    raise RuntimeError(f"{state_id} manifest.{key} 漂移")
            frame = pd.DataFrame(
                snapshot["table_records"],
                columns=snapshot["table_columns"],
            )
            if (
                snapshot["snapshot_format"] != "natural_work_current_v1"
                or snapshot["table_columns"] != schema.attribute_names()
                or len(frame) != dataset["runtime_n_records"]
                or hashlib.sha256(
                    frame.to_csv(index=False).encode("utf-8")
                ).hexdigest()
                != snapshot["current_table_sha256"]
            ):
                raise RuntimeError(f"{state_id} table hash/shape 失败")
            q, residual, _ = evaluate_vectorized(
                frame,
                queries,
                schema,
                target=target,
                n_records=dataset["runtime_n_records"],
                batch_size=256,
                device="numpy",
                want_fitness=True,
                verbose=False,
                residual_geometry="relative",
                residual_geometry_floor=8.0,
            )
            loss = float(compute_loss(target, q))
            normalized_l1 = float(
                np.mean(np.abs(target - q))
                / dataset["runtime_n_records"]
            )
            if (
                not np.array_equal(
                    np.asarray(q, dtype=float),
                    np.asarray(
                        snapshot["current_query_answers"], dtype=float
                    ),
                )
                or not np.allclose(
                    np.asarray(residual, dtype=float),
                    np.asarray(
                        snapshot["current_residual_signal"], dtype=float
                    ),
                    rtol=0.0,
                    atol=1e-15,
                )
                or not math.isclose(
                    loss,
                    float(snapshot["current_squared_loss"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    normalized_l1,
                    float(snapshot["current_normalized_l1"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(snapshot["direction_reference_scale"]),
                    float(trajectory["direction_reference_scale"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise RuntimeError(f"{state_id} q/residual/loss/L1/s0 失败")
            recomputed_states += 1
    return {
        "file_sha256": frozen.file_sha256(path),
        "scientific_sha256": library[
            "state_library_scientific_sha256"
        ],
        "state_count": recomputed_states,
        "gates": gates,
    }


def _gibbs_name(sweeps: int) -> str:
    return f"gibbs_tau_2_sweeps_{sweeps}"


def _joint_name() -> str:
    return "joint_tau_2"


def _weighted_kernel(
    rows: list[dict],
    name: str,
    width_group: str | None = None,
) -> dict:
    totals = {metric: 0.0 for metric in KERNEL_METRICS}
    count = 0
    active_blocks = 0
    for row in rows:
        source = row["probe"]["kernel_summary"]
        if width_group is not None:
            source = row["probe"]["kernel_summary_by_active_width"][
                width_group
            ]
        kernel = source[name]
        kernel_count = int(kernel["participating_active_rows"])
        count += kernel_count
        active_blocks += int(kernel["active_blocks"])
        for metric in KERNEL_METRICS:
            value = float(kernel[metric])
            if not math.isfinite(value):
                raise RuntimeError("kernel summary 包含非有限值")
            totals[metric] += value * kernel_count
    result = {
        "participating_active_rows": count,
        "active_blocks": active_blocks,
    }
    for metric in KERNEL_METRICS:
        result[metric] = totals[metric] / count if count else 0.0
    return result


def _mixing_group(
    rows: list[dict],
    sweeps: int,
    *,
    width_group: str | None = None,
    require_nonempty: bool,
) -> dict:
    baseline = _weighted_kernel(rows, _gibbs_name(0), width_group)
    candidate = _weighted_kernel(rows, _gibbs_name(sweeps), width_group)
    initial_gap = baseline["absolute_expected_direction_gap"]
    remaining_gap = candidate["absolute_expected_direction_gap"]
    recovery = (
        1.0 - remaining_gap / initial_gap if initial_gap > 0.0 else 1.0
    )
    nonempty = candidate["participating_active_rows"] > 0
    gated = require_nonempty or nonempty
    tvd_pass = candidate["tvd_to_joint"] <= frozen.TVD_THRESHOLD
    recovery_pass = recovery >= frozen.RECOVERY_THRESHOLD
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
        "passed": bool(
            (nonempty or not require_nonempty)
            and (not gated or (tvd_pass and recovery_pass))
        ),
    }


def _condition_digest(proposal_sha256: list[str]) -> str:
    digest = hashlib.sha256()
    for value in proposal_sha256:
        digest.update(b"proposal_sha256\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _recompute_dataset(
    reported: dict,
    dataset_name: str,
    dataset: dict,
    expected_state_ids: list[str],
    sweeps: int,
    protocol: dict,
    previous_tvd: dict[str, float],
    expected_conditions: dict[str, dict],
) -> tuple[bool, bool, dict[str, float], dict[str, dict]]:
    rows = reported.get("state_results")
    if not isinstance(rows, list) or [
        row.get("state_id") for row in rows
    ] != expected_state_ids:
        raise RuntimeError(
            f"{dataset_name}/s{sweeps} state 缺失、重复或乱序"
        )
    configs = {_gibbs_name(0), _gibbs_name(sweeps), _joint_name()}
    condition_rows = []
    next_conditions = dict(expected_conditions)
    conditions_equal = True
    for row in rows:
        probe_result = row.get("probe", {})
        if (
            row.get("dataset") != dataset_name
            or probe_result.get("state_sha256") != row.get("state_sha256")
            or probe_result.get("n_proposals")
            != dataset["proposals_per_state"]
            or probe_result.get("rho") != frozen.RHO
            or probe_result.get("eta") != frozen.ETA
            or probe_result.get("mu") != 0.0
            or probe_result.get("probe_alpha") != frozen.FIXED_ALPHA
            or probe_result.get("reference_scale_proposal_index") is not None
            or set(probe_result.get("kernel_summary", {})) != configs
            or set(probe_result.get("proposal_summary", {})) != configs
            or set(
                probe_result.get("kernel_summary_by_active_width", {})
            )
            != set(protocol["active_width_groups"])
        ):
            raise RuntimeError(f"{row.get('state_id')} probe 身份漂移")
        controls = probe_result.get("probe_controls", {})
        expected_controls = {
            "n_records": dataset["runtime_n_records"],
            "rho": frozen.RHO,
            "eta": frozen.ETA,
            "max_factor_order": dataset["max_factor_order"],
            "max_active_attributes": dataset["max_active_attributes"],
            "selection_scale_invariant": True,
            "selection_scale_invariant_min_spread": 1e-3,
            "residual_geometry": "relative",
            "residual_geometry_floor": 8.0,
        }
        _assert_same(
            controls,
            expected_controls,
            f"{row['state_id']}.probe_controls",
        )
        for summary in probe_result["proposal_summary"].values():
            if summary.get("n") != dataset["proposals_per_state"]:
                raise RuntimeError("independent comparison proposal 数漂移")
        identity = probe_result.get("shared_condition_identity", {})
        proposals = identity.get("proposal_sha256")
        if (
            identity.get("format") != "factor_gibbs_shared_condition_v1"
            or not isinstance(proposals, list)
            or len(proposals) != dataset["proposals_per_state"]
            or not all(_valid_sha256(value) for value in proposals)
            or identity.get("scientific_sha256")
            != _condition_digest(proposals)
        ):
            raise RuntimeError(f"{row['state_id']} shared condition 损坏")
        state_id = row["state_id"]
        condition_rows.append({
            "state_id": state_id,
            "proposal_sha256": proposals,
            "scientific_sha256": identity["scientific_sha256"],
        })
        if state_id in expected_conditions:
            conditions_equal &= identity == expected_conditions[state_id]
        else:
            next_conditions[state_id] = identity
        factors = probe_result.get("factor_diagnostics", {})
        if (
            int(factors.get("maximum_active_factor_order", -1))
            > dataset["max_factor_order"]
            or not math.isfinite(
                float(factors.get("exact_energy_max_error", math.inf))
            )
            or not math.isfinite(
                float(factors.get("tvd_snapshot_increase_max", math.inf))
            )
        ):
            raise RuntimeError(f"{row['state_id']} factor diagnostics 无效")
        if (
            factors.get("energy_atol") != frozen.ENERGY_ATOL
            or factors.get("energy_rtol") != frozen.ENERGY_RTOL
        ):
            raise RuntimeError(
                f"{row['state_id']} energy atol/rtol 与冻结协议不一致"
            )
        worst_case = factors.get("exact_energy_worst_case", {})
        worst_abs_diff = float(worst_case.get("abs_diff", math.inf))
        worst_scale = float(worst_case.get("scale", math.inf))
        recorded_ratio = float(
            factors.get("exact_energy_tolerance_ratio_max", math.inf)
        )
        recorded_relative = float(
            factors.get("exact_energy_max_relative_error", math.inf)
        )
        recomputed_ratio = frozen.energy_tolerance_ratio(
            worst_abs_diff, worst_scale
        )
        if recomputed_ratio != recorded_ratio:
            raise RuntimeError(
                f"{row['state_id']} energy tolerance ratio 与 worst-case "
                "分量重算不一致"
            )
        if (
            not math.isfinite(recorded_relative)
            or recorded_relative < 0.0
            or worst_abs_diff
            > float(factors["exact_energy_max_error"])
            or (
                worst_scale > 0.0
                and worst_abs_diff / worst_scale > recorded_relative
            )
            or (worst_scale == 0.0 and worst_abs_diff != 0.0)
        ):
            raise RuntimeError(
                f"{row['state_id']} energy 恒等诊断分量不自洽"
            )
        for group in protocol["active_width_groups"]:
            if set(
                probe_result["kernel_summary_by_active_width"][group]
            ) != configs:
                raise RuntimeError("active width kernel config 漂移")

    global_group = _mixing_group(rows, sweeps, require_nonempty=True)
    stage_groups = {
        group: _mixing_group(
            [row for row in rows if row["state_group"] == group],
            sweeps,
            require_nonempty=True,
        )
        for group in protocol["required_stage_groups"]
    }
    width_groups = {
        group: _mixing_group(
            rows,
            sweeps,
            width_group=group,
            require_nonempty=False,
        )
        for group in protocol["active_width_groups"]
    }
    expected_mixing = {
        "global": global_group,
        "stage_groups": stage_groups,
        "active_width_groups": width_groups,
    }
    _assert_same(
        reported.get("mixing"),
        expected_mixing,
        f"{dataset_name}.mixing",
    )

    probabilities = [
        row["probe"]["probability_diagnostics_by_temperature"]["tau_2"]
        for row in rows
    ]
    logits = [
        row["probe"]["conditional_logit_diagnostics"]["tau_2"]
        for row in rows
    ]
    samplers = [
        row["probe"]["production_sampler_diagnostics"]["tau_2"]
        for row in rows
    ]
    factors = [row["probe"]["factor_diagnostics"] for row in rows]
    probability_gate = all(
        item["distribution_count"] == 0
        or (
            item["all_finite"]
            and item["all_nonnegative"]
            and item["probability_sum_max_error"]
            <= frozen.PROBABILITY_SUM_TOLERANCE
        )
        for item in probabilities
    )
    clip_hits = sum(int(item["clip_hit_count"]) for item in logits)
    comparisons = sum(int(item["comparison_count"]) for item in samplers)
    mismatches = sum(int(item["mismatch_count"]) for item in samplers)
    energy_error = max(
        float(item["exact_energy_max_error"]) for item in factors
    )
    energy_relative_error = max(
        float(item["exact_energy_max_relative_error"])
        for item in factors
    )
    energy_ratio = max(
        frozen.energy_tolerance_ratio(
            float(item["exact_energy_worst_case"]["abs_diff"]),
            float(item["exact_energy_worst_case"]["scale"]),
        )
        for item in factors
    )
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
        item["tvd_snapshot_increase_max"]
        <= frozen.TVD_MONOTONIC_TOLERANCE
        for item in factors
    )
    validity = {
        "all_required_stage_groups_nonempty": all(
            value["nonempty"] for value in stage_groups.values()
        ),
        "probabilities_valid": probability_gate,
        "exact_factor_energy": energy_ratio <= 1.0,
        "production_tape_replay": comparisons > 0 and mismatches == 0,
        "shared_conditions_exact": conditions_equal,
        "tvd_monotonic_across_candidates": monotonic,
        "tvd_monotonic_within_probe": within_probe_monotonic,
        "sweeps_within_hard_cap": sweeps <= 32,
    }
    qualification = {
        "global_mixing": global_group["passed"],
        "all_stage_groups_mixing": all(
            value["passed"] for value in stage_groups.values()
        ),
        "all_nonempty_width_groups_mixing": all(
            value["passed"] for value in width_groups.values()
        ),
        "zero_conditional_clip_hits": clip_hits == 0,
    }
    valid = all(validity.values())
    passed = valid and all(qualification.values())
    numerical = {
        "conditional_clip_hit_count": clip_hits,
        "probability_sum_max_error": max(
            float(item["probability_sum_max_error"])
            for item in probabilities
        ),
        "exact_energy_max_error": energy_error,
        "exact_energy_max_relative_error": energy_relative_error,
        "exact_energy_tolerance_ratio_max": energy_ratio,
        "production_sampler_comparison_count": comparisons,
        "production_sampler_mismatch_count": mismatches,
    }
    for key, expected in (
        ("validity_gates", validity),
        ("qualification_gates", qualification),
        ("numerical_diagnostics", numerical),
        ("shared_condition_rows", condition_rows),
    ):
        _assert_same(
            reported.get(key), expected, f"{dataset_name}.{key}"
        )
    condition_sha = frozen.canonical_sha256(condition_rows)
    if reported.get("shared_condition_scientific_sha256") != condition_sha:
        raise RuntimeError("shared_condition_scientific_sha256 漂移")
    if reported.get("valid") is not valid or reported.get("passed") is not passed:
        raise RuntimeError("dataset valid/passed 漂移")

    cost = reported.get("cost", {})
    expected_cost = {
        "participating_active_rows": sum(
            int(item["active_rows"]) for item in factors
        ),
        "active_attribute_updates_per_sweep": global_group[
            "active_attribute_updates_per_sweep"
        ],
        "gibbs_microsteps": sum(
            int(item["microsteps"]) for item in samplers
        ),
        "factor_count": sum(
            int(item["total_factor_count"]) for item in factors
        ),
        "factor_table_entries": sum(
            int(item["total_factor_table_entries"]) for item in factors
        ),
    }
    for key, expected in expected_cost.items():
        _assert_same(cost.get(key), expected, f"{dataset_name}.cost.{key}")
    timing_keys = {
        "factor_build_elapsed_sec_diagnostic_only",
        "exact_propagation_elapsed_sec_diagnostic_only",
        "production_sample_elapsed_sec_diagnostic_only",
    }
    if set(cost) != set(expected_cost) | timing_keys or any(
        not math.isfinite(float(cost[key])) or float(cost[key]) < 0.0
        for key in timing_keys
    ):
        raise RuntimeError("cost wall-time diagnostics 无效")
    return valid, passed, current_tvd, next_conditions


def _sequence_result(attempts: list[dict]) -> tuple[str, int | None]:
    observed = [item.get("sweeps") for item in attempts]
    if observed != list(frozen.CANDIDATE_SWEEPS[: len(attempts)]):
        raise RuntimeError("候选 sweep 缺失、乱序或跳级")
    invalid = [
        item for item in attempts if item.get("valid") is not True
    ]
    if invalid:
        if len(invalid) != 1 or attempts[-1] is not invalid[0]:
            raise RuntimeError("结构性 invalid 后仍执行更高 sweep")
        return "invalid_or_incomplete", None
    passing = [item for item in attempts if item.get("passed") is True]
    if passing:
        if len(passing) != 1 or attempts[-1] is not passing[0]:
            raise RuntimeError("首次双数据通过后仍执行更高 sweep")
        sweep = passing[0]["sweeps"]
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


def audit_stage4_mixing(
    report_path: str | Path,
    state_library_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, dict]:
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    library_file = Path(state_library_path).resolve()
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise FileExistsError(f"审计输出已存在，不覆盖：{output_file}")
    report = _load_json_strict(report_file)
    library = _load_json_strict(library_file)
    mode = report.get("mode")
    protocol = frozen.stage4_protocol(mode)
    if (
        report.get("report_format") != frozen.REPORT_FORMAT
        or report.get("status") != "complete"
        or report.get("protocol") != protocol
        or report.get("protocol_sha256") != frozen.protocol_sha256(mode)
    ):
        raise RuntimeError("Stage 4 report 外层协议身份失败")
    input_audit, runtime = _input_audit(protocol)
    library_audit = _audit_state_library(
        library_file, library, protocol, input_audit, runtime
    )
    binding = report.get("state_library_binding", {})
    if (
        Path(binding.get("path", "")).resolve() != library_file
        or binding.get("file_sha256") != library_audit["file_sha256"]
        or binding.get("scientific_sha256")
        != library_audit["scientific_sha256"]
        or not binding.get("binding_gates")
        or not all(binding["binding_gates"].values())
    ):
        raise RuntimeError("报告与状态库绑定失败")

    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("报告没有完整 attempts")
    previous_tvd = {name: {} for name in protocol["dataset_order"]}
    expected_conditions = {name: {} for name in protocol["dataset_order"]}
    state_ids = [state["state_id"] for state in library["states"]]
    for attempt in attempts:
        sweeps = attempt.get("sweeps")
        datasets = attempt.get("datasets")
        if (
            sweeps not in frozen.CANDIDATE_SWEEPS
            or not isinstance(datasets, dict)
            or list(datasets) != protocol["dataset_order"]
        ):
            raise RuntimeError("attempt sweep 或双数据顺序无效")
        recomputed_valid = []
        recomputed_passed = []
        for dataset_name in protocol["dataset_order"]:
            dataset = protocol["datasets"][dataset_name]
            expected = [
                state_id
                for state_id in state_ids
                if state_id.startswith(f"{dataset_name}__")
            ]
            valid, passed, tvd, conditions = _recompute_dataset(
                datasets[dataset_name],
                dataset_name,
                dataset,
                expected,
                sweeps,
                protocol,
                previous_tvd[dataset_name],
                expected_conditions[dataset_name],
            )
            previous_tvd[dataset_name] = tvd
            expected_conditions[dataset_name] = conditions
            recomputed_valid.append(valid)
            recomputed_passed.append(passed)
        expected_valid = all(recomputed_valid)
        expected_passed = all(recomputed_passed)
        if (
            attempt.get("valid") is not expected_valid
            or attempt.get("passed") is not expected_passed
        ):
            raise RuntimeError("attempt 双数据 valid/passed 聚合漂移")
    result, selected = _sequence_result(attempts)
    if (
        report.get("attempted_sweeps")
        != [item["sweeps"] for item in attempts]
        or report.get("result") != result
        or report.get("selected_minimal_sufficient_sweeps") != selected
        or result not in protocol["allowed_results"]
    ):
        raise RuntimeError("Stage 4 全局停止规则或结果标签漂移")

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
    execution_sha = frozen.canonical_sha256(
        _without_diagnostics(scientific_payload)
    )
    if report.get("execution_scientific_sha256") != execution_sha:
        raise RuntimeError("execution scientific SHA 漂移")
    expected_formal = bool(
        mode == "qualification"
        and result != "invalid_or_incomplete"
        and report.get("git", {}).get("dirty") is False
        and library.get("formal_result_valid") is True
        and library.get("git", {}).get("dirty") is False
        and report.get("git", {}).get("commit")
        == library.get("git", {}).get("commit")
    )
    if mode == "qualification" and not expected_formal:
        raise RuntimeError(
            "qualification 状态库与报告必须来自同一个 clean commit"
        )
    if report.get("formal_result_valid") is not expected_formal:
        raise RuntimeError("formal_result_valid 漂移")

    audit = {
        "audit_format": frozen.AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "mode": mode,
        "formal_result_valid": expected_formal,
        "protocol_sha256": frozen.protocol_sha256(mode),
        "report_path": str(report_file),
        "report_sha256": frozen.file_sha256(report_file),
        "state_library_path": str(library_file),
        "state_library_sha256": library_audit["file_sha256"],
        "state_library_scientific_sha256": library_audit[
            "scientific_sha256"
        ],
        "execution_scientific_sha256": execution_sha,
        "recomputed": {
            "state_count": library_audit["state_count"],
            "attempted_sweeps": [item["sweeps"] for item in attempts],
            "result": result,
            "selected_minimal_sufficient_sweeps": selected,
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            audit,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(f"Stage 4 独立审计通过：{output_file}", flush=True)
    return output_file, audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    audit_stage4_mixing(
        args.report, args.state_library, args.output
    )


if __name__ == "__main__":
    main()
