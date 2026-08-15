"""运行 Issue #49 协议 v2 的 Stage T、共同状态库与 A0/A1。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__:
    from scripts import build_issue49_unfiltered_state_library as builder
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue49_stage_t_a_protocol as frozen_protocol
    from scripts import probe_factorized_gibbs_mixing as probe
else:
    import build_issue49_unfiltered_state_library as builder
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue49_stage_t_a_protocol as frozen_protocol
    import probe_factorized_gibbs_mixing as probe

from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


REPORT_FORMAT = "issue49_stage_t_a_report_v2"
TEMPERATURES = list(frozen_protocol.TEMPERATURES)
EXPECTED_INPUT_SHA256 = frozen_protocol.EXPECTED_INPUT_SHA256
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
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _required_state_families(temperatures=TEMPERATURES):
    return (
        "initial",
        *(f"mid_source_{_tau_key(tau)}" for tau in temperatures),
        *(f"late_source_{_tau_key(tau)}" for tau in temperatures),
    )


REQUIRED_STATE_FAMILIES = _required_state_families()
REQUIRED_MIXING_GROUPS = ("global", *REQUIRED_STATE_FAMILIES)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_strict(path):
    def reject_constant(value):
        raise ValueError(f"JSON 包含非标准数值常量：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _write_json_atomic(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _git_identity():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    ).stdout.strip()
    return {"commit": commit, "worktree_clean": status == ""}


def _protocol(mode):
    return frozen_protocol.protocol(mode)


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
        "probe_mutation_disabled": protocol["probe_mu"] == 0.0,
        "probe_logit_clip_matches": (
            probe.GIBBS_LOGIT_CLIP == protocol["logit_clip"]
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"冻结协议与实现常量不一致：{failed}")
    return gates


def _load_inputs():
    paths = {
        "schema": REPOSITORY_ROOT / trajectory.SCHEMA_PATH,
        "queries": REPOSITORY_ROOT / trajectory.QUERY_PATH,
        "marginals": REPOSITORY_ROOT / trajectory.MARGINALS_PATH,
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"test_300x10 输入哈希与预注册不一致：{hashes}"
        )
    raw_queries = _load_json_strict(paths["queries"])
    raw_marginals = _load_json_strict(paths["marginals"])
    schema = load_schema(str(paths["schema"]))
    queries = load_queries(str(paths["queries"]))
    target = np.asarray(
        [query["result"] for query in queries], dtype=float
    )
    marginals = load_marginals(str(paths["marginals"]))
    if (
        raw_queries.get("record_count") != trajectory.N_RECORDS
        or raw_queries.get("query_count") != 50
        or raw_marginals.get("n_records") != trajectory.N_RECORDS
        or len(queries) != 50
        or target.shape != (50,)
        or not np.all(np.isfinite(target))
        or len(schema.attribute_names()) != 10
        or set(schema.attribute_names())
        != set(raw_marginals.get("attributes", {}))
    ):
        raise RuntimeError("test_300x10 输入结构与预注册不一致")
    return target, queries, schema, marginals, hashes


def _run_stage_t(target, queries, schema, marginals, protocol):
    rows = []
    state_seed_set = set(protocol["state_library_seeds"])
    for seed in protocol["stage_t_seeds"]:
        for temperature in protocol["source_temperatures"]:
            rows.append(trajectory._run_one(
                target,
                queries,
                schema,
                marginals,
                seed=seed,
                rounds=protocol["rounds"],
                temperature=temperature,
                sweeps=0,
                device=protocol["device"],
                snapshot_rounds=(
                    protocol["snapshot_rounds"]
                    if seed in state_seed_set else None
                ),
            ))
    return rows


def _stage_t_identity_gates(rows, protocol):
    expected_pairs = {
        (seed, temperature)
        for seed in protocol["stage_t_seeds"]
        for temperature in protocol["source_temperatures"]
    }
    actual_pairs = {
        (row.get("seed"), row.get("temperature")) for row in rows
    }
    by_seed = {
        seed: [row for row in rows if row["seed"] == seed]
        for seed in protocol["stage_t_seeds"]
    }
    state_seeds = set(protocol["state_library_seeds"])
    gates = {
        "trajectory_grid_complete": (
            len(rows) == len(expected_pairs)
            and actual_pairs == expected_pairs
        ),
        "independent_unfiltered_kernel": all(
            row["sweeps"] == 0 and row["name"] == "independent"
            for row in rows
        ),
        "all_rounds_complete": all(
            row["rounds_run"] == protocol["rounds"] for row in rows
        ),
        "all_current_loss_histories_complete": all(
            len(row["current_loss_after_round_history"])
            == protocol["rounds"]
            for row in rows
        ),
        "state_snapshot_scope_exact": all(
            (
                row.get("snapshot_rounds")
                == protocol["snapshot_rounds"]
                and len(row.get("state_snapshots", [])) == 3
            ) if row["seed"] in state_seeds else (
                "state_snapshots" not in row
                and "snapshot_rounds" not in row
            )
            for row in rows
        ),
        "initial_state_aligned_within_seed": all(
            len({row["initial_csv_sha256"] for row in seed_rows}) == 1
            and len({row["initial_loss"] for row in seed_rows}) == 1
            for seed_rows in by_seed.values()
        ),
        "primary_rng_endpoint_aligned_within_seed": all(
            len({
                row["primary_rng_state_sha256"] for row in seed_rows
            }) == 1
            for seed_rows in by_seed.values()
        ),
        "direction_reference_scale_aligned_within_seed": all(
            len({row["direction_reference_scale"] for row in seed_rows})
            == 1
            for seed_rows in by_seed.values()
        ),
        "direction_reference_scale_positive_finite": all(
            row["direction_reference_scale"] is not None
            and np.isfinite(row["direction_reference_scale"])
            and row["direction_reference_scale"] > 0.0
            and row["direction_reference_scale_round"] == 0
            for row in rows
        ),
        "trajectory_diagnostics_complete": all(
            row["independent_direction_diagnostics"]["condition_count"] > 0
            and len(row["independent_direction_diagnostics"][
                "clip_hit_conditions"
            ]) == row["independent_direction_diagnostics"][
                "clip_hit_count"
            ]
            for row in rows
        ),
    }
    gates["all_identity_gates_passed"] = all(gates.values())
    if not gates["all_identity_gates_passed"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Stage T 身份门禁失败：{failed}")
    return gates


def _numeric_summary(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _aggregate_stage_t_logit(rows):
    diagnostics = [row["independent_direction_diagnostics"] for row in rows]
    count = int(sum(row["condition_count"] for row in diagnostics))
    hits = int(sum(row["clip_hit_count"] for row in diagnostics))
    maximum_row = max(
        rows,
        key=lambda row: row["independent_direction_diagnostics"][
            "raw_logit_abs_max"
        ],
    )
    maximum_diagnostic = maximum_row["independent_direction_diagnostics"]
    hit_conditions = []
    for run in rows:
        for condition in run["independent_direction_diagnostics"][
            "clip_hit_conditions"
        ]:
            hit_conditions.append({
                "seed": run["seed"],
                "temperature": run["temperature"],
                **condition,
            })
    negative_count = int(sum(
        row["negative_direction_count"] for row in diagnostics
    ))
    positive_count = int(sum(
        row["positive_direction_count"] for row in diagnostics
    ))
    nonempty = [row for row in diagnostics if row["condition_count"]]
    return {
        "condition_count": count,
        "raw_logit_min": float(min(
            row["raw_logit_min"] for row in nonempty
        )),
        "raw_logit_max": float(max(
            row["raw_logit_max"] for row in nonempty
        )),
        "raw_logit_abs_max": float(
            maximum_diagnostic["raw_logit_abs_max"]
        ),
        "raw_logit_abs_max_condition": {
            "seed": maximum_row["seed"],
            "temperature": maximum_row["temperature"],
            **maximum_diagnostic["raw_logit_abs_max_condition"],
        },
        "logit_clip": float(probe.GIBBS_LOGIT_CLIP),
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count),
        "clip_hit_conditions": hit_conditions,
        "raw_logit_strictly_inside_clip": all(
            row["raw_logit_strictly_inside_clip"] for row in diagnostics
        ),
        "conditional_probability_min": float(min(
            row["conditional_probability_min"] for row in nonempty
        )),
        "conditional_probability_max": float(max(
            row["conditional_probability_max"] for row in nonempty
        )),
        "minimum_binary_outcome_probability": float(min(
            row["minimum_binary_outcome_probability"]
            for row in nonempty
        )),
        "conditional_entropy_mean": float(sum(
            row["conditional_entropy_mean"] * row["condition_count"]
            for row in nonempty
        ) / count),
        "negative_direction_count": negative_count,
        "negative_direction_copy_probability": (
            float(sum(
                row["negative_direction_copy_probability"]
                * row["negative_direction_count"]
                for row in diagnostics
                if row["negative_direction_count"]
            ) / negative_count) if negative_count else None
        ),
        "positive_direction_count": positive_count,
        "positive_direction_copy_probability": (
            float(sum(
                row["positive_direction_copy_probability"]
                * row["positive_direction_count"]
                for row in diagnostics
                if row["positive_direction_count"]
            ) / positive_count) if positive_count else None
        ),
        "all_finite": all(row["all_finite"] for row in diagnostics),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"]
            for row in diagnostics
        ),
    }


def _aggregate_stage_t(rows, protocol):
    by_temperature = {}
    for temperature in protocol["source_temperatures"]:
        selected = [row for row in rows if row["temperature"] == temperature]
        by_temperature[_tau_key(temperature)] = {
            "trajectory_count": len(selected),
            "seeds": [row["seed"] for row in selected],
            "all_rounds_complete": all(
                row["rounds_run"] == protocol["rounds"]
                for row in selected
            ),
            "final_current_loss": _numeric_summary([
                row["final_loss"] for row in selected
            ]),
            "late_window_current_loss": _numeric_summary([
                row["late_window_current_loss_mean"] for row in selected
            ]),
            "current_loss_auc": _numeric_summary([
                row["current_loss_auc"] for row in selected
            ]),
            "positive_gain_rate": _numeric_summary([
                row["positive_gain_rate"] for row in selected
            ]),
            "negative_gain_rate": _numeric_summary([
                row["negative_gain_rate"] for row in selected
            ]),
            "mean_positive_gain": _numeric_summary([
                row["mean_positive_gain"] for row in selected
            ]),
            "mean_negative_gain": _numeric_summary([
                row["mean_negative_gain"] for row in selected
            ]),
            "mean_changed_cells": _numeric_summary([
                row["mean_changed_cells"] for row in selected
            ]),
            "independent_direction_diagnostics": (
                _aggregate_stage_t_logit(selected)
            ),
        }
    return {
        "trajectory_count": len(rows),
        "expected_trajectory_count": (
            len(protocol["stage_t_seeds"])
            * len(protocol["source_temperatures"])
        ),
        "by_temperature": by_temperature,
    }


def _validate_reloaded_library(library, protocol, input_hashes):
    required = {
        "state_library_format": builder.STATE_LIBRARY_FORMAT,
        "dataset": "test_300x10",
        "source_kernel": "independent_directional_unfiltered",
        "source_temperatures": protocol["source_temperatures"],
        "source_sweeps": 0,
        "rounds": protocol["rounds"],
        "snapshot_rounds": protocol["snapshot_rounds"],
        "seeds": protocol["state_library_seeds"],
        "input_sha256": input_hashes,
    }
    for key, expected in required.items():
        if library.get(key) != expected:
            raise RuntimeError(
                f"重载状态库 {key}={library.get(key)!r}，期望 {expected!r}"
            )
    states = library.get("states")
    expected_count = (
        len(protocol["state_library_seeds"])
        * (1 + 2 * len(protocol["source_temperatures"]))
    )
    if (
        not isinstance(states, list)
        or len(states) != expected_count
        or len(states) != library.get("expected_state_count")
        or len(states) != library.get("state_count")
        or not library.get("all_gates_passed")
        or not all(library.get("gates", {}).values())
    ):
        raise RuntimeError("重载状态库数量或门禁无效")
    state_ids = [state.get("state_id") for state in states]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("重载状态库包含重复 state_id")
    expected_family_counts = {
        family: len(protocol["state_library_seeds"])
        for family in REQUIRED_STATE_FAMILIES
    }
    actual_family_counts = {
        family: sum(
            state.get("state_family") == family for state in states
        )
        for family in REQUIRED_STATE_FAMILIES
    }
    if actual_family_counts != expected_family_counts:
        raise RuntimeError(
            "重载状态库十一个状态族不完整："
            f"{actual_family_counts}"
        )


def _probe_library(
    library,
    target,
    queries,
    schema,
    protocol,
    *,
    temperatures,
    sweeps,
):
    rows = []
    for state_index, entry in enumerate(library["states"]):
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
            raise RuntimeError(
                f"状态库 wrapper 与快照身份不一致：{entry['state_id']}"
            )
        result = probe._probe_state(
            state,
            target,
            queries,
            schema,
            seed=entry["seed"],
            state_index=state_index,
            state_rounds=entry["state_round"],
            temperatures=temperatures,
            sweeps=sweeps,
            proposals=protocol["proposals_per_state"],
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
            raise RuntimeError(
                f"探针未保持外部状态控制：{entry['state_id']}"
            )
        rows.append({
            "state_id": entry["state_id"],
            "seed": entry["seed"],
            "state_round": entry["state_round"],
            "state_family": entry["state_family"],
            "source_temperature": entry["source_temperature"],
            "state_sha256": snapshot["state_sha256"],
            "probe": result,
        })
    return rows


def _weighted_kernel(rows, config):
    summaries = [row["probe"]["kernel_summary"][config] for row in rows]
    total_rows = int(sum(
        summary["participating_active_rows"] for summary in summaries
    ))
    active_blocks = int(sum(
        summary["active_blocks"] for summary in summaries
    ))
    result = {
        "participating_active_rows": total_rows,
        "active_blocks": active_blocks,
    }
    for metric in KERNEL_METRICS:
        result[metric] = (
            float(sum(
                summary[metric] * summary["participating_active_rows"]
                for summary in summaries
            ) / total_rows)
            if total_rows else None
        )
    return result


def _aggregate_logit(rows, temperature):
    key = _tau_key(temperature)
    values = [
        row["probe"]["conditional_logit_diagnostics"][key]
        for row in rows
    ]
    count = int(sum(value["condition_count"] for value in values))
    hits = int(sum(value["clip_hit_count"] for value in values))
    nonempty = [value for value in values if value["condition_count"]]
    maximum_row = max(
        rows,
        key=lambda row: row["probe"]["conditional_logit_diagnostics"][
            key
        ]["raw_logit_abs_max"],
    )
    maximum = maximum_row["probe"]["conditional_logit_diagnostics"][key]
    hit_conditions = []
    for state_row, value in zip(rows, values):
        for condition in value["clip_hit_conditions"]:
            hit_conditions.append({
                "state_id": state_row["state_id"],
                "seed": state_row["seed"],
                "state_round": state_row["state_round"],
                "state_family": state_row["state_family"],
                **condition,
            })
    return {
        "condition_count": count,
        "raw_logit_min": (
            float(min(value["raw_logit_min"] for value in nonempty))
            if nonempty else None
        ),
        "raw_logit_max": (
            float(max(value["raw_logit_max"] for value in nonempty))
            if nonempty else None
        ),
        "raw_logit_abs_max": float(maximum["raw_logit_abs_max"]),
        "raw_logit_abs_max_condition": (
            {
                "state_id": maximum_row["state_id"],
                "seed": maximum_row["seed"],
                "state_round": maximum_row["state_round"],
                "state_family": maximum_row["state_family"],
                **maximum["raw_logit_abs_max_condition"],
            } if maximum["raw_logit_abs_max_condition"] is not None else None
        ),
        "logit_clip": float(probe.GIBBS_LOGIT_CLIP),
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "clip_hit_conditions": hit_conditions,
        "raw_logit_strictly_inside_clip": all(
            value["raw_logit_strictly_inside_clip"] for value in values
        ),
        "conditional_probability_min": (
            float(min(
                value["conditional_probability_min"]
                for value in nonempty
            )) if nonempty else None
        ),
        "conditional_probability_max": (
            float(max(
                value["conditional_probability_max"]
                for value in nonempty
            )) if nonempty else None
        ),
        "minimum_binary_outcome_probability": (
            float(min(
                value["minimum_binary_outcome_probability"]
                for value in nonempty
            )) if nonempty else None
        ),
        "uniform_condition_entropy_mean": (
            float(sum(
                value["uniform_condition_entropy_mean"]
                * value["condition_count"]
                for value in nonempty
            ) / count) if count else None
        ),
        "uniform_condition_entropy_maximum": float(np.log(2.0)),
        "all_conditionals_bidirectional": all(
            value["all_conditionals_bidirectional"] for value in values
        ),
    }


def _aggregate_factor_diagnostics(rows, temperatures):
    diagnostics = [row["probe"]["factor_diagnostics"] for row in rows]
    return {
        "active_rows": int(sum(row["active_rows"] for row in diagnostics)),
        "exact_energy_max_error": float(max(
            row["exact_energy_max_error"] for row in diagnostics
        )),
        "one_hot_direction_max_error": float(max(
            row["one_hot_direction_max_error"] for row in diagnostics
        )),
        "maximum_active_factor_order": int(max(
            row["maximum_active_factor_order"] for row in diagnostics
        )),
        "tvd_snapshot_increase_max": float(max(
            row["tvd_snapshot_increase_max"] for row in diagnostics
        )),
        "tvd_snapshot_increase_max_by_temperature": {
            _tau_key(temperature): float(max(
                row["tvd_snapshot_increase_max_by_temperature"][
                    _tau_key(temperature)
                ]
                for row in diagnostics
            ))
            for temperature in temperatures
        },
        "factor_build_elapsed_sec": float(sum(
            row["factor_build_elapsed_sec"] for row in diagnostics
        )),
        "exact_finite_state_propagation_elapsed_sec": float(sum(
            row["exact_finite_state_propagation_elapsed_sec"]
            for row in diagnostics
        )),
    }


def _aggregate_probability_diagnostics(rows, temperature=None):
    if temperature is None:
        diagnostics = [
            row["probe"]["probability_diagnostics"] for row in rows
        ]
    else:
        key = _tau_key(temperature)
        diagnostics = [
            row["probe"]["probability_diagnostics_by_temperature"][key]
            for row in rows
        ]
    minima = [
        row["minimum_probability"] for row in diagnostics
        if row["minimum_probability"] is not None
    ]
    maxima = [
        row["maximum_probability"] for row in diagnostics
        if row["maximum_probability"] is not None
    ]
    return {
        "distribution_count": int(sum(
            row["distribution_count"] for row in diagnostics
        )),
        "all_finite": all(row["all_finite"] for row in diagnostics),
        "all_nonnegative": all(
            row["all_nonnegative"] for row in diagnostics
        ),
        "probability_sum_max_error": float(max(
            row["probability_sum_max_error"] for row in diagnostics
        )),
        "minimum_probability": float(min(minima)) if minima else None,
        "maximum_probability": float(max(maxima)) if maxima else None,
    }


def _aggregate_production_sampler(rows, temperature):
    key = _tau_key(temperature)
    diagnostics = [
        row["probe"]["production_sampler_diagnostics"][key]
        for row in rows
    ]
    comparisons = int(sum(row["comparison_count"] for row in diagnostics))
    mismatches = int(sum(row["mismatch_count"] for row in diagnostics))
    return {
        "comparison_count": comparisons,
        "mismatch_count": mismatches,
        "all_exact_tape_replays_match": bool(
            comparisons > 0 and mismatches == 0
        ),
        "microsteps": int(sum(row["microsteps"] for row in diagnostics)),
        "production_sampler_elapsed_sec": float(sum(
            row["production_sampler_elapsed_sec"] for row in diagnostics
        )),
        "exact_tape_replay_elapsed_sec": float(sum(
            row["exact_tape_replay_elapsed_sec"] for row in diagnostics
        )),
    }


def _aggregate_group(name, rows, protocol, temperatures):
    configs = probe._config_names(temperatures, protocol["sweeps"])
    kernel = {config: _weighted_kernel(rows, config) for config in configs}
    proposal_rows = {
        config: [
            proposal_row
            for row in rows
            for proposal_row in row["probe"]["proposal_rows"][config]
        ]
        for config in configs
    }
    proposal_summary = {
        config: probe._summarize_proposals(values)
        for config, values in proposal_rows.items()
    }
    recovery = {}
    for temperature in temperatures:
        baseline = probe._gibbs_name(temperature, 0)
        initial_gap = kernel[baseline]["absolute_expected_direction_gap"]
        for sweep in protocol["sweeps"]:
            config = probe._gibbs_name(temperature, sweep)
            remaining = kernel[config]["absolute_expected_direction_gap"]
            recovery[config] = (
                1.0 - remaining / initial_gap
                if initial_gap is not None and initial_gap > 0.0
                else (1.0 if initial_gap == 0.0 else None)
            )
    return {
        "group": name,
        "state_count": len(rows),
        "state_ids": [row["state_id"] for row in rows],
        "factor_diagnostics": _aggregate_factor_diagnostics(
            rows, temperatures
        ),
        "probability_diagnostics": _aggregate_probability_diagnostics(rows),
        "probability_diagnostics_by_temperature": {
            _tau_key(temperature): _aggregate_probability_diagnostics(
                rows, temperature
            )
            for temperature in temperatures
        },
        "conditional_logit_diagnostics": {
            _tau_key(temperature): _aggregate_logit(rows, temperature)
            for temperature in temperatures
        },
        "production_sampler_diagnostics": {
            _tau_key(temperature): _aggregate_production_sampler(
                rows, temperature
            )
            for temperature in temperatures
        },
        "kernel_summary": kernel,
        "expected_direction_gap_recovery": recovery,
        "proposal_summary": proposal_summary,
    }


def _aggregate_all(state_rows, protocol, temperatures):
    groups = {"global": list(state_rows)}
    for family in REQUIRED_STATE_FAMILIES:
        groups[family] = [
            row for row in state_rows if row["state_family"] == family
        ]
    groups.update({
        "phase_initial": [
            row for row in state_rows if row["state_family"] == "initial"
        ],
        "phase_mid": [
            row for row in state_rows
            if row["state_family"].startswith("mid_source_")
        ],
        "phase_late": [
            row for row in state_rows
            if row["state_family"].startswith("late_source_")
        ],
    })
    for temperature in protocol["source_temperatures"]:
        key = _tau_key(temperature)
        groups[f"source_{key}"] = [
            row for row in state_rows
            if row["state_family"].endswith(f"source_{key}")
        ]
    if any(not values for values in groups.values()):
        empty = [name for name, values in groups.items() if not values]
        raise RuntimeError(f"状态汇总组为空：{empty}")
    return {
        name: _aggregate_group(name, values, protocol, temperatures)
        for name, values in groups.items()
    }


def _all_numeric_finite(value):
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return True
    return value is None or isinstance(value, str)


def _common_semantic_gates(
    library,
    stage_t_rows,
    stage_t_identity_gates,
    a0_rows,
    protocol,
    input_hashes,
):
    factor = _aggregate_factor_diagnostics(
        a0_rows, protocol["evaluation_temperatures"]
    )
    probabilities = _aggregate_probability_diagnostics(a0_rows)
    first_baseline = probe._gibbs_name(
        protocol["evaluation_temperatures"][0], 0
    )
    gates = {
        "input_hashes_match_preregistration": (
            input_hashes == EXPECTED_INPUT_SHA256
        ),
        "stage_t_identity_passed": stage_t_identity_gates[
            "all_identity_gates_passed"
        ],
        "state_library_passed": bool(
            library.get("all_gates_passed")
            and all(library.get("gates", {}).values())
        ),
        "state_count_complete": (
            len(a0_rows) == library["expected_state_count"]
        ),
        "proposal_counts_complete": all(
            row["probe"]["n_proposals"]
            == protocol["proposals_per_state"]
            for row in a0_rows
        ),
        "raw_proposal_rows_complete": all(
            len(proposal_rows) == protocol["proposals_per_state"]
            for row in a0_rows
            for proposal_rows in row["probe"]["proposal_rows"].values()
        ),
        "all_required_families_have_active_rows": all(
            sum(
                row["probe"]["kernel_summary"][first_baseline][
                    "participating_active_rows"
                ]
                for row in a0_rows
                if row["state_family"] == family
            ) > 0
            for family in REQUIRED_STATE_FAMILIES
        ),
        "exact_energy_error_within_tolerance": (
            factor["exact_energy_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "one_hot_error_within_tolerance": (
            factor["one_hot_direction_max_error"]
            <= protocol["energy_tolerance"]
        ),
        "probability_distributions_complete": (
            probabilities["distribution_count"] > 0
        ),
        "probability_values_finite": probabilities["all_finite"],
        "probability_values_nonnegative": probabilities["all_nonnegative"],
        "probability_sums_within_tolerance": (
            probabilities["probability_sum_max_error"]
            <= protocol["probability_sum_tolerance"]
        ),
        "all_numeric_values_finite": _all_numeric_finite({
            "stage_t_rows": stage_t_rows,
            "a0_rows": a0_rows,
        }),
    }
    gates["all_common_semantic_gates_passed"] = all(gates.values())
    return gates


def _classify_a0(
    stage_t_aggregates, a0_rows, common_gates, protocol
):
    temperatures = {}
    for temperature in protocol["evaluation_temperatures"]:
        key = _tau_key(temperature)
        trajectory_logit = stage_t_aggregates["by_temperature"][key][
            "independent_direction_diagnostics"
        ]
        factor_logit = _aggregate_logit(a0_rows, temperature)
        gates = {
            "common_semantic_gates_passed": common_gates[
                "all_common_semantic_gates_passed"
            ],
            "source_trajectories_complete": stage_t_aggregates[
                "by_temperature"
            ][key]["all_rounds_complete"],
            "source_scale_and_logits_finite": (
                trajectory_logit["all_finite"]
            ),
            "source_raw_logits_strictly_inside_clip": (
                trajectory_logit["raw_logit_strictly_inside_clip"]
                and trajectory_logit["clip_hit_count"] == 0
            ),
            "source_conditionals_bidirectional": (
                trajectory_logit["all_conditionals_bidirectional"]
            ),
            "factor_conditions_complete": (
                factor_logit["condition_count"] > 0
            ),
            "factor_raw_logits_strictly_inside_clip": (
                factor_logit["raw_logit_strictly_inside_clip"]
                and factor_logit["clip_hit_count"] == 0
            ),
            "factor_conditionals_bidirectional": (
                factor_logit["all_conditionals_bidirectional"]
                and factor_logit["minimum_binary_outcome_probability"]
                is not None
                and factor_logit["minimum_binary_outcome_probability"] > 0.0
            ),
        }
        eligible = all(gates.values())
        if not common_gates["all_common_semantic_gates_passed"]:
            status = "common_semantic_gate_failed"
        elif eligible:
            status = "eligible_for_mixing"
        else:
            status = "out_of_numerical_domain"
        temperatures[key] = {
            "status": status,
            "eligible_for_mixing": eligible,
            "gates": gates,
            "source_trajectory_logit_diagnostics": trajectory_logit,
            "factor_conditional_logit_diagnostics": factor_logit,
        }
    return {
        "rule": (
            "each tau independently requires all Stage T independent and "
            "all frozen-factor raw |logit|<30, finite probabilities and "
            "strict bidirectional support"
        ),
        "temperatures": temperatures,
        "eligible_temperatures": [
            temperature
            for temperature in protocol["evaluation_temperatures"]
            if temperatures[_tau_key(temperature)]["eligible_for_mixing"]
        ],
    }


def _a1_correctness_gates(a0_rows, a1_rows, aggregates, protocol, a0):
    results = {}
    a0_by_state = {row["state_id"]: row for row in a0_rows}
    a1_by_state = {row["state_id"]: row for row in a1_rows}
    state_identity_aligned = (
        list(a0_by_state) == list(a1_by_state)
        and all(
            a0_by_state[state_id]["state_sha256"]
            == a1_by_state[state_id]["state_sha256"]
            for state_id in a0_by_state
        )
    )
    for temperature in protocol["evaluation_temperatures"]:
        key = _tau_key(temperature)
        if not a0["temperatures"][key]["eligible_for_mixing"]:
            results[key] = {
                "applicable": False,
                "all_correctness_gates_passed": None,
                "gates": {},
            }
            continue
        global_group = aggregates["global"]
        factor = global_group["factor_diagnostics"]
        probabilities = global_group[
            "probability_diagnostics_by_temperature"
        ][key]
        production = global_group["production_sampler_diagnostics"][key]
        replay_exact = all(
            a0_by_state[state_id]["probe"][
                "conditional_logit_diagnostics"
            ][key]
            == a1_by_state[state_id]["probe"][
                "conditional_logit_diagnostics"
            ][key]
            for state_id in a0_by_state
        )
        gates = {
            "a0_replay_state_identity_exact": state_identity_aligned,
            "a0_replay_factor_logits_exact": replay_exact,
            "proposal_counts_complete": all(
                row["probe"]["n_proposals"]
                == protocol["proposals_per_state"]
                for row in a1_rows
            ),
            "all_required_groups_have_active_rows": all(
                aggregates[group]["kernel_summary"][
                    probe._gibbs_name(temperature, 0)
                ]["participating_active_rows"] > 0
                for group in REQUIRED_MIXING_GROUPS
            ),
            "exact_energy_error_within_tolerance": (
                factor["exact_energy_max_error"]
                <= protocol["energy_tolerance"]
            ),
            "one_hot_error_within_tolerance": (
                factor["one_hot_direction_max_error"]
                <= protocol["energy_tolerance"]
            ),
            "tvd_monotonic_within_tolerance": (
                factor["tvd_snapshot_increase_max_by_temperature"][key]
                <= protocol["tvd_monotonic_tolerance"]
            ),
            "probability_distributions_complete": (
                probabilities["distribution_count"] > 0
            ),
            "probability_values_finite": probabilities["all_finite"],
            "probability_values_nonnegative": (
                probabilities["all_nonnegative"]
            ),
            "probability_sums_within_tolerance": (
                probabilities["probability_sum_max_error"]
                <= protocol["probability_sum_tolerance"]
            ),
            "production_sampler_replay_complete": (
                production["comparison_count"] > 0
                and production["microsteps"] > 0
            ),
            "production_sampler_matches_exact_tape_replay": (
                production["all_exact_tape_replays_match"]
                and production["mismatch_count"] == 0
            ),
            "all_numeric_values_finite": _all_numeric_finite({
                "a1_rows": a1_rows,
                "aggregates": aggregates,
            }),
        }
        results[key] = {
            "applicable": True,
            "all_correctness_gates_passed": all(gates.values()),
            "gates": gates,
            "production_sampler_diagnostics": production,
        }
    return results


def _classify(aggregates, a0, correctness, protocol):
    temperatures = {}
    for temperature in protocol["evaluation_temperatures"]:
        key = _tau_key(temperature)
        a0_result = a0["temperatures"][key]
        if not a0_result["eligible_for_mixing"]:
            temperatures[key] = {
                "status": a0_result["status"],
                "minimal_sufficient_sweeps": None,
                "candidates": [],
            }
            continue
        candidates = []
        minimal = None
        correctness_passed = correctness[key][
            "all_correctness_gates_passed"
        ]
        for sweep in protocol["candidate_sweeps"]:
            config = probe._gibbs_name(temperature, sweep)
            group_checks = {}
            for group in REQUIRED_MIXING_GROUPS:
                kernel = aggregates[group]["kernel_summary"][config]
                recovery = aggregates[group][
                    "expected_direction_gap_recovery"
                ][config]
                passed = bool(
                    kernel["participating_active_rows"] > 0
                    and kernel["tvd_to_joint"]
                    <= protocol["tvd_threshold"]
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
            candidate_passed = bool(
                correctness_passed
                and all(row["passed"] for row in group_checks.values())
            )
            candidates.append({
                "sweeps": sweep,
                "groups": group_checks,
                "passed": candidate_passed,
            })
            if candidate_passed and minimal is None:
                minimal = sweep
        if not correctness_passed:
            status = "a1_correctness_gate_failed"
        elif minimal is None:
            status = "not_sufficient_through_32"
        else:
            status = "sufficient_within_grid"
        temperatures[key] = {
            "status": status,
            "minimal_sufficient_sweeps": minimal,
            "candidates": candidates,
        }
    return {
        "selection_rule": (
            "for each A0-eligible tau, choose the first sweep in 8,16,32 "
            "passing TVD<=0.05 and recovery>=0.80 globally and in all "
            "eleven preregistered state families"
        ),
        "required_groups": list(REQUIRED_MIXING_GROUPS),
        "temperatures": temperatures,
    }


def run_stage_a(mode, output_dir):
    protocol = _protocol(mode)
    implementation_gates = _validate_implementation_constants(protocol)
    output_directory = Path(output_dir)
    library_path = output_directory / "state_library.json"
    report_path = output_directory / "stage_t_a_report.json"
    collisions = [
        str(path) for path in (library_path, report_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            f"输出已存在，尚未启动任何轨迹：{collisions}"
        )
    git = _git_identity()
    if mode == "formal" and not git["worktree_clean"]:
        raise RuntimeError("正式 Stage T/A 要求 tracked 工作树干净")
    target, queries, schema, marginals, input_hashes = _load_inputs()
    protocol_identity = {
        "protocol": protocol,
        "input_sha256": input_hashes,
        "git_commit": git["commit"],
    }
    protocol_sha256 = _canonical_sha256(protocol_identity)

    start = time.perf_counter()
    stage_t_rows = _run_stage_t(
        target, queries, schema, marginals, protocol
    )
    stage_t_identity_gates = _stage_t_identity_gates(
        stage_t_rows, protocol
    )
    stage_t_aggregates = _aggregate_stage_t(stage_t_rows, protocol)
    state_runs = {
        seed: {
            row["temperature"]: row
            for row in stage_t_rows if row["seed"] == seed
        }
        for seed in protocol["state_library_seeds"]
    }
    library = builder.build_state_library_from_runs(
        state_runs,
        seeds=protocol["state_library_seeds"],
        rounds=protocol["rounds"],
        snapshot_rounds=protocol["snapshot_rounds"],
        device=protocol["device"],
        source_temperatures=protocol["source_temperatures"],
    )
    library.update({
        "mode": mode,
        "formal_protocol": mode == "formal",
        "input_sha256": input_hashes,
        "git_commit": git["commit"],
        "git_worktree_clean_at_start": git["worktree_clean"],
        "protocol_sha256": protocol_sha256,
    })
    _write_json_atomic(library_path, library)
    reloaded_library = _load_json_strict(library_path)
    _validate_reloaded_library(
        reloaded_library, protocol, input_hashes
    )
    library_sha256 = _sha256_file(library_path)

    a0_rows = _probe_library(
        reloaded_library,
        target,
        queries,
        schema,
        protocol,
        temperatures=protocol["evaluation_temperatures"],
        sweeps=[0],
    )
    common_gates = _common_semantic_gates(
        reloaded_library,
        stage_t_rows,
        stage_t_identity_gates,
        a0_rows,
        protocol,
        input_hashes,
    )
    a0 = _classify_a0(
        stage_t_aggregates, a0_rows, common_gates, protocol
    )
    eligible = a0["eligible_temperatures"]
    if eligible:
        a1_rows = _probe_library(
            reloaded_library,
            target,
            queries,
            schema,
            protocol,
            temperatures=eligible,
            sweeps=protocol["sweeps"],
        )
        aggregates = _aggregate_all(a1_rows, protocol, eligible)
    else:
        a1_rows = []
        aggregates = {}
    correctness = _a1_correctness_gates(
        a0_rows, a1_rows, aggregates, protocol, a0
    )
    selection = _classify(aggregates, a0, correctness, protocol)
    formal_identity_gates = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _protocol("formal"),
        "worktree_clean": git["worktree_clean"],
        "input_hashes_match": input_hashes == EXPECTED_INPUT_SHA256,
    }
    a1_execution_valid = all(
        result["all_correctness_gates_passed"]
        for result in correctness.values()
        if result["applicable"]
    )
    formal_result_valid = bool(
        all(formal_identity_gates.values())
        and common_gates["all_common_semantic_gates_passed"]
        and a1_execution_valid
    )
    report = {
        "report_format": REPORT_FORMAT,
        "status": "complete",
        "experiment": "issue49_high_temperature_factor_gibbs_stage_t_a",
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "interpretation": (
            "formal_preregistered_stage_t_a"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "input_sha256": input_hashes,
        "git": git,
        "command_argv": list(sys.argv),
        "environment": trajectory._environment(protocol["device"]),
        "implementation_gates": implementation_gates,
        "stage_t": {
            "identity_gates": stage_t_identity_gates,
            "aggregates": stage_t_aggregates,
            "trajectories": stage_t_rows,
        },
        "state_library": {
            "path": str(library_path.resolve()),
            "sha256": library_sha256,
            "format": reloaded_library["state_library_format"],
            "state_count": reloaded_library["state_count"],
        },
        "a0": {
            "state_results": a0_rows,
            "classification": a0,
        },
        "a1": {
            "state_results": a1_rows,
            "aggregates": aggregates,
            "correctness_gates": correctness,
            "selection": selection,
        },
        "common_semantic_gates": common_gates,
        "formal_identity_gates": formal_identity_gates,
        "elapsed_sec": float(time.perf_counter() - start),
    }
    _write_json_atomic(report_path, report)
    return report_path, library_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["smoke", "formal"], default="smoke"
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/issue49_high_temperature_factor_gibbs/"
            "stage_t_a_smoke"
        ),
    )
    args = parser.parse_args()
    if args.mode == "formal" and args.output_dir.endswith("stage_t_a_smoke"):
        parser.error("正式模式必须显式提供非 smoke 输出目录")

    report_path, library_path, report = run_stage_a(
        args.mode, args.output_dir
    )
    print("\n===== Issue #49 Stage T/A =====")
    print(f"mode={report['mode']}")
    common_passed = report["common_semantic_gates"][
        "all_common_semantic_gates_passed"
    ]
    print(
        f"common_semantic_gates={common_passed}"
    )
    for temperature, result in report["a1"]["selection"][
        "temperatures"
    ].items():
        print(
            f"{temperature}: status={result['status']} "
            f"minimal_sweeps={result['minimal_sufficient_sweeps']}"
        )
    print(f"state_library={library_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
