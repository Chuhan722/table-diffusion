"""独立复算并审计 Issue #49 协议 v2 的 Stage T/A 输出。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import probe_factorized_gibbs_mixing as probe
else:
    import compare_factorized_gibbs_unfiltered as trajectory
    import probe_factorized_gibbs_mixing as probe

from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


AUDIT_FORMAT = "issue49_stage_t_a_audit_v2"
REPORT_FORMAT = "issue49_stage_t_a_report_v2"
STATE_LIBRARY_FORMAT = "issue49_unfiltered_state_library_v2"
TEMPERATURES = [4.0, 5.0, 6.0, 7.0, 8.0]
SWEEPS = [0, 8, 16, 32]
CANDIDATE_SWEEPS = [8, 16, 32]
EXPECTED_INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "queries": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
}
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
PROPOSAL_METRICS = (
    "gain",
    "linear_gain",
    "quadratic_penalty",
    "gain_per_changed_cell",
    "changed_cells",
    "changed_rows",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _tau_label(temperature):
    return f"{temperature:g}".replace(".", "p")


def _tau_key(temperature):
    return f"tau_{_tau_label(temperature)}"


def _gibbs_name(temperature, sweeps):
    return f"gibbs_tau_{_tau_label(temperature)}_sweeps_{sweeps}"


def _joint_name(temperature):
    return f"joint_tau_{_tau_label(temperature)}"


def _config_names(temperatures, sweeps):
    names = []
    for temperature in temperatures:
        names.extend(
            _gibbs_name(temperature, value) for value in sweeps
        )
        names.append(_joint_name(temperature))
    return names


def _required_state_families():
    return (
        "initial",
        *(f"mid_source_{_tau_key(tau)}" for tau in TEMPERATURES),
        *(f"late_source_{_tau_key(tau)}" for tau in TEMPERATURES),
    )


REQUIRED_STATE_FAMILIES = _required_state_families()
REQUIRED_MIXING_GROUPS = ("global", *REQUIRED_STATE_FAMILIES)


def _expected_protocol(mode):
    if mode == "formal":
        stage_t_seeds = list(range(10))
        state_seeds = [0, 1, 2]
        rounds, snapshots, proposals = 1000, [0, 500, 1000], 200
    elif mode == "smoke":
        stage_t_seeds = [99]
        state_seeds = [99]
        rounds, snapshots, proposals = 12, [0, 6, 12], 2
    else:
        raise RuntimeError(f"未知 Stage T/A mode：{mode!r}")
    return {
        "protocol_version": 2,
        "mode": mode,
        "dataset": "test_300x10",
        "stage_t_seeds": stage_t_seeds,
        "state_library_seeds": state_seeds,
        "rounds": rounds,
        "snapshot_rounds": snapshots,
        "source_temperatures": list(TEMPERATURES),
        "source_sweeps": 0,
        "evaluation_temperatures": list(TEMPERATURES),
        "sweeps": list(SWEEPS),
        "candidate_sweeps": list(CANDIDATE_SWEEPS),
        "proposals_per_state": proposals,
        "rho": 0.01,
        "eta": 0.5,
        "trajectory_mu": 0.01,
        "probe_mu": 0.0,
        "max_factor_order": 3,
        "max_active_attributes": 12,
        "logit_clip": 30.0,
        "device": "numpy",
        "tvd_threshold": 0.05,
        "recovery_threshold": 0.8,
        "energy_tolerance": 1e-10,
        "tvd_monotonic_tolerance": 1e-12,
        "probability_sum_tolerance": 1e-12,
    }


def _load_json_strict(path):
    def reject_constant(value):
        raise ValueError(f"JSON 包含非标准数值常量：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"审计输出已存在，不覆盖：{output}")
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


def _assert_same(actual, expected, path="value"):
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
            actual_fields = (
                set(actual) if isinstance(actual, dict) else type(actual)
            )
            raise RuntimeError(
                f"{path} 字段不一致：{actual_fields} != {set(expected)}"
            )
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


def _all_numeric_finite(value):
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (int, bool)):
        return True
    return value is None or isinstance(value, str)


def _load_frozen_inputs():
    paths = {
        "schema": REPOSITORY_ROOT / "configs/test_300x10/schema.yaml",
        "queries": (
            REPOSITORY_ROOT / "configs/test_300x10/measured_50query.json"
        ),
        "marginals": (
            REPOSITORY_ROOT / "configs/test_300x10/init_marginals.json"
        ),
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"独立审计发现输入哈希变化：{hashes}")
    raw_queries = _load_json_strict(paths["queries"])
    schema = load_schema(str(paths["schema"]))
    queries = load_queries(str(paths["queries"]))
    target = np.asarray([query["result"] for query in queries], dtype=float)
    if (
        raw_queries.get("record_count") != 300
        or raw_queries.get("query_count") != 50
        or len(queries) != 50
        or len(schema.attribute_names()) != 10
        or not np.all(np.isfinite(target))
    ):
        raise RuntimeError("独立审计发现冻结输入结构变化")
    return target, queries, schema, hashes


def _family_identity(family, protocol):
    if family == "initial":
        return protocol["snapshot_rounds"][0], None
    stage, marker, tau_label = family.partition("_source_tau_")
    if marker != "_source_tau_" or stage not in ("mid", "late"):
        raise RuntimeError(f"未知状态族：{family}")
    temperature = float(tau_label.replace("p", "."))
    if temperature not in TEMPERATURES:
        raise RuntimeError(f"未知状态族温度：{family}")
    state_round = (
        protocol["snapshot_rounds"][1]
        if stage == "mid" else protocol["snapshot_rounds"][2]
    )
    return state_round, temperature


def _validate_library(library, protocol, target, queries, schema, hashes):
    expected_count = len(protocol["state_library_seeds"]) * 11
    expected_header = {
        "state_library_format": STATE_LIBRARY_FORMAT,
        "dataset": "test_300x10",
        "source_kernel": "independent_directional_unfiltered",
        "source_temperatures": list(TEMPERATURES),
        "source_sweeps": 0,
        "rounds": protocol["rounds"],
        "snapshot_rounds": protocol["snapshot_rounds"],
        "seeds": protocol["state_library_seeds"],
        "expected_state_count": expected_count,
        "state_count": expected_count,
        "mode": protocol["mode"],
        "formal_protocol": protocol["mode"] == "formal",
        "input_sha256": hashes,
    }
    for key, expected in expected_header.items():
        _assert_same(library.get(key), expected, f"library.{key}")
    states = library.get("states")
    if not isinstance(states, list) or len(states) != expected_count:
        raise RuntimeError("状态库 states 数量错误")
    if not library.get("all_gates_passed") or not all(
        library.get("gates", {}).values()
    ):
        raise RuntimeError("状态库自身门禁未通过")
    state_ids = [row.get("state_id") for row in states]
    if len(set(state_ids)) != len(state_ids):
        raise RuntimeError("状态库 state_id 重复")

    indexed = {}
    for entry in states:
        seed = entry.get("seed")
        family = entry.get("state_family")
        if seed not in protocol["state_library_seeds"]:
            raise RuntimeError(f"状态库 seed 越界：{seed}")
        if family not in REQUIRED_STATE_FAMILIES:
            raise RuntimeError(f"未知状态族：{family}")
        key = (seed, family)
        if key in indexed:
            raise RuntimeError(f"状态库 seed-family 重复：{key}")
        indexed[key] = entry
        state_round, source_temperature = _family_identity(family, protocol)
        expected_id = f"seed_{seed}_{family}_round_{state_round}"
        if (
            entry.get("state_round") != state_round
            or entry.get("source_temperature") != source_temperature
            or entry.get("state_id") != expected_id
            or entry.get("shared_source_temperatures")
            != (list(TEMPERATURES) if family == "initial" else None)
        ):
            raise RuntimeError(f"状态族身份错误：{entry.get('state_id')}")
        snapshot = entry.get("snapshot")
        _, controls = probe._restore_current_snapshot(
            snapshot, target, queries, schema, device="numpy"
        )
        expected_snapshot_temperature = (
            TEMPERATURES[0]
            if source_temperature is None else source_temperature
        )
        if (
            controls["source_seed"] != seed
            or controls["state_round"] != state_round
            or controls["state_sha256"] != snapshot["state_sha256"]
            or snapshot["source_sweeps"] != 0
            or snapshot["source_rounds"] != protocol["rounds"]
            or snapshot["state_kind"] != "current"
            or snapshot["source_temperature"]
            != expected_snapshot_temperature
            or snapshot["direction_reference_scale_round"] != 0
            or snapshot["gibbs_rng_state_sha256"] is not None
        ):
            raise RuntimeError(f"快照身份错误：{entry['state_id']}")
        alpha_round = min(state_round, protocol["rounds"] - 1)
        expected_alpha = 2.0 + 8.0 * (
            alpha_round / (protocol["rounds"] - 1)
        )
        _assert_same(
            snapshot["donor_alpha"],
            expected_alpha,
            f"{entry['state_id']}.snapshot.donor_alpha",
        )

    seed_rows = library.get("seed_rows")
    if not isinstance(seed_rows, list) or len(seed_rows) != len(
        protocol["state_library_seeds"]
    ):
        raise RuntimeError("状态库 seed_rows 不完整")
    seed_row_map = {row.get("seed"): row for row in seed_rows}
    if set(seed_row_map) != set(protocol["state_library_seeds"]):
        raise RuntimeError("状态库 seed_rows seeds 不匹配")
    for seed in protocol["state_library_seeds"]:
        entries = {
            family: indexed[(seed, family)]
            for family in REQUIRED_STATE_FAMILIES
        }
        initial = entries["initial"]["snapshot"]
        row = seed_row_map[seed]
        _assert_same(
            row.get("state_ids"),
            [entries[family]["state_id"] for family in REQUIRED_STATE_FAMILIES],
            f"seed_{seed}.state_ids",
        )
        if set(row.get("source_trajectories", {})) != {
            _tau_key(tau) for tau in TEMPERATURES
        }:
            raise RuntimeError(f"seed {seed} 来源轨迹身份不完整")
        for temperature in TEMPERATURES:
            label = _tau_key(temperature)
            late = entries[f"late_source_{label}"]["snapshot"]
            expected = {
                "source_temperature": temperature,
                "source_sweeps": 0,
                "rounds_run": protocol["rounds"],
                "initial_loss": initial["current_loss"],
                "final_loss": late["current_loss"],
                "initial_state_sha256": initial["state_sha256"],
                "final_state_sha256": late["state_sha256"],
                "direction_reference_scale": initial[
                    "direction_reference_scale"
                ],
                "primary_rng_endpoint_sha256": late[
                    "primary_rng_state_sha256"
                ],
            }
            _assert_same(
                row["source_trajectories"][label],
                expected,
                f"seed_{seed}.{label}",
            )
        all_snapshots = [
            entries[family]["snapshot"] for family in REQUIRED_STATE_FAMILIES
        ]
        if not (
            initial["direction_reference_scale"] > 0.0
            and all(
                snapshot["direction_reference_scale"]
                == initial["direction_reference_scale"]
                for snapshot in all_snapshots
            )
            and all(
                len({
                    entries[f"{stage}_source_{_tau_key(tau)}"][
                        "snapshot"
                    ]["donor_alpha"]
                    for tau in TEMPERATURES
                }) == 1
                for stage in ("mid", "late")
            )
        ):
            raise RuntimeError(f"seed {seed} 的 s0/alpha 未对齐")
    return True


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
    maximum_run = max(
        rows,
        key=lambda row: row["independent_direction_diagnostics"][
            "raw_logit_abs_max"
        ],
    )
    maximum = maximum_run["independent_direction_diagnostics"]
    hit_conditions = [
        {
            "seed": run["seed"],
            "temperature": run["temperature"],
            **condition,
        }
        for run in rows
        for condition in run["independent_direction_diagnostics"][
            "clip_hit_conditions"
        ]
    ]
    negative_count = int(sum(
        row["negative_direction_count"] for row in diagnostics
    ))
    positive_count = int(sum(
        row["positive_direction_count"] for row in diagnostics
    ))
    nonempty = [row for row in diagnostics if row["condition_count"]]
    return {
        "condition_count": count,
        "raw_logit_min": float(min(row["raw_logit_min"] for row in nonempty)),
        "raw_logit_max": float(max(row["raw_logit_max"] for row in nonempty)),
        "raw_logit_abs_max": float(maximum["raw_logit_abs_max"]),
        "raw_logit_abs_max_condition": {
            "seed": maximum_run["seed"],
            "temperature": maximum_run["temperature"],
            **maximum["raw_logit_abs_max_condition"],
        },
        "logit_clip": 30.0,
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
            row["minimum_binary_outcome_probability"] for row in nonempty
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
                for row in diagnostics if row["negative_direction_count"]
            ) / negative_count) if negative_count else None
        ),
        "positive_direction_count": positive_count,
        "positive_direction_copy_probability": (
            float(sum(
                row["positive_direction_copy_probability"]
                * row["positive_direction_count"]
                for row in diagnostics if row["positive_direction_count"]
            ) / positive_count) if positive_count else None
        ),
        "all_finite": all(row["all_finite"] for row in diagnostics),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"] for row in diagnostics
        ),
    }


def _stage_t_recompute(rows, protocol):
    expected_pairs = [
        (seed, temperature)
        for seed in protocol["stage_t_seeds"]
        for temperature in TEMPERATURES
    ]
    if [(row.get("seed"), row.get("temperature")) for row in rows] != (
        expected_pairs
    ):
        raise RuntimeError("Stage T 轨迹顺序或网格不完整")
    by_seed = {
        seed: [row for row in rows if row["seed"] == seed]
        for seed in protocol["stage_t_seeds"]
    }
    state_seeds = set(protocol["state_library_seeds"])
    gates = {
        "trajectory_grid_complete": len(rows) == len(expected_pairs),
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
                row.get("snapshot_rounds") == protocol["snapshot_rounds"]
                and len(row.get("state_snapshots", [])) == 3
            ) if row["seed"] in state_seeds else (
                "state_snapshots" not in row
                and "snapshot_rounds" not in row
            )
            for row in rows
        ),
        "initial_state_aligned_within_seed": all(
            len({row["initial_csv_sha256"] for row in values}) == 1
            and len({row["initial_loss"] for row in values}) == 1
            for values in by_seed.values()
        ),
        "primary_rng_endpoint_aligned_within_seed": all(
            len({row["primary_rng_state_sha256"] for row in values}) == 1
            for values in by_seed.values()
        ),
        "direction_reference_scale_aligned_within_seed": all(
            len({row["direction_reference_scale"] for row in values}) == 1
            for values in by_seed.values()
        ),
        "direction_reference_scale_positive_finite": all(
            row["direction_reference_scale"] is not None
            and math.isfinite(row["direction_reference_scale"])
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
    aggregates = {"trajectory_count": len(rows)}
    aggregates["expected_trajectory_count"] = len(expected_pairs)
    aggregates["by_temperature"] = {}
    for temperature in TEMPERATURES:
        selected = [row for row in rows if row["temperature"] == temperature]
        aggregates["by_temperature"][_tau_key(temperature)] = {
            "trajectory_count": len(selected),
            "seeds": [row["seed"] for row in selected],
            "all_rounds_complete": all(
                row["rounds_run"] == protocol["rounds"] for row in selected
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
    return gates, aggregates


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
            ) / total) if total else 0.0
        )
    return result


def _proposal_summary(rows):
    result = {"n": len(rows)}
    for metric in PROPOSAL_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    result.update({
        "positive_gain_rate": float(np.mean(gains > 0.0)),
        "zero_gain_rate": float(np.mean(gains == 0.0)),
        "negative_gain_rate": float(np.mean(gains < 0.0)),
    })
    return result


def _validate_state_results(
    state_rows, library, temperatures, sweeps, proposals
):
    entries = {row["state_id"]: row for row in library["states"]}
    if [row.get("state_id") for row in state_rows] != list(entries):
        raise RuntimeError("逐状态报告顺序或身份与状态库不一致")
    configs = _config_names(temperatures, sweeps)
    for row in state_rows:
        entry = entries[row["state_id"]]
        if any(
            row[key] != entry[key]
            for key in (
                "seed", "state_round", "state_family", "source_temperature"
            )
        ) or row["state_sha256"] != entry["snapshot"]["state_sha256"]:
            raise RuntimeError(f"逐状态 wrapper 身份错误：{row['state_id']}")
        result = row["probe"]
        if (
            result["n_proposals"] != proposals
            or result["state_sha256"] != row["state_sha256"]
            or set(result["proposal_rows"]) != set(configs)
            or set(result["kernel_summary"]) != set(configs)
            or set(result["proposal_summary"]) != set(configs)
            or set(result["conditional_logit_diagnostics"])
            != {_tau_key(tau) for tau in temperatures}
        ):
            raise RuntimeError(f"逐状态探针配置错误：{row['state_id']}")
        for config in configs:
            proposal_rows = result["proposal_rows"][config]
            if len(proposal_rows) != proposals:
                raise RuntimeError(f"proposal 行数错误：{row['state_id']}")
            _assert_same(
                result["kernel_summary"][config],
                _kernel_from_proposal_rows(proposal_rows),
                f"{row['state_id']}.kernel.{config}",
            )
            _assert_same(
                result["proposal_summary"][config],
                _proposal_summary(proposal_rows),
                f"{row['state_id']}.proposal.{config}",
            )
        for temperature in temperatures:
            key = _tau_key(temperature)
            logit = result["conditional_logit_diagnostics"][key]
            if (
                len(logit["clip_hit_conditions"])
                != logit["clip_hit_count"]
                or logit["logit_clip"] != 30.0
                or logit["raw_logit_strictly_inside_clip"]
                != (logit["raw_logit_abs_max"] < 30.0)
            ):
                raise RuntimeError(
                    f"条件 logit 诊断内部不一致：{row['state_id']} {key}"
                )
    return True


def _weighted_kernel(rows, config):
    summaries = [row["probe"]["kernel_summary"][config] for row in rows]
    total = int(sum(row["participating_active_rows"] for row in summaries))
    result = {
        "participating_active_rows": total,
        "active_blocks": int(sum(row["active_blocks"] for row in summaries)),
    }
    for metric in KERNEL_METRICS:
        result[metric] = (
            float(sum(
                row[metric] * row["participating_active_rows"]
                for row in summaries
            ) / total) if total else None
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
        key=lambda row: row["probe"]["conditional_logit_diagnostics"][key][
            "raw_logit_abs_max"
        ],
    )
    maximum = maximum_row["probe"]["conditional_logit_diagnostics"][key]
    hit_conditions = [
        {
            "state_id": state_row["state_id"],
            "seed": state_row["seed"],
            "state_round": state_row["state_round"],
            "state_family": state_row["state_family"],
            **condition,
        }
        for state_row, value in zip(rows, values)
        for condition in value["clip_hit_conditions"]
    ]
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
        "logit_clip": 30.0,
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "clip_hit_conditions": hit_conditions,
        "raw_logit_strictly_inside_clip": all(
            value["raw_logit_strictly_inside_clip"] for value in values
        ),
        "conditional_probability_min": (
            float(min(
                value["conditional_probability_min"] for value in nonempty
            )) if nonempty else None
        ),
        "conditional_probability_max": (
            float(max(
                value["conditional_probability_max"] for value in nonempty
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


def _aggregate_factor(rows, temperatures):
    values = [row["probe"]["factor_diagnostics"] for row in rows]
    return {
        "active_rows": int(sum(row["active_rows"] for row in values)),
        "exact_energy_max_error": float(max(
            row["exact_energy_max_error"] for row in values
        )),
        "one_hot_direction_max_error": float(max(
            row["one_hot_direction_max_error"] for row in values
        )),
        "maximum_active_factor_order": int(max(
            row["maximum_active_factor_order"] for row in values
        )),
        "tvd_snapshot_increase_max": float(max(
            row["tvd_snapshot_increase_max"] for row in values
        )),
        "tvd_snapshot_increase_max_by_temperature": {
            _tau_key(tau): float(max(
                row["tvd_snapshot_increase_max_by_temperature"][_tau_key(tau)]
                for row in values
            ))
            for tau in temperatures
        },
        "factor_build_elapsed_sec": float(sum(
            row["factor_build_elapsed_sec"] for row in values
        )),
        "exact_finite_state_propagation_elapsed_sec": float(sum(
            row["exact_finite_state_propagation_elapsed_sec"]
            for row in values
        )),
    }


def _aggregate_probability(rows, temperature=None):
    if temperature is None:
        values = [row["probe"]["probability_diagnostics"] for row in rows]
    else:
        key = _tau_key(temperature)
        values = [
            row["probe"]["probability_diagnostics_by_temperature"][key]
            for row in rows
        ]
    minima = [
        row["minimum_probability"] for row in values
        if row["minimum_probability"] is not None
    ]
    maxima = [
        row["maximum_probability"] for row in values
        if row["maximum_probability"] is not None
    ]
    return {
        "distribution_count": int(sum(
            row["distribution_count"] for row in values
        )),
        "all_finite": all(row["all_finite"] for row in values),
        "all_nonnegative": all(row["all_nonnegative"] for row in values),
        "probability_sum_max_error": float(max(
            row["probability_sum_max_error"] for row in values
        )),
        "minimum_probability": float(min(minima)) if minima else None,
        "maximum_probability": float(max(maxima)) if maxima else None,
    }


def _aggregate_production(rows, temperature):
    key = _tau_key(temperature)
    values = [
        row["probe"]["production_sampler_diagnostics"][key]
        for row in rows
    ]
    comparisons = int(sum(row["comparison_count"] for row in values))
    mismatches = int(sum(row["mismatch_count"] for row in values))
    return {
        "comparison_count": comparisons,
        "mismatch_count": mismatches,
        "all_exact_tape_replays_match": bool(
            comparisons > 0 and mismatches == 0
        ),
        "microsteps": int(sum(row["microsteps"] for row in values)),
        "production_sampler_elapsed_sec": float(sum(
            row["production_sampler_elapsed_sec"] for row in values
        )),
        "exact_tape_replay_elapsed_sec": float(sum(
            row["exact_tape_replay_elapsed_sec"] for row in values
        )),
    }


def _aggregate_group(name, rows, protocol, temperatures):
    configs = _config_names(temperatures, protocol["sweeps"])
    kernels = {config: _weighted_kernel(rows, config) for config in configs}
    raw_proposals = {
        config: [
            proposal
            for row in rows
            for proposal in row["probe"]["proposal_rows"][config]
        ]
        for config in configs
    }
    recovery = {}
    for temperature in temperatures:
        baseline = _gibbs_name(temperature, 0)
        initial_gap = kernels[baseline]["absolute_expected_direction_gap"]
        for sweep in protocol["sweeps"]:
            config = _gibbs_name(temperature, sweep)
            remaining = kernels[config]["absolute_expected_direction_gap"]
            recovery[config] = (
                1.0 - remaining / initial_gap
                if initial_gap is not None and initial_gap > 0.0
                else (1.0 if initial_gap == 0.0 else None)
            )
    return {
        "group": name,
        "state_count": len(rows),
        "state_ids": [row["state_id"] for row in rows],
        "factor_diagnostics": _aggregate_factor(rows, temperatures),
        "probability_diagnostics": _aggregate_probability(rows),
        "probability_diagnostics_by_temperature": {
            _tau_key(tau): _aggregate_probability(rows, tau)
            for tau in temperatures
        },
        "conditional_logit_diagnostics": {
            _tau_key(tau): _aggregate_logit(rows, tau)
            for tau in temperatures
        },
        "production_sampler_diagnostics": {
            _tau_key(tau): _aggregate_production(rows, tau)
            for tau in temperatures
        },
        "kernel_summary": kernels,
        "expected_direction_gap_recovery": recovery,
        "proposal_summary": {
            config: _proposal_summary(values)
            for config, values in raw_proposals.items()
        },
    }


def _aggregate_all(rows, protocol, temperatures):
    groups = {"global": list(rows)}
    for family in REQUIRED_STATE_FAMILIES:
        groups[family] = [
            row for row in rows if row["state_family"] == family
        ]
    groups.update({
        "phase_initial": [
            row for row in rows if row["state_family"] == "initial"
        ],
        "phase_mid": [
            row for row in rows
            if row["state_family"].startswith("mid_source_")
        ],
        "phase_late": [
            row for row in rows
            if row["state_family"].startswith("late_source_")
        ],
    })
    for temperature in TEMPERATURES:
        key = _tau_key(temperature)
        groups[f"source_{key}"] = [
            row for row in rows
            if row["state_family"].endswith(f"source_{key}")
        ]
    if any(not values for values in groups.values()):
        raise RuntimeError("A1 汇总存在空状态组")
    return {
        name: _aggregate_group(name, values, protocol, temperatures)
        for name, values in groups.items()
    }


def _common_gates(
    library, stage_rows, stage_gates, a0_rows, protocol, hashes
):
    factor = _aggregate_factor(a0_rows, TEMPERATURES)
    probabilities = _aggregate_probability(a0_rows)
    baseline = _gibbs_name(TEMPERATURES[0], 0)
    gates = {
        "input_hashes_match_preregistration": hashes == EXPECTED_INPUT_SHA256,
        "stage_t_identity_passed": stage_gates[
            "all_identity_gates_passed"
        ],
        "state_library_passed": bool(
            library.get("all_gates_passed")
            and all(library.get("gates", {}).values())
        ),
        "state_count_complete": len(a0_rows) == library["expected_state_count"],
        "proposal_counts_complete": all(
            row["probe"]["n_proposals"] == protocol["proposals_per_state"]
            for row in a0_rows
        ),
        "raw_proposal_rows_complete": all(
            len(values) == protocol["proposals_per_state"]
            for row in a0_rows
            for values in row["probe"]["proposal_rows"].values()
        ),
        "all_required_families_have_active_rows": all(
            sum(
                row["probe"]["kernel_summary"][baseline][
                    "participating_active_rows"
                ]
                for row in a0_rows if row["state_family"] == family
            ) > 0
            for family in REQUIRED_STATE_FAMILIES
        ),
        "exact_energy_error_within_tolerance": (
            factor["exact_energy_max_error"] <= protocol["energy_tolerance"]
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
            "stage_t_rows": stage_rows,
            "a0_rows": a0_rows,
        }),
    }
    gates["all_common_semantic_gates_passed"] = all(gates.values())
    return gates


def _classify_a0(stage_aggregates, a0_rows, common, protocol):
    results = {}
    for temperature in TEMPERATURES:
        key = _tau_key(temperature)
        source = stage_aggregates["by_temperature"][key][
            "independent_direction_diagnostics"
        ]
        factor = _aggregate_logit(a0_rows, temperature)
        gates = {
            "common_semantic_gates_passed": common[
                "all_common_semantic_gates_passed"
            ],
            "source_trajectories_complete": stage_aggregates[
                "by_temperature"
            ][key]["all_rounds_complete"],
            "source_scale_and_logits_finite": source["all_finite"],
            "source_raw_logits_strictly_inside_clip": (
                source["raw_logit_strictly_inside_clip"]
                and source["clip_hit_count"] == 0
            ),
            "source_conditionals_bidirectional": source[
                "all_conditionals_bidirectional"
            ],
            "factor_conditions_complete": factor["condition_count"] > 0,
            "factor_raw_logits_strictly_inside_clip": (
                factor["raw_logit_strictly_inside_clip"]
                and factor["clip_hit_count"] == 0
            ),
            "factor_conditionals_bidirectional": (
                factor["all_conditionals_bidirectional"]
                and factor["minimum_binary_outcome_probability"] is not None
                and factor["minimum_binary_outcome_probability"] > 0.0
            ),
        }
        eligible = all(gates.values())
        status = (
            "common_semantic_gate_failed"
            if not common["all_common_semantic_gates_passed"]
            else (
                "eligible_for_mixing"
                if eligible else "out_of_numerical_domain"
            )
        )
        results[key] = {
            "status": status,
            "eligible_for_mixing": eligible,
            "gates": gates,
            "source_trajectory_logit_diagnostics": source,
            "factor_conditional_logit_diagnostics": factor,
        }
    return {
        "rule": (
            "each tau independently requires all Stage T independent and "
            "all frozen-factor raw |logit|<30, finite probabilities and "
            "strict bidirectional support"
        ),
        "temperatures": results,
        "eligible_temperatures": [
            tau for tau in TEMPERATURES
            if results[_tau_key(tau)]["eligible_for_mixing"]
        ],
    }


def _correctness(a0_rows, a1_rows, aggregates, protocol, a0):
    results = {}
    a0_map = {row["state_id"]: row for row in a0_rows}
    a1_map = {row["state_id"]: row for row in a1_rows}
    identity = (
        list(a0_map) == list(a1_map)
        and all(
            a0_map[state]["state_sha256"] == a1_map[state]["state_sha256"]
            for state in a0_map
        )
    )
    for temperature in TEMPERATURES:
        key = _tau_key(temperature)
        if not a0["temperatures"][key]["eligible_for_mixing"]:
            results[key] = {
                "applicable": False,
                "all_correctness_gates_passed": None,
                "gates": {},
            }
            continue
        factor = aggregates["global"]["factor_diagnostics"]
        probabilities = aggregates["global"][
            "probability_diagnostics_by_temperature"
        ][key]
        production = aggregates["global"][
            "production_sampler_diagnostics"
        ][key]
        gates = {
            "a0_replay_state_identity_exact": identity,
            "a0_replay_factor_logits_exact": all(
                a0_map[state]["probe"]["conditional_logit_diagnostics"][key]
                == a1_map[state]["probe"]["conditional_logit_diagnostics"][key]
                for state in a0_map
            ),
            "proposal_counts_complete": all(
                row["probe"]["n_proposals"]
                == protocol["proposals_per_state"] for row in a1_rows
            ),
            "all_required_groups_have_active_rows": all(
                aggregates[group]["kernel_summary"][
                    _gibbs_name(temperature, 0)
                ]["participating_active_rows"] > 0
                for group in REQUIRED_MIXING_GROUPS
            ),
            "exact_energy_error_within_tolerance": (
                factor["exact_energy_max_error"] <= protocol["energy_tolerance"]
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
            "probability_values_nonnegative": probabilities["all_nonnegative"],
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


def _selection(aggregates, a0, correctness, protocol):
    results = {}
    for temperature in TEMPERATURES:
        key = _tau_key(temperature)
        if not a0["temperatures"][key]["eligible_for_mixing"]:
            results[key] = {
                "status": a0["temperatures"][key]["status"],
                "minimal_sufficient_sweeps": None,
                "candidates": [],
            }
            continue
        candidates = []
        minimal = None
        correct = correctness[key]["all_correctness_gates_passed"]
        for sweep in CANDIDATE_SWEEPS:
            config = _gibbs_name(temperature, sweep)
            groups = {}
            for group in REQUIRED_MIXING_GROUPS:
                kernel = aggregates[group]["kernel_summary"][config]
                recovery = aggregates[group][
                    "expected_direction_gap_recovery"
                ][config]
                passed = bool(
                    kernel["participating_active_rows"] > 0
                    and kernel["tvd_to_joint"] <= protocol["tvd_threshold"]
                    and recovery is not None
                    and recovery >= protocol["recovery_threshold"]
                )
                groups[group] = {
                    "participating_active_rows": kernel[
                        "participating_active_rows"
                    ],
                    "tvd_to_joint": kernel["tvd_to_joint"],
                    "expected_direction_gap_recovery": recovery,
                    "passed": passed,
                }
            passed = bool(correct and all(row["passed"] for row in groups.values()))
            candidates.append({
                "sweeps": sweep,
                "groups": groups,
                "passed": passed,
            })
            if passed and minimal is None:
                minimal = sweep
        status = (
            "a1_correctness_gate_failed"
            if not correct else (
                "not_sufficient_through_32"
                if minimal is None else "sufficient_within_grid"
            )
        )
        results[key] = {
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
        "temperatures": results,
    }


def audit_stage_a(report_path, library_path, output_path):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    library_file = Path(library_path).resolve()
    report = _load_json_strict(report_file)
    library = _load_json_strict(library_file)
    if report.get("report_format") != REPORT_FORMAT:
        raise RuntimeError("Stage T/A 报告格式版本不匹配")
    if report.get("status") != "complete":
        raise RuntimeError("Stage T/A 报告未完整结束")
    mode = report.get("mode")
    protocol = _expected_protocol(mode)
    _assert_same(report.get("protocol"), protocol, "report.protocol")
    _assert_same(
        report.get("experiment"),
        "issue49_high_temperature_factor_gibbs_stage_t_a",
        "report.experiment",
    )
    _assert_same(
        report.get("interpretation"),
        (
            "formal_preregistered_stage_t_a"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "report.interpretation",
    )
    implementation_gates = {
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
    if not all(implementation_gates.values()):
        raise RuntimeError("审计发现冻结协议与实现常量不一致")
    _assert_same(
        report.get("implementation_gates"),
        implementation_gates,
        "report.implementation_gates",
    )

    target, queries, schema, hashes = _load_frozen_inputs()
    _assert_same(report.get("input_sha256"), hashes, "report.input_sha256")
    expected_protocol_sha = _canonical_sha256({
        "protocol": protocol,
        "input_sha256": hashes,
        "git_commit": report["git"]["commit"],
    })
    _assert_same(
        report.get("protocol_sha256"),
        expected_protocol_sha,
        "report.protocol_sha256",
    )
    if Path(report["state_library"]["path"]).resolve() != library_file:
        raise RuntimeError("报告绑定的状态库路径与审计输入不同")
    library_sha = _sha256_file(library_file)
    _assert_same(
        report["state_library"]["sha256"],
        library_sha,
        "report.state_library.sha256",
    )
    _assert_same(
        report["state_library"]["format"],
        STATE_LIBRARY_FORMAT,
        "report.state_library.format",
    )
    _assert_same(
        report["state_library"]["state_count"],
        library.get("state_count"),
        "report.state_library.state_count",
    )
    _assert_same(
        library.get("protocol_sha256"),
        expected_protocol_sha,
        "library.protocol_sha256",
    )
    _assert_same(
        library.get("git_commit"), report["git"]["commit"],
        "library.git_commit",
    )
    _assert_same(
        library.get("git_worktree_clean_at_start"),
        report["git"]["worktree_clean"],
        "library.git_worktree_clean_at_start",
    )
    _validate_library(library, protocol, target, queries, schema, hashes)

    stage_rows = report["stage_t"]["trajectories"]
    stage_gates, stage_aggregates = _stage_t_recompute(stage_rows, protocol)
    _assert_same(
        report["stage_t"]["identity_gates"],
        stage_gates,
        "report.stage_t.identity_gates",
    )
    _assert_same(
        report["stage_t"]["aggregates"],
        stage_aggregates,
        "report.stage_t.aggregates",
    )

    a0_rows = report["a0"]["state_results"]
    _validate_state_results(
        a0_rows,
        library,
        TEMPERATURES,
        [0],
        protocol["proposals_per_state"],
    )
    common = _common_gates(
        library, stage_rows, stage_gates, a0_rows, protocol, hashes
    )
    _assert_same(
        report["common_semantic_gates"],
        common,
        "report.common_semantic_gates",
    )
    a0 = _classify_a0(stage_aggregates, a0_rows, common, protocol)
    _assert_same(
        report["a0"]["classification"],
        a0,
        "report.a0.classification",
    )

    eligible = a0["eligible_temperatures"]
    a1_rows = report["a1"]["state_results"]
    if eligible:
        _validate_state_results(
            a1_rows,
            library,
            eligible,
            SWEEPS,
            protocol["proposals_per_state"],
        )
        aggregates = _aggregate_all(a1_rows, protocol, eligible)
    else:
        if a1_rows:
            raise RuntimeError("A0 无合格 tau 时 A1 不应有逐状态结果")
        aggregates = {}
    _assert_same(
        report["a1"]["aggregates"],
        aggregates,
        "report.a1.aggregates",
    )
    correctness = _correctness(
        a0_rows, a1_rows, aggregates, protocol, a0
    )
    _assert_same(
        report["a1"]["correctness_gates"],
        correctness,
        "report.a1.correctness_gates",
    )
    selection = _selection(aggregates, a0, correctness, protocol)
    _assert_same(
        report["a1"]["selection"],
        selection,
        "report.a1.selection",
    )

    formal_identity = {
        "mode_is_formal": mode == "formal",
        "formal_parameters_exact": protocol == _expected_protocol("formal"),
        "worktree_clean": report["git"]["worktree_clean"],
        "input_hashes_match": hashes == EXPECTED_INPUT_SHA256,
    }
    _assert_same(
        report["formal_identity_gates"],
        formal_identity,
        "report.formal_identity_gates",
    )
    a1_valid = all(
        result["all_correctness_gates_passed"]
        for result in correctness.values() if result["applicable"]
    )
    formal_result_valid = bool(
        all(formal_identity.values())
        and common["all_common_semantic_gates_passed"]
        and a1_valid
    )
    _assert_same(
        report["formal_result_valid"],
        formal_result_valid,
        "report.formal_result_valid",
    )
    audit = {
        "audit_format": AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "report_path": str(report_file),
        "report_sha256": _sha256_file(report_file),
        "state_library_path": str(library_file),
        "state_library_sha256": library_sha,
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "protocol_sha256": expected_protocol_sha,
        "checks": {
            "frozen_protocol_exact": True,
            "frozen_inputs_exact": True,
            "state_library_identity_reloaded": True,
            "stage_t_recomputed": True,
            "a0_recomputed_per_temperature": True,
            "a1_aggregates_recomputed_from_proposal_rows": True,
            "production_sampler_counters_recomputed": True,
            "selection_recomputed": True,
        },
        "a0": a0,
        "selection": selection,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    _write_json_atomic(output_path, audit)
    return Path(output_path), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_stage_a(
        args.report, args.state_library, args.output
    )
    print("\n===== Issue #49 Stage T/A audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
