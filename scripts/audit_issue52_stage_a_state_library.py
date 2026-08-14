"""Independent audit of the Issue #52 Stage A common state library."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import subprocess
import time

import pandas as pd

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue52_protocol as frozen_protocol
    from scripts import run_issue49_stage_a as common
else:
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue52_protocol as frozen_protocol
    import run_issue49_stage_a as common


AUDIT_FORMAT = "issue52_stage_a_state_library_audit_v1"
LIBRARY_FORMAT = "issue52_stage_a_state_library_v1"
STAGE_T_REPORT_FORMAT = "issue52_stage_t_report_v1"
STAGE_T_AUDIT_FORMAT = "issue52_stage_t_audit_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _assert_same(actual, expected, path):
    if actual != expected:
        raise RuntimeError(f"状态库审计重算不一致：{path}")


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _valid_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _commit_is_ancestor(ancestor, descendant):
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _source_binding(report_file, audit_file, report, audit, protocol):
    report_sha256 = common._sha256_file(report_file)
    audit_sha256 = common._sha256_file(audit_file)
    stage_t_protocol = frozen_protocol.stage_t_protocol(protocol["mode"])
    expected_protocol_sha256 = common._canonical_sha256({
        "protocol": stage_t_protocol,
        "input_sha256": frozen_protocol.EXPECTED_INPUT_SHA256,
        "git_commit": report.get("git", {}).get("commit"),
    })
    frozen_hashes = protocol["expected_source_sha256"]
    gates = {
        "report_format_exact": (
            report.get("report_format") == STAGE_T_REPORT_FORMAT
        ),
        "report_status_complete": report.get("status") == "complete",
        "report_mode_exact": report.get("mode") == protocol["mode"],
        "report_protocol_exact": report.get("protocol") == stage_t_protocol,
        "report_protocol_sha256_exact": (
            report.get("protocol_sha256") == expected_protocol_sha256
        ),
        "report_input_sha256_exact": (
            report.get("input_sha256")
            == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "report_identity_gates_passed": report.get("stage_t", {}).get(
            "identity_gates", {}
        ).get("all_identity_gates_passed") is True,
        "audit_format_exact": (
            audit.get("audit_format") == STAGE_T_AUDIT_FORMAT
        ),
        "audit_status_complete": audit.get("status") == "complete",
        "audit_passed": audit.get("passed") is True,
        "audit_report_sha256_exact": (
            audit.get("report_sha256") == report_sha256
        ),
        "audit_protocol_sha256_exact": (
            audit.get("protocol_sha256") == expected_protocol_sha256
        ),
        "audit_input_sha256_exact": (
            audit.get("input_sha256")
            == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "audit_formal_flag_matches_report": (
            audit.get("formal_result_valid")
            == report.get("formal_result_valid")
        ),
        "audit_git_matches_report": (
            audit.get("git", {}).get("commit")
            == report.get("git", {}).get("commit")
        ),
        "audit_trajectory_sha_matches_report": (
            audit.get("recomputed", {}).get(
                "trajectory_scientific_sha256"
            )
            == report.get("execution", {}).get(
                "trajectory_scientific_sha256"
            )
        ),
        "formal_source_hashes_exact": (
            frozen_hashes is None
            or frozen_hashes == {
                "report": report_sha256,
                "audit": audit_sha256,
            }
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"状态库审计发现 Stage T 来源绑定失败：{failed}")
    identity = {
        "stage_t_report_sha256": report_sha256,
        "stage_t_audit_sha256": audit_sha256,
        "stage_t_protocol_sha256": expected_protocol_sha256,
        "stage_t_trajectory_scientific_sha256": report["execution"][
            "trajectory_scientific_sha256"
        ],
        "stage_t_git_commit": report["git"]["commit"],
        "input_sha256": report["input_sha256"],
    }
    return identity, gates


def _source_rows(report, protocol):
    indexed = {}
    for row in report.get("stage_t", {}).get("trajectories", []):
        run = row.get("run", {})
        pair = (run.get("seed"), run.get("temperature"))
        if pair in indexed:
            raise RuntimeError(f"状态库审计发现来源轨迹重复：{pair}")
        indexed[pair] = row
    result = {}
    for seed in protocol["state_library_seeds"]:
        result[seed] = {}
        for temperature in protocol["source_temperatures"]:
            pair = (seed, temperature)
            if pair not in indexed:
                raise RuntimeError(f"状态库审计发现来源轨迹缺失：{pair}")
            result[seed][temperature] = indexed[pair]
    return result


def _snapshot_map(run, protocol):
    snapshots = run.get("state_snapshots")
    if not isinstance(snapshots, list):
        raise RuntimeError("状态库审计发现来源快照缺失")
    indexed = {}
    for snapshot in snapshots:
        state_round = snapshot.get("state_round")
        if state_round in indexed:
            raise RuntimeError(f"状态库审计发现来源 round 重复：{state_round}")
        indexed[state_round] = snapshot
    if list(indexed) != protocol["snapshot_rounds"]:
        raise RuntimeError("状态库审计发现来源快照轮次或顺序不完整")
    return indexed


def _validate_snapshot_direct(
    snapshot,
    run,
    protocol,
    *,
    target,
    queries,
    schema,
):
    required = {
        "snapshot_format",
        "source_seed",
        "source_rounds",
        "state_round",
        "state_kind",
        "source_temperature",
        "source_sweeps",
        "donor_alpha",
        "current_loss",
        "state_sha256",
        "primary_rng_state_sha256",
        "gibbs_rng_state_sha256",
        "table_columns",
        "table_records",
        "direction_reference_scale",
        "direction_reference_scale_round",
    }
    if not isinstance(snapshot, dict) or required - set(snapshot):
        raise RuntimeError("状态库审计发现快照字段缺失")
    try:
        frame = pd.DataFrame(
            snapshot["table_records"], columns=snapshot["table_columns"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("状态库审计无法重建快照表") from exc
    counts, _, _ = trajectory.evaluate_vectorized(
        frame,
        queries,
        schema,
        target=target,
        n_records=trajectory.N_RECORDS,
        batch_size=256,
        device=protocol["device"],
        want_fitness=False,
        verbose=False,
    )
    recomputed_loss = float(trajectory.compute_loss(target, counts))
    state_round = snapshot.get("state_round")
    history = run.get("current_loss_after_round_history", [])
    expected_loss = (
        run.get("initial_loss")
        if state_round == 0 else history[state_round - 1]
    )
    expected_alpha = trajectory._donor_alpha(
        min(state_round, protocol["source_rounds"] - 1),
        protocol["source_rounds"],
    )
    valid = bool(
        snapshot.get("snapshot_format")
        == trajectory.CURRENT_SNAPSHOT_FORMAT
        and snapshot.get("source_seed") == run.get("seed")
        and snapshot.get("source_rounds") == protocol["source_rounds"]
        and state_round in protocol["snapshot_rounds"]
        and snapshot.get("state_kind") == "current"
        and snapshot.get("source_temperature") == run.get("temperature")
        and snapshot.get("source_sweeps") == 0
        and snapshot.get("gibbs_rng_state_sha256") is None
        and snapshot.get("direction_reference_scale")
        == run.get("direction_reference_scale")
        and snapshot.get("direction_reference_scale_round")
        == run.get("direction_reference_scale_round") == 0
        and snapshot.get("table_columns") == schema.attribute_names()
        and len(frame) == trajectory.N_RECORDS
        and _valid_sha256(snapshot.get("state_sha256"))
        and trajectory._frame_sha256(frame)
        == snapshot.get("state_sha256")
        and _valid_sha256(snapshot.get("primary_rng_state_sha256"))
        and math.isclose(
            float(snapshot.get("current_loss")),
            float(expected_loss),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            recomputed_loss,
            float(snapshot.get("current_loss")),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(snapshot.get("donor_alpha")),
            expected_alpha,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and (
            state_round != 0
            or snapshot.get("state_sha256")
            == run.get("initial_csv_sha256")
        )
        and (
            state_round != protocol["source_rounds"]
            or (
                snapshot.get("state_sha256")
                == run.get("final_csv_sha256")
                and snapshot.get("primary_rng_state_sha256")
                == run.get("primary_rng_state_sha256")
            )
        )
    )
    if not valid:
        raise RuntimeError(
            "状态库快照查询 loss 或来源身份不一致："
            f"seed={run.get('seed')} tau={run.get('temperature')} "
            f"round={state_round}"
        )


def _without_source_temperature(snapshot):
    return {
        key: value for key, value in snapshot.items()
        if key != "source_temperature"
    }


def _trajectory_identity(row, snapshots):
    run = row["run"]
    return {
        "task_id": row["task_id"],
        "source_temperature": float(run["temperature"]),
        "source_sweeps": int(run["sweeps"]),
        "rounds_run": int(run["rounds_run"]),
        "initial_loss": float(run["initial_loss"]),
        "final_loss": float(run["final_loss"]),
        "initial_state_sha256": run["initial_csv_sha256"],
        "final_state_sha256": run["final_csv_sha256"],
        "direction_reference_scale": float(
            run["direction_reference_scale"]
        ),
        "primary_rng_endpoint_sha256": run[
            "primary_rng_state_sha256"
        ],
        "snapshot_state_sha256": {
            str(state_round): snapshot["state_sha256"]
            for state_round, snapshot in snapshots.items()
        },
    }


def _state_entry(
    seed,
    state_round,
    snapshot,
    *,
    source_temperature,
    source_temperatures,
    shared,
):
    if shared:
        family = "initial"
        state_id = f"seed_{seed}_initial_round_0"
    else:
        tau = _tau_key(source_temperature)
        family = f"round_{state_round}_source_{tau}"
        state_id = f"seed_{seed}_{family}"
    return {
        "state_id": state_id,
        "seed": int(seed),
        "state_round": int(state_round),
        "state_family": family,
        "source_temperature": (
            None if shared else float(source_temperature)
        ),
        "shared_source_temperatures": (
            list(source_temperatures) if shared else None
        ),
        "snapshot": copy.deepcopy(snapshot),
    }


def _expected_state_ids(protocol):
    state_ids = []
    for seed in protocol["state_library_seeds"]:
        state_ids.append(f"seed_{seed}_initial_round_0")
        for state_round in protocol["snapshot_rounds"][1:]:
            for temperature in protocol["source_temperatures"]:
                state_ids.append(
                    f"seed_{seed}_round_{state_round}_source_"
                    f"{_tau_key(temperature)}"
                )
    return state_ids


def _derive_expected(
    report,
    protocol,
    source_identity,
    source_binding_gates,
    *,
    target,
    queries,
    schema,
):
    rows_by_seed = _source_rows(report, protocol)
    temperatures = protocol["source_temperatures"]
    snapshot_rounds = protocol["snapshot_rounds"]
    states = []
    seed_rows = []
    raw_count = 0
    for seed in protocol["state_library_seeds"]:
        rows = rows_by_seed[seed]
        snapshots = {
            temperature: _snapshot_map(rows[temperature]["run"], protocol)
            for temperature in temperatures
        }
        raw_count += sum(len(value) for value in snapshots.values())
        for temperature in temperatures:
            for snapshot in snapshots[temperature].values():
                _validate_snapshot_direct(
                    snapshot,
                    rows[temperature]["run"],
                    protocol,
                    target=target,
                    queries=queries,
                    schema=schema,
                )
        canonical_temperature = protocol[
            "canonical_round0_source_temperature"
        ]
        canonical_initial = snapshots[canonical_temperature][0]
        canonical_without_temperature = _without_source_temperature(
            canonical_initial
        )
        seed_gates = {
            "source_temperature_grid_exact": list(rows) == temperatures,
            "all_source_trajectories_independent": all(
                rows[temperature]["kernel"] == "independent"
                and rows[temperature]["run"]["sweeps"] == 0
                for temperature in temperatures
            ),
            "initial_snapshots_equal_except_source_temperature": all(
                _without_source_temperature(snapshots[temperature][0])
                == canonical_without_temperature
                for temperature in temperatures
            ),
            "initial_state_aligned": len({
                rows[temperature]["run"]["initial_csv_sha256"]
                for temperature in temperatures
            }) == 1,
            "initial_loss_aligned": len({
                rows[temperature]["run"]["initial_loss"]
                for temperature in temperatures
            }) == 1,
            "direction_reference_scale_aligned": len({
                rows[temperature]["run"]["direction_reference_scale"]
                for temperature in temperatures
            }) == 1,
            "primary_rng_endpoint_aligned": len({
                rows[temperature]["run"]["primary_rng_state_sha256"]
                for temperature in temperatures
            }) == 1,
            "snapshot_rounds_exact": all(
                list(snapshots[temperature]) == snapshot_rounds
                for temperature in temperatures
            ),
        }
        if not all(seed_gates.values()):
            raise RuntimeError(f"状态库审计发现 seed {seed} 去重门禁失败")
        seed_states = [_state_entry(
            seed,
            0,
            canonical_initial,
            source_temperature=canonical_temperature,
            source_temperatures=temperatures,
            shared=True,
        )]
        for state_round in snapshot_rounds[1:]:
            for temperature in temperatures:
                seed_states.append(_state_entry(
                    seed,
                    state_round,
                    snapshots[temperature][state_round],
                    source_temperature=temperature,
                    source_temperatures=temperatures,
                    shared=False,
                ))
        states.extend(seed_states)
        seed_rows.append({
            "seed": seed,
            "state_ids": [state["state_id"] for state in seed_states],
            "source_trajectories": {
                _tau_key(temperature): _trajectory_identity(
                    rows[temperature], snapshots[temperature]
                )
                for temperature in temperatures
            },
            "gates": seed_gates,
            "all_gates_passed": True,
        })
    state_ids = [state["state_id"] for state in states]
    manifest = {
        "raw_source_snapshot_count": raw_count,
        "deduplicated_state_count": len(states),
        "expected_state_count": protocol["expected_unique_state_count"],
        "state_ids_in_fixed_order": state_ids,
        "state_sha256_by_id": {
            state["state_id"]: state["snapshot"]["state_sha256"]
            for state in states
        },
        "state_count_by_seed": {
            str(seed): sum(state["seed"] == seed for state in states)
            for seed in protocol["state_library_seeds"]
        },
        "state_count_by_round": {
            str(state_round): sum(
                state["state_round"] == state_round for state in states
            )
            for state_round in snapshot_rounds
        },
        "round0_deduplicated_count": sum(
            state["state_round"] == 0 for state in states
        ),
    }
    gates = {
        "source_artifact_binding_passed": all(
            source_binding_gates.values()
        ),
        "raw_snapshot_count_exact": (
            raw_count == protocol["expected_raw_snapshot_count"]
        ),
        "deduplicated_state_count_exact": (
            len(states) == protocol["expected_unique_state_count"]
        ),
        "state_ids_unique": len(state_ids) == len(set(state_ids)),
        "state_order_exact": state_ids == _expected_state_ids(protocol),
        "one_round0_state_per_seed": (
            manifest["round0_deduplicated_count"]
            == len(protocol["state_library_seeds"])
        ),
        "all_seed_gates_passed": all(
            row["all_gates_passed"] for row in seed_rows
        ),
        "all_snapshots_query_loss_recomputed": True,
    }
    return {
        "source_identity": source_identity,
        "source_binding_gates": source_binding_gates,
        "states": states,
        "seed_rows": seed_rows,
        "manifest": manifest,
        "gates": gates,
        "rows_by_seed": rows_by_seed,
    }


def _scientific_payload(library):
    return {
        "state_library_format": library["state_library_format"],
        "mode": library["mode"],
        "protocol": library["protocol"],
        "protocol_sha256": library["protocol_sha256"],
        "source_identity": library["source_identity"],
        "states": library["states"],
        "seed_rows": library["seed_rows"],
        "manifest": library["manifest"],
    }


def _validate_materialized_snapshots(
    states,
    rows_by_seed,
    protocol,
    *,
    target,
    queries,
    schema,
):
    for state in states:
        if not isinstance(state, dict):
            raise RuntimeError("状态库 states 中存在非 object 条目")
        seed = state.get("seed")
        source_temperature = state.get("source_temperature")
        if source_temperature is None:
            source_temperature = protocol[
                "canonical_round0_source_temperature"
            ]
        try:
            row = rows_by_seed[seed][source_temperature]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("状态库条目的来源 seed/tau 无效") from exc
        _validate_snapshot_direct(
            state.get("snapshot"),
            row["run"],
            protocol,
            target=target,
            queries=queries,
            schema=schema,
        )


def audit_state_library(
    report_path,
    stage_t_audit_path,
    library_path,
    output_path,
):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    stage_t_audit_file = Path(stage_t_audit_path).resolve()
    library_file = Path(library_path).resolve()
    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(f"状态库审计输出已存在，不覆盖：{output_file}")
    report = common._load_json_strict(report_file)
    stage_t_audit = common._load_json_strict(stage_t_audit_file)
    library = common._load_json_strict(library_file)
    if library.get("state_library_format") != LIBRARY_FORMAT:
        raise RuntimeError("未知 Issue #52 Stage A 状态库格式")
    mode = library.get("mode")
    protocol = frozen_protocol.stage_a_state_library_protocol(mode)
    _assert_same(library.get("protocol"), protocol, "library.protocol")
    _assert_same(report.get("mode"), mode, "source_report.mode")
    source_identity, source_binding_gates = _source_binding(
        report_file,
        stage_t_audit_file,
        report,
        stage_t_audit,
        protocol,
    )
    target, queries, schema, _, input_sha256 = common._load_inputs()
    _assert_same(
        input_sha256, source_identity["input_sha256"], "current_inputs"
    )
    expected = _derive_expected(
        report,
        protocol,
        source_identity,
        source_binding_gates,
        target=target,
        queries=queries,
        schema=schema,
    )
    _validate_materialized_snapshots(
        library.get("states", []),
        expected["rows_by_seed"],
        protocol,
        target=target,
        queries=queries,
        schema=schema,
    )
    _assert_same(
        library.get("source_identity"), source_identity, "source_identity"
    )
    _assert_same(
        library.get("source_binding_gates"),
        source_binding_gates,
        "source_binding_gates",
    )
    _assert_same(library.get("states"), expected["states"], "states")
    _assert_same(
        library.get("seed_rows"), expected["seed_rows"], "seed_rows"
    )
    _assert_same(
        library.get("manifest"), expected["manifest"], "manifest"
    )
    _assert_same(library.get("gates"), expected["gates"], "gates")
    if not all(expected["gates"].values()):
        raise RuntimeError("状态库审计重算的全局门禁未全部通过")

    builder_git = library.get("builder_git", {})
    expected_protocol_sha256 = common._canonical_sha256({
        "protocol": protocol,
        "source_identity": source_identity,
        "builder_git_commit": builder_git.get("commit"),
    })
    _assert_same(
        library.get("protocol_sha256"),
        expected_protocol_sha256,
        "protocol_sha256",
    )
    source_commit_is_ancestor = _commit_is_ancestor(
        source_identity["stage_t_git_commit"], builder_git.get("commit", "")
    )
    formal_gates = {
        "mode_is_formal": mode == "formal",
        "formal_protocol_exact": (
            protocol
            == frozen_protocol.stage_a_state_library_protocol("formal")
        ),
        "source_report_formal_result_valid": (
            report.get("formal_result_valid") is True
        ),
        "source_audit_formal_result_valid": (
            stage_t_audit.get("formal_result_valid") is True
        ),
        "frozen_source_hashes_exact": source_binding_gates[
            "formal_source_hashes_exact"
        ],
        "builder_worktree_clean": (
            builder_git.get("worktree_clean") is True
        ),
        "source_commit_is_ancestor": source_commit_is_ancestor,
        "input_sha256_exact": (
            input_sha256 == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "state_library_gates_passed": all(expected["gates"].values()),
    }
    _assert_same(
        library.get("formal_identity_gates"),
        formal_gates,
        "formal_identity_gates",
    )
    formal_result_valid = all(formal_gates.values())
    _assert_same(
        library.get("formal_result_valid"),
        formal_result_valid,
        "formal_result_valid",
    )
    expected_scientific_sha256 = common._canonical_sha256(
        _scientific_payload(library)
    )
    _assert_same(
        library.get("state_library_scientific_sha256"),
        expected_scientific_sha256,
        "state_library_scientific_sha256",
    )

    current_git = common._git_identity()
    checks = {
        "library_status_complete": library.get("status") == "complete",
        "interpretation_exact": library.get("interpretation") == (
            "formal_audited_state_library_source"
            if mode == "formal" else "pipeline_smoke_only_not_evidence"
        ),
        "source_binding_passed": all(source_binding_gates.values()),
        "all_state_gates_passed": all(expected["gates"].values()),
        "all_materialized_query_losses_recomputed": True,
        "scientific_sha256_exact": (
            library.get("state_library_scientific_sha256")
            == expected_scientific_sha256
        ),
        "builder_commit_matches_current_checkout": (
            builder_git.get("commit") == current_git["commit"]
        ),
        "current_worktree_clean_when_formal": (
            mode != "formal" or current_git["worktree_clean"]
        ),
        "formal_flag_exact": (
            library.get("formal_result_valid") == formal_result_valid
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Issue #52 Stage A 状态库独立审计失败：{failed}")
    audit = {
        "audit_format": AUDIT_FORMAT,
        "status": "complete",
        "passed": True,
        "formal_result_valid": formal_result_valid,
        "library_path": str(library_file),
        "library_sha256": common._sha256_file(library_file),
        "state_library_scientific_sha256": expected_scientific_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "source_identity": source_identity,
        "git": current_git,
        "checks": checks,
        "recomputed": {
            "manifest": expected["manifest"],
            "gates": expected["gates"],
            "state_ids_in_fixed_order": [
                state["state_id"] for state in expected["states"]
            ],
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    common._write_json_atomic(output_file, audit)
    return output_file, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--stage-t-audit", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, audit = audit_state_library(
        args.report,
        args.stage_t_audit,
        args.library,
        args.output,
    )
    print("\n===== Issue #52 Stage A State Library Audit =====")
    print(f"passed={audit['passed']}")
    print(f"formal_result_valid={audit['formal_result_valid']}")
    print(f"library_sha256={audit['library_sha256']}")
    print(f"audit={output}")


if __name__ == "__main__":
    main()
