"""Materialize the audited Issue #52 Stage A common state library."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import subprocess
import time

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue52_protocol as frozen_protocol
    from scripts import probe_factorized_gibbs_mixing as probe
    from scripts import run_issue49_stage_a as common
else:
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue52_protocol as frozen_protocol
    import probe_factorized_gibbs_mixing as probe
    import run_issue49_stage_a as common


LIBRARY_FORMAT = "issue52_stage_a_state_library_v1"
STAGE_T_REPORT_FORMAT = "issue52_stage_t_report_v1"
STAGE_T_AUDIT_FORMAT = "issue52_stage_t_audit_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _tau_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _commit_is_ancestor(ancestor, descendant):
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _valid_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_artifact_binding(report_path, audit_path, report, audit, protocol):
    report_sha256 = common._sha256_file(report_path)
    audit_sha256 = common._sha256_file(audit_path)
    stage_t_protocol = frozen_protocol.stage_t_protocol(protocol["mode"])
    expected_protocol_sha256 = common._canonical_sha256({
        "protocol": stage_t_protocol,
        "input_sha256": frozen_protocol.EXPECTED_INPUT_SHA256,
        "git_commit": report.get("git", {}).get("commit"),
    })
    frozen_source_hashes = protocol["expected_source_sha256"]
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
            frozen_source_hashes is None
            or frozen_source_hashes == {
                "report": report_sha256,
                "audit": audit_sha256,
            }
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage T 来源产物绑定失败：{failed}")
    return {
        "stage_t_report_sha256": report_sha256,
        "stage_t_audit_sha256": audit_sha256,
        "stage_t_protocol_sha256": expected_protocol_sha256,
        "stage_t_trajectory_scientific_sha256": report["execution"][
            "trajectory_scientific_sha256"
        ],
        "stage_t_git_commit": report["git"]["commit"],
        "input_sha256": report["input_sha256"],
    }, gates


def _source_runs(report, protocol):
    rows = report.get("stage_t", {}).get("trajectories", [])
    indexed = {}
    for row in rows:
        run = row.get("run", {})
        pair = (run.get("seed"), run.get("temperature"))
        if pair in indexed:
            raise RuntimeError(f"Stage T 来源轨迹重复：{pair}")
        indexed[pair] = row
    expected_pairs = [
        (seed, temperature)
        for seed in protocol["state_library_seeds"]
        for temperature in protocol["source_temperatures"]
    ]
    missing = [pair for pair in expected_pairs if pair not in indexed]
    if missing:
        raise RuntimeError(f"Stage T 状态来源轨迹不完整：{missing}")
    return {
        seed: {
            temperature: indexed[(seed, temperature)]
            for temperature in protocol["source_temperatures"]
        }
        for seed in protocol["state_library_seeds"]
    }


def _snapshot_map(run, protocol):
    snapshots = run.get("state_snapshots")
    if not isinstance(snapshots, list):
        raise RuntimeError("Stage T 来源轨迹没有完整 current-state 快照")
    indexed = {}
    for snapshot in snapshots:
        state_round = snapshot.get("state_round")
        if state_round in indexed:
            raise RuntimeError(f"来源快照 round 重复：{state_round}")
        indexed[state_round] = snapshot
    if list(indexed) != protocol["snapshot_rounds"]:
        raise RuntimeError(
            "来源快照轮次或顺序不完整："
            f"得到 {list(indexed)}，要求 {protocol['snapshot_rounds']}"
        )
    return indexed


def _validate_snapshot(
    snapshot,
    run,
    protocol,
    *,
    target,
    queries,
    schema,
):
    state, controls = probe._restore_current_snapshot(
        snapshot,
        target,
        queries,
        schema,
        device=protocol["device"],
    )
    state_round = controls["state_round"]
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
        run.get("seed") == controls["source_seed"]
        and run.get("temperature") == controls["source_temperature"]
        and run.get("sweeps") == protocol["source_sweeps"] == 0
        and run.get("rounds_run") == protocol["source_rounds"]
        and controls["source_rounds"] == protocol["source_rounds"]
        and state_round in protocol["snapshot_rounds"]
        and controls["source_sweeps"] == 0
        and controls["gibbs_rng_state_sha256"] is None
        and controls["direction_reference_scale"]
        == run.get("direction_reference_scale")
        and controls["direction_reference_scale_round"]
        == run.get("direction_reference_scale_round") == 0
        and math.isclose(
            controls["current_loss"],
            float(expected_loss),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            controls["probe_alpha"],
            expected_alpha,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and (
            state_round != 0
            or controls["state_sha256"] == run.get("initial_csv_sha256")
        )
        and (
            state_round != protocol["source_rounds"]
            or (
                controls["state_sha256"] == run.get("final_csv_sha256")
                and controls["primary_rng_state_sha256"]
                == run.get("primary_rng_state_sha256")
            )
        )
    )
    if not valid:
        raise RuntimeError(
            "来源快照身份或轨迹对齐失败："
            f"seed={run.get('seed')} tau={run.get('temperature')} "
            f"round={state_round}"
        )
    return state, controls


def _round_zero_without_temperature(snapshot):
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


def _derive_library_payload(
    report,
    protocol,
    source_identity,
    source_binding_gates,
    *,
    target,
    queries,
    schema,
):
    rows_by_seed = _source_runs(report, protocol)
    states = []
    seed_rows = []
    raw_snapshot_count = 0
    temperatures = protocol["source_temperatures"]
    rounds = protocol["snapshot_rounds"]
    for seed in protocol["state_library_seeds"]:
        rows = rows_by_seed[seed]
        snapshots = {
            temperature: _snapshot_map(rows[temperature]["run"], protocol)
            for temperature in temperatures
        }
        raw_snapshot_count += sum(len(value) for value in snapshots.values())
        for temperature in temperatures:
            run = rows[temperature]["run"]
            for snapshot in snapshots[temperature].values():
                _validate_snapshot(
                    snapshot,
                    run,
                    protocol,
                    target=target,
                    queries=queries,
                    schema=schema,
                )

        canonical_temperature = protocol[
            "canonical_round0_source_temperature"
        ]
        canonical_initial = snapshots[canonical_temperature][0]
        initial_without_temperature = _round_zero_without_temperature(
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
                _round_zero_without_temperature(snapshots[temperature][0])
                == initial_without_temperature
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
                list(snapshots[temperature]) == rounds
                for temperature in temperatures
            ),
        }
        if not all(seed_gates.values()):
            failed = [
                name for name, passed in seed_gates.items() if not passed
            ]
            raise RuntimeError(f"seed {seed} 状态去重门禁失败：{failed}")

        seed_states = [_state_entry(
            seed,
            0,
            canonical_initial,
            source_temperature=canonical_temperature,
            source_temperatures=temperatures,
            shared=True,
        )]
        for state_round in rounds[1:]:
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
        "raw_source_snapshot_count": raw_snapshot_count,
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
            for state_round in rounds
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
            raw_snapshot_count == protocol["expected_raw_snapshot_count"]
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
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Issue #52 状态库全局门禁失败：{failed}")
    return {
        "source_identity": source_identity,
        "source_binding_gates": source_binding_gates,
        "states": states,
        "seed_rows": seed_rows,
        "manifest": manifest,
        "gates": gates,
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


def build_state_library(report_path, stage_t_audit_path, output_path):
    started = time.perf_counter()
    report_file = Path(report_path).resolve()
    audit_file = Path(stage_t_audit_path).resolve()
    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(f"状态库输出已存在，不覆盖：{output_file}")
    report = common._load_json_strict(report_file)
    audit = common._load_json_strict(audit_file)
    mode = report.get("mode")
    protocol = frozen_protocol.stage_a_state_library_protocol(mode)
    current_git = common._git_identity()
    if mode == "formal" and not current_git["worktree_clean"]:
        raise RuntimeError("正式 Stage A 状态库要求当前工作树干净")
    source_identity, source_binding_gates = _source_artifact_binding(
        report_file, audit_file, report, audit, protocol
    )
    target, queries, schema, _, input_sha256 = common._load_inputs()
    if input_sha256 != source_identity["input_sha256"]:
        raise RuntimeError("当前公开输入与 Stage T 来源输入不一致")
    source_commit_is_ancestor = _commit_is_ancestor(
        source_identity["stage_t_git_commit"], current_git["commit"]
    )
    if not source_commit_is_ancestor:
        raise RuntimeError("Stage T 来源 commit 不是当前构建 commit 的祖先")
    payload = _derive_library_payload(
        report,
        protocol,
        source_identity,
        source_binding_gates,
        target=target,
        queries=queries,
        schema=schema,
    )
    protocol_sha256 = common._canonical_sha256({
        "protocol": protocol,
        "source_identity": source_identity,
        "builder_git_commit": current_git["commit"],
    })
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
            audit.get("formal_result_valid") is True
        ),
        "frozen_source_hashes_exact": source_binding_gates[
            "formal_source_hashes_exact"
        ],
        "builder_worktree_clean": current_git["worktree_clean"],
        "source_commit_is_ancestor": source_commit_is_ancestor,
        "input_sha256_exact": (
            input_sha256 == frozen_protocol.EXPECTED_INPUT_SHA256
        ),
        "state_library_gates_passed": all(payload["gates"].values()),
    }
    formal_result_valid = all(formal_gates.values())
    library = {
        "state_library_format": LIBRARY_FORMAT,
        "status": "complete",
        "experiment": "issue52_stage_a_common_current_state_library",
        "mode": mode,
        "formal_result_valid": formal_result_valid,
        "interpretation": (
            "formal_audited_state_library_source"
            if mode == "formal"
            else "pipeline_smoke_only_not_evidence"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "source_identity": payload["source_identity"],
        "builder_git": current_git,
        "source_binding_gates": payload["source_binding_gates"],
        "gates": payload["gates"],
        "formal_identity_gates": formal_gates,
        "manifest": payload["manifest"],
        "states": payload["states"],
        "seed_rows": payload["seed_rows"],
    }
    library["state_library_scientific_sha256"] = common._canonical_sha256(
        _scientific_payload(library)
    )
    library["elapsed_sec"] = float(time.perf_counter() - started)
    common._write_json_atomic(output_file, library)
    return output_file, library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--stage-t-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, library = build_state_library(
        args.report, args.stage_t_audit, args.output
    )
    print("\n===== Issue #52 Stage A State Library =====")
    print(f"mode={library['mode']}")
    print(f"states={library['manifest']['deduplicated_state_count']}")
    print(f"gates={all(library['gates'].values())}")
    print(f"formal_result_valid={library['formal_result_valid']}")
    print(
        "scientific_sha256="
        f"{library['state_library_scientific_sha256']}"
    )
    print(f"library={output}")


if __name__ == "__main__":
    main()
