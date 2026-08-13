"""组装 Issue #49 高温独立方向无门控 current-state 状态库。"""

import numpy as np

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
else:
    import compare_factorized_gibbs_unfiltered as trajectory


STATE_LIBRARY_FORMAT = "issue49_unfiltered_state_library_v1"
SOURCE_TEMPERATURES = (4.0, 8.0)


def _temperature_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _validate_protocol(seeds, rounds, snapshot_rounds, device):
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
            for seed in seeds
        )
    ):
        raise ValueError("seeds 必须非空、非负且不重复")
    if (
        isinstance(rounds, (bool, np.bool_))
        or not isinstance(rounds, (int, np.integer))
        or rounds <= 1
    ):
        raise ValueError("rounds 必须是大于 1 的整数")
    normalized = trajectory._normalize_snapshot_rounds(
        snapshot_rounds, int(rounds)
    )
    if (
        normalized is None
        or len(normalized) != 3
        or normalized[0] != 0
        or normalized[-1] != rounds
        or not 0 < normalized[1] < rounds
    ):
        raise ValueError(
            "状态库要求恰好三个快照：round 0、一个中期轮次和末轮"
        )
    if device not in ("numpy", "cpu", "cuda"):
        raise ValueError("device 必须是 numpy、cpu 或 cuda")
    return [int(seed) for seed in seeds], int(rounds), normalized


def _snapshot_map(run, expected_rounds):
    snapshots = run.get("state_snapshots")
    if not isinstance(snapshots, list):
        raise RuntimeError("来源轨迹没有返回 current-state 快照")
    indexed = {}
    for snapshot in snapshots:
        state_round = snapshot.get("state_round")
        if state_round in indexed:
            raise RuntimeError(f"来源轨迹包含重复快照：round {state_round}")
        indexed[state_round] = snapshot
    if tuple(sorted(indexed)) != tuple(expected_rounds):
        raise RuntimeError(
            "来源轨迹快照轮次不完整："
            f"得到 {sorted(indexed)}，要求 {list(expected_rounds)}"
        )
    return indexed


def _seed_gates(seed, rounds, snapshot_rounds, runs, snapshots):
    temperatures = SOURCE_TEMPERATURES
    initial = [snapshots[tau][0] for tau in temperatures]
    reference_scale = [
        runs[tau]["direction_reference_scale"] for tau in temperatures
    ]
    comparable_initial_fields = (
        "snapshot_format",
        "source_seed",
        "source_rounds",
        "state_round",
        "state_kind",
        "source_sweeps",
        "donor_alpha",
        "current_loss",
        "best_loss_so_far_diagnostic_only",
        "state_sha256",
        "primary_rng_state_sha256",
        "gibbs_rng_state_sha256",
        "table_columns",
        "table_records",
        "direction_reference_scale",
        "direction_reference_scale_round",
    )
    gates = {
        "source_trajectories_complete": set(runs) == set(temperatures),
        "rounds_complete": all(
            runs[tau]["rounds_run"] == rounds for tau in temperatures
        ),
        "independent_source_kernel": all(
            runs[tau]["sweeps"] == 0 for tau in temperatures
        ),
        "snapshot_protocol_exact": all(
            snapshot["snapshot_format"]
            == trajectory.CURRENT_SNAPSHOT_FORMAT
            and snapshot["source_seed"] == seed
            and snapshot["source_rounds"] == rounds
            and snapshot["state_round"] == state_round
            and snapshot["state_kind"] == "current"
            and snapshot["source_temperature"] == tau
            and snapshot["source_sweeps"] == 0
            for tau in temperatures
            for state_round, snapshot in snapshots[tau].items()
        ),
        "initial_snapshots_exact_except_temperature": all(
            initial[0][field] == initial[1][field]
            for field in comparable_initial_fields
        ),
        "initial_table_aligned": (
            runs[temperatures[0]]["initial_csv_sha256"]
            == runs[temperatures[1]]["initial_csv_sha256"]
            == initial[0]["state_sha256"]
        ),
        "initial_loss_aligned": (
            runs[temperatures[0]]["initial_loss"]
            == runs[temperatures[1]]["initial_loss"]
            == initial[0]["current_loss"]
        ),
        "initial_primary_rng_aligned": (
            initial[0]["primary_rng_state_sha256"]
            == initial[1]["primary_rng_state_sha256"]
        ),
        "primary_rng_endpoint_aligned": (
            runs[temperatures[0]]["primary_rng_state_sha256"]
            == runs[temperatures[1]]["primary_rng_state_sha256"]
        ),
        "direction_reference_scale_aligned": (
            reference_scale[0] == reference_scale[1]
        ),
        "direction_reference_scale_positive_finite": all(
            scale is not None and np.isfinite(scale) and scale > 0.0
            for scale in reference_scale
        ),
        "direction_reference_scale_discovered_round_zero": all(
            snapshot["direction_reference_scale_round"] == 0
            for tau in temperatures
            for snapshot in snapshots[tau].values()
        ),
        "direction_reference_scale_fixed_in_snapshots": all(
            snapshot["direction_reference_scale"]
            == runs[tau]["direction_reference_scale"]
            for tau in temperatures
            for snapshot in snapshots[tau].values()
        ),
        "alpha_schedule_aligned": all(
            snapshots[temperatures[0]][state_round]["donor_alpha"]
            == snapshots[temperatures[1]][state_round]["donor_alpha"]
            for state_round in snapshot_rounds
        ),
        "terminal_snapshot_matches_run": all(
            snapshots[tau][rounds]["state_sha256"]
            == runs[tau]["final_csv_sha256"]
            and snapshots[tau][rounds]["current_loss"]
            == runs[tau]["final_loss"]
            and snapshots[tau][rounds]["primary_rng_state_sha256"]
            == runs[tau]["primary_rng_state_sha256"]
            for tau in temperatures
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(
            f"seed {seed} 状态库身份门禁失败：{failed}"
        )
    return gates


def _trajectory_identity(run):
    return {
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
    }


def _state_entry(seed, stage, source_temperature, snapshot, *, shared=False):
    temperature_label = (
        "shared"
        if shared else _temperature_key(source_temperature)
    )
    family = (
        "initial"
        if shared else f"{stage}_source_{temperature_label}"
    )
    return {
        "state_id": (
            f"seed_{seed}_{family}_round_{snapshot['state_round']}"
        ),
        "seed": int(seed),
        "state_round": int(snapshot["state_round"]),
        "state_family": family,
        "source_temperature": (
            None if shared else float(source_temperature)
        ),
        "shared_source_temperatures": (
            list(SOURCE_TEMPERATURES) if shared else None
        ),
        "snapshot": snapshot,
    }


def build_state_library(
    target,
    queries,
    schema,
    marginals,
    *,
    seeds,
    rounds,
    snapshot_rounds,
    device,
):
    seeds, rounds, snapshot_rounds = _validate_protocol(
        seeds, rounds, snapshot_rounds, device
    )
    states = []
    seed_rows = []
    for seed in seeds:
        runs = {}
        snapshots = {}
        for temperature in SOURCE_TEMPERATURES:
            run = trajectory._run_one(
                target,
                queries,
                schema,
                marginals,
                seed=seed,
                rounds=rounds,
                temperature=temperature,
                sweeps=0,
                device=device,
                snapshot_rounds=snapshot_rounds,
            )
            runs[temperature] = run
            snapshots[temperature] = _snapshot_map(
                run, snapshot_rounds
            )

        gates = _seed_gates(
            seed, rounds, snapshot_rounds, runs, snapshots
        )
        seed_states = [_state_entry(
            seed,
            "initial",
            SOURCE_TEMPERATURES[0],
            snapshots[SOURCE_TEMPERATURES[0]][0],
            shared=True,
        )]
        for state_round in snapshot_rounds[1:]:
            stage = "late" if state_round == rounds else "mid"
            for temperature in SOURCE_TEMPERATURES:
                seed_states.append(_state_entry(
                    seed,
                    stage,
                    temperature,
                    snapshots[temperature][state_round],
                ))
        states.extend(seed_states)
        seed_rows.append({
            "seed": seed,
            "state_ids": [row["state_id"] for row in seed_states],
            "source_trajectories": {
                _temperature_key(temperature): _trajectory_identity(
                    runs[temperature]
                )
                for temperature in SOURCE_TEMPERATURES
            },
            "gates": gates,
            "all_gates_passed": True,
        })

    expected_state_count = len(seeds) * 5
    state_ids = [state["state_id"] for state in states]
    global_gates = {
        "expected_state_count": len(states) == expected_state_count,
        "state_ids_unique": len(state_ids) == len(set(state_ids)),
        "all_seed_gates_passed": all(
            row["all_gates_passed"] for row in seed_rows
        ),
    }
    if not all(global_gates.values()):
        raise RuntimeError(f"状态库全局门禁失败：{global_gates}")
    return {
        "state_library_format": STATE_LIBRARY_FORMAT,
        "dataset": "test_300x10",
        "source_kernel": "independent_directional_unfiltered",
        "source_temperatures": list(SOURCE_TEMPERATURES),
        "source_sweeps": 0,
        "rounds": rounds,
        "snapshot_rounds": list(snapshot_rounds),
        "seeds": seeds,
        "expected_state_count": expected_state_count,
        "state_count": len(states),
        "states": states,
        "seed_rows": seed_rows,
        "gates": global_gates,
        "all_gates_passed": True,
    }
