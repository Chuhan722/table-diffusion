"""组装 Issue #49 高温独立方向无门控 current-state 状态库。"""

import numpy as np

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
    from scripts import issue49_stage_t_a_protocol as frozen_protocol
else:
    import compare_factorized_gibbs_unfiltered as trajectory
    import issue49_stage_t_a_protocol as frozen_protocol


STATE_LIBRARY_FORMAT = "issue49_unfiltered_state_library_v2"
SOURCE_TEMPERATURES = frozen_protocol.TEMPERATURES


def _temperature_key(temperature):
    return f"tau_{temperature:g}".replace(".", "p")


def _normalize_temperatures(source_temperatures):
    values = tuple(float(value) for value in source_temperatures)
    if (
        not values
        or len(set(values)) != len(values)
        or any(not np.isfinite(value) or value < 0.0 for value in values)
    ):
        raise ValueError("source_temperatures 必须非空、非负、有限且不重复")
    return values


def _validate_protocol(
    seeds, rounds, snapshot_rounds, device, source_temperatures
):
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
    temperatures = _normalize_temperatures(source_temperatures)
    return (
        [int(seed) for seed in seeds],
        int(rounds),
        normalized,
        temperatures,
    )


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


def _seed_gates(
    seed, rounds, snapshot_rounds, runs, snapshots, temperatures
):
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
            snapshot[field] == initial[0][field]
            for snapshot in initial[1:]
            for field in comparable_initial_fields
        ),
        "initial_table_aligned": (
            all(
                runs[tau]["initial_csv_sha256"]
                == initial[0]["state_sha256"]
                for tau in temperatures
            )
        ),
        "initial_loss_aligned": (
            all(
                runs[tau]["initial_loss"]
                == initial[0]["current_loss"]
                for tau in temperatures
            )
        ),
        "initial_primary_rng_aligned": (
            all(
                snapshot["primary_rng_state_sha256"]
                == initial[0]["primary_rng_state_sha256"]
                for snapshot in initial[1:]
            )
        ),
        "primary_rng_endpoint_aligned": (
            len({
                runs[tau]["primary_rng_state_sha256"]
                for tau in temperatures
            }) == 1
        ),
        "direction_reference_scale_aligned": (
            len(set(reference_scale)) == 1
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
            snapshots[tau][state_round]["donor_alpha"]
            == snapshots[temperatures[0]][state_round]["donor_alpha"]
            for tau in temperatures[1:]
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


def _state_entry(
    seed,
    stage,
    source_temperature,
    snapshot,
    *,
    source_temperatures,
    shared=False,
):
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
            list(source_temperatures) if shared else None
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
    source_temperatures=SOURCE_TEMPERATURES,
):
    seeds, rounds, snapshot_rounds, temperatures = _validate_protocol(
        seeds,
        rounds,
        snapshot_rounds,
        device,
        source_temperatures,
    )
    runs_by_seed = {}
    for seed in seeds:
        runs = {}
        for temperature in temperatures:
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
        runs_by_seed[seed] = runs

    return build_state_library_from_runs(
        runs_by_seed,
        seeds=seeds,
        rounds=rounds,
        snapshot_rounds=snapshot_rounds,
        device=device,
        source_temperatures=temperatures,
    )


def build_state_library_from_runs(
    runs_by_seed,
    *,
    seeds,
    rounds,
    snapshot_rounds,
    device,
    source_temperatures=SOURCE_TEMPERATURES,
):
    seeds, rounds, snapshot_rounds, temperatures = _validate_protocol(
        seeds,
        rounds,
        snapshot_rounds,
        device,
        source_temperatures,
    )
    if set(runs_by_seed) != set(seeds):
        raise RuntimeError(
            "预计算来源轨迹 seeds 不完整："
            f"得到 {sorted(runs_by_seed)}，要求 {seeds}"
        )

    states = []
    seed_rows = []
    for seed in seeds:
        runs = runs_by_seed[seed]
        if set(runs) != set(temperatures):
            raise RuntimeError(
                f"seed {seed} 预计算来源温度不完整：{sorted(runs)}"
            )
        snapshots = {
            temperature: _snapshot_map(
                runs[temperature], snapshot_rounds
            )
            for temperature in temperatures
        }

        gates = _seed_gates(
            seed,
            rounds,
            snapshot_rounds,
            runs,
            snapshots,
            temperatures,
        )
        seed_states = [_state_entry(
            seed,
            "initial",
            temperatures[0],
            snapshots[temperatures[0]][0],
            source_temperatures=temperatures,
            shared=True,
        )]
        for state_round in snapshot_rounds[1:]:
            stage = "late" if state_round == rounds else "mid"
            for temperature in temperatures:
                seed_states.append(_state_entry(
                    seed,
                    stage,
                    temperature,
                    snapshots[temperature][state_round],
                    source_temperatures=temperatures,
                ))
        states.extend(seed_states)
        seed_rows.append({
            "seed": seed,
            "state_ids": [row["state_id"] for row in seed_states],
            "source_trajectories": {
                _temperature_key(temperature): _trajectory_identity(
                    runs[temperature]
                )
                for temperature in temperatures
            },
            "gates": gates,
            "all_gates_passed": True,
        })

    expected_state_count = len(seeds) * (1 + 2 * len(temperatures))
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
        "source_temperatures": list(temperatures),
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
