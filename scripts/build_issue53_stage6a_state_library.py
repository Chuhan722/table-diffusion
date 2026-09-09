#!/usr/bin/env python3
"""Build the frozen Issue #53 Stage 6A independent-state library.

This collector only materializes measured-workload current states. It does
not generate frozen proposals, compute B/C classifications, read offline
reference data, or feed diagnostics back into a source trajectory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss, compute_residual
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema

if __package__:
    from scripts import issue53_stage6a_protocol as protocol
    from scripts import run_issue53_fixed_alpha_calibration as fixed_inputs
else:
    import issue53_stage6a_protocol as protocol
    import run_issue53_fixed_alpha_calibration as fixed_inputs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATE_LIBRARY_FORMAT = "issue53_stage6a_state_library_v1"
STATE_LIBRARY_SHARD_FORMAT = "issue53_stage6a_state_library_seed_shard_v1"
STATE_LIBRARY_FILENAME = "state_library.json"
STATE_LIBRARY_SHARD_DIRECTORY = "state_library_shards"
NORMAL_TERMINATION_REASONS = (
    "fit_target_reached",
    "early_stopped",
    "resource_cap_reached",
)
SOURCE_BOUNDARY = {
    "kernel": "independent_s0",
    "terminal_current": True,
    "post_proposal_gate": False,
    "reference_table_read": False,
    "offline_metric_read": False,
    "proposal_probe_run": False,
}


def _json_safe(value: Any, path: tuple[str, ...] = ()) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(
                f"strict JSON 禁止 NaN：path={'.'.join(path)}"
            )
        if math.isinf(value):
            if value > 0 and path and path[-1] == "tol":
                return "positive_infinity"
            raise ValueError(
                f"非 tol 字段禁止 infinity：path={'.'.join(path)}"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, path + (str(key),))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path + (str(index),))
            for index, item in enumerate(value)
        ]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), path)
    if hasattr(value, "item"):
        return _json_safe(value.item(), path)
    raise TypeError(f"不可严格 JSON 序列化：{type(value)!r}")


def _strict_load_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON 包含非标准数值：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("状态库 JSON 根必须是 object")
    return value


def _exclusive_write_json(path: str | Path, value: Any) -> Path:
    """Atomically publish strict JSON without replacing an existing path."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"状态库输出已存在，不覆盖：{output}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"状态库输出已存在，不覆盖：{output}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _git_identity(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_clean_including_untracked": not status,
        "status": status,
    }


def _mode_seeds(mode: str) -> tuple[int, ...]:
    if mode == "formal":
        return protocol.FORMAL_SEEDS
    if mode == "smoke":
        return (protocol.SMOKE_SEED,)
    raise ValueError("mode 必须是 formal 或 smoke")


def _select_seeds(
    mode: str,
    selected_seeds: Sequence[int] | None,
) -> tuple[tuple[int, ...], str, str]:
    allowed = _mode_seeds(mode)
    if selected_seeds is None:
        return allowed, "full", STATE_LIBRARY_FORMAT
    selected = tuple(selected_seeds)
    if (
        len(selected) != 1
        or isinstance(selected[0], bool)
        or selected[0] not in allowed
    ):
        raise ValueError(
            "selected_seeds 必须恰好包含一个冻结 seed"
        )
    return selected, "seed_shard", STATE_LIBRARY_SHARD_FORMAT


def _expected_state_ids(seeds: Sequence[int]) -> list[str]:
    return [
        f"{dataset}__seed_{seed}__{group}"
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for group in protocol.STATE_GROUPS
    ]


def _expected_trajectory_keys(
    seeds: Sequence[int],
) -> list[tuple[str, int]]:
    return [
        (dataset, seed)
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
    ]


def _runtime_target(
    raw_target: Sequence[int],
    dataset_name: str,
    mode: str,
) -> np.ndarray:
    dataset = protocol.DATASETS[dataset_name]
    runtime_n = protocol.source_generator_params(
        dataset_name, _mode_seeds(mode)[0], mode=mode
    )["n_records"]
    target = np.asarray(raw_target, dtype=float)
    if runtime_n != dataset["n_records"]:
        target = target * (runtime_n / dataset["n_records"])
    if not np.all(np.isfinite(target)):
        raise RuntimeError(f"{dataset_name} runtime target 非有限")
    if mode == "formal" and not np.all(target == np.rint(target)):
        raise RuntimeError(f"{dataset_name} formal target 必须是整数 counts")
    return target


def _audit_runtime_inputs(
    root: Path,
    mode: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    input_audit: dict[str, Any] = {}
    runtime_inputs: dict[str, Any] = {}
    runtime_targets: dict[str, Any] = {}
    for dataset_name in protocol.DATASET_ORDER:
        observed = fixed_inputs._audit_dataset(root, dataset_name)
        expected = protocol.DATASETS[dataset_name]
        if (
            observed["sha256"] != expected["input_sha256"]
            or observed["query_count"] != expected["query_count"]
            or observed["order_counts"] != expected["query_order_counts"]
            or observed["query_identity_sha256"]
            != expected["query_identity_sha256"]
            or observed["target_vector_sha256"]
            != expected["target_vector_sha256"]
        ):
            raise RuntimeError(f"{dataset_name} Stage 6A 输入身份漂移")
        target = _runtime_target(
            observed["targets"], dataset_name, mode
        )
        input_audit[dataset_name] = {
            key: value
            for key, value in observed.items()
            if key not in {"queries", "targets"}
        }
        runtime_inputs[dataset_name] = observed
        runtime_targets[dataset_name] = {
            "runtime_n_records": int(
                protocol.source_generator_params(
                    dataset_name,
                    _mode_seeds(mode)[0],
                    mode=mode,
                )["n_records"]
            ),
            "target_values": target.tolist(),
            "target_vector_sha256": protocol.canonical_sha256(
                target.tolist()
            ),
            "source_target_vector_sha256": (
                expected["target_vector_sha256"]
            ),
        }
    return (
        _json_safe(input_audit),
        runtime_inputs,
        _json_safe(runtime_targets),
    )


def _validate_formal_execution(
    mode: str,
    git: dict[str, Any],
    confirmed_execution_commit: str | None,
    *,
    require_cuda: bool,
) -> dict[str, Any]:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if mode == "smoke":
        return environment
    if mode != "formal":
        raise ValueError("mode 必须是 formal 或 smoke")
    if not git["worktree_clean_including_untracked"]:
        raise RuntimeError(
            "formal Stage 6A 状态库要求包含 untracked 在内的 clean worktree"
        )
    if (
        not confirmed_execution_commit
        or confirmed_execution_commit != git["commit"]
    ):
        raise PermissionError(
            "formal Stage 6A 必须显式确认当前 clean execution commit"
        )
    if not require_cuda:
        return environment

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise RuntimeError(
            "formal Stage 6A seed shard 必须显式且只暴露一张 GPU"
        )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("formal nltcs CUDA 路径要求 torch") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "formal nltcs CUDA 路径要求进程内恰好一张可用 GPU"
        )
    environment.update({
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
    })
    return environment


def _select_milestones(snapshots: Sequence[dict[str, Any]]) -> list[dict]:
    if len(snapshots) < len(protocol.STATE_GROUPS):
        raise RuntimeError(
            "来源轨迹不足五个互异 natural-work states"
        )
    state_indices = [item.get("state_index") for item in snapshots]
    rounds = [item.get("round") for item in snapshots]
    works = [float(item.get("normalized_work")) for item in snapshots]
    if (
        state_indices != sorted(set(state_indices))
        or rounds != sorted(rounds)
        or works != sorted(works)
        or snapshots[0].get("phase") != "initial"
        or snapshots[0].get("round") != 0
        or snapshots[-1].get("termination_reason")
        not in NORMAL_TERMINATION_REASONS
    ):
        raise RuntimeError(
            "natural-work snapshots 的顺序、initial 或 terminal 身份无效"
        )
    terminal_work = works[-1]
    if not math.isfinite(terminal_work) or terminal_work <= 0.0:
        raise RuntimeError("terminal natural work 必须为正有限值")

    target_works = [
        terminal_work * protocol.STATE_TARGET_FRACTIONS[group]
        for group in protocol.PRIMARY_STATE_GROUPS[:-1]
    ]
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for indices in itertools.combinations(
        range(1, len(snapshots) - 1), 3
    ):
        error = sum(
            abs(works[index] - target)
            for index, target in zip(indices, target_works)
        )
        candidates.append((error, indices))
    if not candidates:
        raise RuntimeError(
            "没有三个互异 interior natural-work states"
        )
    _, interior = min(candidates, key=lambda item: (item[0], item[1]))
    selected_indices = (0, *interior, len(snapshots) - 1)

    selected = []
    for group, index in zip(protocol.STATE_GROUPS, selected_indices):
        target_fraction = protocol.STATE_TARGET_FRACTIONS[group]
        target_work = terminal_work * target_fraction
        selected.append({
            "state_group": group,
            "target_fraction": target_fraction,
            "target_normalized_work": target_work,
            "selection_absolute_work_error": abs(
                works[index] - target_work
            ),
            "source_snapshot_index": index,
            "snapshot": snapshots[index],
        })
    return selected


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _snapshot_frame(snapshot: dict[str, Any]) -> pd.DataFrame:
    columns = snapshot.get("table_columns")
    records = snapshot.get("table_records")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
        or not isinstance(records, list)
    ):
        raise RuntimeError("natural-work snapshot table payload 无效")
    return pd.DataFrame.from_records(records, columns=columns)


def _validate_snapshot(
    snapshot: dict[str, Any],
    *,
    dataset_name: str,
    runtime_n_records: int,
    target: np.ndarray,
    direction_reference_scale: float,
    formal: bool,
) -> np.ndarray:
    query_count = int(protocol.DATASETS[dataset_name]["query_count"])
    frame = _snapshot_frame(snapshot)
    answers = np.asarray(snapshot.get("current_query_answers"))
    residual_signal = np.asarray(snapshot.get("current_residual_signal"))
    if (
        snapshot.get("snapshot_format") != "natural_work_current_v1"
        or len(frame) != runtime_n_records
        or answers.shape != (query_count,)
        or residual_signal.shape != (query_count,)
        or answers.dtype.kind not in "iuf"
        or residual_signal.dtype.kind not in "iuf"
        or not np.all(np.isfinite(answers))
        or not np.all(np.isfinite(residual_signal))
        or fixed_inputs._frame_sha256(frame)
        != snapshot.get("current_table_sha256")
        or not _is_sha256(snapshot.get("primary_rng_state_sha256"))
        or snapshot.get("factorized_gibbs_rng_state_sha256") is not None
        or snapshot.get("direction_reference_scale")
        != direction_reference_scale
    ):
        raise RuntimeError(
            f"{dataset_name} natural-work snapshot 内容或身份无效"
        )
    if formal and not np.all(answers == np.rint(answers)):
        raise RuntimeError(
            f"{dataset_name} formal current query counts 必须为整数"
        )
    count_residual = target - answers.astype(float, copy=False)
    if formal and not np.all(count_residual == np.rint(count_residual)):
        raise RuntimeError(
            f"{dataset_name} formal count residual 必须为整数"
        )
    expected_residual_signal = compute_residual(
        target,
        answers,
        runtime_n_records,
        geometry="relative",
        geometry_floor=8.0,
    )
    expected_loss = compute_loss(target, answers)
    expected_normalized_l1 = float(
        np.mean(np.abs(count_residual)) / runtime_n_records
    )
    if (
        not np.array_equal(
            residual_signal.astype(float, copy=False),
            expected_residual_signal,
        )
        or snapshot.get("current_squared_loss") != expected_loss
        or snapshot.get("current_normalized_l1")
        != expected_normalized_l1
    ):
        raise RuntimeError(
            f"{dataset_name} snapshot query diagnostics 重算失败"
        )
    return count_residual


def _validate_run(
    dataset_name: str,
    seed: int,
    output: pd.DataFrame,
    diagnostics: dict[str, Any],
    snapshots: Sequence[dict[str, Any]],
    *,
    mode: str,
    target: np.ndarray,
) -> None:
    expected = protocol.source_generator_params(
        dataset_name, seed, mode=mode
    )
    params = diagnostics.get("params", {})
    rounds = diagnostics.get("rounds_run")
    direction_scale = diagnostics.get("direction_reference_scale")
    gates = {
        "terminal_current": (
            diagnostics.get("output_table_identity")
            == "terminal_current"
        ),
        "normal_termination": (
            diagnostics.get("termination_reason")
            in NORMAL_TERMINATION_REASONS
        ),
        "rounds_type": (
            isinstance(rounds, int)
            and not isinstance(rounds, bool)
            and 0 < rounds <= expected["n_rounds"]
        ),
        "all_proposals_applied": (
            diagnostics.get("accept_history") == [True] * rounds
            if isinstance(rounds, int) and rounds >= 0
            else False
        ),
        "one_attempt_per_round": (
            diagnostics.get("proposal_attempts_history") == [1] * rounds
            if isinstance(rounds, int) and rounds >= 0
            else False
        ),
        "first_attempt_applied": (
            diagnostics.get("accepted_attempt_history") == [1] * rounds
            if isinstance(rounds, int) and rounds >= 0
            else False
        ),
        "candidate_count": (
            diagnostics.get("candidate_evaluation_count") == rounds
        ),
        "transition_count": (
            diagnostics.get("transition_clock_count") == rounds
        ),
        "seed": params.get("seed") == seed,
        "n_records": params.get("n_records") == expected["n_records"],
        "tolerance": (
            isinstance(params.get("tol"), float)
            and math.isinf(params["tol"])
            and params["tol"] > 0
        ),
        "no_retry": params.get("max_retries") == 0,
        "rho": params.get("rho") == protocol.RHO,
        "eta": params.get("eta") == protocol.ETA,
        "mu": params.get("mu") == protocol.TRAJECTORY_MU,
        "fixed_alpha": params.get("fixed_alpha") == protocol.FIXED_ALPHA,
        "independent_kernel": (
            params.get("factorized_gibbs_sweeps") == 0
        ),
        "scale_invariant": (
            params.get("selection_scale_invariant") is True
        ),
        "relative_residual": (
            params.get("residual_geometry") == "relative"
            and params.get("residual_geometry_floor") == 8.0
        ),
        "direction": (
            params.get("residual_directed_diffusion") is True
            and params.get("diffusion_direction_strength")
            == protocol.TAU
            and params.get("diffusion_direction_normalization")
            == "initial_rms"
        ),
        "no_annealing_or_cooling": (
            params.get("rho_anneal_end") is None
            and params.get("residual_self_cooling") is None
        ),
        "snapshots_enabled": (
            params.get("record_natural_work_snapshots") is True
        ),
        "stationarity_disabled": (
            params.get("record_stationarity_trace") is False
        ),
        "patience": (
            params.get("inner_early_stopping_patience_ticks")
            == protocol.PATIENCE_TICKS
        ),
        "direction_scale": (
            isinstance(direction_scale, float)
            and math.isfinite(direction_scale)
            and direction_scale > 0.0
        ),
        "snapshot_count": len(snapshots) >= len(protocol.STATE_GROUPS),
        "output_equals_final": (
            fixed_inputs._frame_sha256(output)
            == fixed_inputs._frame_sha256(
                diagnostics.get("final_table")
            )
            if isinstance(diagnostics.get("final_table"), pd.DataFrame)
            else False
        ),
        "terminal_snapshot_hash": (
            bool(snapshots)
            and fixed_inputs._frame_sha256(output)
            == snapshots[-1].get("current_table_sha256")
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(
            f"{dataset_name}/seed{seed} source trajectory 身份失败："
            f"{failed}"
        )

    for snapshot in snapshots:
        _validate_snapshot(
            snapshot,
            dataset_name=dataset_name,
            runtime_n_records=expected["n_records"],
            target=target,
            direction_reference_scale=direction_scale,
            formal=mode == "formal",
        )
    snapshot_candidate_counts = [
        item.get("candidate_evaluation_count_cumulative")
        for item in snapshots
    ]
    snapshot_participating_rows = [
        item.get("cumulative_participating_rows")
        for item in snapshots
    ]
    snapshot_works = [item.get("normalized_work") for item in snapshots]
    if (
        snapshot_candidate_counts
        != sorted(snapshot_candidate_counts)
        or snapshot_candidate_counts[0] != 0
        or snapshot_candidate_counts[-1] != rounds
        or snapshot_participating_rows
        != sorted(snapshot_participating_rows)
        or any(
            work != participating_rows / expected["n_records"]
            for work, participating_rows in zip(
                snapshot_works, snapshot_participating_rows
            )
        )
    ):
        raise RuntimeError(
            f"{dataset_name}/seed{seed} snapshot natural-work clock 失败"
        )
    _select_milestones(snapshots)


def _trajectory_scientific_payload(
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "dataset",
        "seed",
        "runtime_n_records",
        "runtime_device",
        "runtime_target_sha256",
        "source_generator_params",
        "initial_table_sha256",
        "primary_rng_post_initialization_state_sha256",
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
    return {key: trajectory[key] for key in keys}


def _state_scientific_payload(
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_id": state["state_id"],
        "dataset": state["dataset"],
        "seed": state["seed"],
        "state_group": state["state_group"],
        "target_fraction": state["target_fraction"],
        "target_normalized_work": state["target_normalized_work"],
        "selection_absolute_work_error": state[
            "selection_absolute_work_error"
        ],
        "source_snapshot_index": state["source_snapshot_index"],
        "current_count_residual": state["current_count_residual"],
        "snapshot": state["snapshot"],
    }


def scientific_payload(library: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_library_format": library["state_library_format"],
        "mode": library["mode"],
        "artifact_scope": library["artifact_scope"],
        "selected_seeds": library["selected_seeds"],
        "protocol_sha256": library["protocol_sha256"],
        "input_audit": library["input_audit"],
        "runtime_targets": library["runtime_targets"],
        "trajectories": [
            _trajectory_scientific_payload(row)
            for row in library["trajectories"]
        ],
        "states": [
            _state_scientific_payload(row)
            for row in library["states"]
        ],
    }


def _validate_library_structure(
    library: dict[str, Any],
    *,
    mode: str,
    artifact_scope: str,
    selected_seeds: Sequence[int],
) -> None:
    expected_format = (
        STATE_LIBRARY_FORMAT
        if artifact_scope == "full"
        else STATE_LIBRARY_SHARD_FORMAT
    )
    expected_ids = _expected_state_ids(selected_seeds)
    expected_trajectories = _expected_trajectory_keys(selected_seeds)
    manifest = library.get("manifest")
    if (
        library.get("state_library_format") != expected_format
        or library.get("status") != "complete"
        or library.get("mode") != mode
        or library.get("artifact_scope") != artifact_scope
        or library.get("selected_seeds") != list(selected_seeds)
        or library.get("protocol")
        != protocol.frozen_protocol_manifest()
        or library.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or not isinstance(manifest, dict)
        or manifest.get("state_count") != len(expected_ids)
        or manifest.get("state_ids_in_fixed_order") != expected_ids
        or manifest.get("dataset_order")
        != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(selected_seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or [row.get("state_id") for row in library.get("states", [])]
        != expected_ids
        or [
            (row.get("dataset"), row.get("seed"))
            for row in library.get("trajectories", [])
        ]
        != expected_trajectories
    ):
        raise RuntimeError("Stage 6A 状态库结构、覆盖或协议绑定失败")
    source_shards = manifest.get("source_seed_shard_sha256")
    if artifact_scope == "seed_shard" and source_shards != {}:
        raise RuntimeError("seed shard 不得递归绑定 source shards")
    if artifact_scope == "full" and not isinstance(source_shards, dict):
        raise RuntimeError("full 状态库 source shard manifest 无效")

    input_audit = library.get("input_audit")
    if (
        not isinstance(input_audit, dict)
        or set(input_audit) != set(protocol.DATASET_ORDER)
    ):
        raise RuntimeError("Stage 6A input audit 覆盖失败")
    for dataset_name in protocol.DATASET_ORDER:
        row = input_audit[dataset_name]
        expected_dataset = protocol.DATASETS[dataset_name]
        try:
            observed_order_counts = {
                int(order): count
                for order, count in row["order_counts"].items()
            }
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"{dataset_name} input audit order counts 无效"
            ) from error
        if (
            row.get("sha256") != expected_dataset["input_sha256"]
            or row.get("query_count")
            != expected_dataset["query_count"]
            or row.get("query_identity_sha256")
            != expected_dataset["query_identity_sha256"]
            or row.get("target_vector_sha256")
            != expected_dataset["target_vector_sha256"]
            or observed_order_counts
            != expected_dataset["query_order_counts"]
        ):
            raise RuntimeError(f"{dataset_name} input audit 身份失败")

    runtime_targets = library.get("runtime_targets")
    if (
        not isinstance(runtime_targets, dict)
        or set(runtime_targets) != set(protocol.DATASET_ORDER)
    ):
        raise RuntimeError("Stage 6A runtime targets 覆盖失败")
    targets: dict[str, np.ndarray] = {}
    for dataset_name in protocol.DATASET_ORDER:
        row = runtime_targets[dataset_name]
        values = row.get("target_values")
        target = np.asarray(values)
        expected_n_records = protocol.source_generator_params(
            dataset_name, _mode_seeds(mode)[0], mode=mode
        )["n_records"]
        if (
            target.shape
            != (protocol.DATASETS[dataset_name]["query_count"],)
            or target.dtype.kind not in "iuf"
            or not np.all(np.isfinite(target))
            or row.get("runtime_n_records") != expected_n_records
            or row.get("source_target_vector_sha256")
            != protocol.DATASETS[dataset_name][
                "target_vector_sha256"
            ]
            or row.get("target_vector_sha256")
            != protocol.canonical_sha256(list(values))
            or (
                mode == "formal"
                and not np.all(target == np.rint(target))
            )
        ):
            raise RuntimeError(
                f"{dataset_name} runtime target manifest 无效"
            )
        targets[dataset_name] = target.astype(float, copy=False)

    if library.get("source_boundary") != SOURCE_BOUNDARY:
        raise RuntimeError("Stage 6A measured-only/no-gate source boundary 失败")

    trajectories = library["trajectories"]
    state_by_id = {
        row["state_id"]: row for row in library["states"]
    }
    for trajectory in trajectories:
        dataset_name = trajectory["dataset"]
        seed = trajectory["seed"]
        expected_params = _json_safe(
            protocol.source_generator_params(
                dataset_name, seed, mode=mode
            )
        )
        expected_selected_ids = [
            f"{dataset_name}__seed_{seed}__{group}"
            for group in protocol.STATE_GROUPS
        ]
        scale = trajectory.get("direction_reference_scale")
        snapshot_manifest = trajectory.get(
            "natural_work_snapshot_manifest"
        )
        if (
            trajectory.get("runtime_n_records")
            != expected_params["n_records"]
            or trajectory.get("runtime_device")
            != expected_params["device"]
            or trajectory.get("runtime_target_sha256")
            != runtime_targets[dataset_name]["target_vector_sha256"]
            or trajectory.get("source_generator_params")
            != expected_params
            or not _is_sha256(
                trajectory.get("initial_table_sha256")
            )
            or not _is_sha256(
                trajectory.get(
                    "primary_rng_post_initialization_state_sha256"
                )
            )
            or not _is_sha256(
                trajectory.get("terminal_table_sha256")
            )
            or not _is_sha256(
                trajectory.get("primary_rng_endpoint_sha256")
            )
            or trajectory.get("candidate_evaluations")
            != trajectory.get("rounds_run")
            or trajectory.get("termination_reason")
            not in NORMAL_TERMINATION_REASONS
            or not isinstance(scale, float)
            or not math.isfinite(scale)
            or scale <= 0.0
            or trajectory.get("selected_state_ids")
            != expected_selected_ids
            or not isinstance(snapshot_manifest, list)
            or trajectory.get("recorded_natural_work_state_count")
            != len(snapshot_manifest)
        ):
            raise RuntimeError(
                f"{dataset_name}/seed{seed} trajectory artifact 身份失败"
            )
        selected = _select_milestones(snapshot_manifest)
        for expected_item, state_id in zip(
            selected, expected_selected_ids
        ):
            state = state_by_id[state_id]
            snapshot = state["snapshot"]
            source_snapshot = snapshot_manifest[
                expected_item["source_snapshot_index"]
            ]
            if (
                state["state_group"]
                != expected_item["state_group"]
                or state["target_fraction"]
                != expected_item["target_fraction"]
                or state["target_normalized_work"]
                != expected_item["target_normalized_work"]
                or state["selection_absolute_work_error"]
                != expected_item["selection_absolute_work_error"]
                or state["source_snapshot_index"]
                != expected_item["source_snapshot_index"]
                or any(
                    snapshot.get(key) != source_snapshot.get(key)
                    for key in (
                        "state_index",
                        "round",
                        "phase",
                        "normalized_work",
                        "termination_reason",
                        "current_table_sha256",
                        "primary_rng_state_sha256",
                        "direction_reference_scale",
                    )
                )
            ):
                raise RuntimeError(
                    f"{state_id} milestone selection 身份失败"
                )

    trajectory_scales = {
        (row["dataset"], row["seed"]): row[
            "direction_reference_scale"
        ]
        for row in library["trajectories"]
    }
    for state in library["states"]:
        dataset_name = state["dataset"]
        seed = state["seed"]
        target = targets[dataset_name]
        snapshot = state.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"{state['state_id']} snapshot 缺失")
        residual = _validate_snapshot(
            snapshot,
            dataset_name=dataset_name,
            runtime_n_records=runtime_targets[dataset_name][
                "runtime_n_records"
            ],
            target=target,
            direction_reference_scale=trajectory_scales[
                (dataset_name, seed)
            ],
            formal=mode == "formal",
        )
        stored_residual = np.asarray(state.get("current_count_residual"))
        if (
            stored_residual.shape != residual.shape
            or not np.array_equal(stored_residual, residual)
        ):
            raise RuntimeError(
                f"{state['state_id']} count residual 身份失败"
            )

    observed_scientific_sha = protocol.canonical_sha256(
        scientific_payload(library)
    )
    if (
        library.get("state_library_scientific_sha256")
        != observed_scientific_sha
    ):
        raise RuntimeError("Stage 6A 状态库 scientific SHA-256 失败")
    expected_formal = mode == "formal"
    if library.get("formal_result_valid") is not expected_formal:
        raise RuntimeError("Stage 6A formal_result_valid 身份失败")


def build_plan(mode: str) -> dict[str, Any]:
    observed_protocol_sha = protocol.protocol_sha256()
    if observed_protocol_sha != protocol.FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "Stage 6A frozen protocol manifest SHA-256 漂移"
        )
    seeds = _mode_seeds(mode)
    return {
        "mode": "plan_only_no_input_read_no_generation",
        "requested_mode": mode,
        "formal_result_valid": False,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "state_library_format": STATE_LIBRARY_FORMAT,
        "dataset_order": list(protocol.DATASET_ORDER),
        "seed_order": list(seeds),
        "state_group_order": list(protocol.STATE_GROUPS),
        "trajectory_count": len(protocol.DATASET_ORDER) * len(seeds),
        "state_count": (
            len(protocol.DATASET_ORDER)
            * len(seeds)
            * len(protocol.STATE_GROUPS)
        ),
        "seed_shards": [
            {
                "shard_index": index,
                "seed": seed,
                "trajectory_count": len(protocol.DATASET_ORDER),
                "state_count": (
                    len(protocol.DATASET_ORDER)
                    * len(protocol.STATE_GROUPS)
                ),
            }
            for index, seed in enumerate(seeds)
        ],
        "generation_started": False,
        "formal_confirmation_consumed": False,
    }


def build_state_library(
    mode: str,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
    selected_seeds: Sequence[int] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Collect all frozen states or one deterministic seed shard."""

    protocol.require_formal_confirmation(
        mode, confirmed_protocol_sha256
    )
    seeds, artifact_scope, library_format = _select_seeds(
        mode, selected_seeds
    )
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"状态库输出已存在，不覆盖：{output}")
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    git = _git_identity()
    environment = _validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=True,
    )
    input_audit, runtime_inputs, runtime_targets = (
        _audit_runtime_inputs(REPOSITORY_ROOT, mode)
    )
    input_audit = _json_safe(input_audit)
    runtime_targets = _json_safe(runtime_targets)

    trajectories: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    started = time.perf_counter()
    for dataset_name in protocol.DATASET_ORDER:
        dataset = protocol.DATASETS[dataset_name]
        observed = runtime_inputs[dataset_name]
        target = np.asarray(
            runtime_targets[dataset_name]["target_values"],
            dtype=float,
        )
        schema = load_schema(str(REPOSITORY_ROOT / dataset["schema"]))
        queries = load_queries(str(REPOSITORY_ROOT / dataset["queries"]))
        marginals = load_marginals(
            str(REPOSITORY_ROOT / dataset["marginals"])
        )
        if len(queries) != dataset["query_count"]:
            raise RuntimeError(f"{dataset_name} runtime query count 漂移")
        if observed["queries"] != queries:
            raise RuntimeError(
                f"{dataset_name} audited/runtime ordered queries 漂移"
            )

        for seed in seeds:
            expected_params = protocol.source_generator_params(
                dataset_name, seed, mode=mode
            )
            n_records = int(expected_params.pop("n_records"))
            trajectory_started = time.perf_counter()
            output_table, diagnostics = run_evolution(
                target,
                queries,
                schema,
                n_records=n_records,
                marginals=marginals,
                **expected_params,
            )
            elapsed = time.perf_counter() - trajectory_started
            final_table = diagnostics.get("final_table")
            snapshots = diagnostics.get("natural_work_snapshots")
            if (
                not isinstance(final_table, pd.DataFrame)
                or not isinstance(snapshots, list)
            ):
                raise RuntimeError(
                    f"{dataset_name}/seed{seed} 未返回 final table/snapshots"
                )
            pd.testing.assert_frame_equal(
                output_table.reset_index(drop=True),
                final_table.reset_index(drop=True),
            )
            _validate_run(
                dataset_name,
                seed,
                final_table,
                diagnostics,
                snapshots,
                mode=mode,
                target=target,
            )
            selected = _select_milestones(snapshots)
            selected_state_ids = []
            direction_scale = float(
                diagnostics["direction_reference_scale"]
            )
            for item in selected:
                group = item["state_group"]
                state_identifier = (
                    f"{dataset_name}__seed_{seed}__{group}"
                )
                selected_state_ids.append(state_identifier)
                snapshot = _json_safe(item["snapshot"])
                count_residual = _validate_snapshot(
                    snapshot,
                    dataset_name=dataset_name,
                    runtime_n_records=n_records,
                    target=target,
                    direction_reference_scale=direction_scale,
                    formal=mode == "formal",
                )
                states.append({
                    "state_id": state_identifier,
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
                    "current_count_residual": (
                        count_residual.tolist()
                    ),
                    "snapshot": snapshot,
                })

            terminal = snapshots[-1]
            source_params = protocol.source_generator_params(
                dataset_name, seed, mode=mode
            )
            trajectory = {
                "dataset": dataset_name,
                "seed": int(seed),
                "runtime_n_records": n_records,
                "runtime_device": source_params["device"],
                "runtime_target_sha256": runtime_targets[dataset_name][
                    "target_vector_sha256"
                ],
                "source_generator_params": _json_safe(source_params),
                "initial_table_sha256": diagnostics[
                    "initial_table_sha256"
                ],
                "primary_rng_post_initialization_state_sha256": (
                    diagnostics[
                        "primary_rng_post_initialization_state_sha256"
                    ]
                ),
                "terminal_table_sha256": fixed_inputs._frame_sha256(
                    final_table
                ),
                "rounds_run": int(diagnostics["rounds_run"]),
                "candidate_evaluations": int(
                    diagnostics["candidate_evaluation_count"]
                ),
                "termination_reason": diagnostics[
                    "termination_reason"
                ],
                "terminal_normalized_work": float(
                    terminal["normalized_work"]
                ),
                "terminal_squared_loss": float(
                    diagnostics["final_current_squared_loss"]
                ),
                "terminal_normalized_l1": float(
                    diagnostics["final_current_normalized_l1"]
                ),
                "direction_reference_scale": direction_scale,
                "primary_rng_endpoint_sha256": diagnostics[
                    "primary_rng_state_sha256"
                ],
                "recorded_natural_work_state_count": len(snapshots),
                "natural_work_snapshot_manifest": [
                    {
                        key: snapshot[key]
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
                            "candidate_evaluation_count_cumulative",
                            "direction_reference_scale",
                        )
                    }
                    for snapshot in snapshots
                ],
                "selected_state_ids": selected_state_ids,
                "elapsed_sec_diagnostic_only": elapsed,
            }
            trajectories.append(_json_safe(trajectory))
            print(
                f"[Stage6A {mode} {dataset_name} seed={seed}] "
                f"{trajectory['termination_reason']} "
                f"rounds={trajectory['rounds_run']} "
                f"work={trajectory['terminal_normalized_work']:.4f} "
                f"states={len(snapshots)} elapsed={elapsed:.2f}s",
                flush=True,
            )

    expected_ids = _expected_state_ids(seeds)
    if (
        [row["state_id"] for row in states] != expected_ids
        or [
            (row["dataset"], row["seed"]) for row in trajectories
        ]
        != _expected_trajectory_keys(seeds)
    ):
        raise RuntimeError("Stage 6A state/trajectory 固定顺序失败")

    library = {
        "state_library_format": library_format,
        "status": "complete",
        "mode": mode,
        "artifact_scope": artifact_scope,
        "selected_seeds": list(seeds),
        "formal_result_valid": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": _json_safe(environment),
        "input_audit": input_audit,
        "runtime_targets": runtime_targets,
        "source_boundary": dict(SOURCE_BOUNDARY),
        "trajectories": trajectories,
        "states": states,
        "manifest": {
            "state_count": len(states),
            "state_ids_in_fixed_order": expected_ids,
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "source_seed_shard_sha256": {},
        },
        "elapsed_sec_diagnostic_only": (
            time.perf_counter() - started
        ),
    }
    library["state_library_scientific_sha256"] = (
        protocol.canonical_sha256(scientific_payload(library))
    )
    _validate_library_structure(
        library,
        mode=mode,
        artifact_scope=artifact_scope,
        selected_seeds=seeds,
    )
    published = _exclusive_write_json(output, library)
    print(f"Stage 6A 状态库：{published}", flush=True)
    return published, library


def aggregate_state_library_shards(
    mode: str,
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Combine exactly one seed shard for every frozen mode seed."""

    protocol.require_formal_confirmation(
        mode, confirmed_protocol_sha256
    )
    seeds = _mode_seeds(mode)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"状态库输出已存在，不覆盖：{output}")
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    git = _git_identity()
    environment = _validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=False,
    )
    expected_input_audit, _, expected_runtime_targets = (
        _audit_runtime_inputs(REPOSITORY_ROOT, mode)
    )
    expected_input_audit = _json_safe(expected_input_audit)
    expected_runtime_targets = _json_safe(expected_runtime_targets)

    indexed: dict[int, dict[str, Any]] = {}
    indexed_paths: dict[int, Path] = {}
    input_audit = None
    runtime_targets = None
    elapsed = 0.0
    source_environments = {}
    for raw_path in shard_paths:
        path = Path(raw_path).resolve()
        shard = _strict_load_json(path)
        shard_seeds = shard.get("selected_seeds")
        if (
            not isinstance(shard_seeds, list)
            or len(shard_seeds) != 1
            or shard_seeds[0] not in seeds
        ):
            raise RuntimeError(f"Stage 6A seed shard seed 无效：{path}")
        seed = shard_seeds[0]
        if seed in indexed:
            raise RuntimeError(f"Stage 6A seed shard 重复：{seed}")
        _validate_library_structure(
            shard,
            mode=mode,
            artifact_scope="seed_shard",
            selected_seeds=(seed,),
        )
        shard_git = shard.get("git", {})
        if (
            shard_git.get("commit") != git["commit"]
            or (
                mode == "formal"
                and (
                    not shard_git.get(
                        "worktree_clean_including_untracked"
                    )
                    or shard.get("formal_result_valid") is not True
                )
            )
        ):
            raise RuntimeError(
                f"Stage 6A seed shard execution identity 失败：{path}"
            )
        if input_audit is None:
            input_audit = shard["input_audit"]
            runtime_targets = shard["runtime_targets"]
        elif (
            shard["input_audit"] != input_audit
            or shard["runtime_targets"] != runtime_targets
        ):
            raise RuntimeError("Stage 6A seed shards 输入/target 不一致")
        indexed[seed] = shard
        indexed_paths[seed] = path
        source_environments[str(seed)] = shard["environment"]
        elapsed += float(shard["elapsed_sec_diagnostic_only"])

    if set(indexed) != set(seeds):
        raise RuntimeError(
            "Stage 6A seed shards 必须恰好覆盖全部冻结 seeds"
        )
    if (
        input_audit != expected_input_audit
        or runtime_targets != expected_runtime_targets
    ):
        raise RuntimeError(
            "Stage 6A seed shards 与当前冻结 measured inputs 不一致"
        )
    trajectory_index = {
        (row["dataset"], row["seed"]): row
        for shard in indexed.values()
        for row in shard["trajectories"]
    }
    state_index = {
        row["state_id"]: row
        for shard in indexed.values()
        for row in shard["states"]
    }
    expected_trajectory_keys = _expected_trajectory_keys(seeds)
    expected_state_ids = _expected_state_ids(seeds)
    if (
        set(trajectory_index) != set(expected_trajectory_keys)
        or set(state_index) != set(expected_state_ids)
    ):
        raise RuntimeError(
            "Stage 6A seed shards trajectory/state 覆盖失败"
        )
    trajectories = [
        trajectory_index[key] for key in expected_trajectory_keys
    ]
    states = [state_index[key] for key in expected_state_ids]

    library = {
        "state_library_format": STATE_LIBRARY_FORMAT,
        "status": "complete",
        "mode": mode,
        "artifact_scope": "full",
        "selected_seeds": list(seeds),
        "formal_result_valid": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": {
            "aggregation": _json_safe(environment),
            "source_seed_shards": source_environments,
        },
        "input_audit": input_audit,
        "runtime_targets": runtime_targets,
        "source_boundary": dict(SOURCE_BOUNDARY),
        "trajectories": trajectories,
        "states": states,
        "manifest": {
            "state_count": len(states),
            "state_ids_in_fixed_order": expected_state_ids,
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "source_seed_shard_sha256": {
                str(seed): protocol.file_sha256(indexed_paths[seed])
                for seed in seeds
            },
        },
        "elapsed_sec_diagnostic_only": elapsed,
    }
    library["state_library_scientific_sha256"] = (
        protocol.canonical_sha256(scientific_payload(library))
    )
    _validate_library_structure(
        library,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    published = _exclusive_write_json(output, library)
    print(f"Stage 6A 聚合状态库：{published}", flush=True)
    return published, library


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--shard-index", type=int)
    collect_parser.add_argument("--confirmed-protocol-sha256")
    collect_parser.add_argument("--confirmed-execution-commit")

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--shards", nargs="+", required=True)
    aggregate_parser.add_argument("--confirmed-protocol-sha256")
    aggregate_parser.add_argument("--confirmed-execution-commit")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(
            json.dumps(
                build_plan(args.mode),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        return
    if args.command == "aggregate":
        aggregate_state_library_shards(
            args.mode,
            args.shards,
            args.output,
            confirmed_protocol_sha256=(
                args.confirmed_protocol_sha256
            ),
            confirmed_execution_commit=(
                args.confirmed_execution_commit
            ),
        )
        return

    selected = None
    if args.shard_index is not None:
        seeds = _mode_seeds(args.mode)
        if not 0 <= args.shard_index < len(seeds):
            raise ValueError("--shard-index 超出冻结 seed 范围")
        selected = (seeds[args.shard_index],)
    build_state_library(
        args.mode,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
        selected_seeds=selected,
    )


if __name__ == "__main__":
    main()
