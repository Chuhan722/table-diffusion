"""Issue #52 CPU trajectory execution at independent ``(seed, config)`` granularity.

This module is infrastructure only.  It does not define an experiment grid or
run any research seeds; future Stage T/B runners construct the frozen task list
and call :func:`run_trajectory_tasks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Literal, Optional, Sequence

import numpy as np

if __package__:
    from scripts import compare_factorized_gibbs_unfiltered as trajectory
else:
    import compare_factorized_gibbs_unfiltered as trajectory
from table_diffevo.experiment_parallel import run_ordered_process_tasks


Kernel = Literal["independent", "factor"]


@dataclass(frozen=True)
class TrajectoryTask:
    """One deterministic CPU trajectory addressed by seed and configuration."""

    config_id: str
    kernel: Kernel
    seed: int
    rounds: int
    temperature: float
    sweeps: int
    factor_builder: str = "legacy_rowwise"
    record_state_hashes: bool = False
    snapshot_rounds: Optional[tuple[int, ...]] = None

    @property
    def task_id(self) -> str:
        return f"seed_{self.seed}__{self.config_id}"


def _validate_task(task: TrajectoryTask) -> None:
    if not isinstance(task, TrajectoryTask):
        raise TypeError(f"任务必须是 TrajectoryTask，得到 {type(task)!r}")
    if not task.config_id or not isinstance(task.config_id, str):
        raise ValueError("config_id 必须是非空字符串")
    if task.kernel not in ("independent", "factor"):
        raise ValueError(f"未知 kernel：{task.kernel!r}")
    if (
        isinstance(task.seed, (bool, np.bool_))
        or not isinstance(task.seed, (int, np.integer))
        or task.seed < 0
    ):
        raise ValueError(f"seed 必须是非负整数，得到 {task.seed!r}")
    if (
        isinstance(task.rounds, (bool, np.bool_))
        or not isinstance(task.rounds, (int, np.integer))
        or task.rounds < 1
    ):
        raise ValueError(f"rounds 必须是正整数，得到 {task.rounds!r}")
    if (
        isinstance(task.temperature, (bool, np.bool_))
        or not isinstance(task.temperature, (int, float, np.integer, np.floating))
        or not math.isfinite(float(task.temperature))
        or task.temperature <= 0
    ):
        raise ValueError(
            f"temperature 必须是正有限数，得到 {task.temperature!r}"
        )
    if (
        isinstance(task.sweeps, (bool, np.bool_))
        or not isinstance(task.sweeps, (int, np.integer))
        or task.sweeps < 0
    ):
        raise ValueError(f"sweeps 必须是非负整数，得到 {task.sweeps!r}")
    if task.kernel == "independent" and task.sweeps != 0:
        raise ValueError("independent 任务的 sweeps 必须为 0")
    if task.kernel == "factor" and task.sweeps <= 0:
        raise ValueError("factor 任务的 sweeps 必须为正整数")
    if task.factor_builder not in ("legacy_rowwise", "compiled_batch"):
        raise ValueError(f"未知 factor_builder：{task.factor_builder!r}")
    if task.snapshot_rounds is not None:
        values = task.snapshot_rounds
        if (
            not isinstance(values, tuple)
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in values
            )
            or len(values) != len(set(values))
            or tuple(sorted(values)) != values
            or any(value < 0 or value > task.rounds for value in values)
        ):
            raise ValueError(
                "snapshot_rounds 必须是 0..rounds 内、严格递增且不重复的"
                "整数 tuple"
            )


def _execute_trajectory_task(
    task: TrajectoryTask,
    *,
    target,
    queries,
    schema,
    marginals,
):
    run = trajectory._run_one(
        target,
        queries,
        schema,
        marginals,
        seed=int(task.seed),
        rounds=int(task.rounds),
        temperature=float(task.temperature),
        sweeps=int(task.sweeps),
        device="numpy",
        factor_builder=task.factor_builder,
        record_state_hashes=bool(task.record_state_hashes),
        snapshot_rounds=task.snapshot_rounds,
    )
    return {
        "task_id": task.task_id,
        "config_id": task.config_id,
        "kernel": task.kernel,
        "run": run,
    }


def run_trajectory_tasks(
    target,
    queries,
    schema,
    marginals,
    tasks: Sequence[TrajectoryTask],
    *,
    max_workers: int,
):
    """Execute Issue #52 NumPy/CPU trajectories in stable task order."""
    task_list = list(tasks)
    for task in task_list:
        _validate_task(task)
    task_ids = [task.task_id for task in task_list]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("(seed, config_id) 任务身份不得重复")

    worker = partial(
        _execute_trajectory_task,
        target=target,
        queries=queries,
        schema=schema,
        marginals=marginals,
    )
    rows = run_ordered_process_tasks(
        worker,
        task_list,
        max_workers=max_workers,
    )
    if [row.get("task_id") for row in rows] != task_ids:
        raise RuntimeError("并行执行返回顺序或任务身份发生变化")
    for task, row in zip(task_list, rows):
        run = row.get("run", {})
        if (
            row.get("config_id") != task.config_id
            or row.get("kernel") != task.kernel
            or run.get("seed") != task.seed
            or run.get("temperature") != float(task.temperature)
            or run.get("sweeps") != task.sweeps
        ):
            raise RuntimeError(f"任务结果身份不一致：{task.task_id}")
    return rows
