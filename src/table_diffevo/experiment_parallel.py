"""Deterministic process-level execution helpers for research trajectories.

The helpers in this module parallelize independent tasks only.  They preserve
the caller's task order and deliberately use the ``spawn`` start method so a
worker cannot inherit mutable RNG or numerical-library state from its parent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")

MAX_EXPERIMENT_WORKERS = 8


def validate_max_workers(max_workers: int) -> int:
    """Return a validated worker count in the frozen Issue #52 range 1..8."""
    if (
        isinstance(max_workers, (bool, np.bool_))
        or not isinstance(max_workers, (int, np.integer))
        or not 1 <= int(max_workers) <= MAX_EXPERIMENT_WORKERS
    ):
        raise ValueError(
            "max_workers 必须是 1..8 的整数，"
            f"得到 {max_workers!r}"
        )
    return int(max_workers)


def run_ordered_process_tasks(
    worker: Callable[[TaskT], ResultT],
    tasks: Iterable[TaskT],
    *,
    max_workers: int,
) -> list[ResultT]:
    """Run picklable tasks and return results in the original task order.

    ``max_workers=1`` is the canonical serial path.  Counts above one use a
    spawn-based :class:`~concurrent.futures.ProcessPoolExecutor`; only the wall
    clock and completion order may then differ.  The returned list always
    follows the input order, independently of actual completion order.
    """
    workers = validate_max_workers(max_workers)
    task_list = list(tasks)
    if not task_list:
        return []
    if workers == 1 or len(task_list) == 1:
        return [worker(task) for task in task_list]

    context = multiprocessing.get_context("spawn")
    effective_workers = min(workers, len(task_list))
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=context,
    ) as executor:
        futures = [executor.submit(worker, task) for task in task_list]
        results: list[ResultT] = []
        try:
            for index, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    for pending in futures[index + 1:]:
                        pending.cancel()
                    raise RuntimeError(
                        f"并行任务 index={index} 执行失败："
                        f"{task_list[index]!r}"
                    ) from exc
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
    return results


def _is_runtime_field(key: object) -> bool:
    return isinstance(key, str) and (
        key == "elapsed_sec" or key.endswith("_elapsed_sec")
    )


def scientific_payload(value: Any) -> Any:
    """Return a JSON-compatible view with wall-clock fields removed.

    Scientific equivalence is intentionally strict: task order, trajectories,
    hashes, RNG endpoints, diagnostics, and every other non-timing field must
    match exactly.  Only keys named ``elapsed_sec`` or ending in
    ``_elapsed_sec`` are ignored.
    """
    if isinstance(value, Mapping):
        return {
            str(key): scientific_payload(item)
            for key, item in value.items()
            if not _is_runtime_field(key)
        }
    if isinstance(value, np.ndarray):
        return scientific_payload(value.tolist())
    if isinstance(value, np.generic):
        return scientific_payload(value.item())
    if isinstance(value, (list, tuple)):
        return [scientific_payload(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def scientific_sha256(value: Any) -> str:
    """Hash all scientific fields after removing wall-clock diagnostics."""
    encoded = json.dumps(
        scientific_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_scientifically_equal(expected: Any, actual: Any) -> str:
    """Raise if two serial/parallel outputs differ outside timing fields."""
    expected_payload = scientific_payload(expected)
    actual_payload = scientific_payload(actual)
    if expected_payload != actual_payload:
        raise AssertionError(
            "串行与并行科学字段不一致："
            f"serial={scientific_sha256(expected)}，"
            f"parallel={scientific_sha256(actual)}"
        )
    return scientific_sha256(expected)
