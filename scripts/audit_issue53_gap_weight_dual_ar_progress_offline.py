#!/usr/bin/env python3
"""只读重算旧 A/R 轨迹在相对初始进度坐标下的通道关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import issue53_gap_weight_dual_ar_screen_protocol as prior


CONTRACT_VERSION = "issue53-gap-weight-dual-ar-progress-offline-audit-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _targets(repository_root: Path, dataset: str) -> list[int]:
    spec = prior.DATASETS[dataset]
    path = repository_root / spec["queries"]
    if file_sha256(path) != spec["input_sha256"]["queries"]:
        raise RuntimeError(f"{dataset} 查询文件 SHA-256 漂移")
    rows = json.loads(path.read_text(encoding="utf-8"))["queries"]
    targets = [row["result"] for row in rows]
    if prior.canonical_sha256(targets) != spec["target_vector_sha256"]:
        raise RuntimeError(f"{dataset} 目标向量身份漂移")
    return targets


def _channels(
    counts: list[int], targets: list[int], n_records: int
) -> tuple[float, float | None]:
    if len(counts) != len(targets) or not counts:
        raise RuntimeError("查询计数向量长度漂移")
    count_array = np.asarray(counts, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    errors = np.abs(count_array - target_array)
    denominators = np.full(len(targets), float(n_records), dtype=np.float64)
    absolute = float(np.mean(errors / denominators, dtype=np.float64))
    positive = target_array > 0.0
    if not np.any(positive):
        return absolute, None
    inverse = 1.0 / target_array[positive]
    inverse_normalizer = float(np.sum(inverse, dtype=np.float64))
    weights = np.zeros_like(target_array, dtype=np.float64)
    weights[positive] = (
        inverse / inverse_normalizer / float(n_records)
    )
    relative = float(np.sum(
        errors * weights,
        dtype=np.float64,
    ))
    return absolute, relative


def _dominant(absolute: float, relative: float | None) -> str:
    if relative is None or absolute > relative:
        return "absolute"
    if relative > absolute:
        return "relative"
    return "tie"


def _row(
    item: dict[str, Any],
    *,
    targets: list[int],
    n_records: int,
    absolute_initial: float,
    relative_initial: float | None,
) -> dict[str, Any]:
    absolute, relative = _channels(
        item["query_answers"], targets, n_records
    )
    absolute_progress = absolute / absolute_initial
    relative_progress = (
        relative / relative_initial
        if relative is not None and relative_initial is not None else None
    )
    return {
        "kind": item["kind"],
        "phase": item["phase"],
        "round": int(item["round"]),
        "state_index": int(item["state_index"]),
        "absolute_raw": absolute,
        "relative_raw": relative,
        "raw_dominant_channel": _dominant(absolute, relative),
        "absolute_relative_to_initial": absolute_progress,
        "relative_relative_to_initial": relative_progress,
        "progress_dominant_channel": _dominant(
            absolute_progress, relative_progress
        ),
    }


def build_audit(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    datasets = {}
    for dataset in prior.DATASET_ORDER:
        targets = _targets(root, dataset)
        spec = prior.DATASETS[dataset]
        artifact = (
            root
            / prior.OUTPUT_DIR
            / "cases"
            / f"seed_{prior.DEVELOPMENT_SEED}__{prior.CANDIDATE_ARM}__{dataset}"
            / "checkpoint_query_answers.json"
        )
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if (
            payload["dataset"] != dataset
            or payload["seed"] != prior.DEVELOPMENT_SEED
            or payload["target_vector_sha256"]
            != spec["target_vector_sha256"]
            or payload["query_count"] != len(targets)
        ):
            raise RuntimeError(f"{dataset} 历史检查点身份漂移")
        initial = payload["fixed_checkpoints"][0]
        if initial["round"] != 0 or initial["phase"] != "initial":
            raise RuntimeError(f"{dataset} 缺少冻结初始检查点")
        absolute_initial, relative_initial = _channels(
            initial["query_answers"], targets, int(spec["n_records"])
        )
        if absolute_initial <= 0.0:
            raise RuntimeError(f"{dataset} A_init 不是正数")
        if any(target > 0 for target in targets) and (
            relative_initial is None or relative_initial <= 0.0
        ):
            raise RuntimeError(f"{dataset} R_init 不是正数")
        rows = [
            _row(
                item,
                targets=targets,
                n_records=int(spec["n_records"]),
                absolute_initial=absolute_initial,
                relative_initial=relative_initial,
            )
            for item in payload["fixed_checkpoints"]
        ]
        terminal = _row(
            payload["terminal"],
            targets=targets,
            n_records=int(spec["n_records"]),
            absolute_initial=absolute_initial,
            relative_initial=relative_initial,
        )
        datasets[dataset] = {
            "source_checkpoint_artifact": str(artifact.relative_to(root)),
            "source_checkpoint_artifact_sha256": file_sha256(artifact),
            "n_records": int(spec["n_records"]),
            "query_count": len(targets),
            "zero_target_query_count": sum(target == 0 for target in targets),
            "positive_target_query_count": sum(target > 0 for target in targets),
            "absolute_initial": absolute_initial,
            "relative_initial": relative_initial,
            "fixed_checkpoints": rows,
            "terminal": terminal,
            "summary": {
                "raw_absolute_dominant_fixed_checkpoint_count": sum(
                    row["raw_dominant_channel"] == "absolute" for row in rows
                ),
                "raw_relative_dominant_fixed_checkpoint_count": sum(
                    row["raw_dominant_channel"] == "relative" for row in rows
                ),
                "progress_absolute_dominant_fixed_checkpoint_count": sum(
                    row["progress_dominant_channel"] == "absolute"
                    for row in rows
                ),
                "progress_relative_dominant_fixed_checkpoint_count": sum(
                    row["progress_dominant_channel"] == "relative"
                    for row in rows
                ),
                "terminal_dominance_changed": (
                    terminal["raw_dominant_channel"]
                    != terminal["progress_dominant_channel"]
                ),
            },
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "formula": {
            "absolute": "sum_all(abs(target-count))/(N*J)",
            "relative": (
                "sum_positive(abs(target-count)/target)"
                "/(N*sum_positive(1/target))"
            ),
            "progress_aggregation": "max(A/A_init,R/R_init)",
            "reference_source": "historical_round_0_initial_checkpoint",
            "zero_target_policy": "absolute_channel_only",
        },
        "datasets": datasets,
        "audit": {
            "read_only": True,
            "runtime_float64_reduction_order_reproduced": True,
            "historical_candidate_checkpoint_answers_only": True,
            "new_candidate_generated": False,
            "raw_reference_table_accessed": False,
            "quality_evaluation_performed": False,
            "parameter_search_performed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    print(strict_json_text(build_audit(args.repository_root)), end="")


if __name__ == "__main__":
    main()
