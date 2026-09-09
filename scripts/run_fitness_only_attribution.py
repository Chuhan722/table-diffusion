#!/usr/bin/env python3
"""Run the result-blind fitness-only attribution development screen.

The command has two deliberately separate phases:

``plan``
    Emits the frozen protocol without reading any input data or result.
``run``
    Validates the generation inputs, runs paired ``residual``/``equal`` arms,
    and only *after every generation trajectory has finished* loads held-out
    answers and the raw reference tables for offline quality evaluation.

This is a development screen, not a formal multi-seed claim.  It has no
quality-based promotion gate and never overwrites an existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_paired_fitness_only_attribution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.quality import (
    evaluate_quality_snapshot,
    query_error_metrics,
    query_fingerprint,
    validate_query_partition,
)
from table_diffevo.queries import evaluate_table, load_data, load_queries
from table_diffevo.schema import load_schema


PROTOCOL_VERSION = "fitness-only-attribution-development-v1"
OUTPUT_DIR = Path("outputs/fitness_only_attribution_dev_seed9908_v1")
SEED = 9908
N_ROUNDS = 3000
FROZEN_PROTOCOL_SHA256 = (
    "8a538ae3d03f2d1dce9b4122ba567bf404f59c45dedbd381ee52c612b3b2e323"
)

ARMS = ("residual", "equal")

# Insertion order is part of the execution protocol.  Generation is completed
# for both datasets before any held-out/reference input is opened.
DATASETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict([
    (
        "test_300x10",
        {
            "schema": Path("configs/test_300x10/schema.yaml"),
            "measured": Path(
                "configs/test_300x10/measured_50query_30_15_5.json"
            ),
            "marginals": Path("configs/test_300x10/init_marginals.json"),
            "heldout": Path("configs/test_300x10/heldout_issue53_v1.json"),
            "reference": Path("data/test_300x10/test_300x10.csv"),
            "safety": Path("configs/test_300x10/measured_50query.json"),
            "safety_source": "frozen_one_way_file",
            "n_records": 300,
            "device": "numpy",
            "expected_query_count": 50,
            "input_sha256": {
                "schema": (
                    "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
                ),
                "measured": (
                    "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
                ),
                "marginals": (
                    "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
                ),
                "heldout": (
                    "300bffea1f3d9105ad8f1840d50a900616115659065efec35b3c02f7a38cc1e0"
                ),
                "reference": (
                    "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
                ),
                "safety": (
                    "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88"
                ),
            },
        },
    ),
    (
        "nltcs",
        {
            "schema": Path("configs/nltcs/schema.yaml"),
            "measured": Path("configs/nltcs/measured_1000query.json"),
            "marginals": Path("configs/nltcs/init_marginals.json"),
            "heldout": Path("configs/nltcs/heldout_issue53_v1.json"),
            "reference": Path("data/nltcs/nltcs.csv"),
            "safety": Path("configs/nltcs/init_marginals.json"),
            "safety_source": "marginal_values_and_counts",
            "n_records": 16181,
            "device": "cuda",
            "expected_query_count": 1001,
            "input_sha256": {
                "schema": (
                    "5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4"
                ),
                "measured": (
                    "b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f"
                ),
                "marginals": (
                    "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e"
                ),
                "heldout": (
                    "a025b5b4d2d44261b03075d4aca030cc10fc0c1d99051241a671f465644003eb"
                ),
                "reference": (
                    "7d185b8a065e051341e581ba65b27007d72e5d2b67adfffacf213117799edf7c"
                ),
                "safety": (
                    "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e"
                ),
            },
        },
    ),
])


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def _git_text(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _jsonable(value: Any) -> Any:
    """Convert diagnostics to strict JSON without silently writing NaN."""

    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            raise ValueError("诊断数据包含 NaN")
        if math.isinf(number):
            return "positive_infinity" if number > 0 else "negative_infinity"
        return number
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"无法序列化诊断类型: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "development_attribution_only_no_formal_quality_claim",
        "seed": SEED,
        "n_rounds": N_ROUNDS,
        "arms": list(ARMS),
        "dataset_order": list(DATASETS),
        "datasets": [
            {
                "name": name,
                "schema": str(spec["schema"]),
                "measured": str(spec["measured"]),
                "marginals": str(spec["marginals"]),
                "heldout": str(spec["heldout"]),
                "reference": str(spec["reference"]),
                "safety": str(spec["safety"]),
                "safety_source": spec["safety_source"],
                "n_records": spec["n_records"],
                "device": spec["device"],
                "expected_query_count": spec["expected_query_count"],
                "input_sha256": spec["input_sha256"],
            }
            for name, spec in DATASETS.items()
        ],
        "generation_config": {
            "init_method": "marginal",
            "eval_method": "vectorized",
            "rho": 0.01,
            "eta": 0.5,
            "mu": 0.01,
            "distance_mode": "geometric",
            "selection_scale_invariant": True,
            "selection_scale_invariant_min_spread": 1e-3,
            "alpha_schedule_mode": "fixed",
            "fixed_alpha": 16.0,
            "lambda_param": 0.5,
            "delta": 0.05,
            "winsorize_quantiles": [0.01, 0.99],
            "residual_geometry": "relative",
            "residual_geometry_floor": 8.0,
            "exclude_self": True,
            "tol": "positive_infinity",
            "max_retries": 0,
            "residual_directed_diffusion": False,
            "factorized_gibbs_sweeps": 0,
            "gap_l1_sweeps": 0,
            "candidate_budget": None,
            "residual_self_cooling": None,
            "rho_anneal_end": None,
            "stop_on_exact_residual": False,
            "inner_early_stopping_patience_ticks": None,
            "horizon_invariant": True,
            "return_identity": "terminal_current",
        },
        "pairing": {
            "same_seed_and_frozen_marginals": True,
            "same_initial_table_hash": True,
            "same_post_initialization_rng_hash": True,
            "same_final_rng_hash": True,
            "same_candidate_count_and_round_count": True,
            "only_treatment_difference": "row_fitness",
        },
        "phase_boundary": {
            "heldout_and_reference_loaded_only_after_all_generation": True,
            "online_quality_control": False,
            "parameter_retuning_allowed": False,
            "output_overwrite_allowed": False,
        },
        "quality": {
            "primary_role": "descriptive_development_screen",
            "metrics": [
            "measured",
            "measured_by_order",
            "target_frequency_buckets",
            "heldout_3way",
            "heldout_4way",
                "one_way_safety",
                "schema_validity",
                "diversity",
                "reference_support_offline_only",
            ],
            "lower_is_better_error": True,
            "promotion_gate": None,
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_json_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "fitness-only 开发协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    """Build a plan without reading any input file or result artifact."""

    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": _json_protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "trajectory_count": len(DATASETS) * len(ARMS),
        "generation_started": False,
    }


def _source_snapshot(root: Path) -> dict[str, Any]:
    tracked_sources = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/fitness_only.py"),
    )
    runner_path = Path(__file__).resolve().relative_to(root)
    paths = (*tracked_sources, runner_path)
    return {
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "source_sha256": {
            str(path): _sha256_file(root / path) for path in paths
        },
        "git_status_porcelain": _git_text(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
    }


def _audit_source_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if before["git_commit"] != after["git_commit"]:
        raise RuntimeError("生成期间 git commit 发生变化")
    if before["source_sha256"] != after["source_sha256"]:
        raise RuntimeError("生成期间核心源码发生变化")


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象: {path}")
    return value


def _audit_generation_inputs(root: Path) -> dict[str, dict[str, Any]]:
    """Validate only schema/measured/marginal inputs before generation."""

    audits: dict[str, dict[str, Any]] = {}
    for name, spec in DATASETS.items():
        observed_hashes = {
            key: _sha256_file(root / spec[key])
            for key in ("schema", "measured", "marginals")
        }
        expected_hashes = {
            key: spec["input_sha256"][key]
            for key in observed_hashes
        }
        if observed_hashes != expected_hashes:
            raise RuntimeError(
                f"{name} generation 输入 SHA 漂移: "
                f"expected={expected_hashes}, observed={observed_hashes}"
            )

        payload = _load_json_object(root / spec["measured"])
        queries = payload.get("queries")
        if (
            not isinstance(queries, list)
            or len(queries) != spec["expected_query_count"]
        ):
            raise RuntimeError(f"{name} measured 查询数量漂移")
        fingerprints = [query_fingerprint(query) for query in queries]
        if len(set(fingerprints)) != len(fingerprints):
            raise RuntimeError(f"{name} measured 查询存在重复语义")
        targets = []
        for index, query in enumerate(queries):
            result = query.get("result")
            if (
                isinstance(result, bool)
                or not isinstance(result, (int, float))
                or not np.isfinite(result)
                or result < 0
            ):
                raise ValueError(f"{name} measured target 非法: index={index}")
            targets.append(float(result))
        if int(payload.get("record_count", -1)) != spec["n_records"]:
            raise RuntimeError(f"{name} measured record_count 漂移")
        audits[name] = {
            "generation_input_sha256": observed_hashes,
            "query_count": len(queries),
            "query_identity_sha256": _sha256_bytes(
                "\n".join(fingerprints).encode("ascii")
            ),
            "queries": queries,
            "targets": targets,
            "target_vector_sha256": _sha256_bytes(
                _strict_json_bytes(targets)
            ),
        }
    return audits


def _fitness_config(spec: Mapping[str, Any]) -> FitnessOnlyConfig:
    return FitnessOnlyConfig(
        n_rounds=N_ROUNDS,
        seed=SEED,
        device=spec["device"],
        eval_method="vectorized",
        batch_size=256,
        init_method="marginal",
        log_every=100,
        rho=0.01,
        eta=0.5,
        mu=0.01,
        lambda_param=0.5,
        fixed_alpha=16.0,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        selection_scale_invariant_min_spread=1e-3,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
        exclude_self=True,
        record_transition_clocks=False,
    )


def _run_generation(
    root: Path,
    staging: Path,
    audits: Mapping[str, Mapping[str, Any]],
    config_factory: Any = None,
) -> dict[str, Any]:
    if config_factory is None:
        config_factory = _fitness_config
    results: dict[str, Any] = {}
    for name, spec in DATASETS.items():
        schema = load_schema(str(root / spec["schema"]))
        marginals = load_marginals(str(root / spec["marginals"]))
        config = config_factory(spec)
        started = time.perf_counter()
        runs, pairing = run_paired_fitness_only_attribution(
            np.asarray(audits[name]["targets"], dtype=float),
            list(audits[name]["queries"]),
            schema,
            int(spec["n_records"]),
            config=config,
            marginals=marginals,
        )
        elapsed = time.perf_counter() - started
        dataset_dir = staging / "generation" / name
        dataset_dir.mkdir(parents=True, exist_ok=False)
        _write_json(dataset_dir / "pairing.json", pairing)
        arm_results = {}
        for arm in ARMS:
            table, diagnostics = runs[arm]
            table = table.reset_index(drop=True)
            table_path = dataset_dir / f"{arm}_terminal_current.csv"
            table.to_csv(table_path, index=False)
            _write_json(
                dataset_dir / f"{arm}_diagnostics.json",
                diagnostics,
            )
            arm_results[arm] = {
                "terminal_table_sha256": _frame_sha256(table),
                "rounds_run": int(diagnostics["rounds_run"]),
                "candidate_evaluation_count": int(
                    diagnostics["candidate_evaluation_count"]
                ),
                "termination_reason": diagnostics["termination_reason"],
                "output_table_identity": diagnostics[
                    "output_table_identity"
                ],
                "final_current_squared_loss": float(
                    diagnostics["final_current_squared_loss"]
                ),
                "final_current_normalized_l1": float(
                    diagnostics["final_current_normalized_l1"]
                ),
                "best_loss_diagnostic_only": float(
                    diagnostics["best_loss_diagnostic_only"]
                ),
                "elapsed_sec": float(diagnostics["elapsed_sec"]),
            }
        results[name] = {
            "dataset": name,
            "seed": SEED,
            "n_records": int(spec["n_records"]),
            "query_count": int(audits[name]["query_count"]),
            "device": spec["device"],
            "elapsed_sec_wall": float(elapsed),
            "pairing": pairing,
            "arms": arm_results,
        }
        print(
            f"[{name}] generation complete in {elapsed:.1f}s",
            flush=True,
        )
    return results


def _target_bucket(target: float) -> str:
    if target == 0:
        return "target_0"
    if target <= 8:
        return "target_1_8"
    if target <= 50:
        return "target_9_50"
    if target <= 500:
        return "target_51_500"
    if target <= 2000:
        return "target_501_2000"
    return "target_over_2000"


def _grouped_error_metrics(
    queries: Sequence[dict[str, Any]],
    target: Sequence[float],
    answers: Sequence[float],
    n_records: int,
) -> dict[str, Any]:
    target_values = np.asarray(target, dtype=float)
    answer_values = np.asarray(answers, dtype=float)
    groups: dict[str, list[int]] = {}
    for index, query in enumerate(queries):
        order = len(query.get("conditions", []))
        groups.setdefault(f"{order}way", []).append(index)
    bucket_groups: dict[str, list[int]] = {}
    for index, value in enumerate(target_values):
        bucket_groups.setdefault(_target_bucket(float(value)), []).append(index)

    def metrics_for(indices: Iterable[int]) -> dict[str, Any]:
        selected = np.asarray(list(indices), dtype=int)
        if len(selected) == 0:
            return {"query_count": 0}
        result = query_error_metrics(
            target_values[selected],
            answer_values[selected],
            n_records,
        )
        abs_errors = np.abs(target_values[selected] - answer_values[selected])
        result.update({
            "absolute_error_mean": float(np.mean(abs_errors)),
            "absolute_error_median": float(np.median(abs_errors)),
            "absolute_error_p90": float(np.percentile(abs_errors, 90)),
            "absolute_error_max": float(np.max(abs_errors)),
        })
        return result

    return {
        "by_order": {
            key: metrics_for(indices)
            for key, indices in sorted(groups.items())
        },
        "by_target_bucket": {
            key: metrics_for(indices)
            for key, indices in sorted(bucket_groups.items())
        },
    }


def _load_one_way_safety(
    root: Path,
    name: str,
) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    """Build the fixed one-way safety workload after generation only."""

    spec = DATASETS[name]
    safety_hash = _sha256_file(root / spec["safety"])
    if safety_hash != spec["input_sha256"]["safety"]:
        raise RuntimeError(f"{name} one-way safety 输入 SHA 漂移")

    if spec["safety_source"] == "frozen_one_way_file":
        payload = _load_json_object(root / spec["safety"])
        all_queries = payload.get("queries")
        if not isinstance(all_queries, list):
            raise RuntimeError(f"{name} one-way safety 文件格式错误")
        queries = [
            query for query in all_queries
            if len(query.get("conditions", [])) == 1
        ]
        targets = [float(query["result"]) for query in queries]
    elif spec["safety_source"] == "marginal_values_and_counts":
        marginals = load_marginals(str(root / spec["safety"]))
        queries = []
        targets = []
        for attribute, attribute_spec in marginals["attributes"].items():
            values = attribute_spec.get("values")
            counts = attribute_spec.get("counts")
            if not isinstance(values, list) or not isinstance(counts, list):
                raise RuntimeError(f"{name} marginal one-way 数据格式错误")
            if len(values) != len(counts):
                raise RuntimeError(f"{name} marginal values/counts 数量不一致")
            for value, count in zip(values, counts):
                queries.append({
                    "conditions": [{
                        "attribute": attribute,
                        "operator": "==",
                        "value": value,
                    }],
                })
                targets.append(float(count))
    else:
        raise RuntimeError(f"未知 one-way safety 来源: {spec['safety_source']}")

    if not queries or len(queries) != 25 and name == "test_300x10":
        raise RuntimeError(f"{name} one-way safety 数量错误: {len(queries)}")
    if name == "nltcs" and len(queries) != 32:
        raise RuntimeError(f"{name} one-way safety 数量错误: {len(queries)}")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError(f"{name} one-way safety 存在重复语义")
    return queries, np.asarray(targets, dtype=float), safety_hash


def _evaluate_one(
    root: Path,
    name: str,
    arm: str,
    generation_audit: Mapping[str, Any],
    generation_result: Mapping[str, Any],
    staging: Path,
) -> dict[str, Any]:
    """Load held-out/reference only in the post-generation phase."""

    spec = DATASETS[name]
    expected_hashes = spec["input_sha256"]
    heldout_hash = _sha256_file(root / spec["heldout"])
    reference_hash = _sha256_file(root / spec["reference"])
    if heldout_hash != expected_hashes["heldout"]:
        raise RuntimeError(f"{name} heldout SHA 漂移")
    if reference_hash != expected_hashes["reference"]:
        raise RuntimeError(f"{name} reference SHA 漂移")

    heldout_payload = _load_json_object(root / spec["heldout"])
    heldout_queries = heldout_payload.get("queries")
    if not isinstance(heldout_queries, list) or len(heldout_queries) != 1024:
        raise RuntimeError(f"{name} heldout 查询数量漂移")
    heldout_targets = []
    for query in heldout_queries:
        result = query.get("result")
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError(f"{name} heldout target 非法")
        heldout_targets.append(float(result))
    heldout_orders = [len(query.get("conditions", [])) for query in heldout_queries]
    if heldout_orders.count(3) != 512 or heldout_orders.count(4) != 512:
        raise RuntimeError(f"{name} heldout 3/4-way 分组漂移")
    validate_query_partition(
        list(generation_audit["queries"]),
        heldout_queries,
    )

    table_path = (
        staging / "generation" / name / f"{arm}_terminal_current.csv"
    )
    table = pd.read_csv(table_path)
    schema = load_schema(str(root / spec["schema"]))
    if _frame_sha256(table) != generation_result["terminal_table_sha256"]:
        raise RuntimeError(f"{name}/{arm} terminal table SHA 漂移")
    if len(table) != spec["n_records"]:
        raise RuntimeError(f"{name}/{arm} terminal row count 漂移")

    measured_queries = list(generation_audit["queries"])
    measured_targets = np.asarray(generation_audit["targets"], dtype=float)
    measured_answers = np.asarray(evaluate_table(table, measured_queries), dtype=float)
    heldout_answers = np.asarray(evaluate_table(table, heldout_queries), dtype=float)
    safety_queries, safety_targets, safety_hash = _load_one_way_safety(
        root,
        name,
    )
    validate_query_partition(measured_queries, safety_queries)
    validate_query_partition(heldout_queries, safety_queries)
    safety_answers = np.asarray(evaluate_table(table, safety_queries), dtype=float)
    reference = load_data(str(root / spec["reference"]))
    quality = evaluate_quality_snapshot(
        table,
        schema,
        measured_queries,
        measured_targets,
        heldout_queries,
        heldout_targets,
        reference_table=reference,
    )
    quality["measured_by_order_and_target_bucket"] = _grouped_error_metrics(
        measured_queries,
        measured_targets,
        measured_answers,
        int(spec["n_records"]),
    )
    quality["heldout_by_order_and_target_bucket"] = _grouped_error_metrics(
        heldout_queries,
        heldout_targets,
        heldout_answers,
        int(spec["n_records"]),
    )
    quality["one_way_safety"] = query_error_metrics(
        safety_targets,
        safety_answers,
        int(spec["n_records"]),
    )
    quality["one_way_safety"]["absolute_error_mean"] = float(
        np.mean(np.abs(safety_targets - safety_answers))
    )
    quality["one_way_safety"]["absolute_error_median"] = float(
        np.median(np.abs(safety_targets - safety_answers))
    )
    quality["one_way_safety"]["absolute_error_p90"] = float(
        np.percentile(np.abs(safety_targets - safety_answers), 90)
    )
    quality["one_way_safety"]["absolute_error_max"] = float(
        np.max(np.abs(safety_targets - safety_answers))
    )
    quality["one_way_safety_by_target_bucket"] = _grouped_error_metrics(
        safety_queries,
        safety_targets,
        safety_answers,
        int(spec["n_records"]),
    )
    quality["evaluation_inputs"] = {
        "measured_sha256": expected_hashes["measured"],
        "heldout_sha256": heldout_hash,
        "reference_sha256": reference_hash,
        "one_way_safety_sha256": safety_hash,
        "reference_loaded_after_generation": True,
    }
    quality_dir = staging / "quality" / name
    quality_dir.mkdir(parents=True, exist_ok=True)
    _write_json(quality_dir / f"{arm}.json", quality)
    return quality


def _metric_path(payload: Mapping[str, Any], *path: str) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _paired_quality_comparison(
    quality_by_dataset: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for name, arms in quality_by_dataset.items():
        residual = arms["residual"]
        equal = arms["equal"]
        comparison[name] = {
            "delta_residual_minus_equal": {
                "measured_normalized_l1_mean": (
                    _metric_path(residual, "measured", "normalized_l1_mean")
                    - _metric_path(equal, "measured", "normalized_l1_mean")
                ),
                "heldout_3way_normalized_l1_mean": (
                    _metric_path(residual, "heldout", "3way", "normalized_l1_mean")
                    - _metric_path(equal, "heldout", "3way", "normalized_l1_mean")
                ),
                "heldout_4way_normalized_l1_mean": (
                    _metric_path(residual, "heldout", "4way", "normalized_l1_mean")
                    - _metric_path(equal, "heldout", "4way", "normalized_l1_mean")
                ),
                "one_way_normalized_l1_mean": (
                    _metric_path(residual, "one_way_safety", "normalized_l1_mean")
                    - _metric_path(equal, "one_way_safety", "normalized_l1_mean")
                ),
            },
            "interpretation": (
                "descriptive_only_lower_is_better; "
                "no_promotion_gate_in_development_screen"
            ),
        }
    return comparison


def run(confirmed_protocol_sha256: str) -> Path:
    expected_protocol = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected_protocol:
        raise ValueError(
            "protocol SHA-256 确认值不一致："
            f"expected={expected_protocol}, received={confirmed_protocol_sha256}"
        )
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")

    source_before = _source_snapshot(root)
    generation_audits = _audit_generation_inputs(root)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        generation_results = _run_generation(
            root,
            staging,
            generation_audits,
        )
        source_after_generation = _source_snapshot(root)
        _audit_source_unchanged(source_before, source_after_generation)

        # This is the explicit phase boundary: only now open held-out answers
        # and raw reference tables.
        quality_by_dataset: dict[str, dict[str, Any]] = {}
        for name in DATASETS:
            quality_by_dataset[name] = {}
            for arm in ARMS:
                quality_by_dataset[name][arm] = _evaluate_one(
                    root,
                    name,
                    arm,
                    generation_audits[name],
                    generation_results[name]["arms"][arm],
                    staging,
                )

        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected_protocol,
            "protocol": _json_protocol_manifest(),
            "source_before_generation": source_before,
            "source_after_generation": source_after_generation,
            "generation_inputs": {
                name: {
                    key: value
                    for key, value in audit.items()
                    if key not in {"queries", "targets"}
                }
                for name, audit in generation_audits.items()
            },
            "generation": generation_results,
            "quality": quality_by_dataset,
            "paired_quality_comparison": _paired_quality_comparison(
                quality_by_dataset
            ),
            "summary": {
                "development_only": True,
                "formal_claim_allowed": False,
                "all_generation_terminal_current": True,
                "all_generation_fixed_rounds": all(
                    generation_results[name]["arms"][arm]["rounds_run"]
                    == N_ROUNDS
                    for name in DATASETS
                    for arm in ARMS
                ),
                "all_generation_unconditional": True,
                "parameter_retuning_performed": False,
                "reference_loaded_after_generation": True,
            },
            "completed_at_unix": time.time(),
        }
        _write_json(staging / "report.json", report)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "report.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    report_path = run(args.confirm_protocol_sha)
    print(f"fitness-only attribution report -> {report_path}")


if __name__ == "__main__":
    main()
