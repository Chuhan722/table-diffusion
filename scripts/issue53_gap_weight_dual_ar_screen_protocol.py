"""Issue #53 B+C 问题一：A/R 双通道单种子开发筛查协议。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from scripts import issue53_gap_weight_r8_screen_protocol as r8
from scripts import issue53_gap_weight_sqrt_screen_protocol as sqrt


PROTOCOL_VERSION = "issue53-gap-weight-dual-ar-max-screen-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_BC问题一AR双通道单种子开发筛查结果前协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "b24a2b05fe892acbb4244c133688eeb96a9be4835ae8fc63958e55c0254d1395"
)
FROZEN_PROTOCOL_SHA256 = (
    "85016eb9b91364006537b4ccadd71bea8685b502a96aa05d88f73c700d75532d"
)

DESIGN_DOC = Path(
    "docs/设计/Issue53_BC问题一AR双通道查询权重设计冻结.md"
)
DESIGN_DOC_SHA256 = (
    "c21b23709618c02c48bbc3be34856bc8d6297a5c454b058cd7973d337be488e0"
)

TARGET_WEIGHT_AUDIT = Path(
    "docs/设计/Issue53_BC问题一AR双通道目标权重离线审计.json"
)
TARGET_WEIGHT_AUDIT_SHA256 = (
    "06b4d50a6af9ef250e472cccf5e2af7a6faa4cbca4fa1f993a3a4f26b9a145d4"
)

DEVELOPMENT_SEED = 9908
ROUND_CAP = r8.ROUND_CAP
CANDIDATE_BUDGET = r8.CANDIDATE_BUDGET
PATIENCE_TICKS = r8.PATIENCE_TICKS
CHECKPOINT_ROUNDS = r8.CHECKPOINT_ROUNDS
MAX_WORKERS = r8.MAX_WORKERS

DUAL_WEIGHTING = "dual_abs_relative_max"
CANDIDATE_ARM = "gap_dual_abs_relative_max_s8"
DATASET_ORDER = r8.DATASET_ORDER
ARM_ORDER = (CANDIDATE_ARM,)
DATASETS = r8.DATASETS
TARGET_COUNT_BINS = r8.TARGET_COUNT_BINS
NLTCS_COMMON_BIN = r8.NLTCS_COMMON_BIN
NLTCS_RARE_BIN = r8.NLTCS_RARE_BIN

NLTCS_RARE_RATIO_MAX = 1.25
TEST_OVERALL_RATIO_MAX = 1.05
ONE_WAY_RATIO_MAX = 1.05

OUTPUT_DIR = Path(
    "outputs/issue53_gap_weight_dual_ar_max_screen_seed9908_v1"
)

BASELINE_ARTIFACTS = {
    "legacy_and_r8": {
        "collection": {
            "path": r8.OUTPUT_DIR / "collection_report.json",
            "sha256": (
                "35e0b5c594dd8b502198b58690ca720449d2283702817acae843b11968bd3eb8"
            ),
        },
        "evaluation": {
            "path": r8.OUTPUT_DIR / "evaluation_report.json",
            "sha256": (
                "548b611b3762cd5489fcbd97d787ea9e91f987c5a94bed0b73e6782f6108bc4c"
            ),
        },
        "metrics_csv": {
            "path": r8.OUTPUT_DIR / "screen_metrics.csv",
            "sha256": (
                "b6d1efff24202a74003cee01eb05568db64b071aab3a47a86e3cafa4bc27b660"
            ),
        },
        "independent_audit": {
            "path": r8.OUTPUT_DIR / "independent_audit.json",
            "sha256": (
                "9e9e29c39cadc756f29dd67cc1bb17eeccccace32a167225383fca81e5ce33e6"
            ),
        },
    },
    "sqrt": {
        "collection": {
            "path": sqrt.OUTPUT_DIR / "collection_report.json",
            "sha256": (
                "50f0c3561fa9614abc85eacb014566e3354e987a647784f27453419ea3e882f4"
            ),
        },
        "evaluation": {
            "path": sqrt.OUTPUT_DIR / "evaluation_report.json",
            "sha256": (
                "e6d8110f521e366130949b4b4d3ac460b34eb7109d3e4a8f5518d96e92df3fad"
            ),
        },
        "metrics_csv": {
            "path": sqrt.OUTPUT_DIR / "screen_metrics.csv",
            "sha256": (
                "b5dd01a653544bb7a3a98317ba0e708d2e89a10b70679c57c6046cf339b072e6"
            ),
        },
        "independent_audit": {
            "path": sqrt.OUTPUT_DIR / "independent_audit.json",
            "sha256": (
                "534b19e8f866f6ef81352e695e82891113d31158d084ea1ec7b7a5184ad7ed21"
            ),
        },
    },
}

EXPECTED_WEIGHT_AUDIT = {
    "test_300x10": {
        "query_count": 50,
        "n_records": 300,
        "minimum_target": 0,
        "maximum_target": 88,
        "zero_target_query_count": 3,
        "positive_target_query_count": 47,
        "absolute_unit_error_weight": 6.666666666666667e-05,
        "uniform_error_quantum": 0.0033333333333333335,
        "relative_inverse_target_normalizer": 5.243603286242844,
        "minimum_positive_relative_beta": 0.002167142658074474,
        "maximum_positive_relative_beta": 0.19070855391055372,
        "relative_positive_weight_ratio": 88.0,
    },
    "nltcs": {
        "query_count": 1001,
        "n_records": 16181,
        "minimum_target": 12,
        "maximum_target": 13092,
        "zero_target_query_count": 0,
        "positive_target_query_count": 1001,
        "absolute_unit_error_weight": 6.17391384340275e-08,
        "uniform_error_quantum": 6.180087757246153e-05,
        "relative_inverse_target_normalizer": 3.8332971794614985,
        "minimum_positive_relative_beta": 1.992606367380902e-05,
        "maximum_positive_relative_beta": 0.02173933546812564,
        "relative_positive_weight_ratio": 1091.0,
    },
}

IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": (
            "7d675c24858397be6d41e5c9ef37d34e541e0a06c3f2aa81ee4092490a3d64c1"
        ),
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": (
            "388239802a607f0f8259958ab70e6b502356c1abb7e13fe02f4d5d3f0167dcb2"
        ),
    },
}

Dataset = Literal["test_300x10", "nltcs"]


@dataclass(frozen=True)
class DualArScreenTask:
    dataset: Dataset
    arm: str
    seed: int
    rounds: int

    @property
    def task_id(self) -> str:
        return f"seed_{self.seed}__{self.arm}__{self.dataset}"


@dataclass(frozen=True)
class DualArScreenPlan:
    seed: int
    tasks: tuple[DualArScreenTask, ...]
    protocol_frozen: bool = True
    runner_wired: bool = False
    generation_started: bool = False


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return value


def task_plan() -> DualArScreenPlan:
    tasks = tuple(
        DualArScreenTask(
            dataset=dataset,
            arm=CANDIDATE_ARM,
            seed=DEVELOPMENT_SEED,
            rounds=ROUND_CAP,
        )
        for dataset in DATASET_ORDER
    )
    if len(tasks) != 2 or len({task.task_id for task in tasks}) != 2:
        raise RuntimeError("A/R 双通道筛查任务矩阵不完整")
    return DualArScreenPlan(seed=DEVELOPMENT_SEED, tasks=tasks)


def common_generator_params() -> dict[str, Any]:
    return r8.common_generator_params()


def arm_kernel_params(arm: str) -> dict[str, Any]:
    if arm != CANDIDATE_ARM:
        raise ValueError(f"未知 A/R 双通道筛查方法：{arm!r}")
    return {
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_use_compiled_workload": False,
        "gap_l1_sweeps": 8,
        "gap_l1_weighting": DUAL_WEIGHTING,
        "gap_l1_max_weight_ratio": None,
    }


def task_generator_params(dataset: str, arm: str) -> dict[str, Any]:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"未知 A/R 双通道筛查数据集：{dataset!r}")
    params = common_generator_params()
    params.update(arm_kernel_params(arm))
    params.update({
        "seed": DEVELOPMENT_SEED,
        "n_records": int(DATASETS[dataset]["n_records"]),
        "factorized_gibbs_max_order": int(
            DATASETS[dataset]["max_factor_order"]
        ),
    })
    return params


def generator_params_manifest(dataset: str, arm: str) -> dict[str, Any]:
    params = task_generator_params(dataset, arm)
    nonfinite = [
        key
        for key, value in params.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if nonfinite != ["tol"] or params["tol"] != float("inf"):
        raise RuntimeError("A/R 双通道筛查生成参数只允许 tol=+inf")
    manifest = _jsonable(params)
    _strict_json_bytes(manifest)
    return manifest


def generator_params_manifest_matrix() -> dict[str, dict[str, Any]]:
    return {
        task.task_id: generator_params_manifest(task.dataset, task.arm)
        for task in task_plan().tasks
    }


def generator_params_manifest_sha256() -> str:
    return canonical_sha256(generator_params_manifest_matrix())


def _load_query_rows(root: Path, dataset: str) -> list[dict[str, Any]]:
    spec = DATASETS[dataset]
    query_path = root / spec["queries"]
    if file_sha256(query_path) != spec["input_sha256"]["queries"]:
        raise RuntimeError(f"{dataset} 查询文件 SHA-256 漂移")
    payload = json.loads(query_path.read_text(encoding="utf-8"))
    rows = payload.get("queries")
    if not isinstance(rows, list) or len(rows) != spec["query_count"]:
        raise RuntimeError(f"{dataset} 查询数量漂移")
    targets = []
    for index, row in enumerate(rows):
        target = row.get("result")
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 0 <= target <= spec["n_records"]
        ):
            raise RuntimeError(f"{dataset} 第 {index} 条目标不是非负整数计数")
        targets.append(target)
    if canonical_sha256(targets) != spec["target_vector_sha256"]:
        raise RuntimeError(f"{dataset} 目标向量身份漂移")
    return rows


def _bin_for_target(dataset: str, target: int) -> dict[str, Any]:
    matches = [
        spec
        for spec in TARGET_COUNT_BINS[dataset]
        if spec["lower_inclusive"] <= target <= spec["upper_inclusive"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{dataset} 目标 {target} 未唯一进入冻结分箱")
    return matches[0]


def _summary(
    targets: list[int], *, n_records: int, query_count: int
) -> dict[str, Any]:
    positive_targets = [target for target in targets if target > 0]
    inverse_normalizer = math.fsum(
        1.0 / target for target in positive_targets
    )
    betas = [
        (1.0 / target) / inverse_normalizer for target in positive_targets
    ]
    return {
        "query_count": query_count,
        "n_records": n_records,
        "minimum_target": min(targets),
        "maximum_target": max(targets),
        "zero_target_query_count": query_count - len(positive_targets),
        "positive_target_query_count": len(positive_targets),
        "absolute_unit_error_weight": 1.0 / (n_records * query_count),
        "uniform_error_quantum": 1.0 / n_records,
        "relative_inverse_target_normalizer": inverse_normalizer,
        "minimum_positive_relative_beta": min(betas),
        "maximum_positive_relative_beta": max(betas),
        "relative_positive_weight_ratio": max(betas) / min(betas),
    }


def target_weight_audit(repository_root: str | Path) -> dict[str, Any]:
    """只从冻结查询目标独立构造 A/R 权重，不导入核实现。"""

    root = Path(repository_root).resolve()
    datasets: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        spec = DATASETS[dataset]
        query_rows = _load_query_rows(root, dataset)
        targets = [int(row["result"]) for row in query_rows]
        n_records = int(spec["n_records"])
        query_count = len(targets)
        positive_targets = [target for target in targets if target > 0]
        inverse_normalizer = math.fsum(
            1.0 / target for target in positive_targets
        )
        audited_rows = []
        membership_by_bin: dict[str, list[dict[str, Any]]] = {
            item["name"]: [] for item in TARGET_COUNT_BINS[dataset]
        }
        for index, (query, target) in enumerate(zip(query_rows, targets)):
            bin_spec = _bin_for_target(dataset, target)
            relative_raw_weight = 1.0 / target if target > 0 else None
            relative_beta = (
                relative_raw_weight / inverse_normalizer
                if relative_raw_weight is not None
                else 0.0
            )
            audited_rows.append({
                "index": index,
                "id": query.get("id"),
                "target": target,
                "target_bin": bin_spec["name"],
                "absolute_unit_error_weight": 1.0 / (
                    n_records * query_count
                ),
                "relative_raw_inverse_target_weight": relative_raw_weight,
                "relative_normalized_beta": relative_beta,
                "relative_unit_error_weight": relative_beta / n_records,
                "relative_channel_member": target > 0,
            })
            membership_by_bin[bin_spec["name"]].append({
                "index": index,
                "id": query.get("id"),
                "target": target,
            })

        bin_summaries = {}
        for bin_spec in TARGET_COUNT_BINS[dataset]:
            name = bin_spec["name"]
            members = membership_by_bin[name]
            if (
                len(members) != bin_spec["query_count"]
                or canonical_sha256(members) != bin_spec["membership_sha256"]
            ):
                raise RuntimeError(f"{dataset}/{name} 冻结分箱身份漂移")
            rows = [row for row in audited_rows if row["target_bin"] == name]
            positive_rows = [
                row for row in rows if row["relative_channel_member"]
            ]
            bin_summaries[name] = {
                "query_count": len(rows),
                "zero_target_query_count": len(rows) - len(positive_rows),
                "positive_target_query_count": len(positive_rows),
                "minimum_target": min(row["target"] for row in rows),
                "maximum_target": max(row["target"] for row in rows),
                "absolute_unit_error_weight": 1.0 / (
                    n_records * query_count
                ),
                "minimum_positive_relative_beta": (
                    min(row["relative_normalized_beta"] for row in positive_rows)
                    if positive_rows else None
                ),
                "maximum_positive_relative_beta": (
                    max(row["relative_normalized_beta"] for row in positive_rows)
                    if positive_rows else None
                ),
                "lower_inclusive": bin_spec["lower_inclusive"],
                "upper_inclusive": bin_spec["upper_inclusive"],
                "membership_sha256": bin_spec["membership_sha256"],
            }

        summary = _summary(
            targets, n_records=n_records, query_count=query_count
        )
        for key, expected in EXPECTED_WEIGHT_AUDIT[dataset].items():
            if summary[key] != expected:
                raise RuntimeError(
                    f"{dataset} A/R 权重审计漂移：{key}="
                    f"{summary[key]!r}，预期 {expected!r}"
                )
        datasets[dataset] = {
            "query_file": str(spec["queries"]),
            "query_file_sha256": spec["input_sha256"]["queries"],
            "target_vector_sha256": spec["target_vector_sha256"],
            "summary": summary,
            "target_bins": bin_summaries,
            "queries": audited_rows,
        }

    return {
        "contract_version": "issue53-gap-weight-dual-ar-target-audit-v1",
        "formula": {
            "absolute": "sum_all(abs(target-count))/(N*J)",
            "relative": (
                "sum_positive(abs(target-count)/target)"
                "/(N*sum_positive(1/target))"
            ),
            "aggregation": "max(absolute,relative)",
            "zero_target_policy": "absolute_channel_only",
            "empty_positive_target_policy": "relative_zero_then_absolute",
            "tunable_parameter_count": 0,
        },
        "datasets": datasets,
        "audit": {
            "query_targets_only": True,
            "kernel_implementation_imported": False,
            "raw_reference_data_accessed": False,
            "legacy_r8_or_sqrt_terminal_tables_accessed": False,
            "candidate_generation_started": False,
            "candidate_results_accessed": False,
            "all_query_identities_verified": True,
            "all_target_identities_verified": True,
            "all_frozen_bin_identities_verified": True,
        },
    }


def _validated_ratio(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or math.isnan(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} 必须是非负数或正无穷")
    return float(value)


def classify_screen(
    *,
    execution_valid: bool,
    nltcs_measured_l1_ratio: Any,
    nltcs_common_bin_ratio: Any,
    nltcs_rare_vs_legacy_ratio: Any,
    nltcs_rare_vs_sqrt_ratio: Any,
    test_measured_l1_ratio: Any,
    one_way_ratio_by_dataset: Mapping[str, Any],
) -> str:
    if not isinstance(execution_valid, (bool, np.bool_)):
        raise ValueError("execution_valid 必须是布尔值")
    if not bool(execution_valid):
        return "execution_invalid"
    if not isinstance(one_way_ratio_by_dataset, Mapping) or set(
        one_way_ratio_by_dataset
    ) != set(DATASET_ORDER):
        raise ValueError("one_way_ratio_by_dataset 必须恰好覆盖两个数据集")
    nltcs_l1 = _validated_ratio(
        nltcs_measured_l1_ratio, "nltcs_measured_l1_ratio"
    )
    nltcs_common = _validated_ratio(
        nltcs_common_bin_ratio, "nltcs_common_bin_ratio"
    )
    rare_legacy = _validated_ratio(
        nltcs_rare_vs_legacy_ratio, "nltcs_rare_vs_legacy_ratio"
    )
    rare_sqrt = _validated_ratio(
        nltcs_rare_vs_sqrt_ratio, "nltcs_rare_vs_sqrt_ratio"
    )
    test_l1 = _validated_ratio(
        test_measured_l1_ratio, "test_measured_l1_ratio"
    )
    one_way = {
        dataset: _validated_ratio(
            one_way_ratio_by_dataset[dataset],
            f"one_way_ratio_by_dataset[{dataset!r}]",
        )
        for dataset in DATASET_ORDER
    }
    if nltcs_l1 >= 1.0 or nltcs_common >= 1.0:
        return "common_mechanism_not_retained"
    if rare_legacy > NLTCS_RARE_RATIO_MAX or rare_sqrt >= 1.0:
        return "rare_query_protection_not_recovered"
    if test_l1 > TEST_OVERALL_RATIO_MAX or any(
        value > ONE_WAY_RATIO_MAX for value in one_way.values()
    ):
        return "measured_or_one_way_safety_risk"
    return "advance_to_fresh_seed_confirmation"


def _flatten_baselines() -> dict[str, dict[str, Any]]:
    return {
        f"{family}_{name}": value
        for family, artifacts in BASELINE_ARTIFACTS.items()
        for name, value in artifacts.items()
    }


def frozen_protocol_manifest() -> dict[str, Any]:
    plan = task_plan()
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "gap_weight_problem_1_dual_ar_max_single_seed_screen",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "design_document": {
            "path": str(DESIGN_DOC),
            "sha256": DESIGN_DOC_SHA256,
        },
        "target_weight_audit": {
            "path": str(TARGET_WEIGHT_AUDIT),
            "sha256": TARGET_WEIGHT_AUDIT_SHA256,
        },
        "claim_boundary": {
            "development_screen_only": True,
            "candidate_selected_after_r8_and_sqrt_results": True,
            "formal_superiority_claim_allowed": False,
            "public_default_change_allowed": False,
        },
        "formula": {
            "weighting": DUAL_WEIGHTING,
            "absolute": "sum_all(abs(target-count))/(N*J)",
            "relative": (
                "sum_positive(abs(target-count)/target)"
                "/(N*sum_positive(1/target))"
            ),
            "aggregation": "max",
            "zero_target_policy": "absolute_channel_only",
            "empty_positive_target_policy": "relative_zero_then_absolute",
            "floor_applied": False,
            "max_weight_ratio": None,
            "smoothing_count": None,
            "tunable_parameter_count": 0,
        },
        "task_matrix": {
            "datasets": list(DATASET_ORDER),
            "arms": list(ARM_ORDER),
            "seed": DEVELOPMENT_SEED,
            "seed_reused_for_paired_development_diagnosis": True,
            "round_cap": ROUND_CAP,
            "candidate_budget": CANDIDATE_BUDGET,
            "trajectory_count": len(plan.tasks),
            "task_ids": [task.task_id for task in plan.tasks],
            "legacy_r8_and_sqrt_rerun_allowed": False,
        },
        "baseline_artifacts": _jsonable(BASELINE_ARTIFACTS),
        "datasets": {
            name: {
                "schema": str(DATASETS[name]["schema"]),
                "queries": str(DATASETS[name]["queries"]),
                "marginals": str(DATASETS[name]["marginals"]),
                "reference_evaluation_only": str(DATASETS[name]["reference"]),
                "n_records": DATASETS[name]["n_records"],
                "query_count": DATASETS[name]["query_count"],
                "query_identity_sha256": DATASETS[name]["query_identity_sha256"],
                "target_vector_sha256": DATASETS[name]["target_vector_sha256"],
                "generation_input_sha256": {
                    key: DATASETS[name]["input_sha256"][key]
                    for key in ("schema", "queries", "marginals")
                },
                "expected_weight_audit": EXPECTED_WEIGHT_AUDIT[name],
            }
            for name in DATASET_ORDER
        },
        "target_count_bins": _jsonable(TARGET_COUNT_BINS),
        "generator_params_by_task": generator_params_manifest_matrix(),
        "generator_params_manifest_sha256": (
            generator_params_manifest_sha256()
        ),
        "implementation_sources": _jsonable(IMPLEMENTATION_SOURCES),
        "invariants": {
            "only_c_query_objective_changes": True,
            "b_residual_channel_unchanged": True,
            "gap_l1_sweeps": 8,
            "microsteps": "8*K",
            "one_unconditional_candidate_per_round": True,
            "patience_ticks": PATIENCE_TICKS,
            "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
            "terminal_current_output": True,
            "legacy_bounded_and_sqrt_compatibility_required": True,
        },
        "screen_gates": {
            "nltcs_measured_l1_ratio_strictly_less_than": 1.0,
            "nltcs_common_bin_ratio_strictly_less_than": 1.0,
            "nltcs_rare_vs_legacy_ratio_max": NLTCS_RARE_RATIO_MAX,
            "nltcs_rare_vs_sqrt_ratio_strictly_less_than": 1.0,
            "test_measured_l1_ratio_max": TEST_OVERALL_RATIO_MAX,
            "one_way_ratio_each_dataset_max": ONE_WAY_RATIO_MAX,
            "classification_order": [
                "execution_invalid",
                "common_mechanism_not_retained",
                "rare_query_protection_not_recovered",
                "measured_or_one_way_safety_risk",
                "advance_to_fresh_seed_confirmation",
            ],
        },
        "information_flow": {
            "generation_reads_reference_or_unseen_queries": False,
            "complete_candidate_collection_before_quality_evaluation": True,
            "quality_evaluation_separately_authorized": True,
            "automatic_generation_allowed": False,
            "automatic_evaluation_allowed": False,
            "automatic_parameter_search_allowed": False,
        },
    }


def assert_frozen_protocol_identity(repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    for path, expected in (
        (PROTOCOL_DOC, PROTOCOL_DOC_SHA256),
        (DESIGN_DOC, DESIGN_DOC_SHA256),
        (TARGET_WEIGHT_AUDIT, TARGET_WEIGHT_AUDIT_SHA256),
    ):
        if file_sha256(root / path) != expected:
            raise RuntimeError(f"冻结文件 SHA-256 漂移：{path}")
    for spec in IMPLEMENTATION_SOURCES.values():
        if file_sha256(root / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"实现源码 SHA-256 漂移：{spec['path']}")
    for artifact in _flatten_baselines().values():
        if file_sha256(root / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"复用基线 SHA-256 漂移：{artifact['path']}")
    if canonical_sha256(frozen_protocol_manifest()) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("A/R 双通道冻结协议清单 SHA-256 漂移")


def require_run_confirmation(value: str) -> None:
    if value != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("必须显式确认当前 A/R 双通道冻结协议 SHA-256")


def build_plan() -> dict[str, Any]:
    plan = task_plan()
    return {
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol": frozen_protocol_manifest(),
        "seed": plan.seed,
        "tasks": [task.__dict__ | {"task_id": task.task_id} for task in plan.tasks],
        "runner_wired": plan.runner_wired,
        "generation_started": plan.generation_started,
        "raw_reference_data_accessed": False,
        "candidate_results_accessed": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    audit = subparsers.add_parser("target-audit")
    audit.add_argument("--repository-root", default=".")
    audit.add_argument("--output")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(strict_json_text(build_plan()), end="")
        return
    rendered = strict_json_text(target_weight_audit(args.repository_root))
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"拒绝覆盖目标权重审计：{output}")
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
