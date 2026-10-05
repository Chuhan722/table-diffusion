"""Issue #53 B+C 问题一：平方根查询权重单种子开发筛查协议。"""

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


PROTOCOL_VERSION = "issue53-gap-weight-sqrt-target-screen-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_BC问题一平方根查询权重单种子开发筛查结果前协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "2f227272cbbb49b260b4fdb52519896a290dc2eb1d94e4ce8ffe4533efdfda9b"
)
FROZEN_PROTOCOL_SHA256 = (
    "ecb85159f109f12a4b4e84e56c40ba8675ba315b1da05da3f68277a6c5fd3049"
)

TARGET_WEIGHT_AUDIT = Path(
    "docs/设计/Issue53_BC问题一平方根查询权重离线审计.json"
)
TARGET_WEIGHT_AUDIT_SHA256 = (
    "fe806fa7fd08b46fbdc31df988573a137bfa1bb2ec6d32782e4645a927e1d434"
)

DEVELOPMENT_SEED = 9908
ROUND_CAP = r8.ROUND_CAP
CANDIDATE_BUDGET = r8.CANDIDATE_BUDGET
PATIENCE_TICKS = r8.PATIENCE_TICKS
CHECKPOINT_ROUNDS = r8.CHECKPOINT_ROUNDS
MAX_WORKERS = r8.MAX_WORKERS

SQRT_WEIGHTING = "sqrt_target_relative"
CANDIDATE_ARM = "gap_sqrt_target_s8"
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
    "outputs/issue53_gap_weight_sqrt_target_screen_seed9908_v1"
)

BASELINE_DIR = r8.OUTPUT_DIR
BASELINE_ARTIFACTS = {
    "collection": {
        "path": BASELINE_DIR / "collection_report.json",
        "sha256": (
            "35e0b5c594dd8b502198b58690ca720449d2283702817acae843b11968bd3eb8"
        ),
    },
    "evaluation": {
        "path": BASELINE_DIR / "evaluation_report.json",
        "sha256": (
            "548b611b3762cd5489fcbd97d787ea9e91f987c5a94bed0b73e6782f6108bc4c"
        ),
    },
    "metrics_csv": {
        "path": BASELINE_DIR / "screen_metrics.csv",
        "sha256": (
            "b6d1efff24202a74003cee01eb05568db64b071aab3a47a86e3cafa4bc27b660"
        ),
    },
    "independent_audit": {
        "path": BASELINE_DIR / "independent_audit.json",
        "sha256": (
            "9e9e29c39cadc756f29dd67cc1bb17eeccccace32a167225383fca81e5ce33e6"
        ),
    },
}

EXPECTED_WEIGHT_AUDIT = {
    "test_300x10": {
        "query_count": 50,
        "minimum_target": 0,
        "maximum_target": 88,
        "minimum_denominator": 1.0,
        "maximum_denominator": 9.38083151964686,
        "actual_weight_ratio": 9.38083151964686,
    },
    "nltcs": {
        "query_count": 1001,
        "minimum_target": 12,
        "maximum_target": 13092,
        "minimum_denominator": 3.4641016151377544,
        "maximum_denominator": 114.42027792310242,
        "actual_weight_ratio": 33.03028912982749,
    },
}

IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": (
            "1d44b117bda2d1b2816d2489fc26b2b7c45e891985de8932175b40aeb761ef6a"
        ),
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": (
            "dcd604431737e18199d526186bce45be27bdbbb26af28dff63219a4c7dc3f77a"
        ),
    },
}

Dataset = Literal["test_300x10", "nltcs"]


@dataclass(frozen=True)
class SqrtWeightingScreenTask:
    dataset: Dataset
    arm: str
    seed: int
    rounds: int

    @property
    def task_id(self) -> str:
        return f"seed_{self.seed}__{self.arm}__{self.dataset}"


@dataclass(frozen=True)
class SqrtWeightingScreenPlan:
    seed: int
    tasks: tuple[SqrtWeightingScreenTask, ...]
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
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


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


def task_plan() -> SqrtWeightingScreenPlan:
    tasks = tuple(
        SqrtWeightingScreenTask(
            dataset=dataset,
            arm=CANDIDATE_ARM,
            seed=DEVELOPMENT_SEED,
            rounds=ROUND_CAP,
        )
        for dataset in DATASET_ORDER
    )
    if len(tasks) != 2 or len({task.task_id for task in tasks}) != 2:
        raise RuntimeError("平方根权重筛查任务矩阵不完整")
    return SqrtWeightingScreenPlan(seed=DEVELOPMENT_SEED, tasks=tasks)


def common_generator_params() -> dict[str, Any]:
    return r8.common_generator_params()


def arm_kernel_params(arm: str) -> dict[str, Any]:
    if arm != CANDIDATE_ARM:
        raise ValueError(f"未知平方根权重筛查方法：{arm!r}")
    return {
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_use_compiled_workload": False,
        "gap_l1_sweeps": 8,
        "gap_l1_weighting": SQRT_WEIGHTING,
        "gap_l1_max_weight_ratio": None,
    }


def task_generator_params(dataset: str, arm: str) -> dict[str, Any]:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"未知平方根权重筛查数据集：{dataset!r}")
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
        raise RuntimeError("平方根权重筛查生成参数只允许 tol=+inf")
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
            raise RuntimeError(f"{dataset} 第 {index} 条目标不是整数计数")
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


def _range_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [int(row["target"]) for row in rows]
    denominators = [float(row["denominator"]) for row in rows]
    weights = [float(row["unit_error_weight"]) for row in rows]
    return {
        "query_count": len(rows),
        "minimum_target": min(targets),
        "maximum_target": max(targets),
        "minimum_denominator": min(denominators),
        "maximum_denominator": max(denominators),
        "minimum_unit_error_weight": min(weights),
        "maximum_unit_error_weight": max(weights),
        # 与核诊断保持同一运算顺序，避免两次倒数引入末位舍入差。
        "actual_weight_ratio": max(denominators) / min(denominators),
    }


def target_weight_audit(repository_root: str | Path) -> dict[str, Any]:
    """独立读取冻结目标并计算平方根分母；不导入核实现或参考表。"""

    root = Path(repository_root).resolve()
    datasets: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        query_rows = _load_query_rows(root, dataset)
        audited_rows = []
        membership_by_bin: dict[str, list[dict[str, Any]]] = {
            spec["name"]: [] for spec in TARGET_COUNT_BINS[dataset]
        }
        for index, query in enumerate(query_rows):
            target = int(query["result"])
            bin_spec = _bin_for_target(dataset, target)
            denominator = math.sqrt(max(target, 1))
            audited_rows.append({
                "index": index,
                "id": query.get("id"),
                "target": target,
                "target_bin": bin_spec["name"],
                "denominator": denominator,
                "unit_error_weight": 1.0 / denominator,
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
                or canonical_sha256(members)
                != bin_spec["membership_sha256"]
            ):
                raise RuntimeError(f"{dataset}/{name} 冻结分箱身份漂移")
            weighted_members = [
                row for row in audited_rows if row["target_bin"] == name
            ]
            bin_summaries[name] = {
                **_range_summary(weighted_members),
                "lower_inclusive": bin_spec["lower_inclusive"],
                "upper_inclusive": bin_spec["upper_inclusive"],
                "membership_sha256": bin_spec["membership_sha256"],
            }

        summary = _range_summary(audited_rows)
        expected = EXPECTED_WEIGHT_AUDIT[dataset]
        for key, value in expected.items():
            if summary[key] != value:
                raise RuntimeError(
                    f"{dataset} 平方根权重审计漂移：{key}="
                    f"{summary[key]!r}，预期 {value!r}"
                )
        datasets[dataset] = {
            "query_file": str(DATASETS[dataset]["queries"]),
            "query_file_sha256": DATASETS[dataset]["input_sha256"]["queries"],
            "target_vector_sha256": DATASETS[dataset]["target_vector_sha256"],
            "summary": summary,
            "target_bins": bin_summaries,
            "queries": audited_rows,
        }

    return {
        "contract_version": "issue53-gap-weight-sqrt-target-audit-v1",
        "formula": {
            "denominator": "sqrt(max(target_count,1))",
            "unit_error_weight": "1/denominator",
            "target_count_quantum": 1,
            "tunable_parameter_count": 0,
        },
        "datasets": datasets,
        "audit": {
            "query_targets_only": True,
            "kernel_implementation_imported": False,
            "raw_reference_data_accessed": False,
            "legacy_or_r8_terminal_tables_accessed": False,
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
    nltcs_rare_vs_r8_ratio: Any,
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
    rare_r8 = _validated_ratio(
        nltcs_rare_vs_r8_ratio, "nltcs_rare_vs_r8_ratio"
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
    if rare_legacy > NLTCS_RARE_RATIO_MAX or rare_r8 >= 1.0:
        return "rare_query_protection_not_recovered"
    if test_l1 > TEST_OVERALL_RATIO_MAX or any(
        value > ONE_WAY_RATIO_MAX for value in one_way.values()
    ):
        return "measured_or_one_way_safety_risk"
    return "advance_to_fresh_seed_confirmation"


def frozen_protocol_manifest() -> dict[str, Any]:
    plan = task_plan()
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "gap_weight_problem_1_sqrt_target_single_seed_screen",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "target_weight_audit": {
            "path": str(TARGET_WEIGHT_AUDIT),
            "sha256": TARGET_WEIGHT_AUDIT_SHA256,
        },
        "claim_boundary": {
            "development_screen_only": True,
            "candidate_selected_after_r8_result": True,
            "formal_superiority_claim_allowed": False,
            "public_default_change_allowed": False,
        },
        "formula": {
            "weighting": SQRT_WEIGHTING,
            "denominator": "sqrt(max(target_count,1))",
            "target_count_quantum": 1,
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
            "legacy_and_r8_rerun_allowed": False,
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
            "only_c_query_denominator_changes": True,
            "b_residual_channel_unchanged": True,
            "gap_l1_sweeps": 8,
            "microsteps": "8*K",
            "one_unconditional_candidate_per_round": True,
            "patience_ticks": PATIENCE_TICKS,
            "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
            "terminal_current_output": True,
            "legacy_and_bounded_compatibility_required": True,
        },
        "screen_gates": {
            "nltcs_measured_l1_ratio_strictly_less_than": 1.0,
            "nltcs_common_bin_ratio_strictly_less_than": 1.0,
            "nltcs_rare_vs_legacy_ratio_max": NLTCS_RARE_RATIO_MAX,
            "nltcs_rare_vs_r8_ratio_strictly_less_than": 1.0,
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
        (TARGET_WEIGHT_AUDIT, TARGET_WEIGHT_AUDIT_SHA256),
    ):
        if file_sha256(root / path) != expected:
            raise RuntimeError(f"冻结文件 SHA-256 漂移：{path}")
    for spec in IMPLEMENTATION_SOURCES.values():
        if file_sha256(root / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"实现源码 SHA-256 漂移：{spec['path']}")
    for artifact in BASELINE_ARTIFACTS.values():
        if file_sha256(root / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"复用基线 SHA-256 漂移：{artifact['path']}")
    if canonical_sha256(frozen_protocol_manifest()) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("平方根权重冻结协议清单 SHA-256 漂移")


def require_run_confirmation(value: str) -> None:
    if value != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("必须显式确认当前平方根权重冻结协议 SHA-256")


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
    if args.output is None:
        print(rendered, end="")
        return
    output = Path(args.output)
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"拒绝覆盖内容不同的既有审计：{output}")
    else:
        temporary = output.with_name(f".{output.name}.tmp")
        if temporary.exists():
            raise RuntimeError(f"审计临时文件已经存在：{temporary}")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(f"target weight audit -> {output}")
    print(f"SHA-256 -> {file_sha256(output)}")


if __name__ == "__main__":
    main()
