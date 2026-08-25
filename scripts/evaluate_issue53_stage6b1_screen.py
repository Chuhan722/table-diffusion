#!/usr/bin/env python3
"""按结果前冻结门槛评价第 6B-1 阶段正式固定状态筛查。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from scripts import audit_issue53_stage6b1_structure as structural_auditor
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import calibrate_issue53_stage6b1_gap_l1 as calibrator
    from scripts import collect_issue53_stage6b1_screen as collector
    from scripts import issue53_stage6b1_protocol as protocol
else:
    import audit_issue53_stage6b1_structure as structural_auditor
    import build_issue53_stage6a_state_library as state_builder
    import calibrate_issue53_stage6b1_gap_l1 as calibrator
    import collect_issue53_stage6b1_screen as collector
    import issue53_stage6b1_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FORMAT = "issue53_stage6b1_frozen_evaluation_v1"


@dataclass(frozen=True)
class ExactErrorSystem:
    target_numerators: tuple[int, ...]
    source_denominator: int
    reduced_divisors: tuple[int, ...]
    term_denominators: tuple[int, ...]
    common_multiple: int
    weights: tuple[int, ...]
    error_denominator: int

    def units(self, counts: Sequence[int]) -> int:
        if len(counts) != len(self.target_numerators):
            raise ValueError("查询计数长度与精确目标不一致")
        total = 0
        for q, target, divisor, weight in zip(
            counts,
            self.target_numerators,
            self.reduced_divisors,
            self.weights,
        ):
            if isinstance(q, bool) or not isinstance(q, (int, np.integer)):
                raise ValueError("查询计数必须是整数")
            residual = abs(target - int(q) * self.source_denominator)
            if residual % divisor:
                raise RuntimeError("精确误差约分除数身份失败")
            total += (residual // divisor) * weight
        return total

    def fraction(self, units: int) -> Fraction:
        return Fraction(int(units), self.error_denominator)


def build_exact_error_system(
    source_target: Sequence[int],
    runtime_n_records: int,
    source_n_records: int,
) -> ExactErrorSystem:
    source = [int(value) for value in source_target]
    if not source:
        raise ValueError("至少需要一个查询")
    if runtime_n_records <= 0 or source_n_records <= 0:
        raise ValueError("记录数必须为正整数")
    targets = [value * runtime_n_records for value in source]
    raw_denominators = [
        max(target, int(protocol.GAP_L1_FLOOR) * source_n_records)
        for target in targets
    ]
    divisors = [
        math.gcd(math.gcd(abs(target), source_n_records), denominator)
        for target, denominator in zip(targets, raw_denominators)
    ]
    term_denominators = [
        denominator // divisor
        for denominator, divisor in zip(raw_denominators, divisors)
    ]
    common_multiple = math.lcm(*term_denominators)
    weights = [
        common_multiple // denominator for denominator in term_denominators
    ]
    return ExactErrorSystem(
        target_numerators=tuple(targets),
        source_denominator=int(source_n_records),
        reduced_divisors=tuple(divisors),
        term_denominators=tuple(term_denominators),
        common_multiple=common_multiple,
        weights=tuple(weights),
        error_denominator=len(source) * common_multiple,
    )


def _exact_record(value: Fraction) -> dict[str, Any]:
    sign = "-" if value.numerator < 0 else ""
    numerator_hex = sign + format(abs(value.numerator), "x")
    return {
        "numerator_hex": numerator_hex,
        "denominator_hex": format(value.denominator, "x"),
        "float_reading_only": float(value),
    }


def _strip_diagnostic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_diagnostic(item)
            for key, item in value.items()
            if not key.endswith("_diagnostic_only") and key != "environment"
        }
    if isinstance(value, list):
        return [_strip_diagnostic(item) for item in value]
    return value


def scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return _strip_diagnostic({
        key: item
        for key, item in value.items()
        if key not in {
            "evaluation_scientific_sha256",
            "artifact_paths_diagnostic_only",
            "environment",
            "elapsed_sec_diagnostic_only",
        }
    })


def _validate_execution(
    mode: str,
    confirmed_execution_commit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    git = state_builder._git_identity(REPOSITORY_ROOT)
    if not git["worktree_clean_including_untracked"]:
        raise RuntimeError("冻结评价要求包含未跟踪文件在内的干净工作树")
    if confirmed_execution_commit != git["commit"]:
        raise PermissionError("必须精确确认当前干净实现提交")
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=False,
    )
    return git, environment


def _load_chain(
    mode: str,
    calibration_path: str | Path,
    collection_path: str | Path,
    structural_audit_path: str | Path,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    paths = {
        "calibration": Path(calibration_path).resolve(),
        "collection": Path(collection_path).resolve(),
        "structural_audit": Path(structural_audit_path).resolve(),
    }
    hashes = {name: protocol.file_sha256(path) for name, path in paths.items()}
    calibration = state_builder._strict_load_json(paths["calibration"])
    collection = state_builder._strict_load_json(paths["collection"])
    audit = state_builder._strict_load_json(paths["structural_audit"])
    calibrator.validate_calibration_manifest(calibration, mode=mode)
    collector.validate_collection(collection, mode=mode)
    structural_auditor.validate_structural_audit(audit, mode=mode)
    if (
        any(
            artifact.get("git", {}).get("commit") != git_commit
            for artifact in (calibration, collection, audit)
        )
        or collection["calibration_artifact"]["file_sha256"]
        != hashes["calibration"]
        or audit["artifact_identity"]["calibration_file_sha256"]
        != hashes["calibration"]
        or audit["artifact_identity"]["collection_file_sha256"]
        != hashes["collection"]
        or audit.get("audit_passed") is not True
    ):
        raise RuntimeError("冻结评价输入产物链或结构审计资格失败")
    return calibration, collection, audit, hashes


def _source_targets(dataset: str) -> list[int]:
    path = REPOSITORY_ROOT / protocol.DATASETS[dataset]["queries"]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return [int(query["result"]) for query in value["queries"]]


def _apply_sparse_delta(
    before: Sequence[int],
    sparse: Sequence[Sequence[int]],
) -> list[int]:
    result = [int(value) for value in before]
    seen = set()
    for item in sparse:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("稀疏查询改变量结构无效")
        index, delta = item
        if (
            isinstance(index, bool)
            or isinstance(delta, bool)
            or not isinstance(index, int)
            or not isinstance(delta, int)
            or not 0 <= index < len(result)
            or index in seen
            or delta == 0
        ):
            raise ValueError("稀疏查询改变量索引/数值无效")
        seen.add(index)
        result[index] += delta
    return result


def _mean(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("精确均值分母不足")
    return sum(values, Fraction(0, 1)) / len(values)


def _comparison(
    new_values: Mapping[int, Fraction],
    baseline_values: Mapping[int, Fraction],
    *,
    relation: str,
) -> dict[str, Any]:
    seeds = tuple(new_values)
    if seeds != tuple(baseline_values):
        raise RuntimeError("随机种子精确汇总覆盖不一致")
    new_mean = _mean([new_values[seed] for seed in seeds])
    baseline_mean = _mean([baseline_values[seed] for seed in seeds])
    if relation == "strict_lower":
        per_seed = {
            seed: new_values[seed] < baseline_values[seed] for seed in seeds
        }
        aggregate = new_mean < baseline_mean
    elif relation == "noninferior_1p05":
        bound = Fraction(105, 100)
        per_seed = {
            seed: new_values[seed] <= bound * baseline_values[seed]
            for seed in seeds
        }
        aggregate = new_mean <= bound * baseline_mean
    else:
        raise ValueError("未知冻结比较关系")
    stable_count = sum(per_seed.values())
    return {
        "relation": relation,
        "new_equal_seed_mean": _exact_record(new_mean),
        "baseline_equal_seed_mean": _exact_record(baseline_mean),
        "per_seed": {
            str(seed): {
                "new": _exact_record(new_values[seed]),
                "baseline": _exact_record(baseline_values[seed]),
                "passes": per_seed[seed],
            }
            for seed in seeds
        },
        "passing_seed_count": stable_count,
        "passes": bool(
            aggregate and stable_count >= protocol.SEED_STABILITY_MINIMUM
        ),
    }


def _retention_comparison(
    pair_metrics: Sequence[dict[str, Any]],
    baseline: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    per_seed = {}
    missing = []
    for seed in seeds:
        rows = [row for row in pair_metrics if row["seed"] == seed]
        selected = [
            row for row in rows if row["gain_units"][baseline] > 0
        ]
        denominator = sum(
            row["gain_units"][baseline] for row in selected
        )
        if denominator == 0:
            missing.append(seed)
            continue
        numerator = sum(
            max(row["gain_units"][protocol.ARM_GAP_L1], 0)
            for row in selected
        )
        per_seed[seed] = Fraction(numerator, denominator)
    support = not missing
    if support:
        equal_seed_mean = _mean(list(per_seed.values()))
        passing = sum(
            value >= Fraction(90, 100) for value in per_seed.values()
        )
        passes = bool(
            equal_seed_mean >= Fraction(95, 100)
            and passing >= protocol.SEED_STABILITY_MINIMUM
        )
    else:
        equal_seed_mean = None
        passing = 0
        passes = False
    return {
        "support_sufficient": support,
        "missing_positive_gain_seed_denominators": missing,
        "equal_seed_mean": (
            _exact_record(equal_seed_mean)
            if equal_seed_mean is not None else None
        ),
        "per_seed": {
            str(seed): _exact_record(value)
            for seed, value in per_seed.items()
        },
        "passing_seed_count_at_0p90": passing,
        "passes": passes,
    }


def evaluate_dataset(
    dataset: str,
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    state_index = {
        row["state_id"]: row for row in collection["state_manifest"]
    }
    # 运行记录数直接由冻结模式决定：状态清单的计数表行数不落盘，来源协议
    # 已把正式 N 和 smoke N 绑定。
    mode = collection["mode"]
    runtime_n = (
        protocol.DATASETS[dataset]["n_records"] if mode == "formal" else 128
    )
    source_n = protocol.DATASETS[dataset]["n_records"]
    system = build_exact_error_system(
        _source_targets(dataset), runtime_n, source_n
    )
    seeds = protocol.mode_seeds(mode)
    metrics = []
    for pair in collection["pairs"]:
        if pair["dataset"] != dataset:
            continue
        state_id = protocol.state_id(
            dataset,
            int(pair["source_seed"]),
            pair["state_group"],
            mode=mode,
        )
        current_counts = state_index[state_id]["current_query_counts"]
        current_units = system.units(current_counts)
        arm_units = {}
        gain_units = {}
        full_units = {}
        for arm in protocol.ARMS:
            record = pair["arms"][arm]
            copy_counts = _apply_sparse_delta(
                current_counts, record["query_delta"]["copy_only"]
            )
            full_counts = _apply_sparse_delta(
                current_counts, record["query_delta"]["full"]
            )
            arm_units[arm] = system.units(copy_counts)
            full_units[arm] = system.units(full_counts)
            gain_units[arm] = current_units - arm_units[arm]
        metrics.append({
            "pair_id": pair["pair_id"],
            "seed": int(pair["source_seed"]),
            "state_group": pair["state_group"],
            "eligible": pair["address_status"] != "already_exact_deterministic_no_op",
            "current_units": current_units,
            "copy_units": arm_units,
            "full_units": full_units,
            "gain_units": gain_units,
        })

    def seed_means(
        groups: Sequence[str],
        arm: str,
        field: str,
        transform=lambda value: value,
    ) -> dict[int, Fraction]:
        result = {}
        for seed in seeds:
            values = [
                transform(row[field][arm])
                for row in metrics
                if row["eligible"]
                and row["seed"] == seed
                and row["state_group"] in groups
            ]
            # 若一个 seed 的相关状态全部已经精确命中，三组确定性相同；用零
            # 表示这一无风险等价，不把地址伪造成效果证据。
            result[seed] = (
                Fraction(sum(values), len(values) * system.error_denominator)
                if values else Fraction(0, 1)
            )
        return result

    baselines = (protocol.ARM_INDEPENDENT, protocol.ARM_FACTOR)
    primary_gap = {}
    harm = {}
    retention = {}
    initial_safety = {}
    full_safety = {}
    new_primary_copy = seed_means(
        protocol.PRIMARY_STATE_GROUPS, protocol.ARM_GAP_L1, "copy_units"
    )
    new_primary_harm = seed_means(
        protocol.PRIMARY_STATE_GROUPS,
        protocol.ARM_GAP_L1,
        "gain_units",
        transform=lambda value: max(-value, 0),
    )
    new_initial = seed_means(
        ("initial",), protocol.ARM_GAP_L1, "copy_units"
    )
    new_full = seed_means(
        protocol.PRIMARY_STATE_GROUPS, protocol.ARM_GAP_L1, "full_units"
    )
    primary_rows = [
        row for row in metrics
        if row["eligible"] and row["state_group"] in protocol.PRIMARY_STATE_GROUPS
    ]
    for baseline in baselines:
        baseline_primary = seed_means(
            protocol.PRIMARY_STATE_GROUPS, baseline, "copy_units"
        )
        primary_gap[baseline] = _comparison(
            new_primary_copy, baseline_primary, relation="strict_lower"
        )
        baseline_harm = seed_means(
            protocol.PRIMARY_STATE_GROUPS,
            baseline,
            "gain_units",
            transform=lambda value: max(-value, 0),
        )
        harm_support = any(value > 0 for value in baseline_harm.values())
        harm[baseline] = {
            "support_sufficient": harm_support,
            **_comparison(
                new_primary_harm,
                baseline_harm,
                relation="strict_lower",
            ),
        }
        if not harm_support:
            harm[baseline]["passes"] = False
            harm[baseline]["support_label"] = "insufficient_harm_support"
        retention[baseline] = _retention_comparison(
            primary_rows, baseline, seeds
        )
        initial_safety[baseline] = _comparison(
            new_initial,
            seed_means(("initial",), baseline, "copy_units"),
            relation="noninferior_1p05",
        )
        full_safety[baseline] = _comparison(
            new_full,
            seed_means(
                protocol.PRIMARY_STATE_GROUPS, baseline, "full_units"
            ),
            relation="noninferior_1p05",
        )

    stable_gap_pass = all(primary_gap[arm]["passes"] for arm in baselines)
    transition_support = all(
        harm[arm]["support_sufficient"]
        and retention[arm]["support_sufficient"]
        for arm in baselines
    )
    harm_pass = all(harm[arm]["passes"] for arm in baselines)
    retention_pass = all(retention[arm]["passes"] for arm in baselines)
    initial_pass = all(initial_safety[arm]["passes"] for arm in baselines)
    full_pass = all(full_safety[arm]["passes"] for arm in baselines)
    label = protocol.classify_dataset(
        execution_label=None,
        stable_gap_error_gain=stable_gap_pass,
        transition_support=transition_support,
        harm_suppression=harm_pass,
        good_step_retention=retention_pass,
        initial_safety=initial_pass,
        full_safety=full_pass,
    )
    return {
        "dataset": dataset,
        "exact_error_system": {
            "source_denominator": source_n,
            "runtime_n_records": runtime_n,
            "query_count": len(system.target_numerators),
            "common_multiple_hex": format(system.common_multiple, "x"),
            "error_denominator_hex": format(system.error_denominator, "x"),
        },
        "eligible_primary_pair_count": len(primary_rows),
        "primary_gap_error": primary_gap,
        "negative_harm": harm,
        "good_step_retention": retention,
        "initial_safety": initial_safety,
        "full_safety": full_safety,
        "frozen_boolean_summary": {
            "stable_gap_error_gain": stable_gap_pass,
            "transition_support": transition_support,
            "harm_suppression": harm_pass,
            "good_step_retention": retention_pass,
            "initial_safety": initial_pass,
            "full_safety": full_pass,
        },
        "dataset_classification": label,
        "address_exact_metric_sha256": protocol.canonical_sha256([
            {
                "pair_id": row["pair_id"],
                "current_units_hex": format(row["current_units"], "x"),
                "copy_units_hex": {
                    arm: format(row["copy_units"][arm], "x")
                    for arm in protocol.ARMS
                },
                "full_units_hex": {
                    arm: format(row["full_units"][arm], "x")
                    for arm in protocol.ARMS
                },
            }
            for row in metrics
        ]),
    }


def validate_evaluation(value: Mapping[str, Any], *, mode: str) -> None:
    if (
        value.get("evaluation_format") != EVALUATION_FORMAT
        or value.get("status") != "complete"
        or value.get("mode") != mode
        or value.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or value.get("evaluation_scientific_sha256")
        != protocol.canonical_sha256(scientific_payload(value))
        or (mode == "smoke" and (
            value.get("formal_result_valid") is not False
            or value.get("mechanism_evidence_emitted") is not False
            or value.get("final_screen_classification") is not None
            or value.get("dataset_evaluations") is not None
        ))
    ):
        raise RuntimeError("第 6B-1 阶段冻结评价结构/身份失败")


def evaluate_screen(
    mode: str,
    calibration_path: str | Path,
    collection_path: str | Path,
    structural_audit_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None,
    confirmed_execution_commit: str | None,
) -> tuple[Path, dict[str, Any]]:
    protocol.require_run_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"冻结评价输出已存在，不覆盖：{output}")
    git, environment = _validate_execution(
        mode, confirmed_execution_commit
    )
    calibration, collection, audit, hashes = _load_chain(
        mode,
        calibration_path,
        collection_path,
        structural_audit_path,
        git["commit"],
    )
    started = time.perf_counter()
    # smoke 仍执行全部精确算术，以验证评价器接线；结果只保留校验哈希，禁止
    # 发布逐数据效果、门槛真假或机制分类。
    computed = {
        dataset: evaluate_dataset(dataset, collection)
        for dataset in protocol.DATASET_ORDER
    }
    if mode == "formal":
        labels = {
            dataset: computed[dataset]["dataset_classification"]
            for dataset in protocol.DATASET_ORDER
        }
        final = protocol.classify_cross_dataset(labels)
        dataset_evaluations: Any = computed
        smoke_validation = None
        evidence_emitted = True
    else:
        final = None
        dataset_evaluations = None
        smoke_validation = {
            "exact_evaluator_completed": True,
            "dataset_count": len(computed),
            "internal_result_sha256_not_mechanism_evidence": (
                protocol.canonical_sha256(computed)
            ),
            "threshold_results_suppressed": True,
        }
        evidence_emitted = False
    value = {
        "evaluation_format": EVALUATION_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "artifact_role": (
            "formal_frozen_development_screen_evaluation"
            if mode == "formal" else "pipeline_smoke_only"
        ),
        "mechanism_evidence_emitted": evidence_emitted,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": state_builder._json_safe(environment),
        "artifact_identity": {
            **{f"{name}_file_sha256": digest for name, digest in hashes.items()},
            "calibration_scientific_sha256": calibration[
                "calibration_scientific_sha256"
            ],
            "collection_scientific_sha256": collection[
                "collection_scientific_sha256"
            ],
            "structural_audit_scientific_sha256": audit[
                "structural_audit_scientific_sha256"
            ],
        },
        "dataset_evaluations": dataset_evaluations,
        "smoke_pipeline_validation": smoke_validation,
        "final_screen_classification": final,
        "default_kernel_changed": False,
        "artifact_paths_diagnostic_only": {
            "calibration": str(Path(calibration_path).resolve()),
            "collection": str(Path(collection_path).resolve()),
            "structural_audit": str(Path(structural_audit_path).resolve()),
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    value["evaluation_scientific_sha256"] = protocol.canonical_sha256(
        scientific_payload(value)
    )
    validate_evaluation(value, mode=mode)
    published = state_builder._exclusive_write_json(output, value)
    return published, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--collection", required=True)
    run.add_argument("--structural-audit", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--confirmed-protocol-sha256", required=True)
    run.add_argument("--confirmed-execution-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        plan = protocol.build_plan(args.mode)
        plan.update({
            "evaluation_kind": "frozen_exact_rational_seed_equal_weight",
            "source_read_started": False,
            "evaluation_started": False,
        })
        print(json.dumps(
            plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ))
        return
    evaluate_screen(
        args.mode,
        args.calibration,
        args.collection,
        args.structural_audit,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
