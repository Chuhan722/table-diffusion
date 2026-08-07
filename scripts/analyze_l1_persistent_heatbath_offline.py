"""对 L1 持久热浴阶段 I 输出做生成后独立质量评价。

本脚本先只用公开输入重建初始表与两种最终合成表，确认哈希、查询答案和 loss；
全部生成状态通过审计后才读取真实参考表。离线指标不参与生成、参数选择、早停、
checkpoint 或输出表选择。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

if __package__:
    from scripts import analyze_persistent_heatbath_offline as base_analysis
    from scripts import compare_factorized_gibbs_closed_loop as offline_metrics
    from scripts import probe_l1_persistent_heatbath as experiment
else:
    import analyze_persistent_heatbath_offline as base_analysis
    import compare_factorized_gibbs_closed_loop as offline_metrics
    import probe_l1_persistent_heatbath as experiment
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.persistent_heatbath import (
    initialize_persistent_heatbath_state,
)
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


FORMAL_INPUT = experiment.FORMAL_OUTPUT
FORMAL_OUTPUT = Path(
    "outputs/l1_persistent_workload_heatbath/"
    "offline_stage1_20seed_3000step_tau1.json"
)
REAL_DATA_PATH = Path("data/test_300x10/test_300x10.csv")
QUALITY_METRIC_DIRECTIONS = base_analysis.QUALITY_METRIC_DIRECTIONS
MEASURED_REQUIRED_WINS = 14
THREE_WAY_REQUIRED_WINS = 14
THREE_WAY_REQUIRED_NONWORSE_VS_INITIAL = 11
MINIMUM_ENTROPY_RATIO = 0.50
MINIMUM_UPHILL_MASS = 0.01


def _summarize(values):
    result = base_analysis._summarize(values)
    array = np.asarray(values, dtype=float)
    result["median"] = float(np.median(array))
    return result


def _paired_summary(candidate, reference, metric):
    candidate_values = np.asarray(
        [row[metric] for row in candidate], dtype=float
    )
    reference_values = np.asarray(
        [row[metric] for row in reference], dtype=float
    )
    if candidate_values.shape != reference_values.shape:
        raise ValueError(f"{metric} 的配对形状不一致")
    differences = candidate_values - reference_values
    lower_is_better = QUALITY_METRIC_DIRECTIONS[metric]
    improved = differences < 0.0 if lower_is_better else differences > 0.0
    worsened = differences > 0.0 if lower_is_better else differences < 0.0
    standard_deviation = (
        float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    )
    mean = float(differences.mean())
    if standard_deviation > 0.0:
        effect_size = float(mean / standard_deviation)
    elif mean == 0.0:
        effect_size = 0.0
    else:
        effect_size = None
    return {
        **_summarize(differences),
        "candidate_minus_reference": True,
        "lower_is_better": lower_is_better,
        "improved": int(np.sum(improved)),
        "ties": int(np.sum(differences == 0.0)),
        "nonworse": int(np.sum(~worsened)),
        "worsened": int(np.sum(worsened)),
        "paired_cohens_dz": effect_size,
    }


def _final_state(run, variant, schema, queries, target, n_records):
    trajectory = run[variant]
    names = schema.attribute_names()
    records = trajectory["final_table_records"]
    if (
        not isinstance(records, list)
        or len(records) != n_records
        or any(
            not isinstance(record, dict) or set(record) != set(names)
            for record in records
        )
    ):
        raise ValueError(f"seed {run['seed']} {variant} 最终表结构无效")
    state = initialize_persistent_heatbath_state(
        pd.DataFrame(records, columns=names), schema, queries, target
    )
    if (
        experiment.common._frame_sha256(state.table)
        != trajectory["final_table_sha256"]
        or [int(value) for value in state.query_answers]
        != trajectory["final_query_answers"]
        or not experiment.common._float_close(
            state.loss, trajectory["loss_history"][-1]
        )
        or not experiment.common._float_close(
            experiment._normalized_l1(
                target, state.query_answers, n_records
            ),
            trajectory["normalized_l1_history"][-1],
        )
    ):
        raise ValueError(f"seed {run['seed']} {variant} 最终表审计失败")
    return state


def _reconstruct_generated_states(payload, schema, queries, marginals):
    target = np.asarray(payload["target"], dtype=float)
    n_records = int(payload["protocol"]["n_records"])
    states = {name: [] for name in ("initial", "baseline", "candidate")}
    for run in payload["runs"]:
        initial_table = init_synthetic_table(
            n_records,
            schema,
            np.random.default_rng(int(run["seed"])),
            marginals=marginals,
        )
        initial_state = initialize_persistent_heatbath_state(
            initial_table, schema, queries, target
        )
        if (
            experiment.common._frame_sha256(initial_state.table)
            != run["initial_table_sha256"]
            or [int(value) for value in initial_state.query_answers]
            != run["initial_query_answers"]
            or not experiment.common._float_close(
                initial_state.loss, run["initial_loss"]
            )
            or not experiment.common._float_close(
                experiment._normalized_l1(
                    target, initial_state.query_answers, n_records
                ),
                run["initial_normalized_l1"],
            )
        ):
            raise ValueError(f"seed {run['seed']} 初始表审计失败")
        states["initial"].append(initial_state)
        for variant in ("baseline", "candidate"):
            states[variant].append(_final_state(
                run, variant, schema, queries, target, n_records
            ))
    return states


def _classification(by_variant, paired, generation_payload):
    candidate = by_variant["candidate"]
    initial = by_variant["initial"]
    baseline = by_variant["baseline"]
    candidate_vs_initial = paired["candidate_minus_initial"]
    candidate_vs_baseline = paired["candidate_minus_baseline"]
    measured_initial = candidate_vs_initial["training_normalized_l1"]
    measured_baseline = candidate_vs_baseline["training_normalized_l1"]
    three_way_initial = candidate_vs_initial["unmeasured_3way_l1"]
    three_way_baseline = candidate_vs_baseline["unmeasured_3way_l1"]

    measured_gate = bool(
        candidate["training_normalized_l1"]["mean"]
        < initial["training_normalized_l1"]["mean"]
        and measured_initial["improved"] >= MEASURED_REQUIRED_WINS
        and candidate["training_normalized_l1"]["mean"]
        < baseline["training_normalized_l1"]["mean"]
        and measured_baseline["improved"] >= MEASURED_REQUIRED_WINS
    )
    three_way_gate = bool(
        candidate["unmeasured_3way_l1"]["mean"]
        <= initial["unmeasured_3way_l1"]["mean"]
        and three_way_initial["nonworse"]
        >= THREE_WAY_REQUIRED_NONWORSE_VS_INITIAL
        and candidate["unmeasured_3way_l1"]["mean"]
        < baseline["unmeasured_3way_l1"]["mean"]
        and three_way_baseline["improved"] >= THREE_WAY_REQUIRED_WINS
    )
    candidate_mechanism = generation_payload["aggregate"][
        "paired_metrics"
    ]
    entropy_mean = candidate_mechanism[
        "conditional_normalized_entropy_mean"
    ]["candidate"]["mean"]
    uphill_mean = candidate_mechanism[
        "uphill_probability_mass_mean"
    ]["candidate"]["mean"]
    collapse_risk = bool(
        entropy_mean < MINIMUM_ENTROPY_RATIO
        or uphill_mean < MINIMUM_UPHILL_MASS
    )
    semantic_gates = bool(
        generation_payload["aggregate"][
            "all_diagnostic_gates_passed"
        ]
        and generation_payload["independent_audit"]["passed"]
    )
    if not semantic_gates or not measured_gate:
        label = "not_supported"
    elif collapse_risk:
        label = "exploration_collapse_risk"
    elif three_way_gate:
        label = "target_aligned_and_independent_quality_not_degraded"
    else:
        label = "training_target_only"
    return {
        "label": label,
        "semantic_gates_passed": semantic_gates,
        "measured_l1_gate_passed": measured_gate,
        "unmeasured_3way_gate_passed": three_way_gate,
        "exploration_collapse_risk": collapse_risk,
        "candidate_entropy_ratio_mean": float(entropy_mean),
        "candidate_l1_uphill_probability_mass_mean": float(uphill_mean),
        "thresholds": {
            "measured_required_improvements": MEASURED_REQUIRED_WINS,
            "three_way_required_improvements_vs_square": (
                THREE_WAY_REQUIRED_WINS
            ),
            "three_way_required_nonworse_vs_initial": (
                THREE_WAY_REQUIRED_NONWORSE_VS_INITIAL
            ),
            "minimum_entropy_ratio": MINIMUM_ENTROPY_RATIO,
            "minimum_uphill_probability_mass": MINIMUM_UPHILL_MASS,
        },
    }


def build_analysis(payload, schema, queries, marginals, reference, states):
    """在生成状态完全固定后计算指标和预注册阶段 I 分类。"""
    if not isinstance(reference, pd.DataFrame) or len(reference) == 0:
        raise ValueError("reference 必须是非空 DataFrame")
    columns = schema.attribute_names()
    if list(reference.columns) != columns:
        raise ValueError("reference 列及顺序必须与 schema 完全一致")
    if len(reference) != int(payload["protocol"]["n_records"]):
        raise ValueError("reference 行数与公开记录数不一致")
    if set(states) != {"initial", "baseline", "candidate"}:
        raise ValueError("states 缺少初始、baseline 或 candidate")
    if any(
        len(variant_states) != len(payload["runs"])
        for variant_states in states.values()
    ):
        raise ValueError("states 与生成 seed 数量不一致")
    domains = offline_metrics._discretization_domains(marginals)
    measured_triples = offline_metrics._measured_cell_keys(
        queries, marginals, order=3
    )
    target = np.asarray(payload["target"], dtype=float)
    n_records = int(payload["protocol"]["n_records"])
    rows = {name: [] for name in states}
    for name, variant_states in states.items():
        for run, state in zip(payload["runs"], variant_states):
            offline = offline_metrics._offline_metrics(
                reference,
                state.table,
                marginals,
                domains,
                measured_triples,
            )
            rows[name].append({
                "seed": int(run["seed"]),
                **base_analysis._flatten_metrics(
                    target, n_records, state, offline
                ),
            })
    by_variant = {
        name: {
            metric: _summarize([row[metric] for row in variant_rows])
            for metric in QUALITY_METRIC_DIRECTIONS
        }
        for name, variant_rows in rows.items()
    }
    paired = {}
    for reference_name in ("initial", "baseline"):
        paired[f"candidate_minus_{reference_name}"] = {
            metric: _paired_summary(
                rows["candidate"], rows[reference_name], metric
            )
            for metric in QUALITY_METRIC_DIRECTIONS
        }
    classification = _classification(by_variant, paired, payload)
    return {
        "reference_rows": int(len(reference)),
        "reference_scope": (
            "single offline reference table; no independent train/test split"
        ),
        "measured_3way_cells_excluded": int(len(measured_triples)),
        "metric_order": [
            "unmeasured_3way_l1",
            "training_normalized_l1",
            "unmeasured_4way_l1",
            "training_workload_loss",
            "support_and_joint_diagnostics",
        ],
        "rows": rows,
        "by_variant": by_variant,
        "paired": paired,
        "classification": classification,
    }


def _load_fixed_inputs():
    schema = load_schema(str(experiment.SCHEMA_PATH))
    queries = load_queries(str(experiment.QUERY_PATH))
    marginals = load_marginals(str(experiment.MARGINALS_PATH))
    return schema, queries, marginals


def _validate_source_payload(payload):
    if (
        not experiment._formal_payload_matches(payload)
        or payload.get("aggregate", {}).get("classification")
        != "generation_complete_pending_offline_quality"
        or payload.get("aggregate", {}).get(
            "all_diagnostic_gates_passed"
        ) is not True
        or payload.get("independent_audit", {}).get("passed") is not True
    ):
        raise ValueError("输入不是通过语义门禁的 L1 阶段 I 正式输出")


def _recompute(input_path):
    payload = experiment.common._load_json_strict(input_path)
    _validate_source_payload(payload)
    schema, queries, marginals = _load_fixed_inputs()
    states = _reconstruct_generated_states(
        payload, schema, queries, marginals
    )
    if any(len(value) != len(payload["runs"]) for value in states.values()):
        raise RuntimeError("生成状态数量审计失败")
    # 隐私边界：全部合成状态验证完毕后，才从这里读取真实参考表。
    reference = pd.read_csv(REAL_DATA_PATH)[schema.attribute_names()]
    analysis = build_analysis(
        payload, schema, queries, marginals, reference, states
    )
    return payload, analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FORMAL_INPUT)
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument(
        "--audit-existing",
        type=Path,
        help="重算并只读审计已有离线评价 JSON",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    environment = None
    if args.audit_existing is None:
        environment = experiment.common._environment_snapshot("cpu")
        if not environment["git_worktree_clean_including_untracked"]:
            raise RuntimeError(
                "正式离线评价要求当前提交对应的工作树完全干净"
            )
    payload, analysis = _recompute(args.input)
    source_hash = experiment.common._sha256_file(args.input)
    reference_hash = experiment.common._sha256_file(REAL_DATA_PATH)

    if args.audit_existing is not None:
        existing = experiment.common._load_json_strict(args.audit_existing)
        passed = bool(
            existing.get("source", {}).get("generation_result_sha256")
            == source_hash
            and existing.get("source", {}).get("reference_sha256")
            == reference_hash
            and existing.get("analysis") == analysis
        )
        audit = {
            "passed": passed,
            "generation_result_sha256": source_hash,
            "reference_sha256": reference_hash,
            "checked_seeds": len(payload["runs"]),
            "checked_tables": 3 * len(payload["runs"]),
        }
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        if not passed:
            raise RuntimeError("已有离线评价独立审计失败")
        return

    report = {
        "experiment": "l1_persistent_workload_heatbath_offline_stage1",
        "boundary": (
            "post-generation evaluation only; never used by generation, "
            "parameter selection, early stopping, or output selection"
        ),
        "environment": environment,
        "source": {
            "generation_result_path": str(args.input),
            "generation_result_sha256": source_hash,
            "generation_commit": payload["environment"]["git_commit"],
            "generation_classification": payload["aggregate"][
                "classification"
            ],
            "reference_path": str(REAL_DATA_PATH),
            "reference_sha256": reference_hash,
            "public_input_sha256": payload["public_input_sha256"],
        },
        "analysis": analysis,
        "elapsed_sec": float(time.perf_counter() - started),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    experiment.common._write_json_atomic(args.output, report)
    print(f"classification={analysis['classification']['label']}")
    print(f"output={args.output}")
    print(f"sha256={experiment.common._sha256_file(args.output)}")
    print(json.dumps({
        "measured_l1": {
            name: analysis["by_variant"][name][
                "training_normalized_l1"
            ]["mean"]
            for name in ("initial", "baseline", "candidate")
        },
        "unmeasured_3way_l1": {
            name: analysis["by_variant"][name][
                "unmeasured_3way_l1"
            ]["mean"]
            for name in ("initial", "baseline", "candidate")
        },
        "classification": analysis["classification"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
