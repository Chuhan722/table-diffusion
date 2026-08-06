"""对持久化热浴正式输出做生成后离线质量评价。

本脚本只读取已经完成并通过审计的生成结果。它先用公开输入重建所有初始表和最终
合成表，确认哈希、查询答案与 loss；此后才读取真实参考表。离线指标不参与生成、
正式分类、参数选择、早停或输出表选择。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from scripts import compare_factorized_gibbs_closed_loop as offline_metrics
    from scripts import probe_persistent_heatbath as experiment
else:
    import compare_factorized_gibbs_closed_loop as offline_metrics
    import probe_persistent_heatbath as experiment
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.persistent_heatbath import (
    initialize_persistent_heatbath_state,
)
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


FORMAL_INPUT = experiment.FORMAL_OUTPUT
FORMAL_OUTPUT = Path(
    "outputs/persistent_workload_heatbath/"
    "offline_formal_20seed_3000step_tau1.json"
)
REAL_DATA_PATH = Path("data/test_300x10/test_300x10.csv")
QUALITY_METRIC_DIRECTIONS = {
    "training_workload_loss": True,
    "training_normalized_l1": True,
    "unmeasured_3way_l1": True,
    "unmeasured_4way_l1": True,
    "raw_joint_tvd": True,
    "binned_joint_tvd": True,
    "raw_unique_states": False,
    "binned_unique_states": False,
    "raw_support_overlap": False,
    "binned_support_overlap": False,
    "raw_missing_reference_mass": True,
    "raw_novel_synthetic_mass": True,
}


def _summarize(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("汇总输入必须是非空有限一维数组")
    mean = float(array.mean())
    if len(array) == 1:
        interval = [mean, mean]
        standard_deviation = 0.0
    else:
        standard_deviation = float(array.std(ddof=1))
        standard_error = standard_deviation / np.sqrt(len(array))
        critical = float(stats.t.ppf(0.975, df=len(array) - 1))
        interval = [
            float(mean - critical * standard_error),
            float(mean + critical * standard_error),
        ]
    return {
        "n": int(len(array)),
        "mean": mean,
        "std": standard_deviation,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean_t_interval_95": interval,
        "values": array.tolist(),
    }


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
    return {
        **_summarize(differences),
        "candidate_minus_reference": True,
        "lower_is_better": lower_is_better,
        "improved": int(np.sum(improved)),
        "ties": int(np.sum(differences == 0.0)),
        "worsened": int(np.sum(worsened)),
    }


def _flatten_metrics(target, n_records, state, offline):
    absolute_errors = np.abs(
        np.asarray(target, dtype=float)
        - state.query_answers.astype(float, copy=False)
    )
    return {
        "training_workload_loss": float(
            compute_loss(target, state.query_answers)
        ),
        "training_normalized_l1": float(
            np.mean(absolute_errors) / n_records
        ),
        "unmeasured_3way_l1": float(
            offline["unmeasured_3way"]["mean"]
        ),
        "unmeasured_4way_l1": float(
            offline["unmeasured_4way"]["mean"]
        ),
        "raw_joint_tvd": float(offline["raw_joint"]["tvd"]),
        "binned_joint_tvd": float(offline["binned_joint"]["tvd"]),
        "raw_unique_states": int(offline["raw_joint"]["n_unique"]),
        "binned_unique_states": int(
            offline["binned_joint"]["n_unique"]
        ),
        "raw_support_overlap": int(
            offline["raw_joint"]["support_overlap"]
        ),
        "binned_support_overlap": int(
            offline["binned_joint"]["support_overlap"]
        ),
        "raw_missing_reference_mass": float(
            offline["raw_joint"]["missing_reference_mass"]
        ),
        "raw_novel_synthetic_mass": float(
            offline["raw_joint"]["novel_synthetic_mass"]
        ),
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
    table = pd.DataFrame(records, columns=names)
    state = initialize_persistent_heatbath_state(
        table, schema, queries, target
    )
    if (
        experiment._frame_sha256(state.table)
        != trajectory["final_table_sha256"]
        or [int(value) for value in state.query_answers]
        != trajectory["final_query_answers"]
        or not experiment._float_close(
            state.loss, trajectory["loss_history"][-1]
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
            experiment._frame_sha256(initial_state.table)
            != run["initial_table_sha256"]
            or [int(value) for value in initial_state.query_answers]
            != run["initial_query_answers"]
            or not experiment._float_close(
                initial_state.loss, run["initial_loss"]
            )
        ):
            raise ValueError(f"seed {run['seed']} 初始表审计失败")
        states["initial"].append(initial_state)
        for variant in ("baseline", "candidate"):
            states[variant].append(_final_state(
                run, variant, schema, queries, target, n_records
            ))
    return states


def _relative_change(candidate, reference):
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else None
    return float((candidate - reference) / abs(reference))


def build_analysis(payload, schema, queries, marginals, reference, states):
    """在生成状态已固定后计算确定性的离线质量汇总。"""
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
                **_flatten_metrics(
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

    candidate = by_variant["candidate"]
    initial = by_variant["initial"]
    risk_changes = {
        metric: _relative_change(
            candidate[metric]["mean"], initial[metric]["mean"]
        )
        for metric in (
            "unmeasured_3way_l1",
            "unmeasured_4way_l1",
            "binned_joint_tvd",
            "raw_unique_states",
        )
    }
    risk_flags = {
        "unmeasured_3way_worse_over_5pct": bool(
            risk_changes["unmeasured_3way_l1"] is not None
            and risk_changes["unmeasured_3way_l1"] > 0.05
        ),
        "unmeasured_4way_worse_over_5pct": bool(
            risk_changes["unmeasured_4way_l1"] is not None
            and risk_changes["unmeasured_4way_l1"] > 0.05
        ),
        "binned_joint_tvd_worse_over_5pct": bool(
            risk_changes["binned_joint_tvd"] is not None
            and risk_changes["binned_joint_tvd"] > 0.05
        ),
        "raw_unique_states_drop_over_5pct": bool(
            risk_changes["raw_unique_states"] is not None
            and risk_changes["raw_unique_states"] < -0.05
        ),
    }
    return {
        "reference_rows": int(len(reference)),
        "reference_scope": (
            "single offline reference table; no independent train/test split"
        ),
        "measured_3way_cells_excluded": int(len(measured_triples)),
        "rows": rows,
        "by_variant": by_variant,
        "paired": paired,
        "candidate_vs_initial_relative_changes": risk_changes,
        "candidate_vs_initial_risk_flags": {
            **risk_flags,
            "any_full_quality_risk": bool(any(risk_flags.values())),
        },
    }


def _load_fixed_inputs():
    schema = load_schema(str(experiment.SCHEMA_PATH))
    queries = load_queries(str(experiment.QUERY_PATH))
    marginals = load_marginals(str(experiment.MARGINALS_PATH))
    return schema, queries, marginals


def _validate_source_payload(payload):
    valid_classifications = {
        "supports_persistent_heatbath_smoke",
        "persistent_heatbath_smoke_inconclusive",
        "persistent_heatbath_smoke_not_supported",
    }
    if (
        payload.get("experiment") != "persistent_workload_heatbath"
        or payload.get("formal_protocol") is not True
        or not experiment._formal_payload_matches(payload)
        or payload.get("aggregate", {}).get("classification")
        not in valid_classifications
        or payload.get("aggregate", {}).get(
            "all_diagnostic_gates_passed"
        ) is not True
        or payload.get("independent_audit", {}).get("passed") is not True
    ):
        raise ValueError("输入不是通过门禁的持久化热浴正式输出")


def _recompute(input_path):
    payload = experiment._load_json_strict(input_path)
    _validate_source_payload(payload)
    schema, queries, marginals = _load_fixed_inputs()
    states = _reconstruct_generated_states(
        payload, schema, queries, marginals
    )
    if any(len(variant) != len(payload["runs"]) for variant in states.values()):
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
        environment = experiment._environment_snapshot("cpu")
        if not environment["git_worktree_clean_including_untracked"]:
            raise RuntimeError(
                "正式离线评价要求当前提交对应的工作树完全干净"
            )
    payload, analysis = _recompute(args.input)
    source_hash = experiment._sha256_file(args.input)
    reference_hash = experiment._sha256_file(REAL_DATA_PATH)

    if args.audit_existing is not None:
        existing = experiment._load_json_strict(args.audit_existing)
        passed = bool(
            existing.get("source", {}).get("generation_result_sha256")
            == source_hash
            and existing.get("source", {}).get("reference_sha256")
            == reference_hash
            and json.dumps(
                existing.get("analysis"),
                sort_keys=True,
                separators=(",", ":"),
            ) == json.dumps(
                analysis, sort_keys=True, separators=(",", ":")
            )
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
        "experiment": "persistent_workload_heatbath_offline_evaluation",
        "boundary": (
            "post-generation diagnostic only; not used by generation, "
            "formal classification, parameter selection, early stopping, "
            "or output selection"
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
    experiment._write_json_atomic(args.output, report)
    print(f"output={args.output}")
    print(f"sha256={experiment._sha256_file(args.output)}")
    print(json.dumps(
        {
            "risk_flags": analysis[
                "candidate_vs_initial_risk_flags"
            ],
            "training_loss": {
                name: analysis["by_variant"][name][
                    "training_workload_loss"
                ]["mean"]
                for name in ("initial", "baseline", "candidate")
            },
            "unmeasured_3way_l1": {
                name: analysis["by_variant"][name][
                    "unmeasured_3way_l1"
                ]["mean"]
                for name in ("initial", "baseline", "candidate")
            },
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
