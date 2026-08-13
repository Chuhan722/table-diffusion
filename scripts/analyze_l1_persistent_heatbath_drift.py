"""对阶段 I 正式轨迹做公开输入范围内的事后期望漂移诊断。

本脚本不读取真实参考表，也不改变预注册分类。它从公开输入重建每个初始表，枚举
所有坐标的完整候选能量，精确计算随机扫描一步漂移以及漂移变号的临界 tau；同时
复算正式轨迹中已经记录的条件期望漂移。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from scripts import probe_l1_persistent_heatbath as experiment
else:
    import probe_l1_persistent_heatbath as experiment
from table_diffevo.generator import init_synthetic_table
from table_diffevo.persistent_heatbath import (
    ENERGY_MODE_NORMALIZED_L1,
    ENERGY_MODE_SQUARED,
    build_persistent_heatbath_conditional,
    initial_gain_rms_scale,
    initialize_persistent_heatbath_state,
)


FORMAL_INPUT = experiment.FORMAL_OUTPUT
FORMAL_OUTPUT = Path(
    "outputs/l1_persistent_workload_heatbath/"
    "posthoc_initial_expected_drift.json"
)
ROOT_ITERATIONS = 80
DRIFT_TOLERANCE = 1e-15


def _candidate_energy_sets(state, schema, queries, target, energy_mode):
    energies = []
    candidate_evaluations = 0
    query_evaluations = 0
    for coordinate in range(len(state.table) * schema.n_blocks()):
        row, attribute = divmod(coordinate, schema.n_blocks())
        conditional = build_persistent_heatbath_conditional(
            state,
            schema,
            queries,
            target,
            row_index=row,
            attribute_index=attribute,
            inverse_temperature=0.0,
            energy_mode=energy_mode,
        )
        energies.append(conditional.candidate_energies.copy())
        candidate_evaluations += conditional.candidate_state_evaluations
        query_evaluations += conditional.query_indicator_evaluations
    return {
        "source_energy": experiment._energy_from_state(
            state, target, energy_mode
        ),
        "candidate_energies": energies,
        "coordinates": len(energies),
        "candidate_state_evaluations": int(candidate_evaluations),
        "query_indicator_evaluations": int(query_evaluations),
    }


def _random_scan_drift(energy_sets, inverse_energy_scale):
    alpha = float(inverse_energy_scale)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("inverse_energy_scale 必须是非负有限数值")
    source = float(energy_sets["source_energy"])
    drifts = []
    minimum_probability = 1.0
    derivative_terms = []
    for raw in energy_sets["candidate_energies"]:
        energies = np.asarray(raw, dtype=float)
        centered = -alpha * (energies - float(np.min(energies)))
        weights = np.exp(centered)
        probabilities = weights / float(np.sum(weights))
        expected = float(np.dot(probabilities, energies))
        drifts.append(expected - source)
        minimum_probability = min(
            minimum_probability, float(np.min(probabilities))
        )
        derivative_terms.append(float(np.dot(
            probabilities, (energies - expected) ** 2
        )))
    values = np.asarray(drifts, dtype=float)
    return {
        "mean_drift": float(values.mean()),
        "positive_coordinate_fraction": float(np.mean(values > 0.0)),
        "zero_coordinate_fraction": float(np.mean(values == 0.0)),
        "negative_coordinate_fraction": float(np.mean(values < 0.0)),
        "minimum_coordinate_drift": float(values.min()),
        "maximum_coordinate_drift": float(values.max()),
        "minimum_conditional_probability": float(minimum_probability),
        "derivative_wrt_inverse_scale": -float(np.mean(derivative_terms)),
    }


def _critical_tau(energy_sets, scale):
    s0 = float(scale)
    if not np.isfinite(s0) or s0 <= 0.0:
        raise ValueError("scale 必须是有限正数")
    source = float(energy_sets["source_energy"])
    zero = _random_scan_drift(energy_sets, 0.0)["mean_drift"]
    limit = float(np.mean([
        np.min(values) for values in energy_sets["candidate_energies"]
    ])) - source
    if zero <= 0.0:
        return {
            "exists_finite": True,
            "critical_tau": 0.0,
            "drift_at_critical": float(zero),
            "infinite_inverse_scale_limit_drift": float(limit),
        }
    if limit >= -DRIFT_TOLERANCE:
        return {
            "exists_finite": False,
            "critical_tau": None,
            "drift_at_critical": None,
            "infinite_inverse_scale_limit_drift": float(limit),
        }

    low_tau = 0.0
    high_tau = 1.0
    high_drift = _random_scan_drift(
        energy_sets, high_tau / s0
    )["mean_drift"]
    while high_drift > 0.0:
        high_tau *= 2.0
        if not np.isfinite(high_tau):
            raise ValueError("临界 tau 搜索超出 float64 范围")
        high_drift = _random_scan_drift(
            energy_sets, high_tau / s0
        )["mean_drift"]
    for _ in range(ROOT_ITERATIONS):
        middle = 0.5 * (low_tau + high_tau)
        drift = _random_scan_drift(
            energy_sets, middle / s0
        )["mean_drift"]
        if drift > 0.0:
            low_tau = middle
        else:
            high_tau = middle
    critical = float(high_tau)
    at_critical = _random_scan_drift(
        energy_sets, critical / s0
    )["mean_drift"]
    return {
        "exists_finite": True,
        "critical_tau": critical,
        "drift_at_critical": float(at_critical),
        "infinite_inverse_scale_limit_drift": float(limit),
    }


def _trajectory_drift(run, variant):
    trajectory = run[variant]
    before = np.asarray(trajectory["energy_history"][:-1], dtype=float)
    expected = np.asarray(
        trajectory["expected_energy_history"], dtype=float
    )
    if before.shape != expected.shape or len(before) != run["n_steps"]:
        raise ValueError(
            f"seed {run['seed']} {variant} 条件期望历史长度不一致"
        )
    drift = expected - before
    return {
        "mean_conditional_expected_drift": float(drift.mean()),
        "positive_step_fraction": float(np.mean(drift > 0.0)),
        "minimum_step_drift": float(drift.min()),
        "maximum_step_drift": float(drift.max()),
        "realized_final_minus_initial": float(
            trajectory["energy_history"][-1]
            - trajectory["energy_history"][0]
        ),
    }


def _summary(rows, field):
    values = [row[field] for row in rows]
    return {
        **experiment.common._summarize(values),
        "median": float(np.median(values)),
        "positive": int(np.sum(np.asarray(values) > 0.0)),
        "zero": int(np.sum(np.asarray(values) == 0.0)),
        "negative": int(np.sum(np.asarray(values) < 0.0)),
    }


def build_analysis(payload, schema, queries, target, marginals):
    modes = {
        "baseline": ENERGY_MODE_SQUARED,
        "candidate": ENERGY_MODE_NORMALIZED_L1,
    }
    rows = {variant: [] for variant in modes}
    for run in payload["runs"]:
        seed = int(run["seed"])
        table = init_synthetic_table(
            int(payload["protocol"]["n_records"]),
            schema,
            np.random.default_rng(seed),
            marginals=marginals,
        )
        state = initialize_persistent_heatbath_state(
            table, schema, queries, target
        )
        if experiment.common._frame_sha256(state.table) != run[
            "initial_table_sha256"
        ]:
            raise ValueError(f"seed {seed} 初始表哈希不一致")
        for variant, mode in modes.items():
            scale = initial_gain_rms_scale(
                state,
                schema,
                queries,
                target,
                energy_mode=mode,
            )
            if scale != run["initial_gain_rms_scales"][variant]:
                raise ValueError(f"seed {seed} {variant} 尺度不一致")
            energy_sets = _candidate_energy_sets(
                state, schema, queries, target, mode
            )
            formal = _random_scan_drift(
                energy_sets,
                float(run["tau"]) / float(scale["scale"]),
            )
            critical = _critical_tau(energy_sets, scale["scale"])
            rows[variant].append({
                "seed": seed,
                "energy_mode": mode,
                "scale": float(scale["scale"]),
                "formal_tau": float(run["tau"]),
                "initial_random_scan": formal,
                "critical": critical,
                "trajectory": _trajectory_drift(run, variant),
                "coordinates": energy_sets["coordinates"],
                "candidate_state_evaluations": energy_sets[
                    "candidate_state_evaluations"
                ],
                "query_indicator_evaluations": energy_sets[
                    "query_indicator_evaluations"
                ],
            })

    by_variant = {}
    for variant, variant_rows in rows.items():
        critical_values = [
            row["critical"]["critical_tau"]
            for row in variant_rows
            if row["critical"]["exists_finite"]
        ]
        by_variant[variant] = {
            "initial_mean_drift": _summary([
                {
                    "value": row["initial_random_scan"]["mean_drift"]
                }
                for row in variant_rows
            ], "value"),
            "initial_positive_coordinate_fraction": _summary([
                {
                    "value": row["initial_random_scan"][
                        "positive_coordinate_fraction"
                    ]
                }
                for row in variant_rows
            ], "value"),
            "trajectory_mean_conditional_expected_drift": _summary([
                {
                    "value": row["trajectory"][
                        "mean_conditional_expected_drift"
                    ]
                }
                for row in variant_rows
            ], "value"),
            "trajectory_realized_final_minus_initial": _summary([
                {
                    "value": row["trajectory"][
                        "realized_final_minus_initial"
                    ]
                }
                for row in variant_rows
            ], "value"),
            "finite_critical_tau": {
                "count": len(critical_values),
                "summary": (
                    {
                        **experiment.common._summarize(critical_values),
                        "median": float(np.median(critical_values)),
                    }
                    if critical_values else None
                ),
            },
        }
    return {
        "scope": (
            "posthoc mechanism diagnostic from public inputs and recorded "
            "trajectories; does not change the preregistered classification"
        ),
        "rows": rows,
        "by_variant": by_variant,
    }


def _validate_source(payload):
    if (
        not experiment._formal_payload_matches(payload)
        or payload.get("aggregate", {}).get(
            "all_diagnostic_gates_passed"
        ) is not True
        or payload.get("independent_audit", {}).get("passed") is not True
    ):
        raise ValueError("输入不是通过语义门禁的阶段 I 正式输出")


def _recompute(input_path):
    payload = experiment.common._load_json_strict(input_path)
    _validate_source(payload)
    schema, queries, target, marginals = experiment._load_public_inputs()
    input_paths = {
        "schema": experiment.SCHEMA_PATH,
        "queries": experiment.QUERY_PATH,
        "marginals": experiment.MARGINALS_PATH,
    }
    if any(
        experiment.common._sha256_file(path)
        != payload["public_input_sha256"].get(name)
        for name, path in input_paths.items()
    ):
        raise ValueError("公开输入哈希与正式输出不一致")
    return payload, build_analysis(
        payload, schema, queries, target, marginals
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FORMAL_INPUT)
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument(
        "--audit-existing", type=Path, help="只读复算已有漂移诊断"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    environment = None
    if args.audit_existing is None:
        environment = experiment.common._environment_snapshot("cpu")
        if not environment["git_worktree_clean_including_untracked"]:
            raise RuntimeError("正式事后诊断要求当前工作树完全干净")
    payload, analysis = _recompute(args.input)
    source_hash = experiment.common._sha256_file(args.input)
    if args.audit_existing is not None:
        existing = experiment.common._load_json_strict(args.audit_existing)
        passed = bool(
            existing.get("source", {}).get("generation_result_sha256")
            == source_hash
            and existing.get("analysis") == analysis
        )
        result = {
            "passed": passed,
            "generation_result_sha256": source_hash,
            "checked_seeds": len(payload["runs"]),
            "checked_initial_coordinate_sets": 2 * len(payload["runs"]),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not passed:
            raise RuntimeError("已有漂移诊断复算失败")
        return

    report = {
        "experiment": "l1_persistent_heatbath_posthoc_drift",
        "environment": environment,
        "source": {
            "generation_result_path": str(args.input),
            "generation_result_sha256": source_hash,
            "generation_commit": payload["environment"]["git_commit"],
            "preregistered_classification_unchanged": True,
        },
        "analysis": analysis,
        "elapsed_sec": float(time.perf_counter() - started),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    experiment.common._write_json_atomic(args.output, report)
    print(f"output={args.output}")
    print(f"sha256={experiment.common._sha256_file(args.output)}")
    print(json.dumps(
        analysis["by_variant"], ensure_ascii=False, indent=2
    ))


if __name__ == "__main__":
    main()
