"""在共同冻结状态上检验曲率收益的阶段交互。

脚本复用 Issue #18 已验证的严格配对探针。运行时只使用公开 schema、记录数、
预定义查询、精确 target、1-way marginal 与合成状态；真实参考表没有输入路径。
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import stats

if __package__:
    from scripts import probe_generation_curvature_gibbs as frozen
else:
    import probe_generation_curvature_gibbs as frozen

from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


FORMAL_SEEDS = list(range(30, 40))
FORMAL_STATE_ROUNDS = [0, 500]
FORMAL_PROPOSALS = 200
FORMAL_TEMPERATURE = 2.0
FORMAL_SWEEPS = 8
FORMAL_MAX_FACTOR_ORDER = 3
FORMAL_LOGIT_CLIP = float(frozen.FORMAL_LOGIT_CLIP)
FORMAL_RHO = 0.01
FORMAL_ETA = 0.5
FORMAL_BASELINE_CURVATURE = 0.0
FORMAL_CANDIDATE_CURVATURE = 1.0
FORMAL_OUTPUT = Path(
    "outputs/curvature_state_interaction/"
    "formal_10seed_2state_200p_tau2_sweep8.json"
)
REQUIRED_INTERACTION_WINS = 8
PRIMARY_METRICS = (
    "linear_gain",
    "self_penalty",
    "cross_penalty",
    "quadratic_penalty",
    "net_gain",
    "positive_gain",
    "changed_rows",
    "copied_cells",
    "mask_hamming_cells",
    "mean_copy_blocks_per_participant",
)
BOOLEAN_STATE_GATES = (
    "primary_rng_aligned",
    "gibbs_rng_aligned",
    "gamma_zero_frame_exact",
    "gamma_zero_mask_exact",
    "gamma_zero_diagnostics_exact",
    "internal_initial_masks_aligned",
    "initial_mask_replay_exact",
    "logit_clip_not_hit",
    "all_conditionals_bidirectional",
)
MAX_ERROR_STATE_GATES = (
    "gamma_zero_conditional_probability_max_error",
    "initial_query_delta_max_error",
    "final_query_delta_max_error",
    "candidate_energy_identity_max_error",
    "row_delta_sum_max_error",
    "quadratic_identity_max_error",
    "gain_identity_max_error",
    "linear_query_consistency_max_error",
)


def _frozen_probe_constants_match():
    return (
        frozen.RHO == FORMAL_RHO
        and frozen.ETA == FORMAL_ETA
        and frozen.BASELINE_CURVATURE == FORMAL_BASELINE_CURVATURE
        and frozen.CANDIDATE_CURVATURE == FORMAL_CANDIDATE_CURVATURE
        and frozen.FORMAL_LOGIT_CLIP == FORMAL_LOGIT_CLIP == 30.0
    )


def _mean_t_interval(values, confidence=0.95):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("区间输入必须是非空有限一维数组")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 (0, 1)")
    mean = float(array.mean())
    if len(array) == 1:
        return [mean, mean]
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    if standard_error == 0.0:
        return [mean, mean]
    critical = float(stats.t.ppf(
        0.5 + confidence / 2.0,
        df=len(array) - 1,
    ))
    return [
        float(mean - critical * standard_error),
        float(mean + critical * standard_error),
    ]


def _seed_summary(values):
    array = np.asarray(values, dtype=float)
    summary = frozen._summarize_values(array)
    summary.update({
        "mean_t_interval_95": _mean_t_interval(array),
        "positive": int(np.sum(array > 0.0)),
        "zero": int(np.sum(array == 0.0)),
        "negative": int(np.sum(array < 0.0)),
        "independent_unit": "seed",
    })
    return summary


def _classify_stage_interaction(
    interaction_summary,
    *,
    expected_seed_count=len(FORMAL_SEEDS),
    required_direction_count=REQUIRED_INTERACTION_WINS,
):
    if interaction_summary.get("n") != expected_seed_count:
        raise ValueError("正式阶段交互分类要求固定数量的独立 seed")
    if not 1 <= required_direction_count <= expected_seed_count:
        raise ValueError("方向一致 seed 门槛超出有效范围")
    lower, upper = interaction_summary["mean_t_interval_95"]
    if (
        lower > 0.0
        and interaction_summary["positive"] >= required_direction_count
    ):
        return "curvature_advantage_strengthens_late"
    if (
        upper < 0.0
        and interaction_summary["negative"] >= required_direction_count
    ):
        return "curvature_advantage_weakens_late"
    return "state_interaction_inconclusive"


def _state_index(states, seeds):
    expected = {
        (int(seed), int(rounds))
        for seed in seeds
        for rounds in FORMAL_STATE_ROUNDS
    }
    indexed = {}
    for state in states:
        key = (int(state["seed"]), int(state["state_rounds"]))
        if key in indexed:
            raise ValueError(f"重复的 seed-state：{key}")
        indexed[key] = state
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"seed-state 集合不完整：missing={missing}, extra={extra}"
        )
    return indexed


def _metric_difference(state, metric):
    try:
        value = state["paired"][metric]["difference"]["mean"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"状态缺少 {metric} 配对均值") from exc
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{metric} 配对均值必须有限")
    return value


def _build_seed_stage_rows(states, seeds, n_queries):
    if not isinstance(n_queries, (int, np.integer)) or n_queries <= 0:
        raise ValueError("n_queries 必须为正整数")
    indexed = _state_index(states, seeds)
    rows = []
    for seed in seeds:
        stage_rows = {
            rounds: indexed[(int(seed), rounds)]
            for rounds in FORMAL_STATE_ROUNDS
        }
        stages = {}
        for rounds, state in stage_rows.items():
            loss = float(state["state_loss"])
            if not np.isfinite(loss) or loss < 0.0:
                raise ValueError("冻结状态 loss 必须是非负有限值")
            stages[str(rounds)] = {
                "state_sha256": state["state_sha256"],
                "state_loss": loss,
                "residual_rms": float(np.sqrt(2.0 * loss / n_queries)),
                "metric_differences": {
                    metric: _metric_difference(state, metric)
                    for metric in PRIMARY_METRICS
                },
                "probe_elapsed_sec": float(state["elapsed_sec"]),
                "state_generation_elapsed_sec": float(
                    state["state_generation"]["elapsed_sec"]
                ),
            }
        interactions = {
            metric: (
                stages["500"]["metric_differences"][metric]
                - stages["0"]["metric_differences"][metric]
            )
            for metric in PRIMARY_METRICS
        }
        rows.append({
            "seed": int(seed),
            "stages": stages,
            "interactions_late_minus_initial": interactions,
        })
    return rows


def _metric_stage_seed_summaries(seed_stage_rows):
    summaries = {}
    for metric in PRIMARY_METRICS:
        initial = [
            row["stages"]["0"]["metric_differences"][metric]
            for row in seed_stage_rows
        ]
        late = [
            row["stages"]["500"]["metric_differences"][metric]
            for row in seed_stage_rows
        ]
        interaction = [
            row["interactions_late_minus_initial"][metric]
            for row in seed_stage_rows
        ]
        summaries[metric] = {
            "initial_curvature_difference": _seed_summary(initial),
            "late_curvature_difference": _seed_summary(late),
            "late_minus_initial_interaction": _seed_summary(interaction),
            "initial_values_by_seed": initial,
            "late_values_by_seed": late,
            "interaction_values_by_seed": interaction,
        }
    return summaries


def _stage_aggregates(states, seeds, n_queries):
    indexed = _state_index(states, seeds)
    result = {}
    for rounds in FORMAL_STATE_ROUNDS:
        stage_states = [indexed[(int(seed), rounds)] for seed in seeds]
        baseline_rows = [
            row for state in stage_states for row in state["baseline_rows"]
        ]
        candidate_rows = [
            row for state in stage_states for row in state["candidate_rows"]
        ]
        metrics = tuple(stage_states[0]["paired"])
        if any(tuple(state["paired"]) != metrics for state in stage_states):
            raise ValueError("各 seed-state 的配对指标集合或顺序不一致")
        paired = {
            metric: frozen._paired(candidate_rows, baseline_rows, metric)
            for metric in metrics
        }
        paired["query_delta_l2_squared"] = frozen._paired(
            [
                {"value": 2.0 * row["quadratic_penalty"]}
                for row in candidate_rows
            ],
            [
                {"value": 2.0 * row["quadratic_penalty"]}
                for row in baseline_rows
            ],
            "value",
        )
        state_losses = [float(state["state_loss"]) for state in stage_states]
        residual_rms = [
            float(np.sqrt(2.0 * value / n_queries))
            for value in state_losses
        ]
        result[str(rounds)] = {
            "n_seed_states": len(stage_states),
            "n_paired_proposals": len(baseline_rows),
            "state_loss_by_seed": _seed_summary(state_losses),
            "state_residual_rms_by_seed": _seed_summary(residual_rms),
            "probe_elapsed_sec_by_seed": _seed_summary([
                state["elapsed_sec"] for state in stage_states
            ]),
            "state_generation_elapsed_sec_by_seed": _seed_summary([
                state["state_generation"]["elapsed_sec"]
                for state in stage_states
            ]),
            "paired_proposal_metrics": paired,
            "baseline_conditional": frozen._conditional_summary(
                baseline_rows
            ),
            "candidate_conditional": frozen._conditional_summary(
                candidate_rows
            ),
        }
    return result


def _all_numeric_values_finite(value):
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return True
    return value is None or isinstance(value, str)


def _aggregate_gates(states, seeds, proposals):
    indexed = _state_index(states, seeds)
    ordered = [
        indexed[(int(seed), rounds)]
        for seed in seeds
        for rounds in FORMAL_STATE_ROUNDS
    ]
    initialization_aligned = all(
        indexed[(int(seed), 0)]["state_sha256"]
        == indexed[(int(seed), 500)]["state_generation"].get(
            "initial_table_sha256"
        )
        for seed in seeds
    )
    generation_protocol_complete = all(
        indexed[(int(seed), 0)]["state_generation"].get("method")
        == "marginal_initialization"
        and indexed[(int(seed), 0)]["state_generation"].get("rounds") == 0
        and indexed[(int(seed), 500)]["state_generation"].get("method")
        == "standard_closed_loop_best"
        and indexed[(int(seed), 500)]["state_generation"].get("rounds")
        == 500
        and indexed[(int(seed), 500)]["state_generation"].get(
            "rounds_run"
        ) == 500
        and not indexed[(int(seed), 500)]["state_generation"].get(
            "stopped_early"
        )
        for seed in seeds
    )
    late_state_loss_generation_max_error = max(
        abs(
            float(indexed[(int(seed), 500)]["state_loss"])
            - float(
                indexed[(int(seed), 500)]["state_generation"]["best_loss"]
            )
        )
        for seed in seeds
    )
    direction_scale_shared = all(
        indexed[(int(seed), 0)]["direction_reference_scale"]
        == indexed[(int(seed), 500)]["direction_reference_scale"]
        == indexed[(int(seed), 500)]["state_generation"].get(
            "direction_reference_scale"
        )
        for seed in seeds
    )
    direction_scale_positive_finite = all(
        np.isfinite(indexed[(int(seed), rounds)][
            "direction_reference_scale"
        ])
        and indexed[(int(seed), rounds)]["direction_reference_scale"] > 0.0
        for seed in seeds
        for rounds in FORMAL_STATE_ROUNDS
    )
    raw_row_counts_complete = all(
        len(state["baseline_rows"]) == proposals
        and len(state["candidate_rows"]) == proposals
        and len(state["pair_rows"]) == proposals
        for state in ordered
    )
    proposal_indices_aligned = all(
        [row["proposal_index"] for row in state["baseline_rows"]]
        == list(range(proposals))
        == [row["proposal_index"] for row in state["candidate_rows"]]
        == [row["proposal_index"] for row in state["pair_rows"]]
        for state in ordered
    )
    gates = {
        "state_set_complete": len(ordered) == 2 * len(seeds),
        "proposal_counts_complete": all(
            state["n_proposals"] == proposals for state in ordered
        ),
        "raw_row_counts_complete": raw_row_counts_complete,
        "proposal_indices_aligned": proposal_indices_aligned,
        "state_initialization_aligned": initialization_aligned,
        "state_generation_protocol_complete": generation_protocol_complete,
        "late_state_loss_generation_max_error": (
            late_state_loss_generation_max_error
        ),
        "direction_reference_scale_shared": direction_scale_shared,
        "direction_reference_scale_positive_finite": (
            direction_scale_positive_finite
        ),
        "all_numeric_values_finite": _all_numeric_values_finite(ordered),
    }
    for name in BOOLEAN_STATE_GATES:
        gates[name] = all(state["gates"][name] for state in ordered)
    for name in MAX_ERROR_STATE_GATES:
        gates[name] = max(state["gates"][name] for state in ordered)
    gates["conditional_logit_abs_max"] = max(
        state["gates"]["conditional_logit_abs_max"] for state in ordered
    )
    return gates


def _diagnostic_gate_passed(gates):
    return (
        gates["state_set_complete"]
        and gates["proposal_counts_complete"]
        and gates["raw_row_counts_complete"]
        and gates["proposal_indices_aligned"]
        and gates["state_initialization_aligned"]
        and gates["state_generation_protocol_complete"]
        and gates["late_state_loss_generation_max_error"] <= 1e-10
        and gates["direction_reference_scale_shared"]
        and gates["direction_reference_scale_positive_finite"]
        and gates["all_numeric_values_finite"]
        and all(gates[name] for name in BOOLEAN_STATE_GATES)
        and gates[
            "gamma_zero_conditional_probability_max_error"
        ] == 0.0
        and all(
            gates[name] <= 1e-10
            for name in MAX_ERROR_STATE_GATES
            if name != "gamma_zero_conditional_probability_max_error"
        )
    )


def _write_json_atomic(path, payload, *, overwrite=False):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，不覆盖：{output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--proposals", type=int, default=FORMAL_PROPOSALS)
    parser.add_argument(
        "--temperature", type=float, default=FORMAL_TEMPERATURE
    )
    parser.add_argument("--sweeps", type=int, default=FORMAL_SWEEPS)
    parser.add_argument(
        "--max-factor-order", type=int, default=FORMAL_MAX_FACTOR_ORDER
    )
    parser.add_argument(
        "--logit-clip", type=float, default=FORMAL_LOGIT_CLIP
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument("--output", default=str(FORMAL_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error("--seeds 必须非空、非负且不重复")
    if args.proposals <= 0:
        parser.error("--proposals 必须为正整数")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")
    if not 1 <= args.max_factor_order <= 8:
        parser.error("--max-factor-order 必须在 1..8 内")
    if not np.isfinite(args.logit_clip) or args.logit_clip <= 0.0:
        parser.error("--logit-clip 必须是正有限数值")

    formal_protocol_matches = (
        args.seeds == FORMAL_SEEDS
        and args.proposals == FORMAL_PROPOSALS
        and args.temperature == FORMAL_TEMPERATURE
        and args.sweeps == FORMAL_SWEEPS
        and args.max_factor_order == FORMAL_MAX_FACTOR_ORDER
        and args.logit_clip == FORMAL_LOGIT_CLIP
        and args.device == "cuda"
    )
    output = Path(args.output)
    if output.exists() and (formal_protocol_matches or not args.overwrite):
        raise FileExistsError(f"输出已存在，不覆盖：{output}")
    if not _frozen_probe_constants_match():
        raise RuntimeError("既有冻结探针常量与预注册协议不一致")

    environment = frozen._environment_snapshot(args.device)
    if formal_protocol_matches and not environment["git_worktree_clean"]:
        raise RuntimeError("正式协议要求 tracked 工作树干净")
    if args.device == "cuda" and not environment["cuda_available"]:
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备")

    schema = load_schema(str(frozen.SCHEMA_PATH))
    queries = load_queries(str(frozen.QUERY_PATH))
    target = np.asarray(
        [query["result"] for query in queries], dtype=float
    )
    marginals = load_marginals(str(frozen.MARGINALS_PATH))
    if (
        len(queries) != 50
        or target.shape != (50,)
        or not np.all(np.isfinite(target))
        or len(schema.attribute_names()) != 10
        or marginals.get("n_records") != frozen.N_RECORDS
        or set(marginals.get("attributes", {}))
        != set(schema.attribute_names())
    ):
        raise ValueError("test_300x10 的公开输入与协议不一致")

    experiment_start = time.perf_counter()
    states = []
    for seed in args.seeds:
        prepared_states = []
        for state_index, state_rounds in enumerate(FORMAL_STATE_ROUNDS):
            generation_start = time.perf_counter()
            state, generation = frozen._make_state(
                target,
                queries,
                schema,
                marginals,
                seed=seed,
                rounds=state_rounds,
                temperature=args.temperature,
                device=args.device,
            )
            generation = dict(generation)
            generation["elapsed_sec"] = (
                time.perf_counter() - generation_start
            )
            prepared_states.append(
                (state_index, state_rounds, state, generation)
            )

        reference_scale, _ = frozen._resolve_seed_state_controls(
            prepared_states,
            require_complete=formal_protocol_matches,
        )
        for state_index, state_rounds, state, generation in prepared_states:
            result = frozen._probe_state(
                state,
                target,
                queries,
                schema,
                seed=seed,
                state_index=state_index,
                state_rounds=state_rounds,
                proposals=args.proposals,
                temperature=args.temperature,
                sweeps=args.sweeps,
                max_factor_order=args.max_factor_order,
                device=args.device,
                fixed_reference_scale=reference_scale,
                logit_clip=args.logit_clip,
            )
            result["state_generation"] = generation
            states.append(result)
            print(
                f"seed={seed:02d} state={state_rounds:03d} "
                "曲率净收益差="
                f"{result['paired']['net_gain']['difference']['mean']:+.4f}",
                flush=True,
            )

    seed_stage_rows = _build_seed_stage_rows(
        states, args.seeds, len(queries)
    )
    metric_stage_seed_summaries = _metric_stage_seed_summaries(
        seed_stage_rows
    )
    primary_interaction = metric_stage_seed_summaries["net_gain"][
        "late_minus_initial_interaction"
    ]
    gates = _aggregate_gates(states, args.seeds, args.proposals)
    diagnostic_gate_passed = _diagnostic_gate_passed(gates)
    classification = (
        _classify_stage_interaction(primary_interaction)
        if formal_protocol_matches and diagnostic_gate_passed
        else (
            "diagnostic_gate_failed"
            if formal_protocol_matches else "non_formal_run_no_decision"
        )
    )
    summary = {
        "experiment": "curvature_common_state_stage_interaction",
        "issue": 30,
        "status": "complete",
        "formal_protocol_matches": formal_protocol_matches,
        "diagnostic_gate_passed": diagnostic_gate_passed,
        "classification": classification,
        "scope": "frozen_common_state_diagnostic_no_schedule_or_acceptance",
        "dataset": "test_300x10",
        "n_records": frozen.N_RECORDS,
        "n_queries": len(queries),
        "seeds": args.seeds,
        "state_rounds": FORMAL_STATE_ROUNDS,
        "proposals_per_state": args.proposals,
        "n_seed_states": len(states),
        "n_paired_proposals": sum(
            state["n_proposals"] for state in states
        ),
        "temperature": args.temperature,
        "sweeps": args.sweeps,
        "baseline_curvature_weight": FORMAL_BASELINE_CURVATURE,
        "candidate_curvature_weight": FORMAL_CANDIDATE_CURVATURE,
        "max_factor_order": args.max_factor_order,
        "gibbs_logit_clip": float(args.logit_clip),
        "rho": FORMAL_RHO,
        "eta": FORMAL_ETA,
        "proposal_mu": 0.0,
        "device": args.device,
        "independent_unit": "seed",
        "preregistered_classification_rule": {
            "confidence": 0.95,
            "interval": "two_sided_student_t_over_seed_interactions",
            "required_direction_count": REQUIRED_INTERACTION_WINS,
            "expected_seed_count": len(FORMAL_SEEDS),
        },
        "real_data_access": "none",
        "environment": environment,
        "public_input_sha256": {
            str(path): frozen._sha256_file(path)
            for path in (
                frozen.SCHEMA_PATH,
                frozen.QUERY_PATH,
                frozen.MARGINALS_PATH,
            )
        },
        "elapsed_sec": time.perf_counter() - experiment_start,
        "gates": gates,
        "primary_interaction": primary_interaction,
        "metric_stage_seed_summaries": metric_stage_seed_summaries,
        "stage_aggregates": _stage_aggregates(
            states, args.seeds, len(queries)
        ),
        "seed_stage_rows": seed_stage_rows,
        "states": states,
    }
    _write_json_atomic(
        output,
        summary,
        overwrite=args.overwrite and not formal_protocol_matches,
    )

    net_gain = metric_stage_seed_summaries["net_gain"]
    print("\n===== 曲率收益共同状态阶段交互 =====")
    print(
        "初始态曲率净收益差："
        f"{net_gain['initial_curvature_difference']['mean']:+.6g}，"
        "95%区间="
        f"{net_gain['initial_curvature_difference']['mean_t_interval_95']}"
    )
    print(
        "500 轮态曲率净收益差："
        f"{net_gain['late_curvature_difference']['mean']:+.6g}，"
        "95%区间="
        f"{net_gain['late_curvature_difference']['mean_t_interval_95']}"
    )
    print(
        "晚期减初始交互："
        f"{primary_interaction['mean']:+.6g}，"
        f"95%区间={primary_interaction['mean_t_interval_95']}，"
        f"{primary_interaction['positive']}/"
        f"{primary_interaction['zero']}/"
        f"{primary_interaction['negative']}"
    )
    print(f"诊断门禁：{diagnostic_gate_passed}")
    print(f"预注册分类：{classification}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
