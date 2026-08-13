"""运行平方能量与 normalized L1 持久热浴的阶段 I 配对实验。

生成阶段只读取公开 schema、公开记录数、预定义查询、已提供 target 与初始化
marginal。脚本不读取真实参考表，不执行接受、回滚、早停或 best checkpoint 选择。
真实表上的高阶质量由独立离线脚本在全部生成结束后评价。
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

if __package__:
    from scripts import probe_persistent_heatbath as common
else:
    import probe_persistent_heatbath as common
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.persistent_heatbath import (
    ENERGY_MODE_NORMALIZED_L1,
    ENERGY_MODE_SQUARED,
    build_persistent_heatbath_conditional,
    initial_gain_rms_scale,
    initialize_persistent_heatbath_state,
    legal_attribute_values,
    persistent_heatbath_step,
    verify_persistent_heatbath_state,
)
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import AttributeBlock, Schema, load_schema


SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
QUERY_PATH = Path("configs/test_300x10/measured_50query.json")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
N_RECORDS = 300
FORMAL_SEEDS = list(range(60, 80))
FORMAL_STEPS = 3000
FORMAL_TAIL = 750
FORMAL_TAU = 1.0
FORMAL_VERIFY_EVERY = 100
FORMAL_DEVICE = "cpu"
FORMAL_OUTPUT = Path(
    "outputs/l1_persistent_workload_heatbath/"
    "formal_stage1_20seed_3000step_tau1.json"
)
ORACLE_INVERSE_SCALES = (0.0, 0.7, 1.3)
IDENTITY_TOLERANCE = 1e-12


def _address_seed(seed):
    sequence = np.random.SeedSequence([
        int(seed),
        0x4C31504552534953,
        0x54454E5448454154,
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _normalized_l1(target, answers, n_records):
    values = np.abs(
        np.asarray(target, dtype=float)
        - np.asarray(answers, dtype=float)
    )
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("normalized L1 要求非空一维 workload")
    result = float(np.mean(values) / int(n_records))
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("normalized L1 必须是有限非负数值")
    return result


def _energy_from_state(state, target, energy_mode):
    if energy_mode == ENERGY_MODE_SQUARED:
        return float(state.loss) / len(state.table)
    if energy_mode == ENERGY_MODE_NORMALIZED_L1:
        return _normalized_l1(target, state.query_answers, len(state.table))
    raise ValueError(f"未知能量模式：{energy_mode!r}")


def _oracle_problem():
    schema = Schema([
        AttributeBlock(
            name="a", type="categorical", description="", values=[0, 1]
        ),
        AttributeBlock(
            name="b", type="categorical", description="", values=[0, 1]
        ),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
    ]
    return schema, queries, np.asarray([1, 1, 1], dtype=float)


def _oracle_table(bits):
    return pd.DataFrame({
        "a": [bits[0], bits[2]],
        "b": [bits[1], bits[3]],
    })


def _is_irreducible(matrix):
    reached = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for destination in np.flatnonzero(matrix[source] > 0.0):
            destination = int(destination)
            if destination not in reached:
                reached.add(destination)
                frontier.append(destination)
    return len(reached) == len(matrix)


def run_exact_oracle():
    """穷举 16 个有序表状态，验证 L1 核的完整 Gibbs 语义。"""
    states = tuple(itertools.product((0, 1), repeat=4))
    indices = {state: index for index, state in enumerate(states)}
    schema, queries, target = _oracle_problem()
    results = []
    expected_by_scale = {}
    maximum_increment_error = 0.0
    maximum_energy_error = 0.0
    maximum_conditional_error = 0.0

    for inverse_scale in ORACLE_INVERSE_SCALES:
        matrix = np.zeros((len(states), len(states)), dtype=float)
        energies = np.zeros(len(states), dtype=float)
        conditionals = {}
        for state_index, bits in enumerate(states):
            state = initialize_persistent_heatbath_state(
                _oracle_table(bits), schema, queries, target
            )
            energies[state_index] = _normalized_l1(
                target, state.query_answers, len(state.table)
            )
            for coordinate in range(4):
                row, attribute = divmod(coordinate, 2)
                conditional = build_persistent_heatbath_conditional(
                    state,
                    schema,
                    queries,
                    target,
                    row_index=row,
                    attribute_index=attribute,
                    inverse_temperature=inverse_scale,
                    energy_mode=ENERGY_MODE_NORMALIZED_L1,
                )
                conditionals[(state_index, coordinate)] = conditional
                for choice, value in enumerate(conditional.values):
                    next_bits = list(bits)
                    next_bits[coordinate] = value
                    matrix[state_index, indices[tuple(next_bits)]] += (
                        conditional.probabilities[choice] / 4.0
                    )
                    candidate = state.table.copy(deep=True)
                    candidate.iat[row, attribute] = value
                    full_answers = evaluate_table(candidate, queries)
                    full_energy = _normalized_l1(
                        target, full_answers, len(candidate)
                    )
                    maximum_increment_error = max(
                        maximum_increment_error,
                        float(np.max(np.abs(
                            state.query_answers
                            + conditional.query_deltas[choice]
                            - full_answers
                        ))),
                    )
                    maximum_energy_error = max(
                        maximum_energy_error,
                        abs(
                            conditional.candidate_energies[choice]
                            - full_energy
                        ),
                        abs(
                            conditional.energy_gains[choice]
                            - (energies[state_index] - full_energy)
                        ),
                    )

        weights = np.exp(-inverse_scale * energies)
        stationary = weights / weights.sum()
        conditional_expected = {}
        for (state_index, coordinate), conditional in conditionals.items():
            alternatives = []
            for value in conditional.values:
                alternative = list(states[state_index])
                alternative[coordinate] = value
                alternatives.append(indices[tuple(alternative)])
            oracle = stationary[alternatives]
            oracle = oracle / oracle.sum()
            maximum_conditional_error = max(
                maximum_conditional_error,
                float(np.max(np.abs(
                    conditional.probabilities - oracle
                ))),
            )
            conditional_expected[(state_index, coordinate)] = (
                conditional.expected_energy
            )
        expected_by_scale[inverse_scale] = conditional_expected
        flow = stationary[:, None] * matrix
        results.append({
            "inverse_energy_scale": float(inverse_scale),
            "row_sum_max_error": float(np.max(np.abs(
                matrix.sum(axis=1) - 1.0
            ))),
            "minimum_positive_transition": float(
                matrix[matrix > 0.0].min()
            ),
            "all_state_self_loops_positive": bool(
                np.all(np.diag(matrix) > 0.0)
            ),
            "irreducible": bool(_is_irreducible(matrix)),
            "detailed_balance_max_error": float(np.max(np.abs(
                flow - flow.T
            ))),
            "stationarity_max_error": float(np.max(np.abs(
                stationary @ matrix - stationary
            ))),
        })

    expected_monotonic = True
    derivative_max_error = 0.0
    for key in expected_by_scale[0.0]:
        expected = [
            expected_by_scale[value][key]
            for value in ORACLE_INVERSE_SCALES
        ]
        expected_monotonic = expected_monotonic and all(
            expected[index] + IDENTITY_TOLERANCE >= expected[index + 1]
            for index in range(len(expected) - 1)
        )
        state_index, coordinate = key
        state = initialize_persistent_heatbath_state(
            _oracle_table(states[state_index]), schema, queries, target
        )
        row, attribute = divmod(coordinate, 2)

        def conditional(value):
            return build_persistent_heatbath_conditional(
                state,
                schema,
                queries,
                target,
                row_index=row,
                attribute_index=attribute,
                inverse_temperature=value,
                energy_mode=ENERGY_MODE_NORMALIZED_L1,
            )

        center = conditional(0.7)
        epsilon = 1e-5
        numerical = (
            conditional(0.7 + epsilon).expected_energy
            - conditional(0.7 - epsilon).expected_energy
        ) / (2.0 * epsilon)
        variance = float(np.dot(
            center.probabilities,
            (center.candidate_energies - center.expected_energy) ** 2,
        ))
        derivative_max_error = max(
            derivative_max_error, abs(numerical + variance)
        )

    passed = bool(
        maximum_increment_error <= IDENTITY_TOLERANCE
        and maximum_energy_error <= IDENTITY_TOLERANCE
        and maximum_conditional_error <= IDENTITY_TOLERANCE
        and expected_monotonic
        and derivative_max_error <= 1e-9
        and all(
            result["row_sum_max_error"] <= IDENTITY_TOLERANCE
            and result["minimum_positive_transition"] > 0.0
            and result["all_state_self_loops_positive"]
            and result["irreducible"]
            and result["detailed_balance_max_error"]
            <= IDENTITY_TOLERANCE
            and result["stationarity_max_error"] <= IDENTITY_TOLERANCE
            for result in results
        )
    )
    return {
        "states": len(states),
        "coordinates_per_state": 4,
        "inverse_energy_scales": list(ORACLE_INVERSE_SCALES),
        "by_inverse_energy_scale": results,
        "query_increment_max_error": float(maximum_increment_error),
        "energy_identity_max_error": float(maximum_energy_error),
        "conditional_probability_max_error": float(
            maximum_conditional_error
        ),
        "expected_energy_monotonic": bool(expected_monotonic),
        "derivative_identity_max_error": float(derivative_max_error),
        "passed": passed,
    }


def _state_audit(state, schema, queries, target, step):
    return {
        **common._state_audit(state, schema, queries, target, step),
        "normalized_l1": _normalized_l1(
            target, state.query_answers, len(state.table)
        ),
    }


def _empty_trajectory(state, target, energy_mode):
    return {
        "energy_mode": energy_mode,
        "initial_query_answers": [
            int(value) for value in state.query_answers
        ],
        "loss_history": [float(state.loss)],
        "normalized_l1_history": [
            _normalized_l1(target, state.query_answers, len(state.table))
        ],
        "energy_history": [
            _energy_from_state(state, target, energy_mode)
        ],
        "loss_gain_history": [],
        "energy_gain_history": [],
        "changed_history": [],
        "choice_index_history": [],
        "query_delta_sparse_history": [],
        "expected_energy_history": [],
        "reference_expected_energy_history": [],
        "expected_energy_gain_over_reference_history": [],
        "conditional_normalized_entropy_history": [],
        "uphill_probability_mass_history": [],
        "probability_min_history": [],
        "probability_max_history": [],
        "candidate_state_evaluations_history": [],
        "query_indicator_evaluations_history": [],
        "full_state_audits": [],
        "kernel_elapsed_sec": 0.0,
    }


def _append_step(trajectory, state, target, diagnostics, elapsed):
    trajectory["loss_history"].append(float(diagnostics["loss_after"]))
    trajectory["normalized_l1_history"].append(
        _normalized_l1(target, state.query_answers, len(state.table))
    )
    trajectory["energy_history"].append(float(diagnostics["energy_after"]))
    trajectory["loss_gain_history"].append(float(diagnostics["gain"]))
    trajectory["energy_gain_history"].append(
        float(diagnostics["energy_gain"])
    )
    trajectory["changed_history"].append(bool(diagnostics["changed"]))
    trajectory["choice_index_history"].append(
        int(diagnostics["choice_index"])
    )
    trajectory["query_delta_sparse_history"].append(
        common._sparse_delta(diagnostics["query_delta"])
    )
    for target_name, source_name in (
        ("expected_energy_history", "expected_energy"),
        (
            "reference_expected_energy_history",
            "reference_expected_energy",
        ),
        (
            "expected_energy_gain_over_reference_history",
            "expected_energy_gain_over_reference",
        ),
        (
            "conditional_normalized_entropy_history",
            "conditional_normalized_entropy",
        ),
        ("uphill_probability_mass_history", "uphill_probability_mass"),
        ("probability_min_history", "probability_min"),
        ("probability_max_history", "probability_max"),
    ):
        trajectory[target_name].append(float(diagnostics[source_name]))
    trajectory["candidate_state_evaluations_history"].append(
        int(diagnostics["candidate_state_evaluations"])
    )
    trajectory["query_indicator_evaluations_history"].append(
        int(diagnostics["query_indicator_evaluations"])
    )
    trajectory["kernel_elapsed_sec"] += float(elapsed)


def _trajectory_metrics(trajectory, tail):
    energy = np.asarray(trajectory["energy_history"], dtype=float)
    normalized_l1 = np.asarray(
        trajectory["normalized_l1_history"], dtype=float
    )
    loss = np.asarray(trajectory["loss_history"], dtype=float)
    gains = np.asarray(trajectory["energy_gain_history"], dtype=float)
    changed = np.asarray(trajectory["changed_history"], dtype=bool)
    if len(energy) != len(normalized_l1) or len(energy) != len(loss):
        raise ValueError("轨迹状态历史长度不一致")
    if len(energy) - 1 < tail:
        raise ValueError("轨迹短于主终点尾窗")
    positive = gains > 0.0
    negative = gains < 0.0
    zero = gains == 0.0
    return {
        "tail_mean_energy": float(energy[-tail:].mean()),
        "final_energy": float(energy[-1]),
        "trajectory_mean_energy": float(energy[1:].mean()),
        "diagnostic_best_energy": float(energy.min()),
        "tail_mean_normalized_l1": float(normalized_l1[-tail:].mean()),
        "final_normalized_l1": float(normalized_l1[-1]),
        "trajectory_mean_normalized_l1": float(
            normalized_l1[1:].mean()
        ),
        "final_square_loss": float(loss[-1]),
        "positive_energy_steps": int(positive.sum()),
        "zero_energy_steps": int(zero.sum()),
        "negative_energy_steps": int(negative.sum()),
        "changed_value_rate": float(changed.mean()),
        "expected_energy_gain_over_reference_mean": float(np.mean(
            trajectory["expected_energy_gain_over_reference_history"]
        )),
        "conditional_normalized_entropy_mean": float(np.mean(
            trajectory["conditional_normalized_entropy_history"]
        )),
        "uphill_probability_mass_mean": float(np.mean(
            trajectory["uphill_probability_mass_history"]
        )),
        "probability_min": float(np.min(
            trajectory["probability_min_history"]
        )),
        "probability_max": float(np.max(
            trajectory["probability_max_history"]
        )),
        "candidate_state_evaluations": int(np.sum(
            trajectory["candidate_state_evaluations_history"]
        )),
        "query_indicator_evaluations": int(np.sum(
            trajectory["query_indicator_evaluations_history"]
        )),
        "kernel_elapsed_sec": float(trajectory["kernel_elapsed_sec"]),
        "full_table_state_audits": int(len(
            trajectory["full_state_audits"]
        )),
    }


def run_seed(
    seed,
    schema,
    queries,
    target,
    marginals,
    *,
    steps,
    tail,
    tau,
    verify_every,
):
    """共享初始表、坐标与 Gumbel 输入，比较平方能量和 L1 能量。"""
    if (
        isinstance(steps, bool)
        or not isinstance(steps, (int, np.integer))
        or steps <= 0
    ):
        raise ValueError("steps 必须是正整数")
    if (
        isinstance(tail, bool)
        or not isinstance(tail, (int, np.integer))
        or not 1 <= tail <= steps
    ):
        raise ValueError("tail 必须位于 [1, steps]")
    if (
        isinstance(verify_every, bool)
        or not isinstance(verify_every, (int, np.integer))
        or verify_every <= 0
    ):
        raise ValueError("verify_every 必须是正整数")
    if (
        isinstance(tau, (bool, np.bool_))
        or not isinstance(tau, (int, float, np.integer, np.floating))
        or not np.isfinite(tau)
        or tau < 0.0
    ):
        raise ValueError("tau 必须是非负有限数值")
    start = time.perf_counter()
    initial_table = init_synthetic_table(
        N_RECORDS,
        schema,
        np.random.default_rng(seed),
        marginals=marginals,
    )
    initial_state = initialize_persistent_heatbath_state(
        initial_table, schema, queries, target
    )
    states = {
        "baseline": initial_state.copy(),
        "candidate": initial_state.copy(),
    }
    modes = {
        "baseline": ENERGY_MODE_SQUARED,
        "candidate": ENERGY_MODE_NORMALIZED_L1,
    }
    scales = {}
    scale_elapsed = {}
    for variant, mode in modes.items():
        scale_start = time.perf_counter()
        scales[variant] = initial_gain_rms_scale(
            initial_state,
            schema,
            queries,
            target,
            energy_mode=mode,
        )
        scale_elapsed[variant] = float(
            time.perf_counter() - scale_start
        )
        if scales[variant]["scale"] <= 0.0:
            raise ValueError(f"{variant} 初始能量变化 RMS 必须严格为正")
    inverse_scales = {
        variant: float(tau / scales[variant]["scale"])
        for variant in modes
    }
    trajectories = {
        variant: _empty_trajectory(states[variant], target, modes[variant])
        for variant in modes
    }
    for variant in modes:
        trajectories[variant]["full_state_audits"].append(
            _state_audit(
                states[variant], schema, queries, target, step=0
            )
        )

    rng_seed = _address_seed(seed)
    rngs = {
        variant: np.random.default_rng(rng_seed) for variant in modes
    }
    initial_rng_sha256 = common._rng_state_sha256(rngs["baseline"])
    coordinate_history = []
    random_digest = hashlib.sha256()
    random_inputs_aligned = True
    maximum_loss_identity_error = 0.0
    maximum_energy_identity_error = 0.0
    expectation_violation_max = -np.inf
    mode_mismatch = False

    for step in range(1, steps + 1):
        coordinates = {
            variant: int(rng.integers(
                0, N_RECORDS * schema.n_blocks()
            ))
            for variant, rng in rngs.items()
        }
        coordinate = coordinates["baseline"]
        attribute = coordinate % schema.n_blocks()
        domain_size = len(legal_attribute_values(
            schema.attributes[attribute]
        ))
        gumbels = {
            variant: rng.gumbel(size=domain_size)
            for variant, rng in rngs.items()
        }
        aligned = (
            coordinates["baseline"] == coordinates["candidate"]
            and np.array_equal(gumbels["baseline"], gumbels["candidate"])
        )
        random_inputs_aligned = random_inputs_aligned and aligned
        if not aligned:
            raise RuntimeError("配对随机坐标或 Gumbel 输入发生错位")
        coordinate_history.append(coordinate)
        random_digest.update(
            np.asarray([coordinate], dtype=np.int64).tobytes()
        )
        random_digest.update(np.ascontiguousarray(
            gumbels["baseline"], dtype=np.float64
        ).tobytes())

        for variant in modes:
            state = states[variant]
            before_energy = _energy_from_state(
                state, target, modes[variant]
            )
            kernel_start = time.perf_counter()
            diagnostics = persistent_heatbath_step(
                state,
                schema,
                queries,
                target,
                inverse_temperature=inverse_scales[variant],
                rng=rngs[variant],
                coordinate_index=coordinate,
                gumbels=gumbels[variant],
                energy_mode=modes[variant],
            )
            kernel_elapsed = time.perf_counter() - kernel_start
            after_energy = _energy_from_state(
                state, target, modes[variant]
            )
            _append_step(
                trajectories[variant],
                state,
                target,
                diagnostics,
                kernel_elapsed,
            )
            maximum_loss_identity_error = max(
                maximum_loss_identity_error,
                abs(
                    diagnostics["gain"]
                    - (
                        diagnostics["loss_before"]
                        - diagnostics["loss_after"]
                    )
                ),
            )
            maximum_energy_identity_error = max(
                maximum_energy_identity_error,
                abs(diagnostics["energy_before"] - before_energy),
                abs(diagnostics["energy_after"] - after_energy),
                abs(
                    diagnostics["energy_gain"]
                    - (before_energy - after_energy)
                ),
            )
            expectation_violation_max = max(
                expectation_violation_max,
                -diagnostics["expected_energy_gain_over_reference"],
            )
            mode_mismatch = (
                mode_mismatch
                or diagnostics["energy_mode"] != modes[variant]
            )
        if step % verify_every == 0 or step == steps:
            for variant in modes:
                trajectories[variant]["full_state_audits"].append(
                    _state_audit(
                        states[variant],
                        schema,
                        queries,
                        target,
                        step,
                    )
                )

    final_rng_sha256 = {
        variant: common._rng_state_sha256(rng)
        for variant, rng in rngs.items()
    }
    for variant in modes:
        state = states[variant]
        trajectory = trajectories[variant]
        trajectory["metrics"] = _trajectory_metrics(trajectory, tail)
        trajectory["final_table_sha256"] = common._frame_sha256(
            state.table
        )
        trajectory["final_table_records"] = common._frame_records(
            state.table
        )
        trajectory["final_query_answers"] = [
            int(value) for value in state.query_answers
        ]

    expected_audits = steps // verify_every + 1
    if steps % verify_every:
        expected_audits += 1
    gates = {
        "initial_scales_positive": bool(all(
            scale["scale"] > 0.0 for scale in scales.values()
        )),
        "initial_states_aligned": bool(
            trajectories["baseline"]["full_state_audits"][0][
                "table_sha256"
            ]
            == trajectories["candidate"]["full_state_audits"][0][
                "table_sha256"
            ]
        ),
        "random_inputs_aligned": bool(random_inputs_aligned),
        "random_rng_endpoints_aligned": bool(
            final_rng_sha256["baseline"]
            == final_rng_sha256["candidate"]
        ),
        "full_audit_counts_complete": bool(all(
            len(trajectory["full_state_audits"]) == expected_audits
            for trajectory in trajectories.values()
        )),
        "all_full_state_audits_exact": bool(all(
            audit["query_answer_max_abs_error"] == 0
            and audit["loss_abs_error"] <= IDENTITY_TOLERANCE
            for trajectory in trajectories.values()
            for audit in trajectory["full_state_audits"]
        )),
        "loss_identity_max_error": float(maximum_loss_identity_error),
        "energy_identity_max_error": float(maximum_energy_identity_error),
        "conditional_expectation_violation_max": float(
            expectation_violation_max
        ),
        "all_probabilities_strictly_positive": bool(all(
            trajectory["metrics"]["probability_min"] > 0.0
            for trajectory in trajectories.values()
        )),
        "energy_modes_exact": bool(not mode_mismatch),
    }
    return {
        "seed": int(seed),
        "n_steps": int(steps),
        "tail_window": int(tail),
        "tau": float(tau),
        "initial_table_sha256": common._frame_sha256(initial_table),
        "initial_loss": float(initial_state.loss),
        "initial_normalized_l1": _normalized_l1(
            target, initial_state.query_answers, len(initial_state.table)
        ),
        "initial_query_answers": [
            int(value) for value in initial_state.query_answers
        ],
        "initial_gain_rms_scales": scales,
        "scale_elapsed_sec": scale_elapsed,
        "inverse_energy_scales": inverse_scales,
        "pair_rng_seed": int(rng_seed),
        "pair_rng_initial_state_sha256": initial_rng_sha256,
        "rng_final_state_sha256": final_rng_sha256,
        "random_input_sha256": random_digest.hexdigest(),
        "coordinate_history": coordinate_history,
        "baseline": trajectories["baseline"],
        "candidate": trajectories["candidate"],
        "gates": gates,
        "elapsed_sec": float(time.perf_counter() - start),
    }


def _paired_metric_summary(runs, metric, *, lower_is_better):
    baseline = np.asarray([
        run["baseline"]["metrics"][metric] for run in runs
    ], dtype=float)
    candidate = np.asarray([
        run["candidate"]["metrics"][metric] for run in runs
    ], dtype=float)
    differences = candidate - baseline
    improved = differences < 0.0 if lower_is_better else differences > 0.0
    worsened = differences > 0.0 if lower_is_better else differences < 0.0
    return {
        "baseline": common._summarize(baseline),
        "candidate": common._summarize(candidate),
        "candidate_minus_baseline": {
            **common._summarize(differences),
            "median": float(np.median(differences)),
            "wins": int(np.sum(improved)),
            "ties": int(np.sum(differences == 0.0)),
            "losses": int(np.sum(worsened)),
            "lower_is_better": bool(lower_is_better),
        },
        "baseline_by_seed": baseline.tolist(),
        "candidate_by_seed": candidate.tolist(),
        "difference_by_seed": differences.tolist(),
    }


def aggregate_results(runs, exact_oracle, expected_seeds, steps, *, formal):
    metric_directions = {
        "final_normalized_l1": True,
        "tail_mean_normalized_l1": True,
        "final_square_loss": True,
        "conditional_normalized_entropy_mean": False,
        "uphill_probability_mass_mean": False,
        "kernel_elapsed_sec": True,
        "candidate_state_evaluations": True,
        "query_indicator_evaluations": True,
    }
    run_gates_passed = all(
        run["n_steps"] == steps
        and run["baseline"]["energy_mode"] == ENERGY_MODE_SQUARED
        and run["candidate"]["energy_mode"]
        == ENERGY_MODE_NORMALIZED_L1
        and all((
            run["gates"]["initial_scales_positive"],
            run["gates"]["initial_states_aligned"],
            run["gates"]["random_inputs_aligned"],
            run["gates"]["random_rng_endpoints_aligned"],
            run["gates"]["full_audit_counts_complete"],
            run["gates"]["all_full_state_audits_exact"],
            run["gates"]["loss_identity_max_error"]
            <= IDENTITY_TOLERANCE,
            run["gates"]["energy_identity_max_error"]
            <= IDENTITY_TOLERANCE,
            run["gates"]["conditional_expectation_violation_max"]
            <= IDENTITY_TOLERANCE,
            run["gates"]["all_probabilities_strictly_positive"],
            run["gates"]["energy_modes_exact"],
        ))
        for run in runs
    )
    gates = {
        "exact_oracle_passed": bool(exact_oracle["passed"]),
        "seed_set_complete": sorted(run["seed"] for run in runs)
        == sorted(expected_seeds),
        "all_runs_full_length": all(
            run["n_steps"] == steps for run in runs
        ),
        "all_run_semantic_gates_passed": bool(run_gates_passed),
    }
    all_gates_passed = bool(all(gates.values()))
    if not all_gates_passed:
        classification = "implementation_or_experiment_failure"
    elif formal:
        classification = "generation_complete_pending_offline_quality"
    else:
        classification = "exploratory_protocol_no_formal_classification"
    return {
        "classification": classification,
        "diagnostic_gates": gates,
        "all_diagnostic_gates_passed": all_gates_passed,
        "paired_metrics": {
            metric: _paired_metric_summary(
                runs, metric, lower_is_better=lower_is_better
            )
            for metric, lower_is_better in metric_directions.items()
        },
        "optimized_energy_by_variant": {
            variant: {
                metric: common._summarize([
                    run[variant]["metrics"][metric] for run in runs
                ])
                for metric in (
                    "final_energy",
                    "tail_mean_energy",
                    "trajectory_mean_energy",
                )
            }
            for variant in ("baseline", "candidate")
        },
        "initial_normalized_l1": common._summarize([
            run["initial_normalized_l1"] for run in runs
        ]),
        "scale_by_variant": {
            variant: common._summarize([
                run["initial_gain_rms_scales"][variant]["scale"]
                for run in runs
            ])
            for variant in ("baseline", "candidate")
        },
    }


def _float_close(left, right):
    return common._float_close(left, right, IDENTITY_TOLERANCE)


def _step_mismatch(trajectory, index, state, target, diagnostics):
    expected = {
        "loss_history": diagnostics["loss_after"],
        "energy_history": diagnostics["energy_after"],
        "loss_gain_history": diagnostics["gain"],
        "energy_gain_history": diagnostics["energy_gain"],
        "expected_energy_history": diagnostics["expected_energy"],
        "reference_expected_energy_history": diagnostics[
            "reference_expected_energy"
        ],
        "expected_energy_gain_over_reference_history": diagnostics[
            "expected_energy_gain_over_reference"
        ],
        "conditional_normalized_entropy_history": diagnostics[
            "conditional_normalized_entropy"
        ],
        "uphill_probability_mass_history": diagnostics[
            "uphill_probability_mass"
        ],
        "probability_min_history": diagnostics["probability_min"],
        "probability_max_history": diagnostics["probability_max"],
    }
    for name, value in expected.items():
        offset = 1 if name in ("loss_history", "energy_history") else 0
        if not _float_close(trajectory[name][index + offset], value):
            return name
    normalized_l1 = _normalized_l1(
        target, state.query_answers, len(state.table)
    )
    if not _float_close(
        trajectory["normalized_l1_history"][index + 1], normalized_l1
    ):
        return "normalized_l1_history"
    exact = {
        "changed_history": bool(diagnostics["changed"]),
        "choice_index_history": int(diagnostics["choice_index"]),
        "query_delta_sparse_history": common._sparse_delta(
            diagnostics["query_delta"]
        ),
        "candidate_state_evaluations_history": int(
            diagnostics["candidate_state_evaluations"]
        ),
        "query_indicator_evaluations_history": int(
            diagnostics["query_indicator_evaluations"]
        ),
    }
    for name, value in exact.items():
        if trajectory[name][index] != value:
            return name
    return None


def _formal_payload_matches(payload):
    protocol = payload.get("protocol", {})
    return bool(
        payload.get("experiment") == "l1_persistent_workload_heatbath"
        and payload.get("formal_protocol") is True
        and protocol.get("n_records") == N_RECORDS
        and protocol.get("seeds") == FORMAL_SEEDS
        and protocol.get("steps") == FORMAL_STEPS
        and protocol.get("tail_window") == FORMAL_TAIL
        and protocol.get("tau") == FORMAL_TAU
        and protocol.get("verify_every") == FORMAL_VERIFY_EVERY
        and protocol.get("device") == FORMAL_DEVICE
        and protocol.get("baseline_energy_mode") == ENERGY_MODE_SQUARED
        and protocol.get("candidate_energy_mode")
        == ENERGY_MODE_NORMALIZED_L1
        and protocol.get("acceptance_or_checkpoint_selection") is False
    )


def independent_audit(payload, *, input_paths):
    """从公开输入、seed 和稀疏历史独立重放全部正式转移。

    审计口径（2026-08-12 审查修复）：target 从已哈希的 queries 重新派生并
    绑定比较，protocol.n_records 与 marginals 公开记录数比对；step 0
    checkpoint 与重建初始状态显式比较；checked_* 计数只在对应项实际审计
    成功后递增。
    """
    failures = []
    checked_seed_trajectories = 0
    checked_transitions = 0
    checked_public_initial_states = 0
    checked_random_schedules = 0
    checked_final_tables = 0
    required_paths = {"schema", "queries", "marginals"}
    if set(input_paths) != required_paths:
        raise ValueError("input_paths 必须包含 schema、queries 与 marginals")
    try:
        schema = load_schema(str(input_paths["schema"]))
        queries = load_queries(str(input_paths["queries"]))
        marginals = load_marginals(str(input_paths["marginals"]))
        target = np.asarray(payload["target"], dtype=float)
        # target 绑定：payload 自带 target 必须逐元素等于已哈希 queries 的
        # result；否则结果只与自带 target 自洽，不对应声明的公开查询目标。
        expected_target = np.asarray(
            [query["result"] for query in queries], dtype=float
        )
        if not np.array_equal(target, expected_target):
            failures.append({"reason": "target_public_input_mismatch"})
        recorded_hashes = payload["public_input_sha256"]
        if any(
            common._sha256_file(path) != recorded_hashes.get(name)
            for name, path in input_paths.items()
        ):
            failures.append({"reason": "public_input_hash_mismatch"})
    except Exception as error:
        return {
            "passed": False,
            "checked_seed_trajectories": 0,
            "checked_transitions": 0,
            "failures": [{
                "reason": "public_input_load_failure",
                "error_type": type(error).__name__,
            }],
        }
    steps = int(payload["protocol"]["steps"])
    tail = int(payload["protocol"]["tail_window"])
    verify_every = int(payload["protocol"]["verify_every"])
    n_records = int(payload["protocol"]["n_records"])
    if int(marginals.get("n_records", -1)) != n_records:
        failures.append({"reason": "marginal_record_count_mismatch"})
    if payload.get("exact_oracle") != run_exact_oracle():
        failures.append({"reason": "exact_oracle_mismatch"})

    for run in payload.get("runs", []):
        seed = int(run["seed"])
        try:
            if (
                int(run["n_steps"]) != steps
                or int(run["tail_window"]) != tail
                or not _float_close(run["tau"], payload["protocol"]["tau"])
            ):
                failures.append({
                    "seed": seed,
                    "reason": "run_protocol_metadata_mismatch",
                })
            initial_table = init_synthetic_table(
                n_records,
                schema,
                np.random.default_rng(seed),
                marginals=marginals,
            )
            initial_state = initialize_persistent_heatbath_state(
                initial_table, schema, queries, target
            )
            if (
                common._frame_sha256(initial_table)
                != run["initial_table_sha256"]
                or [int(value) for value in initial_state.query_answers]
                != run["initial_query_answers"]
                or not _float_close(initial_state.loss, run["initial_loss"])
                or not _float_close(
                    _normalized_l1(
                        target, initial_state.query_answers, n_records
                    ),
                    run["initial_normalized_l1"],
                )
            ):
                failures.append({
                    "seed": seed,
                    "reason": "regenerated_initial_state_mismatch",
                })
            else:
                checked_public_initial_states += 1

            modes = {
                "baseline": ENERGY_MODE_SQUARED,
                "candidate": ENERGY_MODE_NORMALIZED_L1,
            }
            inverse_scales = {}
            for variant, mode in modes.items():
                scale = initial_gain_rms_scale(
                    initial_state,
                    schema,
                    queries,
                    target,
                    energy_mode=mode,
                )
                if scale != run["initial_gain_rms_scales"][variant]:
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "initial_scale_mismatch",
                    })
                inverse_scales[variant] = run["tau"] / scale["scale"]
                if not _float_close(
                    inverse_scales[variant],
                    run["inverse_energy_scales"][variant],
                ):
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "inverse_scale_mismatch",
                    })

            replay_rng = np.random.default_rng(_address_seed(seed))
            if (
                run["pair_rng_seed"] != _address_seed(seed)
                or common._rng_state_sha256(replay_rng)
                != run["pair_rng_initial_state_sha256"]
            ):
                failures.append({
                    "seed": seed,
                    "reason": "random_schedule_initial_state_mismatch",
                })
            schedule = []
            random_digest = hashlib.sha256()
            for recorded_coordinate in run["coordinate_history"]:
                coordinate = int(replay_rng.integers(
                    0, n_records * schema.n_blocks()
                ))
                attribute = coordinate % schema.n_blocks()
                domain_size = len(legal_attribute_values(
                    schema.attributes[attribute]
                ))
                gumbels = replay_rng.gumbel(size=domain_size)
                schedule.append((coordinate, gumbels.copy()))
                random_digest.update(
                    np.asarray([coordinate], dtype=np.int64).tobytes()
                )
                random_digest.update(np.ascontiguousarray(
                    gumbels, dtype=np.float64
                ).tobytes())
                if coordinate != int(recorded_coordinate):
                    failures.append({
                        "seed": seed,
                        "reason": "random_schedule_coordinate_mismatch",
                    })
                    break
            final_rng_hash = common._rng_state_sha256(replay_rng)
            if (
                len(schedule) != steps
                or random_digest.hexdigest() != run["random_input_sha256"]
                or any(
                    run["rng_final_state_sha256"][variant]
                    != final_rng_hash
                    for variant in modes
                )
            ):
                failures.append({
                    "seed": seed,
                    "reason": "random_schedule_replay_mismatch",
                })
            else:
                checked_random_schedules += 1

            expected_checkpoints = [0]
            expected_checkpoints.extend(range(
                verify_every, steps + 1, verify_every
            ))
            if expected_checkpoints[-1] != steps:
                expected_checkpoints.append(steps)
            for variant, mode in modes.items():
                trajectory = run[variant]
                state = initial_state.copy()
                per_step_histories = (
                    "loss_gain_history",
                    "energy_gain_history",
                    "changed_history",
                    "choice_index_history",
                    "query_delta_sparse_history",
                    "expected_energy_history",
                    "reference_expected_energy_history",
                    "expected_energy_gain_over_reference_history",
                    "conditional_normalized_entropy_history",
                    "uphill_probability_mass_history",
                    "probability_min_history",
                    "probability_max_history",
                    "candidate_state_evaluations_history",
                    "query_indicator_evaluations_history",
                )
                if (
                    trajectory.get("energy_mode") != mode
                    or len(trajectory.get("loss_history", [])) != steps + 1
                    or len(trajectory.get("energy_history", []))
                    != steps + 1
                    or len(trajectory.get("normalized_l1_history", []))
                    != steps + 1
                    or any(
                        len(trajectory.get(name, [])) != steps
                        for name in per_step_histories
                    )
                    or trajectory["initial_query_answers"]
                    != [int(value) for value in state.query_answers]
                    or not _float_close(
                        trajectory["loss_history"][0], state.loss
                    )
                    or not _float_close(
                        trajectory["normalized_l1_history"][0],
                        _normalized_l1(
                            target, state.query_answers, n_records
                        ),
                    )
                    or not _float_close(
                        trajectory["energy_history"][0],
                        _energy_from_state(state, target, mode),
                    )
                ):
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "trajectory_structure_or_initial_mismatch",
                    })
                    continue
                checkpoints = {
                    int(row["step"]): row
                    for row in trajectory["full_state_audits"]
                }
                if list(checkpoints) != expected_checkpoints:
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "checkpoint_schedule_mismatch",
                    })
                # step 0 checkpoint 显式绑定重建初始状态：只在转移后比较会
                # 让初始 checkpoint 永远不被验证（审查修复）。
                if 0 not in checkpoints or checkpoints[0] != _state_audit(
                    state, schema, queries, target, 0
                ):
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "initial_checkpoint_mismatch",
                    })
                replay_broken = False
                for index, (coordinate, gumbels) in enumerate(schedule):
                    diagnostics = persistent_heatbath_step(
                        state,
                        schema,
                        queries,
                        target,
                        inverse_temperature=inverse_scales[variant],
                        rng=np.random.default_rng(0),
                        coordinate_index=coordinate,
                        gumbels=gumbels,
                        energy_mode=mode,
                    )
                    mismatch = _step_mismatch(
                        trajectory, index, state, target, diagnostics
                    )
                    if mismatch is not None:
                        failures.append({
                            "seed": seed,
                            "variant": variant,
                            "step": index + 1,
                            "reason": "transition_replay_mismatch",
                            "field": mismatch,
                        })
                        replay_broken = True
                        break
                    checked_transitions += 1
                    step_number = index + 1
                    if step_number in checkpoints:
                        audit = _state_audit(
                            state,
                            schema,
                            queries,
                            target,
                            step_number,
                        )
                        if audit != checkpoints[step_number]:
                            failures.append({
                                "seed": seed,
                                "variant": variant,
                                "step": step_number,
                                "reason": "checkpoint_replay_mismatch",
                            })
                            replay_broken = True
                            break
                if not replay_broken:
                    checked_seed_trajectories += 1
                if (
                    common._frame_sha256(state.table)
                    != trajectory["final_table_sha256"]
                    or common._frame_records(state.table)
                    != trajectory["final_table_records"]
                    or [int(value) for value in state.query_answers]
                    != trajectory["final_query_answers"]
                ):
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "final_state_mismatch",
                    })
                elif not replay_broken:
                    checked_final_tables += 1
                recomputed = _trajectory_metrics(trajectory, tail)
                if recomputed != trajectory["metrics"]:
                    failures.append({
                        "seed": seed,
                        "variant": variant,
                        "reason": "trajectory_metric_mismatch",
                    })
            gates = run.get("gates", {})
            if not (
                gates.get("initial_scales_positive") is True
                and gates.get("initial_states_aligned") is True
                and gates.get("random_inputs_aligned") is True
                and gates.get("random_rng_endpoints_aligned") is True
                and gates.get("full_audit_counts_complete") is True
                and gates.get("all_full_state_audits_exact") is True
                and gates.get("all_probabilities_strictly_positive") is True
                and gates.get("energy_modes_exact") is True
                and gates.get("loss_identity_max_error", np.inf)
                <= IDENTITY_TOLERANCE
                and gates.get("energy_identity_max_error", np.inf)
                <= IDENTITY_TOLERANCE
                and gates.get(
                    "conditional_expectation_violation_max", np.inf
                ) <= IDENTITY_TOLERANCE
            ):
                failures.append({
                    "seed": seed,
                    "reason": "recorded_semantic_gate_failure",
                })
        except Exception as error:
            failures.append({
                "seed": seed,
                "reason": "seed_replay_failure",
                "error_type": type(error).__name__,
            })

    expected_runs = payload["protocol"]["seeds"]
    if sorted(run["seed"] for run in payload.get("runs", [])) != sorted(
        expected_runs
    ):
        failures.append({"reason": "seed_set_mismatch"})
    recomputed_aggregate = aggregate_results(
        payload.get("runs", []),
        payload["exact_oracle"],
        expected_runs,
        steps,
        formal=bool(payload.get("formal_protocol")),
    )
    if recomputed_aggregate != payload.get("aggregate"):
        failures.append({"reason": "aggregate_mismatch"})
    return {
        "passed": not failures,
        # 全部计数只在对应项实际审计成功后递增；异常或提前退出不再报告满额。
        "checked_seed_trajectories": int(checked_seed_trajectories),
        "checked_transitions": int(checked_transitions),
        "checked_public_initial_states": int(checked_public_initial_states),
        "checked_random_schedules": int(checked_random_schedules),
        "checked_final_tables": int(checked_final_tables),
        "failures": failures,
    }


def _formal_protocol_matches(args):
    return bool(
        args.seeds == FORMAL_SEEDS
        and args.steps == FORMAL_STEPS
        and args.tail == FORMAL_TAIL
        and args.tau == FORMAL_TAU
        and args.verify_every == FORMAL_VERIFY_EVERY
        and args.device == FORMAL_DEVICE
    )


def _load_public_inputs():
    schema = load_schema(str(SCHEMA_PATH))
    queries = load_queries(str(QUERY_PATH))
    target = np.asarray(
        [query["result"] for query in queries], dtype=float
    )
    marginals = load_marginals(str(MARGINALS_PATH))
    if (
        schema.n_blocks() != 10
        or len(queries) != 50
        or target.shape != (50,)
        or not np.all(np.isfinite(target))
        or marginals.get("n_records") != N_RECORDS
        or set(marginals.get("attributes", {}))
        != set(schema.attribute_names())
    ):
        raise ValueError("test_300x10 的公开输入与协议不一致")
    return schema, queries, target, marginals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-existing",
        type=Path,
        help="只读重审已有 JSON，不运行实验或覆盖输出",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--steps", type=int, default=FORMAL_STEPS)
    parser.add_argument("--tail", type=int, default=FORMAL_TAIL)
    parser.add_argument("--tau", type=float, default=FORMAL_TAU)
    parser.add_argument(
        "--verify-every", type=int, default=FORMAL_VERIFY_EVERY
    )
    parser.add_argument("--device", choices=["cpu"], default=FORMAL_DEVICE)
    parser.add_argument("--output", default=str(FORMAL_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_paths = {
        "schema": SCHEMA_PATH,
        "queries": QUERY_PATH,
        "marginals": MARGINALS_PATH,
    }
    if args.audit_existing is not None:
        payload = common._load_json_strict(args.audit_existing)
        audit = independent_audit(payload, input_paths=input_paths)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"input={args.audit_existing}")
        print(f"sha256={common._sha256_file(args.audit_existing)}")
        if not audit["passed"]:
            raise RuntimeError("已有输出独立审计未通过")
        return

    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error("--seeds 必须非空、非负且不重复")
    if args.steps <= 0:
        parser.error("--steps 必须为正整数")
    if not 1 <= args.tail <= args.steps:
        parser.error("--tail 必须位于 [1, steps]")
    if not np.isfinite(args.tau) or args.tau < 0.0:
        parser.error("--tau 必须是非负有限数值")
    if args.verify_every <= 0:
        parser.error("--verify-every 必须为正整数")

    formal = _formal_protocol_matches(args)
    output = Path(args.output)
    if output.exists() and (formal or not args.overwrite):
        raise FileExistsError(f"输出已存在，不覆盖：{output}")
    environment = common._environment_snapshot(args.device)
    if formal and not environment[
        "git_worktree_clean_including_untracked"
    ]:
        raise RuntimeError("正式协议要求当前提交对应的工作树完全干净")
    schema, queries, target, marginals = _load_public_inputs()
    exact_oracle = run_exact_oracle()
    if not exact_oracle["passed"]:
        raise RuntimeError("L1 的 16 状态精确 oracle 未通过")

    experiment_start = time.perf_counter()
    runs = []
    for index, seed in enumerate(args.seeds, start=1):
        run = run_seed(
            seed,
            schema,
            queries,
            target,
            marginals,
            steps=args.steps,
            tail=args.tail,
            tau=args.tau,
            verify_every=args.verify_every,
        )
        runs.append(run)
        print(
            f"[{index}/{len(args.seeds)}] seed={seed} measured L1 "
            f"{run['baseline']['metrics']['final_normalized_l1']:.6f} -> "
            f"{run['candidate']['metrics']['final_normalized_l1']:.6f} "
            f"({run['elapsed_sec']:.2f}s)",
            flush=True,
        )
    aggregate = aggregate_results(
        runs, exact_oracle, args.seeds, args.steps, formal=formal
    )
    payload = {
        "experiment": "l1_persistent_workload_heatbath",
        "research_boundary": (
            "exact-target no-noise stage-I energy-alignment experiment; "
            "not a full-generator comparison and not a DP claim"
        ),
        "formal_protocol": bool(formal),
        "protocol": {
            "schema_path": str(SCHEMA_PATH),
            "query_path": str(QUERY_PATH),
            "marginals_path": str(MARGINALS_PATH),
            "n_records": N_RECORDS,
            "seeds": list(args.seeds),
            "steps": int(args.steps),
            "tail_window": int(args.tail),
            "tau": float(args.tau),
            "verify_every": int(args.verify_every),
            "device": args.device,
            "baseline_energy_mode": ENERGY_MODE_SQUARED,
            "candidate_energy_mode": ENERGY_MODE_NORMALIZED_L1,
            "scale": (
                "per-seed initial nonzero single-coordinate energy-gain RMS; "
                "computed once per mode and frozen"
            ),
            "paired_randomness": (
                "same initial table, coordinates, and per-domain Gumbels"
            ),
            "output_state": "current table after the final microstep",
            "acceptance_or_checkpoint_selection": False,
            "real_reference_loaded_during_generation": False,
        },
        "environment": environment,
        "public_input_sha256": {
            name: common._sha256_file(path)
            for name, path in input_paths.items()
        },
        "target": target.tolist(),
        "exact_oracle": exact_oracle,
        "runs": runs,
        "aggregate": aggregate,
        "elapsed_sec": float(time.perf_counter() - experiment_start),
    }
    payload["independent_audit"] = independent_audit(
        payload, input_paths=input_paths
    )
    if not payload["independent_audit"]["passed"]:
        raise RuntimeError("保存前独立审计未通过")
    common._write_json_atomic(
        output,
        payload,
        overwrite=args.overwrite and not formal,
    )
    print(f"classification={aggregate['classification']}")
    print(f"output={output}")
    print(f"sha256={common._sha256_file(output)}")


if __name__ == "__main__":
    main()
