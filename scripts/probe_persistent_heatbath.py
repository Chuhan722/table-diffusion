"""运行持久化 workload 能量热浴扩散的精确 oracle 与配对冒烟实验。

生成阶段只读取公开 schema、公开记录数、预定义查询、已发布 target 与 1-way
marginal。脚本不接受真实训练/测试表路径，不执行接受、回滚、早停或 best 选择。
"""

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.persistent_heatbath import (
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
FORMAL_SEEDS = list(range(40, 60))
FORMAL_STEPS = 3000
FORMAL_TAIL = 750
FORMAL_TAU = 1.0
FORMAL_VERIFY_EVERY = 100
FORMAL_DEVICE = "cpu"
FORMAL_OUTPUT = Path(
    "outputs/persistent_workload_heatbath/"
    "formal_20seed_3000step_tau1.json"
)
ORACLE_BETAS = (0.0, 0.7, 1.3)
IDENTITY_TOLERANCE = 1e-12
REQUIRED_WINS = 14
MIN_SUPPORT_REDUCTION = 0.05


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame):
    return _sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def _rng_state_sha256(rng):
    serialized = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(serialized.encode("utf-8"))


def _git_text(*args):
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip()


def _environment_snapshot(device):
    commit_code, commit = _git_text("rev-parse", "HEAD")
    status_code, status = _git_text(
        "status", "--porcelain", "--untracked-files=no"
    )
    full_status_code, full_status = _git_text(
        "status", "--porcelain", "--untracked-files=normal"
    )
    return {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit if commit_code == 0 else None,
        "git_tracked_worktree_clean": status_code == 0 and status == "",
        "git_worktree_clean_including_untracked": (
            full_status_code == 0 and full_status == ""
        ),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _address_seed(seed):
    sequence = np.random.SeedSequence([
        int(seed),
        0x50455253495354,
        0x48454154424154,
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _sparse_delta(delta):
    values = np.asarray(delta, dtype=np.int64)
    indices = np.flatnonzero(values)
    return [[int(index), int(values[index])] for index in indices]


def _frame_records(frame):
    return [
        {
            column: (
                value.item() if isinstance(value, np.generic) else value
            )
            for column, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _load_json_strict(path):
    def reject_constant(value):
        raise ValueError(f"JSON 包含非标准数值常量：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _mean_t_interval(values, confidence=0.95):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("区间输入必须是非空有限一维数组")
    mean = float(array.mean())
    if len(array) == 1:
        return [mean, mean]
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    if standard_error == 0.0:
        return [mean, mean]
    critical = float(stats.t.ppf(
        0.5 + confidence / 2.0, df=len(array) - 1
    ))
    return [
        float(mean - critical * standard_error),
        float(mean + critical * standard_error),
    ]


def _summarize(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("汇总输入必须是非空有限一维数组")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean_t_interval_95": _mean_t_interval(array),
    }


def _binary_oracle_problem():
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
        for target in np.flatnonzero(matrix[source] > 0.0):
            target = int(target)
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return len(reached) == len(matrix)


def run_exact_oracle():
    """完整枚举 16 个有序表状态并返回数值语义门禁。"""
    states = tuple(itertools.product((0, 1), repeat=4))
    state_index = {state: index for index, state in enumerate(states)}
    schema, queries, target = _binary_oracle_problem()
    results = []
    expected_by_beta = {}
    derivative_max_error = 0.0
    global_increment_error = 0.0
    global_loss_identity_error = 0.0
    global_conditional_error = 0.0

    for beta in ORACLE_BETAS:
        matrix = np.zeros((16, 16), dtype=float)
        losses = np.zeros(16, dtype=float)
        conditionals = {}
        for index, bits in enumerate(states):
            state = initialize_persistent_heatbath_state(
                _oracle_table(bits), schema, queries, target
            )
            losses[index] = state.loss
            for coordinate in range(4):
                row, attribute = divmod(coordinate, 2)
                conditional = build_persistent_heatbath_conditional(
                    state,
                    schema,
                    queries,
                    target,
                    row_index=row,
                    attribute_index=attribute,
                    inverse_temperature=beta,
                )
                conditionals[(index, coordinate)] = conditional
                for choice, value in enumerate(conditional.values):
                    next_bits = list(bits)
                    next_bits[coordinate] = value
                    matrix[index, state_index[tuple(next_bits)]] += (
                        conditional.probabilities[choice] / 4.0
                    )
                    candidate = state.table.copy(deep=True)
                    candidate.iat[row, attribute] = value
                    full_answers = evaluate_table(candidate, queries)
                    full_loss = compute_loss(target, full_answers)
                    global_increment_error = max(
                        global_increment_error,
                        float(np.max(np.abs(
                            state.query_answers
                            + conditional.query_deltas[choice]
                            - full_answers
                        ))),
                    )
                    global_loss_identity_error = max(
                        global_loss_identity_error,
                        abs(float(conditional.candidate_losses[choice]) - full_loss),
                        abs(float(conditional.gains[choice]) - (state.loss - full_loss)),
                    )
        weights = np.exp(-beta * losses / 2.0)
        stationary = weights / weights.sum()
        conditional_expected = {}
        for (index, coordinate), conditional in conditionals.items():
            alternatives = []
            for value in conditional.values:
                bits = list(states[index])
                bits[coordinate] = value
                alternatives.append(state_index[tuple(bits)])
            oracle = stationary[alternatives]
            oracle = oracle / oracle.sum()
            global_conditional_error = max(
                global_conditional_error,
                float(np.max(np.abs(conditional.probabilities - oracle))),
            )
            conditional_expected[(index, coordinate)] = (
                conditional.expected_loss
            )
        expected_by_beta[beta] = conditional_expected
        flow = stationary[:, None] * matrix
        all_allowed_positive = True
        all_disallowed_zero = True
        for source, source_bits in enumerate(states):
            for destination, destination_bits in enumerate(states):
                hamming = sum(
                    left != right
                    for left, right in zip(source_bits, destination_bits)
                )
                if hamming <= 1:
                    all_allowed_positive = (
                        all_allowed_positive
                        and matrix[source, destination] > 0.0
                    )
                else:
                    all_disallowed_zero = (
                        all_disallowed_zero
                        and matrix[source, destination] == 0.0
                    )
        result = {
            "beta": beta,
            "row_sum_max_error": float(np.max(np.abs(matrix.sum(axis=1) - 1.0))),
            "minimum_positive_transition": float(matrix[matrix > 0.0].min()),
            "all_state_self_loops_positive": bool(np.all(np.diag(matrix) > 0.0)),
            "all_allowed_single_coordinate_transitions_positive": bool(
                all_allowed_positive
            ),
            "all_multi_coordinate_transitions_zero": bool(
                all_disallowed_zero
            ),
            "irreducible": _is_irreducible(matrix),
            "detailed_balance_max_error": float(np.max(np.abs(flow - flow.T))),
            "stationarity_max_error": float(np.max(np.abs(
                stationary @ matrix - stationary
            ))),
        }
        results.append(result)

    monotonic = True
    for key in expected_by_beta[0.0]:
        values = [expected_by_beta[beta][key] for beta in ORACLE_BETAS]
        monotonic = monotonic and all(
            values[index] + IDENTITY_TOLERANCE >= values[index + 1]
            for index in range(len(values) - 1)
        )

        index, coordinate = key
        bits = states[index]
        state = initialize_persistent_heatbath_state(
            _oracle_table(bits), schema, queries, target
        )
        row, attribute = divmod(coordinate, 2)
        center = build_persistent_heatbath_conditional(
            state, schema, queries, target,
            row_index=row, attribute_index=attribute,
            inverse_temperature=0.7,
        )
        epsilon = 1e-5
        upper = build_persistent_heatbath_conditional(
            state, schema, queries, target,
            row_index=row, attribute_index=attribute,
            inverse_temperature=0.7 + epsilon,
        )
        lower = build_persistent_heatbath_conditional(
            state, schema, queries, target,
            row_index=row, attribute_index=attribute,
            inverse_temperature=0.7 - epsilon,
        )
        numerical = (
            upper.expected_loss - lower.expected_loss
        ) / (2.0 * epsilon)
        variance = float(np.dot(
            center.probabilities,
            (center.candidate_losses - center.expected_loss) ** 2,
        ))
        derivative_max_error = max(
            derivative_max_error,
            abs(numerical + variance / 2.0),
        )

    passed = (
        all(result["row_sum_max_error"] <= IDENTITY_TOLERANCE for result in results)
        and all(result["minimum_positive_transition"] > 0.0 for result in results)
        and all(result["all_state_self_loops_positive"] for result in results)
        and all(
            result["all_allowed_single_coordinate_transitions_positive"]
            for result in results
        )
        and all(
            result["all_multi_coordinate_transitions_zero"]
            for result in results
        )
        and all(result["irreducible"] for result in results)
        and all(result["detailed_balance_max_error"] <= IDENTITY_TOLERANCE for result in results)
        and all(result["stationarity_max_error"] <= IDENTITY_TOLERANCE for result in results)
        and global_increment_error <= IDENTITY_TOLERANCE
        and global_loss_identity_error <= IDENTITY_TOLERANCE
        and global_conditional_error <= IDENTITY_TOLERANCE
        and monotonic
        and derivative_max_error <= 1e-9
    )
    return {
        "states": 16,
        "coordinates_per_state": 4,
        "betas": list(ORACLE_BETAS),
        "by_beta": results,
        "conditional_probability_max_error": global_conditional_error,
        "query_increment_max_error": global_increment_error,
        "loss_gain_identity_max_error": global_loss_identity_error,
        "expected_loss_monotonic": bool(monotonic),
        "derivative_identity_max_error": float(derivative_max_error),
        "passed": bool(passed),
    }


def _state_audit(state, schema, queries, target, step):
    verification = verify_persistent_heatbath_state(
        state, schema, queries, target
    )
    return {
        "step": int(step),
        "table_sha256": _frame_sha256(state.table),
        "query_answers": [int(value) for value in state.query_answers],
        "recorded_loss": float(state.loss),
        **verification,
    }


def _empty_trajectory(state):
    return {
        "initial_query_answers": [int(value) for value in state.query_answers],
        "loss_history": [float(state.loss)],
        "gain_history": [],
        "changed_history": [],
        "choice_index_history": [],
        "query_delta_sparse_history": [],
        "conditional_expected_state_gain_history": [],
        "expected_gain_over_reference_history": [],
        "conditional_normalized_entropy_history": [],
        "uphill_probability_mass_history": [],
        "probability_min_history": [],
        "probability_max_history": [],
        "candidate_state_evaluations_history": [],
        "query_indicator_evaluations_history": [],
        "full_state_audits": [],
        "kernel_elapsed_sec": 0.0,
    }


def _append_step(trajectory, diagnostics, elapsed):
    trajectory["loss_history"].append(float(diagnostics["loss_after"]))
    trajectory["gain_history"].append(float(diagnostics["gain"]))
    trajectory["changed_history"].append(bool(diagnostics["changed"]))
    trajectory["choice_index_history"].append(
        int(diagnostics["choice_index"])
    )
    trajectory["query_delta_sparse_history"].append(
        _sparse_delta(diagnostics["query_delta"])
    )
    trajectory["conditional_expected_state_gain_history"].append(
        float(diagnostics["loss_before"] - diagnostics["expected_loss"])
    )
    trajectory["expected_gain_over_reference_history"].append(
        float(diagnostics["expected_gain_over_reference"])
    )
    trajectory["conditional_normalized_entropy_history"].append(
        float(diagnostics["conditional_normalized_entropy"])
    )
    trajectory["uphill_probability_mass_history"].append(
        float(diagnostics["uphill_probability_mass"])
    )
    trajectory["probability_min_history"].append(
        float(diagnostics["probability_min"])
    )
    trajectory["probability_max_history"].append(
        float(diagnostics["probability_max"])
    )
    trajectory["candidate_state_evaluations_history"].append(
        int(diagnostics["candidate_state_evaluations"])
    )
    trajectory["query_indicator_evaluations_history"].append(
        int(diagnostics["query_indicator_evaluations"])
    )
    trajectory["kernel_elapsed_sec"] += float(elapsed)


def _trajectory_metrics(trajectory, tail):
    losses = np.asarray(trajectory["loss_history"], dtype=float)
    gains = np.asarray(trajectory["gain_history"], dtype=float)
    changed = np.asarray(trajectory["changed_history"], dtype=bool)
    post_step = losses[1:]
    if len(post_step) < tail:
        raise ValueError("轨迹短于主终点尾窗")
    positive = gains > 0.0
    negative = gains < 0.0
    zero = gains == 0.0
    return {
        "tail_mean_loss": float(post_step[-tail:].mean()),
        "final_loss": float(losses[-1]),
        "trajectory_mean_loss": float(post_step.mean()),
        "diagnostic_best_loss": float(losses.min()),
        "positive_steps": int(positive.sum()),
        "zero_steps": int(zero.sum()),
        "negative_steps": int(negative.sum()),
        "positive_gain_mean": (
            float(gains[positive].mean()) if np.any(positive) else 0.0
        ),
        "negative_gain_abs_mean": (
            float((-gains[negative]).mean()) if np.any(negative) else 0.0
        ),
        "changed_value_rate": float(changed.mean()),
        "conditional_expected_state_gain_mean": float(np.mean(
            trajectory["conditional_expected_state_gain_history"]
        )),
        "expected_gain_over_reference_mean": float(np.mean(
            trajectory["expected_gain_over_reference_history"]
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
    """以共同坐标/Gumbel 随机量运行一个配对 seed。"""
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
    baseline_state = initial_state.copy()
    candidate_state = initial_state.copy()
    scale_start = time.perf_counter()
    scale = initial_gain_rms_scale(
        initial_state, schema, queries, target
    )
    scale_elapsed = time.perf_counter() - scale_start
    candidate_beta = tau / scale["scale"] if scale["scale"] > 0.0 else 0.0

    baseline = _empty_trajectory(baseline_state)
    candidate = _empty_trajectory(candidate_state)
    baseline["full_state_audits"].append(
        _state_audit(baseline_state, schema, queries, target, 0)
    )
    candidate["full_state_audits"].append(
        _state_audit(candidate_state, schema, queries, target, 0)
    )
    coordinate_history = []
    random_digest = hashlib.sha256()
    rng_seed = _address_seed(seed)
    baseline_rng = np.random.default_rng(rng_seed)
    candidate_rng = np.random.default_rng(rng_seed)
    initial_rng_sha256 = _rng_state_sha256(baseline_rng)
    random_inputs_aligned = True
    max_gain_identity_error = 0.0
    expectation_violation_max = -np.inf

    for step in range(1, steps + 1):
        n_coordinates = N_RECORDS * schema.n_blocks()
        baseline_coordinate = int(baseline_rng.integers(0, n_coordinates))
        candidate_coordinate = int(candidate_rng.integers(0, n_coordinates))
        attribute = baseline_coordinate % schema.n_blocks()
        domain_size = len(legal_attribute_values(schema.attributes[attribute]))
        baseline_gumbels = baseline_rng.gumbel(size=domain_size)
        candidate_gumbels = candidate_rng.gumbel(size=domain_size)
        aligned = (
            baseline_coordinate == candidate_coordinate
            and np.array_equal(baseline_gumbels, candidate_gumbels)
        )
        random_inputs_aligned = random_inputs_aligned and aligned
        if not aligned:
            raise RuntimeError("配对随机坐标或 Gumbel 输入发生错位")
        coordinate_history.append(baseline_coordinate)
        random_digest.update(
            np.asarray([baseline_coordinate], dtype=np.int64).tobytes()
        )
        random_digest.update(
            np.ascontiguousarray(baseline_gumbels, dtype=np.float64).tobytes()
        )

        for state, beta, rng, gumbels, trajectory in (
            (
                baseline_state,
                0.0,
                baseline_rng,
                baseline_gumbels,
                baseline,
            ),
            (
                candidate_state,
                candidate_beta,
                candidate_rng,
                candidate_gumbels,
                candidate,
            ),
        ):
            kernel_start = time.perf_counter()
            diagnostics = persistent_heatbath_step(
                state,
                schema,
                queries,
                target,
                inverse_temperature=beta,
                rng=rng,
                coordinate_index=baseline_coordinate,
                gumbels=gumbels,
            )
            kernel_elapsed = time.perf_counter() - kernel_start
            _append_step(trajectory, diagnostics, kernel_elapsed)
            max_gain_identity_error = max(
                max_gain_identity_error,
                abs(
                    diagnostics["gain"]
                    - (
                        diagnostics["loss_before"]
                        - diagnostics["loss_after"]
                    )
                ),
            )
            expectation_violation_max = max(
                expectation_violation_max,
                diagnostics["expected_loss"]
                - diagnostics["reference_expected_loss"],
            )
        if step % verify_every == 0 or step == steps:
            baseline["full_state_audits"].append(
                _state_audit(
                    baseline_state, schema, queries, target, step
                )
            )
            candidate["full_state_audits"].append(
                _state_audit(
                    candidate_state, schema, queries, target, step
                )
            )

    baseline_final_rng = _rng_state_sha256(baseline_rng)
    candidate_final_rng = _rng_state_sha256(candidate_rng)
    baseline["metrics"] = _trajectory_metrics(baseline, tail)
    candidate["metrics"] = _trajectory_metrics(candidate, tail)
    baseline["final_table_sha256"] = _frame_sha256(baseline_state.table)
    candidate["final_table_sha256"] = _frame_sha256(candidate_state.table)
    baseline["final_table_records"] = _frame_records(baseline_state.table)
    candidate["final_table_records"] = _frame_records(candidate_state.table)
    baseline["final_query_answers"] = [
        int(value) for value in baseline_state.query_answers
    ]
    candidate["final_query_answers"] = [
        int(value) for value in candidate_state.query_answers
    ]
    audit_count = steps // verify_every + 1
    if steps % verify_every:
        audit_count += 1
    gates = {
        "initial_scale_positive": bool(scale["scale"] > 0.0),
        "initial_states_aligned": bool(
            baseline["full_state_audits"][0]["table_sha256"]
            == candidate["full_state_audits"][0]["table_sha256"]
        ),
        "random_inputs_aligned": bool(random_inputs_aligned),
        "random_rng_endpoints_aligned": bool(
            baseline_final_rng == candidate_final_rng
        ),
        "full_audit_counts_complete": bool(
            len(baseline["full_state_audits"]) == audit_count
            == len(candidate["full_state_audits"])
        ),
        "all_full_state_audits_exact": bool(all(
            audit["query_answer_max_abs_error"] == 0
            and audit["loss_abs_error"] <= IDENTITY_TOLERANCE
            for trajectory in (baseline, candidate)
            for audit in trajectory["full_state_audits"]
        )),
        "gain_identity_max_error": float(max_gain_identity_error),
        "conditional_expectation_violation_max": float(
            expectation_violation_max
        ),
        "all_probabilities_strictly_positive": bool(
            baseline["metrics"]["probability_min"] > 0.0
            and candidate["metrics"]["probability_min"] > 0.0
        ),
    }
    return {
        "seed": int(seed),
        "n_steps": int(steps),
        "tail_window": int(tail),
        "tau": float(tau),
        "initial_table_sha256": _frame_sha256(initial_table),
        "initial_loss": float(initial_state.loss),
        "initial_query_answers": [
            int(value) for value in initial_state.query_answers
        ],
        "initial_gain_rms_scale": scale,
        "scale_elapsed_sec": float(scale_elapsed),
        "baseline_inverse_temperature": 0.0,
        "candidate_inverse_temperature": float(candidate_beta),
        "pair_rng_seed": int(rng_seed),
        "pair_rng_initial_state_sha256": initial_rng_sha256,
        "baseline_rng_final_state_sha256": baseline_final_rng,
        "candidate_rng_final_state_sha256": candidate_final_rng,
        "random_input_sha256": random_digest.hexdigest(),
        "coordinate_history": coordinate_history,
        "baseline": baseline,
        "candidate": candidate,
        "gates": gates,
        "elapsed_sec": float(time.perf_counter() - start),
    }


def _paired_metric_summary(runs, metric):
    baseline = [run["baseline"]["metrics"][metric] for run in runs]
    candidate = [run["candidate"]["metrics"][metric] for run in runs]
    differences = [right - left for left, right in zip(baseline, candidate)]
    return {
        "baseline": _summarize(baseline),
        "candidate": _summarize(candidate),
        "candidate_minus_baseline": {
            **_summarize(differences),
            "wins": int(np.sum(np.asarray(differences) < 0.0)),
            "ties": int(np.sum(np.asarray(differences) == 0.0)),
            "losses": int(np.sum(np.asarray(differences) > 0.0)),
            "lower_is_better": True,
        },
        "baseline_by_seed": [float(value) for value in baseline],
        "candidate_by_seed": [float(value) for value in candidate],
        "difference_by_seed": [float(value) for value in differences],
    }


def _variant_metric_summary(runs, metric):
    return {
        variant: _summarize([
            run[variant]["metrics"][metric] for run in runs
        ])
        for variant in ("baseline", "candidate")
    }


def aggregate_results(
    runs,
    exact_oracle,
    expected_seeds,
    steps,
    *,
    classify=True,
):
    primary = _paired_metric_summary(runs, "tail_mean_loss")
    baseline_mean = primary["baseline"]["mean"]
    candidate_mean = primary["candidate"]["mean"]
    reduction = (
        (baseline_mean - candidate_mean) / baseline_mean
        if baseline_mean > 0.0 else 0.0
    )
    wins = primary["candidate_minus_baseline"]["wins"]
    run_gates_passed = all(
        run["n_steps"] == steps
        and all((
            run["gates"]["initial_scale_positive"],
            run["gates"]["initial_states_aligned"],
            run["gates"]["random_inputs_aligned"],
            run["gates"]["random_rng_endpoints_aligned"],
            run["gates"]["full_audit_counts_complete"],
            run["gates"]["all_full_state_audits_exact"],
            run["gates"]["gain_identity_max_error"] <= IDENTITY_TOLERANCE,
            run["gates"]["conditional_expectation_violation_max"]
            <= IDENTITY_TOLERANCE,
            run["gates"]["all_probabilities_strictly_positive"],
        ))
        for run in runs
    )
    diagnostic_gates = {
        "exact_oracle_passed": bool(exact_oracle["passed"]),
        "seed_set_complete": sorted(run["seed"] for run in runs)
        == sorted(expected_seeds),
        "all_runs_full_length": all(run["n_steps"] == steps for run in runs),
        "all_run_semantic_gates_passed": bool(run_gates_passed),
    }
    all_gates_passed = all(diagnostic_gates.values())
    if not all_gates_passed:
        classification = "implementation_or_experiment_failure"
    elif not classify:
        classification = "exploratory_protocol_no_formal_classification"
    elif candidate_mean >= baseline_mean or wins <= 10:
        classification = "persistent_heatbath_smoke_not_supported"
    elif reduction >= MIN_SUPPORT_REDUCTION and wins >= REQUIRED_WINS:
        classification = "supports_persistent_heatbath_smoke"
    else:
        classification = "persistent_heatbath_smoke_inconclusive"

    candidate_entropy = _summarize([
        run["candidate"]["metrics"][
            "conditional_normalized_entropy_mean"
        ] for run in runs
    ])
    candidate_uphill = _summarize([
        run["candidate"]["metrics"]["uphill_probability_mass_mean"]
        for run in runs
    ])
    return {
        "primary_metric": "post_step_tail_mean_current_workload_loss",
        "primary": primary,
        "candidate_relative_reduction": float(reduction),
        "classification": classification,
        "classification_thresholds": {
            "minimum_relative_reduction": MIN_SUPPORT_REDUCTION,
            "minimum_wins": REQUIRED_WINS,
            "not_supported_maximum_wins": 10,
        },
        "secondary": {
            metric: _paired_metric_summary(runs, metric)
            for metric in ("final_loss", "trajectory_mean_loss")
        },
        "trajectory_diagnostics": {
            metric: _variant_metric_summary(runs, metric)
            for metric in (
                "diagnostic_best_loss",
                "positive_steps",
                "zero_steps",
                "negative_steps",
                "positive_gain_mean",
                "negative_gain_abs_mean",
                "changed_value_rate",
                "conditional_expected_state_gain_mean",
                "expected_gain_over_reference_mean",
                "conditional_normalized_entropy_mean",
                "uphill_probability_mass_mean",
                "candidate_state_evaluations",
                "query_indicator_evaluations",
                "kernel_elapsed_sec",
                "full_table_state_audits",
            )
        },
        "seed_runtime": {
            "scale_elapsed_sec": _summarize([
                run["scale_elapsed_sec"] for run in runs
            ]),
            "paired_run_elapsed_sec": _summarize([
                run["elapsed_sec"] for run in runs
            ]),
        },
        "candidate_risks": {
            "conditional_normalized_entropy": candidate_entropy,
            "uphill_probability_mass": candidate_uphill,
            "conditional_concentration_risk": bool(
                candidate_entropy["mean"] < 0.50
            ),
            "uphill_support_mass_risk": bool(
                candidate_uphill["mean"] < 0.01
            ),
        },
        "diagnostic_gates": diagnostic_gates,
        "all_diagnostic_gates_passed": bool(all_gates_passed),
    }


def _float_close(left, right, tolerance=IDENTITY_TOLERANCE):
    return bool(np.isclose(
        float(left), float(right), rtol=0.0, atol=tolerance
    ))


def _replay_step_mismatch(trajectory, index, diagnostics):
    if int(trajectory["choice_index_history"][index]) != int(
        diagnostics["choice_index"]
    ):
        return "choice_index"
    if bool(trajectory["changed_history"][index]) != bool(
        diagnostics["changed"]
    ):
        return "changed"
    if trajectory["query_delta_sparse_history"][index] != _sparse_delta(
        diagnostics["query_delta"]
    ):
        return "query_delta"
    if int(trajectory["candidate_state_evaluations_history"][index]) != int(
        diagnostics["candidate_state_evaluations"]
    ):
        return "candidate_state_evaluations"
    if int(trajectory["query_indicator_evaluations_history"][index]) != int(
        diagnostics["query_indicator_evaluations"]
    ):
        return "query_indicator_evaluations"
    float_fields = {
        "loss_before": (
            trajectory["loss_history"][index], diagnostics["loss_before"]
        ),
        "loss_after": (
            trajectory["loss_history"][index + 1],
            diagnostics["loss_after"],
        ),
        "gain": (
            trajectory["gain_history"][index], diagnostics["gain"]
        ),
        "conditional_expected_state_gain": (
            trajectory["conditional_expected_state_gain_history"][index],
            diagnostics["loss_before"] - diagnostics["expected_loss"],
        ),
        "expected_gain_over_reference": (
            trajectory["expected_gain_over_reference_history"][index],
            diagnostics["expected_gain_over_reference"],
        ),
        "conditional_normalized_entropy": (
            trajectory["conditional_normalized_entropy_history"][index],
            diagnostics["conditional_normalized_entropy"],
        ),
        "uphill_probability_mass": (
            trajectory["uphill_probability_mass_history"][index],
            diagnostics["uphill_probability_mass"],
        ),
        "probability_min": (
            trajectory["probability_min_history"][index],
            diagnostics["probability_min"],
        ),
        "probability_max": (
            trajectory["probability_max_history"][index],
            diagnostics["probability_max"],
        ),
    }
    for name, (recorded, recomputed) in float_fields.items():
        if not _float_close(recorded, recomputed):
            return name
    return None


def _formal_payload_matches(payload):
    protocol = payload.get("protocol", {})
    return (
        protocol.get("n_records") == N_RECORDS
        and protocol.get("seeds") == FORMAL_SEEDS
        and protocol.get("steps") == FORMAL_STEPS
        and protocol.get("tail_window") == FORMAL_TAIL
        and protocol.get("tau") == FORMAL_TAU
        and protocol.get("verify_every") == FORMAL_VERIFY_EVERY
        and protocol.get("device") == FORMAL_DEVICE
        and protocol.get("acceptance_or_checkpoint_selection") is False
    )


def independent_audit(payload, *, input_paths=None):
    """只读取保存结果与公开输入，独立复算状态、随机量和聚合。"""
    failures = []
    target = np.asarray(payload.get("target", []), dtype=float)
    steps = int(payload["protocol"]["steps"])
    tail = int(payload["protocol"]["tail_window"])
    n_records = int(payload["protocol"]["n_records"])
    n_queries = len(target)
    public_inputs = None
    if payload.get("formal_protocol") and not _formal_payload_matches(payload):
        failures.append({"reason": "formal_protocol_metadata_mismatch"})
    if input_paths is not None:
        required_paths = {"schema", "queries", "marginals"}
        if set(input_paths) != required_paths:
            failures.append({"reason": "public_input_path_set_mismatch"})
        else:
            recomputed_hashes = {
                name: _sha256_file(path) for name, path in input_paths.items()
            }
            if recomputed_hashes != payload["public_input_sha256"]:
                failures.append({"reason": "public_input_hash_mismatch"})
            try:
                schema = load_schema(str(input_paths["schema"]))
                queries = load_queries(str(input_paths["queries"]))
                marginals = load_marginals(str(input_paths["marginals"]))
                input_target = np.asarray(
                    [query["result"] for query in queries], dtype=float
                )
                if not np.array_equal(input_target, target):
                    failures.append({"reason": "target_public_input_mismatch"})
                if marginals.get("n_records") != n_records:
                    failures.append({
                        "reason": "marginal_record_count_mismatch"
                    })
                public_inputs = (schema, queries, marginals)
                recomputed_oracle = run_exact_oracle()
                if json.dumps(
                    recomputed_oracle,
                    sort_keys=True,
                    separators=(",", ":"),
                ) != json.dumps(
                    payload["exact_oracle"],
                    sort_keys=True,
                    separators=(",", ":"),
                ):
                    failures.append({"reason": "exact_oracle_mismatch"})
            except Exception as error:
                failures.append({
                    "reason": "public_input_load_failure",
                    "error_type": type(error).__name__,
                })

    for run in payload["runs"]:
        if (
            int(run["n_steps"]) != steps
            or int(run["tail_window"]) != tail
            or not _float_close(run["tau"], payload["protocol"]["tau"])
        ):
            failures.append({
                "seed": run.get("seed"),
                "reason": "run_protocol_metadata_mismatch",
            })
        replayed_final_states = {}
        if public_inputs is not None:
            schema, queries, marginals = public_inputs
            try:
                seed = int(run["seed"])
                initial_table = init_synthetic_table(
                    n_records,
                    schema,
                    np.random.default_rng(seed),
                    marginals=marginals,
                )
                initial_state = initialize_persistent_heatbath_state(
                    initial_table, schema, queries, target
                )
                initial_scale = initial_gain_rms_scale(
                    initial_state, schema, queries, target
                )
                expected_beta = (
                    float(run["tau"]) / initial_scale["scale"]
                    if initial_scale["scale"] > 0.0 else 0.0
                )
                initial_answers = [
                    int(value) for value in initial_state.query_answers
                ]
                if (
                    _frame_sha256(initial_table)
                    != run["initial_table_sha256"]
                    or initial_answers != run["initial_query_answers"]
                    or not _float_close(initial_state.loss, run["initial_loss"])
                    or initial_scale != run["initial_gain_rms_scale"]
                    or not _float_close(
                        expected_beta,
                        run["candidate_inverse_temperature"],
                    )
                    or not _float_close(
                        run["baseline_inverse_temperature"], 0.0
                    )
                ):
                    failures.append({
                        "seed": seed,
                        "reason": "regenerated_initial_state_mismatch",
                    })

                expected_pair_seed = _address_seed(seed)
                replay_rng = np.random.default_rng(expected_pair_seed)
                replay_digest = hashlib.sha256()
                random_schedule = []
                coordinates = run["coordinate_history"]
                if (
                    int(run["pair_rng_seed"]) != expected_pair_seed
                    or _rng_state_sha256(replay_rng)
                    != run["pair_rng_initial_state_sha256"]
                    or len(coordinates) != steps
                ):
                    failures.append({
                        "seed": seed,
                        "reason": "random_schedule_initial_state_mismatch",
                    })
                else:
                    random_schedule_matches = True
                    n_coordinates = n_records * schema.n_blocks()
                    for recorded_coordinate in coordinates:
                        coordinate = int(replay_rng.integers(
                            0, n_coordinates
                        ))
                        attribute = coordinate % schema.n_blocks()
                        domain_size = len(legal_attribute_values(
                            schema.attributes[attribute]
                        ))
                        gumbels = replay_rng.gumbel(size=domain_size)
                        random_schedule.append((coordinate, gumbels.copy()))
                        random_schedule_matches = (
                            random_schedule_matches
                            and coordinate == int(recorded_coordinate)
                        )
                        replay_digest.update(
                            np.asarray(
                                [coordinate], dtype=np.int64
                            ).tobytes()
                        )
                        replay_digest.update(
                            np.ascontiguousarray(
                                gumbels, dtype=np.float64
                            ).tobytes()
                        )
                    final_rng_sha256 = _rng_state_sha256(replay_rng)
                    if (
                        not random_schedule_matches
                        or replay_digest.hexdigest()
                        != run["random_input_sha256"]
                        or final_rng_sha256
                        != run["baseline_rng_final_state_sha256"]
                        or final_rng_sha256
                        != run["candidate_rng_final_state_sha256"]
                    ):
                        failures.append({
                            "seed": seed,
                            "reason": "random_schedule_replay_mismatch",
                        })

                    verify_every = int(payload["protocol"]["verify_every"])
                    expected_checkpoints = [0]
                    expected_checkpoints.extend(range(
                        verify_every, steps + 1, verify_every
                    ))
                    if expected_checkpoints[-1] != steps:
                        expected_checkpoints.append(steps)
                    for variant, beta in (
                        ("baseline", 0.0),
                        ("candidate", expected_beta),
                    ):
                        trajectory = run[variant]
                        checkpoints = {
                            int(checkpoint["step"]): checkpoint
                            for checkpoint in trajectory[
                                "full_state_audits"
                            ]
                        }
                        recorded_checkpoint_steps = [
                            int(checkpoint["step"])
                            for checkpoint in trajectory[
                                "full_state_audits"
                            ]
                        ]
                        if recorded_checkpoint_steps != expected_checkpoints:
                            failures.append({
                                "seed": seed,
                                "variant": variant,
                                "reason": "checkpoint_schedule_mismatch",
                            })
                        state = initial_state.copy()
                        initial_checkpoint = checkpoints.get(0, {})
                        if (
                            initial_checkpoint.get("table_sha256")
                            != _frame_sha256(state.table)
                        ):
                            failures.append({
                                "seed": seed,
                                "variant": variant,
                                "reason": "initial_table_checkpoint_mismatch",
                            })
                        audit_rng = np.random.default_rng(0)
                        for step_index, (
                            coordinate, gumbels
                        ) in enumerate(random_schedule):
                            diagnostics = persistent_heatbath_step(
                                state,
                                schema,
                                queries,
                                target,
                                inverse_temperature=beta,
                                rng=audit_rng,
                                coordinate_index=coordinate,
                                gumbels=gumbels,
                            )
                            mismatch = _replay_step_mismatch(
                                trajectory, step_index, diagnostics
                            )
                            if mismatch is not None:
                                failures.append({
                                    "seed": seed,
                                    "variant": variant,
                                    "step": step_index + 1,
                                    "reason": "transition_replay_mismatch",
                                    "field": mismatch,
                                })
                                break
                            step_number = step_index + 1
                            if step_number in checkpoints:
                                checkpoint = checkpoints[step_number]
                                if (
                                    checkpoint["table_sha256"]
                                    != _frame_sha256(state.table)
                                    or checkpoint["query_answers"]
                                    != [
                                        int(value)
                                        for value in state.query_answers
                                    ]
                                    or not _float_close(
                                        checkpoint["recorded_loss"],
                                        state.loss,
                                    )
                                ):
                                    failures.append({
                                        "seed": seed,
                                        "variant": variant,
                                        "step": step_number,
                                        "reason": (
                                            "replayed_checkpoint_mismatch"
                                        ),
                                    })
                                    break
                        replayed_final_states[variant] = state
            except Exception as error:
                failures.append({
                    "seed": run.get("seed"),
                    "reason": "public_initial_or_random_audit_failure",
                    "error_type": type(error).__name__,
                })

        for variant in ("baseline", "candidate"):
            trajectory = run[variant]
            query_answers = np.asarray(
                trajectory["initial_query_answers"], dtype=np.int64
            )
            losses = trajectory["loss_history"]
            gains = trajectory["gain_history"]
            sparse_history = trajectory["query_delta_sparse_history"]
            per_step_histories = (
                "gain_history",
                "changed_history",
                "choice_index_history",
                "query_delta_sparse_history",
                "conditional_expected_state_gain_history",
                "expected_gain_over_reference_history",
                "conditional_normalized_entropy_history",
                "uphill_probability_mass_history",
                "probability_min_history",
                "probability_max_history",
                "candidate_state_evaluations_history",
                "query_indicator_evaluations_history",
            )
            if not (
                len(losses) == steps + 1
                and all(
                    len(trajectory[name]) == steps
                    for name in per_step_histories
                )
                and query_answers.shape == (n_queries,)
            ):
                failures.append({
                    "seed": run["seed"],
                    "variant": variant,
                    "reason": "history_length_or_shape",
                })
                continue
            checkpoints = {
                int(audit["step"]): audit
                for audit in trajectory["full_state_audits"]
            }
            initial_loss = compute_loss(target, query_answers)
            if (
                0 not in checkpoints
                or checkpoints[0]["query_answers"]
                != [int(value) for value in query_answers]
                or not _float_close(losses[0], initial_loss)
                or not _float_close(
                    checkpoints[0]["recorded_loss"], initial_loss
                )
            ):
                failures.append({
                    "seed": run["seed"],
                    "variant": variant,
                    "reason": "initial_checkpoint_mismatch",
                })
            for step, sparse in enumerate(sparse_history, start=1):
                seen = set()
                for index, delta in sparse:
                    index, delta = int(index), int(delta)
                    if (
                        not 0 <= index < n_queries
                        or index in seen
                        or delta not in (-1, 1)
                    ):
                        failures.append({
                            "seed": run["seed"],
                            "variant": variant,
                            "step": step,
                            "reason": "invalid_sparse_query_delta",
                        })
                        break
                    seen.add(index)
                    query_answers[index] += delta
                if np.any(query_answers < 0) or np.any(
                    query_answers > payload["protocol"]["n_records"]
                ):
                    failures.append({
                        "seed": run["seed"],
                        "variant": variant,
                        "step": step,
                        "reason": "query_answer_out_of_range",
                    })
                    break
                recomputed_loss = compute_loss(target, query_answers)
                if not _float_close(recomputed_loss, losses[step]):
                    failures.append({
                        "seed": run["seed"],
                        "variant": variant,
                        "step": step,
                        "reason": "recomputed_loss_mismatch",
                    })
                    break
                if not _float_close(
                    gains[step - 1], losses[step - 1] - losses[step]
                ):
                    failures.append({
                        "seed": run["seed"],
                        "variant": variant,
                        "step": step,
                        "reason": "gain_identity_mismatch",
                    })
                    break
                if step in checkpoints:
                    checkpoint = checkpoints[step]
                    if (
                        checkpoint["query_answers"]
                        != [int(value) for value in query_answers]
                        or not _float_close(
                            checkpoint["recorded_loss"], recomputed_loss
                        )
                        or checkpoint["query_answer_max_abs_error"] != 0
                        or checkpoint["loss_abs_error"] > IDENTITY_TOLERANCE
                    ):
                        failures.append({
                            "seed": run["seed"],
                            "variant": variant,
                            "step": step,
                            "reason": "full_checkpoint_mismatch",
                        })
                        break
            recomputed_metrics = _trajectory_metrics(trajectory, tail)
            for name, value in recomputed_metrics.items():
                recorded = trajectory["metrics"][name]
                if isinstance(value, int):
                    matches = int(recorded) == value
                else:
                    matches = _float_close(recorded, value)
                if not matches:
                    failures.append({
                        "seed": run["seed"],
                        "variant": variant,
                        "reason": f"metric_mismatch:{name}",
                    })
                    break
            if [int(value) for value in query_answers] != trajectory[
                "final_query_answers"
            ]:
                failures.append({
                    "seed": run["seed"],
                    "variant": variant,
                    "reason": "final_query_answers_mismatch",
                })

            if public_inputs is not None:
                schema, queries, _ = public_inputs
                try:
                    names = schema.attribute_names()
                    records = trajectory["final_table_records"]
                    if (
                        not isinstance(records, list)
                        or len(records) != n_records
                        or any(
                            not isinstance(record, dict)
                            or set(record) != set(names)
                            for record in records
                        )
                    ):
                        raise ValueError("最终表记录结构不匹配")
                    final_table = pd.DataFrame(records, columns=names)
                    final_state = initialize_persistent_heatbath_state(
                        final_table, schema, queries, target
                    )
                    final_answers = [
                        int(value) for value in final_state.query_answers
                    ]
                    final_checkpoint = trajectory["full_state_audits"][-1]
                    replayed_state = replayed_final_states.get(variant)
                    if (
                        _frame_sha256(final_state.table)
                        != trajectory["final_table_sha256"]
                        or final_answers != trajectory["final_query_answers"]
                        or not _float_close(
                            final_state.loss, trajectory["loss_history"][-1]
                        )
                        or int(final_checkpoint["step"]) != steps
                        or final_checkpoint["table_sha256"]
                        != trajectory["final_table_sha256"]
                        or replayed_state is None
                        or _frame_sha256(replayed_state.table)
                        != trajectory["final_table_sha256"]
                    ):
                        failures.append({
                            "seed": run["seed"],
                            "variant": variant,
                            "reason": "reconstructed_final_table_mismatch",
                        })
                except Exception as error:
                    failures.append({
                        "seed": run.get("seed"),
                        "variant": variant,
                        "reason": "final_table_audit_failure",
                        "error_type": type(error).__name__,
                    })

    recomputed_aggregate = aggregate_results(
        payload["runs"],
        payload["exact_oracle"],
        payload["protocol"]["seeds"],
        steps,
        classify=payload.get("formal_protocol", True),
    )
    if json.dumps(
        recomputed_aggregate, sort_keys=True, separators=(",", ":")
    ) != json.dumps(
        payload["aggregate"], sort_keys=True, separators=(",", ":")
    ):
        failures.append({"reason": "aggregate_mismatch"})

    return {
        "passed": not failures,
        "checked_seed_trajectories": int(2 * len(payload["runs"])),
        "checked_transitions": int(2 * len(payload["runs"]) * steps),
        "checked_public_initial_states": int(
            len(payload["runs"]) if public_inputs is not None else 0
        ),
        "checked_random_schedules": int(
            len(payload["runs"]) if public_inputs is not None else 0
        ),
        "checked_final_tables": int(
            2 * len(payload["runs"]) if public_inputs is not None else 0
        ),
        "checked_exact_oracle": bool(input_paths is not None),
        "failures": failures,
    }


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


def _formal_protocol_matches(args):
    return (
        args.seeds == FORMAL_SEEDS
        and args.steps == FORMAL_STEPS
        and args.tail == FORMAL_TAIL
        and args.tau == FORMAL_TAU
        and args.verify_every == FORMAL_VERIFY_EVERY
        and args.device == FORMAL_DEVICE
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-existing",
        type=Path,
        help="只读重审一个已有 JSON，不运行实验或覆盖输出",
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

    if args.audit_existing is not None:
        payload = _load_json_strict(args.audit_existing)
        input_paths = {
            "schema": SCHEMA_PATH,
            "queries": QUERY_PATH,
            "marginals": MARGINALS_PATH,
        }
        audit = independent_audit(payload, input_paths=input_paths)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"input={args.audit_existing}")
        print(f"sha256={_sha256_file(args.audit_existing)}")
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
    environment = _environment_snapshot(args.device)
    if formal and not environment[
        "git_worktree_clean_including_untracked"
    ]:
        raise RuntimeError("正式协议要求当前提交对应的工作树完全干净")

    schema = load_schema(str(SCHEMA_PATH))
    queries = load_queries(str(QUERY_PATH))
    target = np.asarray([query["result"] for query in queries], dtype=float)
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

    input_paths = {
        "schema": SCHEMA_PATH,
        "queries": QUERY_PATH,
        "marginals": MARGINALS_PATH,
    }
    exact_oracle = run_exact_oracle()
    if not exact_oracle["passed"]:
        raise RuntimeError("16 状态精确 oracle 未通过")
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
            f"[{index}/{len(args.seeds)}] seed={seed} "
            f"tail {run['baseline']['metrics']['tail_mean_loss']:.4f} -> "
            f"{run['candidate']['metrics']['tail_mean_loss']:.4f} "
            f"({run['elapsed_sec']:.2f}s)",
            flush=True,
        )

    aggregate = aggregate_results(
        runs,
        exact_oracle,
        args.seeds,
        args.steps,
        classify=formal,
    )
    payload = {
        "experiment": "persistent_workload_heatbath",
        "research_boundary": (
            "operator smoke against beta=0 reference; not a comparison with "
            "the full generator and not a DP claim"
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
            "baseline": "beta=0 uniform legal-value resampling",
            "candidate": "beta=tau/initial_nonzero_gain_over_n_rms",
            "output_state": "current table after the final microstep",
            "acceptance_or_checkpoint_selection": False,
        },
        "environment": environment,
        "public_input_sha256": {
            name: _sha256_file(path) for name, path in input_paths.items()
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
    _write_json_atomic(
        output,
        payload,
        overwrite=args.overwrite and not formal,
    )
    print(f"classification={aggregate['classification']}")
    print(f"output={output}")
    print(f"sha256={_sha256_file(output)}")


if __name__ == "__main__":
    main()
