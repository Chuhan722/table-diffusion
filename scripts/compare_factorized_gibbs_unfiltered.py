"""关闭整代接受，比较独立核与低阶因子 Gibbs 核的长期动力学。

两侧使用相同温度、初始化、donor 机制、轮数和主随机流。candidate 只增加固定数量
的随机扫描 Gibbs sweep；额外 Gibbs 随机量来自独立流，不会错位后续 donor、独立
mask 初值或 mutation。每个原始 proposal 无条件成为下一状态，主终点是最终当前表。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from table_diffevo.directional_diffusion import (
    bernoulli_entropy,
    compute_copy_direction_scores,
    direction_rms_scale,
    tilted_copy_probabilities,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.factorized_diffusion import (
    DEFAULT_LOGIT_CLIP,
    compile_mask_workload,
    evolve_step_factorized_gibbs,
)
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
N_RECORDS = 300
RHO = 0.01
ETA = 0.5
MU = 0.01
GIBBS_LOGIT_CLIP = DEFAULT_LOGIT_CLIP
CURRENT_SNAPSHOT_FORMAT = "issue49_unfiltered_current_v1"


def _git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _environment(device):
    result = {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device in ("cuda", "cpu"):
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda_runtime"] = torch.version.cuda
        if device == "cuda":
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            result["cuda_device_capability"] = list(
                torch.cuda.get_device_capability(0)
            )
    return result


def _gibbs_seed(seed):
    sequence = np.random.SeedSequence([int(seed), 0x4749424253])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _rng_state_sha256(rng):
    serialized = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _frame_sha256(frame):
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


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


def _donor_alpha(round_index, rounds):
    progress = round_index / (rounds - 1) if rounds > 1 else 1.0
    return 2.0 + 8.0 * progress


def _empty_independent_direction_diagnostics():
    return {
        "condition_count": 0,
        "raw_logit_min": None,
        "raw_logit_max": None,
        "raw_logit_abs_max": 0.0,
        "raw_logit_abs_max_condition": None,
        "clip_hit_count": 0,
        "clip_hit_conditions": [],
        "conditional_probability_min": None,
        "conditional_probability_max": None,
        "minimum_binary_outcome_probability": None,
        "conditional_entropy_sum": 0.0,
        "negative_direction_count": 0,
        "negative_direction_probability_sum": 0.0,
        "positive_direction_count": 0,
        "positive_direction_probability_sum": 0.0,
        "all_finite": True,
        "all_conditionals_bidirectional": True,
    }


def _accumulate_independent_direction_diagnostics(
    accumulator,
    directions,
    differs,
    strength,
    *,
    round_index,
    attribute_names,
    eta=ETA,
    logit_clip=GIBBS_LOGIT_CLIP,
):
    direction_values = np.asarray(directions, dtype=float)
    active = np.asarray(differs, dtype=bool)
    if direction_values.shape != active.shape:
        raise ValueError("方向量与 recipient/donor 差异 mask 形状不一致")
    if not np.all(np.isfinite(direction_values)):
        raise ValueError("独立方向量必须全部有限")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("独立方向核强度必须是非负有限值")
    if not np.any(active):
        return

    base_logit = float(np.log(eta) - np.log1p(-eta))
    with np.errstate(over="ignore", invalid="ignore"):
        raw_matrix = base_logit + float(strength) * direction_values
    raw = raw_matrix[active]
    if not np.all(np.isfinite(raw)):
        accumulator["all_finite"] = False
        raise ValueError("独立方向核原始 logit 超出 float64 可表示范围")
    probabilities = tilted_copy_probabilities(
        eta, direction_values[active], float(strength)
    )
    if not np.all(np.isfinite(probabilities)):
        accumulator["all_finite"] = False
        raise ValueError("独立方向核条件概率不是有限值")

    count = int(raw.size)
    minimum = float(raw.min())
    maximum = float(raw.max())
    absolute = np.abs(raw)
    maximum_flat_index = int(np.argmax(absolute))
    active_indices = np.argwhere(active)
    maximum_row, maximum_attribute = active_indices[maximum_flat_index]
    maximum_absolute = float(absolute[maximum_flat_index])
    if (
        accumulator["raw_logit_abs_max_condition"] is None
        or maximum_absolute > accumulator["raw_logit_abs_max"]
    ):
        accumulator["raw_logit_abs_max"] = maximum_absolute
        accumulator["raw_logit_abs_max_condition"] = {
            "round": int(round_index),
            "row": int(maximum_row),
            "attribute_index": int(maximum_attribute),
            "attribute": attribute_names[int(maximum_attribute)],
            "direction": float(direction_values[
                maximum_row, maximum_attribute
            ]),
            "raw_logit": float(raw_matrix[
                maximum_row, maximum_attribute
            ]),
        }

    accumulator["condition_count"] += count
    accumulator["raw_logit_min"] = (
        minimum
        if accumulator["raw_logit_min"] is None
        else min(accumulator["raw_logit_min"], minimum)
    )
    accumulator["raw_logit_max"] = (
        maximum
        if accumulator["raw_logit_max"] is None
        else max(accumulator["raw_logit_max"], maximum)
    )
    probability_minimum = float(probabilities.min())
    probability_maximum = float(probabilities.max())
    minimum_outcome = float(np.min(np.minimum(
        probabilities, 1.0 - probabilities
    )))
    accumulator["conditional_probability_min"] = (
        probability_minimum
        if accumulator["conditional_probability_min"] is None
        else min(
            accumulator["conditional_probability_min"],
            probability_minimum,
        )
    )
    accumulator["conditional_probability_max"] = (
        probability_maximum
        if accumulator["conditional_probability_max"] is None
        else max(
            accumulator["conditional_probability_max"],
            probability_maximum,
        )
    )
    accumulator["minimum_binary_outcome_probability"] = (
        minimum_outcome
        if accumulator["minimum_binary_outcome_probability"] is None
        else min(
            accumulator["minimum_binary_outcome_probability"],
            minimum_outcome,
        )
    )
    accumulator["conditional_entropy_sum"] += float(
        bernoulli_entropy(probabilities).sum()
    )
    interior = (probabilities > 0.0) & (probabilities < 1.0)
    accumulator["all_conditionals_bidirectional"] &= bool(
        np.all(interior)
    )

    active_directions = direction_values[active]
    negative = active_directions < 0.0
    positive = active_directions > 0.0
    accumulator["negative_direction_count"] += int(negative.sum())
    accumulator["negative_direction_probability_sum"] += float(
        probabilities[negative].sum()
    )
    accumulator["positive_direction_count"] += int(positive.sum())
    accumulator["positive_direction_probability_sum"] += float(
        probabilities[positive].sum()
    )

    hit_indices = np.argwhere(active & (np.abs(raw_matrix) >= logit_clip))
    accumulator["clip_hit_count"] += int(len(hit_indices))
    for row_index, attribute_index in hit_indices:
        accumulator["clip_hit_conditions"].append({
            "round": int(round_index),
            "row": int(row_index),
            "attribute_index": int(attribute_index),
            "attribute": attribute_names[int(attribute_index)],
            "direction": float(direction_values[
                row_index, attribute_index
            ]),
            "raw_logit": float(raw_matrix[row_index, attribute_index]),
        })


def _finalize_independent_direction_diagnostics(
    accumulator, *, logit_clip=GIBBS_LOGIT_CLIP
):
    count = int(accumulator["condition_count"])
    negative_count = int(accumulator["negative_direction_count"])
    positive_count = int(accumulator["positive_direction_count"])
    hits = int(accumulator["clip_hit_count"])
    return {
        "condition_count": count,
        "raw_logit_min": accumulator["raw_logit_min"],
        "raw_logit_max": accumulator["raw_logit_max"],
        "raw_logit_abs_max": float(accumulator["raw_logit_abs_max"]),
        "raw_logit_abs_max_condition": accumulator[
            "raw_logit_abs_max_condition"
        ],
        "logit_clip": float(logit_clip),
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "clip_hit_conditions": list(accumulator["clip_hit_conditions"]),
        "raw_logit_strictly_inside_clip": bool(
            count == 0 or accumulator["raw_logit_abs_max"] < logit_clip
        ),
        "conditional_probability_min": accumulator[
            "conditional_probability_min"
        ],
        "conditional_probability_max": accumulator[
            "conditional_probability_max"
        ],
        "minimum_binary_outcome_probability": accumulator[
            "minimum_binary_outcome_probability"
        ],
        "conditional_entropy_mean": (
            float(accumulator["conditional_entropy_sum"] / count)
            if count else None
        ),
        "negative_direction_count": negative_count,
        "negative_direction_copy_probability": (
            float(
                accumulator["negative_direction_probability_sum"]
                / negative_count
            ) if negative_count else None
        ),
        "positive_direction_count": positive_count,
        "positive_direction_copy_probability": (
            float(
                accumulator["positive_direction_probability_sum"]
                / positive_count
            ) if positive_count else None
        ),
        "all_finite": bool(accumulator["all_finite"]),
        "all_conditionals_bidirectional": bool(
            accumulator["all_conditionals_bidirectional"]
        ),
    }


def _empty_factor_conditional_diagnostics():
    return {
        "condition_count": 0,
        "raw_logit_min": None,
        "raw_logit_max": None,
        "raw_logit_abs_max": 0.0,
        "raw_logit_abs_max_condition": None,
        "clip_hit_count": 0,
        "clip_hit_conditions": [],
        "conditional_probability_min": None,
        "conditional_probability_max": None,
        "minimum_binary_outcome_probability": None,
        "conditional_entropy_sum": 0.0,
        "all_finite": True,
        "all_conditionals_bidirectional": True,
    }


def _accumulate_factor_conditional_diagnostics(
    accumulator, update, *, round_index, logit_clip=GIBBS_LOGIT_CLIP
):
    count = int(update["condition_count"])
    if update["logit_clip"] != float(logit_clip):
        raise RuntimeError("实际 Gibbs 条件诊断的 clip 与轨迹协议不一致")
    if count == 0:
        return
    if (
        len(update["clip_hit_conditions"])
        != update["clip_hit_count"]
        or update["conditional_entropy_mean"] is None
    ):
        raise RuntimeError("实际 Gibbs 条件诊断内部不完整")
    maximum_context = {
        "round": int(round_index),
        **update["raw_logit_abs_max_condition"],
    }
    if (
        accumulator["raw_logit_abs_max_condition"] is None
        or update["raw_logit_abs_max"]
        > accumulator["raw_logit_abs_max"]
    ):
        accumulator["raw_logit_abs_max"] = float(
            update["raw_logit_abs_max"]
        )
        accumulator["raw_logit_abs_max_condition"] = maximum_context
    accumulator["condition_count"] += count
    accumulator["raw_logit_min"] = (
        float(update["raw_logit_min"])
        if accumulator["raw_logit_min"] is None
        else min(accumulator["raw_logit_min"], update["raw_logit_min"])
    )
    accumulator["raw_logit_max"] = (
        float(update["raw_logit_max"])
        if accumulator["raw_logit_max"] is None
        else max(accumulator["raw_logit_max"], update["raw_logit_max"])
    )
    accumulator["clip_hit_count"] += int(update["clip_hit_count"])
    accumulator["clip_hit_conditions"].extend({
        "round": int(round_index),
        **condition,
    } for condition in update["clip_hit_conditions"])
    accumulator["conditional_probability_min"] = (
        float(update["conditional_probability_min"])
        if accumulator["conditional_probability_min"] is None
        else min(
            accumulator["conditional_probability_min"],
            update["conditional_probability_min"],
        )
    )
    accumulator["conditional_probability_max"] = (
        float(update["conditional_probability_max"])
        if accumulator["conditional_probability_max"] is None
        else max(
            accumulator["conditional_probability_max"],
            update["conditional_probability_max"],
        )
    )
    accumulator["minimum_binary_outcome_probability"] = (
        float(update["minimum_binary_outcome_probability"])
        if accumulator["minimum_binary_outcome_probability"] is None
        else min(
            accumulator["minimum_binary_outcome_probability"],
            update["minimum_binary_outcome_probability"],
        )
    )
    accumulator["conditional_entropy_sum"] += float(
        update["conditional_entropy_sum"]
    )
    accumulator["all_finite"] &= bool(update["all_finite"])
    accumulator["all_conditionals_bidirectional"] &= bool(
        update["all_conditionals_bidirectional"]
    )


def _finalize_factor_conditional_diagnostics(
    accumulator, *, logit_clip=GIBBS_LOGIT_CLIP
):
    count = int(accumulator["condition_count"])
    hits = int(accumulator["clip_hit_count"])
    return {
        "condition_count": count,
        "raw_logit_min": accumulator["raw_logit_min"],
        "raw_logit_max": accumulator["raw_logit_max"],
        "raw_logit_abs_max": float(accumulator["raw_logit_abs_max"]),
        "raw_logit_abs_max_condition": accumulator[
            "raw_logit_abs_max_condition"
        ],
        "logit_clip": float(logit_clip),
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "clip_hit_conditions": list(accumulator["clip_hit_conditions"]),
        "raw_logit_strictly_inside_clip": bool(
            count == 0 or accumulator["raw_logit_abs_max"] < logit_clip
        ),
        "conditional_probability_min": accumulator[
            "conditional_probability_min"
        ],
        "conditional_probability_max": accumulator[
            "conditional_probability_max"
        ],
        "minimum_binary_outcome_probability": accumulator[
            "minimum_binary_outcome_probability"
        ],
        "conditional_entropy_mean": (
            float(accumulator["conditional_entropy_sum"] / count)
            if count else None
        ),
        "all_finite": bool(accumulator["all_finite"]),
        "all_conditionals_bidirectional": bool(
            accumulator["all_conditionals_bidirectional"]
        ),
    }


def _normalize_snapshot_rounds(snapshot_rounds, rounds):
    if snapshot_rounds is None:
        return None
    try:
        values = list(snapshot_rounds)
    except TypeError as exc:
        raise ValueError("snapshot_rounds 必须是轮数序列") from exc
    normalized = []
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("snapshot_rounds 必须只包含整数")
        normalized.append(int(value))
    if len(set(normalized)) != len(normalized):
        raise ValueError("snapshot_rounds 不得重复")
    if any(value < 0 or value > rounds for value in normalized):
        raise ValueError(f"snapshot_rounds 必须位于 0..{rounds}")
    return tuple(sorted(normalized))


def _capture_current_snapshot(
    state,
    rng,
    gibbs_rng,
    *,
    seed,
    state_round,
    rounds,
    current_loss,
    best_loss,
    temperature,
    sweeps,
):
    alpha_round = min(state_round, rounds - 1)
    return {
        "snapshot_format": CURRENT_SNAPSHOT_FORMAT,
        "source_seed": int(seed),
        "source_rounds": int(rounds),
        "state_round": int(state_round),
        "state_kind": "current",
        "source_temperature": float(temperature),
        "source_sweeps": int(sweeps),
        "donor_alpha": float(_donor_alpha(alpha_round, rounds)),
        "current_loss": float(current_loss),
        "best_loss_so_far_diagnostic_only": float(best_loss),
        "state_sha256": _frame_sha256(state),
        "primary_rng_state_sha256": _rng_state_sha256(rng),
        "gibbs_rng_state_sha256": (
            _rng_state_sha256(gibbs_rng)
            if gibbs_rng is not None else None
        ),
        "table_columns": list(state.columns),
        "table_records": _frame_records(state),
    }


def _run_one(
    target,
    queries,
    schema,
    marginals,
    *,
    seed,
    rounds,
    temperature,
    sweeps,
    device,
    factor_builder="legacy_rowwise",
    record_state_hashes=False,
    snapshot_rounds=None,
):
    if factor_builder not in ("legacy_rowwise", "compiled_batch"):
        raise ValueError(f"未知因子构造器：{factor_builder!r}")
    requested_snapshot_rounds = _normalize_snapshot_rounds(
        snapshot_rounds, rounds
    )
    snapshot_round_set = (
        set(requested_snapshot_rounds)
        if requested_snapshot_rounds is not None else None
    )
    rng = np.random.default_rng(seed)
    gibbs_rng = (
        np.random.default_rng(_gibbs_seed(seed)) if sweeps > 0 else None
    )
    state = init_synthetic_table(
        N_RECORDS, schema, rng, marginals=marginals
    )
    q, residual, fitness = evaluate_vectorized(
        state,
        queries,
        schema,
        target=target,
        n_records=N_RECORDS,
        batch_size=256,
        device=device,
        want_fitness=True,
        verbose=False,
    )
    initial_loss = float(compute_loss(target, q))
    initial_csv_sha256 = _frame_sha256(state)
    best_loss = initial_loss
    direction_reference_scale = None
    direction_reference_scale_round = None
    loss_history = []
    gain_history = []
    changed_cells_history = []
    unique_history = []
    factor_build_elapsed = 0.0
    gibbs_sample_elapsed = 0.0
    active_gibbs_rows = 0
    active_blocks = 0
    factor_count = 0
    factor_table_entries = 0
    gibbs_microsteps = 0
    factor_model_builds = 0
    condition_evaluation_batches = 0
    compiled_validation_elapsed = 0.0
    direction_elapsed = 0.0
    independent_direction_diagnostics = (
        _empty_independent_direction_diagnostics()
    )
    factor_conditional_diagnostics = (
        _empty_factor_conditional_diagnostics()
    )
    state_sha256_history = []
    trajectory_audit_elapsed = 0.0
    snapshot_capture_elapsed = 0.0
    state_snapshots = []
    if snapshot_round_set is not None and 0 in snapshot_round_set:
        snapshot_start = time.perf_counter()
        state_snapshots.append(_capture_current_snapshot(
            state,
            rng,
            gibbs_rng,
            seed=seed,
            state_round=0,
            rounds=rounds,
            current_loss=initial_loss,
            best_loss=best_loss,
            temperature=temperature,
            sweeps=sweeps,
        ))
        snapshot_capture_elapsed += time.perf_counter() - snapshot_start
    start = time.perf_counter()
    workload_compile_elapsed = 0.0
    compiled_workload = None
    if sweeps > 0 and factor_builder == "compiled_batch":
        compile_start = time.perf_counter()
        compiled_workload = compile_mask_workload(
            schema, queries, max_factor_order=3
        )
        workload_compile_elapsed = time.perf_counter() - compile_start

    for round_index in range(rounds):
        current_loss = float(compute_loss(target, q))
        loss_history.append(current_loss)
        unique_history.append(int(len(state.value_counts())))
        if np.all(residual == 0.0):
            break

        use_torch = device in ("cuda", "cpu")
        distances = pairwise_block_distance(
            state,
            state,
            schema,
            device=device,
            return_tensor=use_torch,
        )
        alpha = _donor_alpha(round_index, rounds)
        probabilities = compute_sampling_probs(
            fitness,
            distances,
            beta=1.0,
            h=0.8,
            device=device,
            distance_mode="geometric",
            lambda_param=0.5,
            alpha=alpha,
            delta=0.05,
            winsorize_quantiles=(0.01, 0.99),
            exclude_self=True,
        )
        donor_indices = sample_donors(probabilities, rng, device=device)
        donors = state.iloc[donor_indices].reset_index(drop=True)

        direction_start = time.perf_counter()
        directions = compute_copy_direction_scores(
            state,
            donors,
            schema,
            queries,
            residual,
            batch_size=256,
            device=device,
        )
        direction_elapsed += time.perf_counter() - direction_start
        differs = np.column_stack([
            state[attr].reset_index(drop=True).to_numpy()
            != donors[attr].to_numpy()
            for attr in schema.attribute_names()
        ])
        if direction_reference_scale is None:
            candidate_scale = direction_rms_scale(directions[differs])
            if candidate_scale > 0.0:
                direction_reference_scale = candidate_scale
                direction_reference_scale_round = round_index
        effective_strength = (
            temperature / direction_reference_scale
            if direction_reference_scale is not None else 0.0
        )
        _accumulate_independent_direction_diagnostics(
            independent_direction_diagnostics,
            directions,
            differs,
            effective_strength,
            round_index=round_index,
            attribute_names=schema.attribute_names(),
        )

        proposal, update_diagnostics = evolve_step_factorized_gibbs(
            state,
            donors,
            schema,
            queries,
            residual,
            rho=RHO,
            eta=ETA,
            mu=MU,
            copy_direction_scores=directions,
            copy_direction_strength=effective_strength,
            n_sweeps=sweeps,
            rng=rng,
            gibbs_rng=gibbs_rng,
            max_factor_order=3,
            gibbs_logit_clip=GIBBS_LOGIT_CLIP,
            compiled_workload=compiled_workload,
        )
        _accumulate_factor_conditional_diagnostics(
            factor_conditional_diagnostics,
            update_diagnostics["factor_conditional_logit_diagnostics"],
            round_index=round_index,
        )
        proposal_q, proposal_residual, proposal_fitness = evaluate_vectorized(
            proposal,
            queries,
            schema,
            target=target,
            n_records=N_RECORDS,
            batch_size=256,
            device=device,
            want_fitness=True,
            verbose=False,
        )
        proposal_loss = float(compute_loss(target, proposal_q))
        gain_history.append(current_loss - proposal_loss)
        changed_cells_history.append(int(
            (
                proposal.reset_index(drop=True)
                != state.reset_index(drop=True)
            ).to_numpy().sum()
        ))
        best_loss = min(best_loss, proposal_loss)
        factor_build_elapsed += update_diagnostics[
            "factor_build_elapsed_sec"
        ]
        gibbs_sample_elapsed += update_diagnostics[
            "gibbs_sample_elapsed_sec"
        ]
        active_gibbs_rows += update_diagnostics["active_gibbs_rows"]
        active_blocks += update_diagnostics["active_blocks"]
        factor_count += update_diagnostics["factor_count"]
        factor_table_entries += update_diagnostics["factor_table_entries"]
        gibbs_microsteps += update_diagnostics["gibbs_microsteps"]
        factor_model_builds += update_diagnostics["factor_model_builds"]
        condition_evaluation_batches += update_diagnostics[
            "condition_evaluation_batches"
        ]
        compiled_validation_elapsed += update_diagnostics[
            "compiled_validation_elapsed_sec"
        ]

        # 核心条件：不检查 proposal_loss，不重试，不回滚，无条件进入下一状态。
        state = proposal
        q = proposal_q
        residual = proposal_residual
        fitness = proposal_fitness
        if record_state_hashes:
            audit_start = time.perf_counter()
            state_sha256_history.append(_frame_sha256(state))
            trajectory_audit_elapsed += time.perf_counter() - audit_start
        completed_round = round_index + 1
        if (
            snapshot_round_set is not None
            and completed_round in snapshot_round_set
        ):
            snapshot_start = time.perf_counter()
            state_snapshots.append(_capture_current_snapshot(
                state,
                rng,
                gibbs_rng,
                seed=seed,
                state_round=completed_round,
                rounds=rounds,
                current_loss=proposal_loss,
                best_loss=best_loss,
                temperature=temperature,
                sweeps=sweeps,
            ))
            snapshot_elapsed = time.perf_counter() - snapshot_start
            snapshot_capture_elapsed += snapshot_elapsed
            trajectory_audit_elapsed += snapshot_elapsed

    if requested_snapshot_rounds is not None:
        captured_rounds = {
            snapshot["state_round"] for snapshot in state_snapshots
        }
        missing_rounds = sorted(snapshot_round_set - captured_rounds)
        if missing_rounds:
            raise RuntimeError(
                "轨迹提前停止，未生成请求的 current-state 快照："
                f"{missing_rounds}"
            )
        fixed_scale = (
            float(direction_reference_scale)
            if direction_reference_scale is not None else None
        )
        for snapshot in state_snapshots:
            snapshot["direction_reference_scale"] = fixed_scale
            snapshot["direction_reference_scale_round"] = (
                int(direction_reference_scale_round)
                if direction_reference_scale_round is not None else None
            )

    final_loss = float(compute_loss(target, q))
    raw_elapsed = time.perf_counter() - start
    elapsed = raw_elapsed - trajectory_audit_elapsed
    gains = np.asarray(gain_history, dtype=float)
    losses = np.asarray(loss_history, dtype=float)
    positive_gains = gains[gains > 0.0]
    negative_gains = gains[gains < 0.0]
    completed_current_losses = np.asarray(
        (
            loss_history[1:len(gain_history)] + [final_loss]
            if gain_history else []
        ),
        dtype=float,
    )
    current_loss_path = np.asarray(
        [initial_loss, *completed_current_losses.tolist()], dtype=float
    )
    late_window = completed_current_losses[-250:]
    label = "independent" if sweeps == 0 else f"gibbs_{sweeps}_sweeps"
    result = {
        "seed": int(seed),
        "name": label,
        "factor_builder": (
            factor_builder if sweeps > 0 else "not_used"
        ),
        "temperature": float(temperature),
        "sweeps": int(sweeps),
        "rounds_run": len(gain_history),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss_diagnostic_only": best_loss,
        "final_change_pct": (final_loss / initial_loss - 1.0) * 100.0,
        "mean_raw_gain": float(gains.mean()) if len(gains) else 0.0,
        "mean_trajectory_loss": (
            float(losses.mean()) if len(losses) else final_loss
        ),
        "late_100_mean_loss": (
            float(losses[-100:].mean()) if len(losses) else final_loss
        ),
        "late_250_mean_loss": (
            float(losses[-250:].mean()) if len(losses) else final_loss
        ),
        "late_window_rounds": int(len(late_window)),
        "late_window_current_loss_mean": (
            float(late_window.mean()) if len(late_window) else final_loss
        ),
        "current_loss_auc": float(np.sum(
            0.5 * (current_loss_path[:-1] + current_loss_path[1:])
        )),
        "positive_gain_rate": (
            float(np.mean(gains > 0.0)) if len(gains) else 0.0
        ),
        "negative_gain_rate": (
            float(np.mean(gains < 0.0)) if len(gains) else 0.0
        ),
        "mean_positive_gain": (
            float(positive_gains.mean()) if len(positive_gains) else 0.0
        ),
        "mean_negative_gain": (
            float(negative_gains.mean()) if len(negative_gains) else 0.0
        ),
        "mean_changed_cells": (
            float(np.mean(changed_cells_history))
            if changed_cells_history else 0.0
        ),
        "maximum_loss": float(max(loss_history + [final_loss])),
        "final_unique_states": int(len(state.value_counts())),
        "mean_unique_states": float(np.mean(unique_history)),
        "direction_reference_scale": direction_reference_scale,
        "direction_reference_scale_round": direction_reference_scale_round,
        "independent_direction_diagnostics": (
            _finalize_independent_direction_diagnostics(
                independent_direction_diagnostics
            )
        ),
        "factor_conditional_logit_diagnostics": (
            _finalize_factor_conditional_diagnostics(
                factor_conditional_diagnostics
            )
        ),
        "direction_elapsed_sec": direction_elapsed,
        "factor_build_elapsed_sec": factor_build_elapsed,
        "compiled_validation_elapsed_sec": compiled_validation_elapsed,
        "workload_compile_elapsed_sec": workload_compile_elapsed,
        "factor_pipeline_elapsed_sec": (
            workload_compile_elapsed
            + compiled_validation_elapsed
            + factor_build_elapsed
        ),
        "gibbs_sample_elapsed_sec": gibbs_sample_elapsed,
        "active_gibbs_rows": active_gibbs_rows,
        "active_blocks": active_blocks,
        "factor_count": factor_count,
        "factor_table_entries": factor_table_entries,
        "gibbs_microsteps": gibbs_microsteps,
        "factor_model_builds": factor_model_builds,
        "condition_evaluation_batches": condition_evaluation_batches,
        "compiled_unique_conditions": (
            compiled_workload.n_unique_conditions
            if compiled_workload is not None else 0
        ),
        "elapsed_sec": elapsed,
        "raw_elapsed_sec": raw_elapsed,
        "trajectory_audit_elapsed_sec": trajectory_audit_elapsed,
        "primary_rng_state_sha256": _rng_state_sha256(rng),
        "gibbs_rng_state_sha256": (
            _rng_state_sha256(gibbs_rng) if gibbs_rng is not None else None
        ),
        "initial_csv_sha256": initial_csv_sha256,
        "final_csv_sha256": _frame_sha256(state),
        "state_sha256_history": state_sha256_history,
        "loss_history": loss_history,
        "current_loss_after_round_history": (
            completed_current_losses.tolist()
        ),
        "gain_history": gain_history,
        "changed_cells_history": changed_cells_history,
    }
    if requested_snapshot_rounds is not None:
        result.update({
            "snapshot_rounds": list(requested_snapshot_rounds),
            "snapshot_capture_elapsed_sec": snapshot_capture_elapsed,
            "state_snapshots": state_snapshots,
        })
    print(
        f"seed={seed:02d} {label:<16} "
        f"{result['factor_builder']:<16} "
        f"loss={initial_loss:.1f}->{final_loss:.1f} "
        f"({result['final_change_pct']:+.1f}%) "
        f"raw_pos={result['positive_gain_rate']:.1%} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _aggregate(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values.tolist(),
    }


def _paired(candidate, baseline, key, lower_is_better):
    candidate_values = np.asarray([row[key] for row in candidate], dtype=float)
    baseline_values = np.asarray([row[key] for row in baseline], dtype=float)
    difference = candidate_values - baseline_values
    if len(difference) < 2 or np.all(difference == 0.0):
        paired_t = None
        paired_p = None
    else:
        statistic, p_value = stats.ttest_rel(
            candidate_values, baseline_values
        )
        paired_t = float(statistic) if np.isfinite(statistic) else None
        paired_p = float(p_value) if np.isfinite(p_value) else None
    better = difference < 0.0 if lower_is_better else difference > 0.0
    worse = difference > 0.0 if lower_is_better else difference < 0.0
    return {
        "candidate_mean": float(candidate_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "mean_difference": float(difference.mean()),
        "paired_t": paired_t,
        "paired_p": paired_p,
        "wins": int(np.sum(better)),
        "ties": int(np.sum(difference == 0.0)),
        "losses": int(np.sum(worse)),
        "values": difference.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--factor-builder",
        choices=["legacy_rowwise", "compiled_batch"],
        default="legacy_rowwise",
    )
    parser.add_argument("--record-state-hashes", action="store_true")
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default="outputs/factorized_gibbs/unfiltered_tau2_sweep8.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds 必须为正整数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)
    runs = {"independent": [], f"gibbs_{args.sweeps}_sweeps": []}
    experiment_start = time.perf_counter()
    for seed in args.seeds:
        runs["independent"].append(_run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=0,
            device=args.device,
            factor_builder=args.factor_builder,
            record_state_hashes=args.record_state_hashes,
        ))
        runs[f"gibbs_{args.sweeps}_sweeps"].append(_run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=args.sweeps,
            device=args.device,
            factor_builder=args.factor_builder,
            record_state_hashes=args.record_state_hashes,
        ))

    metrics = (
        "initial_loss",
        "final_loss",
        "best_loss_diagnostic_only",
        "final_change_pct",
        "mean_raw_gain",
        "mean_trajectory_loss",
        "late_100_mean_loss",
        "late_250_mean_loss",
        "positive_gain_rate",
        "negative_gain_rate",
        "mean_positive_gain",
        "mean_negative_gain",
        "mean_changed_cells",
        "maximum_loss",
        "final_unique_states",
        "mean_unique_states",
        "direction_elapsed_sec",
        "factor_build_elapsed_sec",
        "compiled_validation_elapsed_sec",
        "workload_compile_elapsed_sec",
        "factor_pipeline_elapsed_sec",
        "gibbs_sample_elapsed_sec",
        "active_gibbs_rows",
        "active_blocks",
        "factor_count",
        "factor_table_entries",
        "gibbs_microsteps",
        "factor_model_builds",
        "condition_evaluation_batches",
        "elapsed_sec",
        "raw_elapsed_sec",
        "trajectory_audit_elapsed_sec",
    )
    aggregate = {
        name: {key: _aggregate(rows, key) for key in metrics}
        for name, rows in runs.items()
    }
    candidate_name = f"gibbs_{args.sweeps}_sweeps"
    comparisons = {
        "final_loss": _paired(
            runs[candidate_name], runs["independent"], "final_loss", True
        ),
        "mean_raw_gain": _paired(
            runs[candidate_name],
            runs["independent"],
            "mean_raw_gain",
            False,
        ),
        "mean_trajectory_loss": _paired(
            runs[candidate_name],
            runs["independent"],
            "mean_trajectory_loss",
            True,
        ),
        "late_100_mean_loss": _paired(
            runs[candidate_name],
            runs["independent"],
            "late_100_mean_loss",
            True,
        ),
        "late_250_mean_loss": _paired(
            runs[candidate_name],
            runs["independent"],
            "late_250_mean_loss",
            True,
        ),
        "positive_gain_rate": _paired(
            runs[candidate_name],
            runs["independent"],
            "positive_gain_rate",
            False,
        ),
        "final_unique_states": _paired(
            runs[candidate_name],
            runs["independent"],
            "final_unique_states",
            False,
        ),
    }
    primary_rng_aligned = all(
        baseline["primary_rng_state_sha256"]
        == candidate["primary_rng_state_sha256"]
        for baseline, candidate in zip(
            runs["independent"], runs[candidate_name]
        )
    )
    initial_loss_aligned = all(
        baseline["initial_loss"] == candidate["initial_loss"]
        for baseline, candidate in zip(
            runs["independent"], runs[candidate_name]
        )
    )
    direction_scale_aligned = all(
        baseline["direction_reference_scale"]
        == candidate["direction_reference_scale"]
        for baseline, candidate in zip(
            runs["independent"], runs[candidate_name]
        )
    )
    summary = {
        "experiment": "factorized_gibbs_unfiltered_dynamics",
        "scope": (
            "same_temperature_and_primary_rng_every_raw_proposal_becomes_"
            "next_state_no_loss_acceptance"
        ),
        "primary_endpoint": "final_current_loss_not_best_loss",
        "dataset": "test_300x10",
        "n_rounds": args.rounds,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "candidate_sweeps": args.sweeps,
        "factor_builder": args.factor_builder,
        "record_state_hashes": args.record_state_hashes,
        "rho": RHO,
        "eta": ETA,
        "mu": MU,
        "gibbs_logit_clip": GIBBS_LOGIT_CLIP,
        "device": args.device,
        "git_commit": _git_commit(),
        "command_argv": sys.argv,
        "environment": _environment(args.device),
        "primary_rng_aligned_all_seeds": primary_rng_aligned,
        "initial_loss_aligned_all_seeds": initial_loss_aligned,
        "direction_reference_scale_aligned_all_seeds": (
            direction_scale_aligned
        ),
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n===== 无整代接受的最终当前表 =====")
    for name in runs:
        final = aggregate[name]["final_loss"]
        positive = aggregate[name]["positive_gain_rate"]
        elapsed = aggregate[name]["elapsed_sec"]
        print(
            f"{name:<18} final_loss={final['mean']:.2f}±{final['std']:.2f} "
            f"raw_pos={positive['mean']:.1%} "
            f"elapsed={elapsed['mean']:.1f}s"
        )
    print(f"主随机流逐种子对齐：{primary_rng_aligned}")
    print(f"初始 loss 逐种子对齐：{initial_loss_aligned}")
    print(f"首轮方向尺度逐种子对齐：{direction_scale_aligned}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
