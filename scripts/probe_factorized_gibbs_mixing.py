"""冻结状态下精确测量低阶因子随机扫描 Gibbs 的混合与 proposal 质量。

候选从现有独立定向 mask 分布出发，执行固定数量的随机坐标 Gibbs 微步。小表实验
通过完整状态转移精确传播分布，因此混合指标没有 Monte Carlo 误差；完整 proposal
只在生成后离线评价，不执行整代接受、重试、变异或 best 选择。
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.factorized_diffusion import (
    DEFAULT_LOGIT_CLIP,
    build_sparse_mask_energy,
    evaluate_sparse_mask_energies,
    propagate_random_scan_distribution,
    sparse_single_directions,
)
from table_diffevo.generator import init_synthetic_table
from table_diffevo.joint_diffusion import (
    additive_mask_directions,
    baseline_mask_log_probabilities,
    compute_joint_mask_landscapes,
    gibbs_mask_log_probabilities,
)
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


def _frame_sha256(frame):
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json_strict(path):
    def reject_constant(value):
        raise ValueError(f"快照包含非标准数值常量：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _snapshot_integer(snapshot, key):
    value = snapshot[key]
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"快照字段 {key} 必须是整数")
    return int(value)


def _restore_current_snapshot(
    snapshot,
    target,
    queries,
    schema,
    *,
    device,
):
    if not isinstance(snapshot, dict):
        raise ValueError("外部快照必须是 JSON object")
    required = {
        "snapshot_format",
        "source_seed",
        "source_rounds",
        "state_round",
        "state_kind",
        "source_temperature",
        "source_sweeps",
        "donor_alpha",
        "current_loss",
        "state_sha256",
        "primary_rng_state_sha256",
        "gibbs_rng_state_sha256",
        "table_columns",
        "table_records",
        "direction_reference_scale",
        "direction_reference_scale_round",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise ValueError(f"外部快照缺少字段：{missing}")
    if snapshot["snapshot_format"] != CURRENT_SNAPSHOT_FORMAT:
        raise ValueError("外部快照格式版本不匹配")
    if snapshot["state_kind"] != "current":
        raise ValueError("外部快照必须保存 current state")

    source_seed = _snapshot_integer(snapshot, "source_seed")
    source_rounds = _snapshot_integer(snapshot, "source_rounds")
    state_round = _snapshot_integer(snapshot, "state_round")
    source_sweeps = _snapshot_integer(snapshot, "source_sweeps")
    scale_round = _snapshot_integer(
        snapshot, "direction_reference_scale_round"
    )
    if source_seed < 0 or source_rounds <= 0:
        raise ValueError("快照 source seed/rounds 无效")
    if not 0 <= state_round <= source_rounds:
        raise ValueError("快照 state_round 超出来源轨迹")
    if source_sweeps < 0 or scale_round < 0:
        raise ValueError("快照 sweeps 或 s0 发现轮次无效")

    source_temperature = float(snapshot["source_temperature"])
    probe_alpha = float(snapshot["donor_alpha"])
    recorded_loss = float(snapshot["current_loss"])
    reference_scale = float(snapshot["direction_reference_scale"])
    if (
        not np.isfinite(source_temperature)
        or source_temperature < 0.0
        or not np.isfinite(probe_alpha)
        or probe_alpha <= 0.0
        or not np.isfinite(recorded_loss)
        or recorded_loss < 0.0
        or not np.isfinite(reference_scale)
        or reference_scale <= 0.0
    ):
        raise ValueError("快照温度、alpha、loss 或 s0 无效")

    expected_columns = schema.attribute_names()
    if snapshot["table_columns"] != expected_columns:
        raise ValueError("快照表格列名或顺序与 schema 不一致")
    records = snapshot["table_records"]
    if not isinstance(records, list) or len(records) != N_RECORDS:
        raise ValueError(f"快照必须包含 {N_RECORDS} 条表格记录")
    if any(
        not isinstance(row, dict) or list(row) != expected_columns
        for row in records
    ):
        raise ValueError("快照记录的字段或顺序与 schema 不一致")
    state = pd.DataFrame(records, columns=expected_columns)

    recorded_hash = snapshot["state_sha256"]
    if not _is_sha256(recorded_hash):
        raise ValueError("快照 state_sha256 格式无效")
    actual_hash = _frame_sha256(state)
    if actual_hash != recorded_hash:
        raise ValueError("快照表格哈希核验失败")
    if not _is_sha256(snapshot["primary_rng_state_sha256"]):
        raise ValueError("快照主 RNG 哈希格式无效")
    gibbs_rng_hash = snapshot["gibbs_rng_state_sha256"]
    if gibbs_rng_hash is not None and not _is_sha256(gibbs_rng_hash):
        raise ValueError("快照 Gibbs RNG 哈希格式无效")

    q, _, _ = evaluate_vectorized(
        state,
        queries,
        schema,
        target=target,
        n_records=N_RECORDS,
        batch_size=256,
        device=device,
        want_fitness=False,
        verbose=False,
    )
    recomputed_loss = float(compute_loss(target, q))
    if abs(recomputed_loss - recorded_loss) > 1e-10:
        raise ValueError("快照 current loss 重新计算不一致")

    controls = {
        "snapshot_format": CURRENT_SNAPSHOT_FORMAT,
        "source_seed": source_seed,
        "source_rounds": source_rounds,
        "source_temperature": source_temperature,
        "source_sweeps": source_sweeps,
        "state_round": state_round,
        "state_sha256": actual_hash,
        "current_loss": recomputed_loss,
        "probe_alpha": probe_alpha,
        "direction_reference_scale": reference_scale,
        "direction_reference_scale_round": scale_round,
        "primary_rng_state_sha256": snapshot[
            "primary_rng_state_sha256"
        ],
        "gibbs_rng_state_sha256": gibbs_rng_hash,
    }
    return state, controls


def _load_current_snapshot(path, target, queries, schema, *, device):
    snapshot = _load_json_strict(path)
    return _restore_current_snapshot(
        snapshot, target, queries, schema, device=device
    )


def _tau_label(tau):
    return f"{tau:g}".replace(".", "p")


def _gibbs_name(tau, sweeps):
    return f"gibbs_tau_{_tau_label(tau)}_sweeps_{sweeps}"


def _joint_name(tau):
    return f"joint_tau_{_tau_label(tau)}"


def _config_names(temperatures, sweeps):
    names = []
    for tau in temperatures:
        names.extend(_gibbs_name(tau, sweep) for sweep in sweeps)
        names.append(_joint_name(tau))
    return names


def _address_seed(seed, state_index, proposal_index, stream):
    sequence = np.random.SeedSequence(
        [int(seed), int(state_index), int(proposal_index), int(stream)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _make_baseline_state(
    target, queries, schema, marginals, seed, rounds, device
):
    if rounds == 0:
        return init_synthetic_table(
            N_RECORDS,
            schema,
            np.random.default_rng(seed),
            marginals=marginals,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = run_evolution(
            target,
            queries,
            schema,
            n_records=N_RECORDS,
            n_rounds=rounds,
            seed=seed,
            beta=1.0,
            h=0.8,
            rho=RHO,
            eta=ETA,
            mu=0.01,
            device=device,
            eval_method="vectorized",
            batch_size=256,
            init_method="marginal",
            marginals=marginals,
            log_every=rounds + 1,
            distance_mode="geometric",
            lambda_param=0.5,
            alpha_min=2.0,
            alpha_max=10.0,
            delta=0.05,
            winsorize_quantiles=(0.01, 0.99),
            exclude_self=True,
            max_retries=0,
        )
    return state


def _empty_kernel_accumulator():
    return {
        "rows": 0,
        "active_blocks": 0,
        "tvd_to_joint_sum": 0.0,
        "kl_to_joint_sum": 0.0,
        "kl_to_reference_sum": 0.0,
        "entropy_sum": 0.0,
        "joint_entropy_sum": 0.0,
        "expected_direction_sum": 0.0,
        "joint_expected_direction_sum": 0.0,
        "absolute_expected_direction_gap_sum": 0.0,
        "negative_mass_sum": 0.0,
        "joint_negative_mass_sum": 0.0,
    }


def _empty_logit_accumulator():
    return {
        "condition_count": 0,
        "raw_logit_min": None,
        "raw_logit_max": None,
        "raw_logit_abs_max": 0.0,
        "clip_hit_count": 0,
        "conditional_probability_min": None,
        "conditional_probability_max": None,
        "minimum_binary_outcome_probability": None,
        "uniform_condition_entropy_sum": 0.0,
        "all_conditionals_bidirectional": True,
    }


def _raw_conditional_logits(masks, directions, eta, strength):
    checked_masks = np.asarray(masks)
    values = np.asarray(directions, dtype=float)
    if (
        checked_masks.ndim != 2
        or checked_masks.dtype.kind not in "biuf"
        or not np.all(np.isfinite(checked_masks))
        or np.any((checked_masks != 0) & (checked_masks != 1))
    ):
        raise ValueError("masks 必须是有限 0/1 二维数组")
    checked_masks = checked_masks.astype(bool, copy=False)
    if values.shape != (len(checked_masks),):
        raise ValueError("directions 必须与完整 mask 数量一致")
    if not np.all(np.isfinite(values)):
        raise ValueError("directions 必须全部有限")
    if (
        not np.isfinite(eta)
        or not 0.0 < eta < 1.0
        or not np.isfinite(strength)
        or strength < 0.0
    ):
        raise ValueError("eta/strength 必须是有效有限值")
    n_active = checked_masks.shape[1]
    if len(checked_masks) != 1 << n_active:
        raise ValueError("masks 必须完整枚举全部状态")
    if n_active == 0:
        return np.empty(0, dtype=float)

    base_logit = float(np.log(eta) - np.log1p(-eta))
    state_indices = np.arange(len(checked_masks), dtype=np.intp)
    logits = []
    for variable in range(n_active):
        lower = state_indices[~checked_masks[:, variable]]
        upper = lower | (1 << variable)
        with np.errstate(over="ignore", invalid="ignore"):
            variable_logits = base_logit + strength * (
                values[upper] - values[lower]
            )
        if not np.all(np.isfinite(variable_logits)):
            raise ValueError("未截断条件 logit 超出 float64 可表示范围")
        logits.append(variable_logits)
    return np.concatenate(logits)


def _accumulate_logit_diagnostics(accumulator, raw_logits, logit_clip):
    values = np.asarray(raw_logits, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("raw_logits 必须是有限一维数组")
    if (
        not np.isfinite(logit_clip)
        or logit_clip <= 0.0
    ):
        raise ValueError("logit_clip 必须是正有限值")
    if values.size == 0:
        return
    minimum = float(values.min())
    maximum = float(values.max())
    accumulator["condition_count"] += int(values.size)
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
    accumulator["raw_logit_abs_max"] = max(
        accumulator["raw_logit_abs_max"],
        float(np.max(np.abs(values))),
    )
    accumulator["clip_hit_count"] += int(np.sum(
        np.abs(values) > logit_clip
    ))
    effective_logits = np.clip(values, -logit_clip, logit_clip)
    probabilities = np.empty_like(effective_logits)
    positive = effective_logits >= 0.0
    probabilities[positive] = 1.0 / (
        1.0 + np.exp(-effective_logits[positive])
    )
    exponentials = np.exp(effective_logits[~positive])
    probabilities[~positive] = exponentials / (1.0 + exponentials)
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
    interior = (probabilities > 0.0) & (probabilities < 1.0)
    entropies = np.zeros_like(probabilities)
    entropies[interior] = -(
        probabilities[interior] * np.log(probabilities[interior])
        + (1.0 - probabilities[interior])
        * np.log1p(-probabilities[interior])
    )
    accumulator["uniform_condition_entropy_sum"] += float(
        entropies.sum()
    )
    accumulator["all_conditionals_bidirectional"] &= bool(
        np.all(interior)
    )


def _finalize_logit_diagnostics(accumulator, logit_clip):
    count = int(accumulator["condition_count"])
    hits = int(accumulator["clip_hit_count"])
    return {
        "condition_count": count,
        "raw_logit_min": accumulator["raw_logit_min"],
        "raw_logit_max": accumulator["raw_logit_max"],
        "raw_logit_abs_max": float(accumulator["raw_logit_abs_max"]),
        "logit_clip": float(logit_clip),
        "clip_hit_count": hits,
        "clip_hit_rate": float(hits / count) if count else 0.0,
        "raw_logit_strictly_inside_clip": bool(
            count == 0
            or accumulator["raw_logit_abs_max"] < logit_clip
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
        "uniform_condition_entropy_mean": (
            float(accumulator["uniform_condition_entropy_sum"] / count)
            if count else None
        ),
        "uniform_condition_entropy_maximum": float(np.log(2.0)),
        "all_conditionals_bidirectional": bool(
            accumulator["all_conditionals_bidirectional"]
        ),
    }


def _safe_kl(probabilities, reference):
    positive = probabilities > 0.0
    return float(np.dot(
        probabilities[positive],
        np.log(probabilities[positive]) - np.log(reference[positive]),
    ))


def _distribution_metrics(probabilities, joint, reference, directions):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    joint = np.asarray(joint, dtype=float)
    joint = joint / joint.sum()
    reference = np.asarray(reference, dtype=float)
    reference = reference / reference.sum()
    scale = max(1.0, float(np.max(np.abs(directions))))
    negative = directions < -1e-12 * scale
    positive = probabilities > 0.0
    joint_positive = joint > 0.0
    expected_direction = float(np.dot(probabilities, directions))
    joint_expected_direction = float(np.dot(joint, directions))
    return {
        "tvd_to_joint": float(0.5 * np.abs(probabilities - joint).sum()),
        "kl_to_joint": _safe_kl(probabilities, joint),
        "kl_to_reference": _safe_kl(probabilities, reference),
        "entropy": float(-np.dot(
            probabilities[positive], np.log(probabilities[positive])
        )),
        "joint_entropy": float(-np.dot(
            joint[joint_positive], np.log(joint[joint_positive])
        )),
        "expected_direction": expected_direction,
        "joint_expected_direction": joint_expected_direction,
        "absolute_expected_direction_gap": abs(
            joint_expected_direction - expected_direction
        ),
        "negative_mass": float(probabilities[negative].sum()),
        "joint_negative_mass": float(joint[negative].sum()),
    }


def _accumulate_kernel(accumulator, metrics, n_active):
    accumulator["rows"] += 1
    accumulator["active_blocks"] += int(n_active)
    for key, value in metrics.items():
        accumulator[f"{key}_sum"] += float(value)


def _finalize_kernel(accumulator):
    rows = accumulator["rows"]
    if rows == 0:
        return {
            "participating_active_rows": 0,
            "active_blocks": 0,
            "tvd_to_joint": 0.0,
            "kl_to_joint": 0.0,
            "kl_to_reference": 0.0,
            "entropy": 0.0,
            "joint_entropy": 0.0,
            "expected_direction": 0.0,
            "joint_expected_direction": 0.0,
            "absolute_expected_direction_gap": 0.0,
            "negative_mass": 0.0,
            "joint_negative_mass": 0.0,
        }
    result = {
        "participating_active_rows": rows,
        "active_blocks": accumulator["active_blocks"],
    }
    for key in (
        "tvd_to_joint",
        "kl_to_joint",
        "kl_to_reference",
        "entropy",
        "joint_entropy",
        "expected_direction",
        "joint_expected_direction",
        "absolute_expected_direction_gap",
        "negative_mass",
        "joint_negative_mass",
    ):
        result[key] = accumulator[f"{key}_sum"] / rows
    return result


def _apply_selected_mask(
    proposal, donors, row_index, active_attribute_indices, mask, attr_names
):
    for local_index, attr_index in enumerate(active_attribute_indices):
        if mask[local_index]:
            attr = attr_names[attr_index]
            proposal.at[row_index, attr] = donors.at[row_index, attr]


def _sample_index(probabilities, gumbels):
    probabilities = np.asarray(probabilities, dtype=float)
    scores = np.full_like(probabilities, -np.inf)
    positive = probabilities > 0.0
    scores[positive] = np.log(probabilities[positive])
    return int(np.argmax(scores + gumbels))


def _measure_proposal(state, proposal, q, loss, target, queries, schema, device):
    proposal_q, _, _ = evaluate_vectorized(
        proposal,
        queries,
        schema,
        batch_size=256,
        device=device,
        want_fitness=False,
        verbose=False,
    )
    proposal_loss = compute_loss(target, proposal_q)
    delta_q = proposal_q - q
    linear = float(np.dot(target - q, delta_q))
    quadratic = float(0.5 * np.dot(delta_q, delta_q))
    changed = proposal.reset_index(drop=True) != state.reset_index(drop=True)
    changed_cells = int(changed.to_numpy().sum())
    gain = float(loss - proposal_loss)
    return {
        "gain": gain,
        "linear_gain": linear,
        "quadratic_penalty": quadratic,
        "gain_per_changed_cell": (
            gain / changed_cells if changed_cells else 0.0
        ),
        "changed_cells": changed_cells,
        "changed_rows": int(changed.any(axis=1).sum()),
    }


def _summarize_proposals(rows):
    result = {"n": len(rows)}
    for key in (
        "gain",
        "linear_gain",
        "quadratic_penalty",
        "gain_per_changed_cell",
        "changed_cells",
        "changed_rows",
    ):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    result["positive_gain_rate"] = float(np.mean(gains > 0.0))
    result["zero_gain_rate"] = float(np.mean(gains == 0.0))
    result["negative_gain_rate"] = float(np.mean(gains < 0.0))
    return result


def _paired(candidate_rows, baseline_rows):
    differences = np.asarray([
        candidate["gain"] - baseline["gain"]
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ], dtype=float)
    return {
        "mean_gain_difference": float(differences.mean()),
        "std_gain_difference": (
            float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
        ),
        "median_gain_difference": float(np.median(differences)),
        "wins": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "losses": int(np.sum(differences < 0.0)),
        "values": differences.tolist(),
    }


def _probe_state(
    state,
    target,
    queries,
    schema,
    *,
    seed,
    state_index,
    state_rounds,
    temperatures,
    sweeps,
    proposals,
    device,
    max_active_attributes,
    external_snapshot_controls=None,
):
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
    loss = compute_loss(target, q)
    use_torch = device in ("cuda", "cpu")
    distances = pairwise_block_distance(
        state, state, schema, device=device, return_tensor=use_torch
    )
    if external_snapshot_controls is None:
        probe_alpha = 2.0 if state_rounds == 0 else 10.0
        direction_reference_scale = None
        reference_scale_proposal_index = None
    else:
        controls = external_snapshot_controls
        if (
            controls.get("source_seed") != seed
            or controls.get("state_round") != state_rounds
            or controls.get("state_sha256") != _frame_sha256(state)
            or abs(float(controls.get("current_loss", np.inf)) - loss)
            > 1e-10
        ):
            raise ValueError("外部快照身份与 probe state 不一致")
        probe_alpha = float(controls["probe_alpha"])
        direction_reference_scale = float(
            controls["direction_reference_scale"]
        )
        if (
            not np.isfinite(probe_alpha)
            or probe_alpha <= 0.0
            or not np.isfinite(direction_reference_scale)
            or direction_reference_scale <= 0.0
        ):
            raise ValueError("外部快照的 probe alpha 或 s0 无效")
        reference_scale_proposal_index = None
    sampling_probabilities = compute_sampling_probs(
        fitness,
        distances,
        beta=1.0,
        h=0.8,
        device=device,
        distance_mode="geometric",
        lambda_param=0.5,
        alpha=probe_alpha,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        exclude_self=True,
    )

    attr_names = schema.attribute_names()
    configs = _config_names(temperatures, sweeps)
    proposal_rows = {name: [] for name in configs}
    global_kernel = {
        name: _empty_kernel_accumulator() for name in configs
    }
    global_logit = {
        tau: _empty_logit_accumulator() for tau in temperatures
    }
    exact_energy_max_error = 0.0
    one_hot_max_error = 0.0
    factor_count_sum = 0
    factor_table_entries_sum = 0
    maximum_factor_order = 0
    active_factor_rows = 0
    tvd_snapshot_increase_max = 0.0
    factor_build_elapsed = 0.0
    exact_propagation_elapsed = 0.0
    probe_start = time.perf_counter()

    for proposal_index in range(proposals):
        donor_rng = np.random.default_rng(
            _address_seed(seed, state_index, proposal_index, 0)
        )
        donor_idx = sample_donors(
            sampling_probabilities, donor_rng, device=device
        )
        donors = state.iloc[donor_idx].reset_index(drop=True)
        directions = compute_copy_direction_scores(
            state,
            donors,
            schema,
            queries,
            residual,
            batch_size=256,
            device=device,
        )
        differs = np.column_stack([
            state[attr].reset_index(drop=True).to_numpy()
            != donors[attr].to_numpy()
            for attr in attr_names
        ])
        active_directions = directions[differs]
        if direction_reference_scale is None:
            candidate_scale = direction_rms_scale(active_directions)
            if candidate_scale > 0.0:
                direction_reference_scale = candidate_scale
                reference_scale_proposal_index = proposal_index

        participation_rng = np.random.default_rng(
            _address_seed(seed, state_index, proposal_index, 1)
        )
        participating_rows = np.flatnonzero(
            participation_rng.random(N_RECORDS) < RHO
        )
        landscapes = compute_joint_mask_landscapes(
            state,
            donors,
            participating_rows,
            schema,
            queries,
            residual,
            batch_size=256,
            device=device,
            max_active_attributes=max_active_attributes,
        )

        generated = {
            name: state.reset_index(drop=True).copy() for name in configs
        }
        proposal_kernel = {
            name: _empty_kernel_accumulator() for name in configs
        }
        gumbel_rng = np.random.default_rng(
            _address_seed(seed, state_index, proposal_index, 2)
        )

        for landscape in landscapes:
            row_index = landscape.row_index
            masks = landscape.masks
            n_active = masks.shape[1]
            if n_active == 0:
                continue

            build_start = time.perf_counter()
            model = build_sparse_mask_energy(
                state.iloc[[row_index]],
                donors.iloc[[row_index]],
                schema,
                queries,
                residual,
                max_factor_order=3,
            )
            factor_build_elapsed += time.perf_counter() - build_start
            if not np.array_equal(
                model.active_attribute_indices,
                landscape.active_attribute_indices,
            ):
                raise RuntimeError(
                    "稀疏因子与完整 oracle 的活跃属性顺序不一致"
                )
            factor_energies = evaluate_sparse_mask_energies(model, masks)
            exact_energy_max_error = max(
                exact_energy_max_error,
                float(np.max(np.abs(
                    factor_energies - landscape.directions
                ))),
            )
            singles = sparse_single_directions(model)
            one_hot_max_error = max(
                one_hot_max_error,
                float(np.max(np.abs(
                    singles
                    - directions[row_index, landscape.active_attribute_indices]
                ))),
            )
            factor_count_sum += len(model.factors)
            factor_table_entries_sum += sum(
                len(factor.values) for factor in model.factors
            )
            maximum_factor_order = max(
                maximum_factor_order, model.max_active_query_order
            )
            active_factor_rows += 1

            reference_log = baseline_mask_log_probabilities(masks, ETA)
            reference = np.exp(reference_log)
            additive = additive_mask_directions(masks, singles)
            gumbels = gumbel_rng.gumbel(size=len(masks))

            for tau in temperatures:
                strength = (
                    tau / direction_reference_scale
                    if direction_reference_scale is not None else 0.0
                )
                raw_logits = _raw_conditional_logits(
                    masks, factor_energies, ETA, strength
                )
                _accumulate_logit_diagnostics(
                    global_logit[tau], raw_logits, GIBBS_LOGIT_CLIP
                )
                independent = np.exp(gibbs_mask_log_probabilities(
                    reference_log, additive, strength
                ))
                joint = np.exp(gibbs_mask_log_probabilities(
                    reference_log, factor_energies, strength
                ))
                variants = {}
                current = independent
                previous_sweep = 0
                previous_tvd = None
                for sweep in sweeps:
                    if sweep > previous_sweep:
                        propagation_start = time.perf_counter()
                        current = propagate_random_scan_distribution(
                            model,
                            current,
                            ETA,
                            strength,
                            (sweep - previous_sweep) * n_active,
                            max_active_attributes=max_active_attributes,
                            logit_clip=GIBBS_LOGIT_CLIP,
                        )
                        exact_propagation_elapsed += (
                            time.perf_counter() - propagation_start
                        )
                    name = _gibbs_name(tau, sweep)
                    variants[name] = current.copy()
                    tvd = float(0.5 * np.abs(current - joint).sum())
                    if previous_tvd is not None:
                        tvd_snapshot_increase_max = max(
                            tvd_snapshot_increase_max, tvd - previous_tvd
                        )
                    previous_tvd = tvd
                    previous_sweep = sweep
                variants[_joint_name(tau)] = joint

                for name, probabilities in variants.items():
                    metrics = _distribution_metrics(
                        probabilities,
                        joint,
                        reference,
                        factor_energies,
                    )
                    _accumulate_kernel(
                        proposal_kernel[name], metrics, n_active
                    )
                    _accumulate_kernel(
                        global_kernel[name], metrics, n_active
                    )
                    selected_index = _sample_index(probabilities, gumbels)
                    _apply_selected_mask(
                        generated[name],
                        donors,
                        row_index,
                        landscape.active_attribute_indices,
                        masks[selected_index],
                        attr_names,
                    )

        for name, proposal in generated.items():
            measurement = _measure_proposal(
                state,
                proposal,
                q,
                loss,
                target,
                queries,
                schema,
                device,
            )
            measurement["kernel"] = _finalize_kernel(
                proposal_kernel[name]
            )
            proposal_rows[name].append(measurement)

    kernel_summary = {
        name: _finalize_kernel(accumulator)
        for name, accumulator in global_kernel.items()
    }
    proposal_summary = {
        name: _summarize_proposals(rows)
        for name, rows in proposal_rows.items()
    }
    conditional_logit_diagnostics = {
        f"tau_{_tau_label(tau)}": _finalize_logit_diagnostics(
            global_logit[tau], GIBBS_LOGIT_CLIP
        )
        for tau in temperatures
    }
    paired = {}
    recovery = {}
    for tau in temperatures:
        baseline_name = _gibbs_name(tau, 0)
        oracle_name = _joint_name(tau)
        initial_gap = kernel_summary[baseline_name][
            "absolute_expected_direction_gap"
        ]
        for sweep in sweeps:
            name = _gibbs_name(tau, sweep)
            paired[f"{name}_vs_{baseline_name}"] = _paired(
                proposal_rows[name], proposal_rows[baseline_name]
            )
            paired[f"{name}_vs_{oracle_name}"] = _paired(
                proposal_rows[name], proposal_rows[oracle_name]
            )
            remaining_gap = kernel_summary[name][
                "absolute_expected_direction_gap"
            ]
            recovery[name] = (
                1.0 - remaining_gap / initial_gap
                if initial_gap > 0.0 else 1.0
            )

    result = {
        "seed": int(seed),
        "state_rounds": int(state_rounds),
        "state_loss": float(loss),
        "probe_alpha": probe_alpha,
        "state_sha256": _frame_sha256(state),
        "n_proposals": int(proposals),
        "rho": RHO,
        "eta": ETA,
        "mu": 0.0,
        "direction_reference_scale": direction_reference_scale,
        "reference_scale_proposal_index": reference_scale_proposal_index,
        "factor_diagnostics": {
            "active_rows": active_factor_rows,
            "exact_energy_max_error": exact_energy_max_error,
            "one_hot_direction_max_error": one_hot_max_error,
            "mean_factor_count": (
                factor_count_sum / active_factor_rows
                if active_factor_rows else 0.0
            ),
            "mean_factor_table_entries": (
                factor_table_entries_sum / active_factor_rows
                if active_factor_rows else 0.0
            ),
            "maximum_active_factor_order": maximum_factor_order,
            "tvd_snapshot_increase_max": tvd_snapshot_increase_max,
            "factor_build_elapsed_sec": factor_build_elapsed,
            "exact_finite_state_propagation_elapsed_sec": (
                exact_propagation_elapsed
            ),
        },
        "conditional_logit_diagnostics": conditional_logit_diagnostics,
        "kernel_summary": kernel_summary,
        "proposal_summary": proposal_summary,
        "paired": paired,
        "expected_direction_gap_recovery": recovery,
        "proposal_rows": proposal_rows,
        "elapsed_sec": time.perf_counter() - probe_start,
    }
    if external_snapshot_controls is not None:
        result["external_snapshot_controls"] = dict(
            external_snapshot_controls
        )

    print(
        f"seed={seed:02d} state_rounds={state_rounds} "
        f"state_loss={loss:.1f} scale={direction_reference_scale}",
        flush=True,
    )
    for tau in temperatures:
        for sweep in sweeps:
            name = _gibbs_name(tau, sweep)
            kernel = kernel_summary[name]
            gain = proposal_summary[name]["gain"]["mean"]
            print(
                f"  tau={tau:g} sweep={sweep:<2} "
                f"TVD={kernel['tvd_to_joint']:.4f} "
                f"recovery={recovery[name]:.1%} gain={gain:+.2f}",
                flush=True,
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--state-rounds", nargs="+", type=int, default=[0, 100]
    )
    parser.add_argument("--proposals", type=int, default=200)
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[1.0, 2.0]
    )
    parser.add_argument(
        "--sweeps", nargs="+", type=int, default=[0, 1, 2, 4, 8]
    )
    parser.add_argument(
        "--max-active-attributes", type=int, default=12
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="numpy"
    )
    parser.add_argument(
        "--output",
        default="outputs/factorized_gibbs/frozen_mixing.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if any(value < 0 for value in args.state_rounds):
        parser.error("--state-rounds 必须为非负整数")
    if args.proposals <= 0:
        parser.error("--proposals 必须为正整数")
    if len(set(args.temperatures)) != len(args.temperatures):
        parser.error("--temperatures 不得重复")
    if any(
        not np.isfinite(value) or value < 0.0
        for value in args.temperatures
    ):
        parser.error("--temperatures 必须全部为非负有限数值")
    if args.sweeps != sorted(set(args.sweeps)) or not args.sweeps:
        parser.error("--sweeps 必须是严格递增且不重复的非空整数序列")
    if args.sweeps[0] != 0 or any(value < 0 for value in args.sweeps):
        parser.error("--sweeps 必须从 0 开始且全部非负")
    if args.max_active_attributes < 0:
        parser.error("--max-active-attributes 必须为非负整数")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)

    states = []
    experiment_start = time.perf_counter()
    for seed in args.seeds:
        for state_index, state_rounds in enumerate(args.state_rounds):
            state = _make_baseline_state(
                target,
                queries,
                schema,
                marginals,
                seed,
                state_rounds,
                args.device,
            )
            states.append(_probe_state(
                state,
                target,
                queries,
                schema,
                seed=seed,
                state_index=state_index,
                state_rounds=state_rounds,
                temperatures=args.temperatures,
                sweeps=args.sweeps,
                proposals=args.proposals,
                device=args.device,
                max_active_attributes=args.max_active_attributes,
            ))

    summary = {
        "experiment": "factorized_random_scan_gibbs_frozen_mixing",
        "scope": (
            "same_state_donor_participation_and_gumbels_exact_finite_state_"
            "propagation_no_mutation_no_generation_acceptance"
        ),
        "dataset": "test_300x10",
        "seeds": args.seeds,
        "state_rounds": args.state_rounds,
        "n_proposals_per_state": args.proposals,
        "temperatures": args.temperatures,
        "sweeps": args.sweeps,
        "sweep_definition": (
            "k uniformly random coordinates with replacement for k active bits"
        ),
        "initial_distribution": (
            "same-strength independent additive directional mask kernel"
        ),
        "device": args.device,
        "git_commit": _git_commit(),
        "command_argv": sys.argv,
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "max_active_attributes": args.max_active_attributes,
        "gibbs_logit_clip": GIBBS_LOGIT_CLIP,
        "states": states,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
