"""冻结诊断因子 Gibbs 的整代查询步幅与二次过冲来源。

脚本只读取公开 schema、记录数、预定义查询、精确 target、1-way marginal 和
合成状态。冻结 proposal 不执行 generation acceptance，也不读取真实参考表。
"""

import argparse
import contextlib
from datetime import datetime
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
    tilted_copy_probabilities,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.factorized_diffusion import (
    DEFAULT_LOGIT_CLIP,
    evolve_step_factorized_gibbs,
)
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.step_diagnostics import (
    compute_row_query_deltas,
    decompose_query_step,
)
from table_diffevo.vectorized_eval import evaluate_vectorized


SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
QUERY_PATH = Path("configs/test_300x10/measured_50query.json")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
N_RECORDS = 300
FORMAL_SEEDS = [0, 1, 2]
FORMAL_STATE_ROUNDS = [0, 500]
FORMAL_PROPOSALS = 200
FORMAL_TEMPERATURE = 2.0
FORMAL_SWEEPS = 8
RHO = 0.01
ETA = 0.5


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame):
    return _sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def _array_sha256(values):
    array = np.ascontiguousarray(values)
    payload = (
        str(array.dtype).encode()
        + repr(array.shape).encode()
        + array.tobytes()
    )
    return _sha256_bytes(payload)


def _rng_state_sha256(rng):
    serialized = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(serialized.encode("utf-8"))


def _address_seed(seed, state_index, proposal_index, stream):
    sequence = np.random.SeedSequence([
        int(seed),
        int(state_index),
        int(proposal_index),
        int(stream),
        0x53544550,
    ])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


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
    status_code, status = _git_text("status", "--porcelain")
    snapshot = {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit if commit_code == 0 else None,
        "git_worktree_clean": status_code == 0 and status == "",
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        snapshot.update({
            "torch": None,
            "torch_cuda_runtime": None,
            "cuda_available": False,
            "gpu": None,
        })
    else:
        cuda_available = bool(torch.cuda.is_available())
        snapshot.update({
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": (
                torch.cuda.get_device_name(0)
                if device == "cuda" and cuda_available else None
            ),
        })
    return snapshot


def _make_state(
    target,
    queries,
    schema,
    marginals,
    *,
    seed,
    rounds,
    temperature,
    device,
):
    if rounds == 0:
        state = init_synthetic_table(
            N_RECORDS,
            schema,
            np.random.default_rng(seed),
            marginals=marginals,
        )
        return state, {
            "method": "marginal_initialization",
            "rounds": 0,
            "best_loss": None,
            "rounds_run": 0,
            "stopped_early": False,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        state, diagnostics = run_evolution(
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
            tol=1e-9,
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
            residual_directed_diffusion=True,
            diffusion_direction_strength=temperature,
            diffusion_direction_normalization="initial_rms",
            factorized_gibbs_sweeps=0,
            factorized_gibbs_max_order=3,
        )
    return state, {
        "method": "standard_closed_loop_best",
        "rounds": int(rounds),
        "best_loss": float(diagnostics["best_loss"]),
        "rounds_run": int(diagnostics["rounds_run"]),
        "stopped_early": bool(diagnostics["stopped_early"]),
        "initial_table_sha256": diagnostics["initial_table_sha256"],
        "primary_rng_state_sha256": diagnostics[
            "primary_rng_state_sha256"
        ],
        "direction_reference_scale": diagnostics[
            "direction_reference_scale"
        ],
    }


def _copy_masks(current, proposal, donors, participate, attr_names):
    current_values = current[attr_names].reset_index(drop=True).to_numpy()
    proposal_values = proposal[attr_names].reset_index(drop=True).to_numpy()
    donor_values = donors[attr_names].reset_index(drop=True).to_numpy()
    differs = current_values != donor_values
    changed = current_values != proposal_values
    participate = np.asarray(participate, dtype=bool)
    if participate.shape != (len(current),):
        raise ValueError("participate 必须与表行数一致")
    if np.any(changed & (proposal_values != donor_values)):
        raise RuntimeError("mu=0 诊断中出现非 donor 复制变化")
    if np.any(changed & ~participate[:, None]):
        raise RuntimeError("未参与记录发生了变化")
    return differs, changed


def _replay_independent_initial_mask(
    update_seed,
    differs,
    direction_scores,
    strength,
):
    """从配对 update seed 重放参与量、独立初始 mask 和主 RNG 端点。"""
    differs = np.asarray(differs, dtype=bool)
    scores = np.asarray(direction_scores, dtype=float)
    if differs.ndim != 2 or scores.ndim != 2:
        raise ValueError("differs/direction_scores 必须是二维数组")
    expected_shape = (N_RECORDS, differs.shape[1])
    if differs.shape != expected_shape or scores.shape != expected_shape:
        raise ValueError(
            "differs/direction_scores 必须与公开记录数和属性数一致"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("direction_scores 必须全部有限")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("strength 必须是非负有限数值")

    replay_rng = np.random.default_rng(update_seed)
    participate = replay_rng.random(N_RECORDS) < RHO
    initial_mask = np.zeros_like(differs)
    for attribute_index in range(differs.shape[1]):
        probabilities = (
            ETA
            if strength == 0.0
            else tilted_copy_probabilities(
                ETA,
                scores[:, attribute_index],
                strength,
            )
        )
        initial_mask[:, attribute_index] = (
            replay_rng.random(N_RECORDS) < probabilities
        )
    initial_mask &= differs
    # 正式诊断 mu=0，但核心仍为 mutation 抽一次 N 维随机量；重放这次消耗后，
    # RNG 端点应与两侧更新完全一致，且不会产生后续属性/取值抽样。
    mutation_rows = participate & (replay_rng.random(N_RECORDS) < 0.0)
    if np.any(mutation_rows):
        raise RuntimeError("mu=0 重放不应产生变异行")
    return participate, initial_mask, _rng_state_sha256(replay_rng)


def _measure_proposal(
    current,
    proposal,
    q,
    loss,
    target,
    queries,
    schema,
    *,
    device,
):
    row_deltas = compute_row_query_deltas(current, proposal, queries)
    decomposition = decompose_query_step(row_deltas, target - q)
    proposal_q, _, _ = evaluate_vectorized(
        proposal,
        queries,
        schema,
        n_records=N_RECORDS,
        batch_size=256,
        device=device,
        want_fitness=False,
        verbose=False,
    )
    direct_delta = proposal_q.astype(float) - q.astype(float)
    delta_error = float(np.max(np.abs(
        direct_delta - decomposition["delta_q"]
    ))) if len(direct_delta) else 0.0
    proposal_loss = float(compute_loss(target, proposal_q))
    direct_gain = float(loss - proposal_loss)
    gain_error = abs(direct_gain - decomposition["net_gain"])
    quadratic_identity_error = abs(
        decomposition["quadratic_penalty"]
        - decomposition["self_penalty"]
        - decomposition["cross_penalty"]
    )
    changed_rows = np.any(row_deltas != 0, axis=1)
    return {
        "linear_gain": decomposition["linear_gain"],
        "self_penalty": decomposition["self_penalty"],
        "cross_penalty": decomposition["cross_penalty"],
        "quadratic_penalty": decomposition["quadratic_penalty"],
        "net_gain": decomposition["net_gain"],
        "positive_gain": decomposition["net_gain"] > 0.0,
        "zero_gain": decomposition["net_gain"] == 0.0,
        "negative_gain": decomposition["net_gain"] < 0.0,
        "changed_query_rows": int(changed_rows.sum()),
        "delta_q_sha256": _array_sha256(
            decomposition["delta_q"].astype(np.float64)
        ),
        "row_delta_sum_max_error": delta_error,
        "quadratic_identity_error": quadratic_identity_error,
        "gain_identity_error": gain_error,
    }


def _mask_metrics(differs, initial_mask, final_mask, participate):
    participant_indices = np.flatnonzero(participate)
    active = differs[participant_indices]
    initial = initial_mask[participant_indices]
    final = final_mask[participant_indices]
    active_counts = active.sum(axis=1, dtype=np.int64)
    initial_counts = initial.sum(axis=1, dtype=np.int64)
    final_counts = final.sum(axis=1, dtype=np.int64)
    hamming_counts = np.logical_xor(initial, final).sum(
        axis=1, dtype=np.int64
    )
    return {
        "participating_rows": int(len(participant_indices)),
        "participant_indices": participant_indices.tolist(),
        "active_block_counts": active_counts.tolist(),
        "initial_copy_block_counts": initial_counts.tolist(),
        "final_copy_block_counts": final_counts.tolist(),
        "mask_hamming_counts": hamming_counts.tolist(),
        "active_blocks": int(active_counts.sum()),
        "initial_copied_cells": int(initial_counts.sum()),
        "final_copied_cells": int(final_counts.sum()),
        "mask_hamming_cells": int(hamming_counts.sum()),
        "mean_active_blocks_per_participant": (
            float(active_counts.mean()) if len(active_counts) else 0.0
        ),
        "mean_initial_copy_blocks_per_participant": (
            float(initial_counts.mean()) if len(initial_counts) else 0.0
        ),
        "mean_final_copy_blocks_per_participant": (
            float(final_counts.mean()) if len(final_counts) else 0.0
        ),
        "mean_mask_hamming_per_participant": (
            float(hamming_counts.mean()) if len(hamming_counts) else 0.0
        ),
        "changed_rows": int(np.any(final, axis=1).sum()),
    }


def _summarize_mask_pairs(pair_rows):
    metrics = (
        "active_blocks",
        "initial_copied_cells",
        "final_copied_cells",
        "mask_hamming_cells",
        "mean_active_blocks_per_participant",
        "mean_initial_copy_blocks_per_participant",
        "mean_final_copy_blocks_per_participant",
        "mean_mask_hamming_per_participant",
        "changed_rows",
    )
    return {
        metric: _summarize_values([
            row["mask"][metric] for row in pair_rows
        ])
        for metric in metrics
    }


def _summarize_values(values):
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("汇总值必须非空且全部有限")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _paired(candidate, baseline, metric):
    candidate_values = np.asarray(
        [row[metric] for row in candidate], dtype=float
    )
    baseline_values = np.asarray(
        [row[metric] for row in baseline], dtype=float
    )
    if (
        len(candidate_values) == 0
        or len(candidate_values) != len(baseline_values)
        or not np.all(np.isfinite(candidate_values))
        or not np.all(np.isfinite(baseline_values))
    ):
        raise ValueError(f"{metric} 的配对值无效")
    difference = candidate_values - baseline_values
    return {
        "n": int(len(difference)),
        "baseline": _summarize_values(baseline_values),
        "candidate": _summarize_values(candidate_values),
        "difference": _summarize_values(difference),
        "candidate_larger": int(np.sum(difference > 0.0)),
        "ties": int(np.sum(difference == 0.0)),
        "candidate_smaller": int(np.sum(difference < 0.0)),
        "differences": difference.tolist(),
    }


def _choose_source(global_paired, state_paired):
    total = global_paired["quadratic_penalty"]["difference"]["mean"]
    self_difference = global_paired["self_penalty"]["difference"]["mean"]
    cross_difference = global_paired["cross_penalty"]["difference"]["mean"]
    self_dominant_states = sum(
        state["self_penalty"]["difference"]["mean"]
        > state["cross_penalty"]["difference"]["mean"]
        for state in state_paired
    )
    cross_dominant_states = sum(
        state["cross_penalty"]["difference"]["mean"]
        > state["self_penalty"]["difference"]["mean"]
        for state in state_paired
    )
    self_share = self_difference / total if total > 0.0 else None
    cross_share = cross_difference / total if total > 0.0 else None
    if (
        total > 0.0
        and self_difference > 0.0
        and self_share >= 2.0 / 3.0
        and self_dominant_states >= 4
    ):
        decision = "supports_per_row_or_fixed_cardinality_normalization"
    elif (
        total > 0.0
        and cross_difference > 0.0
        and cross_share >= 2.0 / 3.0
        and cross_dominant_states >= 4
    ):
        decision = "supports_cross_row_time_step_normalization"
    else:
        decision = "mixed_or_inconclusive_source"
    return {
        "decision": decision,
        "quadratic_penalty_difference": total,
        "self_penalty_difference": self_difference,
        "cross_penalty_difference": cross_difference,
        "self_share_of_positive_total": self_share,
        "cross_share_of_positive_total": cross_share,
        "self_dominant_seed_states": self_dominant_states,
        "cross_dominant_seed_states": cross_dominant_states,
        "n_seed_states": len(state_paired),
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
    proposals,
    temperature,
    sweeps,
    max_factor_order,
    device,
    fixed_reference_scale=None,
):
    if (
        fixed_reference_scale is not None
        and (
            not np.isfinite(fixed_reference_scale)
            or fixed_reference_scale <= 0.0
        )
    ):
        raise ValueError(
            "fixed_reference_scale 必须是正有限数值或 None"
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
    loss = float(compute_loss(target, q))
    use_torch = device in ("cuda", "cpu")
    distances = pairwise_block_distance(
        state, state, schema, device=device, return_tensor=use_torch
    )
    alpha = 2.0 if state_rounds == 0 else 10.0
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
    attr_names = schema.attribute_names()
    baseline_rows = []
    candidate_rows = []
    pair_rows = []
    reference_scale = fixed_reference_scale
    reference_scale_proposal_index = None
    reference_scale_source = (
        "standard_closed_loop_initial_rms"
        if fixed_reference_scale is not None
        else "first_nonzero_frozen_proposal"
    )
    probe_start = time.perf_counter()

    for proposal_index in range(proposals):
        donor_seed = _address_seed(seed, state_index, proposal_index, 0)
        donor_rng = np.random.default_rng(donor_seed)
        donor_idx = sample_donors(probabilities, donor_rng, device=device)
        donors = state.iloc[donor_idx].reset_index(drop=True)
        direction_scores = compute_copy_direction_scores(
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
        active_directions = direction_scores[differs]
        if reference_scale is None:
            candidate_scale = direction_rms_scale(active_directions)
            if candidate_scale > 0.0:
                reference_scale = candidate_scale
                reference_scale_proposal_index = proposal_index
        strength = (
            temperature / reference_scale
            if reference_scale is not None else 0.0
        )

        update_seed = _address_seed(seed, state_index, proposal_index, 1)
        gibbs_seed = _address_seed(seed, state_index, proposal_index, 2)
        baseline_rng = np.random.default_rng(update_seed)
        candidate_rng = np.random.default_rng(update_seed)
        candidate_gibbs_rng = np.random.default_rng(gibbs_seed)
        gibbs_initial_state = _rng_state_sha256(candidate_gibbs_rng)

        baseline, baseline_diagnostics = evolve_step_factorized_gibbs(
            state,
            donors,
            schema,
            queries,
            residual,
            rho=RHO,
            eta=ETA,
            mu=0.0,
            copy_direction_scores=direction_scores,
            copy_direction_strength=strength,
            n_sweeps=0,
            rng=baseline_rng,
            max_factor_order=max_factor_order,
            gibbs_logit_clip=DEFAULT_LOGIT_CLIP,
        )
        candidate, candidate_diagnostics = evolve_step_factorized_gibbs(
            state,
            donors,
            schema,
            queries,
            residual,
            rho=RHO,
            eta=ETA,
            mu=0.0,
            copy_direction_scores=direction_scores,
            copy_direction_strength=strength,
            n_sweeps=sweeps,
            rng=candidate_rng,
            gibbs_rng=candidate_gibbs_rng,
            max_factor_order=max_factor_order,
            gibbs_logit_clip=DEFAULT_LOGIT_CLIP,
        )
        baseline_rng_hash = _rng_state_sha256(baseline_rng)
        candidate_rng_hash = _rng_state_sha256(candidate_rng)
        if baseline_rng_hash != candidate_rng_hash:
            raise RuntimeError("0/8 sweep 的主 RNG 端点不一致")

        participate, replayed_initial_mask, replayed_rng_hash = (
            _replay_independent_initial_mask(
                update_seed,
                differs,
                direction_scores,
                strength,
            )
        )
        if replayed_rng_hash != baseline_rng_hash:
            raise RuntimeError("独立初始 mask 重放与主 RNG 端点不一致")
        expected_participating = int(participate.sum())
        if (
            baseline_diagnostics["participating_rows"]
            != expected_participating
            or candidate_diagnostics["participating_rows"]
            != expected_participating
        ):
            raise RuntimeError("参与行重建与更新诊断不一致")

        baseline_differs, baseline_applied_mask = _copy_masks(
            state, baseline, donors, participate, attr_names
        )
        candidate_differs, candidate_applied_mask = _copy_masks(
            state, candidate, donors, participate, attr_names
        )
        if not np.array_equal(baseline_differs, candidate_differs):
            raise RuntimeError("两侧 active block 不一致")
        expected_applied_initial_mask = (
            replayed_initial_mask & participate[:, None]
        )
        if not np.array_equal(
            baseline_applied_mask, expected_applied_initial_mask
        ):
            raise RuntimeError("baseline 落地编辑与重放的独立初始 mask 不一致")
        candidate_final_mask = replayed_initial_mask.copy()
        candidate_final_mask[participate] = candidate_applied_mask[
            participate
        ]

        baseline_measurement = _measure_proposal(
            state,
            baseline,
            q,
            loss,
            target,
            queries,
            schema,
            device=device,
        )
        candidate_measurement = _measure_proposal(
            state,
            candidate,
            q,
            loss,
            target,
            queries,
            schema,
            device=device,
        )
        mask_metrics = _mask_metrics(
            differs,
            replayed_initial_mask,
            candidate_final_mask,
            participate,
        )
        baseline_measurement.update({
            "seed": int(seed),
            "state_index": int(state_index),
            "state_rounds": int(state_rounds),
            "proposal_index": int(proposal_index),
            "variant": "independent_0_sweeps",
            "donor_indices_sha256": _array_sha256(
                donor_idx.astype(np.int64)
            ),
            "participation_sha256": _array_sha256(participate),
            "initial_mask_sha256": _array_sha256(replayed_initial_mask),
            "final_mask_sha256": _array_sha256(replayed_initial_mask),
            "applied_mask_sha256": _array_sha256(
                baseline_applied_mask
            ),
            "primary_rng_state_sha256": baseline_rng_hash,
            "gibbs_initial_rng_state_sha256": None,
            "gibbs_final_rng_state_sha256": None,
            "copied_cells": int(baseline_applied_mask.sum()),
            "changed_rows": int(
                np.any(baseline_applied_mask, axis=1).sum()
            ),
            "mean_copy_blocks_per_participant": (
                mask_metrics[
                    "mean_initial_copy_blocks_per_participant"
                ]
            ),
            "factor_count": 0,
            "factor_table_entries": 0,
            "gibbs_microsteps": 0,
            "factor_build_elapsed_sec": 0.0,
            "gibbs_sample_elapsed_sec": 0.0,
        })
        candidate_measurement.update({
            "seed": int(seed),
            "state_index": int(state_index),
            "state_rounds": int(state_rounds),
            "proposal_index": int(proposal_index),
            "variant": f"factorized_{sweeps}_sweeps",
            "donor_indices_sha256": _array_sha256(
                donor_idx.astype(np.int64)
            ),
            "participation_sha256": _array_sha256(participate),
            "initial_mask_sha256": _array_sha256(replayed_initial_mask),
            "final_mask_sha256": _array_sha256(candidate_final_mask),
            "applied_mask_sha256": _array_sha256(
                candidate_applied_mask
            ),
            "primary_rng_state_sha256": candidate_rng_hash,
            "gibbs_initial_rng_state_sha256": gibbs_initial_state,
            "gibbs_final_rng_state_sha256": _rng_state_sha256(
                candidate_gibbs_rng
            ),
            "copied_cells": int(candidate_applied_mask.sum()),
            "changed_rows": int(
                np.any(candidate_applied_mask, axis=1).sum()
            ),
            "mean_copy_blocks_per_participant": (
                mask_metrics[
                    "mean_final_copy_blocks_per_participant"
                ]
            ),
            "factor_count": int(candidate_diagnostics["factor_count"]),
            "factor_table_entries": int(
                candidate_diagnostics["factor_table_entries"]
            ),
            "gibbs_microsteps": int(
                candidate_diagnostics["gibbs_microsteps"]
            ),
            "factor_build_elapsed_sec": float(
                candidate_diagnostics["factor_build_elapsed_sec"]
            ),
            "gibbs_sample_elapsed_sec": float(
                candidate_diagnostics["gibbs_sample_elapsed_sec"]
            ),
        })
        baseline_rows.append(baseline_measurement)
        candidate_rows.append(candidate_measurement)
        pair_rows.append({
            "seed": int(seed),
            "state_index": int(state_index),
            "state_rounds": int(state_rounds),
            "proposal_index": int(proposal_index),
            "update_seed": int(update_seed),
            "gibbs_seed": int(gibbs_seed),
            "direction_strength": float(strength),
            "mask": mask_metrics,
            "primary_rng_aligned": (
                baseline_rng_hash == candidate_rng_hash
            ),
            "donor_aligned": (
                baseline_measurement["donor_indices_sha256"]
                == candidate_measurement["donor_indices_sha256"]
            ),
            "participation_aligned": (
                baseline_measurement["participation_sha256"]
                == candidate_measurement["participation_sha256"]
            ),
            "initial_mask_aligned": (
                baseline_measurement["final_mask_sha256"]
                == candidate_measurement["initial_mask_sha256"]
            ),
            "initial_mask_replay_rng_aligned": (
                replayed_rng_hash == baseline_rng_hash
            ),
            "baseline_applied_initial_mask_aligned": np.array_equal(
                baseline_applied_mask,
                expected_applied_initial_mask,
            ),
        })

    metrics = (
        "linear_gain",
        "self_penalty",
        "cross_penalty",
        "quadratic_penalty",
        "net_gain",
        "positive_gain",
        "changed_rows",
        "changed_query_rows",
        "copied_cells",
        "mean_copy_blocks_per_participant",
        "factor_count",
        "factor_table_entries",
        "gibbs_microsteps",
        "factor_build_elapsed_sec",
        "gibbs_sample_elapsed_sec",
    )
    paired = {
        metric: _paired(candidate_rows, baseline_rows, metric)
        for metric in metrics
    }
    gates = {
        "primary_rng_aligned": all(
            row["primary_rng_aligned"] for row in pair_rows
        ),
        "donor_aligned": all(row["donor_aligned"] for row in pair_rows),
        "participation_aligned": all(
            row["participation_aligned"] for row in pair_rows
        ),
        "initial_mask_aligned": all(
            row["initial_mask_aligned"] for row in pair_rows
        ),
        "initial_mask_replay_rng_aligned": all(
            row["initial_mask_replay_rng_aligned"] for row in pair_rows
        ),
        "baseline_applied_initial_mask_aligned": all(
            row["baseline_applied_initial_mask_aligned"]
            for row in pair_rows
        ),
        "row_delta_sum_max_error": max(
            row["row_delta_sum_max_error"]
            for row in baseline_rows + candidate_rows
        ),
        "quadratic_identity_max_error": max(
            row["quadratic_identity_error"]
            for row in baseline_rows + candidate_rows
        ),
        "gain_identity_max_error": max(
            row["gain_identity_error"]
            for row in baseline_rows + candidate_rows
        ),
    }
    return {
        "seed": int(seed),
        "state_index": int(state_index),
        "state_rounds": int(state_rounds),
        "state_sha256": _frame_sha256(state),
        "state_loss": loss,
        "direction_reference_scale": reference_scale,
        "direction_reference_scale_source": reference_scale_source,
        "reference_scale_proposal_index": reference_scale_proposal_index,
        "n_proposals": int(proposals),
        "elapsed_sec": time.perf_counter() - probe_start,
        "gates": gates,
        "paired": paired,
        "mask_summary": _summarize_mask_pairs(pair_rows),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "pair_rows": pair_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument(
        "--state-rounds",
        nargs="+",
        type=int,
        default=FORMAL_STATE_ROUNDS,
    )
    parser.add_argument(
        "--proposals", type=int, default=FORMAL_PROPOSALS
    )
    parser.add_argument(
        "--temperature", type=float, default=FORMAL_TEMPERATURE
    )
    parser.add_argument("--sweeps", type=int, default=FORMAL_SWEEPS)
    parser.add_argument("--max-factor-order", type=int, default=3)
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default="outputs/factorized_step_overshoot/formal.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error("--seeds 必须非空、非负且不重复")
    if (
        not args.state_rounds
        or len(set(args.state_rounds)) != len(args.state_rounds)
        or any(rounds < 0 for rounds in args.state_rounds)
    ):
        parser.error("--state-rounds 必须非空、非负且不重复")
    if args.proposals <= 0:
        parser.error("--proposals 必须为正整数")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")
    if not 1 <= args.max_factor_order <= 8:
        parser.error("--max-factor-order 必须在 1..8 内")

    formal_protocol_matches = (
        args.seeds == FORMAL_SEEDS
        and args.state_rounds == FORMAL_STATE_ROUNDS
        and args.proposals == FORMAL_PROPOSALS
        and args.temperature == FORMAL_TEMPERATURE
        and args.sweeps == FORMAL_SWEEPS
        and args.max_factor_order == 3
        and args.device == "cuda"
    )
    output_path = Path(args.output)
    if output_path.exists() and (
        formal_protocol_matches or not args.overwrite
    ):
        raise FileExistsError(f"输出已存在，不覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    environment = _environment_snapshot(args.device)
    if formal_protocol_matches and not environment["git_worktree_clean"]:
        raise RuntimeError("正式协议要求 tracked 工作树干净")
    if args.device == "cuda" and not environment["cuda_available"]:
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备")

    schema = load_schema(str(SCHEMA_PATH))
    queries = load_queries(str(QUERY_PATH))
    target = np.asarray([query["result"] for query in queries], dtype=float)
    marginals = load_marginals(str(MARGINALS_PATH))
    if (
        len(queries) != 50
        or target.shape != (50,)
        or not np.all(np.isfinite(target))
        or len(schema.attribute_names()) != 10
        or marginals.get("n_records") != N_RECORDS
        or set(marginals.get("attributes", {}))
        != set(schema.attribute_names())
    ):
        raise ValueError("test_300x10 的公开输入与协议不一致")

    states = []
    all_baseline = []
    all_candidate = []
    all_pair_rows = []
    state_initialization_aligned = True
    experiment_start = time.perf_counter()
    for seed in args.seeds:
        prepared_states = []
        for state_index, state_rounds in enumerate(args.state_rounds):
            state, generation = _make_state(
                target,
                queries,
                schema,
                marginals,
                seed=seed,
                rounds=state_rounds,
                temperature=args.temperature,
                device=args.device,
            )
            prepared_states.append(
                (state_index, state_rounds, state, generation)
            )

        fixed_reference_scale = next(
            (
                generation.get("direction_reference_scale")
                for _, _, _, generation in prepared_states
                if generation.get("direction_reference_scale") is not None
            ),
            None,
        )
        initial_state_hashes = [
            _frame_sha256(state)
            for _, state_rounds, state, _ in prepared_states
            if state_rounds == 0
        ]
        generated_initial_hashes = [
            generation.get("initial_table_sha256")
            for _, state_rounds, _, generation in prepared_states
            if state_rounds > 0
            and generation.get("initial_table_sha256") is not None
        ]
        if initial_state_hashes and generated_initial_hashes:
            seed_initialization_aligned = all(
                value == initial_state_hashes[0]
                for value in generated_initial_hashes
            )
            if not seed_initialization_aligned:
                raise RuntimeError(
                    "直接 marginal 状态与标准闭环初始表不一致"
                )
            state_initialization_aligned &= seed_initialization_aligned

        for (
            state_index,
            state_rounds,
            state,
            generation,
        ) in prepared_states:
            state_result = _probe_state(
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
                fixed_reference_scale=fixed_reference_scale,
            )
            state_result["state_generation"] = generation
            states.append(state_result)
            all_baseline.extend(state_result["baseline_rows"])
            all_candidate.extend(state_result["candidate_rows"])
            all_pair_rows.extend(state_result["pair_rows"])
            quadratic = state_result["paired"][
                "quadratic_penalty"
            ]["difference"]["mean"]
            self_difference = state_result["paired"][
                "self_penalty"
            ]["difference"]["mean"]
            cross_difference = state_result["paired"][
                "cross_penalty"
            ]["difference"]["mean"]
            print(
                f"seed={seed:02d} state={state_rounds:03d} "
                f"ΔQ={quadratic:+.4f} "
                f"Δself={self_difference:+.4f} "
                f"Δcross={cross_difference:+.4f}",
                flush=True,
            )

    metrics = tuple(states[0]["paired"])
    global_paired = {
        metric: _paired(all_candidate, all_baseline, metric)
        for metric in metrics
    }
    state_paired = [state["paired"] for state in states]
    source = _choose_source(global_paired, state_paired)
    gates = {
        "state_initialization_aligned": state_initialization_aligned,
        "all_primary_rng_aligned": all(
            state["gates"]["primary_rng_aligned"] for state in states
        ),
        "all_donors_aligned": all(
            state["gates"]["donor_aligned"] for state in states
        ),
        "all_participation_aligned": all(
            state["gates"]["participation_aligned"] for state in states
        ),
        "all_initial_masks_aligned": all(
            state["gates"]["initial_mask_aligned"] for state in states
        ),
        "all_initial_mask_replay_rng_aligned": all(
            state["gates"]["initial_mask_replay_rng_aligned"]
            for state in states
        ),
        "all_baseline_applied_initial_masks_aligned": all(
            state["gates"]["baseline_applied_initial_mask_aligned"]
            for state in states
        ),
        "row_delta_sum_max_error": max(
            state["gates"]["row_delta_sum_max_error"] for state in states
        ),
        "quadratic_identity_max_error": max(
            state["gates"]["quadratic_identity_max_error"]
            for state in states
        ),
        "gain_identity_max_error": max(
            state["gates"]["gain_identity_max_error"] for state in states
        ),
    }
    gate_passed = (
        gates["state_initialization_aligned"]
        and gates["all_primary_rng_aligned"]
        and gates["all_donors_aligned"]
        and gates["all_participation_aligned"]
        and gates["all_initial_masks_aligned"]
        and gates["all_initial_mask_replay_rng_aligned"]
        and gates["all_baseline_applied_initial_masks_aligned"]
        and gates["row_delta_sum_max_error"] <= 1e-10
        and gates["quadratic_identity_max_error"] <= 1e-10
        and gates["gain_identity_max_error"] <= 1e-10
    )
    final_decision = (
        source["decision"]
        if formal_protocol_matches and gate_passed
        else (
            "diagnostic_gate_failed"
            if formal_protocol_matches else "non_formal_run_no_decision"
        )
    )
    summary = {
        "experiment": "factorized_gibbs_generation_step_overshoot",
        "issue": 17,
        "formal_protocol_matches": formal_protocol_matches,
        "diagnostic_gate_passed": gate_passed,
        "decision": final_decision,
        "decision_rule": (
            "component share >= 2/3 of positive total and component "
            "dominates in at least 4/6 seed-states"
        ),
        "scope": "fixed_exact_target_frozen_proposals_no_acceptance",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "seeds": args.seeds,
        "state_rounds": args.state_rounds,
        "proposals_per_state": args.proposals,
        "n_seed_states": len(states),
        "n_paired_proposals": len(all_baseline),
        "temperature": args.temperature,
        "baseline_sweeps": 0,
        "candidate_sweeps": args.sweeps,
        "max_factor_order": args.max_factor_order,
        "gibbs_logit_clip": float(DEFAULT_LOGIT_CLIP),
        "rho": RHO,
        "eta": ETA,
        "mu": 0.0,
        "device": args.device,
        "real_data_access": "none",
        "state_initialization_aligned": state_initialization_aligned,
        "environment": environment,
        "public_input_sha256": {
            str(path): _sha256_file(path)
            for path in (SCHEMA_PATH, QUERY_PATH, MARGINALS_PATH)
        },
        "elapsed_sec": time.perf_counter() - experiment_start,
        "gates": gates,
        "source_attribution": source,
        "global_mask_summary": _summarize_mask_pairs(all_pair_rows),
        "global_paired": global_paired,
        "states": states,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\n===== 整代步幅诊断 =====")
    for metric in (
        "linear_gain",
        "self_penalty",
        "cross_penalty",
        "quadratic_penalty",
        "net_gain",
        "copied_cells",
    ):
        comparison = global_paired[metric]
        print(
            f"{metric:<24} "
            f"{comparison['baseline']['mean']:.6g} -> "
            f"{comparison['candidate']['mean']:.6g} "
            f"(Δ={comparison['difference']['mean']:+.6g})"
        )
    print(f"诊断门禁：{gate_passed}")
    print(f"来源判断：{final_decision}")
    print(f"详细结果：{output_path}")


if __name__ == "__main__":
    main()
