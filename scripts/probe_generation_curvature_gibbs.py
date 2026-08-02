"""冻结比较整代一阶 Gibbs 与曲率感知 Gibbs。

脚本只使用公开 schema、记录数、预定义查询、精确 target、1-way marginal 与
合成状态。所有 proposal 都只离线测量，不执行 generation acceptance，也没有真实
参考表输入路径。
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
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.factorized_diffusion import evolve_step_factorized_gibbs
from table_diffevo.generation_curvature import (
    evolve_step_generation_curvature_gibbs,
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
BASELINE_CURVATURE = 0.0
CANDIDATE_CURVATURE = 1.0
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
        0x4355525645,
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
        raise RuntimeError("mu=0 探针中出现非 donor 复制变化")
    if np.any(changed & ~participate[:, None]):
        raise RuntimeError("未参与记录发生了变化")
    return differs, changed


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
    row_delta_error = float(np.max(np.abs(
        direct_delta - decomposition["delta_q"]
    ))) if len(direct_delta) else 0.0
    proposal_loss = float(compute_loss(target, proposal_q))
    direct_gain = float(loss - proposal_loss)
    return {
        "linear_gain": decomposition["linear_gain"],
        "self_penalty": decomposition["self_penalty"],
        "cross_penalty": decomposition["cross_penalty"],
        "quadratic_penalty": decomposition["quadratic_penalty"],
        "net_gain": decomposition["net_gain"],
        "positive_gain": decomposition["net_gain"] > 0.0,
        "zero_gain": decomposition["net_gain"] == 0.0,
        "negative_gain": decomposition["net_gain"] < 0.0,
        "delta_q": decomposition["delta_q"].tolist(),
        "delta_q_sha256": _array_sha256(
            decomposition["delta_q"].astype(np.float64)
        ),
        "row_delta_sum_max_error": row_delta_error,
        "quadratic_identity_error": abs(
            decomposition["quadratic_penalty"]
            - decomposition["self_penalty"]
            - decomposition["cross_penalty"]
        ),
        "gain_identity_error": abs(
            direct_gain - decomposition["net_gain"]
        ),
    }


def _mask_metrics(differs, initial_mask, final_mask, participate):
    participants = np.flatnonzero(participate)
    active = differs[participants]
    initial = initial_mask[participants]
    final = final_mask[participants]
    active_counts = active.sum(axis=1, dtype=np.int64)
    initial_counts = initial.sum(axis=1, dtype=np.int64)
    final_counts = final.sum(axis=1, dtype=np.int64)
    hamming = np.logical_xor(initial, final).sum(axis=1, dtype=np.int64)
    return {
        "participating_rows": int(len(participants)),
        "active_blocks": int(active_counts.sum()),
        "initial_copied_cells": int(initial_counts.sum()),
        "final_copied_cells": int(final_counts.sum()),
        "mask_hamming_cells": int(hamming.sum()),
        "changed_rows": int(np.any(final, axis=1).sum()),
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
            float(hamming.mean()) if len(hamming) else 0.0
        ),
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


def _conditional_summary(rows):
    counts = np.asarray([
        row["conditional_probability_count"] for row in rows
    ], dtype=np.int64)
    total = int(counts.sum())
    nonempty = [
        row for row in rows if row["conditional_probability_count"] > 0
    ]
    if total == 0:
        return {
            "n_microsteps": 0,
            "mean_entropy": None,
            "min_probability": None,
            "max_probability": None,
            "all_bidirectional": True,
        }
    entropy_sum = sum(
        row["conditional_entropy_mean"]
        * row["conditional_probability_count"]
        for row in nonempty
    )
    return {
        "n_microsteps": total,
        "mean_entropy": float(entropy_sum / total),
        "min_probability": float(min(
            row["conditional_probability_min"] for row in nonempty
        )),
        "max_probability": float(max(
            row["conditional_probability_max"] for row in nonempty
        )),
        "all_bidirectional": all(
            row["all_conditionals_bidirectional"] for row in rows
        ),
    }


def _decision_from_state_results(states, late_paired, initial_paired):
    late_states = [state for state in states if state["state_rounds"] == 500]
    improved_states = sum(
        state["paired"]["net_gain"]["difference"]["mean"] > 0.0
        for state in late_states
    )
    positive_rate_non_decreasing_states = sum(
        state["paired"]["positive_gain"]["difference"]["mean"] >= 0.0
        for state in late_states
    )
    late_mean_difference = late_paired["net_gain"]["difference"]["mean"]
    late_positive_rate_difference = late_paired[
        "positive_gain"
    ]["difference"]["mean"]
    if (
        improved_states == 3
        and late_mean_difference > 0.0
        and late_positive_rate_difference >= 0.0
        and positive_rate_non_decreasing_states >= 2
    ):
        decision = "supports_late_curvature_kernel"
    elif late_mean_difference > 0.0 and improved_states >= 2:
        decision = "late_curvature_inconclusive"
    else:
        decision = "late_curvature_not_supported"

    initial_baseline = initial_paired["net_gain"]["baseline"]["mean"]
    initial_candidate = initial_paired["net_gain"]["candidate"]["mean"]
    initial_relative_change = (
        (initial_candidate - initial_baseline) / abs(initial_baseline)
        if initial_baseline != 0.0 else None
    )
    initial_deterioration_risk = (
        initial_relative_change is not None
        and initial_relative_change < -0.05
    )

    baseline_entropy = _conditional_summary([
        row
        for state in late_states
        for row in state["baseline_rows"]
    ])
    candidate_entropy = _conditional_summary([
        row
        for state in late_states
        for row in state["candidate_rows"]
    ])
    entropy_relative_change = (
        (
            candidate_entropy["mean_entropy"]
            - baseline_entropy["mean_entropy"]
        ) / baseline_entropy["mean_entropy"]
        if baseline_entropy["mean_entropy"] not in (None, 0.0)
        else None
    )
    entropy_risk = (
        entropy_relative_change is not None
        and entropy_relative_change < -0.10
    )
    return {
        "decision": decision,
        "late_improved_seed_states": improved_states,
        "late_positive_rate_non_decreasing_seed_states": (
            positive_rate_non_decreasing_states
        ),
        "n_late_seed_states": len(late_states),
        "late_net_gain_difference": late_mean_difference,
        "late_positive_gain_rate_difference": late_positive_rate_difference,
        "initial_net_gain_relative_change": initial_relative_change,
        "initial_deterioration_over_5pct_risk": initial_deterioration_risk,
        "late_baseline_conditional": baseline_entropy,
        "late_candidate_conditional": candidate_entropy,
        "late_conditional_entropy_relative_change": entropy_relative_change,
        "conditional_entropy_concentration_risk": entropy_risk,
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
        raise ValueError("fixed_reference_scale 必须是正有限数值或 None")
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
    donor_probabilities = compute_sampling_probs(
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
        donor_idx = sample_donors(
            donor_probabilities, donor_rng, device=device
        )
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
        initial_rng = np.random.default_rng(update_seed)
        reference_rng = np.random.default_rng(update_seed)
        baseline_rng = np.random.default_rng(update_seed)
        candidate_rng = np.random.default_rng(update_seed)
        reference_gibbs_rng = np.random.default_rng(gibbs_seed)
        baseline_gibbs_rng = np.random.default_rng(gibbs_seed)
        candidate_gibbs_rng = np.random.default_rng(gibbs_seed)
        gibbs_initial_hash = _rng_state_sha256(reference_gibbs_rng)

        initial, initial_diagnostics = evolve_step_factorized_gibbs(
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
            rng=initial_rng,
            max_factor_order=max_factor_order,
        )
        reference, reference_diagnostics = evolve_step_factorized_gibbs(
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
            rng=reference_rng,
            gibbs_rng=reference_gibbs_rng,
            max_factor_order=max_factor_order,
        )
        baseline, baseline_diagnostics = (
            evolve_step_generation_curvature_gibbs(
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
                curvature_weight=BASELINE_CURVATURE,
                rng=baseline_rng,
                gibbs_rng=baseline_gibbs_rng,
                max_factor_order=max_factor_order,
            )
        )
        candidate, candidate_diagnostics = (
            evolve_step_generation_curvature_gibbs(
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
                curvature_weight=CANDIDATE_CURVATURE,
                rng=candidate_rng,
                gibbs_rng=candidate_gibbs_rng,
                max_factor_order=max_factor_order,
            )
        )

        primary_hashes = {
            _rng_state_sha256(value)
            for value in (
                initial_rng,
                reference_rng,
                baseline_rng,
                candidate_rng,
            )
        }
        gibbs_hashes = {
            _rng_state_sha256(value)
            for value in (
                reference_gibbs_rng,
                baseline_gibbs_rng,
                candidate_gibbs_rng,
            )
        }
        participation_rng = np.random.default_rng(update_seed)
        participate = participation_rng.random(N_RECORDS) < RHO
        expected_participating = int(participate.sum())
        if any(
            diagnostics["participating_rows"] != expected_participating
            for diagnostics in (
                initial_diagnostics,
                reference_diagnostics,
                baseline_diagnostics,
                candidate_diagnostics,
            )
        ):
            raise RuntimeError("参与行重建与更新诊断不一致")

        initial_differs, initial_mask = _copy_masks(
            state, initial, donors, participate, attr_names
        )
        reference_differs, reference_mask = _copy_masks(
            state, reference, donors, participate, attr_names
        )
        baseline_differs, baseline_mask = _copy_masks(
            state, baseline, donors, participate, attr_names
        )
        candidate_differs, candidate_mask = _copy_masks(
            state, candidate, donors, participate, attr_names
        )
        if not (
            np.array_equal(initial_differs, reference_differs)
            and np.array_equal(initial_differs, baseline_differs)
            and np.array_equal(initial_differs, candidate_differs)
        ):
            raise RuntimeError("各变体 active block 不一致")

        initial_measurement = _measure_proposal(
            state, initial, q, loss, target, queries, schema, device=device
        )
        baseline_measurement = _measure_proposal(
            state, baseline, q, loss, target, queries, schema, device=device
        )
        candidate_measurement = _measure_proposal(
            state, candidate, q, loss, target, queries, schema, device=device
        )
        baseline_mask_metrics = _mask_metrics(
            differs, initial_mask, baseline_mask, participate
        )
        candidate_mask_metrics = _mask_metrics(
            differs, initial_mask, candidate_mask, participate
        )
        baseline_candidate_hamming = int(np.logical_xor(
            baseline_mask, candidate_mask
        ).sum())
        common_keys = (
            "participating_rows",
            "active_gibbs_rows",
            "active_blocks",
            "factor_count",
            "factor_table_entries",
            "gibbs_microsteps",
        )
        gamma_zero_diagnostics_equal = all(
            baseline_diagnostics[key] == reference_diagnostics[key]
            for key in common_keys
        )
        initial_internal_delta = np.asarray(
            baseline_diagnostics["initial_query_delta"], dtype=float
        )
        candidate_initial_internal_delta = np.asarray(
            candidate_diagnostics["initial_query_delta"], dtype=float
        )
        baseline_final_internal_delta = np.asarray(
            baseline_diagnostics["final_query_delta"], dtype=float
        )
        candidate_final_internal_delta = np.asarray(
            candidate_diagnostics["final_query_delta"], dtype=float
        )
        direct_initial_delta = np.asarray(
            initial_measurement["delta_q"], dtype=float
        )
        direct_baseline_delta = np.asarray(
            baseline_measurement["delta_q"], dtype=float
        )
        direct_candidate_delta = np.asarray(
            candidate_measurement["delta_q"], dtype=float
        )
        initial_delta_error = max(
            float(np.max(np.abs(
                initial_internal_delta - direct_initial_delta
            ))),
            float(np.max(np.abs(
                candidate_initial_internal_delta - direct_initial_delta
            ))),
        )
        final_delta_error = max(
            float(np.max(np.abs(
                baseline_final_internal_delta - direct_baseline_delta
            ))),
            float(np.max(np.abs(
                candidate_final_internal_delta - direct_candidate_delta
            ))),
        )
        candidate_energy_error = abs(
            N_RECORDS * candidate_diagnostics["final_generation_energy"]
            - candidate_measurement["net_gain"]
        )
        initial_mask_hash = _array_sha256(initial_mask)
        internal_initial_masks_aligned = (
            baseline_diagnostics["initial_copy_mask_sha256"]
            == initial_mask_hash
            == candidate_diagnostics["initial_copy_mask_sha256"]
        )

        for measurement, diagnostics, mask_metrics, final_mask, variant in (
            (
                baseline_measurement,
                baseline_diagnostics,
                baseline_mask_metrics,
                baseline_mask,
                "generation_linear_gamma0",
            ),
            (
                candidate_measurement,
                candidate_diagnostics,
                candidate_mask_metrics,
                candidate_mask,
                "generation_curvature_gamma1",
            ),
        ):
            internal_initial_delta = np.asarray(
                diagnostics["initial_query_delta"], dtype=np.float64
            )
            internal_final_delta = np.asarray(
                diagnostics["final_query_delta"], dtype=np.float64
            )
            measurement.update({
                "seed": int(seed),
                "state_index": int(state_index),
                "state_rounds": int(state_rounds),
                "proposal_index": int(proposal_index),
                "variant": variant,
                "donor_indices_sha256": _array_sha256(
                    donor_idx.astype(np.int64)
                ),
                "participation_sha256": _array_sha256(participate),
                "initial_mask_sha256": _array_sha256(initial_mask),
                "final_mask_sha256": _array_sha256(final_mask),
                "initial_query_delta_sha256": _array_sha256(
                    internal_initial_delta
                ),
                "final_internal_query_delta_sha256": _array_sha256(
                    internal_final_delta
                ),
                "primary_rng_state_sha256": next(iter(primary_hashes)),
                "gibbs_initial_rng_state_sha256": gibbs_initial_hash,
                "gibbs_final_rng_state_sha256": next(iter(gibbs_hashes)),
                "copied_cells": mask_metrics["final_copied_cells"],
                "changed_rows": mask_metrics["changed_rows"],
                "mask_hamming_cells": mask_metrics[
                    "mask_hamming_cells"
                ],
                "mean_copy_blocks_per_participant": mask_metrics[
                    "mean_final_copy_blocks_per_participant"
                ],
                "factor_count": diagnostics["factor_count"],
                "factor_table_entries": diagnostics[
                    "factor_table_entries"
                ],
                "query_factor_count": diagnostics["query_factor_count"],
                "query_factor_table_entries": diagnostics[
                    "query_factor_table_entries"
                ],
                "gibbs_microsteps": diagnostics["gibbs_microsteps"],
                "factor_build_elapsed_sec": diagnostics[
                    "factor_build_elapsed_sec"
                ],
                "gibbs_sample_elapsed_sec": diagnostics[
                    "gibbs_sample_elapsed_sec"
                ],
                "conditional_probability_count": diagnostics[
                    "conditional_probability_count"
                ],
                "conditional_probability_min": diagnostics[
                    "conditional_probability_min"
                ],
                "conditional_probability_max": diagnostics[
                    "conditional_probability_max"
                ],
                "conditional_entropy_mean": diagnostics[
                    "conditional_entropy_mean"
                ],
                "all_conditionals_bidirectional": diagnostics[
                    "all_conditionals_bidirectional"
                ],
                "curvature_weight": diagnostics["curvature_weight"],
                "initial_generation_energy": diagnostics[
                    "initial_generation_energy"
                ],
                "generation_energy": diagnostics[
                    "final_generation_energy"
                ],
                "linear_query_consistency_max_error": diagnostics[
                    "linear_query_consistency_max_error"
                ],
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
            "primary_rng_aligned": len(primary_hashes) == 1,
            "gibbs_rng_aligned": len(gibbs_hashes) == 1,
            "gamma_zero_frame_exact": baseline.equals(reference),
            "gamma_zero_mask_exact": np.array_equal(
                baseline_mask, reference_mask
            ),
            "gamma_zero_diagnostics_exact": gamma_zero_diagnostics_equal,
            "internal_initial_masks_aligned": (
                internal_initial_masks_aligned
            ),
            "initial_query_delta_max_error": initial_delta_error,
            "final_query_delta_max_error": final_delta_error,
            "candidate_energy_identity_error": candidate_energy_error,
            "baseline_candidate_mask_hamming": (
                baseline_candidate_hamming
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
        "copied_cells",
        "mask_hamming_cells",
        "mean_copy_blocks_per_participant",
        "factor_count",
        "factor_table_entries",
        "query_factor_count",
        "query_factor_table_entries",
        "gibbs_microsteps",
        "factor_build_elapsed_sec",
        "gibbs_sample_elapsed_sec",
    )
    paired = {
        metric: _paired(candidate_rows, baseline_rows, metric)
        for metric in metrics
    }
    all_measurements = baseline_rows + candidate_rows
    gates = {
        "primary_rng_aligned": all(
            row["primary_rng_aligned"] for row in pair_rows
        ),
        "gibbs_rng_aligned": all(
            row["gibbs_rng_aligned"] for row in pair_rows
        ),
        "gamma_zero_frame_exact": all(
            row["gamma_zero_frame_exact"] for row in pair_rows
        ),
        "gamma_zero_mask_exact": all(
            row["gamma_zero_mask_exact"] for row in pair_rows
        ),
        "gamma_zero_diagnostics_exact": all(
            row["gamma_zero_diagnostics_exact"] for row in pair_rows
        ),
        "internal_initial_masks_aligned": all(
            row["internal_initial_masks_aligned"] for row in pair_rows
        ),
        "initial_query_delta_max_error": max(
            row["initial_query_delta_max_error"] for row in pair_rows
        ),
        "final_query_delta_max_error": max(
            row["final_query_delta_max_error"] for row in pair_rows
        ),
        "candidate_energy_identity_max_error": max(
            row["candidate_energy_identity_error"] for row in pair_rows
        ),
        "row_delta_sum_max_error": max(
            row["row_delta_sum_max_error"] for row in all_measurements
        ),
        "quadratic_identity_max_error": max(
            row["quadratic_identity_error"] for row in all_measurements
        ),
        "gain_identity_max_error": max(
            row["gain_identity_error"] for row in all_measurements
        ),
        "linear_query_consistency_max_error": max(
            row["linear_query_consistency_max_error"]
            for row in all_measurements
        ),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"]
            for row in all_measurements
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
        "baseline_conditional": _conditional_summary(baseline_rows),
        "candidate_conditional": _conditional_summary(candidate_rows),
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
    parser.add_argument("--proposals", type=int, default=FORMAL_PROPOSALS)
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
        default="outputs/generation_curvature_gibbs/formal.json",
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
    if not 0 <= args.max_factor_order <= 8:
        parser.error("--max-factor-order 必须在 0..8 内")

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
            aligned = all(
                value == initial_state_hashes[0]
                for value in generated_initial_hashes
            )
            if not aligned:
                raise RuntimeError(
                    "直接 marginal 状态与标准闭环初始表不一致"
                )
            state_initialization_aligned &= aligned

        for state_index, state_rounds, state, generation in prepared_states:
            result = _probe_state(
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
            result["state_generation"] = generation
            states.append(result)
            print(
                f"seed={seed:02d} state={state_rounds:03d} "
                f"gain Δ={result['paired']['net_gain']['difference']['mean']:+.4f} "
                f"positive Δ={result['paired']['positive_gain']['difference']['mean']:+.4f}",
                flush=True,
            )

    all_baseline = [
        row for state in states for row in state["baseline_rows"]
    ]
    all_candidate = [
        row for state in states for row in state["candidate_rows"]
    ]
    metrics = tuple(states[0]["paired"])
    global_paired = {
        metric: _paired(all_candidate, all_baseline, metric)
        for metric in metrics
    }
    late_baseline = [
        row
        for state in states
        if state["state_rounds"] == 500
        for row in state["baseline_rows"]
    ]
    late_candidate = [
        row
        for state in states
        if state["state_rounds"] == 500
        for row in state["candidate_rows"]
    ]
    initial_baseline = [
        row
        for state in states
        if state["state_rounds"] == 0
        for row in state["baseline_rows"]
    ]
    initial_candidate = [
        row
        for state in states
        if state["state_rounds"] == 0
        for row in state["candidate_rows"]
    ]
    late_paired = (
        {
            metric: _paired(late_candidate, late_baseline, metric)
            for metric in metrics
        }
        if late_baseline else None
    )
    initial_paired = (
        {
            metric: _paired(initial_candidate, initial_baseline, metric)
            for metric in metrics
        }
        if initial_baseline else None
    )
    decision = (
        _decision_from_state_results(states, late_paired, initial_paired)
        if late_paired is not None and initial_paired is not None
        else {
            "decision": "incomplete_state_set_no_formal_decision",
            "late_improved_seed_states": None,
            "late_positive_rate_non_decreasing_seed_states": None,
            "n_late_seed_states": sum(
                state["state_rounds"] == 500 for state in states
            ),
            "late_net_gain_difference": None,
            "late_positive_gain_rate_difference": None,
            "initial_net_gain_relative_change": None,
            "initial_deterioration_over_5pct_risk": None,
            "late_baseline_conditional": None,
            "late_candidate_conditional": None,
            "late_conditional_entropy_relative_change": None,
            "conditional_entropy_concentration_risk": None,
        }
    )
    gates = {
        "state_initialization_aligned": state_initialization_aligned,
        "all_primary_rng_aligned": all(
            state["gates"]["primary_rng_aligned"] for state in states
        ),
        "all_gibbs_rng_aligned": all(
            state["gates"]["gibbs_rng_aligned"] for state in states
        ),
        "gamma_zero_frame_exact": all(
            state["gates"]["gamma_zero_frame_exact"] for state in states
        ),
        "gamma_zero_mask_exact": all(
            state["gates"]["gamma_zero_mask_exact"] for state in states
        ),
        "gamma_zero_diagnostics_exact": all(
            state["gates"]["gamma_zero_diagnostics_exact"]
            for state in states
        ),
        "internal_initial_masks_aligned": all(
            state["gates"]["internal_initial_masks_aligned"]
            for state in states
        ),
        "initial_query_delta_max_error": max(
            state["gates"]["initial_query_delta_max_error"]
            for state in states
        ),
        "final_query_delta_max_error": max(
            state["gates"]["final_query_delta_max_error"]
            for state in states
        ),
        "candidate_energy_identity_max_error": max(
            state["gates"]["candidate_energy_identity_max_error"]
            for state in states
        ),
        "row_delta_sum_max_error": max(
            state["gates"]["row_delta_sum_max_error"]
            for state in states
        ),
        "quadratic_identity_max_error": max(
            state["gates"]["quadratic_identity_max_error"]
            for state in states
        ),
        "gain_identity_max_error": max(
            state["gates"]["gain_identity_max_error"]
            for state in states
        ),
        "linear_query_consistency_max_error": max(
            state["gates"]["linear_query_consistency_max_error"]
            for state in states
        ),
        "all_conditionals_bidirectional": all(
            state["gates"]["all_conditionals_bidirectional"]
            for state in states
        ),
    }
    gate_passed = (
        gates["state_initialization_aligned"]
        and gates["all_primary_rng_aligned"]
        and gates["all_gibbs_rng_aligned"]
        and gates["gamma_zero_frame_exact"]
        and gates["gamma_zero_mask_exact"]
        and gates["gamma_zero_diagnostics_exact"]
        and gates["internal_initial_masks_aligned"]
        and gates["initial_query_delta_max_error"] <= 1e-10
        and gates["final_query_delta_max_error"] <= 1e-10
        and gates["candidate_energy_identity_max_error"] <= 1e-10
        and gates["row_delta_sum_max_error"] <= 1e-10
        and gates["quadratic_identity_max_error"] <= 1e-10
        and gates["gain_identity_max_error"] <= 1e-10
        and gates["linear_query_consistency_max_error"] <= 1e-10
        and gates["all_conditionals_bidirectional"]
    )
    final_decision = (
        decision["decision"]
        if formal_protocol_matches and gate_passed
        else (
            "diagnostic_gate_failed"
            if formal_protocol_matches else "non_formal_run_no_decision"
        )
    )
    summary = {
        "experiment": "generation_curvature_gibbs_frozen",
        "issue": 18,
        "formal_protocol_matches": formal_protocol_matches,
        "diagnostic_gate_passed": gate_passed,
        "decision": final_decision,
        "scope": "fixed_exact_target_frozen_proposals_no_acceptance",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "seeds": args.seeds,
        "state_rounds": args.state_rounds,
        "proposals_per_state": args.proposals,
        "n_seed_states": len(states),
        "n_paired_proposals": len(all_baseline),
        "temperature": args.temperature,
        "sweeps": args.sweeps,
        "baseline_curvature_weight": BASELINE_CURVATURE,
        "candidate_curvature_weight": CANDIDATE_CURVATURE,
        "max_factor_order": args.max_factor_order,
        "rho": RHO,
        "eta": ETA,
        "mu": 0.0,
        "device": args.device,
        "real_data_access": "none",
        "environment": environment,
        "public_input_sha256": {
            str(path): _sha256_file(path)
            for path in (SCHEMA_PATH, QUERY_PATH, MARGINALS_PATH)
        },
        "elapsed_sec": time.perf_counter() - experiment_start,
        "gates": gates,
        "preregistered_decision": decision,
        "global_paired": global_paired,
        "initial_state_paired": initial_paired,
        "late_state_paired": late_paired,
        "global_baseline_conditional": _conditional_summary(all_baseline),
        "global_candidate_conditional": _conditional_summary(all_candidate),
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

    print("\n===== 整代曲率 Gibbs 冻结实验 =====")
    for label, paired in (
        ("初始态", initial_paired),
        ("500 轮态", late_paired),
        ("全局", global_paired),
    ):
        if paired is None:
            continue
        comparison = paired["net_gain"]
        positive = paired["positive_gain"]
        print(
            f"{label}: gain "
            f"{comparison['baseline']['mean']:.6g} -> "
            f"{comparison['candidate']['mean']:.6g} "
            f"(Δ={comparison['difference']['mean']:+.6g}), "
            f"positive Δ={positive['difference']['mean']:+.6g}"
        )
    print(f"诊断门禁：{gate_passed}")
    print(f"预注册判断：{final_decision}")
    print(f"详细结果：{output_path}")


if __name__ == "__main__":
    main()
