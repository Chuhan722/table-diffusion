"""比较 gamma=0/1 整代曲率 Gibbs 的无接受长期动力学。

两侧使用相同公开输入、初始化、主随机流和按轮地址化的 Gibbs 随机源。每个
proposal 无条件成为下一状态；loss 只记录，不参与接受、停止、重试或输出选择。
脚本不提供真实训练/测试表路径。
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.factorized_diffusion import (
    DEFAULT_LOGIT_CLIP,
    evolve_step_factorized_gibbs,
)
from table_diffevo.generation_curvature import (
    evolve_step_generation_curvature_gibbs,
)
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized


SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
QUERY_PATH = Path("configs/test_300x10/measured_50query.json")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
N_RECORDS = 300
FORMAL_SEEDS = list(range(20))
FORMAL_ROUNDS = 1000
FORMAL_TEMPERATURE = 2.0
FORMAL_SWEEPS = 8
FORMAL_REFERENCE_ROUNDS = 20
MAX_FACTOR_ORDER = 3
GIBBS_LOGIT_CLIP = float(DEFAULT_LOGIT_CLIP)
RHO = 0.01
ETA = 0.5
MU = 0.01
BASELINE_CURVATURE = 0.0
CANDIDATE_CURVATURE = 1.0
PRIMARY_TAIL_WINDOW = 250
SECONDARY_TAIL_WINDOW = 100


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


def _query_vector_sha256(values):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("查询向量必须是一维整数数组")
    canonical = np.ascontiguousarray(array, dtype="<i8")
    payload = (
        np.asarray(canonical.shape, dtype="<i8").tobytes()
        + canonical.tobytes()
    )
    return _sha256_bytes(payload)


def _rng_state_json(rng):
    return json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )


def _rng_state_sha256(rng):
    return _sha256_bytes(_rng_state_json(rng).encode("utf-8"))


def _gibbs_round_seed(seed, round_index):
    sequence = np.random.SeedSequence([
        int(seed),
        int(round_index),
        0x435552564544594E,
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
    result = {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit if commit_code == 0 else None,
        "git_worktree_clean": status_code == 0 and status == "",
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        result.update({
            "torch": None,
            "torch_cuda_runtime": None,
            "cuda_available": False,
            "gpu": None,
        })
    else:
        cuda_available = bool(torch.cuda.is_available())
        result.update({
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": (
                torch.cuda.get_device_name(0)
                if device == "cuda" and cuda_available else None
            ),
        })
    return result


def _evaluate(state, target, queries, schema, device):
    return evaluate_vectorized(
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


def _sample_round_donors(
    state,
    fitness,
    schema,
    *,
    rng,
    round_index,
    rounds,
    device,
):
    use_torch = device in ("cuda", "cpu")
    distances = pairwise_block_distance(
        state,
        state,
        schema,
        device=device,
        return_tensor=use_torch,
    )
    progress = round_index / (rounds - 1) if rounds > 1 else 1.0
    alpha = 2.0 + 8.0 * progress
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
    return donors, donor_indices


def _direction_for_round(
    state,
    donors,
    residual,
    schema,
    queries,
    *,
    temperature,
    reference_scale,
    device,
):
    start = time.perf_counter()
    directions = compute_copy_direction_scores(
        state,
        donors,
        schema,
        queries,
        residual,
        batch_size=256,
        device=device,
    )
    elapsed = time.perf_counter() - start
    attr_names = schema.attribute_names()
    differs = np.column_stack([
        state[attr].reset_index(drop=True).to_numpy()
        != donors[attr].to_numpy()
        for attr in attr_names
    ])
    discovered_scale = reference_scale
    if discovered_scale is None:
        candidate_scale = direction_rms_scale(directions[differs])
        if candidate_scale > 0.0:
            discovered_scale = candidate_scale
    strength = (
        temperature / discovered_scale
        if discovered_scale is not None else 0.0
    )
    return directions, discovered_scale, strength, elapsed


def _verify_gamma_zero_reference(
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
):
    """逐轮对拍 gamma=0 研究更新与既有因子 Gibbs。"""
    reference_rng = np.random.default_rng(seed)
    candidate_rng = np.random.default_rng(seed)
    reference = init_synthetic_table(
        N_RECORDS, schema, reference_rng, marginals=marginals
    )
    candidate = init_synthetic_table(
        N_RECORDS, schema, candidate_rng, marginals=marginals
    )
    if not reference.equals(candidate):
        raise RuntimeError("gamma=0 预检的初始表不一致")
    reference_q, reference_residual, reference_fitness = _evaluate(
        reference, target, queries, schema, device
    )
    candidate_q, candidate_residual, candidate_fitness = _evaluate(
        candidate, target, queries, schema, device
    )
    if (
        not np.array_equal(reference_q, candidate_q)
        or not np.array_equal(reference_residual, candidate_residual)
        or not np.array_equal(reference_fitness, candidate_fitness)
        or _rng_state_sha256(reference_rng)
        != _rng_state_sha256(candidate_rng)
    ):
        raise RuntimeError("gamma=0 预检的初始化状态不一致")
    reference_scale = None
    candidate_scale = None
    loss_history = [float(compute_loss(target, reference_q))]
    gibbs_seed_digest = hashlib.sha256()
    gibbs_endpoint_digest = hashlib.sha256()
    common_keys = (
        "participating_rows",
        "active_gibbs_rows",
        "active_blocks",
        "factor_count",
        "factor_table_entries",
        "gibbs_microsteps",
    )

    for round_index in range(rounds):
        reference_donors, reference_indices = _sample_round_donors(
            reference,
            reference_fitness,
            schema,
            rng=reference_rng,
            round_index=round_index,
            rounds=rounds,
            device=device,
        )
        candidate_donors, candidate_indices = _sample_round_donors(
            candidate,
            candidate_fitness,
            schema,
            rng=candidate_rng,
            round_index=round_index,
            rounds=rounds,
            device=device,
        )
        if not np.array_equal(reference_indices, candidate_indices):
            raise RuntimeError("gamma=0 预检的 donor 索引不一致")
        reference_directions, discovered_scale, reference_strength, _ = (
            _direction_for_round(
                reference,
                reference_donors,
                reference_residual,
                schema,
                queries,
                temperature=temperature,
                reference_scale=reference_scale,
                device=device,
            )
        )
        (
            candidate_directions,
            candidate_discovered_scale,
            candidate_strength,
            _,
        ) = _direction_for_round(
            candidate,
            candidate_donors,
            candidate_residual,
            schema,
            queries,
            temperature=temperature,
            reference_scale=candidate_scale,
            device=device,
        )
        if (
            not np.array_equal(reference_directions, candidate_directions)
            or discovered_scale != candidate_discovered_scale
            or reference_strength != candidate_strength
        ):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮方向状态不一致"
            )
        reference_scale = discovered_scale
        candidate_scale = candidate_discovered_scale
        gibbs_seed = _gibbs_round_seed(seed, round_index)
        gibbs_seed_digest.update(np.uint64(gibbs_seed).tobytes())
        reference_gibbs = np.random.default_rng(gibbs_seed)
        candidate_gibbs = np.random.default_rng(gibbs_seed)
        reference_proposal, reference_diagnostics = (
            evolve_step_factorized_gibbs(
                reference,
                reference_donors,
                schema,
                queries,
                reference_residual,
                rho=RHO,
                eta=ETA,
                mu=MU,
                copy_direction_scores=reference_directions,
                copy_direction_strength=reference_strength,
                n_sweeps=sweeps,
                rng=reference_rng,
                gibbs_rng=reference_gibbs,
                max_factor_order=MAX_FACTOR_ORDER,
                gibbs_logit_clip=GIBBS_LOGIT_CLIP,
            )
        )
        candidate_proposal, candidate_diagnostics = (
            evolve_step_generation_curvature_gibbs(
                candidate,
                candidate_donors,
                schema,
                queries,
                candidate_residual,
                rho=RHO,
                eta=ETA,
                mu=MU,
                copy_direction_scores=candidate_directions,
                copy_direction_strength=candidate_strength,
                n_sweeps=sweeps,
                curvature_weight=BASELINE_CURVATURE,
                rng=candidate_rng,
                gibbs_rng=candidate_gibbs,
                max_factor_order=MAX_FACTOR_ORDER,
                gibbs_logit_clip=GIBBS_LOGIT_CLIP,
            )
        )
        if not reference_proposal.equals(candidate_proposal):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮最终表不一致"
            )
        if (
            _rng_state_sha256(reference_rng)
            != _rng_state_sha256(candidate_rng)
        ):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮主 RNG 不一致"
            )
        if (
            _rng_state_sha256(reference_gibbs)
            != _rng_state_sha256(candidate_gibbs)
        ):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮 Gibbs RNG 不一致"
            )
        gibbs_endpoint_digest.update(
            _rng_state_json(reference_gibbs).encode("utf-8")
        )
        if any(
            reference_diagnostics[key] != candidate_diagnostics[key]
            for key in common_keys
        ):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮共同诊断不一致"
            )
        if (
            candidate_diagnostics[
                "gamma_zero_reference_probability_max_error"
            ] != 0.0
            or candidate_diagnostics["conditional_logit_clipped_count"] != 0
            or not candidate_diagnostics["all_conditionals_bidirectional"]
        ):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮条件概率门禁失败"
            )

        reference = reference_proposal
        candidate = candidate_proposal
        reference_q, reference_residual, reference_fitness = _evaluate(
            reference, target, queries, schema, device
        )
        candidate_q, candidate_residual, candidate_fitness = _evaluate(
            candidate, target, queries, schema, device
        )
        reference_loss = float(compute_loss(target, reference_q))
        candidate_loss = float(compute_loss(target, candidate_q))
        if (
            not np.array_equal(reference_q, candidate_q)
            or not np.array_equal(reference_residual, candidate_residual)
            or not np.array_equal(reference_fitness, candidate_fitness)
            or reference_loss != candidate_loss
        ):
            raise RuntimeError(
                f"gamma=0 预检第 {round_index} 轮评价状态不一致"
            )
        loss_history.append(reference_loss)

    return {
        "passed": True,
        "seed": int(seed),
        "rounds": int(rounds),
        "final_table_sha256": _frame_sha256(reference),
        "loss_history": loss_history,
        "primary_rng_state_sha256": _rng_state_sha256(reference_rng),
        "gibbs_round_seed_sha256": gibbs_seed_digest.hexdigest(),
        "gibbs_round_endpoint_sha256": gibbs_endpoint_digest.hexdigest(),
        "direction_reference_scale": reference_scale,
        "table_exact_rounds": int(rounds),
        "loss_exact_rounds": int(rounds),
        "primary_rng_exact_rounds": int(rounds),
        "gibbs_rng_exact_rounds": int(rounds),
        "common_diagnostics_exact_rounds": int(rounds),
        "gamma_zero_conditional_probability_max_error": 0.0,
        "logit_clip_hits": 0,
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
    curvature_weight,
    device,
    record_query_clock=False,
):
    if not isinstance(record_query_clock, (bool, np.bool_)):
        raise ValueError("record_query_clock 必须是布尔值")
    rng = np.random.default_rng(seed)
    state = init_synthetic_table(
        N_RECORDS, schema, rng, marginals=marginals
    )
    initial_table_sha256 = _frame_sha256(state)
    q, residual, fitness = _evaluate(
        state, target, queries, schema, device
    )
    initial_loss = float(compute_loss(target, q))
    loss_history = [initial_loss]
    gain_history = []
    changed_cells_history = []
    unique_history = [int(len(state.value_counts()))]
    zero_residual_rounds = 0
    direction_reference_scale = None
    direction_reference_scale_round = None
    direction_elapsed = 0.0
    factor_build_elapsed = 0.0
    gibbs_sample_elapsed = 0.0
    participating_rows = 0
    active_gibbs_rows = 0
    active_blocks = 0
    factor_count = 0
    factor_table_entries = 0
    query_factor_count = 0
    query_factor_table_entries = 0
    gibbs_microsteps = 0
    conditional_count = 0
    conditional_entropy_sum = 0.0
    conditional_probability_min = None
    conditional_probability_max = None
    conditional_logit_abs_max = 0.0
    conditional_logit_clipped_count = 0
    all_conditionals_bidirectional = True
    gamma_zero_probability_max_error = 0.0
    conditional_probability_count_history = []
    conditional_entropy_mean_history = []
    conditional_probability_min_history = []
    conditional_probability_max_history = []
    conditional_logit_abs_max_history = []
    conditional_logit_clipped_count_history = []
    conditional_bidirectional_history = []
    if record_query_clock:
        count_residual = target.astype(float, copy=False) - q
        count_residual_l2_squared_history = [float(
            np.dot(count_residual, count_residual)
        )]
        query_count_history = [
            q.astype(np.int64, copy=False).tolist()
        ]
        query_state_sha256_history = [_query_vector_sha256(q)]
        query_delta_l2_squared_history = []
        linear_gain_history = []
        quadratic_cost_history = []
        gain_identity_error_history = []
        cumulative_query_quadratic_variation_history = [0.0]
    gibbs_seed_digest = hashlib.sha256()
    gibbs_endpoint_digest = hashlib.sha256()
    start = time.perf_counter()

    for round_index in range(rounds):
        if np.all(residual == 0.0):
            zero_residual_rounds += 1
        donors, _ = _sample_round_donors(
            state,
            fitness,
            schema,
            rng=rng,
            round_index=round_index,
            rounds=rounds,
            device=device,
        )
        directions, discovered_scale, strength, elapsed = (
            _direction_for_round(
                state,
                donors,
                residual,
                schema,
                queries,
                temperature=temperature,
                reference_scale=direction_reference_scale,
                device=device,
            )
        )
        direction_elapsed += elapsed
        if (
            direction_reference_scale is None
            and discovered_scale is not None
        ):
            direction_reference_scale_round = round_index
        direction_reference_scale = discovered_scale

        gibbs_seed = _gibbs_round_seed(seed, round_index)
        gibbs_seed_digest.update(np.uint64(gibbs_seed).tobytes())
        gibbs_rng = np.random.default_rng(gibbs_seed)
        proposal, diagnostics = evolve_step_generation_curvature_gibbs(
            state,
            donors,
            schema,
            queries,
            residual,
            rho=RHO,
            eta=ETA,
            mu=MU,
            copy_direction_scores=directions,
            copy_direction_strength=strength,
            n_sweeps=sweeps,
            curvature_weight=curvature_weight,
            rng=rng,
            gibbs_rng=gibbs_rng,
            max_factor_order=MAX_FACTOR_ORDER,
            gibbs_logit_clip=GIBBS_LOGIT_CLIP,
        )
        gibbs_endpoint_digest.update(
            _rng_state_json(gibbs_rng).encode("utf-8")
        )
        proposal_q, proposal_residual, proposal_fitness = _evaluate(
            proposal, target, queries, schema, device
        )
        proposal_loss = float(compute_loss(target, proposal_q))
        actual_gain = loss_history[-1] - proposal_loss
        gain_history.append(actual_gain)
        if record_query_clock:
            query_delta = (
                proposal_q.astype(float, copy=False)
                - q.astype(float, copy=False)
            )
            count_residual = target.astype(float, copy=False) - q
            query_delta_l2_squared = float(
                np.dot(query_delta, query_delta)
            )
            linear_gain = float(np.dot(count_residual, query_delta))
            quadratic_cost = 0.5 * query_delta_l2_squared
            identity_error = float(
                actual_gain - (linear_gain - quadratic_cost)
            )
            query_delta_l2_squared_history.append(
                query_delta_l2_squared
            )
            linear_gain_history.append(linear_gain)
            quadratic_cost_history.append(quadratic_cost)
            gain_identity_error_history.append(identity_error)
            cumulative_query_quadratic_variation_history.append(float(
                cumulative_query_quadratic_variation_history[-1]
                + query_delta_l2_squared
            ))
            proposal_count_residual = (
                target.astype(float, copy=False) - proposal_q
            )
            count_residual_l2_squared_history.append(float(
                np.dot(
                    proposal_count_residual,
                    proposal_count_residual,
                )
            ))
            query_state_sha256_history.append(
                _query_vector_sha256(proposal_q)
            )
            query_count_history.append(
                proposal_q.astype(np.int64, copy=False).tolist()
            )
        changed_cells_history.append(int(
            (
                proposal.reset_index(drop=True)
                != state.reset_index(drop=True)
            ).to_numpy().sum()
        ))
        state = proposal
        q = proposal_q
        residual = proposal_residual
        fitness = proposal_fitness
        loss_history.append(proposal_loss)
        unique_history.append(int(len(state.value_counts())))

        factor_build_elapsed += diagnostics["factor_build_elapsed_sec"]
        gibbs_sample_elapsed += diagnostics["gibbs_sample_elapsed_sec"]
        participating_rows += diagnostics["participating_rows"]
        active_gibbs_rows += diagnostics["active_gibbs_rows"]
        active_blocks += diagnostics["active_blocks"]
        factor_count += diagnostics["factor_count"]
        factor_table_entries += diagnostics["factor_table_entries"]
        query_factor_count += diagnostics["query_factor_count"]
        query_factor_table_entries += diagnostics[
            "query_factor_table_entries"
        ]
        gibbs_microsteps += diagnostics["gibbs_microsteps"]
        count = diagnostics["conditional_probability_count"]
        conditional_probability_count_history.append(int(count))
        conditional_entropy_mean_history.append(
            diagnostics["conditional_entropy_mean"]
        )
        conditional_probability_min_history.append(
            diagnostics["conditional_probability_min"]
        )
        conditional_probability_max_history.append(
            diagnostics["conditional_probability_max"]
        )
        conditional_logit_abs_max_history.append(
            diagnostics["conditional_logit_abs_max"]
        )
        conditional_logit_clipped_count_history.append(int(
            diagnostics["conditional_logit_clipped_count"]
        ))
        conditional_bidirectional_history.append(bool(
            diagnostics["all_conditionals_bidirectional"]
        ))
        conditional_count += count
        if count:
            conditional_entropy_sum += (
                diagnostics["conditional_entropy_mean"] * count
            )
            conditional_probability_min = (
                diagnostics["conditional_probability_min"]
                if conditional_probability_min is None
                else min(
                    conditional_probability_min,
                    diagnostics["conditional_probability_min"],
                )
            )
            conditional_probability_max = (
                diagnostics["conditional_probability_max"]
                if conditional_probability_max is None
                else max(
                    conditional_probability_max,
                    diagnostics["conditional_probability_max"],
                )
            )
            conditional_logit_abs_max = max(
                conditional_logit_abs_max,
                diagnostics["conditional_logit_abs_max"],
            )
        conditional_logit_clipped_count += diagnostics[
            "conditional_logit_clipped_count"
        ]
        all_conditionals_bidirectional &= diagnostics[
            "all_conditionals_bidirectional"
        ]
        if curvature_weight == 0.0:
            gamma_zero_probability_max_error = max(
                gamma_zero_probability_max_error,
                diagnostics[
                    "gamma_zero_reference_probability_max_error"
                ],
            )

    elapsed = time.perf_counter() - start
    post_update_losses = np.asarray(loss_history[1:], dtype=float)
    gains = np.asarray(gain_history, dtype=float)
    positive_gains = gains[gains > 0.0]
    negative_gains = gains[gains < 0.0]
    tail_counts = conditional_probability_count_history[
        -PRIMARY_TAIL_WINDOW:
    ]
    tail_entropy = conditional_entropy_mean_history[
        -PRIMARY_TAIL_WINDOW:
    ]
    tail_conditional_count = int(sum(tail_counts))
    tail_conditional_entropy_mean = (
        float(sum(
            entropy * count
            for entropy, count in zip(tail_entropy, tail_counts)
            if count > 0
        ) / tail_conditional_count)
        if tail_conditional_count else None
    )
    label = (
        "generation_linear_gamma0"
        if curvature_weight == 0.0
        else "generation_curvature_gamma1"
    )
    result = {
        "seed": int(seed),
        "name": label,
        "curvature_weight": float(curvature_weight),
        "temperature": float(temperature),
        "sweeps": int(sweeps),
        "rounds_run": len(gain_history),
        "initial_loss": initial_loss,
        "final_loss": float(loss_history[-1]),
        "best_loss_diagnostic_only": float(min(loss_history)),
        "mean_trajectory_loss": float(post_update_losses.mean()),
        "late_100_mean_loss": float(
            post_update_losses[-SECONDARY_TAIL_WINDOW:].mean()
        ),
        "late_250_mean_loss": float(
            post_update_losses[-PRIMARY_TAIL_WINDOW:].mean()
        ),
        "mean_raw_gain": float(gains.mean()),
        "positive_gain_rate": float(np.mean(gains > 0.0)),
        "zero_gain_rate": float(np.mean(gains == 0.0)),
        "negative_gain_rate": float(np.mean(gains < 0.0)),
        "mean_positive_gain": (
            float(positive_gains.mean()) if len(positive_gains) else 0.0
        ),
        "mean_negative_gain": (
            float(negative_gains.mean()) if len(negative_gains) else 0.0
        ),
        "maximum_loss": float(max(loss_history)),
        "mean_changed_cells": float(np.mean(changed_cells_history)),
        "initial_unique_states": int(unique_history[0]),
        "final_unique_states": int(unique_history[-1]),
        "mean_unique_states": float(np.mean(unique_history[1:])),
        "zero_residual_rounds": int(zero_residual_rounds),
        "direction_reference_scale": direction_reference_scale,
        "direction_reference_scale_round": (
            int(direction_reference_scale_round)
            if direction_reference_scale_round is not None else None
        ),
        "direction_elapsed_sec": float(direction_elapsed),
        "factor_build_elapsed_sec": float(factor_build_elapsed),
        "gibbs_sample_elapsed_sec": float(gibbs_sample_elapsed),
        "participating_rows": int(participating_rows),
        "active_gibbs_rows": int(active_gibbs_rows),
        "active_blocks": int(active_blocks),
        "factor_count": int(factor_count),
        "factor_table_entries": int(factor_table_entries),
        "query_factor_count": int(query_factor_count),
        "query_factor_table_entries": int(query_factor_table_entries),
        "gibbs_microsteps": int(gibbs_microsteps),
        "conditional_probability_count": int(conditional_count),
        "conditional_entropy_mean": (
            float(conditional_entropy_sum / conditional_count)
            if conditional_count else None
        ),
        "late_250_conditional_probability_count": (
            tail_conditional_count
        ),
        "late_250_conditional_entropy_mean": (
            tail_conditional_entropy_mean
        ),
        "conditional_probability_min": conditional_probability_min,
        "conditional_probability_max": conditional_probability_max,
        "conditional_logit_abs_max": float(conditional_logit_abs_max),
        "conditional_logit_clipped_count": int(
            conditional_logit_clipped_count
        ),
        "all_conditionals_bidirectional": bool(
            all_conditionals_bidirectional
        ),
        "gamma_zero_reference_probability_max_error": (
            float(gamma_zero_probability_max_error)
            if curvature_weight == 0.0 else None
        ),
        "elapsed_sec": float(elapsed),
        "initial_table_sha256": initial_table_sha256,
        "final_table_sha256": _frame_sha256(state),
        "primary_rng_state_sha256": _rng_state_sha256(rng),
        "gibbs_round_seed_sha256": gibbs_seed_digest.hexdigest(),
        "gibbs_round_endpoint_sha256": gibbs_endpoint_digest.hexdigest(),
        "loss_history": loss_history,
        "gain_history": gain_history,
        "changed_cells_history": changed_cells_history,
        "unique_states_history": unique_history,
        "conditional_probability_count_history": (
            conditional_probability_count_history
        ),
        "conditional_entropy_mean_history": conditional_entropy_mean_history,
        "conditional_probability_min_history": (
            conditional_probability_min_history
        ),
        "conditional_probability_max_history": (
            conditional_probability_max_history
        ),
        "conditional_logit_abs_max_history": (
            conditional_logit_abs_max_history
        ),
        "conditional_logit_clipped_count_history": (
            conditional_logit_clipped_count_history
        ),
        "conditional_bidirectional_history": (
            conditional_bidirectional_history
        ),
    }
    if record_query_clock:
        result.update({
            "query_clock_recorded": True,
            "query_count_history": query_count_history,
            "query_state_sha256_history": query_state_sha256_history,
            "count_residual_l2_squared_history": (
                count_residual_l2_squared_history
            ),
            "query_delta_l2_squared_history": (
                query_delta_l2_squared_history
            ),
            "linear_gain_history": linear_gain_history,
            "quadratic_cost_history": quadratic_cost_history,
            "gain_identity_error_history": (
                gain_identity_error_history
            ),
            "gain_identity_max_abs_error": float(max(
                (abs(value) for value in gain_identity_error_history),
                default=0.0,
            )),
            "cumulative_query_quadratic_variation_history": (
                cumulative_query_quadratic_variation_history
            ),
        })
    print(
        f"seed={seed:02d} {label:<28} "
        f"tail250={result['late_250_mean_loss']:.3f} "
        f"final={result['final_loss']:.3f} "
        f"raw_pos={result['positive_gain_rate']:.1%} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _summarize(values):
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("汇总值必须非空且全部有限")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _paired(candidate, baseline, metric, *, lower_is_better):
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
    if lower_is_better is None:
        preference = "descriptive_only"
        wins = None
        ties = None
        losses = None
    else:
        better = difference < 0.0 if lower_is_better else difference > 0.0
        worse = difference > 0.0 if lower_is_better else difference < 0.0
        preference = (
            "lower_is_better" if lower_is_better else "higher_is_better"
        )
        wins = int(np.sum(better))
        ties = int(np.sum(difference == 0.0))
        losses = int(np.sum(worse))
    return {
        "metric": metric,
        "preference": preference,
        "lower_is_better": lower_is_better,
        "baseline": _summarize(baseline_values),
        "candidate": _summarize(candidate_values),
        "difference": _summarize(difference),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _aggregate_conditional(rows):
    total = sum(row["conditional_probability_count"] for row in rows)
    nonempty = [
        row for row in rows if row["conditional_probability_count"] > 0
    ]
    if total == 0:
        return {
            "n_microsteps": 0,
            "mean_entropy": None,
            "min_probability": None,
            "max_probability": None,
            "max_abs_logit": 0.0,
            "logit_clip_hits": 0,
            "all_bidirectional": True,
        }
    entropy_sum = sum(
        row["conditional_entropy_mean"]
        * row["conditional_probability_count"]
        for row in nonempty
    )
    return {
        "n_microsteps": int(total),
        "mean_entropy": float(entropy_sum / total),
        "min_probability": float(min(
            row["conditional_probability_min"] for row in nonempty
        )),
        "max_probability": float(max(
            row["conditional_probability_max"] for row in nonempty
        )),
        "max_abs_logit": float(max(
            row["conditional_logit_abs_max"] for row in rows
        )),
        "logit_clip_hits": int(sum(
            row["conditional_logit_clipped_count"] for row in rows
        )),
        "all_bidirectional": all(
            row["all_conditionals_bidirectional"] for row in rows
        ),
    }


def _aggregate_tail_conditional(rows, window=PRIMARY_TAIL_WINDOW):
    """按每条轨迹末尾固定窗口聚合逐轮条件诊断。"""
    if window <= 0:
        raise ValueError("条件诊断窗口必须为正整数")
    counts = []
    entropies = []
    minimums = []
    maximums = []
    logits = []
    clipped_counts = []
    bidirectional = []
    history_keys = (
        "conditional_probability_count_history",
        "conditional_entropy_mean_history",
        "conditional_probability_min_history",
        "conditional_probability_max_history",
        "conditional_logit_abs_max_history",
        "conditional_logit_clipped_count_history",
        "conditional_bidirectional_history",
    )
    for row in rows:
        lengths = {len(row[key]) for key in history_keys}
        if len(lengths) != 1 or lengths != {row["rounds_run"]}:
            raise ValueError("逐轮条件诊断长度与实际轮数不一致")
        start = max(0, row["rounds_run"] - window)
        counts.extend(row[history_keys[0]][start:])
        entropies.extend(row[history_keys[1]][start:])
        minimums.extend(row[history_keys[2]][start:])
        maximums.extend(row[history_keys[3]][start:])
        logits.extend(row[history_keys[4]][start:])
        clipped_counts.extend(row[history_keys[5]][start:])
        bidirectional.extend(row[history_keys[6]][start:])

    total = int(sum(counts))
    nonempty_indices = [
        index for index, count in enumerate(counts) if count > 0
    ]
    observed_logits = [
        logits[index]
        for index in nonempty_indices
        if logits[index] is not None
    ]
    if any(count < 0 for count in counts):
        raise ValueError("条件概率计数不能为负")
    if total == 0:
        return {
            "window": int(window),
            "n_round_observations": int(len(counts)),
            "n_microsteps": 0,
            "mean_entropy": None,
            "min_probability": None,
            "max_probability": None,
            "max_abs_logit": float(max(observed_logits, default=0.0)),
            "logit_clip_hits": int(sum(clipped_counts)),
            "all_bidirectional": all(bidirectional),
        }
    if any(
        entropies[index] is None
        or minimums[index] is None
        or maximums[index] is None
        for index in nonempty_indices
    ):
        raise ValueError("非空条件诊断缺少熵或概率范围")
    entropy_sum = sum(
        entropies[index] * counts[index]
        for index in nonempty_indices
    )
    return {
        "window": int(window),
        "n_round_observations": int(len(counts)),
        "n_microsteps": total,
        "mean_entropy": float(entropy_sum / total),
        "min_probability": float(min(
            minimums[index] for index in nonempty_indices
        )),
        "max_probability": float(max(
            maximums[index] for index in nonempty_indices
        )),
        "max_abs_logit": float(max(observed_logits, default=0.0)),
        "logit_clip_hits": int(sum(clipped_counts)),
        "all_bidirectional": all(bidirectional),
    }


def _decision_from_runs(baseline, candidate, comparisons):
    primary = comparisons["late_250_mean_loss"]
    baseline_mean = primary["baseline"]["mean"]
    candidate_mean = primary["candidate"]["mean"]
    relative_change = (
        candidate_mean / baseline_mean - 1.0
        if baseline_mean != 0.0 else None
    )
    wins = primary["wins"]
    if (
        relative_change is not None
        and relative_change <= -0.05
        and wins >= 14
    ):
        decision = "supports_unfiltered_curvature_dynamics"
    elif candidate_mean < baseline_mean and wins >= 11:
        decision = "curvature_dynamics_inconclusive"
    else:
        decision = "curvature_dynamics_not_supported"

    baseline_conditional = _aggregate_tail_conditional(baseline)
    candidate_conditional = _aggregate_tail_conditional(candidate)
    entropy_relative_change = (
        candidate_conditional["mean_entropy"]
        / baseline_conditional["mean_entropy"] - 1.0
        if baseline_conditional["mean_entropy"] not in (None, 0.0)
        else None
    )
    baseline_unique = float(np.mean([
        row["final_unique_states"] for row in baseline
    ]))
    candidate_unique = float(np.mean([
        row["final_unique_states"] for row in candidate
    ]))
    unique_relative_change = (
        candidate_unique / baseline_unique - 1.0
        if baseline_unique != 0.0 else None
    )
    baseline_maximum = float(np.mean([
        row["maximum_loss"] for row in baseline
    ]))
    candidate_maximum = float(np.mean([
        row["maximum_loss"] for row in candidate
    ]))
    maximum_relative_change = (
        candidate_maximum / baseline_maximum - 1.0
        if baseline_maximum != 0.0 else None
    )
    return {
        "decision": decision,
        "primary_metric": "late_250_mean_loss",
        "primary_relative_change": relative_change,
        "primary_wins": wins,
        "primary_ties": primary["ties"],
        "primary_losses": primary["losses"],
        "baseline_late_250_conditional": baseline_conditional,
        "candidate_late_250_conditional": candidate_conditional,
        "conditional_entropy_relative_change": entropy_relative_change,
        "conditional_entropy_concentration_risk": (
            entropy_relative_change is not None
            and entropy_relative_change < -0.10
        ),
        "baseline_final_unique_states_mean": baseline_unique,
        "candidate_final_unique_states_mean": candidate_unique,
        "final_unique_states_relative_change": unique_relative_change,
        "support_contraction_risk": (
            unique_relative_change is not None
            and unique_relative_change < -0.05
        ),
        "baseline_maximum_loss_mean": baseline_maximum,
        "candidate_maximum_loss_mean": candidate_maximum,
        "maximum_loss_relative_change": maximum_relative_change,
        "stage_explosion_risk": (
            maximum_relative_change is not None
            and maximum_relative_change > 0.20
        ),
    }


def _assert_finite(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{path} 包含非有限数值 {value!r}")


def _write_json(path, payload):
    _assert_finite(payload)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    parser.add_argument(
        "--temperature", type=float, default=FORMAL_TEMPERATURE
    )
    parser.add_argument("--sweeps", type=int, default=FORMAL_SWEEPS)
    parser.add_argument(
        "--reference-rounds", type=int, default=FORMAL_REFERENCE_ROUNDS
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/generation_curvature_dynamics/"
            "formal_20seed_1000r_tau2_sweep8.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error("--seeds 必须非空、非负且不重复")
    if args.rounds < PRIMARY_TAIL_WINDOW:
        parser.error(
            f"--rounds 不得小于主窗口 {PRIMARY_TAIL_WINDOW}"
        )
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")
    if args.reference_rounds <= 0:
        parser.error("--reference-rounds 必须为正整数")

    formal_protocol_matches = (
        args.seeds == FORMAL_SEEDS
        and args.rounds == FORMAL_ROUNDS
        and args.temperature == FORMAL_TEMPERATURE
        and args.sweeps == FORMAL_SWEEPS
        and args.reference_rounds == FORMAL_REFERENCE_ROUNDS
        and args.device == "cuda"
    )
    output = Path(args.output)
    if output.exists() and (
        formal_protocol_matches or not args.overwrite
    ):
        raise FileExistsError(f"输出已存在，不覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

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

    experiment_start = time.perf_counter()
    preflight = _verify_gamma_zero_reference(
        target,
        queries,
        schema,
        marginals,
        seed=0,
        rounds=args.reference_rounds,
        temperature=args.temperature,
        sweeps=args.sweeps,
        device=args.device,
    )
    runs = {"baseline": [], "candidate": []}
    partial = {
        "experiment": "generation_curvature_unfiltered_dynamics",
        "issue": 24,
        "status": "running",
        "formal_protocol_matches": formal_protocol_matches,
        "environment": environment,
        "preflight": preflight,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "runs": runs,
    }
    _write_json(output, partial)

    for seed in args.seeds:
        runs["baseline"].append(_run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=args.sweeps,
            curvature_weight=BASELINE_CURVATURE,
            device=args.device,
        ))
        partial["completed_baseline_seeds"] = len(runs["baseline"])
        _write_json(output, partial)
        runs["candidate"].append(_run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=args.sweeps,
            curvature_weight=CANDIDATE_CURVATURE,
            device=args.device,
        ))
        partial["completed_candidate_seeds"] = len(runs["candidate"])
        _write_json(output, partial)

    metric_directions = {
        "initial_loss": None,
        "final_loss": True,
        "best_loss_diagnostic_only": True,
        "mean_trajectory_loss": True,
        "late_100_mean_loss": True,
        "late_250_mean_loss": True,
        "mean_raw_gain": False,
        "positive_gain_rate": False,
        "zero_gain_rate": None,
        "negative_gain_rate": True,
        "mean_positive_gain": False,
        "mean_negative_gain": False,
        "maximum_loss": True,
        "mean_changed_cells": None,
        "final_unique_states": False,
        "mean_unique_states": False,
        "conditional_entropy_mean": False,
        "late_250_conditional_entropy_mean": False,
        "conditional_logit_abs_max": None,
        "direction_elapsed_sec": True,
        "factor_build_elapsed_sec": True,
        "gibbs_sample_elapsed_sec": True,
        "gibbs_microsteps": None,
        "elapsed_sec": True,
    }
    comparisons = {
        metric: _paired(
            runs["candidate"],
            runs["baseline"],
            metric,
            lower_is_better=lower_is_better,
        )
        for metric, lower_is_better in metric_directions.items()
    }
    decision = _decision_from_runs(
        runs["baseline"], runs["candidate"], comparisons
    )
    gates = {
        "gamma_zero_reference_preflight": preflight["passed"],
        "initial_table_aligned": all(
            baseline["initial_table_sha256"]
            == candidate["initial_table_sha256"]
            for baseline, candidate in zip(
                runs["baseline"], runs["candidate"]
            )
        ),
        "initial_loss_aligned": all(
            baseline["initial_loss"] == candidate["initial_loss"]
            for baseline, candidate in zip(
                runs["baseline"], runs["candidate"]
            )
        ),
        "direction_reference_scale_aligned": all(
            baseline["direction_reference_scale"]
            == candidate["direction_reference_scale"]
            and baseline["direction_reference_scale_round"] == 0
            and candidate["direction_reference_scale_round"] == 0
            for baseline, candidate in zip(
                runs["baseline"], runs["candidate"]
            )
        ),
        "primary_rng_endpoint_aligned": all(
            baseline["primary_rng_state_sha256"]
            == candidate["primary_rng_state_sha256"]
            for baseline, candidate in zip(
                runs["baseline"], runs["candidate"]
            )
        ),
        "gibbs_round_seed_aligned": all(
            baseline["gibbs_round_seed_sha256"]
            == candidate["gibbs_round_seed_sha256"]
            for baseline, candidate in zip(
                runs["baseline"], runs["candidate"]
            )
        ),
        "all_trajectories_complete": all(
            row["rounds_run"] == args.rounds
            for rows in runs.values() for row in rows
        ),
        "gamma_zero_conditional_probability_max_error": max(
            row["gamma_zero_reference_probability_max_error"]
            for row in runs["baseline"]
        ),
        "logit_clip_hits": sum(
            row["conditional_logit_clipped_count"]
            for rows in runs.values() for row in rows
        ),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"]
            for rows in runs.values() for row in rows
        ),
        "max_abs_effective_logit": max(
            row["conditional_logit_abs_max"]
            for rows in runs.values() for row in rows
        ),
    }
    diagnostic_gate_passed = (
        gates["gamma_zero_reference_preflight"]
        and gates["initial_table_aligned"]
        and gates["initial_loss_aligned"]
        and gates["direction_reference_scale_aligned"]
        and gates["primary_rng_endpoint_aligned"]
        and gates["gibbs_round_seed_aligned"]
        and gates["all_trajectories_complete"]
        and gates[
            "gamma_zero_conditional_probability_max_error"
        ] == 0.0
        and gates["logit_clip_hits"] == 0
        and gates["all_conditionals_bidirectional"]
    )
    final_decision = (
        decision["decision"]
        if formal_protocol_matches and diagnostic_gate_passed
        else (
            "diagnostic_gate_failed"
            if formal_protocol_matches else "non_formal_run_no_decision"
        )
    )
    summary = {
        "experiment": "generation_curvature_unfiltered_dynamics",
        "issue": 24,
        "status": "complete",
        "formal_protocol_matches": formal_protocol_matches,
        "diagnostic_gate_passed": diagnostic_gate_passed,
        "decision": final_decision,
        "scope": (
            "fixed_exact_target_every_proposal_becomes_next_state_"
            "no_loss_acceptance"
        ),
        "primary_endpoint": "late_250_mean_current_loss",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "sweeps": args.sweeps,
        "reference_rounds": args.reference_rounds,
        "baseline_curvature_weight": BASELINE_CURVATURE,
        "candidate_curvature_weight": CANDIDATE_CURVATURE,
        "max_factor_order": MAX_FACTOR_ORDER,
        "rho": RHO,
        "eta": ETA,
        "mu": MU,
        "gibbs_logit_clip": GIBBS_LOGIT_CLIP,
        "gibbs_rng_addressing": "seed_and_round_index",
        "device": args.device,
        "real_data_access": "none",
        "environment": environment,
        "public_input_sha256": {
            str(path): _sha256_file(path)
            for path in (SCHEMA_PATH, QUERY_PATH, MARGINALS_PATH)
        },
        "preflight": preflight,
        "gates": gates,
        "preregistered_decision": decision,
        "comparisons": comparisons,
        "baseline_conditional": _aggregate_conditional(runs["baseline"]),
        "candidate_conditional": _aggregate_conditional(runs["candidate"]),
        "baseline_late_250_conditional": _aggregate_tail_conditional(
            runs["baseline"]
        ),
        "candidate_late_250_conditional": _aggregate_tail_conditional(
            runs["candidate"]
        ),
        "runs": runs,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    _write_json(output, summary)

    primary = comparisons["late_250_mean_loss"]
    final = comparisons["final_loss"]
    print("\n===== 整代曲率 Gibbs 无接受长期动力学 =====")
    print(
        "末 250 轮平均当前 loss: "
        f"{primary['baseline']['mean']:.6g} -> "
        f"{primary['candidate']['mean']:.6g}, "
        f"Δ={primary['difference']['mean']:+.6g}, "
        f"{primary['wins']}/{primary['ties']}/{primary['losses']}"
    )
    print(
        "最终当前 loss: "
        f"{final['baseline']['mean']:.6g} -> "
        f"{final['candidate']['mean']:.6g}, "
        f"Δ={final['difference']['mean']:+.6g}"
    )
    print(f"诊断门禁：{diagnostic_gate_passed}")
    print(f"预注册判断：{final_decision}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
