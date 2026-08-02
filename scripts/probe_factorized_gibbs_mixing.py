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

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.factorized_diffusion import (
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


def _git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
    probe_alpha = 2.0 if state_rounds == 0 else 10.0
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
    direction_reference_scale = None
    reference_scale_proposal_index = None
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
        "state_sha256": hashlib.sha256(
            state.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
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
        "kernel_summary": kernel_summary,
        "proposal_summary": proposal_summary,
        "paired": paired,
        "expected_direction_gap_recovery": recovery,
        "proposal_rows": proposal_rows,
        "elapsed_sec": time.perf_counter() - probe_start,
    }

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
        "states": states,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
