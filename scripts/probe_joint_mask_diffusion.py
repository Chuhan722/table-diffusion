"""冻结状态下测量独立单块分解与精确联合 mask 扩散核的差距。

每个 proposal 固定当前表、残差、donor、参与记录和 Gumbel 随机量；baseline、
独立加性 Gibbs 核与完整 hybrid Gibbs 核只改变 mask 分布，不执行 generation
acceptance。主实验关闭变异，避免把无方向变异混入复制核归因。
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
    tilted_copy_probabilities,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.generator import init_synthetic_table
from table_diffevo.joint_diffusion import (
    additive_mask_directions,
    baseline_mask_log_probabilities,
    categorical_entropy,
    categorical_kl,
    compute_joint_mask_landscapes,
    gibbs_mask_log_probabilities,
    mask_distribution_diagnostics,
    match_gibbs_strength_for_kl,
    sample_mask_index,
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


def _tau_label(tau):
    return f"{tau:g}".replace(".", "p")


def _config_names(temperatures):
    names = ["baseline"]
    for tau in temperatures:
        label = _tau_label(tau)
        names.extend([
            f"independent_tau_{label}",
            f"joint_tau_{label}",
            f"independent_matched_tau_{label}",
            f"joint_matched_tau_{label}",
        ])
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
        "kl_total": 0.0,
        "active_blocks": 0,
        "entropy_total": 0.0,
        "baseline_entropy_total": 0.0,
        "expected_exact_direction": 0.0,
        "expected_additive_direction": 0.0,
        "selected_exact_direction": 0.0,
        "selected_additive_direction": 0.0,
        "negative_mass_sum": 0.0,
        "neutral_mass_sum": 0.0,
        "positive_mass_sum": 0.0,
        "active_rows": 0,
        "effective_tau_values": [],
        "matched_kl_errors": [],
    }


def _accumulate_kernel(
    accumulator,
    *,
    log_probabilities,
    reference_log_probabilities,
    exact_directions,
    additive_directions,
    selected_index,
    active_blocks,
    effective_tau,
    matched_target_kl=None,
):
    diagnostics = mask_distribution_diagnostics(
        log_probabilities,
        reference_log_probabilities,
        exact_directions,
    )
    probabilities = np.exp(log_probabilities)
    accumulator["kl_total"] += diagnostics["kl_to_baseline"]
    accumulator["active_blocks"] += int(active_blocks)
    accumulator["entropy_total"] += diagnostics["entropy"]
    accumulator["baseline_entropy_total"] += categorical_entropy(
        reference_log_probabilities
    )
    accumulator["expected_exact_direction"] += diagnostics[
        "expected_direction"
    ]
    accumulator["expected_additive_direction"] += float(
        np.dot(probabilities, additive_directions)
    )
    accumulator["selected_exact_direction"] += float(
        exact_directions[selected_index]
    )
    accumulator["selected_additive_direction"] += float(
        additive_directions[selected_index]
    )
    if active_blocks:
        accumulator["negative_mass_sum"] += diagnostics[
            "negative_direction_mass"
        ]
        accumulator["neutral_mass_sum"] += diagnostics[
            "neutral_direction_mass"
        ]
        accumulator["positive_mass_sum"] += diagnostics[
            "positive_direction_mass"
        ]
        accumulator["active_rows"] += 1
        accumulator["effective_tau_values"].append(float(effective_tau))
        if matched_target_kl is not None:
            accumulator["matched_kl_errors"].append(abs(
                diagnostics["kl_to_baseline"] - matched_target_kl
            ))


def _finalize_kernel_accumulator(accumulator):
    active_blocks = accumulator["active_blocks"]
    active_rows = accumulator["active_rows"]
    reference_entropy = accumulator["baseline_entropy_total"]
    effective_taus = accumulator["effective_tau_values"]
    matched_errors = accumulator["matched_kl_errors"]
    return {
        "kernel_kl_total": accumulator["kl_total"],
        "kernel_kl_per_active_block": (
            accumulator["kl_total"] / active_blocks
            if active_blocks else 0.0
        ),
        "kernel_entropy_total": accumulator["entropy_total"],
        "kernel_entropy_ratio_to_baseline": (
            accumulator["entropy_total"] / reference_entropy
            if reference_entropy else 1.0
        ),
        "expected_exact_direction": accumulator[
            "expected_exact_direction"
        ],
        "expected_additive_direction": accumulator[
            "expected_additive_direction"
        ],
        "selected_exact_direction": accumulator[
            "selected_exact_direction"
        ],
        "selected_additive_direction": accumulator[
            "selected_additive_direction"
        ],
        "negative_exact_direction_mass": (
            accumulator["negative_mass_sum"] / active_rows
            if active_rows else 0.0
        ),
        "neutral_exact_direction_mass": (
            accumulator["neutral_mass_sum"] / active_rows
            if active_rows else 1.0
        ),
        "positive_exact_direction_mass": (
            accumulator["positive_mass_sum"] / active_rows
            if active_rows else 0.0
        ),
        "participating_active_rows": active_rows,
        "active_blocks": active_blocks,
        "effective_tau_mean": (
            float(np.mean(effective_taus)) if effective_taus else 0.0
        ),
        "effective_tau_min": (
            float(np.min(effective_taus)) if effective_taus else 0.0
        ),
        "effective_tau_max": (
            float(np.max(effective_taus)) if effective_taus else 0.0
        ),
        "matched_kl_error_max": (
            float(np.max(matched_errors)) if matched_errors else 0.0
        ),
    }


def _apply_selected_mask(
    proposal, donors, row_index, active_attribute_indices, mask, attr_names
):
    for local_index, attr_index in enumerate(active_attribute_indices):
        if mask[local_index]:
            attr = attr_names[attr_index]
            proposal.at[row_index, attr] = donors.at[row_index, attr]


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


def _summarize(rows):
    summary = {"n": len(rows)}
    keys = (
        "gain",
        "linear_gain",
        "quadratic_penalty",
        "gain_per_changed_cell",
        "changed_cells",
        "changed_rows",
        "kernel_kl_total",
        "kernel_kl_per_active_block",
        "kernel_entropy_ratio_to_baseline",
        "expected_exact_direction",
        "expected_additive_direction",
        "selected_exact_direction",
        "negative_exact_direction_mass",
        "positive_exact_direction_mass",
        "effective_tau_mean",
        "matched_kl_error_max",
    )
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    summary["positive_gain_rate"] = float(np.mean(gains > 0.0))
    summary["zero_gain_rate"] = float(np.mean(gains == 0.0))
    summary["negative_gain_rate"] = float(np.mean(gains < 0.0))
    return summary


def _paired_summary(candidate_rows, baseline_rows):
    differences = np.asarray([
        candidate["gain"] - baseline["gain"]
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ])
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


def _summarize_interactions(rows):
    active_rows = sum(row["active_rows"] for row in rows)
    if active_rows == 0:
        return {
            "active_rows": 0,
            "q0_mean_absolute_interaction": 0.0,
            "q0_rms_interaction": 0.0,
            "q0_nonadditive_mask_mass": 0.0,
            "max_absolute_interaction": 0.0,
            "one_hot_direction_max_error": 0.0,
            "independent_clipped_marginal_max_error": 0.0,
            "mean_mask_states_per_active_row": 0.0,
            "max_mask_states": 1,
        }
    return {
        "active_rows": active_rows,
        "q0_mean_absolute_interaction": (
            sum(row["q0_absolute_interaction_sum"] for row in rows)
            / active_rows
        ),
        "q0_rms_interaction": float(np.sqrt(
            sum(row["q0_squared_interaction_sum"] for row in rows)
            / active_rows
        )),
        "q0_nonadditive_mask_mass": (
            sum(row["q0_nonadditive_mass_sum"] for row in rows)
            / active_rows
        ),
        "max_absolute_interaction": max(
            row["max_absolute_interaction"] for row in rows
        ),
        "one_hot_direction_max_error": max(
            row["one_hot_direction_max_error"] for row in rows
        ),
        "independent_clipped_marginal_max_error": max(
            row["independent_clipped_marginal_max_error"] for row in rows
        ),
        "mean_mask_states_per_active_row": (
            sum(row["mask_states_sum"] for row in rows) / active_rows
        ),
        "max_mask_states": max(row["max_mask_states"] for row in rows),
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
    configs = _config_names(temperatures)
    rows = {name: [] for name in configs}
    interaction_rows = []
    direction_reference_scale = None
    reference_scale_proposal_index = None
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
        participate = participation_rng.random(N_RECORDS) < RHO
        participating_rows = np.flatnonzero(participate)
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
        accumulators = {
            name: _empty_kernel_accumulator() for name in configs
        }
        interaction = {
            "active_rows": 0,
            "q0_absolute_interaction_sum": 0.0,
            "q0_squared_interaction_sum": 0.0,
            "q0_nonadditive_mass_sum": 0.0,
            "max_absolute_interaction": 0.0,
            "one_hot_direction_max_error": 0.0,
            "independent_clipped_marginal_max_error": 0.0,
            "mask_states_sum": 0,
            "max_mask_states": 1,
        }
        gumbel_rng = np.random.default_rng(
            _address_seed(seed, state_index, proposal_index, 2)
        )

        for landscape in landscapes:
            row_index = landscape.row_index
            active_indices = landscape.active_attribute_indices
            n_active = len(active_indices)
            masks = landscape.masks
            exact = landscape.directions
            single = directions[row_index, active_indices]
            additive = additive_mask_directions(masks, single)
            interaction_direction = exact - additive
            reference = baseline_mask_log_probabilities(masks, ETA)
            gumbels = gumbel_rng.gumbel(size=len(masks))

            variants = {
                "baseline": (reference, 0.0, None),
            }
            for tau in temperatures:
                label = _tau_label(tau)
                effective_strength = (
                    tau / direction_reference_scale
                    if direction_reference_scale is not None else 0.0
                )
                independent = gibbs_mask_log_probabilities(
                    reference, additive, effective_strength
                )
                joint = gibbs_mask_log_probabilities(
                    reference, exact, effective_strength
                )
                independent_kl = categorical_kl(independent, reference)
                joint_kl = categorical_kl(joint, reference)
                common_kl = min(independent_kl, joint_kl)
                independent_matched_strength = (
                    match_gibbs_strength_for_kl(
                        reference, additive, common_kl
                    )
                    if common_kl > 0.0 else 0.0
                )
                joint_matched_strength = (
                    match_gibbs_strength_for_kl(
                        reference, exact, common_kl
                    )
                    if common_kl > 0.0 else 0.0
                )
                independent_matched = gibbs_mask_log_probabilities(
                    reference, additive, independent_matched_strength
                )
                joint_matched = gibbs_mask_log_probabilities(
                    reference, exact, joint_matched_strength
                )
                variants.update({
                    f"independent_tau_{label}": (
                        independent, tau, None
                    ),
                    f"joint_tau_{label}": (joint, tau, None),
                    f"independent_matched_tau_{label}": (
                        independent_matched,
                        independent_matched_strength
                        * (direction_reference_scale or 1.0),
                        common_kl,
                    ),
                    f"joint_matched_tau_{label}": (
                        joint_matched,
                        joint_matched_strength
                        * (direction_reference_scale or 1.0),
                        common_kl,
                    ),
                })

                # 理论独立 Gibbs 核与现有截断 logistic 核只应有数值饱和微差。
                if n_active:
                    theoretical_marginals = (
                        np.exp(independent)[:, None]
                        * masks.astype(float)
                    ).sum(axis=0)
                    clipped_marginals = tilted_copy_probabilities(
                        ETA, single, effective_strength
                    )
                    marginal_error = float(np.max(np.abs(
                        theoretical_marginals - clipped_marginals
                    )))
                    interaction[
                        "independent_clipped_marginal_max_error"
                    ] = max(
                        interaction[
                            "independent_clipped_marginal_max_error"
                        ],
                        marginal_error,
                    )

            for name, (
                log_probabilities,
                effective_tau,
                matched_target_kl,
            ) in variants.items():
                selected_index = sample_mask_index(
                    log_probabilities, gumbels
                )
                _apply_selected_mask(
                    generated[name],
                    donors,
                    row_index,
                    active_indices,
                    masks[selected_index],
                    attr_names,
                )
                _accumulate_kernel(
                    accumulators[name],
                    log_probabilities=log_probabilities,
                    reference_log_probabilities=reference,
                    exact_directions=exact,
                    additive_directions=additive,
                    selected_index=selected_index,
                    active_blocks=n_active,
                    effective_tau=effective_tau,
                    matched_target_kl=matched_target_kl,
                )

            if n_active:
                reference_probabilities = np.exp(reference)
                absolute_interaction = np.abs(interaction_direction)
                scale = max(1.0, float(np.max(np.abs(exact))))
                nonadditive = absolute_interaction > 1e-12 * scale
                one_hot_indices = 1 << np.arange(n_active)
                one_hot_error = float(np.max(np.abs(
                    exact[one_hot_indices] - single
                )))
                interaction["active_rows"] += 1
                interaction["q0_absolute_interaction_sum"] += float(
                    np.dot(reference_probabilities, absolute_interaction)
                )
                interaction["q0_squared_interaction_sum"] += float(
                    np.dot(reference_probabilities, interaction_direction ** 2)
                )
                interaction["q0_nonadditive_mass_sum"] += float(
                    reference_probabilities[nonadditive].sum()
                )
                interaction["max_absolute_interaction"] = max(
                    interaction["max_absolute_interaction"],
                    float(np.max(absolute_interaction)),
                )
                interaction["one_hot_direction_max_error"] = max(
                    interaction["one_hot_direction_max_error"],
                    one_hot_error,
                )
                interaction["mask_states_sum"] += len(masks)
                interaction["max_mask_states"] = max(
                    interaction["max_mask_states"], len(masks)
                )

        interaction_rows.append(interaction)
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
            measurement.update(
                _finalize_kernel_accumulator(accumulators[name])
            )
            rows[name].append(measurement)

    paired = {}
    for tau in temperatures:
        label = _tau_label(tau)
        paired[f"joint_vs_independent_tau_{label}"] = _paired_summary(
            rows[f"joint_tau_{label}"],
            rows[f"independent_tau_{label}"],
        )
        paired[f"joint_vs_independent_matched_tau_{label}"] = (
            _paired_summary(
                rows[f"joint_matched_tau_{label}"],
                rows[f"independent_matched_tau_{label}"],
            )
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
        "configs": {
            name: _summarize(config_rows)
            for name, config_rows in rows.items()
        },
        "paired_joint_vs_independent": paired,
        "interaction_summary": _summarize_interactions(interaction_rows),
        "proposal_rows": rows,
        "interaction_rows": interaction_rows,
        "elapsed_sec": time.perf_counter() - probe_start,
    }
    print(
        f"seed={seed:02d} state_rounds={state_rounds} "
        f"state_loss={loss:.1f} scale={direction_reference_scale}",
        flush=True,
    )
    for tau in temperatures:
        label = _tau_label(tau)
        independent = result["configs"][f"independent_tau_{label}"]
        joint = result["configs"][f"joint_tau_{label}"]
        matched = result["paired_joint_vs_independent"][
            f"joint_vs_independent_matched_tau_{label}"
        ]
        print(
            f"  tau={tau:g} same: {independent['gain']['mean']:+.2f}"
            f" -> {joint['gain']['mean']:+.2f}; "
            f"matched delta={matched['mean_gain_difference']:+.2f} "
            f"({matched['wins']}/{matched['ties']}/{matched['losses']})",
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
        "--temperatures", nargs="+", type=float, default=[1.0, 4.0, 8.0]
    )
    parser.add_argument(
        "--max-active-attributes", type=int, default=12
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="numpy"
    )
    parser.add_argument(
        "--output",
        default="outputs/joint_mask_diffusion/frozen_probe.json",
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
                proposals=args.proposals,
                device=args.device,
                max_active_attributes=args.max_active_attributes,
            ))

    summary = {
        "experiment": "joint_mask_diffusion_frozen_state_probe",
        "scope": (
            "same_state_donor_participation_and_gumbels_no_mutation_"
            "no_generation_acceptance"
        ),
        "dataset": "test_300x10",
        "seeds": args.seeds,
        "state_rounds": args.state_rounds,
        "n_proposals_per_state": args.proposals,
        "temperatures": args.temperatures,
        "matched_kl_rule": (
            "per recipient use min(same_tau_independent_kl, "
            "same_tau_joint_kl), then solve both temperatures"
        ),
        "device": args.device,
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
