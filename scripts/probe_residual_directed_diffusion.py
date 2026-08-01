"""在完全相同的冻结状态与 donor 上探测扩散算子的原始提案。

每个种子构造两个只依赖已发布 target 的状态：1-way marginal 初始表，以及原始
baseline 演化若干轮后的 best 表。对每个冻结状态重复抽 donor；各方向强度共享
donor、参与/复制/变异随机流，不执行整代接受。这样可以把算子因果效应与不同
运行轨迹到达的状态差异分开。
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
    tilted_copy_probabilities,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.evolution import run_evolution
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.update import evolve_step
from table_diffevo.vectorized_eval import evaluate_vectorized


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
N_RECORDS = 300
RHO = 0.01
ETA = 0.5
MU = 0.01


def _strength_name(strength):
    return f"strength_{strength:g}".replace(".", "p")


def _address_seed(seed, state_index, proposal_index, stream):
    sequence = np.random.SeedSequence(
        [int(seed), int(state_index), int(proposal_index), int(stream)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _summarize(rows):
    summary = {"n": len(rows)}
    for key in (
        "gain",
        "linear_gain",
        "quadratic_penalty",
        "gain_per_changed_cell",
        "changed_cells",
        "changed_rows",
    ):
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    summary["positive_rate"] = float(np.mean(gains > 0.0))
    summary["zero_rate"] = float(np.mean(gains == 0.0))
    summary["negative_rate"] = float(np.mean(gains < 0.0))
    return summary


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
            mu=MU,
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


def _probe_state(
    state,
    target,
    queries,
    schema,
    *,
    seed,
    state_index,
    state_rounds,
    strengths,
    proposals,
    device,
    normalization,
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
    probs = compute_sampling_probs(
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

    configs = ["baseline"] + [_strength_name(value) for value in strengths]
    rows = {name: [] for name in configs}
    endpoint_exact = True
    direction_reference_scale = None
    reference_scale_proposal_index = None
    direction_probability = {
        _strength_name(value): {
            "negative_probability": [],
            "positive_probability": [],
            "neutral_probability": [],
        }
        for value in strengths
    }

    for proposal_index in range(proposals):
        donor_rng = np.random.default_rng(
            _address_seed(seed, state_index, proposal_index, 0)
        )
        donor_idx = sample_donors(probs, donor_rng, device=device)
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
            for attr in schema.attribute_names()
        ])
        active = directions[differs]
        if (
            normalization == "initial_rms"
            and direction_reference_scale is None
        ):
            candidate_scale = direction_rms_scale(active)
            if candidate_scale > 0.0:
                direction_reference_scale = candidate_scale
                reference_scale_proposal_index = proposal_index

        proposal_seed = _address_seed(seed, state_index, proposal_index, 1)
        baseline = evolve_step(
            state,
            donors,
            schema,
            rho=RHO,
            eta=ETA,
            mu=MU,
            rng=np.random.default_rng(proposal_seed),
        )
        baseline_csv = baseline.to_csv(index=False)

        generated = {"baseline": baseline}
        for strength in strengths:
            name = _strength_name(strength)
            effective_strength = (
                strength / direction_reference_scale
                if normalization == "initial_rms"
                and direction_reference_scale is not None
                else (0.0 if normalization == "initial_rms" else strength)
            )
            generated[name] = evolve_step(
                state,
                donors,
                schema,
                rho=RHO,
                eta=ETA,
                mu=MU,
                rng=np.random.default_rng(proposal_seed),
                copy_direction_scores=directions,
                copy_direction_strength=effective_strength,
            )
            probabilities = tilted_copy_probabilities(
                ETA, active, effective_strength
            )
            for label, mask in (
                ("negative_probability", active < 0.0),
                ("positive_probability", active > 0.0),
                ("neutral_probability", active == 0.0),
            ):
                if np.any(mask):
                    direction_probability[name][label].append(
                        float(np.mean(probabilities[mask]))
                    )

        endpoint_exact = endpoint_exact and (
            generated["strength_0"].to_csv(index=False) == baseline_csv
        )
        for name, proposal in generated.items():
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
            rows[name].append({
                "gain": gain,
                "linear_gain": linear,
                "quadratic_penalty": quadratic,
                "gain_per_changed_cell": (
                    gain / changed_cells if changed_cells else 0.0
                ),
                "changed_cells": changed_cells,
                "changed_rows": int(changed.any(axis=1).sum()),
            })

    probability_summary = {}
    for name, groups in direction_probability.items():
        probability_summary[name] = {
            key: (float(np.mean(values)) if values else None)
            for key, values in groups.items()
        }

    result = {
        "seed": int(seed),
        "state_rounds": int(state_rounds),
        "state_loss": float(loss),
        "probe_alpha": probe_alpha,
        "state_sha256": hashlib.sha256(
            state.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "n_proposals": int(proposals),
        "direction_normalization": normalization,
        "direction_reference_scale": direction_reference_scale,
        "reference_scale_proposal_index": reference_scale_proposal_index,
        "strength_zero_exact": bool(endpoint_exact),
        "configs": {name: _summarize(values) for name, values in rows.items()},
        "mean_copy_probability_by_direction": probability_summary,
    }
    print(
        f"seed={seed:02d} baseline_state_rounds={state_rounds} "
        f"state_loss={loss:.1f}",
        flush=True,
    )
    for name in configs:
        summary = result["configs"][name]
        print(
            f"  {name:<14} raw_gain={summary['gain']['mean']:+.2f} "
            f"positive={summary['positive_rate']:.1%} "
            f"changed={summary['changed_cells']['mean']:.1f}",
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--state-rounds", nargs="+", type=int, default=[0, 100]
    )
    parser.add_argument("--proposals", type=int, default=100)
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=[0.0, 1.0]
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "initial_rms"],
        default="initial_rms",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default="outputs/residual_directed_diffusion_small/frozen_probe.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if any(value < 0 for value in args.state_rounds):
        parser.error("--state-rounds 必须为非负整数")
    if args.proposals <= 0:
        parser.error("--proposals 必须为正数")
    if 0.0 not in args.strengths:
        parser.error("--strengths 必须包含 0")
    if len(set(args.strengths)) != len(args.strengths):
        parser.error("--strengths 不得重复")
    if any(not np.isfinite(value) or value < 0.0 for value in args.strengths):
        parser.error("--strengths 必须全部为非负有限数值")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)

    states = []
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
                strengths=args.strengths,
                proposals=args.proposals,
                device=args.device,
                normalization=args.normalization,
            ))

    summary = {
        "experiment": "residual_directed_diffusion_frozen_state_probe",
        "scope": "same_state_same_donor_same_random_rolls_no_generation_acceptance",
        "dataset": "test_300x10",
        "seeds": args.seeds,
        "state_rounds": args.state_rounds,
        "n_proposals_per_state": args.proposals,
        "strengths": args.strengths,
        "normalization": args.normalization,
        "device": args.device,
        "all_strength_zero_exact": all(
            state["strength_zero_exact"] for state in states
        ),
        "states": states,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"全部 strength=0 端点等价：{summary['all_strength_zero_exact']}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
