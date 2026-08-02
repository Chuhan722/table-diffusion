"""对一张指定合成表做无接受、共同随机流的方向扩散提案探针。

输入状态、schema、workload 和 target 都显式提供；脚本不读取真实训练/测试表。
适合在大表的已有 baseline checkpoint 上比较不同方向强度，而不让运行轨迹差异
混入算子因果判断。
"""

import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import (
    bernoulli_entropy,
    compute_copy_direction_scores,
    direction_rms_scale,
    tilted_copy_probabilities,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.update import evolve_step
from table_diffevo.vectorized_eval import evaluate_vectorized


def _name(strength):
    return "baseline" if strength is None else (
        f"strength_{strength:g}".replace(".", "p")
    )


def _address_seed(seed, proposal_index, stream):
    sequence = np.random.SeedSequence(
        [int(seed), int(proposal_index), int(stream)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(*args):
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip()


def _environment_snapshot(device):
    commit_code, commit = _git_text("rev-parse", "HEAD")
    status_code, status = _git_text("status", "--porcelain")
    snapshot = {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit if commit_code == 0 else None,
        "git_worktree_clean": status_code == 0 and status == "",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requested_device": device,
    }
    try:
        import torch
    except ImportError:
        snapshot.update({"torch": None, "cuda_available": False, "gpu": None})
    else:
        cuda_available = bool(torch.cuda.is_available())
        snapshot.update({
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "gpu": (
                torch.cuda.get_device_name(0)
                if device == "cuda" and cuda_available else None
            ),
        })
    return snapshot


def _summary(rows):
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
    result["positive_rate"] = float(np.mean(gains > 0.0))
    result["zero_rate"] = float(np.mean(gains == 0.0))
    result["negative_rate"] = float(np.mean(gains < 0.0))
    return result


def _paired_gain_summary(candidate_rows, baseline_rows):
    differences = np.asarray([
        candidate["gain"] - baseline["gain"]
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ], dtype=float)
    return {
        "mean": float(differences.mean()),
        "std": (
            float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
        ),
        "median": float(np.median(differences)),
        "min": float(differences.min()),
        "max": float(differences.max()),
        "wins": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "losses": int(np.sum(differences < 0.0)),
        "values": differences.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--schema", default="configs/nltcs/schema.yaml")
    parser.add_argument(
        "--queries", default="configs/nltcs/measured_1000query.json"
    )
    parser.add_argument("--proposals", type=int, default=100)
    parser.add_argument("--expected-records", type=int, default=16_181)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0]
    )
    parser.add_argument(
        "--direction-normalization",
        choices=["none", "initial_rms"],
        default="initial_rms",
    )
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.proposals <= 0:
        parser.error("--proposals 必须为正数")
    if args.expected_records <= 0:
        parser.error("--expected-records 必须为正数")
    if 0.0 not in args.strengths:
        parser.error("--strengths 必须包含 0")
    if len(set(args.strengths)) != len(args.strengths):
        parser.error("--strengths 不得重复")
    if any(not np.isfinite(value) or value < 0.0 for value in args.strengths):
        parser.error("--strengths 必须全部为非负有限数值")
    if not 0.0 <= args.rho <= 1.0:
        parser.error("--rho 必须在 [0,1]")
    if not 0.0 <= args.eta <= 1.0:
        parser.error("--eta 必须在 [0,1]")
    if not 0.0 <= args.mu <= 1.0:
        parser.error("--mu 必须在 [0,1]")
    if args.alpha < 0.0 or not np.isfinite(args.alpha):
        parser.error("--alpha 必须是非负有限数值")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    environment = _environment_snapshot(args.device)
    if args.device == "cuda" and not environment["cuda_available"]:
        parser.error("--device=cuda，但当前可见设备中 CUDA 不可用")

    schema = load_schema(args.schema)
    queries = load_queries(args.queries)
    target = np.asarray([query["result"] for query in queries], dtype=float)
    state = pd.read_csv(args.state)
    columns = schema.attribute_names()
    if list(state.columns) != columns:
        raise ValueError("输入状态列名或顺序与 schema 不一致")
    n_records = len(state)
    if n_records != args.expected_records:
        raise ValueError(
            f"输入状态有 {n_records} 行，预期 {args.expected_records} 行；"
            "请确认它与 target 使用同一公开记录数"
        )

    q, residual, fitness = evaluate_vectorized(
        state,
        queries,
        schema,
        target=target,
        n_records=n_records,
        batch_size=256,
        device=args.device,
        want_fitness=True,
        verbose=False,
    )
    state_loss = compute_loss(target, q)
    use_torch = args.device in ("cuda", "cpu")
    distances = pairwise_block_distance(
        state, state, schema, device=args.device, return_tensor=use_torch
    )
    probs = compute_sampling_probs(
        fitness,
        distances,
        beta=1.0,
        h=0.8,
        device=args.device,
        distance_mode="geometric",
        lambda_param=0.5,
        alpha=args.alpha,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        exclude_self=True,
    )

    names = ["baseline"] + [_name(value) for value in args.strengths]
    rows = {name: [] for name in names}
    probabilities = {
        _name(value): {"negative": [], "positive": [], "neutral": []}
        for value in args.strengths
    }
    direction_entropy = {
        _name(value): [] for value in args.strengths
    }
    endpoint_exact = True
    direction_reference_scale = None
    reference_scale_proposal_index = None

    for proposal_index in range(args.proposals):
        donor_rng = np.random.default_rng(
            _address_seed(args.seed, proposal_index, 0)
        )
        donor_idx = sample_donors(probs, donor_rng, device=args.device)
        donors = state.iloc[donor_idx].reset_index(drop=True)
        directions = compute_copy_direction_scores(
            state,
            donors,
            schema,
            queries,
            residual,
            batch_size=256,
            device=args.device,
        )
        differs = np.column_stack([
            state[attr].reset_index(drop=True).to_numpy()
            != donors[attr].to_numpy()
            for attr in columns
        ])
        active = directions[differs]
        if (
            args.direction_normalization == "initial_rms"
            and direction_reference_scale is None
        ):
            candidate_scale = direction_rms_scale(active)
            if candidate_scale > 0.0:
                direction_reference_scale = candidate_scale
                reference_scale_proposal_index = proposal_index
        proposal_seed = _address_seed(args.seed, proposal_index, 1)
        generated = {
            "baseline": evolve_step(
                state,
                donors,
                schema,
                rho=args.rho,
                eta=args.eta,
                mu=args.mu,
                rng=np.random.default_rng(proposal_seed),
            )
        }
        for strength in args.strengths:
            name = _name(strength)
            if args.direction_normalization == "initial_rms":
                effective_strength = (
                    strength / direction_reference_scale
                    if direction_reference_scale is not None else 0.0
                )
            else:
                effective_strength = strength
            generated[name] = evolve_step(
                state,
                donors,
                schema,
                rho=args.rho,
                eta=args.eta,
                mu=args.mu,
                rng=np.random.default_rng(proposal_seed),
                copy_direction_scores=directions,
                copy_direction_strength=effective_strength,
            )
            copy_probs = tilted_copy_probabilities(
                args.eta, active, effective_strength
            )
            if len(copy_probs):
                direction_entropy[name].append(
                    float(np.mean(bernoulli_entropy(copy_probs)))
                )
            for label, mask in (
                ("negative", active < 0.0),
                ("positive", active > 0.0),
                ("neutral", active == 0.0),
            ):
                if np.any(mask):
                    probabilities[name][label].append(
                        float(np.mean(copy_probs[mask]))
                    )

        endpoint_exact = endpoint_exact and generated["strength_0"].equals(
            generated["baseline"]
        )
        for name, proposal in generated.items():
            proposal_q, _, _ = evaluate_vectorized(
                proposal,
                queries,
                schema,
                batch_size=256,
                device=args.device,
                want_fitness=False,
                verbose=False,
            )
            proposal_loss = compute_loss(target, proposal_q)
            delta_q = proposal_q - q
            linear = float(np.dot(target - q, delta_q))
            quadratic = float(0.5 * np.dot(delta_q, delta_q))
            gain = float(state_loss - proposal_loss)
            changed = proposal.reset_index(drop=True) != state.reset_index(drop=True)
            changed_cells = int(changed.to_numpy().sum())
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

    result = {
        "experiment": "residual_directed_diffusion_fixed_state_probe",
        "scope": "same_state_same_donor_same_random_rolls_no_generation_acceptance",
        "state": str(Path(args.state)),
        "state_sha256": hashlib.sha256(
            state.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "input_sha256": {
            "state": _sha256_file(args.state),
            "schema": _sha256_file(args.schema),
            "queries": _sha256_file(args.queries),
        },
        "environment": environment,
        "state_loss": float(state_loss),
        "n_records": n_records,
        "expected_records": args.expected_records,
        "n_queries": len(queries),
        "n_proposals": args.proposals,
        "seed": args.seed,
        "strengths": args.strengths,
        "direction_normalization": args.direction_normalization,
        "direction_reference_scale": direction_reference_scale,
        "reference_scale_proposal_index": reference_scale_proposal_index,
        "effective_direction_strengths": {
            _name(strength): (
                strength / direction_reference_scale
                if (
                    args.direction_normalization == "initial_rms"
                    and direction_reference_scale is not None
                ) else (
                    0.0
                    if args.direction_normalization == "initial_rms"
                    else strength
                )
            )
            for strength in args.strengths
        },
        "rho": args.rho,
        "eta": args.eta,
        "mu": args.mu,
        "alpha": args.alpha,
        "device": args.device,
        "strength_zero_exact": bool(endpoint_exact),
        "configs": {name: _summary(values) for name, values in rows.items()},
        "paired_gain_improvement_vs_baseline": {
            _name(strength): _paired_gain_summary(
                rows[_name(strength)], rows["baseline"]
            )
            for strength in args.strengths
        },
        "proposal_rows": rows,
        "mean_copy_entropy": {
            name: (float(np.mean(values)) if values else None)
            for name, values in direction_entropy.items()
        },
        "mean_copy_probability_by_direction": {
            name: {
                label: (float(np.mean(values)) if values else None)
                for label, values in groups.items()
            }
            for name, groups in probabilities.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"state loss={state_loss:.1f}")
    for name in names:
        summary = result["configs"][name]
        print(
            f"{name:<14} raw_gain={summary['gain']['mean']:+.1f} "
            f"positive={summary['positive_rate']:.1%} "
            f"gain/cell={summary['gain_per_changed_cell']['mean']:+.1f} "
            f"changed={summary['changed_cells']['mean']:.1f}"
        )
    print(f"strength=0 精确等价：{endpoint_exact}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
