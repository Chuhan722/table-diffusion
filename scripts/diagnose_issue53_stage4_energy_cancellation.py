"""Issue #53 Stage 4 能量恒等差异的只读验尸诊断。

对指定状态位级复现 probe 的 factor/oracle 能量比对，定位最大差异元素，
并把该元素的两条计算路径拆解到逐查询项，量化大项相消（catastrophic
cancellation）结构。只读取已归档产物与冻结协议，不产生任何资格结论，
不修改资格管线。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import issue53_stage4_protocol as frozen
from scripts import build_issue53_stage4_state_library as library_builder
from scripts import probe_factorized_gibbs_mixing as probe
from scripts import run_issue53_stage4_mixing as runner
from table_diffevo.factorized_diffusion import (
    build_sparse_mask_energy,
    evaluate_sparse_mask_energies,
)
from table_diffevo.joint_diffusion import compute_joint_mask_landscapes
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import (
    evaluate_directional_potential,
    evaluate_vectorized,
)
from table_diffevo.distance import pairwise_block_distance


def _single_row_query_indicators(frame, queries, schema) -> np.ndarray:
    """对单行 DataFrame 返回每个查询的 0/1 指示向量（精确整数计数）。"""
    counts, _, _ = evaluate_vectorized(
        frame,
        queries,
        schema,
        target=None,
        n_records=len(frame),
        want_fitness=False,
        device="numpy",
        verbose=False,
    )
    indicators = np.asarray(counts, dtype=float)
    if not np.all(np.isin(indicators, (0.0, 1.0))):
        raise RuntimeError("单行查询指示必须是 0/1")
    return indicators


def diagnose(
    library_path: str | Path,
    report_path: str | Path,
    state_id: str,
    output_path: str | Path,
    device: str = "numpy",
) -> tuple[Path, dict]:
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise FileExistsError(f"诊断输出已存在，不覆盖：{output_file}")

    report = runner._load_json_strict(report_path)
    mode = report["mode"]
    _, library, _ = runner._validate_library(mode, library_path, None)
    protocol = frozen.stage4_protocol(mode)

    entry = next(
        (
            (index, item)
            for index, item in enumerate(library["states"])
            if item["state_id"] == state_id
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"状态 {state_id!r} 不在状态库中")
    state_index, entry = entry

    recorded = next(
        (
            row["probe"]["factor_diagnostics"]
            for attempt in report["attempts"]
            for dataset_result in attempt["datasets"].values()
            for row in dataset_result["state_results"]
            if row["state_id"] == state_id
        ),
        None,
    )
    if recorded is None:
        raise ValueError(f"状态 {state_id!r} 不在 mixing 报告中")

    dataset_name = entry["dataset"]
    dataset = protocol["datasets"][dataset_name]
    schema = load_schema(str(REPOSITORY_ROOT / dataset["schema"]))
    queries = load_queries(str(REPOSITORY_ROOT / dataset["queries"]))
    raw_target = np.asarray(
        [query["result"] for query in queries], dtype=float
    )
    target = library_builder._runtime_target(raw_target, dataset)

    frame, controls = runner._restore_state(
        entry, dataset, target, queries, schema
    )
    seed = int(entry["seed"])
    n_records = dataset["runtime_n_records"]

    _, residual, fitness = evaluate_vectorized(
        frame,
        queries,
        schema,
        target=target,
        n_records=n_records,
        batch_size=256,
        device=device,
        want_fitness=True,
        verbose=False,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
    )
    distances = pairwise_block_distance(
        frame, frame, schema, device=device,
        return_tensor=device in ("cuda", "cpu"),
    )
    sampling_probabilities = compute_sampling_probs(
        fitness,
        distances,
        beta=1.0,
        h=0.8,
        device=device,
        distance_mode="geometric",
        lambda_param=0.5,
        alpha=float(controls["probe_alpha"]),
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        exclude_self=True,
        scale_invariant=True,
        scale_invariant_min_spread=1e-3,
    )

    worst = {
        "abs_diff": -1.0,
        "proposal_index": None,
        "row_index": None,
        "mask_index": None,
        "factor_energy": None,
        "oracle_direction": None,
        "landscape_scale": None,
    }
    max_abs_diff = 0.0
    for proposal_index in range(dataset["proposals_per_state"]):
        donor_rng = np.random.default_rng(
            probe._address_seed(seed, state_index, proposal_index, 0)
        )
        donor_idx = sample_donors(
            sampling_probabilities, donor_rng, device=device
        )
        donors = frame.iloc[donor_idx].reset_index(drop=True)
        participation_rng = np.random.default_rng(
            probe._address_seed(seed, state_index, proposal_index, 1)
        )
        participating_rows = np.flatnonzero(
            participation_rng.random(n_records) < frozen.RHO
        )
        landscapes = compute_joint_mask_landscapes(
            frame,
            donors,
            participating_rows,
            schema,
            queries,
            residual,
            batch_size=256,
            device=device,
            max_active_attributes=dataset["max_active_attributes"],
        )
        for landscape in landscapes:
            if landscape.masks.shape[1] == 0:
                continue
            model = build_sparse_mask_energy(
                frame.iloc[[landscape.row_index]],
                donors.iloc[[landscape.row_index]],
                schema,
                queries,
                residual,
                max_factor_order=dataset["max_factor_order"],
            )
            factor_energies = evaluate_sparse_mask_energies(
                model, landscape.masks
            )
            abs_diff = np.abs(factor_energies - landscape.directions)
            local_worst = int(np.argmax(abs_diff))
            max_abs_diff = max(max_abs_diff, float(np.max(abs_diff)))
            if float(abs_diff[local_worst]) > worst["abs_diff"]:
                worst = {
                    "abs_diff": float(abs_diff[local_worst]),
                    "proposal_index": proposal_index,
                    "row_index": int(landscape.row_index),
                    "mask_index": local_worst,
                    "mask": landscape.masks[local_worst].tolist(),
                    "active_attributes": list(
                        landscape.active_attributes
                    ),
                    "factor_energy": float(factor_energies[local_worst]),
                    "oracle_direction": float(
                        landscape.directions[local_worst]
                    ),
                    "landscape_scale": float(
                        max(
                            abs(float(factor_energies[local_worst])),
                            abs(float(landscape.directions[local_worst])),
                        )
                    ),
                    "donor_row": donors.iloc[landscape.row_index][
                        list(schema.attribute_names())
                    ].to_dict(),
                }

    reproduction = {
        "recorded_exact_energy_max_error": float(
            recorded["exact_energy_max_error"]
        ),
        "replayed_exact_energy_max_error": max_abs_diff,
        "bitwise_equal": max_abs_diff
        == float(recorded["exact_energy_max_error"]),
        "recorded_worst_case": dict(recorded["exact_energy_worst_case"]),
    }

    # ---- worst 元素拆解：逐查询项与三种求和路径 ----
    attr_names = list(schema.attribute_names())
    base_row = frame.iloc[[worst["row_index"]]].reset_index(drop=True)
    hybrid_row = base_row.copy()
    for local_index, attr in enumerate(worst["active_attributes"]):
        if worst["mask"][local_index]:
            hybrid_row.at[0, attr] = worst["donor_row"][attr]

    a_base = _single_row_query_indicators(base_row, queries, schema)
    a_hybrid = _single_row_query_indicators(hybrid_row, queries, schema)
    weighted_residual = np.asarray(residual, dtype=float)
    terms_base = weighted_residual * a_base
    terms_hybrid = weighted_residual * a_hybrid
    diff_terms = weighted_residual * (a_hybrid - a_base)
    changed = np.flatnonzero(a_hybrid != a_base)

    potential_pair = evaluate_directional_potential(
        pd.concat([base_row, hybrid_row], ignore_index=True),
        queries,
        schema,
        residual,
        batch_size=256,
        device=device,
        verbose=False,
    )
    potential_base = float(potential_pair[0])
    potential_hybrid = float(potential_pair[1])
    oracle_direction = potential_hybrid - potential_base

    exact_direction = math.fsum(
        float(diff_terms[index]) for index in changed
    )
    sum_abs_base = math.fsum(abs(float(v)) for v in terms_base)
    sum_abs_hybrid = math.fsum(abs(float(v)) for v in terms_hybrid)
    magnitude = max(sum_abs_base, sum_abs_hybrid)
    eps = float(np.finfo(np.float64).eps)

    factor_error = abs(worst["factor_energy"] - exact_direction)
    oracle_error = abs(worst["oracle_direction"] - exact_direction)
    decomposition = {
        "query_count": len(queries),
        "changed_query_count": int(changed.size),
        "changed_query_indices": changed.tolist(),
        "changed_query_terms": [
            float(diff_terms[index]) for index in changed
        ],
        "potential_base": potential_base,
        "potential_hybrid": potential_hybrid,
        "oracle_direction_from_pair": oracle_direction,
        "exact_direction_fsum": exact_direction,
        "factor_energy": worst["factor_energy"],
        "oracle_direction_from_landscape": worst["oracle_direction"],
        "sum_abs_terms_base": sum_abs_base,
        "sum_abs_terms_hybrid": sum_abs_hybrid,
        "max_abs_term": float(np.max(np.abs(terms_base))),
        "cancellation_ratio": (
            magnitude / abs(exact_direction)
            if exact_direction != 0.0
            else math.inf
        ),
        "one_ulp_of_summand_magnitude": eps * magnitude,
        "factor_vs_exact_abs_error": factor_error,
        "oracle_vs_exact_abs_error": oracle_error,
        "abs_diff_factor_vs_oracle": worst["abs_diff"],
        "oracle_error_in_ulp_of_magnitude": (
            oracle_error / (eps * magnitude) if magnitude > 0.0 else 0.0
        ),
        "factor_error_in_ulp_of_magnitude": (
            factor_error / (eps * magnitude) if magnitude > 0.0 else 0.0
        ),
    }

    verdict = {
        "cancellation_confirmed": bool(
            decomposition["cancellation_ratio"] > 1e6
            and oracle_error > 10.0 * factor_error
        ),
        "dominant_error_side": (
            "oracle_large_sum" if oracle_error > factor_error else "factor"
        ),
        "summary": (
            "oracle 侧对 ~{:.2e} 量级的项做大和相减，结果仅 ~{:.2e}；"
            "相消比 ~{:.1e}。oracle 相对精确差 {:.3e}（≈{:.1f} ulp of 项量级），"
            "factor 相对精确差 {:.3e}。".format(
                decomposition["max_abs_term"],
                abs(exact_direction),
                decomposition["cancellation_ratio"],
                oracle_error,
                decomposition["oracle_error_in_ulp_of_magnitude"],
                factor_error,
            )
        ),
    }

    diagnosis = {
        "diagnosis_format": "issue53_stage4_energy_cancellation_v1",
        "state_id": state_id,
        "dataset": dataset_name,
        "mode": mode,
        "device": device,
        "protocol_sha256": frozen.protocol_sha256(mode),
        "state_library_sha256": library.get(
            "state_library_scientific_sha256"
        ),
        "report_execution_sha256": report.get(
            "execution_scientific_sha256"
        ),
        "reproduction": reproduction,
        "worst_element": worst,
        "decomposition": decomposition,
        "verdict": verdict,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            diagnosis, ensure_ascii=False, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    return output_file, diagnosis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--state-id", default="nltcs__seed_326__initial"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", choices=("numpy", "cuda", "cpu"), default="numpy"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_file, diagnosis = diagnose(
        args.state_library,
        args.report,
        args.state_id,
        args.output,
        device=args.device,
    )
    reproduction = diagnosis["reproduction"]
    decomposition = diagnosis["decomposition"]
    verdict = diagnosis["verdict"]
    print(f"状态：{diagnosis['state_id']}（device={diagnosis['device']}）")
    print(
        "复现 max_abs_diff："
        f"replayed={reproduction['replayed_exact_energy_max_error']:.17e} "
        f"recorded={reproduction['recorded_exact_energy_max_error']:.17e} "
        f"bitwise_equal={reproduction['bitwise_equal']}"
    )
    print(
        "worst 元素："
        f"proposal={diagnosis['worst_element']['proposal_index']} "
        f"row={diagnosis['worst_element']['row_index']} "
        f"mask={diagnosis['worst_element']['mask_index']} "
        f"factor={decomposition['factor_energy']:.17e} "
        f"oracle={decomposition['oracle_direction_from_landscape']:.17e}"
    )
    print(
        f"potential_base={decomposition['potential_base']:.17e} "
        f"potential_hybrid={decomposition['potential_hybrid']:.17e}"
    )
    print(
        f"exact(fsum)={decomposition['exact_direction_fsum']:.17e} "
        f"changed_queries={decomposition['changed_query_count']}"
    )
    print(
        f"Σ|terms|={decomposition['sum_abs_terms_base']:.6e} "
        f"max|term|={decomposition['max_abs_term']:.6e} "
        f"相消比={decomposition['cancellation_ratio']:.3e}"
    )
    print(
        f"oracle 误差={decomposition['oracle_vs_exact_abs_error']:.3e} "
        f"(≈{decomposition['oracle_error_in_ulp_of_magnitude']:.1f} ulp) "
        f"factor 误差={decomposition['factor_vs_exact_abs_error']:.3e} "
        f"(≈{decomposition['factor_error_in_ulp_of_magnitude']:.1f} ulp)"
    )
    print(
        f"结论：cancellation_confirmed={verdict['cancellation_confirmed']} "
        f"dominant={verdict['dominant_error_side']}"
    )
    print(verdict["summary"])
    print(f"诊断已写入：{output_file}")


if __name__ == "__main__":
    main()
