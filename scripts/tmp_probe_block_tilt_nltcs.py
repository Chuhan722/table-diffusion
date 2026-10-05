#!/usr/bin/env python3
"""分科倾斜复制（block-tilted copy）nltcs 三臂摸底（临时诊断，不入正式协议）。

三臂同种子（9908）、同正式配置（512 池 = 480 all-2way + 32 one-way 真答案，
7000 轮上限，v3 ρ 三段式 1050/700/0.001，早停 patience 6）：

- baseline    : block_score_tilt_strength = 0（与历史路径逐位一致）
- mild        : strength = 0.5, bounds (0.3, 0.7) —— 温和倾斜
- aggressive  : strength = 2.0, bounds (0.2, 0.8) —— 顶格倾斜（大量硬币贴夹带边）

评估口径与正式报告对齐：in-pool 512 与冻结 heldout 3/4-way 的
normalized L1（|q−y|/N）分布 + tilt 诊断摘要 + 墙钟。仅摸底，无任何门槛
或正式声明；结果好坏都只用于决定是否走正式结果前协议。

用法：CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python scripts/tmp_probe_block_tilt_nltcs.py --arm mild
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_fitness_only_evolution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "tmp_block_tilt_probe_nltcs_seed9908"

SEED = 9908
N_RECORDS = 16181
N_ROUNDS = 7000

# 每臂 = FitnessOnlyConfig 字段覆盖；未覆盖处用正式配置缺省。
_TILT2 = dict(
    block_score_tilt_strength=2.0, block_score_tilt_bounds=(0.2, 0.8)
)
_DEEP_ANNEAL = dict(
    # 深退火：rho 几何降到 1e-5（前 ~670 轮与正式 700 轮到 1e-3 同速，
    # 之后继续下潜），mu 同步几何降到 1e-6（末期变异≈关）。上限 2 万轮，
    # patience 6 早停兜底。
    n_rounds=20000,
    rho_anneal_rounds=2000,
    rho_anneal_end=1e-5,
    mu_anneal_start_round=1050,
    mu_anneal_rounds=2000,
    mu_anneal_end=1e-6,
)
_VALUE_GUIDANCE = dict(
    # 值引导核：λ 与 ρ/μ 退火同窗线性升温（1050 起 2000 轮升到 4.0），
    # 流量降·精度升接力——末期低流量高保真照账写值。
    value_guidance_strength=4.0,
    value_guidance_warmup_start_round=1050,
    value_guidance_warmup_rounds=2000,
)
ARMS = {
    "baseline": {},
    "mild": dict(
        block_score_tilt_strength=0.5, block_score_tilt_bounds=(0.3, 0.7)
    ),
    "aggressive": dict(_TILT2),
    "s4": dict(
        block_score_tilt_strength=4.0, block_score_tilt_bounds=(0.1, 0.9)
    ),
    "s8": dict(
        block_score_tilt_strength=8.0, block_score_tilt_bounds=(0.05, 0.95)
    ),
    "L0": dict(lottery_first_donor_selection=True),
    "L1": dict(lottery_first_donor_selection=True, **_DEEP_ANNEAL),
    "L2": dict(lottery_first_donor_selection=True, **_DEEP_ANNEAL, **_TILT2),
    # V 系列：残差引导值分布核（V0=L1 复用已有结果）
    "V1": dict(
        lottery_first_donor_selection=True, **_DEEP_ANNEAL, **_VALUE_GUIDANCE
    ),
    "V2": dict(
        lottery_first_donor_selection=True, **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        value_guidance_drop_donor=True,
        # drop_donor 臂在 λ=0 保温段只剩微变异（无供体=近冻结、loss 不动），
        # patience 会在 warmup 前掐掉（首跑 602 轮早停实证）。关掉早停让
        # 对照跑满，与 V1 同时间表、只差 drop_donor 一个开关。
        inner_early_stopping_patience_ticks=None,
    ),
    # V3-V5：修 λ×ρ 窗口错位（λ 升到顶时 ρ 已把流量掐到 0.16 行/轮）。
    # V3=抬流量地板（验流量瓶颈）；V4=+加力（验力度）；V5=+提前升温（验交接窗）。
    "V3": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "rho_anneal_end": 1e-4,  # 末期 ~1.6 行/轮，给引导留流量
    },
    "V4": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "rho_anneal_end": 1e-4,
        "value_guidance_strength": 8.0,  # 末期照账率 98%→99.9%
    },
    "V5": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "rho_anneal_end": 1e-4,
        "value_guidance_strength": 8.0,
        "value_guidance_warmup_start_round": 500,  # 提前交接（2500 到顶）
    },
    # V6-V8：λ×gain 尺度失配修正验证（gain 行内差中位数 ~5.45e-4，
    # λ=4 实际倾斜仅 0.2%——V3≡V4 逐位一致实锤）。V1 底座只拉 λ_max：
    # λ=500/2000/8000 → 典型倾斜 e^0.27/e^1.1/e^4.4 ≈ 1.3x/3x/80x。
    "V6": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "value_guidance_strength": 500.0,
    },
    "V7": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "value_guidance_strength": 2000.0,
    },
    "V8": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "value_guidance_strength": 8000.0,
    },
    # V9：自适应尺度（λ_eff=λ_t/p90|wr_t|）。λ=4 恢复"照账倍数"语义
    # （≈e^4 倍照账），早期账多自动温柔、末期账平自动加压。对照 V8
    # （固定 λ=8000）验证归一化语义与自动变压收益。
    "V9": {
        "lottery_first_donor_selection": True,
        **_DEEP_ANNEAL, **_VALUE_GUIDANCE,
        "value_guidance_adaptive_scale": True,
    },
}


def build_pool():
    """正式 512 池：480 all-2way 真答案 + 32 one-way 边缘（同 profile 脚本）。"""
    marginals = load_marginals(str(ROOT / "configs/nltcs/init_marginals.json"))
    payload = json.loads(
        (ROOT / "configs/nltcs/all2way_issue53_v1.json").read_text()
    )
    queries = list(payload["queries"])
    assert len(queries) == 480
    targets = [float(q["result"]) for q in queries]

    index = 0
    for attribute, spec in marginals["attributes"].items():
        for value, count in zip(spec["values"], spec["counts"]):
            index += 1
            queries.append({
                "id": f"OW{index:04d}",
                "type": "single",
                "expression": f"{attribute} == {value}",
                "conditions": [
                    {"attribute": attribute, "operator": "==", "value": value}
                ],
                "result": float(count),
            })
            targets.append(float(count))
    assert len(queries) == 512
    return queries, np.asarray(targets, dtype=float), marginals


def normalized_l1_stats(target, current, n_records):
    err = np.abs(np.asarray(current, dtype=float) - np.asarray(target)) / n_records
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(err.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    arm = ARMS[args.arm]

    schema = load_schema(str(ROOT / "configs/nltcs/schema.yaml"))
    queries, targets, marginals = build_pool()

    base_fields = dict(
        n_rounds=N_ROUNDS,
        seed=SEED,
        device=args.device,
        eval_method="vectorized",
        batch_size=256,
        init_method="marginal",
        log_every=500,
        rho=0.01,
        eta=0.5,
        mu=0.01,
        rho_anneal_start_round=1050,
        rho_anneal_rounds=700,
        rho_anneal_end=0.001,
        lambda_param=0.5,
        fixed_alpha=16.0,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        selection_scale_invariant_min_spread=1e-3,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
        exclude_self=True,
        inner_early_stopping_patience_ticks=6,
    )
    base_fields.update(arm)
    config = FitnessOnlyConfig(**base_fields)

    start = time.perf_counter()
    table, diagnostics = run_fitness_only_evolution(
        targets, queries, schema, N_RECORDS,
        config=config, fitness_mode="residual", marginals=marginals,
    )
    walltime = time.perf_counter() - start

    # 生成后评估：in-pool 512 + 冻结 heldout 3/4-way（评估阶段才加载）
    q_pool = evaluate_vectorized(
        table, queries, schema, want_fitness=False, device=args.device,
        verbose=False,
    )[0]
    heldout = json.loads(
        (ROOT / "configs/nltcs/heldout_issue53_v1.json").read_text()
    )
    heldout_queries = list(heldout["queries"])
    heldout_targets = np.asarray(
        [float(q["result"]) for q in heldout_queries], dtype=float
    )
    q_heldout = evaluate_vectorized(
        table, heldout_queries, schema, want_fitness=False,
        device=args.device, verbose=False,
    )[0]
    kinds = np.asarray([q["type"] for q in heldout_queries])

    bt = diagnostics["block_score_tilt"]
    tilt_summary = {
        "enabled": bt["enabled"],
        "strength": bt["strength"],
        "bounds": bt["bounds"],
        "reference_scale": bt["reference_scale"],
    }
    if bt["enabled"]:
        mad = bt["mean_abs_scaled_delta_history"]
        lo = bt["clip_lo_rate_history"]
        hi = bt["clip_hi_rate_history"]
        tilt_summary.update({
            "mean_abs_scaled_delta_first_last": (mad[0], mad[-1]),
            "clip_lo_rate_first_last": (lo[0], lo[-1]),
            "clip_hi_rate_first_last": (hi[0], hi[-1]),
        })

    vg = diagnostics["value_guidance"]
    vg_summary = {
        "enabled": vg["enabled"],
        "strength": vg["strength"],
        "warmup_start_round": vg["warmup_start_round"],
        "warmup_rounds": vg["warmup_rounds"],
        "drop_donor": vg["drop_donor"],
    }
    if vg["enabled"]:
        lam = vg["lambda_history"]
        vg_summary.update({
            "lambda_first_last": (lam[0], lam[-1]),
            "gain_recompute_count": vg["gain_recompute_count"],
            "structure_compile_elapsed_sec": round(
                vg["structure_compile_elapsed_sec"], 4
            ),
        })
        if vg.get("adaptive_scale"):
            sc = vg["scale_history"]
            lam_eff = [
                l / s for l, s in zip(lam, sc)
            ]
            vg_summary.update({
                "adaptive_scale": True,
                "scale_first_last": (sc[0], sc[-1]),
                "lambda_eff_first_last": (lam_eff[0], lam_eff[-1]),
                "lambda_eff_max": max(lam_eff),
            })

    report = {
        "arm": args.arm,
        "config_overrides": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in arm.items()
        },
        "seed": SEED,
        "device": args.device,
        "rounds_run": diagnostics.get("rounds_run"),
        "termination_reason": diagnostics.get("termination_reason"),
        "walltime_sec": round(walltime, 1),
        "final_loss": diagnostics["loss_history"][-1],
        # GSD 口径主指标：480 格 all-2way 家族（build_pool 前 480 条）
        "family480_normalized_l1": normalized_l1_stats(
            targets[:480], q_pool[:480], N_RECORDS
        ),
        "in_pool_normalized_l1": normalized_l1_stats(
            targets, q_pool, N_RECORDS
        ),
        "heldout_normalized_l1": {
            "3way": normalized_l1_stats(
                heldout_targets[kinds == "heldout_3way"],
                q_heldout[kinds == "heldout_3way"], N_RECORDS,
            ),
            "4way": normalized_l1_stats(
                heldout_targets[kinds == "heldout_4way"],
                q_heldout[kinds == "heldout_4way"], N_RECORDS,
            ),
            "combined": normalized_l1_stats(
                heldout_targets, q_heldout, N_RECORDS
            ),
        },
        "block_score_tilt": tilt_summary,
        "value_guidance": vg_summary,
        "references": {
            "formal_engine_family480_mean": 1.3493191603320767e-4,
            "formal_engine_family480_max": 1.545022e-3,
            "gsd_family480_mean": 1.030015e-6,
            "gsd_family480_max": 6.180088e-5,
            "pgm_family480_mean": 3.574150752940692e-4,
            "pgm_family480_max": 1.977628e-3,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.arm}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已写出 {out_path}")


if __name__ == "__main__":
    main()
