"""固定预算分配考卷（第一步，非自适应）：一阶全测花大头，小对少测，其余不测靠生成器填。

规则只用公开信息（字段域大小、行数 N、预算 rho），不看数据，不花隐私预算：
  1. 一阶 m 个全测，sigma1；
  2. 二阶按格数 k_a*k_b 从小到大取前 K 对，sigma2；
  3. 其余对不测，引擎 --init-oneway 从一阶独立表起步自己填。

隐私记账与 AIM 同口径（unbounded，增删一行，高斯 L2 敏感度 1）：每个边际花 1/(2 sigma^2)，
账本 m/(2 sigma1^2) + K/(2 sigma2^2) = rho = cdp_rho(epsilon, delta)，写进 README 并断言。
注意本仓 make_noisy_exam 旧卷用 bounded 口径（敏感度根号二），同 rho 下 sigma 大根号二，两者不可混比。

参数自动档（--f1 auto / --pairs auto）用的误差模型：
  没测的对 (a,b) 用带噪一阶独立填充，期望 L1/N ≈ tau + kappa·(k_a+k_b)·sigma1·√(2/π)/N，
  tau 为先验常数（一对边际与独立积的 L1 距离，adult 真一阶独立实测 0.112，默认 0.11），
  kappa 为一阶噪声传到没测对的系数（第 0 步回放实测：引擎去噪后一阶误差约 0.35~0.43 倍地板，
  传播再打 0.38 折，合计约 0.15；少测对时去噪弱，默认取 0.25）；
  测过的对期望 L1/N ≈ gamma·k_a·k_b·sigma2·√(2/π)/N，gamma 为引擎推断相对直测地板的增益
  （第 0 步回放实测 0.64~0.70，默认 0.65）。
  给定 K，拉格朗日最优 sigma1:sigma2 = (m/(kappa·A))^(1/3) : (K/(gamma·B))^(1/3)，
  A = Σ_a (u_a+1)·k_a（u_a 为含 a 的没测对数，+1 计一阶自身），B = Σ_{测过} k_a·k_b；K 取总误差最小者。

用法：
  python scripts/make_alloc_exam.py --data adult --epsilon 0.1 --noise-seed 1 --f1 auto --pairs auto
  python scripts/make_alloc_exam.py --data adult --epsilon 0.1 --noise-seed 1 --f1 0.5 --pairs 18 --tag f50k18
  python scripts/make_alloc_exam.py --data adult --epsilon 0.1 --plan-only      # 只打印分配表不出卷
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from make_noisy_exam import cdp_rho  # noqa: E402

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def read_table(path: Path) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """读整数编码 CSV；域 = 0..max（与 AIM 同口径，data/<name>/<name>.csv 已是连续编码）。"""
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[int(v) for v in r] for r in reader if r]
    arr = np.asarray(rows, dtype=np.int64)
    domains = {c: [str(v) for v in range(int(arr[:, j].max()) + 1)] for j, c in enumerate(header)}
    return header, arr, domains


def split_for_k(m: int, rho: float, k_sizes: list[int], pairs_sorted: list[tuple[int, int]],
                K: int, gamma: float, kappa: float) -> tuple[float, float, float]:
    """给定 K 的拉格朗日最优分配，返回 (sigma1, sigma2, f1)。K=0 时 sigma2=inf。"""
    measured = pairs_sorted[:K]
    cnt = [0] * m
    for a, b in measured:
        cnt[a] += 1
        cnt[b] += 1
    # +1：一阶边际自身的误差也计一份，避免全测时 A=0 把一阶预算压成零
    A = kappa * sum(((m - 1) - cnt[a] + 1) * k_sizes[a] for a in range(m))
    if K == 0:
        return math.sqrt(m / (2.0 * rho)), float("inf"), 1.0
    B = gamma * sum(k_sizes[a] * k_sizes[b] for a, b in measured)
    t2 = (m ** (1 / 3) * A ** (2 / 3) + K ** (1 / 3) * B ** (2 / 3)) / (2.0 * rho)
    t = math.sqrt(t2)
    s1 = t * (m / A) ** (1 / 3)
    s2 = t * (K / B) ** (1 / 3)
    f1 = m / (2.0 * s1**2) / rho
    return s1, s2, f1


def expected_error(m: int, n: int, k_sizes: list[int], pairs_sorted: list[tuple[int, int]],
                   K: int, s1: float, s2: float, tau: float, gamma: float, kappa: float) -> tuple[float, float, float]:
    """误差模型：返回 (二阶均 L1/N, 测过对均, 没测对均)。"""
    P = len(pairs_sorted)
    measured = pairs_sorted[:K]
    unmeasured = pairs_sorted[K:]
    e_meas = [gamma * k_sizes[a] * k_sizes[b] * s2 * SQRT_2_OVER_PI / n for a, b in measured]
    e_un = [tau + kappa * (k_sizes[a] + k_sizes[b]) * s1 * SQRT_2_OVER_PI / n for a, b in unmeasured]
    tot = (sum(e_meas) + sum(e_un)) / P
    return tot, (np.mean(e_meas) if e_meas else float("nan")), (np.mean(e_un) if e_un else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description="固定预算分配考卷生成")
    ap.add_argument("--data", required=True, help="干净数据目录名，如 adult")
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--noise-seed", type=int, default=1)
    ap.add_argument("--f1", default="auto", help="一阶预算占比，数值或 auto")
    ap.add_argument("--pairs", default="auto", help="测的二阶对数 K，整数或 auto")
    ap.add_argument("--tau", type=float, default=0.11, help="没测对的独立填充先验误差 L1/N")
    ap.add_argument("--gamma", type=float, default=0.65, help="测过对引擎推断相对直测地板的增益系数")
    ap.add_argument("--kappa", type=float, default=0.25, help="一阶噪声传到没测对的系数")
    ap.add_argument("--tag", default="", help="目录名后缀，如 f50k18")
    ap.add_argument("--plan-only", action="store_true", help="只打印分配表，不出卷")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_dir = REPO / "data" / args.data
    header, arr, domains = read_table(src_dir / f"{args.data}.csv")
    m, n = len(header), int(arr.shape[0])
    k_sizes = [len(domains[c]) for c in header]
    rho = float(cdp_rho(args.epsilon, args.delta))
    pairs_sorted = sorted(combinations(range(m), 2), key=lambda p: (k_sizes[p[0]] * k_sizes[p[1]], p))
    P = len(pairs_sorted)

    # 分配表：每个 K 的最优拆分与模型误差
    plan = []
    for K in range(0, P + 1):
        s1, s2, f1 = split_for_k(m, rho, k_sizes, pairs_sorted, K, args.gamma, args.kappa)
        tot, em, eu = expected_error(m, n, k_sizes, pairs_sorted, K, s1, s2, args.tau, args.gamma, args.kappa)
        plan.append((K, s1, s2, f1, tot, em, eu))
    best = min(plan, key=lambda r: r[4])

    if args.pairs == "auto":
        K = best[0]
    else:
        K = int(args.pairs)
        if not 0 <= K <= P:
            raise SystemExit(f"--pairs 须在 0..{P}")
    if args.f1 == "auto":
        s1, s2, f1 = split_for_k(m, rho, k_sizes, pairs_sorted, K, args.gamma, args.kappa)
    else:
        f1 = float(args.f1)
        if not 0.0 < f1 <= 1.0 or (K > 0 and f1 >= 1.0):
            raise SystemExit("--f1 须在 (0,1]，K>0 时须 <1")
        s1 = math.sqrt(m / (2.0 * f1 * rho))
        s2 = math.sqrt(K / (2.0 * (1.0 - f1) * rho)) if K > 0 else float("inf")
    tot, em, eu = expected_error(m, n, k_sizes, pairs_sorted, K, s1, s2, args.tau, args.gamma, args.kappa)
    rho_used = m / (2.0 * s1**2) + (K / (2.0 * s2**2) if K > 0 else 0.0)
    if abs(rho_used - rho) > 1e-9 * rho:
        raise SystemExit(f"账本不平：{rho_used:.9g} != {rho:.9g}")

    print(f"{args.data}: m={m} N={n} 对数={P} ε={args.epsilon} δ={args.delta} ρ={rho:.6g}（AIM 口径，每边际 1/(2σ²)）")
    print(f"域大小 {k_sizes}")
    print(f"{'K':>4}{'第K对格数':>9}{'σ1':>8}{'σ2':>8}{'一阶占比':>8}{'模型二阶均':>10}{'测过均':>9}{'没测均':>9}")
    show = sorted(set([0, 1, 5, 10, 15, 18, 20, 25, 30, 33, 40, 50, 60, 80, 100, P, best[0], K]))
    for r in plan:
        if r[0] in show:
            cK = k_sizes[pairs_sorted[r[0] - 1][0]] * k_sizes[pairs_sorted[r[0] - 1][1]] if r[0] else 0
            mark = " <- 自动档" if r[0] == best[0] else ""
            s2txt = f"{r[2]:8.1f}" if math.isfinite(r[2]) else f"{'-':>8}"
            print(f"{r[0]:>4}{cK:>9}{r[1]:8.1f}{s2txt}{r[3]:8.2f}{r[4]*1e6:10.0f}{(r[5]*1e6 if r[5]==r[5] else 0):9.0f}{(r[6]*1e6 if r[6]==r[6] else 0):9.0f}{mark}")
    print(f"选定：K={K} 一阶占比 f1={f1:.3f} σ1={s1:.2f} σ2={s2:.2f}（{'自动' if args.pairs=='auto' else '指定'} K，{'自动' if args.f1=='auto' else '指定'} f1）"
          f" 模型二阶均 {tot*1e6:.0f}；账本 {m}/(2·{s1:.2f}²)+{K}/(2·{s2:.2f}²)={rho_used:.6g}=ρ ✓")
    if args.plan_only:
        return

    # 出卷：一阶 m 个 + 前 K 对，计数 + N(0, σ)，不裁剪（与 AIM 测量同口径）
    rng = np.random.default_rng(args.noise_seed)
    queries = []
    n1 = n2 = 0
    for j, c in enumerate(header):
        true = np.bincount(arr[:, j], minlength=k_sizes[j]).astype(np.float64)
        noisy = true + rng.normal(0.0, s1, true.size)
        for v in range(k_sizes[j]):
            n1 += 1
            queries.append({
                "id": f"A1_{n1:05d}", "type": "eq1way", "expression": f"{c} == {v}",
                "conditions": [{"attribute": c, "operator": "==", "value": str(v)}],
                "result": float(noisy[v]), "sigma": float(s1),
            })
    measured_pairs = pairs_sorted[:K]
    for a, b in measured_pairs:
        ka, kb = k_sizes[a], k_sizes[b]
        true = np.bincount(arr[:, a] * kb + arr[:, b], minlength=ka * kb).astype(np.float64)
        noisy = true + rng.normal(0.0, s2, true.size)
        for va in range(ka):
            for vb in range(kb):
                n2 += 1
                queries.append({
                    "id": f"A2_{n2:05d}", "type": "eq2way",
                    "expression": f"{header[a]} == {va} AND {header[b]} == {vb}",
                    "conditions": [
                        {"attribute": header[a], "operator": "==", "value": str(va)},
                        {"attribute": header[b], "operator": "==", "value": str(vb)},
                    ],
                    "result": float(noisy[va * kb + vb]), "sigma": float(s2),
                })

    eps_txt = f"{args.epsilon:g}".replace(".", "p")
    out_name = f"{args.data}_alloc{eps_txt}{args.tag}_s{args.noise_seed}"
    out_dir = REPO / "data" / out_name
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} 已存在，加 --force 覆盖")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copyfile(src_dir / f"{args.data}.csv", out_dir / f"{out_name}.csv")
    md5 = hashlib.md5((out_dir / f"{out_name}.csv").read_bytes()).hexdigest()
    pair_names = [f"{header[a]}×{header[b]}({k_sizes[a]*k_sizes[b]})" for a, b in measured_pairs]
    payload = {
        "dataset": f"{out_name}.csv", "record_count": n, "query_count": len(queries),
        "result_unit": "records",
        "description": (
            f"固定分配考卷：一阶 {m} 个 σ={s1:.4g} + 格数最小 {K} 对 σ={s2:.4g}，其余 {P-K} 对不测；"
            f"ε={args.epsilon} δ={args.delta} ρ={rho:.6g} AIM 口径（增删一行，每边际 1/(2σ²)），不裁剪"
        ),
        "allocation": {
            "epsilon": args.epsilon, "delta": args.delta, "rho": rho, "accounting": "unbounded_l2_1",
            "f1": f1, "K": K, "sigma1": s1, "sigma2": s2, "tau": args.tau, "gamma": args.gamma, "kappa": args.kappa,
            "f1_mode": args.f1, "pairs_mode": args.pairs,
            "measured_pairs": [[header[a], header[b]] for a, b in measured_pairs],
            "ledger": {"oneway": m / (2.0 * s1**2), "twoway": (K / (2.0 * s2**2) if K > 0 else 0.0), "total": rho_used},
            "model_error": {"twoway_mean": tot, "measured_mean": em, "unmeasured_mean": eu},
        },
        "queries": queries,
    }
    exam_out = out_dir / f"measured_{len(queries)}query.json"
    exam_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    readme = (
        f"# {out_name} 固定分配考卷\n\n"
        f"1. 来源，data/{args.data} 真表原样复制 md5 {md5}，一阶 {m} 个 + 格数最小 {K}/{P} 对直接加高斯噪声，其余对不测。\n"
        f"2. 账本（AIM 口径，增删一行，高斯 L2 敏感度 1，每边际 1/(2σ²)）：ε={args.epsilon} δ={args.delta} "
        f"cdp_rho ρ={rho:.6g}；一阶 σ1={s1:.4g} 花 {m/(2*s1**2):.6g}（{f1:.1%}），二阶 σ2={s2:.4g} 花 "
        f"{(K/(2*s2**2) if K>0 else 0):.6g}（{1-f1:.1%}），合计 {rho_used:.6g} = ρ。\n"
        f"3. 规则参数：f1 {args.f1}，K {args.pairs}，tau {args.tau}，gamma {args.gamma}，kappa {args.kappa}；模型预估二阶均 {tot*1e6:.0f}×1e-6"
        f"（测过 {em*1e6 if em==em else 0:.0f}，没测 {eu*1e6 if eu==eu else 0:.0f}）。\n"
        f"4. 测的对：{', '.join(pair_names) if pair_names else '无'}。\n"
        f"5. 噪声种子 {args.noise_seed}，numpy default_rng，先一阶后二阶按上列顺序逐格生成，不裁剪。\n"
        f"6. 用法：run_plants.py --data {out_name} --init-oneway --exam-weights sigma --struct-ruler ding（不带 --ji-rho）。\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"题数 {len(queries)}（一阶 {n1} 二阶 {n2}）-> {exam_out}")


if __name__ == "__main__":
    main()
