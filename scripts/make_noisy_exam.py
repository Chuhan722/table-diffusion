"""噪声考卷生成器，按 GSD 官方口径给全二阶训练卷加高斯噪声。

口径逐条对齐官方 private_gsd 库 chained_adaptive_statistics 的 oneshot 流程，
epsilon 与 delta 用 snsynth 的 cdp_rho 换算成零集中预算 rho，
每个二阶边际均分 rho，边际灵敏度取根号二除 N 的频率口径，
sigma 等于根号下灵敏度平方除以二倍每边际预算，
频率加高斯后裁剪到零一区间，乘 N 回计数口径写进噪声考卷。
分箱当公开预处理不花预算，与两家同餐同简化。

噪声由本脚本固定种子生成落盘，我们与 GSD 同吃一份，消掉噪声手气，
噪声顺序按考卷条目顺序逐格，条目本身按字段对加格子升序与两家格子序一致。

须知，cdp_rho 换算内置 IBM cdp2adp 参考实现，与官方所用 snsynth 同源零偏差，
./.venv/bin/python scripts/make_noisy_exam.py \
    --data nltcs --epsilon 1 --delta 1e-5 --noise-seed 1
输出目录 data/<data>_eps<E><tag>_s<seed> 必须不存在，只写噪声训练卷和说明。
本脚本只读取输入考卷，使用其中声明的 record_count，不读取或复制原始表。
考卷须为 JSON 对象，record_count 为正整数（不接受浮点数或布尔值），
result_unit 必须为 records；不合规元数据在计算和新建产物之前拒绝。
生成请使用公开输入入口 python -m resevo，显式提供公开 schema/N；
新目录不再适配依赖同名原表 CSV 的历史生成入口。
目录仍含可复现噪声信息，仅用于基准内部对照，不能作为 DP 发布包。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cdp_delta(rho: float, eps: float) -> float:
    """零集中差分隐私换 (eps,delta) 的紧公式，逐行同 IBM cdp2adp 参考实现。

    snsynth.utils.cdp_rho 亦抄自同一参考实现，本地内置以免拖依赖，
    换算口径与官方 GSD 用的完全一致。
    """
    import math

    assert rho >= 0 and eps >= 0
    if rho == 0:
        return 0.0
    amin, amax = 1.01, (eps + 1.0) / (2.0 * rho) + 2.0
    alpha = amax
    for _ in range(1000):
        alpha = (amin + amax) / 2.0
        derivative = (2.0 * alpha - 1.0) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    delta = math.exp(
        (alpha - 1.0) * (alpha * rho - eps) + alpha * math.log1p(-1.0 / alpha)
    ) / (alpha - 1.0)
    return min(delta, 1.0)


def cdp_rho(epsilon: float, delta: float) -> float:
    """二分找最大 rho 使换算 delta 不超标，同 IBM cdp2adp 与 snsynth。"""
    assert epsilon >= 0 and delta > 0
    if delta >= 1:
        return 0.0
    rhomin, rhomax = 0.0, epsilon + 1.0
    for _ in range(1000):
        rho = (rhomin + rhomax) / 2.0
        if _cdp_delta(rho, epsilon) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="data 下包含唯一 measured_*query.json 的目录名，无需原表")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument(
        "--reserve-rho", type=float, default=0.0,
        help="从总预算预留给结构基准 S* 购买的 rho（统一账本口径），考卷只分剩余预算",
    )
    parser.add_argument(
        "--tag", default="",
        help="目录名标记，插在 eps 段后，如 u 得 <data>_eps1u_s<seed>，防覆盖旧卷",
    )
    args = parser.parse_args()
    if args.epsilon <= 0 or not (0 < args.delta < 1):
        raise SystemExit("epsilon 须为正，delta 须落在 (0,1)")
    if args.reserve_rho < 0:
        raise SystemExit("reserve-rho 须非负")

    src_dir = REPO / "data" / args.data
    found = sorted(src_dir.glob("measured_*query.json"))
    if len(found) != 1:
        raise SystemExit(f"{src_dir} 下 measured_*query.json 须恰有一个")
    with open(found[0], encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise SystemExit("输入考卷必须是 JSON 对象")
    if payload.get("result_unit") != "records":
        raise SystemExit('输入考卷 result_unit 必须为 "records"')
    n_rows = payload.get("record_count")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise SystemExit("输入考卷 record_count 必须为正整数，不接受浮点数、布尔值或字符串")
    queries = payload["queries"]

    # 按字段对分组，组内条目顺序即格子顺序，组序即考卷条目顺序
    pair_of = []
    for q in queries:
        conds = q["conditions"]
        if len(conds) != 2 or any(c["operator"] != "==" for c in conds):
            raise SystemExit("噪声口径只支持全二阶等值训练卷")
        pair_of.append((conds[0]["attribute"], conds[1]["attribute"]))
    pairs = list(dict.fromkeys(pair_of))
    m = len(pairs)

    rho_total = float(cdp_rho(epsilon=args.epsilon, delta=args.delta))
    rho = rho_total - args.reserve_rho
    if rho <= 0:
        raise SystemExit("预留后考卷预算须为正")
    rho_per = rho / m
    sensitivity = np.sqrt(2.0) / n_rows
    sigma = float(np.sqrt(sensitivity**2 / (2.0 * rho_per)))

    rng = np.random.default_rng(args.noise_seed)
    freq = np.array([float(q["result"]) for q in queries]) / n_rows
    raw = freq + rng.normal(0.0, sigma, size=len(queries))
    clipped = int(((raw < 0.0) | (raw > 1.0)).sum())
    noisy_freq = np.clip(raw, 0.0, 1.0)

    eps_txt = f"{args.epsilon:g}".replace(".", "p")
    out_name = f"{args.data}_eps{eps_txt}{args.tag}_s{args.noise_seed}"
    out_dir = REPO / "data" / out_name
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"拒绝复用已有输出路径：{out_dir}；请使用新的 --tag，旧内容不会被清理") from exc

    for q, nf in zip(queries, noisy_freq):
        q["result"] = float(nf * n_rows)
    reserve_txt = (
        f"总 rho {rho_total:.6g} 预留 {args.reserve_rho:.6g} 给 S* 购买，考卷 rho {rho:.6g}，"
        if args.reserve_rho > 0 else f"rho {rho:.6g}，"
    )
    payload["description"] = (
        payload.get("description", "")
        + f" 噪声版，epsilon {args.epsilon} delta {args.delta} {reserve_txt}"
        f"每边际 rho {rho_per:.6g}，频率 sigma {sigma:.6g}，噪声种子 {args.noise_seed}，"
        "高斯加噪后频率裁剪到零一乘 N 回计数，口径同官方 GSD oneshot。"
    )
    exam_out = out_dir / found[0].name
    with open(exam_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    with open(out_dir / "README.md", "w", encoding="utf-8") as fh:
        fh.write(
            f"# {out_name} 噪声考卷\n\n"
            f"1. 来源，data/{args.data} 的训练卷加高斯噪声；只读取考卷，不读取或复制原表。\n"
            f"2. 口径，epsilon {args.epsilon}，delta {args.delta}，cdp_rho 换算总 rho {rho_total:.6g}，"
            f"预留 {args.reserve_rho:.6g} 给结构基准 S* 购买，考卷 rho {rho:.6g}，"
            f"边际数 {m} 均分，每边际 rho {rho_per:.6g}，灵敏度 根号2/{n_rows}，"
            f"频率 sigma {sigma:.6g}，计数 sigma {sigma * n_rows:.4f}。\n"
            f"3. 噪声种子 {args.noise_seed}，numpy default_rng，按考卷条目顺序逐格一次生成。\n"
            f"4. 裁剪触发 {clipped} 格，生成结束后独立用真实目录的考卷评价，本目录只供拟合。\n"
            "5. 生成使用 python -m resevo，显式提供公开 schema/N；不适配依赖同名原表 CSV 的旧入口。\n"
            "6. 仍含可复现噪声信息，仅用于基准内部对照，不能作为 DP 发布包。\n"
        )
    print(
        f"rho {rho:.6g} 边际 {m} 每边际 {rho_per:.6g} 频率sigma {sigma:.6g} "
        f"计数sigma {sigma * n_rows:.2f} 裁剪 {clipped} 格"
    )
    print(f"已写出 {out_dir}")


if __name__ == "__main__":
    main()
