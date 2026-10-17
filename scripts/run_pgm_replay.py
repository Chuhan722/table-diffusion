"""独立 PGM 回放：把 AIM 测量录像原样喂给 Private-PGM（mbi MirrorDescent）出表。

第一阶段 1B 消融的核对臂。AIM = 选测器 + PGM，录像里的测量列表就是 AIM 末次
estimate 收到的那一列（含重复团、各自 sigma），本脚本复刻 AIM 收尾那两行：
    model = MirrorDescent().estimate(domain, measurements, iters=max_iters)
    synth = model.synthetic_data(rows=N)
区别仅在没有前面几十轮的 warm_start 势能（从零起步跑 iters 轮）。known_total 默认喂真表
行数 N（我们引擎与 GSD 都拿 N，三方同口径）；--known-total auto 则与 AIM 原样留空自估。
若此臂与同一跑的 AIM 终表各轴相差 ≤2%，即证实「AIM 终表 ≈ 独立 PGM」，1B 里的
对照臂可直接用 AIM 终表；差得多则以本臂为 PGM 对照（同录像、同迭代预算）。

用法（A6000 aim2 环境，纯 CPU）：
  PYTHONPATH=/root/private-pgm/src /root/miniconda3/envs/aim2/bin/python scripts/run_pgm_replay.py \
      --dump results/aimreplay/adult_eps0p1_aimfull_seed1_measurements.json \
      --seed 1 --out results/stage1/adult_eps0p1_pgmrep_seed1.csv
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd


def _project_simplex(v: np.ndarray, total: float) -> np.ndarray:
    """逐字复制 resevo.initialization._project_simplex（本脚本在 aim2 环境跑，不装 resevo）。"""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    idx = np.arange(1, len(u) + 1)
    cond = u - cssv / idx > 0
    rho = idx[cond][-1]
    tau = cssv[rho - 1] / rho
    return np.maximum(v - tau, 0.0)


def _pool_vector(y: np.ndarray, sigma: float, total: float, z: float) -> np.ndarray:
    """逐字复制 resevo.denoise.pool_vector 的一阶均分规则（σ 同格同值）。"""
    known = np.abs(y) >= z * sigma
    if known.all():
        return y.copy()
    out = np.zeros_like(y)
    out[known] = np.maximum(y[known], 0.0)
    rest = float(total) - out[known].sum()
    if rest < 0.0:
        return _project_simplex(y, float(total))
    out[~known] = rest / int((~known).sum())
    return out


def pool_oneway_records(records: list[dict], total: int, z: float, scope: str) -> tuple[list[dict], dict]:
    """给 PGM 喂与我们 --oneway-pool/--oneway-pool-scope 完全相同的清洗后一阶。

    同字段重复一阶测量先逆方差合并（与 aim_replay_exam.py 同式），再按 z·σ 规则清洗，
    以一条测量（清洗后向量，合并后 σ）替换原来那几条；高斯似然下合并前后等价，
    被清洗字段以外的测量逐字不动。scope=uncovered 只处理没出现在任何 ≥2 阶团里的字段。
    """
    covered = {a for r in records if len(r["clique"]) >= 2 for a in r["clique"]}
    groups: dict[str, list[dict]] = {}
    for r in records:
        if len(r["clique"]) == 1:
            groups.setdefault(r["clique"][0], []).append(r)
    stats = {"fields": [], "cells": 0, "pooled": 0}
    replaced: dict[str, dict] = {}
    for attr, recs in groups.items():
        if scope == "uncovered" and attr in covered:
            continue
        prec = np.array([1.0 / float(r["sigma"]) ** 2 for r in recs])
        ys = np.array([np.asarray(r["y"], dtype=np.float64) for r in recs])
        y = (ys * prec[:, None]).sum(axis=0) / prec.sum()
        sigma = float(1.0 / np.sqrt(prec.sum()))
        c = _pool_vector(y, sigma, total, z)
        stats["fields"].append(attr)
        stats["cells"] += int(y.size)
        stats["pooled"] += int((np.abs(y) < z * sigma).sum())
        replaced[attr] = {**recs[0], "y": c.tolist(), "sigma": sigma}
    out, seen = [], set()
    for r in records:
        if len(r["clique"]) == 1 and r["clique"][0] in replaced:
            if r["clique"][0] not in seen:
                seen.add(r["clique"][0])
                out.append(replaced[r["clique"][0]])
            continue
        out.append(r)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="AIM 测量录像 -> 独立 Private-PGM 出表")
    ap.add_argument("--dump", required=True, help="run_aim_full.py --dump-measurements 产物")
    ap.add_argument("--seed", type=int, default=1, help="合成采样随机种子（numpy 全局）")
    ap.add_argument("--iters", type=int, default=1000, help="MirrorDescent 迭代数上限，AIM 默认 max_iters=1000")
    ap.add_argument("--tol", type=float, default=None,
                    help="跑到收敛：每块（mbi CALLBACK_EVERY 步）相对损失改善 ≤ tol 连续 patience 块即停，不给则固定跑满 iters")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--rows", type=int, default=0, help="合成行数，0 为录像里的真表行数")
    ap.add_argument("--known-total", choices=["rows", "auto"], default="rows",
                    help="模型总量：rows 喂真表行数（与我们引擎、GSD 同口径）；auto 留空由 mbi 从噪声测量自估（AIM 原样）")
    ap.add_argument("--max-model-mb", type=float, default=4096.0,
                    help="联结树参数表内存上限 MB，超限直接退出记录为证据")
    ap.add_argument("--oneway-pool", type=float, default=0.0,
                    help="与 run_plants.py 同名开关：一阶平分清洗阈值 z（|y|<z·σ 的格子平分剩余人数），0 关")
    ap.add_argument("--oneway-pool-scope", choices=["all", "uncovered"], default="all",
                    help="清洗范围：all 全部一阶字段；uncovered 只清洗没出现在任何 ≥2 阶团里的字段")
    ap.add_argument("--stop-floor-mult", type=float, default=0.0,
                    help="地板停倍数 m：mbi 损失 0.5·Σ((Qμ−y)/σ)² 首次 ≤ m·0.5·格数 即停（格数=所有测量向量长度之和，"
                         "重复团分开计）；0 关。开着时 tol/patience 与 iters 作为兜底仍生效，停法写入 *_stop.json")
    ap.add_argument("--check-every", type=int, default=50,
                    help="每多少步查一次地板（须整除 50；tol 仍固定每 50 步比一次 = mbi CALLBACK_EVERY 原语义）；"
                         "1 即逐步查，与我们引擎逐轮查地板同口径")
    ap.add_argument("--curve", default=None, help="可选：逐块损失曲线 CSV（iters,loss,elapsed_s）")
    ap.add_argument("--out", required=True, help="输出合成表 CSV")
    args = ap.parse_args()

    from mbi import Domain, estimation, junction_tree
    from mbi.marginal_loss import LinearMeasurement

    with open(args.dump, "r", encoding="utf-8") as fh:
        dump = json.load(fh)
    columns = list(dump["columns"])
    domain = Domain(columns, [int(s) for s in dump["domain_sizes"]])
    rows = args.rows or int(dump["num_rows"])

    records = list(dump["measurements"])
    if args.oneway_pool > 0.0:
        records, pst = pool_oneway_records(records, int(dump["num_rows"]), args.oneway_pool, args.oneway_pool_scope)
        print(f"一阶清洗 z={args.oneway_pool} scope={args.oneway_pool_scope}：字段 {pst['fields']}，"
              f"{pst['cells']} 格中 {pst['pooled']} 格未测出被平分；测量条数 {len(dump['measurements'])} -> {len(records)}",
              flush=True)

    measurements = []
    for rec in records:
        cl = tuple(rec["clique"])
        shape = [int(domain[a]) for a in cl]
        if shape != [int(s) for s in rec["shape"]]:
            raise SystemExit(f"录像团 {cl} 形状 {rec['shape']} 与域 {shape} 不符")
        y = np.asarray(rec["y"], dtype=np.float64)
        if y.size != int(np.prod(shape)):
            raise SystemExit(f"录像团 {cl} 长度 {y.size} 与域 {shape} 不符")
        measurements.append(LinearMeasurement(y, cl, stddev=float(rec["sigma"])))
    n1 = sum(1 for m in measurements if len(m.clique) == 1)
    cliques = [m.clique for m in measurements]
    model_mb = junction_tree.hypothetical_model_size(domain, cliques)
    print(f"{dump['data']} eps={dump['epsilon']} 录像种子={dump['seed']} 测量 {len(measurements)} 条"
          f"（一阶 {n1}，高阶 {len(measurements) - n1}），联结树参数表 {model_mb:.1f} MB，"
          f"iters={args.iters} tol={args.tol} rows={rows} known_total={args.known_total}", flush=True)
    if model_mb > args.max_model_mb:
        raise SystemExit(f"树宽爆炸: 参数表 {model_mb:.1f} MB 超上限 {args.max_model_mb:.0f} MB")

    t0 = time.time()
    known_total = float(rows) if args.known_total == "rows" else None
    n_cells = int(sum(m.noisy_measurement.size for m in measurements))
    floor = args.stop_floor_mult * 0.5 * n_cells if args.stop_floor_mult > 0.0 else None
    if 50 % args.check_every:
        raise SystemExit(f"--check-every {args.check_every} 须整除 50")
    if floor is None and args.check_every == 50 and args.curve is None:
        model = estimation.MirrorDescent().estimate(domain, measurements, known_total=known_total, iters=args.iters,
                                                    tol=args.tol, patience=args.patience)
        stop = None
    else:
        print(f"地板停 m={args.stop_floor_mult}：格数 {n_cells}，地板 {floor if floor is not None else '关'}，"
              f"每 {args.check_every} 步查一次", flush=True)
        model, stop = fit_with_stops(estimation.MirrorDescent(), domain, measurements, known_total, args.iters,
                                     args.tol, args.patience, floor, args.check_every, args.curve)
    fit_s = time.time() - t0
    np.random.seed(args.seed)
    synth = model.synthetic_data(rows=rows)
    out = pd.DataFrame({c: np.asarray(synth.data[c]) for c in columns})
    out.to_csv(args.out, index=False)
    print(f"拟合 {fit_s:.0f} 秒，合成 {len(out)} 行 -> {args.out}", flush=True)
    if stop is not None:
        stop.update({"fit_s": fit_s, "n_cells": n_cells, "stop_floor_mult": args.stop_floor_mult,
                     "iters_cap": args.iters, "tol": args.tol, "patience": args.patience,
                     "check_every": args.check_every, "known_total": known_total})
        stop_path = args.out[:-4] + "_stop.json" if args.out.endswith(".csv") else args.out + "_stop.json"
        with open(stop_path, "w", encoding="utf-8") as fh:
            json.dump(stop, fh, ensure_ascii=False, indent=1)
        ratio = stop["final_loss"] / floor if floor else float("nan")
        print(f"停法 {stop['stop_reason']}，{stop['iters_done']} 步，起步损失 {stop['initial_loss']:.1f}，"
              f"终损失 {stop['final_loss']:.1f}，÷地板 {ratio:.4f} -> {stop_path}", flush=True)


def fit_with_stops(est, domain, measurements, known_total, iters, tol, patience, floor, block, curve_path):
    """逐字复刻 mbi Estimator.estimate() 的循环（_init → 分块 _step → tol 早停 → _finalize），只多两件事：
    每块查一次「损失 ≤ 地板」即停，并把每块的损失记下来。block=50 时每块与 mbi _multi_step 同为 50 步 scan。"""
    import math

    import jax
    import jax.numpy as jnp
    from mbi import marginal_loss
    from mbi.estimation import minimum_variance_unbiased_total

    if known_total is None:
        known_total = minimum_variance_unbiased_total(measurements)
    loss_fn = marginal_loss.from_linear_measurements(measurements, domain)
    state = est._init(domain, loss_fn, known_total)
    state = jax.tree.map(jnp.copy, state)

    @jax.jit
    def multi_step(state, loss_fn, known_total):
        def step(s, _):
            return est._step(s, loss_fn, known_total, ()), None
        return jax.lax.scan(step, state, None, length=block)[0]

    t0 = time.time()
    initial_loss = float(state.loss)
    curve = [(0, initial_loss, 0.0)]
    prev_loss, stalls, done, reason = float("inf"), 0, 0, "iter_limit"
    for _ in range(math.ceil(iters / block)):
        state = multi_step(state, loss_fn, known_total)
        done += block
        loss = float(state.loss)
        curve.append((done, loss, time.time() - t0))
        if floor is not None and loss <= floor:
            reason = "floor_reached"
            break
        # tol 仍按 mbi 原语义每 50 步（CALLBACK_EVERY）比一次，与 block 大小无关（block 需整除 50）
        if tol is not None and done % 50 == 0:
            converged = prev_loss - loss <= tol * max(abs(loss), 1.0)
            stalls = stalls + 1 if converged else 0
            prev_loss = loss
            if stalls >= patience:
                reason = "loss_plateau"
                break
    model = est._finalize(state, known_total, constraints=())
    if curve_path:
        with open(curve_path, "w", encoding="utf-8") as fh:
            fh.write("iters,loss,elapsed_s\n")
            for it, ls, el in curve:
                fh.write(f"{it},{ls!r},{el:.3f}\n")
    stop = {"stop_reason": reason, "iters_done": done, "initial_loss": initial_loss, "final_loss": float(state.loss),
            "floor": floor, "final_over_floor": (float(state.loss) / floor) if floor else None}
    return model, stop


if __name__ == "__main__":
    main()
