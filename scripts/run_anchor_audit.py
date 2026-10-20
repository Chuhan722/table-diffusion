"""四条记录小表例子的端到端逐数字对账报告。

把实现算出的每个量与 docs/锚点数据.md 抄录的文档纸面值逐项比较并打印。
用法  ./.venv/bin/python scripts/run_anchor_audit.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resevo.candidates import BlockSupport, full_single_row_supports, normalize_block
from resevo.engine import build_kernel
from resevo.gain import block_deltas, block_gains
from resevo.state import make_workload, table_answers, table_loss, table_residual
from resevo.stepsize import analytic_step
from resevo.tilt import calibrate_beta, tilted_distributions

import itertools

FEATURES = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
TARGET = np.array([3.0, 1.0, 1.0])
WEIGHTS = np.ones(3)
IDS = np.array([0, 1, 2, 3])

checks = []


def check(name, got, expect, tol):
    got_arr = np.atleast_1d(np.asarray(got, dtype=float))
    expect_arr = np.atleast_1d(np.asarray(expect, dtype=float))
    diff = float(np.max(np.abs(got_arr - expect_arr)))
    ok = diff <= tol
    checks.append(ok)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}  最大绝对差 {diff:.3e}  容差 {tol:.0e}")
    if not ok:
        print(f"       实现值 {got}\n       文档值 {expect}")


def main():
    workload = make_workload(FEATURES, TARGET, WEIGHTS)
    print("== 账本 ==")
    check("整表答案 q", table_answers(workload, IDS), [2, 2, 1], 0)
    check("残差 e", table_residual(workload, IDS), [1, -1, 0], 0)
    check("损失 L", table_loss(workload, IDS), 1.0, 0)
    stepped = IDS.copy()
    stepped[1] = 2
    check("手算锚 01->10 后损失", table_loss(workload, stepped), 0.0, 0)

    print("== 增益矩阵 16 项 ==")
    e = table_residual(workload, IDS)
    gain_anchor = np.array(
        [[0, -1.5, 0.5, -1.5], [0.5, 0, 1, 0], [-1.5, -3, 0, -2], [-1.5, -2, 0, 0]]
    )
    gains_by_state = np.empty((4, 4))
    for i in range(4):
        src = (int(IDS[i]),)
        outcomes = (src,) + tuple((u,) for u in range(4) if (u,) != src)
        d = block_deltas(FEATURES, src, outcomes)
        g = block_gains(d, e, WEIGHTS)
        for u, gv in zip(outcomes, g):
            gains_by_state[i, u[0]] = gv
    check("增益矩阵", gains_by_state, gain_anchor, 1e-12)

    print("== 熵校准 ==")
    blocks = [normalize_block(s, IDS, 4) for s in full_single_row_supports(4, 4)]
    deltas = [block_deltas(FEATURES, nb.outcomes[0], nb.outcomes) for nb in blocks]
    gains = [block_gains(d, e, WEIGHTS) for d in deltas]
    refs = [nb.reference for nb in blocks]
    check("参考分布 R 行1", refs[0], [0.9, 1 / 30, 1 / 30, 1 / 30], 1e-15)
    tilt = calibrate_beta(gains, refs, 1.0, alpha=0.5)
    check("最大增益和 M", tilt.max_gain_sum, 1.5, 1e-12)
    check("要求 delta", tilt.required_gain, 0.75, 1e-12)
    d_table = {0: -0.366667, 1: 0.065895, 2: 0.261787, 3: 0.493206, 4: 0.754670, 8: 1.316503}
    got = [tilted_distributions(gains, refs, b)[1] for b in d_table]
    check("D(beta) 六点表", got, list(d_table.values()), 5e-7)
    check("D(3.9814026412)", tilted_distributions(gains, refs, 3.9814026412)[1], 0.75, 1e-9)
    check("beta 根", tilt.beta, 3.981402641214968, 1e-9)
    check("D(beta*)", tilt.direction_gain, 0.75, 1e-12)

    print("== 整代矩 步长 最终核 ==")
    result = build_kernel(workload, IDS)
    b = np.array([float(bl.rates.sum()) for bl in result.blocks])
    check("离开速率 b", b, [0.213419, 0.696332, 0.000107527, 0.0358140], 1e-6)
    check("跨行交互 C", result.interaction, 0.1548345932, 1e-9)
    check("步长 h", result.step, 1.4360970925, 1e-9)
    k_anchor = np.array(
        [
            [0.693510, 0.000107, 0.306277, 0.000107],
            [0.118242, 0.0, 0.865607, 0.016152],
            [0.000136, 3.4552e-7, 0.999846, 0.0000185],
            [0.000131, 0.0000179, 0.051284, 0.948568],
        ]
    )
    check("最终核 K 16 格", result.row_marginals, k_anchor, 1e-6)
    rho = np.array([float(bl.probabilities[1:].sum()) for bl in result.blocks])
    check("行变化概率 rho", rho, [0.306490, 1.0, 0.0001544, 0.0514324], 1e-6)
    check("期望改行数", result.expected_changed_rows, 1.358077, 1e-6)

    print("== 期望损失恒等式 ==")
    check("解析 E[L]", result.expected_loss, 0.242254152907815, 1e-12)
    combos = []
    for picks in itertools.product(*[range(len(bl.outcomes)) for bl in result.blocks]):
        p = 1.0
        nxt = IDS.copy()
        for bl, j in zip(result.blocks, picks):
            p *= float(bl.probabilities[j])
            nxt[list(bl.rows)] = bl.outcomes[j]
        combos.append((p, table_loss(workload, nxt)))
    check("256 枚举平均损失=解析值", sum(p * l for p, l in combos), result.expected_loss, 1e-12)
    check("损失上升概率", sum(p for p, l in combos if l > 1.0), 0.0227002, 1e-6)
    check("解析上界 L-hD/2", result.analytic_upper_bound, 1.0 - result.step * 0.75 / 2, 1e-12)

    print("== B_max=0.25 变体 ==")
    capped = build_kernel(workload, IDS, max_expected_rows=0.25)
    check("预算步长 h", capped.step, 0.2643622352, 1e-9)
    check("期望改行数", capped.expected_changed_rows, 0.25, 1e-12)
    check("E[L]", capped.expected_loss, 0.8125493094, 1e-9)

    print("== 独立小例锚 ==")
    for eps, anchor in [(1e-1, 21.9722), (1e-3, 2197.2246), (1e-6, 2197224.5773)]:
        t = calibrate_beta([np.array([0.0, eps])], [np.array([0.9, 0.1])], 1.0, alpha=0.5)
        check(f"微小增益 eps={eps:g} 的 beta", t.beta, anchor, anchor * 1e-5)
    check("三行例步长 h*", analytic_step(1.0, 1.5, 3.0), 0.25, 0)

    two_row = build_kernel(
        make_workload(FEATURES, np.array([1.0, 1.0, 0.0]), np.array([2.0, 2.0, 1.0])),
        np.array([0, 3]),
        supports=[BlockSupport((0, 1), tuple(itertools.product(range(4), repeat=2)))],
    )
    check("双行块 E[L]", two_row.expected_loss, 0.0040808634, 1e-9)

    total = len(checks)
    passed = sum(checks)
    print(f"\n对账结果 {passed}/{total} 项通过")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
