"""随机小表多轮演化冒烟，打印逐轮解析量与实际损失，验证期望下降。

用法  ./.venv/bin/python scripts/smoke_descent.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resevo.engine import build_kernel, sample_next
from resevo.state import make_workload, table_loss


def build_case(rng, num_fields=3, num_rows=16):
    m = 2**num_fields
    states = [tuple((x >> k) & 1 for k in range(num_fields)) for x in range(m)]
    predicates = [lambda s, k=k: s[k] == 1 for k in range(num_fields)]
    predicates.append(lambda s: all(v == 1 for v in s))
    features = np.array([[float(p(s)) for p in predicates] for s in states])
    true_ids = rng.integers(0, m, size=num_rows)
    target = features[true_ids].sum(axis=0)
    weights = rng.uniform(0.5, 2.0, size=features.shape[1]).round(3)
    start = rng.integers(0, m, size=num_rows)
    return make_workload(features, target, weights), start


def main():
    seed = 20260911
    rng = np.random.default_rng(seed)
    workload, ids = build_case(rng)
    print(f"随机小表冒烟，8 状态 4 查询 16 行，种子 {seed}")
    print(f"{'轮':>3} {'当前损失':>12} {'beta':>10} {'D':>10} {'C':>10} {'h':>8} {'解析E[L]':>12} {'状态':>6}")
    for k in range(30):
        result = build_kernel(workload, ids)
        print(
            f"{k:>3} {result.old_loss:>12.6f} {result.beta:>10.4f} "
            f"{result.direction_gain:>10.6f} {result.interaction:>10.6f} "
            f"{result.step:>8.4f} {result.expected_loss:>12.6f} {result.status:>6}"
        )
        if result.status == "no_positive_direction":
            print("冻结停止，本次支持没有正的一阶方向")
            break
        assert result.expected_loss < result.old_loss
        ids = sample_next(result, ids, rng)
    print(f"最终损失 {table_loss(workload, ids):.6f}")

    # 蒙特卡洛验证第一轮精确恒等式
    rng2 = np.random.default_rng(seed)
    workload2, ids2 = build_case(rng2)
    result = build_kernel(workload2, ids2)
    n = 20000
    samples = np.array(
        [table_loss(workload2, sample_next(result, ids2, rng2)) for _ in range(n)]
    )
    mc = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(n)
    print(
        f"第一轮解析 E[L]={result.expected_loss:.6f}  "
        f"蒙特卡洛 {n} 次均值 {mc:.6f}  差 {abs(mc - result.expected_loss):.2e}  "
        f"标准误差 {se:.2e}  偏离 {abs(mc - result.expected_loss) / se:.2f} 个标准误差"
    )


if __name__ == "__main__":
    main()
