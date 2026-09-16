"""期望下降验证，蒙特卡洛对齐精确恒等式与随机小表多轮下降。"""
import numpy as np

from resevo.engine import build_kernel, evolve, sample_next
from resevo.state import make_workload, table_loss


def test_montecarlo_matches_expected_loss(four_record):
    # 四条记录单步 5000 次抽样，平均新损失落在精确期望的蒙特卡洛误差内
    workload, ids = four_record
    result = build_kernel(workload, ids)
    rng = np.random.default_rng(20260911)
    losses = np.array(
        [table_loss(workload, sample_next(result, ids, rng)) for _ in range(5000)]
    )
    se = losses.std(ddof=1) / np.sqrt(len(losses))
    assert abs(losses.mean() - result.expected_loss) < 4 * se
    # 单次损失上升的样本必须真实保留，不被筛除
    assert (losses > result.old_loss).any()


def random_reachable_case(rng, num_fields=3, num_rows=16):
    """随机小表，目标取自一张随机真表的答案，保证损失可压到很低。"""
    m = 2**num_fields
    states = [tuple((x >> k) & 1 for k in range(num_fields)) for x in range(m)]
    predicates = [
        lambda s, k=k: s[k] == 1 for k in range(num_fields)
    ] + [lambda s: all(v == 1 for v in s)]
    features = np.array([[float(p(s)) for p in predicates] for s in states])
    true_ids = rng.integers(0, m, size=num_rows)
    target = features[true_ids].sum(axis=0)
    weights = rng.uniform(0.5, 2.0, size=features.shape[1]).round(3)
    start_ids = rng.integers(0, m, size=num_rows)
    return make_workload(features, target, weights), start_ids


def test_expected_loss_decreases_every_round():
    # 每一轮在抽样前就有解析保证 E[L'] <= L - hD/2 < L
    rng = np.random.default_rng(20260911)
    workload, ids = random_reachable_case(rng)
    out = evolve(workload, ids, 30, rng)
    for rec in out.records:
        if rec.status == "ok":
            assert rec.expected_loss <= rec.old_loss - rec.step * rec.direction_gain / 2 + 1e-12
            assert rec.expected_loss < rec.old_loss


def test_multi_seed_average_descent():
    # 十个种子的随机小表，30 轮后平均损失显著低于初始
    initial, final = [], []
    for seed in range(10):
        rng = np.random.default_rng(20260911 + seed)
        workload, ids = random_reachable_case(rng)
        L0 = table_loss(workload, ids)
        out = evolve(workload, ids, 30, rng)
        initial.append(L0)
        final.append(table_loss(workload, out.state_ids))
    assert np.mean(final) < 0.3 * np.mean(initial)


def test_freeze_reports_stop_reason():
    # 目标可达时多数运行冻结在零残差附近，停止原因必须是明确的停滞报告
    rng = np.random.default_rng(20260915)
    workload, ids = random_reachable_case(rng, num_fields=2, num_rows=6)
    out = evolve(workload, ids, 200, rng)
    if out.stop_reason == "no_positive_direction":
        assert out.records[-1].status == "no_positive_direction"
    else:
        assert len(out.records) == 200
