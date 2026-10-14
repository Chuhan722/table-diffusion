"""引擎分相计时器（不改引擎源码，只在导入后给关键函数套计时壳，GPU 相前后同步）。

用法（与 run_plants.py 参数逐字相同）：
  .venv/bin/python -u -B logs_eps/prof/phase_timer.py --data ... --rounds ... <其余 run_plants 参数>
每相累计墙钟秒数与调用次数在收官时打到 stderr（以 "PHASE" 开头的行）。
"""
from __future__ import annotations

import os
import runpy
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from resevo import batchkernel, batchmenu, dataset, gpukernel, grouping, structure  # noqa: E402

cp = None
if "--gpu" in sys.argv:
    import cupy as cp  # noqa: E402

ACC: "OrderedDict[str, list[float]]" = OrderedDict()


def _sync():
    if cp is not None:
        cp.cuda.Device().synchronize()


DEPTH = [0]            # 计时壳嵌套深度，只有最外层相计入“本轮已计时”
ROUND_TIMED = [0.0]    # 本轮最外层相累计（provider 壳每轮开头清零）
ROUND_MARKS: list[tuple[float, float]] = []  # (轮起点时刻, 上一轮已计时秒数)
EVENTS: list[tuple[str, float, float, int]] = []  # 前两轮的逐相事件 (标签, 起, 止, 深度)，排查首轮用


def wrap(obj, name, label, sync=True):
    fn = getattr(obj, name)
    acc = ACC.setdefault(label, [0.0, 0])
    wait = ACC.setdefault("gpu.前置同步等待(入相前排空设备，不计入各相)", [0.0, 0]) if sync else None

    def inner(*a, **k):
        if sync:
            t0 = time.perf_counter()
            _sync()
            wait[0] += time.perf_counter() - t0
            wait[1] += 1
        t = time.perf_counter()
        DEPTH[0] += 1
        try:
            return fn(*a, **k)
        finally:
            if sync:
                _sync()
            DEPTH[0] -= 1
            d = time.perf_counter() - t
            acc[0] += d
            acc[1] += 1
            if DEPTH[0] == 0:
                ROUND_TIMED[0] += d
            if len(ROUND_MARKS) <= 2:
                EVENTS.append((label, t, t + d, DEPTH[0]))

    setattr(obj, name, inner)


# GPU 核各相（build 总 = 条件计数 + 残差损失 + 答案差 + 半空间差 + 括根 + 其余装配/漂移）
G = gpukernel.GpuBatchContext
wrap(G, "build", "gpu.build 总")
wrap(G, "_condition_counts", "gpu.1 条件计数 cnt(G×Q)")
wrap(G, "_residual_loss", "gpu.2 残差/损失/组错位分(G×Q)")
wrap(G, "_delta_parts", "gpu.3 菜单答案差 delta")
# 第五十六步提速后的直算通路（旧代码没有这些方法，套壳时跳过）
for nm, label in (
    ("_feats_direct", "gpu.1' 直算特征 feats(G×Q)"),
    ("_delta_direct", "gpu.3' 直算答案差 delta"),
    ("_direct_entries", "gpu.3'' 直算条目(含分段)"),
    ("_state_terms", "gpu.0 整表项 state_terms(cnt/feats+残差)"),
    ("_residual_from_feats", "gpu.0b 残差/损失/组分 两次gemv(G×Q)"),
    ("_hs_scores", "gpu.0c 半空间整数分"),
    # 第五十九步 乙：稀疏整表项/漂移项（旧代码没有，套壳时跳过）
    ("_state_terms_sparse", "gpu.0s 稀疏整表项(整数计数+聚合)"),
    ("_drift_sparse", "gpu.6 漂移项(稀疏合并+分段)"),
    ("_merge_keys", "gpu.6' 键合并(排序+游程和)"),
):
    if hasattr(G, nm):
        wrap(G, nm, label)
wrap(G, "_halfspace_delta", "gpu.4 半空间差")
wrap(G, "_calibrate_flat", "gpu.5 括根(步长 beta)")
# CPU 相
# batchkernel 按名导入了这些函数，要在 batchkernel 的命名空间里套壳才生效
wrap(batchkernel, "group_state_ids", "cpu.分组 group_state_ids", sync=False)
wrap(batchkernel, "generate_batch_menu", "cpu.出菜单 generate_batch_menu", sync=False)
wrap(batchkernel, "build_paired_supports", "cpu.配对 build_paired_supports", sync=False)
wrap(batchkernel, "build_rescue_menu", "cpu.救援菜单 build_rescue_menu", sync=False)
wrap(batchkernel, "sample_batch_next", "cpu.抽样落表 sample_batch_next", sync=False)
wrap(dataset.StateRegistry, "register_table", "cpu.注册新状态 register_table", sync=False)
wrap(batchmenu.CodeBook, "sync", "cpu.码本同步 codebook.sync", sync=False)
wrap(dataset.StateRegistry, "build_workload", "启动.build_workload", sync=False)
for cls_name in ("StructShaper",):
    cls = getattr(structure, cls_name, None)
    if cls is not None:
        for nm in ("temperature_dev", "path_factors_dev"):
            if hasattr(cls, nm):
                wrap(cls, nm, f"gpu.结构尺 {nm}")
# 其它继承了 temperature_dev 的尺类（组合尺等）
for nm, cls in vars(structure).items():
    if isinstance(cls, type) and cls is not getattr(structure, "StructShaper", None):
        if "temperature_dev" in vars(cls):
            wrap(cls, "temperature_dev", f"gpu.结构尺 {nm}.temperature_dev")

# 计划提供器整体（含分组+挑组+出菜单）：包 evolve_batch 把 plan_provider 套壳
_orig_evolve = batchkernel.evolve_batch


def _evolve(state_ids, num_rounds, rng, plan_provider, *a, **k):
    acc = ACC.setdefault("cpu.计划 provider 总(分组+挑组+菜单)", [0.0, 0])

    class P:
        def __call__(self, *pa, **pk):
            t = time.perf_counter()
            ROUND_MARKS.append((t, ROUND_TIMED[0]))  # 轮起点；上一轮的已计时量随手记下
            ROUND_TIMED[0] = 0.0
            try:
                return plan_provider(*pa, **pk)
            finally:
                d = time.perf_counter() - t
                acc[0] += d
                acc[1] += 1
                ROUND_TIMED[0] += d
                if len(ROUND_MARKS) <= 2:
                    EVENTS.append(("provider", t, t + d, -1))

        def __getattr__(self, item):
            return getattr(plan_provider, item)

    acc_e = ACC.setdefault("evolve_batch 总", [0.0, 0])
    t = time.perf_counter()
    try:
        return _orig_evolve(state_ids, num_rounds, rng, P(), *a, **k)
    finally:
        _sync()
        acc_e[0] += time.perf_counter() - t
        acc_e[1] += 1


batchkernel.evolve_batch = _evolve

if os.environ.get("PHASE_GC", "") == "off":
    import gc

    gc.disable()
    print("PHASE_GC=off：已关闭循环垃圾回收", file=sys.stderr)

T0 = time.perf_counter()
try:
    sys.argv = [str(REPO / "scripts" / "run_plants.py")] + sys.argv[1:]
    runpy.run_path(str(REPO / "scripts" / "run_plants.py"), run_name="__main__")
finally:
    total = time.perf_counter() - T0
    print(f"PHASE 进程总墙钟 {total:.1f}s", file=sys.stderr)
    ev = ACC.get("evolve_batch 总", [0.0, 0])[0]
    print(f"PHASE 启动+收尾(总-evolve) {total - ev:.1f}s", file=sys.stderr)
    for label, (sec, n) in ACC.items():
        per = sec / n * 1000 if n else 0.0
        share = sec / ev * 100 if ev else 0.0
        print(f"PHASE {label:44s} {sec:8.1f}s  {n:7d}次  {per:8.1f}ms/次  占evolve {share:5.1f}%",
              file=sys.stderr)
    if len(ROUND_MARKS) > 2:
        # 逐轮：轮长 = 相邻两次 provider 起点之差；相外 = 轮长 − 本轮最外层各相之和（末轮无终点不算）
        durs = np.diff([m[0] for m in ROUND_MARKS]) * 1000
        timed = np.array([m[1] for m in ROUND_MARKS[1:]]) * 1000
        gap = durs - timed
        q = lambda x: " / ".join(f"{v:.1f}" for v in np.percentile(x, [0, 10, 50, 90, 100]))
        print(f"ROUND 轮长 ms 最小/p10/中位/p90/最大 {q(durs)}  均值 {durs.mean():.1f}", file=sys.stderr)
        print(f"ROUND 相外 ms 最小/p10/中位/p90/最大 {q(gap)}  均值 {gap.mean():.1f}  合计 {gap.sum()/1000:.1f}s", file=sys.stderr)
        top = np.argsort(-gap)[:8]
        print("ROUND 相外最大的 8 轮 (轮号:相外ms):", " ".join(f"{i+1}:{gap[i]:.0f}" for i in top), file=sys.stderr)
    if EVENTS:
        # 前两轮时间线：只列最外层（深度 0 或 provider），相邻事件之间的空档即相外代码
        ev0 = sorted((e for e in EVENTS if e[3] <= 0), key=lambda e: e[1])
        t0 = ROUND_MARKS[0][0]
        prev_end = t0
        for label, a, b, _ in ev0:
            if a - prev_end > 0.001:
                print(f"EV   {(prev_end - t0) * 1000:9.1f}  空档 {(a - prev_end) * 1000:8.1f} ms", file=sys.stderr)
            print(f"EV   {(a - t0) * 1000:9.1f}  {label[:40]:40s} {(b - a) * 1000:8.1f} ms", file=sys.stderr)
            prev_end = max(prev_end, b)
