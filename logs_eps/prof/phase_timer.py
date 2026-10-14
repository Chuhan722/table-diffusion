"""引擎分相计时器（不改引擎源码，只在导入后给关键函数套计时壳，GPU 相前后同步）。

用法（与 run_plants.py 参数逐字相同）：
  .venv/bin/python -u -B logs_eps/prof/phase_timer.py --data ... --rounds ... <其余 run_plants 参数>
每相累计墙钟秒数与调用次数在收官时打到 stderr（以 "PHASE" 开头的行）。
"""
from __future__ import annotations

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


def wrap(obj, name, label, sync=True):
    fn = getattr(obj, name)
    acc = ACC.setdefault(label, [0.0, 0])

    def inner(*a, **k):
        if sync:
            _sync()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if sync:
                _sync()
            acc[0] += time.perf_counter() - t
            acc[1] += 1

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
            try:
                return plan_provider(*pa, **pk)
            finally:
                acc[0] += time.perf_counter() - t
                acc[1] += 1

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
