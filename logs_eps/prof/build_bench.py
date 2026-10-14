"""GPU build 微基准：抓一轮稳态 build 的入参，原地重跑 N 次，量墙钟 / CPU 线程时间 / 核发射数。

用法（与 run_plants.py 参数逐字相同，另加环境变量）：
  BENCH_ROUND=20 BENCH_REPEAT=30 .venv/bin/python -u -B logs_eps/prof/build_bench.py --gpu ... <run_plants 参数>
设备同步改为阻塞等待（cudaDeviceScheduleBlockingSync），同步点上 CPU 不自旋，
线程 CPU 时间 ≈ 纯主机侧开销（Python + 发射），墙钟 − CPU ≈ 等 GPU 的时间。
抓到入参后基准完即退出，不跑完整场。结果以 "BENCH" 开头打到 stderr。
"""
from __future__ import annotations

import ctypes
import glob
import os
import runpy
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# 先于任何 CUDA 初始化设置调度标志：4 = cudaDeviceScheduleBlockingSync
_libs = glob.glob(str(REPO / ".venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"))
if _libs and os.environ.get("BENCH_SPIN", "") != "1":
    _rt = ctypes.CDLL(_libs[0])
    _err = _rt.cudaSetDeviceFlags(ctypes.c_uint(4))
    print(f"BENCH cudaSetDeviceFlags(BlockingSync) -> {_err}", file=sys.stderr)

import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402

from resevo import gpukernel  # noqa: E402

ROUND = int(os.environ.get("BENCH_ROUND", "20"))
REPEAT = int(os.environ.get("BENCH_REPEAT", "30"))

# 操作计数：用 sys.setprofile 数 build 期间对 cupy 扩展函数/方法的 C 调用（≈ 每次 cupy 运算一条，核发射数的下界）
OPS = [0, 0]  # [cupy C 调用, 其中 asnumpy/同步类]
_SYNC_NAMES = {"asnumpy", "get", "synchronize", "item", "tolist", "nonzero", "flatnonzero", "bincount", "unique"}


def _prof(frame, event, arg):
    if event != "c_call":
        return
    owner = getattr(arg, "__self__", None)
    mod = getattr(type(owner), "__module__", None) if owner is not None else getattr(arg, "__module__", None)
    if mod and mod.startswith("cupy"):
        OPS[0] += 1
        if getattr(arg, "__name__", "") in _SYNC_NAMES:
            OPS[1] += 1


G = gpukernel.GpuBatchContext
_orig_build = G.build
CALLS = [0]


def _bench(self, *a, **k):
    CALLS[0] += 1
    if CALLS[0] < ROUND:
        return _orig_build(self, *a, **k)
    grouped, menu = a[0], a[1]
    print(f"BENCH 抓到第 {CALLS[0]} 轮 build：组数 {grouped.num_groups}，路径数 {menu.num_paths}，"
          f"题数 {self.num_queries}，稀疏 {self.sparse}", file=sys.stderr)
    # 预热一次（也验证可重入）
    r0 = _orig_build(self, *a, **k)
    cp.cuda.Device().synchronize()
    walls, cpus = [], []
    for _ in range(REPEAT):
        t0, c0 = time.perf_counter(), time.thread_time()
        r = _orig_build(self, *a, **k)
        cp.cuda.Device().synchronize()
        walls.append(time.perf_counter() - t0)
        cpus.append(time.thread_time() - c0)
        assert r.step == r0.step and r.old_loss == r0.old_loss
    OPS[0] = OPS[1] = 0
    sys.setprofile(_prof)
    _orig_build(self, *a, **k)
    sys.setprofile(None)
    w = np.array(walls) * 1000
    c = np.array(cpus) * 1000
    print(f"BENCH build ×{REPEAT}：墙钟 中位 {np.median(w):.1f} ms（p10 {np.percentile(w,10):.1f} p90 {np.percentile(w,90):.1f}），"
          f"CPU 线程 中位 {np.median(c):.1f} ms，等 GPU ≈ {np.median(w - c):.1f} ms", file=sys.stderr)
    print(f"BENCH 每次 build cupy C 调用 {OPS[0]} 次，其中同步类（asnumpy/nonzero/bincount/unique…） {OPS[1]} 次", file=sys.stderr)
    # 各子相的墙钟 / CPU 拆分（逐个套壳，单独同步）
    for nm in ("_state_terms", "_delta_direct", "_direct_entries", "_halfspace_delta",
               "_calibrate_flat", "_drift_sparse", "_merge_keys"):
        if not hasattr(G, nm):
            continue
        acc = {"w": 0.0, "c": 0.0, "n": 0}
        fn = getattr(G, nm)

        def mk(fn, acc):
            def inner(self2, *aa, **kk):
                cp.cuda.Device().synchronize()
                t0, c0 = time.perf_counter(), time.thread_time()
                out = fn(self2, *aa, **kk)
                cp.cuda.Device().synchronize()
                acc["w"] += time.perf_counter() - t0
                acc["c"] += time.thread_time() - c0
                acc["n"] += 1
                return out
            return inner
        setattr(G, nm, mk(fn, acc))
        for _ in range(REPEAT):
            _orig_build(self, *a, **k)
        setattr(G, nm, fn)
        n = max(acc["n"], 1)
        print(f"BENCH   {nm:18s} 每次 build 调 {n / REPEAT:.1f} 次  墙钟 {acc['w'] / REPEAT * 1000:6.1f} ms"
              f"  CPU {acc['c'] / REPEAT * 1000:6.1f} ms", file=sys.stderr)
    sys.stderr.flush()
    os._exit(0)


G.build = _bench

sys.argv = [str(REPO / "scripts" / "run_plants.py")] + sys.argv[1:]
runpy.run_path(str(REPO / "scripts" / "run_plants.py"), run_name="__main__")
