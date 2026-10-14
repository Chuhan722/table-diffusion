"""GPU build 逐位核对：同一轮入参分别喂给当前 gpukernel 与旧版（resevo/_gpukernel_old.py），逐轮比对输出。

用法（与 run_plants.py 参数逐字相同）：
  .venv/bin/python -u -B logs_eps/prof/build_bitcheck.py --gpu ... <run_plants 参数>
旧版文件由 git show HEAD:src/resevo/gpukernel.py 导出放到 src/resevo/_gpukernel_old.py（临时，不入库）。
每轮打印 "BITCHECK" 行到 stderr，任一字段不逐位相同即列出字段名。
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from resevo import gpukernel  # noqa: E402
from resevo import _gpukernel_old as old  # noqa: E402

G = gpukernel.GpuBatchContext
OldG = old.GpuBatchContext
_orig_init = G.__init__
_orig_build = G.build
STATE = {"round": 0, "bad": 0, "old": None}
FIELDS = ("old_loss", "beta", "direction_gain", "interaction", "step", "expected_loss",
          "analytic_upper_bound", "expected_changed_rows", "max_gain_sum", "required_gain", "status")


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    self._bitcheck_init = (a, k)


def _same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x = np.asarray(x)
        y = np.asarray(y)
        return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(
            x.view(np.uint8) if x.dtype.kind == "f" else x,
            y.view(np.uint8) if y.dtype.kind == "f" else y,
        )
    return x == y


def _build(self, *a, **k):
    STATE["round"] += 1
    r = _orig_build(self, *a, **k)
    if STATE["old"] is None:
        ia, ik = self._bitcheck_init
        STATE["old"] = OldG(*ia, **ik)
    ro = STATE["old"].build(*a, **k)
    bad = []
    for nm in FIELDS:
        if hasattr(r, nm) and not _same(getattr(r, nm), getattr(ro, nm)):
            bad.append(nm)
    for nm in ("probabilities", "offsets", "group_scores"):
        if hasattr(r, nm) and not _same(getattr(r, nm), getattr(ro, nm)):
            bad.append(nm)
    if bad:
        STATE["bad"] += 1
        detail = []
        for nm in bad:
            x, y = getattr(r, nm), getattr(ro, nm)
            if isinstance(x, np.ndarray):
                d = np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))
                detail.append(f"{nm}: 最大差 {d.max():.3e} 条数 {(d > 0).sum()}")
            else:
                detail.append(f"{nm}: {x!r} vs {y!r}")
        print(f"BITCHECK 第 {STATE['round']} 轮 不同: " + "; ".join(detail), file=sys.stderr)
    else:
        print(f"BITCHECK 第 {STATE['round']} 轮 逐位相同 (路径 {a[1].num_paths}, 状态 {r.status})",
              file=sys.stderr)
    sys.stderr.flush()
    return r


G.__init__ = _init
G.build = _build

import atexit  # noqa: E402


@atexit.register
def _summary():
    print(f"BITCHECK 汇总：共 {STATE['round']} 轮，不同 {STATE['bad']} 轮", file=sys.stderr)
    sys.stderr.flush()


sys.argv = [str(REPO / "scripts" / "run_plants.py")] + sys.argv[1:]
runpy.run_path(str(REPO / "scripts" / "run_plants.py"), run_name="__main__")
