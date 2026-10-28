"""重放400轮公开输入验证器至指定抽样点，保存状态/核/RNG与实际二项参数。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code-root", type=Path, required=True)
parser.add_argument("--worker", type=Path, required=True)
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--round-index", type=int, required=True)
args, worker_args = parser.parse_known_args()
if not 0 <= args.round_index < 400:
    parser.error("round-index 须在已声明的400轮协议范围内")
if args.worker.name not in ("verify_resevo_migration.py", "verify_resevo_gpu_recheck.py"):
    parser.error("worker 须为仓库已有的迁移或非整数GPU验证器")
if args.capture.exists():
    raise FileExistsError(f"拒绝覆盖已有捕获：{args.capture}")
sys.path.insert(0, str(args.code_root.resolve() / "src"))
import numpy as np
from resevo import batchkernel
from resevo.gpukernel import GpuBatchContext

state = {"round": -1, "active": False, "capture": None}
original_evolve = batchkernel.evolve_batch
original_cpu_build = batchkernel.build_batch_kernel
original_gpu_build = GpuBatchContext.build
original_sample = batchkernel.sample_batch_next


class CapturedBoundary(Exception):
    """完成请求的有界重放，不是生成器数值失败。"""


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def table_digest(registry, ids):
    rows = [registry.state_tuple(int(value)) for value in ids]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def evolve(*values, **kwargs):
    state["round"] = -1
    state["active"] = values[1] == 400  # 不把验证器的两轮预热当作正式轨迹。
    return original_evolve(*values, **kwargs)


def cpu_build(*values, **kwargs):
    state["round"] += 1
    return original_cpu_build(*values, **kwargs)


def gpu_build(*values, **kwargs):
    state["round"] += 1
    return original_gpu_build(*values, **kwargs)


def sample(result, ids, rng, registry):
    if not state["active"] or state["round"] != args.round_index:
        return original_sample(result, ids, rng, registry)
    starts = result.offsets[:-1]
    weights = result.probabilities.copy()
    weights[starts] = 0.0
    mass = np.add.reduceat(weights, starts)
    old_move = np.minimum(1.0, np.maximum(0.0, 1.0 - result.probabilities[starts]))
    limited = (mass > 0) & (old_move == 0)
    capture = dict(
        round_index=state["round"], current_ids_sha256=digest(ids),
        current_table_sha256=table_digest(registry, ids),
        probabilities_sha256=digest(result.probabilities),
        counts_sha256=digest(result.grouped.counts), offsets_sha256=digest(result.offsets),
        rng_before=rng.bit_generator.state, limited_segments=int(limited.sum()),
        limited_mass=mass[limited].tolist(),
    )

    class ObservedRng:
        def binomial(self, n, p):
            capture["effective_limited_p"] = np.asarray(p)[limited].tolist()
            return rng.binomial(n, p)

        def multinomial(self, n, p):
            return rng.multinomial(n, p)

    sampled = original_sample(result, ids, ObservedRng(), registry)
    capture["next_ids_sha256"] = digest(sampled)
    capture["next_table_sha256"] = table_digest(registry, sampled)
    capture["rng_after"] = rng.bit_generator.state
    capture["status"] = "captured"
    state["capture"] = capture
    raise CapturedBoundary()


batchkernel.evolve_batch = evolve
batchkernel.build_batch_kernel = cpu_build
GpuBatchContext.build = gpu_build
batchkernel.sample_batch_next = sample
sys.argv = [str(args.worker), "--code-root", str(args.code_root), *worker_args]
try:
    runpy.run_path(str(args.worker), run_name="__main__")
except CapturedBoundary:
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    with args.capture.open("x", encoding="utf-8") as stream:
        json.dump(state["capture"], stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("CAPTURED", args.round_index)
else:
    raise RuntimeError("未到达指定抽样点，不能生成捕获成功记录")
