#!/usr/bin/env python3
"""第 6B-1B 阶段 NLTCS 显卡环境检查。

本脚本只构造人工张量，不读取数据集、第 6A 阶段状态、供体、
候选地址或第 6B-1A 阶段结果。它不生成候选表，也不消耗任何
冻结随机地址。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys


EXPECTED_PHYSICAL_GPU = "1"
EXPECTED_GPU_UUID = "GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-physical-gpu",
        default=EXPECTED_PHYSICAL_GPU,
        help="CUDA_VISIBLE_DEVICES 必须精确匹配的物理显卡编号",
    )
    return parser


def validate_gpu_environment(
    expected_physical_gpu: str = EXPECTED_PHYSICAL_GPU,
) -> dict[str, object]:
    """验证并返回可嵌入五段产物的设备身份。"""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_physical_gpu:
        raise RuntimeError(
            "必须通过 CUDA_VISIBLE_DEVICES 精确且只暴露物理 "
            f"{expected_physical_gpu} 号显卡"
        )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("第 6B-1B 阶段显卡路径需要 PyTorch") from error

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("必须只有一张可见且可用的 CUDA 显卡")

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    uuid = f"GPU-{properties.uuid}"
    if uuid != EXPECTED_GPU_UUID or properties.name != EXPECTED_GPU_NAME:
        raise RuntimeError("物理 1 号显卡的 UUID（唯一编号）或名称不匹配")

    # 只验证未来新核必需的整数计数、布尔条件、双精度
    # 误差和软概率基本算子；不引入任何真实查询或随机数。
    counts = torch.tensor([3, 7, 11, 13], dtype=torch.int64, device=device)
    deltas = torch.tensor([1, -2, 0, 3], dtype=torch.int64, device=device)
    targets = torch.tensor(
        [4.0, 6.0, 11.0, 15.0], dtype=torch.float64, device=device
    )
    denominators = torch.maximum(
        targets, torch.full_like(targets, 8.0)
    )
    candidate = counts + deltas
    error = torch.mean(
        torch.abs(targets - candidate.to(torch.float64)) / denominators
    )
    score = torch.tensor(0.125, dtype=torch.float64, device=device)
    probability = torch.sigmoid(2.0 * score)
    finite = bool(torch.isfinite(error) & torch.isfinite(probability))
    bidirectional = bool((probability > 0.0) & (probability < 1.0))
    torch.cuda.synchronize(device)

    if not finite or not bidirectional:
        raise RuntimeError("显卡双精度误差/软概率微型检查失败")

    return {
        "expected_physical_gpu": expected_physical_gpu,
        "physical_gpu_uuid": uuid,
        "visible_logical_gpu_count": torch.cuda.device_count(),
        "logical_gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_compile_available": hasattr(torch, "compile"),
        "triton_available": importlib.util.find_spec("triton") is not None,
        "torch_inductor_available": (
            importlib.util.find_spec("torch._inductor") is not None
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "required_dtypes_verified": ["int64", "bool", "float64"],
        "float64_error_reading": error.item(),
        "float64_probability_reading": probability.item(),
        "formal_dataset_read": False,
        "frozen_address_consumed": False,
        "candidate_generation_performed": False,
    }


def main() -> None:
    args = _parser().parse_args()
    result = {
        "status": "ready_for_gpu_kernel_validation",
        **validate_gpu_environment(args.expected_physical_gpu),
        "gpu_kernel_implemented": True,
    }
    print(json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
