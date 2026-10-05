"""Issue #53 联合完整长轨迹的结果盲任务清单基础设施。

本模块只负责一次性列全 ``test_300x10``、``nltcs`` 和三组方法的
配对任务。它不加载数据、不调用生成器、不规定正式随机种子/轮数，也不运行实验。
后续冻结协议和执行器只能消费完整清单，不能先看一个数据集的结果再删减另一个。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np


Dataset = Literal["test_300x10", "nltcs"]
Arm = Literal["factor_b_s8", "independent_b_s0", "gap_l1_global_s8"]

DATASET_ORDER: tuple[Dataset, ...] = ("test_300x10", "nltcs")
ARM_ORDER: tuple[Arm, ...] = (
    "factor_b_s8",
    "independent_b_s0",
    "gap_l1_global_s8",
)

NO_GATE_KERNEL_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "tol": float("inf"),
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "return_final_table": True,
    }
)

_ARM_KERNEL_PARAMS: Mapping[Arm, Mapping[str, object]] = MappingProxyType(
    {
        "factor_b_s8": MappingProxyType(
            {
                "factorized_gibbs_sweeps": 8,
                "factorized_gibbs_use_compiled_workload": True,
                "gap_l1_sweeps": 0,
            }
        ),
        "independent_b_s0": MappingProxyType(
            {
                "factorized_gibbs_sweeps": 0,
                "factorized_gibbs_use_compiled_workload": False,
                "gap_l1_sweeps": 0,
            }
        ),
        "gap_l1_global_s8": MappingProxyType(
            {
                "factorized_gibbs_sweeps": 0,
                "factorized_gibbs_use_compiled_workload": False,
                "gap_l1_sweeps": 8,
            }
        ),
    }
)


@dataclass(frozen=True)
class JointTrajectoryTask:
    """一个由数据集、方法、随机种子和固定轮数唯一定位的完整轨迹。"""

    dataset: Dataset
    arm: Arm
    seed: int
    rounds: int

    @property
    def task_id(self) -> str:
        return f"seed_{self.seed}__{self.arm}__{self.dataset}"


@dataclass(frozen=True)
class JointTrajectoryPlan:
    """完整且在读取任何结果前构造的两数据三组任务计划。"""

    seeds: tuple[int, ...]
    rounds_by_dataset: Mapping[Dataset, int]
    tasks: tuple[JointTrajectoryTask, ...]
    protocol_frozen: bool = False
    generation_started: bool = False


def _validated_seed(value: object) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"seed 必须是非负整数，得到 {value!r}")
    return int(value)


def _validated_rounds(value: object, dataset: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(
            f"{dataset} 的 rounds 必须是正整数，得到 {value!r}"
        )
    return int(value)


def _validated_rounds_by_dataset(
    rounds_by_dataset: Mapping[str, object],
) -> Mapping[Dataset, int]:
    if not isinstance(rounds_by_dataset, Mapping):
        raise TypeError("rounds_by_dataset 必须是映射")
    keys = tuple(rounds_by_dataset)
    if set(keys) != set(DATASET_ORDER) or len(keys) != len(DATASET_ORDER):
        raise ValueError(
            "rounds_by_dataset 必须且只能同时包含 "
            f"{list(DATASET_ORDER)}"
        )
    validated = {
        dataset: _validated_rounds(rounds_by_dataset[dataset], dataset)
        for dataset in DATASET_ORDER
    }
    return MappingProxyType(validated)


def arm_kernel_params(arm: Arm) -> dict[str, object]:
    """返回一组方法相对共同完整生成配置的唯一核差量。"""

    if arm not in ARM_ORDER:
        raise ValueError(f"未知长轨迹方法：{arm!r}")
    params = dict(NO_GATE_KERNEL_PARAMS)
    params.update(_ARM_KERNEL_PARAMS[arm])
    return params


def build_joint_trajectory_plan(
    *,
    seeds: Sequence[int],
    rounds_by_dataset: Mapping[str, object],
) -> JointTrajectoryPlan:
    """在结果读取前一次性构造两数据、三方法的完整配对矩阵。

    排序固定为 ``seed -> arm -> dataset``，因此同一随机种子和方法下的
    两个数据集相邻。调用方没有数据集或方法子集参数，不能构造只跑一半的计划。
    """

    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds 必须是非空整数序列")
    validated_seeds = tuple(_validated_seed(seed) for seed in seeds)
    if not validated_seeds:
        raise ValueError("seeds 不得为空")
    if len(set(validated_seeds)) != len(validated_seeds):
        raise ValueError("seeds 不得重复")
    validated_rounds = _validated_rounds_by_dataset(rounds_by_dataset)

    tasks = tuple(
        JointTrajectoryTask(
            dataset=dataset,
            arm=arm,
            seed=seed,
            rounds=validated_rounds[dataset],
        )
        for seed in validated_seeds
        for arm in ARM_ORDER
        for dataset in DATASET_ORDER
    )
    expected_count = len(validated_seeds) * len(ARM_ORDER) * len(DATASET_ORDER)
    if len(tasks) != expected_count or len({task.task_id for task in tasks}) != len(
        tasks
    ):
        raise AssertionError("联合长轨迹任务矩阵不完整或地址不唯一")

    return JointTrajectoryPlan(
        seeds=validated_seeds,
        rounds_by_dataset=validated_rounds,
        tasks=tasks,
    )


def plan_manifest(plan: JointTrajectoryPlan) -> dict[str, object]:
    """生成不读取数据或结果的可序列化任务清单。"""

    if not isinstance(plan, JointTrajectoryPlan):
        raise TypeError("plan 必须是 JointTrajectoryPlan")
    return {
        "mode": "plan_only_no_data_read_no_generation",
        "datasets": list(DATASET_ORDER),
        "arms": list(ARM_ORDER),
        "seeds": list(plan.seeds),
        "rounds_by_dataset": dict(plan.rounds_by_dataset),
        "trajectory_count": len(plan.tasks),
        "task_ids": [task.task_id for task in plan.tasks],
        "protocol_frozen": plan.protocol_frozen,
        "generation_started": plan.generation_started,
        "partial_dataset_plan_allowed": False,
        "result_dependent_task_removal_allowed": False,
    }
