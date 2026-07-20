"""
随机种子管理工具

确保每个实验完全可复现（铁律 4）。

## 为什么需要固定随机种子？

算法中多处涉及随机操作：
- 初始化合成表（如果从随机表开始）
- 选择参考记录（donor selection，根据适应度概率抽样）
- 决定是否复制属性（copy rate）
- 决定是否变异（mutation rate）
- 变异时选择新值

如果不固定种子，每次运行这些随机选择都不一样，导致：
- 同样的查询目标 + 同样的参数，可能生成完全不同的合成表
- 无法判断是算法改进了，还是只是运气好
- 科学实验无法复现

固定种子后，所有随机选择的顺序完全一样 → 实验完全可复现。

## 使用方式

当前采用旧方式（全局 seed）：简单直接，适合单次实验。

在主生成器开头调用一次：
    from table_diffevo.utils import set_seed

    set_seed(42)  # 种子值通常从配置文件读取
    # 之后所有 np.random.* 调用都是确定性的

## 新方式（get_rng）说明

如果后续需要模块间随机流隔离，可改用 get_rng 创建独立生成器。
当前阶段不使用，保持简单。
"""
import numpy as np


def set_seed(seed: int) -> None:
    """
    固定所有随机源的种子，确保实验可复现（旧方式，全局状态）。

    工作原理：
    - 计算机的"随机数"其实是伪随机：用数学算法从种子生成数列
    - 相同的种子 → 相同的数列
    - 不同的种子 → 不同的数列

    示例：
        np.random.seed(42)
        print(np.random.randn(3))  # [ 0.4967 -0.1383  0.6477]

        np.random.seed(42)  # 重置到同一起点
        print(np.random.randn(3))  # [ 0.4967 -0.1383  0.6477]  ← 完全一样

    Parameters
    ----------
    seed : int
        全局随机种子，通常从配置文件读取

    Notes
    -----
    - 应在实验最开始、任何随机操作之前调用一次
    - 如需对比多个种子的影响，每次实验开头用不同种子重新调用
    - 全局方式的缺点：模块间共享状态，可能互相影响；但足够简单直接
    """
    np.random.seed(seed)
    # 如果后续用到 Python 内置 random 模块，取消下面注释：
    # import random
    # random.seed(seed)


def get_rng(seed: int) -> np.random.Generator:
    """
    创建一个独立的随机数生成器（推荐用于模块化代码）。

    Parameters
    ----------
    seed : int
        种子值

    Returns
    -------
    np.random.Generator
        NumPy 新式随机数生成器实例

    Notes
    -----
    相比全局 set_seed，这种方式更模块化：
    - 每个组件可以有自己的 rng，避免全局状态污染
    - 便于并行实验（每个进程用不同 rng）

    用法示例：
        rng = get_rng(42)
        samples = rng.choice(data, size=10)
    """
    return np.random.default_rng(seed)
