"""
实验日志记录器

为接受规则对照实验和 α 调度实验提供结构化日志。
支持三个层级的日志：每轮（round）、每块（block）、探测（probe）。
"""
import json
import csv
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class RoundLog:
    """每轮日志记录"""
    seed: int
    arm: str  # 实验臂名称（如 "A0", "A1", "B0", "B2"）
    round: int
    block: int  # 当前块编号
    alpha: float
    u: float  # 归一化 α
    L1_current: float
    best_L1: float
    Q_current: float
    accepted: bool
    delta_L1: float  # L1_new - L1_old
    delta_Q: float   # Q_new - Q_old
    candidate_evaluations: int  # 累计候选评估次数


@dataclass
class BlockLog:
    """每块日志记录"""
    seed: int
    arm: str
    block: int
    block_start_L1: float
    block_end_L1: float
    block_improvement: float  # start - end (正数表示改善)
    acceptance_rate: float  # 块内接受率
    stall_count: int  # 当前停滞计数
    cooldown_remaining: int  # 剩余冷却块数
    probe_triggered: bool


@dataclass
class ProbeLog:
    """探测日志记录"""
    seed: int
    arm: str
    probe_id: int
    checkpoint_block: int  # 触发探测时的块编号
    checkpoint_L1: float   # checkpoint 时的 best_L1
    direction: str  # "DOWN", "HOLD", "UP"
    branch_seed: Optional[int]  # 分支随机种子（若使用）
    branch_budget: int  # 分支消耗的候选评估数
    branch_final_L1: float
    branch_final_Q: float
    winner: bool  # 是否为获胜分支
    winner_reason: Optional[str]  # 获胜原因（仅 winner=True 时有值）


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, output_dir: Path, force_overwrite: bool = False):
        """
        Parameters
        ----------
        output_dir : Path
            日志输出目录
        force_overwrite : bool, default=False
            是否允许覆盖非空目录。

            **定位（重要）**：`force_overwrite` 只是**非正式、best-effort** 的
            覆盖工具，用于本地调试/重跑时复用同一目录。它**不提供崩溃安全保证**
            （见 `save()` / `_publish()` 的中断残留说明）。**正式实验请勿复用目录**
            ——每次用一个新的（唯一/时间戳）`output_dir`，让首次发布走单步原子
            `os.replace` 路径，从根本上避开非空复用时的两步 rename 窗口。
        """
        self.output_dir = Path(output_dir)

        # 检查目录是否非空
        if self.output_dir.exists() and not force_overwrite:
            existing_files = list(self.output_dir.iterdir())
            if existing_files:
                raise ValueError(
                    f"输出目录非空: {self.output_dir}\n"
                    f"包含 {len(existing_files)} 个文件/目录。\n"
                    f"请使用不同的输出目录或传入 force_overwrite=True"
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 三个日志缓冲区
        self.round_logs: List[RoundLog] = []
        self.block_logs: List[BlockLog] = []
        self.probe_logs: List[ProbeLog] = []

        # 统计信息
        self.stats: Dict[str, Any] = {}

    def log_round(self, **kwargs):
        """记录一轮日志"""
        log = RoundLog(**kwargs)
        self.round_logs.append(log)

    def log_block(self, **kwargs):
        """记录一块日志"""
        log = BlockLog(**kwargs)
        self.block_logs.append(log)

    def log_probe(self, **kwargs):
        """记录探测日志"""
        log = ProbeLog(**kwargs)
        self.probe_logs.append(log)

    def add_stat(self, key: str, value: Any):
        """添加统计信息"""
        self.stats[key] = value

    def save(self):
        """保存所有日志到文件（目录级 staging 发布）。

        策略：把本次要输出的全部文件（各 CSV + summary.json）先用**最终
        文件名**写进一个唯一暂存目录，再把整个目录换到 output_dir。因此单个
        文件层面不会出现新旧拼接的半成品。

        发布分两种情形，崩溃安全保证**不同**（详见 `_publish()`）：
        - output_dir 为空（首次保存）：单步 os.replace，**真正零窗口原子**，
          对进程终止/断电也安全。**正式实验应始终走这条路径**（用新目录）。
        - output_dir 非空（force_overwrite 复用）：两步 rename——先把旧目录
          挪成备份，再把暂存挪进来。这条路径**不是崩溃安全**的：两次 rename
          之间正式目录短暂不存在，若此刻被 KeyboardInterrupt / 进程终止
          （`kill`）打断，正式目录会缺失，磁盘上只剩 `.backup-*` 与
          `.staging-*`。普通 `OSError`（如第二步 rename 失败）会回滚备份并抛出，
          但中断/终止无法用 try/except 覆盖。故 `force_overwrite` 仅作非正式
          best-effort 工具，正式实验请勿复用目录。中断残留的恢复办法见下方
          「中断恢复」。

        上一轮遗留的陈旧类别文件（如本次无 probe 时的旧 probes.csv）会随
        「整个旧目录被丢弃」而自动消失，无需再逐个清理。

        中断恢复（仅非空复用路径可能触发）
        --------------------------------
        若发现正式 output_dir 缺失、同级只剩 `.<name>.backup-*` / `.<name>.staging-*`：
        - `.backup-*` 是**完整的上一版**——把它改名回 output_dir 即可恢复旧数据；
        - `.staging-*` 是**完整的这一版**——若确认要用新数据，改名它即可；
        - 二者都在时按需二选一，另一个删除。之后重跑本身也会因目录状态异常而
          暴露问题，不会静默产出错误结果。
        """
        # 1. 先序列化 + 严格校验 summary（在碰任何磁盘之前）。
        #    allow_nan=False 作为兜底：任何漏网的非有限值都会在此处抛错，
        #    而不是写出 NaN/Infinity 这类非法 JSON。
        stats_serializable = self._make_serializable(self.stats)
        try:
            summary_text = json.dumps(stats_serializable, indent=2, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"统计信息无法序列化为 JSON: {e}")

        # 2. 建唯一暂存目录（与 output_dir 同级，保证同一文件系统 → rename
        #    才是原子的；名字唯一，不与他人碰撞）。
        parent = self.output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.output_dir.name}.staging-",
                                        dir=parent))
        # mkdtemp 建的是 0700；发布后它就是 output_dir，需按 umask 摆回常规
        # 权限（默认 0755），否则共享机器上同组同事读不到日志。
        _umask = os.umask(0)
        os.umask(_umask)
        os.chmod(staging, 0o777 & ~_umask)

        # 3. 把本次所有文件用最终名字写进暂存目录。
        try:
            csv_specs = [
                (self.round_logs, "rounds.csv", RoundLog.__annotations__.keys()),
                (self.block_logs, "blocks.csv", BlockLog.__annotations__.keys()),
                (self.probe_logs, "probes.csv", ProbeLog.__annotations__.keys()),
            ]
            for logs, name, fieldnames in csv_specs:
                if logs:
                    self._write_csv(staging / name, logs, fieldnames)

            with open(staging / "summary.json", "w") as f:
                f.write(summary_text)

            # 4. 整组发布。
            self._publish(staging)
        except Exception:
            # 发布未完成 → 清理暂存目录，正式目录保持不动。
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _publish(self, staging: Path):
        """把暂存目录换到 output_dir。

        - output_dir 为空：单步 `os.replace`，零窗口原子，崩溃安全。
        - output_dir 非空：两步 rename（旧→备份，暂存→正式）+ 普通异常回滚。
          **不是崩溃安全**——两步之间正式目录短暂缺失，KeyboardInterrupt /
          进程终止会绕过 `except`，留下 `.backup-*` / `.staging-*`（恢复办法见
          `save()` 的「中断恢复」）。此路径仅供 `force_overwrite` 非正式复用，
          正式实验用新目录走上面的单步原子路径。
        """
        target = self.output_dir
        # output_dir 在构造时已建好；判断它当前是否为空。
        is_empty = not any(target.iterdir())

        if is_empty:
            # 空目录：直接原子替换，零中间态。
            os.replace(staging, target)
            return

        # 非空：先把旧目录挪成唯一备份名，再把暂存挪进来。
        # 注意：这两步之间 target 短暂不存在——这是一个非崩溃安全窗口。普通
        # OSError 会被下面 except 回滚，但 KeyboardInterrupt/进程终止无法覆盖，
        # 届时只剩 .backup-*（完整旧版）与 .staging-*（完整新版）。故此路径仅
        # 供 force_overwrite 非正式复用，正式实验走空目录单步原子路径。
        backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.backup-",
                                       dir=target.parent))
        backup.rmdir()  # 只借这个唯一名字，rename 需要目标不存在
        os.replace(target, backup)           # 第一步：旧 → 备份
        try:
            os.replace(staging, target)       # 第二步：暂存 → 正式
        except Exception:
            os.replace(backup, target)        # 回滚：备份 → 正式（仅普通异常可达）
            raise
        # 发布已成功；删备份是尽力而为，失败绝不影响已发布的新数据。
        try:
            shutil.rmtree(backup)
        except Exception:
            pass

    def _write_csv(self, path: Path, logs: List, fieldnames):
        """把 CSV 写入暂存目录中的最终文件名。"""
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for log in logs:
                writer.writerow(asdict(log))

    def _make_serializable(self, obj):
        """将 numpy 类型转换为 Python 原生类型，并处理非有限值"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            val = obj.item()
            # 处理非有限值
            if isinstance(val, float):
                if math.isnan(val):
                    return None
                elif math.isinf(val):
                    return None
            return val
        elif isinstance(obj, np.ndarray):
            # 递归处理，确保数组内的 NaN/Infinity 同样转为 None
            return [self._make_serializable(v) for v in obj.tolist()]
        elif isinstance(obj, float):
            # 处理 Python 原生 float 的非有限值
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        else:
            return obj
