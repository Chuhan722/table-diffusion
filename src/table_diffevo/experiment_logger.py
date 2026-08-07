"""
实验日志记录器

为接受规则对照实验和 α 调度实验提供结构化日志。
支持三个层级的日志：每轮（round）、每块（block）、探测（probe）。
"""
import json
import csv
import math
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
            是否允许覆盖非空目录
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

    # 本记录器管理的全部输出文件名（用于清理陈旧类别）
    _MANAGED_FILES = ("rounds.csv", "blocks.csv", "probes.csv", "summary.json")

    def save(self):
        """保存所有日志到文件。

        采用「全部构造并校验 → 一次性发布」的策略：任何一步（CSV 构造、
        summary.json 序列化）失败都在发布任何正式文件之前抛出，绝不留下
        半成品。summary.json 的序列化在写入任何 CSV 临时文件之前就先行
        校验，避免「CSV 已发布、JSON 才失败」导致的不一致结果。
        """
        # 1. 先序列化 + 严格校验 summary（在写入/发布任何文件之前）
        #    allow_nan=False 作为兜底：任何漏网的非有限值都会在此处抛错，
        #    而不是写出 NaN/Infinity 这类非法 JSON。
        stats_serializable = self._make_serializable(self.stats)
        try:
            summary_text = json.dumps(stats_serializable, indent=2, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"统计信息无法序列化为 JSON: {e}")

        # 2. 全部写入临时文件（不触碰既有正式文件）
        pending: List[tuple] = []  # (temp_path, final_path)
        try:
            csv_specs = [
                (self.round_logs, "rounds.csv", RoundLog.__annotations__.keys()),
                (self.block_logs, "blocks.csv", BlockLog.__annotations__.keys()),
                (self.probe_logs, "probes.csv", ProbeLog.__annotations__.keys()),
            ]
            for logs, name, fieldnames in csv_specs:
                if logs:
                    final = self.output_dir / name
                    temp = self._write_csv_temp(final, logs, fieldnames)
                    pending.append((temp, final))

            summary_final = self.output_dir / "summary.json"
            summary_temp = summary_final.with_suffix(".json.tmp")
            with open(summary_temp, "w") as f:
                f.write(summary_text)
            pending.append((summary_temp, summary_final))

            # 3. 全部临时文件就绪 → 一次性发布（替换）
            for temp, final in pending:
                temp.replace(final)
        except Exception:
            # 清理所有已写入的临时文件，正式文件保持不动
            for temp, _ in pending:
                if temp.exists():
                    temp.unlink()
            raise

        # 4. 清理「本次未写入」的陈旧类别文件（force_overwrite 复用目录时，
        #    上一轮遗留的 probes.csv 等不应残留）
        written = {final.name for _, final in pending}
        for name in self._MANAGED_FILES:
            if name not in written:
                stale = self.output_dir / name
                if stale.exists():
                    stale.unlink()

    def _write_csv_temp(self, path: Path, logs: List, fieldnames) -> Path:
        """把 CSV 写入临时文件并返回其路径（不发布）。

        失败时清理自身的临时文件后向上抛出。
        """
        temp_path = path.with_suffix('.csv.tmp')
        try:
            with open(temp_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for log in logs:
                    writer.writerow(asdict(log))
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        return temp_path

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
