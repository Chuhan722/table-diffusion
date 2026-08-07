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

    def save(self):
        """保存所有日志到文件"""
        # 保存每轮日志为 CSV
        if self.round_logs:
            round_csv = self.output_dir / "rounds.csv"
            self._write_csv_atomic(round_csv, self.round_logs, RoundLog.__annotations__.keys())

        # 保存每块日志为 CSV
        if self.block_logs:
            block_csv = self.output_dir / "blocks.csv"
            self._write_csv_atomic(block_csv, self.block_logs, BlockLog.__annotations__.keys())

        # 保存探测日志为 CSV
        if self.probe_logs:
            probe_csv = self.output_dir / "probes.csv"
            self._write_csv_atomic(probe_csv, self.probe_logs, ProbeLog.__annotations__.keys())

        # 保存统计信息为 JSON
        stats_json = self.output_dir / "summary.json"
        stats_serializable = self._make_serializable(self.stats)
        # 预校验 JSON 可序列化性
        try:
            json.dumps(stats_serializable)
        except (TypeError, ValueError) as e:
            raise ValueError(f"统计信息无法序列化为 JSON: {e}")
        self._write_json_atomic(stats_json, stats_serializable)

    def _write_csv_atomic(self, path: Path, logs: List, fieldnames):
        """原子写入 CSV 文件"""
        temp_path = path.with_suffix('.csv.tmp')
        try:
            with open(temp_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for log in logs:
                    writer.writerow(asdict(log))
            temp_path.replace(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _write_json_atomic(self, path: Path, data: dict):
        """原子写入 JSON 文件"""
        temp_path = path.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

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
            return obj.tolist()
        elif isinstance(obj, float):
            # 处理 Python 原生 float 的非有限值
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        else:
            return obj
