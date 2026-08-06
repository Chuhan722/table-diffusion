"""
实验配置管理

标准化实验参数结构，支持 YAML 序列化/反序列化和参数验证。
"""
import yaml
from pathlib import Path
from typing import Optional, Literal, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class AcceptanceRuleConfig:
    """接受规则配置"""
    rule: Literal["A0", "A1"]
    eps_L1: Optional[float] = None  # A1 需要
    eps_Q: float = 0.0  # 默认值，所有规则都用


@dataclass
class AlphaScheduleConfig:
    """α 调度配置"""
    mode: Literal["fixed", "round_schedule", "probe"]
    alpha_value: Optional[float] = None  # fixed 模式需要
    alpha_min: float = 2.0
    alpha_max: float = 10.0
    # probe 模式参数
    W: Optional[int] = None  # 块大小（轮数）或 B_block（候选评估数）
    P: int = 3  # 停滞触发块数
    H: int = 2  # 探测分支块数
    s: float = 0.10  # 归一化步长
    C: int = 2  # 冷却块数


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str
    target_path: str
    measured_target_path: str
    init_marginals_path: str
    n_records: int


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    experiment_name: str
    data: DataConfig
    acceptance_rule: AcceptanceRuleConfig
    alpha_schedule: AlphaScheduleConfig
    seeds: List[int]
    n_rounds: int
    output_dir: str
    # 演化参数（从现有实验继承）
    beta: float = 1.0
    eta: float = 0.5
    h: float = 0.8
    mu: float = 0.01
    lambda_: float = 0.5
    delta: float = 0.05
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)

    def validate(self):
        """验证配置的合理性"""
        errors = []

        # 验证接受规则
        if self.acceptance_rule.rule in ["A1"]:
            if self.acceptance_rule.eps_L1 is None:
                errors.append(f"{self.acceptance_rule.rule} 需要指定 eps_L1")

        # 验证 α 调度
        if self.alpha_schedule.mode == "fixed":
            if self.alpha_schedule.alpha_value is None:
                errors.append("fixed 模式需要指定 alpha_value")
        elif self.alpha_schedule.mode == "probe":
            if self.alpha_schedule.W is None:
                errors.append("probe 模式需要指定 W（块大小）")

        # 验证种子数量
        if len(self.seeds) == 0:
            errors.append("至少需要 1 个 seed")

        # 验证轮数
        if self.n_rounds <= 0:
            errors.append("n_rounds 必须为正数")

        # 验证 alpha 范围
        if self.alpha_schedule.alpha_min >= self.alpha_schedule.alpha_max:
            errors.append("alpha_min 必须小于 alpha_max")

        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            data = yaml.safe_load(f)

        # 递归构建嵌套 dataclass
        config = cls(
            experiment_name=data["experiment_name"],
            data=DataConfig(**data["data"]),
            acceptance_rule=AcceptanceRuleConfig(**data["acceptance_rule"]),
            alpha_schedule=AlphaScheduleConfig(**data["alpha_schedule"]),
            seeds=data["seeds"],
            n_rounds=data["n_rounds"],
            output_dir=data["output_dir"],
            **{k: v for k, v in data.items() if k in [
                "beta", "eta", "h", "mu", "lambda_", "delta", "winsorize_limits"
            ]}
        )

        config.validate()
        return config

    def to_yaml(self, path: Path):
        """保存配置到 YAML 文件"""
        # 转换为字典
        data = {
            "experiment_name": self.experiment_name,
            "data": asdict(self.data),
            "acceptance_rule": asdict(self.acceptance_rule),
            "alpha_schedule": asdict(self.alpha_schedule),
            "seeds": self.seeds,
            "n_rounds": self.n_rounds,
            "output_dir": self.output_dir,
            "beta": self.beta,
            "eta": self.eta,
            "h": self.h,
            "mu": self.mu,
            "lambda_": self.lambda_,
            "delta": self.delta,
            "winsorize_limits": list(self.winsorize_limits)
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
