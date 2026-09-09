"""严格的 fitness-only 扩散演化入口。

这个模块把研究主张变成 fail-closed 的可执行合同。**默认合同**（
``value_guidance_strength=0`` 且 ``block_score_tilt_strength=0``）：动态
residual 只允许进入行适应度；donor 选定后使用盲独立复制核（``eta/mu``
固定；``rho`` 恒定，或走预冻结的纯轮数驱动三段式时间表：保温 H 轮 →
几何降温 D 轮 → 地板，公式不含总轮数、不读取残差）；每个 proposal 无
条件成为下一状态；唯一停止条件是固定轮数；主返回恒为 terminal
current。``equal`` 对照只把行适应度替换为常数，结构距离、时间表和全部
随机机制保持不变，因此与 ``residual`` 臂之间只差适应度信号。

**引导扩展核不是 fitness-only**：``value_guidance_strength>0``（残差
逐格增益进值分布，``transition_kernel='value_guided'``）或
``block_score_tilt_strength>0``（残差块分数倾斜复制开关，
``transition_kernel='block_score_tilted'``）属于"生成前分布塑形"路线，
残差经适应度之外的通道驱动转移核。此时产物合同据实声明
``residual_driving_channels`` 追加对应通道、``transition_kernel`` 改为
对应核名，审计器按请求配置重算期望并逐字段对拍——λ>0 的运行若携带
``blind_independent`` 声明即审计 failure（fail-closed，防伪造）。引导核
下 ``equal`` 对照臂无法定义（equal 臂禁用一切残差信号，而引导增益正是
残差信号），单臂 equal 与配对归因均在入口 fail-closed 拒绝。
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from table_diffevo.evolution import FITNESS_ONLY_MODES, run_evolution
from table_diffevo.schema import Schema


FitnessMode = Literal["residual", "equal"]


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class FitnessOnlyConfig:
    """两条归因臂共享的、与结果无关的运行配置。"""

    n_rounds: int
    seed: int
    device: Literal["numpy", "cpu", "cuda"] = "numpy"
    eval_method: Literal["vectorized", "legacy"] = "vectorized"
    batch_size: int = 256
    init_method: Literal["random", "marginal"] = "marginal"
    log_every: int = 100
    rho: float = 0.01
    eta: float = 0.5
    mu: float = 0.01
    rho_anneal_start_round: Optional[int] = None
    rho_anneal_rounds: Optional[int] = None
    rho_anneal_end: Optional[float] = None
    eta_anneal_start_round: Optional[int] = None
    eta_anneal_rounds: Optional[int] = None
    eta_anneal_end: Optional[float] = None
    mu_anneal_start_round: Optional[int] = None
    mu_anneal_rounds: Optional[int] = None
    mu_anneal_end: Optional[float] = None
    mw_query_weight_eta: Optional[float] = None
    mw_signal_cap: Optional[float] = None
    mw_weight_cap: Optional[float] = None
    mw_start_round: Optional[int] = None
    lambda_param: float = 0.5
    fixed_alpha: float = 16.0
    delta: float = 0.05
    winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)
    selection_scale_invariant_min_spread: float = 1e-3
    residual_geometry: Literal[
        "absolute", "sqrt_relative", "relative"
    ] = "relative"
    residual_geometry_floor: float = 8.0
    exclude_self: bool = True
    lottery_first_donor_selection: bool = False
    block_score_tilt_strength: float = 0.0
    block_score_tilt_bounds: Tuple[float, float] = (0.3, 0.7)
    value_guidance_strength: float = 0.0
    value_guidance_warmup_start_round: Optional[int] = None
    value_guidance_warmup_rounds: Optional[int] = None
    value_guidance_drop_donor: bool = False
    value_guidance_adaptive_scale: bool = False
    record_transition_clocks: bool = False
    inner_early_stopping_patience_ticks: Optional[int] = None

    def validate(self) -> None:
        """拒绝会改变合同语义或造成隐式退化的配置。"""

        errors = []
        if (
            isinstance(self.n_rounds, (bool, np.bool_))
            or not isinstance(self.n_rounds, (int, np.integer))
            or self.n_rounds <= 0
        ):
            errors.append("n_rounds 必须是正整数")
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
        ):
            errors.append("seed 必须是整数")
        if self.device not in {"numpy", "cpu", "cuda"}:
            errors.append("device 必须是 numpy、cpu 或 cuda")
        if self.eval_method not in {"vectorized", "legacy"}:
            errors.append("eval_method 必须是 vectorized 或 legacy")
        if (
            isinstance(self.batch_size, (bool, np.bool_))
            or not isinstance(self.batch_size, (int, np.integer))
            or self.batch_size <= 0
        ):
            errors.append("batch_size 必须是正整数")
        if self.init_method not in {"random", "marginal"}:
            errors.append("init_method 只允许 random 或 marginal")
        if (
            isinstance(self.log_every, (bool, np.bool_))
            or not isinstance(self.log_every, (int, np.integer))
            or self.log_every < 0
        ):
            errors.append("log_every 必须是非负整数")

        for name, value, lower_open in (
            ("rho", self.rho, True),
            ("eta", self.eta, False),
            ("mu", self.mu, False),
        ):
            valid = (
                _is_finite_number(value)
                and (value > 0.0 if lower_open else value >= 0.0)
                and value <= 1.0
            )
            if not valid:
                interval = "(0, 1]" if lower_open else "[0, 1]"
                errors.append(f"{name} 必须位于 {interval}")

        schedule_values = (
            self.rho_anneal_start_round,
            self.rho_anneal_rounds,
            self.rho_anneal_end,
        )
        schedule_flags = [value is not None for value in schedule_values]
        if any(schedule_flags) and not all(schedule_flags):
            errors.append(
                "三段式时间表必须同时提供 rho_anneal_start_round、"
                "rho_anneal_rounds、rho_anneal_end，或三者全为 None"
            )
        elif all(schedule_flags):
            if (
                isinstance(self.rho_anneal_start_round, (bool, np.bool_))
                or not isinstance(
                    self.rho_anneal_start_round, (int, np.integer)
                )
                or self.rho_anneal_start_round < 0
            ):
                errors.append("rho_anneal_start_round 必须是非负整数")
            if (
                isinstance(self.rho_anneal_rounds, (bool, np.bool_))
                or not isinstance(
                    self.rho_anneal_rounds, (int, np.integer)
                )
                or self.rho_anneal_rounds < 1
            ):
                errors.append("rho_anneal_rounds 必须是正整数")
            if (
                not _is_finite_number(self.rho_anneal_end)
                or not (
                    _is_finite_number(self.rho)
                    and 0.0 < self.rho_anneal_end <= self.rho
                )
            ):
                errors.append("rho_anneal_end 必须位于 (0, rho]")

        eta_schedule_values = (
            self.eta_anneal_start_round,
            self.eta_anneal_rounds,
            self.eta_anneal_end,
        )
        eta_schedule_flags = [
            value is not None for value in eta_schedule_values
        ]
        if any(eta_schedule_flags) and not all(eta_schedule_flags):
            errors.append(
                "eta 三段式时间表必须同时提供 eta_anneal_start_round、"
                "eta_anneal_rounds、eta_anneal_end，或三者全为 None"
            )
        elif all(eta_schedule_flags):
            if (
                isinstance(self.eta_anneal_start_round, (bool, np.bool_))
                or not isinstance(
                    self.eta_anneal_start_round, (int, np.integer)
                )
                or self.eta_anneal_start_round < 0
            ):
                errors.append("eta_anneal_start_round 必须是非负整数")
            if (
                isinstance(self.eta_anneal_rounds, (bool, np.bool_))
                or not isinstance(
                    self.eta_anneal_rounds, (int, np.integer)
                )
                or self.eta_anneal_rounds < 1
            ):
                errors.append("eta_anneal_rounds 必须是正整数")
            if (
                not _is_finite_number(self.eta_anneal_end)
                or not (
                    _is_finite_number(self.eta)
                    and 0.0 < self.eta_anneal_end <= self.eta
                )
            ):
                errors.append("eta_anneal_end 必须位于 (0, eta]")

        mu_schedule_values = (
            self.mu_anneal_start_round,
            self.mu_anneal_rounds,
            self.mu_anneal_end,
        )
        mu_schedule_flags = [
            value is not None for value in mu_schedule_values
        ]
        if any(mu_schedule_flags) and not all(mu_schedule_flags):
            errors.append(
                "mu 三段式时间表必须同时提供 mu_anneal_start_round、"
                "mu_anneal_rounds、mu_anneal_end，或三者全为 None"
            )
        elif all(mu_schedule_flags):
            if (
                isinstance(self.mu_anneal_start_round, (bool, np.bool_))
                or not isinstance(
                    self.mu_anneal_start_round, (int, np.integer)
                )
                or self.mu_anneal_start_round < 0
            ):
                errors.append("mu_anneal_start_round 必须是非负整数")
            if (
                isinstance(self.mu_anneal_rounds, (bool, np.bool_))
                or not isinstance(
                    self.mu_anneal_rounds, (int, np.integer)
                )
                or self.mu_anneal_rounds < 1
            ):
                errors.append("mu_anneal_rounds 必须是正整数")
            if (
                not _is_finite_number(self.mu_anneal_end)
                or not (
                    _is_finite_number(self.mu)
                    and 0.0 < self.mu_anneal_end <= self.mu
                )
            ):
                errors.append("mu_anneal_end 必须位于 (0, mu]")

        mw_values = (
            self.mw_query_weight_eta,
            self.mw_signal_cap,
            self.mw_weight_cap,
            self.mw_start_round,
        )
        mw_flags = [value is not None for value in mw_values]
        if any(mw_flags) and not all(mw_flags):
            errors.append(
                "MW 查询权重必须同时提供 mw_query_weight_eta、"
                "mw_signal_cap、mw_weight_cap、mw_start_round，"
                "或四者全为 None"
            )
        elif all(mw_flags):
            if (
                not _is_finite_number(self.mw_query_weight_eta)
                or self.mw_query_weight_eta <= 0.0
            ):
                errors.append("mw_query_weight_eta 必须是正有限数")
            if (
                not _is_finite_number(self.mw_signal_cap)
                or self.mw_signal_cap <= 0.0
            ):
                errors.append("mw_signal_cap 必须是正有限数")
            if (
                not _is_finite_number(self.mw_weight_cap)
                or self.mw_weight_cap <= 1.0
            ):
                errors.append("mw_weight_cap 必须是大于 1 的有限数")
            if (
                isinstance(self.mw_start_round, (bool, np.bool_))
                or not isinstance(self.mw_start_round, (int, np.integer))
                or self.mw_start_round < 0
            ):
                errors.append("mw_start_round 必须是非负整数")

        for name, value in (
            ("fixed_alpha", self.fixed_alpha),
            (
                "selection_scale_invariant_min_spread",
                self.selection_scale_invariant_min_spread,
            ),
        ):
            if (
                not _is_finite_number(value)
                or value <= 0.0
            ):
                errors.append(f"{name} 必须是正有限数")
        if (
            not _is_finite_number(self.lambda_param)
            or not 0.0 <= self.lambda_param <= 1.0
        ):
            errors.append("lambda_param 必须位于 [0, 1]")
        if (
            not _is_finite_number(self.delta)
            or not 0.0 < self.delta < 1.0
        ):
            errors.append("delta 必须位于 (0, 1)")
        if (
            not isinstance(self.winsorize_quantiles, tuple)
            or len(self.winsorize_quantiles) != 2
            or not all(
                _is_finite_number(x) for x in self.winsorize_quantiles
            )
            or not 0.0 <= self.winsorize_quantiles[0]
            < self.winsorize_quantiles[1] <= 1.0
        ):
            errors.append(
                "winsorize_quantiles 必须是满足 0<=low<high<=1 的二元组"
            )
        if self.residual_geometry not in {
            "absolute",
            "sqrt_relative",
            "relative",
        }:
            errors.append("residual_geometry 非法")
        if self.residual_geometry in {"sqrt_relative", "relative"} and (
            not _is_finite_number(self.residual_geometry_floor)
            or self.residual_geometry_floor <= 0.0
        ):
            errors.append("residual_geometry_floor 必须是正有限数")
        if not isinstance(self.exclude_self, (bool, np.bool_)):
            errors.append("exclude_self 必须是布尔值")
        if not isinstance(
            self.lottery_first_donor_selection, (bool, np.bool_)
        ):
            errors.append("lottery_first_donor_selection 必须是布尔值")
        if (
            not _is_finite_number(self.block_score_tilt_strength)
            or self.block_score_tilt_strength < 0.0
        ):
            errors.append("block_score_tilt_strength 必须是非负有限数值")
        elif self.block_score_tilt_strength > 0.0:
            # fail-closed：分科倾斜的组合边界与主循环校验一致（见
            # evolution.run_evolution），在配置层提前拒绝。
            if self.eval_method != "vectorized":
                errors.append(
                    "block_score_tilt_strength>0 要求 eval_method='vectorized'"
                )
            if self.mw_query_weight_eta is not None:
                errors.append(
                    "block_score_tilt_strength>0 与 MW 查询权重互斥"
                )
            if self.eta_anneal_end is not None:
                errors.append("block_score_tilt_strength>0 与 eta 退火互斥")
            if (
                not isinstance(self.block_score_tilt_bounds, (tuple, list))
                or len(self.block_score_tilt_bounds) != 2
                or not all(
                    _is_finite_number(v)
                    for v in self.block_score_tilt_bounds
                )
            ):
                errors.append(
                    "block_score_tilt_bounds 必须是有限数值 (lo, hi) 二元组"
                )
            elif not (
                0.0
                <= float(self.block_score_tilt_bounds[0])
                <= float(self.eta)
                <= float(self.block_score_tilt_bounds[1])
                <= 1.0
            ):
                errors.append(
                    "block_score_tilt_bounds 必须满足 0 ≤ lo ≤ eta ≤ hi ≤ 1"
                )
        if (
            not _is_finite_number(self.value_guidance_strength)
            or self.value_guidance_strength < 0.0
        ):
            errors.append("value_guidance_strength 必须是非负有限数值")
        elif self.value_guidance_strength > 0.0:
            # fail-closed：值引导核的组合边界与主循环校验一致（见
            # evolution.run_evolution），在配置层提前拒绝。
            if self.eval_method != "vectorized":
                errors.append(
                    "value_guidance_strength>0 要求 eval_method='vectorized'"
                )
            if self.mw_query_weight_eta is not None:
                errors.append(
                    "value_guidance_strength>0 与 MW 查询权重互斥"
                )
            if self.eta_anneal_end is not None:
                errors.append("value_guidance_strength>0 与 eta 退火互斥")
            if self.block_score_tilt_strength > 0.0:
                errors.append("value_guidance_strength>0 与分科倾斜互斥")
        else:
            if self.value_guidance_drop_donor:
                errors.append(
                    "value_guidance_drop_donor=True 需要 "
                    "value_guidance_strength>0"
                )
            if self.value_guidance_adaptive_scale:
                errors.append(
                    "value_guidance_adaptive_scale=True 需要 "
                    "value_guidance_strength>0"
                )
            if (
                self.value_guidance_warmup_start_round is not None
                or self.value_guidance_warmup_rounds is not None
            ):
                errors.append(
                    "value_guidance warmup 参数需要 value_guidance_strength>0"
                )
        if not isinstance(self.value_guidance_drop_donor, (bool, np.bool_)):
            errors.append("value_guidance_drop_donor 必须是布尔值")
        if not isinstance(
            self.value_guidance_adaptive_scale, (bool, np.bool_)
        ):
            errors.append("value_guidance_adaptive_scale 必须是布尔值")
        if self.value_guidance_warmup_start_round is not None and (
            isinstance(
                self.value_guidance_warmup_start_round, (bool, np.bool_)
            )
            or not isinstance(
                self.value_guidance_warmup_start_round, (int, np.integer)
            )
            or self.value_guidance_warmup_start_round < 0
        ):
            errors.append(
                "value_guidance_warmup_start_round 必须是非负整数或 None"
            )
        if self.value_guidance_warmup_rounds is not None and (
            isinstance(self.value_guidance_warmup_rounds, (bool, np.bool_))
            or not isinstance(
                self.value_guidance_warmup_rounds, (int, np.integer)
            )
            or self.value_guidance_warmup_rounds < 1
        ):
            errors.append("value_guidance_warmup_rounds 必须是正整数或 None")
        if (self.value_guidance_warmup_start_round is None) != (
            self.value_guidance_warmup_rounds is None
        ):
            errors.append(
                "value_guidance_warmup_start_round 与 "
                "value_guidance_warmup_rounds 必须成对提供或都为 None"
            )
        if not isinstance(self.record_transition_clocks, (bool, np.bool_)):
            errors.append("record_transition_clocks 必须是布尔值")
        if self.inner_early_stopping_patience_ticks is not None and (
            isinstance(
                self.inner_early_stopping_patience_ticks, (bool, np.bool_)
            )
            or not isinstance(
                self.inner_early_stopping_patience_ticks, (int, np.integer)
            )
            or self.inner_early_stopping_patience_ticks <= 0
        ):
            errors.append(
                "inner_early_stopping_patience_ticks 必须是正整数或 None"
            )
        if errors:
            raise ValueError("fitness-only 配置验证失败：" + "；".join(errors))


def build_fitness_only_kwargs(
    config: FitnessOnlyConfig,
    *,
    fitness_mode: FitnessMode,
    marginals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造主循环参数；所有被禁止的旁路在这里显式关闭。"""

    config.validate()
    if fitness_mode not in FITNESS_ONLY_MODES:
        raise ValueError(
            f"fitness_mode 必须是 {FITNESS_ONLY_MODES} 之一，"
            f"得到 {fitness_mode!r}"
        )
    if config.init_method == "marginal" and marginals is None:
        raise ValueError("marginal 初始化要求显式提供冻结的 marginals")
    if config.init_method == "random" and marginals is not None:
        raise ValueError("random 初始化不允许提供 marginals")

    return {
        "n_rounds": int(config.n_rounds),
        "seed": int(config.seed),
        "rho": float(config.rho),
        "eta": float(config.eta),
        "mu": float(config.mu),
        "tol": float("inf"),
        "device": config.device,
        "eval_method": config.eval_method,
        "batch_size": int(config.batch_size),
        "init_method": config.init_method,
        "marginals": marginals,
        "log_every": int(config.log_every),
        "distance_mode": "geometric",
        "lambda_param": float(config.lambda_param),
        "delta": float(config.delta),
        "winsorize_quantiles": tuple(config.winsorize_quantiles),
        "exclude_self": bool(config.exclude_self),
        "lottery_first_donor_selection": bool(
            config.lottery_first_donor_selection
        ),
        "block_score_tilt_strength": float(config.block_score_tilt_strength),
        "block_score_tilt_bounds": (
            tuple(float(v) for v in config.block_score_tilt_bounds)
            if config.block_score_tilt_strength > 0.0 else None
        ),
        "value_guidance_strength": float(config.value_guidance_strength),
        "value_guidance_warmup_start_round": (
            int(config.value_guidance_warmup_start_round)
            if config.value_guidance_warmup_start_round is not None
            else None
        ),
        "value_guidance_warmup_rounds": (
            int(config.value_guidance_warmup_rounds)
            if config.value_guidance_warmup_rounds is not None else None
        ),
        "value_guidance_drop_donor": bool(config.value_guidance_drop_donor),
        "value_guidance_adaptive_scale": bool(
            config.value_guidance_adaptive_scale
        ),
        "max_retries": 0,
        "residual_directed_diffusion": False,
        "diffusion_direction_strength": 0.0,
        "diffusion_direction_normalization": "none",
        "diffusion_direction_logit_clip": None,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_use_compiled_workload": False,
        "gap_l1_sweeps": 0,
        "candidate_budget": None,
        "residual_self_cooling": None,
        "self_cooling_stop_ratio": None,
        "rho_anneal_end": (
            float(config.rho_anneal_end)
            if config.rho_anneal_end is not None else None
        ),
        "rho_anneal_rounds": (
            int(config.rho_anneal_rounds)
            if config.rho_anneal_rounds is not None else None
        ),
        "rho_anneal_start_round": (
            int(config.rho_anneal_start_round)
            if config.rho_anneal_start_round is not None else None
        ),
        "eta_anneal_end": (
            float(config.eta_anneal_end)
            if config.eta_anneal_end is not None else None
        ),
        "eta_anneal_rounds": (
            int(config.eta_anneal_rounds)
            if config.eta_anneal_rounds is not None else None
        ),
        "eta_anneal_start_round": (
            int(config.eta_anneal_start_round)
            if config.eta_anneal_start_round is not None else None
        ),
        "mu_anneal_end": (
            float(config.mu_anneal_end)
            if config.mu_anneal_end is not None else None
        ),
        "mu_anneal_rounds": (
            int(config.mu_anneal_rounds)
            if config.mu_anneal_rounds is not None else None
        ),
        "mu_anneal_start_round": (
            int(config.mu_anneal_start_round)
            if config.mu_anneal_start_round is not None else None
        ),
        "mw_query_weight_eta": (
            float(config.mw_query_weight_eta)
            if config.mw_query_weight_eta is not None else None
        ),
        "mw_signal_cap": (
            float(config.mw_signal_cap)
            if config.mw_signal_cap is not None else None
        ),
        "mw_weight_cap": (
            float(config.mw_weight_cap)
            if config.mw_weight_cap is not None else None
        ),
        "mw_start_round": (
            int(config.mw_start_round)
            if config.mw_start_round is not None else None
        ),
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": float(
            config.selection_scale_invariant_min_spread
        ),
        "residual_geometry": config.residual_geometry,
        "residual_geometry_floor": float(config.residual_geometry_floor),
        "return_final_table": False,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": float(config.fixed_alpha),
        "record_transition_clocks": bool(config.record_transition_clocks),
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": False,
        # 引擎合同：启用 inner A/B/C 早停时 A（loss=0 停止）必须同时开启。
        "stop_on_exact_residual": (
            config.inner_early_stopping_patience_ticks is not None
        ),
        "horizon_invariant": True,
        "inner_early_stopping_patience_ticks": (
            int(config.inner_early_stopping_patience_ticks)
            if config.inner_early_stopping_patience_ticks is not None
            else None
        ),
        "fitness_only_mode": fitness_mode,
    }


def _audit_fitness_only_run(
    diagnostics: Dict[str, Any],
    config: FitnessOnlyConfig,
    fitness_mode: FitnessMode,
) -> None:
    """运行后再次核验合同，防止主循环未来改动后静默漂移。"""

    contract = diagnostics.get("fitness_only_contract")
    value_guidance_on = config.value_guidance_strength > 0.0
    block_tilt_on = config.block_score_tilt_strength > 0.0
    # 与合同生成端同一逻辑重算期望（fail-closed）：λ>0 时残差经值引导
    # 进入转移核，通道/核名必须据实；产物若仍声明 blind_independent
    # 即为伪造，记 failure。
    expected_channels = ["fitness"] if fitness_mode == "residual" else []
    if value_guidance_on:
        expected_channels = expected_channels + ["value_guidance"]
    if block_tilt_on:
        expected_channels = expected_channels + ["block_score_tilt"]
    expected_kernel = (
        "value_guided"
        if value_guidance_on
        else "block_score_tilted"
        if block_tilt_on
        else "blind_independent"
    )
    early_stopping_enabled = (
        config.inner_early_stopping_patience_ticks is not None
    )
    rounds_run = diagnostics.get("rounds_run")
    failures = []
    if diagnostics.get("output_table_identity") != "terminal_current":
        failures.append("主输出不是 terminal current")
    if early_stopping_enabled:
        reason = diagnostics.get("termination_reason")
        # 引擎 A/B/C 早停状态机的全部终止标签：A=fit_target_reached、
        # B=early_stopped、C=resource_cap_reached（跑满外部预算上限）。
        if reason not in {
            "fit_target_reached",
            "early_stopped",
            "resource_cap_reached",
        }:
            failures.append("终止原因不在早停合同允许集合内")
        if (
            isinstance(rounds_run, bool)
            or not isinstance(rounds_run, int)
            or rounds_run < 1
            or rounds_run > config.n_rounds
        ):
            failures.append("实际轮数超出固定预算上限范围")
        elif (
            reason == "resource_cap_reached"
            and rounds_run != config.n_rounds
        ):
            failures.append("触顶终止但实际轮数不等于预算上限")
        if diagnostics.get("candidate_evaluation_count") != rounds_run:
            failures.append("候选评价数不等于实际轮数")
        if bool(diagnostics.get("stopped_early")) != (
            reason in {"fit_target_reached", "early_stopped"}
        ):
            failures.append("stopped_early 与终止原因不一致")
        inner = diagnostics.get("inner_early_stopping")
        if (
            not isinstance(inner, dict)
            or not inner.get("enabled")
            or inner.get("patience_ticks")
            != config.inner_early_stopping_patience_ticks
        ):
            failures.append("inner early stopping 诊断缺失或与配置不一致")
    else:
        if diagnostics.get("termination_reason") != "max_rounds":
            failures.append("终止原因不是固定轮数")
        if rounds_run != config.n_rounds:
            failures.append("实际轮数不等于固定预算")
        if diagnostics.get("candidate_evaluation_count") != config.n_rounds:
            failures.append("候选评价数不等于固定轮数")
        if diagnostics.get("stopped_early"):
            failures.append("发生了结果相关提前停止")
    if not all(diagnostics.get("accept_history", [])):
        failures.append("存在未接续 proposal")
    if diagnostics.get("direction_evaluation_count") != 0:
        failures.append("执行了残差定向复制评价")
    if diagnostics.get("factorized_gibbs_microsteps") != 0:
        failures.append("执行了 factorized Gibbs")
    if diagnostics.get("gap_l1_microsteps") != 0:
        failures.append("执行了 C/gap 微步")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        failures.append("缺少 fitness-only 合同诊断")
    elif (
        contract.get("fitness_mode") != fitness_mode
        or contract.get("residual_driving_channels") != expected_channels
        or contract.get("transition_kernel") != expected_kernel
        or contract.get("proposal_transition") != "unconditional"
        or contract.get("termination_rule") != (
            "inner_early_stopping_a_b_c"
            if early_stopping_enabled
            else "fixed_n_rounds"
        )
        or contract.get("output_identity") != "terminal_current"
    ):
        failures.append("fitness-only 合同诊断与请求不一致")
    if fitness_mode == "equal" and any(
        value is not None and value != 0.0
        for value in diagnostics.get("donor_fitness_history", [])
    ):
        # lottery 换位口径下零中签轮记 None，不属于非零 fitness 违约。
        failures.append("equal 对照出现了非零 donor fitness")
    run_params = diagnostics.get("params", {})
    if bool(run_params.get("lottery_first_donor_selection")) != bool(
        config.lottery_first_donor_selection
    ):
        failures.append("lottery_first_donor_selection 与请求配置不一致")
    if float(
        run_params.get("block_score_tilt_strength", 0.0)
    ) != float(config.block_score_tilt_strength):
        failures.append("block_score_tilt_strength 与请求配置不一致")
    if config.block_score_tilt_strength > 0.0:
        tilt_diag = diagnostics.get("block_score_tilt")
        if not isinstance(tilt_diag, dict) or not tilt_diag.get("enabled"):
            failures.append("分科倾斜已请求但诊断显示未启用")
        elif tilt_diag.get("reference_scale") is None:
            failures.append("分科倾斜启用但参考尺度未标定")
        expected_bounds = tuple(
            float(v) for v in config.block_score_tilt_bounds
        )
        actual_bounds = run_params.get("block_score_tilt_bounds")
        if actual_bounds is None or tuple(
            float(v) for v in actual_bounds
        ) != expected_bounds:
            failures.append("block_score_tilt_bounds 与请求配置不一致")
    if float(
        run_params.get("value_guidance_strength", 0.0)
    ) != float(config.value_guidance_strength):
        failures.append("value_guidance_strength 与请求配置不一致")
    if value_guidance_on:
        # V9 型运行的科学口径由归一化/去供体/warmup 共同定义，任一
        # 谎报都会让不同实验设定的产物不可分辨——全参数对拍。
        for key, expected in (
            (
                "value_guidance_adaptive_scale",
                bool(config.value_guidance_adaptive_scale),
            ),
            (
                "value_guidance_drop_donor",
                bool(config.value_guidance_drop_donor),
            ),
            (
                "value_guidance_warmup_start_round",
                config.value_guidance_warmup_start_round,
            ),
            (
                "value_guidance_warmup_rounds",
                config.value_guidance_warmup_rounds,
            ),
        ):
            if run_params.get(key) != expected:
                failures.append(f"{key} 与请求配置不一致")
        vg_diag = diagnostics.get("value_guidance")
        if not isinstance(vg_diag, dict) or not vg_diag.get("enabled"):
            failures.append("值引导已请求但诊断显示未启用")
        else:
            if float(vg_diag.get("strength", 0.0)) != float(
                config.value_guidance_strength
            ):
                failures.append("值引导诊断 strength 与请求配置不一致")
            if bool(vg_diag.get("adaptive_scale")) != bool(
                config.value_guidance_adaptive_scale
            ):
                failures.append(
                    "值引导诊断 adaptive_scale 与请求配置不一致"
                )
            if bool(vg_diag.get("drop_donor")) != bool(
                config.value_guidance_drop_donor
            ):
                failures.append("值引导诊断 drop_donor 与请求配置不一致")
            if (
                vg_diag.get("warmup_start_round")
                != config.value_guidance_warmup_start_round
                or vg_diag.get("warmup_rounds")
                != config.value_guidance_warmup_rounds
            ):
                failures.append("值引导诊断 warmup 参数与请求配置不一致")
    schedule_history = diagnostics.get("rho_schedule_history")
    expected_schedule_length = (
        rounds_run if early_stopping_enabled else config.n_rounds
    )
    if (
        not isinstance(schedule_history, list)
        or len(schedule_history) != expected_schedule_length
    ):
        failures.append("rho_schedule_history 缺失或长度不等于实际轮数")
    else:
        rho0 = float(config.rho)
        for t, actual in enumerate(schedule_history):
            if config.rho_anneal_end is not None:
                anneal_t = t - int(config.rho_anneal_start_round)
                anneal_progress = min(
                    1.0,
                    max(0.0, anneal_t / int(config.rho_anneal_rounds)),
                )
                expected = rho0 * (
                    float(config.rho_anneal_end) / rho0
                ) ** anneal_progress
            else:
                expected = rho0
            if actual != expected:
                failures.append(
                    f"rho_schedule_history 第 {t} 轮与预冻结时间表公式不一致"
                )
                break
    mu_history = diagnostics.get("mu_schedule_history")
    if (
        not isinstance(mu_history, list)
        or len(mu_history) != expected_schedule_length
    ):
        failures.append("mu_schedule_history 缺失或长度不等于实际轮数")
    else:
        mu0 = float(config.mu)
        for t, actual in enumerate(mu_history):
            if config.mu_anneal_end is not None:
                mu_anneal_t = t - int(config.mu_anneal_start_round)
                mu_anneal_progress = min(
                    1.0,
                    max(0.0, mu_anneal_t / int(config.mu_anneal_rounds)),
                )
                expected = mu0 * (
                    float(config.mu_anneal_end) / mu0
                ) ** mu_anneal_progress
            else:
                expected = mu0
            if actual != expected:
                failures.append(
                    f"mu_schedule_history 第 {t} 轮与预冻结时间表公式不一致"
                )
                break
    if float(config.value_guidance_strength) > 0.0:
        vg_block = diagnostics.get("value_guidance")
        vg_history = (
            vg_block.get("lambda_history")
            if isinstance(vg_block, dict) else None
        )
        if (
            not isinstance(vg_history, list)
            or len(vg_history) != expected_schedule_length
        ):
            failures.append(
                "value_guidance.lambda_history 缺失或长度不等于实际轮数"
            )
        else:
            lam_max = float(config.value_guidance_strength)
            for t, actual in enumerate(vg_history):
                if config.value_guidance_warmup_rounds is not None:
                    vg_progress = min(
                        1.0,
                        max(
                            0.0,
                            (
                                t
                                - int(
                                    config.value_guidance_warmup_start_round
                                )
                            )
                            / int(config.value_guidance_warmup_rounds),
                        ),
                    )
                    expected = lam_max * vg_progress
                else:
                    expected = lam_max
                if actual != expected:
                    failures.append(
                        f"value_guidance.lambda_history 第 {t} 轮与预冻结"
                        "时间表公式不一致"
                    )
                    break
        if bool(config.value_guidance_adaptive_scale):
            vg_scales = (
                vg_block.get("scale_history")
                if isinstance(vg_block, dict) else None
            )
            if (
                not isinstance(vg_scales, list)
                or len(vg_scales) != expected_schedule_length
            ):
                failures.append(
                    "value_guidance.scale_history 缺失或长度不等于实际轮数"
                    "（adaptive_scale=True 时逐轮必录）"
                )
            elif not all(
                isinstance(s, (int, float))
                and np.isfinite(s)
                and s > 0.0
                for s in vg_scales
            ):
                failures.append(
                    "value_guidance.scale_history 含非有限或非正尺度"
                )
    if failures:
        raise RuntimeError("fitness-only 运行后审计失败：" + "；".join(failures))


def run_fitness_only_evolution(
    target: np.ndarray,
    queries: list[dict[str, Any]],
    schema: Schema,
    n_records: int,
    *,
    config: FitnessOnlyConfig,
    fitness_mode: FitnessMode = "residual",
    marginals: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """运行一条严格 fitness-only 轨迹并执行合同自审计。"""

    kwargs = build_fitness_only_kwargs(
        config,
        fitness_mode=fitness_mode,
        marginals=marginals,
    )
    table, diagnostics = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=n_records,
        **kwargs,
    )
    _audit_fitness_only_run(diagnostics, config, fitness_mode)
    return table, diagnostics


def run_paired_fitness_only_attribution(
    target: np.ndarray,
    queries: list[dict[str, Any]],
    schema: Schema,
    n_records: int,
    *,
    config: FitnessOnlyConfig,
    marginals: Optional[Dict[str, Any]] = None,
) -> tuple[
    Dict[str, tuple[pd.DataFrame, Dict[str, Any]]],
    Dict[str, Any],
]:
    """运行 residual/equal 两臂并验证初始化与随机流地址严格配对。"""

    if config.inner_early_stopping_patience_ticks is not None:
        raise ValueError(
            "配对归因合同要求两臂固定同长轨迹；"
            "inner early stopping 只允许单臂运行"
        )
    if (
        config.value_guidance_strength > 0.0
        or config.block_score_tilt_strength > 0.0
    ):
        # fail-closed：引导核让残差进入转移核，equal 臂（禁用一切残差
        # 信号）无法定义，两臂归因失去"只差适应度信号"的合同前提。
        raise ValueError(
            "配对归因合同仅对盲独立复制核定义；"
            "value_guidance_strength/block_score_tilt_strength 必须为 0"
        )
    runs = {
        mode: run_fitness_only_evolution(
            target,
            queries,
            schema,
            n_records,
            config=config,
            fitness_mode=mode,
            marginals=marginals,
        )
        for mode in FITNESS_ONLY_MODES
    }
    residual_diag = runs["residual"][1]
    equal_diag = runs["equal"][1]
    paired_fields = (
        "initial_table_sha256",
        "primary_rng_post_initialization_state_sha256",
        "primary_rng_state_sha256",
        "candidate_evaluation_count",
        "rounds_run",
        "rho_schedule_history",
    )
    mismatches = [
        field
        for field in paired_fields
        if residual_diag.get(field) != equal_diag.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "fitness-only 配对随机地址不一致：" + "、".join(mismatches)
        )
    pairing = {
        "paired": True,
        "seed": int(config.seed),
        "arms": list(FITNESS_ONLY_MODES),
        "only_treatment_difference": "row_fitness",
        "shared_structure_distance": True,
        "shared_blind_independent_kernel": True,
        "shared_rho_schedule": {
            "rho_anneal_start_round": config.rho_anneal_start_round,
            "rho_anneal_rounds": config.rho_anneal_rounds,
            "rho_anneal_end": config.rho_anneal_end,
        },
        "shared_mu_schedule": {
            "mu_anneal_start_round": config.mu_anneal_start_round,
            "mu_anneal_rounds": config.mu_anneal_rounds,
            "mu_anneal_end": config.mu_anneal_end,
        },
        "shared_initial_table_sha256": residual_diag["initial_table_sha256"],
        "shared_primary_rng_endpoint_sha256": residual_diag[
            "primary_rng_state_sha256"
        ],
        "fixed_rounds": int(config.n_rounds),
    }
    return runs, pairing
