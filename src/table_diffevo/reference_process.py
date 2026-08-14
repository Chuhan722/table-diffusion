"""Horizon-invariant current-state process for Issue #53 Stage 1."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import (
    DEFAULT_DIRECTION_LOGIT_CLIP,
)
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import Schema
from table_diffevo.stationarity import StationarityTrace


REFERENCE_PROCESS_CONTRACT_VERSION = "issue53-stage1-v1"
STATIONARITY_CALIBRATION_CONTRACT_VERSION = "issue53-stage2a-v1"

_ENFORCED_ARGUMENTS = {
    "distance_mode",
    "alpha_schedule_mode",
    "alpha_min",
    "alpha_max",
    "fixed_alpha",
    "rho",
    "eta",
    "mu",
    "tol",
    "max_retries",
    "residual_directed_diffusion",
    "diffusion_direction_strength",
    "diffusion_direction_normalization",
    "diffusion_direction_reference_scale",
    "diffusion_direction_logit_clip",
    "residual_self_cooling",
    "self_cooling_monotone",
    "self_cooling_stop_ratio",
    "rho_anneal_end",
    "rho_anneal_rounds",
    "return_final_table",
    "record_transition_clocks",
    "record_stationarity_trace",
    "stop_on_exact_residual",
    "horizon_invariant",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"reference diagnostics 包含不可 JSON 序列化类型：{type(value)!r}"
    )


def _run_reference_process(
    target: np.ndarray,
    queries: List[Dict[str, Any]],
    schema: Schema,
    n_records: int,
    n_rounds: int,
    seed: int,
    *,
    fixed_alpha: float,
    rho: float,
    eta: float,
    mu: float,
    diffusion_direction_strength: float,
    diffusion_direction_reference_scale: float,
    diffusion_direction_logit_clip: Optional[float] = (
        DEFAULT_DIRECTION_LOGIT_CLIP
    ),
    _record_stationarity_trace: bool,
    _stop_on_exact_residual: bool,
    _contract_version: str,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[StationarityTrace]]:
    """Internal shared implementation for Stage 1 and Stage 2A wrappers."""
    overlap = sorted(_ENFORCED_ARGUMENTS.intersection(kwargs))
    if overlap:
        raise ValueError(
            "以下参数由 horizon-invariant 入口强制管理，不得覆盖："
            + ", ".join(overlap)
        )
    if isinstance(n_rounds, bool) or not isinstance(
        n_rounds, (int, np.integer)
    ) or n_rounds <= 0:
        raise ValueError("n_rounds 必须是正整数最大预算")

    _, diagnostics = run_evolution(
        target,
        queries,
        schema,
        n_records,
        n_rounds=int(n_rounds),
        seed=seed,
        distance_mode="geometric",
        fixed_alpha=fixed_alpha,
        alpha_schedule_mode="fixed",
        rho=rho,
        eta=eta,
        mu=mu,
        tol=float("inf"),
        max_retries=0,
        residual_directed_diffusion=True,
        diffusion_direction_strength=diffusion_direction_strength,
        diffusion_direction_normalization="fixed",
        diffusion_direction_reference_scale=(
            diffusion_direction_reference_scale
        ),
        diffusion_direction_logit_clip=diffusion_direction_logit_clip,
        residual_self_cooling=None,
        self_cooling_monotone=False,
        self_cooling_stop_ratio=None,
        rho_anneal_end=None,
        rho_anneal_rounds=None,
        return_final_table=True,
        record_transition_clocks=True,
        record_stationarity_trace=_record_stationarity_trace,
        stop_on_exact_residual=_stop_on_exact_residual,
        horizon_invariant=True,
        **kwargs,
    )
    final_table = diagnostics.pop("final_table")
    stationarity_trace = diagnostics.pop("stationarity_trace", None)
    diagnostics["params"]["tol"] = "positive_infinity_no_gate"
    for legacy_best_key in (
        "best_loss",
        "normalized_l1_error",
        "normalized_l1_median",
        "normalized_l1_p90",
        "normalized_l1_max",
    ):
        diagnostics.pop(legacy_best_key)
    diagnostics["reference_process_contract"] = {
        "version": _contract_version,
        "output_state_role": "final_current",
        "historical_best_role": "diagnostic_only",
        "fixed_alpha_role": "convergence_calibration_only_not_selected",
        "n_rounds_role": "maximum_budget_only",
        "prefix_invariance_required": True,
        "exact_residual_stop": (
            "enabled_legacy_stage1"
            if _stop_on_exact_residual
            else "disabled_stage2a_stationarity_calibration"
        ),
        "stationarity_trace_role": (
            "returned_separately_not_in_json_diagnostics"
            if _record_stationarity_trace else "not_recorded"
        ),
    }
    serialized = json.dumps(
        diagnostics,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    diagnostics = json.loads(serialized)
    return (
        final_table.reset_index(drop=True),
        diagnostics,
        stationarity_trace,
    )


def run_horizon_invariant_evolution(
    target: np.ndarray,
    queries: List[Dict[str, Any]],
    schema: Schema,
    n_records: int,
    n_rounds: int,
    seed: int,
    *,
    fixed_alpha: float,
    rho: float,
    eta: float,
    mu: float,
    diffusion_direction_strength: float,
    diffusion_direction_reference_scale: float,
    diffusion_direction_logit_clip: Optional[float] = (
        DEFAULT_DIRECTION_LOGIT_CLIP
    ),
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the Stage 1 fixed-parameter process and return final current state."""
    final_table, diagnostics, trace = _run_reference_process(
        target,
        queries,
        schema,
        n_records,
        n_rounds,
        seed,
        fixed_alpha=fixed_alpha,
        rho=rho,
        eta=eta,
        mu=mu,
        diffusion_direction_strength=diffusion_direction_strength,
        diffusion_direction_reference_scale=(
            diffusion_direction_reference_scale
        ),
        diffusion_direction_logit_clip=diffusion_direction_logit_clip,
        _record_stationarity_trace=False,
        _stop_on_exact_residual=True,
        _contract_version=REFERENCE_PROCESS_CONTRACT_VERSION,
        **kwargs,
    )
    if trace is not None:  # pragma: no cover - internal fail-closed assertion
        raise RuntimeError("Stage 1 reference process 不应返回 stationarity trace")
    return final_table, diagnostics


def run_stationarity_calibration_evolution(
    target: np.ndarray,
    queries: List[Dict[str, Any]],
    schema: Schema,
    n_records: int,
    n_rounds: int,
    seed: int,
    *,
    fixed_alpha: float,
    rho: float,
    eta: float,
    mu: float,
    diffusion_direction_strength: float,
    diffusion_direction_reference_scale: float,
    diffusion_direction_logit_clip: Optional[float] = (
        DEFAULT_DIRECTION_LOGIT_CLIP
    ),
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any], StationarityTrace]:
    """Run a fixed-budget Stage 2A trace with legacy exact-target stop disabled."""
    final_table, diagnostics, trace = _run_reference_process(
        target,
        queries,
        schema,
        n_records,
        n_rounds,
        seed,
        fixed_alpha=fixed_alpha,
        rho=rho,
        eta=eta,
        mu=mu,
        diffusion_direction_strength=diffusion_direction_strength,
        diffusion_direction_reference_scale=(
            diffusion_direction_reference_scale
        ),
        diffusion_direction_logit_clip=diffusion_direction_logit_clip,
        _record_stationarity_trace=True,
        _stop_on_exact_residual=False,
        _contract_version=STATIONARITY_CALIBRATION_CONTRACT_VERSION,
        **kwargs,
    )
    if trace is None:  # pragma: no cover - internal fail-closed assertion
        raise RuntimeError("Stage 2A calibration process 缺少 stationarity trace")
    return final_table, diagnostics, trace
