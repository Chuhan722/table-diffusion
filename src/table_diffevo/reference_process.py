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


REFERENCE_PROCESS_CONTRACT_VERSION = "issue53-stage1-v1"

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
    "horizon_invariant",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"reference diagnostics 包含不可 JSON 序列化类型：{type(value)!r}"
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
    """Run an opt-in fixed-parameter process and return the final current table.

    The wrapper owns all parameters that can violate the Stage 1 identity.  It
    deliberately rejects duplicate overrides rather than silently allowing a
    caller to weaken the fail-closed contract through ``kwargs``.
    """
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
        horizon_invariant=True,
        **kwargs,
    )
    final_table = diagnostics.pop("final_table")
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
        "version": REFERENCE_PROCESS_CONTRACT_VERSION,
        "output_state_role": "final_current",
        "historical_best_role": "diagnostic_only",
        "fixed_alpha_role": "convergence_calibration_only_not_selected",
        "n_rounds_role": "maximum_budget_only",
        "prefix_invariance_required": True,
    }
    serialized = json.dumps(
        diagnostics,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    diagnostics = json.loads(serialized)
    return final_table.reset_index(drop=True), diagnostics
