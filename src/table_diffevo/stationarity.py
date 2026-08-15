"""Issue #53 Stage 2A current-state trace and offline replay tools.

The detector in this module is deliberately parameter-blind: it consumes only
an observed current-state trajectory plus an explicit detector configuration.
It does not know the initializer, random seed, transition kernel, or any
generation parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.quality import query_fingerprint


STATIONARITY_TRACE_CONTRACT_VERSION = "issue53-stage2a-trace-v1"
STATIONARITY_REPLAY_CONTRACT_VERSION = "issue53-stage2a-replay-v2"
STATIONARITY_RANGE_EVIDENCE_CONTRACT_VERSION = (
    "issue53-stage2b-range-evidence-v1"
)
STATIONARITY_QUERY_MAX_RANGE_EVIDENCE_CONTRACT_VERSION = (
    "issue53-stage2b-query-max-range-evidence-v1"
)
STATIONARITY_QUERY_MAX_REPLAY_CONTRACT_VERSION = (
    "issue53-stage2b-query-max-replay-v1"
)

_QUERY_ARRAY_FILENAME = "measured_query_answers.npz"
_TRACE_METADATA_FILENAME = "stationarity_trace.json"
_QUERY_ARRAY_KEY = "measured_query_answers"

_TERMINATION_REASONS = {
    "in_progress",
    "max_rounds",
    "candidate_budget",
    "exact_residual",
    "self_cooling_ratio",
    "stationary_qualified",
    "stalled",
}

_OBSERVATION_KEYS = {
    "state_index",
    "round_index",
    "phase",
    "current_normalized_l1",
    "current_squared_loss",
    "unique_row_count",
    "unique_row_rate",
    "empirical_row_entropy",
    "normalized_row_entropy",
    "effective_unique_row_ratio",
    "proposal_attempt_count",
    "proposal_accepted",
    "applied_attempt_index",
    "attempted_participating_row_count",
    "applied_participating_row_count",
    "actual_changed_row_count",
    "actual_changed_cell_count",
    "actual_changed_query_count",
    "normalized_query_l1_movement_mean",
    "gibbs_microstep_count_attempted",
    "gibbs_microstep_count_applied",
    "candidate_evaluation_count_cumulative",
    "state_changed",
    "current_table_sha256",
    "primary_rng_state_sha256",
    "factorized_gibbs_rng_state_sha256",
}

_TRACE_METADATA_KEYS = {
    "contract_version",
    "n_records",
    "query_count",
    "state_count",
    "post_round_count",
    "query_identity_sha256",
    "target_identity_sha256",
    "termination_reason",
    "observations",
    "query_array",
}

_QUERY_ARRAY_METADATA_KEYS = {
    "filename",
    "key",
    "shape",
    "dtype",
    "sha256",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_exact_keys(
    value: Any,
    expected: set[str],
    name: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是对象")
    missing = sorted(expected.difference(value))
    if missing:
        raise ValueError(f"{name} 缺少字段：" + ", ".join(missing))
    unknown = sorted(set(value).difference(expected))
    if unknown:
        raise ValueError(f"{name} 包含未知字段：" + ", ".join(unknown))


def ordered_query_identity_sha256(
    queries: Sequence[Dict[str, Any]],
) -> str:
    """Hash query semantics in vector-column order."""
    fingerprints = [query_fingerprint(query) for query in queries]
    if not fingerprints:
        raise ValueError("queries 不能为空")
    return _sha256_bytes(_strict_json_bytes(fingerprints))


def target_answer_identity_sha256(target: Sequence[float]) -> str:
    """Hash a finite one-dimensional target vector deterministically."""
    values = np.asarray(target, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("target 必须是非空一维数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("target 必须全部为有限数值")
    return _sha256_bytes(
        _strict_json_bytes([float(value) for value in values])
    )


def stationarity_row_diversity_metrics(
    frame: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute the source-free row diversity required by Stage 2A."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("current_table 必须是 DataFrame")
    n_records = len(frame)
    if n_records <= 0:
        raise ValueError("current_table 不能为空")

    counts = frame.value_counts(dropna=False, sort=False).to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    row_entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if n_records == 1:
        normalized_entropy = 1.0
    else:
        normalized_entropy = float(row_entropy / np.log(n_records))
    effective_unique_rows = float(np.exp(row_entropy))
    unique_count = int(len(counts))
    return {
        "unique_row_count": unique_count,
        "unique_row_rate": float(unique_count / n_records),
        "empirical_row_entropy": row_entropy,
        "normalized_row_entropy": normalized_entropy,
        "effective_unique_row_ratio": float(
            effective_unique_rows / n_records
        ),
    }


def build_stationarity_observation(
    *,
    frame: pd.DataFrame,
    target: Sequence[float],
    current_query_answers: Sequence[float],
    n_records: int,
    squared_loss: float,
    state_index: int,
    round_index: int,
    phase: str,
    proposal_attempt_count: int,
    proposal_accepted: bool,
    applied_attempt_index: int,
    attempted_participating_row_count: int,
    applied_participating_row_count: int,
    actual_changed_row_count: int,
    actual_changed_cell_count: int,
    actual_changed_query_count: int,
    normalized_query_l1_movement_mean: float,
    gibbs_microstep_count_attempted: int,
    gibbs_microstep_count_applied: int,
    candidate_evaluation_count_cumulative: int,
    current_table_sha256: str,
    primary_rng_state_sha256: str,
    factorized_gibbs_rng_state_sha256: str | None,
) -> Dict[str, Any]:
    """Build one deterministic state observation without evaluating queries."""
    if phase not in {"initial", "post_round"}:
        raise ValueError("phase 必须是 initial 或 post_round")
    answers = np.asarray(current_query_answers, dtype=float)
    if answers.ndim != 1 or len(answers) == 0:
        raise ValueError("current_query_answers 必须是非空一维数组")
    if not np.all(np.isfinite(answers)):
        raise ValueError("current_query_answers 必须全部为有限数值")
    if len(frame) != n_records:
        raise ValueError("current_table 行数必须等于 n_records")
    diversity = stationarity_row_diversity_metrics(frame)
    changed = int(actual_changed_cell_count) > 0
    observation = {
        "state_index": int(state_index),
        "round_index": int(round_index),
        "phase": phase,
        "current_normalized_l1": compute_normalized_l1(
            np.asarray(target, dtype=float), answers, n_records
        ),
        "current_squared_loss": float(squared_loss),
        **diversity,
        "proposal_attempt_count": int(proposal_attempt_count),
        "proposal_accepted": bool(proposal_accepted),
        "applied_attempt_index": int(applied_attempt_index),
        "attempted_participating_row_count": int(
            attempted_participating_row_count
        ),
        "applied_participating_row_count": int(
            applied_participating_row_count
        ),
        "actual_changed_row_count": int(actual_changed_row_count),
        "actual_changed_cell_count": int(actual_changed_cell_count),
        "actual_changed_query_count": int(actual_changed_query_count),
        "normalized_query_l1_movement_mean": float(
            normalized_query_l1_movement_mean
        ),
        "gibbs_microstep_count_attempted": int(
            gibbs_microstep_count_attempted
        ),
        "gibbs_microstep_count_applied": int(
            gibbs_microstep_count_applied
        ),
        "candidate_evaluation_count_cumulative": int(
            candidate_evaluation_count_cumulative
        ),
        "state_changed": bool(changed),
        "current_table_sha256": current_table_sha256,
        "primary_rng_state_sha256": primary_rng_state_sha256,
        "factorized_gibbs_rng_state_sha256": (
            factorized_gibbs_rng_state_sha256
        ),
    }
    _validate_observation(observation, n_records=n_records)
    return observation


def _validate_sha256(value: Any, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} 必须是 SHA-256 十六进制字符串")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 SHA-256 十六进制字符串") from exc


def _validate_observation(
    observation: Dict[str, Any],
    *,
    n_records: int,
) -> None:
    _validate_exact_keys(observation, _OBSERVATION_KEYS, "轨迹观测")
    if observation["phase"] not in {"initial", "post_round"}:
        raise ValueError("轨迹 phase 非法")

    nonnegative_integer_keys = (
        "state_index",
        "round_index",
        "unique_row_count",
        "proposal_attempt_count",
        "applied_attempt_index",
        "attempted_participating_row_count",
        "applied_participating_row_count",
        "actual_changed_row_count",
        "actual_changed_cell_count",
        "actual_changed_query_count",
        "gibbs_microstep_count_attempted",
        "gibbs_microstep_count_applied",
        "candidate_evaluation_count_cumulative",
    )
    for key in nonnegative_integer_keys:
        value = observation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} 必须是非负整数")

    finite_nonnegative_keys = (
        "current_normalized_l1",
        "current_squared_loss",
        "empirical_row_entropy",
        "normalized_query_l1_movement_mean",
    )
    for key in finite_nonnegative_keys:
        value = observation[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值")
        if value < 0.0:
            raise ValueError(f"{key} 必须非负")

    for key in (
        "unique_row_rate",
        "normalized_row_entropy",
        "effective_unique_row_ratio",
    ):
        value = observation[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值")
        if not 0.0 <= value <= 1.0 + 1e-12:
            raise ValueError(f"{key} 必须位于 [0, 1]")

    if observation["unique_row_count"] > n_records:
        raise ValueError("unique_row_count 不能超过 n_records")
    if observation["applied_participating_row_count"] > n_records:
        raise ValueError("applied_participating_row_count 不能超过 n_records")
    if observation["applied_participating_row_count"] > observation[
        "attempted_participating_row_count"
    ]:
        raise ValueError("applied participating rows 不能超过 attempted work")
    if observation["actual_changed_row_count"] > n_records:
        raise ValueError("actual_changed_row_count 不能超过 n_records")
    if observation["actual_changed_row_count"] > observation[
        "applied_participating_row_count"
    ]:
        raise ValueError("actual changed rows 不能超过 applied participating rows")
    if observation["attempted_participating_row_count"] > (
        observation["proposal_attempt_count"] * n_records
    ):
        raise ValueError("attempted_participating_row_count 超过尝试上限")
    if not isinstance(observation["proposal_accepted"], bool):
        raise ValueError("proposal_accepted 必须是布尔值")
    if not isinstance(observation["state_changed"], bool):
        raise ValueError("state_changed 必须是布尔值")
    if observation["state_changed"] != (
        observation["actual_changed_cell_count"] > 0
    ):
        raise ValueError("state_changed 与 actual_changed_cell_count 不一致")
    if (observation["actual_changed_row_count"] > 0) != (
        observation["actual_changed_cell_count"] > 0
    ):
        raise ValueError("actual changed rows 与 changed cells 不一致")
    if not observation["proposal_accepted"] and (
        observation["applied_attempt_index"] != 0
        or observation["applied_participating_row_count"] != 0
        or observation["actual_changed_row_count"] != 0
        or observation["actual_changed_cell_count"] != 0
        or observation["actual_changed_query_count"] != 0
        or observation["normalized_query_l1_movement_mean"] != 0.0
        or observation["gibbs_microstep_count_applied"] != 0
    ):
        raise ValueError("未接受 proposal 时实际状态运动必须为 0")
    if observation["proposal_accepted"] and not (
        1
        <= observation["applied_attempt_index"]
        <= observation["proposal_attempt_count"]
    ):
        raise ValueError("接受 proposal 时 applied_attempt_index 非法")
    if observation["gibbs_microstep_count_applied"] > observation[
        "gibbs_microstep_count_attempted"
    ]:
        raise ValueError("applied Gibbs microsteps 不能超过 attempted work")
    if observation["phase"] == "initial" and (
        observation["proposal_attempt_count"] != 0
        or observation["proposal_accepted"]
        or observation["gibbs_microstep_count_attempted"] != 0
        or observation["candidate_evaluation_count_cumulative"] != 0
    ):
        raise ValueError("initial 状态不能携带 proposal 工作量")
    _validate_sha256(
        observation["current_table_sha256"], "current_table_sha256"
    )
    _validate_sha256(
        observation["primary_rng_state_sha256"],
        "primary_rng_state_sha256",
    )
    _validate_sha256(
        observation["factorized_gibbs_rng_state_sha256"],
        "factorized_gibbs_rng_state_sha256",
        optional=True,
    )
    json.dumps(observation, ensure_ascii=False, allow_nan=False)


@dataclass
class StationarityTrace:
    """A versioned current-state trace with query vectors stored separately."""

    n_records: int
    query_identity_sha256: str
    target_identity_sha256: str
    observations: List[Dict[str, Any]]
    measured_query_answers: np.ndarray
    termination_reason: str
    contract_version: str = STATIONARITY_TRACE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.observations, list):
            raise ValueError("observations 必须是列表")
        self.measured_query_answers = np.asarray(
            self.measured_query_answers, dtype=np.float64
        ).copy()
        self.observations = [dict(row) for row in self.observations]
        self.validate()

    @property
    def query_count(self) -> int:
        return int(self.measured_query_answers.shape[1])

    @property
    def state_count(self) -> int:
        return len(self.observations)

    @property
    def post_round_count(self) -> int:
        return sum(row["phase"] == "post_round" for row in self.observations)

    def post_round_positions(self) -> List[int]:
        return [
            position
            for position, row in enumerate(self.observations)
            if row["phase"] == "post_round"
        ]

    def validate(self) -> None:
        if self.contract_version != STATIONARITY_TRACE_CONTRACT_VERSION:
            raise ValueError(
                "不支持的 stationarity trace contract version："
                f"{self.contract_version!r}"
            )
        if isinstance(self.n_records, bool) or not isinstance(
            self.n_records, int
        ) or self.n_records <= 0:
            raise ValueError("n_records 必须是正整数")
        _validate_sha256(
            self.query_identity_sha256, "query_identity_sha256"
        )
        _validate_sha256(
            self.target_identity_sha256, "target_identity_sha256"
        )
        if self.termination_reason not in _TERMINATION_REASONS:
            raise ValueError(
                f"未知 termination_reason：{self.termination_reason!r}"
            )
        answers = self.measured_query_answers
        if answers.ndim != 2 or answers.shape[0] != len(self.observations):
            raise ValueError("measured_query_answers 与 observations 形状不一致")
        if answers.shape[1] <= 0 or not np.all(np.isfinite(answers)):
            raise ValueError("measured_query_answers 必须是有限非空二维数组")
        if np.any(answers < 0.0) or np.any(answers > self.n_records):
            raise ValueError("measured_query_answers 必须是 [0, N] 查询计数")
        if not self.observations:
            raise ValueError("stationarity trace 至少需要初始状态")

        for expected_index, observation in enumerate(self.observations):
            _validate_observation(observation, n_records=self.n_records)
            if observation["state_index"] != expected_index:
                raise ValueError("state_index 必须从 0 连续递增")
            expected_phase = "initial" if expected_index == 0 else "post_round"
            if observation["phase"] != expected_phase:
                raise ValueError("只有首个状态可以是 initial")
            expected_round = expected_index
            if observation["round_index"] != expected_round:
                raise ValueError("round_index 必须与真实状态转移序号对齐")
            if observation["actual_changed_query_count"] > self.query_count:
                raise ValueError("actual_changed_query_count 超过 query_count")
            if expected_index > 0:
                if observation["proposal_attempt_count"] <= 0:
                    raise ValueError("post_round 必须包含至少一次 proposal attempt")
                previous_candidate_count = self.observations[
                    expected_index - 1
                ]["candidate_evaluation_count_cumulative"]
                if observation[
                    "candidate_evaluation_count_cumulative"
                ] - previous_candidate_count != observation[
                    "proposal_attempt_count"
                ]:
                    raise ValueError(
                        "candidate evaluation count 与 proposal attempts 不一致"
                    )
                previous_answers = answers[expected_index - 1]
                current_answers = answers[expected_index]
                delta = current_answers - previous_answers
                expected_changed_queries = int(np.count_nonzero(delta))
                expected_movement = float(
                    np.mean(np.abs(delta)) / self.n_records
                )
                if observation["actual_changed_query_count"] != (
                    expected_changed_queries
                ):
                    raise ValueError(
                        "actual_changed_query_count 与 query vector 不一致"
                    )
                if expected_changed_queries > 0 and not observation[
                    "state_changed"
                ]:
                    raise ValueError("query vector 改变时 current table 必须改变")
                if not np.isclose(
                    observation["normalized_query_l1_movement_mean"],
                    expected_movement,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    raise ValueError(
                        "normalized query movement 与 query vector 不一致"
                    )
                table_hash_changed = (
                    observation["current_table_sha256"]
                    != self.observations[expected_index - 1][
                        "current_table_sha256"
                    ]
                )
                if observation["state_changed"] != table_hash_changed:
                    raise ValueError("state_changed 与 current table hash 不一致")
        json.dumps(self.observations, ensure_ascii=False, allow_nan=False)


def save_stationarity_trace(
    trace: StationarityTrace,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Save one trace into a new directory without pickle or overwrites."""
    trace.validate()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"轨迹输出目录已存在：{destination}")
    destination.mkdir(parents=True)
    array_path = destination / _QUERY_ARRAY_FILENAME
    metadata_path = destination / _TRACE_METADATA_FILENAME
    try:
        with array_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{_QUERY_ARRAY_KEY: trace.measured_query_answers},
            )
        array_sha256 = _sha256_bytes(array_path.read_bytes())
        metadata = {
            "contract_version": trace.contract_version,
            "n_records": trace.n_records,
            "query_count": trace.query_count,
            "state_count": trace.state_count,
            "post_round_count": trace.post_round_count,
            "query_identity_sha256": trace.query_identity_sha256,
            "target_identity_sha256": trace.target_identity_sha256,
            "termination_reason": trace.termination_reason,
            "observations": trace.observations,
            "query_array": {
                "filename": _QUERY_ARRAY_FILENAME,
                "key": _QUERY_ARRAY_KEY,
                "shape": list(trace.measured_query_answers.shape),
                "dtype": str(trace.measured_query_answers.dtype),
                "sha256": array_sha256,
            },
        }
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        "metadata_path": str(metadata_path),
        "query_array_path": str(array_path),
        "query_array_sha256": array_sha256,
    }


def load_stationarity_trace(input_dir: str | Path) -> StationarityTrace:
    """Load and strictly validate a trace written by save_stationarity_trace."""
    source = Path(input_dir)
    metadata_path = source / _TRACE_METADATA_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取严格 stationarity trace metadata") from exc
    _validate_exact_keys(metadata, _TRACE_METADATA_KEYS, "trace metadata")
    array_info = metadata.get("query_array")
    _validate_exact_keys(
        array_info,
        _QUERY_ARRAY_METADATA_KEYS,
        "query_array metadata",
    )
    if array_info.get("filename") != _QUERY_ARRAY_FILENAME:
        raise ValueError("query array 文件名与契约不一致")
    if array_info.get("key") != _QUERY_ARRAY_KEY:
        raise ValueError("query array key 与契约不一致")
    array_path = source / _QUERY_ARRAY_FILENAME
    try:
        payload = array_path.read_bytes()
    except OSError as exc:
        raise ValueError("无法读取 measured query answer array") from exc
    if _sha256_bytes(payload) != array_info.get("sha256"):
        raise ValueError("measured query answer array SHA-256 不一致")
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != {_QUERY_ARRAY_KEY}:
                raise ValueError("query array archive 包含未知字段")
            answers = np.asarray(archive[_QUERY_ARRAY_KEY]).copy()
    except (OSError, ValueError) as exc:
        raise ValueError("无法严格加载 measured query answer array") from exc
    if list(answers.shape) != array_info.get("shape"):
        raise ValueError("query array shape 与 metadata 不一致")
    if str(answers.dtype) != array_info.get("dtype"):
        raise ValueError("query array dtype 与 metadata 不一致")
    if answers.dtype != np.dtype(np.float64):
        raise ValueError("query array dtype 必须是 float64")

    trace = StationarityTrace(
        n_records=metadata.get("n_records"),
        query_identity_sha256=metadata.get("query_identity_sha256"),
        target_identity_sha256=metadata.get("target_identity_sha256"),
        observations=metadata.get("observations"),
        measured_query_answers=answers,
        termination_reason=metadata.get("termination_reason"),
        contract_version=metadata.get("contract_version"),
    )
    if trace.query_count != metadata.get("query_count"):
        raise ValueError("query_count 与 trace 不一致")
    if trace.state_count != metadata.get("state_count"):
        raise ValueError("state_count 与 trace 不一致")
    if trace.post_round_count != metadata.get("post_round_count"):
        raise ValueError("post_round_count 与 trace 不一致")
    return trace


def _trace_identity_sha256(trace: StationarityTrace) -> str:
    """Bind one replay result to the exact trace contents it consumed."""
    answer_bytes = np.ascontiguousarray(
        trace.measured_query_answers, dtype="<f8"
    ).tobytes(order="C")
    identity = {
        "contract_version": trace.contract_version,
        "n_records": trace.n_records,
        "query_identity_sha256": trace.query_identity_sha256,
        "target_identity_sha256": trace.target_identity_sha256,
        "termination_reason": trace.termination_reason,
        "observations": trace.observations,
        "measured_query_answers": {
            "shape": list(trace.measured_query_answers.shape),
            "dtype": "float64",
            "sha256": _sha256_bytes(answer_bytes),
        },
    }
    return _sha256_bytes(_strict_json_bytes(identity))


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} 必须是非负有限数值")
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} 必须是非负有限数值")
    return converted


@dataclass(frozen=True)
class StationarityDetectorConfig:
    """Explicit Stage 2A method settings; no production defaults are supplied.

    L1 location is the arithmetic window mean. L1 spread is the window
    P90-P10 inter-percentile range. Movement requires both sustained active
    rounds and a non-microscopic mean changed-row fraction in every window.
    """

    window_size: int
    query_mean_shift_tolerance: float
    query_p95_shift_tolerance: float
    l1_mean_shift_tolerance: float
    l1_p90_minus_p10_shift_tolerance: float
    unique_row_rate_tolerance: float
    normalized_row_entropy_tolerance: float
    minimum_active_round_rate: float
    minimum_mean_changed_row_fraction: float
    stall_patience_checks: int

    def __post_init__(self) -> None:
        if isinstance(self.window_size, bool) or not isinstance(
            self.window_size, int
        ) or self.window_size < 2:
            raise ValueError("window_size 必须是至少为 2 的整数")
        if isinstance(self.stall_patience_checks, bool) or not isinstance(
            self.stall_patience_checks, int
        ) or self.stall_patience_checks <= 0:
            raise ValueError("stall_patience_checks 必须是正整数")
        for name in (
            "query_mean_shift_tolerance",
            "query_p95_shift_tolerance",
            "l1_mean_shift_tolerance",
            "l1_p90_minus_p10_shift_tolerance",
            "unique_row_rate_tolerance",
            "normalized_row_entropy_tolerance",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        for name in (
            "minimum_active_round_rate",
            "minimum_mean_changed_row_fraction",
        ):
            movement_rate = _finite_nonnegative(getattr(self, name), name)
            if not 0.0 < movement_rate <= 1.0:
                raise ValueError(f"{name} 必须位于 (0, 1]")
            object.__setattr__(self, name, movement_rate)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "window_size": self.window_size,
            "query_mean_shift_tolerance": self.query_mean_shift_tolerance,
            "query_p95_shift_tolerance": self.query_p95_shift_tolerance,
            "l1_mean_shift_tolerance": self.l1_mean_shift_tolerance,
            "l1_p90_minus_p10_shift_tolerance": (
                self.l1_p90_minus_p10_shift_tolerance
            ),
            "unique_row_rate_tolerance": self.unique_row_rate_tolerance,
            "normalized_row_entropy_tolerance": (
                self.normalized_row_entropy_tolerance
            ),
            "minimum_active_round_rate": self.minimum_active_round_rate,
            "minimum_mean_changed_row_fraction": (
                self.minimum_mean_changed_row_fraction
            ),
            "stall_patience_checks": self.stall_patience_checks,
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result


@dataclass(frozen=True)
class QueryMaxStationarityDetectorConfig:
    """Versioned extension that adds a worst-query drift guard.

    The frozen Stage 2B detector config remains an unchanged nested value so
    old replay and validation protocols cannot silently acquire the new rule.
    """

    base_config: StationarityDetectorConfig
    query_max_shift_tolerance: float

    def __post_init__(self) -> None:
        if type(self.base_config) is not StationarityDetectorConfig:
            raise ValueError(
                "base_config 必须是原版 StationarityDetectorConfig"
            )
        object.__setattr__(
            self,
            "query_max_shift_tolerance",
            _finite_nonnegative(
                self.query_max_shift_tolerance,
                "query_max_shift_tolerance",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            **self.base_config.to_dict(),
            "query_max_shift_tolerance": self.query_max_shift_tolerance,
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result


@dataclass
class StationarityReplayResult:
    """Deterministic result of replaying a detector over one trace."""

    status: str
    candidate_state_index: int | None
    candidate_round_index: int | None
    checks: List[Dict[str, Any]]
    trace: Dict[str, Any]
    detector_config: Dict[str, Any]
    contract_version: str = STATIONARITY_REPLAY_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "contract_version": self.contract_version,
            "status": self.status,
            "candidate_state_index": self.candidate_state_index,
            "candidate_round_index": self.candidate_round_index,
            "trace": self.trace,
            "detector_config": self.detector_config,
            "checks": self.checks,
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result


def _max_pairwise_difference(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.max(array) - np.min(array))


def _window_summary(
    trace: StationarityTrace,
    positions: Sequence[int],
) -> Dict[str, Any]:
    answers = trace.measured_query_answers[np.asarray(positions)]
    rows = [trace.observations[position] for position in positions]
    l1 = np.asarray([row["current_normalized_l1"] for row in rows])
    return {
        "query_mean": np.mean(answers / trace.n_records, axis=0),
        "l1_mean": float(np.mean(l1)),
        "l1_p90_minus_p10": float(
            np.percentile(l1, 90, method="linear")
            - np.percentile(l1, 10, method="linear")
        ),
        "l1_p95": float(np.percentile(l1, 95, method="linear")),
        "unique_row_rate": float(
            np.mean([row["unique_row_rate"] for row in rows])
        ),
        "normalized_row_entropy": float(
            np.mean([row["normalized_row_entropy"] for row in rows])
        ),
        "active_round_rate": float(np.mean([
            row["actual_changed_row_count"] > 0 for row in rows
        ])),
        "mean_changed_row_fraction": float(np.mean([
            row["actual_changed_row_count"] / trace.n_records
            for row in rows
        ])),
    }


def _range_evidence_from_summaries(
    summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return threshold-free evidence for exactly three state windows."""
    if len(summaries) != 3:
        raise ValueError("量程证据必须恰好比较三个窗口")
    query_mean_shifts = []
    query_p95_shifts = []
    for left_index in range(len(summaries)):
        for right_index in range(left_index + 1, len(summaries)):
            shift = np.abs(
                summaries[left_index]["query_mean"]
                - summaries[right_index]["query_mean"]
            )
            query_mean_shifts.append(float(np.mean(shift)))
            query_p95_shifts.append(
                float(np.percentile(shift, 95, method="linear"))
            )

    query_mean_shift = max(query_mean_shifts)
    query_p95_shift = max(query_p95_shifts)
    l1_mean_shift = _max_pairwise_difference(
        [summary["l1_mean"] for summary in summaries]
    )
    l1_p90_minus_p10_shift = _max_pairwise_difference(
        [summary["l1_p90_minus_p10"] for summary in summaries]
    )
    unique_row_rate_shift = _max_pairwise_difference(
        [summary["unique_row_rate"] for summary in summaries]
    )
    normalized_row_entropy_shift = _max_pairwise_difference(
        [summary["normalized_row_entropy"] for summary in summaries]
    )
    window_active_round_rates = [
        summary["active_round_rate"] for summary in summaries
    ]
    window_mean_changed_row_fractions = [
        summary["mean_changed_row_fraction"] for summary in summaries
    ]
    minimum_observed_active_round_rate = min(window_active_round_rates)
    minimum_observed_mean_changed_row_fraction = min(
        window_mean_changed_row_fractions
    )
    return {
        "query_mean_shift": query_mean_shift,
        "query_p95_shift": query_p95_shift,
        "l1_mean_shift": l1_mean_shift,
        "l1_p90_minus_p10_shift": l1_p90_minus_p10_shift,
        "window_l1_means": [
            summary["l1_mean"] for summary in summaries
        ],
        "window_l1_p90_minus_p10": [
            summary["l1_p90_minus_p10"] for summary in summaries
        ],
        "window_l1_p95": [
            summary["l1_p95"] for summary in summaries
        ],
        "unique_row_rate_shift": unique_row_rate_shift,
        "normalized_row_entropy_shift": normalized_row_entropy_shift,
        "window_active_round_rates": window_active_round_rates,
        "window_mean_changed_row_fractions": (
            window_mean_changed_row_fractions
        ),
        "minimum_observed_active_round_rate": (
            minimum_observed_active_round_rate
        ),
        "minimum_observed_mean_changed_row_fraction": (
            minimum_observed_mean_changed_row_fraction
        ),
    }


def _query_max_range_evidence_from_summaries(
    summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extend the unchanged range evidence with worst-coordinate drift."""
    evidence = _range_evidence_from_summaries(summaries)
    query_max_shifts = []
    for left_index in range(len(summaries)):
        for right_index in range(left_index + 1, len(summaries)):
            shift = np.abs(
                summaries[left_index]["query_mean"]
                - summaries[right_index]["query_mean"]
            )
            query_max_shifts.append(float(np.max(shift)))
    evidence["query_max_shift"] = max(query_max_shifts)
    return evidence


def _range_evidence(
    trace: StationarityTrace,
    windows: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    summaries = [_window_summary(trace, positions) for positions in windows]
    return _range_evidence_from_summaries(summaries)


def _stability_evidence(
    trace: StationarityTrace,
    windows: Sequence[Sequence[int]],
    config: StationarityDetectorConfig,
) -> Dict[str, Any]:
    evidence = _range_evidence(trace, windows)
    evidence["stable"] = bool(
        evidence["query_mean_shift"]
        <= config.query_mean_shift_tolerance
        and evidence["query_p95_shift"]
        <= config.query_p95_shift_tolerance
        and evidence["l1_mean_shift"]
        <= config.l1_mean_shift_tolerance
        and evidence["l1_p90_minus_p10_shift"]
        <= config.l1_p90_minus_p10_shift_tolerance
        and evidence["unique_row_rate_shift"]
        <= config.unique_row_rate_tolerance
        and evidence["normalized_row_entropy_shift"]
        <= config.normalized_row_entropy_tolerance
    )
    evidence["movement_sufficient"] = bool(
        evidence["minimum_observed_active_round_rate"]
        >= config.minimum_active_round_rate
        and evidence["minimum_observed_mean_changed_row_fraction"]
        >= config.minimum_mean_changed_row_fraction
    )
    return evidence


def _query_max_stability_evidence(
    trace: StationarityTrace,
    windows: Sequence[Sequence[int]],
    config: QueryMaxStationarityDetectorConfig,
) -> Dict[str, Any]:
    summaries = [_window_summary(trace, positions) for positions in windows]
    evidence = _query_max_range_evidence_from_summaries(summaries)
    base = config.base_config
    evidence["stable"] = bool(
        evidence["query_mean_shift"] <= base.query_mean_shift_tolerance
        and evidence["query_p95_shift"] <= base.query_p95_shift_tolerance
        and evidence["query_max_shift"]
        <= config.query_max_shift_tolerance
        and evidence["l1_mean_shift"] <= base.l1_mean_shift_tolerance
        and evidence["l1_p90_minus_p10_shift"]
        <= base.l1_p90_minus_p10_shift_tolerance
        and evidence["unique_row_rate_shift"]
        <= base.unique_row_rate_tolerance
        and evidence["normalized_row_entropy_shift"]
        <= base.normalized_row_entropy_tolerance
    )
    evidence["movement_sufficient"] = bool(
        evidence["minimum_observed_active_round_rate"]
        >= base.minimum_active_round_rate
        and evidence["minimum_observed_mean_changed_row_fraction"]
        >= base.minimum_mean_changed_row_fraction
    )
    return evidence


def collect_stationarity_range_evidence(
    trace: StationarityTrace,
    window_sizes: Sequence[int],
) -> List[Dict[str, Any]]:
    """Collect block-aligned three-window evidence without thresholds.

    This Stage 2B entry point deliberately has no detector configuration and
    returns no pass/fail, stationarity, or stall classification.  Its formulas
    and block alignment are shared with :func:`replay_stationarity` so range
    finding cannot silently measure a different statistic from the detector.
    """
    trace.validate()
    sizes = list(window_sizes)
    if not sizes:
        raise ValueError("window_sizes 不能为空")
    for window_size in sizes:
        if isinstance(window_size, bool) or not isinstance(
            window_size, (int, np.integer)
        ) or int(window_size) < 2:
            raise ValueError("每个 window_size 必须是至少为 2 的整数")
    normalized_sizes = [int(value) for value in sizes]
    if len(set(normalized_sizes)) != len(normalized_sizes):
        raise ValueError("window_sizes 不得重复")

    positions = trace.post_round_positions()
    checks: List[Dict[str, Any]] = []
    for window_size in normalized_sizes:
        completed_blocks = len(positions) // window_size
        block_positions = [
            positions[
                block_index * window_size:(block_index + 1) * window_size
            ]
            for block_index in range(completed_blocks)
        ]
        block_summaries = [
            _window_summary(trace, block) for block in block_positions
        ]
        for block_count in range(3, completed_blocks + 1):
            windows = block_positions[block_count - 3:block_count]
            summaries = block_summaries[block_count - 3:block_count]
            terminal_position = windows[-1][-1]
            terminal_observation = trace.observations[terminal_position]
            round_ranges = [
                [
                    int(trace.observations[window[0]]["round_index"]),
                    int(trace.observations[window[-1]]["round_index"]),
                ]
                for window in windows
            ]
            checks.append({
                "completed_block_count": int(block_count),
                "window_size": int(window_size),
                "state_index": int(terminal_observation["state_index"]),
                "round_index": int(terminal_observation["round_index"]),
                "window_round_ranges": round_ranges,
                **_range_evidence_from_summaries(summaries),
            })
    json.dumps(checks, ensure_ascii=False, allow_nan=False)
    return checks


def collect_query_max_stationarity_range_evidence(
    trace: StationarityTrace,
    window_sizes: Sequence[int],
) -> List[Dict[str, Any]]:
    """Collect threshold-free evidence including worst-query drift.

    This is a separate versioned entry point.  The original Stage 2B range
    evidence and replay outputs remain byte-for-byte unchanged.
    """
    trace.validate()
    sizes = list(window_sizes)
    if not sizes:
        raise ValueError("window_sizes 不能为空")
    for window_size in sizes:
        if isinstance(window_size, bool) or not isinstance(
            window_size, (int, np.integer)
        ) or int(window_size) < 2:
            raise ValueError("每个 window_size 必须是至少为 2 的整数")
    normalized_sizes = [int(value) for value in sizes]
    if len(set(normalized_sizes)) != len(normalized_sizes):
        raise ValueError("window_sizes 不得重复")

    positions = trace.post_round_positions()
    checks: List[Dict[str, Any]] = []
    for window_size in normalized_sizes:
        completed_blocks = len(positions) // window_size
        block_positions = [
            positions[
                block_index * window_size:(block_index + 1) * window_size
            ]
            for block_index in range(completed_blocks)
        ]
        block_summaries = [
            _window_summary(trace, block) for block in block_positions
        ]
        for block_count in range(3, completed_blocks + 1):
            windows = block_positions[block_count - 3:block_count]
            summaries = block_summaries[block_count - 3:block_count]
            terminal_position = windows[-1][-1]
            terminal_observation = trace.observations[terminal_position]
            round_ranges = [
                [
                    int(trace.observations[window[0]]["round_index"]),
                    int(trace.observations[window[-1]]["round_index"]),
                ]
                for window in windows
            ]
            checks.append({
                "completed_block_count": int(block_count),
                "window_size": int(window_size),
                "state_index": int(terminal_observation["state_index"]),
                "round_index": int(terminal_observation["round_index"]),
                "window_round_ranges": round_ranges,
                **_query_max_range_evidence_from_summaries(summaries),
            })
    json.dumps(checks, ensure_ascii=False, allow_nan=False)
    return checks


def replay_stationarity(
    trace: StationarityTrace,
    config: StationarityDetectorConfig,
) -> StationarityReplayResult:
    """Replay the three-window detector without reading generation metadata."""
    trace.validate()
    if not isinstance(config, StationarityDetectorConfig):
        raise ValueError("config 必须是 StationarityDetectorConfig")
    trace_descriptor = {
        "contract_version": trace.contract_version,
        "trace_identity_sha256": _trace_identity_sha256(trace),
        "query_identity_sha256": trace.query_identity_sha256,
        "target_identity_sha256": trace.target_identity_sha256,
        "n_records": trace.n_records,
        "query_count": trace.query_count,
        "state_count": trace.state_count,
        "post_round_count": trace.post_round_count,
        "termination_reason": trace.termination_reason,
    }
    detector_config = config.to_dict()

    def make_result(
        status: str,
        candidate_state_index: int | None,
        candidate_round_index: int | None,
    ) -> StationarityReplayResult:
        return StationarityReplayResult(
            status=status,
            candidate_state_index=candidate_state_index,
            candidate_round_index=candidate_round_index,
            checks=checks,
            trace=trace_descriptor,
            detector_config=detector_config,
        )

    positions = trace.post_round_positions()
    completed_blocks = len(positions) // config.window_size
    checks: List[Dict[str, Any]] = []
    moving_stability_streak = 0
    insufficient_movement_streak = 0

    for block_count in range(3, completed_blocks + 1):
        windows = []
        for block_index in range(block_count - 3, block_count):
            start = block_index * config.window_size
            end = start + config.window_size
            windows.append(positions[start:end])
        evidence = _stability_evidence(trace, windows, config)
        terminal_position = windows[-1][-1]
        terminal_observation = trace.observations[terminal_position]
        if evidence["stable"] and evidence["movement_sufficient"]:
            moving_stability_streak += 1
            insufficient_movement_streak = 0
            check_status = "stability_pass"
        elif evidence["stable"]:
            moving_stability_streak = 0
            insufficient_movement_streak += 1
            check_status = "insufficient_movement"
        else:
            moving_stability_streak = 0
            insufficient_movement_streak = 0
            check_status = "running"
        check = {
            "completed_block_count": int(block_count),
            "window_size": int(config.window_size),
            "state_index": int(terminal_observation["state_index"]),
            "round_index": int(terminal_observation["round_index"]),
            **evidence,
            "moving_stability_streak": int(moving_stability_streak),
            "insufficient_movement_streak": int(
                insufficient_movement_streak
            ),
            "check_status": check_status,
        }
        checks.append(check)
        if moving_stability_streak >= 2:
            return make_result(
                "stationary_qualified",
                check["state_index"],
                check["round_index"],
            )
        if (
            insufficient_movement_streak
            >= config.stall_patience_checks
        ):
            return make_result(
                "stalled",
                check["state_index"],
                check["round_index"],
            )

    if trace.termination_reason in {"max_rounds", "candidate_budget"}:
        status = "horizon_reached"
    elif trace.termination_reason != "in_progress":
        status = "terminated_before_qualification"
    elif completed_blocks < 3:
        status = "collecting"
    elif checks and checks[-1]["check_status"] == "insufficient_movement":
        status = "insufficient_movement"
    else:
        status = "running"
    return make_result(status, None, None)


def replay_query_max_stationarity(
    trace: StationarityTrace,
    config: QueryMaxStationarityDetectorConfig,
) -> StationarityReplayResult:
    """Replay the versioned detector with a worst-query drift guard."""
    trace.validate()
    if not isinstance(config, QueryMaxStationarityDetectorConfig):
        raise ValueError(
            "config 必须是 QueryMaxStationarityDetectorConfig"
        )
    trace_descriptor = {
        "contract_version": trace.contract_version,
        "trace_identity_sha256": _trace_identity_sha256(trace),
        "query_identity_sha256": trace.query_identity_sha256,
        "target_identity_sha256": trace.target_identity_sha256,
        "n_records": trace.n_records,
        "query_count": trace.query_count,
        "state_count": trace.state_count,
        "post_round_count": trace.post_round_count,
        "termination_reason": trace.termination_reason,
    }
    detector_config = config.to_dict()

    def make_result(
        status: str,
        candidate_state_index: int | None,
        candidate_round_index: int | None,
    ) -> StationarityReplayResult:
        return StationarityReplayResult(
            status=status,
            candidate_state_index=candidate_state_index,
            candidate_round_index=candidate_round_index,
            checks=checks,
            trace=trace_descriptor,
            detector_config=detector_config,
            contract_version=(
                STATIONARITY_QUERY_MAX_REPLAY_CONTRACT_VERSION
            ),
        )

    base = config.base_config
    positions = trace.post_round_positions()
    completed_blocks = len(positions) // base.window_size
    checks: List[Dict[str, Any]] = []
    moving_stability_streak = 0
    insufficient_movement_streak = 0

    for block_count in range(3, completed_blocks + 1):
        windows = []
        for block_index in range(block_count - 3, block_count):
            start = block_index * base.window_size
            end = start + base.window_size
            windows.append(positions[start:end])
        evidence = _query_max_stability_evidence(trace, windows, config)
        terminal_position = windows[-1][-1]
        terminal_observation = trace.observations[terminal_position]
        if evidence["stable"] and evidence["movement_sufficient"]:
            moving_stability_streak += 1
            insufficient_movement_streak = 0
            check_status = "stability_pass"
        elif evidence["stable"]:
            moving_stability_streak = 0
            insufficient_movement_streak += 1
            check_status = "insufficient_movement"
        else:
            moving_stability_streak = 0
            insufficient_movement_streak = 0
            check_status = "running"
        check = {
            "completed_block_count": int(block_count),
            "window_size": int(base.window_size),
            "state_index": int(terminal_observation["state_index"]),
            "round_index": int(terminal_observation["round_index"]),
            **evidence,
            "moving_stability_streak": int(moving_stability_streak),
            "insufficient_movement_streak": int(
                insufficient_movement_streak
            ),
            "check_status": check_status,
        }
        checks.append(check)
        if moving_stability_streak >= 2:
            return make_result(
                "stationary_qualified",
                check["state_index"],
                check["round_index"],
            )
        if insufficient_movement_streak >= base.stall_patience_checks:
            return make_result(
                "stalled",
                check["state_index"],
                check["round_index"],
            )

    if trace.termination_reason in {"max_rounds", "candidate_budget"}:
        status = "horizon_reached"
    elif trace.termination_reason != "in_progress":
        status = "terminated_before_qualification"
    elif completed_blocks < 3:
        status = "collecting"
    elif checks and checks[-1]["check_status"] == "insufficient_movement":
        status = "insufficient_movement"
    else:
        status = "running"
    return make_result(status, None, None)
