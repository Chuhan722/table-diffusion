"""Result-blind protocol for Issue #53 Stage 6A overshoot diagnosis."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "issue53-stage6a-overshoot-diagnosis-v1"
PROTOCOL_DOC = Path(
    "docs/设计/"
    "Issue53_Stage6A无门控proposal过冲来源诊断结果前冻结协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "2fcf5060fb21d1d8cff8bd082b5f9c5fa25bbbc1e9f3d842dd8bedc68977cc4a"
)
PROTOCOL_DOC_COMMIT = "825af5f42364f977a58822c342c716e5b043b05d"

# This value is deliberately excluded from frozen_protocol_manifest() to
# avoid a self-reference. It is filled after the manifest is frozen.
FROZEN_PROTOCOL_SHA256 = (
    "66f32914a82a62b56269beb7c95e3cf7f2b662a80da22df8825b525d5b70b638"
)

STAGE5_PROTOCOL_DOC_SHA256 = (
    "4d0ffb8ebf77006becea00849eef452174559aea62faeb86dd70c227e3fc7fab"
)
STAGE5_PROTOCOL_MANIFEST_SHA256 = (
    "1d447be0fb0ce9a2c7707abd2e259ed6ea41edbf3780f30426e320b1bad94f1c"
)
STAGE5_EXECUTION_COMMIT = "a5455ca1574a45acbbbe68abe3100a97e4b976ba"
STAGE5_COLLECTION_SHA256 = (
    "375377849aaec401ec6e2dcd29ed850f180c4204c17ed785f67fba2f8f6c506d"
)
STAGE5_EVALUATION_SHA256 = (
    "8368c58d462a9f4b540f32c0d815faf93c436a7b23fef7fa852cda013a8f94f5"
)
STAGE5_AUDIT_SHA256 = (
    "c14fb651466e641b4bf75d0b5966aed4b4a2dcd8f588da2cf5862aaa035e34bb"
)

DATASET_ORDER = ("test_300x10", "nltcs")
FORMAL_SEEDS = tuple(range(348, 353))
SMOKE_SEED = 9906
EXCLUDED_ISSUE53_SEED_RANGE = (313, 347)
STATE_GROUPS = (
    "initial",
    "work_q25",
    "work_q50",
    "work_q75",
    "terminal",
)
PRIMARY_STATE_GROUPS = (
    "work_q25",
    "work_q50",
    "work_q75",
    "terminal",
)
STATE_TARGET_FRACTIONS = {
    "initial": 0.0,
    "work_q25": 0.25,
    "work_q50": 0.50,
    "work_q75": 0.75,
    "terminal": 1.0,
}

RHO = 0.01
ETA = 0.5
TRAJECTORY_MU = 0.01
COPY_ONLY_MU = 0.0
FULL_PROPOSAL_MU = 0.01
TAU = 2.0
FIXED_ALPHA = 16.0
PATIENCE_TICKS = 6
RESOURCE_CAP = 6000
SMOKE_RESOURCE_CAP = 500
SMOKE_N_RECORDS = 128
SMOKE_PROPOSALS_PER_STATE = 2
MIN_FAILURES_PER_SEED = 10
REQUIRED_SEED_MAJORITY = 4

PROPOSAL_STREAMS = ("donor", "update")
ADDRESS_DOMAIN = 0x53364131
OUTPUT_DIR = Path("outputs/issue53_stage6a_overshoot_diagnosis_v1")
SMOKE_OUTPUT_DIR = Path(
    "outputs/issue53_stage6a_overshoot_diagnosis_smoke_v1"
)

FULL_GAIN_CATEGORIES = (
    "improving",
    "unchanged",
    "exact_balance",
    "direction_failure",
    "curvature_overrun",
)
CURVATURE_ROLE_CATEGORIES = (
    "self_sufficient_overrun",
    "cross_breaks_tie",
    "cross_decisive_overrun",
)
GEOMETRY_LABELS = ("direction_failure", "curvature_overrun")
CROSS_LABELS = ("self_sufficient_overrun", "cross_required_overrun")
MUTATION_FAILURE_LABELS = (
    "mutation_created_failure",
    "copy_already_failed",
)
ALLOWED_OVERALL_RESULTS = (
    "invalid_or_incomplete",
    "insufficient_negative_support",
    "no_shared_failure_geometry",
    "shared_direction_failure",
    "shared_curvature_overrun",
)

DATASETS = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path(
            "configs/test_300x10/measured_50query_30_15_5.json"
        ),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "device": "numpy",
        "query_count": 50,
        "query_order_counts": {2: 30, 3: 15, 4: 5},
        "max_factor_order": 4,
        "proposals_per_state": 200,
        "query_identity_sha256": (
            "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
        ),
        "target_vector_sha256": (
            "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
        ),
        "input_sha256": {
            "schema": (
                "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
            ),
            "queries": (
                "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
            ),
            "marginals": (
                "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
            ),
        },
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        "device": "cuda",
        "query_count": 1001,
        "query_order_counts": {2: 479, 3: 522},
        "max_factor_order": 3,
        "proposals_per_state": 20,
        "query_identity_sha256": (
            "48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429"
        ),
        "target_vector_sha256": (
            "f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d"
        ),
        "input_sha256": {
            "schema": (
                "5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4"
            ),
            "queries": (
                "b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f"
            ),
            "marginals": (
                "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e"
            ),
        },
    },
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and math.isinf(value):
        return "positive_infinity"
    return value


def _require_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是精确整数")
    return int(value)


def _validate_mode_seed(mode: str, seed: int) -> int:
    seed = _require_integer("seed", seed)
    if mode == "formal":
        allowed = FORMAL_SEEDS
    elif mode == "smoke":
        allowed = (SMOKE_SEED,)
    else:
        raise ValueError("mode 必须是 formal 或 smoke")
    if seed not in allowed:
        raise ValueError(f"seed 不在 {mode} 冻结矩阵中：{seed}")
    return seed


def proposals_per_state(dataset: str, *, mode: str = "formal") -> int:
    if dataset not in DATASETS:
        raise ValueError(f"未知 dataset：{dataset!r}")
    if mode == "formal":
        return int(DATASETS[dataset]["proposals_per_state"])
    if mode == "smoke":
        return SMOKE_PROPOSALS_PER_STATE
    raise ValueError("mode 必须是 formal 或 smoke")


def state_id(dataset: str, seed: int, state_group: str, *, mode: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"未知 dataset：{dataset!r}")
    seed = _validate_mode_seed(mode, seed)
    if state_group not in STATE_GROUPS:
        raise ValueError(f"未知 state_group：{state_group!r}")
    return f"{dataset}__seed_{seed}__{state_group}"


def expected_state_ids(*, mode: str = "formal") -> tuple[str, ...]:
    seeds = FORMAL_SEEDS if mode == "formal" else (SMOKE_SEED,)
    if mode not in {"formal", "smoke"}:
        raise ValueError("mode 必须是 formal 或 smoke")
    return tuple(
        state_id(dataset, seed, group, mode=mode)
        for dataset in DATASET_ORDER
        for seed in seeds
        for group in STATE_GROUPS
    )


def proposal_address_seed(
    dataset: str,
    seed: int,
    state_group: str,
    proposal_index: int,
    stream: str,
    *,
    mode: str = "formal",
) -> int:
    """Return one frozen uint64 SeedSequence address."""

    if dataset not in DATASETS:
        raise ValueError(f"未知 dataset：{dataset!r}")
    seed = _validate_mode_seed(mode, seed)
    if state_group not in STATE_GROUPS:
        raise ValueError(f"未知 state_group：{state_group!r}")
    proposal_index = _require_integer("proposal_index", proposal_index)
    limit = proposals_per_state(dataset, mode=mode)
    if not 0 <= proposal_index < limit:
        raise ValueError(
            f"proposal_index 必须在 [0, {limit})：{proposal_index}"
        )
    if stream not in PROPOSAL_STREAMS:
        raise ValueError(f"未知 proposal RNG stream：{stream!r}")
    entropy = [
        ADDRESS_DOMAIN,
        53,
        6,
        1,
        DATASET_ORDER.index(dataset),
        seed,
        STATE_GROUPS.index(state_group),
        proposal_index,
        PROPOSAL_STREAMS.index(stream),
    ]
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def source_generator_params(
    dataset: str,
    seed: int,
    *,
    mode: str = "formal",
) -> dict[str, Any]:
    """Return the exact independent source-trajectory arguments."""

    if dataset not in DATASETS:
        raise ValueError(f"未知 dataset：{dataset!r}")
    seed = _validate_mode_seed(mode, seed)
    formal = mode == "formal"
    cap = RESOURCE_CAP if formal else SMOKE_RESOURCE_CAP
    dataset_config = DATASETS[dataset]
    return {
        "n_records": (
            int(dataset_config["n_records"])
            if formal
            else SMOKE_N_RECORDS
        ),
        "n_rounds": cap,
        "seed": seed,
        "beta": 1.0,
        "h": 0.8,
        "rho": RHO,
        "eta": ETA,
        "mu": TRAJECTORY_MU,
        "tol": float("inf"),
        "device": (
            str(dataset_config["device"]) if formal else "numpy"
        ),
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "marginal",
        "log_every": cap + 1,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": FIXED_ALPHA,
        "alpha_max": FIXED_ALPHA,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": TAU,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": int(
            dataset_config["max_factor_order"]
        ),
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": cap,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": FIXED_ALPHA,
        "record_transition_clocks": True,
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": True,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def classify_full_proposal(b2: int, c2: int) -> str:
    """Classify one full proposal using exact doubled integer units."""

    b2 = _require_integer("B2", b2)
    c2 = _require_integer("C2", c2)
    if c2 < 0:
        raise ValueError("C2 必须非负")
    if c2 == 0 and b2 != 0:
        raise ValueError(
            "C2=0 时 exact query delta 为零，因此 B2 也必须为零"
        )
    g2 = b2 - c2
    if g2 > 0:
        return "improving"
    if g2 == 0:
        return "unchanged" if c2 == 0 else "exact_balance"
    return "direction_failure" if b2 <= 0 else "curvature_overrun"


def classify_curvature_role(
    b2: int,
    cself2: int,
    ccross2: int,
) -> str:
    """Classify whether row cross terms are required for an overrun."""

    b2 = _require_integer("B2", b2)
    cself2 = _require_integer("Cself2", cself2)
    ccross2 = _require_integer("Ccross2", ccross2)
    if cself2 < 0 or cself2 + ccross2 < 0:
        raise ValueError("Cself2 与总 C2 必须非负")
    if classify_full_proposal(b2, cself2 + ccross2) != (
        "curvature_overrun"
    ):
        raise ValueError("cross role 只对 curvature_overrun 定义")
    without_cross_g2 = b2 - cself2
    if without_cross_g2 < 0:
        return "self_sufficient_overrun"
    if without_cross_g2 == 0:
        return "cross_breaks_tie"
    return "cross_decisive_overrun"


def classify_mutation_failure(gcopy2: int, gfull2: int) -> str:
    """Attribute one full negative proposal to copy or mutation."""

    gcopy2 = _require_integer("Gcopy2", gcopy2)
    gfull2 = _require_integer("Gfull2", gfull2)
    if gfull2 >= 0:
        raise ValueError("mutation failure 来源只对 full G2<0 定义")
    if gcopy2 >= 0:
        return "mutation_created_failure"
    return "copy_already_failed"


def classify_mutation_sign_transition(gcopy2: int, gfull2: int) -> str:
    """Return the complete copy/full gain-sign transition."""

    gcopy2 = _require_integer("Gcopy2", gcopy2)
    gfull2 = _require_integer("Gfull2", gfull2)

    def sign(value: int) -> str:
        if value < 0:
            return "negative"
        if value > 0:
            return "positive"
        return "zero"

    return f"{sign(gcopy2)}_to_{sign(gfull2)}"


def exact_doubled_decomposition(
    row_query_deltas: np.ndarray,
    count_residual: np.ndarray,
) -> dict[str, Any]:
    """Compute exact B2/C2/G2 and row self/cross terms in int64."""

    raw_deltas = np.asarray(row_query_deltas)
    raw_residual = np.asarray(count_residual)
    if (
        raw_deltas.ndim != 2
        or raw_deltas.dtype.kind not in "iu"
        or raw_deltas.dtype.kind == "b"
    ):
        raise ValueError("row_query_deltas 必须是整数二维数组")
    if (
        raw_residual.shape != (raw_deltas.shape[1],)
        or raw_residual.dtype.kind not in "iu"
        or raw_residual.dtype.kind == "b"
    ):
        raise ValueError("count_residual 必须是匹配查询数的整数一维数组")
    deltas = raw_deltas.astype(np.int64, copy=False)
    residual = raw_residual.astype(np.int64, copy=False)

    row_count, query_count = deltas.shape
    max_delta = int(np.max(np.abs(deltas))) if deltas.size else 0
    max_residual = int(np.max(np.abs(residual))) if residual.size else 0
    max_delta_q = row_count * max_delta
    int64_max = np.iinfo(np.int64).max
    bounds = {
        "b2": 2 * query_count * max_residual * max_delta_q,
        "c2": query_count * max_delta_q * max_delta_q,
        "cself2": row_count * query_count * max_delta * max_delta,
    }
    if any(value > int64_max for value in bounds.values()):
        raise OverflowError("精确 B2/C2 分解存在 int64 溢出风险")

    delta_q = deltas.sum(axis=0, dtype=np.int64)
    b2 = 2 * int(np.dot(residual, delta_q))
    c2 = int(np.dot(delta_q, delta_q))
    cself2 = int(np.einsum("ij,ij->", deltas, deltas))
    ccross2 = c2 - cself2
    g2 = b2 - c2
    return {
        "delta_q": delta_q,
        "b2": b2,
        "cself2": cself2,
        "ccross2": ccross2,
        "c2": c2,
        "g2": g2,
        "category": classify_full_proposal(b2, c2),
    }


def validate_sequential_identity(
    *,
    bcopy2: int,
    ccopy2: int,
    bmutation_given_copy2: int,
    cmutation2: int,
    bfull2: int,
    cfull2: int,
    copy_mutation_interaction2: int,
) -> dict[str, int]:
    """Fail closed unless copy/mutation/full doubled identities are exact."""

    values = {
        name: _require_integer(name, value)
        for name, value in {
            "bcopy2": bcopy2,
            "ccopy2": ccopy2,
            "bmutation_given_copy2": bmutation_given_copy2,
            "cmutation2": cmutation2,
            "bfull2": bfull2,
            "cfull2": cfull2,
            "copy_mutation_interaction2": (
                copy_mutation_interaction2
            ),
        }.items()
    }
    if values["ccopy2"] < 0 or values["cmutation2"] < 0:
        raise ValueError("copy 与 mutation 自身 C2 必须非负")
    expected_bfull2 = (
        values["bcopy2"]
        + values["bmutation_given_copy2"]
        + values["copy_mutation_interaction2"]
    )
    expected_cfull2 = (
        values["ccopy2"]
        + values["cmutation2"]
        + values["copy_mutation_interaction2"]
    )
    if (
        values["bfull2"] != expected_bfull2
        or values["cfull2"] != expected_cfull2
        or values["cfull2"] < 0
    ):
        raise ValueError("copy/mutation/full 的 B2 或 C2 恒等式失败")
    gcopy2 = values["bcopy2"] - values["ccopy2"]
    gmutation2 = (
        values["bmutation_given_copy2"] - values["cmutation2"]
    )
    gfull2 = values["bfull2"] - values["cfull2"]
    if gfull2 != gcopy2 + gmutation2:
        raise ValueError("Gfull2 != Gcopy2 + Gmutation_given_copy2")
    return {
        "gcopy2": gcopy2,
        "gmutation_given_copy2": gmutation2,
        "gfull2": gfull2,
    }


def classify_dataset_dominance(
    counts_by_seed: Mapping[int, Mapping[str, int]],
    labels: Sequence[str],
    *,
    minimum_per_seed: int = MIN_FAILURES_PER_SEED,
) -> dict[str, Any]:
    """Apply the frozen 4/5 plus equal-seed-share majority rule."""

    labels = tuple(labels)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("dominance axis 必须恰有两个不同 labels")
    if (
        isinstance(minimum_per_seed, bool)
        or not isinstance(minimum_per_seed, Integral)
        or minimum_per_seed < 1
    ):
        raise ValueError("minimum_per_seed 必须是正整数")
    if set(counts_by_seed) != set(FORMAL_SEEDS):
        raise ValueError("counts_by_seed 必须恰好覆盖五个正式 seeds")

    normalized: dict[int, dict[str, int]] = {}
    denominators: dict[int, int] = {}
    seed_labels: dict[int, str | None] = {}
    shares: dict[int, dict[str, Fraction]] = {}
    for seed in FORMAL_SEEDS:
        row = counts_by_seed[seed]
        if set(row) != set(labels):
            raise ValueError(f"seed {seed} 必须恰好覆盖两个 axis labels")
        normalized[seed] = {
            label: _require_integer(f"{seed}/{label}", row[label])
            for label in labels
        }
        if any(value < 0 for value in normalized[seed].values()):
            raise ValueError("dominance counts 必须非负")
        denominator = sum(normalized[seed].values())
        denominators[seed] = denominator
        if denominator == 0:
            shares[seed] = {
                label: Fraction(0, 1) for label in labels
            }
        else:
            shares[seed] = {
                label: Fraction(normalized[seed][label], denominator)
                for label in labels
            }
        left, right = (normalized[seed][label] for label in labels)
        seed_labels[seed] = (
            labels[0]
            if left > right
            else labels[1] if right > left else None
        )

    insufficient = [
        seed
        for seed, denominator in denominators.items()
        if denominator < minimum_per_seed
    ]
    mean_shares = {
        label: sum(
            (shares[seed][label] for seed in FORMAL_SEEDS),
            Fraction(0, 1),
        )
        / len(FORMAL_SEEDS)
        for label in labels
    }
    support_counts = {
        label: sum(seed_labels[seed] == label for seed in FORMAL_SEEDS)
        for label in labels
    }
    selected = None
    if not insufficient:
        candidates = [
            label
            for label in labels
            if support_counts[label] >= REQUIRED_SEED_MAJORITY
            and mean_shares[label] > Fraction(1, 2)
        ]
        if len(candidates) > 1:
            raise AssertionError("互斥二元 axis 不应同时产生两个多数")
        selected = candidates[0] if candidates else None

    return {
        "status": (
            "insufficient_support"
            if insufficient
            else "supported" if selected is not None else "no_stable_label"
        ),
        "label": selected,
        "minimum_per_seed": int(minimum_per_seed),
        "insufficient_seeds": insufficient,
        "denominators": denominators,
        "seed_labels": seed_labels,
        "support_counts": support_counts,
        "mean_shares": {
            label: {
                "numerator": value.numerator,
                "denominator": value.denominator,
                "value": float(value),
            }
            for label, value in mean_shares.items()
        },
    }


def shared_geometry_result(
    dataset_results: Mapping[str, Mapping[str, Any]],
) -> str:
    """Map two frozen dataset dominance results to the overall label."""

    if set(dataset_results) != set(DATASET_ORDER):
        raise ValueError("dataset_results 必须恰好覆盖两个冻结数据集")
    rows = [dataset_results[name] for name in DATASET_ORDER]
    if any(row.get("status") == "insufficient_support" for row in rows):
        return "insufficient_negative_support"
    labels = [
        row.get("label") if row.get("status") == "supported" else None
        for row in rows
    ]
    if labels == ["direction_failure", "direction_failure"]:
        return "shared_direction_failure"
    if labels == ["curvature_overrun", "curvature_overrun"]:
        return "shared_curvature_overrun"
    return "no_shared_failure_geometry"


def verify_input_file_identities(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root)
    audit = {}
    for dataset_name in DATASET_ORDER:
        dataset = DATASETS[dataset_name]
        observed = {
            key: file_sha256(root / dataset[key])
            for key in ("schema", "queries", "marginals")
        }
        if observed != dataset["input_sha256"]:
            raise RuntimeError(f"{dataset_name} 冻结输入 SHA-256 漂移")
        audit[dataset_name] = observed
    return audit


def frozen_protocol_manifest() -> dict[str, Any]:
    source_template = source_generator_params(
        DATASET_ORDER[0], FORMAL_SEEDS[0]
    ).copy()
    for key in (
        "n_records",
        "seed",
        "device",
        "factorized_gibbs_max_order",
    ):
        source_template.pop(key)
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6a_gate_free_proposal_overshoot_diagnosis",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
            "commit": PROTOCOL_DOC_COMMIT,
        },
        "upstream_stage5": {
            "protocol_document_sha256": STAGE5_PROTOCOL_DOC_SHA256,
            "protocol_manifest_sha256": (
                STAGE5_PROTOCOL_MANIFEST_SHA256
            ),
            "execution_commit": STAGE5_EXECUTION_COMMIT,
            "collection_sha256": STAGE5_COLLECTION_SHA256,
            "evaluation_sha256": STAGE5_EVALUATION_SHA256,
            "independent_audit_sha256": STAGE5_AUDIT_SHA256,
            "frozen_result": "no_shared_factor_support",
            "source_kernel": "independent_s0",
        },
        "formal_result_valid": True,
        "formal_requires_separate_user_authorization": True,
        "datasets": {
            name: _jsonable(DATASETS[name]) for name in DATASET_ORDER
        },
        "dataset_order": list(DATASET_ORDER),
        "formal_seeds": list(FORMAL_SEEDS),
        "excluded_issue53_seed_range": list(
            EXCLUDED_ISSUE53_SEED_RANGE
        ),
        "smoke_seed": SMOKE_SEED,
        "state_groups": list(STATE_GROUPS),
        "primary_state_groups": list(PRIMARY_STATE_GROUPS),
        "state_target_fractions": STATE_TARGET_FRACTIONS,
        "state_selection_rule": (
            "initial and terminal exact; choose three distinct recorded "
            "natural-work states minimizing total absolute distance to "
            "25%, 50%, and 75% of terminal work, with earlier-index "
            "lexicographic tie break"
        ),
        "source_trajectory_count": (
            len(DATASET_ORDER) * len(FORMAL_SEEDS)
        ),
        "state_count": len(expected_state_ids()),
        "proposal_pair_count": sum(
            len(FORMAL_SEEDS)
            * len(STATE_GROUPS)
            * proposals_per_state(dataset)
            for dataset in DATASET_ORDER
        ),
        "proposal_table_result_count": 2
        * sum(
            len(FORMAL_SEEDS)
            * len(STATE_GROUPS)
            * proposals_per_state(dataset)
            for dataset in DATASET_ORDER
        ),
        "source_generator_common": _jsonable(source_template),
        "source_runtime": {
            name: {
                "n_records": DATASETS[name]["n_records"],
                "device": DATASETS[name]["device"],
                "factorized_gibbs_max_order": (
                    DATASETS[name]["max_factor_order"]
                ),
            }
            for name in DATASET_ORDER
        },
        "no_gate_contract": {
            "tolerance": "positive_infinity",
            "max_retries": 0,
            "one_finite_proposal_unconditionally_applied": True,
            "terminal_current": True,
            "proposal_result_filtering": False,
            "retry_after_bc": False,
            "rollback": False,
            "best_or_shadow_selection": False,
            "probe_feedback_to_source_trajectory": False,
        },
        "probe": {
            "kernel": "independent_directional_copy",
            "rho": RHO,
            "eta": ETA,
            "tau": TAU,
            "copy_only_mu": COPY_ONLY_MU,
            "full_proposal_mu": FULL_PROPOSAL_MU,
            "rng_address": {
                "method": "numpy_seed_sequence_uint64",
                "domain": ADDRESS_DOMAIN,
                "streams": list(PROPOSAL_STREAMS),
                "copy_full_update_stream_cloned": True,
            },
            "frozen_state_no_closed_loop_feedback": True,
            "retain_every_addressed_pair": True,
        },
        "exact_diagnostics": {
            "integer_unit": "doubled_loss_gain",
            "b2": "2 * residual dot delta_q",
            "c2": "delta_q dot delta_q",
            "g2": "b2 - c2",
            "cself2": "sum_i row_delta_i dot row_delta_i",
            "ccross2": "c2 - cself2",
            "classification_epsilon": 0,
            "full_categories": list(FULL_GAIN_CATEGORIES),
            "curvature_role_categories": list(
                CURVATURE_ROLE_CATEGORIES
            ),
            "sequential_gain_identity": (
                "gfull2 = gcopy2 + gmutation_given_copy2"
            ),
            "quadratic_interaction_identity": (
                "cfull2 = ccopy2 + cmutation2 + "
                "copy_mutation_interaction2"
            ),
        },
        "analysis": {
            "independent_unit": "source_trajectory_seed",
            "minimum_failures_per_seed": MIN_FAILURES_PER_SEED,
            "required_seed_majority": REQUIRED_SEED_MAJORITY,
            "seed_count": len(FORMAL_SEEDS),
            "dataset_support_rule": (
                "all seeds meet minimum; same strict seed majority in at "
                "least 4/5; equal-seed mean share strictly above 0.5"
            ),
            "geometry_labels": list(GEOMETRY_LABELS),
            "cross_labels": list(CROSS_LABELS),
            "mutation_failure_labels": list(
                MUTATION_FAILURE_LABELS
            ),
            "initial_descriptive_only": True,
            "seed_extension_on_insufficient": False,
            "proposal_extension_on_insufficient": False,
            "allowed_overall_results": list(ALLOWED_OVERALL_RESULTS),
        },
        "information_boundary": {
            "generation_reads_measured_workload_only": True,
            "reference_table_forbidden": True,
            "heldout_forbidden": True,
            "offline_safety_forbidden": True,
            "collector_final_classification_forbidden": True,
            "evaluator_after_complete_structural_audit_only": True,
        },
        "smoke": {
            "formal_result_valid": False,
            "seed": SMOKE_SEED,
            "n_records": SMOKE_N_RECORDS,
            "resource_cap": SMOKE_RESOURCE_CAP,
            "proposals_per_state": SMOKE_PROPOSALS_PER_STATE,
        },
        "stage6b": {
            "automatically_authorized": False,
            "formal_seed_reuse_forbidden": True,
            "post_proposal_gate_forbidden": True,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> None:
    root = Path(repository_root)
    observed_doc_sha = file_sha256(root / PROTOCOL_DOC)
    if observed_doc_sha != PROTOCOL_DOC_SHA256:
        raise RuntimeError(
            "Stage 6A protocol document SHA-256 漂移："
            f"{observed_doc_sha}"
        )
    observed_protocol_sha = protocol_sha256()
    if observed_protocol_sha != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "Stage 6A protocol manifest SHA-256 漂移："
            f"{observed_protocol_sha}"
        )
    verify_input_file_identities(root)


def require_formal_confirmation(
    mode: str,
    confirmed_protocol_sha256: str | None,
) -> None:
    """Fail closed before any formal seed is instantiated."""

    if mode == "smoke":
        return
    if mode != "formal":
        raise ValueError("mode 必须是 formal 或 smoke")
    if confirmed_protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError(
            "formal Stage 6A 尚未获单独显式授权；必须传入冻结协议 "
            f"SHA-256：{FROZEN_PROTOCOL_SHA256}"
        )
