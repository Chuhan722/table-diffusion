"""Issue #53 A/R 相对初始进度第 232 轮 logit 截断短诊断协议。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import (
    issue53_gap_weight_dual_ar_progress_screen_execution_protocol as source,
)


PROTOCOL_VERSION = "issue53-gap-weight-dual-ar-progress-clip-diagnostic-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_BC问题一AR相对初始进度logit截断短诊断协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "bb779b3cb02845c66658ee71cf66564ed0c356676d98acb91fba22c1af96f99b"
)
FROZEN_PROTOCOL_SHA256 = (
    "acaaa1378b976db30b445df7024d75c73c5383d0d7097d5fe969fb2f71215da5"
)

SOURCE_EXECUTION_COMMIT = "b3c0df95f7d735cabca0fadf543f460ae3990269"
SOURCE_EXECUTION_PROTOCOL_SHA256 = (
    "56f852fb925f2ec2785a3a7f56593a51f73d4b39a1649db82725cb1076647685"
)
SOURCE_SCIENTIFIC_PROTOCOL_SHA256 = (
    "8c90f51b09179c738f6ea5a45110873844061e2ba1adc4b5a253ac6875f87e46"
)
SOURCE_GENERATOR_PARAMS_SHA256 = (
    "980298ca6db5e08f5f49a995aaadd89b913d580bdea4263f472d954da210c6df"
)

DATASET = "nltcs"
ARM = source.ARM_GAP
SEED = source.DEVELOPMENT_SEED
DIAGNOSTIC_ROUNDS = 232
PREFAIL_ZERO_CLIP_ROUNDS = 231
OUTPUT_DIR = Path(
    "outputs/issue53_gap_weight_dual_ar_progress_clip_diagnostic_nltcs_seed9908_v1"
)
REPORT = "diagnostic_report.json"

EXPECTED_INITIAL_TABLE_SHA256 = (
    "173c0ad24e9e1b3ec4e30bc27c5325c34a664968f366254a84e9c14665cbd40b"
)
EXPECTED_PRIMARY_RNG_POST_INITIALIZATION_SHA256 = (
    "cdf19822cb312a91493ba3540ace38a938c8ea3fd2bc64bbc225707e056124da"
)

FAILED_STAGING_ARTIFACTS = {
    "staging_manifest": {
        "path": Path(
            "outputs/issue53_gap_weight_dual_ar_progress_max_screen_seed9908_v2_shards/"
            ".local_rtx4090.staging-g0u4tj4p/staging_manifest.json"
        ),
        "sha256": (
            "165d3dc1974c6fb4c18296bd49b8731722dd772e4435e16b2268bb1b19d92cd6"
        ),
    },
    "test_case_manifest": {
        "path": Path(
            "outputs/issue53_gap_weight_dual_ar_progress_max_screen_seed9908_v2_shards/"
            ".local_rtx4090.staging-g0u4tj4p/cases/"
            "seed_9908__gap_dual_abs_relative_progress_max_s8__test_300x10/"
            "case_manifest.json"
        ),
        "sha256": (
            "732b8693a0ee36a3ecc7490b54ee4edaaf5fe8f2a761ac6d20fa857492c15fe2"
        ),
    },
    "test_checkpoint_answers": {
        "path": Path(
            "outputs/issue53_gap_weight_dual_ar_progress_max_screen_seed9908_v2_shards/"
            ".local_rtx4090.staging-g0u4tj4p/cases/"
            "seed_9908__gap_dual_abs_relative_progress_max_s8__test_300x10/"
            "checkpoint_query_answers.json"
        ),
        "sha256": (
            "bfd5feb43e117337b27a69480703df9f257f76eb96c07eaa6279c4dce926013e"
        ),
    },
    "test_terminal_current": {
        "path": Path(
            "outputs/issue53_gap_weight_dual_ar_progress_max_screen_seed9908_v2_shards/"
            ".local_rtx4090.staging-g0u4tj4p/cases/"
            "seed_9908__gap_dual_abs_relative_progress_max_s8__test_300x10/"
            "terminal_current.csv"
        ),
        "sha256": (
            "86573ffa1df5d3aadcace1a4039220c06d92225ea0fd7e5d0387695ad4dfe017"
        ),
    },
    "test_transition_audit": {
        "path": Path(
            "outputs/issue53_gap_weight_dual_ar_progress_max_screen_seed9908_v2_shards/"
            ".local_rtx4090.staging-g0u4tj4p/cases/"
            "seed_9908__gap_dual_abs_relative_progress_max_s8__test_300x10/"
            "transition_audit.json"
        ),
        "sha256": (
            "2c95ee3ce0ccc686248419eda961278422f11e14046c1482f669693caeebb994"
        ),
    },
}

IMPLEMENTATION_SOURCES = {
    "source_scientific_protocol": {
        "path": Path(
            "scripts/issue53_gap_weight_dual_ar_progress_screen_protocol.py"
        ),
        "sha256": (
            "9da9ce118b4d455c7d53b1941638f2fdbd0dac948bbb3f5a6b465aa96b778f3b"
        ),
    },
    "source_execution_protocol": {
        "path": Path(
            "scripts/issue53_gap_weight_dual_ar_progress_screen_execution_protocol.py"
        ),
        "sha256": (
            "607618edae050f2b788c74bde57922c194672670ce1f5556e061424d62721fd5"
        ),
    },
    "source_collector": {
        "path": Path(
            "scripts/run_issue53_gap_weight_dual_ar_progress_screen.py"
        ),
        "sha256": (
            "48955bf678ac19d3a0be293321ba5fc77462321175a9f467ec9335e2ef9d86d1"
        ),
    },
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": (
            "7fffd0330bf7c00dfb352112b4d15675d06e9942ccc80a8895947197fdf600ee"
        ),
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": (
            "31237867ee9d6b7aeac810e94e4912c84eda29a79a443e1b863e2d04c269f590"
        ),
    },
    "diagnostic_runner": {
        "path": Path(
            "scripts/diagnose_issue53_gap_weight_dual_ar_progress_clip.py"
        ),
        "sha256": (
            "7e665b533217895957aa97325b98b43710c4a59e1a8d3ea017cf7fb36b5b0eaa"
        ),
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


def strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    return source.file_sha256(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return value


def generator_params() -> dict[str, Any]:
    params = source.task_generator_params(DATASET, ARM, SEED)
    params.update({
        "n_rounds": DIAGNOSTIC_ROUNDS,
        "candidate_budget": DIAGNOSTIC_ROUNDS,
    })
    return params


def generator_params_manifest() -> dict[str, Any]:
    manifest = _jsonable(generator_params())
    _strict_json_bytes(manifest)
    return manifest


def frozen_protocol_manifest() -> dict[str, Any]:
    source_params = source.generator_params_manifest(DATASET, ARM, SEED)
    diagnostic_params = generator_params_manifest()
    changed = {
        key: {"source": source_params[key], "diagnostic": diagnostic_params[key]}
        for key in source_params
        if source_params[key] != diagnostic_params[key]
    }
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "dual_ar_progress_round_232_clip_mechanism_diagnostic",
        "document": {"path": str(PROTOCOL_DOC), "sha256": PROTOCOL_DOC_SHA256},
        "source_failed_execution": {
            "execution_commit": SOURCE_EXECUTION_COMMIT,
            "execution_protocol_sha256": SOURCE_EXECUTION_PROTOCOL_SHA256,
            "scientific_protocol_sha256": SOURCE_SCIENTIFIC_PROTOCOL_SHA256,
            "generator_params_manifest_sha256": SOURCE_GENERATOR_PARAMS_SHA256,
            "first_failed_guard_round": DIAGNOSTIC_ROUNDS,
            "failure_class_deduced_from_return_contract": "gap_logit_clip_hit",
            "complete_collection_published": False,
            "quality_accessed": False,
        },
        "task": {
            "dataset": DATASET,
            "arm": ARM,
            "seed": SEED,
            "rounds": DIAGNOSTIC_ROUNDS,
            "prefail_zero_clip_rounds": PREFAIL_ZERO_CLIP_ROUNDS,
            "generator_params": diagnostic_params,
            "only_generator_param_changes": changed,
        },
        "instrumentation": {
            "production_generator_modified": False,
            "scoped_runtime_wrappers": [
                "_evolve_step_gap_l1_global_cuda",
                "_record_channel_dominance",
                "_build_scan_diagnostics",
            ],
            "original_functions_called_exactly_once": True,
            "random_state_modified": False,
            "returned_values_modified": False,
            "capture_round": DIAGNOSTIC_ROUNDS,
            "coordinate_and_roll_tape_replayed_from_copied_rng_state": True,
        },
        "hard_checks": {
            "applied_rounds_exact": DIAGNOSTIC_ROUNDS,
            "unique_unconditional_candidate_each_round": True,
            "microsteps_equal_8k_each_round": True,
            "nonfinite_conditions_total": 0,
            "exact_zero_or_one_probabilities_total": 0,
            "clip_hits_rounds_1_through_231": 0,
            "clip_hits_round_232_strictly_positive": True,
            "clipped_step_vector_count_matches_round_count": True,
            "abs_normalized_score_strictly_above": 15.0,
            "initial_table_sha256": EXPECTED_INITIAL_TABLE_SHA256,
            "primary_rng_post_initialization_sha256": (
                EXPECTED_PRIMARY_RNG_POST_INITIALIZATION_SHA256
            ),
            "initial_channel_reference": (
                source.EXPECTED_INITIAL_CHANNEL_REFERENCES[DATASET]
            ),
        },
        "failed_staging_artifacts": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in FAILED_STAGING_ARTIFACTS.items()
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "output": {
            "directory": str(OUTPUT_DIR),
            "report": REPORT,
            "atomic_new_only": True,
            "terminal_table_persisted": False,
            "checkpoint_answers_persisted": False,
        },
        "information_boundary": {
            "raw_reference_access_allowed": False,
            "quality_metric_computation_allowed": False,
            "unseen_query_evaluation_allowed": False,
            "method_comparison_allowed": False,
            "parameter_retuning_allowed": False,
            "automatic_full_rerun_allowed": False,
            "automatic_quality_evaluation_allowed": False,
        },
        "authorization": {
            "explicit_user_continue_received_after_failure": True,
            "matching_protocol_sha_required": True,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root).resolve()
    source.assert_frozen_protocol_identity(root)
    if source.FROZEN_PROTOCOL_SHA256 != SOURCE_EXECUTION_PROTOCOL_SHA256:
        raise RuntimeError("短诊断继承的 v2 执行协议身份漂移")
    if source.SCIENTIFIC_PROTOCOL_SHA256 != SOURCE_SCIENTIFIC_PROTOCOL_SHA256:
        raise RuntimeError("短诊断继承的 v2 科学协议身份漂移")
    if source.generator_params_manifest_sha256() != SOURCE_GENERATOR_PARAMS_SHA256:
        raise RuntimeError("短诊断继承的 v2 生成参数矩阵漂移")
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("短诊断协议文档 SHA-256 漂移")
    for name, item in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"短诊断实现源码漂移：{name}")
    for name, item in FAILED_STAGING_ARTIFACTS.items():
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"v2 失败现场漂移：{name}")
    changed = frozen_protocol_manifest()["task"]["only_generator_param_changes"]
    if set(changed) != {"n_rounds", "candidate_budget"}:
        raise RuntimeError("短诊断生成参数不只改变两个资源上限")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "短诊断冻结清单漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def require_confirmation(value: str | None) -> None:
    if value != FROZEN_PROTOCOL_SHA256:
        raise PermissionError("短诊断运行必须精确确认冻结协议 SHA-256")


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_generator_or_quality_access",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol": frozen_protocol_manifest(),
        "generation_started": False,
        "raw_reference_data_accessed": False,
        "quality_metrics_computed": False,
    }
