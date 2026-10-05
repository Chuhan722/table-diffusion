"""Issue #53 第 6D 阶段两数据三方法正式效果的结果前冻结协议。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import issue53_stage6c_joint_trajectories as joint

PROTOCOL_VERSION = "issue53-stage6d-joint-formal-effect-v5"
PROTOCOL_DOC = Path("docs/设计/Issue53_Stage6D两数据三方法正式闭环效果结果前协议.md")
PROTOCOL_DOC_SHA256 = "842a251ed9c607250da9c17c70c422336ef055601b24dc1e443f24447f324344"

# 清单本身不包含该常量，避免自指。全部源码和文档身份确定后再填入。
FROZEN_PROTOCOL_SHA256 = (
    "835a2165beebe754707a942ae473bb0432b13ec7c914a1caac7b32a6a1a5a523"
)

FORMAL_SEEDS = tuple(range(353, 358))
EXCLUDED_SEED_RANGES = ((338, 347), (348, 352))
EXCLUDED_SMOKE_SEEDS = (9907,)
ROUNDS = 2_500
CANDIDATE_BUDGET = 2_500
CHECKPOINT_ROUNDS = (0, 500, 1_000, 1_500, 2_000, 2_500)
MAX_WORKERS = 2
STABLE_WIN_MINIMUM = 4
LOWER_RISK_RATIO_MAX = 1.05
HIGHER_QUALITY_RATIO_MIN = 0.95

OUTPUT_DIR = Path("outputs/issue53_stage6d_joint_formal_effect_v5")
SHARD_OUTPUT_ROOT = Path("outputs/issue53_stage6d_joint_formal_effect_v5_shards")
COLLECTION_REPORT = "collection_report.json"
SHARD_REPORT = "shard_report.json"
EVALUATION_REPORT = "evaluation_report.json"
L1_RESULTS_CSV = "l1_results.csv"
AUDIT_REPORT = "independent_audit.json"
POSITIVE_INFINITY_MANIFEST_SENTINEL = "positive_infinity"

ARM_GAP = "gap_l1_global_s8"
BASELINE_ARMS = ("factor_b_s8", "independent_b_s0")

EXPECTED_SOFTWARE = {
    "python_major_minor": "3.11",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "torch": "2.13.0+cu130",
    "cuda_runtime": "13.0",
}

LOCAL_SHARD = "local_rtx4090"
A6000_SHARD = "remote_a6000"
SHARD_ORDER = (LOCAL_SHARD, A6000_SHARD)

# 用户在结果前最终确定 21/9。每项是不可拆分的（seed, dataset）三方法配对块。
SHARD_BLOCKS = {
    LOCAL_SHARD: (
        (354, "test_300x10"),
        (355, "test_300x10"),
        (355, "nltcs"),
        (356, "test_300x10"),
        (356, "nltcs"),
        (357, "test_300x10"),
        (357, "nltcs"),
    ),
    A6000_SHARD: (
        (353, "test_300x10"),
        (353, "nltcs"),
        (354, "nltcs"),
    ),
}

EXECUTION_SHARDS = {
    LOCAL_SHARD: {
        "hostname": "linyao-system",
        "task_count": 21,
        "paired_block_count": 7,
        "max_workers": 2,
        "expected_gpu": {
            "physical_index": 1,
            "cuda_visible_devices": "1",
            "uuid": "GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02",
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total_mib": 24564,
            "process_device": "cuda:0",
            "visible_device_count": 1,
        },
    },
    A6000_SHARD: {
        "hostname": "Cardiff_VM_6",
        "task_count": 9,
        "paired_block_count": 3,
        "max_workers": 2,
        "expected_gpu": {
            "physical_index": 0,
            "cuda_visible_devices": "0",
            "uuid": "GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed",
            "name": "NVIDIA RTX A6000",
            "memory_total_mib": 46068,
            "process_device": "cuda:0",
            "visible_device_count": 1,
        },
    },
}

DATASETS: dict[str, dict[str, Any]] = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path("configs/test_300x10/measured_50query_30_15_5.json"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "reference": Path("data/test_300x10/test_300x10.csv"),
        "n_records": 300,
        "query_count": 50,
        "order_counts": {2: 30, 3: 15, 4: 5},
        "max_factor_order": 4,
        "query_identity_sha256": (
            "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
        ),
        "target_vector_sha256": (
            "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
        ),
        "trace_query_identity_sha256": (
            "ff593d1aab304b867358670a46f3907238fc30c6d10c334091e4b09828ee104a"
        ),
        "trace_target_vector_sha256": (
            "33c796bb984c36773e347249c319fb39d421a022cf06f9adefa54406a544e358"
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
            "reference": (
                "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
            ),
        },
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "reference": Path("data/nltcs/nltcs.train.data"),
        "n_records": 16_181,
        "query_count": 1_001,
        "order_counts": {2: 479, 3: 522},
        "max_factor_order": 3,
        "query_identity_sha256": (
            "48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429"
        ),
        "target_vector_sha256": (
            "f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d"
        ),
        "trace_query_identity_sha256": (
            "252ad578c3e0a72477186dff6532007959d4618ef54fa1ab5c86b32ed34f1cf6"
        ),
        "trace_target_vector_sha256": (
            "810c79bc0c259fbb643fa3e4a5a24dc16647630576657206a364c5e5b4d175fd"
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
            "reference": (
                "e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30"
            ),
        },
    },
}

TEST_IDENTITY_ARTIFACT = Path("configs/test_300x10/issue53_query_workload_ab_v1.json")
TEST_IDENTITY_ARTIFACT_SHA256 = (
    "a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36"
)
TEST_GROUP_ORDER = (
    "one_way_safety",
    "common_unseen_2way",
    "fixed_heldout_3way",
    "fixed_heldout_4way",
)
TEST_GROUP_COUNTS = {
    "one_way_safety": 25,
    "common_unseen_2way": 521,
    "fixed_heldout_3way": 512,
    "fixed_heldout_4way": 512,
}
TEST_GROUP_IDENTITIES = {
    "one_way_safety": (
        "b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c"
    ),
    "common_unseen_2way": (
        "fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2"
    ),
    "fixed_heldout_3way": (
        "d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a"
    ),
    "fixed_heldout_4way": (
        "2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171"
    ),
}
NLTCS_GROUP_COUNTS = {
    "one_way_safety": 32,
    "unmeasured_3way": 3_958,
    "all_4way": 29_120,
}
NLTCS_GROUP_IDENTITIES = {
    "one_way_safety": (
        "bbc8fc5d1b1ed0e5cd318a2168fe3887297b1c6aa33634736d0c693e96785c13"
    ),
    "unmeasured_3way": (
        "9c43437d6366e3cce0438fdf79e104d70ebabc112db9236b3feef5220b5eb588"
    ),
    "all_4way": ("1b92f8d80e775cffd637450d3d5015c78d43f7d9a870faf1603c99c88ec5d408"),
}

# 正式协议同时绑定采集、评价、独立复核和它们复用的既有只读评价基础设施。
# 哈希在文件全部实现并审查后填入。
IMPLEMENTATION_SOURCES = {
    "full_generator": {
        "path": Path("src/table_diffevo/evolution.py"),
        "sha256": "cb0cb9fd19f7172c957cbd28543a9a80630d68f3ffe97e4ccc079601a0a8801b",
    },
    "shared_update_plan": {
        "path": Path("src/table_diffevo/update.py"),
        "sha256": "505c6ce582dfbc6cf597305ca54eb384fe9a460e0abc0da5efbf50414fcae541",
    },
    "gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": "8c0fd1f8233edf8c756d9036e0ebce7289f45c5f7c5ece100fd7acc585d83b47",
    },
    "joint_task_matrix": {
        "path": Path("scripts/issue53_stage6c_joint_trajectories.py"),
        "sha256": "37070be03d0b2d8121553f49f01079e37744eb418143f380160b058c9671a194",
    },
    "gpu_runtime_helpers": {
        "path": Path("scripts/run_issue53_stage6c_joint_smoke.py"),
        "sha256": "a676a9f2c29a9481dda66e842abeece9e3a46ad2f67d4bb91ce7df520446a864",
    },
    "parallel_execution": {
        "path": Path("src/table_diffevo/experiment_parallel.py"),
        "sha256": "44d6fd8a56903b2952b78621029126ce07c9e5e2f35ad3101233c5d243bac137",
    },
    "stationarity_trace": {
        "path": Path("src/table_diffevo/stationarity.py"),
        "sha256": "a280c18e630beb8f5342fa17a743b3847426e821cf66b2ca7a4f69a8cb0b2152",
    },
    "result_blind_query_identity": {
        "path": Path("scripts/freeze_issue53_test_query_workload_ab.py"),
        "sha256": "cfaf56569999438798d2673eee3d282814b714b08b7d192c046a443bdfd56410",
    },
    "offline_evaluation_helpers": {
        "path": Path("scripts/evaluate_issue53_fixed_alpha_calibration.py"),
        "sha256": "df41d09ec23e9272af762ae43c4379dda1190f1d5fae2a0ca3700569309afc3e",
    },
    "collector": {
        "path": Path("scripts/run_issue53_stage6d_formal.py"),
        "sha256": "d020648c2ee9c3808e370ddab7fca74a26ddd412ea28c4643e3f9b280071f88b",
    },
    "evaluator": {
        "path": Path("scripts/evaluate_issue53_stage6d_formal.py"),
        "sha256": "4a18e49bc931446a978e1a1e7c3ae4c89684698a9d1782a4885b56e725ebc51b",
    },
    "independent_auditor": {
        "path": Path("scripts/audit_issue53_stage6d_formal.py"),
        "sha256": "901059965dc081e4dc707852411cfb018f475d5fb601e2ae3ffda5e415498205",
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
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return POSITIVE_INFINITY_MANIFEST_SENTINEL if value > 0 else "negative_infinity"
    return value


def common_generator_params() -> dict[str, Any]:
    """返回正式轨迹除数据、方法核和随机种子外的共同参数。"""

    return {
        "n_rounds": ROUNDS,
        "beta": 1.0,
        "h": 0.8,
        "rho": 0.01,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "device": "cuda",
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "marginal",
        "log_every": ROUNDS + 1,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": 16.0,
        "alpha_max": 16.0,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "retry_rho_decay": 0.5,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_logit_clip": 30.0,
        "candidate_budget": CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 16.0,
        "record_transition_clocks": True,
        # 只用于提取预先固定的查询答案检查点，不影响随机流或状态更新。
        "record_stationarity_trace": True,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": None,
    }


def task_generator_params(dataset: str, arm: str, seed: int) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"未知正式数据集：{dataset!r}")
    if arm not in joint.ARM_ORDER:
        raise ValueError(f"未知正式方法：{arm!r}")
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"随机种子不在正式集合：{seed!r}")
    params = common_generator_params()
    params.update(joint.arm_kernel_params(arm))
    params.update(
        {
            "seed": int(seed),
            "n_records": int(DATASETS[dataset]["n_records"]),
            "factorized_gibbs_max_order": int(DATASETS[dataset]["max_factor_order"]),
        }
    )
    return params


def _nonfinite_float_items(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], float]]:
    found: list[tuple[tuple[str, ...], float]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_nonfinite_float_items(item, (*path, str(key))))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_nonfinite_float_items(item, (*path, str(index))))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append((path, value))
    return found


def generator_params_manifest(dataset: str, arm: str, seed: int) -> dict[str, Any]:
    """返回单个正式案例可严格写入 JSON（结构化文本）的运行参数清单。

    运行时 ``tol=+inf`` 是冻结的无门控设置。只有该字段允许使用非有限运行值，
    并在清单中固定表示为 ``positive_infinity（正无穷）``；其他非有限参数必须在
    启动显卡轨迹前失败关闭。
    """

    params = task_generator_params(dataset, arm, seed)
    nonfinite = _nonfinite_float_items(params)
    if (
        len(nonfinite) != 1
        or nonfinite[0][0] != ("tol",)
        or not math.isinf(nonfinite[0][1])
        or nonfinite[0][1] <= 0
    ):
        raise RuntimeError(
            "正式生成参数只允许 tol=+inf，"
            f"得到 {[(path, str(item)) for path, item in nonfinite]}"
        )
    manifest = _jsonable(params)
    if manifest.get("tol") != POSITIVE_INFINITY_MANIFEST_SENTINEL:
        raise RuntimeError("正式无门控 tol 清单编码漂移")
    _strict_json_bytes(manifest)
    return manifest


def generator_params_manifest_matrix() -> dict[str, dict[str, Any]]:
    """返回30条正式任务在运行前必须统一绑定的参数清单。"""

    return {
        task.task_id: generator_params_manifest(task.dataset, task.arm, task.seed)
        for task in task_plan().tasks
    }


def generator_params_manifest_sha256() -> str:
    return canonical_sha256(generator_params_manifest_matrix())


def task_plan() -> joint.JointTrajectoryPlan:
    return joint.build_joint_trajectory_plan(
        seeds=FORMAL_SEEDS,
        rounds_by_dataset={dataset: ROUNDS for dataset in joint.DATASET_ORDER},
    )


def tasks_for_shard(shard_id: str) -> tuple[joint.JointTrajectoryTask, ...]:
    """按全局冻结顺序返回某个执行分片的完整三方法配对任务。"""

    if shard_id not in SHARD_ORDER:
        raise ValueError(f"未知第 6D 执行分片：{shard_id!r}")
    blocks = set(SHARD_BLOCKS[shard_id])
    tasks = tuple(
        task for task in task_plan().tasks if (task.seed, task.dataset) in blocks
    )
    expected = EXECUTION_SHARDS[shard_id]
    if len(tasks) != expected["task_count"]:
        raise RuntimeError(f"{shard_id} 冻结任务数量漂移")
    for seed, dataset in SHARD_BLOCKS[shard_id]:
        paired = [
            task for task in tasks if task.seed == seed and task.dataset == dataset
        ]
        if {task.arm for task in paired} != set(joint.ARM_ORDER):
            raise RuntimeError(f"{shard_id}/{seed}/{dataset} 三方法配对不完整")
    return tasks


def task_shard_id(task: joint.JointTrajectoryTask) -> str:
    matched = [
        shard_id
        for shard_id in SHARD_ORDER
        if (task.seed, task.dataset) in SHARD_BLOCKS[shard_id]
    ]
    if len(matched) != 1:
        raise RuntimeError(f"正式任务没有唯一执行分片：{task.task_id}")
    return matched[0]


def shard_assignment_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_ids: list[str] = []
    all_blocks: list[tuple[int, str]] = []
    for shard_id in SHARD_ORDER:
        tasks = tasks_for_shard(shard_id)
        task_ids = [task.task_id for task in tasks]
        blocks = [
            {"seed": seed, "dataset": dataset}
            for seed, dataset in SHARD_BLOCKS[shard_id]
        ]
        result[shard_id] = {
            "hostname": EXECUTION_SHARDS[shard_id]["hostname"],
            "task_count": len(task_ids),
            "paired_block_count": len(blocks),
            "max_workers": EXECUTION_SHARDS[shard_id]["max_workers"],
            "expected_gpu": dict(EXECUTION_SHARDS[shard_id]["expected_gpu"]),
            "paired_blocks": blocks,
            "task_ids_in_global_order": task_ids,
        }
        all_ids.extend(task_ids)
        all_blocks.extend(SHARD_BLOCKS[shard_id])
    expected_ids = [task.task_id for task in task_plan().tasks]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(expected_ids):
        raise RuntimeError("21/9 分片没有恰好覆盖30条正式任务")
    expected_blocks = {
        (seed, dataset) for seed in FORMAL_SEEDS for dataset in joint.DATASET_ORDER
    }
    if len(all_blocks) != len(set(all_blocks)) or set(all_blocks) != expected_blocks:
        raise RuntimeError("21/9 分片没有恰好覆盖10个三方法配对块")
    return result


def shard_assignment_sha256() -> str:
    return canonical_sha256(shard_assignment_manifest())


def frozen_protocol_manifest() -> dict[str, Any]:
    plan = task_plan()
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6D_joint_full_trajectory_formal_effect",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
        },
        "target_kind": "exact_source_query_counts_not_noisy",
        "task_matrix": {
            "datasets": list(joint.DATASET_ORDER),
            "arms": list(joint.ARM_ORDER),
            "candidate_arm": ARM_GAP,
            "baseline_arms": list(BASELINE_ARMS),
            "seeds": list(FORMAL_SEEDS),
            "excluded_seed_ranges_inclusive": [
                list(pair) for pair in EXCLUDED_SEED_RANGES
            ],
            "excluded_smoke_seeds": list(EXCLUDED_SMOKE_SEEDS),
            "rounds": ROUNDS,
            "candidate_budget": CANDIDATE_BUDGET,
            "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
            "task_order": "seed_then_arm_then_dataset",
            "task_ids": [task.task_id for task in plan.tasks],
            "trajectory_count": len(plan.tasks),
            "all_tasks_declared_before_any_result": True,
            "partial_dataset_or_arm_plan_allowed": False,
            "result_dependent_task_removal_allowed": False,
            "execution_shard_order": list(SHARD_ORDER),
            "execution_shard_assignment": shard_assignment_manifest(),
            "execution_shard_assignment_sha256": shard_assignment_sha256(),
            "all_dataset_seed_triplets_single_shard": True,
        },
        "datasets": {
            name: {
                "schema": str(spec["schema"]),
                "queries": str(spec["queries"]),
                "marginals": str(spec["marginals"]),
                "reference_evaluation_only": str(spec["reference"]),
                "n_records": spec["n_records"],
                "query_count": spec["query_count"],
                "order_counts": spec["order_counts"],
                "max_factor_order": spec["max_factor_order"],
                "query_identity_sha256": spec["query_identity_sha256"],
                "target_vector_sha256": spec["target_vector_sha256"],
                "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
                "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
                "input_sha256": dict(spec["input_sha256"]),
            }
            for name, spec in DATASETS.items()
        },
        "dataset_identity_contract": {
            "result_blind_query_set": (
                "sorted_condition_query_fingerprints_in_vector_order"
            ),
            "result_blind_integer_target": "canonical_json_integer_vector",
            "stationarity_ordered_query": (
                "table_diffevo.stationarity.ordered_query_identity_sha256"
            ),
            "stationarity_float_target": (
                "table_diffevo.stationarity.target_answer_identity_sha256"
            ),
            "all_four_identities_required": True,
            "cross_convention_hash_equality_expected": False,
        },
        "generator_common": _jsonable(common_generator_params()),
        "arm_kernel_params": {
            arm: _jsonable(joint.arm_kernel_params(arm)) for arm in joint.ARM_ORDER
        },
        "no_gate_contract": {
            "tol_positive_infinity": True,
            "tol_manifest_representation": POSITIVE_INFINITY_MANIFEST_SENTINEL,
            "only_tol_may_be_nonfinite_in_runtime_params": True,
            "max_retries": 0,
            "one_candidate_per_round": True,
            "accept_or_reject": False,
            "rollback": False,
            "winner_selection": False,
            "terminal_current_output": True,
            "exact_zero_pre_donor_stop_preserved": True,
            "participant_bias_added": False,
            "donor_before_participant_order_changed": False,
            "gap_microsteps_identity": "8*K",
        },
        "primary_quality": {
            "terminal_identity": "last_actual_current_table",
            "integer_metric": "sum_j_abs_target_minus_answer",
            "normalized_l1_formula": "sum_abs_error/(query_count*n_records)",
            "gap_e_formula": "mean(abs_error/max(target,8))",
            "historical_best_allowed": False,
            "checkpoint_selection_allowed": False,
            "stable_win_minimum": STABLE_WIN_MINIMUM,
            "paired_seed_count": len(FORMAL_SEEDS),
            "must_beat_each_baseline_separately": True,
            "must_pass_on_each_dataset_separately": True,
            "cross_dataset_weighted_score_allowed": False,
        },
        "l1_output": {
            "evaluation_json": EVALUATION_REPORT,
            "long_form_csv": L1_RESULTS_CSV,
            "per_case_terminal": True,
            "per_seed_pairwise_differences": True,
            "five_seed_summaries": True,
            "fixed_checkpoint_curve": list(CHECKPOINT_ROUNDS),
        },
        "offline_evaluation": {
            "generation_may_read_reference": False,
            "evaluation_reads_reference_after_complete_collection": True,
            "test_identity_artifact": {
                "path": str(TEST_IDENTITY_ARTIFACT),
                "sha256": TEST_IDENTITY_ARTIFACT_SHA256,
            },
            "test_groups": {
                "order": list(TEST_GROUP_ORDER),
                "counts": dict(TEST_GROUP_COUNTS),
                "identity_sha256": dict(TEST_GROUP_IDENTITIES),
            },
            "nltcs_groups": {
                "counts": dict(NLTCS_GROUP_COUNTS),
                "identity_sha256": dict(NLTCS_GROUP_IDENTITIES),
            },
            "safety_aggregation": "five_seed_arithmetic_mean_ratio",
            "lower_is_better_ratio_max": LOWER_RISK_RATIO_MAX,
            "higher_is_better_ratio_min": HIGHER_QUALITY_RATIO_MIN,
            "valid_row_rate_required": 1.0,
        },
        "implementation_sources": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in IMPLEMENTATION_SOURCES.items()
        },
        "information_flow": {
            "collection_emits_partial_comparison": False,
            "collection_reads_raw_reference": False,
            "evaluation_requires_complete_collection_sha_confirmation": True,
            "auditor_recomputes_from_terminal_tables_and_checkpoint_vectors": True,
            "auditor_reruns_generation": False,
            "post_result_threshold_retuning_allowed": False,
        },
        "execution": {
            "output_dir": str(OUTPUT_DIR),
            "shard_output_root": str(SHARD_OUTPUT_ROOT),
            "shard_report": SHARD_REPORT,
            "shard_count": len(SHARD_ORDER),
            "local_task_count": EXECUTION_SHARDS[LOCAL_SHARD]["task_count"],
            "a6000_task_count": EXECUTION_SHARDS[A6000_SHARD]["task_count"],
            "expected_software": dict(EXPECTED_SOFTWARE),
            "multiprocessing_start_method": "spawn",
            "generator_params_manifest_sha256": (generator_params_manifest_sha256()),
            "all_generator_manifests_preflighted_before_gpu": True,
            "completed_case_directories_resumed_without_rerun": True,
            "stale_temporary_case_directories_auto_deleted": False,
            "task_reordering": False,
            "adaptive_concurrency": False,
            "automatic_cpu_fallback": False,
            "each_dataset_seed_triplet_runs_on_one_gpu": True,
            "gpu_monitor_physical_index_from_frozen_shard": True,
            "shard_assignment_result_dependent": False,
            "merge_requires_both_complete_shards": True,
            "partial_shard_comparison_allowed": False,
            "exclusive_idle_gpu_required": True,
            "formal_output_overwrite_allowed": False,
        },
        "authorization_boundary": {
            "protocol_frozen": True,
            "collector_evaluator_auditor_wired": True,
            "formal_collection_authorized_at_freeze": False,
            "formal_evaluation_requires_collection_confirmation": True,
            "public_default_kernel_changed": False,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> str:
    root = Path(repository_root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("第 6D 正式协议文档 SHA-256 漂移")
    for name, binding in IMPLEMENTATION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"第 6D 正式实现源码漂移：{name}")
    if file_sha256(root / TEST_IDENTITY_ARTIFACT) != TEST_IDENTITY_ARTIFACT_SHA256:
        raise RuntimeError("第 6D test 离线查询身份文件 SHA-256 漂移")
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "第 6D 正式协议清单漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan(repository_root: str | Path) -> dict[str, Any]:
    protocol_sha = assert_frozen_protocol_identity(repository_root)
    plan = task_plan()
    return {
        "mode": "plan_only_no_dataset_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "task_ids": [task.task_id for task in plan.tasks],
        "trajectory_count": len(plan.tasks),
        "seeds": list(FORMAL_SEEDS),
        "rounds": ROUNDS,
        "checkpoint_rounds": list(CHECKPOINT_ROUNDS),
        "max_workers": MAX_WORKERS,
        "output_dir": str(OUTPUT_DIR),
        "shard_output_root": str(SHARD_OUTPUT_ROOT),
        "shard_assignment_sha256": shard_assignment_sha256(),
        "shards": shard_assignment_manifest(),
        "l1_results_file": L1_RESULTS_CSV,
        "generation_started": False,
        "formal_collection_authorized": False,
    }


def require_run_confirmation(confirmed_protocol_sha256: str | None) -> None:
    if confirmed_protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError("正式采集需要用户另行授权并确认完整协议 SHA-256")


def require_collection_confirmation(confirmed_sha256: str | None) -> None:
    if not isinstance(confirmed_sha256, str) or len(confirmed_sha256) != 64:
        raise PermissionError("正式评价需要显式确认完整 collection 报告 SHA-256")
