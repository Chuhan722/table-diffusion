#!/usr/bin/env python
"""残差信号几何正式实验（Issue #57 预注册协议）。

诊断背景（dev，见 docs/设计/残差信号几何_绝对与相对适应度口径.md）：
绝对残差口径把大计数查询压到抽样涨落之下（bootstrap iid 参考线
0.001598±0.000174，平台 0.000864-0.0011 已低于该线），而对稀有模式
查询系统性无力（p<0.05 查询占平台残差 L1 44.8%，bootstrap 对照
22.3%）。相对残差近似 KL 梯度口径，dev 三种子（42..44）配对改善
-70.5%。

五臂（唯一变量 = residual_geometry 及 floor；其余共享 #48 v3 冻结的
no_gate_si 配置：tol=inf、si α≡16、min_spread=1e-3、ds=2.0、rho=0.01、
marginal 初始化）：
- absolute：现状口径（baseline）；
- relative_f8：相对残差 floor=8（主 candidate，dev 定标）；
- relative_f1 / relative_f4 / relative_f16：floor 敏感性次要臂。

判定（运行前冻结，主判定数据集 nltcs，最终表 measured L1 五种子）：
1. 主判定：relative_f8 配对 5/5 种子低于 absolute 且均值改善 ≥30%
   → supports_relative_geometry；3-4/5 或均值改善 <30% → mixed；
   否则 not_supported。
2. floor 稳健性（观察，不进主判定）：若 f8 不是四个 floor 臂中均值
   最优，结果标注 floor_suboptimal 提示（供后续定标，不改变主分类）。
3. 质量风险带：train 未测量 3/4-way L1 与分箱 TVD 相对 absolute 劣化
   >5% 报警；报警时 supports 降级为 supports_with_quality_risk。
4. test_300x10 辅助独立判定（冒烟一致性），不合并主结论。

种子 200..204（与 dev 42..44 不重叠）；2000 轮固定预算（同轮数口径，
不含墙钟等价声明）；生成只读公开输入，参考表仅在全部生成完成后离线
读取。失败结果全部保留。
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from scripts import compare_factorized_gibbs_closed_loop as offline_helpers
else:
    import compare_factorized_gibbs_closed_loop as offline_helpers
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

FORMAL_SEEDS = [200, 201, 202, 203, 204]
FORMAL_ROUNDS = 2000
FROZEN_SI_ALPHA = 16.0   # #48 v3 冻结
FROZEN_MIN_SPREAD = 1e-3  # #48 v3 冻结
PRIMARY_CANDIDATE = "relative_f8"
PRIMARY_BASELINE = "absolute"
PRIMARY_MIN_WINS = 5
PRIMARY_MIN_IMPROVEMENT = 0.30  # 均值改善 ≥30%
MIXED_MIN_WINS = 3
QUALITY_RISK_REL = 0.05

DATASETS = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path("configs/test_300x10/measured_50query.json"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "device": "numpy",
        "references": {"reference": Path("data/test_300x10/test_300x10.csv")},
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        "device": "cuda",
        # 一次实验一份源数据：queries/marginals/target/n_records 全部
        # 来自 train，离线参考因此只用 train。
        "references": {
            "train": Path("data/nltcs/nltcs.train.data"),
        },
    },
}

SHARED_PARAMS = dict(
    rho=0.01, beta=1.0, h=0.8, eta=0.5, mu=0.01, lambda_param=0.5, delta=0.05,
    winsorize_quantiles=(0.01, 0.99), distance_mode="geometric",
    init_method="marginal", residual_directed_diffusion=True,
    diffusion_direction_strength=2.0,
    diffusion_direction_normalization="initial_rms",
    exclude_self=True, tol=float("inf"),
    selection_scale_invariant=True,
    selection_scale_invariant_min_spread=FROZEN_MIN_SPREAD,
    alpha_min=FROZEN_SI_ALPHA, alpha_max=FROZEN_SI_ALPHA,
)

ARMS = {
    "absolute": dict(residual_geometry="absolute"),
    "relative_f8": dict(
        residual_geometry="relative", residual_geometry_floor=8.0,
    ),
    "relative_f1": dict(
        residual_geometry="relative", residual_geometry_floor=1.0,
    ),
    "relative_f4": dict(
        residual_geometry="relative", residual_geometry_floor=4.0,
    ),
    "relative_f16": dict(
        residual_geometry="relative", residual_geometry_floor=16.0,
    ),
}

OUTPUT_PATH = Path(
    "outputs/residual_geometry/"
    f"formal_residual_geometry_{len(FORMAL_SEEDS)}seed_{FORMAL_ROUNDS}round"
    ".json"
)

# 冻结的离线参考表身份（fail-closed，Issue #60）：离线评价阶段读取的
# 参考文件必须精确匹配，不符即协议违规终止（参考表仍只在该数据集全部
# 生成结束后读取，隐私边界不变）。
EXPECTED_REFERENCE_SHA256 = {
    "test_300x10": {
        "reference": (
            "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
        ),
    },
    "nltcs": {
        "train": (
            "e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30"
        ),
    },
}


def _protocol_identity():
    """协议身份的 canonical 表示（Issue #60：协议漂移必须显式可见）。

    覆盖：种子/轮数/全部臂与共享参数/判定阈值/冻结输入与参考哈希。
    任何字段变化都会改变 protocol_sha256，正式运行要求与
    FROZEN_PROTOCOL_SHA256 一致——修改协议必须同步更新该常量，
    等价于显式重新预注册。
    """
    def _canon(value):
        if isinstance(value, dict):
            return {str(k): _canon(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [_canon(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and value == float("inf"):
            return "inf"
        return value

    return {
        "issue": 57,
        "primary_dataset": "nltcs",
        "seeds": FORMAL_SEEDS,
        "rounds": FORMAL_ROUNDS,
        "frozen_si_alpha": FROZEN_SI_ALPHA,
        "frozen_min_spread": FROZEN_MIN_SPREAD,
        "arms": _canon(ARMS),
        "shared_params": _canon(SHARED_PARAMS),
        "datasets": _canon({
            name: {
                key: value for key, value in spec.items()
                if key != "device"  # 设备不属于协议身份
            }
            for name, spec in DATASETS.items()
        }),
        "judgement_thresholds": {
            "primary_baseline": PRIMARY_BASELINE,
            "primary_candidate": PRIMARY_CANDIDATE,
            "primary_min_wins": PRIMARY_MIN_WINS,
            "primary_min_improvement": PRIMARY_MIN_IMPROVEMENT,
            "mixed_min_wins": MIXED_MIN_WINS,
            "quality_risk_rel": QUALITY_RISK_REL,
        },
        "expected_input_sha256": _canon(EXPECTED_INPUT_SHA256),
        "expected_reference_sha256": _canon(EXPECTED_REFERENCE_SHA256),
    }


def canonical_protocol_manifest():
    """唯一的规范协议清单（PR #62 五轮审查不变量 2）。

    生成端把这份清单原样写入正式产物的 ``protocol`` 字段；审计端从
    冻结代码重建同一清单后做整份 canonical 精确比较（缺字段、多字段
    或任一字段不同都拒绝），不再依靠分散的逐字段检查。JSON 序列化
    往返以保证与产物读回值可直接比较。
    """
    return json.loads(json.dumps(
        _protocol_identity(), ensure_ascii=False, sort_keys=True,
    ))


def protocol_sha256():
    """从入库代码独立复算的冻结协议常量 SHA-256。

    边界（有意）：只覆盖协议常量（种子/轮数/臂/共享参数/判定阈值/
    冻结输入与参考哈希），不覆盖判定函数实现本身（其正确性由脚本级
    测试锚定）也不含 device（运行环境单独记录于 provenance.environment）。
    """
    payload = json.dumps(
        _protocol_identity(), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 冻结协议身份（运行时 fail-closed 对拍；协议修改必须同步更新 = 显式
# 重新预注册）。数值在实现后由 protocol_sha256() 一次性生成。
FROZEN_PROTOCOL_SHA256 = (
    "71f6cac87d1f99c7eb8afc6b72f4b36c19a246fef98a816052dbfaee622ce73a"
)

# 产物结构版本（PR #62 五轮不变量 1）。
ARTIFACT_SCHEMA_VERSION = "residual-geometry-formal-artifact-v2"

# 已知 legacy 正式产物白名单（PR #62 二轮审查）：protocol_sha256 机制
# 引入前生成的正式 JSON，按整体文件字节 SHA-256 登记；审计器只接受
# 名单内的缺字段文件，其余一律 fail-closed 拒绝。
# 51aff541…：首个正式实验产物（commit aac1aff，五种子 supports，
# 已归档 docs/实验结果/，PR #59 审查复核过）。
KNOWN_LEGACY_ARTIFACT_SHA256 = {
    "51aff5414eb15c9cfdda496dc1549c6fba7216043159bd377be429fb11443f64",
}

# ---- 第六轮最终合同：正式产物冻结结构（唯一生产验证入口的合同常量）----
# 任何字段集合的增删都属于产物结构变更，必须升级
# ARTIFACT_SCHEMA_VERSION 并显式过 review，等价于重新预注册产物格式。

FORMAL_TOP_LEVEL_FIELDS = frozenset({
    "artifact_schema_version", "protocol", "run_config", "provenance",
    "datasets",
})
FORMAL_AUDIT_SECTION_FIELDS = frozenset({"source_sha256", "source_kind"})
FORMAL_AUDIT_SOURCE_KINDS = frozenset({"v2", "legacy_migrated"})
FORMAL_RUN_CONFIG_FIELDS = frozenset({"seeds", "rounds", "datasets"})
FORMAL_PROVENANCE_FIELDS = frozenset({
    "git_commit", "git_dirty", "protocol_sha256", "protocol_match",
    "formal", "started_at", "finished_at", "environment", "input_sha256",
    "input_hash_mismatches", "command",
})
FORMAL_DATASET_FIELDS = frozenset({
    "initial_state", "reference_sha256", "runs", "judgment",
})
FORMAL_INITIAL_STATE_FIELDS = frozenset({
    "measured_l1_mean", "measured_l1_by_seed", "loss_by_seed", "note",
})
FORMAL_RUN_RECORD_FIELDS = frozenset({
    "dataset", "arm", "seed", "rounds_run", "candidate_evaluations",
    "pre_final_proposal_loss", "final_loss", "final_table_measured_l1",
    "best_loss", "rare_query_mean_abs_residual",
    "common_query_mean_abs_residual", "exact_match_queries",
    "row_max_prob_mean_final", "effective_donors_mean_final",
    "tail_mean_pre_proposal_loss", "final_table_sha256", "elapsed_sec",
    "offline",
})
# 生成端允许为 None 的诊断字段（无稀有查询/无历史时）；其余数值必需有限。
FORMAL_RUN_NULLABLE_FIELDS = frozenset({
    "rare_query_mean_abs_residual", "common_query_mean_abs_residual",
    "row_max_prob_mean_final", "effective_donors_mean_final",
})
FORMAL_RUN_FLOAT_FIELDS = frozenset({
    "pre_final_proposal_loss", "final_loss", "final_table_measured_l1",
    "best_loss", "tail_mean_pre_proposal_loss", "elapsed_sec",
})
FORMAL_OFFLINE_FLOAT_FIELDS = frozenset({
    "unmeasured_3way_l1", "unmeasured_4way_l1", "raw_joint_tvd",
    "binned_joint_tvd",
})
FORMAL_OFFLINE_INT_FIELDS = frozenset({
    "raw_unique_states", "raw_support_overlap",
})
FORMAL_OFFLINE_METRIC_FIELDS = (
    FORMAL_OFFLINE_FLOAT_FIELDS | FORMAL_OFFLINE_INT_FIELDS
)
# initial_state 重算对拍公差（同机确定性重算，历史锚定 rtol=1e-12）。
INITIAL_STATE_RTOL = 1e-12
INITIAL_STATE_ATOL = 1e-15


class FormalArtifactError(ValueError):
    """正式产物违反冻结结构合同（第六轮统一验证入口）。"""


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _is_finite_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def _is_strict_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, bool
    )


def _is_sha256_hex(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def recompute_initial_state(ds_name):
    """按冻结协议重算 n_rounds=0 初始状态（种子相关，可独立复现）。

    固定使用 FORMAL_SEEDS 与协议共享参数——不信任产物自报的种子
    （PR #62 三轮意见 1）。返回结构与生成端 initial_state 一致。
    """
    spec = DATASETS[ds_name]
    schema = load_schema(str(spec["schema"]))
    queries = load_queries(str(spec["queries"]))
    marginals = load_marginals(str(spec["marginals"]))
    target = np.asarray([q["result"] for q in queries], dtype=float)
    first_arm = next(iter(ARMS.values()))
    l1_by_seed = {}
    loss_by_seed = {}
    for seed in FORMAL_SEEDS:
        _, diag0 = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=spec["n_records"], n_rounds=0, seed=seed,
            marginals=marginals, log_every=-1, device=spec["device"],
            return_final_table=True, **{**SHARED_PARAMS, **first_arm},
        )
        table0 = diag0.pop("final_table")
        q0 = evaluate_table(table0, queries)
        loss_by_seed[str(seed)] = float(compute_loss(target, q0))
        l1_by_seed[str(seed)] = float(
            compute_normalized_l1(target, q0, spec["n_records"])
        )
    return {
        "measured_l1_by_seed": l1_by_seed,
        "loss_by_seed": loss_by_seed,
    }


def validate_formal_artifact(payload, *, source_kind):
    """唯一生产验证入口（第六轮最终合同）。

    生成器必须在正式结果写入前调用（source_kind="generator"）；审计器
    必须在输出审计结果前对完整输出调用（source_kind="audit"，此时
    payload 额外携带冻结结构的 audit 段）。测试直接调用本入口，不得
    重新实现另一套验证逻辑。

    纯验证：不修改 payload；违反冻结结构合同的任何一条即抛
    FormalArtifactError（调用方 fail-closed 处理）。规则覆盖：

    1.  顶层字段集合精确等于冻结结构（缺失/多余/冲突字段全拒）；
    2.  artifact_schema_version 精确匹配；
    3.  protocol 与冻结代码重建的 canonical 清单整份完全相等；
    4.  run_config 精确等于冻结协议（seeds 列表逐项、rounds、
        datasets 排序后精确比较，缺项/重复/顺序错误全拒）；
    5.  formal is True、protocol_match is True、git_dirty is False
        （必须是真正的布尔值）；protocol_sha256 精确等于冻结常量；
    6.  input_sha256 与 reference_sha256 整份完全相等（缺少或增加
        任何键都拒绝）；
    7.  数据集集合精确一致；每个数据集字段集合精确等于冻结结构
        （initial_state / reference_sha256 / runs / judgment）；
    8.  新版只允许 judgment 拼写（字段集合精确比较天然拒绝旧拼写与
        双拼写；本入口不做任何迁移或补齐）；
    9.  每条运行记录字段集合与类型精确匹配；seed×arm 组合恰好一次；
    10. 必需数值为有限数且非布尔；rounds_run/数据集名/种子/臂必须
        与所属任务一致；
    11. 判定为非空映射，并无条件重算后完全一致；
    12. initial_state 必须已存在（缺失即拒，不补写），并按冻结种子
        重算后逐种子对拍 measured_l1 与 loss。
    """
    if source_kind not in ("generator", "audit"):
        raise ValueError(f"source_kind 非法: {source_kind!r}")

    def fail(msg):
        raise FormalArtifactError(f"正式产物验证失败: {msg}")

    if not isinstance(payload, dict):
        fail(f"顶层必须是映射，得到 {type(payload).__name__}")
    expected_top = set(FORMAL_TOP_LEVEL_FIELDS)
    if source_kind == "audit":
        expected_top.add("audit")
    if set(payload) != expected_top:
        fail(
            f"顶层字段集合不符——缺失 {sorted(expected_top - set(payload))} "
            f"多余 {sorted(set(payload) - expected_top)}"
        )
    if source_kind == "audit":
        audit = payload["audit"]
        if not isinstance(audit, dict) or set(audit) != set(
            FORMAL_AUDIT_SECTION_FIELDS
        ):
            fail("audit 段字段集合不符或类型错误")
        if not _is_sha256_hex(audit["source_sha256"]):
            fail("audit.source_sha256 必须是 64 位十六进制字符串")
        if audit["source_kind"] not in FORMAL_AUDIT_SOURCE_KINDS:
            fail(f"audit.source_kind 非法: {audit['source_kind']!r}")

    if payload["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        fail(
            f"产物结构版本 {payload['artifact_schema_version']!r} != "
            f"当前 {ARTIFACT_SCHEMA_VERSION!r}"
        )

    expected_manifest = canonical_protocol_manifest()
    declared = payload["protocol"]
    if not isinstance(declared, dict):
        fail("protocol 必须是映射")
    if _canonical_json(declared) != _canonical_json(expected_manifest):
        recorded_keys, expected_keys = set(declared), set(expected_manifest)
        if recorded_keys != expected_keys:
            detail = (
                f"缺失 {sorted(expected_keys - recorded_keys)} "
                f"多余 {sorted(recorded_keys - expected_keys)}"
            )
        else:
            detail = str(sorted(
                key for key in expected_keys
                if _canonical_json(declared[key])
                != _canonical_json(expected_manifest[key])
            ))
        fail(f"protocol 清单与冻结代码重建值整份对拍失败（差异：{detail}）")

    rc = payload["run_config"]
    if not isinstance(rc, dict) or set(rc) != set(FORMAL_RUN_CONFIG_FIELDS):
        fail("run_config 字段集合不符或类型错误")
    seeds = rc["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != len(FORMAL_SEEDS)
        or any(not _is_strict_int(s) for s in seeds)
        or [int(s) for s in seeds] != list(FORMAL_SEEDS)
    ):
        fail(f"run_config.seeds {seeds!r} != 冻结 {FORMAL_SEEDS}")
    if not _is_strict_int(rc["rounds"]) or rc["rounds"] != FORMAL_ROUNDS:
        fail(f"run_config.rounds {rc['rounds']!r} != 冻结 {FORMAL_ROUNDS}")
    if rc["datasets"] != sorted(DATASETS):
        fail(
            f"run_config.datasets {rc['datasets']!r} != 冻结 "
            f"{sorted(DATASETS)}（缺项/重复/顺序错误均拒绝）"
        )

    prov = payload["provenance"]
    if not isinstance(prov, dict):
        fail(f"provenance 必须是映射，得到 {type(prov).__name__}")
    if set(prov) != set(FORMAL_PROVENANCE_FIELDS):
        fail(
            "provenance 字段集合不符——缺失 "
            f"{sorted(set(FORMAL_PROVENANCE_FIELDS) - set(prov))} 多余 "
            f"{sorted(set(prov) - set(FORMAL_PROVENANCE_FIELDS))}"
        )
    if prov["formal"] is not True:
        fail(f"provenance.formal 必须是布尔 True，得到 {prov['formal']!r}")
    if prov["protocol_match"] is not True:
        fail(
            "provenance.protocol_match 必须是布尔 True，得到 "
            f"{prov['protocol_match']!r}"
        )
    if prov["git_dirty"] is not False:
        fail(
            "provenance.git_dirty 必须是布尔 False，得到 "
            f"{prov['git_dirty']!r}"
        )
    if prov["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        fail(
            f"provenance.protocol_sha256 {prov['protocol_sha256']!r} != "
            "冻结常量"
        )
    if not (
        isinstance(prov["git_commit"], str)
        and len(prov["git_commit"]) == 40
        and all(c in "0123456789abcdef" for c in prov["git_commit"])
    ):
        fail("provenance.git_commit 必须是 40 位十六进制提交哈希")
    for key in ("started_at", "finished_at", "command"):
        if not isinstance(prov[key], str) or not prov[key]:
            fail(f"provenance.{key} 必须是非空字符串")
    if not isinstance(prov["environment"], dict) or not prov["environment"]:
        fail("provenance.environment 必须是非空映射")
    if prov["input_hash_mismatches"] != []:
        fail(
            "formal 产物 provenance.input_hash_mismatches 必须是空列表，"
            f"得到 {prov['input_hash_mismatches']!r}"
        )
    expected_inputs = json.loads(_canonical_json(
        {k: dict(v) for k, v in EXPECTED_INPUT_SHA256.items()}
    ))
    if _canonical_json(prov["input_sha256"]) != _canonical_json(
        expected_inputs
    ):
        fail(
            "provenance.input_sha256 与冻结 EXPECTED_INPUT_SHA256 整份"
            "对拍失败（缺少或增加任何键、任何值差异都拒绝）"
        )

    datasets = payload["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        fail(
            f"datasets 集合 {sorted(datasets) if isinstance(datasets, dict) else datasets!r} "
            f"!= 冻结 {sorted(DATASETS)}"
        )
    for name in sorted(DATASETS):
        ds = datasets[name]
        if not isinstance(ds, dict) or set(ds) != set(FORMAL_DATASET_FIELDS):
            fail(
                f"datasets[{name}] 字段集合不符（必须恰好为 "
                f"{sorted(FORMAL_DATASET_FIELDS)}；旧拼写 judgement、"
                "多余或缺失字段均拒绝）"
            )
        expected_refs = json.loads(_canonical_json(
            dict(EXPECTED_REFERENCE_SHA256[name])
        ))
        if _canonical_json(ds["reference_sha256"]) != _canonical_json(
            expected_refs
        ):
            fail(
                f"datasets[{name}].reference_sha256 与冻结常量整份对拍"
                "失败（缺少或增加任何键都拒绝）"
            )
        runs = ds["runs"]
        if not isinstance(runs, list) or not runs:
            fail(f"datasets[{name}].runs 必须是非空列表")
        expected_ref_names = set(DATASETS[name]["references"])
        combo_counts = {}
        for idx, run in enumerate(runs):
            where = f"datasets[{name}].runs[{idx}]"
            if not isinstance(run, dict) or set(run) != set(
                FORMAL_RUN_RECORD_FIELDS
            ):
                fail(
                    f"{where} 字段集合不符——缺失 "
                    f"{sorted(set(FORMAL_RUN_RECORD_FIELDS) - set(run)) if isinstance(run, dict) else '全部'}"
                    f" 多余 {sorted(set(run) - set(FORMAL_RUN_RECORD_FIELDS)) if isinstance(run, dict) else ''}"
                )
            if run["dataset"] != name:
                fail(f"{where}.dataset {run['dataset']!r} != 所属 {name}")
            if run["arm"] not in ARMS:
                fail(f"{where}.arm {run['arm']!r} 不在冻结臂集合中")
            if not _is_strict_int(run["seed"]) or (
                run["seed"] not in FORMAL_SEEDS
            ):
                fail(f"{where}.seed {run['seed']!r} 不在冻结种子中")
            if not _is_strict_int(run["rounds_run"]) or (
                run["rounds_run"] != FORMAL_ROUNDS
            ):
                fail(
                    f"{where}.rounds_run {run['rounds_run']!r} != 冻结轮数 "
                    f"{FORMAL_ROUNDS}"
                )
            if not _is_strict_int(run["candidate_evaluations"]) or (
                run["candidate_evaluations"] != FORMAL_ROUNDS
            ):
                fail(
                    f"{where}.candidate_evaluations "
                    f"{run['candidate_evaluations']!r} != 冻结协议候选"
                    f"评价数 {FORMAL_ROUNDS}（max_retries=0 下每轮恰好"
                    "一次）"
                )
            if not _is_strict_int(run["exact_match_queries"]) or (
                run["exact_match_queries"] < 0
            ):
                fail(f"{where}.exact_match_queries 必须是非负整数")
            for key in sorted(FORMAL_RUN_FLOAT_FIELDS):
                if not _is_finite_number(run[key]):
                    fail(f"{where}.{key}={run[key]!r} 缺失或非有限数值")
            if run["elapsed_sec"] < 0:
                fail(f"{where}.elapsed_sec 必须非负")
            for key in sorted(FORMAL_RUN_NULLABLE_FIELDS):
                if run[key] is not None and not _is_finite_number(run[key]):
                    fail(
                        f"{where}.{key}={run[key]!r} 必须是有限数值或 null"
                    )
            if not _is_sha256_hex(run["final_table_sha256"]):
                fail(f"{where}.final_table_sha256 必须是 64 位十六进制")
            offline = run["offline"]
            if not isinstance(offline, dict) or set(offline) != (
                expected_ref_names
            ):
                fail(
                    f"{where}.offline 参考名集合 != 冻结 "
                    f"{sorted(expected_ref_names)}"
                )
            for ref_name in sorted(expected_ref_names):
                metrics = offline[ref_name]
                if not isinstance(metrics, dict) or set(metrics) != set(
                    FORMAL_OFFLINE_METRIC_FIELDS
                ):
                    fail(
                        f"{where}.offline[{ref_name}] 指标字段集合不符"
                        f"（必须恰好为 {sorted(FORMAL_OFFLINE_METRIC_FIELDS)}）"
                    )
                for key in sorted(FORMAL_OFFLINE_FLOAT_FIELDS):
                    if not _is_finite_number(metrics[key]):
                        fail(
                            f"{where}.offline[{ref_name}].{key}="
                            f"{metrics[key]!r} 缺失或非有限数值"
                        )
                for key in sorted(FORMAL_OFFLINE_INT_FIELDS):
                    if not _is_strict_int(metrics[key]) or metrics[key] < 0:
                        fail(
                            f"{where}.offline[{ref_name}].{key}="
                            f"{metrics[key]!r} 必须是非负整数"
                        )
            combo = (int(run["seed"]), str(run["arm"]))
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        expected_combos = {
            (seed, arm) for seed in FORMAL_SEEDS for arm in ARMS
        }
        if set(combo_counts) != expected_combos:
            fail(
                f"datasets[{name}] seed×arm 组合集合不符——缺失 "
                f"{sorted(expected_combos - set(combo_counts))[:5]} 多余 "
                f"{sorted(set(combo_counts) - expected_combos)[:5]}"
            )
        duplicated = {
            combo: count for combo, count in combo_counts.items()
            if count != 1
        }
        if duplicated:
            fail(
                f"datasets[{name}] 存在重复 seed×arm 运行记录："
                f"{sorted(duplicated.items())[:5]}"
            )

        ist = ds["initial_state"]
        if not isinstance(ist, dict) or set(ist) != set(
            FORMAL_INITIAL_STATE_FIELDS
        ):
            fail(
                f"datasets[{name}].initial_state 缺失或字段集合不符"
                "（新版缺失直接拒绝，审计不补写）"
            )
        expected_seed_keys = {str(seed) for seed in FORMAL_SEEDS}
        for map_key in ("measured_l1_by_seed", "loss_by_seed"):
            mapping = ist[map_key]
            if not isinstance(mapping, dict) or set(mapping) != (
                expected_seed_keys
            ):
                fail(
                    f"datasets[{name}].initial_state.{map_key} 种子键集合"
                    f" != 冻结 {sorted(expected_seed_keys)}"
                )
            for seed_key in sorted(expected_seed_keys):
                if not _is_finite_number(mapping[seed_key]):
                    fail(
                        f"datasets[{name}].initial_state.{map_key}"
                        f"[{seed_key}] 非有限数值"
                    )
        if not _is_finite_number(ist["measured_l1_mean"]) or not np.isclose(
            ist["measured_l1_mean"],
            float(np.mean(list(ist["measured_l1_by_seed"].values()))),
            rtol=INITIAL_STATE_RTOL, atol=INITIAL_STATE_ATOL,
        ):
            fail(
                f"datasets[{name}].initial_state.measured_l1_mean 与"
                " by_seed 均值不一致"
            )
        if not isinstance(ist["note"], str) or not ist["note"]:
            fail(f"datasets[{name}].initial_state.note 必须是非空字符串")

        judgment = ds["judgment"]
        if not isinstance(judgment, dict) or not judgment:
            fail(
                f"datasets[{name}].judgment 必须是非空映射，得到 "
                f"{type(judgment).__name__}"
            )
        recomputed = _judge(runs)
        if _canonical_json(judgment) != _canonical_json(
            json.loads(_canonical_json(recomputed))
        ):
            fail(f"datasets[{name}].judgment 与无条件重算结果不一致")

    # 规则 12（放最后：结构反例快速失败，昂贵重算只对结构合法产物执行）
    for name in sorted(DATASETS):
        rebuilt = recompute_initial_state(name)
        ist = payload["datasets"][name]["initial_state"]
        for map_key in ("measured_l1_by_seed", "loss_by_seed"):
            for seed_key, value in rebuilt[map_key].items():
                recorded = ist[map_key][seed_key]
                if not np.isclose(
                    recorded, value,
                    rtol=INITIAL_STATE_RTOL, atol=INITIAL_STATE_ATOL,
                ):
                    fail(
                        f"datasets[{name}].initial_state.{map_key}"
                        f"[{seed_key}] 重算对拍不一致（记录 {recorded} vs"
                        f" 重算 {value}）"
                    )


# 冻结的预期输入身份（fail-closed）：正式运行必须精确匹配这组公开输入。
# 与首次正式产物（commit aac1aff）记录的 input_sha256 一致。
EXPECTED_INPUT_SHA256 = {
    "test_300x10": {
        "schema": (
            "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
        ),
        "queries": (
            "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88"
        ),
        "marginals": (
            "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
        ),
    },
    "nltcs": {
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
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame):
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _environment():
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    # PR #62 审查意见：协议 SHA 有意不含 device（冻结协议常量边界），
    # 实际运行环境（device/torch/CUDA/GPU）单独入档供跨环境复现判读。
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch"] = None
    return env


def _load_reference(path, columns):
    # .data 等无表头文件必须显式 header=None，否则首行数据被当表头
    # 吃掉（行数 -1，触发行数一致性校验失败）。与 v3 协议脚本一致。
    if str(path).endswith(".csv"):
        frame = pd.read_csv(path)
        if list(frame.columns) != columns:
            frame.columns = columns
    else:
        frame = pd.read_csv(path, header=None, names=columns)
    return frame[columns]


def _load_verified_reference(dataset_name, ref_name, path, columns):
    """一次读取参考文件原始字节 → 先验 SHA-256 → 再从同一份字节解析。

    fail-closed（Issue #60 + PR #62 审查意见 1/4）：哈希校验先于任何
    内容使用，消除"读取与验哈希之间"的变化窗口；本函数是生产与测试
    共用的唯一校验入口。未登记预期值的数据集跳过哈希对拍（其正式
    身份由协议 SHA 罩住——登记后才可能 formal）。

    Returns (frame, sha256_hex)。
    """
    import io

    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_refs = EXPECTED_REFERENCE_SHA256.get(dataset_name)
    if expected_refs is not None:
        expected = expected_refs.get(ref_name)
        if digest != expected:
            raise RuntimeError(
                f"[{dataset_name}] 参考 {ref_name} SHA-256 与冻结值不符："
                f"实际 {digest[:12]}… != 预期 {str(expected)[:12]}…"
            )
    buffer = io.BytesIO(payload)
    if str(path).endswith(".csv"):
        frame = pd.read_csv(buffer)
        if list(frame.columns) != columns:
            frame.columns = columns
    else:
        frame = pd.read_csv(buffer, header=None, names=columns)
    return frame[columns], digest


def _run_dataset(name, spec, seeds, rounds):
    schema = load_schema(str(spec["schema"]))
    queries = load_queries(str(spec["queries"]))
    marginals = load_marginals(str(spec["marginals"]))
    target = np.asarray([q["result"] for q in queries], dtype=float)
    n_records = spec["n_records"]
    columns_check = schema.attribute_names()
    # 源数据一致性校验：生成前只允许用公开元信息。
    with open(spec["queries"], "r", encoding="utf-8") as handle:
        query_meta = json.load(handle)
    meta_count = query_meta.get("record_count")
    if meta_count is not None and int(meta_count) != n_records:
        raise RuntimeError(
            f"[{name}] queries record_count {meta_count} 与 n_records "
            f"{n_records} 不一致，违反源数据规则"
        )
    # initial_state 原生输出（n_rounds=0，只依赖公开输入；
    # scripts/audit_formal_json.py 可独立复验）。
    init_l1_by_seed = {}
    init_loss_by_seed = {}
    for seed in seeds:
        _, diag0 = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=n_records, n_rounds=0, seed=seed,
            marginals=marginals, log_every=-1, device=spec["device"],
            return_final_table=True,
            **SHARED_PARAMS, **ARMS[PRIMARY_BASELINE],
        )
        table0 = diag0.pop("final_table")
        q0 = evaluate_table(table0, queries)
        init_loss_by_seed[str(seed)] = float(compute_loss(q0, target))
        init_l1_by_seed[str(seed)] = float(
            compute_normalized_l1(q0, target, n_records=n_records)
        )
    initial_state = {
        "measured_l1_mean": float(np.mean(list(init_l1_by_seed.values()))),
        "measured_l1_by_seed": init_l1_by_seed,
        "loss_by_seed": init_loss_by_seed,
        "note": "n_rounds=0 的 marginal 初始化状态（种子相关）",
    }
    runs = []
    tables = {}
    for seed in seeds:
        for arm, extra in ARMS.items():
            start = time.perf_counter()
            _, diag = run_evolution(
                target=target, queries=queries, schema=schema,
                n_records=n_records, n_rounds=rounds, seed=seed,
                marginals=marginals, log_every=0, device=spec["device"],
                return_final_table=True,
                **SHARED_PARAMS, **extra,
            )
            elapsed = time.perf_counter() - start
            losses = diag["loss_history"]
            final_table = diag.pop("final_table")
            final_q = evaluate_table(final_table, queries)
            final_loss = float(compute_loss(final_q, target))
            final_l1 = float(
                compute_normalized_l1(final_q, target, n_records=n_records)
            )
            tables[(seed, arm)] = final_table
            if len(final_table) != n_records or (
                list(final_table.columns) != columns_check
            ):
                raise RuntimeError(
                    f"[{name} seed={seed} {arm}] 合成表行数/列名与源数据"
                    "不一致"
                )
            raw_abs = np.abs(target - final_q)
            p_rate = target / n_records
            rare_mask = p_rate < 0.05
            runs.append({
                "dataset": name,
                "arm": arm,
                "seed": seed,
                "rounds_run": diag["rounds_run"],
                "candidate_evaluations": diag["candidate_evaluation_count"],
                "pre_final_proposal_loss": float(losses[-1]),
                "final_loss": final_loss,
                "final_table_measured_l1": final_l1,
                "best_loss": float(diag["best_loss"]),
                # 稀有查询残差诊断（本实验的机制指标）：p<0.05 与其余
                # 查询的平均绝对计数残差。
                "rare_query_mean_abs_residual": (
                    float(raw_abs[rare_mask].mean())
                    if rare_mask.any() else None
                ),
                "common_query_mean_abs_residual": (
                    float(raw_abs[~rare_mask].mean())
                    if (~rare_mask).any() else None
                ),
                "exact_match_queries": int((raw_abs == 0).sum()),
                "row_max_prob_mean_final": (
                    float(diag["row_max_prob_mean_history"][-1])
                    if diag.get("row_max_prob_mean_history") else None
                ),
                "effective_donors_mean_final": (
                    float(diag["effective_donors_mean_history"][-1])
                    if diag.get("effective_donors_mean_history") else None
                ),
                # 口径注记：loss_history 为 round-start/pre-proposal 状态窗口。
                "tail_mean_pre_proposal_loss": float(np.mean(losses[-100:])),
                "final_table_sha256": _frame_sha256(final_table),
                "elapsed_sec": round(elapsed, 1),
            })
            print(
                f"[{name} seed={seed} {arm}] final={final_loss:.4g} "
                f"L1={final_l1:.6f} ({elapsed:.0f}s)",
                flush=True,
            )
    # 隐私边界：该数据集全部生成完成后才读取真实参考表做离线评价。
    columns = schema.attribute_names()
    domains = offline_helpers._discretization_domains(marginals)
    measured_triples = offline_helpers._measured_cell_keys(
        queries, marginals, order=3
    )
    references = {}
    reference_sha256 = {}
    for ref_name, path in spec["references"].items():
        # 先验 SHA 再解析（fail-closed，生产与测试共用同一入口）。
        frame, digest = _load_verified_reference(
            name, ref_name, path, columns
        )
        references[ref_name] = frame
        reference_sha256[ref_name] = digest
    for ref_name, reference in references.items():
        if len(reference) != n_records:
            raise RuntimeError(
                f"[{name}] 参考 {ref_name} 行数 {len(reference)} 与"
                f" n_records {n_records} 不一致，违反源数据规则"
            )
        if list(reference.columns) != columns:
            raise RuntimeError(f"[{name}] 参考 {ref_name} 列名不一致")
    for run in runs:
        table = tables[(run["seed"], run["arm"])]
        run["offline"] = {}
        for ref_name, reference in references.items():
            metrics = offline_helpers._offline_metrics(
                reference, table, marginals, domains, measured_triples
            )
            run["offline"][ref_name] = {
                "unmeasured_3way_l1": float(
                    metrics["unmeasured_3way"]["mean"]
                ),
                "unmeasured_4way_l1": float(
                    metrics["unmeasured_4way"]["mean"]
                ),
                "raw_joint_tvd": float(metrics["raw_joint"]["tvd"]),
                "binned_joint_tvd": float(metrics["binned_joint"]["tvd"]),
                "raw_unique_states": int(metrics["raw_joint"]["n_unique"]),
                "raw_support_overlap": int(
                    metrics["raw_joint"]["support_overlap"]
                ),
            }
    # reference_sha256 已在 _load_verified_reference 中"先验后用"得到，
    # 此处不再重复读文件或后置校验（PR #62 审查意见 4）。
    return runs, initial_state, reference_sha256


def _assert_offline_complete(name, runs, formal):
    """fail-closed（Issue #60）：正式运行中任何 run 缺失或非有限的
    离线指标视为协议违规直接失败，而不是被 _judge 静默跳过降级为
    无风险（NaN/inf 同样拒绝，PR #62 审查意见 2）。"""
    required = (
        "unmeasured_3way_l1", "unmeasured_4way_l1", "binned_joint_tvd",
    )
    missing = []
    for run in runs:
        offline = run.get("offline") or {}
        for ref_name, metrics in offline.items():
            for key in required:
                value = metrics.get(key)
                if (
                    value is None
                    or not isinstance(
                        value, (int, float, np.integer, np.floating)
                    )
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                ):
                    missing.append(
                        f"seed={run['seed']} arm={run['arm']} "
                        f"{ref_name}/{key}={value!r}"
                    )
        if not offline:
            missing.append(f"seed={run['seed']} arm={run['arm']} 无 offline")
    if missing and formal:
        raise RuntimeError(
            f"[{name}] 正式运行存在缺失或非有限离线指标（fail-closed）：\n  "
            + "\n  ".join(missing[:10])
        )
    if missing:
        print(
            f"[{name}] 警告：{len(missing)} 项离线指标缺失（探索模式继续，"
            "质量风险判定将跳过缺失项）",
            flush=True,
        )


def _arm_metric_by_seed(runs, arm, metric):
    return {
        run["seed"]: run[metric]
        for run in runs if run["arm"] == arm
    }


def _arm_mean(runs, arm, metric):
    values = [run[metric] for run in runs if run["arm"] == arm]
    return float(np.mean(values))


def _offline_mean(runs, arm, ref_name, metric):
    values = [
        run["offline"][ref_name][metric]
        for run in runs if run["arm"] == arm and ref_name in run["offline"]
    ]
    return float(np.mean(values)) if values else None


def _judge(runs):
    # 离线参考名从 runs 自动推断（每数据集只有一个参考：train/reference），
    # 保持与 scripts/audit_formal_json.py 的单参 _judge(runs) 接口兼容。
    ref_names = [
        next(iter(run["offline"]))
        for run in runs if run.get("offline")
    ]
    ref_name = ref_names[0] if ref_names else None
    metric = "final_table_measured_l1"
    base_by_seed = _arm_metric_by_seed(runs, PRIMARY_BASELINE, metric)
    cand_by_seed = _arm_metric_by_seed(runs, PRIMARY_CANDIDATE, metric)
    seeds = sorted(base_by_seed)
    wins = sum(
        1 for seed in seeds if cand_by_seed[seed] < base_by_seed[seed]
    )
    base_mean = float(np.mean([base_by_seed[s] for s in seeds]))
    cand_mean = float(np.mean([cand_by_seed[s] for s in seeds]))
    improvement = (
        (base_mean - cand_mean) / base_mean if base_mean > 0 else 0.0
    )
    primary_pass = (
        wins >= PRIMARY_MIN_WINS and improvement >= PRIMARY_MIN_IMPROVEMENT
    )
    mixed = (not primary_pass) and (
        wins >= MIXED_MIN_WINS and cand_mean < base_mean
    )
    # floor 稳健性（观察项，不进主判定）
    floor_means = {
        arm: _arm_mean(runs, arm, metric)
        for arm in ARMS if arm != PRIMARY_BASELINE
    }
    floor_best = min(floor_means, key=floor_means.get)
    floor_suboptimal = floor_best != PRIMARY_CANDIDATE
    # 质量风险带：train 侧未测量 3/4-way 与分箱 TVD 相对 absolute
    # 劣化 >QUALITY_RISK_REL 报警。
    risks = {}
    for quality_metric in (
        "unmeasured_3way_l1", "unmeasured_4way_l1", "binned_joint_tvd",
    ):
        base_q = _offline_mean(runs, PRIMARY_BASELINE, ref_name, quality_metric)
        cand_q = _offline_mean(runs, PRIMARY_CANDIDATE, ref_name, quality_metric)
        if base_q is None or cand_q is None or base_q <= 0:
            risks[quality_metric] = None
            continue
        rel = (cand_q - base_q) / base_q
        risks[quality_metric] = {
            "baseline_mean": base_q,
            "candidate_mean": cand_q,
            "relative_change": float(rel),
            "flagged": bool(rel > QUALITY_RISK_REL),
        }
    any_quality_risk = any(
        item is not None and item["flagged"] for item in risks.values()
    )
    if primary_pass:
        classification = "supports_relative_geometry"
        if any_quality_risk:
            classification += "_with_quality_risk"
    elif mixed:
        classification = "mixed"
    else:
        classification = "not_supported"
    return {
        "metric": metric,
        "primary_baseline": PRIMARY_BASELINE,
        "primary_candidate": PRIMARY_CANDIDATE,
        "baseline_l1_by_seed": {str(s): base_by_seed[s] for s in seeds},
        "candidate_l1_by_seed": {str(s): cand_by_seed[s] for s in seeds},
        "paired_wins": int(wins),
        "n_seeds": len(seeds),
        "baseline_mean": base_mean,
        "candidate_mean": cand_mean,
        "relative_improvement": float(improvement),
        "primary_pass": bool(primary_pass),
        "floor_arm_means": floor_means,
        "floor_best_arm": floor_best,
        "floor_suboptimal_flag": bool(floor_suboptimal),
        "quality_risks": risks,
        "any_quality_risk": bool(any_quality_risk),
        "classification": classification,
        "thresholds": {
            "primary_min_wins": PRIMARY_MIN_WINS,
            "primary_min_improvement": PRIMARY_MIN_IMPROVEMENT,
            "mixed_min_wins": MIXED_MIN_WINS,
            "quality_risk_rel": QUALITY_RISK_REL,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="允许脏工作树运行（输出无条件标记 formal=false）",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=FORMAL_SEEDS,
        help="种子列表（偏离预注册值时输出标记 formal=false）",
    )
    parser.add_argument(
        "--rounds", type=int, default=FORMAL_ROUNDS,
        help="每个 run 的轮数（偏离预注册值时输出标记 formal=false）",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS),
        help="要运行的数据集（偏离预注册全集时输出标记 formal=false）",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="输出 JSON 路径（已存在时拒绝覆盖）",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"输出 {args.output} 已存在，拒绝覆盖；请换路径或手动移除"
        )

    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise SystemExit("工作树不干净；正式运行要求干净树（或 --allow-dirty）")

    # fail-closed（Issue #60）：协议身份对拍。协议常量（臂/参数/阈值/
    # 冻结哈希）任何变化都会改变 protocol_sha256；正式运行要求与
    # FROZEN_PROTOCOL_SHA256 一致——修改协议必须同步更新常量（等价于
    # 显式重新预注册）。探索模式（--allow-dirty）允许继续但 formal=false。
    current_protocol_sha = protocol_sha256()
    protocol_match = current_protocol_sha == FROZEN_PROTOCOL_SHA256
    if not protocol_match and not args.allow_dirty:
        raise SystemExit(
            "协议身份与 FROZEN_PROTOCOL_SHA256 不符，拒绝正式运行"
            "（协议已修改？请显式重新预注册并更新冻结常量；探索性运行"
            f"请加 --allow-dirty）。当前协议 SHA: {current_protocol_sha}"
        )

    matches_prereg = (
        not dirty
        and list(args.seeds) == FORMAL_SEEDS
        and args.rounds == FORMAL_ROUNDS
        and sorted(args.datasets) == sorted(DATASETS)
    )
    formal = bool(
        matches_prereg and not args.allow_dirty and protocol_match
    )

    input_sha256 = {}
    input_hash_mismatches = []
    for name in args.datasets:
        spec = DATASETS[name]
        input_sha256[name] = {
            "schema": _sha256_file(spec["schema"]),
            "queries": _sha256_file(spec["queries"]),
            "marginals": _sha256_file(spec["marginals"]),
        }
        expected = EXPECTED_INPUT_SHA256.get(name)
        if expected is not None:
            for kind, digest in input_sha256[name].items():
                if digest != expected[kind]:
                    input_hash_mismatches.append(
                        f"{name}/{kind}: 实际 {digest[:12]}… != 冻结 "
                        f"{expected[kind][:12]}…"
                    )
    # fail-closed：公开输入与冻结身份不符时拒绝正式运行（预注册协议
    # 绑定这组输入）。--allow-dirty 的探索模式允许继续（formal 已为
    # false），但把偏差记录进输出。
    if input_hash_mismatches and not args.allow_dirty:
        raise SystemExit(
            "输入身份与冻结的 EXPECTED_INPUT_SHA256 不符，拒绝正式运行"
            "（探索性运行请加 --allow-dirty）：\n  "
            + "\n  ".join(input_hash_mismatches)
        )
    if input_hash_mismatches:
        formal = False

    result = {
        # 产物结构版本（PR #62 五轮不变量 1）：审计端按版本执行统一
        # 结构验证。formal 产物的 protocol 字段即 canonical 清单本身；
        # 探索性运行（偏离预注册 seeds/rounds/datasets）在 run_config
        # 中记录实际参数，protocol 清单仍为冻结值供对拍。
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol": canonical_protocol_manifest(),
        "run_config": {
            "seeds": list(args.seeds),
            "rounds": args.rounds,
            # 排序后的确定形态（冻结结构合同：顺序错误也拒绝）
            "datasets": sorted(args.datasets),
        },
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": dirty,
            "protocol_sha256": current_protocol_sha,
            "protocol_match": protocol_match,
            "formal": formal,
            "started_at": datetime.now().astimezone().isoformat(),
            "environment": _environment(),
            "input_sha256": input_sha256,
            "input_hash_mismatches": input_hash_mismatches,
            "command": " ".join(sys.argv),
        },
        "datasets": {},
    }

    for name in args.datasets:
        spec = DATASETS[name]
        print(f"=== 数据集 {name} ===", flush=True)
        runs, initial_state, reference_sha256 = _run_dataset(
            name, spec, args.seeds, args.rounds
        )
        _assert_offline_complete(name, runs, formal)
        result["datasets"][name] = {
            "initial_state": initial_state,
            "reference_sha256": reference_sha256,
            "runs": runs,
            # 字段名与 scripts/audit_formal_json.py 及既有正式协议一致
            # （judgment，无 e）。首次正式产物（aac1aff）使用旧拼写
            # judgement，审计器已兼容读取。
            "judgment": _judge(runs),
        }
        print(
            f"[{name}] 判定: "
            f"{result['datasets'][name]['judgment']['classification']}",
            flush=True,
        )

    result["provenance"]["finished_at"] = (
        datetime.now().astimezone().isoformat()
    )
    # 第六轮最终合同：生成器必须在正式结果写入前调用唯一验证入口。
    # 任何冻结结构违规都在落盘前 fail-closed（探索性运行不受此约束，
    # 其 formal=False 本身即被审计器拒绝）。
    if formal:
        validate_formal_artifact(result, source_kind="generator")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.output} (formal={formal})", flush=True)


if __name__ == "__main__":
    main()
