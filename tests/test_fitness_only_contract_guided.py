"""引导扩展核的合同据实与审计防伪测试（PR #73 审查 F1）。

历史缺陷：``value_guidance_strength>0`` 让残差经逐格值分布进入转移核，
但产物合同仍硬编码 ``transition_kernel='blind_independent'`` 与
``residual_driving_channels=['fitness']``，且审计照常通过——"证件造假"。
本文件锁死修复后的行为：合同据实声明、审计器按请求配置重算期望并
fail-closed 拒绝伪造声明。
"""

import numpy as np
import pytest

from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    _audit_fitness_only_run,
    run_fitness_only_evolution,
    run_paired_fitness_only_attribution,
)
from table_diffevo.schema import AttributeBlock, Schema


def _problem():
    schema = Schema([
        AttributeBlock(
            name="a",
            type="categorical",
            description="a",
            values=["0", "1"],
        ),
        AttributeBlock(
            name="b",
            type="categorical",
            description="b",
            values=["0", "1"],
        ),
        AttributeBlock(
            name="c",
            type="categorical",
            description="c",
            values=["0", "1"],
        ),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": "1"},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
            {"attribute": "b", "operator": "==", "value": "1"},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
            {"attribute": "c", "operator": "==", "value": "1"},
        ]},
    ]
    target = np.asarray([8.0, 3.0, 3.0, 6.0])
    return schema, queries, target


def _config(**overrides):
    values = {
        "n_rounds": 7,
        "seed": 23,
        "init_method": "random",
        "log_every": 100,
        "rho": 0.5,
        "eta": 0.5,
        "mu": 0.01,
    }
    values.update(overrides)
    return FitnessOnlyConfig(**values)


def _run(config, fitness_mode="residual"):
    schema, queries, target = _problem()
    return run_fitness_only_evolution(
        target,
        schema=schema,
        queries=queries,
        n_records=12,
        config=config,
        fitness_mode=fitness_mode,
    )


def test_blind_default_contract_unchanged():
    """λ=0 回归锚：默认合同与历史产物口径逐字段一致。"""
    _, diagnostics = _run(_config())
    contract = diagnostics["fitness_only_contract"]
    assert contract["residual_driving_channels"] == ["fitness"]
    assert contract["transition_kernel"] == "blind_independent"


def test_value_guided_contract_declares_channel_and_kernel():
    """λ>0（residual 臂）：合同必须据实声明值引导通道与核名。"""
    _, diagnostics = _run(
        _config(value_guidance_strength=4.0, value_guidance_adaptive_scale=True)
    )
    contract = diagnostics["fitness_only_contract"]
    assert contract["residual_driving_channels"] == [
        "fitness",
        "value_guidance",
    ]
    assert contract["transition_kernel"] == "value_guided"
    vg = diagnostics["value_guidance"]
    assert vg["enabled"] is True
    assert float(vg["strength"]) == 4.0
    assert float(diagnostics["params"]["value_guidance_strength"]) == 4.0


def test_value_guided_equal_arm_rejected_fail_closed():
    """equal 臂 + λ>0 必须被拒绝：equal 臂禁用一切残差信号，而值引导
    gain 正是残差信号——引擎 fail-closed 拒绝，杜绝"带引导的对照臂"。"""
    with pytest.raises(ValueError, match="equal 臂禁用一切残差"):
        _run(_config(value_guidance_strength=2.0), fitness_mode="equal")


def test_block_score_tilt_contract_declares_channel_and_kernel():
    """块倾斜同族问题：残差块分数倾斜复制开关，同样不得声明盲核。"""
    _, diagnostics = _run(_config(block_score_tilt_strength=1.0))
    contract = diagnostics["fitness_only_contract"]
    assert contract["residual_driving_channels"] == [
        "fitness",
        "block_score_tilt",
    ]
    assert contract["transition_kernel"] == "block_score_tilted"


def test_audit_rejects_forged_blind_contract_when_guided():
    """防伪核心：λ>0 的产物若携带 blind_independent 声明，审计必须拒绝。"""
    config = _config(value_guidance_strength=4.0)
    _, diagnostics = _run(config)
    forged = dict(diagnostics)
    forged["fitness_only_contract"] = dict(
        diagnostics["fitness_only_contract"],
        residual_driving_channels=["fitness"],
        transition_kernel="blind_independent",
    )
    with pytest.raises(RuntimeError, match="合同诊断与请求不一致"):
        _audit_fitness_only_run(forged, config, "residual")


def test_audit_rejects_lambda_mismatch_in_params():
    """产物 params 谎报 λ（如篡改为 0 冒充盲核运行）必须被审计拒绝。"""
    config = _config(value_guidance_strength=4.0)
    _, diagnostics = _run(config)
    forged = dict(diagnostics)
    forged["params"] = dict(
        diagnostics["params"], value_guidance_strength=0.0
    )
    with pytest.raises(
        RuntimeError, match="value_guidance_strength 与请求配置不一致"
    ):
        _audit_fitness_only_run(forged, config, "residual")


def test_audit_rejects_missing_value_guidance_diagnostics():
    """λ>0 但产物缺值引导诊断段：fail-closed 拒绝。"""
    config = _config(value_guidance_strength=4.0)
    _, diagnostics = _run(config)
    forged = dict(diagnostics)
    forged.pop("value_guidance", None)
    with pytest.raises(RuntimeError, match="值引导已请求但诊断显示未启用"):
        _audit_fitness_only_run(forged, config, "residual")


def test_audit_rejects_adaptive_scale_forgery():
    """归一化方式是 V9 型运行的科学口径核心：params 或诊断段谎报
    adaptive_scale（如手调 λ 冒充自适应）必须被审计拒绝。"""
    config = _config(
        value_guidance_strength=4.0, value_guidance_adaptive_scale=True
    )
    _, diagnostics = _run(config)

    forged_params = dict(diagnostics)
    forged_params["params"] = dict(
        diagnostics["params"], value_guidance_adaptive_scale=False
    )
    with pytest.raises(
        RuntimeError, match="value_guidance_adaptive_scale 与请求配置不一致"
    ):
        _audit_fitness_only_run(forged_params, config, "residual")

    forged_diag = dict(diagnostics)
    forged_diag["value_guidance"] = dict(
        diagnostics["value_guidance"], adaptive_scale=False
    )
    with pytest.raises(
        RuntimeError, match="诊断 adaptive_scale 与请求配置不一致"
    ):
        _audit_fitness_only_run(forged_diag, config, "residual")


def test_audit_rejects_tilt_bounds_forgery():
    """块倾斜 bounds 决定倾斜范围口径，params 谎报必须被审计拒绝。"""
    config = _config(block_score_tilt_strength=1.0)
    _, diagnostics = _run(config)
    forged = dict(diagnostics)
    forged["params"] = dict(
        diagnostics["params"], block_score_tilt_bounds=(0.1, 0.9)
    )
    with pytest.raises(
        RuntimeError, match="block_score_tilt_bounds 与请求配置不一致"
    ):
        _audit_fitness_only_run(forged, config, "residual")


def test_paired_attribution_rejects_guided_kernels():
    """配对归因合同仅对盲核定义（equal 对照臂无法带残差引导）；
    引导配置必须在入口 fail-closed 拒绝，盲核配对元数据保持历史口径。"""
    schema, queries, target = _problem()
    _, pairing = run_paired_fitness_only_attribution(
        target,
        schema=schema,
        queries=queries,
        n_records=12,
        config=_config(),
    )
    assert pairing["shared_blind_independent_kernel"] is True

    for overrides in (
        {"value_guidance_strength": 2.0},
        {"block_score_tilt_strength": 1.0},
    ):
        with pytest.raises(ValueError, match="仅对盲独立复制核定义"):
            run_paired_fitness_only_attribution(
                target,
                schema=schema,
                queries=queries,
                n_records=12,
                config=_config(**overrides),
            )
