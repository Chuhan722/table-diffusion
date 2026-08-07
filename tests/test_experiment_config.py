"""
测试实验配置管理
"""
import pytest
from pathlib import Path
from table_diffevo.experiment_config import (
    ExperimentConfig,
    AcceptanceRuleConfig,
    AlphaScheduleConfig,
    DataConfig
)


def test_config_validation_a0_valid():
    """A0 规则的有效配置"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    # 应该不抛出异常
    config.validate()


def test_config_validation_a1_requires_eps_l1():
    """A1 规则必须指定 eps_L1"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A1", eps_L1=None),  # 缺少 eps_L1
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    with pytest.raises(ValueError, match="A1 需要指定 eps_L1"):
        config.validate()


def test_config_validation_a1_with_eps_l1_valid():
    """A1 规则指定 eps_L1 后有效"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A1", eps_L1=1e-5, eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    # 应该不抛出异常
    config.validate()


def test_config_validation_fixed_alpha_requires_value():
    """fixed 模式必须指定 alpha_value"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=None),  # 缺少值
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    with pytest.raises(ValueError, match="fixed 模式需要指定 alpha_value"):
        config.validate()


def test_config_validation_probe_requires_w():
    """probe 模式必须指定 probe_block_candidate_budget"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="probe", probe_block_candidate_budget=None),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    with pytest.raises(ValueError, match="probe 模式需要指定 probe_block_candidate_budget"):
        config.validate()


def test_config_validation_empty_seeds():
    """种子列表不能为空"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[],  # 空列表
        n_rounds=100,
        output_dir="output"
    )

    with pytest.raises(ValueError, match="至少需要 1 个 seed"):
        config.validate()


def test_config_validation_invalid_n_rounds():
    """n_rounds 必须为正数"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=0,  # 非正数
        output_dir="output"
    )

    with pytest.raises(ValueError, match="n_rounds 必须 > 0"):
        config.validate()


def test_config_validation_invalid_alpha_range():
    """alpha_min 必须小于 alpha_max"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(
            mode="fixed",
            alpha_value=5.0,
            alpha_min=10.0,
            alpha_max=2.0  # min > max
        ),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    with pytest.raises(ValueError, match="alpha_min 必须小于 alpha_max"):
        config.validate()


def test_config_yaml_roundtrip(tmp_path):
    """测试 YAML 序列化/反序列化"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42, 43],
        n_rounds=100,
        output_dir="output"
    )

    # 保存
    yaml_path = tmp_path / "config.yaml"
    config.to_yaml(yaml_path)

    # 加载
    loaded = ExperimentConfig.from_yaml(yaml_path)

    # 验证
    assert loaded.experiment_name == config.experiment_name
    assert loaded.seeds == config.seeds
    assert loaded.acceptance_rule.rule == config.acceptance_rule.rule
    assert loaded.alpha_schedule.mode == config.alpha_schedule.mode
    assert loaded.alpha_schedule.alpha_value == config.alpha_schedule.alpha_value
    assert loaded.data.dataset_name == config.data.dataset_name
    assert loaded.data.n_records == config.data.n_records


def test_config_yaml_with_probe_mode(tmp_path):
    """测试 probe 模式的 YAML 序列化"""
    config = ExperimentConfig(
        experiment_name="test_probe",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A1", eps_L1=1e-5, eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(
            mode="probe",
            probe_block_candidate_budget=20,
            probe_P=3,
            probe_H_candidate_budget=2,
            probe_s=0.10,
            probe_C=2
        ),
        seeds=[42],
        n_rounds=500,
        output_dir="output"
    )

    # 保存
    yaml_path = tmp_path / "probe_config.yaml"
    config.to_yaml(yaml_path)

    # 加载
    loaded = ExperimentConfig.from_yaml(yaml_path)

    # 验证 probe 参数
    assert loaded.alpha_schedule.mode == "probe"
    assert loaded.alpha_schedule.probe_block_candidate_budget == 20
    assert loaded.alpha_schedule.probe_P == 3
    assert loaded.alpha_schedule.probe_H_candidate_budget == 2
    assert loaded.alpha_schedule.probe_s == 0.10
    assert loaded.alpha_schedule.probe_C == 2


def test_config_default_evolution_params():
    """测试默认演化参数"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    # 验证默认值
    assert config.beta == 1.0
    assert config.eta == 0.5
    assert config.h == 0.8
    assert config.mu == 0.01
    assert config.lambda_ == 0.5
    assert config.delta == 0.05
    assert config.winsorize_limits == (0.01, 0.99)
    assert config.rho == 0.01  # 新增默认值


def test_config_validation_negative_epsilon():
    """eps_L1 和 eps_Q 不能为负"""
    # 负 eps_L1
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A1", eps_L1=-0.01, eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="eps_L1 必须 >= 0"):
        config.validate()

    # 负 eps_Q
    config2 = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=-1.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="eps_Q 必须 >= 0"):
        config2.validate()


def test_config_validation_zero_n_records():
    """n_records 必须 > 0"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=0  # 非法值
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="n_records 必须 > 0"):
        config.validate()


def test_config_validation_probe_params():
    """probe 模式参数校验"""
    # probe_block_candidate_budget = 0
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="probe", probe_block_candidate_budget=0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="probe_block_candidate_budget 必须 > 0"):
        config.validate()

    # probe_s = 2.0（超出范围）
    config2 = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="probe", probe_block_candidate_budget=20, probe_s=2.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="probe_s 必须在 \\(0, 1\\) 范围内"):
        config2.validate()


def test_config_validation_unknown_rule():
    """未知接受规则"""
    # 注意：由于 Literal 类型，这个在运行时不会自动拒绝
    # 但 validate() 应该能捕获
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    # 手动设置非法值（绕过类型检查）
    config.acceptance_rule.rule = "A99"
    with pytest.raises(ValueError, match="未知接受规则"):
        config.validate()


def test_config_yaml_unknown_key(tmp_path):
    """YAML 包含未知键应拒绝"""
    yaml_path = tmp_path / "bad_config.yaml"
    with open(yaml_path, 'w') as f:
        f.write("""
experiment_name: test
data:
  dataset_name: nltcs
  init_marginals_path: data/marginals.json
  n_records: 16181
acceptance_rule:
  rule: A0
  eps_Q: 0.0
alpha_schedule:
  mode: fixed
  alpha_value: 5.0
seeds: [42]
n_rounds: 100
output_dir: output
unknown_field: 123  # 未知键
""")

    with pytest.raises(ValueError, match="配置文件包含未知键"):
        ExperimentConfig.from_yaml(yaml_path)


def test_config_validation_negative_candidate_budget():
    """candidate_budget 若指定则必须 > 0"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output",
        candidate_budget=-1,  # 非法值
    )
    with pytest.raises(ValueError, match="candidate_budget 若指定则必须 > 0"):
        config.validate()


def test_config_validation_unknown_device():
    """device 必须在白名单内"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            init_marginals_path="data/marginals.json",
            n_records=16181,
            device="not-a-device",  # 非法值
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output",
    )
    with pytest.raises(ValueError, match="未知 device"):
        config.validate()


def test_config_validation_valid_devices():
    """cpu/cuda/numpy 三个合法设备都应通过"""
    for dev in ("cpu", "cuda", "numpy"):
        config = ExperimentConfig(
            experiment_name="test",
            data=DataConfig(
                dataset_name="nltcs",
                init_marginals_path="data/marginals.json",
                n_records=16181,
                device=dev,
            ),
            acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
            alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
            seeds=[42],
            n_rounds=100,
            output_dir="output",
        )
        config.validate()  # 不应抛出


def _write_minimal_data_files(tmp_path):
    """在 tmp 目录写一份最小 schema + query 文件，返回两者路径。

    自包含，不依赖仓库内某个数据集是否存在。
    """
    import json
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "attributes:\n"
        "- name: attr_1\n"
        "  type: categorical\n"
        "  values: [0, 1]\n"
        "- name: attr_2\n"
        "  type: categorical\n"
        "  values: [0, 1]\n"
    )
    query_path = tmp_path / "queries.json"
    queries = {
        "queries": [
            {"id": "Q1", "conditions": [
                {"attribute": "attr_1", "operator": "==", "value": 0}], "result": 10},
            {"id": "Q2", "conditions": [
                {"attribute": "attr_2", "operator": "==", "value": 1}], "result": 7},
        ]
    }
    query_path.write_text(json.dumps(queries))
    return schema_path, query_path


def test_to_run_evolution_kwargs_covers_signature(tmp_path):
    """to_run_evolution_kwargs 应覆盖 run_evolution 的所有必填参数，
    且不产生签名之外的未知键（防止映射漂移）。"""
    import inspect
    from table_diffevo.evolution import run_evolution

    schema_path, query_path = _write_minimal_data_files(tmp_path)
    config = ExperimentConfig(
        experiment_name="map_test",
        data=DataConfig(
            dataset_name="toy",
            schema_path=str(schema_path),
            query_path=str(query_path),
            init_marginals_path="",
            n_records=100,
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="fixed", alpha_value=5.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output",
    )

    kwargs = config.to_run_evolution_kwargs(seed=42)

    sig = inspect.signature(run_evolution)
    params = sig.parameters
    # 所有无默认值的必填参数都必须被提供
    required = [
        name for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [name for name in required if name not in kwargs]
    assert not missing, f"缺少 run_evolution 必填参数: {missing}"

    # 不能有 run_evolution 签名之外的未知键
    unknown = [k for k in kwargs if k not in params]
    assert not unknown, f"kwargs 含 run_evolution 未知键: {unknown}"


def test_to_run_evolution_kwargs_maps_values(tmp_path):
    """to_run_evolution_kwargs 应把配置字段的值正确映射到对应参数。"""
    schema_path, query_path = _write_minimal_data_files(tmp_path)
    config = ExperimentConfig(
        experiment_name="map_test",
        data=DataConfig(
            dataset_name="toy",
            schema_path=str(schema_path),
            query_path=str(query_path),
            init_marginals_path="",
            n_records=100,
            device="cpu",
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(
            mode="fixed", alpha_value=5.0, alpha_min=3.0, alpha_max=9.0),
        seeds=[7],
        n_rounds=123,
        output_dir="output",
        rho=0.02,
        candidate_budget=456,
    )

    kwargs = config.to_run_evolution_kwargs(seed=7)

    # 数据类字段
    assert kwargs["n_records"] == 100
    assert kwargs["device"] == "cpu"
    # 实验字段
    assert kwargs["n_rounds"] == 123
    assert kwargs["rho"] == 0.02
    assert kwargs["seed"] == 7
    assert kwargs["candidate_budget"] == 456
    # alpha 调度字段
    assert kwargs["alpha_min"] == 3.0
    assert kwargs["alpha_max"] == 9.0
    # target 从 query 的 result 派生
    assert list(kwargs["target"]) == [10, 7]
