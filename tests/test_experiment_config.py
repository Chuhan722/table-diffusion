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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
    """probe 模式必须指定 W"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            target_path="data/target.json",
            measured_target_path="data/measured.json",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="probe", W=None),  # 缺少 W
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )

    with pytest.raises(ValueError, match="probe 模式需要指定 W"):
        config.validate()


def test_config_validation_empty_seeds():
    """种子列表不能为空"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A1", eps_L1=1e-5, eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(
            mode="probe",
            W=20,
            P=3,
            H=2,
            s=0.10,
            C=2
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
    assert loaded.alpha_schedule.W == 20
    assert loaded.alpha_schedule.P == 3
    assert loaded.alpha_schedule.H == 2
    assert loaded.alpha_schedule.s == 0.10
    assert loaded.alpha_schedule.C == 2


def test_config_default_evolution_params():
    """测试默认演化参数"""
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
    # W = 0
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            target_path="data/target.json",
            measured_target_path="data/measured.json",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="probe", W=0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="W 必须 > 0"):
        config.validate()

    # s = 2.0（超出范围）
    config2 = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            target_path="data/target.json",
            measured_target_path="data/measured.json",
            init_marginals_path="data/marginals.json",
            n_records=16181
        ),
        acceptance_rule=AcceptanceRuleConfig(rule="A0", eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(mode="probe", W=20, s=2.0),
        seeds=[42],
        n_rounds=100,
        output_dir="output"
    )
    with pytest.raises(ValueError, match="s 必须在 \\(0, 1\\) 范围内"):
        config2.validate()


def test_config_validation_unknown_rule():
    """未知接受规则"""
    # 注意：由于 Literal 类型，这个在运行时不会自动拒绝
    # 但 validate() 应该能捕获
    config = ExperimentConfig(
        experiment_name="test",
        data=DataConfig(
            dataset_name="nltcs",
            target_path="data/target.json",
            measured_target_path="data/measured.json",
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
  target_path: data/target.json
  measured_target_path: data/measured.json
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

