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


@pytest.mark.parametrize("section,body", [
    (
        "data",
        """
experiment_name: t
data:
  dataset_name: nltcs
  init_marginals_path: x
  n_records: 100
  bogus_nested: 1
acceptance_rule: {rule: A0, eps_Q: 0.0}
alpha_schedule: {mode: fixed, alpha_value: 5.0}
seeds: [42]
n_rounds: 100
output_dir: out
""",
    ),
    (
        "acceptance_rule",
        """
experiment_name: t
data: {dataset_name: nltcs, init_marginals_path: x, n_records: 100}
acceptance_rule: {rule: A0, eps_Q: 0.0, typo_key: 1}
alpha_schedule: {mode: fixed, alpha_value: 5.0}
seeds: [42]
n_rounds: 100
output_dir: out
""",
    ),
    (
        "alpha_schedule",
        """
experiment_name: t
data: {dataset_name: nltcs, init_marginals_path: x, n_records: 100}
acceptance_rule: {rule: A0, eps_Q: 0.0}
alpha_schedule: {mode: fixed, alpha_value: 5.0, W: 20}
seeds: [42]
n_rounds: 100
output_dir: out
""",
    ),
])
def test_config_yaml_nested_unknown_key(tmp_path, section, body):
    """嵌套节里的未知键也应被拒（干净的 ValueError，而非原始 TypeError）。

    顶层未知键早有护栏，但嵌套 dict（data/acceptance_rule/alpha_schedule）里
    拼错的键会被 DataConfig(**...) 等原样展开抛 TypeError，定位差、不符合
    “拒绝未知 YAML 键”的契约（见 #36 复现）。这里锚定三个节都给出带节名的
    ValueError。alpha_schedule 用旧字段名 `W`（现为 probe_block_candidate_budget）
    模拟真实迁移遗漏。
    """
    yaml_path = tmp_path / "bad_nested.yaml"
    yaml_path.write_text(body)
    with pytest.raises(ValueError, match=rf"配置节 '{section}' 包含未知键"):
        ExperimentConfig.from_yaml(yaml_path)


def test_config_yaml_nested_section_wrong_type(tmp_path):
    """嵌套节若不是映射（写成标量/列表）应给出清晰 ValueError，而非晦涩的展开错误。"""
    yaml_path = tmp_path / "bad_type.yaml"
    yaml_path.write_text("""
experiment_name: t
data: "should-be-a-mapping"
acceptance_rule: {rule: A0, eps_Q: 0.0}
alpha_schedule: {mode: fixed, alpha_value: 5.0}
seeds: [42]
n_rounds: 100
output_dir: out
""")
    with pytest.raises(ValueError, match=r"配置节 'data' 必须是键值映射"):
        ExperimentConfig.from_yaml(yaml_path)


@pytest.mark.parametrize("section_line", [
    "data:",                 # 显式为空（YAML null）
    "data: null",            # 显式 null
    # 整节缺失由下方单独用例覆盖（无法用一行替换表达）
])
def test_config_yaml_nested_section_empty(tmp_path, section_line):
    """嵌套节写成空/null 应给出清晰 ValueError，而非 DataConfig(**None) 的晦涩 TypeError。

    与 test_config_yaml_nested_section_wrong_type 同源：坏 YAML 必须落到带节名的
    ValueError。原实现对 None 直接 continue，会把空节漏到下方展开处抛 TypeError。
    """
    yaml_path = tmp_path / "empty_section.yaml"
    yaml_path.write_text(f"""
experiment_name: t
{section_line}
acceptance_rule: {{rule: A0, eps_Q: 0.0}}
alpha_schedule: {{mode: fixed, alpha_value: 5.0}}
seeds: [42]
n_rounds: 100
output_dir: out
""")
    with pytest.raises(ValueError, match=r"配置节 'data' 缺失或为空"):
        ExperimentConfig.from_yaml(yaml_path)


def test_config_yaml_nested_section_missing(tmp_path):
    """整个必填嵌套节缺失应给出清晰 ValueError，而非 KeyError。"""
    yaml_path = tmp_path / "missing_section.yaml"
    yaml_path.write_text("""
experiment_name: t
acceptance_rule: {rule: A0, eps_Q: 0.0}
alpha_schedule: {mode: fixed, alpha_value: 5.0}
seeds: [42]
n_rounds: 100
output_dir: out
""")
    with pytest.raises(ValueError, match=r"配置节 'data' 缺失或为空"):
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


def _make_config(tmp_path, *, rule="A0", mode="round_schedule", **overrides):
    """构造一个最小合法 ExperimentConfig，供 fail-closed 测试复用。"""
    schema_path, query_path = _write_minimal_data_files(tmp_path)
    kwargs = dict(
        experiment_name="map_test",
        data=DataConfig(
            dataset_name="toy",
            schema_path=str(schema_path),
            query_path=str(query_path),
            init_marginals_path="",
            n_records=100,
        ),
        acceptance_rule=AcceptanceRuleConfig(rule=rule, eps_L1=1e-5, eps_Q=0.0),
        alpha_schedule=AlphaScheduleConfig(
            mode=mode,
            alpha_value=5.0 if mode == "fixed" else None,
            probe_block_candidate_budget=20 if mode == "probe" else None,
        ),
        seeds=[42],
        n_rounds=100,
        output_dir="output",
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_to_run_evolution_kwargs_fail_closed_on_acceptance_rule(tmp_path):
    """阶段 0：A0/A1 尚未接入主循环，to_run_evolution_kwargs 必须报错而非静默运行。

    锚定 fail-closed 契约——绝不让配置了 A0/A1 的实验静默按默认判据跑。
    真正的接受规则接入留待阶段 1（PR #37）。
    """
    for rule in ("A0", "A1"):
        config = _make_config(tmp_path, rule=rule, mode="round_schedule")
        with pytest.raises(NotImplementedError, match="acceptance_rule"):
            config.to_run_evolution_kwargs(seed=42)


def test_to_run_evolution_kwargs_fail_closed_on_alpha_mode(tmp_path):
    """阶段 0：fixed/probe α 调度尚未接入主循环，必须报错而非错误映射。

    锚定 fail-closed 契约——fixed 不能被当成线性轮数调度静默跑（旧 bug），
    probe 的探测语义更未实现。真正接入留待阶段 2-5。
    """
    for mode in ("fixed", "probe"):
        config = _make_config(tmp_path, rule="A0", mode=mode)
        with pytest.raises(NotImplementedError, match="alpha_schedule.mode"):
            config.to_run_evolution_kwargs(seed=42)


def test_to_run_evolution_kwargs_fail_closed_is_whitelist_not_blacklist(tmp_path):
    """fail-closed 用白名单判定：非法/未预期的口径值也必须被挡下，不能 fail-open。

    防回归——若 guard 写成黑名单（`if rule in ("A0","A1")`），一个拼错的规则值
    会溜过护栏、用主循环默认判据静默运行。这里直接构造非白名单值验证被拒。
    绕过 validate()（模拟直接调用 to_run_evolution_kwargs 的路径）。
    """
    # 非白名单的接受规则值（round_schedule 已接入，落到规则护栏）
    config = _make_config(tmp_path, rule="A0", mode="round_schedule")
    config.acceptance_rule.rule = "GARBAGE"
    with pytest.raises(NotImplementedError, match="acceptance_rule"):
        config.to_run_evolution_kwargs(seed=42)

    # 非白名单的 α 模式值
    config2 = _make_config(tmp_path, rule="A0", mode="round_schedule")
    config2.alpha_schedule.mode = "some_future_mode"
    with pytest.raises(NotImplementedError, match="alpha_schedule.mode"):
        config2.to_run_evolution_kwargs(seed=42)


def test_to_run_evolution_kwargs_fails_before_file_io(tmp_path):
    """fail-closed 护栏必须在加载 schema/queries 等文件之前触发。

    guard 早于 I/O，既能给出清晰的“未接入”信息，也避免对不存在的数据文件
    先报一个误导性的 FileNotFoundError。用一个指向不存在文件的配置验证：
    应抛 NotImplementedError（护栏），而不是文件相关错误。
    """
    config = _make_config(tmp_path, rule="A0", mode="round_schedule")
    config.data.schema_path = "/nonexistent/schema.json"
    config.data.query_path = "/nonexistent/queries.json"
    with pytest.raises(NotImplementedError):
        config.to_run_evolution_kwargs(seed=42)


# 仓库根目录（tests/ 的上一级）
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_example_yamls():
    """收集仓库内随包提供的示例实验配置 YAML。"""
    return sorted((_REPO_ROOT / "experiments" / "configs").glob("*.yaml"))


def test_repo_example_configs_exist():
    """至少应存在一份示例配置（否则下面的加载测试会静默空跑）。"""
    yamls = _repo_example_yamls()
    assert yamls, "experiments/configs/ 下未找到任何示例 YAML"


@pytest.mark.parametrize(
    "yaml_path", _repo_example_yamls(), ids=lambda p: p.name
)
def test_repo_example_config_loads_and_validates(yaml_path):
    """逐个加载仓库内示例 YAML：from_yaml 内部会跑 validate()。

    回归 PR #36 第四轮问题 3——示例配置曾引用已删字段（target_path）导致
    加载即抛 TypeError。此测试确保随仓库提供的每个示例都能被真实加载并通过
    校验，示例与当前 schema 不脱节。
    """
    config = ExperimentConfig.from_yaml(yaml_path)
    # from_yaml 已调用 validate()；再断言关键字段被正确解析。
    assert config.experiment_name
    assert config.acceptance_rule.rule in ("A0", "A1")
    assert config.alpha_schedule.mode in ("fixed", "round_schedule", "probe")
