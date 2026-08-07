"""
测试实验日志记录器
"""
import pytest
from pathlib import Path
import json
import csv
import numpy as np
from table_diffevo.experiment_logger import (
    ExperimentLogger,
    RoundLog,
    BlockLog,
    ProbeLog
)


def test_round_log_creation():
    """测试 RoundLog 创建"""
    log = RoundLog(
        seed=42,
        arm="A0",
        round=10,
        block=2,
        alpha=5.0,
        u=0.5,
        L1_current=0.05,
        best_L1=0.04,
        Q_current=1000.0,
        accepted=True,
        delta_L1=-0.01,
        delta_Q=-50.0,
        candidate_evaluations=1000
    )

    assert log.seed == 42
    assert log.arm == "A0"
    assert log.round == 10
    assert log.accepted is True


def test_block_log_creation():
    """测试 BlockLog 创建"""
    log = BlockLog(
        seed=42,
        arm="B2",
        block=5,
        block_start_L1=0.10,
        block_end_L1=0.09,
        block_improvement=0.01,
        acceptance_rate=0.75,
        stall_count=0,
        cooldown_remaining=2,
        probe_triggered=False
    )

    assert log.block == 5
    assert log.block_improvement == 0.01
    assert log.probe_triggered is False


def test_probe_log_creation():
    """测试 ProbeLog 创建"""
    log = ProbeLog(
        seed=42,
        arm="B2",
        probe_id=1,
        checkpoint_block=10,
        checkpoint_L1=0.05,
        direction="DOWN",
        branch_seed=100,
        branch_budget=200,
        branch_final_L1=0.045,
        branch_final_Q=800.0,
        winner=True,
        winner_reason="best L1 improvement"
    )

    assert log.probe_id == 1
    assert log.direction == "DOWN"
    assert log.winner is True


def test_logger_basic_flow(tmp_path):
    """测试日志记录器基本流程"""
    logger = ExperimentLogger(tmp_path)

    # 记录一轮
    logger.log_round(
        seed=42, arm="A0", round=1, block=0,
        alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
        Q_current=5000.0, accepted=True,
        delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100
    )

    # 记录一块
    logger.log_block(
        seed=42, arm="A0", block=0,
        block_start_L1=0.1, block_end_L1=0.09,
        block_improvement=0.01, acceptance_rate=0.8,
        stall_count=0, cooldown_remaining=0,
        probe_triggered=False
    )

    # 添加统计
    logger.add_stat("final_best_L1", 0.05)

    # 保存
    logger.save()

    # 验证文件存在
    assert (tmp_path / "rounds.csv").exists()
    assert (tmp_path / "blocks.csv").exists()
    assert (tmp_path / "summary.json").exists()

    # 验证内容
    with open(tmp_path / "summary.json") as f:
        stats = json.load(f)
        assert stats["final_best_L1"] == 0.05


def test_logger_csv_content(tmp_path):
    """测试 CSV 内容正确性"""
    logger = ExperimentLogger(tmp_path)

    # 记录两轮
    logger.log_round(
        seed=42, arm="A0", round=1, block=0,
        alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
        Q_current=5000.0, accepted=True,
        delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100
    )
    logger.log_round(
        seed=42, arm="A0", round=2, block=0,
        alpha=2.0, u=0.0, L1_current=0.09, best_L1=0.09,
        Q_current=4500.0, accepted=True,
        delta_L1=-0.01, delta_Q=-500.0, candidate_evaluations=200
    )

    logger.save()

    # 读取 CSV 验证
    with open(tmp_path / "rounds.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["round"] == "1"
        assert rows[1]["round"] == "2"
        assert rows[1]["delta_L1"] == "-0.01"


def test_logger_handles_numpy_types(tmp_path):
    """测试日志记录器处理 numpy 类型"""
    logger = ExperimentLogger(tmp_path)
    logger.add_stat("numpy_int", np.int64(42))
    logger.add_stat("numpy_float", np.float64(3.14))
    logger.add_stat("numpy_array", np.array([1, 2, 3]))

    logger.save()

    with open(tmp_path / "summary.json") as f:
        stats = json.load(f)
        assert stats["numpy_int"] == 42
        assert abs(stats["numpy_float"] - 3.14) < 1e-10
        assert stats["numpy_array"] == [1, 2, 3]


def test_logger_empty_logs(tmp_path):
    """测试空日志不报错"""
    logger = ExperimentLogger(tmp_path)
    logger.add_stat("test", 123)

    # 没有记录任何 round/block/probe，只保存统计
    logger.save()

    # 应该只有 summary.json
    assert (tmp_path / "summary.json").exists()
    assert not (tmp_path / "rounds.csv").exists()
    assert not (tmp_path / "blocks.csv").exists()
    assert not (tmp_path / "probes.csv").exists()


def test_logger_probe_logs(tmp_path):
    """测试探测日志完整性"""
    logger = ExperimentLogger(tmp_path)

    # 记录三个分支
    for direction, winner in [("DOWN", True), ("HOLD", False), ("UP", False)]:
        logger.log_probe(
            seed=42,
            arm="B2",
            probe_id=1,
            checkpoint_block=10,
            checkpoint_L1=0.05,
            direction=direction,
            branch_seed=100,
            branch_budget=200,
            branch_final_L1=0.045 if direction == "DOWN" else 0.048,
            branch_final_Q=800.0,
            winner=winner,
            winner_reason="best L1 improvement" if winner else None
        )

    logger.save()

    # 读取验证
    with open(tmp_path / "probes.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        # 找到获胜分支
        winners = [r for r in rows if r["winner"] == "True"]
        assert len(winners) == 1
        assert winners[0]["direction"] == "DOWN"


def test_logger_nested_dict_serialization(tmp_path):
    """测试嵌套字典的序列化"""
    logger = ExperimentLogger(tmp_path)
    logger.add_stat("nested", {
        "level1": {
            "level2": np.array([1.5, 2.5, 3.5]),
            "int": np.int32(100)
        },
        "list_of_arrays": [np.array([1, 2]), np.array([3, 4])]
    })

    logger.save()

    with open(tmp_path / "summary.json") as f:
        stats = json.load(f)
        assert stats["nested"]["level1"]["level2"] == [1.5, 2.5, 3.5]
        assert stats["nested"]["level1"]["int"] == 100
        assert stats["nested"]["list_of_arrays"] == [[1, 2], [3, 4]]


def test_directory_collision_raises_error(tmp_path):
    """测试目录碰撞时抛出异常"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    # 创建一个非空文件
    (output_dir / "existing.txt").write_text("old data")

    with pytest.raises(ValueError, match="输出目录非空"):
        ExperimentLogger(output_dir)


def test_directory_collision_with_force_overwrite(tmp_path):
    """测试 force_overwrite=True 允许覆盖非空目录"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("old data")

    # 应该不抛出异常
    logger = ExperimentLogger(output_dir, force_overwrite=True)
    assert logger.output_dir == output_dir


def test_empty_directory_accepted(tmp_path):
    """测试空目录可以正常使用"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    # 空目录应该不抛出异常
    logger = ExperimentLogger(output_dir)
    assert logger.output_dir == output_dir


def test_nonexistent_directory_created(tmp_path):
    """测试不存在的目录会被创建"""
    output_dir = tmp_path / "test_output"

    logger = ExperimentLogger(output_dir)
    assert output_dir.exists()
    assert logger.output_dir == output_dir


def test_numpy_bool_serialization(tmp_path):
    """测试 numpy bool 类型的序列化"""
    logger = ExperimentLogger(tmp_path)

    # 添加包含 numpy bool 的统计信息
    logger.add_stat("numpy_true", np.bool_(True))
    logger.add_stat("numpy_false", np.bool_(False))
    logger.save()

    # 验证保存的 JSON
    stats_file = tmp_path / "summary.json"
    with open(stats_file) as f:
        data = json.load(f)

    assert data["numpy_true"] is True
    assert data["numpy_false"] is False
    assert isinstance(data["numpy_true"], bool)
    assert isinstance(data["numpy_false"], bool)


def test_nan_infinity_serialization(tmp_path):
    """测试 NaN 和 Infinity 的序列化"""
    logger = ExperimentLogger(tmp_path)

    # 添加非有限值
    logger.add_stat("nan_value", float('nan'))
    logger.add_stat("inf_value", float('inf'))
    logger.add_stat("neg_inf_value", float('-inf'))
    logger.add_stat("numpy_nan", np.float64('nan'))
    logger.add_stat("numpy_inf", np.float64('inf'))
    logger.save()

    # 验证保存的 JSON
    stats_file = tmp_path / "summary.json"
    with open(stats_file) as f:
        data = json.load(f)

    # 非有限值应该被转换为 None
    assert data["nan_value"] is None
    assert data["inf_value"] is None
    assert data["neg_inf_value"] is None
    assert data["numpy_nan"] is None
    assert data["numpy_inf"] is None


def test_atomic_write_protects_existing_file(tmp_path):
    """测试原子写入在失败时不破坏既有文件"""
    logger = ExperimentLogger(tmp_path)

    # 先保存一次正常数据
    logger.add_stat("value", 42)
    logger.save()

    # 读取原始内容
    stats_file = tmp_path / "summary.json"
    original_content = stats_file.read_text()

    # 创建一个新的 logger 实例，并尝试保存不可序列化的数据
    logger2 = ExperimentLogger(tmp_path, force_overwrite=True)
    logger2.add_stat("normal", 123)

    # 手动注入一个复杂对象（绕过 _make_serializable）
    class UnserializableObject:
        pass
    logger2.stats["bad"] = UnserializableObject()

    # 保存应该失败
    with pytest.raises(ValueError, match="统计信息无法序列化为 JSON"):
        logger2.save()

    # 验证原始文件未被破坏
    assert stats_file.read_text() == original_content


def test_no_temp_file_residue_after_success(tmp_path):
    """测试成功写入后不留下临时文件"""
    logger = ExperimentLogger(tmp_path)

    logger.log_round(
        seed=42, arm="A0", round=1, block=0,
        alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
        Q_current=5000.0, accepted=True,
        delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100
    )
    logger.add_stat("count", 100)
    logger.save()

    # 检查没有 .tmp 文件残留
    temp_files = list(tmp_path.glob("*.tmp"))
    assert len(temp_files) == 0


def test_nested_numpy_types(tmp_path):
    """测试嵌套结构中的 numpy 类型"""
    logger = ExperimentLogger(tmp_path)

    # 嵌套结构
    logger.add_stat("nested", {
        "bool": np.bool_(True),
        "int": np.int64(42),
        "float": np.float64(3.14),
        "list": [np.int32(1), np.int32(2)],
        "nan": np.float64('nan')
    })
    logger.save()

    stats_file = tmp_path / "summary.json"
    with open(stats_file) as f:
        data = json.load(f)

    assert data["nested"]["bool"] is True
    assert data["nested"]["int"] == 42
    assert data["nested"]["float"] == 3.14
    assert data["nested"]["list"] == [1, 2]
    assert data["nested"]["nan"] is None


def test_bad_summary_leaves_no_half_baked_csv(tmp_path):
    """回归：summary 序列化失败时，不得发布半成品 rounds.csv。

    save() 必须先构造并校验 summary（在发布任何 CSV 之前），因此一个
    无法序列化的 summary 应让整个 save() 失败且不留下任何正式文件。
    """
    logger = ExperimentLogger(tmp_path)

    # 有正常的 round 日志（本应写成 rounds.csv）
    logger.log_round(
        seed=42, arm="A0", round=1, block=0,
        alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
        Q_current=5000.0, accepted=True,
        delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100,
    )

    # 但 summary 含不可序列化对象
    class Unserializable:
        pass
    logger.stats["bad"] = Unserializable()

    with pytest.raises(ValueError, match="统计信息无法序列化为 JSON"):
        logger.save()

    # 关键断言：不得留下半成品 rounds.csv，也不得留下 summary.json 或临时文件
    assert not (tmp_path / "rounds.csv").exists()
    assert not (tmp_path / "summary.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_force_overwrite_clears_stale_category(tmp_path):
    """回归：force_overwrite 复用目录时，须清理本次未写入的陈旧类别文件。

    第一次运行有 probe 日志（产出 probes.csv）；第二次运行没有 probe，
    则旧的 probes.csv 不应残留，避免分析脚本读到过期数据。
    """
    # 第一次：含 probe 日志
    logger1 = ExperimentLogger(tmp_path)
    logger1.log_round(
        seed=42, arm="B2", round=1, block=0,
        alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
        Q_current=5000.0, accepted=True,
        delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100,
    )
    logger1.log_probe(
        seed=42, arm="B2", probe_id=0, checkpoint_block=1,
        checkpoint_L1=0.1, direction="UP", branch_seed=None,
        branch_budget=50, branch_final_L1=0.08, branch_final_Q=4000.0,
        winner=True, winner_reason="l1_improvement",
    )
    logger1.add_stat("run", 1)
    logger1.save()
    assert (tmp_path / "probes.csv").exists()

    # 第二次：复用同目录，但没有 probe 日志
    logger2 = ExperimentLogger(tmp_path, force_overwrite=True)
    logger2.log_round(
        seed=43, arm="B2", round=1, block=0,
        alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
        Q_current=5000.0, accepted=True,
        delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100,
    )
    logger2.add_stat("run", 2)
    logger2.save()

    # 陈旧的 probes.csv 应被清理；rounds/summary 应为本次内容
    assert not (tmp_path / "probes.csv").exists()
    assert (tmp_path / "rounds.csv").exists()
    with open(tmp_path / "summary.json") as f:
        assert json.load(f)["run"] == 2


def test_array_with_non_finite_serialized(tmp_path):
    """回归：ndarray 内的 NaN/Infinity 也要转成 None（递归处理）。

    此前 ndarray 分支直接 tolist() 未递归，导致数组内的非有限值写出
    非法 JSON。现在应递归转换，且 allow_nan=False 兜底不会写出 NaN。
    """
    logger = ExperimentLogger(tmp_path)
    logger.add_stat("arr", np.array([1.0, float('nan'), float('inf'), float('-inf'), 2.0]))
    logger.add_stat("nested_arr", {"inner": np.array([float('nan'), 3.0])})
    logger.save()

    stats_file = tmp_path / "summary.json"
    # 文件内容必须是合法 JSON（不含裸 NaN/Infinity）
    raw = stats_file.read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw

    with open(stats_file) as f:
        data = json.load(f)
    assert data["arr"] == [1.0, None, None, None, 2.0]
    assert data["nested_arr"]["inner"] == [None, 3.0]
