"""
测试结果落盘

锚定 save_run 的文件结构、内容一致性、重名后缀递增。
"""
import os
import json
import numpy as np
import pandas as pd
import pytest
from table_diffevo.io import (
    save_run, _make_run_dir, create_parent_dir, save_summary,
)
from datetime import datetime


def make_result():
    """构造一份 best_S 和 diagnostics"""
    best_S = pd.DataFrame({
        "age": [20, 30, 40],
        "edu": ["low", "mid", "high"],
    })
    diagnostics = {
        "loss_history": [100.0, 80.0, 60.0],
        "best_loss": 60.0,
        "rounds_run": 3,
        "stopped_early": False,
        "accept_history": [True, True, False],
    }
    return best_S, diagnostics


class TestSaveRun:
    """保存主流程"""

    def test_creates_folder_and_files(self, tmp_path):
        """保存后文件夹和两个文件都存在"""
        best_S, diag = make_result()
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))
        assert os.path.isdir(run_dir)
        assert os.path.isfile(os.path.join(run_dir, "best_synthetic.csv"))
        assert os.path.isfile(os.path.join(run_dir, "diagnostics.json"))

    def test_csv_content_matches(self, tmp_path):
        """CSV 读回内容与 best_S 一致"""
        best_S, diag = make_result()
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))
        loaded = pd.read_csv(os.path.join(run_dir, "best_synthetic.csv"))
        pd.testing.assert_frame_equal(loaded, best_S)

    def test_json_content_matches(self, tmp_path):
        """JSON 读回内容与 diagnostics 一致"""
        best_S, diag = make_result()
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))
        with open(os.path.join(run_dir, "diagnostics.json"), encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == diag

    def test_returns_folder_path(self, tmp_path):
        """返回实际文件夹路径"""
        best_S, diag = make_result()
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))
        assert str(tmp_path) in run_dir


class TestFolderNaming:
    """文件夹命名与后缀"""

    def test_suffix_starts_at_zero(self, tmp_path):
        """首个文件夹后缀为 _0"""
        best_S, diag = make_result()
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))
        assert os.path.basename(run_dir).endswith("_0")

    def test_suffix_increments_on_collision(self, tmp_path):
        """连续保存两次，后缀递增 _0、_1"""
        best_S, diag = make_result()
        # 用固定时间强制重名
        now = datetime(2026, 7, 21, 14, 30)
        d0 = _make_run_dir(str(tmp_path), now)
        d1 = _make_run_dir(str(tmp_path), now)
        assert os.path.basename(d0) == "2026-07-21_1430_0"
        assert os.path.basename(d1) == "2026-07-21_1430_1"

    def test_name_format(self, tmp_path):
        """文件夹名格式 YYYY-MM-DD_HHMM_N"""
        now = datetime(2026, 1, 5, 9, 3)
        run_dir = _make_run_dir(str(tmp_path), now)
        assert os.path.basename(run_dir) == "2026-01-05_0903_0"


class TestExplicitRunDir:
    """save_run 传入 run_dir 时直接写入该目录（多种子场景）"""

    def test_writes_to_given_dir(self, tmp_path):
        """指定 run_dir 时结果写入该目录，不自建时间文件夹"""
        best_S, diag = make_result()
        target = str(tmp_path / "2026-07-24_0411" / "0-0")
        run_dir = save_run(best_S, diag, run_dir=target)
        assert run_dir == target
        assert os.path.isfile(os.path.join(target, "best_synthetic.csv"))
        assert os.path.isfile(os.path.join(target, "diagnostics.json"))

    def test_creates_nested_dir(self, tmp_path):
        """run_dir 的父目录不存在时也能创建"""
        best_S, diag = make_result()
        target = str(tmp_path / "a" / "b" / "c")
        save_run(best_S, diag, run_dir=target)
        assert os.path.isdir(target)

    def test_none_keeps_old_behavior(self, tmp_path):
        """run_dir=None 时沿用旧行为（自建带后缀的时间文件夹）"""
        best_S, diag = make_result()
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path), run_dir=None)
        assert os.path.basename(run_dir).endswith("_0")


class TestMultiSeedLayout:
    """多种子父文件夹 + summary"""

    def test_parent_dir_no_suffix(self, tmp_path):
        """父文件夹按时间命名、不加数字后缀"""
        parent = create_parent_dir(outputs_dir=str(tmp_path))
        assert os.path.isdir(parent)
        # 名字形如 YYYY-MM-DD_HHMM，末尾不是 _0/_1
        assert not os.path.basename(parent)[-2:].startswith("_")

    def test_seed_subdirs_layout(self, tmp_path):
        """父/{顺序}-{种子}/ 结构成立"""
        best_S, diag = make_result()
        parent = create_parent_dir(outputs_dir=str(tmp_path))
        for i, seed in enumerate([0, 1, 2]):
            save_run(best_S, diag,
                     run_dir=os.path.join(parent, f"{i}-{seed}"))
        assert sorted(os.listdir(parent)) == ["0-0", "1-1", "2-2"]

    def test_save_summary(self, tmp_path):
        """summary.json 写入父文件夹且内容可读回"""
        parent = create_parent_dir(outputs_dir=str(tmp_path))
        summary = {
            "seeds": [0, 1],
            "aggregate": {"best_loss": {"mean": 1.0, "std": 0.1,
                                        "min": 0.9, "max": 1.1}},
        }
        path = save_summary(parent, summary)
        assert path == os.path.join(parent, "summary.json")
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == summary


class TestNumpyTypes:
    """numpy 类型可序列化"""

    def test_numpy_types_in_diagnostics(self, tmp_path):
        """diagnostics 含 numpy 类型也能存"""
        best_S = pd.DataFrame({"x": [1, 2]})
        diag = {
            "best_loss": np.float64(42.5),
            "rounds_run": np.int64(10),
            "stopped_early": np.bool_(True),
            "loss_history": np.array([1.0, 2.0]),
        }
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))
        with open(os.path.join(run_dir, "diagnostics.json"), encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["best_loss"] == 42.5
        assert loaded["rounds_run"] == 10
        assert loaded["stopped_early"] is True
        assert loaded["loss_history"] == [1.0, 2.0]


class TestIntegration:
    """与主循环端到端"""

    def test_save_real_run(self, tmp_path):
        """真实跑一次演化并保存"""
        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_queries
        from table_diffevo.evolution import run_evolution

        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        target = np.array([q["result"] for q in queries])

        best_S, diag = run_evolution(
            target, queries, schema, n_records=300, n_rounds=10, seed=0
        )
        run_dir = save_run(best_S, diag, outputs_dir=str(tmp_path))

        # 合成表可读回，形状正确
        loaded = pd.read_csv(os.path.join(run_dir, "best_synthetic.csv"))
        assert loaded.shape == (300, 10)
        # 诊断可读回
        with open(os.path.join(run_dir, "diagnostics.json"), encoding="utf-8") as f:
            loaded_diag = json.load(f)
        assert len(loaded_diag["loss_history"]) == diag["rounds_run"]
