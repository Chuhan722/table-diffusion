"""necessity ablation 协议脚本的正式身份判定测试（第四轮审查）。"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "probe_necessity_ablation.py"
)


@pytest.fixture(scope="module")
def ablation_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "probe_necessity_ablation_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module, monkeypatch, tmp_path, argv, dirty):
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["probe_necessity_ablation.py", "--output", str(out)] + argv,
    )
    monkeypatch.setattr(
        module, "_environment",
        lambda: {
            "git_commit": "test",
            "git_worktree_clean_including_untracked": not dirty,
            "python": "x", "numpy": "x", "pandas": "x",
            "platform": "x", "argv": sys.argv,
            "started_at": "now",
        },
    )
    # 只验证正式身份判定，不真正跑实验：数据集清空（formal 的
    # datasets 比较对空集自动成立，seeds/rounds 判定不受影响）。
    monkeypatch.setattr(module, "DATASETS", {})
    module.main()
    import json
    return json.loads(out.read_text())


def test_allow_dirty_forces_informal_even_with_formal_params(
    ablation_module, monkeypatch, tmp_path
):
    """--allow-dirty 时即使 seeds/rounds/datasets 与预注册一致也必须
    formal_protocol=false（第四轮审查意见 2）。"""
    module = ablation_module
    payload = _run_main(
        module, monkeypatch, tmp_path,
        argv=["--allow-dirty"],
        dirty=True,
    )
    assert payload["formal_protocol"] is False


def test_clean_tree_with_formal_params_is_formal(
    ablation_module, monkeypatch, tmp_path
):
    module = ablation_module
    payload = _run_main(module, monkeypatch, tmp_path, argv=[], dirty=False)
    assert payload["formal_protocol"] is True


def test_dirty_tree_without_allow_dirty_refuses(
    ablation_module, monkeypatch, tmp_path
):
    module = ablation_module
    with pytest.raises(RuntimeError, match="干净"):
        _run_main(module, monkeypatch, tmp_path, argv=[], dirty=True)


def test_non_formal_seeds_marked_informal(
    ablation_module, monkeypatch, tmp_path
):
    module = ablation_module
    payload = _run_main(
        module, monkeypatch, tmp_path,
        argv=["--seeds", "42"],
        dirty=False,
    )
    assert payload["formal_protocol"] is False
