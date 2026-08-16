"""scale invariant 正式协议脚本的正式身份与安全判定测试（第二轮审查）。"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "probe_scale_invariant_formal.py"
)


@pytest.fixture(scope="module")
def formal_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "probe_scale_invariant_formal_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module, monkeypatch, tmp_path, argv, dirty, out_name="o.json"):
    out = tmp_path / out_name
    monkeypatch.setattr(
        sys, "argv",
        ["probe_scale_invariant_formal.py", "--out", str(out)] + argv,
    )
    monkeypatch.setattr(
        module, "_environment",
        lambda: {
            "git_commit": "test",
            "git_worktree_clean_including_untracked": not dirty,
            "python": "x", "numpy": "x", "pandas": "x",
            "platform": "x", "argv": sys.argv, "started_at": "now",
        },
    )
    monkeypatch.setattr(module, "DATASETS", {})
    module.main()
    return json.loads(out.read_text())


def test_allow_dirty_forces_informal_even_with_formal_params(
    formal_module, monkeypatch, tmp_path
):
    payload = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=["--allow-dirty"], dirty=False,
    )
    assert payload["formal"] is False


def test_clean_tree_formal_params_is_formal(
    formal_module, monkeypatch, tmp_path
):
    assert formal_module.OUTPUT_PATH.name == (
        "formal_scale_invariant_v3_5seed_2000round.json"
    )
    payload = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=[], dirty=False,
    )
    assert payload["formal"] is True
    assert payload["protocol_deviations"] == []


def test_non_formal_rounds_marked_informal(
    formal_module, monkeypatch, tmp_path
):
    payload = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=["--rounds", "30"], dirty=False,
    )
    assert payload["formal"] is False
    assert any("rounds" in d for d in payload["protocol_deviations"])


def test_dirty_tree_without_flag_refuses(
    formal_module, monkeypatch, tmp_path
):
    with pytest.raises(SystemExit):
        _run_main(
            formal_module, monkeypatch, tmp_path,
            argv=[], dirty=True,
        )


def test_existing_output_refuses_overwrite(
    formal_module, monkeypatch, tmp_path
):
    out = tmp_path / "exists.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        _run_main(
            formal_module, monkeypatch, tmp_path,
            argv=[], dirty=False, out_name="exists.json",
        )
