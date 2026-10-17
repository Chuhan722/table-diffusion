"""第二段真题锁定（工作笔记第六十五步）：answers_for_rows 与负载同值；锁定开关只改第二段、第二段地板停关闭、核对量自洽。"""
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from resevo.dataset import StateRegistry, load_queries, load_table, target_from_specs
from resevo.state import table_answers

ROOT = Path(__file__).resolve().parent.parent
SRC_CSV = ROOT / "data" / "test_300x10" / "test_300x10.csv"


def _load_run_plants():
    spec = importlib.util.spec_from_file_location("run_plants_lock", ROOT / "scripts" / "run_plants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_data(tmp_path: Path, name: str = "t300") -> Path:
    """等值格子考卷：全部字段一阶 + 三对二阶，带噪、带 sigma；其余 42 对没测，纤维阶段有题可补。"""
    data_dir = tmp_path / name
    data_dir.mkdir()
    shutil.copy(SRC_CSV, data_dir / f"{name}.csv")
    schema, rows = load_table(str(data_dir / f"{name}.csv"))
    rng = np.random.default_rng(3)
    queries = []
    sigma = 4.0

    def add(attrs):
        import itertools

        doms = [schema.domains[schema.field_index(a)] for a in attrs]
        for vals in itertools.product(*doms):
            true = sum(all(r[schema.field_index(a)] == v for a, v in zip(attrs, vals)) for r in rows)
            queries.append(dict(
                id=f"Q{len(queries):04d}", type=f"eq{len(attrs)}way",
                expression=" AND ".join(f"{a} == {v}" for a, v in zip(attrs, vals)),
                conditions=[dict(attribute=a, operator="==", value=v) for a, v in zip(attrs, vals)],
                result=float(true + rng.normal(0.0, sigma)), sigma=sigma,
            ))

    for f in schema.fields:
        add([f])
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        add([schema.fields[a], schema.fields[b]])
    with open(data_dir / "measured_eq.json", "w", encoding="utf-8") as fh:
        json.dump(dict(dataset=f"{name}.csv", record_count=len(rows), query_count=len(queries),
                       result_unit="records", queries=queries), fh)
    return data_dir


def _run(rp, tmp_path: Path, tag: str, extra: list[str], stop_floor: float) -> Path:
    out = tmp_path / "out" / tag
    out.mkdir(parents=True)
    argv = [
        "run_plants.py", "--data", "t300", "--exam", "measured_eq.json", "--batched", "--uniform-init",
        "--rounds", "25", "--fiber-passes", "3", "--fiber-rounds", "15", "--exam-weights", "sigma",
        "--stop-floor", str(stop_floor), "--save-table", str(out / "t.csv"), "--out", str(out / "t_curve.csv"),
    ] + extra
    old_argv, old_root = sys.argv, rp.DATA_ROOT
    sys.argv, rp.DATA_ROOT = argv, tmp_path
    try:
        rp.main()
    finally:
        sys.argv, rp.DATA_ROOT = old_argv, old_root
    return out


def _final_stop_reason(curve: Path) -> str:
    with open(curve, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    final = [r for r in rows if r and r[0] == "final"]
    assert len(final) == 1
    return final[0][7]


def test_answers_for_rows_matches_workload():
    """分块求答案与注册后负载的 table_answers 逐位一致（含区间与半空间题）。"""
    schema, rows = load_table(str(SRC_CSV))
    specs = load_queries(str(ROOT / "data" / "test_300x10" / "measured_50query.json"))
    reg = StateRegistry(schema, specs)
    ids = reg.register_table(rows)
    via_workload = table_answers(reg.build_workload(target_from_specs(specs), np.ones(len(specs))), ids)
    fresh = StateRegistry(schema, specs)
    assert np.array_equal(fresh.answers_for_rows(rows, chunk=64), via_workload)
    assert np.array_equal(fresh.answers_for_rows(rows), via_workload)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("lock")
    _make_data(tmp_path)
    rp = _load_run_plants()
    # 地板设得很高：第一段立刻 floor_reached；第二段若仍带地板也会立刻停，由此检验第二段地板已关
    off = _run(rp, tmp_path, "off", [], stop_floor=1e12)
    on = _run(rp, tmp_path, "on", ["--fiber-lock-real"], stop_floor=1e12)
    return off, on


def _curve_sans_time(curve: Path) -> list[list[str]]:
    with open(curve, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    k = rows[0].index("elapsed_s")
    return [r[:k] + r[k + 1:] for r in rows]


def test_lock_leaves_stage1_untouched(runs):
    """锁定只改第二段：第一段终表逐字节相同，第一段曲线除耗时列外逐字相同。"""
    off, on = runs
    assert (off / "t_stage1.csv").read_bytes() == (on / "t_stage1.csv").read_bytes()
    assert _curve_sans_time(off / "t_curve_stage1.csv") == _curve_sans_time(on / "t_curve_stage1.csv")
    assert _final_stop_reason(off / "t_curve_stage1.csv") == "floor_reached"


def test_stage2_floor_and_lock_check(runs):
    """不锁定：第二段无地板，不因地板收工。锁定：第二段地板 = 每题一个人 ½Σw，第一段的 1e12 不再沿用；
    第二段起步损失恰为纯伪题损失，第一段真题损失与结算一致，核对文件齐全。"""
    off, on = runs
    assert _final_stop_reason(off / "t_curve.csv") != "floor_reached"
    assert not (off / "t_lock.json").exists()
    info = json.loads((on / "t_lock.json").read_text(encoding="utf-8"))
    assert info["stage1_stop_reason"] == "floor_reached"
    assert info["stage2_stop_reason"] in ("floor_reached", "round_limit", "loss_plateau", "no_positive_direction", "rescue_exhausted")
    assert info["num_pseudo"] > 0
    data_dir = on.parent.parent / "t300"
    specs = load_queries(str(data_dir / "measured_eq.json"))
    sig = np.array([q.sigma for q in specs])
    w_real = (sig.min() / sig) ** 2
    # 伪题 σ = 二阶题 σ 中位数 ×1，其权重按合卷（真题 + 伪题）的 σ_min 归一
    sig2 = np.median([q.sigma for q in specs if len(q.conditions) == 2])
    w_pseudo = (sig.min() / sig2) ** 2
    assert info["stage2_floor"] == pytest.approx(0.5 * (w_real.sum() + info["num_pseudo"] * w_pseudo), rel=1e-9)
    assert info["stage2_over_floor"] == pytest.approx(info["stage2_final_loss"] / info["stage2_floor"])
    if info["stage2_stop_reason"] == "floor_reached":
        assert info["stage2_final_loss"] <= info["stage2_floor"]
    else:
        assert info["stage2_final_loss"] > info["stage2_floor"] or info["stage2_rounds"] == 15
    assert info["stage2_initial_loss"] == pytest.approx(info["stage2_initial_pseudo_loss"], rel=1e-9)
    assert info["stage1_real_loss"] == pytest.approx(info["stage1_final_loss"], rel=1e-9)
    assert info["final_over_stage1"] == pytest.approx(info["final_real_loss"] / info["stage1_real_loss"])
    assert info["final_over_floor"] == pytest.approx(info["final_real_loss"] / 1e12)


def test_lock_targets_equal_stage1_counts(runs):
    """锁定后的第二段真题目标 = 第一段终表自己的计数：按第一段终表重算真题损失应为 0。"""
    _, on = runs
    rp = _load_run_plants()
    schema, rows = load_table(str(on / "t_stage1.csv"))
    data_dir = on.parent.parent / "t300"
    specs = load_queries(str(data_dir / "measured_eq.json"))
    reg = StateRegistry(schema, specs)
    answers = reg.answers_for_rows(rows)
    info = json.loads((on / "t_lock.json").read_text(encoding="utf-8"))
    sig = np.array([q.sigma for q in specs])
    w = (sig.min() / sig) ** 2
    assert rp.weighted_half_loss(target_from_specs(specs), answers, w) == pytest.approx(info["stage1_real_loss"], rel=1e-9)
