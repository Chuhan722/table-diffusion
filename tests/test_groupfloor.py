"""分档地板（工作笔记第六十八步）：默认关闭零改变；分档损失之和 = 总损失；分档停与分档快照的轮号、表一致；
σ 分档与小档合并；运行脚本端到端（CPU 批量路径）。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from test_batchkernel import _scene
from test_earlystop import _evolve
from test_fiber_lock import _make_data

ROOT = Path(__file__).resolve().parent.parent


def _groups_for_scene(registry_weights, n_groups=2):
    """场景里 6 题分两档（前三题一档、后三题一档），地板随便取正数。"""
    q = len(registry_weights)
    index = np.array([0] * (q // 2) + [1] * (q - q // 2), dtype=np.int64)
    return index


def test_group_default_off_bitwise_identical():
    """不给分档参数：轨迹、终表逐位同，返回值 group_losses 为 None。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    base = _evolve(r1, ids1, w1)
    assert base.group_losses is None
    r2, ids2, w2, _ = _scene(2, num_rows=30)
    idx = _groups_for_scene(w2)
    on = _evolve(r2, ids2, w2, group_index=idx, group_floors=np.array([1e-9, 1e-9]))
    # 地板极小永不过线：只记账不停，轨迹与终表逐位同
    assert [r.old_loss for r in on.records] == [r.old_loss for r in base.records]
    assert np.array_equal(on.state_ids, base.state_ids)
    assert on.stop_reason == base.stop_reason


def test_group_losses_sum_to_old_loss():
    """每轮各档损失之和与该轮起点损失一致（CPU 全表负载轮）。"""
    r, ids, w, _ = _scene(3, num_rows=30)
    idx = _groups_for_scene(w)
    out = _evolve(r, ids, w, rounds=30, group_index=idx, group_floors=np.array([1e-9, 1e-9]))
    assert out.group_losses and len(out.group_losses) == len(out.records)
    by_round = {r_.round_index: r_.old_loss for r_ in out.records}
    for k, per in out.group_losses:
        assert per.shape == (2,)
        assert per.sum() == pytest.approx(by_round[k], rel=1e-12, abs=1e-9)


def _first_pass_round(out, floors):
    for k, per in out.group_losses:
        if np.all(per <= floors):
            return k
    return None


def test_group_stop_and_snapshot_agree_with_trajectory():
    """分档停：停在首个全档过线的轮，返回该轮起点表；分档快照不停时在同一轮交出同一张表；
    地板取自全程记账，令其中一档为瓶颈（另一档早已过线）。"""
    r0, ids0, w0, _ = _scene(5, num_rows=30)
    idx = _groups_for_scene(w0)
    full = _evolve(r0, ids0, w0, rounds=40, group_index=idx, group_floors=np.array([1e-9, 1e-9]))
    per0 = full.group_losses[0][1]
    per_last = full.group_losses[-1][1]
    # 档 0 地板放宽（起点就过线），档 1 地板取起终中点：档 1 是瓶颈
    floors = np.array([per0[0] * 2.0, (per0[1] + per_last[1]) / 2.0])
    k_expect = _first_pass_round(full, floors)
    assert k_expect is not None and k_expect > 0
    r1, ids1, w1, _ = _scene(5, num_rows=30)
    stopped = _evolve(r1, ids1, w1, rounds=40, group_index=idx, group_floors=floors, stop_floor_groups=True)
    assert stopped.stop_reason == "group_floor_reached"
    assert len(stopped.records) - 1 == k_expect
    assert [r.old_loss for r in stopped.records] == [r.old_loss for r in full.records[: k_expect + 1]]
    fired = []
    r2, ids2, w2, _ = _scene(5, num_rows=30)
    cont = _evolve(
        r2, ids2, w2, rounds=40, group_index=idx, group_floors=floors,
        group_snapshot_hook=lambda k, loss, per, ids: fired.append((k, loss, per, ids)),
    )
    assert cont.stop_reason == full.stop_reason and len(cont.records) == len(full.records)
    assert len(fired) == 1
    k, loss, per, ids = fired[0]
    assert k == k_expect
    assert loss == full.records[k].old_loss
    assert np.all(per <= floors)
    assert np.array_equal(ids, stopped.state_ids)
    # 快照表 = 跑 k 轮后的表
    r3, ids3, w3, _ = _scene(5, num_rows=30)
    prefix = _evolve(r3, ids3, w3, rounds=k)
    assert np.array_equal(ids, prefix.state_ids)


def test_group_total_floor_differs_from_group_rule():
    """总和到地板 ≠ 每档到地板：以档 1 瓶颈的地板为例，总和过线的轮早于全档过线的轮。"""
    r0, ids0, w0, _ = _scene(5, num_rows=30)
    idx = _groups_for_scene(w0)
    full = _evolve(r0, ids0, w0, rounds=40, group_index=idx, group_floors=np.array([1e-9, 1e-9]))
    per0, per_last = full.group_losses[0][1], full.group_losses[-1][1]
    floors = np.array([per0[0] * 2.0, (per0[1] + per_last[1]) / 2.0])
    k_group = _first_pass_round(full, floors)
    k_total = next(k for k, per in full.group_losses if per.sum() <= floors.sum())
    assert k_total <= k_group


def test_group_validation():
    r, ids, w, _ = _scene(6, num_rows=20)
    idx = _groups_for_scene(w)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=5, group_index=idx)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=5, stop_floor_groups=True)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=5, group_index=idx, group_floors=np.array([1.0]))
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=5, group_index=idx, group_floors=np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=5, group_index=idx[:-1], group_floors=np.array([1.0, 1.0]))


def _load_run_plants():
    spec = importlib.util.spec_from_file_location("run_plants_groupfloor", ROOT / "scripts" / "run_plants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Q:
    def __init__(self, sigma):
        self.sigma = sigma


def test_sigma_groups_merge_and_floors():
    """σ 相同为一档、小档并入最近相邻档、各档地板之和 = ½Σwσ²、题多于门槛时不并。"""
    rp = _load_run_plants()
    sig = [1.0] * 40 + [2.0] * 3 + [5.0] * 50 + [20.0] * 2
    specs = [_Q(s) for s in sig]
    w = (min(sig) / np.array(sig)) ** 2
    info = rp.sigma_groups(specs, w, 30)
    assert len(info["table"]) == 2
    assert info["table"][0]["sigmas"] == [1.0, 2.0] and info["table"][0]["questions"] == 43
    assert info["table"][1]["sigmas"] == [5.0, 20.0] and info["table"][1]["questions"] == 52
    assert info["floors"].sum() == pytest.approx(0.5 * float(np.sum(w * np.array(sig) ** 2)))
    assert np.array_equal(np.bincount(info["index"]), [43, 52])
    one = rp.sigma_groups(specs, w, 1)
    assert len(one["table"]) == 4
    single = rp.sigma_groups([_Q(3.0)] * 10, np.ones(10), 30)
    assert len(single["table"]) == 1 and single["floors"][0] == pytest.approx(0.5 * 10 * 9.0)
    with pytest.raises(ValueError):
        rp.sigma_groups([_Q(None)] * 3, np.ones(3), 1)


def _run(rp, tmp_path: Path, tag: str, extra: list[str]) -> Path:
    out = tmp_path / "out" / tag
    out.mkdir(parents=True)
    argv = [
        "run_plants.py", "--data", "t300", "--exam", "measured_eq.json", "--batched", "--uniform-init",
        "--rounds", "25", "--exam-weights", "sigma",
        "--save-table", str(out / "t.csv"), "--out", str(out / "t_curve.csv"),
    ] + extra
    old_argv, old_root = sys.argv, rp.DATA_ROOT
    sys.argv, rp.DATA_ROOT = argv, tmp_path
    try:
        rp.main()
    finally:
        sys.argv, rp.DATA_ROOT = old_argv, old_root
    return out


def _curve_sans_time(curve: Path) -> list[list[str]]:
    with open(curve, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    k = rows[0].index("elapsed_s")
    return [r[:k] + r[k + 1:] for r in rows]


def test_runner_group_floor_end_to_end(tmp_path):
    """脚本端到端：开分档快照不停 → 曲线与终表同默认跑；分档曲线各档之和 = 曲线损失、floor 行 = 各档地板；
    开分档停 → 停止原因 group_floor_reached 且停在分档曲线首个全档过线的轮；参数互斥校验。"""
    _make_data(tmp_path)
    rp = _load_run_plants()
    base = _run(rp, tmp_path, "base", [])
    snap = _run(rp, tmp_path, "snap", ["--snapshot-group-floor", "--snapshot-prefix", str(tmp_path / "out" / "snap" / "s"), "--floor-group-min", "1"])
    assert (base / "t.csv").read_bytes() == (snap / "t.csv").read_bytes()
    assert _curve_sans_time(base / "t_curve.csv") == _curve_sans_time(snap / "t_curve.csv")
    with open(snap / "t_curve_groups.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body, floor_row = rows[0], rows[1:-1], rows[-1]
    assert floor_row[0] == "floor"
    meta = json.loads((snap / "t_curve_groups.json").read_text(encoding="utf-8"))
    # 考卷 σ 全同：只有一档，地板 = ½Σwσ² = ½·题数·σ²
    assert len(meta["groups"]) == 1 and len(header) == 2
    floors = np.array([float(x) for x in floor_row[1:]])
    curve = {r[0]: float(r[1]) for r in _curve_sans_time(snap / "t_curve.csv")[1:] if r[0] != "final"}
    for r in body:
        assert sum(float(x) for x in r[1:]) == pytest.approx(curve[r[0]], rel=1e-9)
    # 单档时分档停 ≡ 硬地板停：若全程过线，停止轮与 --stop-floor 合计地板一致
    first = next((r[0] for r in body if np.all(np.array([float(x) for x in r[1:]]) <= floors)), None)
    stop = _run(rp, tmp_path, "stop", ["--stop-floor-groups", "--floor-group-min", "1"])
    reason = [r for r in _curve_sans_time(stop / "t_curve.csv") if r[0] == "final"][0][7]
    if first is None:
        assert reason != "group_floor_reached"
    else:
        assert reason == "group_floor_reached"
        n_rounds = len([r for r in _curve_sans_time(stop / "t_curve.csv")[1:] if r[0] != "final"])
        assert n_rounds - 1 == int(first)
        hard = _run(rp, tmp_path, "hard", ["--stop-floor", f"{floors.sum():.12g}"])
        assert (hard / "t.csv").read_bytes() == (stop / "t.csv").read_bytes()
        snapG = snap / "s_snapG.csv"
        assert snapG.exists() and snapG.read_bytes() == (stop / "t.csv").read_bytes()
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad1", ["--stop-floor-groups", "--stop-floor", "1.0"])
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad2", ["--snapshot-group-floor"])
