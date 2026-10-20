"""分裂探针记账与显著回升停（工作笔记第七十三步）：默认关闭零改变；记账值 = 起点表对模拟卷的 mean|y_B − q|；
分档全过线轮必记且与分档快照同表；全过线后的配对差均值/标准误与暴力重算一致；显著回升停停在
不停跑记账里首个连续 runs 次满足 diff_mean ≥ z·diff_se 的轮、交该轮起点表、停前轨迹逐位同；
运行脚本端到端（CPU 批量路径）：轨迹与终表同默认跑、终表行与暴力计数一致、停法与参数校验。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from test_batchkernel import _scene
from test_earlystop import _evolve
from test_fiber_lock import _make_data
from test_groupfloor import _curve_sans_time, _groups_for_scene, _load_run_plants, _run

from resevo.dataset import evaluate_query, load_queries, load_table
from resevo.state import table_answers

ROOT = Path(__file__).resolve().parent.parent


def _targets_b(registry, seed=7):
    """模拟卷：考卷目标减一份随机扰动，只要与题序对齐即可。"""
    y = np.array([q.result for q in registry.specs], dtype=np.float64)
    return y - np.random.default_rng(seed).normal(0.0, 3.0, size=y.size)


def _l1(workload, ids, yb):
    return float(np.mean(np.abs(yb - table_answers(workload, ids))))


def test_fission_default_off_bitwise_identical():
    """开记账（只记不停）：轨迹、终表、停止原因与不开逐位同。"""
    r1, ids1, w1, _ = _scene(2, num_rows=30)
    base = _evolve(r1, ids1, w1)
    r2, ids2, w2, _ = _scene(2, num_rows=30)
    rows = []
    on = _evolve(
        r2, ids2, w2, fission_targets=_targets_b(r2), fission_every=3,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows.append((k, loss, l1, g)),
    )
    assert [r.old_loss for r in on.records] == [r.old_loss for r in base.records]
    assert np.array_equal(on.state_ids, base.state_ids)
    assert on.stop_reason == base.stop_reason
    assert rows and all(k % 3 == 0 for k, _, _, _ in rows) and not any(g for _, _, _, g in rows)


def test_fission_value_matches_table():
    """记账值 = 该轮起点表对模拟卷的平均绝对误差（CPU 全表负载轮），轮 0 对初表核对，其余轮对轨迹前缀核对。"""
    r, ids, w, _ = _scene(3, num_rows=30)
    yb = _targets_b(r)
    rows = []
    out = _evolve(
        r, ids, w, rounds=12, fission_targets=yb, fission_every=4,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows.append((k, loss, l1)),
    )
    by_round = {r_.round_index: r_.old_loss for r_ in out.records}
    assert [k for k, _, _ in rows] == [k for k in range(len(out.records)) if k % 4 == 0]
    wl = r.build_workload(np.array([q.result for q in r.specs]), w)
    assert rows[0][2] == pytest.approx(_l1(wl, ids, yb), rel=1e-12)
    for k, loss, l1 in rows:
        assert loss == by_round[k]
        r2, ids2, w2, _ = _scene(3, num_rows=30)
        prefix = _evolve(r2, ids2, w2, rounds=k) if k > 0 else None
        ids_k = ids if prefix is None else prefix.state_ids
        assert l1 == pytest.approx(_l1(wl, ids_k, yb), rel=1e-12)


def test_fission_fires_at_group_floor_round():
    """分档全过线轮不在间隔上也记一行，标志位为 1，值与分档快照表一致。"""
    r0, ids0, w0, _ = _scene(5, num_rows=30)
    idx = _groups_for_scene(w0)
    full = _evolve(r0, ids0, w0, rounds=40, group_index=idx, group_floors=np.array([1e-9, 1e-9]))
    per0, per_last = full.group_losses[0][1], full.group_losses[-1][1]
    floors = np.array([per0[0] * 2.0, (per0[1] + per_last[1]) / 2.0])
    r, ids, w, _ = _scene(5, num_rows=30)
    yb = _targets_b(r)
    rows, snaps = [], []
    _evolve(
        r, ids, w, rounds=40, group_index=idx, group_floors=floors,
        group_snapshot_hook=lambda k, loss, per, s_ids: snaps.append((k, s_ids)),
        fission_targets=yb, fission_every=1000,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows.append((k, l1, g)),
    )
    assert len(snaps) == 1
    k_g, ids_g = snaps[0]
    flagged = [(k, l1) for k, l1, g in rows if g]
    assert len(flagged) == 1 and flagged[0][0] == k_g
    wl = r.build_workload(np.array([q.result for q in r.specs]), w)
    assert flagged[0][1] == pytest.approx(_l1(wl, ids_g, yb), rel=1e-12)


def test_fission_validation():
    r, ids, w, _ = _scene(6, num_rows=20)
    yb = _targets_b(r)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=3, fission_targets=yb)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=3, fission_every=2, fission_hook=lambda *a: None)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=3, fission_targets=yb[:-1] if yb.size > 1 else yb, fission_every=0, fission_hook=lambda *a: None)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=3, fission_targets=np.array([np.nan]), fission_every=1, fission_hook=lambda *a: None)


def _diff_stats(e, e_g):
    d = e - e_g
    return float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(d.size))


def _floors_mid_run(seed, rounds=40):
    """让分档全过线落在跑程中段的地板：一档取起点损失的两倍、另一档取起终点均值。"""
    r0, ids0, w0, _ = _scene(seed, num_rows=30)
    idx = _groups_for_scene(w0)
    full = _evolve(r0, ids0, w0, rounds=rounds, group_index=idx, group_floors=np.array([1e-9, 1e-9]))
    per0, per_last = full.group_losses[0][1], full.group_losses[-1][1]
    return idx, np.array([per0[0] * 2.0, (per0[1] + per_last[1]) / 2.0])


def test_fission_diff_stats_match_brute_force():
    """全过线前与全过线轮统计列为 None；其后每条 = 起点表与全过线表逐题误差配对差的均值与标准误（暴力重算）。"""
    idx, floors = _floors_mid_run(5)
    r, ids, w, _ = _scene(5, num_rows=30)
    yb = _targets_b(r)
    rows, snaps = [], []
    out = _evolve(
        r, ids, w, rounds=40, group_index=idx, group_floors=floors,
        group_snapshot_hook=lambda k, loss, per, s_ids: snaps.append((k, s_ids)),
        fission_targets=yb, fission_every=4,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows.append((k, l1, g, dm, ds)),
    )
    assert out.stop_reason != "fission_rise"
    k_g, ids_g = snaps[0]
    wl = r.build_workload(np.array([q.result for q in r.specs]), w)
    e_g = np.abs(yb - table_answers(wl, ids_g))
    after = [row for row in rows if row[0] > k_g]
    assert after, "全过线后须还有记账"
    for k, l1, g, dm, ds in rows:
        if k <= k_g:
            assert dm is None and ds is None
            continue
        r2, ids2, w2, _ = _scene(5, num_rows=30)
        ids_k = _evolve(r2, ids2, w2, rounds=k, group_index=idx, group_floors=floors).state_ids
        e_k = np.abs(yb - table_answers(wl, ids_k))
        m, se = _diff_stats(e_k, e_g)
        assert l1 == pytest.approx(float(np.mean(e_k)), rel=1e-12)
        assert dm == pytest.approx(m, rel=1e-12, abs=1e-15) and ds == pytest.approx(se, rel=1e-12)


def _streak_stop_round(rows, z, runs):
    """按不停跑的记账复算判停轮：连续 runs 条 diff_mean ≥ z·diff_se（se>0）的首个终点。"""
    streak = 0
    for k, _, _, dm, ds in rows:
        if dm is None:
            streak = 0
            continue
        streak = streak + 1 if (ds > 0.0 and dm >= z * ds) else 0
        if streak >= runs:
            return k
    return None


def test_fission_rise_stop_triggers_and_matches_record_only_run():
    """模拟卷取全过线表自己的答案（e_G ≡ 0，之后任何改动都是“更差”）：开停 → 停止原因 fission_rise，
    停在只记不停那趟记账复算出的轮，记录为不停跑的前缀，交出的表 = 该轮起点表；z 极大则永不停且与不停跑逐位同。"""
    idx, floors = _floors_mid_run(5)
    r, ids, w, _ = _scene(5, num_rows=30)
    snaps = []
    _evolve(
        r, ids, w, rounds=40, group_index=idx, group_floors=floors,
        group_snapshot_hook=lambda k, loss, per, s_ids: snaps.append((k, s_ids)),
    )
    k_g, ids_g = snaps[0]
    wl = r.build_workload(np.array([q.result for q in r.specs]), w)
    yb = table_answers(wl, ids_g).astype(np.float64)
    z, runs = 0.5, 2
    rows = []
    r1, ids1, w1, _ = _scene(5, num_rows=30)
    base = _evolve(
        r1, ids1, w1, rounds=40, group_index=idx, group_floors=floors,
        fission_targets=yb, fission_every=1,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows.append((k, l1, g, dm, ds)),
    )
    k_stop = _streak_stop_round(rows, z, runs)
    assert k_stop is not None and k_stop > k_g
    rows_stop = []
    r2, ids2, w2, _ = _scene(5, num_rows=30)
    out = _evolve(
        r2, ids2, w2, rounds=40, group_index=idx, group_floors=floors,
        fission_targets=yb, fission_every=1, fission_stop_z=z, fission_stop_runs=runs,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows_stop.append((k, l1, g, dm, ds)),
    )
    assert out.stop_reason == "fission_rise"
    assert len(out.records) == k_stop + 1
    assert [x.old_loss for x in out.records] == [x.old_loss for x in base.records[: k_stop + 1]]
    assert rows_stop == rows[: len(rows_stop)] and rows_stop[-1][0] == k_stop
    r3, ids3, w3, _ = _scene(5, num_rows=30)
    prefix = _evolve(r3, ids3, w3, rounds=k_stop, group_index=idx, group_floors=floors)
    assert np.array_equal(out.state_ids, prefix.state_ids)
    assert len(out.group_losses) == k_stop + 1 and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(out.group_losses, base.group_losses)
    )
    rows_never = []
    r4, ids4, w4, _ = _scene(5, num_rows=30)
    never = _evolve(
        r4, ids4, w4, rounds=40, group_index=idx, group_floors=floors,
        fission_targets=yb, fission_every=1, fission_stop_z=1e6, fission_stop_runs=runs,
        fission_hook=lambda k, loss, l1, t, g, dm, ds: rows_never.append((k, l1, g, dm, ds)),
    )
    assert never.stop_reason == base.stop_reason and np.array_equal(never.state_ids, base.state_ids)
    assert rows_never == rows


def test_fission_stop_validation():
    r, ids, w, _ = _scene(6, num_rows=20)
    idx = _groups_for_scene(w)
    yb = _targets_b(r)
    fis = dict(fission_targets=yb, fission_every=2, fission_hook=lambda *a: None)
    grp = dict(group_index=idx, group_floors=np.array([1.0, 1.0]))
    for bad in (dict(fission_stop_z=-1.0), dict(fission_stop_z=np.nan), dict(fission_stop_z=1.0, fission_stop_runs=0)):
        with pytest.raises(ValueError):
            _evolve(r, ids, w, rounds=3, **fis, **grp, **bad)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=3, fission_stop_z=1.0, **grp)
    with pytest.raises(ValueError):
        _evolve(r, ids, w, rounds=3, fission_stop_z=1.0, **fis)


def _brute_counts(table_csv: Path, exam_json: Path) -> np.ndarray:
    schema, rows = load_table(str(table_csv))
    specs = load_queries(str(exam_json))
    return np.array([sum(evaluate_query(q, row, schema) for row in rows) for q in specs], dtype=np.float64)


def _make_fission_exam(data_dir: Path) -> Path:
    src = json.loads((data_dir / "measured_eq.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(11)
    for q in src["queries"]:
        q["result"] = float(q["result"] - rng.normal(0.0, float(q["sigma"])))
    out = data_dir / "fisB.json"
    out.write_text(json.dumps(src), encoding="utf-8")
    return out


def test_runner_fission_end_to_end(tmp_path):
    """脚本端到端：开记账 → 曲线与终表同默认跑；CSV 有表头、间隔行、分档全过线行与终表行；
    终表行与分档全过线行的值等于对应表的暴力计数；参数互斥校验。"""
    data_dir = _make_data(tmp_path)
    fis = _make_fission_exam(data_dir)
    rp = _load_run_plants()
    base = _run(rp, tmp_path, "base", [])
    fout = tmp_path / "out" / "fis" / "probe.csv"
    on = _run(rp, tmp_path, "fis", [
        "--fission-exam", str(fis), "--fission-every", "5", "--fission-out", str(fout),
        "--snapshot-group-floor", "--snapshot-prefix", str(tmp_path / "out" / "fis" / "s"), "--floor-group-min", "1",
    ])
    assert (base / "t.csv").read_bytes() == (on / "t.csv").read_bytes()
    assert _curve_sans_time(base / "t_curve.csv") == _curve_sans_time(on / "t_curve.csv")
    with open(fout, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["round", "elapsed_s", "old_loss", "l1_b", "at_group_floor", "diff_mean", "diff_se"]
    body, final = rows[1:-1], rows[-1]
    assert final[0] == "final"
    yb = np.array([q["result"] for q in json.loads(fis.read_text(encoding="utf-8"))["queries"]])
    assert float(final[3]) == pytest.approx(float(np.mean(np.abs(yb - _brute_counts(on / "t.csv", fis)))), rel=1e-12)
    for r in body:
        assert int(r[0]) % 5 == 0 or r[4] == "1"
    flagged = [r for r in body if r[4] == "1"]
    snapG = tmp_path / "out" / "fis" / "s_snapG.csv"
    if snapG.exists():
        assert len(flagged) == 1
        e_g = np.abs(yb - _brute_counts(snapG, fis))
        assert float(flagged[0][3]) == pytest.approx(float(np.mean(e_g)), rel=1e-12)
        k_g = int(flagged[0][0])
        # 统计列：全过线前与全过线轮留空，其后非空；终表行 = 终表对全过线表配对差的暴力重算
        assert all((r[5] == "" and r[6] == "") == (int(r[0]) <= k_g) for r in body)
        m, se = _diff_stats(np.abs(yb - _brute_counts(on / "t.csv", fis)), e_g)
        assert float(final[5]) == pytest.approx(m, rel=1e-12, abs=1e-15) and float(final[6]) == pytest.approx(se, rel=1e-12)
    else:
        assert not flagged
        assert all(r[5] == "" and r[6] == "" for r in rows[1:])
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad1", ["--fission-exam", str(fis)])
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad2", ["--fission-exam", str(fis), "--fission-every", "0", "--fission-out", str(fout)])
    bad_exam = data_dir / "fis_bad.json"
    src = json.loads(fis.read_text(encoding="utf-8"))
    src["queries"] = src["queries"][:-1]
    bad_exam.write_text(json.dumps(src), encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad3", ["--fission-exam", str(bad_exam), "--fission-every", "5", "--fission-out", str(fout)])


def test_runner_fission_rise_stop(tmp_path):
    """脚本端到端开停：模拟卷取全过线表自己的答案 → 停止原因 fission_rise；记账为只记不停那趟的前缀、
    停在其复算出的轮；交出的表 = 触发轮起点表（终表行的统计与触发行逐位同、终表行值与暴力计数一致）；参数校验。"""
    data_dir = _make_data(tmp_path)
    fis = _make_fission_exam(data_dir)
    rp = _load_run_plants()
    snap_args = ["--snapshot-group-floor", "--floor-group-min", "1"]
    fout = tmp_path / "out" / "g" / "probe.csv"
    _run(rp, tmp_path, "g", ["--fission-exam", str(fis), "--fission-every", "1", "--fission-out", str(fout),
                             *snap_args, "--snapshot-prefix", str(tmp_path / "out" / "g" / "s")])
    snapG = tmp_path / "out" / "g" / "s_snapG.csv"
    assert snapG.exists(), "测试数据须在 25 轮内分档全过线"
    src = json.loads(fis.read_text(encoding="utf-8"))
    for q, c in zip(src["queries"], _brute_counts(snapG, fis)):
        q["result"] = float(c)
    fis_g = data_dir / "fisG.json"
    fis_g.write_text(json.dumps(src), encoding="utf-8")
    yb = np.array([q["result"] for q in src["queries"]])
    z, runs = 0.5, 2

    def read_rows(path):
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.reader(fh))[1:]
        return [(r[0], r[2], r[3], r[4], r[5], r[6]) for r in rows]

    def parsed(rows):
        return [(int(r[0]), float(r[2]), r[3], None if r[4] == "" else float(r[4]), None if r[5] == "" else float(r[5]))
                for r in rows if r[0] != "final"]

    f_rec = tmp_path / "out" / "rec" / "probe.csv"
    rec = _run(rp, tmp_path, "rec", ["--fission-exam", str(fis_g), "--fission-every", "1", "--fission-out", str(f_rec),
                                     *snap_args, "--snapshot-prefix", str(tmp_path / "out" / "rec" / "s")])
    rows_rec = read_rows(f_rec)
    k_stop = _streak_stop_round(parsed(rows_rec), z, runs)
    assert k_stop is not None
    f_stop = tmp_path / "out" / "stop" / "probe.csv"
    stop = _run(rp, tmp_path, "stop", ["--fission-exam", str(fis_g), "--fission-every", "1", "--fission-out", str(f_stop),
                                       *snap_args, "--snapshot-prefix", str(tmp_path / "out" / "stop" / "s"),
                                       "--fission-stop-z", str(z), "--fission-stop-runs", str(runs)])
    curve = _curve_sans_time(stop / "t_curve.csv")
    assert [r for r in curve if r[0] == "final"][0][7] == "fission_rise"
    assert len([r for r in curve[1:] if r[0] != "final"]) == k_stop + 1
    rows_stop = read_rows(f_stop)
    body_stop, final_stop = rows_stop[:-1], rows_stop[-1]
    assert body_stop == rows_rec[: len(body_stop)] and body_stop[-1][0] == str(k_stop)
    assert final_stop[2:] == body_stop[-1][2:]  # 交出的表就是触发轮起点表：终表行统计与触发行逐位同
    assert float(final_stop[2]) == pytest.approx(float(np.mean(np.abs(yb - _brute_counts(stop / "t.csv", fis_g)))), rel=1e-12)
    assert (rec / "t.csv").read_bytes() != (stop / "t.csv").read_bytes()
    base = ["--fission-exam", str(fis_g), "--fission-every", "1", "--fission-out", str(f_stop)]
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad1", [*base, "--fission-stop-z", "2"])  # 缺分档快照
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad2", [*snap_args, "--snapshot-prefix", str(tmp_path / "out" / "bad2" / "s"), "--fission-stop-z", "2"])
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad3", [*base, *snap_args, "--snapshot-prefix", str(tmp_path / "out" / "bad3" / "s"), "--fission-stop-z", "-1"])
    with pytest.raises(SystemExit):
        _run(rp, tmp_path, "bad4", [*base, *snap_args, "--snapshot-prefix", str(tmp_path / "out" / "bad4" / "s"), "--fission-stop-z", "2", "--fission-stop-runs", "0"])
