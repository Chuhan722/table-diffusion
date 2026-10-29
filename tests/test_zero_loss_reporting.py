"""零分母只让报数比值未定义，不应阻断合法生成或锁定报告。"""
import csv
from itertools import combinations, product
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_fiber_lock import _load_run_plants

from resevo.dataset import QuerySpec, StateRegistry, TableSchema, target_from_specs
from resevo.fiber import fiber_average, fiber_specs, marg


def _chain():
    return np.repeat(np.array([(a, b, a) for a, b in product(range(2), repeat=2)]), 2, axis=0)


def _specs(rows, schema, sets, shift):
    doms = [len(domain) for domain in schema.domains]
    specs = []
    for cols in sets:
        counts = marg(rows, cols, doms)
        for index, values in enumerate(product(*(schema.domains[j] for j in cols))):
            specs.append(QuerySpec(
                f"Q{len(specs):03d}",
                tuple({"attribute": schema.fields[j], "operator": "==", "value": value}
                      for j, value in zip(cols, values)),
                float(counts[index] + shift), 1.0,
            ))
    return specs


def _run_scene(tmp_path, monkeypatch, kind, backend, *, fission=False):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        try:
            devices = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        if not devices:
            pytest.skip("No CUDA device")
    if kind in ("second_zero", "both_zero"):
        latin = np.array([((b + d) % 3, b, (b + 2 * d) % 3, d) for b, d in product(range(3), repeat=2)])
        rows = np.repeat(latin, [5] + [1] * 8, axis=0)
        doms, sets = [3] * 4, [(0, 1), (1, 2), (2, 3), (0, 3)]
    else:
        rows = _chain()
        doms = [2] * 3
        sets = [(0, 1), (1, 2)] if kind == "first_zero" else list(combinations(range(3), 2))
    source_rows, target_rows = rows, rows
    if kind == "positive_to_zero":
        source_rows = np.array([[0], [1]])
        rows, target_rows = np.zeros((2, 1), dtype=np.int64), np.ones((2, 1), dtype=np.int64)
        doms, sets = [2], [(0,)]
    schema = TableSchema(
        tuple(f"x{j}" for j in range(len(doms))),
        tuple(tuple(map(str, range(d))) for d in doms),
    )
    shift = 0.25 if kind == "second_zero" else 2.0**-30 if kind == "near_positive" else 0.0
    specs = _specs(target_rows, schema, sets, shift)
    source = tmp_path / "data/toy"
    source.mkdir(parents=True)
    for path, table in ((source / "toy.csv", source_rows), (tmp_path / "initial.csv", rows)):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(schema.fields)
            writer.writerows(table.tolist())
    (source / "measured_query.json").write_text(json.dumps({
        "record_count": len(rows), "result_unit": "records",
        "queries": [
            {"id": q.query_id, "conditions": q.conditions, "result": q.result, "sigma": q.sigma}
            for q in specs
        ],
    }))
    output = tmp_path / "out"
    output.mkdir()
    argv = [
        "run_plants.py", "--data", "toy", "--exam", "measured_query.json", "--batched",
        "--init-table", str(tmp_path / "initial.csv"), "--rounds", "3", "--retries", "0",
        "--sample-seed", "7", "--save-table", str(output / "table.csv"),
        "--out", str(output / "curve.csv"),
    ]
    if kind in ("first_zero", "second_zero", "both_zero"):
        argv += ["--fiber-passes", "100", "--fiber-rounds", "3", "--fiber-seed", "7", "--fiber-lock-real"]
    if backend == "gpu":
        argv.append("--gpu")
    if fission:
        validation = json.loads((source / "measured_query.json").read_text())
        for query in validation["queries"]:
            query["result"] += 0.25
        (source / "validation.json").write_text(json.dumps(validation))
        argv += [
            "--fission-exam", str(source / "validation.json"),
            "--fission-every", "1", "--fission-out", str(output / "fission.csv"),
        ]
    input_bytes = {path: path.read_bytes() for path in source.iterdir()}
    rp = _load_run_plants()
    monkeypatch.setattr(rp, "DATA_ROOT", source.parent)
    monkeypatch.setattr(sys, "argv", argv)
    rp.main()
    assert {path: path.read_bytes() for path in source.iterdir()} == input_bytes
    with (output / "table.csv").open(newline="") as stream:
        final = list(csv.reader(stream))
    assert final[0] == list(schema.fields) and len(final) == len(rows) + 1
    assert all(len(row) == len(doms) for row in final[1:])
    assert all(value in schema.domains[j] for row in final[1:] for j, value in enumerate(row))
    assert (output / "curve.csv").exists()
    return rp, output, schema, specs, [tuple(row) for row in final[1:]]


@pytest.mark.parametrize("kind", ("single_zero", "first_zero", "second_zero", "both_zero"))
@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_zero_loss_stages_still_publish_outputs(tmp_path, monkeypatch, capsys, kind, backend):
    rp, output, schema, specs, final = _run_scene(tmp_path, monkeypatch, kind, backend)
    text = capsys.readouterr().out
    assert "初始比 未定义（初始损失为0）" in text
    if kind == "single_zero":
        assert not (output / "table_lock.json").exists()
        assert "第二段" not in text
        return
    info = json.loads((output / "table_lock.json").read_text())
    registry = StateRegistry(schema, specs)
    assert info["final_real_loss"] == rp.weighted_half_loss(
        target_from_specs(specs), registry.answers_for_rows(final), np.ones(len(specs)),
    )
    assert (output / "table_stage1.csv").exists() and (output / "curve_stage1.csv").exists()
    if kind in ("first_zero", "both_zero"):
        assert info["stage1_real_loss"] == 0.0 and info["final_over_stage1"] is None
        assert "第一段终表损失为0，比值未定义" in text
    if kind in ("second_zero", "both_zero"):
        assert info["stage2_initial_loss"] == info["stage2_final_loss"] == 0.0
        assert info["stage2_stop_reason"] == "floor_reached"
    if kind == "second_zero":
        assert info["stage1_real_loss"] == info["final_real_loss"] == 1.125
        assert info["final_over_stage1"] == 1.0
        assert "是第一段终表的 1.0000 倍" in text
    if kind == "both_zero":
        assert info["final_real_loss"] == 0.0


def test_small_positive_initial_loss_keeps_percentage(tmp_path, monkeypatch, capsys):
    _, output, _, _, _ = _run_scene(tmp_path, monkeypatch, "near_positive", "cpu")
    text = capsys.readouterr().out
    assert "初始比 100.0000%" in text and "未定义" not in text
    with (output / "curve.csv").open(newline="") as stream:
        curve = list(csv.DictReader(stream))
    assert 0.0 < float(curve[0]["old_loss"]) < 1e-12


@pytest.mark.parametrize("kind", ("single_zero", "first_zero", "second_zero", "both_zero"))
@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_zero_loss_with_fission_publishes_complete_first_stage_probe(
    tmp_path, monkeypatch, capsys, kind, backend,
):
    """PR81+89: zero loss must still publish both ordinary outputs and the probe's final row."""
    rp, output, schema, specs, _ = _run_scene(
        tmp_path, monkeypatch, kind, backend, fission=True,
    )
    assert "初始比 未定义（初始损失为0）" in capsys.readouterr().out
    stage1_table = output / ("table.csv" if kind == "single_zero" else "table_stage1.csv")
    with stage1_table.open(newline="") as stream:
        table = list(csv.reader(stream))[1:]
    answers = StateRegistry(schema, specs).answers_for_rows([tuple(row) for row in table])
    with (output / "fission.csv").open(newline="") as stream:
        probe = list(csv.DictReader(stream))
    assert probe[0]["round"] == "0"
    assert [row["round"] for row in probe].count("final") == 1
    assert probe[-1]["round"] == "final"
    assert all(np.isfinite(float(row["l1_b"])) for row in probe)
    assert float(probe[-1]["l1_b"]) == np.mean(np.abs(target_from_specs(specs) + 0.25 - answers))
    assert float(probe[-1]["old_loss"]) == rp.weighted_half_loss(
        target_from_specs(specs), answers, np.ones(len(specs)),
    )
    if kind != "single_zero":
        info = json.loads((output / "table_lock.json").read_text())
        if kind in ("first_zero", "both_zero"):
            assert info["stage1_real_loss"] == 0.0 and info["final_over_stage1"] is None
        else:
            assert info["stage2_initial_loss"] == info["stage2_final_loss"] == 0.0


def test_zero_final_loss_keeps_defined_percentage(tmp_path, monkeypatch, capsys):
    _, _, _, _, final = _run_scene(tmp_path, monkeypatch, "positive_to_zero", "cpu")
    text = capsys.readouterr().out
    assert "初始比 0.0000%" in text and "未定义" not in text
    assert final == [("1",), ("1",)]


@pytest.mark.parametrize("shift,changed,target_final", [
    pytest.param(0.0, False, False, id="False-0.0"),
    pytest.param(0.25, False, False, id="False-0.25"),
    pytest.param(0.0, True, False, id="True-0.0"),
    pytest.param(0.25, True, False, id="True-0.25"),
    pytest.param(0.0, True, True, id="defined-zero"),
])
def test_lock_report_handles_undefined_and_normal_ratios(tmp_path, capsys, shift, changed, target_final):
    rp = _load_run_plants()
    schema = TableSchema(("a", "b", "c"), (("0", "1"),) * 3)
    x = _chain()
    rows = [tuple(map(str, row)) for row in x]
    target_rows = x.copy()
    if target_final:
        target_rows[0, 0] = 1
    specs = _specs(target_rows, schema, [(0, 1), (1, 2)], shift)
    y, w = target_from_specs(specs), np.ones(len(specs))
    measured = StateRegistry(schema, specs)
    stage1 = SimpleNamespace(
        y=y, w=w, final_loss=rp.weighted_half_loss(y, measured.answers_for_rows(rows), w), elapsed=0.0,
        out=SimpleNamespace(stop_reason="no_positive_direction", records=[None]),
    )
    lock = rp.lock_real_targets(schema, specs, stage1, rows)
    result = fiber_average(x.copy(), [2] * 3, [(0, 1), (1, 2)], 100, np.random.default_rng(7))
    extra = fiber_specs(result, schema, 1.0)
    combined = lock["locked_specs"] + extra
    registry = StateRegistry(schema, combined)
    y2, w2 = target_from_specs(combined), np.ones(len(combined))
    final_rows = list(rows)
    if changed:
        final_rows[0] = ("1", "0", "0")
    final = SimpleNamespace(
        y=y2, w=w2, registry=registry, elapsed=0.0,
        initial_loss=rp.weighted_half_loss(y2, registry.answers_for_rows(rows), w2),
        final_loss=rp.weighted_half_loss(y2, registry.answers_for_rows(final_rows), w2),
        out=SimpleNamespace(stop_reason="round_limit", records=[None]),
    )
    args = SimpleNamespace(stop_floor=0.0, save_table=str(tmp_path / "table.csv"), out="")
    rp.report_lock_check(args, lock, extra, stage1, final, final_rows, rows, 6.0)
    info = json.loads((tmp_path / "table_lock.json").read_text())
    text = capsys.readouterr().out
    assert info["final_real_loss"] == rp.weighted_half_loss(y, measured.answers_for_rows(final_rows), w)
    if stage1.final_loss == 0.0:
        assert info["final_over_stage1"] is None
        assert "第一段终表损失为0，比值未定义" in text
        assert info["final_real_loss"] == (1.0 if changed else 0.0)
    else:
        ratio = info["final_real_loss"] / info["stage1_real_loss"]
        assert info["final_over_stage1"] == ratio
        assert f"是第一段终表的 {ratio:.4f} 倍" in text and "比值未定义" not in text
        if target_final:
            assert ratio == 0.0
