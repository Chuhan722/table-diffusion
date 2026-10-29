"""二次审查回归：反向覆盖、记录数错配，以及有限目标的均值溢出。"""
import csv
from fractions import Fraction
import json
import sys
import warnings

import numpy as np
import pytest

from test_batchkernel import _scene
from test_earlystop import _evolve
from test_fiber_lock import _make_data
from test_fission_probe import _gpu_or_skip, _make_fission_exam
from test_groupfloor import _load_run_plants


def _runner(tmp_path, monkeypatch):
    data = _make_data(tmp_path)
    fission = _make_fission_exam(data)
    initial = tmp_path / "initial.csv"
    initial.write_bytes((data / "t300.csv").read_bytes())
    products = tmp_path / "products"
    products.mkdir()
    curve, table, prefix = products / "curve.csv", products / "table.csv", products / "s"
    paths = {
        "learning": data / "measured_eq.json", "fission": fission,
        "source": data / "t300.csv", "initial": initial,
        "curve": curve, "table": table, "snapshot": products / "s_snap0.csv",
        "groups_meta": products / "curve_groups.json", "probe": products / "curve_probe.csv",
        "snapshot_log": products / "s_snaps.csv", "stage1_curve": products / "curve_stage1.csv",
        "stage1_table": products / "table_stage1.csv", "stage2_snapshot": products / "s_stage2_snap0.csv",
        "stage2_log": products / "s_stage2_snaps.csv", "lock": products / "table_lock.json",
    }
    argv = [
        "run_plants.py", "--data", "t300", "--exam", "measured_eq.json", "--batched",
        "--init-table", str(initial), "--rounds", "1", "--out", str(curve), "--save-table", str(table),
        "--fission-exam", str(fission), "--fission-every", "1", "--fission-out", str(products / "fission.csv"),
        "--snapshot-group-floor", "--snapshot-loss", "1e12", "--snapshot-prefix", str(prefix),
        "--floor-group-min", "1", "--probe-interval", "1", "--fiber-passes", "1", "--fiber-rounds", "1",
        "--fiber-lock-real",
    ]
    rp = _load_run_plants()
    monkeypatch.setattr(rp, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    return rp, paths, argv


def _reject_without_writes(rp, tmp_path, monkeypatch):
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def unexpected(*args, **kwargs):
        pytest.fail("坏路径/记录数必须在生成开始前被拒绝")

    monkeypatch.setattr(rp, "run_stage", unexpected)
    with pytest.raises(SystemExit) as caught:
        rp.main()
    assert caught.value.code == 2
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("writer", ["curve", "table", "snapshot", "groups_meta"])
def test_outputs_cannot_directly_overwrite_fission_input(tmp_path, monkeypatch, writer):
    rp, paths, argv = _runner(tmp_path, monkeypatch)
    if writer in ("curve", "table"):
        option = "--out" if writer == "curve" else "--save-table"
        argv[argv.index(option) + 1] = str(paths["fission"])
    else:
        paths["fission"].rename(paths[writer])
        argv[argv.index("--fission-exam") + 1] = str(paths[writer])
    _reject_without_writes(rp, tmp_path, monkeypatch)


@pytest.mark.parametrize("writer", ["curve", "table", "snapshot", "groups_meta"])
@pytest.mark.parametrize("input_name", ["learning", "fission", "source", "initial"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_output_aliases_cannot_overwrite_any_input(tmp_path, monkeypatch, writer, input_name, alias):
    rp, paths, _ = _runner(tmp_path, monkeypatch)
    getattr(paths[writer], "symlink_to" if alias == "symlink" else "hardlink_to")(paths[input_name])
    _reject_without_writes(rp, tmp_path, monkeypatch)


@pytest.mark.parametrize("writer", [
    "probe", "snapshot_log", "stage1_curve", "stage1_table", "stage2_snapshot", "stage2_log", "lock",
])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_derived_output_aliases_cannot_overwrite_fission(tmp_path, monkeypatch, writer, alias):
    rp, paths, _ = _runner(tmp_path, monkeypatch)
    getattr(paths[writer], "symlink_to" if alias == "symlink" else "hardlink_to")(paths["fission"])
    _reject_without_writes(rp, tmp_path, monkeypatch)


@pytest.mark.parametrize("writer", ["curve", "snapshot", "probe", "groups_meta", "stage1_curve", "stage2_snapshot"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_normal_outputs_cannot_collide_with_each_other(tmp_path, monkeypatch, writer, alias):
    rp, paths, argv = _runner(tmp_path, monkeypatch)
    if alias == "direct":
        argv[argv.index("--save-table") + 1] = str(paths[writer])
    else:
        paths[writer].write_text("previous output\n")
        getattr(paths["table"], "symlink_to" if alias == "symlink" else "hardlink_to")(paths[writer])
    _reject_without_writes(rp, tmp_path, monkeypatch)


@pytest.mark.parametrize("input_name", ["learning", "fission"])
@pytest.mark.parametrize("record_count", [30, None, True, "300"])
def test_fission_requires_both_exams_to_match_row_count(tmp_path, monkeypatch, input_name, record_count):
    rp, paths, _ = _runner(tmp_path, monkeypatch)
    payload = json.loads(paths[input_name].read_text())
    if record_count is None:
        payload.pop("record_count")
    else:
        payload["record_count"] = record_count
    paths[input_name].write_text(json.dumps(payload))
    _reject_without_writes(rp, tmp_path, monkeypatch)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("target", [1e308, np.finfo(np.float64).max])
def test_large_finite_targets_keep_probe_mean_finite(backend, target):
    if backend == "gpu":
        _gpu_or_skip()
    registry, ids, weights, _ = _scene(2, num_rows=30)
    logged = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _evolve(
            registry, ids, weights, rounds=3, backend=backend,
            fission_targets=np.full(len(registry.specs), target), fission_every=1,
            fission_hook=lambda *args: logged.append(args),
            group_index=np.zeros(len(registry.specs), dtype=np.int64), group_floors=np.array([1e6]),
            fission_stop_z=2.0, fission_stop_runs=1,
        )
    assert out.stop_reason == "round_limit" and len(logged) == 3
    assert all(np.isfinite(row[2]) and row[2] == target for row in logged)
    assert all(row[-2:] == (0.0, 0.0) for row in logged[1:])


@pytest.mark.parametrize("target", [1e308, np.finfo(np.float64).max])
def test_runner_final_fission_row_stays_finite(tmp_path, monkeypatch, target):
    rp, paths, argv = _runner(tmp_path, monkeypatch)
    argv[argv.index("--fiber-passes") + 1] = "0"
    argv.remove("--fiber-lock-real")
    payload = json.loads(paths["fission"].read_text())
    for query in payload["queries"]:
        query["result"] = target
    paths["fission"].write_text(json.dumps(payload))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rp.main()
    with (tmp_path / "products/fission.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["round"] == "final"
    assert all(np.isfinite(float(row["l1_b"])) and float(row["l1_b"]) == target for row in rows)


@pytest.mark.parametrize("collision", ["file_parent", "input_parent", "directory"])
def test_file_and_directory_output_conflicts_are_rejected_early(tmp_path, monkeypatch, collision):
    rp, paths, argv = _runner(tmp_path, monkeypatch)
    if collision == "file_parent":
        argv[argv.index("--save-table") + 1] = str(paths["curve"] / "table.csv")
    elif collision == "input_parent":
        argv[argv.index("--out") + 1] = str(paths["source"] / "curve.csv")
    else:
        argv[argv.index("--out") + 1] = str(tmp_path / "products")
    _reject_without_writes(rp, tmp_path, monkeypatch)


def test_disabled_sidecars_do_not_reserve_output_paths(tmp_path, monkeypatch):
    rp, paths, argv = _runner(tmp_path, monkeypatch)
    for flag in ("--snapshot-group-floor", "--fiber-lock-real"):
        argv.remove(flag)
    for option in ("--snapshot-loss", "--snapshot-prefix"):
        start = argv.index(option)
        del argv[start:start + 2]
    for option in ("--fiber-passes", "--probe-interval"):
        argv[argv.index(option) + 1] = "0"
    argv[argv.index("--out") + 1] = str(tmp_path / "products/curve.json")
    argv[argv.index("--save-table") + 1] = str(paths["probe"])
    rp.main()
    assert paths["probe"].read_text().startswith("age,")
    assert (tmp_path / "products/curve.json").read_text().startswith("round,")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("values", [[0.0, 1.0, 2.0, 7.0], [0.0] * 6, [-2.5, 1.1, 8.0]])
def test_finite_mean_preserves_normal_reduction(backend, values):
    from resevo.fission import finite_mean

    xp = np if backend == "cpu" else _gpu_or_skip()
    array = xp.asarray(values, dtype=xp.float64)
    assert finite_mean(array, xp=xp) == float(xp.mean(array))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_overflow_fallback_matches_independent_rational_mean(backend):
    from resevo.fission import finite_mean

    xp = np if backend == "cpu" else _gpu_or_skip()
    values = [np.finfo(np.float64).max, 1e308, 0.0]
    expected = float(sum(Fraction.from_float(float(value)) for value in values) / len(values))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = finite_mean(xp.asarray(values), xp=xp)
    assert np.isfinite(result) and result == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("values", [[np.inf], [np.nan], [1.0, np.inf, -np.inf]])
def test_nonfinite_fission_diagnostics_are_rejected(backend, values):
    from resevo.fission import finite_mean

    xp = np if backend == "cpu" else _gpu_or_skip()
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:.*非有限"):
        finite_mean(xp.asarray(values), xp=xp)


@pytest.mark.parametrize("backend", ["batch", "gpu"])
def test_bad_fission_mean_keeps_numerical_failure_diagnostics(tmp_path, monkeypatch, recwarn, backend):
    from resevo import batchkernel
    from resevo.engine import NumericalEvolutionError
    from resevo.failure import save_numerical_failure
    from test_sampling_precision import _partial_run

    original = batchkernel.finite_mean
    calls = []

    def corrupt_second_read(values, *, xp):
        calls.append(None)
        if len(calls) == 2:
            values = xp.full_like(values, xp.inf)
        return original(values, xp=xp)

    monkeypatch.setattr(batchkernel, "finite_mean", corrupt_second_read)
    logged = []
    registry, initial, _, run = _partial_run(
        backend, round_limit=2, probe_interval=1,
        fission_targets=np.zeros(2), fission_every=1, fission_hook=lambda *args: logged.append(args),
        group_index=np.zeros(2, dtype=np.int64), group_floors=np.array([1e6]),
    )
    with pytest.raises(NumericalEvolutionError, match="分裂探针均值输入非有限") as caught:
        with save_numerical_failure(registry, tmp_path / "normal.csv"):
            run()
    partial = caught.value.partial_result
    assert len(logged) == 1 and len(partial.records) == 2
    assert partial.probes and partial.sampling_precision and partial.group_losses
    assert not np.array_equal(partial.state_ids, initial)
    assert not (tmp_path / "normal.csv").exists()
    manifest_path, = tmp_path.glob("normal.csv.failed-*/failure.json")
    manifest = json.loads(manifest_path.read_text())
    assert {"probes.csv", "sampling_precision.csv", "group_losses.csv"} <= manifest["files_sha256"].keys()
