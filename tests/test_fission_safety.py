"""PR81 审查反例：文件别名、独立计数、临时注册表和退化配对统计。"""
import csv
import hashlib
import json
import sys

import numpy as np
import pytest

from resevo.batchkernel import BatchRoundPlan, evolve_batch
from resevo.batchmenu import CodeBook, build_query_structure
from resevo.dataset import QuerySpec, StateRegistry, TableSchema
from resevo.editspace import EditBudget
from resevo.pairing import build_rescue_menu
from test_fiber_lock import _make_data
from test_fission_probe import _gpu_or_skip, _make_fission_exam
from test_groupfloor import _load_run_plants


@pytest.mark.parametrize("destination", [
    "learning", "fission", "source", "initial", "curve", "table", "probe",
    "groups", "groups_json", "snapshot", "snapshot_log", "loss_snapshot",
    "stage1_curve", "stage1_probe", "stage1_table", "stage2_snapshot",
    "stage2_snapshot_log", "lock", "existing",
])
def test_runner_rejects_fission_path_conflicts_before_writing(tmp_path, monkeypatch, destination):
    data = _make_data(tmp_path)
    fission = _make_fission_exam(data)
    initial = tmp_path / "initial.csv"
    initial.write_bytes((data / "t300.csv").read_bytes())
    out = tmp_path / "products"
    curve, table, prefix = out / "curve.csv", out / "table.csv", out / "s"
    paths = {
        "learning": data / "measured_eq.json", "fission": fission,
        "source": data / "t300.csv", "initial": initial,
        "curve": curve, "table": table, "probe": out / "curve_probe.csv",
        "groups": out / "curve_groups.csv", "groups_json": out / "curve_groups.json",
        "snapshot": out / "s_snapG.csv", "snapshot_log": out / "s_snaps.csv",
        "loss_snapshot": out / "s_snap0.csv", "stage1_curve": out / "curve_stage1.csv",
        "stage1_probe": out / "curve_stage1_probe.csv", "stage1_table": out / "table_stage1.csv",
        "stage2_snapshot": out / "s_stage2_snap0.csv",
        "stage2_snapshot_log": out / "s_stage2_snaps.csv", "lock": out / "table_lock.json",
        "existing": tmp_path / "previous_fission.csv",
    }
    paths["existing"].write_text("previous result\n", encoding="utf-8")
    before = {p: p.read_bytes() for p in (paths["learning"], fission, paths["source"], initial, paths["existing"])}
    rp = _load_run_plants()
    monkeypatch.setattr(rp, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_plants.py", "--data", "t300", "--exam", "measured_eq.json", "--batched",
        "--init-table", str(initial), "--rounds", "1", "--out", str(curve), "--save-table", str(table),
        "--fission-exam", str(fission), "--fission-every", "1", "--fission-out", str(paths[destination]),
        "--snapshot-group-floor", "--snapshot-loss", "1000000000000", "--snapshot-prefix", str(prefix),
        "--floor-group-min", "1", "--probe-interval", "1", "--fiber-passes", "1", "--fiber-rounds", "1", "--fiber-lock-real",
    ])
    with pytest.raises(SystemExit) as caught:
        rp.main()
    assert caught.value.code == 2
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not out.exists(), "路径冲突必须在任何正常产物写出前拒绝"


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "relative", "dangling"])
def test_runner_rejects_fission_file_aliases(tmp_path, monkeypatch, alias):
    data = _make_data(tmp_path)
    fission = _make_fission_exam(data)
    original = fission.read_bytes()
    target = tmp_path / "alias.csv"
    if alias == "symlink":
        target.symlink_to(fission)
    elif alias == "hardlink":
        target.hardlink_to(fission)
    elif alias == "relative":
        target = data / ".." / "t300" / "fisB.json"
    else:
        target.symlink_to(tmp_path / "not_created.csv")
    rp = _load_run_plants()
    monkeypatch.setattr(rp, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_plants.py", "--data", "t300", "--exam", "measured_eq.json", "--batched", "--uniform-init",
        "--rounds", "1", "--out", str(tmp_path / "curve.csv"),
        "--fission-exam", str(fission), "--fission-every", "1", "--fission-out", str(target),
    ])
    with pytest.raises(SystemExit) as caught:
        rp.main()
    assert caught.value.code == 2
    assert fission.read_bytes() == original
    assert not (tmp_path / "curve.csv").exists()
    assert not (tmp_path / "not_created.csv").exists()


@pytest.mark.parametrize("mode", ["rescue", "tree"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cpu_skips_local_registry_even_when_ids_fit(mode, seed):
    schema = TableSchema(("v",), (("0", "1"),))
    specs = [QuerySpec("q", ({"attribute": "v", "operator": "==", "value": "1"},), 0.0)]
    registry = StateRegistry(schema, specs)
    registry.register_table([("0",)])
    ids = registry.register_table([("1",)] * 5)
    small, local_ids, sub_idx, wl, supports = build_rescue_menu(
        registry, ids, np.array([0.0]), np.ones(1), np.random.default_rng(seed), 1,
        budget=EditBudget(donor_copies=0, joint_edits=0, explore_edits=0),
    )
    # 全局 1 是 v=1，临时 1 是 v=0；旧的矩阵行数检查会通过，却读到相反的状态。
    assert registry.state_tuple(1) == ("1",) and small.state_tuple(1) == ("0",)
    assert wl.features.shape[0] > ids.max()
    plan = BatchRoundPlan(mode, wl, supports=supports, rescue_ids=local_ids,
                          rescue_sub_idx=sub_idx, rescue_registry=small)
    from resevo.candidates import BlockSupport

    complete = registry.build_workload(np.zeros(1), np.ones(1))
    before_full = []

    def provide(current, k, streak):
        if k == 0:
            return plan
        before_full.append(current.copy())
        return BatchRoundPlan("rows", complete, supports=[BlockSupport((i,), ()) for i in range(len(current))])

    logged, snapshots = [], []
    result = evolve_batch(
        ids, 2, np.random.default_rng(seed), provide, registry,
        fission_targets=np.zeros(1), fission_every=1, fission_hook=lambda *args: logged.append(args),
        group_index=np.zeros(1, dtype=np.int64), group_floors=np.array([100.0]),
        group_snapshot_hook=lambda *args: snapshots.append(args),
    )
    assert len(logged) == len(snapshots) == len(result.group_losses) == 1
    actual = float(sum(registry.state_tuple(int(s)) == ("1",) for s in before_full[0]))
    assert logged[0][0] == snapshots[0][0] == result.group_losses[0][0] == 1
    assert logged[0][2] == actual and logged[0][4:] == (True, None, None)
    np.testing.assert_array_equal(snapshots[0][3], before_full[0])
    np.testing.assert_array_equal(result.group_losses[0][1], [actual * actual / 2.0])


@pytest.mark.parametrize("mode", ["generic", "direct", "sparse"])
@pytest.mark.parametrize("training_target", [0.0, 1e16])
def test_gpu_fission_uses_original_answers(mode, training_target):
    cp = _gpu_or_skip()
    from resevo.gpukernel import GpuBatchContext

    schema = TableSchema(("v",), (("0", "1"),))
    specs = [QuerySpec(str(i), ({"attribute": "v", "operator": "==", "value": str(i)},), training_target)
             for i in range(2)]
    registry = StateRegistry(schema, specs)
    ids = registry.register_table([("1",)])
    book = CodeBook(registry)
    ctx = GpuBatchContext(build_query_structure(registry, book), np.full(2, training_target), np.ones(2),
                          direct=mode != "generic", sparse=mode == "sparse")
    got = cp.asnumpy(ctx.fission_errors_of(book.sync(), ids, cp.asarray([0.0, 1.0])))
    np.testing.assert_array_equal(got, [0.0, 0.0])
    opposite = cp.asnumpy(ctx.fission_errors_of(book.sync(), ids, cp.asarray([1.0, 0.0])))
    np.testing.assert_array_equal(opposite, [1.0, 1.0])


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("query_count", [1, 2])
def test_zero_se_does_not_trigger_fission_stop(moving, query_count):
    from resevo.candidates import BlockSupport

    schema = TableSchema(("v",), (("0", "1"),))
    specs = [QuerySpec(str(i), ({"attribute": "v", "operator": "==", "value": "1"},), 3.0)
             for i in range(query_count)]
    registry = StateRegistry(schema, specs)
    ids = registry.register_table([("0",)] * 5)
    destination = int(registry.register_table([("1",)])[0])
    wl = registry.build_workload(np.full(query_count, 3.0), np.ones(query_count))

    def provide(current, k, streak):
        supports = [BlockSupport((i,), ((destination,),) if moving else ()) for i in range(len(current))]
        return BatchRoundPlan("rows", wl, supports=supports)

    logged = []
    result = evolve_batch(
        ids, 5, np.random.default_rng(7), provide, registry, max_frozen_retries=10,
        fission_targets=np.zeros(query_count), fission_every=1, fission_hook=lambda *args: logged.append(args),
        fission_stop_z=2.0, fission_stop_runs=1,
        group_index=np.zeros(query_count, dtype=np.int64), group_floors=np.array([1e6]),
    )
    assert result.stop_reason == "round_limit"
    assert len(logged) == 5 and all(row[-1] == 0.0 for row in logged[1:])
    if moving:
        assert any(row[-2] > 0.0 for row in logged[1:])
    else:
        assert all(row[-2] == 0.0 for row in logged[1:])


def test_runner_never_overwrites_fission_file_created_after_validation(tmp_path, monkeypatch):
    data = _make_data(tmp_path)
    fission = _make_fission_exam(data)
    destination = tmp_path / "fission.csv"
    rp = _load_run_plants()
    original = rp.run_stage

    def intervene(*args, **kwargs):
        destination.write_text("another process wrote this\n", encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(rp, "run_stage", intervene)
    monkeypatch.setattr(rp, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_plants.py", "--data", "t300", "--exam", "measured_eq.json", "--batched", "--uniform-init",
        "--rounds", "1", "--out", str(tmp_path / "curve.csv"),
        "--fission-exam", str(fission), "--fission-every", "1", "--fission-out", str(destination),
    ])
    with pytest.raises(SystemExit) as caught:
        rp.main()
    assert caught.value.code == 2
    assert destination.read_text() == "another process wrote this\n"
    assert not (tmp_path / "curve.csv").exists()


@pytest.mark.parametrize("backend", ["batch", "gpu"])
@pytest.mark.parametrize("unavailable", [False, True])
def test_fission_stop_and_failure_preserve_other_diagnostics(tmp_path, monkeypatch, recwarn, backend, unavailable):
    from resevo.engine import NumericalEvolutionError
    from resevo.failure import save_numerical_failure
    from test_sampling_precision import _partial_run

    if unavailable:
        from resevo import probe

        def fail_probe(*args, **kwargs):
            raise FloatingPointError("numerical_indeterminate: combined fission fixture")

        monkeypatch.setattr(probe, "analytic_step", fail_probe)
    logged = []
    controls = dict(
        probe_interval=1, group_index=np.zeros(2, dtype=np.int64), group_floors=np.array([1e6]),
        fission_targets=np.zeros(2), fission_every=1, fission_hook=lambda *args: logged.append(args),
    )
    _, _, _, run = _partial_run(backend, round_limit=3, fission_stop_z=0.5, fission_stop_runs=1, **controls)
    result = run()
    assert result.stop_reason == "fission_rise" and len(result.records) == 2
    assert result.sampling_precision and result.sampling_precision[0].round_index == 0
    assert result.probes and result.probes[0].diagnostics.status == ("unavailable" if unavailable else "ok")
    rp = _load_run_plants()
    rp.write_curve(tmp_path / "curve.csv", result, result.records[-1].old_loss)
    with (tmp_path / "curve.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["status"] == "fission_rise" and rows[-1]["sampling_limited_segments"] == "1"
    assert (tmp_path / "curve_probe.csv").exists()

    logged.clear()
    registry, _, _, run = _partial_run(backend, fail_after_first=True, **controls)
    with pytest.raises(NumericalEvolutionError):
        with save_numerical_failure(registry, tmp_path / "failed_table.csv"):
            run()
    assert len(logged) == 1 and logged[0][0] == 0
    manifest_path, = tmp_path.glob("failed_table.csv.failed-*/failure.json")
    manifest = json.loads(manifest_path.read_text())
    assert {"probes.csv", "sampling_precision.csv", "group_losses.csv"} <= manifest["files_sha256"].keys()
    assert all(hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() == digest
               for name, digest in manifest["files_sha256"].items())
    assert not (tmp_path / "failed_table.csv").exists()
