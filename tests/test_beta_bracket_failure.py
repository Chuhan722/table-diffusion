"""beta 括根耗尽必须保存最新完成进度，不自动改变热启动或求解策略。"""
import csv
import hashlib
import json

import numpy as np
import pytest

from resevo.batchkernel import BatchRoundPlan, build_batch_kernel, evolve_batch
from resevo.batchmenu import BatchMenu, CodeBook, build_query_structure
from resevo.candidates import BlockSupport
from resevo.dataset import QuerySpec, StateRegistry, TableSchema
from resevo.editspace import tuples_from_ids
from resevo.engine import NumericalEvolutionError, build_kernel
from resevo.failure import save_numerical_failure
from resevo.grouping import group_state_ids
from resevo.state import make_workload
from test_local_gain_freeze import _gpu_context


def _scene(backend, first_weight):
    schema = TableSchema(("a", "b"), (("0", "1"), ("0", "1")))
    specs = [
        QuerySpec(name, ({"attribute": name, "operator": "==", "value": "1"},), 1.0)
        for name in schema.fields
    ]
    registry = StateRegistry(schema, specs)
    ids = registry.register_table([("0", "0")])
    registry.register_table([("1", "0"), ("1", "1")])
    target, weights = np.ones(2), np.array([first_weight, 1.0])
    workload = registry.build_workload(target, weights)
    book = CodeBook(registry)
    qs = build_query_structure(registry, book)
    if backend == "gpu":
        _gpu_context(qs, target, weights)

    def provider(current, round_index, frozen_streak=0):
        field = 0 if round_index == 0 else 1
        menu = BatchMenu(
            np.array([0, 1]), np.array([0]),
            np.array([[field, -1, -1]], dtype=np.int32),
            np.array([[1, -1, -1]], dtype=np.int32), np.ones(1),
        )
        return BatchRoundPlan("batch", workload, group_state_ids(current), menu)

    provider.codebook, provider.structure = book, [qs]
    return registry, ids, workload, book, qs, provider


def _run(scene, backend, rounds, rng):
    registry, ids, _, _, _, provider = scene
    return evolve_batch(ids, rounds, rng, provider, registry, backend=backend)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
@pytest.mark.parametrize("first_weight", (1e-80, 1e80), ids=("hint-too-large", "hint-too-small"))
def test_real_warm_bracket_failure_has_cold_start_control(backend, first_weight):
    scene = _scene(backend, first_weight)
    registry, _, workload, book, qs, provider = scene
    prefix = _run(scene, backend, 1, np.random.default_rng(9001))
    assert tuples_from_ids(registry, prefix.state_ids) == [("1", "0")]
    hint = prefix.records[0].beta
    assert hint > 1e70 if first_weight < 1 else 0 < hint < 1e-70
    plan = provider(prefix.state_ids, 1)

    def build(beta_hint):
        if backend == "gpu":
            return _gpu_context(qs, workload.target, workload.weights).build(
                plan.grouped, plan.menu, book.sync(), beta_hint=beta_hint,
            )
        return build_batch_kernel(
            workload, plan.grouped, plan.menu, qs, book.sync(), beta_hint=beta_hint,
        )

    cold = build(None)
    assert cold.status == "ok" and cold.beta == pytest.approx(2 * np.log(9), rel=1e-9)
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:.*无法括住 beta 根"):
        build(hint)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
@pytest.mark.parametrize("first_weight", (1e-80, 1e80), ids=("hint-too-large", "hint-too-small"))
def test_real_warm_failure_saves_current_table_and_completed_prefix(tmp_path, backend, first_weight):
    reference_scene = _scene(backend, first_weight)
    reference_rng = np.random.default_rng(9002)
    prefix = _run(reference_scene, backend, 1, reference_rng)
    scene = _scene(backend, first_weight)
    registry, original_ids, *_ = scene
    rng = np.random.default_rng(9002)
    output = tmp_path / "table.csv"
    with pytest.raises(NumericalEvolutionError, match="无法括住 beta 根") as caught:
        with save_numerical_failure(registry, output):
            _run(scene, backend, 2, rng)
    result = caught.value.partial_result
    assert result.stop_reason == "numerical_indeterminate"
    assert tuples_from_ids(registry, result.state_ids) == [("1", "0")]
    assert tuples_from_ids(registry, result.state_ids) != tuples_from_ids(registry, original_ids)
    assert len(result.records) == 1
    for field in ("round_index", "old_loss", "beta", "direction_gain", "interaction",
                  "step", "expected_loss", "status", "max_gain_sum", "mode", "candidates", "moved_rows"):
        assert getattr(result.records[0], field) == getattr(prefix.records[0], field)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert not output.exists()
    manifests = list(tmp_path.glob("table.csv.failed-*/failure.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed" and manifest["completed_rounds"] == 1
    assert manifest["normal_output_published"] is False
    assert manifest["checkpoint_phase"] == "before_failed_operation"
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((manifests[0].parent / name).read_bytes()).hexdigest() == digest
    with (manifests[0].parent / "current.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [["a", "b"], ["1", "0"]]
    with (manifests[0].parent / "records.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1 and records[0]["status"] == "ok"


def test_row_calibration_bracket_exhaustion_is_also_classified():
    workload = make_workload([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], [1.0, 1.0])
    supports = [BlockSupport((0,), ((1,), (2,)), (np.nextafter(0.0, 1.0), 1.0))]
    # 合法正mobility在参考归一化中下溢为0，原求解仍耗尽括根；本步只验证分类。
    with np.errstate(divide="ignore", under="ignore"):
        with pytest.raises(FloatingPointError, match="numerical_indeterminate:.*无法括住 beta 根"):
            build_kernel(workload, [0], supports)
