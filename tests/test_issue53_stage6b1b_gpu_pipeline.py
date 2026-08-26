"""第 6B-1B 阶段差量协议、复用接线与独立显卡重放测试。"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    torch = None

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()

from scripts import issue53_stage6b1b_independent_cuda as independent_cuda
from scripts import issue53_stage6b1b_protocol as protocol
from scripts.check_issue53_stage6b1b_nltcs_gpu_environment import (
    validate_gpu_environment,
)
from table_diffevo.gap_l1_diffusion import (
    evolve_step_gap_l1_global,
    isolated_gap_l1_scores,
)
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _schema(*names: str) -> Schema:
    return Schema([
        AttributeBlock(
            name=name,
            type="categorical",
            description=name,
            values=[0, 1],
        )
        for name in names
    ])


def _equals(attribute: str) -> dict:
    return {"attribute": attribute, "operator": "==", "value": 1}


def _case():
    schema = _schema("a", "b", "c")
    queries = [
        {"conditions": [_equals("a")]},
        {"conditions": [_equals("b")]},
        {"conditions": [_equals("a"), _equals("b")]},
        {"conditions": [_equals("b"), _equals("c")]},
        {"conditions": [_equals("a"), _equals("b"), _equals("c")]},
    ]
    current = pd.DataFrame({
        "a": [0, 0, 1, 0, 1, 0],
        "b": [0, 1, 0, 0, 1, 1],
        "c": [0, 0, 0, 1, 0, 1],
    })
    donors = pd.DataFrame({
        "a": [1, 1, 0, 1, 1, 1],
        "b": [1, 0, 1, 1, 0, 0],
        "c": [1, 1, 1, 0, 1, 0],
    })
    counts = evaluate_table(current, queries)
    targets = np.array([3.5, 2.5, 2.0, 1.5, 1.0])
    participate = np.array([True, True, False, True, True, True])
    initial_mask = np.zeros((len(current), 3), dtype=bool)
    initial_mask[0, 1] = True
    initial_mask[3, 2] = True
    return (
        schema,
        queries,
        current,
        donors,
        counts,
        targets,
        participate,
        initial_mask,
    )


def test_delta_protocol_freezes_only_nltcs_gpu_matrix():
    manifest = protocol.frozen_protocol_manifest()
    assert protocol.DATASET_ORDER == ("nltcs",)
    assert len(protocol.expected_pair_ids("formal")) == 500
    assert len(protocol.expected_pair_ids("smoke")) == 10
    assert manifest["formal_matrix"] == {
        "state_count": 25,
        "pair_count": 500,
        "arm_leg_record_count": 3000,
    }
    assert manifest["gpu_delta"]["new_kernel_backend"] == (
        "torch_cuda_float64"
    )
    assert manifest["gpu_delta"]["automatic_cpu_fallback"] is False
    assert manifest["gpu_delta"]["formal_address_batch_size"] == 1
    assert manifest["inherited_method_identity"]["no_gate_contract"] == (
        protocol.NO_GATE_CONTRACT
    )


def test_delta_protocol_document_manifest_and_stage1_entry_are_bound():
    assert protocol.file_sha256(
        REPOSITORY_ROOT / protocol.PROTOCOL_DOC
    ) == protocol.PROTOCOL_DOC_SHA256
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)


def test_pipeline_selector_reuses_existing_calibration_entrypoint():
    environment = os.environ.copy()
    environment["ISSUE53_STAGE6B1_PROTOCOL"] = "stage6b1b"
    environment["PYTHONPATH"] = "src:."
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.calibrate_issue53_stage6b1_gap_l1",
            "plan",
            "--mode",
            "smoke",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["datasets"] == ["nltcs"]
    assert plan["pair_count"] == 10
    assert plan["arm_leg_record_count"] == 60
    assert plan["new_kernel_backend"] == "torch_cuda_float64"
    assert plan["source_read_started"] is False
    assert plan["calibration_started"] is False


def test_independent_cuda_module_does_not_import_production_gap_kernel():
    path = REPOSITORY_ROOT / "scripts/issue53_stage6b1b_independent_cuda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "table_diffevo.gap_l1_diffusion" not in imported
    assert "table_diffevo.vectorized_eval" not in imported


def test_wrong_physical_gpu_selector_is_rejected_before_cuda_use(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="物理 1 号显卡"):
        validate_gpu_environment()


@pytest.mark.parametrize(
    ("nltcs_label", "expected"),
    [
        ("gap_kernel_development_supported", "shared_development_support"),
        (
            "no_stable_gap_error_gain",
            "dataset_dependent_development_support",
        ),
        ("execution_invalid", "inconclusive_or_invalid_screen"),
    ],
)
def test_stage2_combines_with_frozen_stage1_result(nltcs_label, expected):
    assert protocol.classify_stage1({"nltcs": nltcs_label}) == expected


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA（显卡运行时）不可用",
)
def test_production_and_independent_cuda_replay_are_bit_identical():
    torch.use_deterministic_algorithms(True)
    (
        schema,
        queries,
        current,
        donors,
        counts,
        targets,
        participate,
        initial_mask,
    ) = _case()
    seed = 20260826
    production = evolve_step_gap_l1_global(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate=participate,
        initial_mask=initial_mask,
        reference_scale=0.02,
        rng=np.random.default_rng(seed),
        n_sweeps=8,
        device="cuda",
    )
    independent = independent_cuda.replay_gap_l1_cuda(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        participate,
        initial_mask,
        reference_scale=0.02,
        seed=seed,
        n_sweeps=8,
        eta=0.5,
        strength=2.0,
        floor=8.0,
        logit_clip=30.0,
    )
    pd.testing.assert_frame_equal(production[0], independent[0])
    np.testing.assert_array_equal(production[1], independent[1])
    assert production[2]["microstep_trace_sha256"] == (
        independent[2]["trace_sha256"]
    )
    assert production[2]["final_query_counts"] == (
        independent[2]["final_query_counts"].tolist()
    )


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA（显卡运行时）不可用",
)
def test_production_and_independent_cuda_calibration_scores_are_exact():
    torch.use_deterministic_algorithms(True)
    schema, queries, current, donors, counts, targets, *_ = _case()
    exact_numerators = np.array([35, 25, 20, 15, 10], dtype=np.int64)
    production = isolated_gap_l1_scores(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        exact_target_numerators=exact_numerators,
        exact_target_denominator=10,
        device="cuda",
    )
    coordinates, scores, _ = independent_cuda.isolated_gap_l1_scores_cuda(
        current,
        donors,
        schema,
        queries,
        targets,
        counts,
        floor=8.0,
        exact_target_numerators=exact_numerators,
        exact_target_denominator=10,
    )
    np.testing.assert_array_equal(production["coordinates"], coordinates)
    np.testing.assert_array_equal(production["scores"], scores)
