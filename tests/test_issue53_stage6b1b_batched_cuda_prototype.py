"""第 6B-1B 阶段多地址显卡人工原型测试。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.issue53_stage6b1b_batched_cuda_prototype import (
    run_same_workload_batched_prototype,
    run_variable_workloads_batched_prototype,
)
from scripts.issue53_stage6b1b_independent_cuda import (
    replay_gap_l1_batched_cuda,
)
from table_diffevo import gap_l1_diffusion as production
from table_diffevo.gap_l1_diffusion import evolve_step_gap_l1_global
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
    active = participate[:, None] & (
        current.to_numpy() != donors.to_numpy()
    )
    first_mask = np.zeros_like(active)
    first_mask[0, 1] = True
    first_mask[3, 2] = True
    second_mask = active & (
        np.random.default_rng(77).random(active.shape) < 0.5
    )
    return (
        schema,
        queries,
        current,
        donors,
        counts,
        targets,
        participate,
        (first_mask, second_mask),
    )


def _variable_addresses(current):
    donors = (
        pd.DataFrame({
            "a": [1, 1, 0, 1, 1, 1],
            "b": [1, 0, 1, 1, 0, 0],
            "c": [1, 1, 1, 0, 1, 0],
        }),
        current.iloc[[1, 2, 3, 4, 5, 0]].reset_index(drop=True),
        pd.DataFrame(1 - current.to_numpy(), columns=current.columns),
        current.copy(deep=True),
    )
    participates = (
        np.array([True, True, False, True, True, True]),
        np.array([True, False, True, True, False, True]),
        np.array([False, True, False, False, True, False]),
        np.ones(len(current), dtype=bool),
    )
    initial_masks = []
    for index, (donor, participate) in enumerate(
        zip(donors, participates)
    ):
        active = participate[:, None] & (
            current.to_numpy() != donor.to_numpy()
        )
        initial_masks.append(active & (
            np.random.default_rng(900 + index).random(active.shape) < 0.5
        ))
    return donors, participates, tuple(initial_masks)


def test_prototype_stays_isolated_while_audited_batch_entries_are_wired():
    needle = "issue53_stage6b1b_batched_cuda_prototype"
    production_paths = (
        "src/table_diffevo/gap_l1_diffusion.py",
        "scripts/calibrate_issue53_stage6b1_gap_l1.py",
        "scripts/collect_issue53_stage6b1_screen.py",
        "scripts/audit_issue53_stage6b1_structure.py",
        "scripts/evaluate_issue53_stage6b1_screen.py",
        "scripts/audit_issue53_stage6b1_arithmetic.py",
    )
    for relative in production_paths:
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert needle not in text
    batch_entry = "evolve_step_gap_l1_global_batched"
    production_batch_paths = {
        "scripts/collect_issue53_stage6b1_screen.py",
        "scripts/audit_issue53_stage6b1_structure.py",
    }
    for relative in production_paths[1:]:
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert (batch_entry in text) is (relative in production_batch_paths)
    independent_batch_entry = "replay_gap_l1_batched_cuda"
    independent_batch_paths = {
        "scripts/audit_issue53_stage6b1_arithmetic.py",
    }
    for relative in production_paths[1:]:
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert (independent_batch_entry in text) is (
            relative in independent_batch_paths
        )


def test_prototype_rejects_empty_address_batch_before_cuda_use():
    with pytest.raises(ValueError, match="同长度非空序列"):
        run_same_workload_batched_prototype(
            None,
            None,
            None,
            [],
            None,
            None,
            participate=None,
            initial_masks=(),
            reference_scale=1.0,
            seeds=(),
        )


def test_independent_batch_auditor_rejects_empty_batch_before_cuda_use():
    with pytest.raises(ValueError, match="同长度非空序列"):
        replay_gap_l1_batched_cuda(
            None,
            (),
            None,
            (),
            None,
            None,
            (),
            (),
            reference_scale=1.0,
            seeds=(),
            n_sweeps=8,
            eta=0.5,
            strength=2.0,
            floor=8.0,
            logit_clip=30.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA（显卡运行时）不可用"
)
def test_two_address_prototype_matches_existing_single_address_replays():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        (
            schema,
            queries,
            current,
            donors,
            counts,
            targets,
            participate,
            initial_masks,
        ) = _case()
        seeds = (20260826, 20260827)
        references = [
            evolve_step_gap_l1_global(
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
            for initial_mask, seed in zip(initial_masks, seeds)
        ]
        result = run_same_workload_batched_prototype(
            current,
            donors,
            schema,
            queries,
            targets,
            counts,
            participate=participate,
            initial_masks=initial_masks,
            reference_scale=0.02,
            seeds=seeds,
            n_sweeps=8,
        )
    finally:
        torch.use_deterministic_algorithms(previous)

    assert result["batch_size"] == 2
    assert result["production_batch_kernel_called"] is True
    assert result["production_pipeline_enabled"] is False
    assert result["formal_state_or_address_read"] is False
    for index, reference in enumerate(references):
        pd.testing.assert_frame_equal(result["tables"][index], reference[0])
        np.testing.assert_array_equal(result["masks"][index], reference[1])
        np.testing.assert_array_equal(
            result["final_query_counts"][index],
            reference[2]["final_query_counts"],
        )
        assert result["trace_sha256"][index] == (
            reference[2]["microstep_trace_sha256"]
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA（显卡运行时）不可用"
)
def test_variable_address_prototype_preserves_each_reference_chain():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    original_trace = production._update_trace
    try:
        (
            schema,
            queries,
            current,
            _,
            counts,
            targets,
            _,
            _,
        ) = _case()
        donors, participates, initial_masks = _variable_addresses(current)
        seeds = (20260831, 20260832, 20260833, 20260834)
        references = []
        reference_records = []
        for donor, participate, initial_mask, seed in zip(
            donors, participates, initial_masks, seeds
        ):
            captured = []

            def capture(digest, **values):
                captured.append(dict(values))
                return original_trace(digest, **values)

            production._update_trace = capture
            references.append(evolve_step_gap_l1_global(
                current,
                donor,
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
            ))
            reference_records.append(captured)
        production._update_trace = original_trace
        result = run_variable_workloads_batched_prototype(
            current,
            donors,
            schema,
            queries,
            targets,
            counts,
            participates=participates,
            initial_masks=initial_masks,
            reference_scale=0.02,
            seeds=seeds,
            n_sweeps=8,
        )
    finally:
        production._update_trace = original_trace
        torch.use_deterministic_algorithms(previous)

    assert result["batch_size"] == 4
    assert len(set(result["active_switches_k_by_address"])) > 1
    assert 0 in result["active_switches_k_by_address"]
    assert result["padding_microsteps"] > 0
    assert result["different_address_structures_enabled"] is True
    assert result["strict_internal_order_preserved"] is True
    for index, (reference, records) in enumerate(
        zip(references, reference_records)
    ):
        coordinates = np.asarray([
            [record["row_index"], record["attribute_index"]]
            for record in records
        ], dtype=np.int64).reshape((-1, 2))
        random_rolls = np.asarray([
            record["random_roll"] for record in records
        ], dtype=np.float64)
        reference_floats = np.asarray([
            [
                record["e0"],
                record["e1"],
                record["score"],
                record["normalized_score"],
                record["raw_logit"],
                record["probability"],
                record["random_roll"],
            ]
            for record in records
        ], dtype=np.float64).reshape((-1, 7))
        reference_booleans = np.asarray([
            [record["before"], record["after"], record["clipped"]]
            for record in records
        ], dtype=bool).reshape((-1, 3))
        np.testing.assert_array_equal(
            result["coordinates"][index], coordinates
        )
        np.testing.assert_array_equal(
            result["random_rolls"][index], random_rolls
        )
        np.testing.assert_allclose(
            result["float_values"][index],
            reference_floats,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            result["bool_values"][index], reference_booleans
        )
        pd.testing.assert_frame_equal(result["tables"][index], reference[0])
        np.testing.assert_array_equal(result["masks"][index], reference[1])
        np.testing.assert_array_equal(
            result["final_query_counts"][index],
            reference[2]["final_query_counts"],
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA（显卡运行时）不可用"
)
def test_independent_batch_auditor_matches_production_batch_bit_for_bit():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        (
            schema,
            queries,
            current,
            _,
            counts,
            targets,
            _,
            _,
        ) = _case()
        donors, participates, initial_masks = _variable_addresses(current)
        seeds = (20260901, 20260902, 20260903, 20260904)
        production_result = run_variable_workloads_batched_prototype(
            current,
            donors,
            schema,
            queries,
            targets,
            counts,
            participates=participates,
            initial_masks=initial_masks,
            reference_scale=0.02,
            seeds=seeds,
            n_sweeps=8,
        )
        independent_result = replay_gap_l1_batched_cuda(
            current,
            donors,
            schema,
            queries,
            targets,
            counts,
            participates,
            initial_masks,
            reference_scale=0.02,
            seeds=seeds,
            n_sweeps=8,
            eta=0.5,
            strength=2.0,
            floor=8.0,
            logit_clip=30.0,
            capture_trace=True,
        )
    finally:
        torch.use_deterministic_algorithms(previous)

    for key in (
        "batch_size",
        "active_switches_k_by_address",
        "microsteps_by_address",
        "maximum_padded_microsteps",
        "padding_microsteps",
    ):
        assert independent_result[key] == production_result[key]
    np.testing.assert_array_equal(
        independent_result["valid_step_mask"],
        production_result["valid_step_mask"],
    )
    np.testing.assert_array_equal(
        independent_result["masks"], production_result["masks"]
    )
    np.testing.assert_array_equal(
        independent_result["final_query_counts"],
        production_result["final_query_counts"],
    )
    assert independent_result["trace_sha256"] == (
        production_result["trace_sha256"]
    )
    assert independent_result["no_gate"] is True
    assert (
        independent_result[
            "acceptance_rejection_or_selection_performed"
        ]
        is False
    )
    assert independent_result["production_kernel_called"] is False
    assert independent_result["formal_pipeline_enabled"] is False
    assert 0 in independent_result["active_switches_k_by_address"]
    assert independent_result["padding_microsteps"] > 0
    for index in range(independent_result["batch_size"]):
        pd.testing.assert_frame_equal(
            independent_result["tables"][index],
            production_result["tables"][index],
        )
        np.testing.assert_array_equal(
            independent_result["coordinates"][index],
            production_result["coordinates"][index],
        )
        np.testing.assert_array_equal(
            independent_result["random_rolls"][index],
            production_result["random_rolls"][index],
        )
        np.testing.assert_array_equal(
            independent_result["float_values"][index],
            production_result["float_values"][index],
        )
        np.testing.assert_array_equal(
            independent_result["bool_values"][index],
            production_result["bool_values"][index],
        )
        diagnostics = independent_result["diagnostics"][index]
        assert diagnostics["no_gate"] is True
        assert diagnostics["trace_sha256"] == (
            independent_result["trace_sha256"][index]
        )
        assert len(independent_result["trace_records"][index]) == (
            independent_result["microsteps_by_address"][index]
        )
