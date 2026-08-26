"""第 6B-1B 阶段批量协议的无正式数据接线测试。"""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    torch = None

from scripts import audit_issue53_stage6b1_arithmetic as arithmetic
from scripts import audit_issue53_stage6b1_structure as structural
from scripts import collect_issue53_stage6b1_screen as collector
from scripts import issue53_stage6b1b_batched_protocol as protocol
from scripts import issue53_stage6b1b_independent_cuda as independent_cuda
from scripts import issue53_stage6b1b_protocol as old_protocol
from table_diffevo import gap_l1_diffusion as production
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "scripts.calibrate_issue53_stage6b1_gap_l1",
        "scripts.collect_issue53_stage6b1_screen",
        "scripts.audit_issue53_stage6b1_structure",
        "scripts.evaluate_issue53_stage6b1_screen",
        "scripts.audit_issue53_stage6b1_arithmetic",
    ],
)
def test_all_plan_entries_select_batch_protocol_without_source_read(module):
    environment = os.environ.copy()
    environment["ISSUE53_STAGE6B1_PROTOCOL"] = "stage6b1b_batched"
    environment["PYTHONPATH"] = "src:."
    result = subprocess.run(
        [sys.executable, "-m", module, "plan", "--mode", "smoke"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["protocol_sha256"] == protocol.FROZEN_PROTOCOL_SHA256
    assert plan["datasets"] == ["nltcs"]
    assert plan["pair_count"] == 10
    assert plan["gap_l1_address_batch_size"] == 2
    assert plan["gap_l1_state_batch_count"] == 5
    assert plan["production_batch_backend"] == (
        "torch_cuda_float64_batched"
    )
    assert plan["calibration_gap_backend"] == "torch_cuda_float64"
    assert plan["pipeline_wiring_available"] is True
    assert plan["source_read_started"] is False
    assert plan.get("generation_started", False) is False
    assert plan.get("audit_started", False) is False
    assert plan.get("calibration_started", False) is False


def test_batch_protocol_accepts_only_new_frozen_output_paths():
    expected = (
        REPOSITORY_ROOT
        / protocol.SMOKE_OUTPUT_DIR
        / protocol.COLLECTION_FILENAME
    )
    protocol.assert_stage_output_path(
        REPOSITORY_ROOT, "smoke", "collection", expected
    )
    legacy = (
        REPOSITORY_ROOT
        / old_protocol.SMOKE_OUTPUT_DIR
        / old_protocol.COLLECTION_FILENAME
    )
    with pytest.raises(ValueError, match="全新冻结目录"):
        protocol.assert_stage_output_path(
            REPOSITORY_ROOT, "smoke", "collection", legacy
        )


def test_collector_batches_one_state_once_in_proposal_order(monkeypatch):
    monkeypatch.setattr(collector, "protocol", protocol)
    prepared_by_index = {}

    def prepare(_context, proposal_index, **_kwargs):
        item = collector._PreparedPair(
            started=0.0,
            proposal_index=proposal_index,
            replay=SimpleNamespace(
                donors=f"donors-{proposal_index}",
                common_update={
                    "participate": f"participate-{proposal_index}",
                    "copy_masks": f"mask-{proposal_index}",
                },
            ),
            mutation_specs=(),
            independent={},
            factor={},
            gap_seed=100 + proposal_index,
            gap_rng=np.random.default_rng(100 + proposal_index),
            gap_initial_rng_sha=f"initial-{proposal_index}",
        )
        prepared_by_index[proposal_index] = item
        return item

    calls = []

    def batch(current, donors, schema, queries, target, counts, **kwargs):
        calls.append((current, donors, schema, queries, target, counts, kwargs))
        return tuple(
            (f"table-{index}", f"final-mask-{index}", {"index": index})
            for index in range(2)
        )

    def finish(_context, item, scale, gap_result):
        return {
            "proposal_index": item.proposal_index,
            "scale": scale,
            "gap_result": gap_result,
        }

    monkeypatch.setattr(collector, "_prepare_pair", prepare)
    monkeypatch.setattr(collector, "_finish_pair", finish)
    monkeypatch.setattr(
        collector, "evolve_step_gap_l1_global_batched", batch
    )
    context = SimpleNamespace(
        current="current",
        query_counts="counts",
        runtime=SimpleNamespace(
            dataset="nltcs",
            schema="schema",
            queries="queries",
            runtime_target="target",
        ),
    )
    rows = collector._collect_state_batched(
        context,
        0.25,
        mode="smoke",
        factor_compiled="factor-compiled",
        gap_compiled="gap-compiled",
    )

    assert len(calls) == 1
    assert calls[0][1] == ["donors-0", "donors-1"]
    assert calls[0][6]["participates"] == [
        "participate-0",
        "participate-1",
    ]
    assert calls[0][6]["initial_masks"] == ["mask-0", "mask-1"]
    assert calls[0][6]["compiled_workload"] == "gap-compiled"
    assert [row["proposal_index"] for row in rows] == [0, 1]
    assert [row["gap_result"][2]["index"] for row in rows] == [0, 1]
    assert list(prepared_by_index) == [0, 1]


def test_independent_auditor_replays_one_state_as_one_batch(monkeypatch):
    monkeypatch.setattr(arithmetic, "protocol", protocol)
    prepared = []

    def prepare(_context, pair, _scale):
        item = SimpleNamespace(
            pair=pair,
            donors=f"donors-{pair['proposal_index']}",
            participate=f"participate-{pair['proposal_index']}",
            initial_mask=f"mask-{pair['proposal_index']}",
            gap_seed=700 + pair["proposal_index"],
        )
        prepared.append(item)
        return item

    calls = []

    def replay(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "batch_execution_format": protocol.INDEPENDENT_BATCH_AUDIT_FORMAT,
            "batch_size": 2,
            "no_gate": True,
            "production_kernel_called": False,
            "tables": ("table-0", "table-1"),
            "masks": ("final-mask-0", "final-mask-1"),
            "diagnostics": ({"index": 0}, {"index": 1}),
        }

    def finish(_context, item, table, mask, diagnostics):
        return (item.pair["proposal_index"], table, mask, diagnostics["index"])

    monkeypatch.setattr(arithmetic, "_prepare_generated_audit", prepare)
    monkeypatch.setattr(arithmetic, "_finish_generated_audit", finish)
    monkeypatch.setattr(
        arithmetic,
        "_independent_cuda_module",
        lambda: SimpleNamespace(replay_gap_l1_batched_cuda=replay),
    )
    context = SimpleNamespace(
        mode="smoke",
        current="current",
        schema="schema",
        queries="queries",
        target="target",
        q="counts",
    )
    rows = arithmetic._audit_generated_state_batched(
        context,
        ({"proposal_index": 0}, {"proposal_index": 1}),
        0.25,
    )

    assert len(calls) == 1
    assert calls[0][0][1] == ["donors-0", "donors-1"]
    assert calls[0][1]["seeds"] == [700, 701]
    assert [row[0] for row in rows] == [0, 1]
    assert [row[3] for row in rows] == [0, 1]
    assert [item.pair["proposal_index"] for item in prepared] == [0, 1]


def test_structural_auditor_replays_one_state_as_one_batch(monkeypatch):
    monkeypatch.setattr(structural, "protocol", protocol)

    def prepare(_context, pair, **_kwargs):
        index = pair["proposal_index"]
        return SimpleNamespace(
            replay=SimpleNamespace(
                donors=f"donors-{index}",
                common_update={
                    "participate": f"participate-{index}",
                    "copy_masks": f"mask-{index}",
                },
            ),
            gap_seed=900 + index,
            gap_rng=np.random.default_rng(900 + index),
        )

    calls = []

    def batch(*args, **kwargs):
        calls.append((args, kwargs))
        return tuple(
            (f"table-{index}", f"final-mask-{index}", {"index": index})
            for index in range(2)
        )

    def audit(_context, pair, _scale, **kwargs):
        return {
            "proposal_index": pair["proposal_index"],
            "batch_index": kwargs["gap_result"][2]["index"],
        }

    monkeypatch.setattr(structural, "_prepare_structure_pair", prepare)
    monkeypatch.setattr(structural, "_audit_generated_pair", audit)
    monkeypatch.setattr(
        structural, "evolve_step_gap_l1_global_batched", batch
    )
    context = SimpleNamespace(
        current="current",
        query_counts="counts",
        runtime=SimpleNamespace(
            schema="schema",
            queries="queries",
            runtime_target="target",
        ),
    )
    rows = structural._audit_generated_state_batched(
        context,
        [{"proposal_index": 0}, {"proposal_index": 1}],
        0.25,
        mode="smoke",
        factor_compiled="factor-compiled",
        gap_compiled="gap-compiled",
    )

    assert len(calls) == 1
    assert calls[0][0][1] == ["donors-0", "donors-1"]
    assert calls[0][1]["compiled_workload"] == "gap-compiled"
    assert rows == [
        {"proposal_index": 0, "batch_index": 0},
        {"proposal_index": 1, "batch_index": 1},
    ]


def test_collection_validator_requires_frozen_batch_identity(monkeypatch):
    monkeypatch.setattr(collector, "protocol", protocol)
    monkeypatch.setattr(
        protocol, "expected_pair_ids", lambda _mode: ("pair-0", "pair-1")
    )
    pairs = []
    for index in range(2):
        ordinary = {"retained_unconditionally": True}
        gap = {
            "retained_unconditionally": True,
            "kernel_diagnostics": {
                "no_gate": True,
                "gibbs_microsteps": 8,
                "active_switches_k": 1,
                "backend": protocol.PRODUCTION_BATCH_BACKEND,
                "batch_execution": {
                    "format": protocol.PRODUCTION_BATCH_EXECUTION_FORMAT,
                    "batch_size": 2,
                    "batch_index": index,
                    "strict_internal_order_preserved": True,
                },
            },
        }
        pairs.append({
            "pair_id": f"pair-{index}",
            "proposal_index": index,
            "address_status": "generated_unconditionally",
            "retained_unconditionally": True,
            "shared_replay": {
                "new_kernel_device": protocol.PRODUCTION_BATCH_BACKEND
            },
            "arms": {
                protocol.ARM_INDEPENDENT: ordinary,
                protocol.ARM_FACTOR: ordinary,
                protocol.ARM_GAP_L1: gap,
            },
        })
    value = {
        "collection_format": collector.COLLECTION_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "pairs": pairs,
    }
    value["collection_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(value)
    )
    collector.validate_collection(value, mode="smoke")

    pairs[1]["arms"][protocol.ARM_GAP_L1]["kernel_diagnostics"][
        "batch_execution"
    ]["batch_index"] = 0
    value["collection_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(value)
    )
    with pytest.raises(RuntimeError, match="批次身份"):
        collector.validate_collection(value, mode="smoke")


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="CUDA（显卡运行时）不可用",
)
def test_independent_wiring_accepts_real_production_batch_trace(monkeypatch):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        schema = Schema([
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("a", "b")
        ])
        queries = [
            {
                "conditions": [
                    {"attribute": "a", "operator": "==", "value": 1}
                ]
            },
            {
                "conditions": [
                    {"attribute": "b", "operator": "==", "value": 1}
                ]
            },
            {
                "conditions": [
                    {"attribute": "a", "operator": "==", "value": 1},
                    {"attribute": "b", "operator": "==", "value": 1},
                ]
            },
        ]
        current = pd.DataFrame({
            "a": [0, 0, 1, 1],
            "b": [0, 1, 0, 1],
        })
        donors = (
            pd.DataFrame({"a": [1, 1, 0, 0], "b": [1, 0, 1, 0]}),
            current.iloc[[1, 2, 3, 0]].reset_index(drop=True),
        )
        participates = (
            np.array([True, True, False, True]),
            np.array([True, False, True, True]),
        )
        masks = []
        for index in range(2):
            active = participates[index][:, None] & (
                current.to_numpy() != donors[index].to_numpy()
            )
            masks.append(active & (
                np.random.default_rng(40 + index).random(active.shape) < 0.5
            ))
        counts = evaluate_table(current, queries)
        target = np.array([2.0, 2.0, 1.0])
        seeds = (1201, 1202)
        production_results = production.evolve_step_gap_l1_global_batched(
            current,
            donors,
            schema,
            queries,
            target,
            counts,
            participates=participates,
            initial_masks=masks,
            reference_scale=0.1,
            rngs=[np.random.default_rng(seed) for seed in seeds],
            n_sweeps=8,
            device="cuda",
        )
        independent_result = independent_cuda.replay_gap_l1_batched_cuda(
            current,
            donors,
            schema,
            queries,
            target,
            counts,
            participates,
            masks,
            reference_scale=0.1,
            seeds=seeds,
            n_sweeps=8,
            eta=0.5,
            strength=2.0,
            floor=8.0,
            logit_clip=30.0,
        )
    finally:
        torch.use_deterministic_algorithms(previous)

    monkeypatch.setattr(arithmetic, "protocol", protocol)
    context = arithmetic._SourceContext(
        dataset="nltcs",
        seed=9906,
        group="initial",
        mode="smoke",
        current=current,
        schema=schema,
        queries=queries,
        target=target,
        source_target=target.astype(np.int64),
        runtime_n=len(current),
        source_n=len(current),
        q=np.asarray(counts, dtype=np.int64),
        residual=np.zeros(len(queries)),
        fitness=np.zeros(len(current)),
        probabilities=None,
        device="cuda",
        trajectory={},
        source_proposal_state={},
    )
    for index, production_result in enumerate(production_results):
        independent_diagnostics = independent_result["diagnostics"][index]
        pair = {
            "pair_id": f"pair-{index}",
            "proposal_index": index,
            "shared_replay": {
                "new_kernel_device": protocol.PRODUCTION_BATCH_BACKEND
            },
            "arms": {
                protocol.ARM_GAP_L1: {
                    "kernel_diagnostics": production_result[2],
                    "gibbs_rng": {
                        "address_uint64": seeds[index],
                        "initial_state_sha256": independent_diagnostics[
                            "initial_rng_sha256"
                        ],
                        "endpoint_state_sha256": independent_diagnostics[
                            "endpoint_rng_sha256"
                        ],
                    },
                }
            },
        }
        final_counts = np.asarray(
            production_result[2]["final_query_counts"], dtype=np.int64
        )
        prepared = arithmetic._PreparedGeneratedAudit(
            pair=pair,
            donor_indices=np.arange(len(current), dtype=np.int64),
            donors=donors[index],
            participate=participates[index],
            initial_mask=masks[index],
            arm_tables={
                arm: (production_result[0], production_result[0])
                for arm in protocol.ARMS
            },
            arm_counts={
                arm: (final_counts, final_counts) for arm in protocol.ARMS
            },
            gap_seed=seeds[index],
        )
        metric = arithmetic._finish_generated_audit(
            context,
            prepared,
            independent_result["tables"][index],
            independent_result["masks"][index],
            independent_diagnostics,
        )
        assert metric["pair_id"] == f"pair-{index}"
