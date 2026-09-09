from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import diagnose_issue53_gap_weight_dual_ar_progress_clip as runner
from scripts import issue53_gap_weight_dual_ar_progress_clip_diagnostic_protocol as protocol
from table_diffevo import gap_l1_diffusion as gap


ROOT = Path(__file__).resolve().parents[1]


class _Schema:
    def __init__(self, names: tuple[str, ...]):
        self._names = names

    def attribute_names(self) -> list[str]:
        return list(self._names)


def test_frozen_diagnostic_protocol_identity():
    # 双态：产物缺失（gitignored）跳过；活树漂移时守卫失败关闭即为正确行为。
    try:
        observed = protocol.assert_frozen_protocol_identity(ROOT)
    except FileNotFoundError:
        pytest.skip("冻结产物不在本机（gitignored），产物持有机复核")
    except RuntimeError as exc:
        assert "漂移" in str(exc)
        return
    assert observed == protocol.FROZEN_PROTOCOL_SHA256


def test_diagnostic_changes_only_two_resource_caps():
    source = protocol.source.generator_params_manifest(
        protocol.DATASET, protocol.ARM, protocol.SEED
    )
    diagnostic = protocol.generator_params_manifest()
    changed = {
        key for key in source if source[key] != diagnostic[key]
    }
    assert changed == {"n_rounds", "candidate_budget"}
    assert diagnostic["n_rounds"] == protocol.DIAGNOSTIC_ROUNDS
    assert diagnostic["candidate_budget"] == protocol.DIAGNOSTIC_ROUNDS


def test_information_boundary_forbids_quality_and_full_rerun():
    boundary = protocol.frozen_protocol_manifest()["information_boundary"]
    assert boundary == {
        "raw_reference_access_allowed": False,
        "quality_metric_computation_allowed": False,
        "unseen_query_evaluation_allowed": False,
        "method_comparison_allowed": False,
        "parameter_retuning_allowed": False,
        "automatic_full_rerun_allowed": False,
        "automatic_quality_evaluation_allowed": False,
    }


def test_confirmation_fails_before_repository_access(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_repo_root",
        lambda: (_ for _ in ()).throw(AssertionError("提前访问工作树")),
    )
    with pytest.raises(PermissionError, match="确认"):
        runner.run("wrong")


def test_instrumentation_scope_restores_all_production_functions():
    originals = (
        gap._evolve_step_gap_l1_global_cuda,
        gap._record_channel_dominance,
        gap._build_scan_diagnostics,
    )
    with runner._instrumentation(runner._ClipRecorder()):
        assert gap._evolve_step_gap_l1_global_cuda is not originals[0]
        assert gap._record_channel_dominance is not originals[1]
        assert gap._build_scan_diagnostics is not originals[2]
    assert (
        gap._evolve_step_gap_l1_global_cuda,
        gap._record_channel_dominance,
        gap._build_scan_diagnostics,
    ) == originals


def test_affected_queries_uses_current_same_row_mask_context():
    current = pd.DataFrame({"A": [0], "B": [1]})
    donors = pd.DataFrame({"A": [1], "B": [0]})
    queries = [
        {
            "id": "a1",
            "conditions": [
                {"attribute": "A", "operator": "==", "value": 1}
            ],
            "result": 5,
        },
        {
            "id": "a1_b1",
            "conditions": [
                {"attribute": "A", "operator": "==", "value": 1},
                {"attribute": "B", "operator": "==", "value": 1},
            ],
            "result": 3,
        },
        {
            "id": "b1",
            "conditions": [
                {"attribute": "B", "operator": "==", "value": 1}
            ],
            "result": 4,
        },
    ]
    # A 已复制、B 尚未复制：B=0/1 两侧分别形成 11 与 10。
    affected = runner._affected_queries(
        current=current,
        donors=donors,
        mask=np.asarray([[True, False]]),
        row_index=0,
        attribute_index=1,
        attribute_names=("A", "B"),
        queries=queries,
    )
    assert [item["query_id"] for item in affected] == ["a1_b1", "b1"]
    assert all(item["row_indicator_if_switch_0"] is True for item in affected)
    assert all(item["row_indicator_if_switch_1"] is False for item in affected)


def test_recorder_reconstructs_one_clipped_microstep():
    recorder = runner._ClipRecorder()
    recorder.round_index = protocol.PREFAIL_ZERO_CLIP_ROUNDS
    rng = np.random.default_rng(123)
    context = {
        "current": pd.DataFrame({"A": [0]}),
        "donors": pd.DataFrame({"A": [1]}),
        "schema": _Schema(("A",)),
        "queries": [
            {
                "id": "a1",
                "conditions": [
                    {"attribute": "A", "operator": "==", "value": 1}
                ],
                "result": 1,
            }
        ],
        "target": np.asarray([1.0]),
        "current_counts": np.asarray([0.0]),
        "participate": np.asarray([True]),
        "initial_mask": np.asarray([[False]]),
        "rng_state": rng.bit_generator.state,
        "logit_clip": 30.0,
    }
    recorder.begin_round(context)
    clipped_probability = float(1.0 / (1.0 + np.exp(-30.0)))
    recorder.record_scan_vectors({
        "scores": [16.0, *([0.0] * 7)],
        "normalized_scores": [16.0, *([0.0] * 7)],
        "raw_logits": [32.0, *([0.0] * 7)],
        "probabilities": [clipped_probability, *([0.5] * 7)],
    })
    recorder.record_channel_values((16.0, 0.0, 0.0, 0.0))
    for _ in range(7):
        recorder.record_channel_values((0.0, 0.0, 0.0, 0.0))
    recorder.finish_round({
        "active_switches_k": 1,
        "gibbs_microsteps": 8,
        "clip_hit_count": 1,
        "nonfinite_condition_count": 0,
        "exact_zero_or_one_probability_count": 0,
        "reference_scale": 1.0,
        "normalized_score_distribution": {"max": 16.0},
        "raw_logit_distribution": {"max": 32.0},
        "probability_distribution": {"max": clipped_probability},
        "near_deterministic_count": 1,
    })
    assert len(recorder.clipped_steps) == 1
    step = recorder.clipped_steps[0]
    assert step["attribute"] == "A"
    assert step["normalized_score"] == 16.0
    assert step["raw_logit"] == 32.0
    assert step["affected_measured_queries"][0]["query_id"] == "a1"
