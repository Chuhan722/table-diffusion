from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import issue53_gap_weight_r8_screen_protocol as protocol
from scripts import issue53_stage6e_autostop_protocol as stage6e
from table_diffevo import gap_l1_diffusion as gap


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_document_sources_and_manifest_are_frozen():
    # 本筛查线已收束（rare_query_protection_not_recovered）；协议冻结的是
    # 历史实验身份，活实现源码此后继续演进属预期。这里只校验不随活树漂移
    # 的死记录自恰，并断言身份守卫在演进后的树上正确失败关闭。
    assert protocol.file_sha256(REPOSITORY_ROOT / protocol.PROTOCOL_DOC) == (
        protocol.PROTOCOL_DOC_SHA256
    )
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    with pytest.raises(RuntimeError, match="实现源码漂移"):
        protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)


def test_protocol_source_guard_fails_closed(monkeypatch):
    # 活树上多个源已漂移；为单独验证守卫对每个绑定的敏感性，伪造其余
    # 文件全部咬合，只让 gap_kernel 漂移。
    gap_path = (
        REPOSITORY_ROOT
        / protocol.IMPLEMENTATION_SOURCES["gap_kernel"]["path"]
    ).resolve()
    frozen = {
        (REPOSITORY_ROOT / binding["path"]).resolve(): binding["sha256"]
        for binding in protocol.IMPLEMENTATION_SOURCES.values()
    }
    frozen[
        (REPOSITORY_ROOT / protocol.PROTOCOL_DOC).resolve()
    ] = protocol.PROTOCOL_DOC_SHA256

    def drift_one(path):
        resolved = Path(path).resolve()
        if resolved == gap_path:
            return "0" * 64
        return frozen[resolved]

    monkeypatch.setattr(protocol, "file_sha256", drift_one)
    with pytest.raises(RuntimeError, match="实现源码漂移：gap_kernel"):
        protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)


def test_plan_is_exactly_four_tasks_and_read_only(monkeypatch):
    # 收束线：绕过对活树的源码身份校验，只测 plan 本身的只读性与内容。
    monkeypatch.setattr(
        protocol,
        "assert_frozen_protocol_identity",
        lambda _root: protocol.FROZEN_PROTOCOL_SHA256,
    )
    opened: list[Path] = []
    original_open = Path.open

    def record_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    plan = protocol.build_plan(REPOSITORY_ROOT)

    assert plan == {
        "mode": "plan_only_no_dataset_reference_or_generation_access",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "task_ids": [
            "seed_9908__gap_legacy_s8__test_300x10",
            "seed_9908__gap_bounded_r8_s8__test_300x10",
            "seed_9908__gap_legacy_s8__nltcs",
            "seed_9908__gap_bounded_r8_s8__nltcs",
        ],
        "trajectory_count": 4,
        "seed": 9908,
        "reserved_confirmation_seeds": [358, 359, 360, 361, 362],
        "round_cap": 6000,
        "candidate_budget": 6000,
        "patience_ticks": 6,
        "checkpoint_rounds": [0, 100, 250, 500, 1000, 2000, 4000, 6000],
        "max_workers": 2,
        "output_dir": (
            "outputs/issue53_gap_weight_r8_paired_screen_seed9908_v1"
        ),
        "generator_params_manifest_sha256": (
            "7e99f7a63fcaae34d359d2cce697be10ad5c4675363dbaf9276a2be6bd5e5630"
        ),
        "runner_wired": False,
        "generation_started": False,
        "screen_generation_authorized": False,
    }
    forbidden = {
        (REPOSITORY_ROOT / value).resolve()
        for spec in protocol.DATASETS.values()
        for key, value in spec.items()
        if key in {"schema", "queries", "marginals", "reference"}
    }
    forbidden.add((REPOSITORY_ROOT / protocol.TEST_IDENTITY_ARTIFACT).resolve())
    assert not (set(opened) & forbidden)
    # 正式运行已完成，OUTPUT_DIR 在位属预期；plan 只读性已由 opened 记录守护。


def test_common_flow_is_stage6e_and_only_two_weight_fields_differ():
    assert protocol.common_generator_params() == stage6e.common_generator_params()
    for dataset in protocol.DATASET_ORDER:
        legacy = protocol.task_generator_params(dataset, "gap_legacy_s8")
        bounded = protocol.task_generator_params(
            dataset, "gap_bounded_r8_s8"
        )
        assert legacy.keys() == bounded.keys()
        assert {
            key for key in legacy if legacy[key] != bounded[key]
        } == {"gap_l1_weighting", "gap_l1_max_weight_ratio"}
        assert legacy["gap_l1_sweeps"] == bounded["gap_l1_sweeps"] == 8
        assert legacy["factorized_gibbs_sweeps"] == 0
        assert bounded["factorized_gibbs_sweeps"] == 0
        assert legacy["gap_l1_weighting"] == gap.DEFAULT_GAP_L1_WEIGHTING
        assert legacy["gap_l1_max_weight_ratio"] is None
        assert bounded["gap_l1_weighting"] == (
            gap.GAP_L1_WEIGHTING_BOUNDED_RELATIVE
        )
        assert bounded["gap_l1_max_weight_ratio"] == 8.0
        assert legacy["seed"] == bounded["seed"] == 9908


def test_generator_manifests_are_strict_and_predeclared():
    manifests = protocol.generator_params_manifest_matrix()

    assert list(manifests) == [
        task.task_id for task in protocol.task_plan().tasks
    ]
    assert protocol.generator_params_manifest_sha256() == (
        "7e99f7a63fcaae34d359d2cce697be10ad5c4675363dbaf9276a2be6bd5e5630"
    )
    for manifest in manifests.values():
        assert manifest["tol"] == "positive_infinity"
        json.dumps(manifest, allow_nan=False)


def test_bounded_smoothing_uses_float64_division_without_rounding():
    values = protocol.frozen_protocol_manifest()["bounded_weighting"][
        "smoothing_by_dataset"
    ]

    assert values["test_300x10"] == 300 / 7
    assert values["nltcs"] == 16_181 / 7
    assert not values["test_300x10"].is_integer()
    assert not values["nltcs"].is_integer()
    assert (300 + values["test_300x10"]) / values["test_300x10"] == 8.0
    assert math.isclose(
        (16_181 + values["nltcs"]) / values["nltcs"],
        8.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def _query_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def test_target_count_bins_are_complete_disjoint_and_identity_bound():
    for dataset in protocol.DATASET_ORDER:
        rows = _query_rows(REPOSITORY_ROOT / protocol.DATASETS[dataset]["queries"])
        observed_indices: list[int] = []
        for bin_spec in protocol.TARGET_COUNT_BINS[dataset]:
            members = [
                {
                    "index": index,
                    "id": row.get("id"),
                    "target": int(row["result"]),
                }
                for index, row in enumerate(rows)
                if bin_spec["lower_inclusive"]
                <= int(row["result"])
                <= bin_spec["upper_inclusive"]
            ]
            assert len(members) == bin_spec["query_count"]
            assert protocol.canonical_sha256(members) == (
                bin_spec["membership_sha256"]
            )
            observed_indices.extend(item["index"] for item in members)
        assert sorted(observed_indices) == list(range(len(rows)))
        assert len(observed_indices) == len(set(observed_indices))


def test_generation_and_evaluation_input_files_keep_frozen_identities():
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
        for name, expected in spec["input_sha256"].items():
            assert protocol.file_sha256(REPOSITORY_ROOT / spec[name]) == expected
    assert protocol.file_sha256(
        REPOSITORY_ROOT / protocol.TEST_IDENTITY_ARTIFACT
    ) == protocol.TEST_IDENTITY_ARTIFACT_SHA256


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"execution_valid": False}, "execution_invalid"),
        (
            {"nltcs_measured_l1_ratio": 1.0},
            "bounded_weighting_mechanism_not_supported",
        ),
        (
            {"nltcs_common_bin_ratio": 1.0},
            "bounded_weighting_mechanism_not_supported",
        ),
        (
            {"nltcs_rare_bin_ratio": 1.2500001},
            "common_gain_with_rare_query_regression",
        ),
        (
            {"test_measured_l1_ratio": 1.0500001},
            "measured_gain_with_safety_risk",
        ),
        (
            {"one_way_ratio_by_dataset": {"test_300x10": 1.0, "nltcs": 1.06}},
            "measured_gain_with_safety_risk",
        ),
        ({}, "advance_to_five_seed_confirmation"),
    ],
)
def test_screen_classification_uses_frozen_precedence(changes, expected):
    values = {
        "execution_valid": True,
        "nltcs_measured_l1_ratio": 0.99,
        "nltcs_common_bin_ratio": 0.99,
        "nltcs_rare_bin_ratio": 1.25,
        "test_measured_l1_ratio": 1.05,
        "one_way_ratio_by_dataset": {
            "test_300x10": 1.05,
            "nltcs": 1.05,
        },
    }
    values.update(changes)
    assert protocol.classify_screen(**values) == expected


def test_execution_invalid_short_circuits_unavailable_quality_values():
    assert protocol.classify_screen(
        execution_valid=False,
        nltcs_measured_l1_ratio=float("nan"),
        nltcs_common_bin_ratio=float("nan"),
        nltcs_rare_bin_ratio=float("nan"),
        test_measured_l1_ratio=float("nan"),
        one_way_ratio_by_dataset={},
    ) == "execution_invalid"


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"execution_valid": 1}, "布尔"),
        ({"nltcs_measured_l1_ratio": float("nan")}, "非负数"),
        (
            {"one_way_ratio_by_dataset": {"nltcs": 1.0}},
            "恰好覆盖",
        ),
        ({"one_way_ratio_by_dataset": None}, "必须是映射"),
    ],
)
def test_screen_classification_rejects_invalid_inputs(changes, message):
    values = {
        "execution_valid": True,
        "nltcs_measured_l1_ratio": 0.99,
        "nltcs_common_bin_ratio": 0.99,
        "nltcs_rare_bin_ratio": 1.0,
        "test_measured_l1_ratio": 1.0,
        "one_way_ratio_by_dataset": {
            "test_300x10": 1.0,
            "nltcs": 1.0,
        },
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        protocol.classify_screen(**values)


def test_freeze_does_not_authorize_runner_or_generation():
    boundary = protocol.frozen_protocol_manifest()["authorization_boundary"]

    assert boundary["runner_wiring_authorized_at_freeze"] is False
    assert boundary["screen_generation_authorized_at_freeze"] is False
    assert boundary["matching_protocol_hash_alone_authorizes_generation"] is False
    assert boundary["passing_screen_authorizes_confirmation_run"] is False
    with pytest.raises(PermissionError, match="尚未接线"):
        protocol.require_run_confirmation(None)
    with pytest.raises(PermissionError, match="尚未接线"):
        protocol.require_run_confirmation(protocol.FROZEN_PROTOCOL_SHA256)


def test_plan_cli_emits_same_read_only_plan():
    # 收束线：CLI 不经 monkeypatch，身份守卫先于任何输出触发，
    # 预期在演进后的树上失败关闭且不输出计划。
    environment = {"PYTHONPATH": f"{REPOSITORY_ROOT}:{REPOSITORY_ROOT / 'src'}"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.issue53_gap_weight_r8_screen_protocol",
            "plan",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "实现源码漂移" in result.stderr
    assert result.stdout.strip() == ""
