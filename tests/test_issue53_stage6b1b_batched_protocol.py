"""第 6B-1B 阶段显卡批量执行差量协议测试。"""

from pathlib import Path

import pytest

from scripts import issue53_stage6b1b_batched_protocol as protocol
from scripts import issue53_stage6b1b_independent_cuda as independent
from scripts import issue53_stage6b1b_protocol as base_protocol
from table_diffevo import gap_l1_diffusion as production


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_batch_delta_preserves_old_protocol_and_freezes_partition():
    manifest = protocol.frozen_protocol_manifest()
    assert base_protocol.FROZEN_PROTOCOL_SHA256 == (
        "754fbf02fdaeacb716fae0be90ee1b3d8fd27f1b8dd0c8faa1c73a11ddc63387"
    )
    assert base_protocol.frozen_protocol_manifest()["gpu_delta"][
        "formal_address_batch_size"
    ] == 1
    inherited = manifest["inherits_frozen_single_address_protocol"]
    assert inherited["protocol_sha256"] == (
        base_protocol.FROZEN_PROTOCOL_SHA256
    )
    assert inherited["method_parameters_changed"] is False
    assert inherited[
        "queries_addresses_arms_metrics_or_thresholds_changed"
    ] is False
    assert inherited["no_gate_contract"] == base_protocol.NO_GATE_CONTRACT

    partition = manifest["fixed_batch_partition"]
    assert partition["formal_address_batch_size"] == 20
    assert partition["formal_state_batch_count"] == 25
    assert partition["smoke_address_batch_size"] == 2
    assert partition["smoke_state_batch_count"] == 5
    assert partition["address_order"] == "proposal_index_ascending"
    assert partition["cross_state_batching"] is False
    assert partition["sort_or_group_by_k_donor_or_result"] is False
    assert partition["adaptive_batch_resize"] is False
    assert partition["automatic_single_address_fallback"] is False
    assert partition["automatic_cpu_fallback"] is False


def test_batch_execution_and_independent_audit_formats_are_bound():
    manifest = protocol.frozen_protocol_manifest()
    implementation = manifest["batch_execution_implementation"]
    assert production.BATCH_EXECUTION_FORMAT == (
        protocol.PRODUCTION_BATCH_EXECUTION_FORMAT
    )
    assert independent.INDEPENDENT_BATCH_AUDIT_FORMAT == (
        protocol.INDEPENDENT_BATCH_AUDIT_FORMAT
    )
    assert production.TRACE_FORMAT == protocol.MICROSTEP_TRACE_FORMAT
    assert implementation["production_backend"] == (
        "torch_cuda_float64_batched"
    )
    assert implementation["independent_audit_backend"] == (
        "independent_torch_cuda_float64_batched"
    )
    assert implementation[
        "production_imports_independent_batch_arithmetic"
    ] is False
    assert implementation[
        "independent_imports_production_batch_arithmetic"
    ] is False
    assert implementation["artificial_prototype_is_formal_entry"] is False


def test_batch_equivalence_is_exact_without_reusing_single_trace_identity():
    manifest = protocol.frozen_protocol_manifest()
    equivalence = manifest["execution_equivalence"]
    assert equivalence["legacy_single_trace_sha256_must_match"] is False
    assert equivalence["legacy_single_float_absolute_tolerance"] == 1e-12
    assert equivalence["legacy_single_sampling_flip_tolerance"] == 0
    assert equivalence["legacy_single_discrete_outputs_exact"] is True
    assert equivalence[
        "production_vs_independent_batch_float_tolerance"
    ] == 0.0
    assert equivalence["production_vs_independent_batch_exact_fields"] == [
        "valid_step_mask",
        "coordinates",
        "random_rolls",
        "initial_rng_sha256",
        "endpoint_rng_sha256",
        "seven_float64_microstep_values",
        "three_boolean_microstep_values",
        "per_address_trace_sha256",
        "final_mask",
        "final_query_counts",
        "final_table",
    ]
    assert equivalence["nonfinite_value_tolerance"] == 0
    assert equivalence["exact_zero_or_one_probability_tolerance"] == 0
    assert equivalence["full_recount_mismatch_tolerance"] == 0

    no_gate = manifest["no_gate_execution"]
    assert no_gate == {
        "no_gate": True,
        "acceptance_or_rejection": False,
        "candidate_rollback": False,
        "winner_selection": False,
        "result_dependent_address_filtering": False,
        "freeze_address_or_batch_inside_noise_range": False,
    }


def test_batch_protocol_manifest_is_frozen_and_old_source_fails_closed():
    assert protocol.file_sha256(
        REPOSITORY_ROOT / protocol.PROTOCOL_DOC
    ) == protocol.PROTOCOL_DOC_SHA256
    drifted_sources = [
        name
        for name, binding in protocol.BATCH_EXECUTION_SOURCES.items()
        if protocol.file_sha256(REPOSITORY_ROOT / binding["path"])
        != binding["sha256"]
    ]
    assert drifted_sources == ["production_batched_gap_kernel"]
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    try:
        with pytest.raises(
            RuntimeError,
            match="批量执行来源漂移：production_batched_gap_kernel",
        ):
            protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"冻结产物不在本机（gitignored），产物持有机复核：{exc}")


def test_batch_protocol_uses_new_output_dirs_and_cannot_resume_old_run():
    isolation = protocol.frozen_protocol_manifest()["output_isolation"]
    assert isolation["formal_output_dir"] == str(protocol.OUTPUT_DIR)
    assert isolation["smoke_output_dir"] == str(protocol.SMOKE_OUTPUT_DIR)
    assert isolation["legacy_interrupted_output_dir"] == str(
        base_protocol.OUTPUT_DIR
    )
    assert protocol.OUTPUT_DIR != base_protocol.OUTPUT_DIR
    assert protocol.SMOKE_OUTPUT_DIR != base_protocol.SMOKE_OUTPUT_DIR
    assert isolation["resume_or_copy_legacy_interrupted_output"] is False
    assert isolation["fresh_calibration_required"] is True


@pytest.mark.parametrize(
    ("mode", "batch_size", "batch_count", "pair_count"),
    [
        ("formal", 20, 25, 500),
        ("smoke", 2, 5, 10),
    ],
)
def test_batch_plan_is_read_only(mode, batch_size, batch_count, pair_count):
    plan = protocol.build_plan(mode)
    assert plan["mode"] == "plan_only_no_source_read_no_generation"
    assert plan["protocol_sha256"] == protocol.FROZEN_PROTOCOL_SHA256
    assert plan["gap_l1_address_batch_size"] == batch_size
    assert plan["gap_l1_state_batch_count"] == batch_count
    assert plan["pair_count"] == pair_count
    assert plan["pipeline_wiring_available"] is True
    assert plan["formal_pipeline_wired_at_protocol_freeze"] is False
    assert plan["source_read_started"] is False
    assert plan["generation_started"] is False
    assert plan["confirmation_consumed"] is False


def test_batch_protocol_confirmation_is_separate_from_freeze():
    with pytest.raises(PermissionError, match="单独授权"):
        protocol.require_run_confirmation("smoke", None)
    with pytest.raises(PermissionError, match="单独授权"):
        protocol.require_run_confirmation("formal", "wrong")
    protocol.require_run_confirmation(
        "smoke", protocol.FROZEN_PROTOCOL_SHA256
    )


def test_batch_protocol_is_selected_only_through_shared_loader():
    needle = "issue53_stage6b1b_batched_protocol"
    production_paths = (
        "src/table_diffevo/gap_l1_diffusion.py",
        "scripts/issue53_stage6b1b_protocol.py",
        "scripts/calibrate_issue53_stage6b1_gap_l1.py",
        "scripts/collect_issue53_stage6b1_screen.py",
        "scripts/audit_issue53_stage6b1_structure.py",
        "scripts/evaluate_issue53_stage6b1_screen.py",
        "scripts/audit_issue53_stage6b1_arithmetic.py",
    )
    for relative in production_paths:
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert needle not in text
    loader = (
        REPOSITORY_ROOT / "scripts/issue53_stage6b1_protocol_loader.py"
    ).read_text(encoding="utf-8")
    assert '"stage6b1b_batched": "issue53_stage6b1b_batched_protocol"' in loader
