import copy
import json

import pandas as pd
import pytest

from scripts import audit_issue52_stage_a_state_library as auditor
from scripts import audit_issue52_stage_t as stage_t_auditor
from scripts import build_issue52_stage_a_state_library as builder
from scripts import issue52_protocol as protocol
from scripts import probe_factorized_gibbs_mixing as probe
from scripts import run_issue52_stage_t as runner


def _write_payload(path, payload):
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def smoke_library(tmp_path_factory):
    root = tmp_path_factory.mktemp("issue52-stage-a-state-library")
    report_path, report = runner.run_stage_t(
        "smoke", root / "stage_t", max_workers=2
    )
    stage_t_audit_path, _ = stage_t_auditor.audit_stage_t(
        report_path, root / "stage_t_audit.json"
    )
    library_path, library = builder.build_state_library(
        report_path,
        stage_t_audit_path,
        root / "state_library.json",
    )
    audit_path, audit = auditor.audit_state_library(
        report_path,
        stage_t_audit_path,
        library_path,
        root / "state_library_audit.json",
    )
    return {
        "root": root,
        "report_path": report_path,
        "report": report,
        "stage_t_audit_path": stage_t_audit_path,
        "library_path": library_path,
        "library": library,
        "audit_path": audit_path,
        "audit": audit,
    }


def test_formal_state_library_protocol_freezes_48_states_and_source_hashes():
    formal = protocol.stage_a_state_library_protocol("formal")
    smoke = protocol.stage_a_state_library_protocol("smoke")

    assert formal["state_library_seeds"] == [200, 201, 202]
    assert formal["source_temperatures"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert formal["snapshot_rounds"] == [0, 1000, 2000, 3000]
    assert formal["expected_raw_snapshot_count"] == 60
    assert formal["expected_unique_state_count"] == 48
    assert formal["expected_source_sha256"] == {
        "report": (
            "abf7f6e25d86d518ea5255d0c1414e5f"
            "b7b606f9c14876ddfc3c952499dcc665"
        ),
        "audit": (
            "70253a5e6d115bea1bc37463499702ef3"
            "a1e421cbcbfc869d7cd6504408e9705"
        ),
    }
    assert smoke["expected_raw_snapshot_count"] == 20
    assert smoke["expected_unique_state_count"] == 16
    assert smoke["expected_source_sha256"] is None


def test_smoke_library_materializes_fixed_order_and_round0_dedup(
    smoke_library,
):
    library = smoke_library["library"]

    assert library["formal_result_valid"] is False
    assert library["manifest"]["raw_source_snapshot_count"] == 20
    assert library["manifest"]["deduplicated_state_count"] == 16
    assert library["manifest"]["round0_deduplicated_count"] == 1
    assert library["manifest"]["state_count_by_round"] == {
        "0": 1,
        "4": 5,
        "8": 5,
        "12": 5,
    }
    assert all(library["gates"].values())
    assert [state["state_id"] for state in library["states"]] == [
        "seed_9903_initial_round_0",
        *[
            f"seed_9903_round_{state_round}_source_tau_{temperature}"
            for state_round in (4, 8, 12)
            for temperature in (1, 2, 3, 4, 5)
        ],
    ]
    initial = library["states"][0]
    assert initial["source_temperature"] is None
    assert initial["shared_source_temperatures"] == [
        1.0, 2.0, 3.0, 4.0, 5.0
    ]
    assert initial["snapshot"]["source_temperature"] == 1.0
    assert library["seed_rows"][0]["gates"][
        "initial_snapshots_equal_except_source_temperature"
    ] is True


def test_every_materialized_snapshot_is_probe_readable(smoke_library):
    target, queries, schema, _, _ = builder.common._load_inputs()
    for state in smoke_library["library"]["states"]:
        restored, controls = probe._restore_current_snapshot(
            state["snapshot"],
            target,
            queries,
            schema,
            device="numpy",
        )
        assert probe._frame_sha256(restored) == state["snapshot"][
            "state_sha256"
        ]
        assert controls["source_seed"] == state["seed"]
        assert controls["state_round"] == state["state_round"]


def test_independent_audit_passes_and_both_outputs_refuse_overwrite(
    smoke_library,
):
    audit = smoke_library["audit"]
    library = smoke_library["library"]

    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert audit["state_library_scientific_sha256"] == library[
        "state_library_scientific_sha256"
    ]
    assert audit["recomputed"]["manifest"] == library["manifest"]

    with pytest.raises(FileExistsError, match="状态库输出已存在"):
        builder.build_state_library(
            smoke_library["report_path"],
            smoke_library["stage_t_audit_path"],
            smoke_library["library_path"],
        )
    with pytest.raises(FileExistsError, match="审计输出已存在"):
        auditor.audit_state_library(
            smoke_library["report_path"],
            smoke_library["stage_t_audit_path"],
            smoke_library["library_path"],
            smoke_library["audit_path"],
        )


def test_builder_rejects_report_not_bound_to_stage_t_audit(smoke_library):
    tampered = copy.deepcopy(smoke_library["report"])
    tampered["stage_t"]["aggregates"]["trajectory_count"] += 1
    path = _write_payload(
        smoke_library["root"] / "tampered_source_report.json", tampered
    )

    with pytest.raises(RuntimeError, match="来源产物绑定失败"):
        builder.build_state_library(
            path,
            smoke_library["stage_t_audit_path"],
            smoke_library["root"] / "rejected_library.json",
        )


def test_audit_rejects_strong_snapshot_tamper_with_updated_table_hash(
    smoke_library,
):
    tampered = copy.deepcopy(smoke_library["library"])
    snapshot = tampered["states"][1]["snapshot"]
    template = dict(snapshot["table_records"][0])
    for record in snapshot["table_records"]:
        for column, value in template.items():
            record[column] = value
    frame = pd.DataFrame(
        snapshot["table_records"], columns=snapshot["table_columns"]
    )
    snapshot["state_sha256"] = auditor.trajectory._frame_sha256(frame)
    path = _write_payload(
        smoke_library["root"] / "tampered_state_library.json", tampered
    )

    with pytest.raises(RuntimeError, match="查询 loss 或来源身份不一致"):
        auditor.audit_state_library(
            smoke_library["report_path"],
            smoke_library["stage_t_audit_path"],
            path,
            smoke_library["root"] / "tampered_snapshot_audit.json",
        )


def test_audit_rejects_manifest_tamper_even_with_updated_scientific_hash(
    smoke_library,
):
    tampered = copy.deepcopy(smoke_library["library"])
    tampered["manifest"]["deduplicated_state_count"] += 1
    tampered["state_library_scientific_sha256"] = (
        auditor.common._canonical_sha256(auditor._scientific_payload(tampered))
    )
    path = _write_payload(
        smoke_library["root"] / "tampered_manifest_library.json", tampered
    )

    with pytest.raises(RuntimeError, match="manifest"):
        auditor.audit_state_library(
            smoke_library["report_path"],
            smoke_library["stage_t_audit_path"],
            path,
            smoke_library["root"] / "tampered_manifest_audit.json",
        )
