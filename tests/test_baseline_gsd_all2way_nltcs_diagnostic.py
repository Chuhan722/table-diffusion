"""GSD 无噪声 all2way 对照局协议测试（不跑 GSD 子进程）。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import (
    run_baseline_gsd_all2way_nltcs_diagnostic as diag,
)
from scripts import (
    run_baseline_pgm_all2way_nltcs_diagnostic as all2way,
)
from scripts import run_baseline_pgm_nltcs_diagnostic as pgm980


def _repo_root() -> Path:
    return Path(diag.__file__).resolve().parents[1]


def test_protocol_identity_is_frozen():
    assert diag.protocol_sha256() == diag.FROZEN_PROTOCOL_SHA256
    assert diag.assert_frozen_protocol_identity() == (
        diag.FROZEN_PROTOCOL_SHA256
    )


def test_protocol_manifest_pins_same_meal_and_defaults():
    manifest = diag._protocol_manifest()
    generation = manifest["generation"]
    assert generation["noise_model"] == "none_rho_inf_exact_statistics"
    fed = generation["fed_statistics"]
    assert fed["statistic_count"] == 512
    assert fed["one_way_workloads"] == 16
    assert fed["two_way_workloads"] == 120
    params = generation["parameters"]
    assert params["seed"] == 9908
    assert params["num_generations_cap"] == 50_000_000
    assert params["stop_early_threshold"] == 0.0001
    assert params["stop_early_min_generation"] == 16181
    assert params["genetic_operators"] == [
        "mutate", "continuous", "cross", "swap",
    ]
    assert params["sparse_statistics"] is True
    assert params["tuning"] == "none_official_fit_defaults"
    assert generation["upstream_commit"] == (
        "f6150d7821b9675ce9158b456f1a5cff8bc3b3d6"
    )
    assert manifest["generation_inputs_sha256"]["measured_exam"] == (
        all2way.ALL2WAY_SHA256
    )
    assert manifest["fair_comparable_groups"] == [
        "heldout_3way",
        "heldout_4way",
        "heldout_combined",
        "one_way_safety",
    ]
    assert manifest["interpretation"] == (
        "diagnostic_only_lower_is_better_no_promotion_gate"
    )


def test_generation_script_sha_matches_disk():
    script = _repo_root() / diag.GENERATION_SCRIPT_PATH
    assert all2way._sha256_file(script) == (
        diag.GENERATION_SCRIPT_SHA256
    )


def test_environment_audit_passes_and_fails_closed(monkeypatch):
    root = _repo_root()
    audit = diag._audit_gsd_environment(root)
    assert audit["upstream_commit"] == diag.GSD_UPSTREAM_COMMIT
    assert audit["generation_script_sha256"] == (
        diag.GENERATION_SCRIPT_SHA256
    )
    monkeypatch.setattr(
        diag, "GENERATION_SCRIPT_SHA256", "0" * 64
    )
    with pytest.raises(RuntimeError, match="生成脚本 SHA 漂移"):
        diag._audit_gsd_environment(root)


def _good_manifest() -> dict:
    return {
        "statistics": {
            "zero_noise_max_abs_diff": 0.0,
            "statistic_count": 512,
        },
        "parameters": {
            "seed": 9908,
            "num_generations_cap": 50_000_000,
            "stop_early_threshold": 0.0001,
        },
        "runtime": {
            "wall_seconds_fit": 1.0,
            "jax_version": "0.4.18",
            "jax_devices": ["cuda:0"],
        },
        "sanity_family_fit_frequency_l1": {
            "mean": 0.0, "max": 0.0, "note": "sanity",
        },
        "input": {"sha256": pgm980.INPUT_SHA256["reference"]},
        "output": {"row_count": 16181, "sha256": "a" * 64},
    }


def test_manifest_audit_passes_and_fails_closed():
    audit = diag._audit_generation_manifest(_good_manifest())
    assert audit["zero_noise_max_abs_diff"] == 0.0
    assert audit["statistic_count"] == 512
    noisy = _good_manifest()
    noisy["statistics"]["zero_noise_max_abs_diff"] = 1e-9
    with pytest.raises(RuntimeError, match="零噪声"):
        diag._audit_generation_manifest(noisy)
    short = _good_manifest()
    short["output"]["row_count"] = 16180
    with pytest.raises(RuntimeError, match="行数"):
        diag._audit_generation_manifest(short)
    drifted = _good_manifest()
    drifted["statistics"]["statistic_count"] = 480
    with pytest.raises(RuntimeError, match="统计格数"):
        diag._audit_generation_manifest(drifted)
    wrong_seed = _good_manifest()
    wrong_seed["parameters"]["seed"] = 1
    with pytest.raises(RuntimeError, match="seed"):
        diag._audit_generation_manifest(wrong_seed)


def test_load_gsd_table_validates(tmp_path):
    _, names, _ = pgm980._schema_domain(_repo_root())
    good = pd.DataFrame(
        np.zeros((diag.N_RECORDS, len(names)), dtype=int),
        columns=names,
    )
    good_path = tmp_path / "good.csv"
    good.to_csv(good_path, index=False)
    table = diag._load_gsd_table(good_path, names)
    assert list(table.columns) == names
    bad_value = good.copy()
    bad_value.iloc[0, 0] = 2
    bad_path = tmp_path / "bad.csv"
    bad_value.to_csv(bad_path, index=False)
    with pytest.raises(RuntimeError, match="非 0/1"):
        diag._load_gsd_table(bad_path, names)
    short = good.iloc[:-1]
    short_path = tmp_path / "short.csv"
    short.to_csv(short_path, index=False)
    with pytest.raises(RuntimeError, match="行数漂移"):
        diag._load_gsd_table(short_path, names)
    renamed = good.rename(columns={names[0]: "not_a_column"})
    renamed_path = tmp_path / "renamed.csv"
    renamed.to_csv(renamed_path, index=False)
    with pytest.raises(RuntimeError, match="列集合"):
        diag._load_gsd_table(renamed_path, names)


def test_reference_extraction_pins_and_known_values():
    references = diag._load_references(_repo_root())
    engine = references["engine_all2way_pool"]
    assert engine["sha256"] == diag.ENGINE_REPORT_SHA256
    assert engine["family_480_direct"]["mean"] == pytest.approx(
        0.00013493191603320767
    )
    assert engine["snapshot"]["heldout_combined"]["mean"] == (
        pytest.approx(0.0018944020958222606)
    )
    pgm = references["pgm_all2way"]
    assert pgm["sha256"] == diag.PGM_ALL2WAY_REPORT_SHA256
    assert pgm["family_480_direct"]["mean"] == pytest.approx(
        0.000357, abs=5e-6
    )
    assert pgm["snapshot"]["heldout_combined"]["mean"] == (
        pytest.approx(0.001431, abs=5e-6)
    )


def test_comparison_shapes_and_deltas():
    references = diag._load_references(_repo_root())

    def _node(mean):
        return {
            "normalized_l1_mean": mean,
            "normalized_l1_median": mean,
            "normalized_l1_p90": mean,
            "normalized_l1_max": mean,
        }

    fake_quality = {
        "measured": _node(0.00001),
        "heldout": {
            "3way": _node(0.0024),
            "4way": _node(0.0025),
            "combined": _node(0.00245),
        },
        "one_way_safety_by_target_bucket": {
            "by_order": {"1way": _node(0.00002)},
        },
    }
    comparison = diag._comparison(fake_quality, references)
    fair = comparison["normalized_l1_mean_deltas_fair_groups_only"]
    assert set(fair) == set(diag.FAIR_COMPARABLE_GROUPS)
    combined = fair["heldout_combined"]
    assert combined["gsd_all2way"]["mean"] == pytest.approx(0.00245)
    assert combined["gsd_all2way_minus_engine_all2way_pool"] == (
        pytest.approx(0.00245 - 0.0018944020958222606, abs=1e-6)
    )
    assert combined["gsd_all2way_minus_pgm_all2way"] > 0
    family = comparison["family_level_480_direct"]
    assert family["gsd_all2way_on_480_family"]["mean"] == (
        pytest.approx(0.00001)
    )
    assert family["gsd_minus_engine"] < 0
    assert family["gsd_minus_pgm"] < 0
    assert comparison["interpretation"] == (
        "diagnostic_only_lower_is_better_no_promotion_gate"
    )


def test_wrong_confirmation_fails_before_output_creation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(diag, "_repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="确认值不一致"):
        diag.run("0" * 64)
    assert not (tmp_path / diag.OUTPUT_DIR).exists()


def test_existing_output_refused(monkeypatch, tmp_path):
    destination = tmp_path / diag.OUTPUT_DIR
    destination.mkdir(parents=True)
    monkeypatch.setattr(diag, "_repo_root", lambda: tmp_path)
    with pytest.raises(FileExistsError, match="不覆盖"):
        diag.run(diag.FROZEN_PROTOCOL_SHA256)


def test_plan_is_result_blind(monkeypatch):
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(args)
        raise AssertionError("plan 阶段不得读取任何报告或数据")

    monkeypatch.setattr(diag.all2way, "_sha256_file", forbidden)
    monkeypatch.setattr(diag.all2way, "_load_json_object", forbidden)
    plan = diag.build_plan()
    assert plan["protocol_sha256"] == diag.FROZEN_PROTOCOL_SHA256
    assert not forbidden_calls
    payload = json.dumps(plan)
    assert "normalized_l1" not in payload
