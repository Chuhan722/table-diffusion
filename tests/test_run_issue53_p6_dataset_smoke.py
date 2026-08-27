from __future__ import annotations

import pytest

from scripts import run_issue53_p6_dataset_smoke as smoke


def test_plan_is_result_blind_and_fixed(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan must not read or generate")

    monkeypatch.setattr(smoke, "_audit_inputs", forbidden)
    monkeypatch.setattr(smoke, "_load_baselines", forbidden)
    monkeypatch.setattr(smoke, "run_evolution", forbidden)

    plan = smoke.build_plan()

    assert plan["protocol_sha256"] == smoke.FROZEN_PROTOCOL_SHA256
    assert plan["dataset_run_order"] == ["test_300x10", "nltcs"]
    assert plan["trajectory_count"] == 2
    assert plan["parameter_overrides_allowed"] is False
    assert plan["generation_started"] is False


def test_generator_configuration_is_exactly_the_current_p6_main_arm():
    params = smoke.generator_params()

    assert params["seed"] == 200
    assert params["rho"] == 0.01
    assert params["n_rounds"] == 6000
    assert params["candidate_budget"] == 6000
    assert params["inner_early_stopping_patience_ticks"] == 6
    assert params["tol"] == float("inf")
    assert params["max_retries"] == 0
    assert params["alpha_schedule_mode"] == "fixed"
    assert params["fixed_alpha"] == 16.0
    assert params["residual_geometry"] == "relative"
    assert params["residual_geometry_floor"] == 8.0
    assert params["diffusion_direction_normalization"] == "initial_rms"
    assert params["factorized_gibbs_sweeps"] == 0
    assert params["return_final_table"] is True


def test_protocol_identity_and_scope():
    protocol = smoke.frozen_protocol_manifest()

    assert smoke.protocol_sha256() == smoke.FROZEN_PROTOCOL_SHA256
    assert protocol["expected_normalized_work_cap"] == 60.0
    assert protocol["output_identity"] == "terminal_current"
    assert protocol["parameter_retuning_allowed"] is False
    assert protocol["online_l1_used"] is False
    assert protocol["raw_reference_data_accessed"] is False
    assert [item["name"] for item in protocol["datasets"]] == [
        "test_300x10",
        "nltcs",
    ]


def test_cli_has_no_scientific_overrides():
    parser = smoke._build_parser()
    parsed = parser.parse_args(
        ["run", "--confirm-protocol-sha", smoke.FROZEN_PROTOCOL_SHA256]
    )
    assert set(vars(parsed)) == {"command", "confirm_protocol_sha"}
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--confirm-protocol-sha",
                smoke.FROZEN_PROTOCOL_SHA256,
                "--rho",
                "0.1",
            ]
        )


def test_repository_inputs_and_same_seed_baseline_match():
    root = smoke._repo_root()

    assert smoke._audit_inputs(root) == {
        name: spec["sha256"] for name, spec in smoke.DATASETS.items()
    }
    baselines = smoke._load_baselines(root)
    assert baselines["test_300x10"]["final_normalized_l1"] == (0.003666666666666667)
    assert baselines["nltcs"]["final_normalized_l1"] == (0.0002652313387125822)


def test_baseline_hash_drift_fails_closed(monkeypatch):
    monkeypatch.setattr(smoke, "BASELINE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="baseline artifact 身份漂移"):
        smoke._load_baselines(smoke._repo_root())


def test_natural_work_counts_only_applied_attempt():
    assert (
        smoke._applied_rows(
            {"accepted_attempt": 0, "attempts": [{"participating_rows": 5}]}
        )
        == 0
    )
    assert (
        smoke._applied_rows(
            {
                "accepted_attempt": 2,
                "attempts": [
                    {"participating_rows": 3},
                    {"participating_rows": 7},
                ],
            }
        )
        == 7
    )


def test_wrong_sha_fails_before_environment_or_generation(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong SHA must fail first")

    monkeypatch.setattr(smoke, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(smoke, "_environment", forbidden)
    monkeypatch.setattr(smoke, "run_evolution", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        smoke.run("0" * 64)


def test_existing_output_fails_before_environment(tmp_path, monkeypatch):
    (tmp_path / smoke.OUTPUT_DIR).mkdir(parents=True)
    monkeypatch.setattr(smoke, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        smoke,
        "_environment",
        lambda _root: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises(FileExistsError, match="输出已存在"):
        smoke.run(smoke.FROZEN_PROTOCOL_SHA256)


def test_dirty_tree_fails_before_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "_git_text", lambda *_args: "?? dirty.py")
    with pytest.raises(RuntimeError, match="干净工作树"):
        smoke._environment(tmp_path)
