import json

import pytest

from scripts import issue52_protocol as protocol
from scripts import run_issue52_stage_t as runner
from table_diffevo.experiment_parallel import assert_scientifically_equal


def test_formal_protocol_freezes_grid_horizon_and_checkpoints():
    formal = protocol.stage_t_protocol("formal")
    smoke = protocol.stage_t_protocol("smoke")

    assert formal["stage_t_seeds"] == list(range(200, 210))
    assert formal["state_library_seeds"] == [200, 201, 202]
    assert formal["source_temperatures"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert formal["rounds"] == 3000
    assert formal["trend_checkpoints"] == [
        500, 1000, 1500, 2000, 2500, 3000
    ]
    assert formal["snapshot_rounds"] == [0, 1000, 2000, 3000]
    assert formal["late_window_size"] == 500
    assert formal["max_workers_allowed"] == 8
    assert formal["worker_count_is_nonscientific"] is True
    assert set(smoke["stage_t_seeds"]).isdisjoint(formal["stage_t_seeds"])
    assert smoke["rounds"] == 12


def test_task_grid_is_complete_ordered_and_snapshots_are_scoped():
    frozen = protocol.stage_t_protocol("formal")
    tasks = runner._build_tasks(frozen)

    assert len(tasks) == 50
    assert [(task.seed, task.temperature) for task in tasks[:5]] == [
        (200, 1.0), (200, 2.0), (200, 3.0), (200, 4.0), (200, 5.0)
    ]
    assert all(
        task.snapshot_rounds == (0, 1000, 2000, 3000)
        for task in tasks if task.seed in (200, 201, 202)
    )
    assert all(
        task.snapshot_rounds is None
        for task in tasks if task.seed not in (200, 201, 202)
    )
    assert len({task.task_id for task in tasks}) == 50


def test_trend_windows_use_post_round_current_state_and_fixed_late_window():
    frozen = protocol.stage_t_protocol("smoke")
    run = {"current_loss_after_round_history": list(range(1, 13))}

    trend = runner._trend_for_run(run, frozen)

    assert trend["checkpoint_windows"]["2"] == {
        "start_round": 1,
        "end_round": 2,
        "round_count": 2,
        "current_loss_mean": 1.5,
        "current_loss_median": 1.5,
        "current_loss_final": 2.0,
    }
    assert trend["checkpoint_windows"]["12"]["start_round"] == 11
    assert trend["late_window_start_round"] == 11
    assert trend["late_window_current_loss_mean"] == 11.5
    assert trend["clearly_descending_at_horizon"] is False
    assert trend["horizon_interpretation"] == (
        "not_clearly_descending_no_equilibrium_claim"
    )


def test_smoke_pipeline_parallel_matches_serial_and_refuses_overwrite(tmp_path):
    serial_path, serial = runner.run_stage_t(
        "smoke", tmp_path / "serial", max_workers=1
    )
    parallel_path, concurrent = runner.run_stage_t(
        "smoke", tmp_path / "parallel", max_workers=2
    )

    assert serial_path.exists() and parallel_path.exists()
    assert serial["formal_result_valid"] is False
    assert concurrent["formal_result_valid"] is False
    assert concurrent["interpretation"] == "pipeline_smoke_only_not_evidence"
    assert concurrent["execution"]["task_count"] == 5
    assert concurrent["execution"]["requested_max_workers"] == 2
    assert concurrent["stage_t"]["identity_gates"][
        "all_identity_gates_passed"
    ] is True
    assert concurrent["state_library_source_manifest"][
        "raw_snapshot_count_in_trajectories"
    ] == 20
    assert concurrent["state_library_source_manifest"][
        "expected_unique_current_states_after_round0_dedup"
    ] == 16
    assert_scientifically_equal(
        serial["stage_t"]["trajectories"],
        concurrent["stage_t"]["trajectories"],
    )
    reloaded = json.loads(parallel_path.read_text(encoding="utf-8"))
    assert reloaded["protocol"] == runner._protocol("smoke")

    with pytest.raises(FileExistsError, match="尚未启动任何 Stage T 轨迹"):
        runner.run_stage_t("smoke", tmp_path / "parallel", max_workers=2)


def test_formal_dirty_tree_refuses_before_loading_inputs_or_running(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner.common,
        "_git_identity",
        lambda: {"commit": "test", "worktree_clean": False},
    )
    monkeypatch.setattr(
        runner,
        "_load_inputs",
        lambda: pytest.fail("dirty formal run must stop before loading inputs"),
    )

    with pytest.raises(RuntimeError, match="工作树干净"):
        runner.run_stage_t("formal", tmp_path / "formal", max_workers=8)
