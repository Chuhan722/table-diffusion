import operator

import pytest

from scripts import issue52_parallel_trajectories as parallel
from scripts import run_issue49_stage_a as stage_a
from table_diffevo.experiment_parallel import (
    assert_scientifically_equal,
    run_ordered_process_tasks,
    scientific_payload,
    scientific_sha256,
    validate_max_workers,
)


def test_worker_limit_and_ordered_process_map():
    assert validate_max_workers(1) == 1
    assert validate_max_workers(8) == 8
    for invalid in (0, 9, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="1..8"):
            validate_max_workers(invalid)

    tasks = [7, 2, 11, 3]
    assert run_ordered_process_tasks(
        operator.neg, tasks, max_workers=2
    ) == [-7, -2, -11, -3]


def test_trajectory_task_validation_rejects_ambiguous_grid():
    independent = parallel.TrajectoryTask(
        config_id="independent_tau_1",
        kernel="independent",
        seed=9901,
        rounds=2,
        temperature=1.0,
        sweeps=0,
    )
    duplicate = parallel.TrajectoryTask(**independent.__dict__)
    target, queries, schema, marginals, _ = stage_a._load_inputs()
    with pytest.raises(ValueError, match="任务身份不得重复"):
        parallel.run_trajectory_tasks(
            target,
            queries,
            schema,
            marginals,
            [independent, duplicate],
            max_workers=1,
        )

    invalid_factor = parallel.TrajectoryTask(
        config_id="factor_tau_1_sweeps_0",
        kernel="factor",
        seed=9901,
        rounds=2,
        temperature=1.0,
        sweeps=0,
    )
    with pytest.raises(ValueError, match="factor.*sweeps"):
        parallel.run_trajectory_tasks(
            target,
            queries,
            schema,
            marginals,
            [invalid_factor],
            max_workers=1,
        )


def test_real_issue52_trajectories_match_serial_and_parallel_exactly():
    """Two real tiny trajectories cover independent and factor Gibbs paths."""
    target, queries, schema, marginals, _ = stage_a._load_inputs()
    tasks = [
        parallel.TrajectoryTask(
            config_id="independent_tau_1",
            kernel="independent",
            seed=9902,
            rounds=2,
            temperature=1.0,
            sweeps=0,
            record_state_hashes=True,
            snapshot_rounds=(0, 1, 2),
        ),
        parallel.TrajectoryTask(
            config_id="factor_tau_1_sweeps_8",
            kernel="factor",
            seed=9902,
            rounds=2,
            temperature=1.0,
            sweeps=8,
            record_state_hashes=True,
            snapshot_rounds=(0, 1, 2),
        ),
    ]

    serial = parallel.run_trajectory_tasks(
        target,
        queries,
        schema,
        marginals,
        tasks,
        max_workers=1,
    )
    concurrent = parallel.run_trajectory_tasks(
        target,
        queries,
        schema,
        marginals,
        tasks,
        max_workers=2,
    )

    assert [row["task_id"] for row in concurrent] == [
        task.task_id for task in tasks
    ]
    assert serial[0]["run"]["initial_csv_sha256"] == (
        serial[1]["run"]["initial_csv_sha256"]
    )
    assert "elapsed_sec" in serial[0]["run"]
    assert "elapsed_sec" not in scientific_payload(serial[0]["run"])
    assert scientific_sha256(serial) == scientific_sha256(concurrent)
    assert assert_scientifically_equal(serial, concurrent) == (
        scientific_sha256(serial)
    )
