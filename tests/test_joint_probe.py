"""联合 mask 冻结探针的最小端到端回归测试。"""

import numpy as np

import scripts.probe_joint_mask_diffusion as probe
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


def _run_smoke():
    schema = load_schema(probe.SCHEMA_PATH)
    queries = load_queries(probe.QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(probe.MARGINALS_PATH)
    state = init_synthetic_table(
        probe.N_RECORDS,
        schema,
        np.random.default_rng(0),
        marginals=marginals,
    )
    return probe._probe_state(
        state,
        target,
        queries,
        schema,
        seed=0,
        state_index=0,
        state_rounds=0,
        temperatures=[1.0],
        proposals=2,
        device="numpy",
        max_active_attributes=12,
    )


def test_frozen_probe_matches_kl_and_preserves_gain_decomposition():
    result = _run_smoke()

    assert result["n_proposals"] == 2
    assert result["mu"] == 0.0
    interaction = result["interaction_summary"]
    assert interaction["active_rows"] > 0
    assert interaction["q0_nonadditive_mask_mass"] > 0.0
    assert interaction["one_hot_direction_max_error"] < 1e-12
    assert interaction["independent_clipped_marginal_max_error"] < 1e-12

    configs = result["proposal_rows"]
    for rows in configs.values():
        assert len(rows) == 2
        for row in rows:
            assert row["gain"] == row["linear_gain"] - row[
                "quadratic_penalty"
            ]

    baseline = configs["baseline"]
    assert all(row["kernel_kl_total"] == 0.0 for row in baseline)
    independent = configs["independent_matched_tau_1"]
    joint = configs["joint_matched_tau_1"]
    for independent_row, joint_row in zip(independent, joint):
        assert np.isclose(
            independent_row["kernel_kl_total"],
            joint_row["kernel_kl_total"],
            rtol=1e-9,
            atol=1e-10,
        )
        assert independent_row["effective_tau_max"] <= 1.0 + 1e-8
        assert joint_row["effective_tau_max"] <= 1.0 + 1e-8
        assert independent_row["matched_kl_error_max"] < 1e-9
        assert joint_row["matched_kl_error_max"] < 1e-9

    paired = result["paired_joint_vs_independent"][
        "joint_vs_independent_matched_tau_1"
    ]
    assert paired["wins"] + paired["ties"] + paired["losses"] == 2


def test_frozen_probe_is_reproducible_with_addressed_random_streams():
    first = _run_smoke()
    second = _run_smoke()

    assert first["state_sha256"] == second["state_sha256"]
    assert first["direction_reference_scale"] == second[
        "direction_reference_scale"
    ]
    assert first["proposal_rows"] == second["proposal_rows"]
    assert first["interaction_rows"] == second["interaction_rows"]
