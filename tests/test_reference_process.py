"""Issue #53 Stage 1 的前缀不变参考过程契约。"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import table_diffevo.evolution as evolution_module
from table_diffevo.evolution import run_evolution
from table_diffevo.queries import (
    evaluate_table,
    load_data,
    load_queries,
)
from table_diffevo.reference_process import (
    REFERENCE_PROCESS_CONTRACT_VERSION,
    run_horizon_invariant_evolution,
)
from table_diffevo.schema import AttributeBlock, Schema, load_schema


def _table_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reduced_real_problem(n_records: int = 48):
    schema = load_schema("configs/test_300x10/schema.yaml")
    queries = load_queries(
        "configs/test_300x10/measured_50query.json"
    )
    source = load_data(
        "data/test_300x10/test_300x10.csv"
    ).iloc[:n_records].reset_index(drop=True)
    target = evaluate_table(source, queries).astype(float)
    return schema, queries, target, n_records


def _binary_problem():
    schema = Schema([
        AttributeBlock(
            name="a", type="categorical", description="a", values=[0, 1]
        ),
        AttributeBlock(
            name="b", type="categorical", description="b", values=[0, 1]
        ),
        AttributeBlock(
            name="c", type="categorical", description="c", values=[0, 1]
        ),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "c", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1},
            {"attribute": "c", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
            {"attribute": "c", "operator": "==", "value": 1},
        ]},
    ]
    source = pd.DataFrame({
        "a": [0, 1, 0, 1] * 4,
        "b": [0, 0, 1, 1] * 4,
        "c": [0, 1, 1, 0, 1, 0, 0, 1] * 2,
    })
    target = evaluate_table(source, queries).astype(float)
    return schema, queries, target, len(source)


def _reference_kwargs(problem):
    schema, queries, target, n_records = problem
    return {
        "target": target,
        "queries": queries,
        "schema": schema,
        "n_records": n_records,
        "seed": 20260814,
        "fixed_alpha": 6.0,
        "rho": 0.35,
        "eta": 0.45,
        "mu": 0.02,
        "diffusion_direction_strength": 0.8,
        "diffusion_direction_reference_scale": 1.25,
        "diffusion_direction_logit_clip": 9.0,
        "device": "numpy",
        "log_every": 100_000,
    }


class TestHorizonInvariantReference:
    def test_independent_short_run_is_exact_prefix_of_long_run(
        self, monkeypatch
    ):
        short_budget = 4
        long_budget = 7
        common = _reference_kwargs(_reduced_real_problem())

        short_table, short = run_horizon_invariant_evolution(
            n_rounds=short_budget, **common
        )

        original_evolve_step = evolution_module.evolve_step
        long_proposals = []

        def observed_evolve_step(*args, **kwargs):
            result = original_evolve_step(*args, **kwargs)
            proposal = result[0] if isinstance(result, tuple) else result
            long_proposals.append(proposal.copy(deep=True))
            return result

        monkeypatch.setattr(
            evolution_module, "evolve_step", observed_evolve_step
        )
        _, long = run_horizon_invariant_evolution(
            n_rounds=long_budget, **common
        )

        assert short["rounds_run"] == short_budget
        assert long["rounds_run"] == long_budget
        assert short["accept_history"] == [True] * short_budget
        assert short["initial_table_sha256"] == long[
            "initial_table_sha256"
        ]
        assert short["current_state_metrics_history"] == long[
            "current_state_metrics_history"
        ][: short_budget + 1]
        for history_key in (
            "loss_history",
            "alpha_history",
            "rho_schedule_history",
            "accept_history",
            "proposal_attempts_history",
            "accepted_attempt_history",
            "accepted_rho_history",
            "raw_proposal_gain_history",
            "raw_proposal_linear_gain_history",
            "raw_proposal_quadratic_penalty_history",
        ):
            assert short[history_key] == long[history_key][:short_budget]

        short_clocks = short["transition_clock_history"]
        assert short_clocks == long["transition_clock_history"][
            :short_budget
        ]
        assert len(short_clocks) == short_budget
        assert all(
            clock["accepted_attempt"] == 1 for clock in short_clocks
        )
        assert all(len(clock["attempts"]) == 1 for clock in short_clocks)
        assert short["candidate_evaluation_count"] == short_clocks[-1][
            "candidate_evaluation_count_cumulative"
        ]
        assert short["primary_rng_state_sha256"] == long[
            "transition_clock_history"
        ][short_budget - 1]["primary_rng_state_sha256"]
        assert short["factorized_gibbs_rng_state_sha256"] is None
        assert long["transition_clock_history"][short_budget - 1][
            "factorized_gibbs_rng_state_sha256"
        ] is None

        pd.testing.assert_frame_equal(
            short_table, long_proposals[short_budget - 1]
        )
        assert _table_sha256(short_table) == short_clocks[-1][
            "post_current_table_sha256"
        ]

    def test_reference_output_is_current_and_strict_json(self):
        common = _reference_kwargs(_binary_problem())
        common["seed"] = np.int64(20260814)
        table, diagnostics = run_horizon_invariant_evolution(
            n_rounds=3, **common
        )

        assert _table_sha256(table) == diagnostics[
            "transition_clock_history"
        ][-1]["post_current_table_sha256"]
        assert "final_table" not in diagnostics
        for ambiguous_best_key in (
            "best_loss",
            "normalized_l1_error",
            "normalized_l1_median",
            "normalized_l1_p90",
            "normalized_l1_max",
        ):
            assert ambiguous_best_key not in diagnostics
        assert "best_loss_diagnostic_only" in diagnostics
        assert (
            "normalized_l1_at_best_squared_loss_diagnostic_only"
            in diagnostics
        )
        contract = diagnostics["reference_process_contract"]
        assert contract["version"] == REFERENCE_PROCESS_CONTRACT_VERSION
        assert contract["output_state_role"] == "final_current"
        assert contract["n_rounds_role"] == "maximum_budget_only"
        assert diagnostics["params"]["n_rounds"] == 3
        assert isinstance(diagnostics["params"]["seed"], int)
        assert diagnostics["params"]["alpha_schedule_mode"] == "fixed"
        assert diagnostics["params"]["fixed_alpha"] == 6.0
        assert diagnostics["params"]["alpha_min"] is None
        assert diagnostics["params"]["alpha_max"] is None
        assert diagnostics["params"]["tol"] == (
            "positive_infinity_no_gate"
        )
        json.dumps(diagnostics, ensure_ascii=False, allow_nan=False)

    def test_factorized_prefix_preserves_both_rng_streams(self):
        common = _reference_kwargs(_binary_problem())
        common.update({
            "factorized_gibbs_sweeps": 2,
            "factorized_gibbs_max_order": 3,
            "factorized_gibbs_logit_clip": 13.0,
        })

        _, short = run_horizon_invariant_evolution(
            n_rounds=3, **common
        )
        _, long = run_horizon_invariant_evolution(
            n_rounds=5, **common
        )

        assert short["transition_clock_history"] == long[
            "transition_clock_history"
        ][:3]
        terminal_clock = long["transition_clock_history"][2]
        assert short["primary_rng_state_sha256"] == terminal_clock[
            "primary_rng_state_sha256"
        ]
        assert short["factorized_gibbs_rng_state_sha256"] == (
            terminal_clock["factorized_gibbs_rng_state_sha256"]
        )
        assert short["factorized_gibbs_microsteps"] > 0
        first_attempt = short[
            "factorized_gibbs_attempt_diagnostics_history"
        ][0][0]
        assert first_attempt["direction_logit_clip"] == 9.0
        assert first_attempt["gibbs_logit_clip"] == 13.0

    def test_transition_observation_does_not_change_state_or_rng(self):
        schema, queries, target, n_records = _binary_problem()
        common = dict(
            target=target,
            queries=queries,
            schema=schema,
            n_records=n_records,
            n_rounds=4,
            seed=9,
            rho=0.4,
            eta=0.5,
            mu=0.02,
            tol=float("inf"),
            residual_directed_diffusion=True,
            diffusion_direction_strength=0.7,
            diffusion_direction_normalization="fixed",
            diffusion_direction_reference_scale=1.0,
            alpha_schedule_mode="fixed",
            fixed_alpha=4.0,
            horizon_invariant=True,
            return_final_table=True,
            log_every=100_000,
        )

        _, plain = run_evolution(
            record_transition_clocks=False, **common
        )
        _, observed = run_evolution(
            record_transition_clocks=True, **common
        )

        pd.testing.assert_frame_equal(
            plain["final_table"], observed["final_table"]
        )
        assert plain["primary_rng_state_sha256"] == observed[
            "primary_rng_state_sha256"
        ]
        for key in (
            "current_state_metrics_history",
            "loss_history",
            "accept_history",
            "raw_proposal_gain_history",
        ):
            assert plain[key] == observed[key]
        assert plain["transition_clock_history"] == []
        assert len(observed["transition_clock_history"]) == 4


class TestHorizonInvariantGuards:
    @staticmethod
    def _valid_core_kwargs():
        schema, queries, target, n_records = _binary_problem()
        return {
            "target": target,
            "queries": queries,
            "schema": schema,
            "n_records": n_records,
            "n_rounds": 1,
            "seed": 0,
            "distance_mode": "geometric",
            "alpha_schedule_mode": "fixed",
            "fixed_alpha": 4.0,
            "rho": 0.2,
            "tol": float("inf"),
            "max_retries": 0,
            "residual_directed_diffusion": True,
            "diffusion_direction_strength": 0.5,
            "diffusion_direction_normalization": "fixed",
            "diffusion_direction_reference_scale": 1.0,
            "residual_self_cooling": None,
            "rho_anneal_end": None,
            "horizon_invariant": True,
            "log_every": 100_000,
        }

    @pytest.mark.parametrize(
        "changes,match",
        [
            ({"distance_mode": "multiplicative"}, "geometric"),
            (
                {
                    "alpha_schedule_mode": "legacy_linear_horizon",
                    "fixed_alpha": None,
                },
                "alpha",
            ),
            (
                {
                    "diffusion_direction_normalization": "initial_rms",
                    "diffusion_direction_reference_scale": None,
                },
                "fixed s0",
            ),
            ({"tol": 0.0}, "tol"),
            ({"tol": "disabled"}, "tol"),
            ({"max_retries": 1}, "max_retries"),
            ({"residual_self_cooling": 1.0}, "self_cooling"),
            ({"rho_anneal_end": 0.1}, "rho_anneal_end"),
        ],
    )
    def test_core_mode_fails_closed(self, changes, match):
        kwargs = self._valid_core_kwargs()
        kwargs.update(changes)
        with pytest.raises(ValueError, match=match):
            run_evolution(**kwargs)

    @pytest.mark.parametrize(
        "changes,match",
        [
            ({"alpha_schedule_mode": "unknown"}, "alpha_schedule_mode"),
            ({"fixed_alpha": np.inf}, "fixed_alpha"),
            (
                {"diffusion_direction_reference_scale": 0.0},
                "reference_scale",
            ),
            ({"diffusion_direction_logit_clip": 0.0}, "logit_clip"),
            ({"record_transition_clocks": "yes"}, "布尔"),
            ({"horizon_invariant": "yes"}, "布尔"),
            ({"seed": None}, "seed"),
            ({"rho": np.nan}, "rho"),
        ],
    )
    def test_new_parameter_identity_rejects_invalid_values(
        self, changes, match
    ):
        kwargs = self._valid_core_kwargs()
        kwargs.update(changes)
        with pytest.raises(ValueError, match=match):
            run_evolution(**kwargs)

    @pytest.mark.parametrize("n_rounds", [0, -1, True, 1.5])
    def test_reference_requires_positive_integer_budget(self, n_rounds):
        with pytest.raises(ValueError, match="n_rounds"):
            run_horizon_invariant_evolution(
                n_rounds=n_rounds,
                **_reference_kwargs(_binary_problem()),
            )

    def test_reference_rejects_overrides_of_owned_parameters(self):
        with pytest.raises(ValueError, match="强制管理.*distance_mode"):
            run_horizon_invariant_evolution(
                n_rounds=2,
                distance_mode="geometric",
                **_reference_kwargs(_binary_problem()),
            )


class TestLegacyScheduleNegativeControl:
    def test_total_budget_changes_legacy_alpha_and_state_prefix(self):
        schema, queries, target, n_records = _reduced_real_problem()
        common = dict(
            target=target,
            queries=queries,
            schema=schema,
            n_records=n_records,
            seed=20260814,
            rho=0.35,
            eta=0.45,
            mu=0.02,
            tol=float("inf"),
            alpha_min=0.0,
            alpha_max=30.0,
            residual_directed_diffusion=True,
            diffusion_direction_strength=0.8,
            diffusion_direction_normalization="fixed",
            diffusion_direction_reference_scale=1.25,
            record_transition_clocks=True,
            log_every=100_000,
        )

        _, short = run_evolution(n_rounds=4, **common)
        _, long = run_evolution(n_rounds=7, **common)

        assert short["alpha_history"] != long["alpha_history"][:4]
        short_hashes = [
            clock["post_current_table_sha256"]
            for clock in short["transition_clock_history"]
        ]
        long_prefix_hashes = [
            clock["post_current_table_sha256"]
            for clock in long["transition_clock_history"][:4]
        ]
        assert short_hashes[0] == long_prefix_hashes[0]
        assert short_hashes != long_prefix_hashes
