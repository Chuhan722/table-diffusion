from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import freeze_issue53_test_query_workload_ab as freeze
from scripts import run_issue53_test_query_workload_ab as collection

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_is_exact_30_case_numpy_matrix():
    protocol = collection.frozen_protocol_manifest()

    assert collection.protocol_sha256() == collection.FROZEN_PROTOCOL_SHA256
    assert protocol["seeds"] == [318, 319, 320, 321, 322]
    assert protocol["geometries"] == [
        "absolute",
        "sqrt_relative",
        "relative",
    ]
    assert protocol["trajectory_count"] == 30
    assert protocol["common_generator"]["inner_early_stopping_patience_ticks"] == 6
    assert protocol["common_generator"]["n_rounds"] == 6000
    assert protocol["common_generator"]["candidate_budget"] == 6000
    assert protocol["common_generator"]["factorized_gibbs_sweeps"] == 0
    assert protocol["four_way_path_contract"] == {
        "workload_b_four_way_query_count": 5,
        "queries_passed_untruncated_to_objective_and_direction": True,
        "terminal_measured_loss_recomputed_over_all_50_queries": True,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_use_compiled_workload": False,
        "factorized_gibbs_max_order_is_inert": True,
    }


def test_case_order_is_paired_by_geometry_within_each_seed():
    assert collection.CASE_ORDER == (
        ("A", "absolute"),
        ("B", "absolute"),
        ("A", "sqrt_relative"),
        ("B", "sqrt_relative"),
        ("A", "relative"),
        ("B", "relative"),
    )
    plan = collection.build_plan()
    assert [shard["seed"] for shard in plan["shards"]] == list(collection.SEEDS)
    assert [shard["case_count"] for shard in plan["shards"]] == [6] * 5
    assert plan["generation_started"] is False
    assert plan["scientific_overrides_allowed"] is False


def test_plan_does_not_audit_inputs_load_runtime_or_generate(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan must not read inputs or load runtime")

    monkeypatch.setattr(collection, "_audit_inputs", forbidden)
    monkeypatch.setattr(collection, "_load_runtime", forbidden)
    monkeypatch.setattr(collection, "_run_case", forbidden)

    assert collection.build_plan()["mode"] == (
        "plan_only_no_input_or_result_read_no_generation"
    )


def test_formal_input_audit_has_five_four_way_queries_and_no_reference_open(
    monkeypatch,
):
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".csv":
            raise AssertionError(f"collector preflight opened raw CSV: {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    audit = collection._audit_inputs(ROOT)

    assert audit["workloads"]["A"]["four_way_query_count"] == 0
    assert audit["workloads"]["B"]["four_way_query_count"] == 5
    assert audit["workloads"]["B"]["order_counts"] == {2: 30, 3: 15, 4: 5}


def test_generator_rejects_nonfrozen_cases_and_has_inert_max_order():
    params = collection.generator_params(318, "absolute")
    assert params["factorized_gibbs_sweeps"] == 0
    assert params["factorized_gibbs_use_compiled_workload"] is False
    assert params["factorized_gibbs_max_order"] == 3
    with pytest.raises(ValueError, match="seed 不在冻结矩阵"):
        collection.generator_params(999, "absolute")
    with pytest.raises(ValueError, match="geometry 不在冻结矩阵"):
        collection.generator_params(318, "new_geometry")


class _FakeVector(list):
    @property
    def shape(self):
        return (len(self),)


class _FakeNumpy:
    __version__ = "fake"

    @staticmethod
    def asarray(values, dtype=None):
        del dtype
        return _FakeVector(values)


class _FakeFrame:
    def reset_index(self, drop=True):
        assert drop is True
        return self

    def equals(self, other):
        return isinstance(other, _FakeFrame)

    def to_csv(self, path=None, index=False):
        assert index is False
        text = "age,education,employment,income,marital,children,housing,vehicle,health,region\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
            return None
        return text


def _fake_runtime(captured):
    def load_queries(path):
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)["queries"]

    def run_evolution(**kwargs):
        queries = kwargs["queries"]
        target = kwargs["target"]
        captured["query_count"] = len(queries)
        captured["order_counts"] = freeze._order_counts(queries)
        captured["target"] = list(target)
        geometry = kwargs["residual_geometry"]
        frame = _FakeFrame()
        diagnostics = {
            "final_table": frame,
            "output_table_identity": "terminal_current",
            "params": {
                "residual_geometry": geometry,
                "residual_geometry_floor": (
                    None if geometry == "absolute" else 8.0
                ),
                "factorized_gibbs_sweeps": 0,
                "factorized_gibbs_use_compiled_workload": False,
            },
            "factorized_gibbs_factor_count": 0,
            "termination_reason": "early_stopped",
            "rounds_run": 1,
            "transition_clock_history": [{
                "accepted_attempt": 1,
                "attempts": [{"participating_rows": 30}],
            }],
            "accept_history": [True],
            "candidate_evaluation_count": 1,
            "state_evaluation_count": 1,
            "final_current_squared_loss": 0.0,
            "final_current_normalized_l1": 0.0,
            "current_state_metrics_history": [{"current_squared_loss": 0.0}],
            "inner_complete": True,
            "best_loss": 0.0,
            "initial_table_sha256": "1" * 64,
            "primary_rng_post_initialization_state_sha256": "2" * 64,
        }
        return frame, diagnostics

    return SimpleNamespace(
        np=_FakeNumpy(),
        pd=SimpleNamespace(__version__="fake"),
        load_schema=lambda _path: object(),
        load_marginals=lambda _path: object(),
        load_queries=load_queries,
        run_evolution=run_evolution,
        evaluate_table=lambda _table, _queries: captured["target"],
        compute_squared_loss=lambda _target, _answers: 0.0,
        compute_normalized_l1=lambda _target, _answers, _records: 0.0,
    )


def test_workload_b_four_way_queries_reach_full_run_evolution_vector(tmp_path):
    captured = {}
    result = collection._run_case(
        ROOT,
        tmp_path,
        workload="B",
        geometry="absolute",
        seed=318,
        protocol_sha=collection.FROZEN_PROTOCOL_SHA256,
        git_commit="3" * 40,
        runtime=_fake_runtime(captured),
    )

    assert captured["query_count"] == 50
    assert captured["order_counts"] == {2: 30, 3: 15, 4: 5}
    assert captured["target"][-5:] == [0, 1, 4, 3, 0]
    assert result["measured_four_way_query_count"] == 5
    assert result["four_way_queries_in_full_objective_and_early_stop"] is True
    assert result["factorized_gibbs_inactive"] is True


def test_actual_core_four_way_objective_smoke_when_runtime_is_available():
    try:
        runtime = collection._load_runtime()
    except ModuleNotFoundError as exc:
        pytest.skip(f"full project runtime is unavailable: {exc}")

    audit = collection._audit_workload(ROOT, "B")
    spec = collection.WORKLOADS["B"]
    schema = runtime.load_schema(str(ROOT / collection.SCHEMA_PATH))
    queries = runtime.load_queries(str(ROOT / spec["path"]))
    marginals = runtime.load_marginals(str(ROOT / collection.MARGINALS_PATH))
    target = runtime.np.asarray(audit["targets"], dtype=float)
    params = collection.generator_params(318, "absolute")
    params.update({"n_rounds": 1, "candidate_budget": 1, "log_every": 0})

    table, diagnostics = runtime.run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=collection.N_RECORDS,
        marginals=marginals,
        device=collection.DEVICE,
        init_method="marginal",
        **params,
    )
    answers = runtime.np.asarray(runtime.evaluate_table(table, queries), dtype=float)

    assert freeze._order_counts(queries) == {2: 30, 3: 15, 4: 5}
    assert answers.shape == target.shape == (50,)
    assert diagnostics["factorized_gibbs_factor_count"] == 0
    assert runtime.compute_squared_loss(target, answers) == (
        diagnostics["final_current_squared_loss"]
    )


def test_wrong_protocol_sha_fails_before_runtime_or_environment(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong SHA must fail first")

    monkeypatch.setattr(collection, "_load_runtime", forbidden)
    monkeypatch.setattr(collection, "_environment", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        collection.run_shard("0" * 64, 0)


def test_cli_exposes_only_shard_index_and_protocol_confirmation():
    parser = collection._build_parser()
    parsed = parser.parse_args([
        "run-shard",
        "--confirm-protocol-sha",
        collection.FROZEN_PROTOCOL_SHA256,
        "--shard-index",
        "2",
    ])
    assert vars(parsed) == {
        "command": "run-shard",
        "confirm_protocol_sha": collection.FROZEN_PROTOCOL_SHA256,
        "shard_index": 2,
    }
    with pytest.raises(SystemExit):
        parser.parse_args([
            "run-shard",
            "--confirm-protocol-sha",
            collection.FROZEN_PROTOCOL_SHA256,
            "--shard-index",
            "0",
            "--rho",
            "0.1",
        ])
