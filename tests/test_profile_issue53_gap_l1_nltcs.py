import os

import pytest

from scripts import profile_issue53_gap_l1_nltcs as profiler
from table_diffevo.schema import AttributeBlock, Schema


def test_generator_parameters_are_single_trajectory_profile_only():
    params = profiler._generator_parameters()

    assert params["seed"] == 9907
    assert params["n_rounds"] == 100
    assert params["candidate_budget"] == 100
    assert params["gap_l1_sweeps"] == 8
    assert params["factorized_gibbs_sweeps"] == 0
    assert params["device"] == "cuda"
    assert "gap_l1_profile_timing" not in params


def test_timed_generator_preserves_bit_generator_trajectory():
    registry = profiler._TimingRegistry()
    reference = profiler.np.random.default_rng(7)
    timed_base = profiler.np.random.default_rng(7)
    timed = profiler._TimedGenerator(timed_base.bit_generator, registry, "gap_scan_rng")

    assert timed.integers(0, 100) == reference.integers(0, 100)
    assert timed.random() == reference.random()
    assert timed.bit_generator.state == reference.bit_generator.state
    assert len(registry.samples["gap_scan_rng.integers_wall_sec"]) == 1
    assert len(registry.samples["gap_scan_rng.random_wall_sec"]) == 1


def test_instrumentation_context_restores_frozen_bindings():
    registry = profiler._TimingRegistry()
    original_gap = profiler.evolution_module.evolve_step_gap_l1_global
    original_default_rng = profiler.np.random.default_rng

    with profiler._instrument_calls(registry):
        assert profiler.evolution_module.evolve_step_gap_l1_global is not original_gap
        primary = profiler.np.random.default_rng(11)
        gap = profiler.np.random.default_rng(12)
        assert isinstance(primary, profiler._TimedGenerator)
        assert isinstance(gap, profiler._TimedGenerator)

    assert profiler.evolution_module.evolve_step_gap_l1_global is original_gap
    assert profiler.np.random.default_rng is original_default_rng


def test_stage_summary_preserves_totals_and_quantiles():
    summary = profiler.summarize_stage_rows(
        [
            {"round": 1, "kernel": 1.0, "outer": 3.0},
            {"round": 2, "kernel": 2.0, "outer": 5.0},
            {"round": 3, "kernel": 3.0, "outer": 7.0},
        ]
    )

    assert summary["kernel"]["count"] == 3
    assert summary["kernel"]["total"] == 6.0
    assert summary["kernel"]["p50"] == 2.0
    assert summary["outer"]["mean"] == 5.0


def test_stage_summary_rejects_inconsistent_rows():
    with pytest.raises(ValueError, match="字段不一致"):
        profiler.summarize_stage_rows(
            [
                {"round": 1, "kernel": 1.0},
                {"round": 2, "other": 1.0},
            ]
        )


def test_quality_boundary_finds_nested_forbidden_key():
    paths = profiler._forbidden_key_paths(
        {"safe": [{"timing": 1.0}, {"loss_history": [1.0]}]}
    )

    assert paths == ["safe[1].loss_history"]


def test_cuda_external_instrumentation_preserves_tiny_trajectory():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用")
    # 确定性 cuBLAS 要求进程级 workspace 配置（conftest.py 已 setdefault）；
    # 外部注入了其他值时跳过而不是让 torch 抛 RuntimeError。
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        pytest.skip(
            "CUBLAS_WORKSPACE_CONFIG 非确定性取值，无法满足 "
            "torch.use_deterministic_algorithms(True) 的 cuBLAS 要求"
        )
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    schema = Schema(
        [
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("a", "b", "c")
        ]
    )
    queries = [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "b", "operator": "==", "value": 1}]},
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
    ]
    kwargs = {
        "target": profiler.np.asarray([9.0, 11.0, 5.0]),
        "queries": queries,
        "schema": schema,
        "n_records": 20,
        "n_rounds": 2,
        "seed": 20260828,
        "rho": 0.4,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "device": "cuda",
        "distance_mode": "geometric",
        "residual_directed_diffusion": True,
        "gap_l1_sweeps": 8,
        "return_final_table": True,
        "record_transition_clocks": True,
        "stop_on_exact_residual": False,
        "log_every": 100,
    }
    try:
        reference, reference_diagnostics = profiler.run_evolution(**kwargs)
        registry = profiler._TimingRegistry()
        with profiler._instrument_calls(registry):
            observed, observed_diagnostics = profiler.run_evolution(**kwargs)
        registry.finalize_cuda_events()
    finally:
        torch.use_deterministic_algorithms(previous)

    assert observed.equals(reference)
    assert observed_diagnostics["final_table"].equals(
        reference_diagnostics["final_table"]
    )
    assert (
        observed_diagnostics["primary_rng_state_sha256"]
        == (reference_diagnostics["primary_rng_state_sha256"])
    )
    observed_attempts = observed_diagnostics["gap_l1_attempt_diagnostics_history"]
    reference_attempts = reference_diagnostics["gap_l1_attempt_diagnostics_history"]
    assert [row[0]["microstep_trace_sha256"] for row in observed_attempts] == [
        row[0]["microstep_trace_sha256"] for row in reference_attempts
    ]
    assert (
        registry.summary()["outer.evolve_step_gap_l1_global.python_wall_sec"]["count"]
        == 2
    )
