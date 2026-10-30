"""公开输入的只读L1/L2监控记录工具；报警不改变生成轨迹，也不替代正常终表。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from fractions import Fraction
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.cli import load_public_schema
from resevo.dataset import StateRegistry, load_queries, target_from_specs
from resevo.editspace import tuples_from_ids
from resevo.failure import save_numerical_failure
from resevo.quadratic_monitor import IndependentGaussianQuadraticMonitor
from resevo.sampling import sampling_summary
from resevo.state import table_answers


def _dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_table(path, schema, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(schema.fields)
        writer.writerows(rows)


def _finite_mean(values):
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(values))
    if not math.isfinite(result) and np.isfinite(values).all():
        scale = float(np.max(np.abs(values)))
        result = math.fsum(float(value) / scale for value in values) / len(values) * scale
    if not math.isfinite(result):
        raise FloatingPointError("numerical_indeterminate: L1对照均值非有限")
    return result


def _sigmas(specs, name):
    values = np.asarray([np.nan if q.sigma is None else q.sigma for q in specs], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name}每题必须带有限且非负的计数sigma")
    return values


def _records(result):
    records = []
    for record in result.records:
        row = asdict(record)
        if math.isnan(row["moved_rows"]):
            row["moved_rows"] = None
        records.append(row)
    return records


def run_recording(args):
    schema = load_public_schema(args.schema)
    specs = load_queries(str(args.learning), expected_record_count=args.rows)
    target = target_from_specs(specs)
    if not specs or not np.isfinite(target).all():
        raise ValueError("学习目标必须非空且有限")
    learning_sigma = _sigmas(specs, "学习卷")
    exact_scale = sum((Fraction.from_float(float(value)) ** 2 for value in learning_sigma), Fraction(0)) / 2
    try:
        reference_scale = float(exact_scale)
    except OverflowError as exc:
        raise ValueError("学习参考触发尺度超出float64范围") from exc
    if not math.isfinite(reference_scale) or (exact_scale != 0 and reference_scale == 0.0):
        raise ValueError("学习参考触发尺度不可表示")
    weights = np.ones(len(specs))
    validation = validation_sigma = None
    if args.monitor != "off":
        validation_specs = load_queries(str(args.validation), expected_record_count=args.rows)
        if len(validation_specs) != len(specs) or any(
            (a.query_id, a.conditions) != (b.query_id, b.conditions)
            for a, b in zip(specs, validation_specs)
        ):
            raise ValueError("验证卷必须与学习卷同题、同序、同记录数")
        validation = target_from_specs(validation_specs)
        if not np.isfinite(validation).all():
            raise ValueError("验证目标必须有限")
        if args.monitor in ("l2", "both"):
            validation_sigma = _sigmas(validation_specs, "验证卷")

    init_rng = np.random.default_rng(args.seed)
    menu_rng = np.random.default_rng(100000 + args.seed)
    select_rng = np.random.default_rng(200000 + args.seed)
    sample_rng = np.random.default_rng(300000 + args.seed)
    rows = [
        tuple(domain[int(init_rng.integers(len(domain)))] for domain in schema.domains)
        for _ in range(args.rows)
    ]
    registry = StateRegistry(schema, specs)
    registry.set_lazy_features(args.backend == "gpu")
    state_ids = registry.register_table(rows)
    max_looks = (args.rounds + args.interval - 1) // args.interval
    preflight_monitor = None
    if args.monitor in ("l2", "both"):
        preflight_monitor = IndependentGaussianQuadraticMonitor(
            registry.answers_for_rows(rows), validation, validation_sigma, weights,
            error_probability=args.error_probability, max_looks=max_looks,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    _write_table(args.output / "initial.csv", schema, rows)
    _dump(args.output / "configuration.json", {
        "monitor": args.monitor, "record_only": True, "rows": args.rows, "rounds": args.rounds,
        "seed": args.seed, "backend": args.backend, "interval": args.interval,
        "initialization": "uniform", "weights": "unit", "alpha": 0.5, "stay": 0.9,
        "damping": 1.0, "frozen_retries": 6,
        "reference_rule": "first scheduled learning loss <= 0.5*sum(sigma_A^2)",
        "reference_scale": reference_scale, "max_looks": max_looks,
        "error_probability": args.error_probability, "l1_z": args.l1_z, "l1_runs": args.l1_runs,
        "critical_value": None if preflight_monitor is None else preflight_monitor.critical_value,
        "error_probability_per_look": None if preflight_monitor is None else preflight_monitor.error_probability_per_look,
        "input_sha256": {
            "schema": hashlib.sha256(args.schema.read_bytes()).hexdigest(),
            "learning": hashlib.sha256(args.learning.read_bytes()).hexdigest(),
            "validation": None if args.monitor == "off" else hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_model_verified_by_program": False,
    })
    provider = make_batch_provider(
        registry, target, weights, menu_rng, pairing=False,
        defer_workload=args.backend == "gpu", select_rng=select_rng,
    )
    reference_round = None
    reference_answers = reference_errors = None
    l2_monitor = None
    l1_streak = 0
    first_l1_alarm = first_l2_alarm = None
    observations = 0
    answer_seconds = monitor_seconds = 0.0
    terminal = args.output / "terminal.csv"

    with (args.output / "observations.jsonl").open("x", encoding="utf-8") as stream:
        @wraps(provider)
        def observed(current, round_index, frozen_streak=0):
            nonlocal reference_round, reference_answers, reference_errors, l2_monitor
            nonlocal l1_streak, first_l1_alarm, first_l2_alarm, observations, answer_seconds, monitor_seconds
            plan = provider(current, round_index, frozen_streak)
            if round_index % args.interval != 0:
                return plan
            started = time.perf_counter()
            if args.backend == "gpu":
                answers = registry.answers_for_rows(tuples_from_ids(registry, current))
            else:
                answers = table_answers(registry.build_workload(target, weights), current)
            answer_seconds += time.perf_counter() - started
            started = time.perf_counter()
            residual = target - answers
            loss = float(np.dot(residual, residual) / 2)
            if not math.isfinite(loss):
                raise FloatingPointError("numerical_indeterminate: 监控记录的学习损失非有限")
            at_reference = reference_round is None and loss <= reference_scale
            if at_reference:
                reference_round, reference_answers = round_index, answers.copy()
                _write_table(args.output / "reference.csv", schema, tuples_from_ids(registry, current))
                if args.monitor in ("l1", "both"):
                    reference_errors = np.abs(validation - answers)
                if args.monitor in ("l2", "both"):
                    l2_monitor = IndependentGaussianQuadraticMonitor(
                        answers, validation, validation_sigma, weights,
                        error_probability=args.error_probability, max_looks=max_looks,
                    )
            l1 = l2 = None
            if args.monitor in ("l1", "both"):
                errors = np.abs(validation - answers)
                l1 = {"mean_absolute_error": _finite_mean(errors), "difference": None,
                      "sample_se": None, "streak": l1_streak, "alarm": False}
                if reference_errors is not None and not at_reference:
                    delta = errors - reference_errors
                    mean = _finite_mean(delta)
                    se = float(np.std(delta, ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else 0.0
                    if not math.isfinite(se):
                        raise FloatingPointError("numerical_indeterminate: L1对照标准误非有限")
                    l1_streak = l1_streak + 1 if se > 0 and mean >= args.l1_z * se else 0
                    l1.update(difference=mean, sample_se=se, streak=l1_streak, alarm=l1_streak >= args.l1_runs)
                    if l1["alarm"] and first_l1_alarm is None:
                        first_l1_alarm = round_index
                        _write_table(args.output / "l1_first_alarm.csv", schema, tuples_from_ids(registry, current))
            if l2_monitor is not None and not at_reference:
                row = l2_monitor.observe(answers)
                l2 = asdict(row)
                if row.worse and first_l2_alarm is None:
                    first_l2_alarm = round_index
                    _write_table(args.output / "l2_first_alarm.csv", schema, tuples_from_ids(registry, current))
            record = {
                "round": round_index, "learning_loss": loss, "answers": answers.tolist(),
                "at_reference": at_reference, "l1": l1, "l2": l2,
            }
            monitor_seconds += time.perf_counter() - started
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            observations += 1
            return plan

        with save_numerical_failure(registry, terminal):
            result = evolve_batch(
                state_ids, args.rounds, sample_rng, observed, registry,
                alpha=0.5, stay_probability=0.9, damping=1.0,
                max_frozen_retries=6, backend=args.backend,
            )
    final_rows = tuples_from_ids(registry, result.state_ids)
    _write_table(terminal, schema, final_rows)
    status = "off" if args.monitor == "off" else (
        "reference_not_reached" if reference_round is None else
        "reference_only" if observations == 1 or reference_round == (observations - 1) * args.interval else
        "observed"
    )
    summary = {
        "status": "completed", "record_only": True, "monitor_status": status,
        "rows": len(final_rows), "rounds": len(result.records), "stop_reason": result.stop_reason,
        "reference_round": reference_round,
        "reference_answers": None if reference_answers is None else reference_answers.tolist(),
        "first_l1_alarm": first_l1_alarm, "first_l2_alarm": first_l2_alarm,
        "observations": observations, "l2_looks": 0 if l2_monitor is None else l2_monitor.looks_used,
        "max_looks": max_looks, "final_answers": registry.answers_for_rows(final_rows).tolist(),
        "answer_seconds": answer_seconds, "monitor_seconds": monitor_seconds,
        "generation_records": _records(result), "state_ids": result.state_ids.tolist(),
        "sampling_precision": sampling_summary(result.sampling_precision),
        "rng": {
            "initial": init_rng.bit_generator.state, "menu": menu_rng.bit_generator.state,
            "selection": select_rng.bit_generator.state, "sample": sample_rng.bit_generator.state,
        },
    }
    _dump(args.output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--learning", type=Path, required=True, help="固定学习卷，计数单位且每题带sigma")
    parser.add_argument("--validation", type=Path, help="同题同序验证卷；L2要求独立高斯及验证卷自身的sigma")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="全新目录；报警表仅为诊断，terminal.csv不由报警选择")
    parser.add_argument("--monitor", choices=("off", "l1", "l2", "both"), default="off")
    parser.add_argument("--error-probability", type=float, help="L2必填：本参考/轨迹的总误报容许率，不是DP预算")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--l1-z", type=float, default=2.0)
    parser.add_argument("--l1-runs", type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.rows <= 2**50 or args.rounds < 1 or args.interval < 1 or args.seed < 0:
        parser.error("公开行数须为[1,2**50]整数，轮数/间隔须为正，种子须非负")
    if args.monitor == "off" and (args.validation is not None or args.error_probability is not None):
        parser.error("关闭监控时不接受验证卷或误报率参数")
    if args.monitor != "off" and args.validation is None:
        parser.error("启用监控必须显式提供验证卷")
    if args.monitor in ("l2", "both") and args.error_probability is None:
        parser.error("L2监控必须显式提供总误报容许率")
    if args.monitor == "l1" and args.error_probability is not None:
        parser.error("L1经验规则不接受L2误报率参数")
    if not math.isfinite(args.l1_z) or args.l1_z <= 0 or args.l1_runs < 1:
        parser.error("L1对照z须为正有限数，连续次数须为正")
    try:
        summary = run_recording(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: summary[key] for key in (
        "status", "record_only", "monitor_status", "rows", "rounds",
        "stop_reason", "reference_round", "first_l1_alarm", "first_l2_alarm",
    )}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
