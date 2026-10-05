"""无噪声 GSD（private_gsd 遗传搜索）在 nltcs all-2way 同餐上的对照局。

诊断性质（diagnostic_only）：
- 第三方零阶基线：官方 private_gsd 内核（GeneticSD + 链式统计），
  与引擎决胜局（fitness_only_all2way_pool_nltcs_seed9908_v1）和
  PGM-all2way（baseline_pgm_all2way_nltcs_v1）完全同餐——
  喂食 = 480 个 2x2 二维格 + 32 个一维格（512 频率，零噪声 rho=inf）。
- 生成在官方仓库独立 venv 内 subprocess 执行（upstream commit 与
  两处守卫补丁后的文件 SHA 全部钉死，fail-closed）；评估全部回到
  主仓库冻结考卷与主评估器（与两组对照完全同卷同口径）。
- 训练参数照抄官方 GSDSynthesizer.fit 默认（5000 万代上限 + 每
  N'=16181 代早停检查、阈值 1e-4、全量四遗传操作符、sparse 统计、
  N_prime=N），零调参。
- 不做任何晋级判断；数字只进对照表。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts import run_baseline_pgm_all2way_nltcs_diagnostic as all2way
from scripts import run_baseline_pgm_nltcs_diagnostic as pgm980
from scripts import run_fitness_only_attribution as base

PROTOCOL_VERSION = "baseline-gsd-all2way-nltcs-v1"
OUTPUT_DIR = Path("outputs/baseline_gsd_all2way_nltcs_v1")
FROZEN_PROTOCOL_SHA256 = (
    "ca51218c5682362837d6b0fbbc3cd9e855a73c414a81f4a9c5ea49ccac9c0aaf"
)

DATASET = "nltcs"
N_RECORDS = 16181
SEED = 9908
NUM_GENERATIONS_CAP = 50_000_000
EARLY_STOP_THRESHOLD = 0.0001
EXPECTED_STAT_COUNT = 512

ALL2WAY_PATH = all2way.ALL2WAY_PATH
ALL2WAY_SHA256 = all2way.ALL2WAY_SHA256
EXPECTED_CELL_QUERY_COUNT = all2way.EXPECTED_CELL_QUERY_COUNT

# GSD 检出路径与解释器可经环境变量覆盖（审查 F3：去个人硬编码路径）；
# 上游 commit 与补丁文件 SHA 对拍不变，路径可配不放松复现合同。
GSD_REPO = Path(
    os.environ.get("TD_GSD_REPO", "/home/chuhan/projects/private_gsd")
)
GSD_PYTHON = Path(
    os.environ.get("TD_GSD_PYTHON", str(GSD_REPO / ".venv/bin/python"))
)
GSD_UPSTREAM_COMMIT = "f6150d7821b9675ce9158b456f1a5cff8bc3b3d6"
GSD_PATCHED_FILES_SHA256 = {
    "src/genetic_sd/__init__.py": (
        "6546def7d7424a19b13e0783d0e9df270467329ee21d6a9b7029bbe66ff54183"
    ),
    "src/genetic_sd/generator/generator_base.py": (
        "e3af165ac77438aa6cd9876cd67039a0326c8cdf64c17234e80d0529359724b1"
    ),
}
GENERATION_SCRIPT_PATH = Path("scripts/gsd_generate_all2way_noisefree.py")
GENERATION_SCRIPT_SHA256 = (
    "331c446bff02ad4e658b07b4902b57c88fbbcfcf47867eea945d9382181926b9"
)

ENGINE_REPORT_PATH = Path(
    "outputs/fitness_only_all2way_pool_nltcs_seed9908_v1/report.json"
)
ENGINE_REPORT_SHA256 = (
    "e8233ac2cdda3e5e2cd888873bc0268e0fd0333ede06126fbda522595599e346"
)
PGM_ALL2WAY_REPORT_PATH = Path(
    "outputs/baseline_pgm_all2way_nltcs_v1/report.json"
)
PGM_ALL2WAY_REPORT_SHA256 = (
    "b7760bab264a4a88e8ced44d10a139d59ac4be59506c51a6b8ab8341112227bc"
)

FAIR_COMPARABLE_GROUPS = (
    "heldout_3way",
    "heldout_4way",
    "heldout_combined",
    "one_way_safety",
)


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "role": "third_party_zero_order_baseline_diagnostic_only",
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "generation": {
            "method": "private_gsd_genetic_sd_kernel_subprocess",
            "repo": str(GSD_REPO),
            "upstream_commit": GSD_UPSTREAM_COMMIT,
            "patched_files_sha256": dict(GSD_PATCHED_FILES_SHA256),
            "patch_scope": (
                "import guards only (optional GSDSynthesizer import and "
                "fail-closed cdp_rho stub); GeneticSD algorithm untouched"
            ),
            "generation_script": str(GENERATION_SCRIPT_PATH),
            "generation_script_sha256": GENERATION_SCRIPT_SHA256,
            "noise_model": "none_rho_inf_exact_statistics",
            "fed_statistics": {
                "one_way_workloads": 16,
                "two_way_workloads": 120,
                "statistic_count": EXPECTED_STAT_COUNT,
                "same_meal_note": (
                    "matches engine 512-cell pool (480 doubles + 32 "
                    "one-way cells) and pgm-all2way (480 cells + 16 "
                    "one-way marginals) information diet"
                ),
            },
            "parameters": {
                "seed": SEED,
                "num_generations_cap": NUM_GENERATIONS_CAP,
                "stop_early": True,
                "stop_early_threshold": EARLY_STOP_THRESHOLD,
                "stop_early_min_generation": N_RECORDS,
                "genetic_operators": [
                    "mutate",
                    "continuous",
                    "cross",
                    "swap",
                ],
                "sparse_statistics": True,
                "n_prime": N_RECORDS,
                "tuning": "none_official_fit_defaults",
            },
        },
        "generation_inputs_sha256": {
            "data_csv": pgm980.INPUT_SHA256["reference"],
            "measured_exam": ALL2WAY_SHA256,
            "schema": pgm980.INPUT_SHA256["schema"],
            "marginals": pgm980.INPUT_SHA256["marginals"],
        },
        "evaluation": {
            "engine": "main_repo_frozen_evaluator",
            "measured_workload": str(ALL2WAY_PATH),
            "heldout_sha256": pgm980.INPUT_SHA256["heldout"],
            "reference_sha256": pgm980.INPUT_SHA256["reference"],
            "note": (
                "same frozen exams and evaluator as engine showdown and "
                "pgm-all2way arms"
            ),
        },
        "references": {
            "engine_all2way_pool": {
                "path": str(ENGINE_REPORT_PATH),
                "sha256": ENGINE_REPORT_SHA256,
            },
            "pgm_all2way": {
                "path": str(PGM_ALL2WAY_REPORT_PATH),
                "sha256": PGM_ALL2WAY_REPORT_SHA256,
            },
        },
        "fair_comparable_groups": list(FAIR_COMPARABLE_GROUPS),
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def protocol_sha256() -> str:
    return all2way._sha256_bytes(
        all2way._strict_json_bytes(_protocol_manifest())
    )


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "gsd all2way baseline protocol SHA-256 漂移: "
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "protocol_sha256": protocol_sha256(),
        "protocol": _protocol_manifest(),
        "launch_gate": "requires --confirm-protocol-sha256",
    }


def _audit_gsd_environment(root: Path) -> dict[str, Any]:
    if not GSD_PYTHON.exists():
        raise RuntimeError(f"GSD venv python 不存在: {GSD_PYTHON}")
    observed_commit = subprocess.run(
        ["git", "-C", str(GSD_REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != GSD_UPSTREAM_COMMIT:
        raise RuntimeError(
            "GSD upstream commit 漂移: "
            f"expected={GSD_UPSTREAM_COMMIT}, observed={observed_commit}"
        )
    observed_files = {
        rel: all2way._sha256_file(GSD_REPO / rel)
        for rel in GSD_PATCHED_FILES_SHA256
    }
    if observed_files != GSD_PATCHED_FILES_SHA256:
        raise RuntimeError(
            "GSD 补丁文件 SHA 漂移: "
            f"expected={GSD_PATCHED_FILES_SHA256}, "
            f"observed={observed_files}"
        )
    observed_script = all2way._sha256_file(root / GENERATION_SCRIPT_PATH)
    if observed_script != GENERATION_SCRIPT_SHA256:
        raise RuntimeError(
            "GSD 生成脚本 SHA 漂移: "
            f"expected={GENERATION_SCRIPT_SHA256}, "
            f"observed={observed_script}"
        )
    return {
        "gsd_python": str(GSD_PYTHON),
        "upstream_commit": observed_commit,
        "patched_files_sha256": observed_files,
        "generation_script_sha256": observed_script,
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "<unset>"
        ),
    }


def _run_generation_subprocess(
    root: Path, staging: Path
) -> tuple[Path, dict[str, Any], float]:
    generation_dir = staging / "generation"
    generation_dir.mkdir(parents=True, exist_ok=True)
    table_path = generation_dir / "gsd_table.csv"
    manifest_path = generation_dir / "gsd_manifest.json"
    command = [
        str(GSD_PYTHON),
        str(root / GENERATION_SCRIPT_PATH),
        "--data-csv",
        str(root / pgm980.REFERENCE_PATH),
        "--expected-data-sha256",
        pgm980.INPUT_SHA256["reference"],
        "--output-csv",
        str(table_path),
        "--manifest-json",
        str(manifest_path),
        "--seed",
        str(SEED),
        "--num-generations",
        str(NUM_GENERATIONS_CAP),
        "--early-stop-threshold",
        str(EARLY_STOP_THRESHOLD),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    subprocess.run(command, check=True, cwd=str(GSD_REPO), env=env)
    wall_seconds = time.time() - started
    manifest = all2way._load_json_object(manifest_path)
    return table_path, manifest, wall_seconds


def _audit_generation_manifest(
    manifest: Mapping[str, Any]
) -> dict[str, Any]:
    stats = manifest["statistics"]
    params = manifest["parameters"]
    output = manifest["output"]
    if float(stats["zero_noise_max_abs_diff"]) != 0.0:
        raise RuntimeError("manifest 零噪声断言不为 0")
    if int(stats["statistic_count"]) != EXPECTED_STAT_COUNT:
        raise RuntimeError("manifest 统计格数漂移")
    if int(params["seed"]) != SEED:
        raise RuntimeError("manifest seed 漂移")
    if int(params["num_generations_cap"]) != NUM_GENERATIONS_CAP:
        raise RuntimeError("manifest 代数上限漂移")
    if float(params["stop_early_threshold"]) != EARLY_STOP_THRESHOLD:
        raise RuntimeError("manifest 早停阈值漂移")
    if int(output["row_count"]) != N_RECORDS:
        raise RuntimeError("manifest 输出行数漂移")
    if (
        manifest["input"]["sha256"]
        != pgm980.INPUT_SHA256["reference"]
    ):
        raise RuntimeError("manifest 输入数据 SHA 漂移")
    return {
        "zero_noise_max_abs_diff": float(
            stats["zero_noise_max_abs_diff"]
        ),
        "statistic_count": int(stats["statistic_count"]),
        "wall_seconds_fit": float(
            manifest["runtime"]["wall_seconds_fit"]
        ),
        "jax_version": str(manifest["runtime"]["jax_version"]),
        "jax_devices": list(manifest["runtime"]["jax_devices"]),
        "sanity_family_fit_frequency_l1": dict(
            manifest["sanity_family_fit_frequency_l1"]
        ),
        "output_sha256": str(output["sha256"]),
    }


def _load_gsd_table(
    table_path: Path, names: list[str]
) -> pd.DataFrame:
    table = pd.read_csv(table_path)
    if sorted(table.columns) != sorted(names):
        raise RuntimeError("GSD 输出列集合与 schema 不一致")
    table = table[names].astype(int)
    if not np.isin(table.to_numpy(), (0, 1)).all():
        raise RuntimeError("GSD 输出存在非 0/1 取值")
    if len(table) != N_RECORDS:
        raise RuntimeError(
            f"GSD 输出行数漂移: expected={N_RECORDS}, "
            f"observed={len(table)}"
        )
    return table


def _family_snapshot(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        statistic: float(node[f"normalized_l1_{statistic}"])
        for statistic in ("mean", "median", "p90", "max")
    }


def _load_references(root: Path) -> dict[str, Any]:
    engine_sha = all2way._sha256_file(root / ENGINE_REPORT_PATH)
    if engine_sha != ENGINE_REPORT_SHA256:
        raise RuntimeError(
            "引擎决胜局报告 SHA 漂移: "
            f"expected={ENGINE_REPORT_SHA256}, observed={engine_sha}"
        )
    pgm_sha = all2way._sha256_file(root / PGM_ALL2WAY_REPORT_PATH)
    if pgm_sha != PGM_ALL2WAY_REPORT_SHA256:
        raise RuntimeError(
            "PGM-all2way 报告 SHA 漂移: "
            f"expected={PGM_ALL2WAY_REPORT_SHA256}, observed={pgm_sha}"
        )
    engine_report = all2way._load_json_object(root / ENGINE_REPORT_PATH)
    engine_quality = engine_report["quality"]["residual"]
    pgm_report = all2way._load_json_object(
        root / PGM_ALL2WAY_REPORT_PATH
    )
    pgm_quality = pgm_report["quality"]["pgm"]
    if (
        pgm_report["comparison"]["measured_anchors_not_compared"][
            "pgm_all2way_on_480_family"
        ]["mean"]
        != pgm_quality["measured"]["normalized_l1_mean"]
    ):
        raise RuntimeError("PGM-all2way 报告内部 measured 快照不自洽")
    return {
        "engine_all2way_pool": {
            "path": str(ENGINE_REPORT_PATH),
            "sha256": engine_sha,
            "snapshot": pgm980._arm_snapshot(engine_quality),
            "family_480_direct": _family_snapshot(
                engine_quality["all2way_pool"]
            ),
        },
        "pgm_all2way": {
            "path": str(PGM_ALL2WAY_REPORT_PATH),
            "sha256": pgm_sha,
            "snapshot": pgm980._arm_snapshot(pgm_quality),
            "family_480_direct": pgm980._arm_snapshot(pgm_quality)[
                "measured"
            ],
        },
    }


def _comparison(
    gsd_quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    gsd = pgm980._arm_snapshot(gsd_quality)
    fair: dict[str, Any] = {}
    for name in FAIR_COMPARABLE_GROUPS:
        entry: dict[str, Any] = {"gsd_all2way": gsd[name]}
        for ref_name, reference in references.items():
            snapshot = reference["snapshot"][name]
            entry[ref_name] = snapshot
            entry[f"gsd_all2way_minus_{ref_name}"] = float(
                gsd[name]["mean"] - snapshot["mean"]
            )
        fair[name] = entry
    family = {
        "gsd_all2way_on_480_family": gsd["measured"],
        "engine_pool_on_480_family": references[
            "engine_all2way_pool"
        ]["family_480_direct"],
        "pgm_all2way_on_480_family": references["pgm_all2way"][
            "family_480_direct"
        ],
        "gsd_minus_engine": float(
            gsd["measured"]["mean"]
            - references["engine_all2way_pool"]["family_480_direct"][
                "mean"
            ]
        ),
        "gsd_minus_pgm": float(
            gsd["measured"]["mean"]
            - references["pgm_all2way"]["family_480_direct"]["mean"]
        ),
        "note": (
            "all three arms consumed the same 480-cell family; "
            "in-family fit comparison is apples-to-apples"
        ),
    }
    return {
        "normalized_l1_mean_deltas_fair_groups_only": fair,
        "family_level_480_direct": family,
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "gsd all2way baseline protocol SHA-256 确认值不一致"
        )
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = all2way._audit_generation_inputs(root)
    environment = _audit_gsd_environment(root)
    references = _load_references(root)
    _, names, cards = pgm980._schema_domain(root)
    _, one_way_audit, safety = pgm980._build_one_way_measurements(
        root, names, cards
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        table_path, manifest, wall_seconds = (
            _run_generation_subprocess(root, staging)
        )
        manifest_audit = _audit_generation_manifest(manifest)
        table = _load_gsd_table(table_path, names)
        table_sha256 = base._frame_sha256(table)
        heldout_queries, heldout_targets, reference = (
            pgm980._load_heldout_and_reference(root)
        )
        quality = pgm980._evaluate_pgm_table(
            root,
            staging,
            table,
            audit,
            safety,
            heldout_queries,
            heldout_targets,
            reference,
        )
        # 复用评估器默认写旧考卷常数与 pgm.json 文件名；本协议
        # measured=all2way 考卷、方法名=gsd，修正后重写落位。
        quality["evaluation_inputs"]["measured_sha256"] = ALL2WAY_SHA256
        quality["evaluation_inputs"]["measured_workload"] = str(
            ALL2WAY_PATH
        )
        (staging / "quality" / "pgm.json").unlink()
        base._write_json(staging / "quality" / "gsd.json", quality)
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "generation_inputs": {
                "query_count": int(audit["query_count"]),
                "query_identity_sha256": audit["query_identity_sha256"],
                "target_vector_sha256": audit["target_vector_sha256"],
                "input_sha256": dict(audit["input_sha256"]),
            },
            "environment_audit": environment,
            "one_way_measurement_audit": one_way_audit,
            "generation": {
                "subprocess_wall_seconds": wall_seconds,
                "manifest_audit": manifest_audit,
                "table_sha256": table_sha256,
                "row_count": int(len(table)),
            },
            "quality": {"gsd": quality},
            "references": references,
            "comparison": _comparison(quality, references),
            "completed_at_unix": time.time(),
        }
        base._write_json(staging / "report.json", report)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="打印协议 SHA-256")
    sub.add_parser("plan", help="打印协议 manifest")
    run_parser = sub.add_parser("run", help="执行 GSD 对照局")
    run_parser.add_argument(
        "--confirm-protocol-sha256",
        required=True,
        help="逐字确认冻结协议 SHA-256",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "freeze":
        print(protocol_sha256())
        return
    if args.command == "plan":
        print(
            json.dumps(build_plan(), indent=2, sort_keys=True)
        )
        return
    destination = run(args.confirm_protocol_sha256)
    print(f"完成：{destination / 'report.json'}")


if __name__ == "__main__":
    main()
