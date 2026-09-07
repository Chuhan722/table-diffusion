"""无噪声 GSD（private_gsd 遗传搜索）在 plants all-2way 同餐上的对照局。

诊断性质（diagnostic_only）：
- 能力面对决：plants all-2way 全家族（9384 个 2x2 二维格 + 138 个一维格
  = 9522 频率，零噪声 rho=inf）是 PGM 结构性吃不下的一顿饭（2^69 全联合
  格连接树，收据 baseline_pgm_all2way_plants_feasibility_v1），引擎决胜局
  （fitness_only_all2way_pool_plants_seed9908_v1）照吃。本局考 GSD 官方
  内核能不能吃、吃多快、吃完外推如何——与引擎完全同餐，与 PGM 的对照是
  "出席 vs 缺席"（收据引用，不做数值 delta）。
- 生成在官方仓库独立 venv 内 subprocess 执行（upstream commit 与两处守卫
  补丁后的文件 SHA 全部钉死，fail-closed）；评估全部回到主仓库冻结考卷与
  主评估器（与引擎对照完全同卷同口径）。
- 训练参数照抄官方 GSDSynthesizer.fit 默认（5000 万代上限 + 每 N'=17412
  代早停检查、阈值 1e-4、全量四遗传操作符、sparse 统计、N_prime=N），
  零调参。
- 不做任何晋级判断；数字只进对照表。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts import run_baseline_pgm_plants_diagnostic as pgm_plants
from scripts import run_fitness_only_attribution as base
from table_diffevo.quality import query_fingerprint

PROTOCOL_VERSION = "baseline-gsd-all2way-plants-v1"
OUTPUT_DIR = Path("outputs/baseline_gsd_all2way_plants_v1")
FROZEN_PROTOCOL_SHA256 = (
    "46c1459a9271cd4f0dbfdb985e6277a25c64df59bf65ad8681e0b42a36c2333f"
)

DATASET = "plants"
N_RECORDS = 17412
N_ATTRIBUTES = 69
SEED = 9908
NUM_GENERATIONS_CAP = 50_000_000
EARLY_STOP_THRESHOLD = 0.0001
EXPECTED_PAIR_CLIQUE_COUNT = N_ATTRIBUTES * (N_ATTRIBUTES - 1) // 2
EXPECTED_CELL_QUERY_COUNT = EXPECTED_PAIR_CLIQUE_COUNT * 4  # 9384
EXPECTED_CELLS_PER_CLIQUE = 4
EXPECTED_STAT_COUNT = (
    EXPECTED_CELL_QUERY_COUNT + 2 * N_ATTRIBUTES
)  # 9522

ALL2WAY_PATH = Path("configs/plants/all2way_issue53_v1.json")
ALL2WAY_SHA256 = (
    "9dc37994e912ce52c5859a122150144cad487c06d71de6bb0f8bd8a4a584409a"
)

GSD_REPO = Path("/home/chuhan/projects/private_gsd")
GSD_PYTHON = GSD_REPO / ".venv/bin/python"
GSD_UPSTREAM_COMMIT = "f6150d7821b9675ce9158b456f1a5cff8bc3b3d6"
GSD_PATCHED_FILES_SHA256 = {
    "src/genetic_sd/__init__.py": (
        "6546def7d7424a19b13e0783d0e9df270467329ee21d6a9b7029bbe66ff54183"
    ),
    "src/genetic_sd/generator/generator_base.py": (
        "e3af165ac77438aa6cd9876cd67039a0326c8cdf64c17234e80d0529359724b1"
    ),
}
GENERATION_SCRIPT_PATH = Path(
    "scripts/gsd_generate_all2way_noisefree_plants.py"
)
GENERATION_SCRIPT_SHA256 = (
    "304beeee6be16082af164d13b10a61d1c063b36b035c7c139bdc67f61e692fba"
)

ENGINE_REPORT_PATH = Path(
    "outputs/fitness_only_all2way_pool_plants_seed9908_v1/report.json"
)
ENGINE_REPORT_SHA256 = (
    "82d85b8516a6948563f0efd6271c024c96c1b8c5957b13bc7477834461bc6b33"
)
PGM_RECEIPT_PATH = Path(
    "outputs/baseline_pgm_all2way_plants_feasibility_v1/report.json"
)
PGM_RECEIPT_SHA256 = (
    "9e2a5b0d34a2e6255f91e77be569909d5eec19aef0ba654617f587408c893695"
)
PGM_RECEIPT_VERDICT = "baseline_infeasible_no_estimation_attempted"

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
                "one_way_workloads": N_ATTRIBUTES,
                "two_way_workloads": EXPECTED_PAIR_CLIQUE_COUNT,
                "statistic_count": EXPECTED_STAT_COUNT,
                "same_meal_note": (
                    "matches engine 9522-cell pool (9384 doubles + 138 "
                    "one-way cells); pgm is structurally absent from "
                    "this meal with a pinned feasibility receipt"
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
            "data_csv": pgm_plants.INPUT_SHA256["reference"],
            "measured_exam": ALL2WAY_SHA256,
            "schema": pgm_plants.INPUT_SHA256["schema"],
            "marginals": pgm_plants.INPUT_SHA256["marginals"],
        },
        "evaluation": {
            "engine": "main_repo_frozen_evaluator",
            "measured_workload": str(ALL2WAY_PATH),
            "heldout_sha256": pgm_plants.INPUT_SHA256["heldout"],
            "reference_sha256": pgm_plants.INPUT_SHA256["reference"],
            "note": (
                "same frozen exams and evaluator as the engine all2way "
                "pool arm; pgm absent by feasibility receipt"
            ),
        },
        "references": {
            "engine_all2way_pool": {
                "path": str(ENGINE_REPORT_PATH),
                "sha256": ENGINE_REPORT_SHA256,
            },
            "pgm_feasibility_receipt": {
                "path": str(PGM_RECEIPT_PATH),
                "sha256": PGM_RECEIPT_SHA256,
                "verdict": PGM_RECEIPT_VERDICT,
            },
        },
        "fair_comparable_groups": list(FAIR_COMPARABLE_GROUPS),
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def protocol_sha256() -> str:
    return pgm_plants._sha256_bytes(
        pgm_plants._strict_json_bytes(_protocol_manifest())
    )


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "gsd all2way plants baseline protocol SHA-256 漂移: "
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "protocol_sha256": protocol_sha256(),
        "protocol": _protocol_manifest(),
        "launch_gate": "requires --confirm-protocol-sha256",
    }


def _audit_generation_inputs(root: Path) -> dict[str, Any]:
    observed = {
        "schema": pgm_plants._sha256_file(
            root / pgm_plants.SCHEMA_PATH
        ),
        "marginals": pgm_plants._sha256_file(
            root / pgm_plants.MARGINALS_PATH
        ),
        "measured": pgm_plants._sha256_file(root / ALL2WAY_PATH),
    }
    expected = {
        "schema": pgm_plants.INPUT_SHA256["schema"],
        "marginals": pgm_plants.INPUT_SHA256["marginals"],
        "measured": ALL2WAY_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"generation 输入 SHA 漂移: expected={expected}, "
            f"observed={observed}"
        )
    payload = pgm_plants._load_json_object(root / ALL2WAY_PATH)
    queries = payload.get("queries")
    if (
        not isinstance(queries, list)
        or len(queries) != EXPECTED_CELL_QUERY_COUNT
    ):
        raise RuntimeError("plants all2way 查询数量漂移")
    if int(payload.get("record_count", -1)) != N_RECORDS:
        raise RuntimeError("plants all2way record_count 漂移")
    fingerprints = [query_fingerprint(query) for query in queries]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("plants all2way 查询存在重复语义")
    clique_cells: dict[tuple[str, ...], int] = {}
    clique_sums: dict[tuple[str, ...], float] = {}
    targets = []
    for index, query in enumerate(queries):
        conditions = query.get("conditions", [])
        if query.get("type") != "double" or len(conditions) != 2:
            raise RuntimeError(f"all2way 第 {index} 条不是 double")
        clique = tuple(
            sorted(condition["attribute"] for condition in conditions)
        )
        if len(set(clique)) != 2:
            raise RuntimeError(f"all2way 第 {index} 条属性重复")
        result = query.get("result")
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, float))
            or not math.isfinite(float(result))
            or result < 0
        ):
            raise ValueError(f"all2way target 非法: index={index}")
        targets.append(float(result))
        clique_cells[clique] = clique_cells.get(clique, 0) + 1
        clique_sums[clique] = (
            clique_sums.get(clique, 0.0) + float(result)
        )
    if len(clique_cells) != EXPECTED_PAIR_CLIQUE_COUNT:
        raise RuntimeError("all2way 属性对分组数量漂移")
    if any(
        count != EXPECTED_CELLS_PER_CLIQUE
        for count in clique_cells.values()
    ):
        raise RuntimeError("all2way 存在不完整 2x2 列联表分组")
    if any(
        int(total) != N_RECORDS for total in clique_sums.values()
    ):
        raise RuntimeError("all2way 分组四格计数之和不等于 N")
    return {
        "queries": queries,
        "targets": targets,
        "query_count": len(queries),
        "query_identity_sha256": pgm_plants._sha256_bytes(
            "\n".join(fingerprints).encode("ascii")
        ),
        "target_vector_sha256": pgm_plants._sha256_bytes(
            pgm_plants._strict_json_bytes(targets)
        ),
        "input_sha256": observed,
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
        rel: pgm_plants._sha256_file(GSD_REPO / rel)
        for rel in GSD_PATCHED_FILES_SHA256
    }
    if observed_files != GSD_PATCHED_FILES_SHA256:
        raise RuntimeError(
            "GSD 补丁文件 SHA 漂移: "
            f"expected={GSD_PATCHED_FILES_SHA256}, "
            f"observed={observed_files}"
        )
    observed_script = pgm_plants._sha256_file(
        root / GENERATION_SCRIPT_PATH
    )
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
        str(root / pgm_plants.REFERENCE_PATH),
        "--expected-data-sha256",
        pgm_plants.INPUT_SHA256["reference"],
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
    manifest = pgm_plants._load_json_object(manifest_path)
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
        != pgm_plants.INPUT_SHA256["reference"]
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
    engine_sha = pgm_plants._sha256_file(root / ENGINE_REPORT_PATH)
    if engine_sha != ENGINE_REPORT_SHA256:
        raise RuntimeError(
            "引擎 plants all2way 局报告 SHA 漂移: "
            f"expected={ENGINE_REPORT_SHA256}, observed={engine_sha}"
        )
    receipt_sha = pgm_plants._sha256_file(root / PGM_RECEIPT_PATH)
    if receipt_sha != PGM_RECEIPT_SHA256:
        raise RuntimeError(
            "PGM 不可行收据 SHA 漂移: "
            f"expected={PGM_RECEIPT_SHA256}, observed={receipt_sha}"
        )
    engine_report = pgm_plants._load_json_object(
        root / ENGINE_REPORT_PATH
    )
    engine_quality = engine_report["quality"]["residual"]
    receipt = pgm_plants._load_json_object(root / PGM_RECEIPT_PATH)
    receipt_verdict = receipt.get("verdict") or receipt.get(
        "feasibility", {}
    ).get("verdict")
    if receipt_verdict != PGM_RECEIPT_VERDICT:
        raise RuntimeError(
            "PGM 收据 verdict 漂移: "
            f"expected={PGM_RECEIPT_VERDICT}, observed={receipt_verdict}"
        )
    return {
        "engine_all2way_pool": {
            "path": str(ENGINE_REPORT_PATH),
            "sha256": engine_sha,
            "snapshot": pgm_plants._arm_snapshot(engine_quality),
            "family_9384_direct": _family_snapshot(
                engine_quality["all2way_pool"]
            ),
        },
        "pgm_feasibility_receipt": {
            "path": str(PGM_RECEIPT_PATH),
            "sha256": receipt_sha,
            "verdict": receipt_verdict,
            "note": (
                "pgm structurally absent from this meal; receipt "
                "pinned, no numeric deltas"
            ),
        },
    }


def _comparison(
    gsd_quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    gsd = pgm_plants._arm_snapshot(gsd_quality)
    engine = references["engine_all2way_pool"]
    fair: dict[str, Any] = {}
    for name in FAIR_COMPARABLE_GROUPS:
        snapshot = engine["snapshot"][name]
        fair[name] = {
            "gsd_all2way": gsd[name],
            "engine_all2way_pool": snapshot,
            "gsd_all2way_minus_engine_all2way_pool": float(
                gsd[name]["mean"] - snapshot["mean"]
            ),
        }
    family = {
        "gsd_all2way_on_9384_family": gsd["measured"],
        "engine_pool_on_9384_family": engine["family_9384_direct"],
        "gsd_minus_engine": float(
            gsd["measured"]["mean"]
            - engine["family_9384_direct"]["mean"]
        ),
        "note": (
            "both arms consumed the same 9384-cell family; in-family "
            "fit comparison is apples-to-apples"
        ),
    }
    return {
        "normalized_l1_mean_deltas_fair_groups_only": fair,
        "family_level_9384_direct": family,
        "pgm_feasibility_receipt": dict(
            references["pgm_feasibility_receipt"]
        ),
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "gsd all2way plants baseline protocol SHA-256 确认值不一致"
        )
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = _audit_generation_inputs(root)
    environment = _audit_gsd_environment(root)
    references = _load_references(root)
    _, names, cards = pgm_plants._schema_domain(root)
    _, one_way_audit, safety = pgm_plants._build_one_way_measurements(
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
            pgm_plants._load_heldout_and_reference(root)
        )
        quality = pgm_plants._evaluate_pgm_table(
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
    run_parser = sub.add_parser("run", help="执行 GSD plants 对照局")
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
