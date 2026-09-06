#!/usr/bin/env python3
"""Feasibility receipt: Private-PGM cannot consume the plants all-2way family.

The engine-side plants runs now consume the full all-2way family (2346
pairs, 9384 cells).  A fair PGM refit would hand Private-PGM the same
family -- but the graphical model must connect every measured clique, and
the complete pair graph over 69 binary attributes triangulates into a
single 69-attribute clique with 2^69 cells (~4.5 exabytes), structurally
over any memory cap.  This protocol runs ONLY the junction-tree
feasibility precheck (the same ``hypothetical_model_size`` gate the real
plants PGM baseline uses), pins every input, and writes the audited
receipt: ``baseline_infeasible``, no estimation attempted, no synthetic
table produced.

Contrast receipts pinned inside the report: the 980-exam plants PGM
baseline WAS feasible (375 measured cliques fit under the 4096 MB cap),
and the nltcs all-2way refit IS feasible (single 2^16 clique, 0.5 MB) --
the blowup is specific to feeding the full family on a 69-attribute
domain, which is exactly why measurement-selection shells (AIM) exist.
The zeroth-order engine consumes the same family directly because
queries are fitness masks, not graph edges.

Diagnostic only: no noise, no tuning, no formal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from mbi.junction_tree import (  # noqa: E402
    hypothetical_model_size,
    make_junction_tree,
    maximal_cliques,
)

from scripts import run_baseline_pgm_plants_diagnostic as pgm980
from scripts import run_fitness_only_attribution as base
from table_diffevo.quality import query_fingerprint


PROTOCOL_VERSION = "baseline-pgm-all2way-plants-feasibility-v1"
OUTPUT_DIR = Path("outputs/baseline_pgm_all2way_plants_feasibility_v1")
FROZEN_PROTOCOL_SHA256 = (
    "89f5ae06efec7f767f6829f7e82efeaec04570d61e89d851996b8bea9ab6a30e"
)

DATASET = "plants"
N_RECORDS = 17412
N_ATTRIBUTES = 69
FEASIBILITY_CAP_MB = pgm980.FEASIBILITY_CAP_MB

EXPECTED_CELL_QUERY_COUNT = 9384
EXPECTED_PAIR_CLIQUE_COUNT = 2346
EXPECTED_CELLS_PER_CLIQUE = 4

SCHEMA_PATH = pgm980.SCHEMA_PATH
MARGINALS_PATH = pgm980.MARGINALS_PATH
ALL2WAY_PATH = Path("configs/plants/all2way_issue53_v1.json")
ALL2WAY_SHA256 = (
    "9dc37994e912ce52c5859a122150144cad487c06d71de6bb0f8bd8a4a584409a"
)

OLD_PGM_REPORT_PATH = Path("outputs/baseline_pgm_plants_v1/report.json")
OLD_PGM_REPORT_SHA256 = (
    "fbe0c88d793fde6ed3e6b919933b7bedf2d239cb6ea606f3865eadad9f026105"
)


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": (
            "pgm_all2way_infeasibility_receipt_no_estimation_"
            "diagnostic_only"
        ),
        "motivation": (
            "the fair information-parity PGM refit on plants would feed"
            " the full all-2way family; the graphical model must connect"
            " every measured clique, and the complete pair graph over 69"
            " attributes triangulates into one 2^69-cell clique --"
            " structurally over any memory cap; this receipt pins that"
            " fact with the same precheck gate the real plants PGM"
            " baseline used, so comparison tables can cite an audited"
            " number instead of an assertion"
        ),
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "n_attributes": N_ATTRIBUTES,
        "precheck": {
            "method": "junction_tree.hypothetical_model_size",
            "cap_mb": FEASIBILITY_CAP_MB,
            "cap_source": "baseline-pgm-plants-v1 protocol constant",
            "clique_set": (
                "all 2346 attribute pairs (from the frozen all-2way exam)"
                " plus the 69 one-way marginal cliques (the full parity"
                " diet); one-ways cannot reduce the treewidth"
            ),
            "expected_verdict": "infeasible",
            "estimation_attempted": False,
            "synthetic_table_produced": False,
        },
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "measured": str(ALL2WAY_PATH),
            "expected_cell_query_count": EXPECTED_CELL_QUERY_COUNT,
            "expected_pair_clique_count": EXPECTED_PAIR_CLIQUE_COUNT,
            "expected_cells_per_clique": EXPECTED_CELLS_PER_CLIQUE,
            "input_sha256": {
                "schema": pgm980.INPUT_SHA256["schema"],
                "marginals": pgm980.INPUT_SHA256["marginals"],
                "measured": ALL2WAY_SHA256,
            },
        },
        "contrast_receipts": {
            "old_pgm_980_exam_was_feasible": {
                "path": str(OLD_PGM_REPORT_PATH),
                "sha256": OLD_PGM_REPORT_SHA256,
                "note": (
                    "375 measured cliques from the sparse 980 exam fit"
                    " under the same 4096 MB cap and produced the"
                    " baseline_pgm_plants_v1 table"
                ),
            },
            "nltcs_all2way_refit_is_feasible": (
                "the same construction on 16 attributes collapses to one"
                " 2^16 clique (~0.5 MB); see the"
                " baseline-pgm-all2way-nltcs-v1 protocol"
            ),
            "engine_consumed_the_same_family": (
                "fitness_only_all2way_pool_plants_seed9908_v1: 9384-cell"
                " family in-pool, early-stopped 3506/27000, 49 minutes --"
                " queries are fitness masks, not graph edges"
            ),
        },
        "failure_policy": {
            "any_input_sha_drift": "fail_closed_abort",
            "precheck_unexpectedly_feasible": (
                "fail_closed_abort_receipt_claim_would_be_wrong"
            ),
        },
        "interpretation": {
            "diagnostic_only": True,
            "formal_claim_allowed": False,
            "question": (
                "how large is the junction tree Private-PGM would need to"
                " consume the plants all-2way family, and is it under any"
                " realistic memory cap?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "pgm all2way plants feasibility 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": _protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "generation_started": False,
    }


def _audit_inputs(root: Path) -> dict[str, Any]:
    observed = {
        "schema": _sha256_file(root / SCHEMA_PATH),
        "marginals": _sha256_file(root / MARGINALS_PATH),
        "measured": _sha256_file(root / ALL2WAY_PATH),
    }
    expected = {
        "schema": pgm980.INPUT_SHA256["schema"],
        "marginals": pgm980.INPUT_SHA256["marginals"],
        "measured": ALL2WAY_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"输入 SHA 漂移: expected={expected}, observed={observed}"
        )
    payload = _load_json_object(root / ALL2WAY_PATH)
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
    pair_cliques: set[tuple[str, ...]] = set()
    cells_per_pair: dict[tuple[str, ...], int] = {}
    for index, query in enumerate(queries):
        conditions = query.get("conditions", [])
        if query.get("type") != "double" or len(conditions) != 2:
            raise RuntimeError(f"all2way 第 {index} 条不是 double")
        clique = tuple(
            sorted(condition["attribute"] for condition in conditions)
        )
        if len(set(clique)) != 2:
            raise RuntimeError(f"all2way 第 {index} 条属性重复")
        pair_cliques.add(clique)
        cells_per_pair[clique] = cells_per_pair.get(clique, 0) + 1
    if len(pair_cliques) != EXPECTED_PAIR_CLIQUE_COUNT:
        raise RuntimeError(
            f"属性对数量漂移: {len(pair_cliques)}"
        )
    if any(
        count != EXPECTED_CELLS_PER_CLIQUE
        for count in cells_per_pair.values()
    ):
        raise RuntimeError("存在不完整 2x2 列联表分组")
    return {
        "query_count": len(queries),
        "pair_cliques": sorted(pair_cliques),
        "query_identity_sha256": _sha256_bytes(
            "\n".join(fingerprints).encode("ascii")
        ),
        "input_sha256": observed,
    }


def _precheck(root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    domain, names, _cards = pgm980._schema_domain(root)
    cliques = [tuple(clique) for clique in audit["pair_cliques"]] + [
        (name,) for name in names
    ]
    started = time.perf_counter()
    size_mb = float(hypothetical_model_size(domain, cliques))
    tree, _ = make_junction_tree(domain, cliques)
    tree_cliques = maximal_cliques(tree)
    biggest = max(tree_cliques, key=lambda clique: domain.size(clique))
    elapsed = time.perf_counter() - started
    return {
        "clique_count_checked": len(cliques),
        "pair_clique_count": len(audit["pair_cliques"]),
        "one_way_clique_count": len(names),
        "model_size_mb": size_mb,
        "model_size_exabytes_observation_only": size_mb / (1024.0**4),
        "cap_mb": FEASIBILITY_CAP_MB,
        "over_cap_factor": size_mb / FEASIBILITY_CAP_MB,
        "maximal_clique_count": len(tree_cliques),
        "max_clique_attribute_count": len(biggest),
        "max_clique_cells": int(domain.size(biggest)),
        "feasible": size_mb <= FEASIBILITY_CAP_MB,
        "precheck_seconds": elapsed,
    }


def _verify_contrast_receipt(root: Path) -> dict[str, Any]:
    observed = _sha256_file(root / OLD_PGM_REPORT_PATH)
    if observed != OLD_PGM_REPORT_SHA256:
        raise RuntimeError(
            f"旧 PGM 报告 SHA 漂移: expected={OLD_PGM_REPORT_SHA256}, "
            f"observed={observed}"
        )
    report = _load_json_object(root / OLD_PGM_REPORT_PATH)
    old_feasibility = report["feasibility"]
    if old_feasibility.get("feasible") is not True:
        raise RuntimeError("旧 980 局理应可行，对照收据不成立")
    return {
        "path": str(OLD_PGM_REPORT_PATH),
        "sha256": observed,
        "old_exam_feasibility": old_feasibility,
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "pgm all2way plants feasibility protocol SHA-256 确认值不一致"
        )
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = _audit_inputs(root)
    precheck = _precheck(root, audit)
    if precheck["feasible"]:
        raise RuntimeError(
            "预检意外可行——收据主张不成立，fail-closed 拒绝写盘："
            f"{precheck}"
        )
    contrast = _verify_contrast_receipt(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "generation_inputs": {
                "query_count": int(audit["query_count"]),
                "query_identity_sha256": audit["query_identity_sha256"],
                "input_sha256": dict(audit["input_sha256"]),
            },
            "feasibility_precheck": precheck,
            "verdict": "baseline_infeasible_no_estimation_attempted",
            "contrast_receipt_old_pgm_980": contrast,
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
    parser = argparse.ArgumentParser(
        description=(
            "Feasibility receipt: PGM cannot consume the plants all-2way"
            " family"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the frozen read-only plan")
    runner = subparsers.add_parser("run", help="write the receipt")
    runner.add_argument(
        "--confirm-protocol-sha256",
        required=True,
        help="must equal the frozen protocol SHA-256",
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    destination = run(arguments.confirm_protocol_sha256)
    print(f"完成：{destination}")


if __name__ == "__main__":
    main()
