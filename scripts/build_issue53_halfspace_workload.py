#!/usr/bin/env python3
"""Build the deterministic halfspace (non-differentiable) workload for Issue #53.

对应设计稿：docs/设计/半空间不可微查询能力线设计稿.md 第 2 节。

半空间查询 q(w, θ) = #{行 x : w·x ≥ θ} 是清单里唯一不可微（硬阈值）的
查询类型；本脚本为每个已注册数据集（plants / nltcs）构造两档题并冻结：

- A 档 row-sum（权重全 1）：θ 取真实行重量谱的全部非退化格点（两端各留
  0.5% 边距），按 θ 升序交替分配 measured / heldout（两组都覆盖全谱）；
- B 档一般半空间：固定种子生成，权重在 k 列属性子集上取 ±1（k 按数据集
  规格：plants 16、nltcs 8），θ 取该投影分布的分位点（分位数从 [0.2, 0.8]
  均匀抽取），按生成序交替分配。

【与 heldout 构造器的边界差异，明示不藏】本工作负载的查询选择依赖源表
（谱边界与分位点），不是 result-blind 构造：这是诊断线（diagnostic_only、
无噪声内层）专用的能力测试考卷。若未来进入正式 DP 管道，选择规则必须换成
只依赖公共信息的版本（例如由公开 marginals 推导投影界）。

输出每数据集单文件（configs/<dataset>/halfspace_issue53_v1.json），每条 query
携带 role=measured/heldout 与 tier=rowsum/general 标签，runner 按 role 过滤。
支持 --verify-existing 确定性重建复核。
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from table_diffevo.quality import query_fingerprint, validate_query_partition
from table_diffevo.queries import evaluate_table, load_data
from table_diffevo.schema import load_schema


NAMESPACE = "issue53-halfspace-v1"

# A 档：非退化边距（θ 保留的计数须落在 [MARGIN*N, (1-MARGIN)*N]）
ROWSUM_MARGIN = 0.005

# B 档：固定生成参数（k 为数据集相关：plants 16/69≈23% 稀疏；
# nltcs 全表仅 16 列，k=16 会退化为稠密全列，取 k=8=半数列保持稀疏形态，
# k=4 时 ±1 投影只有 9 个档位、θ 谱过粗，故不取比例缩放）
GENERAL_SEED = 20260905
GENERAL_COUNT = 64          # 生成总数，交替分配 → 32 measured + 32 heldout
GENERAL_QUANTILE_RANGE = (0.2, 0.8)
GENERAL_MARGIN = 0.005
GENERAL_MAX_ATTEMPTS = 1000

DATASETS = {
    "plants": {
        "schema": Path("configs/plants/schema.yaml"),
        "source": Path("data/plants/plants.csv"),
        "output": Path("configs/plants/halfspace_issue53_v1.json"),
        "general_k": 16,
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "source": Path("data/nltcs/nltcs.csv"),
        "output": Path("configs/nltcs/halfspace_issue53_v1.json"),
        "general_k": 8,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _halfspace_query(
    attributes: List[str],
    weights: List[int],
    theta: int,
    *,
    tier: str,
    role: str,
    rank: int,
    expression: str,
) -> Dict[str, Any]:
    tier_code = {"rowsum": "A", "general": "B"}[tier]
    role_code = {"measured": "M", "heldout": "H"}[role]
    return {
        "id": f"HS{tier_code}_{role_code}_{rank:04d}",
        "type": "halfspace",
        "tier": tier,
        "role": role,
        "expression": expression,
        "halfspace": {
            "attributes": list(attributes),
            "weights": [int(w) for w in weights],
            "theta": int(theta),
        },
    }


def build_rowsum_tier(source_matrix: np.ndarray, columns: List[str]) -> List[Dict[str, Any]]:
    """A 档：全谱非退化 θ，升序交替分配（偶位 measured / 奇位 heldout）。"""
    n_records = source_matrix.shape[0]
    rowsum = source_matrix.sum(axis=1)
    lower = ROWSUM_MARGIN * n_records
    upper = (1.0 - ROWSUM_MARGIN) * n_records
    thetas = [
        theta
        for theta in range(int(rowsum.min()), int(rowsum.max()) + 2)
        if lower <= int((rowsum >= theta).sum()) <= upper
    ]
    if not thetas:
        raise ValueError("row-sum 谱上没有非退化 θ，检查数据")

    queries = []
    ranks = {"measured": 0, "heldout": 0}
    for index, theta in enumerate(thetas):
        role = "measured" if index % 2 == 0 else "heldout"
        ranks[role] += 1
        queries.append(_halfspace_query(
            columns,
            [1] * len(columns),
            theta,
            tier="rowsum",
            role=role,
            rank=ranks[role],
            expression=f"rowsum >= {theta}",
        ))
    return queries


def build_general_tier(
    source_matrix: np.ndarray, columns: List[str], general_k: int
) -> List[Dict[str, Any]]:
    """B 档：固定种子生成 k 列稀疏 ±1 半空间，θ 取投影分位点，交替分配。"""
    n_records = source_matrix.shape[0]
    rng = np.random.default_rng(GENERAL_SEED)
    lower = GENERAL_MARGIN * n_records
    upper = (1.0 - GENERAL_MARGIN) * n_records
    q_lo, q_hi = GENERAL_QUANTILE_RANGE

    queries: List[Dict[str, Any]] = []
    seen_fingerprints = set()
    ranks = {"measured": 0, "heldout": 0}
    attempts = 0
    while len(queries) < GENERAL_COUNT:
        attempts += 1
        if attempts > GENERAL_MAX_ATTEMPTS:
            raise RuntimeError(
                f"B 档在 {GENERAL_MAX_ATTEMPTS} 次尝试内未能生成 "
                f"{GENERAL_COUNT} 条非退化查询"
            )
        attr_idx = np.sort(rng.choice(len(columns), size=general_k, replace=False))
        weights = rng.choice([-1, 1], size=general_k)
        quantile = float(rng.uniform(q_lo, q_hi))
        projection = source_matrix[:, attr_idx] @ weights
        theta = int(round(float(np.quantile(projection, quantile))))
        count = int((projection >= theta).sum())
        if not (lower <= count <= upper):
            continue  # 退化：重抽（种子流前进，仍然确定性）

        attrs = [columns[i] for i in attr_idx]
        candidate = {
            "type": "halfspace",
            "halfspace": {
                "attributes": attrs,
                "weights": [int(w) for w in weights],
                "theta": theta,
            },
        }
        fingerprint = query_fingerprint(candidate)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)

        index = len(queries)
        role = "measured" if index % 2 == 0 else "heldout"
        ranks[role] += 1
        queries.append(_halfspace_query(
            attrs,
            [int(w) for w in weights],
            theta,
            tier="general",
            role=role,
            rank=ranks[role],
            expression=f"halfspace(k={general_k}) >= {theta}",
        ))
    return queries


def _identity(queries: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(query["fingerprint_sha256"] for query in queries)
        .encode("ascii")
    ).hexdigest()


def build_dataset_workload(dataset: str, specification: Dict[str, Any]) -> Dict[str, Any]:
    schema = load_schema(str(specification["schema"]))
    source = load_data(str(specification["source"]))
    if list(source.columns) != schema.attribute_names():
        raise ValueError(f"{dataset} source 列与 schema 不一致")
    columns = list(source.columns)
    source_matrix = source.to_numpy(dtype=np.int64)
    if not np.isin(source_matrix, (0, 1)).all():
        raise ValueError(f"{dataset} 不是二值表，halfspace 构造规则假设 0/1 数据")

    queries = build_rowsum_tier(source_matrix, columns)
    queries.extend(
        build_general_tier(source_matrix, columns, int(specification["general_k"]))
    )

    # 贴指纹与精确答案（评价走 queries.eval_halfspace_mask 分派）
    answers = evaluate_table(source, queries)
    for query, answer in zip(queries, answers):
        query["fingerprint_sha256"] = query_fingerprint(query)
        query["result"] = int(answer)

    measured = [q for q in queries if q["role"] == "measured"]
    heldout = [q for q in queries if q["role"] == "heldout"]
    partition = validate_query_partition(measured, heldout)

    rowsum_queries = [q for q in queries if q["tier"] == "rowsum"]
    general_queries = [q for q in queries if q["tier"] == "general"]
    return {
        "dataset": dataset,
        "record_count": int(len(source)),
        "query_count": len(queries),
        "result_unit": "records",
        "description": (
            "Issue #53 halfspace (non-differentiable) capability workload; "
            "diagnostic-only exact answers. Selection reads the source "
            "projection spectrum (NOT result-blind); a public-information "
            "selection rule is required before any DP-pipeline use."
        ),
        "construction": {
            "namespace": NAMESPACE,
            "tiers": {
                "rowsum": {
                    "margin": ROWSUM_MARGIN,
                    "count": len(rowsum_queries),
                    "theta_min": min(
                        q["halfspace"]["theta"] for q in rowsum_queries
                    ),
                    "theta_max": max(
                        q["halfspace"]["theta"] for q in rowsum_queries
                    ),
                    "measured_count": sum(
                        1 for q in rowsum_queries if q["role"] == "measured"
                    ),
                    "heldout_count": sum(
                        1 for q in rowsum_queries if q["role"] == "heldout"
                    ),
                },
                "general": {
                    "seed": GENERAL_SEED,
                    "k": int(specification["general_k"]),
                    "weight_values": [-1, 1],
                    "quantile_range": list(GENERAL_QUANTILE_RANGE),
                    "margin": GENERAL_MARGIN,
                    "count": len(general_queries),
                    "measured_count": sum(
                        1 for q in general_queries if q["role"] == "measured"
                    ),
                    "heldout_count": sum(
                        1 for q in general_queries if q["role"] == "heldout"
                    ),
                },
            },
            "measured_query_count": partition["measured_query_count"],
            "heldout_query_count": partition["heldout_query_count"],
            "measured_query_identity_sha256": partition[
                "measured_query_identity_sha256"
            ],
            "heldout_query_identity_sha256": partition[
                "heldout_query_identity_sha256"
            ],
            "query_identity_sha256": _identity(queries),
            "input_sha256": {
                "schema": _sha256_file(specification["schema"]),
                "source": _sha256_file(specification["source"]),
            },
        },
        "queries": queries,
    }


def _serialize_payload(payload: Dict[str, Any]) -> str:
    """Serialize valid JSON with one auditable query per line."""
    queries = payload["queries"]
    metadata = {
        key: value for key, value in payload.items() if key != "queries"
    }
    lines = ["{"]
    for key, value in metadata.items():
        encoded_key = json.dumps(key, ensure_ascii=False)
        encoded_value = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(f"  {encoded_key}: {encoded_value},")
    lines.append('  "queries": [')
    for index, query in enumerate(queries):
        suffix = "," if index + 1 < len(queries) else ""
        encoded_query = json.dumps(
            query,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(f"    {encoded_query}{suffix}")
    lines.extend(["  ]", "}"])
    return "\n".join(lines) + "\n"


def _write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_serialize_payload(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="确定性重建并与已存在的正式文件逐字段比较",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        specification = DATASETS[dataset]
        payload = build_dataset_workload(dataset, specification)
        output = specification["output"]
        if args.verify_existing:
            if _load_json(output) != payload:
                raise RuntimeError(f"{output} 无法由冻结构造器确定性重建")
            print(f"verified {output} sha256={_sha256_file(output)}")
        else:
            _write_payload(output, payload)
            print(f"wrote {output} sha256={_sha256_file(output)}")


if __name__ == "__main__":
    main()
