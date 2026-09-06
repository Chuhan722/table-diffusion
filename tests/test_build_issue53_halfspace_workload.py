"""
半空间 workload 冻结构造器测试（scripts/build_issue53_halfspace_workload.py）

对应设计稿：docs/设计/半空间不可微查询能力线设计稿.md 第 2 节。

验证：
1. 构造确定性：同一源两次构建 payload 逐字段相同
2. 结构合同：A 档全属性全 1 权重、θ 非退化、交替分配覆盖全谱；
   B 档 k=16、权重 ±1、θ 非退化；id/role/tier 一致性
3. 指纹与互斥：全部查询指纹唯一，measured/heldout 语义不相交，
   身份 SHA 与逐条指纹一致
4. 答案正确：result 与 evaluate_table 在源表上的重算逐条一致
5. 序列化：一行一 query 的 JSON 与 payload 往返相等
6. 冻结文件（若已存在）：可由构造器确定性重建
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_issue53_halfspace_workload as builder
from table_diffevo.quality import query_fingerprint
from table_diffevo.queries import evaluate_table, load_data


@pytest.fixture(scope="module")
def payload():
    return builder.build_dataset_workload("plants", builder.DATASETS["plants"])


@pytest.fixture(scope="module")
def source():
    return load_data(str(builder.DATASETS["plants"]["source"]))


class TestConstructionContract:
    def test_deterministic_rebuild(self, payload):
        rebuilt = builder.build_dataset_workload(
            "plants", builder.DATASETS["plants"]
        )
        assert rebuilt == payload

    def test_counts_and_tiers(self, payload):
        queries = payload["queries"]
        assert payload["query_count"] == len(queries)
        tiers = payload["construction"]["tiers"]
        rowsum = [q for q in queries if q["tier"] == "rowsum"]
        general = [q for q in queries if q["tier"] == "general"]
        assert len(rowsum) == tiers["rowsum"]["count"]
        assert len(general) == tiers["general"]["count"] == builder.GENERAL_COUNT
        assert tiers["general"]["measured_count"] == builder.GENERAL_COUNT // 2
        # A 档交替分配：measured 与 heldout 数量差最多 1
        assert abs(
            tiers["rowsum"]["measured_count"] - tiers["rowsum"]["heldout_count"]
        ) <= 1

    def test_rowsum_tier_structure(self, payload, source):
        columns = list(source.columns)
        rowsum = [q for q in payload["queries"] if q["tier"] == "rowsum"]
        n = payload["record_count"]
        thetas = []
        for query in rowsum:
            spec = query["halfspace"]
            assert spec["attributes"] == columns, "A 档必须覆盖全部属性"
            assert set(spec["weights"]) == {1}, "A 档权重必须全 1"
            lower = builder.ROWSUM_MARGIN * n
            assert lower <= query["result"] <= n - lower, "A 档 θ 必须非退化"
            thetas.append(spec["theta"])
        assert thetas == sorted(thetas), "A 档按 θ 升序排列"
        assert len(set(thetas)) == len(thetas)
        # 交替分配：相邻 θ 属于不同 role（两组都覆盖全谱）
        roles = [q["role"] for q in rowsum]
        assert all(a != b for a, b in zip(roles, roles[1:]))

    def test_general_tier_structure(self, payload, source):
        columns = set(source.columns)
        general = [q for q in payload["queries"] if q["tier"] == "general"]
        n = payload["record_count"]
        lower = builder.GENERAL_MARGIN * n
        for query in general:
            spec = query["halfspace"]
            assert len(spec["attributes"]) == builder.GENERAL_K
            assert len(set(spec["attributes"])) == builder.GENERAL_K
            assert set(spec["attributes"]) <= columns
            assert set(spec["weights"]) <= {-1, 1}
            assert lower <= query["result"] <= n - lower, "B 档 θ 必须非退化"

    def test_ids_roles_unique_and_consistent(self, payload):
        queries = payload["queries"]
        ids = [q["id"] for q in queries]
        assert len(set(ids)) == len(ids)
        for query in queries:
            tier_code = {"rowsum": "A", "general": "B"}[query["tier"]]
            role_code = {"measured": "M", "heldout": "H"}[query["role"]]
            assert query["id"].startswith(f"HS{tier_code}_{role_code}_")
            assert query["type"] == "halfspace"


class TestIdentityAndAnswers:
    def test_fingerprints_unique_and_recomputable(self, payload):
        queries = payload["queries"]
        fingerprints = [q["fingerprint_sha256"] for q in queries]
        assert len(set(fingerprints)) == len(fingerprints)
        for query in queries:
            assert query_fingerprint(query) == query["fingerprint_sha256"]

    def test_partition_identities(self, payload):
        construction = payload["construction"]
        queries = payload["queries"]
        measured = [q for q in queries if q["role"] == "measured"]
        heldout = [q for q in queries if q["role"] == "heldout"]
        assert construction["measured_query_count"] == len(measured)
        assert construction["heldout_query_count"] == len(heldout)
        for group, key in (
            (measured, "measured_query_identity_sha256"),
            (heldout, "heldout_query_identity_sha256"),
        ):
            identity = hashlib.sha256("\n".join(
                q["fingerprint_sha256"] for q in group
            ).encode("ascii")).hexdigest()
            assert construction[key] == identity

    def test_results_match_evaluate_table(self, payload, source):
        queries = payload["queries"]
        answers = evaluate_table(source, queries)
        recorded = np.array([q["result"] for q in queries])
        np.testing.assert_array_equal(answers, recorded)


class TestSerializationAndFrozenFile:
    def test_serialization_roundtrip(self, payload):
        text = builder._serialize_payload(payload)
        assert json.loads(text) == payload
        # 一行一 query 的审计格式
        assert text.count('"halfspace":') == payload["query_count"]

    def test_frozen_file_reproducible_if_present(self, payload):
        output = builder.DATASETS["plants"]["output"]
        if not Path(output).exists():
            pytest.skip("正式冻结文件尚未物化")
        frozen = json.loads(Path(output).read_text(encoding="utf-8"))
        assert frozen == payload
