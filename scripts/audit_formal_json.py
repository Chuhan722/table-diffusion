#!/usr/bin/env python
"""正式 JSON 补充字段的独立审计与迁移工具（PR #45/#47 第四轮审查）。

作用（幂等，可重复运行验证）：

1. **test 撤回迁移**：把残留在活动字段的 ``reference_sha256.test`` 移入
   ``test_evaluation_withdrawn.withdrawn_reference_sha256``（历史审计信息，
   不再位于活动参考中）；
2. **字段口径改名**：``runs[*].tail_mean_loss`` →
   ``tail_mean_pre_proposal_loss``（该窗口基于 round-start/pre-proposal
   的 loss_history，不含末轮接受后的真实终态）；
3. **initial_state 重建/验证**：按协议以 n_rounds=0 逐正式种子重算
   marginal 初始化的 measured L1 与 loss——已存在时逐种子对拍（不一致即
   失败退出），缺失时重建写入。这使补充字段可由任何人独立复现；
4. **判定重算**：调用协议模块自身的 ``_judge`` 重算判定并断言分类/标志
   与文件中记录一致（字段迁移不得改变结论）。

用法（在协议所在工作树根目录）：
  PYTHONPATH=src conda run -n gsd python scripts/audit_formal_json.py \
    --protocol probe_gate_free_formal \
    --json outputs/gate_free_self_cooling/formal_5seed_2000round.json
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from table_diffevo.evolution import run_evolution  # noqa: E402
from table_diffevo.marginals import load_marginals  # noqa: E402
from table_diffevo.metrics import compute_normalized_l1  # noqa: E402
from table_diffevo.objective import compute_loss  # noqa: E402
from table_diffevo.queries import evaluate_table, load_queries  # noqa: E402
from table_diffevo.schema import load_schema  # noqa: E402


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_params(protocol):
    shared = getattr(protocol, "SHARED_PARAMS", None)
    if shared is None:
        shared = getattr(protocol, "BASE_PARAMS")
    return shared


def _rebuild_initial_state(protocol, ds_name, seeds):
    spec = protocol.DATASETS[ds_name]
    schema = load_schema(str(spec["schema"]))
    queries = load_queries(str(spec["queries"]))
    marginals = load_marginals(str(spec["marginals"]))
    target = np.asarray([q["result"] for q in queries], dtype=float)
    n_records = spec["n_records"]
    shared = _shared_params(protocol)
    first_arm = next(iter(protocol.ARMS.values()))
    l1_by_seed = {}
    loss_by_seed = {}
    for seed in seeds:
        _, diag0 = run_evolution(
            target=target, queries=queries, schema=schema,
            n_records=n_records, n_rounds=0, seed=seed,
            marginals=marginals, log_every=-1, device=spec["device"],
            return_final_table=True, **{**shared, **first_arm},
        )
        table0 = diag0.pop("final_table")
        q0 = evaluate_table(table0, queries)
        loss_by_seed[str(seed)] = float(compute_loss(target, q0))
        l1_by_seed[str(seed)] = float(
            compute_normalized_l1(target, q0, n_records)
        )
    return {
        "measured_l1_mean": float(np.mean(list(l1_by_seed.values()))),
        "measured_l1_by_seed": l1_by_seed,
        "loss_by_seed": loss_by_seed,
        "note": (
            "n_rounds=0 的 marginal 初始化状态（种子相关）；由"
            " scripts/audit_formal_json.py 可独立重建复验"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    protocol = importlib.import_module(args.protocol)
    json_path = Path(args.json)
    payload = json.loads(json_path.read_text())
    changed = []

    # 0. 协议身份复核（PR #62 审查意见 3）：新产物（provenance 含
    # protocol_sha256）必须与协议模块当前可复算值一致，且 formal=true
    # 时 protocol_match 必须为 true；缺字段的旧正式产物标记 legacy
    # 跳过（其身份由 git_commit + 输入哈希锚定）。
    provenance = payload.get("provenance", {})
    recorded_protocol_sha = provenance.get("protocol_sha256")
    if recorded_protocol_sha is not None:
        if not hasattr(protocol, "protocol_sha256"):
            print("FATAL: 产物记录了 protocol_sha256 但协议模块无法复算")
            sys.exit(1)
        current_sha = protocol.protocol_sha256()
        if recorded_protocol_sha != current_sha:
            print(
                "FATAL: 协议身份不一致——产物 "
                f"{recorded_protocol_sha[:12]}… != 当前协议模块 "
                f"{current_sha[:12]}…（协议已漂移，审计无效）"
            )
            sys.exit(1)
        if provenance.get("formal") and not provenance.get(
            "protocol_match", False
        ):
            print("FATAL: formal=true 但 protocol_match 不为 true")
            sys.exit(1)
        changed.append("provenance: 协议 SHA 复核一致")
    else:
        # legacy 白名单（PR #62 二轮审查）：缺 protocol_sha256 的文件
        # 只有整体字节 SHA 命中协议模块登记的已知旧正式产物才被接受；
        # 未定义白名单的旧协议模块保持原行为（身份由 git_commit +
        # input_sha256 锚定，仅标注）。
        known_legacy = getattr(
            protocol, "KNOWN_LEGACY_ARTIFACT_SHA256", None
        )
        if known_legacy is not None:
            file_sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
            if file_sha not in known_legacy:
                print(
                    "FATAL: 无 protocol_sha256 且文件 SHA "
                    f"{file_sha[:12]}… 不在已知 legacy 产物白名单中"
                    "（损坏或被修改的结果不被接受）"
                )
                sys.exit(1)
            changed.append(
                f"provenance: 已知 legacy 产物（{file_sha[:12]}…）"
            )
        else:
            changed.append(
                "provenance: 无 protocol_sha256 字段（legacy 产物，身份由 "
                "git_commit + input_sha256 锚定）"
            )

    # 0b. 结果完整性复核（PR #62 二轮/三轮审查）：formal=true 的产物
    # 必须覆盖协议预注册的全部数据集与 seed×arm 组合（恰好一次），
    # 每个 run 的主指标与离线指标为真正的有限数值（bool/字符串拒绝），
    # 且产物记录的协议参数与输入/参考身份逐项等于协议模块冻结常量。
    def _is_finite_number(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and np.isfinite(value)
        )

    if provenance.get("formal") and all(
        hasattr(protocol, attr)
        for attr in ("DATASETS", "FORMAL_SEEDS", "ARMS")
    ):
        # 0b-1（三轮意见 1）：产物声明的协议参数必须与冻结常量一致。
        declared = payload.get("protocol", {})
        if list(declared.get("seeds", [])) != list(protocol.FORMAL_SEEDS):
            print(
                "FATAL: formal 产物 protocol.seeds "
                f"{declared.get('seeds')!r} != 冻结 FORMAL_SEEDS "
                f"{list(protocol.FORMAL_SEEDS)!r}"
            )
            sys.exit(1)
        if hasattr(protocol, "FORMAL_ROUNDS") and (
            declared.get("rounds") != protocol.FORMAL_ROUNDS
        ):
            print(
                "FATAL: formal 产物 protocol.rounds "
                f"{declared.get('rounds')!r} != 冻结 FORMAL_ROUNDS "
                f"{protocol.FORMAL_ROUNDS!r}"
            )
            sys.exit(1)
        # 0b-2（三轮意见 1）：输入与参考身份逐项对拍冻结常量；formal
        # 产物缺失这些字段本身即 FATAL。
        if hasattr(protocol, "EXPECTED_INPUT_SHA256"):
            recorded_inputs = provenance.get("input_sha256") or {}
            for ds_name, expected in protocol.EXPECTED_INPUT_SHA256.items():
                got = recorded_inputs.get(ds_name) or {}
                for kind, digest in expected.items():
                    if got.get(kind) != digest:
                        print(
                            f"FATAL: formal 产物 input_sha256[{ds_name}]"
                            f"[{kind}]={got.get(kind)!r} != 冻结 "
                            f"{digest[:12]}…"
                        )
                        sys.exit(1)
        if hasattr(protocol, "EXPECTED_REFERENCE_SHA256"):
            for ds_name, expected in (
                protocol.EXPECTED_REFERENCE_SHA256.items()
            ):
                got = (
                    payload.get("datasets", {})
                    .get(ds_name, {})
                    .get("reference_sha256")
                    or {}
                )
                for ref_name, digest in expected.items():
                    if got.get(ref_name) != digest:
                        print(
                            "FATAL: formal 产物 reference_sha256"
                            f"[{ds_name}][{ref_name}]={got.get(ref_name)!r}"
                            f" != 冻结 {digest[:12]}…"
                        )
                        sys.exit(1)
        expected_datasets = set(protocol.DATASETS)
        got_datasets = set(payload.get("datasets", {}))
        if got_datasets != expected_datasets:
            print(
                "FATAL: formal 产物数据集不完整——期望 "
                f"{sorted(expected_datasets)}，实际 {sorted(got_datasets)}"
            )
            sys.exit(1)
        required_metrics = ("final_table_measured_l1",)
        offline_metrics = (
            "unmeasured_3way_l1", "unmeasured_4way_l1", "binned_joint_tvd",
        )
        for ds_name, ds in payload["datasets"].items():
            # 0b-3（三轮意见 2）：seed×arm 恰好一次——Counter 而非集合，
            # 重复 run 不被吞。
            from collections import Counter

            combo_counts = Counter(
                (run.get("seed"), run.get("arm"))
                for run in ds.get("runs", [])
            )
            expected_combos = {
                (seed, arm)
                for seed in protocol.FORMAL_SEEDS
                for arm in protocol.ARMS
            }
            if set(combo_counts) != expected_combos:
                print(
                    f"FATAL: {ds_name} 的 seed×arm 组合不完整——缺失 "
                    f"{sorted(expected_combos - set(combo_counts))[:5]}"
                )
                sys.exit(1)
            duplicated = {
                combo: count for combo, count in combo_counts.items()
                if count != 1
            }
            if duplicated:
                print(
                    f"FATAL: {ds_name} 存在重复 seed×arm run："
                    f"{sorted(duplicated.items())[:5]}"
                )
                sys.exit(1)
            for run in ds["runs"]:
                # 0b-4（三轮意见 3）：与生成端一致的数值类型检查，
                # bool/字符串/NaN/inf 一律拒绝。
                for key in required_metrics:
                    value = run.get(key)
                    if not _is_finite_number(value):
                        print(
                            f"FATAL: {ds_name} seed={run.get('seed')} "
                            f"arm={run.get('arm')} 指标 {key}={value!r} "
                            "缺失或非有限数值"
                        )
                        sys.exit(1)
                for ref_name, metrics in (run.get("offline") or {}).items():
                    for key in offline_metrics:
                        value = metrics.get(key)
                        if not _is_finite_number(value):
                            print(
                                f"FATAL: {ds_name} seed={run.get('seed')} "
                                f"arm={run.get('arm')} {ref_name}/{key}"
                                f"={value!r} 缺失或非有限数值"
                            )
                            sys.exit(1)
                if not run.get("offline"):
                    print(
                        f"FATAL: {ds_name} seed={run.get('seed')} "
                        f"arm={run.get('arm')} 无 offline 指标"
                    )
                    sys.exit(1)
        changed.append(
            "formal 完整性: 数据集/seed×arm/指标有限性全部通过"
        )

    for ds_name, ds in payload["datasets"].items():
        # 1. test 撤回迁移
        refs = ds.get("reference_sha256", {})
        if "test" in refs:
            withdrawn = ds.setdefault("test_evaluation_withdrawn", {})
            withdrawn["withdrawn_reference_sha256"] = {
                "test": refs.pop("test")
            }
            changed.append(f"{ds_name}: reference_sha256.test → "
                           "test_evaluation_withdrawn")

        # 2. 字段改名
        renamed = 0
        for run in ds.get("runs", []):
            if "tail_mean_loss" in run:
                run["tail_mean_pre_proposal_loss"] = run.pop(
                    "tail_mean_loss"
                )
                renamed += 1
        if renamed:
            changed.append(f"{ds_name}: tail_mean_loss 改名 ×{renamed}")

        # 3. initial_state 重建/验证。formal 产物固定使用协议模块的
        # FORMAL_SEEDS 重建（PR #62 三轮审查意见 1：不信任产物自报的
        # seeds，伪造 protocol.seeds + initial_state 不能通过）；非
        # formal 或旧协议无该常量时退回产物声明值。
        if provenance.get("formal") and hasattr(protocol, "FORMAL_SEEDS"):
            seeds = list(protocol.FORMAL_SEEDS)
        else:
            seeds = payload["protocol"]["seeds"]
        if ds_name in protocol.DATASETS:
            rebuilt = _rebuild_initial_state(protocol, ds_name, seeds)
            existing = ds.get("initial_state")
            if existing is None:
                ds["initial_state"] = rebuilt
                changed.append(f"{ds_name}: initial_state 重建写入")
            else:
                for seed, value in rebuilt["measured_l1_by_seed"].items():
                    old = existing.get("measured_l1_by_seed", {}).get(seed)
                    if old is None or not np.isclose(
                        old, value, rtol=1e-12, atol=1e-15
                    ):
                        print(f"FATAL: {ds_name} initial_state seed {seed} "
                              f"复验不一致（{old} vs {value}）")
                        sys.exit(1)
                existing["note"] = rebuilt["note"]
                changed.append(f"{ds_name}: initial_state 逐种子复验通过")

        # 4. 判定重算断言（兼容早期产物的 judgement 旧拼写）
        judgment_key = "judgment" if "judgment" in ds else (
            "judgement" if "judgement" in ds else None
        )
        old_judgment = ds.get(judgment_key) if judgment_key else None
        if old_judgment is not None:
            new_judgment = protocol._judge(ds["runs"])
            old_str = json.dumps(old_judgment, sort_keys=True)
            new_str = json.dumps(new_judgment, sort_keys=True)
            if old_str != new_str:
                # 允许的差异只有本工具改名的字段不参与判定——逐键比较
                print(f"FATAL: {ds_name} 判定重算与记录不一致")
                sys.exit(1)
            changed.append(f"{ds_name}: 判定重算一致")

    json_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    for line in changed:
        print("·", line)
    print("output=" + str(json_path))
    print("sha256=" + _sha256_file(json_path))


if __name__ == "__main__":
    main()
