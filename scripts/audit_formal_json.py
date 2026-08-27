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
import os
import tempfile
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
    parser.add_argument(
        "--output", default=None,
        help="审计（含迁移）结果输出路径；默认 <json>.audited.json。"
        "原产物文件是强制只读不变量（PR #62 五轮不变量 4）：输出与"
        "输入指向同一物理文件（含路径别名/符号链接/硬链接）一律拒绝",
    )
    args = parser.parse_args()

    protocol = importlib.import_module(args.protocol)
    json_path = Path(args.json)
    output_path = (
        Path(args.output) if args.output
        else json_path.with_suffix(json_path.suffix + ".audited.json")
    )
    # 不变量 4（五轮意见 3）：原产物不可写。resolve() 消解相对路径与
    # 符号链接；samefile 捕获硬链接与其它别名。输出已存在且与输入同
    # 物理文件 → 拒绝；输出不存在时比较解析后的最终路径。
    resolved_input = json_path.resolve(strict=True)
    if output_path.exists():
        if os.path.samefile(json_path, output_path):
            raise SystemExit(
                "FATAL: --output 与输入指向同一物理文件（含别名/链接），"
                "原产物是只读不变量，拒绝审计"
            )
    elif output_path.resolve() == resolved_input:
        raise SystemExit(
            "FATAL: --output 解析后与输入路径相同，拒绝审计"
        )
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
            # 五轮不变量 5：白名单验证原始身份后只迁移内存副本——注入
            # 结构版本、用冻结代码重建 canonical 协议清单、判定字段
            # 改名，随后对迁移结果执行与新产物完全相同的统一验证。
            # 原文件保持逐字节不变。
            if hasattr(protocol, "ARTIFACT_SCHEMA_VERSION"):
                payload["artifact_schema_version"] = (
                    protocol.ARTIFACT_SCHEMA_VERSION
                )
            if hasattr(protocol, "canonical_protocol_manifest"):
                legacy_declared = payload.get("protocol", {})
                payload["run_config"] = {
                    "seeds": legacy_declared.get("seeds"),
                    "rounds": legacy_declared.get("rounds"),
                    "datasets": legacy_declared.get("datasets"),
                }
                payload["protocol"] = (
                    protocol.canonical_protocol_manifest()
                )
            for ds in payload.get("datasets", {}).values():
                if "judgement" in ds and "judgment" not in ds:
                    ds["judgment"] = ds.pop("judgement")
            changed.append("legacy 迁移: 内存副本升级到当前产物结构版本")
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
        # 0b-1（五轮不变量 1+2）：统一结构验证——产物结构版本必须
        # 匹配，protocol 字段与冻结代码重建的唯一 canonical 清单整份
        # 精确比较（缺字段、多字段或任一字段不同都拒绝，不再逐字段）。
        if hasattr(protocol, "ARTIFACT_SCHEMA_VERSION"):
            recorded_version = payload.get("artifact_schema_version")
            if recorded_version != protocol.ARTIFACT_SCHEMA_VERSION:
                print(
                    "FATAL: 产物结构版本 "
                    f"{recorded_version!r} != 当前 "
                    f"{protocol.ARTIFACT_SCHEMA_VERSION!r}"
                )
                sys.exit(1)
        declared = payload.get("protocol")
        if hasattr(protocol, "canonical_protocol_manifest"):
            expected_manifest = protocol.canonical_protocol_manifest()
            if not isinstance(declared, dict):
                print("FATAL: formal 产物缺失 protocol 清单或类型错误")
                sys.exit(1)
            recorded_canon = json.dumps(
                declared, sort_keys=True, ensure_ascii=False,
            )
            expected_canon = json.dumps(
                expected_manifest, sort_keys=True, ensure_ascii=False,
            )
            if recorded_canon != expected_canon:
                recorded_keys = set(declared)
                expected_keys = set(expected_manifest)
                detail = []
                if recorded_keys != expected_keys:
                    detail.append(
                        f"缺失 {sorted(expected_keys - recorded_keys)} "
                        f"多余 {sorted(recorded_keys - expected_keys)}"
                    )
                else:
                    detail.append(str(sorted(
                        key for key in expected_keys
                        if json.dumps(declared[key], sort_keys=True)
                        != json.dumps(expected_manifest[key], sort_keys=True)
                    )))
                print(
                    "FATAL: formal 产物 protocol 清单与冻结代码重建值"
                    f"整份对拍失败（差异字段：{'；'.join(detail)}）"
                )
                sys.exit(1)
            changed.append("protocol: canonical 清单整份对拍一致")
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
                # 0b-5（四轮意见 3）：离线参考名称集合必须与该数据集
                # 冻结的参考名精确一致（任意错误名称不再被静默接受）。
                expected_refs = set(
                    protocol.DATASETS[ds_name].get("references", {})
                )
                got_refs = set(run.get("offline") or {})
                if expected_refs and got_refs != expected_refs:
                    print(
                        f"FATAL: {ds_name} seed={run.get('seed')} "
                        f"arm={run.get('arm')} 离线参考名 "
                        f"{sorted(got_refs)} != 冻结 {sorted(expected_refs)}"
                    )
                    sys.exit(1)
                if not run.get("offline"):
                    print(
                        f"FATAL: {ds_name} seed={run.get('seed')} "
                        f"arm={run.get('arm')} 无 offline 指标"
                    )
                    sys.exit(1)
        changed.append(
            "formal 完整性: 数据集/seed×arm/指标有限性/参考名全部通过"
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

        # 4. 判定重算（五轮不变量 3）：formal 产物必须恰好存在一种
        # 拼写（legacy 已在迁移中改名为 judgment）、值必须是非空映射，
        # 随后无条件重算并精确比较；非 formal 产物保持存在才校验。
        has_new = "judgment" in ds
        has_old = "judgement" in ds
        if provenance.get("formal"):
            if has_new and has_old:
                print(f"FATAL: {ds_name} 同时存在 judgment 与 judgement")
                sys.exit(1)
            if not has_new and not has_old:
                print(f"FATAL: {ds_name} formal 产物缺失判定字段")
                sys.exit(1)
            judgment_key = "judgment" if has_new else "judgement"
            recorded_judgment = ds.get(judgment_key)
            if not isinstance(recorded_judgment, dict) or (
                not recorded_judgment
            ):
                print(
                    f"FATAL: {ds_name} 判定字段必须是非空映射，得到 "
                    f"{type(recorded_judgment).__name__}"
                )
                sys.exit(1)
            recomputed = protocol._judge(ds["runs"])
            if json.dumps(recorded_judgment, sort_keys=True) != (
                json.dumps(recomputed, sort_keys=True)
            ):
                print(f"FATAL: {ds_name} 判定重算与记录不一致")
                sys.exit(1)
            changed.append(f"{ds_name}: 判定无条件重算一致")
        else:
            judgment_key = "judgment" if has_new else (
                "judgement" if has_old else None
            )
            old_judgment = ds.get(judgment_key) if judgment_key else None
            if old_judgment is not None:
                new_judgment = protocol._judge(ds["runs"])
                if json.dumps(old_judgment, sort_keys=True) != (
                    json.dumps(new_judgment, sort_keys=True)
                ):
                    print(f"FATAL: {ds_name} 判定重算与记录不一致")
                    sys.exit(1)
                changed.append(f"{ds_name}: 判定重算一致")

    # 五轮不变量 4：结果先写同目录临时文件再原子替换独立输出文件。
    fd, temp_name = tempfile.mkstemp(
        dir=str(output_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=1, ensure_ascii=False))
        os.replace(temp_name, output_path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    for line in changed:
        print("·", line)
    print("output=" + str(output_path))
    print("sha256=" + _sha256_file(output_path))


if __name__ == "__main__":
    main()
