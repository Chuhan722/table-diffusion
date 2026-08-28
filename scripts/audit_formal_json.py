#!/usr/bin/env python
"""正式产物独立审计器（PR #62 第六轮最终合同）。

职责边界：只审计正式产物（formal artifact）的结构、协议、身份、
完整性与迁移安全。协议模块必须提供唯一生产验证入口
``validate_formal_artifact(payload, *, source_kind)``（生成器写盘前、
审计器输出前、测试全部调用同一入口，不存在第二套验证逻辑）。

流程（fail-closed，任何一步失败即退出码 1 且不产生输出）：

1. 路径不变量：输入与输出指向同一物理文件（相同路径、路径别名、
   符号链接、硬链接）一律拒绝——原产物是只读强不变量；
2. 原文件字节只读取一次：同一份字节同时用于 SHA-256 与 JSON 解析；
3. 新版产物（含 artifact_schema_version）不做任何迁移；旧版产物只有
   整文件 SHA-256 命中协议模块冻结白名单才进入 legacy 迁移，且迁移
   只操作内存副本（注入结构版本/重建协议清单/补齐 provenance 协议
   身份/判定字段改名），原文件始终逐字节不变；
4. 组装带 audit 段（源文件哈希 + 来源类型，无时间戳保证幂等）的完整
   输出后，调用协议模块统一验证入口做整份严格验证；
5. 输出安全：已存在的输出若无法解析或其 audit.source_sha256 与本次
   源文件不同，拒绝覆盖（避免失败后把旧输出误认为本次结果）；验证
   全部通过后才创建输出，先写同目录临时文件再原子替换。

用法（在协议所在工作树根目录）：
  PYTHONPATH=src:scripts conda run -n gsd python scripts/audit_formal_json.py \
    --protocol probe_residual_geometry_formal \
    --json docs/实验结果/formal_residual_geometry_5seed_2000round.json
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fatal(message):
    print(f"FATAL: {message}")
    sys.exit(1)


def _migrate_legacy_in_memory(protocol, payload, notes):
    """白名单命中后的 legacy 内存迁移（原文件不动）。

    迁移语义是"升级到当前产物结构版本"，注入值全部来自冻结协议代码
    （白名单整文件哈希已验证原始身份，协议一致性由历史 review 锚定），
    随后迁移结果与新版产物走完全相同的统一严格验证。
    """
    payload["artifact_schema_version"] = protocol.ARTIFACT_SCHEMA_VERSION
    legacy_declared = payload.get("protocol", {})
    payload["run_config"] = {
        "seeds": legacy_declared.get("seeds"),
        "rounds": legacy_declared.get("rounds"),
        "datasets": sorted(legacy_declared.get("datasets", [])),
    }
    payload["protocol"] = protocol.canonical_protocol_manifest()
    notes.append("legacy 迁移: protocol → 当前 canonical 清单；原声明的"
                 " seeds/rounds/datasets 转存 run_config")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        _fatal("legacy 产物缺失 provenance，无法迁移")
    # 旧版生成于 protocol_sha256 机制引入之前；白名单验证原始身份后，
    # 由冻结代码补齐协议身份三元组（成为可整份验证的新版结构）。
    provenance.setdefault("protocol_sha256", protocol.FROZEN_PROTOCOL_SHA256)
    provenance.setdefault("protocol_match", True)
    provenance.setdefault("input_hash_mismatches", [])
    notes.append("legacy 迁移: provenance 补齐协议身份字段"
                 "（protocol_sha256/protocol_match/input_hash_mismatches）")
    for ds_name, ds in payload.get("datasets", {}).items():
        if isinstance(ds, dict) and "judgement" in ds and (
            "judgment" not in ds
        ):
            ds["judgment"] = ds.pop("judgement")
            notes.append(f"legacy 迁移: {ds_name} judgement → judgment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--output", default=None,
        help="审计结果输出路径；默认 <json>.audited.json。原产物文件是"
        "强制只读不变量：输出与输入指向同一物理文件（含路径别名/符号"
        "链接/硬链接）一律拒绝",
    )
    args = parser.parse_args()

    protocol = importlib.import_module(args.protocol)
    for required_attr in (
        "validate_formal_artifact", "canonical_protocol_manifest",
        "ARTIFACT_SCHEMA_VERSION", "FROZEN_PROTOCOL_SHA256",
        "KNOWN_LEGACY_ARTIFACT_SHA256", "FormalArtifactError",
    ):
        if not hasattr(protocol, required_attr):
            _fatal(
                f"协议模块 {args.protocol} 未提供 {required_attr}——"
                "审计器只支持实现了统一验证入口合同的协议模块"
            )

    json_path = Path(args.json)
    output_path = (
        Path(args.output) if args.output
        else json_path.with_suffix(json_path.suffix + ".audited.json")
    )
    # 不变量：原产物不可写。resolve() 消解相对路径与符号链接；samefile
    # 捕获硬链接与其它别名。输出已存在且与输入同物理文件 → 拒绝；
    # 输出不存在时比较解析后的最终路径。
    resolved_input = json_path.resolve(strict=True)
    if output_path.exists():
        if os.path.samefile(json_path, output_path):
            _fatal(
                "--output 与输入指向同一物理文件（含别名/链接），"
                "原产物是只读不变量，拒绝审计"
            )
    elif output_path.resolve() == resolved_input:
        _fatal("--output 解析后与输入路径相同，拒绝审计")

    # 原文件字节只读一次：同一份字节同时用于哈希与解析（消除
    # 读取-校验间隙）。
    raw_bytes = json_path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        _fatal(f"输入不是合法 JSON: {exc}")
    if not isinstance(payload, dict):
        _fatal("输入顶层必须是 JSON 对象")

    notes = []
    if "artifact_schema_version" in payload:
        source_kind = "v2"
        notes.append("来源: 新版产物（不做迁移）")
    else:
        if source_sha256 not in protocol.KNOWN_LEGACY_ARTIFACT_SHA256:
            _fatal(
                "无 artifact_schema_version 且文件 SHA "
                f"{source_sha256[:12]}… 不在已知 legacy 产物白名单中"
                "（损坏或被修改的结果不被接受）"
            )
        source_kind = "legacy_migrated"
        notes.append(f"来源: 已知 legacy 产物（{source_sha256[:12]}…），"
                     "迁移仅操作内存副本")
        _migrate_legacy_in_memory(protocol, payload, notes)

    # 组装完整输出（audit 段无时间戳：同一源文件两次审计输出逐字节
    # 相同），随后整份通过唯一验证入口——验证全部通过才允许创建输出。
    audited = dict(payload)
    audited["audit"] = {
        "source_sha256": source_sha256,
        "source_kind": source_kind,
    }
    try:
        protocol.validate_formal_artifact(audited, source_kind="audit")
    except protocol.FormalArtifactError as exc:
        _fatal(str(exc))
    notes.append("统一验证入口: 整份严格验证通过（结构/协议/身份/判定"
                 "重算/初始状态重算）")

    # 输出安全：已有输出不属于同一源文件时拒绝覆盖。
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_bytes())
            existing_source = existing["audit"]["source_sha256"]
        except Exception:
            _fatal(
                f"已存在的输出 {output_path} 无法解析为带 audit 段的"
                "审计结果，拒绝覆盖（请手动移除后重试）"
            )
        if existing_source != source_sha256:
            _fatal(
                f"已存在的输出 {output_path} 属于另一源文件"
                f"（{existing_source[:12]}… != {source_sha256[:12]}…），"
                "拒绝覆盖"
            )

    serialized = json.dumps(audited, indent=1, ensure_ascii=False)
    fd, temp_name = tempfile.mkstemp(
        dir=str(output_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temp_name, output_path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    for line in notes:
        print("·", line)
    print("output=" + str(output_path))
    print(
        "sha256="
        + hashlib.sha256(output_path.read_bytes()).hexdigest()
    )


if __name__ == "__main__":
    main()
