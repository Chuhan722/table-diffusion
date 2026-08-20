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
        changed.append(
            "provenance: 无 protocol_sha256 字段（legacy 产物，身份由 "
            "git_commit + input_sha256 锚定）"
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

        # 3. initial_state 重建/验证
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
