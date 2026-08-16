#!/usr/bin/env python
"""在当前 HEAD 重放残差几何正式实验的主判定臂并对拍表哈希（Issue #57）。

用途（PR #59 审查意见 2）：正式结果运行于提交 aac1aff，其后分支合并了
master（含 evolution.py 变化）。本工具在当前 HEAD 以完全相同的协议参数
重放主判定两臂（absolute / relative_f8）× 全部正式种子，对拍正式 JSON
中记录的 final_table_sha256 与 measured L1，逐位一致即证明合并未改变
正式结论所依赖的行为。同时输出正式 JSON 自身的 SHA-256 供归档。

协议参数直接从 scripts/probe_residual_geometry_formal.py 导入（不复制），
保证与正式运行同源。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from scripts import probe_residual_geometry_formal as protocol
else:
    import probe_residual_geometry_formal as protocol
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.schema import load_schema

REPLAY_ARMS = ("absolute", "relative_f8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-json", type=Path, default=protocol.OUTPUT_PATH,
        help="正式结果 JSON 路径（默认预注册输出路径）",
    )
    parser.add_argument(
        "--dataset", default="nltcs", help="重放数据集（默认主判定 nltcs）",
    )
    args = parser.parse_args()

    payload = json.loads(args.formal_json.read_text())
    formal_sha256 = hashlib.sha256(
        args.formal_json.read_bytes()
    ).hexdigest()
    print(f"formal JSON: {args.formal_json}")
    print(f"formal JSON SHA-256: {formal_sha256}")
    print(f"formal commit: {payload['provenance']['git_commit']}")
    print(f"replay commit: {protocol._git('rev-parse', 'HEAD')}")

    spec = protocol.DATASETS[args.dataset]
    schema = load_schema(str(spec["schema"]))
    queries = load_queries(str(spec["queries"]))
    marginals = load_marginals(str(spec["marginals"]))
    target = np.asarray([q["result"] for q in queries], dtype=float)
    n_records = spec["n_records"]
    seeds = payload["protocol"]["seeds"]
    rounds = payload["protocol"]["rounds"]

    recorded = {
        (run["seed"], run["arm"]): run
        for run in payload["datasets"][args.dataset]["runs"]
    }

    mismatches = []
    for seed in seeds:
        for arm in REPLAY_ARMS:
            _, diag = run_evolution(
                target=target, queries=queries, schema=schema,
                n_records=n_records, n_rounds=rounds, seed=seed,
                marginals=marginals, log_every=0, device=spec["device"],
                return_final_table=True,
                **protocol.SHARED_PARAMS, **protocol.ARMS[arm],
            )
            final_table = diag.pop("final_table")
            sha = protocol._frame_sha256(final_table)
            final_q = evaluate_table(final_table, queries)
            l1 = float(
                compute_normalized_l1(final_q, target, n_records=n_records)
            )
            ref = recorded[(seed, arm)]
            sha_ok = sha == ref["final_table_sha256"]
            l1_ok = l1 == ref["final_table_measured_l1"]
            status = "OK" if (sha_ok and l1_ok) else "MISMATCH"
            if status == "MISMATCH":
                mismatches.append((seed, arm))
            print(
                f"[{args.dataset} seed={seed} {arm}] L1={l1:.6f} "
                f"sha={sha[:12]}… vs 记录 {ref['final_table_sha256'][:12]}… "
                f"-> {status}",
                flush=True,
            )

    if mismatches:
        print(f"重放失败：{len(mismatches)} 个 run 与正式记录不一致")
        sys.exit(1)
    print(
        f"重放通过：{len(seeds) * len(REPLAY_ARMS)} 个主判定 run 的最终表"
        "哈希与 measured L1 全部与正式记录逐位一致"
    )


if __name__ == "__main__":
    main()
