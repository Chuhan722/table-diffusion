#!/usr/bin/env python
"""等行数 test 侧离线重评价（PR #45/#47 审查意见，适用三个正式协议）。

问题：nltcs 正式生成表 16181 行，test 参考表 3236 行。原离线评价直接
在不等行数下计算有限样本指标（未测量 3/4-way cell error、joint TVD、
唯一状态数、support overlap），混合了真实分布差异与样本量差异——
`_marginal_cell_error` 的口径本身以两边行数一致为前提。

修正：对每个 (seed, arm) 的最终生成表，按固定评价种子（9001..9005）
无放回抽取与 test 等行数的子样本，重算全部 test 侧指标，报告均值/标准
差/逐种子值；原 test 侧全表指标存档为 `test_fullsize_invalid`（标记
无效，不再作为结论依据）；按协议自身的 _judge 重算判定，旧判定存档为
`judgment_before_test_reeval`。等行数的参考（train、小表 reference）
不受影响，但会全量复算并与存量逐字段对拍（重生成正确性的强验证）。

最终表通过重跑协议参数重生成，并与 JSON 记录的 final_table_sha256
逐一比对（不一致即失败退出）。附带补充初始状态 measured L1
（n_rounds=0，marginal 初始化；PR #47 审查要求的初始对照）。

用法（在协议所在工作树根目录）：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd python \
    scripts/reevaluate_test_offline.py \
    --protocol probe_gate_free_formal \
    --json outputs/gate_free_self_cooling/formal_5seed_2000round.json
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_factorized_gibbs_closed_loop as offline_helpers  # noqa: E402

from table_diffevo.evolution import run_evolution  # noqa: E402
from table_diffevo.marginals import load_marginals  # noqa: E402
from table_diffevo.metrics import compute_normalized_l1  # noqa: E402
from table_diffevo.objective import compute_loss  # noqa: E402
from table_diffevo.queries import evaluate_table, load_queries  # noqa: E402
from table_diffevo.schema import load_schema  # noqa: E402

EVAL_SEEDS = [9001, 9002, 9003, 9004, 9005]

OFFLINE_FIELDS = (
    "unmeasured_3way_l1",
    "unmeasured_4way_l1",
    "raw_joint_tvd",
    "binned_joint_tvd",
    "raw_unique_states",
    "raw_support_overlap",
)


def _frame_sha256(frame):
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True,
    ).stdout.strip()


def _offline_entry(reference, synthetic, marginals, domains, triples):
    metrics = offline_helpers._offline_metrics(
        reference, synthetic, marginals, domains, triples
    )
    return {
        "unmeasured_3way_l1": float(metrics["unmeasured_3way"]["mean"]),
        "unmeasured_4way_l1": float(metrics["unmeasured_4way"]["mean"]),
        "raw_joint_tvd": float(metrics["raw_joint"]["tvd"]),
        "binned_joint_tvd": float(metrics["binned_joint"]["tvd"]),
        "raw_unique_states": int(metrics["raw_joint"]["n_unique"]),
        "raw_support_overlap": int(metrics["raw_joint"]["support_overlap"]),
    }


def _subsampled_offline(reference, synthetic, marginals, domains, triples,
                        eval_seeds):
    per_seed = []
    n_target = len(reference)
    n_source = len(synthetic)
    for eval_seed in eval_seeds:
        rng = np.random.default_rng(eval_seed)
        idx = rng.choice(n_source, size=n_target, replace=False)
        subsample = synthetic.iloc[np.sort(idx)].reset_index(drop=True)
        per_seed.append(
            _offline_entry(reference, subsample, marginals, domains, triples)
        )
    aggregated = {}
    for field in OFFLINE_FIELDS:
        values = [entry[field] for entry in per_seed]
        aggregated[field] = float(np.mean(values))
        aggregated[field + "_std"] = float(np.std(values, ddof=1))
        aggregated[field + "_values"] = [float(v) for v in values]
    aggregated["subsample_rows"] = int(n_target)
    aggregated["source_rows"] = int(n_source)
    aggregated["eval_seeds"] = list(eval_seeds)
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True,
                        help="协议模块名，如 probe_gate_free_formal")
    parser.add_argument("--json", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="30 轮重生成 + 跳过 sha 验证（只验证管线）")
    parser.add_argument("--eval-seeds", nargs="+", type=int,
                        default=EVAL_SEEDS)
    args = parser.parse_args()

    protocol = importlib.import_module(args.protocol)
    shared = getattr(protocol, "SHARED_PARAMS", None)
    if shared is None:
        shared = getattr(protocol, "BASE_PARAMS")

    json_path = Path(args.json)
    payload = json.loads(json_path.read_text())

    for ds_name, ds_payload in payload["datasets"].items():
        spec = protocol.DATASETS[ds_name]
        schema = load_schema(str(spec["schema"]))
        queries = load_queries(str(spec["queries"]))
        marginals = load_marginals(str(spec["marginals"]))
        target = np.asarray([q["result"] for q in queries], dtype=float)
        n_records = spec["n_records"]
        columns = schema.attribute_names()
        domains = offline_helpers._discretization_domains(marginals)
        triples = offline_helpers._measured_cell_keys(
            queries, marginals, order=3
        )
        references = {
            ref_name: protocol._load_reference(path, columns)
            for ref_name, path in spec["references"].items()
        }
        mismatched = {
            ref_name: ref for ref_name, ref in references.items()
            if len(ref) != n_records
        }
        matched = {
            ref_name: ref for ref_name, ref in references.items()
            if len(ref) == n_records
        }
        if not mismatched:
            print(f"[{ds_name}] 所有参考表与生成表等行数，无需重评价")
            continue
        print(
            f"[{ds_name}] 行数不匹配参考: "
            + ", ".join(
                f"{k}({len(v)} vs {n_records})" for k, v in mismatched.items()
            ),
            flush=True,
        )

        # 初始状态 L1（n_rounds=0；PR #47 审查要求的初始对照。marginal
        # 初始化实际是种子相关的，逐正式种子计算并报告均值与逐种子值）
        init_l1_by_seed = {}
        init_loss_by_seed = {}
        for probe_seed in payload["protocol"]["seeds"]:
            _, diag0 = run_evolution(
                target=target, queries=queries, schema=schema,
                n_records=n_records, n_rounds=0, seed=probe_seed,
                marginals=marginals, log_every=-1, device=spec["device"],
                return_final_table=True,
                **{**shared, **next(iter(protocol.ARMS.values()))},
            )
            init_table = diag0.pop("final_table")
            init_q = evaluate_table(init_table, queries)
            init_loss_by_seed[str(probe_seed)] = float(
                compute_loss(target, init_q)
            )
            init_l1_by_seed[str(probe_seed)] = float(
                compute_normalized_l1(target, init_q, n_records)
            )
        ds_payload["initial_state"] = {
            "measured_l1_mean": float(
                np.mean(list(init_l1_by_seed.values()))
            ),
            "measured_l1_by_seed": init_l1_by_seed,
            "loss_by_seed": init_loss_by_seed,
            "note": (
                "n_rounds=0 的 marginal 初始化状态（种子相关）；审查"
                "（PR #47 意见 1）要求的初始对照基准"
            ),
        }
        print(
            f"[{ds_name}] initial L1 mean="
            f"{ds_payload['initial_state']['measured_l1_mean']:.6f}",
            flush=True,
        )

        rounds = 30 if args.smoke else payload["protocol"]["rounds"]
        runs = ds_payload["runs"]
        if args.smoke:
            runs = runs[:2]
        for run in runs:
            arm = run["arm"]
            seed = run["seed"]
            extra = protocol.ARMS[arm]
            t0 = time.time()
            _, diag = run_evolution(
                target=target, queries=queries, schema=schema,
                n_records=n_records, n_rounds=rounds, seed=seed,
                marginals=marginals, log_every=0, device=spec["device"],
                return_final_table=True,
                **{**shared, **extra},
            )
            final_table = diag.pop("final_table")
            regen_sha = _frame_sha256(final_table)
            if not args.smoke:
                if regen_sha != run["final_table_sha256"]:
                    print(
                        f"FATAL: [{ds_name} seed={seed} {arm}] 重生成表 "
                        f"sha256 不匹配（{regen_sha[:12]} vs "
                        f"{run['final_table_sha256'][:12]}），中止",
                    )
                    sys.exit(1)
                # 等行数参考全量复算，与存量对拍（重生成正确性交叉验证）
                for ref_name, reference in matched.items():
                    fresh = _offline_entry(
                        reference, final_table, marginals, domains, triples
                    )
                    stale = run["offline"][ref_name]
                    for field in OFFLINE_FIELDS:
                        if field not in stale:
                            continue
                        if not np.isclose(
                            fresh[field], stale[field],
                            rtol=1e-9, atol=1e-12,
                        ):
                            print(
                                f"FATAL: [{ds_name} seed={seed} {arm}] "
                                f"{ref_name}.{field} 复算不一致 "
                                f"({fresh[field]} vs {stale[field]})，中止",
                            )
                            sys.exit(1)
            for ref_name, reference in mismatched.items():
                stale = run["offline"].get(ref_name)
                if stale is not None:
                    stale = dict(stale)
                    stale["invalid_reason"] = (
                        "全表（16181 行）与 test（3236 行）不等行数直接"
                        "比较，混合分布差异与样本量差异；按审查意见撤回"
                    )
                    run["offline"][ref_name + "_fullsize_invalid"] = stale
                run["offline"][ref_name] = _subsampled_offline(
                    reference, final_table, marginals, domains, triples,
                    args.eval_seeds,
                )
            print(
                f"[{ds_name} seed={seed} {arm}] sha_ok "
                f"un3way(test)={run['offline']['test']['unmeasured_3way_l1']:.6f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        if not args.smoke:
            old_judgment = ds_payload.get("judgment")
            if old_judgment is not None:
                ds_payload["judgment_before_test_reeval"] = old_judgment
            ds_payload["judgment"] = protocol._judge(ds_payload["runs"])
            print(
                f"== {ds_name} 重评判定: "
                + json.dumps(
                    ds_payload["judgment"], ensure_ascii=False
                )[:400],
                flush=True,
            )

    payload["test_reevaluation"] = {
        "eval_seeds": list(args.eval_seeds),
        "method": (
            "对每个 (seed, arm) 最终表按固定评价种子无放回抽取与 test 等"
            "行数子样本，重算全部 test 侧指标并报告均值/标准差/逐种子值"
        ),
        "regenerated_tables_sha256_verified": not args.smoke,
        "git_commit": _git_commit(),
        "reevaluated_at": datetime.now().astimezone().isoformat(),
    }

    if args.smoke:
        out_path = json_path.with_suffix(".smoke.json")
    else:
        out_path = json_path
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("output=" + str(out_path))
    print("sha256=" + _sha256_file(out_path))


if __name__ == "__main__":
    main()
