"""test 数据端到端实验，编辑候选加候选提供器跑满演化并输出损失曲线。

初始表各字段独立均匀随机生成 300 行，不偷看真实表的联合结构，
目标 y 直接取查询负载里的真实答案，权重单位对角，
菜单随机数与抽样随机数分开，冻结重试上限 10。
用法示例，./.venv/bin/python scripts/run_test300.py --rounds 400
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.dataset import (  # noqa: E402
    StateRegistry,
    load_queries,
    load_table,
    query_field_sets,
    target_from_specs,
)
from resevo.editspace import make_edit_provider  # noqa: E402
from resevo.engine import evolve  # noqa: E402
from resevo.pairing import make_paired_provider  # noqa: E402
from resevo.state import table_loss  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_300x10"


def main() -> None:
    parser = argparse.ArgumentParser(description="test_300x10 编辑候选端到端实验")
    parser.add_argument("--rounds", type=int, default=400, help="演化轮数上限")
    parser.add_argument("--menu-seed", type=int, default=20260916, help="菜单随机种子")
    parser.add_argument("--sample-seed", type=int, default=7, help="抽样随机种子")
    parser.add_argument("--init-seed", type=int, default=1, help="初始表随机种子")
    parser.add_argument("--retries", type=int, default=10, help="连续冻结重试上限")
    parser.add_argument(
        "--out", type=str, default="", help="逐轮损失曲线输出 CSV 路径，留空不写文件"
    )
    parser.add_argument(
        "--save-table", type=str, default="", help="演化终表输出 CSV 路径，留空不写文件"
    )
    parser.add_argument(
        "--pairing", action="store_true",
        help="启用配对板块，冻结重试轮把互补行绑成双行块",
    )
    args = parser.parse_args()

    schema, real_rows = load_table(str(DATA_DIR / "test_300x10.csv"))
    specs = load_queries(str(DATA_DIR / "measured_50query.json"))
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    field_sets = [fs for fs in query_field_sets(specs, schema) if len(fs) >= 2]

    registry = StateRegistry(schema, specs)
    init_rng = np.random.default_rng(args.init_seed)
    init_rows = [
        tuple(
            schema.domains[j][int(init_rng.integers(len(schema.domains[j])))]
            for j in range(schema.num_fields)
        )
        for _ in range(len(real_rows))
    ]
    ids = registry.register_table(init_rows)

    factory = make_paired_provider if args.pairing else make_edit_provider
    provider = factory(
        registry, y, w,
        np.random.default_rng(args.menu_seed),
        joint_field_sets=field_sets,
    )
    initial_loss = table_loss(registry.build_workload(y, w), ids)
    print(f"行数 {len(ids)}，查询数 {len(specs)}，初始损失 {initial_loss:.6f}")

    t0 = time.perf_counter()
    out = evolve(
        None, ids, args.rounds,
        np.random.default_rng(args.sample_seed),
        support_provider=provider,
        max_frozen_retries=args.retries,
    )
    elapsed = time.perf_counter() - t0

    final_workload = registry.build_workload(y, w)
    final_loss = table_loss(final_workload, out.state_ids)
    frozen_rounds = sum(1 for r in out.records if r.status == "no_positive_direction")
    print(
        f"实际轮数 {len(out.records)}，停止原因 {out.stop_reason}，"
        f"冻结轮数 {frozen_rounds}，注册状态数 {registry.num_states}"
    )
    print(
        f"最终损失 {final_loss:.6f}，初始比 {final_loss / initial_loss:.4%}，"
        f"耗时 {elapsed:.1f} 秒，平均每轮 {elapsed / len(out.records) * 1000:.0f} 毫秒"
    )
    checkpoints = [r for r in out.records if r.round_index % max(1, len(out.records) // 10) == 0]
    for r in checkpoints:
        print(
            f"  轮 {r.round_index:4d}  损失 {r.old_loss:12.4f}  beta {r.beta:10.4f}  "
            f"步长 {r.step:8.5f}  期望损失 {r.expected_loss:12.4f}  {r.status}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["round", "old_loss", "beta", "direction_gain", "interaction", "step",
                 "expected_loss", "status"]
            )
            for r in out.records:
                writer.writerow(
                    [r.round_index, r.old_loss, r.beta, r.direction_gain,
                     r.interaction, r.step, r.expected_loss, r.status]
                )
            writer.writerow(["final", final_loss, "", "", "", "", "", out.stop_reason])
        print(f"曲线已写入 {out_path}")

    if args.save_table:
        from resevo.editspace import tuples_from_ids

        table_path = Path(args.save_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(schema.fields)
            for row in tuples_from_ids(registry, out.state_ids):
                writer.writerow(row)
        print(f"终表已写入 {table_path}")


if __name__ == "__main__":
    main()
