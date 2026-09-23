"""plants 真数据端到端实验，编辑候选加候选提供器跑满演化并输出损失曲线。

初始表默认按考卷二阶答案边缘化出的一阶比例逐字段独立抽样，
不偷看真实表的联合结构，关联结构留给演化去修，
目标 y 直接取查询负载里的真实答案，权重单位对角，
菜单随机数与抽样随机数分开，冻结重试上限 10。
用法示例，./.venv/bin/python scripts/run_plants.py --rounds 400
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
from resevo.batchkernel import evolve_batch, make_batch_provider  # noqa: E402
from resevo.editspace import make_edit_provider  # noqa: E402
from resevo.engine import evolve  # noqa: E402
from resevo.grouping import evolve_grouped, make_grouped_provider  # noqa: E402
from resevo.initialization import clean_first_order, derive_first_order, derive_first_order_avg, sample_initial_rows  # noqa: E402
from resevo.pairing import PairingBudget, make_paired_provider  # noqa: E402
from resevo.state import table_loss  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def resolve_work_rows(total_rows: int, work_rows: int, work_frac: float) -> int:
    """工作批名额折算，比例开启时按行数乘比例四舍五入且至少一行。

    work_frac 为 0 时沿用 work_rows 绝对值，两者同时给出由参数校验拦截。
    """
    if work_frac <= 0.0:
        return work_rows
    return max(1, int(total_rows * work_frac + 0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="真数据端到端实验")
    parser.add_argument("--data", type=str, default="plants", help="数据目录名，表名须同名")
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
    parser.add_argument(
        "--grouped", action="store_true",
        help="启用重复记录压缩，相同状态的行共享菜单增益概率并按多项分布抽样",
    )
    parser.add_argument(
        "--batched", action="store_true",
        help="启用懒注册批量菜单，候选用差分表示不注册，抽中落地才登记",
    )
    parser.add_argument(
        "--exam", type=str, default="",
        help="训练考卷文件名，在数据目录下，留空自动发现唯一的 measured_*query.json",
    )
    parser.add_argument(
        "--uniform-init", action="store_true",
        help="退回纯均匀随机初始化，默认用考卷二阶边缘化出的一阶比例",
    )
    parser.add_argument(
        "--pairing-backoff", type=int, default=4,
        help="批量路径冻结重试的配对退避周期，前三次都配对之后每 N 次一次，1 即每次都配对",
    )
    parser.add_argument(
        "--pair-rescue-rows", type=int, default=0,
        help="惰性模式冻结救援轮抽此数量的行走子集配对，需 --pairing --gpu，0 关闭",
    )
    parser.add_argument(
        "--rescue-after", type=int, default=6,
        help="连败达此数才首次动用子集救援，之前只刷菜单自愈，零星冻结不烧重炮",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=32,
        help="每次救援允许绑定落回的配对上限，默认守文档预算 32",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="批量轮核构造走 cupy 后端，需 --batched，抽样与回退仍在 CPU",
    )
    parser.add_argument(
        "--work-rows", type=int, default=0,
        help="工作批行数，每轮按错位分挑组上场其余保持，0 为全量，需 --batched",
    )
    parser.add_argument(
        "--work-frac", type=float, default=0.0,
        help="工作批名额占数据行数比例，四舍五入取整且至少一行，0 关闭，与 --work-rows 二选一，需 --batched",
    )
    parser.add_argument(
        "--work-random", type=float, default=0.25,
        help="工作批随机名额比例，防止低分组饿死",
    )
    parser.add_argument(
        "--select-seed", type=int, default=20260917, help="工作批挑组随机种子"
    )
    parser.add_argument(
        "--work-below", type=float, default=0.0,
        help="步长低于该值后才启用工作批，早期全量后期裁组，0 为立即启用",
    )
    parser.add_argument(
        "--retry-select", action="store_true",
        help="冻结重试轮也走错位挑组不回退全量，有损开关需终点加阅卷验收",
    )
    parser.add_argument(
        "--stop-threshold", type=float, default=0.0,
        help="平台早停阈值，窗口相对改进低于该值即停，0 关闭，需 --batched",
    )
    parser.add_argument(
        "--stop-lag", type=int, default=100,
        help="平台早停窗口轮数，每窗比一次窗口内最优损失",
    )
    parser.add_argument(
        "--rescue-stop-window", type=int, default=0,
        help="救援衰竭早停窗口，按救援次数开窗审计性价比，0 关闭，需 --batched",
    )
    parser.add_argument(
        "--rescue-stop-tol", type=float, default=0.02,
        help="救援衰竭早停阈值，窗口内损失相对降幅低于该值即停，默认 0.02",
    )
    parser.add_argument(
        "--gpu-entry-budget", type=int, default=0,
        help="GPU 条目流分块预算，大域宽数据压显存峰值用，0 不分块零改变",
    )
    parser.add_argument(
        "--init-margin-tol", type=float, default=0.0,
        help="初始化一阶边缘化覆盖校验相对容差，噪声考卷用 0.05，默认 0 精确",
    )
    parser.add_argument(
        "--init-clean", action="store_true",
        help="初始化一阶计数清洗，截负加等额摊归一到行数，考卷评分不动，默认关",
    )
    parser.add_argument(
        "--init-avg", action="store_true",
        help="初始化一阶边缘化改全表逆方差加权平均，权重一比伙伴原子数，"
        "替代贪心选首张过关表，缺格表跳过，此模式忽略 --init-margin-tol，默认关走原路径",
    )
    parser.add_argument(
        "--init-table", type=str, default="",
        help="从 CSV 载入初始表跳过抽样初始化，字段与行数须与真表一致，"
        "选择循环热启动用，默认空走原路径",
    )
    parser.add_argument(
        "--noise-floor", type=float, default=0.0,
        help="噪声地板，损失低于此值进入追噪区平台阈值切粗，"
        "取 c 乘格子数乘计数 sigma 平方，默认 0 关闭",
    )
    parser.add_argument(
        "--stop-threshold-noisy", type=float, default=0.0,
        help="追噪区平台粗阈值，与 --noise-floor 同时给出，默认 0 关闭",
    )
    parser.add_argument(
        "--progress", type=int, default=0,
        help="每 N 轮打印一行进度并即时刷出，冻结与救援轮无条件打印，0 静默",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="熵校准增益要求系数，方向收益须达最大可达增益的该比例，落在 (0,1)",
    )
    parser.add_argument(
        "--probe-interval", type=int, default=0,
        help="每 N 轮批量轮跑一次 beta 网格探针记旁路日志，纯只读，0 关闭",
    )
    parser.add_argument(
        "--ref-shape", type=str, default="uniform", choices=["uniform", "distance"],
        help="批量轮参考分布形状，uniform 现状均匀签筒，distance 距离衰减签筒锚定同松紧",
    )
    parser.add_argument(
        "--stay", type=float, default=0.9,
        help="批量轮参考分布源候选保持概率，控制每轮先验松紧，落在 (0,1)",
    )
    parser.add_argument(
        "--damping", type=float, default=1.0,
        help="步长阻尼系数，乘在解析最优步长与可行上限取小之后，1.0 为现状，须为正",
    )
    args = parser.parse_args()
    if args.gpu and not args.batched:
        parser.error("--gpu 只支持批量路径，请同时带 --batched")
    if args.init_avg and args.uniform_init:
        parser.error("--init-avg 与 --uniform-init 互斥，二选一")
    if args.init_table and (args.uniform_init or args.init_avg or args.init_clean):
        parser.error("--init-table 与其他初始化开关互斥")
    if args.work_rows > 0 and not args.batched:
        parser.error("--work-rows 只支持批量路径，请同时带 --batched")
    if args.stop_threshold > 0 and not args.batched:
        parser.error("--stop-threshold 只支持批量路径，请同时带 --batched")
    if args.rescue_stop_window > 0 and not args.batched:
        parser.error("--rescue-stop-window 只支持批量路径，请同时带 --batched")
    if not (0 < args.alpha < 1):
        parser.error("--alpha 必须落在 (0,1)")
    if args.probe_interval > 0 and not args.batched:
        parser.error("--probe-interval 只支持批量路径，请同时带 --batched")
    if args.ref_shape != "uniform" and not args.batched:
        parser.error("--ref-shape 只支持批量路径，请同时带 --batched")
    if not (0 < args.stay < 1):
        parser.error("--stay 必须落在 (0,1)")
    if args.stay != 0.9 and not args.batched:
        parser.error("--stay 只支持批量路径，请同时带 --batched")
    if args.work_frac != 0.0:
        if not (0 < args.work_frac < 1):
            parser.error("--work-frac 必须落在 (0,1)")
        if args.work_rows > 0:
            parser.error("--work-frac 与 --work-rows 二选一，不能同时给")
        if not args.batched:
            parser.error("--work-frac 只支持批量路径，请同时带 --batched")
    if args.damping <= 0:
        parser.error("--damping 必须为正")
    if args.damping != 1.0 and not args.batched:
        parser.error("--damping 只支持批量路径，请同时带 --batched")

    data_dir = DATA_ROOT / args.data
    schema, real_rows = load_table(str(data_dir / f"{args.data}.csv"))
    work_rows_val = resolve_work_rows(len(real_rows), args.work_rows, args.work_frac)
    if args.work_frac > 0:
        print(f"工作批比例 {args.work_frac} 按 {len(real_rows)} 行折算名额 {work_rows_val} 行")
    if args.exam:
        exam_path = data_dir / args.exam
    else:
        found = sorted(data_dir.glob("measured_*query.json"))
        if len(found) != 1:
            parser.error(f"数据目录 {data_dir} 下 measured_*query.json 须恰有一个，实有 {len(found)}")
        exam_path = found[0]
    specs = load_queries(str(exam_path))
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    seen = set()
    field_sets = []
    for fs in query_field_sets(specs, schema):
        # 联合修改来源，二到四字段集合去重，半空间的全字段集合不进来源
        if 2 <= len(fs) <= 4 and fs not in seen:
            seen.add(fs)
            field_sets.append(fs)

    registry = StateRegistry(schema, specs)
    if args.gpu:
        # GPU 后端整轮不读注册表特征，惰性登记省掉每轮特征求值与特征矩阵内存
        registry.set_lazy_features(True)
    init_rng = np.random.default_rng(args.init_seed)
    if args.init_table:
        it_schema, init_rows = load_table(args.init_table)
        if it_schema.fields != schema.fields:
            parser.error("--init-table 字段与真表不一致")
        if len(init_rows) != len(real_rows):
            parser.error(
                f"--init-table 行数 {len(init_rows)} 与真表 {len(real_rows)} 不一致"
            )
        for j in range(schema.num_fields):
            allowed = set(schema.domains[j])
            if not {r[j] for r in init_rows} <= allowed:
                parser.error(f"--init-table 字段 {schema.fields[j]} 出现域外值")
    elif args.uniform_init:
        init_rows = [
            tuple(
                schema.domains[j][int(init_rng.integers(len(schema.domains[j])))]
                for j in range(schema.num_fields)
            )
            for _ in range(len(real_rows))
        ]
    else:
        if args.init_avg:
            marginals = derive_first_order_avg(specs, schema, len(real_rows))
        else:
            marginals = derive_first_order(
                specs, schema, len(real_rows), tol_rows=args.init_margin_tol
            )
        if args.init_clean:
            marginals = clean_first_order(marginals, len(real_rows))
        init_rows = sample_initial_rows(marginals, schema, len(real_rows), init_rng)
    ids = registry.register_table(init_rows)

    factory = make_paired_provider if args.pairing else make_edit_provider
    pairing_budget = PairingBudget(max_pairs=args.max_pairs)
    if args.batched:
        provider = make_batch_provider(
            registry, y, w,
            np.random.default_rng(args.menu_seed),
            joint_field_sets=field_sets,
            pairing=args.pairing,
            pairing_backoff=args.pairing_backoff,
            pairing_budget=pairing_budget,
            defer_workload=args.gpu,
            pair_rescue_rows=args.pair_rescue_rows,
            rescue_after=args.rescue_after,
            rescue_gpu=args.gpu,
            work_rows=work_rows_val,
            work_random_frac=args.work_random,
            select_rng=np.random.default_rng(args.select_seed),
            work_below_step=args.work_below,
            retry_select=args.retry_select,
        )
    elif args.grouped:
        provider = make_grouped_provider(
            registry, y, w,
            np.random.default_rng(args.menu_seed),
            joint_field_sets=field_sets,
            pairing=args.pairing,
        )
    else:
        provider = factory(
            registry, y, w,
            np.random.default_rng(args.menu_seed),
            joint_field_sets=field_sets,
        )
    initial_loss = table_loss(registry.build_workload(y, w), ids)
    print(f"行数 {len(ids)}，查询数 {len(specs)}，初始损失 {initial_loss:.6f}")

    t0 = time.perf_counter()
    if args.batched:
        out = evolve_batch(
            ids, args.rounds,
            np.random.default_rng(args.sample_seed),
            provider, registry,
            alpha=args.alpha,
            max_frozen_retries=args.retries,
            backend="gpu" if args.gpu else "cpu",
            stop_threshold=args.stop_threshold,
            stop_lag=args.stop_lag,
            rescue_stop_window=args.rescue_stop_window,
            rescue_stop_tol=args.rescue_stop_tol,
            gpu_entry_budget=args.gpu_entry_budget,
            noise_floor=args.noise_floor,
            stop_threshold_noisy=args.stop_threshold_noisy,
            progress_every=args.progress,
            probe_interval=args.probe_interval,
            ref_shape=args.ref_shape,
            stay_probability=args.stay,
            damping=args.damping,
        )
    elif args.grouped:
        out = evolve_grouped(
            ids, args.rounds,
            np.random.default_rng(args.sample_seed),
            provider,
            alpha=args.alpha,
            max_frozen_retries=args.retries,
        )
    else:
        out = evolve(
            None, ids, args.rounds,
            np.random.default_rng(args.sample_seed),
            support_provider=provider,
            alpha=args.alpha,
            max_frozen_retries=args.retries,
        )
    elapsed = time.perf_counter() - t0

    if args.gpu:
        # 终点损失也在卡上算，避免为一次报数补算全部欠账特征
        from resevo.gpukernel import GpuBatchContext

        gctx = GpuBatchContext(provider.structure[0], y, w)
        final_loss = gctx.loss_of(provider.codebook.sync(), out.state_ids)
    else:
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
                 "expected_loss", "status", "max_gain_sum"]
            )
            for r in out.records:
                writer.writerow(
                    [r.round_index, r.old_loss, r.beta, r.direction_gain,
                     r.interaction, r.step, r.expected_loss, r.status, r.max_gain_sum]
                )
            writer.writerow(["final", final_loss, "", "", "", "", "", out.stop_reason, ""])
        print(f"曲线已写入 {out_path}")
        if out.probes:
            probe_path = out_path.with_name(out_path.stem + "_probe.csv")
            with open(probe_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "beta_used", "J_used", "beta_best", "J_best"])
                for p in out.probes:
                    writer.writerow(
                        [p.round_index, p.beta_used, p.j_used, p.beta_best, p.j_best]
                    )
            print(f"探针已写入 {probe_path}")

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
