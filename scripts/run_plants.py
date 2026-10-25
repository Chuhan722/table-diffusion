"""plants 真数据端到端实验，编辑候选加候选提供器跑满演化并输出损失曲线。

历史基准适配器，会从原表推断 schema/N；不作为隔离的私有生成入口。
只依赖显式公开输入的基础入口为 python -m resevo。

初始表默认按考卷二阶答案边缘化出的一阶比例逐字段独立抽样，
不偷看真实表的联合结构，关联结构留给演化去修，
目标 y 直接取查询负载里的真实答案，权重单位对角，
菜单随机数与抽样随机数分开，冻结重试上限 10。
用法示例，./.venv/bin/python scripts/run_plants.py --rounds 400
"""
from __future__ import annotations

import argparse
import csv
import math
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
from resevo.editspace import EditBudget, make_edit_provider  # noqa: E402
from resevo.engine import evolve  # noqa: E402
from resevo.grouping import evolve_grouped, make_grouped_provider  # noqa: E402
from resevo.initialization import chow_liu_rows, clean_first_order, derive_first_order, derive_first_order_avg, derive_first_order_direct, sample_initial_rows  # noqa: E402
from resevo.pairing import PairingBudget, make_paired_provider  # noqa: E402
from resevo.state import table_loss  # noqa: E402
from resevo.stepsize import validate_step_parameters  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
FIBER_SEED_BUMP = 100003  # 第二段三个随机种子的偏移，与第一段分流


def resolve_work_rows(total_rows: int, work_rows: int, work_frac: float) -> int:
    """工作批名额折算，比例开启时按行数乘比例四舍五入且至少一行。

    work_frac 为 0 时沿用 work_rows 绝对值，两者同时给出由参数校验拦截。
    """
    if work_frac <= 0.0:
        return work_rows
    return max(1, int(total_rows * work_frac + 0.5))


class StageResult:
    """一段演化的产物：终态、注册表、目标与权重、损失与耗时（普通类，测试用 importlib 载入本脚本时 dataclass 会出错）。"""

    def __init__(self, out, registry, y, w, initial_loss, final_loss, elapsed, num_queries):
        self.out = out
        self.registry = registry
        self.y = y
        self.w = w
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.elapsed = elapsed
        self.num_queries = num_queries


def sigma_groups(specs, w: np.ndarray, min_questions: int) -> dict:
    """按 σ 分档与各档地板（工作笔记第六十八步）。

    σ 相同的题为一档（AIM 每轮一个 σ，一卷只有几档）；题数少于 min_questions 的档并入 σ 最近的
    相邻档（小档的噪声实现离期望太远，当不了判据），反复并到每档都够数或只剩一档。
    各档地板 floor_g = ½ Σ_{i∈g} w_i σ_i²，合计恰为整卷地板 ½ Σ w σ²。
    返回 dict(index=每题档号 int64, floors=各档地板, table=各档明细列表)，档按 σ 升序编号。
    """
    sig = np.array([np.nan if q.sigma is None else float(q.sigma) for q in specs], dtype=np.float64)
    if np.isnan(sig).any() or (sig <= 0.0).any():
        raise ValueError("分档地板要求考卷每题带正的 sigma 字段")
    uniq, inv = np.unique(sig, return_inverse=True)
    n_u = np.bincount(inv, minlength=uniq.size)
    f_u = np.bincount(inv, weights=0.5 * w * sig * sig, minlength=uniq.size)
    # 每档 = 若干个 σ 值（unique 编号）的集合，按 σ 升序排列
    groups: list[list[int]] = [[u] for u in range(uniq.size)]

    def count(g):
        return int(sum(n_u[u] for u in g))

    def rep_sigma(g):
        return float(sum(n_u[u] * uniq[u] for u in g) / count(g))

    while len(groups) > 1 and min(count(g) for g in groups) < min_questions:
        i = min(range(len(groups)), key=lambda j: (count(groups[j]), j))
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        j = min(cands, key=lambda c: (abs(rep_sigma(groups[c]) - rep_sigma(groups[i])), count(groups[c])))
        lo, hi = min(i, j), max(i, j)
        groups[lo] = groups[lo] + groups[hi]
        del groups[hi]
    of_unique = np.empty(uniq.size, dtype=np.int64)
    for g_id, g in enumerate(groups):
        for u in g:
            of_unique[u] = g_id
    index = of_unique[inv].astype(np.int64)
    floors = np.bincount(index, weights=0.5 * w * sig * sig, minlength=len(groups))
    table = []
    for g_id, g in enumerate(groups):
        m = index == g_id
        table.append(dict(
            group=g_id, sigmas=[float(uniq[u]) for u in g], questions=int(m.sum()),
            weight_min=float(w[m].min()), weight_max=float(w[m].max()),
            floor=float(floors[g_id]), share=float(floors[g_id] / floors.sum()),
        ))
    return dict(index=index, floors=floors, table=table)


def describe_groups(info: dict, label: str = "") -> None:
    """分档地板明细打印。"""
    print(f"{label}分档地板：{len(info['table'])} 档（σ 相同为一档，小档并入相邻档），合计 {info['floors'].sum():.6f}")
    for t in info["table"]:
        sig_txt = "/".join(f"{s:.4g}" for s in t["sigmas"])
        w_txt = f"{t['weight_min']:.3g}" if t["weight_min"] == t["weight_max"] else f"{t['weight_min']:.3g}~{t['weight_max']:.3g}"
        print(f"  档{t['group']} σ={sig_txt} 题 {t['questions']} 权重 {w_txt} 地板 {t['floor']:.6f}（{t['share']:.1%}）")


def exam_weights(args, parser, specs, verbose: bool = False) -> np.ndarray:
    """考卷题目权重：--exam-weights none 全 1；sigma 为 (σ_min/σ)²（逆方差，最小 σ 归一到 1）。"""
    w = np.ones(len(specs))
    if args.exam_weights == "sigma":
        sigmas = np.array(
            [np.nan if q.sigma is None else q.sigma for q in specs], dtype=np.float64
        )
        if np.isnan(sigmas).any() or (sigmas <= 0.0).any():
            parser.error("--exam-weights sigma 要求考卷每题带正的 sigma 字段")
        w = (sigmas.min() / sigmas) ** 2
        if verbose:
            print(
                f"查询逆方差加权：sigma 范围 {sigmas.min():.4g}~{sigmas.max():.4g}，"
                f"权重范围 {w.min():.4g}~1，题数 {len(specs)}"
            )
    return w


def run_stage(
    args, parser, schema, specs, init_rows, ji_target, work_rows_val,
    rounds: int, menu_seed: int, sample_seed: int, select_seed: int, label: str = "",
    stop_floor: float | None = None, group_floor: dict | None = None,
) -> StageResult:
    """从给定初始表按考卷 specs 演化到停止，一段完整的"建目标→注册→供给→演化→结算"。

    第一段用考卷原题从初始化表起步；纤维阶段的第二段用原题 + 伪测量题从第一段的终表热启动。
    除 rounds、三个随机种子与地板阈值（stop_floor，None 取 args）外全部参数取 args，两段逐字同配置。
    group_floor 为 sigma_groups 的产物时启用分档地板记账（第六十八步，只第一段给）：
    按 args.stop_floor_groups 决定是否按分档地板收工，按 args.snapshot_group_floor 决定是否在全档过线时落快照。
    """
    if stop_floor is None:
        stop_floor = args.stop_floor
    y = target_from_specs(specs)
    w = exam_weights(args, parser, specs, verbose=True)
    if group_floor is not None:
        describe_groups(group_floor, label)
        if args.stop_floor_groups:
            print(f"{label}分档地板停：每档损失都不高于各自地板即停（由最慢的档说了算），兜底平台早停与轮数上限")
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
    ids = registry.register_table(init_rows)

    factory = make_paired_provider if args.pairing else make_edit_provider
    pairing_budget = PairingBudget(max_pairs=args.max_pairs)
    edit_budget = EditBudget(
        max_edit_fields=args.max_edit_fields,
        donor_copies=args.donor_copies, donor_fields_max=args.donor_fields_max,
    )
    if args.batched:
        tree_sampler = None
        if args.tree_every > 0:
            from resevo.rowtree import TreeRowSampler

            tree_sampler = TreeRowSampler().fit(specs, schema)
            print(f"树采样器就绪，骨架边 {len(tree_sampler.cpt)} 条")
        nullswap = None
        if args.nullswap_every > 0:
            from resevo.nullswap import NullSwapPhase

            nullswap = NullSwapPhase(schema)
            print("换位相位就绪，门槛 -0.05 地板 0.6%")
        provider = make_batch_provider(
            registry, y, w,
            np.random.default_rng(menu_seed),
            budget=edit_budget,
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
            select_rng=np.random.default_rng(select_seed),
            work_below_step=args.work_below,
            retry_select=args.retry_select,
            tree_sampler=tree_sampler,
            tree_every=args.tree_every,
            tree_pool=args.tree_pool,
            tree_per_row=args.tree_per_row,
        )
    elif args.grouped:
        provider = make_grouped_provider(
            registry, y, w,
            np.random.default_rng(menu_seed),
            budget=edit_budget,
            joint_field_sets=field_sets,
            pairing=args.pairing,
        )
    else:
        provider = factory(
            registry, y, w,
            np.random.default_rng(menu_seed),
            budget=edit_budget,
            joint_field_sets=field_sets,
        )
    initial_loss = table_loss(registry.build_workload(y, w), ids)
    print(f"{label}行数 {len(ids)}，查询数 {len(specs)}，初始损失 {initial_loss:.6f}")

    shaper = None
    if args.struct_ruler != "off":
        from resevo.structure import StructShaper

        if args.struct_ruler == "ding+ji" and ji_target is None and args.ji_rho > 0.0:
            shaper = StructShaper(
                "ding", [len(d) for d in schema.domains],
                boost=args.struct_boost, gate=args.struct_gate,
            )
        elif args.struct_ruler == "ding+ji":
            from resevo.structure import CompositeShaper

            sizes = [len(d) for d in schema.domains]
            shaper = CompositeShaper([
                StructShaper(
                    "ding", sizes,
                    boost=args.struct_boost, gate=args.struct_gate,
                ),
                StructShaper(
                    "ji", sizes,
                    boost=args.struct_boost2 or args.struct_boost,
                    gate=args.struct_gate, ji_target=ji_target,
                ),
            ])
        else:
            shaper = StructShaper(
                args.struct_ruler,
                [len(d) for d in schema.domains],
                boost=args.struct_boost,
                gate=args.struct_gate,
                ji_target=ji_target,
            )

    t0 = time.perf_counter()
    gpu_direct = {"auto": None, "direct": True, "generic": False}[args.gpu_delta]
    gpu_sparse = {"auto": None, "on": True, "off": False}[args.gpu_sparse]
    snapshot_hook = None
    group_snapshot_hook = None
    snapshot_losses = getattr(args, "snapshot_losses", None)
    want_group_snap = group_floor is not None and getattr(args, "snapshot_group_floor", False)
    if snapshot_losses or want_group_snap:
        from resevo.editspace import tuples_from_ids

        snap_prefix = Path(args.snapshot_prefix + ("_stage2" if label else ""))
        snap_log = snap_prefix.with_name(snap_prefix.name + "_snaps.csv")

        def write_snapshot(index, threshold, round_index, old_loss, ids):
            """快照落盘：起点表写 CSV，触发记录追加到日志，只读注册表。"""
            snap_path = snap_prefix.with_name(snap_prefix.name + f"_snap{index}.csv")
            write_table(snap_path, schema, tuples_from_ids(registry, ids), f"{label}快照{index}")
            snap_log.parent.mkdir(parents=True, exist_ok=True)
            new_log = not snap_log.exists()
            with open(snap_log, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_log:
                    writer.writerow(["index", "threshold", "round", "old_loss", "elapsed_s"])
                writer.writerow([index, threshold, round_index, old_loss, time.perf_counter() - t0])
            print(
                f"{label}快照{index} 阈值 {threshold:.4f} 轮 {round_index} 损失 {old_loss:.4f}",
                flush=True,
            )

    if snapshot_losses:
        def snapshot_hook(index, threshold, round_index, old_loss, ids):
            write_snapshot(index, threshold, round_index, old_loss, ids)

    if want_group_snap:
        def group_snapshot_hook(round_index, old_loss, per, ids):
            """分档全过线快照（第六十八步）：序号记 G，阈值列记各档地板合计。"""
            write_snapshot("G", float(group_floor["floors"].sum()), round_index, old_loss, ids)
            print(f"{label}分档快照 各档损失÷地板 " + " ".join(f"{r:.3f}" for r in per / group_floor["floors"]), flush=True)

    fission_targets = getattr(args, "fission_targets", None) if not label else None
    fission_hook = None
    fission_ids_g: list = []  # 分档全过线轮起点表（终表行算配对差统计用）
    if fission_targets is not None:
        fission_path = Path(args.fission_out)
        fission_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fission_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["round", "elapsed_s", "old_loss", "l1_b", "at_group_floor", "diff_mean", "diff_se"])

        def fission_hook(round_index, old_loss, l1_b, elapsed_s, at_group_floor, diff_mean, diff_se):
            """分裂探针记账（第七十三步）：每条一行追加，分档全过线轮另打一行；参照未设时统计列留空。"""
            with open(fission_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    round_index, f"{elapsed_s:.3f}", old_loss, repr(float(l1_b)), int(at_group_floor),
                    "" if diff_mean is None else repr(float(diff_mean)), "" if diff_se is None else repr(float(diff_se)),
                ])
            if at_group_floor:
                print(f"{label}分裂探针 分档全过线轮 {round_index} 模拟卷 L1_B {l1_b:.6f}", flush=True)

        if want_group_snap:
            _group_snapshot_hook = group_snapshot_hook

            def group_snapshot_hook(round_index, old_loss, per, ids):
                fission_ids_g.append(ids.copy())
                _group_snapshot_hook(round_index, old_loss, per, ids)

    if args.batched:
        out = evolve_batch(
            ids, rounds,
            np.random.default_rng(sample_seed),
            provider, registry,
            alpha=args.alpha,
            max_frozen_retries=args.retries,
            backend="gpu" if args.gpu else "cpu",
            stop_threshold=args.stop_threshold,
            stop_lag=args.stop_lag,
            rescue_stop_window=args.rescue_stop_window,
            rescue_stop_tol=args.rescue_stop_tol,
            gpu_entry_budget=args.gpu_entry_budget,
            gpu_direct=gpu_direct,
            gpu_sparse=gpu_sparse,
            noise_floor=args.noise_floor,
            stop_threshold_noisy=args.stop_threshold_noisy,
            progress_every=args.progress,
            probe_interval=args.probe_interval,
            ref_shape=args.ref_shape,
            stay_probability=args.stay,
            damping=args.damping,
            struct_shaper=shaper,
            treat_stop_tol=args.treat_stop_tol,
            stop_floor=stop_floor,
            nullswap_phase=nullswap,
            nullswap_every=args.nullswap_every,
            nullswap_after=args.nullswap_after,
            snapshot_losses=snapshot_losses if snapshot_hook is not None else None,
            snapshot_hook=snapshot_hook,
            group_index=None if group_floor is None else group_floor["index"],
            group_floors=None if group_floor is None else group_floor["floors"],
            stop_floor_groups=bool(group_floor is not None and args.stop_floor_groups),
            group_snapshot_hook=group_snapshot_hook,
            fission_targets=fission_targets,
            fission_every=args.fission_every if fission_targets is not None else 0,
            fission_hook=fission_hook,
            fission_stop_z=args.fission_stop_z if fission_targets is not None else 0.0,
            fission_stop_runs=args.fission_stop_runs,
        )
    elif args.grouped:
        out = evolve_grouped(
            ids, rounds,
            np.random.default_rng(sample_seed),
            provider,
            alpha=args.alpha,
            max_frozen_retries=args.retries,
        )
    else:
        out = evolve(
            None, ids, rounds,
            np.random.default_rng(sample_seed),
            support_provider=provider,
            alpha=args.alpha,
            max_frozen_retries=args.retries,
        )
    elapsed = time.perf_counter() - t0

    if args.gpu:
        # 终点损失也在卡上算，避免为一次报数补算全部欠账特征
        from resevo.gpukernel import GpuBatchContext

        gctx = GpuBatchContext(
            provider.structure[0], y, w, direct=gpu_direct, sparse=gpu_sparse,
        )
        final_loss = gctx.loss_of(provider.codebook.sync(), out.state_ids)
    else:
        final_workload = registry.build_workload(y, w)
        final_loss = table_loss(final_workload, out.state_ids)
    frozen_rounds = sum(1 for r in out.records if r.status == "no_positive_direction")
    print(
        f"{label}实际轮数 {len(out.records)}，停止原因 {out.stop_reason}，"
        f"冻结轮数 {frozen_rounds}，注册状态数 {registry.num_states}"
    )
    print(
        f"{label}最终损失 {final_loss:.6f}，初始比 {final_loss / initial_loss:.4%}，"
        f"耗时 {elapsed:.1f} 秒，平均每轮 {elapsed / len(out.records) * 1000:.0f} 毫秒"
    )
    if group_floor is not None and out.group_losses:
        k_last, per = out.group_losses[-1]
        ratios = per / group_floor["floors"]
        print(
            f"{label}分档结算（轮 {k_last} 起点）各档损失÷地板 " + " ".join(f"{r:.3f}" for r in ratios)
            + f"，最慢档 {ratios.max():.3f}，总和÷地板 {per.sum() / group_floor['floors'].sum():.3f}"
        )
    if fission_hook is not None:
        # 终表一行在 CPU 上由行重算（answers_for_rows 与卡上答案同为整数计数），轮号记 final；
        # 有全过线快照时另算终表对全过线表的配对差均值与标准误（与离线判停的 z 值同式）
        from resevo.editspace import tuples_from_ids

        q_final = registry.answers_for_rows(tuples_from_ids(registry, out.state_ids))
        e_final = np.abs(fission_targets - q_final)
        l1_final = float(np.mean(e_final))
        diff_cols = ["", ""]
        if fission_ids_g:
            e_g = np.abs(fission_targets - registry.answers_for_rows(tuples_from_ids(registry, fission_ids_g[0])))
            d = e_final - e_g
            diff_cols = [repr(float(np.mean(d))), repr(float(np.std(d, ddof=1) / np.sqrt(d.size))) if d.size > 1 else "0.0"]
        with open(fission_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["final", f"{elapsed:.3f}", final_loss, repr(l1_final), 0, *diff_cols])
        tail = f"，对全过线表 配对差 {float(diff_cols[0]):+.6f} ± {float(diff_cols[1]):.6f}" if fission_ids_g else ""
        print(f"{label}分裂探针 终表 模拟卷 L1_B {l1_final:.6f}（{len(out.records)} 轮）{tail}", flush=True)
    return StageResult(out, registry, y, w, initial_loss, final_loss, elapsed, len(specs))


def weighted_half_loss(y: np.ndarray, answers: np.ndarray, w: np.ndarray) -> float:
    """引擎记账口径的损失 ½ Σ w (y − q)²，与 state.loss_from_residual 同式。"""
    e = np.asarray(y, dtype=np.float64) - np.asarray(answers, dtype=np.float64)
    return float(np.dot(e * w, e) / 2)


def lock_real_targets(schema, specs, stage1: StageResult, stage1_rows) -> dict:
    """第二段真题锁定：原题目标换成第一段终表自己的计数，题面与 σ 不变（工作笔记第六十五步）。

    返回锁定后的题目与核对量：第一段终表对带噪测量的真题损失（按第一段权重，应与第一段终损失一致）。
    """
    from dataclasses import replace

    real_reg = StateRegistry(schema, specs)
    answers = real_reg.answers_for_rows(stage1_rows)
    locked = [replace(q, result=float(a)) for q, a in zip(specs, answers)]
    real_loss_t1 = weighted_half_loss(stage1.y, answers, stage1.w)
    print(
        f"第二段真题锁定：{len(specs)} 题目标换成第一段终表计数；"
        f"第一段终表真题损失 {real_loss_t1:.6f}（第一段结算 {stage1.final_loss:.6f}）"
    )
    return dict(locked_specs=locked, real_reg=real_reg, answers_t1=answers, real_loss_t1=real_loss_t1)


def report_lock_check(args, lock: dict, extra, stage1: StageResult, final: StageResult, final_rows, stage1_rows,
                      stop_floor2: float = 0.0) -> None:
    """锁定模式收工核对：终表对带噪测量的真题损失 ÷ 第一段终表（÷ 地板），第二段起步损失 = 纯伪题损失，
    第二段终损失 ÷ 第二段地板（每题一个人）；写 *_lock.json。"""
    import json

    n_real = len(lock["locked_specs"])
    answers_final = lock["real_reg"].answers_for_rows(final_rows)
    real_loss_final = weighted_half_loss(stage1.y, answers_final, stage1.w)
    pseudo_reg = StateRegistry(final.registry.schema, extra)
    pseudo_init = weighted_half_loss(
        final.y[n_real:], pseudo_reg.answers_for_rows(stage1_rows), final.w[n_real:]
    )
    # 第一段地板：硬地板停取 --stop-floor；分档地板停取各档地板合计（同为 ½Σwσ²），只作报数
    floor1 = args.stop_floor if args.stop_floor > 0 else float(getattr(args, "group_floor_total", 0.0) or 0.0)
    info = dict(
        num_real=n_real, num_pseudo=len(extra),
        stage1_stop_reason=stage1.out.stop_reason, stage1_rounds=len(stage1.out.records),
        stage1_final_loss=stage1.final_loss, stage1_real_loss=lock["real_loss_t1"], stage1_elapsed_s=stage1.elapsed,
        stage2_stop_reason=final.out.stop_reason, stage2_rounds=len(final.out.records),
        stage2_initial_loss=final.initial_loss, stage2_initial_pseudo_loss=pseudo_init,
        stage2_final_loss=final.final_loss, stage2_elapsed_s=final.elapsed,
        stage2_floor=stop_floor2, stage2_over_floor=final.final_loss / stop_floor2 if stop_floor2 > 0 else None,
        final_real_loss=real_loss_final,
        final_over_stage1=real_loss_final / lock["real_loss_t1"] if lock["real_loss_t1"] > 0 else None,
        stop_floor=floor1,
        stage1_over_floor=lock["real_loss_t1"] / floor1 if floor1 > 0 else None,
        final_over_floor=real_loss_final / floor1 if floor1 > 0 else None,
    )
    floor_txt = (
        f"，÷地板 {info['final_over_floor']:.4f}（第一段终表 {info['stage1_over_floor']:.4f}）"
        if floor1 > 0 else ""
    )
    print(
        f"第二段核对：终表真题损失 {real_loss_final:.6f}，是第一段终表的 {info['final_over_stage1']:.4f} 倍{floor_txt}；"
        f"第二段起步损失 {final.initial_loss:.6f} 对纯伪题损失 {pseudo_init:.6f}，"
        f"第二段终损失 ÷ 第二段地板 {info['stage2_over_floor'] if info['stage2_over_floor'] is None else round(info['stage2_over_floor'], 4)}；"
        f"收工 第一段 {info['stage1_stop_reason']}（{info['stage1_rounds']} 轮）/ 第二段 {info['stage2_stop_reason']}（{info['stage2_rounds']} 轮）"
    )
    base = args.save_table or args.out
    if base:
        p = Path(base)
        p = p.with_name(p.stem + "_lock.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        print(f"锁定核对已写 {p}")


def write_curve(out_path: Path, out, final_loss: float, what: str = "曲线") -> None:
    """演化曲线落盘：逐轮记录 + final 行；有探针再写同名 _probe.csv。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        has_temp = out.temps is not None
        header = ["round", "old_loss", "beta", "direction_gain", "interaction",
                  "step", "expected_loss", "status", "max_gain_sum"]
        if has_temp:
            header.append("temp")
        header += ["mode", "candidates", "moved_rows", "elapsed_s"]  # 第六十步记账列
        writer.writerow(header)
        for i, r in enumerate(out.records):
            row = [r.round_index, r.old_loss, r.beta, r.direction_gain,
                   r.interaction, r.step, r.expected_loss, r.status, r.max_gain_sum]
            if has_temp:
                t = out.temps[i]
                row.append(
                    "|".join(f"{x:.6f}" for x in t)
                    if isinstance(t, tuple) else t
                )
            row += [r.mode, r.candidates, r.moved_rows, r.elapsed]
            writer.writerow(row)
        final_row = ["final", final_loss, "", "", "", "", "", out.stop_reason, ""]
        if has_temp:
            final_row.append("")
        final_row += ["", "", "", ""]
        writer.writerow(final_row)
    print(f"{what}已写入 {out_path}")
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


def write_group_curve(out_path: Path, out, group_floor: dict) -> None:
    """分档地板曲线落盘（第六十八步）：逐轮各档损失 + floor 行（各档地板）；旁边写同名 .json 分档明细。"""
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    floors = group_floor["floors"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round"] + [f"g{i}" for i in range(floors.size)])
        for k, per in (out.group_losses or []):
            writer.writerow([k] + [f"{x:.6f}" for x in per])
        writer.writerow(["floor"] + [f"{x:.6f}" for x in floors])
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dict(stop_reason=out.stop_reason, rounds=len(out.records), groups=group_floor["table"]), f, ensure_ascii=False, indent=1)
    print(f"分档曲线已写入 {out_path}，分档明细 {meta_path}")


def write_table(table_path: Path, schema, rows, what: str = "终表") -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(schema.fields)
        for row in rows:
            writer.writerow(row)
    print(f"{what}已写入 {table_path}")


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
        "--treat-stop-tol", type=float, default=0.0,
        help="治疗平台门阈值，平台停触发时在治尺病情深度窗口间改善"
        "不少于该值即放行续跑，0 关门，需结构尺在场",
    )
    parser.add_argument(
        "--stop-floor", type=float, default=0.0,
        help="硬地板停，损失不高于该值直接收工（一致化卷防背错题，"
        "地板按公开机制参数外算），0 关闭",
    )
    parser.add_argument(
        "--stop-floor-groups", action="store_true",
        help="分档地板停（第六十八步）：第一段按 σ 分档，每档损失都不高于各自地板 ½Σwσ² 才收工，"
        "由最慢的档说了算；只管第一段，与 --stop-floor 二选一，需 --batched 且考卷每题带 sigma",
    )
    parser.add_argument(
        "--floor-group-min", type=int, default=30,
        help="分档地板的小档门槛：题数少于该值的 σ 档并入 σ 最近的相邻档",
    )
    parser.add_argument(
        "--snapshot-group-floor", action="store_true",
        help="分档全过线快照：第一段首个各档损失全不高于各自地板的轮，把起点表写成 前缀_snapG.csv"
        "（不停也记，跑到收敛的对照场用），需 --snapshot-prefix 与 --batched",
    )
    parser.add_argument(
        "--fission-exam", type=str, default="",
        help="分裂探针记账（第七十三步，纯记账不改轨迹）：模拟卷 JSON 路径（题目与考卷同序同题面，"
        "只换 result），第一段每 --fission-every 轮与分档全过线轮用起点表算 mean|y_B−q| 写到 --fission-out，"
        "收工再记终表一行；需 --batched，与 --fission-every、--fission-out 同给",
    )
    parser.add_argument(
        "--fission-every", type=int, default=0,
        help="分裂探针记账间隔（轮），0 关闭",
    )
    parser.add_argument(
        "--fission-out", type=str, default="",
        help="分裂探针记账输出 CSV（round, elapsed_s, old_loss, l1_b, at_group_floor, diff_mean, diff_se；"
        "后两列为起点表对全过线表在模拟卷上的逐题误差配对差均值与标准误，全过线前留空）",
    )
    parser.add_argument(
        "--fission-stop-z", type=float, default=0.0,
        help="显著回升停（第七十三步）：分档全过线后，起点表对全过线表的配对差均值 ≥ 该倍数×标准误、"
        "连续 --fission-stop-runs 次记账即收工（停止原因 fission_rise，交触发轮起点表）；"
        "0 关闭；需分裂探针记账与 --snapshot-group-floor",
    )
    parser.add_argument(
        "--fission-stop-runs", type=int, default=3,
        help="显著回升停的连续记账次数（默认 3）",
    )
    parser.add_argument(
        "--snapshot-loss", type=str, default="",
        help="损失快照阈值，逗号分隔的引擎口径损失值，某轮起点损失首次不高于"
        "某阈值时把该轮起点表写成 CSV（第六十一步，纯记账不改轨迹），"
        "留空关闭，需 --batched 与 --snapshot-prefix",
    )
    parser.add_argument(
        "--snapshot-prefix", type=str, default="",
        help="损失快照输出前缀，表写到 前缀_snap{序号}.csv，"
        "触发记录写到 前缀_snaps.csv，配 --snapshot-loss 或 --snapshot-group-floor",
    )
    parser.add_argument(
        "--tree-every", type=int, default=0,
        help="树采样输血轮周期，每该数轮走一次带树行候选的小注册表配对轮，"
        "树由考卷二阶答案搭 Chow-Liu 骨架，0 关闭，需 --batched --gpu "
        "--pairing 且 --pair-rescue-rows 为正",
    )
    parser.add_argument(
        "--tree-pool", type=int, default=256,
        help="每个输血轮的树采样池行数",
    )
    parser.add_argument(
        "--tree-per-row", type=int, default=2,
        help="每行候选混入的树行数",
    )
    parser.add_argument(
        "--nullswap-every", type=int, default=0,
        help="零空间换位相位周期，每该数轮轮首跑一场保二阶四行换位"
        "（恒温器峰度门槛 -0.05 与接受率自熄内置），损失逐位不变"
        "只修高阶聚簇，0 关闭，需 --batched",
    )
    parser.add_argument(
        "--nullswap-after", type=int, default=0,
        help="换位相位暖场轮数，此前不开场，默认 0 即从首个周期点开始",
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
        "--gpu-delta", choices=("auto", "direct", "generic"), default="auto",
        help="GPU 答案差通路，auto 考卷全单元格时走直算否则通用，"
             "direct 强制直算，generic 强制通用，两通路同种子逐位同结果",
    )
    parser.add_argument(
        "--gpu-sparse", choices=("auto", "on", "off"), default="auto",
        help="GPU 稀疏整表项/漂移项（第五十九步 乙），auto 直算布局且无半空间题时开，"
             "on 强制，off 走稠密老路；数学相同、答案残差逐位同，组分与漂移项浮点"
             "求和顺序不同末位可异，轨迹如换种子；要逐位复现旧表用 off",
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
        "--init-oneway", action="store_true",
        help="初始化一阶直接取卷面一阶等值题（每字段整域覆盖，负计数夹零），"
        "不经二阶边缘化，选测录像回放考卷用，默认关走原路径",
    )
    parser.add_argument(
        "--exam-weights", type=str, default="none", choices=["none", "sigma"],
        help="查询权重，none 全一现状零改变；sigma 按每题 sigma 逆方差加权 (sigma_min/sigma)^2，"
        "考卷每题须带 sigma 字段，混 sigma 考卷用",
    )
    parser.add_argument(
        "--oneway-pool", type=float, default=0.0,
        help="一阶题平分清洗（工作笔记第五十七步第 14、15 条）：|y|<z·sigma 的格子视为未测出，"
        "其余保留 max(y,0)，剩余行数在未测出格子间均分，回写为目标；读卷后初始化前做，"
        "纯后处理零预算，考卷须带 sigma。给 z（惯例 2），默认 0 关闭现状逐位不变",
    )
    parser.add_argument(
        "--oneway-pool-scope", type=str, default="all", choices=["all", "uncovered"],
        help="一阶清洗范围：all 全部字段；uncovered 只清洗没被任何 ≥2 阶题覆盖的字段"
        "（被覆盖字段的一阶由二阶测量决定，清洗多余，第五十七步第 17 条），默认 all",
    )
    parser.add_argument(
        "--oneway-pool-cover-ignore-prefix", type=str, default="",
        help="判定覆盖时忽略题号以此开头的题（后处理补进考卷的伪目标如 P2_ 不是测量，不算覆盖），"
        "默认空现状不变；只在 --oneway-pool-scope uncovered 下有意义",
    )
    parser.add_argument(
        "--pair-pool", type=float, default=0.0,
        help="二阶题清洗，同 --oneway-pool 的规则但未测出格子按清洗后一阶乘积分（没证据当独立），"
        "须与 --oneway-pool 同开。给 z，默认 0 关闭",
    )
    parser.add_argument(
        "--fiber-passes", type=int, default=0,
        help="纤维洗牌补没测的对（第五十八、五十九步）：第一段按考卷收敛后，固定所有测过的字段集合逐格不动，"
        "在每个测过集合的纤维内随机重排其余列，洗 K 趟取平均得没测的二阶边缘，以 P2_ 伪测量题并入考卷，"
        "再从第一段终表热启动演化第二段。给趟数 K（批次用 100），默认 0 关闭现状逐位不变",
    )
    parser.add_argument(
        "--fiber-start-pass", type=int, default=2,
        help="纤维洗牌从第几趟起计入平均（第 1 趟仍带第一段终表的偶然结构，批次口径丢弃），默认 2",
    )
    parser.add_argument(
        "--fiber-sigma-mult", type=float, default=1.0,
        help="伪测量题 σ = 考卷二阶题 σ 中位数 × 此倍数（只定 --exam-weights sigma 下的权重），默认 1",
    )
    parser.add_argument(
        "--fiber-seed", type=int, default=-1,
        help="纤维洗牌随机种子，默认 -1 取 --sample-seed",
    )
    parser.add_argument(
        "--fiber-rounds", type=int, default=0,
        help="第二段最大轮数，默认 0 同 --rounds；两段其余配置逐字相同，第二段三个随机种子各加 %d" % FIBER_SEED_BUMP,
    )
    parser.add_argument(
        "--fiber-lock-real", action="store_true",
        help="第二段真题答案锁定（工作笔记第六十五步定稿）：第二段考卷里原题的目标换成第一段终表自己的计数，"
        "不再追带噪测量，只补没测的对、守住测过的边缘；第二段地板改为「每题平均差一个人」½Σw（第六十六步先导场定），"
        "兜底平台早停与轮数上限；收工后记录终表对带噪测量的真题损失与第一段终表之比，另写 *_lock.json。默认关现状逐位不变",
    )
    parser.add_argument(
        "--init-table", type=str, default="",
        help="从 CSV 载入初始表跳过抽样初始化，字段与行数须与真表一致，"
        "选择循环热启动用，默认空走原路径",
    )
    parser.add_argument(
        "--init-tree", action="store_true",
        help="初始化改 Chow-Liu 树，只用卷面二阶答案建互信息最大生成树树上条件抽样，"
        "与其他初始化开关互斥，默认关走一阶独立抽样",
    )
    parser.add_argument(
        "--donor-copies", type=int, default=8,
        help="供体复制路径数，菜单每组抽的供体行数，默认 8 为现状",
    )
    parser.add_argument(
        "--donor-fields-max", type=int, default=2,
        help="供体复制每条路径最多复制的字段数，批量路径表示上限 3，默认 2 为现状",
    )
    parser.add_argument(
        "--donor-rule", action="store_true",
        help="供体档位规则：复用已结算 S*（同一笔学费），过信噪比门且 S*>1/32"
             "（抄 32 份期望命中≥1 份的机理阈值）则自动加码 d32/dfm3，否则守默认",
    )
    parser.add_argument(
        "--max-edit-fields", type=int, default=8,
        help="单属性修改每组抽的字段数上限，0 关闭单字段路径只留保结构编辑，默认 8 为现状",
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
        help="步长阻尼系数，乘在解析最优步长与可行上限取小之后，须有限且落在 (0,1]，默认 1.0",
    )
    parser.add_argument(
        "--struct-ruler", type=str, default="off",
        choices=["off", "watch", "jia", "bing", "ding", "ji", "ding+ji"],
        help="结构签筒第二级，off 现状零改变，watch 只逐轮记体温不干预，"
        "jia 整行抱团尺，bing 表形状尺，ding 连续恒温尺（bing 无门槛版，"
        "力度随烧度连续，--struct-gate 转义为满速刻度），ji 己尺聚簇尺"
        "（jia 刻度+确诊制+双向连续药力，基准为真表 Simpson，"
        "零噪场精确值，加噪场须另花预算买），ding+ji 组合尺"
        "（两尺各自确诊各自停药，签数因子相乘），需 --batched",
    )
    parser.add_argument(
        "--struct-boost", type=float, default=4.0,
        help="结构签数增幅，变好方向乘该数变坏除该数，须大于 1，默认 4",
    )
    parser.add_argument(
        "--struct-boost2", type=float, default=0.0,
        help="组合尺第二把（己尺）的签数增幅，0 时沿用 --struct-boost",
    )
    parser.add_argument(
        "--struct-gate", type=float, default=-0.05,
        help="恒温开关门槛，体温计（嵌入峰度）低于该值才启用第二级，默认 -0.05",
    )
    parser.add_argument(
        "--ji-rho", type=float, default=0.0,
        help="己尺基准的 zCDP 预算，>0 时对真表 Simpson 加高斯噪"
        "（单行敏感度 2/n），默认 0 为零噪场精确口径",
    )
    args = parser.parse_args()
    if args.gpu and not args.batched:
        parser.error("--gpu 只支持批量路径，请同时带 --batched")
    if args.init_avg and args.uniform_init:
        parser.error("--init-avg 与 --uniform-init 互斥，二选一")
    if args.init_table and (args.uniform_init or args.init_avg or args.init_clean):
        parser.error("--init-table 与其他初始化开关互斥")
    if args.init_oneway and (
        args.uniform_init or args.init_avg or args.init_table or args.init_tree
    ):
        parser.error("--init-oneway 与其他初始化开关互斥")
    if args.init_tree and (
        args.uniform_init or args.init_avg or args.init_clean or args.init_table
    ):
        parser.error("--init-tree 与其他初始化开关互斥")
    if args.donor_copies < 0:
        parser.error("--donor-copies 不能为负")
    if not (1 <= args.donor_fields_max <= 3):
        parser.error("--donor-fields-max 必须落在 1 到 3，批量路径表示上限 3")
    if args.max_edit_fields < 0:
        parser.error("--max-edit-fields 不能为负")
    if args.work_rows > 0 and not args.batched:
        parser.error("--work-rows 只支持批量路径，请同时带 --batched")
    if args.stop_threshold > 0 and not args.batched:
        parser.error("--stop-threshold 只支持批量路径，请同时带 --batched")
    if args.rescue_stop_window > 0 and not args.batched:
        parser.error("--rescue-stop-window 只支持批量路径，请同时带 --batched")
    if args.nullswap_every > 0 and not args.batched:
        parser.error("--nullswap-every 只支持批量路径，请同时带 --batched")
    if args.nullswap_after > 0 and args.nullswap_every <= 0:
        parser.error("--nullswap-after 需要 --nullswap-every 为正")
    if bool(args.snapshot_loss) and not args.snapshot_prefix:
        parser.error("--snapshot-loss 需要 --snapshot-prefix")
    if args.snapshot_group_floor and not args.snapshot_prefix:
        parser.error("--snapshot-group-floor 需要 --snapshot-prefix")
    if args.snapshot_prefix and not (args.snapshot_loss or args.snapshot_group_floor):
        parser.error("--snapshot-prefix 需配 --snapshot-loss 或 --snapshot-group-floor")
    if (args.snapshot_loss or args.snapshot_group_floor) and not args.batched:
        parser.error("快照只支持批量路径，请同时带 --batched")
    fission_on = bool(args.fission_exam) or args.fission_every != 0 or bool(args.fission_out)
    if fission_on and not (args.fission_exam and args.fission_every > 0 and args.fission_out):
        parser.error("--fission-exam、--fission-every（正整数）、--fission-out 须同时给")
    if fission_on and not args.batched:
        parser.error("分裂探针记账只支持批量路径，请同时带 --batched")
    if args.fission_stop_z < 0.0 or not np.isfinite(args.fission_stop_z):
        parser.error("--fission-stop-z 须为非负有限数")
    if args.fission_stop_runs < 1:
        parser.error("--fission-stop-runs 须为正整数")
    if args.fission_stop_z > 0.0 and not (fission_on and args.snapshot_group_floor):
        parser.error("--fission-stop-z 需分裂探针记账（--fission-exam 等）与 --snapshot-group-floor")
    if args.stop_floor_groups and not args.batched:
        parser.error("--stop-floor-groups 只支持批量路径，请同时带 --batched")
    if args.stop_floor_groups and args.stop_floor > 0.0:
        parser.error("--stop-floor-groups 与 --stop-floor 二选一")
    if args.floor_group_min < 1:
        parser.error("--floor-group-min 必须为正")
    snapshot_losses = None
    if args.snapshot_loss:
        try:
            snapshot_losses = [float(x) for x in args.snapshot_loss.split(",") if x.strip()]
        except ValueError:
            parser.error("--snapshot-loss 须为逗号分隔的数值")
        if not snapshot_losses or any(x <= 0.0 for x in snapshot_losses):
            parser.error("--snapshot-loss 的每个阈值必须为正")
    args.snapshot_losses = snapshot_losses
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
    try:
        validate_step_parameters(args.damping)
    except ValueError as exc:
        parser.error(f"--{exc}")
    if args.damping != 1.0 and not args.batched:
        parser.error("--damping 只支持批量路径，请同时带 --batched")
    if args.struct_ruler != "off" and not args.batched:
        parser.error("--struct-ruler 只支持批量路径，请同时带 --batched")
    if args.struct_boost <= 1.0:
        parser.error("--struct-boost 必须大于 1")

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
    if args.pair_pool > 0.0 and args.oneway_pool <= 0.0:
        parser.error("--pair-pool 须与 --oneway-pool 同开")
    if args.oneway_pool > 0.0:
        from resevo.denoise import pool_oneway, pool_pairs

        if any(q.sigma is None for q in specs):
            parser.error("--oneway-pool 要求考卷每题带 sigma 字段")
        specs, pooled_ones, st = pool_oneway(
            specs, schema, len(real_rows), args.oneway_pool, args.oneway_pool_scope,
            args.oneway_pool_cover_ignore_prefix,
        )
        print(
            f"一阶平分清洗 z={args.oneway_pool:g} 范围 {args.oneway_pool_scope}：字段 {st['fields']}，格子 {st['cells']}，"
            f"未测出格子 {st['pooled']}（{st['pooled'] / max(st['cells'], 1):.1%}）"
        )
        if args.pair_pool > 0.0:
            specs, st2 = pool_pairs(specs, schema, len(real_rows), args.pair_pool, pooled_ones)
            print(
                f"二阶独立清洗 z={args.pair_pool:g}：对 {st2['pairs']}，格子 {st2['cells']}，"
                f"未测出格子 {st2['pooled']}（{st2['pooled'] / max(st2['cells'], 1):.1%}）"
            )
    args.fission_targets = None
    if args.fission_exam:
        fpath = Path(args.fission_exam)
        if not fpath.exists():
            fpath = data_dir / args.fission_exam
        if not fpath.exists():
            parser.error(f"--fission-exam 找不到 {args.fission_exam}")
        fspecs = load_queries(str(fpath))
        if len(fspecs) != len(specs):
            parser.error(f"模拟卷题数 {len(fspecs)} 与考卷 {len(specs)} 不一致")
        for a, b in zip(fspecs, specs):
            if a.query_id != b.query_id or a.conditions != b.conditions:
                parser.error(f"模拟卷与考卷题目不对齐：{a.query_id} 对 {b.query_id}")
        args.fission_targets = np.array([q.result for q in fspecs], dtype=np.float64)
        print(f"分裂探针记账：模拟卷 {fpath.name} {len(fspecs)} 题，每 {args.fission_every} 轮与分档全过线轮记一次")
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
    elif args.init_tree:
        init_rows = chow_liu_rows(
            specs, schema, len(real_rows), init_rng, tol_rows=args.init_margin_tol
        )
    else:
        if args.init_oneway:
            marginals = derive_first_order_direct(specs, schema)
        elif args.init_avg:
            marginals = derive_first_order_avg(specs, schema, len(real_rows))
        else:
            marginals = derive_first_order(
                specs, schema, len(real_rows), tol_rows=args.init_margin_tol
            )
        if args.init_clean:
            marginals = clean_first_order(marginals, len(real_rows))
        init_rows = sample_initial_rows(marginals, schema, len(real_rows), init_rng)
    # 己尺基准 S* 统一结算（买断一次，己尺与供体档位共用同一读数）：
    # --ji-rho 0 为零噪场精确值；>0 为部署口径，高斯机制买带噪标量（敏感度 2/n）
    ji_target = None
    if args.struct_ruler in ("ji", "ding+ji") or args.donor_rule:
        from collections import Counter

        cnts = np.array(list(Counter(real_rows).values()), dtype=np.float64)
        ji_target = float((cnts**2).sum() / (len(real_rows) ** 2))
        if args.ji_rho > 0.0:
            sigma = (2.0 / len(real_rows)) / math.sqrt(2.0 * args.ji_rho)
            noise_rng = np.random.default_rng(77000 + args.init_seed)
            noisy = ji_target + float(noise_rng.normal(0.0, sigma))
            # 物理下界：全行唯一表的 Simpson，防负基准
            ji_target = max(noisy, 1.0 / len(real_rows))
            snr = ji_target / sigma
            print(
                f"己尺基准（买入口径）ρ={args.ji_rho:.4e} σ={sigma:.3e} "
                f"S*={ji_target:.6e} 信噪比={snr:.2f}"
            )
            # 信噪比门：买到的基准须统计显著（S* > 3σ）才可信，
            # 判据只依赖已购标量与公开 σ，零额外预算；不过门弃用己尺
            if snr <= 3.0:
                ji_target = None
                print("信噪比门：S* ≤ 3σ，基准不可信，己尺弃用（学费沉没）")
        else:
            print(f"己尺基准（零噪精确口径）S*={ji_target:.6e}")
    if args.donor_rule:
        if ji_target is not None and ji_target > 1.0 / 32.0:
            args.donor_copies = 32
            args.donor_fields_max = 3
            print(f"供体档位门：S*={ji_target:.4e} > 1/32，高重复表，供体加码 d32/dfm3")
        else:
            print("供体档位门：S* 不可信或低重复，供体守默认档")

    group_floor = None
    if args.stop_floor_groups or args.snapshot_group_floor:
        try:
            group_floor = sigma_groups(specs, exam_weights(args, parser, specs), args.floor_group_min)
        except ValueError as exc:
            parser.error(str(exc))
        args.group_floor_total = float(group_floor["floors"].sum())
    stage1 = run_stage(
        args, parser, schema, specs, init_rows, ji_target, work_rows_val,
        args.rounds, args.menu_seed, args.sample_seed, args.select_seed,
        group_floor=group_floor,
    )
    final = stage1
    stage1_rows = None
    if args.fiber_passes > 0:
        from resevo.editspace import tuples_from_ids
        from resevo.fiber import encode_rows, fiber_average, fiber_specs, measured_sets, pseudo_sigma

        stage1_rows = tuples_from_ids(stage1.registry, stage1.out.state_ids)
        t_f = time.perf_counter()
        sets = measured_sets(specs, schema)
        doms = [len(d) for d in schema.domains]
        fiber_seed = args.sample_seed if args.fiber_seed < 0 else args.fiber_seed
        res = fiber_average(
            encode_rows(stage1_rows, schema), doms, sets, args.fiber_passes,
            np.random.default_rng(fiber_seed), args.fiber_start_pass,
        )
        sigma = pseudo_sigma(specs, args.fiber_sigma_mult)
        if args.exam_weights == "sigma" and sigma is None:
            parser.error("--fiber-passes 配 --exam-weights sigma 要求考卷有带 sigma 的二阶题以定伪测量 σ")
        extra = fiber_specs(res, schema, sigma)
        print(
            f"纤维洗牌 {res.passes} 趟（第 {args.fiber_start_pass} 趟起取 {res.averaged} 趟平均）耗时 {time.perf_counter() - t_f:.1f} 秒："
            f"测过集合 {len(sets)} 个逐格不动，没测的对 {len(res.unmeasured_pairs)}/{len(res.unmeasured_pairs) + len(res.measured_pairs)} 补伪测量 {len(extra)} 题，"
            f"伪 σ={'无' if sigma is None else f'{sigma:.4g}'}（二阶 σ 中位数 ×{args.fiber_sigma_mult:g}），种子 {fiber_seed}"
        )
        if not extra:
            # 全二阶考卷（如对 GSD 的卷）每对都测过，无题可补：第二段不跑，产物与单段逐位同
            print("没测的对为 0，无伪测量可补，跳过第二段")
            stage1_rows = None
        else:
            # 第一段的卡上对象先放掉，第二段题数更多
            stage1.registry = None
            if args.gpu:
                import gc

                gc.collect()
                from resevo.gpukernel import _cupy

                _cupy().get_default_memory_pool().free_all_blocks()
            specs2 = list(specs) + extra
            rounds2 = args.fiber_rounds if args.fiber_rounds > 0 else args.rounds
            lock = None
            stop_floor2 = 0.0
            if args.fiber_lock_real:
                lock = lock_real_targets(schema, specs, stage1, stage1_rows)
                specs2 = lock["locked_specs"] + extra
                # 第二段地板 = 每题平均差一个人：½ Σ w·1²。伪题目标是洗牌平均（非整数、无测量噪声），
                # 这是加权平均平方误差尺度的启发式，不保证逐题误差小于一人；第一段噪声地板不适用
                stop_floor2 = 0.5 * float(exam_weights(args, parser, specs2).sum())
                print(f"第二段地板停：每题平均差一个人即停，阈值 ½Σw = {stop_floor2:.6f}（{len(specs2)} 题），兜底平台早停与轮数上限")
            elif args.stop_floor > 0.0:
                # 伪测量题没有地板，第一段的阈值对含伪题的总损失无意义，第二段只留平台停与轮数上限
                print("第二段地板停关闭（伪测量题无地板），只留平台早停与轮数上限")
            final = run_stage(
                args, parser, schema, specs2, stage1_rows, ji_target, work_rows_val,
                rounds2, args.menu_seed + FIBER_SEED_BUMP, args.sample_seed + FIBER_SEED_BUMP,
                args.select_seed + FIBER_SEED_BUMP, label="第二段 ", stop_floor=stop_floor2,
            )
            if lock is not None:
                from resevo.editspace import tuples_from_ids as _tuples

                report_lock_check(args, lock, extra, stage1, final, _tuples(final.registry, final.out.state_ids), stage1_rows, stop_floor2)
    out, registry, y, w = final.out, final.registry, final.y, final.w
    initial_loss, final_loss, elapsed = final.initial_loss, final.final_loss, final.elapsed
    checkpoints = [r for r in out.records if r.round_index % max(1, len(out.records) // 10) == 0]
    for r in checkpoints:
        print(
            f"  轮 {r.round_index:4d}  损失 {r.old_loss:12.4f}  beta {r.beta:10.4f}  "
            f"步长 {r.step:8.5f}  期望损失 {r.expected_loss:12.4f}  {r.status}"
        )

    if args.out:
        out_path = Path(args.out)
        if stage1_rows is not None:
            write_curve(out_path.with_name(out_path.stem + "_stage1" + out_path.suffix), stage1.out, stage1.final_loss, "第一段曲线")
        write_curve(out_path, out, final_loss, "曲线")
        if group_floor is not None:
            write_group_curve(out_path.with_name(out_path.stem + "_groups" + out_path.suffix), stage1.out, group_floor)

    if args.save_table:
        from resevo.editspace import tuples_from_ids

        table_path = Path(args.save_table)
        if stage1_rows is not None:
            write_table(table_path.with_name(table_path.stem + "_stage1" + table_path.suffix), schema, stage1_rows, "第一段终表")
        write_table(table_path, schema, tuples_from_ids(registry, out.state_ids), "终表")


if __name__ == "__main__":
    main()
