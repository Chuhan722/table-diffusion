"""冻结状态筛查，b.md 第六部分的代表状态成本收益表。

拿卡住的终表当考题，每种候选菜单变体只建一次核不抽样，
报有无正方向，预测下降即 old_loss 减 expected_loss，与耗时，
不跑长跑就回答哪种候选构造能解开方向耗尽。
变体，control 现行 32 条随机菜单，k64 预算翻倍，enum 二元字段单翻转全枚举，
directed 正残差查询定向双字段修复，enum+directed 两者并集，pairing 行级配对回退。
用法，./.venv/bin/python scripts/screen_frozen.py --table results/plants_stop_seed1.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.batchkernel import build_batch_kernel  # noqa: E402
from resevo.batchmenu import (  # noqa: E402
    CodeBook,
    _normalize_menu,
    build_query_structure,
    generate_batch_menu,
)
from resevo.dataset import (  # noqa: E402
    StateRegistry,
    load_queries,
    load_table,
    query_field_sets,
    target_from_specs,
)
from resevo.editspace import EditBudget  # noqa: E402
from resevo.engine import build_kernel  # noqa: E402
from resevo.grouping import group_state_ids, grouped_residual  # noqa: E402
from resevo.pairing import make_paired_provider  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "plants"


def load_frozen_rows(path: str, schema) -> list[tuple[str, ...]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        if header != schema.fields:
            raise ValueError("冻结表表头与数据集 schema 不一致")
        return [tuple(row) for row in reader]


def enum_flip_pieces(codes_g, domain_sizes):
    """二元字段单翻转全枚举，每组每个可翻字段恰一条路径。"""
    flip_fields = np.flatnonzero(domain_sizes == 2).astype(np.int32)
    num_groups = codes_g.shape[0]
    n_fields = len(flip_fields)
    group = np.repeat(np.arange(num_groups, dtype=np.int64), n_fields)
    fields = np.full((num_groups * n_fields, 3), -1, dtype=np.int32)
    values = np.full((num_groups * n_fields, 3), -1, dtype=np.int32)
    fields[:, 0] = np.tile(flip_fields, num_groups)
    values[:, 0] = (1 - codes_g[:, flip_fields]).ravel()
    return group, fields, values


def directed_pieces(specs, schema, codebook, codes_g, counts, residual,
                    per_query: int, max_paths: int):
    """正残差等值查询的定向双字段修复，两条件都不满足的组一次翻齐。

    负残差只需拆单条件即单字段翻转，已被全枚举覆盖，这里不重复发。
    """
    fidx = {name: j for j, name in enumerate(schema.fields)}
    pieces_g, pieces_f, pieces_v = [], [], []
    total = 0
    for t in np.argsort(-residual):
        if residual[t] <= 0 or total >= max_paths:
            break
        conds = specs[t].conditions
        if len(conds) != 2 or any(c.get("operator") != "==" for c in conds):
            continue
        j1, j2 = fidx[conds[0]["attribute"]], fidx[conds[1]["attribute"]]
        v1 = codebook.value_maps[j1][conds[0]["value"]]
        v2 = codebook.value_maps[j2][conds[1]["value"]]
        fail_both = np.flatnonzero((codes_g[:, j1] != v1) & (codes_g[:, j2] != v2))
        if len(fail_both) == 0:
            continue
        if len(fail_both) > per_query:
            # 优先重数大的组，一条路径覆盖更多行
            fail_both = fail_both[np.argsort(-counts[fail_both])[:per_query]]
        n = len(fail_both)
        fields = np.full((n, 3), -1, dtype=np.int32)
        values = np.full((n, 3), -1, dtype=np.int32)
        fields[:, 0], fields[:, 1] = j1, j2
        values[:, 0], values[:, 1] = v1, v2
        pieces_g.append(fail_both.astype(np.int64))
        pieces_f.append(fields)
        pieces_v.append(values)
        total += n
    if not pieces_g:
        empty_g = np.empty(0, dtype=np.int64)
        empty = np.empty((0, 3), dtype=np.int32)
        return empty_g, empty, empty
    return (np.concatenate(pieces_g), np.concatenate(pieces_f),
            np.concatenate(pieces_v))


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结状态菜单变体筛查")
    parser.add_argument("--table", type=str, default="results/plants_stop_seed1.csv")
    parser.add_argument("--menu-seeds", type=str, default="1,2,3",
                        help="随机菜单变体的种子列表，逗号分隔")
    parser.add_argument("--per-query", type=int, default=256,
                        help="定向变体每条查询最多发的组数")
    parser.add_argument("--max-paths", type=int, default=300000,
                        help="定向变体路径总数上限")
    parser.add_argument("--pairing", action="store_true", help="加测行级配对回退变体，较慢")
    parser.add_argument("--pair-rows", type=int, default=2000,
                        help="配对变体抽样行数，全表行级稠密特征会耗尽内存")
    parser.add_argument("--pair-seed", type=int, default=1, help="配对抽样与菜单种子")
    parser.add_argument("--skip-menus", action="store_true", help="跳过菜单变体只跑配对")
    parser.add_argument("--out", type=str, default="", help="结果 CSV 输出路径，留空不写")
    args = parser.parse_args()

    schema, _ = load_table(str(DATA_DIR / "plants.csv"))
    specs = load_queries(str(DATA_DIR / "measured_9248query.json"))
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    rows = load_frozen_rows(args.table, schema)

    registry = StateRegistry(schema, specs)
    ids = registry.register_table(rows)
    codebook = CodeBook(registry)
    qs = build_query_structure(registry, codebook)
    grouped = group_state_ids(ids)
    codes = codebook.sync()
    codes_g = codes[grouped.unique_ids]
    workload = registry.build_workload(y, w)
    residual = grouped_residual(workload, grouped)
    half_loss = float(residual @ (w * residual)) / 2.0
    print(f"行数 {len(ids)}，组数 {grouped.num_groups}，查询 {len(specs)}，"
          f"冻结表损失 {half_loss:.1f}，引擎半平方口径")

    seen = set()
    field_sets = []
    for fs in query_field_sets(specs, schema):
        if 2 <= len(fs) <= 4 and fs not in seen:
            seen.add(fs)
            field_sets.append(fs)

    results = []

    def run_batch(tag: str, menu, menu_ms: float) -> None:
        t0 = time.perf_counter()
        res = build_batch_kernel(workload, grouped, menu, qs, codes)
        kernel_ms = (time.perf_counter() - t0) * 1000
        drop = res.old_loss - res.expected_loss
        results.append(
            dict(variant=tag, paths=menu.num_paths, menu_ms=round(menu_ms),
                 kernel_ms=round(kernel_ms), status=res.status,
                 direction_gain=res.direction_gain, interaction=res.interaction,
                 step=res.step, predicted_drop=drop,
                 expected_changed_rows=res.expected_changed_rows)
        )
        print(f"{tag:>16}  路径 {menu.num_paths:>8}  菜单 {menu_ms:6.0f}ms  "
              f"核 {kernel_ms:6.0f}ms  {res.status:<22}  步长 {res.step:.5f}  "
              f"预测下降 {drop:.4f}  期望改行 {res.expected_changed_rows:.2f}")

    seeds = [int(s) for s in args.menu_seeds.split(",") if s]
    if not args.skip_menus:
        for seed in seeds:
            t0 = time.perf_counter()
            menu = generate_batch_menu(codes_g, grouped.counts, codebook.domain_sizes,
                                       np.random.default_rng(seed), None, field_sets)
            run_batch(f"control_s{seed}", menu, (time.perf_counter() - t0) * 1000)
        for seed in seeds:
            t0 = time.perf_counter()
            budget = EditBudget(max_edit_fields=16, donor_copies=16,
                                joint_edits=8, explore_edits=8)
            menu = generate_batch_menu(codes_g, grouped.counts, codebook.domain_sizes,
                                       np.random.default_rng(seed), budget, field_sets)
            run_batch(f"k64_s{seed}", menu, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        eg, ef, ev = enum_flip_pieces(codes_g, codebook.domain_sizes)
        enum_menu = _normalize_menu(eg.copy(), ef.copy(), ev.copy(), codes_g,
                                    grouped.num_groups)
        run_batch("enum", enum_menu, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        dg, df, dv = directed_pieces(specs, schema, codebook, codes_g, grouped.counts,
                                     residual, args.per_query, args.max_paths)
        directed_menu = _normalize_menu(dg.copy(), df.copy(), dv.copy(), codes_g,
                                        grouped.num_groups)
        run_batch("directed", directed_menu, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ug = np.concatenate([eg, dg])
        uf = np.concatenate([ef, df])
        uv = np.concatenate([ev, dv])
        union_menu = _normalize_menu(ug, uf, uv, codes_g, grouped.num_groups)
        run_batch("enum+directed", union_menu, (time.perf_counter() - t0) * 1000)

    if args.pairing:
        # 全表行级稠密特征约四十 GB 必爆内存，改为抽样行 + 伪目标修正，
        # 伪目标 y' = 全表残差 + 子集贡献，使子表残差恒等于全表卡点残差，
        # Gamma 与损失口径均与全表一致，只是配对搭档限于抽到的行。
        t0 = time.perf_counter()
        prng = np.random.default_rng(args.pair_seed)
        n_sub = min(args.pair_rows, len(ids))
        sub_ids = ids[np.sort(prng.choice(len(ids), size=n_sub, replace=False))]
        y_prime = residual + workload.features[sub_ids].sum(axis=0)
        provider = make_paired_provider(registry, y_prime, w, prng,
                                        joint_field_sets=field_sets)
        wl, supports = provider(sub_ids, 0, 1)
        menu_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        res = build_kernel(wl, sub_ids, supports, 0.9, 0.5, 1.0, None)
        kernel_ms = (time.perf_counter() - t0) * 1000
        drop = res.old_loss - res.expected_loss
        tag = f"pairing_{n_sub}r_s{args.pair_seed}"
        results.append(
            dict(variant=tag, paths=len(supports), menu_ms=round(menu_ms),
                 kernel_ms=round(kernel_ms), status=res.status,
                 direction_gain=res.direction_gain, interaction=res.interaction,
                 step=res.step, predicted_drop=drop,
                 expected_changed_rows=res.expected_changed_rows)
        )
        print(f"{tag:>16}  块数 {len(supports):>8}  菜单 {menu_ms:6.0f}ms  "
              f"核 {kernel_ms:6.0f}ms  {res.status:<22}  步长 {res.step:.5f}  "
              f"预测下降 {drop:.4f}  期望改行 {res.expected_changed_rows:.2f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
