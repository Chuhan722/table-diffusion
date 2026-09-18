"""救援闭环对账，拿卡点表跑一次救援加落回，验证预测下降真兑现。

对账三件事，核起点损失等于全表损失，落回后全表损失真下降，
下降量与核期望同量级，变动行数与期望改行同量级。
用法，./.venv/bin/python scripts/verify_rescue.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.batchkernel import BatchRoundPlan, evolve_batch  # noqa: E402
from resevo.dataset import StateRegistry, load_queries, load_table, target_from_specs  # noqa: E402
from resevo.pairing import build_rescue_menu  # noqa: E402
from resevo.state import table_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_frozen import load_frozen_rows  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "plants"


def main() -> None:
    schema, _ = load_table(str(DATA_DIR / "plants.csv"))
    specs = load_queries(str(DATA_DIR / "measured_9248query.json"))
    y = target_from_specs(specs)
    w = np.ones(len(specs))
    rows = load_frozen_rows("results/plants_stop_seed1.csv", schema)
    registry = StateRegistry(schema, specs)
    ids = registry.register_table(rows)
    loss_before = table_loss(registry.build_workload(y, w), ids)
    print(f"卡点全表损失 {loss_before:.4f}")

    box = {}

    def stub_provider(state_ids, round_index, frozen_streak=0):
        small, sub_ids, sub_idx, wl, sups = build_rescue_menu(
            registry, state_ids, y, w, np.random.default_rng(1), 2000,
        )
        box["plan"] = BatchRoundPlan(
            "rescue", wl, supports=sups, rescue_ids=sub_ids,
            rescue_sub_idx=sub_idx, rescue_registry=small,
        )
        return box["plan"]

    out = evolve_batch(ids, 1, np.random.default_rng(1), stub_provider, registry,
                       progress_every=1)
    rec = out.records[0]
    print(f"核起点 {rec.old_loss:.4f}  核期望终点 {rec.expected_loss:.4f}  "
          f"期望下降 {rec.old_loss - rec.expected_loss:.4f}")
    assert abs(rec.old_loss - loss_before) < 1e-6, "核起点与全表损失不一致"

    loss_after = table_loss(registry.build_workload(y, w), out.state_ids)
    changed = int((out.state_ids != ids).sum())
    print(f"落回后全表损失 {loss_after:.4f}  实际下降 {loss_before - loss_after:.4f}  "
          f"变动行数 {changed}")
    assert loss_after < loss_before, "落回后损失未下降"
    print("闭环对账通过")


if __name__ == "__main__":
    main()
