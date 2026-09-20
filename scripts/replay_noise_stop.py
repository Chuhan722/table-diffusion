"""噪声版早停回放器，离线扫参不花卡时。

物理口径按偏差原理，合成表损失等于把真表逐格复刻时的损失，
恰为噪声平方和 D，损失降到 D 以下的部分全是追噪，
D 的理论值等于格子数乘计数 sigma 平方，sigma 只由公开机制参数决定，
sigma_count 等于根号下边际数除以 rho，不碰任何真实答案，隐私口径干净。

回放两种判据在已落盘损失曲线上的停点，
一是现役平台判据精确复刻，相邻两个 lag 轮窗口最优损失相对改进低于阈值停，
二是地板判据，损失首次低于 c 乘 D 理论值即停，扫多档 c，
落地形态是两判据并联先到先停，平台保底防不触发跑满。

用法 ./.venv/bin/python scripts/replay_noise_stop.py \
    --curves results/nltcs_eps1_rescue128stopA_curve_seed1.csv ... \
    --epsilon 1 --delta 1e-5
数据集与边际数格子数从曲线文件名里的数据目录自动识别。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from make_noisy_exam import cdp_rho  # noqa: E402


def load_curve(path: Path) -> list[float]:
    losses = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["round"] == "final":
                continue
            losses.append(float(row["old_loss"]))
    return losses


def exam_shape(data: str) -> tuple[int, int]:
    """从真实考卷数出边际数与格子数。"""
    found = sorted((REPO / "data" / data).glob("measured_*query.json"))
    if len(found) != 1:
        raise SystemExit(f"data/{data} 训练卷须恰有一个")
    with open(found[0], encoding="utf-8") as fh:
        payload = json.load(fh)
    pairs = {
        (q["conditions"][0]["attribute"], q["conditions"][1]["attribute"])
        for q in payload["queries"]
    }
    return len(pairs), len(payload["queries"])


def replay_plateau(losses: list[float], threshold: float, lag: int) -> int:
    """精确复刻 batchkernel 平台判据，返回停止轮，不触发返回最后一轮。"""
    prev_best = None
    window_best = None
    for k, loss in enumerate(losses):
        window_best = loss if window_best is None else min(window_best, loss)
        if (k + 1) % lag == 0:
            if prev_best is not None and (
                prev_best <= 0.0 or (prev_best - window_best) / prev_best < threshold
            ):
                return k
            prev_best = window_best
            window_best = None
    return len(losses) - 1


def replay_floor(losses: list[float], floor: float) -> int | None:
    for k, loss in enumerate(losses):
        if loss <= floor:
            return k
    return None


def replay_two_stage(
    losses: list[float], threshold_fine: float, threshold_coarse: float,
    lag: int, floor: float,
) -> tuple[int, str]:
    """两段式，损失高于地板用细阈值，进入追噪区切粗阈值，返回停轮与段名。"""
    prev_best = None
    window_best = None
    for k, loss in enumerate(losses):
        window_best = loss if window_best is None else min(window_best, loss)
        if (k + 1) % lag == 0:
            armed = window_best <= floor
            threshold = threshold_coarse if armed else threshold_fine
            if prev_best is not None and (
                prev_best <= 0.0 or (prev_best - window_best) / prev_best < threshold
            ):
                return k, ("追噪区" if armed else "平台")
            prev_best = window_best
            window_best = None
    return len(losses) - 1, "跑满"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", nargs="+", required=True)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--stop-threshold", type=float, default=1e-4)
    parser.add_argument("--stop-lag", type=int, default=100)
    parser.add_argument(
        "--coefs", type=float, nargs="+",
        default=[0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--two-stage-coefs", type=float, nargs="+", default=[0.25, 0.3, 0.4, 0.5],
    )
    parser.add_argument(
        "--coarse-thresholds", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2],
    )
    args = parser.parse_args()

    rho = cdp_rho(args.epsilon, args.delta)
    shapes: dict[str, tuple[int, int]] = {}
    for curve in args.curves:
        name = Path(curve).name
        m = re.match(r"([a-z0-9]+)_eps", name)
        if not m:
            raise SystemExit(f"曲线名 {name} 识别不出数据集")
        data = m.group(1)
        if data not in shapes:
            shapes[data] = exam_shape(data)
        n_marg, n_cells = shapes[data]
        sigma_sq = n_marg / rho
        floor_d = n_cells * sigma_sq
        losses = load_curve(Path(curve))
        k_plat = replay_plateau(losses, args.stop_threshold, args.stop_lag)
        print(
            f"\n{name}  边际 {n_marg} 格子 {n_cells} 计数sigma {sigma_sq**0.5:.1f} "
            f"D理论 {floor_d:.4g}"
        )
        print(
            f"  平台判据 停轮 {k_plat} 停损失 {losses[k_plat]:.4g} "
            f"占D {losses[k_plat] / floor_d:.3f}"
        )
        for c in args.coefs:
            k_f = replay_floor(losses, c * floor_d)
            if k_f is None:
                print(f"  地板 c={c:<4} 不触发，平台保底")
                continue
            k_stop = min(k_f, k_plat)
            saved = 1.0 - (k_stop + 1) / (k_plat + 1)
            print(
                f"  地板 c={c:<4} 停轮 {k_f} 停损失 {losses[k_f]:.4g} "
                f"并联停轮 {k_stop} 省轮 {saved:.0%}"
            )
        for c in args.two_stage_coefs:
            for coarse in args.coarse_thresholds:
                k2, seg = replay_two_stage(
                    losses, args.stop_threshold, coarse, args.stop_lag, c * floor_d
                )
                saved = 1.0 - (k2 + 1) / (k_plat + 1)
                print(
                    f"  两段 c={c:<4} 粗阈 {coarse:<6} 停轮 {k2} 段 {seg} "
                    f"停损失 {losses[k2]:.4g} 省轮 {saved:.0%}"
                )


if __name__ == "__main__":
    main()
