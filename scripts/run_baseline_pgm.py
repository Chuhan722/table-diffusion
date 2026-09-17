"""AIM 文献基线，Private-PGM 零噪声吃考卷全二阶边缘一次性拟合抽样。

完整 AIM 是带隐私预算逐轮自适应选边缘的机制，零噪声下选择环节无意义，
文献对照的标准做法即本脚本，把全部二阶边缘一次性喂给其官方估计器
mbi 的 MirrorDescent 拟合图模型再抽样出表，等加噪版本再跑完整 AIM。

运行环境，借旧仓第三方虚拟环境，只用 mbi 与 jax 官方库，不碰旧仓自研代码，
/home/chuhan/projects/table-diffusion-issue53-bounded-gap-r8/.venv/bin/python \
    scripts/run_baseline_pgm.py --seed 1 --out results/pgm_seed1.csv

喂料，考卷 548 条全二阶答案按 45 个字段对组装成完整边缘测量，
外加从二阶边缘化出的 10 条一阶边缘，stddev 统一取 1 仅作占位，零噪声下不影响最优解。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from mbi import Domain, LinearMeasurement, estimation  # noqa: E402

import baseline_common as bc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=3000)
    args = parser.parse_args()

    fields, cats, bins, pair_tables, total = bc.load_exam_structure()
    margins = bc.marginalize_first_order(fields, cats, pair_tables, total)
    _, domains, _ = bc.load_real_domains()

    domain = Domain(tuple(fields), tuple(len(cats[f]) for f in fields))
    measurements = [
        LinearMeasurement(margins[f].astype(np.float64), (f,), stddev=1.0)
        for f in fields
    ]
    for (fa, fb), table in sorted(pair_tables.items()):
        measurements.append(
            LinearMeasurement(
                table.astype(np.float64).ravel(), (fa, fb), stddev=1.0
            )
        )
    n_cells = sum(int(np.asarray(m.noisy_measurement).size) for m in measurements)
    print(f"测量 {len(measurements)} 条，格数 {n_cells}，域大小 {domain.size()}")

    started = time.time()
    model = estimation.MirrorDescent().estimate(
        domain, measurements, known_total=float(total), iters=args.iters
    )
    fit_seconds = time.time() - started

    fit_diffs = []
    for (fa, fb), table in pair_tables.items():
        proj = np.asarray(model.project((fa, fb)).datavector())
        fit_diffs.append(np.abs(proj - table.ravel()))
    fit_all = np.concatenate(fit_diffs)
    print(
        f"拟合 {fit_seconds:.1f} 秒，模型二阶残差 平均 {fit_all.mean():.4f} "
        f"最大 {fit_all.max():.4f}"
    )

    synth = model.synthetic_data(rows=total)
    codes = {f: np.asarray(synth.data[f]) for f in fields}
    mean_d, max_d = bc.binned_counts(codes, pair_tables, cats)
    print(f"抽样后二阶计数偏差 平均 {mean_d:.3f} 最大 {max_d:.0f}")

    rows = bc.decode_rows(codes, fields, cats, bins, domains["age"], args.seed)
    bc.write_table(Path(args.out), rows, fields)
    print(f"已写出 {args.out}，{len(rows)} 行")


if __name__ == "__main__":
    main()
