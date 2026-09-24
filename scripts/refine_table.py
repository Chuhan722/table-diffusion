"""零空间聚簇精修命令行，门槛量测加爬山加停止加零空间自检一条龙。

用法示例:
    python scripts/refine_table.py --input results/nltcs_rescue128stopA_seed1.csv \
        --output results/nltcs_refined_seed1.csv --seed 1

门槛不过（表不欠聚簇）时原样输出，回执打印判定依据。
输出前强制自检全部一阶二阶计数与输入逐格相同，不同则拒绝写出直接报错。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.dataset import load_table  # noqa: E402
from resevo.refine import (  # noqa: E402
    KURT_GATE_DEFAULT,
    STOP_RATE_DEFAULT,
    refine_rows,
    verify_null_space,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="零空间聚簇精修器")
    parser.add_argument("--input", required=True, help="输入表 CSV")
    parser.add_argument("--output", required=True, help="输出表 CSV")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument(
        "--gate", type=float, default=KURT_GATE_DEFAULT,
        help="门槛，嵌入峰度低于该值才施药",
    )
    parser.add_argument(
        "--stop-rate", type=float, default=STOP_RATE_DEFAULT,
        help="停止，单扫接受率跌破该值收手",
    )
    parser.add_argument("--max-sweeps", type=int, default=32, help="扫数上限")
    parser.add_argument(
        "--pairs-per-sweep", type=int, default=0,
        help="每扫抽样字段对数，0 为全部字段对",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="跳过门槛强制施药，验证实验用，常规调用勿开",
    )
    args = parser.parse_args()

    schema, rows = load_table(args.input)
    index = [{v: i for i, v in enumerate(dom)} for dom in schema.domains]
    X0 = np.array(
        [[index[j][row[j]] for j in range(schema.num_fields)] for row in rows],
        dtype=np.int16,
    )
    domain_sizes = [len(dom) for dom in schema.domains]

    def echo(stat):
        print(
            f"扫 {stat.sweep} 接受 {stat.accepted} 提议 {stat.proposals} "
            f"接受率 {stat.rate:.4f} 辛普森 {stat.simpson} 唯一行 {stat.unique_rows}",
            flush=True,
        )

    X1, report = refine_rows(
        X0,
        domain_sizes,
        seed=args.seed,
        kurt_gate=args.gate,
        stop_rate=args.stop_rate,
        max_sweeps=args.max_sweeps,
        pairs_per_sweep=args.pairs_per_sweep,
        force=args.force,
        on_sweep=echo,
    )
    verdict = "施药" if report.treated else "不施药，原样输出"
    if report.forced:
        verdict += "（强制）"
    print(f"门槛 {report.gate:+.3f} 峰度 {report.kurtosis:+.3f} 判定 {verdict}")
    print(f"总扫数 {len(report.sweeps)} 总接受 {report.accepted_total}")

    if not verify_null_space(X0, X1, domain_sizes):
        raise SystemExit("零空间自检失败，一阶或二阶计数发生改变，拒绝写出")
    print("零空间自检通过，全部一阶二阶计数逐格不变")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(schema.fields)
        for r in range(len(rows)):
            writer.writerow(
                schema.domains[c][X1[r, c]] for c in range(schema.num_fields)
            )
    print(f"精修表已写入 {out_path}")


if __name__ == "__main__":
    main()
