"""
构建初始化用的 1-way 边缘测量文件（离线，一次性）

## 这个脚本做什么

读 schema + 查询 + 源数据，产出 configs/<dataset>/init_marginals.json：
- 数值属性：用 marginals.derive_bins_from_queries 从查询造箱，再在源数据上数每箱人数
- 类别属性：每个合法值一个箱，数每个值的人数

产出的文件供运行时 marginals.init_from_marginals 确定性初始化 S_0 用。

## 【警告】此脚本读取源数据

数"每箱/每值人数"是离线预计算的 DP 测量接口（现在噪声=0，DP 阶段在此加噪）。
这与铁律 6 不冲突：铁律 6 管的是"运行期不读私有答案"。运行时（run.py →
run_evolution → init_from_marginals）只读本脚本产出的 json，从不碰源数据。

查询变了必须重跑本脚本重新生成（箱是查询诱导的）。文件里记了 queries_source
便于审计。

## 用法

    conda run -p ./.conda python scripts/build_marginals.py

参数写死在下面常量区（与 run.py 风格一致），改完再跑。
"""
import json
import os

import numpy as np
import pandas as pd

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries, load_data, eval_condition
from table_diffevo.marginals import derive_bins_from_queries


# ========== 参数配置（改这里） ==========
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
DATA_PATH = "data/nltcs/nltcs.csv"
OUT_PATH = "configs/nltcs/init_marginals.json"
DATASET_NAME = "nltcs"
# =======================================


def _count_numeric_bins(col: pd.Series, bins):
    """数每个整数闭箱 [lo, hi] 落入的记录数。"""
    counts = []
    for lo, hi in bins:
        mask = (col >= lo) & (col <= hi)
        counts.append(int(mask.sum()))
    return counts


def _count_categorical_values(col: pd.Series, values):
    """数每个合法值的记录数（按 schema 值顺序）。列值按字符串对齐比较，
    与 queries.eval_condition 对类别列 == 的 _coerce_to_column_type 语义一致。"""
    is_numeric = pd.api.types.is_numeric_dtype(col)
    counts = []
    for v in values:
        if is_numeric:
            cmp = pd.to_numeric(v)
            mask = col == cmp
        else:
            mask = col.astype(str) == str(v)
        counts.append(int(mask.sum()))
    return counts


def build():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    df = load_data(DATA_PATH)  # 【读源数据】离线测量

    # 数值属性：workload 诱导分箱
    numeric_bins = derive_bins_from_queries(queries, schema, verbose=True)

    attributes = {}
    for attr in schema.attributes:
        name = attr.name
        if attr.is_numeric():
            if name not in numeric_bins:
                # 无查询覆盖：不写，运行时该列回退随机
                continue
            bins = numeric_bins[name]
            counts = _count_numeric_bins(df[name], bins)
            attributes[name] = {
                "type": "numeric",
                "bins": [[lo, hi] for lo, hi in bins],
                "counts": counts,
            }
        else:
            counts = _count_categorical_values(df[name], attr.values)
            attributes[name] = {
                "type": "categorical",
                "values": list(attr.values),
                "counts": counts,
            }

    out = {
        "dataset": DATASET_NAME,
        "n_records": int(len(df)),
        "queries_source": QUERY_PATH,
        "attributes": attributes,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 摘要
    print(f"已写出边缘测量文件: {OUT_PATH}")
    print(f"  数据集    : {DATASET_NAME}（N={out['n_records']}）")
    print(f"  造箱依据  : {QUERY_PATH}")
    print(f"  属性覆盖  : {len(attributes)}/{schema.n_blocks()}")
    for name, spec in attributes.items():
        k = len(spec["counts"])
        total = sum(spec["counts"])
        kind = spec["type"]
        print(f"    - {name:12s} {kind:11s} {k} 箱, 计数和={total}")


if __name__ == "__main__":
    build()
