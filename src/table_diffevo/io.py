"""
结果落盘

把一次演化运行的结果保存到 outputs/ 下按时间命名的文件夹。

## 保存内容

每次运行新建一个文件夹 outputs/YYYY-MM-DD_HHMM_N/，内含：
- best_synthetic.csv：最优合成表（best_S）
- diagnostics.json：全部诊断信息（loss_history、best_loss 等）

不保存每一代的完整表（体积大、价值低）。

## 文件夹命名

单次运行（save_run 自建目录，run_dir=None）：
- 时间精确到分钟：YYYY-MM-DD_HHMM
- 加数字后缀避免重名，从 0 起：..._0、..._1、...

多种子运行（run.py 调 create_parent_dir + save_run(run_dir=...)）：
- 父文件夹 outputs/YYYY-MM-DD_HHMM/（不加后缀）
- 各种子子文件夹 {顺序}-{种子}/，如 0-0/、1-1/、2-2/
- 汇总 summary.json 写在父文件夹下

## 职责边界

本模块只负责写盘，不跑演化。主循环 run_evolution 保持只算不写。
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def _json_default(obj: Any) -> Any:
    """让 numpy 类型可被 json 序列化。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"无法序列化类型: {type(obj)}")


def _make_run_dir(outputs_dir: str, now: datetime) -> str:
    """
    创建按时间命名的运行文件夹，加数字后缀避免重名。

    返回实际创建的文件夹路径。
    """
    os.makedirs(outputs_dir, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d_%H%M")

    n = 0
    while True:
        candidate = os.path.join(outputs_dir, f"{stamp}_{n}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        n += 1


def create_parent_dir(outputs_dir: str = "outputs") -> str:
    """
    为一次多种子运行创建日期时间父文件夹 outputs/YYYY-MM-DD_HHMM/。

    与 _make_run_dir 不同：不加数字后缀（按项目约定，同一分钟不防重）。
    各种子的子文件夹（如 0-0/、1-1/）由调用方拼在此目录下。

    Returns
    -------
    str
        创建的父文件夹路径
    """
    os.makedirs(outputs_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    parent = os.path.join(outputs_dir, stamp)
    os.makedirs(parent, exist_ok=True)
    return parent


def save_summary(parent_dir: str, summary: Dict[str, Any]) -> str:
    """
    把多种子运行的汇总写入 parent_dir/summary.json。

    Returns
    -------
    str
        summary.json 的路径
    """
    path = os.path.join(parent_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2,
                  default=_json_default)
    return path


def save_run(
    best_S: pd.DataFrame,
    diagnostics: Dict[str, Any],
    outputs_dir: str = "outputs",
    run_dir: Optional[str] = None,
) -> str:
    """
    把一次演化运行的结果落盘。

    Parameters
    ----------
    best_S : pd.DataFrame
        最优合成表，来自 run_evolution
    diagnostics : dict
        诊断信息，来自 run_evolution
    outputs_dir : str, default "outputs"
        输出根目录，函数会在其下新建按时间命名的子文件夹。
        仅当 run_dir 为 None 时生效。
    run_dir : str or None, default None
        指定结果写入的目录（多种子场景由调用方拼好路径传入）。
        - None（默认）：沿用旧行为，在 outputs_dir 下自建时间命名文件夹
        - 指定路径：直接写入该目录（不存在则创建）

    Returns
    -------
    str
        实际写入的运行文件夹路径（含两个结果文件）

    Notes
    -----
    - best_synthetic.csv：合成表，不含行索引
    - diagnostics.json：全部诊断，UTF-8、缩进 2

    Examples
    --------
    >>> best_S, diag = run_evolution(target, queries, schema, n_records=300)
    >>> folder = save_run(best_S, diag)
    >>> folder
    'outputs/2026-07-21_1430_0'
    """
    if run_dir is None:
        run_dir = _make_run_dir(outputs_dir, datetime.now())
    else:
        os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "best_synthetic.csv")
    best_S.to_csv(csv_path, index=False)

    json_path = os.path.join(run_dir, "diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2,
                  default=_json_default)

    return run_dir
