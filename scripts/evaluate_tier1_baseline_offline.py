"""用本仓库评价器离线评价 Tier-1 基线合成表（Issue #46）。

保持评价器单一来源：基线表与本仓库生成器用完全相同的离线指标实现。
用法（gsd 环境）：
    PYTHONPATH=src conda run -n gsd python \
        scripts/evaluate_tier1_baseline_offline.py outputs/tier1_pgm_baseline
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import compare_factorized_gibbs_closed_loop as helpers
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema

schema = load_schema("configs/nltcs/schema.yaml")
queries = load_queries("configs/nltcs/measured_1000query.json")
marginals = load_marginals("configs/nltcs/init_marginals.json")
columns = schema.attribute_names()
domains = helpers._discretization_domains(marginals)
measured_triples = helpers._measured_cell_keys(queries, marginals, order=3)

references = {
    "train": pd.read_csv("data/nltcs/nltcs.train.data", header=None, names=columns)[columns],
    "test": pd.read_csv("data/nltcs/nltcs.test.data", header=None, names=columns)[columns],
}

pgm_dir = Path(sys.argv[1])
rows = []
for csv_path in sorted(pgm_dir.glob("pgm_synthetic_seed*.csv")):
    table = pd.read_csv(csv_path)[columns]
    entry = {"table": csv_path.name}
    for ref_name, reference in references.items():
        metrics = helpers._offline_metrics(
            reference, table, marginals, domains, measured_triples
        )
        entry[ref_name] = {
            "unmeasured_3way_l1": float(metrics["unmeasured_3way"]["mean"]),
            "unmeasured_4way_l1": float(metrics["unmeasured_4way"]["mean"]),
            "binned_joint_tvd": float(metrics["binned_joint"]["tvd"]),
            "raw_unique_states": int(metrics["raw_joint"]["n_unique"]),
        }
    rows.append(entry)
    print(csv_path.name, "done", flush=True)

summary = {}
for ref_name in references:
    summary[ref_name] = {
        key: float(np.mean([row[ref_name][key] for row in rows]))
        for key in rows[0][ref_name]
    }
print(json.dumps(summary, indent=1))
with open(pgm_dir / "pgm_offline_by_our_evaluator.json", "w") as f:
    json.dump({"rows": rows, "summary": summary}, f, ensure_ascii=False, indent=1)
