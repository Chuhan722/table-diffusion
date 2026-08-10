"""简单测试：验证探测模式是否触发"""
import json
import numpy as np
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution

QUERY_PATH = "configs/test_300x10/measured_50query.json"

schema = load_schema("configs/test_300x10/schema.yaml")
queries = load_queries(QUERY_PATH)
target = np.array([q["result"] for q in queries])

# 合成表行数 = 源数据行数，自动取自查询文件的 record_count（公开元信息），不可手工设定
with open(QUERY_PATH, encoding="utf-8") as f:
    n_records = int(json.load(f)["record_count"])

print("开始运行简单探测测试...")
print(f"  合成表行数: {n_records}（自动取自 {QUERY_PATH} 的 record_count）")
print(f"  块大小: 30 候选评估")
print(f"  停滞阈值 P: 2")
print(f"  停滞相对阈值: 0.05 (提高以便快速触发探测)")
print(f"  轮数: 200\n")

best_S, diagnostics = run_evolution(
    target, queries, schema,
    n_records=n_records,
    n_rounds=200,
    seed=42,
    beta=1.0, h=0.8, rho=0.01, eta=0.5, mu=0.01,
    device='cuda',
    init_method='random',
    log_every=10,
    distance_mode='geometric',
    lambda_param=0.5,
    alpha_min=5.0,
    alpha_max=12.0,
    alpha_schedule_mode='probe',
    alpha_value=5.0,
    delta=0.05,
    winsorize_quantiles=(0.01, 0.99),
    probe_block_candidate_budget=30,
    probe_P=2,
    probe_H_candidate_budget=2,
    probe_s=0.2,
    probe_C=2,
    probe_stall_rel=0.05,  # 停滞相对阈值（提高以便快速触发探测）
    probe_patience=3,
)

print(f"\n完成！")
print(f"  轮数: {diagnostics['rounds_run']}")
print(f"  Best loss: {diagnostics['best_loss']:.2e}")
print(f"  L1: {diagnostics['normalized_l1_error']:.4f}")

ph = diagnostics.get('probe_history', [])
print(f"  探测次数: {len(ph)}")
if ph:
    print(f"  探测轮次: {[p['round'] for p in ph]}")
    print(f"  胜者: {[p['winner'] for p in ph]}")
