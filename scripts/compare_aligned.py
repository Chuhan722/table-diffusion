"""
对齐改动量的对照实验：legacy vs single_block

关键设计：
- legacy: ρ=0.01，参与后期望改 ≈5 块/记录
- single_block: ρ=0.05（5倍），参与后改 1 块/记录
→ 每轮总改动量对齐，验证新方法在速度对齐后能否达到相近的收敛效果
"""
import json
import numpy as np
from pathlib import Path

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution


# 实验配置
SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
N_RECORDS = 300
N_ROUNDS = 100
SEED = 42
ETA = 0.5
MU = 0.01
EPSILON = 0.01  # single_block 的变异占比

# 关键：对齐改动量
RHO_LEGACY = 0.01
RHO_SINGLE_BLOCK = 0.05  # 约5倍，补偿单块改动

OUTPUT_DIR = Path("outputs/aligned_comparison")


def run_experiment(update_mode: str, rho: float) -> dict:
    """跑一次实验并返回诊断。"""
    print(f"\n{'='*60}")
    print(f"运行 {update_mode} 模式（ρ={rho}）...")
    print(f"{'='*60}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])

    best_S, diag = run_evolution(
        target, queries, schema,
        n_records=N_RECORDS,
        n_rounds=N_ROUNDS,
        seed=SEED,
        rho=rho,
        eta=ETA,
        mu=MU,
        update_mode=update_mode,
        epsilon=EPSILON,
        device='numpy',
        log_every=10,
    )

    # 提取关键指标
    result = {
        "update_mode": update_mode,
        "rho": rho,
        "params": diag["params"],
        "initial_loss": diag["loss_history"][0],
        "final_loss": diag["best_loss"],
        "loss_reduction_pct": (1 - diag["best_loss"] / diag["loss_history"][0]) * 100,
        "acceptance_rate": sum(diag["accept_history"]) / len(diag["accept_history"]),
        "rounds_run": diag["rounds_run"],
        "elapsed_sec": diag["elapsed_sec"],
        "normalized_l1_error": diag["normalized_l1_error"],
        "loss_history": diag["loss_history"],
        "accept_history": diag["accept_history"],
    }

    # single_block 特有诊断
    if update_mode == 'single_block':
        result["single_block_diagnostics"] = {
            "participation_rate_mean": np.mean(diag["participation_rate_history"]),
            "copy_attempt_rate_mean": np.mean(diag["copy_attempt_rate_history"]),
            "mutation_attempt_rate_mean": np.mean(diag["mutation_attempt_rate_history"]),
            "accepted_change_rate_mean": np.mean(diag["accepted_change_rate_history"]),
            "empty_copy_set_total": sum(diag["empty_copy_set_count_history"]),
            "empty_copy_set_per_round_mean": np.mean(diag["empty_copy_set_count_history"]),
        }

    # 打印摘要
    print(f"\n结果摘要 ({update_mode}, ρ={rho}):")
    print(f"  初始 loss: {result['initial_loss']:.2e}")
    print(f"  最终 loss: {result['final_loss']:.2e}")
    print(f"  降低: {result['loss_reduction_pct']:.1f}%")
    print(f"  接受率: {result['acceptance_rate']:.1%}")
    print(f"  归一化L1: {result['normalized_l1_error']:.4f}")
    print(f"  耗时: {result['elapsed_sec']:.2f}s")

    if update_mode == 'single_block':
        sb = result["single_block_diagnostics"]
        print(f"\n  single_block 诊断:")
        print(f"    实际参与率: {sb['participation_rate_mean']:.4f} (目标 {rho})")
        print(f"    复制尝试率: {sb['copy_attempt_rate_mean']:.4f}")
        print(f"    变异尝试率: {sb['mutation_attempt_rate_mean']:.4f}")
        print(f"    接受变更率: {sb['accepted_change_rate_mean']:.4f}")
        print(f"    空复制集: {sb['empty_copy_set_total']} 次")

    return result


def compare_results(legacy_result: dict, single_block_result: dict):
    """对比两种模式的结果。"""
    print(f"\n{'='*60}")
    print("对齐改动量后的对比")
    print(f"{'='*60}")

    legacy_loss = legacy_result["final_loss"]
    sb_loss = single_block_result["final_loss"]
    loss_diff_pct = (sb_loss / legacy_loss - 1) * 100

    legacy_acc = legacy_result["acceptance_rate"]
    sb_acc = single_block_result["acceptance_rate"]
    acc_diff_pct = (sb_acc / legacy_acc - 1) * 100

    print(f"\n参与率设置:")
    print(f"  legacy:       ρ={legacy_result['rho']} (期望改 ~5块/记录)")
    print(f"  single_block: ρ={single_block_result['rho']} (改 1块/记录)")
    print(f"  → 对齐每轮总改动量")

    print(f"\n最终 loss:")
    print(f"  legacy:       {legacy_loss:.2e}")
    print(f"  single_block: {sb_loss:.2e}")
    print(f"  差异: {loss_diff_pct:+.1f}%")
    if abs(loss_diff_pct) < 10:
        print(f"  → 差异 <10%，收敛效果相当")
    elif loss_diff_pct < 0:
        print(f"  → single_block 更优")
    else:
        print(f"  → legacy 仍更优")

    print(f"\n接受率:")
    print(f"  legacy:       {legacy_acc:.1%}")
    print(f"  single_block: {sb_acc:.1%}")
    print(f"  差异: {acc_diff_pct:+.1f}%")
    if sb_acc < 0.5:
        print(f"  ⚠ single_block 接受率过低，ρ={single_block_result['rho']} 可能太大")

    print(f"\n归一化L1误差:")
    print(f"  legacy:       {legacy_result['normalized_l1_error']:.4f}")
    print(f"  single_block: {single_block_result['normalized_l1_error']:.4f}")

    # 空复制集分析
    if "single_block_diagnostics" in single_block_result:
        sb = single_block_result["single_block_diagnostics"]
        print(f"\nsingle_block 空复制集: {sb['empty_copy_set_total']} 次")

    print(f"\n结论:")
    if abs(loss_diff_pct) < 10 and sb_acc > 0.5:
        print("  ✓ 对齐改动量后，single_block 能达到相近收敛效果")
        print("  ✓ 新方法的可控性/可解释性优势成立，速度不是瓶颈")
    elif loss_diff_pct > 10:
        print("  ✗ 即使对齐改动量，legacy 仍明显更优")
        print("  → 新方法可能在更新质量上有劣势（不只是数量问题）")
    elif sb_acc < 0.5:
        print("  ⚠ single_block 接受率过低，需降低 ρ 或调整其他参数")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 跑两种模式（对齐改动量）
    legacy_result = run_experiment('legacy', RHO_LEGACY)
    single_block_result = run_experiment('single_block', RHO_SINGLE_BLOCK)

    # 对比分析
    compare_results(legacy_result, single_block_result)

    # 保存完整结果
    output_file = OUTPUT_DIR / "aligned_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "legacy": legacy_result,
            "single_block": single_block_result,
            "config": {
                "rho_legacy": RHO_LEGACY,
                "rho_single_block": RHO_SINGLE_BLOCK,
                "alignment_note": "single_block ρ 约为 legacy 的 5 倍，对齐每轮总改动块数",
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\n完整结果已保存: {output_file}")


if __name__ == "__main__":
    main()
