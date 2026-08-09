"""
从 YAML 配置文件运行扩散演化

用法：
    conda run -p ./.conda python scripts/run_from_config.py [配置文件路径]

    省略路径时用默认的 configs/experiments/nltcs_baseline.yaml。

调参：
    修改 YAML 配置文件，或另建一个再作为参数传入。

注意（接受判据口径）：
    配置里 acceptance_rule.rule 是必填项，因此本脚本总是走 acceptance.py 的
    **严格改善口径**（拒绝平局），而不是主循环历史默认判据
    proposal_loss <= loss + tol（非严格）。二者在边界处理上不同，所以这里跑出的
    A0 结果不能当作历史 baseline 的逐轨迹复现。

优势：
    - 参数集中在 YAML 文件，代码简洁
    - 多个配置文件方便切换对比
    - 配置可版本控制，实验可复现
"""
import argparse
from pathlib import Path

from table_diffevo.experiment_config import ExperimentConfig
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run, create_parent_dir, save_summary


DEFAULT_CONFIG_FILE = "configs/experiments/nltcs_baseline.yaml"


def main(config_file: str = DEFAULT_CONFIG_FILE):
    """主函数"""
    # 加载并验证配置
    print(f"加载配置：{config_file}")
    config = ExperimentConfig.from_yaml(config_file)
    config.validate()
    print(f"✓ 配置验证通过")
    print(f"  实验名称：{config.experiment_name}")
    print(f"  轮数：{config.n_rounds}")
    print(f"  种子：{config.seeds}")
    print(f"  设备：{config.data.device}")
    print()

    # 接受判据披露：配置里的 rule 是必填项，故本脚本总是走 acceptance.py 的
    # 严格改善口径（拒绝平局），与主循环历史默认判据
    # proposal_loss <= loss + tol（非严格）在边界处理上不同。也就是说这里跑出的
    # A0 **不是**历史 baseline 的逐轨迹复现——要复现历史轨迹得让
    # acceptance_rule=None，而配置路径给不出 None。别拿这里的结果当 baseline 对拍。
    print(f"  接受规则：{config.acceptance_rule.rule}"
          f"（严格改善口径，非主循环历史判据）")
    print(f"  eps_L1={config.acceptance_rule.eps_L1}，"
          f"eps_Q={config.acceptance_rule.eps_Q}")
    print()

    # 创建输出目录。create_parent_dir 返回 str，这里统一转成 Path 再拼子目录。
    parent_dir = Path(create_parent_dir(config.output_dir))
    print(f"输出目录：{parent_dir}\n")

    # 多种子实验
    results = []
    for i, seed in enumerate(config.seeds):
        print(f"\n{'='*60}")
        print(f"种子 {seed} ({i+1}/{len(config.seeds)})")
        print(f"{'='*60}\n")

        # 获取参数并设置种子
        kwargs = config.to_run_evolution_kwargs(seed=seed)

        # 运行演化
        best_S, diag = run_evolution(**kwargs)

        # 保存结果。save_run 的第 3 个位置参数是 outputs_dir（会自建时间目录），
        # 必须用关键字 run_dir= 才是"写进这个目录"。
        run_dir = parent_dir / f"{i}-{seed}"
        save_run(best_S, diag, run_dir=str(run_dir))

        results.append({
            "seed": seed,
            "best_loss": diag["best_loss"],
            "normalized_l1_error": diag.get("normalized_l1_error", 0),
            "rounds_run": diag["rounds_run"],
            "stopped_early": diag["stopped_early"],
        })

        print(f"\n种子 {seed} 完成")
        print(f"  最优 loss: {diag['best_loss']:.2e}")
        print(f"  实际轮数: {diag['rounds_run']}/{config.n_rounds}")

    # 保存汇总
    print(f"\n{'='*60}")
    print("实验汇总")
    print(f"{'='*60}\n")

    # save_summary 的签名是 (parent_dir, summary)——只收两个参数，且第一个是目录。
    # 汇总里带上接受判据，避免事后分不清某次结果跑的是哪条口径。
    save_summary(str(parent_dir), {
        "experiment_name": config.experiment_name,
        "acceptance_rule": config.acceptance_rule.rule,
        "eps_L1": config.acceptance_rule.eps_L1,
        "eps_Q": config.acceptance_rule.eps_Q,
        "n_rounds": config.n_rounds,
        "seeds": list(config.seeds),
        "per_seed": results,
    })

    # 打印统计
    import numpy as np
    losses = [r["best_loss"] for r in results]
    print(f"最优 loss:")
    print(f"  均值: {np.mean(losses):.2e}")
    print(f"  标准差: {np.std(losses):.2e}")
    print(f"  最小: {np.min(losses):.2e}")
    print(f"  最大: {np.max(losses):.2e}")

    print(f"\n✓ 所有实验完成")
    print(f"结果保存在：{parent_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 YAML 配置文件运行扩散演化"
    )
    parser.add_argument(
        "config", nargs="?", default=DEFAULT_CONFIG_FILE,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG_FILE}）",
    )
    args = parser.parse_args()
    main(args.config)
