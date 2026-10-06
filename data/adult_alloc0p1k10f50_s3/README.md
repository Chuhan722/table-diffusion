# adult_alloc0p1k10f50_s3 固定分配考卷

1. 来源，data/adult 真表原样复制 md5 b7663cb9c575a67a1a702e34cfc6f0b9，一阶 15 个 + 格数最小 10/105 对直接加高斯噪声，其余对不测。
2. 账本（AIM 口径，增删一行，高斯 L2 敏感度 1，每边际 1/(2σ²)）：ε=0.1 δ=1e-05 cdp_rho ρ=0.000432994；一阶 σ1=186.1 花 0.000216497（50.0%），二阶 σ2=152 花 0.000216497（50.0%），合计 0.000432994 = ρ。
3. 规则参数：f1 0.50，K 10，tau 0.11，gamma 0.65，kappa 0.25；模型预估二阶均 128588×1e-6（测过 30499，没测 138913）。
4. 测的对：attr_10×attr_15(4), attr_9×attr_10(10), attr_9×attr_15(10), attr_8×attr_10(12), attr_8×attr_15(12), attr_6×attr_10(14), attr_6×attr_15(14), attr_10×attr_13(16), attr_13×attr_15(16), attr_2×attr_10(18)。
5. 噪声种子 3，numpy default_rng，先一阶后二阶按上列顺序逐格生成，不裁剪。
6. 用法：run_plants.py --data adult_alloc0p1k10f50_s3 --init-oneway --exam-weights sigma --struct-ruler ding（不带 --ji-rho）。
