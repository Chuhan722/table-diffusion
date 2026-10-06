# adult_alloc0p1k0_s3 固定分配考卷

1. 来源，data/adult 真表原样复制 md5 b7663cb9c575a67a1a702e34cfc6f0b9，一阶 15 个 + 格数最小 0/105 对直接加高斯噪声，其余对不测。
2. 账本（AIM 口径，增删一行，高斯 L2 敏感度 1，每边际 1/(2σ²)）：ε=0.1 δ=1e-05 cdp_rho ρ=0.000432994；一阶 σ1=131.6 花 0.000432994（100.0%），二阶 σ2=inf 花 0（0.0%），合计 0.000432994 = ρ。
3. 规则参数：f1 1，K 0，tau 0.11，gamma 0.65，kappa 0.25；模型预估二阶均 129135×1e-6（测过 0，没测 129135）。
4. 测的对：无。
5. 噪声种子 3，numpy default_rng，先一阶后二阶按上列顺序逐格生成，不裁剪。
6. 用法：run_plants.py --data adult_alloc0p1k0_s3 --init-oneway --exam-weights sigma --struct-ruler ding（不带 --ji-rho）。
