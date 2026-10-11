# adult_aimrep0p3_s3

1. 来源，官方 AIM 录像 adult_eps0p3_aimfull_seed3_measurements.json，ε=0.3 δ=1e-05 seed=3 ρ=0.00330299，测量 45 条。
2. 口径，一阶团 15/15，二阶团 24/105，重复测量合并 6 团，合并后 sigma 范围 43.33~200.9，题数 1737（一阶 178 二阶 1559）。
3. 用法，run_plants.py --data adult_aimrep0p3_s3 --init-oneway --exam-weights sigma，不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。
