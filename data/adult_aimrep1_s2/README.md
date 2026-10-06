# adult_aimrep1_s2

1. 来源，官方 AIM 录像 adult_eps1_aimfull_seed2_measurements.json，ε=1.0 δ=1e-05 seed=2 ρ=0.0305566，测量 61 条。
2. 口径，一阶团 15/15，二阶团 36/105，重复测量合并 10 团，合并后 sigma 范围 12.37~66.06，题数 2949（一阶 178 二阶 2771）。
3. 用法，run_plants.py --data adult_aimrep1_s2 --init-oneway --exam-weights sigma，不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。
