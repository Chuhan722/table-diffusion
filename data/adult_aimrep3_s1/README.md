# adult_aimrep3_s1

1. 来源，官方 AIM 录像 adult_eps3_aimfull_seed1_measurements.json，ε=3.0 δ=1e-05 seed=1 ρ=0.224249，测量 76 条。
2. 口径，一阶团 15/15，二阶团 51/105，重复测量合并 10 团，合并后 sigma 范围 5.452~24.38，题数 4606（一阶 178 二阶 4428）。
3. 用法，run_plants.py --data adult_aimrep3_s1 --init-oneway --exam-weights sigma，不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。
