# adult_aimrep3_s2

1. 来源，官方 AIM 录像 adult_eps3_aimfull_seed2_measurements.json，ε=3.0 δ=1e-05 seed=2 ρ=0.224249，测量 70 条。
2. 口径，一阶团 15/15，二阶团 47/105，重复测量合并 8 团，合并后 sigma 范围 4.737~24.38，题数 4282（一阶 178 二阶 4104）。
3. 用法，run_plants.py --data adult_aimrep3_s2 --init-oneway --exam-weights sigma，不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。
