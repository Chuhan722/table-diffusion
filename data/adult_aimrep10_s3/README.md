# adult_aimrep10_s3

1. 来源，官方 AIM 录像 adult_eps10_aimfull_seed3_measurements.json，ε=10.0 δ=1e-05 seed=3 ρ=1.7827，测量 82 条。
2. 口径，一阶团 15/15，二阶团 60/105，重复测量合并 7 团，合并后 sigma 范围 2.067~8.648，题数 6688（一阶 178 二阶 6510）。
3. 用法，run_plants.py --data adult_aimrep10_s3 --init-oneway --exam-weights sigma，不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。
