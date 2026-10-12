# adult_aimrep10_s1

1. 来源，官方 AIM 录像 adult_eps10_aimfull_seed1_measurements.json，ε=10.0 δ=1e-05 seed=1 ρ=1.7827，测量 83 条。
2. 口径，一阶团 15/15，二阶团 60/105，重复测量合并 8 团，合并后 sigma 范围 1.713~8.648，题数 6767（一阶 178 二阶 6589）。
3. 用法，run_plants.py --data adult_aimrep10_s1 --init-oneway --exam-weights sigma，不带 --ji-rho（预算已被 AIM 花光，S* 不买），隐私账等于 AIM 该跑。
