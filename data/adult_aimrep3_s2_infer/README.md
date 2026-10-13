# adult_aimrep3_s2_infer

1. 来源，adult_aimrep3_s2 考卷原样（4282 题）+ PGM 推断目标。
2. 口径，AIM 没测的对 58/105，每对精确二阶边际逐格成题共 10019 题，sigma 12.19（原二阶中位数 ×1）；推断来自 adult_eps3_pgmexact_seed2_exact.npz（同录像、同配置 PGM 拟合，不出表）。
3. 用法，run_plants.py --data adult_aimrep3_s2_infer --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。
