# adult_aimrep0p1_s3_infer

1. 来源，adult_aimrep0p1_s3 考卷原样（1015 题）+ PGM 推断目标。
2. 口径，AIM 没测的对 87/105，每对精确二阶边际逐格成题共 13286 题，sigma 138.7（原二阶中位数 ×1）；推断来自 adult_eps0p1_pgmexact_seed3_exact.npz（同录像、同配置 PGM 拟合，不出表）。
3. 用法，run_plants.py --data adult_aimrep0p1_s3_infer --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。
