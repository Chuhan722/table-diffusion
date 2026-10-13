# adult_aimrep0p3_s3_infer

1. 来源，adult_aimrep0p3_s3 考卷原样（1737 题）+ PGM 推断目标。
2. 口径，AIM 没测的对 81/105，每对精确二阶边际逐格成题共 12564 题，sigma 50.23（原二阶中位数 ×1）；推断来自 adult_eps0p3_pgmexact_seed3_exact.npz（同录像、同配置 PGM 拟合，不出表）。
3. 用法，run_plants.py --data adult_aimrep0p3_s3_infer --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。
