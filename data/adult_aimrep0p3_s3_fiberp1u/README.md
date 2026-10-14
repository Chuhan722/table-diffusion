# adult_aimrep0p3_s3_fiberp1u

1. 来源，adult_aimrep0p3_s3 考卷原样（1737 题）+ 没测对的目标（fiber_avg）。
2. 口径，AIM 没测的对 81/105，每对二阶边际逐格成题共 12564 题，sigma 50.23（原二阶中位数 ×1）；目标来自 adult_eps0p3_fiberp1u_seed3_targets.npz（我们自己的纤维平均二阶边际（收敛表洗牌平均，不借任何对手产物））。
3. 用法，run_plants.py --data adult_aimrep0p3_s3_fiberp1u --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。
