# adult_aimrep3_s3_fiber

1. 来源，adult_aimrep3_s3 考卷原样（4432 题）+ 没测对的目标（fiber_avg）。
2. 口径，AIM 没测的对 57/105，每对二阶边际逐格成题共 9869 题，sigma 12.19（原二阶中位数 ×1）；目标来自 adult_eps3_fiber_seed3_targets.npz（我们自己的纤维平均二阶边际（收敛表洗牌平均，不借任何对手产物））。
3. 用法，run_plants.py --data adult_aimrep3_s3_fiber --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。
