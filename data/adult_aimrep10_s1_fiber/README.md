# adult_aimrep10_s1_fiber

1. 来源，adult_aimrep10_s1 考卷原样（6767 题）+ 没测对的目标（fiber_avg）。
2. 口径，AIM 没测的对 45/105，每对二阶边际逐格成题共 7534 题，sigma 4.324（原二阶中位数 ×1）；目标来自 adult_eps10_fiber_seed1_targets.npz（我们自己的纤维平均二阶边际（收敛表洗牌平均，不借任何对手产物））。
3. 用法，run_plants.py --data adult_aimrep10_s1_fiber --init-oneway --exam-weights sigma（与 hybrid 臂同，只多这些题）。
