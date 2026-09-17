# plants 真实数据

1. 来源，UCI plants 数据集的 train 切分，取自旧仓 data/plants/plants.csv，只拷数据不拷代码。
2. 规格，17412 行，69 个字段 attr_1 到 attr_69，取值全部 0 或 1，带表头。
3. 校验，md5 为 5acb84905de66184b21b0bef968f60f0。
4. 考卷由 scripts/gen_plants_exam.py 生成，训练卷全二阶格子一阶不出题，保留卷高阶加半空间。
