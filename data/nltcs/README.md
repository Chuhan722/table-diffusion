# nltcs 真实数据

1. 来源，文献标准 nltcs 数据集的 train 切分，取自旧仓 data/nltcs/nltcs.csv，只拷数据不拷代码。
2. 规格，16181 行，16 个字段 attr_1 到 attr_16，取值全部 0 或 1，带表头。
3. 校验，md5 为 9d36c8675596cea393fed26326286572。
4. 考卷由 scripts/gen_plants_exam.py --data nltcs 生成，训练卷全二阶格子 480 条，保留卷三阶四阶各 512 条加半空间 200 条共 1224 条。
