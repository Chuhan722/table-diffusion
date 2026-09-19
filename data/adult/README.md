# adult 真实数据

1. 来源，UCI adult 数据集的 train 切分 adult.data，32561 行 15 字段，缺失问号当独立类别不丢行。
2. 离散化口径由 scripts/prep_adult.py 固化，age 与 fnlwgt 与 hours-per-week 等频 16 桶分位边界去重，capital-gain 与 capital-loss 零单独一桶加非零等频 8 桶，education-num 整数码即值，类别字段去空格按字典序因子化，域宽 2 到 42。
3. 校验，adult.csv md5 为 b7663cb9c575a67a1a702e34cfc6f0b9。
4. 考卷由 scripts/gen_plants_exam.py --data adult 生成，训练卷全二阶格子 14123 条，保留卷三阶四阶各 512 条加半空间 200 条共 1224 条。
