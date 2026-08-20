# plants 数据集

## 来源与引用

公开密度估计基准系列（"twenty datasets" 文献族）中的 plants 数据集，
最初由以下工作整理并发布为二值化基准：

- D. Lowd, J. Davis. "Learning Markov Network Structure with Decision
  Trees." ICDM 2010.（数据整理来源）
- J. Van Haaren, J. Davis. "Markov Network Structure Learning: A
  Randomized Feature Generation Approach." AAAI 2012.（基准系列汇编）

原始数据为 USDA PLANTS Database 的物种-州分布记录的二值化版本
（69 个二值属性）。本仓库使用的副本从公开基准集合仓库
（UCLA-StarAI/Density-Estimation-Datasets，GitHub 公开分发）下载，
train/valid/test 划分沿用原始发布，未做任何修改。

## 授权说明

该基准系列在密度估计与图模型结构学习文献中被广泛公开分发与使用
（上述集合仓库公开托管全部二十个数据集）；底层 USDA PLANTS 数据为
美国政府公开数据。本仓库仅将其用于研究性基准评价。

## 文件与身份（SHA-256）

| 文件 | 行数×列数 | SHA-256 |
|---|---|---|
| plants.train.data | 17412×69 | `1fb1219ff94068d12a563f9e81f8889a1885f41e867884cff608669300c6848f` |
| plants.valid.data | 2321×69 | `47834ed318b2d74f48e4b083226ef47fd446d41986012ceae19d230574821b34` |
| plants.test.data | 3482×69 | `95f53c033cd1d86467aec2610dc5bfe014534206c2da4919f73bcab246ab8755` |
| plants.csv | train 加表头 attr_1..attr_69 | `c7b5cf1e2230df3facf8d4b5d4a077d747a4f84e2dd47ce8779335f553599532` |

格式：无表头逗号分隔 0/1（与 data/nltcs 同系列同格式）。
plants.csv 由 train 原样加表头得到，供离线测量脚本
（scripts/gen_plants_queries_v2.py、scripts/build_marginals.py）使用。

## workload（configs/plants/measured_1000query.json，v2）

980 条查询：double 460 条（φ²=χ²/N 关联度 top 115 属性对，完整 2×2
成组）+ triple 520 条（三阶专属 G² 交互度 top 260 组，组内标准化残差
top-2 且取值模式互异；三属性模式直方图 7/8 覆盖，无 000 塌缩）。
1-way 不计入主 workload——精确 1-way 边缘由
configs/plants/init_marginals.json 作为初始化层单独提供（与 nltcs
口径一致）。设计依据与审查讨论见 PR #61。
