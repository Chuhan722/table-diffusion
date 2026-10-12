# pgmrep10k 两环境互核（同录像、同种子、同 iters=10000；单位 ×10⁻⁶ L1/N）

正式表 = 本机 private-pgm 环境（mbi 0.10.2，jax CPU）；远端 = A6000 aim2 环境（jax 0.11.2，mbi 0.11 系）。远端只跑完 10 场即因主机他人负载被拖慢 20 倍而停。

| 表 | 环境 | measured | 二阶 | 三阶均 | 半空间 |
|---|---|---|---|---|---|
| adult_eps0p1_pgmrep10k_seed1 | 本机 | 952 | 128,098 | 239,000 | 8,012 |
| adult_eps0p1_pgmrep10k_seed1 | 远端 | 950 | 127,803 | 238,273 | 8,109 |
| adult_eps0p1_pgmrep10k_seed1 | 本机−远端 二阶 | | +0.23% | | |
| adult_eps0p1_pgmrep10k_seed2 | 本机 | 1,119 | 150,524 | 265,323 | 9,154 |
| adult_eps0p1_pgmrep10k_seed2 | 远端 | 1,119 | 150,542 | 265,283 | 8,922 |
| adult_eps0p1_pgmrep10k_seed2 | 本机−远端 二阶 | | -0.01% | | |
| adult_eps0p1_pgmrep10k_seed3 | 本机 | 1,024 | 137,720 | 251,005 | 8,851 |
| adult_eps0p1_pgmrep10k_seed3 | 远端 | 1,021 | 137,265 | 250,622 | 8,846 |
| adult_eps0p1_pgmrep10k_seed3 | 本机−远端 二阶 | | +0.33% | | |
| adult_eps0p3_pgmrep10k_seed1 | 本机 | 617 | 82,990 | 171,702 | 4,533 |
| adult_eps0p3_pgmrep10k_seed1 | 远端 | 620 | 83,396 | 172,230 | 4,541 |
| adult_eps0p3_pgmrep10k_seed1 | 本机−远端 二阶 | | -0.49% | | |
| adult_eps0p3_pgmrep10k_seed2 | 本机 | 570 | 76,605 | 164,475 | 4,215 |
| adult_eps0p3_pgmrep10k_seed2 | 远端 | 572 | 76,959 | 165,316 | 4,032 |
| adult_eps0p3_pgmrep10k_seed2 | 本机−远端 二阶 | | -0.46% | | |
| adult_eps0p3_pgmrep10k_seed3 | 本机 | 607 | 81,607 | 168,052 | 5,006 |
| adult_eps0p3_pgmrep10k_seed3 | 远端 | 611 | 82,182 | 168,921 | 5,035 |
| adult_eps0p3_pgmrep10k_seed3 | 本机−远端 二阶 | | -0.70% | | |
| adult_eps1_pgmrep10k_seed1 | 本机 | 346 | 46,490 | 111,548 | 2,670 |
| adult_eps1_pgmrep10k_seed1 | 远端 | 350 | 47,038 | 111,884 | 2,724 |
| adult_eps1_pgmrep10k_seed1 | 本机−远端 二阶 | | -1.17% | | |
| adult_eps1_pgmrep10k_seed2 | 本机 | 349 | 46,912 | 112,664 | 2,591 |
| adult_eps1_pgmrep10k_seed2 | 远端 | 345 | 46,384 | 112,139 | 2,667 |
| adult_eps1_pgmrep10k_seed2 | 本机−远端 二阶 | | +1.14% | | |
| adult_eps1_pgmrep10k_seed3 | 本机 | 347 | 46,738 | 113,037 | 2,454 |
| adult_eps1_pgmrep10k_seed3 | 远端 | 348 | 46,868 | 112,769 | 2,614 |
| adult_eps1_pgmrep10k_seed3 | 本机−远端 二阶 | | -0.28% | | |
| adult_eps3_pgmrep10k_seed1 | 本机 | 234 | 31,491 | 85,990 | 1,968 |
| adult_eps3_pgmrep10k_seed1 | 远端 | 235 | 31,659 | 85,583 | 2,135 |
| adult_eps3_pgmrep10k_seed1 | 本机−远端 二阶 | | -0.53% | | |

二阶相对差：均 -0.19%，绝对值均 0.53%，最大 -1.17%。1000 轮时两环境四位数一致；10000 轮线搜索路径与合成抽样分叉到约 1% 量级，大于抽样种子噪声 ±0.2%，故正式臂统一用本机一套环境。
