#!/bin/bash
# C 组 ε=3 新点位——官方 AIM（A6000 纯 CPU，核 48-55，与 M3 两条 GPU lane 0-47 不相干）：adult ×3 在前（每场约 1 h），nltcs ×3 在后（分钟级）。
# 命令与第五十步/第五十四步官方 AIM 逐字相同（aim2 环境，unbounded + CKS，--max-model-size 80）。产物 results/{ds}_eps3_aimfull_seed{s}.csv。
cd /root/residual-evolution || exit 1
AIM_LANE=48-55
mkdir -p results/aimreplay logs_eps/baseline2
for ds in adult nltcs; do
  for s in 1 2 3; do
    name=${ds}_eps3_aimfull_seed${s}
    [ -s results/${name}.csv ] && { echo "skip $name"; continue; }
    echo "$(date -u +%FT%TZ) START $name lane=$AIM_LANE"
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
      taskset -c $AIM_LANE /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_aim_full.py \
        --data $ds --mechanisms /root/private-pgm/mechanisms --epsilon 3 --delta 1e-5 --seed $s \
        --max-model-size 80 --clear-caches-every 1 --clear-caches-every-projects 500 \
        --out results/${name}.csv --dump-measurements results/aimreplay/${name}_measurements.json \
        > logs_eps/baseline2/${name}.log 2>&1
    echo "$(date -u +%FT%TZ) END $name rc=$?"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE C-AIM"
