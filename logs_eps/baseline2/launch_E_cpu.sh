#!/bin/bash
# E 组纯 CPU 道（A6000 核 48-55）：官方 AIM acsmob ε=10 ×3 → ε=3 ×3 → ε=0.1 ×3（ε=1 时每场 3.6 h，共约 1.5 天）。命令与 C/D 组逐字相同。plants AIM 不跑（每场 22~70 h）。
cd /root/residual-evolution || exit 1
LANE=48-55; LOG=logs_eps/baseline2
mkdir -p results/aimreplay $LOG
for e in 10 3 0.1; do
  tag=${e/./p}
  for s in 1 2 3; do
    name=acsmob_eps${tag}_aimfull_seed${s}
    [ -s results/${name}.csv ] && { echo "skip $name"; continue; }
    echo "$(date -u +%FT%TZ) START $name lane=$LANE"
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
      taskset -c $LANE /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_aim_full.py \
        --data acsmob --mechanisms /root/private-pgm/mechanisms --epsilon $e --delta 1e-5 --seed $s \
        --max-model-size 80 --clear-caches-every 1 --clear-caches-every-projects 500 \
        --out results/${name}.csv --dump-measurements results/aimreplay/${name}_measurements.json \
        > $LOG/${name}.log 2>&1
    echo "$(date -u +%FT%TZ) END $name rc=$?"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE E-CPU"
