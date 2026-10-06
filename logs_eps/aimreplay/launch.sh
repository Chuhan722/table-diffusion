#!/bin/bash
# AIM 录像重跑：adult eps0.1/1 x seed1-3，6 条 9 核 CPU 流全部落在 0-55 内
cd /root/residual-evolution
LANES=(0-8 9-17 18-26 27-35 36-44 45-53)
i=0
for eps in 0.1 1; do
  tag=$([ "$eps" = "0.1" ] && echo 0p1 || echo 1)
  for s in 1 2 3; do
    lane=${LANES[$i]}; i=$((i+1))
    name=adult_eps${tag}_aimfull_seed${s}
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
      setsid nohup taskset -c $lane /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_aim_full.py \
        --data adult --mechanisms /root/private-pgm/mechanisms --epsilon $eps --delta 1e-5 --seed $s \
        --max-model-size 80 --clear-caches-every 1 --clear-caches-every-projects 500 \
        --out results/aimreplay/$name.csv --dump-measurements results/aimreplay/${name}_measurements.json \
        > logs_eps/aimreplay/$name.log 2>&1 < /dev/null &
    echo "launched $name lane=$lane pid=$!"
  done
done
