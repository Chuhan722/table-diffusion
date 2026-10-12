#!/bin/bash
# 独立 PGM 收敛臂：同一录像、同一种子，iters 1000 -> 10000（其余不变），CPU 车道 48-55
cd /root/residual-evolution
export PYTHONPATH=/root/private-pgm/src JAX_PLATFORMS=cpu
for e in 0p1 0p3 1 3 10; do for s in 1 2 3; do
  job=adult_eps${e}_pgmrep10k_seed$s
  echo "$(date -u +%FT%TZ) START $job cpu lane=48-55"
  taskset -c 48-55 /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_pgm_replay.py \
    --dump results/aimreplay/adult_eps${e}_aimfull_seed${s}_measurements.json --seed $s --iters 10000 \
    --known-total rows --out results/stage1/$job.csv > logs_eps/stage1/$job.log 2>&1
  echo "$(date -u +%FT%TZ) END $job rc=$?"
done; done
