#!/bin/bash
# 归属对照：独立收敛 PGM 吃与我们 p1u 臂完全相同的清洗后一阶（--oneway-pool 2 --oneway-pool-scope uncovered），
# 其余与 logs_eps/stage1/run_pgmrep10k.sh 逐字相同（同录像、同种子、iters 10000、known-total rows），CPU 车道 48-55
cd /root/residual-evolution
export PYTHONPATH=/root/private-pgm/src JAX_PLATFORMS=cpu
for e in 0p1 0p3; do for s in 1 2 3; do
  job=adult_eps${e}_pgmrep10kp1u_seed$s
  echo "$(date -u +%FT%TZ) START $job cpu lane=48-55"
  taskset -c 48-55 /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_pgm_replay.py \
    --dump results/aimreplay/adult_eps${e}_aimfull_seed${s}_measurements.json --seed $s --iters 10000 \
    --known-total rows --oneway-pool 2 --oneway-pool-scope uncovered \
    --out results/stage2/$job.csv > logs_eps/stage2/$job.log 2>&1
  echo "$(date -u +%FT%TZ) END $job rc=$?"
done; done
