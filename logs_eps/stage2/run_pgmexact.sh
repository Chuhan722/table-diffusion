#!/bin/bash
# PGM 机制核查：同一拟合（与 logs_eps/stage1/run_pgmrep10k.sh 逐字同配置：同录像、同种子、iters 10000、known-total rows），
# 不出表直接算边际 vs 出表数边际。三条 CPU 车道并行：$1 = A（48-55，eps 10）、B（56-63，eps 3）、C（64-71，eps 1/0.3/0.1）
cd /root/residual-evolution
export PYTHONPATH=/root/private-pgm/src JAX_PLATFORMS=cpu
case "$1" in A) lane=48-55; epss="10";; B) lane=56-63; epss="3";; *) lane=64-71; epss="1 0p3 0p1";; esac
for e in $epss; do for s in 1 2 3; do
  job=adult_eps${e}_pgmexact_seed$s
  echo "$(date -u +%FT%TZ) START $job cpu lane=$lane"
  taskset -c $lane /root/miniconda3/envs/aim2/bin/python -u -B scripts/pgm_exact_marginals.py \
    --dump results/aimreplay/adult_eps${e}_aimfull_seed${s}_measurements.json --seed $s --iters 10000 \
    --known-total rows --heldout data/adult/heldout_1224query.json \
    --out-prefix results/stage2/$job > logs_eps/stage2/$job.log 2>&1
  echo "$(date -u +%FT%TZ) END $job rc=$?"
done; done
