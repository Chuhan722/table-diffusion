#!/bin/bash
# 单任务版：run_pgmexact_one.sh <eps标签> <种子> <cpu车道> [等待的PID]
# 与 run_pgmexact.sh 同命令；若给第 4 个参数，先等该进程退出（接在前一个种子后面跑，不多占车道）
cd /root/residual-evolution
export PYTHONPATH=/root/private-pgm/src JAX_PLATFORMS=cpu
e=$1; s=$2; lane=$3; waitpid=$4
if [ -n "$waitpid" ]; then while kill -0 $waitpid 2>/dev/null; do sleep 30; done; fi
job=adult_eps${e}_pgmexact_seed$s
echo "$(date -u +%FT%TZ) START $job cpu lane=$lane"
taskset -c $lane /root/miniconda3/envs/aim2/bin/python -u -B scripts/pgm_exact_marginals.py \
  --dump results/aimreplay/adult_eps${e}_aimfull_seed${s}_measurements.json --seed $s --iters 10000 \
  --known-total rows --heldout data/adult/heldout_1224query.json \
  --out-prefix results/stage2/$job > logs_eps/stage2/$job.log 2>&1
echo "$(date -u +%FT%TZ) END $job rc=$?"
