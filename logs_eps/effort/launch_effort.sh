#!/bin/bash
# 第六十步：搜索代价曲线（adult 1A，种子 1，五档 ε）。两家吃同一份 measured_2way.json。
#   GSD：launch_E_a6000.sh run_gsd 命令逐字 + --generation-log（gsd_patched 只加记账钩子，每 1000 代记最优 fitness/累计秒/累计候选数）
#   我们：launch_speed3.sh run 命令逐字（1A 命令 + --gpu-sparse auto），曲线新增 mode/candidates/moved_rows/elapsed_s 四列
# 终表与 results/baseline2（GSD）、results/stage1（我们）逐位核对后才算数。GPU0 道 0-23 跑我们，GPU1 道 24-47 跑 GSD，各自空闲门。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/effort/launch_effort.sh < /dev/null > logs_eps/effort/launch_effort.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=results/effort
LOG=logs_eps/effort
SEED=1
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }

run_ours() {  # tag
  local tag=$1 name=adult_eps${1}_ours2way_seed$SEED gpu=0
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_alloc${tag}base_s${SEED} --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-avg --init-clean --exam-weights none --menu-seed $((20260920 + SEED)) --sample-seed $SEED --init-seed $SEED \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA --gpu-sparse auto > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

run_gsd() {  # tag
  local tag=$1 name=adult_eps${1}_gsd_seed$SEED gpu=1
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/root/gsd_patched PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  taskset -c ${LANE[$gpu]} /root/miniconda3/envs/aim/bin/python -u -B scripts/run_baseline_gsd_plants.py \
    --seed $SEED --data adult --noisy-exam data/adult_alloc${tag}base_s${SEED}/measured_2way.json \
    --generation-log $OUT/${name}_gens.csv --log-every 1000 \
    --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

( for t in 0p1 0p3 1 3 10; do run_ours $t; done; echo "$(date -u +%FT%TZ) OURS DONE" ) &
( for t in 0p1 0p3 1 3 10; do run_gsd $t; done; echo "$(date -u +%FT%TZ) GSD DONE" ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE EFFORT"
