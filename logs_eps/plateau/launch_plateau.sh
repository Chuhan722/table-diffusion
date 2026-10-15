#!/bin/bash
# 第六十一步：平台段价值量化（adult 1A，种子 1，五档 ε）。
#   第六十步 ours 命令逐字 + --snapshot-loss/--snapshot-prefix（纯记账，轨迹不变），
#   阈值 = 第六十步该 ε 终损失（引擎口径 old_loss）× 1.2 / 1.1 / 1.05 / 1.02 / 1.01，
#   快照 0..4 对应上述倍数，终表须与 results/effort 逐位一致后才算数。
# GPU0 道 0-23 跑 ε=3/0.1/0.3，GPU1 道 24-47 跑 ε=10/1，各自空闲门。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/plateau/launch_plateau.sh < /dev/null > logs_eps/plateau/launch_plateau.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
declare -A SNAP=(
  [0p1]="846630723.272935,776078163.0001905,740801882.8638182,719636114.7819948,712580858.7547203"
  [0p3]="96564951.11504999,88517871.8554625,84494332.22566874,82080208.4477925,81275500.52183375"
  [1]="8789003.121563593,8056586.194766627,7690377.731368144,7470652.653329054,7397410.960649357"
  [3]="1023073.4422593804,937817.3220710987,895189.2619769579,869612.4259204733,861086.8139016451"
  [10]="117343.8287924863,107565.17639311244,102675.8501934255,99742.25447361336,98764.38923367596"
)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=results/plateau
LOG=logs_eps/plateau
SEED=1
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }

run_ours() {  # tag gpu
  local tag=$1 gpu=$2 name=adult_eps${1}_ours2way_seed$SEED
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_alloc${tag}base_s${SEED} --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-avg --init-clean --exam-weights none --menu-seed $((20260920 + SEED)) --sample-seed $SEED --init-seed $SEED \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    --snapshot-loss "${SNAP[$tag]}" --snapshot-prefix $OUT/$name \
    $EXTRA --gpu-sparse auto > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

( for t in 3 0p1 0p3; do run_ours $t 0; done; echo "$(date -u +%FT%TZ) GPU0 LANE DONE" ) &
( for t in 10 1; do run_ours $t 1; done; echo "$(date -u +%FT%TZ) GPU1 LANE DONE" ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE PLATEAU"
