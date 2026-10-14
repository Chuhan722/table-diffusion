#!/bin/bash
# 第五十九步 甲（出菜单重写 + GPU build 融合核）全程验收：1A 命令逐字（与 launch_speed.sh 相同），
# adult ε=3 与 ε=0.1 各 3 种子，GPU1 道 24-47 顺序跑；产物进 results/speed2/，与 results/speed 的 乙 表配对比对（应逐位相同），同时记整场耗时。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/speed/launch_speed2.sh < /dev/null > logs_eps/speed/launch_speed2.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
LANE=24-47
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=results/speed2
LOG=logs_eps/speed
mkdir -p $OUT $LOG
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local w=0; while :; do if ! gpu_busy; then sleep 60; if ! gpu_busy; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu1 busy, waited ${w}s"; sleep 120; done; }
run() {  # tag seed
  local tag=$1 seed=$2 name=adult_eps${1}_ours2way_jia_seed$2
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle
  echo "$(date -u +%FT%TZ) START $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  taskset -c $LANE .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_alloc${tag}base_s${seed} --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-avg --init-clean --exam-weights none --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA --gpu-sparse auto > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
for s in 1 2 3; do run 3 $s; done
for s in 1 2 3; do run 0p1 $s; done
echo "$(date -u +%FT%TZ) ALL DONE SPEED2"
