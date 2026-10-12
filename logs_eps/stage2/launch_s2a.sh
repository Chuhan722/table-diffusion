#!/bin/bash
# 第二阶段首战 A：测量后处理（平分清洗）消融，adult，吃第一阶段 1B 同一份 AIM 录像考卷。
# 诊断与设计见工作笔记第五十七步第 14–16 条、协议 docs/第一阶段可比性协议.md §4。
# 臂（命令 = launch_stage1.sh run_hybrid 逐字 + 一个开关，其余全同，hybrid 对照臂用第一阶段已有终表不重跑）：
#   p1  <tag> <seed>  hybrid + --oneway-pool 2             一阶题平分清洗
#   p12 <tag> <seed>  hybrid + --oneway-pool 2 --pair-pool 2  一阶 + 二阶（独立分）清洗
# GPU/CPU 纪律同第一阶段：GPU 只用 0/1，别人显存 ≥5 GB 视为在用；GPU 道 0-23 / 24-47。
# 用法（A6000）：cd /root/residual-evolution && nohup bash logs_eps/stage2/launch_s2a.sh [all|gpu0|gpu1] > logs_eps/stage2/launch_s2a.log 2>&1 &
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/stage2/jobs_s2a.txt
OUT=results/stage2
LOG=logs_eps/stage2
mkdir -p $OUT $LOG

next_job() {
  flock "$1.lock" bash -c "l=\$(head -1 $1); if [ -n \"\$l\" ]; then sed -i 1d $1; echo \"\$l\"; fi"
}

gpu_busy() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' ')
  [ -z "$used" ] && return 0
  [ "$used" -ge 5120 ]
}

wait_gpu_idle() {
  local gpu=$1 waited=0
  while :; do
    if ! gpu_busy $gpu; then
      sleep 60
      gpu_busy $gpu || break
    fi
    [ $((waited % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu$gpu 被占用，等待中（已等 $((waited / 60)) min）"
    sleep 120; waited=$((waited + 120))
  done
  [ $waited -gt 0 ] && echo "$(date -u +%FT%TZ) gpu$gpu 空闲，共等 $((waited / 60)) min"
}

run_arm() {  # gpu arm tag seed
  local gpu=$1 arm=$2 tag=$3 seed=$4 name=adult_eps${3}_hybrid${2}_seed$4 pool
  case $arm in
    p1) pool="--oneway-pool 2" ;;
    p12) pool="--oneway-pool 2 --pair-pool 2" ;;
    *) echo "未知臂: $arm"; return ;;
  esac
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  [ -s data/adult_aimrep${tag}_s${seed}/README.md ] || { echo "缺考卷 adult_aimrep${tag}_s${seed}"; return; }
  wait_gpu_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $pool $EXTRA > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

gpu_worker() {
  local gpu=$1 job
  while :; do
    job=$(next_job $JOBS); [ -z "$job" ] && break
    set -- $job
    run_arm $gpu $1 $2 $3
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}

mode=${1:-all}
case $mode in
  all) gpu_worker 0 & gpu_worker 1 & wait ;;
  gpu0) gpu_worker 0 ;;
  gpu1) gpu_worker 1 ;;
  *) echo "用法: $0 [all|gpu0|gpu1]"; exit 1 ;;
esac
echo "$(date -u +%FT%TZ) 全部完成"
