#!/bin/bash
# C 组 ε=3 收尾改在 A6000（用户令 2026-10-08 11:13：本机 4090 当前场跑完即退役，之后全部用 A6000）。
# 等 M3 的 "ALL DONE M3" 后接上，两卡 flock 共抢 jobs_C.txt：
#   aimsel <ds> <eps_tag> <eps> <seed> —— 第 2 步选择器（AIM 选择器 × 我们引擎，Δ₂=1），命令与第五十四步逐字相同
#   alloc  <ds> <eps_tag> <alloc_tag> <seed> —— 我们全测 Δ₂=1
# GPU 只用 0/1；CPU 守 0-47（两 lane 各 24 核）；绝不并发两只引擎于一 lane。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/baseline2/jobs_C.txt
LOG=logs_eps/baseline2
mkdir -p results/alloc results/aimsel logs_eps/alloc logs_eps/aimsel

until grep -q "ALL DONE M3" $LOG/launch_M3.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) M3 已 ALL DONE，接 C 组收尾"

next_job() {
  flock "$JOBS.lock" bash -c "l=\$(head -1 $JOBS); if [ -n \"\$l\" ]; then sed -i 1d $JOBS; echo \"\$l\"; fi"
}

run_aimsel() {  # gpu ds eps_tag eps seed
  local gpu=$1 ds=$2 tag=$3 eps=$4 seed=$5 name=${2}_eps${3}_aimsel_seed$5
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data $ds --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_${ds}${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_alloc() {  # gpu ds eps_tag alloc_tag seed
  local gpu=$1 ds=$2 tag=$3 alloc=$4 seed=$5 name=${2}_eps${3}_alloc_${4}_seed$5
  [ -s results/alloc/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data ${ds}_alloc${tag}${alloc}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/alloc/${name}.csv --out results/alloc/${name}_curve.csv \
    $EXTRA > logs_eps/alloc/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

gpu_worker() {
  local gpu=$1 job
  while :; do
    job=$(next_job); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aimsel) run_aimsel $gpu $2 $3 $4 $5 ;;
      alloc) run_alloc $gpu $2 $3 $4 $5 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}

gpu_worker 0 > $LOG/C_gpu0.log 2>&1 &
gpu_worker 1 > $LOG/C_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE C"
