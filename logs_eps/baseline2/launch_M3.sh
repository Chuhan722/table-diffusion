#!/bin/bash
# 第五十五步 M3（主表 ε=1 口径修正扩到 acsmob/plants，A6000 端；A 组 ALL DONE 后自动接上）：
#   alloc <ds> <eps_tag> <alloc_tag> <seed> —— 我们引擎全测 Δ₂=1（plants 2346 对 f1∈{0.5,0.1,auto}，acsmob 210 对 f1=0.5；auto 版 acsmob 在本机 4090）
#   gsd   <ds> <eps_tag> <seed>             —— GSD 官方 oneshot 同餐，Δ₂=1 全预算二阶卷 data/<ds>_alloc<tag>base_s<seed>/measured_2way.json
# 引擎命令与 adult/nltcs 全测场完全一致（第 0 步混合命令：--init-oneway --exam-weights sigma，组合尺 ding；
# 第四十五步双门在 acsmob/plants 上本就弃己尺守 d8，行为等价）。GPU 只用 0/1；CPU 守 0-47。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/baseline2/jobs_M3.txt
OUT=results/baseline2; LOG=logs_eps/baseline2
mkdir -p $OUT $LOG results/alloc logs_eps/alloc

until grep -q "ALL DONE" $LOG/launch_baseA.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) A 组已 ALL DONE，接 M3"

next_job() {
  flock "$JOBS.lock" bash -c "l=\$(head -1 $JOBS); if [ -n \"\$l\" ]; then sed -i 1d $JOBS; echo \"\$l\"; fi"
}

run_gsd() {  # gpu ds eps_tag seed
  local gpu=$1 ds=$2 tag=$3 seed=$4 name=${2}_eps${3}_gsd_seed$4
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/root/gsd_patched PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  taskset -c ${LANE[$gpu]} /root/miniconda3/envs/aim/bin/python -u -B scripts/run_baseline_gsd_plants.py \
    --seed $seed --data $ds --noisy-exam data/${ds}_alloc${tag}base_s${seed}/measured_2way.json \
    --out $OUT/$name.csv > $LOG/$name.log 2>&1
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
      gsd) run_gsd $gpu $2 $3 $4 ;;
      alloc) run_alloc $gpu $2 $3 $4 $5 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}

gpu_worker 0 > $LOG/M3_gpu0.log 2>&1 &
gpu_worker 1 > $LOG/M3_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE M3"
