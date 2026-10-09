#!/bin/bash
# F 组 acsmob/plants 的 ε=0.3（A6000 GPU0/1，2026-10-09 18:40 北京起；用户令"继续"，补主表最后一列）。队列：acsmob 全测 ×3 → GSD acsmob ×3 → plants 全测 ×3 → GSD plants ×3。引擎为第五十六步提速后新码（结果与旧码逐位同）。
# 两卡 flock 共抢 jobs_F.txt，任务三种（命令与 M3 / C 组逐字相同）：
#   aimsel <ds> <eps_tag> <eps> <seed>      选择器（AIM 选择器 × 我们引擎，Δ₂=1）
#   alloc  <ds> <eps_tag> <alloc_tag> <seed> 我们全测 Δ₂=1（auto 卷）
#   gsd    <ds> <eps_tag> <seed>             GSD 同餐 Δ₂=1（吃 data/{ds}_alloc{tag}base_s{seed}/measured_2way.json）
# GPU 只用 0/1；CPU 守 0-47（两 lane 各 24 核）；绝不并发两只引擎于一 lane。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/baseline2/jobs_F.txt
LOG=logs_eps/baseline2
OUT=results/baseline2
mkdir -p results/alloc results/aimsel results/baseline2 logs_eps/alloc logs_eps/aimsel

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

gpu_worker() {
  local gpu=$1 job
  while :; do
    job=$(next_job); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aimsel) run_aimsel $gpu $2 $3 $4 $5 ;;
      alloc) run_alloc $gpu $2 $3 $4 $5 ;;
      gsd) run_gsd $gpu $2 $3 $4 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}

echo "$(date -u +%FT%TZ) F 组 acsmob/plants ε=0.3 开跑"
gpu_worker 0 > $LOG/F_gpu0.log 2>&1 &
gpu_worker 1 > $LOG/F_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE F"
