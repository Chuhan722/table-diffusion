#!/bin/bash
# 第一步固定分配：GPU0/GPU1 各一条串行队列（jobs_gpu0.txt / jobs_gpu1.txt，每行 "<dataset> <eps_tag> <alloc_tag> <seed>"），CPU 限前 56 核。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-27 [1]=28-55)

run_queue() {
  local gpu=$1
  while read -r ds tag alloc seed; do
    [ -z "$ds" ] && continue
    local name=${ds}_eps${tag}_alloc_${alloc}_seed${seed}
    [ -s results/alloc/${name}.csv ] && { echo "skip $name"; continue; }
    echo "$(date -u +%FT%TZ) START $name gpu$gpu"
    CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
      --data ${ds}_alloc${tag}${alloc}_s${seed} \
      --rounds 30000 --retries 60 --batched --pairing --gpu --work-rows 4096 --work-below 0.2 \
      --stop-threshold 1e-4 --stop-lag 100 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 \
      --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select \
      --init-oneway --exam-weights sigma --struct-ruler ding --struct-boost 4 \
      --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed --progress 200 \
      --save-table results/alloc/${name}.csv --out results/alloc/${name}_curve.csv \
      --gpu-entry-budget 40000000 > logs_eps/alloc/${name}.log 2>&1
    echo "$(date -u +%FT%TZ) END $name rc=$?"
  done < logs_eps/alloc/jobs_gpu${gpu}.txt
}
run_queue 0 > logs_eps/alloc/queue_gpu0.log 2>&1 &
run_queue 1 > logs_eps/alloc/queue_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
