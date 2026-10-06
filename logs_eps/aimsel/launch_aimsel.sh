#!/bin/bash
# 第二步自适应选测（AIM 选择器 + 我们的引擎，unbounded 记账）。等第一步队列 ALL DONE 后再开，
# GPU0 串行 seed 1、3，GPU1 跑 seed 2。adult eps=0.1 三种子。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-27 [1]=28-55)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"

while ! grep -q "ALL DONE" logs_eps/alloc/launch_alloc.log 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) 第一步已完成，开始第二步"

run_one() {  # gpu dataset eps_tag eps seed
  local gpu=$1 ds=$2 tag=$3 eps=$4 seed=$5
  local name=${ds}_eps${tag}_aimsel_seed${seed}
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data $ds --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_${ds}${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
( run_one 0 adult 0p1 0.1 1; run_one 0 adult 0p1 0.1 3 ) > logs_eps/aimsel/queue_gpu0.log 2>&1 &
( run_one 1 adult 0p1 0.1 2 ) > logs_eps/aimsel/queue_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
