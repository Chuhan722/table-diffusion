#!/bin/bash
# 第 0 步回放：AIM 选测录像考卷 + 我们的生成器。GPU0/GPU1 各一条串行队列，CPU 限前 56 核。
# 用法: bash logs_eps/aimreplay/launch_hybrid.sh   （在 /root/residual-evolution 下）
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
GPU0=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed
GPU1=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082

ensure_data() {  # 等 AIM 录像落盘后转考卷（每个 tag/seed 只属于一条队列，无竞争）
  local tag=$1 seed=$2
  local dir=data/adult_aimrep${tag}_s${seed}
  local dump=results/aimreplay/adult_eps${tag}_aimfull_seed${seed}_measurements.json
  local log=logs_eps/aimreplay/adult_eps${tag}_aimfull_seed${seed}.log
  while [ ! -s $dir/README.md ]; do
    if grep -q "录像" $log 2>/dev/null && [ -s $dump ]; then
      .venv/bin/python scripts/aim_replay_exam.py --dump $dump --real data/adult/adult.csv \
        --name adult_aimrep${tag}_s${seed} --force >> logs_eps/aimreplay/convert.log 2>&1
    else
      sleep 60
    fi
  done
}

run_one() {  # tag seed weights(sigma|none) gpu_uuid lane
  local tag=$1 seed=$2 wts=$3 uuid=$4 lane=$5
  ensure_data $tag $seed
  local suffix=hybrid; [ "$wts" = none ] && suffix=hybridnw
  local name=adult_eps${tag}_${suffix}_seed${seed}
  [ -s results/aimreplay/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name"
  CUDA_VISIBLE_DEVICES=$uuid taskset -c $lane .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} \
    --rounds 30000 --retries 60 --batched --pairing --gpu --work-rows 4096 --work-below 0.2 \
    --stop-threshold 1e-4 --stop-lag 100 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 \
    --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select \
    --init-oneway --exam-weights $wts --struct-ruler ding --struct-boost 4 \
    --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed --progress 200 \
    --save-table results/aimreplay/${name}.csv --out results/aimreplay/${name}_curve.csv \
    --gpu-entry-budget 40000000 > logs_eps/aimreplay/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

queue0() {  # GPU0: ε=0.1 s1 s3 → ε=1 s1 s3
  run_one 0p1 1 sigma $GPU0 0-27; run_one 0p1 3 sigma $GPU0 0-27
  run_one 1 1 sigma $GPU0 0-27;   run_one 1 3 sigma $GPU0 0-27
}
queue1() {  # GPU1: ε=0.1 s2 → ε=1 s2 → ε=0.1 不加权消融 s1 s2 s3
  run_one 0p1 2 sigma $GPU1 28-55; run_one 1 2 sigma $GPU1 28-55
  run_one 0p1 1 none $GPU1 28-55; run_one 0p1 2 none $GPU1 28-55; run_one 0p1 3 none $GPU1 28-55
}
queue0 > logs_eps/aimreplay/queue_gpu0.log 2>&1 &
queue1 > logs_eps/aimreplay/queue_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
