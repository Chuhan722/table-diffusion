#!/bin/bash
# 引擎耗时剖析（用户令 10-09 09:05："先看看时间花到哪里了"）。A6000 GPU0，CPU 核 56-79（不占 0-55 的实验道），
# cProfile 跑四段短程：adult ε=3 / acsmob ε=10 各一段冷启动（--init-oneway）和一段终表热启动（--init-table，看后期每轮成本）。
# 命令行参数与 launch_E_a6000.sh 的 run_alloc 逐字相同，只改轮数上限；不写终表不写曲线，不污染 results/。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
export CUDA_VISIBLE_DEVICES=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 100 --gpu-entry-budget 40000000"
P=logs_eps/prof
mkdir -p $P
run() {  # name data rounds init...
  local name=$1 data=$2 rounds=$3; shift 3
  echo "$(date -u +%FT%TZ) START $name"
  taskset -c 56-79 .venv/bin/python -u -B -m cProfile -o $P/$name.prof scripts/run_plants.py \
    --data $data --rounds $rounds --stop-threshold 1e-4 --stop-lag 100 \
    "$@" --exam-weights sigma --menu-seed 20260921 --sample-seed 1 --init-seed 1 \
    $EXTRA > $P/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
run adult3_early  adult_alloc3k105auto_s1  600 --init-oneway
run adult3_late   adult_alloc3k105auto_s1  300 --init-table results/alloc/adult_eps3_alloc_k105auto_seed1.csv
echo "$(date -u +%FT%TZ) ALL DONE PROF2"
