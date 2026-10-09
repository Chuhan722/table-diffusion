#!/bin/bash
# 分相计时（三）：提速 A/B/C 终版代码（并集+切块），A6000 GPU1、CPU 核 56-79；参数与 phase_runs.sh 逐字相同（可与旧 phase_*.log 对照）。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 100 --gpu-entry-budget 40000000"
P=logs_eps/prof
run() {  # name data rounds init...
  local name=$1 data=$2 rounds=$3; shift 3
  echo "$(date -u +%FT%TZ) START $name"
  taskset -c 56-79 .venv/bin/python -u -B $P/phase_timer.py \
    --data $data --rounds $rounds --stop-threshold 1e-4 --stop-lag 100 \
    "$@" --exam-weights sigma --menu-seed 20260921 --sample-seed 1 --init-seed 1 \
    $EXTRA > $P/phase_${name}_new3.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
run acsmob10_late  acsmob_alloc10k210auto_s1  100 --init-table results/alloc/acsmob_eps10_alloc_k210auto_seed1.csv
run adult3_early   adult_alloc3k105auto_s1    300 --init-oneway
run adult3_late    adult_alloc3k105auto_s1    200 --init-table results/alloc/adult_eps3_alloc_k105auto_seed1.csv
run acsmob10_early acsmob_alloc10k210auto_s1  100 --init-oneway
run plants10_early plants_alloc10k2346auto_s1 200 --init-oneway
run nltcs3_early   nltcs_alloc3k120auto_s1    300 --init-oneway
echo "$(date -u +%FT%TZ) ALL DONE PHASE3"
