#!/bin/bash
# GPU build 微基准（甲 GPU 侧动手前）：ε=3 暖、第 20 轮的 build 入参原地重跑 30 次，量 墙钟 / CPU 线程 / cupy 调用数。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/prof/bench_jia2.sh [标签=bench] < /dev/null > logs_eps/prof/bench_jia2_<标签>.log 2>&1 &)
cd /root/residual-evolution || exit 1
TAG=${1:-bench}
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 100 --gpu-entry-budget 40000000"
P=logs_eps/prof
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); [ "$u" -gt 5000 ]; }
if gpu_busy; then echo "gpu1 被占用，退出"; exit 2; fi
for tag in 3 0p1; do
  echo "$(date -u +%FTT%TZ) START eps=$tag load=$(cut -d' ' -f1-3 /proc/loadavg)"
  BENCH_ROUND=20 BENCH_REPEAT=30 taskset -c 24-47 .venv/bin/python -u -B $P/build_bench.py \
    --data adult_alloc${tag}base_s1 --exam measured_2way.json --rounds 300 --stop-threshold 0 \
    --init-table results/stage1/adult_eps${tag}_ours2way_seed1.csv --exam-weights none --menu-seed 20260921 --sample-seed 1 --init-seed 1 \
    $EXTRA 2>&1 | grep -E "^BENCH|Error|error|Traceback" 
  echo "$(date -u +%FT%TZ) END eps=$tag"
done
echo "$(date -u +%FT%TZ) ALL DONE"
