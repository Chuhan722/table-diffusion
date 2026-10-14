#!/bin/bash
# 甲 第一刀疑点排查（二）：ε=3 暖启动 300 轮，逐轮“相外”时间分布；第二场 PHASE_GC=off 关循环垃圾回收作对照。
cd /root/residual-evolution || exit 1
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 100 --gpu-entry-budget 40000000"
P=logs_eps/prof
ARGS="--data adult_alloc3base_s1 --exam measured_2way.json --rounds 300 --stop-threshold 0 --init-table results/stage1/adult_eps3_ours2way_seed1.csv --exam-weights none --menu-seed 20260921 --sample-seed 1 --init-seed 1 $EXTRA"
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); [ "$u" -gt 5000 ] && { echo "gpu1 被占用，退出"; exit 2; }
echo "$(date -u +%FT%TZ) START rounds load=$(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c 24-47 .venv/bin/python -u -B $P/phase_timer.py $ARGS > $P/phase_jia1dbg_rounds.log 2>&1
echo "$(date -u +%FT%TZ) START gcoff load=$(cut -d' ' -f1-3 /proc/loadavg)"
PHASE_GC=off taskset -c 24-47 .venv/bin/python -u -B $P/phase_timer.py $ARGS > $P/phase_jia1dbg_gcoff.log 2>&1
echo "$(date -u +%FT%TZ) ALL DONE"
