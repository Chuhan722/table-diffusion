#!/bin/bash
# 甲 第一刀疑点排查（三）：首轮时间线。新码（出菜单重写）与旧码（dd58bdc 的 batchmenu.py）各跑一场 ε=3 暖启动 300 轮，背靠背。
cd /root/residual-evolution || exit 1
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 100 --gpu-entry-budget 40000000"
P=logs_eps/prof
ARGS="--data adult_alloc3base_s1 --exam measured_2way.json --rounds 300 --stop-threshold 0 --init-table results/stage1/adult_eps3_ours2way_seed1.csv --exam-weights none --menu-seed 20260921 --sample-seed 1 --init-seed 1 $EXTRA"
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); [ "$u" -gt 5000 ] && { echo "gpu1 被占用，退出"; exit 2; }
cp src/resevo/batchmenu.py /tmp/batchmenu_new.py
echo "$(date -u +%FT%TZ) START new load=$(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c 24-47 .venv/bin/python -u -B $P/phase_timer.py $ARGS > $P/phase_jia1dbg_tl_new.log 2>&1
cp /tmp/batchmenu_old.py src/resevo/batchmenu.py
echo "$(date -u +%FT%TZ) START old load=$(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c 24-47 .venv/bin/python -u -B $P/phase_timer.py $ARGS > $P/phase_jia1dbg_tl_old.log 2>&1
cp /tmp/batchmenu_new.py src/resevo/batchmenu.py
echo "$(date -u +%FT%TZ) START new2 load=$(cut -d' ' -f1-3 /proc/loadavg)"
taskset -c 24-47 .venv/bin/python -u -B $P/phase_timer.py $ARGS > $P/phase_jia1dbg_tl_new2.log 2>&1
md5sum src/resevo/batchmenu.py
echo "$(date -u +%FT%TZ) ALL DONE"
