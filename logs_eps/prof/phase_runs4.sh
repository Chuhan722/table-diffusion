#!/bin/bash
# 分相计时（四）：提速甲/乙动手前的基线（笔记第五十九步 12 第 0 步）。
# 1A 命令逐字（GSD 同卷 measured_2way.json、--exam-weights none、EXTRA 同 launch_stage1.sh），
# 用 1A 终表热启动各 300 轮：ε=0.1 与 ε=3 背靠背、再重复一遍看轮间噪声；最后补一场 ε=3 冷启 300 轮对照 10-09 剖析（253 ms/轮）。
# --stop-threshold 0 关平台停，保证整 300 轮都跑到（判停只读损失不碰计算，关掉不改每轮工作量）。
# A6000 GPU1、CPU 道 24-47（与 1A 同道）。用法：cd /root/residual-evolution && (nohup bash logs_eps/prof/phase_runs4.sh < /dev/null > logs_eps/prof/phase_runs4.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 100 --gpu-entry-budget 40000000"
P=logs_eps/prof
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); [ "$u" -gt 5000 ]; }
if gpu_busy; then echo "gpu1 被占用，退出"; exit 2; fi
sleep 60
if gpu_busy; then echo "gpu1 被占用（第二次），退出"; exit 2; fi
run() {  # name tag rounds init...
  local name=$1 tag=$2 rounds=$3; shift 3
  echo "$(date -u +%FT%TZ) START $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  taskset -c 24-47 .venv/bin/python -u -B $P/phase_timer.py \
    --data adult_alloc${tag}base_s1 --exam measured_2way.json --rounds $rounds --stop-threshold 0 \
    "$@" --exam-weights none --menu-seed 20260921 --sample-seed 1 --init-seed 1 \
    $EXTRA > $P/phase_${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
run base_adult0p1_late_a 0p1 300 --init-table results/stage1/adult_eps0p1_ours2way_seed1.csv
run base_adult3_late_a   3   300 --init-table results/stage1/adult_eps3_ours2way_seed1.csv
run base_adult0p1_late_b 0p1 300 --init-table results/stage1/adult_eps0p1_ours2way_seed1.csv
run base_adult3_late_b   3   300 --init-table results/stage1/adult_eps3_ours2way_seed1.csv
run base_adult3_early    3   300 --init-avg --init-clean
echo "$(date -u +%FT%TZ) ALL DONE PHASE4"
