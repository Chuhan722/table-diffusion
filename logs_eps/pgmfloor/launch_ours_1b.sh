#!/bin/bash
# 第六十六步：1B 比生成器（我们 vs PGM，同一 AIM 选测录像）我们臂发射器。
#   第六十五步定稿流程：第一段 --stop-floor 地板停 ×1.0（兜底 plateau 1e-4/100 轮、上限 30000）
#   → 纤维洗牌 100 趟 → 第二段真题锁定 --fiber-lock-real（地板停自动关，plateau+上限停）。
#   比生成器口径：不清洗（无 --oneway-pool），其余与 logs_eps/stage2/launch_full_s1.sh 逐字同。
#   地板阈值来自 logs_eps/pgmfloor/floor_1b.json 的 ours_floor（½·题数·σ_min²）。
#   只用 GPU0/1，谁先空闲（<5 GB 连续两次）用谁；每场结束后再挑。
# 用法（A6000，一条 ssh 发）：
#   cd /root/residual-evolution && (nohup bash logs_eps/pgmfloor/launch_ours_1b.sh "1:1" < /dev/null > logs_eps/pgmfloor/launch_ours_1b.log 2>&1 &)
#   参数为空格分隔的 "ε标签:种子" 列表，默认先导场 1:1。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
declare -A FLOOR=(
  [0p1:1]=6287575.166310625 [0p1:2]=1823340.3762813245 [0p1:3]=5123813.774480508
  [0p3:1]=1399681.5247673127 [0p3:2]=1800172.6178743835 [0p3:3]=1630659.5345760211
  [1:1]=299038.1812299605 [1:2]=225753.08438055194 [1:3]=190205.15277737722
  [3:1]=68465.50849395874 [3:2]=48037.311033374186 [3:3]=51669.8821903612
  [10:1]=9924.017934371434 [10:2]=14727.64151114602 [10:3]=14291.893133479229
)
JOBS=${1:-"1:1"}
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --fiber-passes 100 --fiber-lock-real"
OUT=results/pgmfloor
LOG=logs_eps/pgmfloor
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
pick_gpu() {  # 轮询 GPU0/1，先空闲者（连续两次）胜出，写入 GPU 变量
  local w=0 g
  while :; do
    for g in 0 1; do
      if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then GPU=$g; return; fi; fi
    done
    [ $((w % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu0/1 均被占用，等待中（已等 $((w / 60)) min）"
    sleep 120; w=$((w + 120))
  done
}

for job in $JOBS; do
  tag=${job%%:*}; seed=${job##*:}; name=adult_eps${tag}_oursfloor_seed${seed}
  [ -z "${FLOOR[$job]}" ] && { echo "no floor for $job"; continue; }
  [ -s $OUT/$name.csv ] && { echo "skip $name"; continue; }
  pick_gpu
  mkdir $OUT/.lock_$name 2>/dev/null || { echo "skip(running) $name"; continue; }
  echo "$(date -u +%FT%TZ) START $name gpu$GPU lane=${LANE[$GPU]} floor=${FLOOR[$job]} load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$GPU]} taskset -c ${LANE[$GPU]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --stop-floor ${FLOOR[$job]} \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"; rmdir $OUT/.lock_$name
done
echo "$(date -u +%FT%TZ) 队列完"
