#!/bin/bash
# 第六十七步：1B 比整体方法（我们 vs PGM 原生，同一 AIM 选测录像）我们臂发射器。
#   我们的整体方法（第六十五步定稿）：一阶清洗 z=2（只对未被二阶覆盖的字段）→ 第一段 --stop-floor 地板停 ×1.0
#   （兜底 plateau 1e-4/100 轮、上限 30000）→ 纤维洗牌 100 趟 → 第二段真题锁定 --fiber-lock-real
#   （地板 = 每题平均差一个人 ½Σw，兜底 plateau+上限）。对手 PGM 用第五十一步 pgmrep10k 原生表，不新跑。
#   与 logs_eps/pgmfloor/launch_ours_1b.sh 的差别：加 --oneway-pool 2 --oneway-pool-scope uncovered（清洗只改目标数字，
#   不改 σ 与题数，地板阈值不变，仍用 logs_eps/pgmfloor/floor_1b.json 的 ours_floor）；输出目录与名字改 whole1b/ourswhole。
#   只用 GPU0/1；GPUS 环境变量限定本实例可用的卡（默认 "0 1"），两实例分卡并行时各给一张。
# 用法（A6000，一条 ssh 发，两实例分卡）：
#   cd /root/residual-evolution && (GPUS=0 nohup bash logs_eps/whole1b/launch_ours_whole1b.sh "1:1 1:2 1:3 0p1:1 0p1:2 0p1:3 10:1 10:2" < /dev/null > logs_eps/whole1b/launch_gpu0.log 2>&1 &)
#   cd /root/residual-evolution && (GPUS=1 nohup bash logs_eps/whole1b/launch_ours_whole1b.sh "3:1 3:2 3:3 0p3:1 0p3:2 0p3:3 10:3" < /dev/null > logs_eps/whole1b/launch_gpu1.log 2>&1 &)
#   参数为空格分隔的 "ε标签:种子" 列表，默认全部 15 格。
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
JOBS=${1:-"1:1 1:2 1:3 0p1:1 0p1:2 0p1:3 0p3:1 0p3:2 0p3:3 3:1 3:2 3:3 10:1 10:2 10:3"}
GPUS=${GPUS:-"0 1"}
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --fiber-passes 100 --fiber-lock-real --oneway-pool 2 --oneway-pool-scope uncovered"
OUT=results/whole1b
LOG=logs_eps/whole1b
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
pick_gpu() {  # 轮询 GPU0/1，先空闲者（连续两次）胜出，写入 GPU 变量
  local w=0 g
  while :; do
    for g in $GPUS; do
      if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then GPU=$g; return; fi; fi
    done
    [ $((w % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu($GPUS) 均被占用，等待中（已等 $((w / 60)) min）"
    sleep 120; w=$((w + 120))
  done
}

for job in $JOBS; do
  tag=${job%%:*}; seed=${job##*:}; name=adult_eps${tag}_ourswhole_seed${seed}
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
