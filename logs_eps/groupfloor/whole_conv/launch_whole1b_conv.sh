#!/bin/bash
# 第六十九步补充：1B 整体方法 第一段跑到收敛（去掉 --stop-floor-groups，由平台早停 1e-4/100 轮、上限 30000 收工），
#   其余与 launch_whole1b_gstop.sh / 第六十七步 launch_ours_whole1b.sh 逐字相同：
#   一阶清洗 z=2（只对未被二阶覆盖的字段）→ 第一段收敛 → 纤维洗牌 100 趟 → 第二段真题锁定 --fiber-lock-real（每题平均差一个人地板）。
#   用途：把"第一段停法"的代价单独算干净（分档停 vs 收敛，二者只差这一个开关），并给出带清洗的收敛版 ÷ PGM 原生。
# 用法（A6000，两实例分卡；用户令只用 GPU0/1）：
#   cd /root/residual-evolution && (GPUS=0 nohup bash logs_eps/groupfloor/whole_conv/launch_whole1b_conv.sh "1:1 1:2 1:3 0p1:1 0p1:2 0p1:3 10:1 10:2" < /dev/null > logs_eps/groupfloor/whole_conv/launch_gpu0.log 2>&1 &)
#   cd /root/residual-evolution && (GPUS=1 nohup bash logs_eps/groupfloor/whole_conv/launch_whole1b_conv.sh "3:1 3:2 3:3 0p3:1 0p3:2 0p3:3 10:3" < /dev/null > logs_eps/groupfloor/whole_conv/launch_gpu1.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
JOBS=${1:-"1:1 1:2 1:3 0p1:1 0p1:2 0p1:3 0p3:1 0p3:2 0p3:3 3:1 3:2 3:3 10:1 10:2 10:3"}
GPUS=${GPUS:-"0 1"}
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --fiber-passes 100 --fiber-lock-real --oneway-pool 2 --oneway-pool-scope uncovered"
OUT=results/groupfloor/whole1b_conv
LOG=logs_eps/groupfloor/whole_conv
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
pick_gpu() {  # 轮询本实例可用的卡，先空闲者（连续两次、间隔 60 s）胜出，写入 GPU 变量
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
  tag=${job%%:*}; seed=${job##*:}; name=adult_eps${tag}_oursconv_seed${seed}
  [ -s $OUT/$name.csv ] && { echo "skip $name"; continue; }
  pick_gpu
  mkdir $OUT/.lock_$name 2>/dev/null || { echo "skip(running) $name"; continue; }
  echo "$(date -u +%FT%TZ) START $name gpu$GPU lane=${LANE[$GPU]} load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$GPU]} taskset -c ${LANE[$GPU]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"; rmdir $OUT/.lock_$name
done
echo "$(date -u +%FT%TZ) 队列完"
