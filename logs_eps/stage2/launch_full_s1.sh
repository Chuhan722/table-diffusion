#!/bin/bash
# 第二阶段 B 核对：一条命令版完整配方（清洗 → 收敛 → 纤维洗牌 100 趟 → 热启动再收敛，笔记第五十九步 7），hybridfull 臂，只跑 adult ε=0.1 s1 一道，与 2B-4 的 hybridfiberp1u s1 对分。
# 与 launch_fiberp1u_batch.sh 的差别：数据目录用原卷 adult_aimrep0p1_s1（不用预先补过卷的 _fiberp1u 目录），去掉 --oneway-pool-cover-ignore-prefix（清洗在并卷前对原卷做），加 --fiber-passes 100。
# 用法（A6000，一条 ssh 发）：cd /root/residual-evolution && (nohup bash logs_eps/stage2/launch_full_s1.sh <gpu 0|1> < /dev/null > logs_eps/stage2/launch_full_s1_gpu<N>.log 2>&1 &)
cd /root/residual-evolution || exit 1
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
gpu=$1
JOBS="0p1:1"
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --oneway-pool 2 --oneway-pool-scope uncovered --fiber-passes 100"
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ "$u" -gt 5000 ]; }
wait_gpu_idle() {  # 连续两次空闲才放行（别人占 <5 GB 视为空闲）
  local waited=0
  while :; do
    if ! gpu_busy $gpu; then sleep 60; gpu_busy $gpu || break; fi
    [ $((waited % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu$gpu 被占用，等待中（已等 $((waited / 60)) min）"
    sleep 120; waited=$((waited + 120))
  done
}
for job in $JOBS; do
  tag=${job%%:*}; seed=${job##*:}; name=adult_eps${tag}_hybridfull_seed${seed}
  [ -s results/stage2/$name.csv ] && { echo "skip $name"; continue; }
  wait_gpu_idle
  mkdir results/stage2/.lock_$name 2>/dev/null || { echo "skip(running) $name"; continue; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/stage2/$name.csv --out results/stage2/${name}_curve.csv \
    $EXTRA > logs_eps/stage2/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"; rmdir results/stage2/.lock_$name
done
echo "$(date -u +%FT%TZ) gpu$gpu 队列完"
