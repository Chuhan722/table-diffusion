#!/bin/bash
# 第二阶段 B 批量：没测的对补 PGM 推断目标，hybridinfer 臂，5 ε × 3 种子（ε=10 s1 试点已跑）。
# 用法（A6000，两条 ssh 分别发）：cd /root/residual-evolution && (nohup bash logs_eps/stage2/launch_infer_batch.sh <gpu 0|1> < /dev/null > logs_eps/stage2/launch_infer_batch_gpu<N>.log 2>&1 &)
cd /root/residual-evolution || exit 1
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
gpu=$1
# 按预计时长配平：ε=10 ~31 min、3 ~25、1 ~20、0.3 ~10、0.1 ~6
if [ "$gpu" = "0" ]; then JOBS="10:2 3:1 3:3 1:1 0p3:1 0p1:1 0p1:3"; else JOBS="10:3 3:2 1:2 1:3 0p3:2 0p3:3 0p1:2"; fi
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
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
  tag=${job%%:*}; seed=${job##*:}; name=adult_eps${tag}_hybridinfer_seed${seed}
  [ -s results/stage2/$name.csv ] && { echo "skip $name"; continue; }
  wait_gpu_idle
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed}_infer --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/stage2/$name.csv --out results/stage2/${name}_curve.csv \
    $EXTRA > logs_eps/stage2/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done
echo "$(date -u +%FT%TZ) gpu$gpu 队列完"
