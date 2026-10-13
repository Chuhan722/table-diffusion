#!/bin/bash
# 第二阶段 B 试点：没测的对补 PGM 推断目标。ε=10 种子 1，hybrid 臂命令逐字 + 只换数据目录 adult_aimrep10_s1_infer。
# 用法（A6000）：cd /root/residual-evolution && (nohup bash logs_eps/stage2/run_infer_pilot.sh <gpu 0|1> [tag seed] < /dev/null > logs_eps/stage2/run_infer_pilot.log 2>&1 &)
cd /root/residual-evolution || exit 1
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
gpu=${1:-1}; tag=${2:-10}; seed=${3:-1}
# 与 logs_eps/stage1/launch_stage1.sh 的 EXTRA 逐字同
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
name=adult_eps${tag}_hybridinfer_seed${seed}
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu | tr -d ' ')
[ "$used" -gt 5000 ] && { echo "gpu$gpu 显存已用 ${used} MiB，不发"; exit 2; }
echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
  --data adult_aimrep${tag}_s${seed}_infer --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
  --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
  --save-table results/stage2/$name.csv --out results/stage2/${name}_curve.csv \
  $EXTRA > logs_eps/stage2/$name.log 2>&1
echo "$(date -u +%FT%TZ) END $name rc=$?"
