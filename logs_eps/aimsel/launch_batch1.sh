#!/bin/bash
# 第五十四步第一批（2026-10-07 05:00Z 发射，A6000 端）：
#   1) 第 2 步自适应选测（AIM 选择器 × 我们引擎，Δ₂=1，配置与第五十二/五十三步完全一致）adult：
#      ε=10 s1–3（看自适应是否自动趋近全测）、ε=1 s4–8、ε=0.1 s4–8，共 13 只，任务表 jobs_batch1.txt 按长活优先排序；
#   2) 官方 AIM adult ε=1 / ε=0.1 s4–8 共 10 只（aim2 环境纯 CPU，附录测量录像），与 aimsel 一起凑足 8 种子。
# 三条引擎流（A6000 GPU0、GPU1、本机 4090 GPU0）经 claim_batch1.sh 抢同一份任务表（flock，本机走 ssh）。
# A6000 CPU 占用守在 0-55：引擎两条 lane 各 24 核（0-23、24-47），官方 AIM 一条 lane 8 核（48-55）；绝不并发两只引擎于一 lane。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
AIM_LANE=48-55
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"

run_aimsel() {  # gpu eps_tag eps seed
  local gpu=$1 tag=$2 eps=$3 seed=$4 name=adult_eps${2}_aimsel_seed$4
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data adult --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_adult${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

worker() {
  local gpu=$1 job
  while :; do
    job=$(bash logs_eps/aimsel/claim_batch1.sh a6000gpu$gpu); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aimsel) run_aimsel $gpu $2 $3 $4 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}

aim_worker() {  # 官方 AIM，纯 CPU 顺序跑，ε=1 在前（更长）
  local eps tag s name
  for eps in 1 0.1; do
    tag=$([ "$eps" = "0.1" ] && echo 0p1 || echo 1)
    for s in 4 5 6 7 8; do
      name=adult_eps${tag}_aimfull_seed${s}
      [ -s results/${name}.csv ] && { echo "skip $name"; continue; }
      echo "$(date -u +%FT%TZ) START $name lane=$AIM_LANE"
      env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
          CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
        taskset -c $AIM_LANE /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_aim_full.py \
          --data adult --mechanisms /root/private-pgm/mechanisms --epsilon $eps --delta 1e-5 --seed $s \
          --max-model-size 80 --clear-caches-every 1 --clear-caches-every-projects 500 \
          --out results/${name}.csv --dump-measurements results/aimreplay/${name}_measurements.json \
          > logs_eps/aimsel/${name}.log 2>&1
      echo "$(date -u +%FT%TZ) END $name rc=$?"
    done
  done
  echo "$(date -u +%FT%TZ) 官方 AIM 10 只跑完"
}

worker 0 > logs_eps/aimsel/batch1_gpu0.log 2>&1 &
sleep 20
worker 1 > logs_eps/aimsel/batch1_gpu1.log 2>&1 &
aim_worker > logs_eps/aimsel/batch1_aim.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
