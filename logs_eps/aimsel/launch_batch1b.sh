#!/bin/bash
# 第五十四步第一批 b 段（A6000 端）：填满三条引擎流，不让任何一张卡空转。
# 每条 lane 等本 lane 的 a 段 worker 退出（batch1_gpu$gpu.log 出现"队列空"）后立刻接上，从 jobs_batch1b.txt 抢任务：
#   alloc  <ds> <eps_tag> <alloc_tag> <seed>  —— 第 1 步固定分配全测（make_alloc_exam k105f05/k120f05，Δ₂=1），
#                                              引擎原生配置同第五十二步 launch_alloc.sh；
#   aimsel <ds> <eps_tag> <eps> <seed>        —— 第 2 步 AIM 选择器 × 我们引擎（run_aim_select_v2.py），同 a 段。
# 内容：adult ε=10 全测 ×3（对照 aimsel ε=10 是否趋近全测）；nltcs ε=0.1/1/10 aimsel ×3 与全测 Δ₂=1 ×3（换数据集）。
# CPU 仍守 0-55：引擎 lane 0-23 / 24-47，官方 AIM lane 48-55 由 a 段脚本继续占着。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"

run_aimsel() {  # gpu ds eps_tag eps seed
  local gpu=$1 ds=$2 tag=$3 eps=$4 seed=$5 name=${2}_eps${3}_aimsel_seed$5
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data $ds --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_${ds}${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_alloc() {  # gpu ds eps_tag alloc_tag seed
  local gpu=$1 ds=$2 tag=$3 alloc=$4 seed=$5 name=${2}_eps${3}_alloc_${4}_seed$5
  [ -s results/alloc/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data ${ds}_alloc${tag}${alloc}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/alloc/${name}.csv --out results/alloc/${name}_curve.csv \
    $EXTRA > logs_eps/alloc/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

worker() {
  local gpu=$1 job
  until grep -q "队列空" logs_eps/aimsel/batch1_gpu$gpu.log 2>/dev/null; do sleep 30; done
  echo "$(date -u +%FT%TZ) gpu$gpu a 段已空，接 b 段"
  while :; do
    job=$(bash logs_eps/aimsel/claim_batch1b.sh a6000gpu$gpu); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aimsel) run_aimsel $gpu $2 $3 $4 $5 ;;
      alloc) run_alloc $gpu $2 $3 $4 $5 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu b 段队列空"
}

worker 0 > logs_eps/aimsel/batch1b_gpu0.log 2>&1 &
worker 1 > logs_eps/aimsel/batch1b_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
