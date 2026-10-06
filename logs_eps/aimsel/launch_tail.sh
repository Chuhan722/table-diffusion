#!/bin/bash
# 第一步尾段改排程（2026-10-06 18:55Z）：原队列砍掉 adult ε=1 auto s3 / k58f30 s2,s3 / k105f05 s3，
# 两卡各等自己 lane 上正在跑的 k105f05 引擎（PID 886530 / 886846）退出后，从共享任务表抢任务：
# 第二步 aimsel adult ε=0.1 三种子 + nltcs alloc auto 六场。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-27 [1]=28-55)
declare -A WAITPID=([0]=886530 [1]=886846)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/aimsel/jobs_tail.txt

next_job() {
  flock "$JOBS.lock" bash -c "l=\$(head -1 $JOBS); if [ -n \"\$l\" ]; then sed -i 1d $JOBS; echo \"\$l\"; fi"
}

run_aimsel() {  # gpu seed
  local gpu=$1 seed=$2 name=adult_eps0p1_aimsel_seed$2
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data adult --epsilon 0.1 --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_adult0p1_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_alloc() {  # gpu ds tag seed  (nltcs，无 gpu-entry-budget)
  local gpu=$1 ds=$2 tag=$3 seed=$4 name=${2}_eps${3}_alloc_auto_seed${4}
  [ -s results/alloc/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data ${ds}_alloc${tag}auto_s${seed} \
    --rounds 30000 --retries 60 --batched --pairing --gpu --work-rows 4096 --work-below 0.2 \
    --stop-threshold 1e-4 --stop-lag 100 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 \
    --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select \
    --init-oneway --exam-weights sigma --struct-ruler ding --struct-boost 4 \
    --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed --progress 200 \
    --save-table results/alloc/${name}.csv --out results/alloc/${name}_curve.csv > logs_eps/alloc/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

worker() {
  local gpu=$1
  while kill -0 ${WAITPID[$gpu]} 2>/dev/null; do sleep 30; done
  echo "$(date -u +%FT%TZ) gpu$gpu lane 空出"
  while :; do
    job=$(next_job); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aimsel) run_aimsel $gpu $2 ;;
      alloc) run_alloc $gpu $2 $3 $4 ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}
worker 0 > logs_eps/aimsel/tail_gpu0.log 2>&1 &
worker 1 > logs_eps/aimsel/tail_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
