#!/bin/bash
# 第一阶段：纯生成器消融（adult 五档 ε × 3 种子），协议见 docs/第一阶段可比性协议.md。
# 任务四种：
#   ours2way <tag> <seed>  1A 我们臂：吃 GSD 同一份 data/adult_alloc{tag}base_s{seed}/measured_2way.json（GSD 臂已有，不重跑）
#   hybrid   <tag> <seed>  1B 我们臂：吃 AIM 录像转的 data/adult_aimrep{tag}_s{seed}/（录像没落盘就等）
#   aim10    <seed>        CPU：官方 AIM adult ε=10 复跑并开录像（10-06 那三跑没录），完成即转卷
#   pgmrep   <tag> <seed>  CPU：独立 PGM 吃同一份录像出表（1B 对照臂，秒级）
# GPU 只用 0/1 且空闲才用：每场开跑前查 nvidia-smi，显存占用 >1.5 GB 视为别人在用，连续两次（间隔 60 s）空闲才发射。
# CPU 道 48-55 / 64-71 两条（10-11 机器总负载 61/112，56-63 最忙，让开），GPU 道 0-23 / 24-47 守旧例。引擎命令 = logs_eps/baseline2/launch_E_a6000.sh run_alloc 逐字，只改协议写明的三处。
# 用法（A6000）：cd /root/residual-evolution && nohup bash logs_eps/stage1/launch_stage1.sh [all|gpu|cpu|gpu0|gpu1] > logs_eps/stage1/launch_stage1.log 2>&1 &
#   参数选只起 GPU 道或只起 CPU 道（队列文件 flock 共享，可分进程补起）。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
CPU_LANES=(48-55 64-71)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/stage1/jobs_stage1.txt
CJOBS=logs_eps/stage1/jobs_stage1_cpu.txt
OUT=results/stage1
LOG=logs_eps/stage1
AIMPY=/root/miniconda3/envs/aim2/bin/python
mkdir -p $OUT $LOG results/aimreplay

next_job() {  # $1 = 队列文件
  flock "$1.lock" bash -c "l=\$(head -1 $1); if [ -n \"\$l\" ]; then sed -i 1d $1; echo \"\$l\"; fi"
}

gpu_busy() {  # 别人占着就返回 0
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' ')
  [ -z "$used" ] && return 0
  [ "$used" -gt 1500 ]
}

wait_gpu_idle() {  # 连续两次空闲才放行
  local gpu=$1 waited=0
  while :; do
    if ! gpu_busy $gpu; then
      sleep 60
      gpu_busy $gpu || break
    fi
    [ $((waited % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu$gpu 被占用，等待中（已等 $((waited / 60)) min）"
    sleep 120; waited=$((waited + 120))
  done
  [ $waited -gt 0 ] && echo "$(date -u +%FT%TZ) gpu$gpu 空闲，共等 $((waited / 60)) min"
}

ensure_exam() {  # tag seed：录像落盘则转卷，否则等
  local tag=$1 seed=$2
  # 注意 local 一行里后面的赋值看不到前面刚赋的值（bash 先整体展开），dir 必须另起一行；10-12 02:00 因此栽过：转卷死循环 13 h
  local dir=data/adult_aimrep${tag}_s${seed}
  local dump=results/aimreplay/adult_eps${tag}_aimfull_seed${seed}_measurements.json
  local log=logs_eps/stage1/adult_eps${tag}_aimfull_seed${seed}.log
  while [ ! -s $dir/README.md ]; do
    if [ -s $dump ] && { grep -q "录像" $log 2>/dev/null || [ "$tag" != 10 ]; }; then
      flock $LOG/convert_${tag}_s${seed}.lock .venv/bin/python scripts/aim_replay_exam.py --dump $dump --real data/adult/adult.csv \
        --name adult_aimrep${tag}_s${seed} --force >> $LOG/convert.log 2>&1
    else
      sleep 120
    fi
  done
}

run_ours2way() {  # gpu tag seed
  local gpu=$1 tag=$2 seed=$3 name=adult_eps${2}_ours2way_seed$3
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_gpu_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_alloc${tag}base_s${seed} --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-avg --init-clean --exam-weights none --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_hybrid() {  # gpu tag seed
  local gpu=$1 tag=$2 seed=$3 name=adult_eps${2}_hybrid_seed$3
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  ensure_exam $tag $seed
  wait_gpu_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu lane=${LANE[$gpu]}"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_aim10() {  # lane seed
  local lane=$1 seed=$2 name=adult_eps10_aimfull_seed$2
  local dump=results/aimreplay/${name}_measurements.json
  [ -s $dump ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name cpu lane=$lane"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
    taskset -c $lane $AIMPY -u -B scripts/run_aim_full.py \
      --data adult --mechanisms /root/private-pgm/mechanisms --epsilon 10 --delta 1e-5 --seed $seed \
      --max-model-size 80 --clear-caches-every 1 --clear-caches-every-projects 500 \
      --out results/aimreplay/${name}.csv --dump-measurements $dump > $LOG/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
  ensure_exam 10 $seed
}

run_pgmrep() {  # lane tag seed
  local lane=$1 tag=$2 seed=$3 name=adult_eps${2}_pgmrep_seed$3
  local dump=results/aimreplay/adult_eps${tag}_aimfull_seed${seed}_measurements.json
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  while [ ! -s $dump ]; do sleep 120; done
  echo "$(date -u +%FT%TZ) START $name cpu lane=$lane"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
    taskset -c $lane $AIMPY -u -B scripts/run_pgm_replay.py \
      --dump $dump --seed $seed --iters 1000 --known-total rows --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

gpu_worker() {
  local gpu=$1 job
  while :; do
    job=$(next_job $JOBS); [ -z "$job" ] && break
    set -- $job
    case $1 in
      ours2way) run_ours2way $gpu $2 $3 ;;
      hybrid) run_hybrid $gpu $2 $3 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}

cpu_worker() {
  local lane=$1 job
  while :; do
    job=$(next_job $CJOBS); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aim10) run_aim10 $lane $2 ;;
      pgmrep) run_pgmrep $lane $2 $3 ;;
      *) echo "未知任务: $job" ;;
    esac
  done
  echo "$(date -u +%FT%TZ) cpu lane $lane 队列空"
}

MODE=${1:-all}
echo "$(date -u +%FT%TZ) 第一阶段开跑（$MODE）：GPU 队列 $(wc -l < $JOBS) 场，CPU 队列 $(wc -l < $CJOBS) 场"
case $MODE in all|gpu|gpu0) gpu_worker 0 >> $LOG/stage1_gpu0.log 2>&1 & ;; esac
case $MODE in all|gpu|gpu1) gpu_worker 1 >> $LOG/stage1_gpu1.log 2>&1 & ;; esac
case $MODE in all|cpu) for i in 0 1; do cpu_worker ${CPU_LANES[$i]} >> $LOG/stage1_cpu$i.log 2>&1 & done ;; esac
wait
echo "$(date -u +%FT%TZ) ALL DONE stage1"
