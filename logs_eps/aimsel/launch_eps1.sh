#!/bin/bash
# 第五十三步（2026-10-07 01:30Z 发射）：
#   1) 第 2 步自适应选测 adult ε=1 三种子（run_aim_select_v2.py，Δ₂=1 记账，配置与 ε=0.1 完全一致）；
#   2) 消融 A：ε=0.1 三种子的末轮改成"热启动续跑"（--init-table 循环最后一张中间表 + 同一份 exam_final.json），
#      与现行"--init-oneway 一次性重拟合"对照，复刻第二十一步旧做法；σ 加权消融第五十一步已做（不加权 +17.6%）。
# 两卡从共享任务表抢任务（flock），每卡一条 lane，绝不并发两只引擎。
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-27 [1]=28-55)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/aimsel/jobs_eps1.txt

next_job() {
  flock "$JOBS.lock" bash -c "l=\$(head -1 $JOBS); if [ -n \"\$l\" ]; then sed -i 1d $JOBS; echo \"\$l\"; fi"
}

run_aimsel() {  # gpu eps_tag eps seed
  local gpu=$1 tag=$2 eps=$3 seed=$4 name=adult_eps${2}_aimsel_seed$4
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name gpu$gpu"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data adult --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_adult${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_ablate_ws() {  # gpu seed：ε=0.1 末轮热启动续跑
  local gpu=$1 seed=$2 name=adult_eps0p1_aimselws_seed$2 wd=/root/residual-evolution/runs/aimsel2_adult0p1_s$2
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  local last
  last=$(ls $wd/table_round*.csv | sed 's/.*round\([0-9]*\)\.csv/\1/' | sort -n | tail -1)
  echo "$(date -u +%FT%TZ) START $name gpu$gpu init=table_round$last"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult --exam $wd/exam_final.json --init-table $wd/table_round$last.csv \
    --rounds 30000 --exam-weights sigma --stop-threshold 1e-4 --stop-lag 100 \
    --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/aimsel/${name}.csv --out results/aimsel/${name}_curve.csv \
    $EXTRA > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

worker() {
  local gpu=$1
  while :; do
    job=$(next_job); [ -z "$job" ] && break
    set -- $job
    case $1 in
      aimsel) run_aimsel $gpu $2 $3 $4 ;;
      ablate_ws) run_ablate_ws $gpu $2 ;;
    esac
  done
  echo "$(date -u +%FT%TZ) gpu$gpu 队列空"
}
worker 0 > logs_eps/aimsel/eps1_gpu0.log 2>&1 &
sleep 20
worker 1 > logs_eps/aimsel/eps1_gpu1.log 2>&1 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
