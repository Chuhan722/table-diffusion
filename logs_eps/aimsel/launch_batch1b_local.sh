#!/bin/bash
# 第五十四步第一批 b 段（本机 4090 端）：a 段本机流一退出（launch_batch1_local.log 出现"本机流结束/退出"）立刻接上，
# 经 ssh 调 A6000 上的 claim_batch1b.sh 抢 jobs_batch1b.txt；任务类型与 launch_batch1b.sh 相同（alloc / aimsel）。
# GPU0、CPU 限 16 核（0-15）。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes -p 6006 root@10.8.176.53"

next_job() {
  local out i
  for i in 1 2 3 4 5; do
    out=$($SSH bash /root/residual-evolution/logs_eps/aimsel/claim_batch1b.sh local4090 2>/dev/null) && { echo "$out"; return 0; }
    echo "$(date -u +%FT%TZ) ssh 领取失败第 $i 次，60s 后重试" >&2
    sleep 60
  done
  return 1
}

run_aimsel() {  # ds eps_tag eps seed
  local ds=$1 tag=$2 eps=$3 seed=$4 name=${1}_eps${2}_aimsel_seed$4
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data $ds --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_${ds}${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

run_alloc() {  # ds eps_tag alloc_tag seed
  local ds=$1 tag=$2 alloc=$3 seed=$4 name=${1}_eps${2}_alloc_${3}_seed$4
  [ -s results/alloc/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_plants.py \
    --data ${ds}_alloc${tag}${alloc}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/alloc/${name}.csv --out results/alloc/${name}_curve.csv \
    $EXTRA > logs_eps/alloc/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

until grep -q "本机流结束\|本机流退出" logs_eps/aimsel/launch_batch1_local.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) a 段本机流已退出，接 b 段"
while :; do
  job=$(next_job) || { echo "$(date -u +%FT%TZ) ssh 连续失败，本机流退出"; break; }
  [ -z "$job" ] && break
  set -- $job
  case $1 in
    aimsel) run_aimsel $2 $3 $4 $5 ;;
    alloc) run_alloc $2 $3 $4 $5 ;;
    *) echo "未知任务: $job" ;;
  esac
done
echo "$(date -u +%FT%TZ) 本机 b 段结束（队列空）"
