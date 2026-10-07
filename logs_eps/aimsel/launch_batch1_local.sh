#!/bin/bash
# 第五十四步第一批（本机 4090 端）：第三条引擎流，GPU0、CPU 限 16 核（0-15），
# 经 ssh 调 A6000 上的 claim_batch1.sh 与两条 A6000 lane 抢同一份任务表；ssh 失败重试 5 次后放弃。
# 本机速度约为 A6000 lane 的两倍（ε=1 单种子 65 分钟 vs 132–153 分钟），故本机先启动，抢到最长的 ε=10 s1。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes -p 6006 root@10.8.176.53"

next_job() {
  local out i
  for i in 1 2 3 4 5; do
    out=$($SSH bash /root/residual-evolution/logs_eps/aimsel/claim_batch1.sh local4090 2>/dev/null) && { echo "$out"; return 0; }
    echo "$(date -u +%FT%TZ) ssh 领取失败第 $i 次，60s 后重试" >&2
    sleep 60
  done
  return 1
}

run_aimsel() {  # eps_tag eps seed
  local tag=$1 eps=$2 seed=$3 name=adult_eps${1}_aimsel_seed$3
  [ -s results/aimsel/${name}.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_aim_select_v2.py \
    --data adult --epsilon $eps --seed $seed --engine-rounds 1000 \
    --workdir runs/aimsel2_adult${tag}_s${seed} --out results/aimsel/${name}.csv \
    --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}

while :; do
  job=$(next_job) || { echo "$(date -u +%FT%TZ) ssh 连续失败，本机流退出"; break; }
  [ -z "$job" ] && break
  set -- $job
  case $1 in
    aimsel) run_aimsel $2 $3 $4 ;;
    *) echo "未知任务: $job" ;;
  esac
done
echo "$(date -u +%FT%TZ) 本机流结束（队列空）"
