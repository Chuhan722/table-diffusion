#!/bin/bash
# 第五十四步第一批 c 段（本机 4090 专属，夜间填充）：adult ε=10 的第 2 步自适应选测 s1–3。
# 为什么只放本机：ε=10 时 σ₀=8.65，退火判据阈值 0.8σ₀≈6.9 计数/格，而我们引擎每轮热启动续跑在已测对上的
# 固有变动约 20 计数/格（ε=1 日志实测），退火永远触发不了 → 循环会跑满 ~225 轮（每轮一对，σ 不变），
# 单种子本机约 10 小时、A6000 lane 约 22 小时。A6000 两条 lane 不能被它锁一天，故 05:14Z 杀掉 a 段里的三只，
# 挪到这里，等本机 b 段退出后顺序跑；任务表 jobs_batch1c_local.txt 本机 flock 领取，可随时改。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
JOBS=logs_eps/aimsel/jobs_batch1c_local.txt

next_job() {
  flock "$JOBS.lock" bash -c "l=\$(head -1 $JOBS); if [ -n \"\$l\" ]; then sed -i 1d $JOBS; echo \"\$l\"; fi"
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

until grep -q "本机 b 段结束\|本机流退出" logs_eps/aimsel/launch_batch1b_local.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) b 段已退出，接 c 段"
while :; do
  job=$(next_job); [ -z "$job" ] && break
  set -- $job
  case $1 in
    aimsel) run_aimsel $2 $3 $4 $5 ;;
    *) echo "未知任务: $job" ;;
  esac
done
echo "$(date -u +%FT%TZ) 本机 c 段结束（队列空）"
