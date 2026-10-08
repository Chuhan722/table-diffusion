#!/bin/bash
# C 组 ε=3 本机第三段（4090 GPU0，核 0-15；等第二段结束后接上）：nltcs ε=3 第 2 步选择器(aimsel) ×3（分钟级），补齐 nltcs ε=3 四家同点位。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
mkdir -p results/alloc results/aimsel logs_eps/alloc logs_eps/aimsel
until grep -q "本机 C 组第二段结束" logs_eps/baseline2/launch_C_local2.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) 第二段已收，接第三段"
run() {  # ds eps_tag alloc_tag seed
  local name=${1}_eps${2}_alloc_${3}_seed$4
  [ -s results/alloc/$name.csv ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_plants.py \
    --data ${1}_alloc${2}${3}_s$4 --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + $4)) --sample-seed $4 --init-seed $4 \
    --save-table results/alloc/$name.csv --out results/alloc/${name}_curve.csv \
    $EXTRA > logs_eps/alloc/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
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
for s in 1 2 3; do run_aimsel nltcs 3 3 $s; done
echo "$(date -u +%FT%TZ) 本机 C 组第三段结束"
