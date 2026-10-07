#!/bin/bash
# f1 补数（本机 4090 GPU0，核 0-15）：全测 K=all 下"一阶占比"两种花法对照——adult k105auto(f1=0.015) 三档 ε × 3 种子 + nltcs ε=10 k120auto(f1=0.043) × 3。
# 已有对照：adult k105f05 三档 ε、nltcs k120f05 三档 ε、nltcs auto(=k120auto) ε=0.1/1、plants f05/f10/auto。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
mkdir -p results/alloc logs_eps/alloc
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
for s in 1 2 3; do run adult 1 k105auto $s; done
for s in 1 2 3; do run adult 0p1 k105auto $s; done
for s in 1 2 3; do run nltcs 10 k120auto $s; done
for s in 1 2 3; do run adult 10 k105auto $s; done
echo "$(date -u +%FT%TZ) 本机 f1 补数结束"
