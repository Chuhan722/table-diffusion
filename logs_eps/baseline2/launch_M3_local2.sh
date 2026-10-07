#!/bin/bash
# M3 本机段 v2（4090 GPU0，核 0-15）：plants ε=1 全测 Δ₂=1 三配置 × 三种子（N×Q=1.6 亿格，显存够）。
# acsmob 在 4090 上 OOM（N×Q=8.9 亿格，需 26 GB > 24 GB），已全部移回 A6000 队列。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
mkdir -p results/alloc logs_eps/alloc
for tag in k2346f05 k2346f10 k2346auto; do for seed in 1 2 3; do
  name=plants_eps1_alloc_${tag}_seed$seed
  [ -s results/alloc/$name.csv ] && { echo "skip $name"; continue; }
  echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_plants.py \
    --data plants_alloc1${tag}_s$seed --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/alloc/$name.csv --out results/alloc/${name}_curve.csv \
    $EXTRA > logs_eps/alloc/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done; done
echo "$(date -u +%FT%TZ) 本机 M3 段 v2 结束"
