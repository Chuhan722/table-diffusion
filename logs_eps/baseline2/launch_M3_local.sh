#!/bin/bash
# M3 本机段（4090 GPU0，核 0-15）：等 c 段 adult ε=10 aimsel s1 退出后，接 acsmob ε=1 全测 Δ₂=1 auto-f1 三种子（与 A6000 的 f1=0.5 版对照）。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
mkdir -p results/alloc logs_eps/alloc
until grep -q "END adult_eps10_aimsel_seed1\|本机 c 段结束" logs_eps/aimsel/launch_batch1c_local.log 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) c 段已退出，接 M3 本机段"
for seed in 1 2 3; do
  name=acsmob_eps1_alloc_k210auto_seed$seed
  [ -s results/alloc/$name.csv ] && { echo "skip $name"; continue; }
  echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_plants.py \
    --data acsmob_alloc1k210auto_s$seed --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table results/alloc/$name.csv --out results/alloc/${name}_curve.csv \
    $EXTRA > logs_eps/alloc/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done
echo "$(date -u +%FT%TZ) 本机 M3 段结束"
