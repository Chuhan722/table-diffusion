#!/bin/bash
# 第五十三步：adult ε=1 aimsel s3 改在本机 4090 GPU0 跑（A6000 队列已摘掉），CPU 限 16 核。
cd /home/chuhan/projects0916/residual-evolution || exit 1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
name=adult_eps1_aimsel_seed3
echo "$(date -u +%FT%TZ) START $name local-4090-gpu0"
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv/bin/python -u -B scripts/run_aim_select_v2.py \
  --data adult --epsilon 1 --seed 3 --engine-rounds 1000 \
  --workdir runs/aimsel2_adult1_s3 --out results/aimsel/${name}.csv \
  --engine-extra "$EXTRA" > logs_eps/aimsel/${name}.log 2>&1
echo "$(date -u +%FT%TZ) END $name rc=$?"
