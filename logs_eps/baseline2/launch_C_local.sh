#!/bin/bash
# C 组 ε=3 新点位（本机 4090 GPU0，核 0-15）：adult 全测 K=105 Δ₂=1，k105auto(f1=0.015,σ2 15.41) ×3 在前（与 GSD 同餐 σ2 15.31 对齐），k105f05(σ2 21.64) ×3 在后。定位 adult 全测 vs AIM 的翻转点（门槛模型预测 ε=3 K*=100/105）。
# 配套：AIM 官方 adult/nltcs ε=3 ×3 在 A6000 核 48-55（launch_C_aim.sh），GSD 同餐 adult ε=3 ×3 追加进 A6000 jobs_M3.txt。
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
for s in 1 2 3; do run adult 3 k105auto $s; done
for s in 1 2 3; do run adult 3 k105f05 $s; done
echo "$(date -u +%FT%TZ) 本机 C 组 ε=3 结束"
