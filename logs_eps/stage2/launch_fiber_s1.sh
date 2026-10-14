#!/bin/bash
# 补 ε=10 s1（launch_fiber_batch.sh 的道单沿用上批、上批 s1 是单独试点）：等 gpu0 道单跑完再接在 gpu0 上跑。
cd /root/residual-evolution || exit 1
until grep -q "gpu0 队列完" logs_eps/stage2/launch_fiber_batch_gpu0.log 2>/dev/null; do sleep 120; done
UUID=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '); [ "$u" -gt 5000 ]; }
while gpu_busy || { sleep 60; gpu_busy; }; do sleep 120; done
seed=1; name=adult_eps10_hybridfiber_seed1
echo "$(date -u +%FT%TZ) START $name gpu0 lane=0-23"
CUDA_VISIBLE_DEVICES=$UUID taskset -c 0-23 .venv/bin/python -u -B scripts/run_plants.py \
  --data adult_aimrep10_s1_fiber --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
  --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
  --save-table results/stage2/$name.csv --out results/stage2/${name}_curve.csv \
  $EXTRA > logs_eps/stage2/$name.log 2>&1
echo "$(date -u +%FT%TZ) END $name rc=$?"
