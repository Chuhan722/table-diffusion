#!/bin/bash
# D 组 ε=0.3 纯 CPU 道（A6000 核 48-55）：官方 AIM adult ×3 → nltcs ×3 → PGM 同餐 nltcs ×3。命令与 C 组 launch_C_aim.sh / launch_C_pgm.sh 逐字相同，只换 ε。
cd /root/residual-evolution || exit 1
LANE=48-55; OUT=results/baseline2; LOG=logs_eps/baseline2
mkdir -p results/aimreplay $OUT $LOG
for ds in adult nltcs; do
  for s in 1 2 3; do
    name=${ds}_eps0p3_aimfull_seed${s}
    [ -s results/${name}.csv ] && { echo "skip $name"; continue; }
    echo "$(date -u +%FT%TZ) START $name lane=$LANE"
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src \
      taskset -c $LANE /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_aim_full.py \
        --data $ds --mechanisms /root/private-pgm/mechanisms --epsilon 0.3 --delta 1e-5 --seed $s \
        --max-model-size 80 --clear-caches-every 1 --clear-caches-every-projects 500 \
        --out results/${name}.csv --dump-measurements results/aimreplay/${name}_measurements.json \
        > $LOG/${name}.log 2>&1
    echo "$(date -u +%FT%TZ) END $name rc=$?"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE D-AIM"
for seed in 1 2 3; do
  name=nltcs_eps0p3_pgm_seed$seed; exam=data/nltcs_alloc0p3base_s${seed}/measured_2way.json
  [ -s $OUT/$name.csv ] && { echo "skip $name"; continue; }
  sigma=$(.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['sigma2'])" $exam)
  echo "$(date -u +%FT%TZ) START $name cpu lane=$LANE sigma=$sigma"
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  taskset -c $LANE /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_baseline_pgm.py \
    --exam $exam --sigma $sigma --seed $seed --iters 1000 --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done
echo "$(date -u +%FT%TZ) ALL DONE D-CPU"
