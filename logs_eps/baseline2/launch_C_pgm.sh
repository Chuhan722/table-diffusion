#!/bin/bash
# C 组 ε=3：PGM 同餐 nltcs ×3（A6000 纯 CPU 核 48-55，等 launch_C_aim.sh 的官方 AIM 跑完后接上；命令与 A 组 run_pgm 逐字相同）。
cd /root/residual-evolution || exit 1
PGMLANE=48-55; OUT=results/baseline2; LOG=logs_eps/baseline2
until grep -q "ALL DONE C-AIM" $LOG/launch_C_aim.log 2>/dev/null; do sleep 120; done
for seed in 1 2 3; do
  name=nltcs_eps3_pgm_seed$seed; exam=data/nltcs_alloc3base_s${seed}/measured_2way.json
  [ -s $OUT/$name.csv ] && { echo "skip $name"; continue; }
  sigma=$(.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['sigma2'])" $exam)
  echo "$(date -u +%FT%TZ) START $name cpu lane=$PGMLANE sigma=$sigma"
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=/root/private-pgm/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  taskset -c $PGMLANE /root/miniconda3/envs/aim2/bin/python -u -B scripts/run_baseline_pgm.py \
    --exam $exam --sigma $sigma --seed $seed --iters 1000 --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done
echo "$(date -u +%FT%TZ) ALL DONE C-PGM"
