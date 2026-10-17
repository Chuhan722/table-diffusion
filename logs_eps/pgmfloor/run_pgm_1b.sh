#!/bin/bash
# 第六十六步：1B 比生成器 PGM 臂（本机 CPU，private-pgm 环境，与 pgmrep10k 十五张正式表同一套环境）。
#   同录像、同种子、known-total rows，与 logs_eps/stage1/run_pgmrep10k.sh 逐字同，只多：
#   --stop-floor-mult 1.0（mbi 损失首次 ≤ ½·录像格数即停，逐步查 --check-every 1）
#   兜底 --tol 1e-4 --patience 2（每 50 步相对改善 ≤1e-4 连续两块）与 --iters 10000 上限；停法写 *_stop.json，曲线写 *_curve.csv。
# 用法：cd residual-evolution && bash logs_eps/pgmfloor/run_pgm_1b.sh "1:1" > logs_eps/pgmfloor/run_pgm_1b.log 2>&1
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=/home/chuhan/projects/private-pgm/src JAX_PLATFORMS=cpu
PY=/home/chuhan/projects/private-pgm/.venv/bin/python
JOBS=${1:-"1:1"}
OUT=results/pgmfloor
LOG=logs_eps/pgmfloor
mkdir -p $OUT $LOG
for job in $JOBS; do
  e=${job%%:*}; s=${job##*:}; name=adult_eps${e}_pgmfloor_seed$s
  [ -s $OUT/$name.csv ] && { echo "skip $name"; continue; }
  echo "$(date -u +%FT%TZ) START $name"
  $PY -u -B scripts/run_pgm_replay.py \
    --dump results/aimreplay/adult_eps${e}_aimfull_seed${s}_measurements.json --seed $s --iters 10000 \
    --known-total rows --stop-floor-mult 1.0 --check-every 1 --tol 1e-4 --patience 2 \
    --curve $OUT/${name}_curve.csv --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done
