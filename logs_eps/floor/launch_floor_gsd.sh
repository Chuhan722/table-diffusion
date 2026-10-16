#!/bin/bash
# 第六十二步（GSD 臂）：沿噪声地板倍数取快照（adult，种子 1，五档 ε）。
#   第六十步 run_gsd 命令逐字 + --snapshot-fitness/--snapshot-prefix（gsd_patched 只加快照钩子，
#   在每 1000 代的记账点上复用已读的 best_fitness，首次 ≤ 阈值即把 best_member 落表，每阈值一次）。
#   阈值 = 倍数 × Σσ²/N²（fitness 口径；ℒ = N²·fitness，地板 ℒ = Σσ²），倍数网格同我们臂：
#   5/3/2/1.5/1.3/1.2/1.1/1.05/1/0.95/0.9/0.8/0.7，ε=0.1 去掉够不到的 0.8/0.7（第六十步 GSD 终值 0.825×）。
#   ×1.0 快照即"GSD + 地板停（每 1000 代检查）"会交出的表。GSD 的 GPU 搜索本身非逐位确定，
#   终表与第六十步不同，终 ℒ 之比应在 ±0.5% 内。
# 先等我们臂（launch_floor.sh）全收，再 GPU0 道 0-23 跑 ε=10/0.1，GPU1 道 24-47 跑 ε=3/1/0.3，各自空闲门。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/floor/launch_floor_gsd.sh < /dev/null > logs_eps/floor/launch_floor_gsd.log 2>&1 &)
cd /root/residual-evolution || exit 1
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
declare -A SNAP=(
  [0p1]="8.083765432882725,4.850259259729634,3.2335061731530894,2.425129629864817,2.101779012549508,1.9401037038918536,1.7784283952341995,1.697590740905372,1.6167530865765447,1.5359154322477173,1.4550777779188904"
  [0p3]="1.0597135920977396,0.6358281552586437,0.42388543683909585,0.31791407762932183,0.2755255339454123,0.2543312621034575,0.2331369902615027,0.2225398543405253,0.21194271841954793,0.2013455824985705,0.19074844657759313,0.16955417473563836,0.14835990289368353"
  [1]="0.11454874863354941,0.06872924918012967,0.045819499453419774,0.034364624590064834,0.029782674644722854,0.02749169967205186,0.025200724699380877,0.02405523721304538,0.022909749726709887,0.02176426224037439,0.020618774754038897,0.01832779978136791,0.016036824808696917"
  [3]="0.015608618617244743,0.009365171170346848,0.006243447446897898,0.004682585585173424,0.0040582408404836335,0.0037460684681387382,0.003433896095793844,0.0032778099096213965,0.003121723723448949,0.002965637537276501,0.002809551351104054,0.0024973789787591594,0.0021852066064142637"
  [10]="0.001963442166084204,0.0011780652996505221,0.0007853768664336815,0.0005890326498252611,0.000510494963181893,0.0004712261198602088,0.0004319572765385248,0.0004123228548776828,0.00039268843321684074,0.00037305401155599867,0.00035341958989515665,0.0003141507465734726,0.00027488190325178845"
)
OUT=results/floor
LOG=logs_eps/floor
SEED=1
mkdir -p $OUT $LOG

until grep -q "ALL DONE FLOOR" $LOG/launch_floor.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) ours lane finished, starting GSD"

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }

run_gsd() {  # tag gpu
  local tag=$1 gpu=$2 name=adult_eps${1}_gsd_seed$SEED
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/root/gsd_patched PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  taskset -c ${LANE[$gpu]} /root/miniconda3/envs/aim/bin/python -u -B scripts/run_baseline_gsd_plants.py \
    --seed $SEED --data adult --noisy-exam data/adult_alloc${tag}base_s${SEED}/measured_2way.json \
    --generation-log $OUT/${name}_gens.csv --log-every 1000 \
    --snapshot-fitness "${SNAP[$tag]}" --snapshot-prefix $OUT/$name \
    --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

( for t in 10 0p1; do run_gsd $t 0; done; echo "$(date -u +%FT%TZ) GPU0 GSD LANE DONE" ) &
( for t in 3 1 0p3; do run_gsd $t 1; done; echo "$(date -u +%FT%TZ) GPU1 GSD LANE DONE" ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE FLOOR GSD"
