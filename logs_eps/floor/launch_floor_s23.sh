#!/bin/bash
# 第六十三步：第六十二步同法复跑种子 2、3（adult 1A，五档 ε，两臂，沿噪声地板倍数取快照）。
#   命令与第六十二步逐字相同，只换 SEED（数据目录 adult_alloc*base_s$SEED、菜单/抽样/初始化种子、GSD --seed）；
#   噪声地板 ½Σσ² 三种子逐位相同，阈值网格原样复用（够不到的档不触发）。
#   ×1.0 快照即两臂各自加地板停会交出的表。目的：确认第六十二步两臂同停法对照的比值，不在此定最终配置。
# 车道（各自空闲门，每卡同一时刻只跑一个任务）：
#   GPU0 道 0-23：我们 s2 全档 → GSD s2 ε=10/3 → GSD s3 ε=1/0.3/0.1
#   GPU1 道 24-47：我们 s3 全档 → GSD s3 ε=10/3 → GSD s2 ε=1/0.3/0.1
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/floor/launch_floor_s23.sh < /dev/null > logs_eps/floor/launch_floor_s23.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
declare -A SNAP_OURS=(
  [0p1]="1028467133.7737919,942761539.2926427,899908742.052068,857055944.8114933,814203147.5709186,771350350.330344"
  [0p3]="337058456.7720543,224705637.84803623,168529228.38602716,146058664.60122356,134823382.70882174,123588100.81641993,117970459.87021902,112352818.92401811,106735177.97781721,101117537.0316163,89882255.1392145"
  [1]="60723363.88420613,36434018.330523685,24289345.553682454,18217009.165261842,15788074.609893596,14573607.332209472,13359140.054525351,12751906.415683288,12144672.776841227,11537439.137999166,10930205.499157105,9715738.221472982,8501270.943788858"
  [3]="8274274.833476005,4964564.900085604,3309709.9333904022,2482282.450042802,2151311.4567037616,1985825.9600342412,1820340.4633647213,1737597.7150299612,1654854.9666952011,1572112.218360441,1489369.470025681,1323883.973356161,1158398.4766866406"
  [10]="1040839.071041632,624503.4426249792,416335.6284166528,312251.7213124896,270618.1584708243,249801.37704999166,228984.59562915904,218576.20491874273,208167.8142083264,197759.42349791006,187351.03278749375,166534.25136666113,145717.46994582846"
)
declare -A SNAP_GSD=(
  [0p1]="8.083765432882725,4.850259259729634,3.2335061731530894,2.425129629864817,2.101779012549508,1.9401037038918536,1.7784283952341995,1.697590740905372,1.6167530865765447,1.5359154322477173,1.4550777779188904"
  [0p3]="1.0597135920977396,0.6358281552586437,0.42388543683909585,0.31791407762932183,0.2755255339454123,0.2543312621034575,0.2331369902615027,0.2225398543405253,0.21194271841954793,0.2013455824985705,0.19074844657759313,0.16955417473563836,0.14835990289368353"
  [1]="0.11454874863354941,0.06872924918012967,0.045819499453419774,0.034364624590064834,0.029782674644722854,0.02749169967205186,0.025200724699380877,0.02405523721304538,0.022909749726709887,0.02176426224037439,0.020618774754038897,0.01832779978136791,0.016036824808696917"
  [3]="0.015608618617244743,0.009365171170346848,0.006243447446897898,0.004682585585173424,0.0040582408404836335,0.0037460684681387382,0.003433896095793844,0.0032778099096213965,0.003121723723448949,0.002965637537276501,0.002809551351104054,0.0024973789787591594,0.0021852066064142637"
  [10]="0.001963442166084204,0.0011780652996505221,0.0007853768664336815,0.0005890326498252611,0.000510494963181893,0.0004712261198602088,0.0004319572765385248,0.0004123228548776828,0.00039268843321684074,0.00037305401155599867,0.00035341958989515665,0.0003141507465734726,0.00027488190325178845"
)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=results/floor
LOG=logs_eps/floor
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }

run_ours() {  # tag gpu seed
  local tag=$1 gpu=$2 seed=$3 name=adult_eps${1}_ours2way_seed$3
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_alloc${tag}base_s${seed} --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-avg --init-clean --exam-weights none --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    --snapshot-loss "${SNAP_OURS[$tag]}" --snapshot-prefix $OUT/$name \
    $EXTRA --gpu-sparse auto > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

run_gsd() {  # tag gpu seed
  local tag=$1 gpu=$2 seed=$3 name=adult_eps${1}_gsd_seed$3
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/root/gsd_patched PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  taskset -c ${LANE[$gpu]} /root/miniconda3/envs/aim/bin/python -u -B scripts/run_baseline_gsd_plants.py \
    --seed $seed --data adult --noisy-exam data/adult_alloc${tag}base_s${seed}/measured_2way.json \
    --generation-log $OUT/${name}_gens.csv --log-every 1000 \
    --snapshot-fitness "${SNAP_GSD[$tag]}" --snapshot-prefix $OUT/$name \
    --out $OUT/$name.csv > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

( for t in 3 10 1 0p3 0p1; do run_ours $t 0 2; done
  for t in 10 3; do run_gsd $t 0 2; done
  for t in 1 0p3 0p1; do run_gsd $t 0 3; done
  echo "$(date -u +%FT%TZ) GPU0 LANE DONE" ) &
( for t in 3 10 1 0p3 0p1; do run_ours $t 1 3; done
  for t in 10 3; do run_gsd $t 1 3; done
  for t in 1 0p3 0p1; do run_gsd $t 1 2; done
  echo "$(date -u +%FT%TZ) GPU1 LANE DONE" ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE FLOOR S23"
