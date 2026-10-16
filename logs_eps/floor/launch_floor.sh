#!/bin/bash
# 第六十二步：沿噪声地板倍数取快照、定停点（adult 1A，种子 1，五档 ε）。
#   第六十步 ours 命令逐字 + --snapshot-loss/--snapshot-prefix（纯记账，轨迹不变），
#   阈值 = 噪声地板 ½Σσ²（引擎 old_loss 口径）× 倍数网格 5/3/2/1.5/1.3/1.2/1.1/1.05/1/0.95/0.9/0.8/0.7，
#   每档 ε 只留起点损失之下、终损失之上够得到的倍数（0.1：1.2～0.9 六档；0.3：3～0.8 十一档；1/3/10：十三档全留）。
#   快照 i 对应该 ε 列表第 i 个倍数（见 *_snaps.csv 的 threshold 列 ÷ 地板）；
#   ×1.0 快照即 --stop-floor 硬地板停会交出的表（引擎同一处、同一张 current）。
#   终表须与 results/effort 逐位一致后才算数。
# GPU0 道 0-23 跑 ε=3/0.1/0.3，GPU1 道 24-47 跑 ε=10/1，各自空闲门。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/floor/launch_floor.sh < /dev/null > logs_eps/floor/launch_floor.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
declare -A SNAP=(
  [0p1]="1028467133.7737919,942761539.2926427,899908742.052068,857055944.8114933,814203147.5709186,771350350.330344"
  [0p3]="337058456.7720543,224705637.84803623,168529228.38602716,146058664.60122356,134823382.70882174,123588100.81641993,117970459.87021902,112352818.92401811,106735177.97781721,101117537.0316163,89882255.1392145"
  [1]="60723363.88420613,36434018.330523685,24289345.553682454,18217009.165261842,15788074.609893596,14573607.332209472,13359140.054525351,12751906.415683288,12144672.776841227,11537439.137999166,10930205.499157105,9715738.221472982,8501270.943788858"
  [3]="8274274.833476005,4964564.900085604,3309709.9333904022,2482282.450042802,2151311.4567037616,1985825.9600342412,1820340.4633647213,1737597.7150299612,1654854.9666952011,1572112.218360441,1489369.470025681,1323883.973356161,1158398.4766866406"
  [10]="1040839.071041632,624503.4426249792,416335.6284166528,312251.7213124896,270618.1584708243,249801.37704999166,228984.59562915904,218576.20491874273,208167.8142083264,197759.42349791006,187351.03278749375,166534.25136666113,145717.46994582846"
)
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=results/floor
LOG=logs_eps/floor
SEED=1
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }

run_ours() {  # tag gpu
  local tag=$1 gpu=$2 name=adult_eps${1}_ours2way_seed$SEED
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} .venv/bin/python -u -B scripts/run_plants.py \
    --data adult_alloc${tag}base_s${SEED} --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-avg --init-clean --exam-weights none --menu-seed $((20260920 + SEED)) --sample-seed $SEED --init-seed $SEED \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    --snapshot-loss "${SNAP[$tag]}" --snapshot-prefix $OUT/$name \
    $EXTRA --gpu-sparse auto > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

( for t in 3 0p1 0p3; do run_ours $t 0; done; echo "$(date -u +%FT%TZ) GPU0 LANE DONE" ) &
( for t in 10 1; do run_ours $t 1; done; echo "$(date -u +%FT%TZ) GPU1 LANE DONE" ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE FLOOR"
