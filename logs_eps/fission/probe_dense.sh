#!/bin/bash
# 第七十三步 探针提前停·密集记账（一格）：第七十一步探针命令逐字（学习卷，第一段跑到收敛，落分档过线快照 G），
#   加 --fission-exam/--fission-every/--fission-out 纯记账：每 EVERY 轮与分档过线轮用起点表对模拟卷算 L1_B，收工再记终表一行。
#   跑完 check_dense.py 核对：终表与 G 表 md5 = 第七十一步 results/fission/fis_<格>{,_snapG}.csv；
#   G 行与终表行的 L1_B = 第七十二步 decide.py 的 L1_B_G / L1_B_final（results/autostop/<格>_decision.json）。
# 用法：bash logs_eps/fission/probe_dense.sh 考卷(1A|1B) ε标签 种子 gpu 槽(0|1)
#   核：有 logs_eps/autostop/cpu_mask（56 核）时该道取其中 28 核、槽 0/1 各 14 核起步（看守随后放宽）；没有则固定 HALF_A/HALF_B。
#   EVERY（默认 5）记账间隔；OUT/LOG 环境变量改输出目录（默认 results/fission_dense、logs_eps/fission_dense）；
#   REF 给参照密集目录时核对两趟记账列逐字相同；STOPZ 非空时加 --fission-stop-z STOPZ --fission-stop-runs STOPRUNS（默认 3）开显著回升停。
cd /root/residual-evolution || exit 1
PY=.venv/bin/python
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A HALF_A=([0]=0-11 [1]=24-35)
declare -A HALF_B=([0]=12-23 [1]=36-47)
exam=$1; tag=$2; seed=$3; gpu=$4; slot=${5:-0}
cell=${exam}_eps${tag}_s${seed}
EVERY=${EVERY:-5}
if [ "$slot" = "0" ]; then CP=${HALF_A[$gpu]}; else CP=${HALF_B[$gpu]}; fi
CPU_MODE=fixed
MASKF=logs_eps/autostop/cpu_mask
if [ -s $MASKF ]; then
  IFS=, read -r -a MK < $MASKF
  if [ ${#MK[@]} -ge 56 ]; then
    o=$((gpu * 28 + slot * 14)); join() { local IFS=,; echo "$*"; }
    CP=$(join "${MK[@]:$o:14}"); CPU_MODE=watch56
  fi
fi
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=${OUT:-results/fission_dense}
LOG=${LOG:-logs_eps/fission_dense}
STOPARGS=""
[ -n "$STOPZ" ] && STOPARGS="--fission-stop-z $STOPZ --fission-stop-runs ${STOPRUNS:-3}"
mkdir -p $OUT $LOG
SEEDS="--menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed"
STOP="--rounds 30000 --stop-threshold 1e-4 --stop-lag 100"
if [ "$exam" = "1B" ]; then
  src=adult_aimrep${tag}_s${seed}
  commonP="--data ${src}_fisA $STOP --init-oneway --exam-weights sigma"
  extraP="$EXTRA --oneway-pool 2 --oneway-pool-scope uncovered"
else
  src=adult_alloc${tag}base_s${seed}
  commonP="--data ${src}_fisA --exam measured_2way.json $STOP --init-avg --init-clean --exam-weights none"
  extraP="$EXTRA --gpu-sparse auto"
fi
now() { date +%s.%N; }
dt() { awk "BEGIN{printf \"%.1f\", $1 - $2}"; }
stamp() { date -u +%FT%TZ; }
rm -f $OUT/${cell}_snapG.csv $OUT/$cell.csv $OUT/${cell}_probe.csv
t0=$(now)
echo "$(stamp) [$cell] 开 探针(核 $CP, 每 $EVERY 轮记账) 于 gpu$gpu 槽 $slot（CPU 分法 $CPU_MODE）load=$(cut -d' ' -f1-3 /proc/loadavg)"
CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c $CP $PY -u -B scripts/run_plants.py $commonP $SEEDS \
  --save-table $OUT/$cell.csv --out $OUT/${cell}_curve.csv \
  --snapshot-group-floor --snapshot-prefix $OUT/$cell \
  --fission-exam results/fission/${src}_fisB.json --fission-every $EVERY --fission-out $OUT/${cell}_probe.csv $STOPARGS \
  $extraP > $LOG/$cell.log 2>&1
rc=$?; tE=$(now)
echo "$(stamp) [$cell] 探针完 rc=$rc 墙钟 $(dt $tE $t0) 秒 load=$(cut -d' ' -f1-3 /proc/loadavg)"
$PY logs_eps/fission/check_dense.py $cell $OUT results/fission results/autostop "$REF" results/fission/${src}_fisB.json > $OUT/${cell}_check.json 2> $LOG/${cell}_check.err
echo "$(stamp) [$cell] 核对 $(tr -d '\n' < $OUT/${cell}_check.json)"
cat > $OUT/${cell}_times.json <<EOF
{"cell": "$cell", "exam": "$exam", "tag": "$tag", "seed": $seed, "gpu": $gpu, "slot": $slot, "cores": "$CP", "cpu_mode": "$CPU_MODE",
 "every": $EVERY, "stopz": "${STOPZ:-}", "stopruns": "${STOPRUNS:-}", "t0": $t0, "tE": $tE, "rc": $rc, "wall": $(dt $tE $t0), "load_end": "$(cut -d' ' -f1-3 /proc/loadavg)"}
EOF
