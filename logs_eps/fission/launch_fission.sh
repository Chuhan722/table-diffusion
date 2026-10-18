#!/bin/bash
# 第七十一步 噪声分裂探针试点（adult，种子 1，1A/1B × ε=0.1/1，共 4 格；A6000 GPU0/1）。
#   引擎只看 学习卷 y+e（data/<源>_fisA，由 make_fission.py 生成，σ' = √2σ，地板_A = 2×原地板）；
#   命令 = 各考卷最终方法第一段逐字（1B 含一阶清洗，1A 为 ours2way 命令），去掉硬地板停、跑到平台收敛，
#   加 --snapshot-loss 地板_A × 2.0/1.5/1.3/1.2/1.1/1.0/0.95/0.9/0.8/0.7/0.6/0.5/0.4（高于起点损失的阈值在第 0 轮落表）
#   与 --snapshot-group-floor（首个各档都不高于各自地板_A 的轮 = 地板停会交出的表，_snapG.csv）；终表 = 收敛停会交出的表。
#   阅卷见 score_fission.py：每张快照对 模拟卷 y−e 的损失 L_B 与对真表的损失逐一对照，判"有谷/没谷"是否与真误差一致。
# 规矩：GPU 空闲（<5120 MB 连续两次）才开；GPU0 道核 0-23，GPU1 道核 24-47。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/fission/launch_fission.sh < /dev/null > logs_eps/fission/launch_fission.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
# 地板_A 字符串逐字取自 make_fission.py 的打印（results/fission/floors.json 的 floor_split_str）
declare -A FLOOR=(
  [1B:0p1]=12575150.33262125 [1B:1]=598076.3624599215
  [1A:0p1]=1714111889.622664 [1A:1]=24289345.55369027
)
MULT="1.5 1.3 1.2 1.1 1.0 0.95 0.9 0.8 0.7 0.6 0.5 0.4"   # 快照 0 为 ×2.0（snap_list 置于最前），快照 1.. 依次为此表；×1.0 用地板字符串逐字
SEED=1
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
EXTRA_1B="$EXTRA --oneway-pool 2 --oneway-pool-scope uncovered"
EXTRA_1A="$EXTRA --gpu-sparse auto"
PY=.venv/bin/python
OUT=results/fission
LOG=logs_eps/fission
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }
snap_list() {  # ×2.0 在前，地板字符串逐字作 ×1.0，其余倍数由 python repr 给出
  $PY -c 'import sys; f=sys.argv[1]; ms=sys.argv[2].split(); print(",".join([repr(float(f)*2.0)] + [f if m == "1.0" else repr(float(f) * float(m)) for m in ms]))' "$1" "$MULT"
}

run_job() {  # exam tag gpu
  local exam=$1 tag=$2 gpu=$3 name=fis_${1}_eps${2}_s${SEED} floor=${FLOOR[$1:$2]}
  [ -z "$floor" ] && { echo "no floor for $exam:$tag"; return; }
  [ -s $OUT/$name.csv ] && { echo "skip $name"; return; }
  local data common extra
  if [ "$exam" = "1B" ]; then
    data=adult_aimrep${tag}_s${SEED}_fisA
    common="--data $data --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 --init-oneway --exam-weights sigma"
    extra=$EXTRA_1B
  else
    data=adult_alloc${tag}base_s${SEED}_fisA
    common="--data $data --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 --init-avg --init-clean --exam-weights none"
    extra=$EXTRA_1A
  fi
  wait_idle $gpu
  echo "$(date -u +%FT%TZ) START $name gpu$gpu floor_A=$floor load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c ${LANE[$gpu]} $PY -u -B scripts/run_plants.py \
    $common --menu-seed $((20260920 + SEED)) --sample-seed $SEED --init-seed $SEED \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    --snapshot-loss "$(snap_list $floor)" --snapshot-group-floor --snapshot-prefix $OUT/$name \
    $extra > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

( run_job 1B 0p1 0; run_job 1A 1 0; echo "$(date -u +%FT%TZ) GPU0 LANE DONE" ) &
( run_job 1B 1 1; run_job 1A 0p1 1; echo "$(date -u +%FT%TZ) GPU1 LANE DONE" ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE FISSION"
