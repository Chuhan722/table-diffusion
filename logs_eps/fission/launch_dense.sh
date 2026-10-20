#!/bin/bash
# 第七十三步 探针提前停·密集记账 全网格发射器：30 格（1A/1B × ε=10/3/1/0.3/0.1 × 种子 1/2/3），每格 = probe_dense.sh（只跑探针）。
# 规矩：GPU 只用 0/1；开格前查空闲（<5120 MB 连续两次、间隔 60 s；我们自己已在该卡上跑时不再查）；
#   每卡两槽（同卡两探针并行，各 14 核起步，cpu_watch 看守放宽到 56 核掩码）；四槽共用队列（mkdir 锁抢格），已完成的格跳过。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/fission/launch_dense.sh < /dev/null > logs_eps/fission_dense/launch_dense.log 2>&1 &)
#   参数为空格分隔的 "考卷:ε标签:种子" 列表，默认全部 30 格（长的先跑）。
#   OUT/LOG/REF/STOPZ/STOPRUNS/EVERY 环境变量原样传给 probe_dense.sh（第二趟：OUT=results/fission_dense2 LOG=logs_eps/fission_dense2 REF=results/fission_dense）。
cd /root/residual-evolution || exit 1
ALL=""
for t in 10 3 1 0p3 0p1; do for s in 1 2 3; do ALL="$ALL 1A:$t:$s 1B:$t:$s"; done; done
JOBS=${1:-"$ALL"}
export OUT=${OUT:-results/fission_dense} LOG=${LOG:-logs_eps/fission_dense} REF STOPZ STOPRUNS EVERY
LOCK=$LOG/locks
mkdir -p $OUT $LOG $LOCK
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
ours_on_gpu() {
  # 我们自己的 run_plants 是否已在该卡上（看进程环境里的 CUDA_VISIBLE_DEVICES）；在则不再查空闲（卡是我们先占的）
  local g=$1 pid
  for pid in $(pgrep -f scripts/run_plants.py); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=${UUID[$g]}$" && return 0
  done
  return 1
}
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() {
  local g=$1 w=0
  while :; do
    ours_on_gpu $g && return
    if ! gpu_busy $g; then sleep 60; ours_on_gpu $g && return; if ! gpu_busy $g; then return; fi; fi
    w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120
  done
}
lane() {
  local gpu=$1 slot=$2 job exam tag seed cell
  for job in $JOBS; do
    exam=${job%%:*}; seed=${job##*:}; tag=${job#*:}; tag=${tag%:*}; cell=${exam}_eps${tag}_s${seed}
    [ -s $OUT/${cell}_times.json ] && continue
    mkdir $LOCK/$cell 2>/dev/null || continue
    wait_idle $gpu
    echo "$(date -u +%FT%TZ) START $cell gpu$gpu slot$slot load=$(cut -d' ' -f1-3 /proc/loadavg)"
    bash logs_eps/fission/probe_dense.sh $exam $tag $seed $gpu $slot >> $LOG/cell_$cell.log 2>&1
    echo "$(date -u +%FT%TZ) END $cell rc=$? ok=$(grep -o '"ok": [a-z]*' $OUT/${cell}_check.json 2>/dev/null | cut -d' ' -f2) wall=$(grep -o '"wall": [0-9.]*' $OUT/${cell}_times.json 2>/dev/null | cut -d' ' -f2) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  done
  echo "$(date -u +%FT%TZ) GPU$gpu SLOT$slot DONE"
}
lane 0 0 &
sleep 5
lane 1 0 &
sleep 5
lane 0 1 &
sleep 5
lane 1 1 &
wait
rm -rf $LOCK
echo "$(date -u +%FT%TZ) ALL DONE DENSE"
