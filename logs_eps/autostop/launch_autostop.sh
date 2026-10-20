#!/bin/bash
# 第七十二步 探针并行自动停 全网格发射器：30 格（1A/1B × ε=10/3/1/0.3/0.1 × 种子 1/2/3），每格 = autostop.sh（P 与 F 同卡并行）。
# 规矩：GPU 空闲（<5120 MB 连续两次、间隔 60 s）才开；GPU0 道核 0-23、GPU1 道核 24-47，一道同一时刻只跑一格。两道共用队列（mkdir 锁抢格），已交表的格跳过。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/autostop/launch_autostop.sh < /dev/null > logs_eps/autostop/launch_autostop.log 2>&1 &)
#   参数为空格分隔的 "考卷:ε标签:种子" 列表，默认全部 30 格。
cd /root/residual-evolution || exit 1
ALL=""
for t in 10 3 1 0p3 0p1; do for s in 1 2 3; do ALL="$ALL 1A:$t:$s 1B:$t:$s"; done; done
JOBS=${1:-"$ALL"}
OUT=results/autostop${VARIANT:+_$VARIANT}
LOG=logs_eps/autostop${VARIANT:+_$VARIANT}
LOCK=$LOG/locks
mkdir -p $OUT $LOG $LOCK
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_idle() { local g=$1 w=0; while :; do if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then return; fi; fi; w=$((w+120)); echo "$(date -u +%FT%TZ) gpu$g busy, waited ${w}s"; sleep 120; done; }
lane() {
  local gpu=$1 job exam tag seed cell
  for job in $JOBS; do
    exam=${job%%:*}; seed=${job##*:}; tag=${job#*:}; tag=${tag%:*}; cell=${exam}_eps${tag}_s${seed}
    [ -s $OUT/${cell}_out.csv ] && continue
    mkdir $LOCK/$cell 2>/dev/null || continue
    wait_idle $gpu
    echo "$(date -u +%FT%TZ) START $cell gpu$gpu load=$(cut -d' ' -f1-3 /proc/loadavg)"
    bash logs_eps/autostop/autostop.sh $exam $tag $seed $gpu >> $LOG/cell_$cell.log 2>&1
    echo "$(date -u +%FT%TZ) END $cell rc=$? $(grep -o '"decision": "[a-z]*"' $OUT/${cell}_times.json 2>/dev/null) wall=$(grep -o '"wall": [0-9.]*' $OUT/${cell}_times.json 2>/dev/null | cut -d' ' -f2) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  done
  echo "$(date -u +%FT%TZ) GPU$gpu LANE DONE"
}
lane 0 &
sleep 5
lane 1 &
wait
rm -rf $LOCK
echo "$(date -u +%FT%TZ) ALL DONE AUTOSTOP"
