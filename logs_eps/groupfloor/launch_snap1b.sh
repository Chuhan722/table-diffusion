#!/bin/bash
# 第六十八步 对照场：1B 第一段跑到收敛，沿地板倍数与分档过线点取快照（adult 1B，三种子，五档 ε；本机 4090 GPU0）。
#   命令 = 第六十七步整体方法我们臂（logs_eps/whole1b/launch_ours_whole1b.sh）的第一段逐字（含一阶清洗 --oneway-pool 2 uncovered），
#   去掉 --stop-floor（改由平台早停 1e-4/100 轮、上限 30000 收工）与纤维两段（--fiber-passes/--fiber-lock-real 只管第一段之后）；
#   加 --snapshot-loss 地板 × 1.0/0.9/0.8/0.7/0.6/0.5/0.4（×1.0 的阈值字符串与 FLOOR 逐字同，该快照 _snap0.csv 应与
#   results/whole1b/*_stage1.csv 逐字节同——核对场已证 4090 与 A6000 在 ε=1 种子 1 上逐字节一致）
#   与 --snapshot-group-floor（首个各档都不高于各自地板的轮，_snapG.csv），终表即收敛表。快照纯记账，轨迹不变。
#   分档曲线写 *_curve_groups.csv/.json。
# 规矩：只用本机 GPU0（<5120 MB 连续两次才开），核 0-15（一半 CPU），15 格串行。
# 用法：cd /home/chuhan/projects0916/residual-evolution && (nohup bash logs_eps/groupfloor/launch_snap1b.sh < /dev/null > logs_eps/groupfloor/snap1b_launch.log 2>&1 &)
#   参数为空格分隔的 "ε标签:种子" 列表，默认全部 15 格。
cd /home/chuhan/projects0916/residual-evolution || exit 1
declare -A FLOOR=(
  [0p1:1]=6287575.166310625 [0p1:2]=1823340.3762813245 [0p1:3]=5123813.774480508
  [0p3:1]=1399681.5247673127 [0p3:2]=1800172.6178743835 [0p3:3]=1630659.5345760211
  [1:1]=299038.1812299605 [1:2]=225753.08438055194 [1:3]=190205.15277737722
  [3:1]=68465.50849395874 [3:2]=48037.311033374186 [3:3]=51669.8821903612
  [10:1]=9924.017934371434 [10:2]=14727.64151114602 [10:3]=14291.893133479229
)
MULT="0.9 0.8 0.7 0.6 0.5 0.4"
JOBS=${1:-"1:1 1:2 1:3 0p1:1 0p1:2 0p1:3 0p3:1 0p3:2 0p3:3 3:1 3:2 3:3 10:1 10:2 10:3"}
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --oneway-pool 2 --oneway-pool-scope uncovered"
PY=.venv/bin/python
OUT=results/groupfloor/snap1b
LOG=logs_eps/groupfloor
mkdir -p $OUT $LOG

gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
wait_gpu0() {
  local w=0
  while :; do
    if ! gpu_busy; then sleep 60; if ! gpu_busy; then return; fi; fi
    [ $((w % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu0 被占用，等待中（已等 $((w / 60)) min）"
    sleep 120; w=$((w + 120))
  done
}
snap_list() {  # 地板字符串逐字作 ×1.0，其余倍数由 python repr 给出
  $PY -c 'import sys; f=sys.argv[1]; print(",".join([f] + [repr(float(f) * float(m)) for m in sys.argv[2].split()]))' "$1" "$MULT"
}

for job in $JOBS; do
  tag=${job%%:*}; seed=${job##*:}; name=adult_eps${tag}_snap_seed${seed}
  [ -z "${FLOOR[$job]}" ] && { echo "no floor for $job"; continue; }
  [ -s $OUT/$name.csv ] && { echo "skip $name"; continue; }
  wait_gpu0
  echo "$(date -u +%FT%TZ) START $name gpu0 floor=${FLOOR[$job]} load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 $PY -u -B scripts/run_plants.py \
    --data adult_aimrep${tag}_s${seed} --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    --snapshot-loss "$(snap_list ${FLOOR[$job]})" --snapshot-group-floor --snapshot-prefix $OUT/$name \
    $EXTRA > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
done
echo "$(date -u +%FT%TZ) 队列完"
