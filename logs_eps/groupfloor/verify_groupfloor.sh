#!/bin/bash
# 第六十八步 核对场（只核对实现，不出结论）：
#   A. 1B ε=1 种子 1 整体方法命令（第六十七步逐字）+ --snapshot-group-floor：默认路径逐字节不变
#      （第一段表 md5 20f9b4ed…、终表 98785ff3… 与 results/whole1b 同），分档曲线各档之和 = 曲线损失。
#   B. 1A ε=1 种子 1 ours2way 命令（第六十二步逐字，去快照）+ --stop-floor-groups：单档等价硬地板停，
#      第一段表应与 results/floor/adult_eps1_ours2way_seed1_snap8.csv（×1.0 快照，md5 0eab060e…）逐字节同。
#   C. 1B ε=1 种子 1 整体方法命令，--stop-floor 换成 --stop-floor-groups：分档停冒烟场（看停点、各档比、耗时）。
# 只用空闲的 GPU0/1（<5120 MB 连续两次），三场串行一道。
# 用法：cd /root/residual-evolution && (nohup bash logs_eps/groupfloor/verify_groupfloor.sh < /dev/null > logs_eps/groupfloor/verify_groupfloor.log 2>&1 &)
cd /root/residual-evolution || exit 1
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
GPUS=${GPUS:-"0 1"}
EXTRA_1B="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --fiber-passes 100 --fiber-lock-real --oneway-pool 2 --oneway-pool-scope uncovered"
EXTRA_1A="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --gpu-sparse auto"
COMMON_1B="--data adult_aimrep1_s1 --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 --init-oneway --exam-weights sigma --menu-seed 20260921 --sample-seed 1 --init-seed 1"
COMMON_1A="--data adult_alloc1base_s1 --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 --init-avg --init-clean --exam-weights none --menu-seed 20260921 --sample-seed 1 --init-seed 1"
FLOOR_1B=299038.1812299605
OUT=results/groupfloor/verify
LOG=logs_eps/groupfloor
mkdir -p $OUT $LOG
gpu_busy() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '); [ -z "$u" ] && return 0; [ "$u" -ge 5120 ]; }
pick_gpu() {
  local w=0 g
  while :; do
    for g in $GPUS; do
      if ! gpu_busy $g; then sleep 60; if ! gpu_busy $g; then GPU=$g; return; fi; fi
    done
    [ $((w % 1800)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu($GPUS) 均被占用，等待中（已等 $((w / 60)) min）"
    sleep 120; w=$((w + 120))
  done
}
run() {
  local name=$1; shift
  pick_gpu
  echo "$(date -u +%FT%TZ) START $name gpu$GPU load=$(cut -d' ' -f1-3 /proc/loadavg)"
  CUDA_VISIBLE_DEVICES=${UUID[$GPU]} taskset -c ${LANE[$GPU]} .venv/bin/python -u -B scripts/run_plants.py "$@" > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
# A
run verify_1b_snapon $COMMON_1B --stop-floor $FLOOR_1B \
  --save-table $OUT/eps1s1_snapon.csv --out $OUT/eps1s1_snapon_curve.csv \
  --snapshot-group-floor --snapshot-prefix $OUT/eps1s1_snapon $EXTRA_1B
# B
run verify_1a_gstop $COMMON_1A --stop-floor-groups \
  --save-table $OUT/eps1s1_1a_gstop.csv --out $OUT/eps1s1_1a_gstop_curve.csv $EXTRA_1A
# C
run verify_1b_gstop $COMMON_1B --stop-floor-groups \
  --save-table $OUT/eps1s1_gstop.csv --out $OUT/eps1s1_gstop_curve.csv $EXTRA_1B
echo "== md5 =="
md5sum results/whole1b/adult_eps1_ourswhole_seed1_stage1.csv $OUT/eps1s1_snapon_stage1.csv \
       results/whole1b/adult_eps1_ourswhole_seed1.csv $OUT/eps1s1_snapon.csv \
       results/floor/adult_eps1_ours2way_seed1_snap8.csv $OUT/eps1s1_1a_gstop.csv
echo "== 分档曲线核对（A）=="
.venv/bin/python - <<'EOF'
import csv
rows = list(csv.reader(open("results/groupfloor/verify/eps1s1_snapon_curve_stage1.csv", encoding="utf-8")))
loss = {r[0]: float(r[1]) for r in rows[1:] if r[0] != "final"}
g = list(csv.reader(open("results/groupfloor/verify/eps1s1_snapon_curve_groups.csv", encoding="utf-8")))
body = [r for r in g[1:] if r[0] != "floor"]
floors = [float(x) for x in g[-1][1:]]
worst = max(abs(sum(float(x) for x in r[1:]) - loss[r[0]]) / loss[r[0]] for r in body)
print(f"分档记账轮数 {len(body)} / 曲线轮数 {len(loss)}，各档之和 vs 曲线损失 最大相对差 {worst:.3e}")
last = [float(x) for x in body[-1][1:]]
print("地板停那轮各档损失÷地板:", " ".join(f"{a / b:.3f}" for a, b in zip(last, floors)), f"总和÷地板 {sum(last) / sum(floors):.4f}")
EOF
echo "$(date -u +%FT%TZ) 核对场完"
