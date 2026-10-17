#!/bin/bash
# 第六十八步 核对场（本机 4090 GPU0 版，10-17 用户令"在 4090 上跑吧，只用 gpu0"）。只核对实现，不出结论。
#   A6000 被他人占满，改在本机跑；跨卡浮点可能在近平手抽签处分岔（第四十九步教训），所以同卡参照由旧代码
#   （git worktree /tmp/resevo_old = HEAD~1，与 A6000 上产出第六十七步结果的工作副本同 md5）现跑，
#   与 A6000 结果的 md5 只作跨卡旁证，不作判据。
#   R  旧代码：1B ε=1 种子 1 整体方法命令（第六十七步逐字）→ 同卡参照 ref_1b{_stage1,}.csv
#   A  新代码：同命令 + --snapshot-group-floor → 第一段表与终表须与 R 逐字节同；各档之和须等于曲线损失
#   B1 新代码：1A ε=1 种子 1 第六十二步命令（去快照）+ --stop-floor 12144672.776841227（硬地板停）
#   B2 新代码：同命令 + --stop-floor-groups（单档，应等价硬地板停）→ 表须与 B1 逐字节同
#   C  新代码：1B ε=1 种子 1 整体方法命令，--stop-floor 换 --stop-floor-groups（分档停冒烟：停点、各档比、耗时）
# 规矩：只用 GPU0（<5120 MB 连续两次才开），核 0-15（一半 CPU）。
# 用法：cd /home/chuhan/projects0916/residual-evolution && (nohup bash logs_eps/groupfloor/verify_groupfloor_4090.sh < /dev/null > logs_eps/groupfloor/verify_groupfloor_4090.log 2>&1 &)
NEW=/home/chuhan/projects0916/residual-evolution
OLD=/tmp/resevo_old
cd $NEW || exit 1
PY=$NEW/.venv/bin/python
EXTRA_1B="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --fiber-passes 100 --fiber-lock-real --oneway-pool 2 --oneway-pool-scope uncovered"
EXTRA_1A="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000 --gpu-sparse auto"
COMMON_1B="--data adult_aimrep1_s1 --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 --init-oneway --exam-weights sigma --menu-seed 20260921 --sample-seed 1 --init-seed 1"
COMMON_1A="--data adult_alloc1base_s1 --exam measured_2way.json --rounds 30000 --stop-threshold 1e-4 --stop-lag 100 --init-avg --init-clean --exam-weights none --menu-seed 20260921 --sample-seed 1 --init-seed 1"
FLOOR_1B=299038.1812299605
FLOOR_1A=12144672.776841227
OUT=$NEW/results/groupfloor/verify4090
LOG=$NEW/logs_eps/groupfloor
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
run() {  # run <代码目录> <场名> <参数…>
  local code=$1 name=$2; shift 2
  wait_gpu0
  echo "$(date -u +%FT%TZ) START $name code=$code load=$(cut -d' ' -f1-3 /proc/loadavg)"
  ( cd $code && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 $PY -u -B scripts/run_plants.py "$@" > $LOG/4090_$name.log 2>&1 )
  echo "$(date -u +%FT%TZ) END $name rc=$?"
}
# R 同卡参照（旧代码）
run $OLD ref_1b $COMMON_1B --stop-floor $FLOOR_1B \
  --save-table $OUT/ref_1b.csv --out $OUT/ref_1b_curve.csv $EXTRA_1B
# A
run $NEW verify_1b_snapon $COMMON_1B --stop-floor $FLOOR_1B \
  --save-table $OUT/eps1s1_snapon.csv --out $OUT/eps1s1_snapon_curve.csv \
  --snapshot-group-floor --snapshot-prefix $OUT/eps1s1_snapon $EXTRA_1B
# B1 / B2
run $NEW verify_1a_floor $COMMON_1A --stop-floor $FLOOR_1A \
  --save-table $OUT/eps1s1_1a_floor.csv --out $OUT/eps1s1_1a_floor_curve.csv $EXTRA_1A
run $NEW verify_1a_gstop $COMMON_1A --stop-floor-groups \
  --save-table $OUT/eps1s1_1a_gstop.csv --out $OUT/eps1s1_1a_gstop_curve.csv $EXTRA_1A
# C
run $NEW verify_1b_gstop $COMMON_1B --stop-floor-groups \
  --save-table $OUT/eps1s1_gstop.csv --out $OUT/eps1s1_gstop_curve.csv $EXTRA_1B
echo "== md5（判据：A 两表 = R 两表；B2 = B1）=="
md5sum $OUT/ref_1b_stage1.csv $OUT/eps1s1_snapon_stage1.csv $OUT/ref_1b.csv $OUT/eps1s1_snapon.csv \
       $OUT/eps1s1_1a_floor.csv $OUT/eps1s1_1a_gstop.csv
echo "== md5（旁证：跨卡 vs A6000 结果 20f9b4ed… / 98785ff3… / 0eab060e…）=="
md5sum $NEW/results/whole1b/adult_eps1_ourswhole_seed1_stage1.csv $NEW/results/whole1b/adult_eps1_ourswhole_seed1.csv \
       $NEW/results/floor/adult_eps1_ours2way_seed1_snap8.csv
echo "== 分档曲线核对（A）=="
$PY - <<EOF
import csv
out = "$OUT"
rows = list(csv.reader(open(f"{out}/eps1s1_snapon_curve_stage1.csv", encoding="utf-8")))
loss = {r[0]: float(r[1]) for r in rows[1:] if r[0] != "final"}
g = list(csv.reader(open(f"{out}/eps1s1_snapon_curve_groups.csv", encoding="utf-8")))
body = [r for r in g[1:] if r[0] != "floor"]
floors = [float(x) for x in g[-1][1:]]
worst = max(abs(sum(float(x) for x in r[1:]) - loss[r[0]]) / loss[r[0]] for r in body)
print(f"分档记账轮数 {len(body)} / 曲线轮数 {len(loss)}，各档之和 vs 曲线损失 最大相对差 {worst:.3e}")
last = [float(x) for x in body[-1][1:]]
print("地板停那轮各档损失÷地板:", " ".join(f"{a / b:.3f}" for a, b in zip(last, floors)), f"总和÷地板 {sum(last) / sum(floors):.4f}")
EOF
echo "$(date -u +%FT%TZ) 核对场完"
