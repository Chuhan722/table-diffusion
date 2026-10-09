#!/bin/bash
# 第五十六步提速 A/B/C 验收（二）：多字段路径改"各槽边际并集"后的终版代码，A6000 GPU1，核 56-79。
# 1) nltcs ε=3 s1 完整复现（新代码）比归档表与曲线（归档产自 A6000 GPU1，2026-10-08）；
# 2) acsmob ε=10 s1 终表续跑 100 轮：旧代码 vs 新代码（8 万组大表场景）；
# 3) acsmob ε=10 s1 冷启 200 轮：旧 vs 新；
# 4) plants ε=10 s1 完整复现（终版新代码）比归档表与曲线。
# 命令与 launch_E_a6000.sh run_alloc 逐字相同，只改输出路径、轮数上限与初始化来源。
set -u
cd /root/residual-evolution
export CUDA_VISIBLE_DEVICES=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082
CORES=56-79
OUT=results/prof
LOG=logs_eps/prof
mkdir -p $OUT
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"

run_one() {  # script_path data rounds seed name init...
  local script=$1 data=$2 rounds=$3 seed=$4 name=$5; shift 5
  echo "$(date -u +%FT%TZ) START $name"
  taskset -c $CORES .venv/bin/python -u -B $script \
    --data $data --rounds $rounds --stop-threshold 1e-4 --stop-lag 100 \
    "$@" --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA > $LOG/validate_$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? $(grep -h '耗时' $LOG/validate_$name.log | tail -1)"
}
cmp_pair() {  # label a b
  cmp -s $2 $3 && echo "$1 IDENTICAL" || echo "$1 DIFFER"
}

# 1) nltcs ε=3 s1 完整复现
run_one scripts/run_plants.py nltcs_alloc3k120auto_s1 30000 1 nltcs3_new_full --init-oneway
md5sum $OUT/nltcs3_new_full.csv results/alloc/nltcs_eps3_alloc_k120auto_seed1.csv
cmp_pair "NLTCS3 TABLE" $OUT/nltcs3_new_full.csv results/alloc/nltcs_eps3_alloc_k120auto_seed1.csv
cmp_pair "NLTCS3 CURVE" $OUT/nltcs3_new_full_curve.csv results/alloc/nltcs_eps3_alloc_k120auto_seed1_curve.csv

# 2) acsmob ε=10 s1 终表续跑 100 轮，旧 vs 新
INIT=results/alloc/acsmob_eps10_alloc_k210auto_seed1.csv
run_one /root/re_old/scripts/run_plants.py acsmob_alloc10k210auto_s1 100 1 acsmob10_old_late100 --init-table $INIT
run_one scripts/run_plants.py acsmob_alloc10k210auto_s1 100 1 acsmob10_new_late100 --init-table $INIT
md5sum $OUT/acsmob10_old_late100.csv $OUT/acsmob10_new_late100.csv
cmp_pair "ACSMOB LATE100 TABLE" $OUT/acsmob10_old_late100.csv $OUT/acsmob10_new_late100.csv
cmp_pair "ACSMOB LATE100 CURVE" $OUT/acsmob10_old_late100_curve.csv $OUT/acsmob10_new_late100_curve.csv

# 3) acsmob ε=10 s1 冷启 200 轮，旧 vs 新
run_one /root/re_old/scripts/run_plants.py acsmob_alloc10k210auto_s1 200 1 acsmob10_old_early200 --init-oneway
run_one scripts/run_plants.py acsmob_alloc10k210auto_s1 200 1 acsmob10_new_early200 --init-oneway
md5sum $OUT/acsmob10_old_early200.csv $OUT/acsmob10_new_early200.csv
cmp_pair "ACSMOB EARLY200 TABLE" $OUT/acsmob10_old_early200.csv $OUT/acsmob10_new_early200.csv
cmp_pair "ACSMOB EARLY200 CURVE" $OUT/acsmob10_old_early200_curve.csv $OUT/acsmob10_new_early200_curve.csv

# 4) plants ε=10 s1 完整复现（终版）
run_one scripts/run_plants.py plants_alloc10k2346auto_s1 30000 1 plants10_new_full2 --init-oneway
md5sum $OUT/plants10_new_full2.csv results/alloc/plants_eps10_alloc_k2346auto_seed1.csv
cmp_pair "PLANTS10v2 TABLE" $OUT/plants10_new_full2.csv results/alloc/plants_eps10_alloc_k2346auto_seed1.csv
cmp_pair "PLANTS10v2 CURVE" $OUT/plants10_new_full2_curve.csv results/alloc/plants_eps10_alloc_k2346auto_seed1_curve.csv
echo "$(date -u +%FT%TZ) ALL DONE VALIDATE2"
