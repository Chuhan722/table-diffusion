#!/bin/bash
# 第五十六步提速 A/B/C 验收（三）：终版代码（并集 + 整段直算后切块 + 空块防护），A6000 GPU0，核 56-79。
# 与已跑过的旧代码产物/归档逐位比：adult ε=3 s1 300 轮、acsmob ε=10 s1 终表续 100 轮、nltcs ε=3 s1 完整、plants ε=10 s1 完整。
set -u
cd /root/residual-evolution
export CUDA_VISIBLE_DEVICES=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed
CORES=56-79
OUT=results/prof
LOG=logs_eps/prof
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
cmp_pair() { cmp -s $2 $3 && echo "$1 IDENTICAL" || echo "$1 DIFFER"; }

run_one scripts/run_plants.py adult_alloc3k105auto_s1 300 1 adult3_new300b --init-oneway
cmp_pair "ADULT300b TABLE" $OUT/adult3_new300b.csv $OUT/adult3_old300.csv
cmp_pair "ADULT300b CURVE" $OUT/adult3_new300b_curve.csv $OUT/adult3_old300_curve.csv

run_one scripts/run_plants.py acsmob_alloc10k210auto_s1 100 1 acsmob10_new_late100b --init-table results/alloc/acsmob_eps10_alloc_k210auto_seed1.csv
cmp_pair "ACSMOB LATE100b TABLE" $OUT/acsmob10_new_late100b.csv $OUT/acsmob10_old_late100.csv
cmp_pair "ACSMOB LATE100b CURVE" $OUT/acsmob10_new_late100b_curve.csv $OUT/acsmob10_old_late100_curve.csv

run_one scripts/run_plants.py nltcs_alloc3k120auto_s1 30000 1 nltcs3_new_fullb --init-oneway
cmp_pair "NLTCS3b TABLE" $OUT/nltcs3_new_fullb.csv results/alloc/nltcs_eps3_alloc_k120auto_seed1.csv
cmp_pair "NLTCS3b CURVE" $OUT/nltcs3_new_fullb_curve.csv results/alloc/nltcs_eps3_alloc_k120auto_seed1_curve.csv

run_one scripts/run_plants.py plants_alloc10k2346auto_s1 30000 1 plants10_new_full3 --init-oneway
cmp_pair "PLANTS10v3 TABLE" $OUT/plants10_new_full3.csv results/alloc/plants_eps10_alloc_k2346auto_seed1.csv
cmp_pair "PLANTS10v3 CURVE" $OUT/plants10_new_full3_curve.csv results/alloc/plants_eps10_alloc_k2346auto_seed1_curve.csv
echo "$(date -u +%FT%TZ) ALL DONE VALIDATE3"
