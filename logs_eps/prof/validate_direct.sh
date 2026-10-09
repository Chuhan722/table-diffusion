#!/bin/bash
# 第五十六步提速 A/B/C 验收：旧代码 vs 新代码同种子逐位对比（A6000 GPU0，核 56-79）。
# 1) adult ε=3 s1 截 300 轮：旧代码（/root/re_old）与新代码（直算自动识别）各跑一遍，比表与曲线 md5；
# 2) plants ε=10 s1 完整复现：新代码跑完整场，比对已归档的 results/alloc 表与曲线。
# 命令与 launch_E_a6000.sh run_alloc 逐字相同，只改输出路径与轮数上限。
set -u
cd /root/residual-evolution
export CUDA_VISIBLE_DEVICES=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed
CORES=56-79
OUT=results/prof
LOG=logs_eps/prof
mkdir -p $OUT
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"

run_one() {  # script_path data rounds seed name
  local script=$1 data=$2 rounds=$3 seed=$4 name=$5
  echo "$(date -u +%FT%TZ) START $name"
  taskset -c $CORES .venv/bin/python -u -B $script \
    --data $data --rounds $rounds --stop-threshold 1e-4 --stop-lag 100 \
    --init-oneway --exam-weights sigma --menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed \
    --save-table $OUT/$name.csv --out $OUT/${name}_curve.csv \
    $EXTRA > $LOG/validate_$name.log 2>&1
  echo "$(date -u +%FT%TZ) END $name rc=$? $(grep -h '耗时' $LOG/validate_$name.log | tail -1)"
}

# 1) adult ε=3 s1，300 轮，旧 vs 新
run_one /root/re_old/scripts/run_plants.py adult_alloc3k105auto_s1 300 1 adult3_old300
run_one scripts/run_plants.py adult_alloc3k105auto_s1 300 1 adult3_new300
md5sum $OUT/adult3_old300.csv $OUT/adult3_new300.csv $OUT/adult3_old300_curve.csv $OUT/adult3_new300_curve.csv
cmp -s $OUT/adult3_old300.csv $OUT/adult3_new300.csv && echo "ADULT300 TABLE IDENTICAL" || echo "ADULT300 TABLE DIFFER"
cmp -s $OUT/adult3_old300_curve.csv $OUT/adult3_new300_curve.csv && echo "ADULT300 CURVE IDENTICAL" || echo "ADULT300 CURVE DIFFER"

# 2) plants ε=10 s1 完整复现（新代码）
run_one scripts/run_plants.py plants_alloc10k2346auto_s1 30000 1 plants10_new_full
md5sum $OUT/plants10_new_full.csv results/alloc/plants_eps10_alloc_k2346auto_seed1.csv
cmp -s $OUT/plants10_new_full.csv results/alloc/plants_eps10_alloc_k2346auto_seed1.csv && echo "PLANTS10 TABLE IDENTICAL" || echo "PLANTS10 TABLE DIFFER"
cmp -s $OUT/plants10_new_full_curve.csv results/alloc/plants_eps10_alloc_k2346auto_seed1_curve.csv && echo "PLANTS10 CURVE IDENTICAL" || echo "PLANTS10 CURVE DIFFER"
echo "$(date -u +%FT%TZ) ALL DONE VALIDATE"
