#!/bin/bash
# 第七十二步 探针并行自动停（一格）：探针 P 与正式跑 F 同一块卡同时开，P 收敛后判停法，再决定交 F 的地板表还是 F 的终表。
#   P = 第七十一步探针命令逐字（学习卷 data/<源>_fisA，第一段跑到收敛，只落分档过线快照 G）；
#   F = 最终方法收敛停命令逐字（1A：第六十二步 ours2way 命令；1B：第六十九步 launch_whole1b_conv.sh 命令，含一阶清洗与第二段）
#       加 --snapshot-group-floor（纯记账，轨迹不变）；F 的 G 快照 = 地板停第一段会交出的表。
#   判停法：decide.py 对模拟卷 y−e 算 L1_B，G 低于终 → 地板停，否则收敛停。
#   地板停：1A 直接交 F 的 G 快照（1A 无第二段），F 若未收敛即杀掉；
#           1B 杀掉 F，另跑地板停整套命令（第六十九步 launch_whole1b_gstop.sh 逐字），与现有地板停臂逐位同。
#   收敛停：等 F 收尾，交 F 的终表。
#   计时：t0 = 两进程启动，墙钟 = 交表时刻 − t0；各阶段时刻写 results/autostop/<格>_times.json。
# 用法：bash logs_eps/autostop/autostop.sh 考卷(1A|1B) ε标签 种子 gpu
#   环境变量 CORES_P / CORES_F：两进程各自的核（默认把该卡的 24 核对半分：P 前 12、F 后 12；FULL=1 则两者都用该卡整条 24 核）；NICE_F：F 的 nice 值（默认 0；给 19 则 P 优先、F 用剩余 CPU）；
#   FORCE=floor|conv 强制走某分支（只为验证分支）；VARIANT=名 时产物改写到 results/autostop_名 与 logs_eps/autostop_名（对照不同核分配用）。
#   STOPZ 非空（第七十三步 探针提前停）：P 加分裂探针记账与显著回升停（--fission-exam 模拟卷 --fission-every ${EVERY:-5}
#   --fission-stop-z STOPZ --fission-stop-runs ${STOPRUNS:-3}）；P 以 fission_rise 收工 → 判地板停（不等收敛、不跑 decide.py），
#   否则（跑到收敛）→ 判收敛停，decide.py 只作参考写 <格>_decision_endpoint.json。分支动作与原来相同。
cd /root/residual-evolution || exit 1
PY=.venv/bin/python
export CUDA_PATH=/root/residual-evolution/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime
declare -A UUID=([0]=GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed [1]=GPU-114f21d5-e47f-903c-a164-83a9f9dfa082)
declare -A LANE=([0]=0-23 [1]=24-47)
declare -A HALF_A=([0]=0-11 [1]=24-35)
declare -A HALF_B=([0]=12-23 [1]=36-47)
exam=$1; tag=$2; seed=$3; gpu=$4
cell=${exam}_eps${tag}_s${seed}
if [ -n "$FULL" ]; then CP=${CORES_P:-${LANE[$gpu]}}; CF=${CORES_F:-${LANE[$gpu]}}; else CP=${CORES_P:-${HALF_A[$gpu]}}; CF=${CORES_F:-${HALF_B[$gpu]}}; fi
CPU_MODE=fixed
# 10-19 用户令"看哪个空闲用哪个，总数不超过一半"：有 cpu_watch.py 写的 56 核掩码时，本道取其中 28 核，P/F 各 14 核起步
# （起步掩码定线程数），看守随后把两进程放宽到整张 56 核掩码并每分钟重绑；S 同理。没掩码文件则沿用上面的固定分法。
MASKF=logs_eps/autostop/cpu_mask
if [ -z "$CORES_P$CORES_F$FULL" ] && [ -s $MASKF ]; then
  IFS=, read -r -a MK < $MASKF
  if [ ${#MK[@]} -ge 56 ]; then
    o=$((gpu * 28)); join() { local IFS=,; echo "$*"; }
    CP=$(join "${MK[@]:$o:14}"); CF=$(join "${MK[@]:$((o + 14)):14}"); CS=$(join "${MK[@]:$o:28}"); CPU_MODE=watch56
  fi
fi
CS=${CS:-${LANE[$gpu]}}
NF=${NICE_F:-0}
EXTRA="--gpu --batched --pairing --work-rows 4096 --work-below 0.2 --retries 60 --pair-rescue-rows 2000 --rescue-after 6 --max-pairs 128 --rescue-stop-window 10 --rescue-stop-tol 0.02 --retry-select --struct-ruler ding --struct-boost 4 --progress 200 --gpu-entry-budget 40000000"
OUT=results/autostop${VARIANT:+_$VARIANT}
LOG=logs_eps/autostop${VARIANT:+_$VARIANT}
mkdir -p $OUT $LOG
SEEDS="--menu-seed $((20260920 + seed)) --sample-seed $seed --init-seed $seed"
STOP="--rounds 30000 --stop-threshold 1e-4 --stop-lag 100"
if [ "$exam" = "1B" ]; then
  src=adult_aimrep${tag}_s${seed}
  commonP="--data ${src}_fisA $STOP --init-oneway --exam-weights sigma"
  commonF="--data ${src} $STOP --init-oneway --exam-weights sigma"
  extraP="$EXTRA --oneway-pool 2 --oneway-pool-scope uncovered"
  extraF="$EXTRA --fiber-passes 100 --fiber-lock-real --oneway-pool 2 --oneway-pool-scope uncovered"
else
  src=adult_alloc${tag}base_s${seed}
  commonP="--data ${src}_fisA --exam measured_2way.json $STOP --init-avg --init-clean --exam-weights none"
  commonF="--data ${src} --exam measured_2way.json $STOP --init-avg --init-clean --exam-weights none"
  extraP="$EXTRA --gpu-sparse auto"
  extraF="$EXTRA --gpu-sparse auto"
fi
now() { date +%s.%N; }
dt() { awk "BEGIN{printf \"%.1f\", $1 - $2}"; }
stamp() { date -u +%FT%TZ; }
rm -f $OUT/P_${cell}_snapG.csv $OUT/F_${cell}_snapG.csv $OUT/${cell}_out.csv
t0=$(now)
echo "$(stamp) [$cell] 开 P(核 $CP) 与 F(核 $CF, nice $NF) 于 gpu$gpu（CPU 分法 $CPU_MODE）"
PROBE=""
[ -n "$STOPZ" ] && PROBE="--fission-exam results/fission/${src}_fisB.json --fission-every ${EVERY:-5} --fission-out $OUT/P_${cell}_probe.csv --fission-stop-z $STOPZ --fission-stop-runs ${STOPRUNS:-3}"
CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c $CP $PY -u -B scripts/run_plants.py $commonP $SEEDS \
  --save-table $OUT/P_$cell.csv --out $OUT/P_${cell}_curve.csv \
  --snapshot-group-floor --snapshot-prefix $OUT/P_$cell $PROBE $extraP > $LOG/P_$cell.log 2>&1 &
pidP=$!
CUDA_VISIBLE_DEVICES=${UUID[$gpu]} nice -n $NF taskset -c $CF $PY -u -B scripts/run_plants.py $commonF $SEEDS \
  --save-table $OUT/F_$cell.csv --out $OUT/F_${cell}_curve.csv \
  --snapshot-group-floor --snapshot-prefix $OUT/F_$cell $extraF > $LOG/F_$cell.log 2>&1 &
pidF=$!
wait $pidP; rcP=$?; tP=$(now)
echo "$(stamp) [$cell] P 完 rc=$rcP 用时 $(dt $tP $t0) 秒"
reasonP=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['stop_reason'])" $OUT/P_${cell}_curve_groups.json 2>/dev/null)
if [ -n "$STOPZ" ]; then
  # 第七十三步：判停只看 P 的停止原因；显著回升停触发 → 地板停，收敛 → 收敛停（终点判停法只作参考）
  if [ "$reasonP" = "fission_rise" ]; then dec=floor; else dec=conv; fi
  $PY - "$OUT/P_${cell}_probe.csv" "$OUT/${cell}_decision.json" "$dec" "$reasonP" "$STOPZ" "${STOPRUNS:-3}" <<'PYEOF'
import csv, json, sys
probe, out, dec, reason, z, runs = sys.argv[1:7]
rows = list(csv.DictReader(open(probe, encoding="utf-8")))
g = [r for r in rows if r["at_group_floor"] == "1"]
fin = [r for r in rows if r["round"] == "final"]
num = [r for r in rows if r["round"] != "final"]
res = dict(decision=dec, rule="fission_rise", stop_reason_P=reason, z=float(z), runs=int(runs), questions=None,
           G_round=int(g[0]["round"]) if g else None, G_elapsed_s=float(g[0]["elapsed_s"]) if g else None,
           L1_B_G=float(g[0]["l1_b"]) if g else None,
           P_last_round=int(num[-1]["round"]) if num else None, P_last_elapsed_s=float(num[-1]["elapsed_s"]) if num else None,
           L1_B_final=float(fin[0]["l1_b"]) if fin else None,
           diff_mean=float(fin[0]["diff_mean"]) if fin and fin[0]["diff_mean"] != "" else None,
           diff_se=float(fin[0]["diff_se"]) if fin and fin[0]["diff_se"] != "" else None)
res["z_final"] = res["diff_mean"] / res["diff_se"] if res["diff_se"] else None
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PYEOF
  tD=$(now)
  if [ "$dec" = "conv" ]; then
    $PY logs_eps/autostop/decide.py results/fission/${src}_fisB.json $OUT/P_${cell}_snapG.csv $OUT/P_$cell.csv $OUT/${cell}_decision_endpoint.json > /dev/null 2>&1 &
  fi
else
  dec=$($PY logs_eps/autostop/decide.py results/fission/${src}_fisB.json $OUT/P_${cell}_snapG.csv $OUT/P_$cell.csv $OUT/${cell}_decision.json | tail -1)
  tD=$(now)
fi
case "$dec" in floor|conv) ;; *) echo "$(stamp) [$cell] 判定失败（得到 '$dec'），按收敛停交表并记 error"; dec=error;; esac
[ -n "$FORCE" ] && { echo "$(stamp) [$cell] 判为 $dec，FORCE=$FORCE 强制"; dec=$FORCE; }
echo "$(stamp) [$cell] 判停法 $dec（P 停止原因 ${reasonP:-?}，判定用时 $(dt $tD $tP) 秒）"
rcF=""; rcS=""; tF=""; tS=""; killedF=0
if [ "$dec" = "floor" ]; then
  while [ ! -s $OUT/F_${cell}_snapG.csv ] && kill -0 $pidF 2>/dev/null; do sleep 1; done
  if kill -0 $pidF 2>/dev/null; then kill $pidF; killedF=1; fi
  wait $pidF 2>/dev/null; rcF=$?; tF=$(now)
  echo "$(stamp) [$cell] F 收 rc=$rcF 杀掉=$killedF"
  if [ "$exam" = "1B" ]; then
    CUDA_VISIBLE_DEVICES=${UUID[$gpu]} taskset -c $CS $PY -u -B scripts/run_plants.py $commonF $SEEDS --stop-floor-groups \
      --save-table $OUT/S_$cell.csv --out $OUT/S_${cell}_curve.csv $extraF > $LOG/S_$cell.log 2>&1
    rcS=$?; tS=$(now); cp $OUT/S_$cell.csv $OUT/${cell}_out.csv
    echo "$(stamp) [$cell] 地板停整套 S 完 rc=$rcS 用时 $(dt $tS $tF) 秒"
  else
    cp $OUT/F_${cell}_snapG.csv $OUT/${cell}_out.csv
  fi
else
  wait $pidF; rcF=$?; tF=$(now)
  echo "$(stamp) [$cell] F 完 rc=$rcF 用时 $(dt $tF $t0) 秒"
  cp $OUT/F_$cell.csv $OUT/${cell}_out.csv
fi
tE=$(now)
gsnap=$(stat -c %Y $OUT/F_${cell}_snapG.csv 2>/dev/null || echo "")
cat > $OUT/${cell}_times.json <<EOF
{"cell": "$cell", "exam": "$exam", "tag": "$tag", "seed": $seed, "gpu": $gpu, "cores_P": "$CP", "cores_F": "$CF", "nice_F": $NF, "cpu_mode": "$CPU_MODE",
 "decision": "$dec", "force": "${FORCE:-}", "stopz": "${STOPZ:-}", "stopruns": "${STOPRUNS:-}", "every": "${EVERY:-}", "reasonP": "${reasonP:-}", "t0": $t0, "tP": $tP, "tD": $tD, "tF": ${tF:-null}, "tS": ${tS:-null}, "tE": $tE,
 "F_snapG_mtime": ${gsnap:-null}, "rcP": $rcP, "rcF": ${rcF:-null}, "rcS": ${rcS:-null}, "killedF": $killedF,
 "wall": $(dt $tE $t0), "load_end": "$(cut -d' ' -f1-3 /proc/loadavg)"}
EOF
echo "$(stamp) [$cell] 交表 ${cell}_out.csv 墙钟 $(dt $tE $t0) 秒"
