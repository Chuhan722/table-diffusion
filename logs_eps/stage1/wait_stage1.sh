#!/bin/bash
# 本机守望：每 10 min 数一次 A6000 上第一阶段的产物，齐了就 tar 拉回（结果 + 录像 + 日志），缺什么打印什么。
# 齐 = results/stage1 下 ours2way 15 + hybrid 15 + pgmrep 15 张表，results/aimreplay 下 adult ε=10 录像 3 份。
# 用法: bash logs_eps/stage1/wait_stage1.sh [最长等待小时，默认 10]
cd "$(dirname "$0")/../.." || exit 1
R="ssh -o ConnectTimeout=15 -p 6006 root@10.8.176.53"
MAXH=${1:-10}
t0=$(date +%s)
while :; do
  st=$($R 'cd /root/residual-evolution && echo "$(ls results/stage1 | grep -c "_ours2way_seed[123].csv") $(ls results/stage1 | grep -c "_hybrid_seed[123].csv") $(ls results/stage1 | grep -c "_pgmrep_seed[123].csv") $(ls results/aimreplay | grep -c "adult_eps10_aimfull_seed[123]_measurements.json") gpuq=$(wc -l < logs_eps/stage1/jobs_stage1.txt) cpuq=$(wc -l < logs_eps/stage1/jobs_stage1_cpu.txt) running=$(pgrep -fc "scripts/run_(plants|aim_full|pgm_replay)") load=$(cut -d" " -f1 /proc/loadavg)"' 2>/dev/null)
  echo "$(date +%H:%M) ours2way/hybrid/pgmrep/aim10dump: $st"
  set -- $st
  if [ "$1" = 15 ] && [ "$2" = 15 ] && [ "$3" = 15 ] && [ "$4" = 3 ]; then
    echo "齐了，拉回…"
    $R 'cd /root/residual-evolution && tar czf - results/stage1 results/aimreplay/adult_eps10_aimfull_seed*.csv results/aimreplay/adult_eps10_aimfull_seed*_measurements.json data/adult_aimrep10_s1 data/adult_aimrep10_s2 data/adult_aimrep10_s3 logs_eps/stage1/*.log logs_eps/stage1/jobs_stage1*.txt' | tar xzf -
    echo "$(date +%H:%M) 拉回完成：$(ls results/stage1/*.csv | grep -vc curve) 张表"
    exit 0
  fi
  if [ $(( $(date +%s) - t0 )) -gt $(( MAXH * 3600 )) ]; then echo "超时 ${MAXH}h，未齐"; exit 1; fi
  sleep 600
done
