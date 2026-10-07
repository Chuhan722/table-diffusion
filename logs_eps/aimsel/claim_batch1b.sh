#!/bin/bash
# 第五十四步第一批（b 段，接在 jobs_batch1 之后）：原子领取 jobs_batch1b.txt 首行（flock），队列空则无输出。
# A6000 两条 GPU lane 直接调用，本机 4090 经 ssh 调用，三方抢同一份任务表。用法：claim_batch1.sh <领取者名>
JOBS=/root/residual-evolution/logs_eps/aimsel/jobs_batch1b.txt
(
  flock 9
  l=$(head -1 "$JOBS")
  if [ -n "$l" ]; then
    sed -i 1d "$JOBS"
    echo "$(date -u +%FT%TZ) ${1:-?} 领取: $l" >> "${JOBS%.txt}_claims.log"
    echo "$l"
  fi
) 9>"$JOBS.lock"
