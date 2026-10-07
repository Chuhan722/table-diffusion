#!/bin/bash
# 等 b 段三条流全部结束（A6000 两卡 ALL DONE + 本机“本机 b 段结束/退出”），然后拉回 b 段产物。
cd /home/chuhan/projects0916/residual-evolution
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -p 6006 root@10.8.176.53 'grep -c "ALL DONE" /root/residual-evolution/logs_eps/aimsel/launch_batch1b.log 2>/dev/null' 2>/dev/null)
  l=$(grep -c '本机 b 段结束\|本机流退出' logs_eps/aimsel/launch_batch1b_local.log 2>/dev/null)
  if [ "${r:-0}" -ge 1 ] && [ "${l:-0}" -ge 1 ]; then break; fi
  sleep 300
done
echo "$(date -u +%FT%TZ) b 段全部结束，开始拉取"
ssh -o BatchMode=yes -p 6006 root@10.8.176.53 'cd /root/residual-evolution && tar czf - results/alloc/adult_eps10_alloc_k105f05_seed*.csv results/alloc/nltcs_eps*_alloc_k120f05_seed*.csv results/aimsel/nltcs_eps*_aimsel_seed*.csv runs/aimsel2_nltcs*/aim_log.json runs/aimsel2_nltcs*/exam_final.json runs/aimsel2_nltcs*/engine_log.txt logs_eps/alloc/*.log logs_eps/aimsel/nltcs_eps*_aimsel_seed*.log logs_eps/aimsel/launch_batch1b.log logs_eps/aimsel/batch1b_gpu*.log logs_eps/aimsel/jobs_batch1b_claims.log 2>/dev/null' | tar xzf - 2>&1 | grep -v 'Removing leading' | head
echo "$(date -u +%FT%TZ) 拉取完成"
ls results/alloc/adult_eps10_alloc_k105f05_seed*.csv results/alloc/nltcs_eps*_alloc_k120f05_seed*.csv results/aimsel/nltcs_eps*_aimsel_seed*.csv 2>/dev/null | wc -l
