#!/bin/bash
# 等 A6000 M3 段(ALL DONE M3)后拉回 acsmob 引擎表 + GSD plants/acsmob 表与日志。
cd /home/chuhan/projects0916/residual-evolution
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -p 6006 root@10.8.176.53 'grep -c "ALL DONE M3" /root/residual-evolution/logs_eps/baseline2/launch_M3.log 2>/dev/null' 2>/dev/null)
  if [ "${r:-0}" -ge 1 ]; then break; fi
  sleep 600
done
echo "$(date -u +%FT%TZ) M3 A6000 段全部结束，开始拉取"
ssh -o BatchMode=yes -p 6006 root@10.8.176.53 'cd /root/residual-evolution && tar czf - results/alloc/acsmob_eps1_alloc_*.csv logs_eps/alloc/acsmob_eps1_alloc_*.log results/baseline2/plants_eps1_gsd_seed*.csv results/baseline2/acsmob_eps1_gsd_seed*.csv logs_eps/baseline2/plants_eps1_gsd_seed*.log logs_eps/baseline2/acsmob_eps1_gsd_seed*.log logs_eps/baseline2/launch_M3.log logs_eps/baseline2/M3_gpu*.log 2>/dev/null' | tar xzf - 2>&1 | grep -v 'Removing leading' | head
echo "$(date -u +%FT%TZ) 拉取完成"
ls results/alloc/acsmob_eps1_alloc_*seed?.csv results/baseline2/*_eps1_gsd_seed*.csv 2>/dev/null
