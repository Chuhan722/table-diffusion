#!/bin/bash
# 等 A6000 上 A 组（GSD/PGM 同餐 Δ₂=1 + adult ε=1 k105f05 seed3）全部结束，然后拉回产物。
cd /home/chuhan/projects0916/residual-evolution
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -p 6006 root@10.8.176.53 'grep -c "ALL DONE" /root/residual-evolution/logs_eps/baseline2/launch_baseA.log 2>/dev/null' 2>/dev/null)
  if [ "${r:-0}" -ge 1 ]; then break; fi
  sleep 300
done
echo "$(date -u +%FT%TZ) A 组全部结束，开始拉取"
ssh -o BatchMode=yes -p 6006 root@10.8.176.53 'cd /root/residual-evolution && tar czf - results/baseline2/*.csv results/alloc/adult_eps1_alloc_k105f05_seed3*.csv logs_eps/alloc/adult_eps1_alloc_k105f05_seed3.log logs_eps/baseline2/*.log 2>/dev/null' | tar xzf - 2>&1 | grep -v 'Removing leading' | head
echo "$(date -u +%FT%TZ) 拉取完成"
ls results/baseline2/*.csv 2>/dev/null | wc -l
ls results/alloc/adult_eps1_alloc_k105f05_seed3*.csv 2>/dev/null
