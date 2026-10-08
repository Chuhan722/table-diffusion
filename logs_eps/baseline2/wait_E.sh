#!/bin/bash
# E 组守望：每 15 分钟查一次；GPU 道 "ALL DONE E" 与 CPU 道 "ALL DONE E-CPU" 都出现后拉回全部 acsmob/plants ε=0.1/3/10 产物。
cd /home/chuhan/projects0916/residual-evolution
R="ssh -o BatchMode=yes -p 6006 root@10.8.176.53"
until $R "grep -q 'ALL DONE E' /root/residual-evolution/logs_eps/baseline2/launch_E_a6000.log && grep -q 'ALL DONE E-CPU' /root/residual-evolution/logs_eps/baseline2/launch_E_cpu.log" 2>/dev/null; do sleep 900; done
echo "$(date -u +%FT%TZ) E 组全收，开始拉取"
$R "cd /root/residual-evolution && tar czf - results/alloc/acsmob_eps{0p1,3,10}_* results/alloc/plants_eps{0p1,3,10}_* results/baseline2/acsmob_eps{0p1,3,10}_gsd_* results/baseline2/plants_eps{0p1,3,10}_gsd_* results/acsmob_eps{0p1,3,10}_aimfull_* logs_eps/alloc/acsmob_eps{0p1,3,10}_*.log logs_eps/alloc/plants_eps{0p1,3,10}_*.log logs_eps/baseline2/acsmob_eps{0p1,3,10}_*.log logs_eps/baseline2/plants_eps{0p1,3,10}_*.log logs_eps/baseline2/E_gpu*.log logs_eps/baseline2/launch_E_*.log 2>/dev/null" | tar xzf -
echo "$(date -u +%FT%TZ) 拉取完成"; ls results/alloc/acsmob_eps{0p1,3,10}_*.csv results/alloc/plants_eps{0p1,3,10}_*.csv results/baseline2/*eps{0p1,3,10}_gsd*.csv results/acsmob_eps{0p1,3,10}_aimfull*.csv 2>/dev/null | grep -v curve
$R "cat /root/residual-evolution/logs_eps/baseline2/E_gpu0.log /root/residual-evolution/logs_eps/baseline2/E_gpu1.log /root/residual-evolution/logs_eps/baseline2/launch_E_cpu.log"
