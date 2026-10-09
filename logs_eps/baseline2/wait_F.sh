#!/bin/bash
# F 组守望：每 15 分钟查一次；GPU 道 "ALL DONE F" 与 CPU 道 "ALL DONE F-CPU" 都出现后拉回全部 acsmob/plants ε=0.3 产物。
cd /home/chuhan/projects0916/residual-evolution
R="ssh -o BatchMode=yes -p 6006 root@10.8.176.53"
until $R "grep -q 'ALL DONE F' /root/residual-evolution/logs_eps/baseline2/launch_F_a6000.log && grep -q 'ALL DONE F-CPU' /root/residual-evolution/logs_eps/baseline2/launch_F_cpu.log" 2>/dev/null; do sleep 900; done
echo "$(date -u +%FT%TZ) F 组全收，开始拉取"
$R "cd /root/residual-evolution && tar czf - results/alloc/acsmob_eps0p3_* results/alloc/plants_eps0p3_* results/baseline2/acsmob_eps0p3_gsd_* results/baseline2/plants_eps0p3_gsd_* results/acsmob_eps0p3_aimfull_* logs_eps/alloc/acsmob_eps0p3_*.log logs_eps/alloc/plants_eps0p3_*.log logs_eps/baseline2/acsmob_eps0p3_*.log logs_eps/baseline2/plants_eps0p3_*.log logs_eps/baseline2/F_gpu*.log logs_eps/baseline2/launch_F_*.log 2>/dev/null" | tar xzf -
echo "$(date -u +%FT%TZ) 拉取完成"; ls results/alloc/acsmob_eps0p3_*.csv results/alloc/plants_eps0p3_*.csv results/baseline2/*eps0p3_gsd*.csv results/acsmob_eps0p3_aimfull*.csv 2>/dev/null | grep -v curve
$R "cat /root/residual-evolution/logs_eps/baseline2/F_gpu0.log /root/residual-evolution/logs_eps/baseline2/F_gpu1.log /root/residual-evolution/logs_eps/baseline2/launch_F_cpu.log"
