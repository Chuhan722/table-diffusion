#!/bin/bash
cd /home/chuhan/projects0916/residual-evolution
R="ssh -o BatchMode=yes -p 6006 root@10.8.176.53"
until $R "grep -q 'ALL DONE D' /root/residual-evolution/logs_eps/baseline2/launch_D_a6000.log && grep -q 'ALL DONE D-CPU' /root/residual-evolution/logs_eps/baseline2/launch_D_cpu.log" 2>/dev/null; do sleep 600; done
echo "$(date -u +%FT%TZ) D 组全收，开始拉取"
$R "cd /root/residual-evolution && tar czf - results/alloc/*eps0p3* results/aimsel/*eps0p3* results/baseline2/*eps0p3* results/*eps0p3_aimfull* logs_eps/alloc/*eps0p3*.log logs_eps/aimsel/*eps0p3*.log logs_eps/baseline2/*eps0p3*.log logs_eps/baseline2/D_gpu*.log logs_eps/baseline2/launch_D_*.log runs/aimsel2_adult0p3_s*/aim_log.json runs/aimsel2_adult0p3_s*/exam_final.json runs/aimsel2_nltcs0p3_s*/aim_log.json 2>/dev/null" | tar xzf -
echo "$(date -u +%FT%TZ) 拉取完成"; ls results/alloc/*eps0p3*.csv results/aimsel/*eps0p3*.csv results/baseline2/*eps0p3*.csv results/*eps0p3_aimfull*.csv | grep -v curve
$R "cat /root/residual-evolution/logs_eps/baseline2/D_gpu0.log /root/residual-evolution/logs_eps/baseline2/D_gpu1.log /root/residual-evolution/logs_eps/baseline2/launch_D_cpu.log"
