#!/bin/bash
# 乙 改后复测：5 场稀疏开（auto）+ 2 场同负载稠密对照（--gpu-sparse off，只跑 ε=3/ε=0.1 热启动 a）
cd /root/residual-evolution
bash logs_eps/prof/phase_runs4.sh yi
sed -e "s/^run adult0p1_late_b.*//" -e "s/^run adult3_late_b.*//" -e "s/^run adult3_early.*//" logs_eps/prof/phase_runs4.sh > /tmp/phase_runs4_ctrl.sh
bash /tmp/phase_runs4_ctrl.sh yioff --gpu-sparse off
echo "$(date -u +%FT%TZ) ALL DONE YI BATCH"
