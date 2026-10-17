"""第六十七步诊断：新整体方法（第一段地板停表 newT1 / 终表 newFinal）对 旧收敛流水线（第一段收敛表 oldT1conv = results/stage1 hybrid、终表 oldFinal = results/stage2 hybridfiber）与 PGM 原生，二阶总体 / 测过的对 / 没测的对（×10⁻⁶，三种子均）。用法：.venv/bin/python logs_eps/whole1b/diag_vs_old.py"""
import sys, re, subprocess
from pathlib import Path
ROOT = Path('/home/chuhan/projects0916/residual-evolution')
ARMS = {
 'newT1': 'results/whole1b/adult_eps{t}_ourswhole_seed{s}_stage1.csv',
 'newFinal': 'results/whole1b/adult_eps{t}_ourswhole_seed{s}.csv',
 'oldT1conv': 'results/stage1/adult_eps{t}_hybrid_seed{s}.csv',
 'oldFinal': 'results/stage2/adult_eps{t}_hybridfiber_seed{s}.csv',
 'pgm10k': 'results/stage1/adult_eps{t}_pgmrep10k_seed{s}.csv',
}
def groups(t, s):
    dump = ROOT / f'results/aimreplay/adult_eps{t}_aimfull_seed{s}_measurements.json'
    arms = {a: ROOT / p.format(t=t, s=s) for a, p in ARMS.items()}
    arms = {a: p for a, p in arms.items() if p.exists()}
    cmd = [sys.executable, str(ROOT/'scripts/score_pair_groups.py'), '--real', str(ROOT/'data/adult/adult.csv'), '--dump', str(dump)] + [f'{a}={p}' for a, p in arms.items()]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    rows = {}
    for line in out.splitlines():
        m = re.match(r'(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$', line)
        if m and m.group(1) != '表':
            rows[m.group(1)] = [float(m.group(i)) for i in (2, 3, 4)]
    return rows
import numpy as np
print("ε | 臂 | 二阶总体 | 测过的对 | 没测的对")
for t in ['0p1', '0p3', '1', '3', '10']:
    acc = {a: [] for a in ARMS}
    for s in (1, 2, 3):
        r = groups(t, s)
        for a in ARMS:
            if a in r:
                acc[a].append(r[a])
    for a in ARMS:
        if acc[a]:
            m = np.mean(np.array(acc[a]), axis=0)
            print(f'{t} | {a:9s} | {m[0]:8.0f} | {m[1]:8.0f} | {m[2]:8.0f} | n={len(acc[a])}')
    print()
