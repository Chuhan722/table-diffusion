#!/usr/bin/env python3
"""第七十二步 CPU 调度看守（10-19 用户令"看哪个空闲用哪个，总数不超过一半"）。

整机 112 核与别人共用、无隔离。本看守让我们所有 run_plants 进程只在"别人最闲的 56 核"（恰好一半）上跑：
  每 SAMPLE 秒采样 /proc/stat 逐核忙碌度，扣掉我们自己的 CPU 时间（按当前掩码均摊）得到别人的忙碌度；
  每 RESELECT 秒重选 56 核（换核要闲出 HYST 以上才换，免得来回跳），写到 MASK 文件；
  每 INTERVAL 秒把活着、且已启动 ≥ WIDEN_AFTER 秒的进程全线程重绑到当前 56 核（taskset -a -p）。
启动时线程数由当时的亲和掩码决定（libgomp 在进程起步时读），所以 autostop.sh 先给每个进程 14 核起步、看守再放宽到 56 核，
避免 4 个进程各开 56 线程挤 56 核。只改调度，不改任何计算，结果逐位不变。
用法：cd /root/residual-evolution && nohup python3 logs_eps/autostop/cpu_watch.py > /dev/null 2>&1 &
  --once：只采样、打印并写一次掩码（不重绑、不循环）。
没有发射器（launch_autostop.sh / launch_dense.sh）也没有 run_plants 进程时自动退出。日志 logs_eps/autostop/cpu_watch.log。
"""
import os
import subprocess
import sys
import time

NCPU = os.cpu_count() or 112
BUDGET = NCPU // 2
MASK = "logs_eps/autostop/cpu_mask"
LOG = "logs_eps/autostop/cpu_watch.log"
SAMPLE, INTERVAL, RESELECT, HYST, WIDEN_AFTER = 5, 60, 300, 0.10, 60


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def procs(pattern: str) -> list:
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd:
            out.append(int(pid))
    return out


def our_pids() -> list:
    return procs("scripts/run_plants.py")


def launcher_alive() -> bool:
    return bool(procs("logs_eps/autostop/launch_autostop.sh") or procs("logs_eps/fission/launch_dense.sh"))


def cpu_stat() -> dict:
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            v = list(map(int, f[1:]))
            d[int(f[0][3:])] = (sum(v), v[3] + v[4])  # (total, idle+iowait)
    return d


def our_ticks(pids: list) -> int:
    s = 0
    for pid in pids:
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            s += int(f[11]) + int(f[12])  # utime + stime（USER_HZ）
        except (OSError, IndexError, ValueError):
            pass
    return s


def age_s(pid: int) -> float:
    try:
        start = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19])
        up = float(open("/proc/uptime").read().split()[0])
        return up - start / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return 0.0


def read_mask() -> list:
    try:
        return [int(x) for x in open(MASK).read().strip().split(",") if x]
    except OSError:
        return []


def write_mask(cores: list) -> None:
    tmp = MASK + ".tmp"
    with open(tmp, "w") as f:
        f.write(",".join(map(str, sorted(cores))) + "\n")
    os.replace(tmp, MASK)


def measure(mask: list) -> dict:
    """返回逐核"别人的忙碌度"（0～1）：总忙碌度减去我们自己的份额（按掩码均摊）。"""
    pids = our_pids()
    a, ta = cpu_stat(), our_ticks(pids)
    time.sleep(SAMPLE)
    b, tb = cpu_stat(), our_ticks(pids)
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a}
    per_core_ticks = sum(b[c][0] - a[c][0] for c in a) / len(a)
    ours = (tb - ta) / max(1, per_core_ticks)  # 我们占满了几个核
    inside = set(mask) if mask else set(range(NCPU))
    share = ours / len(inside)
    others = {c: max(0.0, busy[c] - (share if c in inside else 0.0)) for c in busy}
    others["_ours"] = ours
    others["_busy_all"] = sum(busy.values()) / len(busy)
    return others


def select(cur: list, others: dict) -> list:
    """从当前 56 核出发：里面最忙的和外面最闲的配对，外面比里面闲 HYST 以上才换。"""
    cores = [c for c in others if isinstance(c, int)]
    if not cur:
        return sorted(sorted(cores, key=lambda c: others[c])[:BUDGET])
    inside = sorted(cur, key=lambda c: -others[c])
    outside = sorted((c for c in cores if c not in cur), key=lambda c: others[c])
    new = set(cur)
    for ci, co in zip(inside, outside):
        if others[co] < others[ci] - HYST:
            new.discard(ci)
            new.add(co)
        else:
            break
    for c in outside:  # 不足 56 核时用外面最闲的补齐
        if len(new) >= BUDGET:
            break
        new.add(c)
    return sorted(new)[:BUDGET]


def pin(pids: list, mask: list) -> int:
    n = 0
    spec = ",".join(map(str, mask))
    for pid in pids:
        if age_s(pid) < WIDEN_AFTER:
            continue
        r = subprocess.run(["taskset", "-a", "-p", "-c", spec, str(pid)], capture_output=True, text=True)
        if r.returncode == 0:
            n += 1
    return n


def main() -> None:
    once = "--once" in sys.argv
    mask = read_mask()
    others = measure(mask)
    new = select(mask, others)
    write_mask(new)
    log(f"起步 整机忙碌 {others['_busy_all']:.0%} 我们占 {others['_ours']:.1f} 核；选 {len(new)} 核 别人平均忙 "
        f"{sum(others[c] for c in new) / len(new):.0%}（外面 {sum(others[c] for c in range(NCPU) if c not in new) / max(1, NCPU - len(new)):.0%}）：{','.join(map(str, new))}")
    if once:
        return
    mask = new
    last_sel = time.time()
    while True:
        pids = our_pids()
        if not pids and not launcher_alive():
            log("没有 run_plants 也没有 launch_autostop.sh，退出")
            return
        n = pin(pids, mask)
        if time.time() - last_sel >= RESELECT:
            others = measure(mask)
            new = select(mask, others)
            swapped = sorted(set(new) - set(mask))
            if swapped:
                write_mask(new)
                mask = new
            log(f"重估 整机忙碌 {others['_busy_all']:.0%} 我们占 {others['_ours']:.1f} 核 进程 {len(pids)} 重绑 {n}；"
                f"掩码内别人平均忙 {sum(others[c] for c in mask) / len(mask):.0%}，换入 {len(swapped)} 核"
                + (f"：{','.join(map(str, swapped))}" if swapped else ""))
            last_sel = time.time()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
