from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

ROOT = Path("/root/residual-evolution")
LOGS = ROOT / "logs_eps" / "endpoints"
STAGING = ROOT / "results" / ".eps_endpoints_pending"
STATUS = LOGS / "status.json"
GPU_UUIDS = {
    0: "GPU-24b178f1-5d73-6405-752d-3c3aa98e83ed",
    1: "GPU-114f21d5-e47f-903c-a164-83a9f9dfa082",
}
# CPU 严守一半: 四条流全部落在 0-55 (112 核的一半), 每流 14 核; AIM 实测仅用 ~3.6 核.
CPU_LANES = {0: "0-13", 1: "14-27", 2: "28-41", 3: "42-55"}
# v1 控制器两条 28 核流的亲和掩码, 收养旧进程时映射到 v2 流位.
LEGACY_LANES = {"0-27": 0, "28-55": 2}
VENVPY = ROOT / ".venv/bin/python"
GSDPY = Path("/root/miniconda3/envs/aim/bin/python")
AIMPY = Path("/root/miniconda3/envs/aim2/bin/python")
SCHEDULER = "v2: per-GPU heavy(rule)+light(gsd) slots, 4 CPU lanes in 0-55, orphan adoption"


@dataclass(frozen=True)
class Job:
    dataset: str
    tag: str
    method: str
    seed: int

    def __post_init__(self) -> None:
        if self.dataset not in ("nltcs", "adult") or self.tag not in ("0p1", "10"):
            raise ValueError("This runner is restricted to the approved endpoint datasets/budgets")
        if self.seed not in (1, 2, 3) or self.method not in ("rule", "gsd", "aimfull", "pgm"):
            raise ValueError("Unsupported endpoint seed/method")
        if self.dataset == "adult" and self.method == "pgm":
            raise ValueError("Adult PGM is excluded because its junction tree is infeasible")

    @property
    def name(self) -> str:
        return f"{self.dataset}_eps{self.tag}_{self.method}_seed{self.seed}"

    @property
    def output(self) -> Path:
        return ROOT / "results" / f"{self.name}.csv"

    @property
    def uses_gpu(self) -> bool:
        return self.method in ("rule", "gsd")

    @property
    def heavy(self) -> bool:
        # 我们的 adult 场约 17.6GB 显存且 GPU 繁忙, 每卡一场; GSD <1GB 且 ~30% 利用率, 可伴跑.
        return self.method == "rule"


@dataclass(frozen=True)
class Orphan:
    pid: int
    start_ticks: str
    started_at: str
    gpu: int | None
    cpu_lane: int | None
    cpus: str | None
    argv: tuple


class Detach(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit(message: str) -> None:
    try:
        print(f"{now()} {message}", flush=True)
    except OSError:
        pass


def load_schemas() -> dict:
    schemas = {}
    for dataset in ("nltcs", "adult"):
        path = ROOT / "data" / dataset / f"{dataset}.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            fields = next(reader)
            domains = [set() for _ in fields]
            count = 0
            for row in reader:
                if len(row) != len(fields):
                    raise ValueError(f"Malformed source table: {path}")
                count += 1
                for values, value in zip(domains, row):
                    values.add(value)
        schemas[dataset] = (fields, count, domains)
    return schemas


def validate_table(path: Path, schema: tuple) -> None:
    fields, expected_rows, domains = schema
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != fields:
            raise ValueError(f"Column mismatch: {path}")
        count = 0
        for row in reader:
            count += 1
            if len(row) != len(fields):
                raise ValueError(f"Malformed row {count}: {path}")
            if any(value not in domain for value, domain in zip(row, domains)):
                raise ValueError(f"Out-of-domain value in row {count}: {path}")
    if count != expected_rows:
        raise ValueError(f"Row count {count}, expected {expected_rows}: {path}")


def exam_metadata(job: Job, reserved: bool) -> tuple[Path, float]:
    suffix = "u" if reserved else ""
    directory = ROOT / "data" / f"{job.dataset}_eps{job.tag}{suffix}_s{job.seed}"
    exams = sorted(directory.glob("measured_*query.json"))
    if len(exams) != 1:
        raise ValueError(f"Expected exactly one noisy exam: {directory}")
    text = (directory / "README.md").read_text()
    pattern = r"预留 ([0-9.eE+-]+) 给" if reserved else r"计数 sigma ([0-9.eE+-]+)"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Missing budget metadata: {directory}")
    value = float(match.group(1))
    if value <= 0:
        raise ValueError(f"Nonpositive budget/noise metadata: {directory}")
    return exams[0], value


SUCCESS_RECORD = re.compile(r"^(\S+) (?:rc=0(?:\s|$)|adopted=1 .*\bvalidation=ok\b)")


def prepare() -> tuple[list[Job], set[str], dict]:
    schemas = load_schemas()
    done_path = ROOT / "logs_eps" / "eps.done"
    successful = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            match = SUCCESS_RECORD.match(line)
            if match is not None:
                successful.add(match.group(1))
    jobs = [
        Job(dataset, tag, method, seed)
        for dataset in ("nltcs", "adult")
        for tag in ("0p1", "10")
        for method in (("rule", "gsd", "aimfull", "pgm") if dataset == "nltcs"
                       else ("rule", "gsd", "aimfull"))
        for seed in (1, 2, 3)
    ]
    completed = set()
    for job in jobs:
        if job.method in ("rule", "gsd", "pgm"):
            exam_metadata(job, reserved=job.method == "rule")
        if not job.output.exists():
            continue
        validate_table(job.output, schemas[job.dataset])
        legacy_name = f"{job.dataset}_eps{job.tag}_{job.method}_s{job.seed}"
        if job.name not in successful and legacy_name not in successful:
            raise ValueError(f"Existing output has no successful exit record: {job.output}")
        completed.add(job.name)
    return jobs, completed, schemas


def gpu_snapshot(index: int) -> tuple[int, int, list[str]]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index),
         "--query-gpu=uuid,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError(f"Unexpected GPU {index} status: {result.stdout!r}")
    uuid, utilization, memory = (value.strip() for value in rows[0])
    if uuid != GPU_UUIDS[index]:
        raise ValueError(f"GPU {index} UUID changed: {uuid}")
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    pids = [
        row[1].strip()
        for row in csv.reader(apps.stdout.splitlines())
        if len(row) == 2 and row[0].strip() == uuid
    ]
    return int(utilization), int(memory), pids


def gpu_available(index: int, ours: int) -> tuple[bool, str]:
    """ours = 本控制器当前记在该卡上的在跑场数. 为 0 时沿用 v1 严格空闲判据;
    否则只要卡上的计算进程数不超过我们自己的场数就视为无外人 (nvidia-smi 报宿主 PID, 不能逐一比对)."""
    utilization, memory, pids = gpu_snapshot(index)
    if ours == 0:
        ok = utilization == 0 and memory <= 100 and not pids
    else:
        ok = len(pids) <= ours
    return ok, f"gpu={index} utilization={utilization} memory={memory} pids={pids} ours={ours}"


def command_for(job: Job, gpu: int | None, cpu_lane: int | None) -> tuple[list[str], dict]:
    output = STAGING / f"{job.name}.csv"
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    if job.uses_gpu:
        if gpu not in GPU_UUIDS:
            raise ValueError("GPU jobs require physical GPU 0 or 1")
        env["CUDA_VISIBLE_DEVICES"] = GPU_UUIDS[gpu]
        env["CUDA_PATH"] = str(ROOT / ".venv/lib/python3.10/site-packages/nvidia/cuda_runtime")
    else:
        if cpu_lane not in CPU_LANES:
            raise ValueError("CPU jobs require a valid CPU lane")
        env.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                    "PYTHONPATH": "/root/private-pgm/src"})
    if job.method == "rule":
        _, rho = exam_metadata(job, reserved=True)
        tolerance = "0.5" if job.dataset == "adult" and job.tag == "0p1" else "0.05"
        command = [
            str(VENVPY), "-u", "-B", "scripts/run_plants.py",
            "--data", f"{job.dataset}_eps{job.tag}u_s{job.seed}",
            "--rounds", "30000", "--retries", "60", "--batched", "--pairing", "--gpu",
            "--work-rows", "4096", "--work-below", "0.2",
            "--stop-threshold", "1e-4", "--stop-lag", "100",
            "--pair-rescue-rows", "2000", "--rescue-after", "6", "--max-pairs", "128",
            "--rescue-stop-window", "10", "--rescue-stop-tol", "0.02", "--retry-select",
            "--init-margin-tol", tolerance, "--ji-rho", str(rho),
            "--struct-ruler", "ding+ji", "--struct-boost", "4", "--struct-boost2", "8",
            "--donor-rule", "--menu-seed", str(20260920 + job.seed),
            "--sample-seed", str(job.seed), "--init-seed", str(job.seed), "--progress", "200",
            "--save-table", str(output), "--out", str(STAGING / f"{job.name}_curve.csv"),
        ]
        if job.dataset == "adult":
            command += ["--gpu-entry-budget", "40000000"]
    elif job.method == "gsd":
        exam, _ = exam_metadata(job, reserved=False)
        env.update({"JAX_PLATFORMS": "cuda,cpu", "PYTHONPATH": "/root/gsd_patched"})
        command = [
            str(GSDPY), "-u", "-B", "scripts/run_baseline_gsd_plants.py",
            "--seed", str(job.seed), "--data", job.dataset,
            "--noisy-exam", str(exam), "--out", str(output),
        ]
    elif job.method == "aimfull":
        command = [
            str(AIMPY), "-u", "-B", "scripts/run_aim_full.py", "--data", job.dataset,
            "--mechanisms", "/root/private-pgm/mechanisms",
            "--epsilon", "0.1" if job.tag == "0p1" else "10",
            "--delta", "1e-5", "--seed", str(job.seed), "--max-model-size", "80",
            "--clear-caches-every", "1", "--clear-caches-every-projects", "500",
            "--out", str(output),
        ]
    elif job.method == "pgm":
        exam, sigma = exam_metadata(job, reserved=False)
        command = [
            str(AIMPY), "-u", "-B", "scripts/run_baseline_pgm.py",
            "--exam", str(exam), "--sigma", str(sigma), "--seed", str(job.seed),
            "--iters", "1000", "--out", str(output),
        ]
    else:
        raise ValueError(f"Unsupported method: {job.method}")
    if cpu_lane is not None:
        command = ["taskset", "-c", CPU_LANES[cpu_lane]] + command
    return command, env


def bare_command(job: Job) -> tuple:
    """子进程 exec 之后的 argv (taskset 原地 exec 成 python, 前缀消失), 用于在 /proc 里认领."""
    command, _ = command_for(job, 0 if job.uses_gpu else None, None if job.uses_gpu else 0)
    if command[:1] == ["taskset"]:
        command = command[3:]
    return tuple(command)


def proc_read(pid: int, name: str) -> bytes | None:
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None


def process_argv(pid: int) -> tuple | None:
    raw = proc_read(pid, "cmdline")
    if not raw:
        return None
    return tuple(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def process_start_ticks(pid: int) -> str | None:
    raw = proc_read(pid, "stat")
    if not raw:
        return None
    return raw.decode("utf-8", "replace").rsplit(")", 1)[1].split()[19]


def process_alive(orphan: Orphan) -> bool:
    return (process_start_ticks(orphan.pid) == orphan.start_ticks
            and process_argv(orphan.pid) == orphan.argv)


def boot_time() -> float:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    raise RuntimeError("btime missing from /proc/stat")


def scan_orphans(jobs: list[Job], completed: set[str]) -> dict[str, Orphan]:
    """认领上一代控制器留下的在跑进程: argv 逐项等于本控制器会生成的命令才算同一场."""
    expected = {bare_command(job): job for job in jobs if job.name not in completed}
    found: dict[str, Orphan] = {}
    btime = boot_time()
    ticks = os.sysconf("SC_CLK_TCK")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        pid = int(entry.name)
        argv = process_argv(pid)
        job = expected.get(argv) if argv else None
        if job is None:
            continue
        raw_env = proc_read(pid, "environ") or b""
        env = dict(part.decode("utf-8", "replace").split("=", 1)
                   for part in raw_env.split(b"\0") if b"=" in part)
        status = (proc_read(pid, "status") or b"").decode("utf-8", "replace")
        cpus = next((line.split(":", 1)[1].strip() for line in status.splitlines()
                     if line.startswith("Cpus_allowed_list:")), None)
        gpu = lane = None
        if job.uses_gpu:
            uuid = env.get("CUDA_VISIBLE_DEVICES", "")
            matches = [index for index, value in GPU_UUIDS.items() if value == uuid]
            if len(matches) != 1:
                raise RuntimeError(f"Live {job.name} pid={pid} not bound to GPU 0/1: {uuid!r}")
            gpu = matches[0]
        else:
            lane = LEGACY_LANES.get(cpus)
            if lane is None:
                lane = next((index for index, value in CPU_LANES.items() if value == cpus), None)
            if lane is None:
                raise RuntimeError(f"Live {job.name} pid={pid} has unexpected affinity {cpus!r}")
        if job.name in found:
            raise RuntimeError(f"Multiple live processes for {job.name}: {found[job.name].pid}, {pid}")
        start_ticks = process_start_ticks(pid)
        if start_ticks is None:
            continue
        started = datetime.fromtimestamp(btime + int(start_ticks) / ticks, timezone.utc)
        found[job.name] = Orphan(pid, start_ticks, started.isoformat(timespec="seconds"),
                                 gpu, lane, cpus, argv)
    return found


def run(jobs: list[Job], completed: set[str], schemas: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    lock_handle = (LOGS / "controller.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    orphans = scan_orphans(jobs, completed)
    by_name = {job.name: job for job in jobs}
    guard = threading.RLock()
    children: dict[str, subprocess.Popen] = {}
    stop = threading.Event()
    detach = threading.Event()
    retried: set[str] = set()
    records = {job.name: {"state": "succeeded" if job.name in completed else "pending"}
               for job in jobs}
    # 收养的在跑进程先登记为 running, 工作线程启动后查卡时才不会把自己人当外人.
    for name, orphan in orphans.items():
        job = by_name[name]
        slot = ("heavy" if job.heavy else "light") if job.uses_gpu else f"cpu{orphan.cpu_lane}"
        records[name] = {
            "state": "running", "pid": orphan.pid, "adopted": True,
            "started_at": orphan.started_at, "adopted_at": now(), "gpu": orphan.gpu,
            "cpu_lane": orphan.cpu_lane, "cpus": orphan.cpus, "slot": slot,
            "command": list(orphan.argv), "log": str(LOGS / f"{name}.log"),
        }
    state = {
        "controller_pid": os.getpid(), "started_at": now(), "phase": "running",
        "scheduler": SCHEDULER, "budgets": [0.1, 10], "allowed_gpus": [0, 1],
        "cpu_cores": "0-55", "cpu_lanes": CPU_LANES, "initial_completed": len(completed),
        "adopted": sorted(orphans), "jobs": records, "gpu_waiting": {}, "worker_errors": [],
    }

    def save() -> None:
        state["updated_at"] = now()
        state["counts"] = {
            name: sum(record["state"] == name for record in records.values())
            for name in ("pending", "running", "succeeded", "failed")
        }
        temporary = STATUS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(STATUS)

    def running_on_gpu(index: int) -> int:
        with guard:
            return sum(record["state"] == "running" and record.get("gpu") == index
                       for record in records.values())

    def finish(job: Job, returncode: int | None, started: float, gpu, cpu_lane,
               adopted: bool) -> bool:
        error = None
        if returncode == 0 or (adopted and returncode is None):
            try:
                validate_table(STAGING / f"{job.name}.csv", schemas[job.dataset])
                if job.method == "rule":
                    curve = STAGING / f"{job.name}_curve.csv"
                    if not curve.is_file() or curve.stat().st_size == 0:
                        raise ValueError(f"Missing evolution curve: {curve}")
            except (ValueError, csv.Error, UnicodeError, FileNotFoundError) as exc:
                error = str(exc)
        else:
            error = f"Process exited with code {returncode}"
        if error is None and job.output.exists():
            error = f"Refusing to replace an existing output: {job.output}"
        with guard:
            children.pop(job.name, None)
            record = records[job.name]
            record.update({"finished_at": now(), "elapsed_seconds": time.monotonic() - started,
                           "returncode": returncode})
            tag = f"adopted=1 rc={'unknown' if returncode is None else returncode}" if adopted \
                else f"rc={returncode}"
            if error is None:
                (STAGING / f"{job.name}.csv").replace(job.output)
                if job.method == "rule":
                    (STAGING / f"{job.name}_curve.csv").replace(
                        ROOT / "results" / f"{job.name}_curve.csv")
                record["state"] = "succeeded"
                line = f"{job.name} {tag} validation=ok gpu={gpu} cpu_lane={cpu_lane}"
            else:
                record["state"] = "failed"
                record["error"] = error
                line = f"{job.name} {tag} validation=failed gpu={gpu} cpu_lane={cpu_lane}"
            # 账本先落盘, 再打印, 避免 v1 那种 stdout 断管吞掉 eps.done 记录.
            with (ROOT / "logs_eps" / "eps.done").open("a") as handle:
                handle.write(line + "\n")
            save()
        if error is None:
            emit(f"DONE {job.name} elapsed={time.monotonic() - started:.1f}s adopted={adopted}")
            return True
        emit(f"FAILED {job.name} error={error} log={LOGS / f'{job.name}.log'}")
        try:
            tail = (LOGS / f"{job.name}.log").read_text(errors="replace").splitlines()[-25:]
            emit("\n".join(tail))
        except OSError:
            pass
        return False

    def execute(job: Job, gpu: int | None = None, cpu_lane: int | None = None,
                slot: str | None = None) -> bool | None:
        """True 成功, False 失败, None 表示控制器脱离 (子进程留跑)."""
        command, overrides = command_for(job, gpu, cpu_lane)
        environment = os.environ.copy()
        environment.update(overrides)
        logfile = LOGS / f"{job.name}.log"
        started = time.monotonic()
        with logfile.open("a") as handle:
            handle.write(f"\n{now()} COMMAND {json.dumps(command)}\n")
            handle.write(f"ENV {json.dumps(overrides, sort_keys=True)}\n")
            handle.flush()
            with guard:
                if stop.is_set():
                    return None
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=subprocess.STDOUT)
                children[job.name] = process
                records[job.name] = {
                    "state": "running", "pid": process.pid, "started_at": now(),
                    "gpu": gpu, "cpu_lane": cpu_lane, "slot": slot, "command": command,
                    "environment": overrides, "log": str(logfile),
                }
                save()
            emit(f"START {job.name} pid={process.pid} gpu={gpu} cpu_lane={cpu_lane} slot={slot}")
            while process.poll() is None:
                if detach.is_set():
                    emit(f"DETACH {job.name} pid={process.pid} left running")
                    return None
                time.sleep(2)
        return finish(job, process.returncode, started, gpu, cpu_lane, adopted=False)

    def adopt(job: Job, orphan: Orphan) -> bool | None:
        started = time.monotonic()
        record = records[job.name]
        emit(f"ADOPT {job.name} pid={orphan.pid} gpu={orphan.gpu} cpu_lane={orphan.cpu_lane} "
             f"cpus={orphan.cpus} started_at={orphan.started_at} slot={record.get('slot')}")
        while process_alive(orphan):
            if detach.is_set():
                emit(f"DETACH {job.name} pid={orphan.pid} left running")
                return None
            time.sleep(2)
        return finish(job, None, started, orphan.gpu, orphan.cpu_lane, adopted=True)

    pending = [job for job in jobs if job.name not in completed and job.name not in orphans]
    heavy_jobs: queue.Queue = queue.Queue()
    light_jobs: queue.Queue = queue.Queue()
    cpu_jobs: queue.Queue = queue.Queue()
    # 长场优先 (LPT): ε=10 的 adult 我们场最长, 先占重槽; ε=0.1 的 adult 实测约 13 分钟收尾填空.
    heavy_pending = sorted((job for job in pending if job.uses_gpu and job.heavy),
                           key=lambda job: (job.dataset != "adult", job.tag != "10", job.seed))
    light_pending = sorted((job for job in pending if job.uses_gpu and not job.heavy),
                           key=lambda job: (job.dataset != "adult", job.tag != "10", job.seed))
    cpu_pending = sorted((job for job in pending if not job.uses_gpu),
                         key=lambda job: (job.method != "pgm", job.dataset != "adult",
                                          job.tag != "10", job.seed))
    for job in heavy_pending:
        heavy_jobs.put(job)
    for job in light_pending:
        light_jobs.put(job)
    for job in cpu_pending:
        cpu_jobs.put(job)

    def take(*queues: queue.Queue) -> Job | None:
        for candidate in queues:
            try:
                return candidate.get_nowait()
            except queue.Empty:
                continue
        return None

    def adopt_or_requeue(job: Job, orphan: Orphan) -> None:
        # 收养的孤儿进程若未留下合格表 (多半是被外力中止), 允许原位重跑一次;
        # 本控制器自己启动的场失败不重试, 口径同 v1.
        outcome = adopt(job, orphan)
        if outcome is not False or job.name in retried:
            return
        retried.add(job.name)
        emit(f"REQUEUE {job.name} after adopted process failed validation")
        (heavy_jobs if job.heavy else light_jobs if job.uses_gpu else cpu_jobs).put(job)

    def wait_for_gpu(index: int, label: str) -> bool:
        while not stop.is_set():
            ok, detail = gpu_available(index, running_on_gpu(index))
            if ok:
                with guard:
                    state["gpu_waiting"].pop(label, None)
                    save()
                return True
            with guard:
                if label not in state["gpu_waiting"]:
                    emit(f"WAIT {label} {detail}")
                state["gpu_waiting"][label] = detail
                save()
            stop.wait(30)
        return False

    def gpu_worker(index: int, slot: str) -> None:
        label = f"gpu{index}-{slot}"
        for name, orphan in orphans.items():
            job = by_name[name]
            if orphan.gpu == index and job.heavy == (slot == "heavy"):
                adopt_or_requeue(job, orphan)
        while not stop.is_set():
            job = take(heavy_jobs, light_jobs) if slot == "heavy" else take(light_jobs)
            if job is None:
                return
            if not wait_for_gpu(index, label):
                return
            execute(job, gpu=index, slot=slot)

    def cpu_worker(lane: int) -> None:
        for name, orphan in orphans.items():
            if orphan.cpu_lane == lane:
                adopt_or_requeue(by_name[name], orphan)
        while not stop.is_set():
            job = take(cpu_jobs)
            if job is None:
                return
            execute(job, cpu_lane=lane, slot=f"cpu{lane}")

    def guarded(target, *args) -> None:
        try:
            target(*args)
        except Exception as exc:  # 单个工作线程出错不拖垮全批
            emit(f"WORKER_ERROR {target.__name__}{args}: {exc!r}")
            with guard:
                state["worker_errors"].append(f"{now()} {target.__name__}{args}: {exc!r}")
                save()

    def interrupted(signum, frame) -> None:
        raise KeyboardInterrupt(f"Controller received signal {signum}")

    def detach_requested(signum, frame) -> None:
        raise Detach(f"Controller received detach signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGUSR1, detach_requested)
    with guard:
        save()
    emit(f"PLAN total={len(jobs)} existing={len(completed)} adopted={len(orphans)} "
         f"pending_heavy={len(heavy_pending)} pending_light={len(light_pending)} "
         f"pending_cpu={len(cpu_pending)} budgets=0.1,10 scheduler={SCHEDULER!r}")
    pool = ThreadPoolExecutor(max_workers=8)
    futures = [pool.submit(guarded, gpu_worker, index, slot)
               for index in (0, 1) for slot in ("heavy", "light")]
    futures += [pool.submit(guarded, cpu_worker, lane) for lane in CPU_LANES]
    detaching = False
    try:
        for future in as_completed(futures):
            future.result()
    except Detach as exc:
        detaching = True
        emit(f"DETACH requested: {exc}; leaving running jobs alive for the next controller")
    finally:
        stop.set()
        if detaching:
            detach.set()
        else:
            with guard:
                running = list(children.values())
                adopted_pids = [record["pid"] for record in records.values()
                                if record["state"] == "running" and record.get("adopted")]
            for process in running:
                if process.poll() is None:
                    process.terminate()
            for pid in adopted_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for process in running:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        pool.shutdown(wait=True)
        with guard:
            if detaching:
                state["phase"] = "detached"
            else:
                state["phase"] = "finished" if all(record["state"] == "succeeded"
                                                 for record in records.values()) else "incomplete"
            state["finished_at"] = now()
            save()
        lock_handle.close()
    if detaching:
        return
    failed = [name for name, record in records.items() if record["state"] != "succeeded"]
    if failed:
        raise RuntimeError(f"Endpoint jobs not completed: {failed}")
    emit("ALL_ENDPOINTS_COMPLETE completed=42 budgets=0.1,10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    jobs, completed, schemas = prepare()
    if args.inspect:
        orphans = scan_orphans(jobs, completed)
        print(json.dumps({
            "total": len(jobs), "completed": sorted(completed),
            "adoptable": {name: {"pid": orphan.pid, "gpu": orphan.gpu, "cpu_lane": orphan.cpu_lane,
                                 "cpus": orphan.cpus, "started_at": orphan.started_at}
                          for name, orphan in sorted(orphans.items())},
            "pending": [
                {"name": job.name,
                 "resource": ("gpu-heavy" if job.heavy else "gpu-light") if job.uses_gpu else "cpu",
                 "command": command_for(job, 0 if job.uses_gpu else None,
                                        None if job.uses_gpu else 0)[0]}
                for job in jobs if job.name not in completed and job.name not in orphans
            ],
        }, indent=2, ensure_ascii=False))
        return
    run(jobs, completed, schemas)


if __name__ == "__main__":
    main()
