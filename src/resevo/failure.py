"""数值中断的诊断落盘；不发布正常终表，也不把失败转换为成功。"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from .dataset import StateRegistry
from .editspace import tuples_from_ids
from .engine import NumericalEvolutionError, ProbeRecord, RoundRecord
from .sampling import SAMPLING_CONTRACT, SamplingPrecision


def write_numerical_failure(
    registry: StateRegistry,
    error: NumericalEvolutionError,
    output: str | Path | None = None,
    *,
    stage: str = "generation",
) -> Path:
    """保存当前合成表和已有曲线；唯一目录避免覆盖此前失败，manifest 最后写。"""
    result = error.partial_result
    if result.stop_reason != "numerical_indeterminate":
        raise ValueError("失败快照必须携带 numerical_indeterminate 状态")
    base = Path(output) if output is not None else Path("outputs") / "resevo"
    rows = tuples_from_ids(registry, result.state_ids)
    base.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{base.name[:40]}.failed-", dir=base.parent))
    hashes = {}

    def save_csv(name, header, values):
        path = directory / name
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(values)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save_records(name, records, record_type):
        names = [field.name for field in fields(record_type)]
        save_csv(name, names, ([getattr(record, key) for key in names] for record in records))

    save_csv("current.csv", registry.schema.fields, rows)
    save_records("records.csv", result.records, RoundRecord)
    if result.probes is not None:
        save_records("probes.csv", result.probes, ProbeRecord)
    if result.sampling_precision:
        save_records("sampling_precision.csv", result.sampling_precision, SamplingPrecision)
    if result.temps is not None:
        save_csv("temperatures.csv", ["round", "temperature"], enumerate(result.temps))
    if result.group_losses:
        size = len(result.group_losses[0][1])
        save_csv(
            "group_losses.csv", ["round"] + [f"g{i}" for i in range(size)],
            ([k, *losses] for k, losses in result.group_losses),
        )
    manifest = {
        "format_version": 1,
        "status": "failed",
        "stop_reason": result.stop_reason,
        "error": str(error),
        "stage": stage,
        "rows": len(rows),
        "completed_rounds": len(result.records),
        "checkpoint_phase": "before_failed_operation",
        "normal_output_published": False,
        "resumable": False,
        "sampling_contract": SAMPLING_CONTRACT,
        "files_sha256": hashes,
    }
    manifest_path = directory / "failure.json.tmp"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest_path.rename(directory / "failure.json")
    return directory


@contextmanager
def save_numerical_failure(
    registry: StateRegistry, output: str | Path | None = None, *, stage: str = "generation",
):
    """驱动共用的失败出口；保存失败也继续抛错，不打印成功提示。"""
    try:
        yield
    except NumericalEvolutionError as exc:
        directory = write_numerical_failure(registry, exc, output, stage=stage)
        print(f"数值中断，失败诊断已保存：{directory}；未发布正常终表", file=sys.stderr)
        raise
