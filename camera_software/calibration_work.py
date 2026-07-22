"""Durable catalog and filesystem homes for calibration-room work."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping


CALIBRATION_WORK_SCHEMA_VERSION = 1


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def calibration_work_key(mode_key: str, manifest: Mapping[str, Any]) -> str:
    """Stable identity for a calibration configuration, not for one run."""

    identity = {
        "mode": str(mode_key),
        "parameters": dict(manifest.get("parameters", {})),
        "render_product": dict(manifest.get("render_product", {})),
        "resolved_ray_trace_settings": dict(
            manifest.get("resolved_ray_trace_settings", {})
        ),
        "exposure_control_settings": dict(
            manifest.get("exposure_control_settings", {})
        ),
        "wave_solver": dict(manifest.get("wave_solver", {})),
    }
    digest = hashlib.sha256(
        json.dumps(_canonical(identity), separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"calibration-{mode_key}-{digest}"


@dataclass
class CalibrationWorkRecord:
    work_key: str
    mode_key: str
    display_name: str
    work_dir: str
    manifest_path: str
    status: str = "queued"
    launch_count: int = 0
    latest_sequence: int = 0
    completed_runs: int = 0
    preview_path: str = ""
    linear_path: str = ""
    result_manifest_path: str = ""
    error: str = ""
    updated_at_s: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)


class CalibrationWorkCatalog:
    """Atomic catalog whose records are shown beneath Work / Assets."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.path = os.path.join(self.root, "catalog.json")
        self._lock = threading.RLock()
        self._records: dict[str, CalibrationWorkRecord] = {}
        os.makedirs(self.root, exist_ok=True)
        if os.path.isfile(self.path):
            self._load()
            changed = False
            for record in self._records.values():
                if record.status == "working":
                    record.status = "checkpointed"
                    changed = True
            if changed:
                self._save()

    def begin(
        self,
        mode_key: str,
        display_name: str,
        manifest: Mapping[str, Any],
        sequence: int,
    ) -> CalibrationWorkRecord:
        key = calibration_work_key(mode_key, manifest)
        work_dir = os.path.join(self.root, key)
        os.makedirs(work_dir, exist_ok=True)
        manifest_path = os.path.join(work_dir, "work_manifest.json")
        with self._lock:
            record = self._records.get(key) or CalibrationWorkRecord(
                work_key=key,
                mode_key=str(mode_key),
                display_name=str(display_name),
                work_dir=work_dir,
                manifest_path=manifest_path,
            )
            record.status = "working"
            record.launch_count += 1
            record.latest_sequence = int(sequence)
            record.error = ""
            record.updated_at_s = time.time()
            record.history.append({
                "sequence": int(sequence), "status": "working",
                "started_at_s": record.updated_at_s,
            })
            self._records[key] = record
            self._write_json(manifest_path, {
                "schema_version": CALIBRATION_WORK_SCHEMA_VERSION,
                "work_key": key,
                "resume_policy": "continue-compatible-retained-evidence",
                "calibration": _canonical(dict(manifest)),
            })
            self._save()
            return record

    def complete(
        self,
        work_key: str,
        sequence: int,
        *,
        preview_path: str,
        linear_path: str,
        result_manifest_path: str,
        elapsed_s: float,
    ) -> CalibrationWorkRecord:
        with self._lock:
            record = self._records[str(work_key)]
            record.status = "complete"
            record.completed_runs += 1
            record.latest_sequence = int(sequence)
            record.preview_path = os.path.abspath(preview_path) if preview_path else ""
            record.linear_path = os.path.abspath(linear_path) if linear_path else ""
            record.result_manifest_path = (
                os.path.abspath(result_manifest_path) if result_manifest_path else ""
            )
            record.updated_at_s = time.time()
            for event in reversed(record.history):
                if int(event.get("sequence", -1)) == int(sequence):
                    event.update({
                        "status": "complete", "elapsed_s": float(elapsed_s),
                        "preview_path": record.preview_path,
                        "linear_path": record.linear_path,
                        "result_manifest_path": record.result_manifest_path,
                        "completed_at_s": record.updated_at_s,
                    })
                    break
            self._save()
            return record

    def fail(self, work_key: str, sequence: int, error: str) -> None:
        with self._lock:
            record = self._records.get(str(work_key))
            if record is None:
                return
            record.status = "failed"
            record.error = str(error)
            record.updated_at_s = time.time()
            for event in reversed(record.history):
                if int(event.get("sequence", -1)) == int(sequence):
                    event.update({"status": "failed", "error": str(error)})
                    break
            self._save()

    def checkpoint(self, work_key: str, sequence: int) -> None:
        """Mark interrupted work resumable without discarding its evidence."""
        with self._lock:
            record = self._records.get(str(work_key))
            if record is None:
                return
            record.status = "checkpointed"
            record.updated_at_s = time.time()
            for event in reversed(record.history):
                if int(event.get("sequence", -1)) == int(sequence):
                    event.update({"status": "checkpointed"})
                    break
            self._save()

    def snapshot(self) -> tuple[CalibrationWorkRecord, ...]:
        with self._lock:
            return tuple(
                self._records[key] for key in sorted(
                    self._records,
                    key=lambda item: self._records[item].updated_at_s,
                    reverse=True,
                )
            )

    @staticmethod
    def _write_json(path: str, payload: Mapping[str, Any]) -> None:
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _save(self) -> None:
        self._write_json(self.path, {
            "schema_version": CALIBRATION_WORK_SCHEMA_VERSION,
            "records": [asdict(record) for record in self.snapshot()],
        })

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("schema_version", -1)) != CALIBRATION_WORK_SCHEMA_VERSION:
            raise ValueError("unsupported calibration work catalog schema")
        for item in payload.get("records", ()):
            record = CalibrationWorkRecord(**dict(item))
            self._records[record.work_key] = record


__all__ = [
    "CalibrationWorkCatalog", "CalibrationWorkRecord", "calibration_work_key",
]
