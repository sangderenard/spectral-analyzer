"""Develop multiple spectral scene-order jobs through need-weighted ray packets.

One GPU-owning renderer subprocess is active at a time. After every bounded
recursive-sensor epoch burst, its accumulation is checkpointed and the global
scheduler reconsiders every unfinished scene. This gives N logical jobs smooth
progress without multiplying resident GL contexts and SSBO arenas.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from camera_software.progressive_exposure import ExposureProgressEvent
from camera_software.render_work_scheduler import (
    NeedWeightedPacketScheduler,
    RenderPacketJob,
)
from scene_orders import (
    composition_metadata,
    load_order,
    order_runtime_settings,
    resolved_jobs,
)


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", help="schema_version=1 scene-order JSON")
    parser.add_argument("--job", action="append", default=[],
                        help="Job id to develop; repeatable. Defaults to all jobs.")
    parser.add_argument("--out-dir", default="exposures/scene_packets")
    parser.add_argument("--state", default="",
                        help="Scheduler checkpoint JSON; defaults under out-dir.")
    parser.add_argument("--packet-epochs", type=int, default=1,
                        help="Recursive sensor epochs granted per scheduling quantum.")
    parser.add_argument("--epochs-per-job", type=int, default=0,
                        help="Target epochs per job; 0 uses exposure.sensor_sweeps.")
    parser.add_argument("--max-packets", type=int, default=0,
                        help="Stop after this many packets in this invocation; 0 finishes.")
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true",
                        help="Discard an existing scheduler checkpoint.")
    parser.add_argument("--no-vcm", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the next allocation without launching the renderer.")
    return parser.parse_args(argv)


def _selected(package: dict, ids: list[str]) -> list[dict]:
    jobs = resolved_jobs(package)
    if not ids:
        return jobs
    wanted = set(map(str, ids))
    selected = [job for job in jobs if str(job["id"]) in wanted]
    missing = sorted(wanted - {str(job["id"]) for job in selected})
    if missing:
        raise ValueError(f"unknown requested job ids: {missing}")
    return selected


def _display_raster_to_native_square(array: np.ndarray) -> np.ndarray:
    """Invert the renderer's square-native to rectangular-display mapping."""

    display = np.asarray(array)
    if display.ndim not in (2, 3):
        raise ValueError("display sensor raster must be HxW or HxWxC")
    height, width = display.shape[:2]
    resolution = max(width, height)
    source_y = np.minimum(
        ((np.arange(resolution) + 0.5) * height / resolution).astype(np.int64),
        height - 1,
    )
    source_x = np.minimum(
        ((np.arange(resolution) + 0.5) * width / resolution).astype(np.int64),
        width - 1,
    )
    square = display[source_y[:, None], source_x[None, :]]
    return np.ascontiguousarray(np.flip(np.swapaxes(square, 0, 1), axis=0))


def _prepare_restore(job: RenderPacketJob, job_dir: str) -> tuple[str, str, str] | None:
    if not job.sensor_sum_path or not job.exposure_weight_path:
        return None
    if not os.path.isfile(job.sensor_sum_path) or not os.path.isfile(
        job.exposure_weight_path
    ):
        return None
    restore_sum = os.path.join(job_dir, "restore_sensor_sum_native.npy")
    restore_weight = os.path.join(job_dir, "restore_sensor_weight_native.npy")
    dirty = os.path.join(job_dir, "restore_sensor_dirty_native.npy")
    np.save(
        restore_sum,
        _display_raster_to_native_square(
            np.load(job.sensor_sum_path, allow_pickle=False)
        ),
        allow_pickle=False,
    )
    np.save(
        restore_weight,
        _display_raster_to_native_square(
            np.load(job.exposure_weight_path, allow_pickle=False)
        ),
        allow_pickle=False,
    )
    np.save(dirty, np.empty(0, np.uint32), allow_pickle=False)
    return restore_sum, restore_weight, dirty


def _target_packets(order_job: dict, override: int) -> int:
    if int(override) > 0:
        return int(override)
    return max(1, int(order_runtime_settings(order_job)["sensor_sweeps"]))


def _job_weight(order_job: dict) -> float:
    exposure = dict(order_job.get("exposure", {}))
    return float(exposure.get("scheduler_weight", exposure.get("work_weight", 1.0)))


def _new_scheduler(jobs: list[dict], epochs_per_job: int) -> NeedWeightedPacketScheduler:
    return NeedWeightedPacketScheduler(
        RenderPacketJob(
            job_id=str(job["id"]),
            target_packets=_target_packets(job, epochs_per_job),
            weight=_job_weight(job),
            metadata={"token": str(job.get("token", ""))},
        )
        for job in jobs
    )


def _load_or_create_scheduler(
    state_path: str,
    jobs: list[dict],
    epochs_per_job: int,
    *,
    resume: bool,
) -> NeedWeightedPacketScheduler:
    expected = [str(job["id"]) for job in jobs]
    if resume and os.path.isfile(state_path):
        scheduler = NeedWeightedPacketScheduler.load(state_path)
        if list(scheduler.order) != expected:
            raise ValueError(
                "scheduler checkpoint job order differs from the selected scene jobs"
            )
        for order_job in jobs:
            state = scheduler.jobs[str(order_job["id"])]
            target = _target_packets(order_job, epochs_per_job)
            if state.target_packets != target:
                raise ValueError(
                    f"scheduler target changed for {state.job_id!r}: "
                    f"{state.target_packets} != {target}; use --no-resume"
                )
        return scheduler
    return _new_scheduler(jobs, epochs_per_job)


def _write_job_manifests(
    order_path: str, out_dir: str, jobs: list[dict]
) -> dict[str, str]:
    directories = {}
    for job in jobs:
        job_id = str(job["id"])
        job_dir = os.path.abspath(os.path.join(out_dir, job_id))
        directories[job_id] = job_dir
        os.makedirs(job_dir, exist_ok=True)
        with open(
            os.path.join(job_dir, "resolved_scene_order.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump({
                "source_order": order_path,
                "resolved_job": job,
                "runtime": order_runtime_settings(job),
            }, handle, indent=2)
        with open(
            os.path.join(job_dir, "composition_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(composition_metadata(job), handle, indent=2)
    return directories


def _run_packet(
    *,
    order_path: str,
    state: RenderPacketJob,
    job_dir: str,
    packet_epochs: int,
    no_vcm: bool,
) -> ExposureProgressEvent:
    progress_dir = os.path.join(job_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    command = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "exposure_render_demo.py"),
        "--scene-order", order_path,
        "--scene-job", state.job_id,
        "--integrator", "bdpt",
        "--backend", "cpp",
        "--frames", "1",
        "--gpu-resident",
        "--no-window",
        "--save-files",
        "--out-dir", job_dir,
        "--progress-dir", progress_dir,
        "--progress-exposure-id", state.job_id,
        "--bdpt-native-packages", "1",
        "--no-convergence-drive-batches",
    ]
    if no_vcm:
        command.append("--no-vcm")

    environment = os.environ.copy()
    environment["SPECTRAL_SENSOR_MAX_EPOCHS"] = str(packet_epochs)
    environment["SPECTRAL_SENSOR_PERSISTENT_EPOCHS"] = "1"
    environment["SPECTRAL_SENSOR_STEPS_PER_LAYER"] = "1"
    environment["SPECTRAL_PROGRESS_RETAIN_LAYERS"] = "2"
    environment["SPECTRAL_EXPOSURE_SEED_OFFSET"] = str(
        state.completed_packets * 1_000_003
    )
    restore = _prepare_restore(state, job_dir)
    if restore is not None:
        environment["SPECTRAL_SENSOR_RESTORE_SUM"] = restore[0]
        environment["SPECTRAL_SENSOR_RESTORE_WEIGHT"] = restore[1]
        environment["SPECTRAL_SENSOR_DIRTY_SITES"] = restore[2]

    latest: ExposureProgressEvent | None = None
    process = subprocess.Popen(
        command,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{state.job_id}] {line}", end="", flush=True)
        event = ExposureProgressEvent.from_line(line.rstrip("\r\n"))
        if event is not None:
            latest = event
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if latest is None:
        raise RuntimeError(
            f"job {state.job_id!r} completed a packet without a progress checkpoint"
        )
    if not latest.sensor_sum_path or not latest.exposure_weight_path:
        raise RuntimeError(
            f"job {state.job_id!r} did not publish resumable sensor evidence"
        )
    return latest


def main(argv: list[str] | None = None) -> int:
    args = _args(sys.argv[1:] if argv is None else argv)
    if args.packet_epochs <= 0:
        raise ValueError("--packet-epochs must be positive")
    if args.max_failures <= 0:
        raise ValueError("--max-failures must be positive")
    order_path = os.path.abspath(args.order)
    jobs = _selected(load_order(order_path), args.job)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.abspath(
        args.state or os.path.join(out_dir, "holistic_scheduler.json")
    )
    directories = _write_job_manifests(order_path, out_dir, jobs)
    scheduler = _load_or_create_scheduler(
        state_path,
        jobs,
        args.epochs_per_job,
        resume=not args.no_resume,
    )

    packets_run = 0
    while not scheduler.done:
        if args.max_packets > 0 and packets_run >= args.max_packets:
            break
        state = scheduler.next_job()
        if state is None:
            break
        packet_epochs = min(args.packet_epochs, state.remaining_packets)
        print(
            f"SCHEDULE job={state.job_id!r} packet={packet_epochs} "
            f"need={state.effective_need():.4f} "
            f"developed={state.completed_packets}/{state.target_packets}",
            flush=True,
        )
        if args.dry_run:
            scheduler.save(state_path)
            return 0
        try:
            event = _run_packet(
                order_path=order_path,
                state=state,
                job_dir=directories[state.job_id],
                packet_epochs=packet_epochs,
                no_vcm=bool(args.no_vcm),
            )
        except Exception as exc:
            failed = scheduler.fail_packet(state.job_id, f"{type(exc).__name__}: {exc}")
            scheduler.save(state_path)
            if failed.failures >= args.max_failures:
                raise
            continue
        scheduler.complete_packet(
            state.job_id,
            packets=packet_epochs,
            priority_map_path=event.priority_map_path,
            sensor_sum_path=event.sensor_sum_path,
            exposure_weight_path=event.exposure_weight_path,
            linear_accumulation_path=event.linear_accumulation_path,
            message=event.message,
        )
        scheduler.save(state_path)
        packets_run += 1
        print(
            f"CHECKPOINT job={state.job_id!r} "
            f"developed={state.completed_packets}/{state.target_packets} "
            f"remaining_jobs={sum(not job.done for job in scheduler.jobs.values())}",
            flush=True,
        )

    scheduler.save(state_path)
    print(
        f"SCHEDULER {'COMPLETE' if scheduler.done else 'PAUSED'} "
        f"packets_this_run={packets_run} state={state_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
