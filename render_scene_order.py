"""Validate, compile, or render a declarative spectral scene-order package."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np

from scene_orders import (
    compile_job,
    composition_metadata,
    load_order,
    order_runtime_settings,
    resolved_jobs,
)


def _args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", help="Path to a schema_version=1 scene-order JSON file")
    p.add_argument("--job", action="append", default=[],
                   help="Job id to process; repeatable. Omit to process every job.")
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true", help="Validate schema only")
    action.add_argument("--compile-only", action="store_true",
                        help="Compile geometry/materials and save manifests/NPZ meshes")
    action.add_argument("--render", action="store_true",
                        help="Render every selected job through native thick-lens BDPT")
    p.add_argument("--out-dir", default="exposures/scene_orders")
    p.add_argument("--no-vcm", action="store_true")
    return p.parse_args(argv)


def _selected(package: dict, ids: list[str]) -> list[dict]:
    jobs = resolved_jobs(package)
    if not ids:
        return jobs
    wanted = set(ids)
    selected = [job for job in jobs if str(job["id"]) in wanted]
    missing = sorted(wanted - {str(job["id"]) for job in selected})
    if missing:
        raise ValueError(f"unknown requested job ids: {missing}")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = _args(sys.argv[1:] if argv is None else argv)
    order_path = os.path.abspath(args.order)
    package = load_order(order_path)
    jobs = _selected(package, args.job)
    if args.validate:
        for job in jobs:
            print(f"VALID job={job['id']!r} token={job['token']!r} runtime={order_runtime_settings(job)}")
        return 0

    if args.compile_only:
        import exposure_render_demo as exposure
        base = exposure._build_thick_lens_lab_tracer_scene()
        for job in jobs:
            scene, report = compile_job(base, job)
            job_dir = os.path.join(args.out_dir, str(job["id"]))
            os.makedirs(job_dir, exist_ok=True)
            manifest = {
                "order": order_path,
                "runtime": order_runtime_settings(job),
                "compile": dataclasses.asdict(report),
                "bounds_min": np.asarray(scene.bounds_min).tolist(),
                "bounds_max": np.asarray(scene.bounds_max).tolist(),
                "source_triangles": int(scene.src_tri_idx.size),
                "camera_groups": {k: int(v.size) for k, v in scene.camera_tri_groups.items()},
                "composition": composition_metadata(job),
            }
            with open(os.path.join(job_dir, "compile_manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            np.savez_compressed(
                os.path.join(job_dir, "compiled_scene.npz"),
                verts=scene.verts, normals=scene.normals, mat_idx=scene.mat_idx,
                mat_buf=scene.mat_buf, src_tri_idx=scene.src_tri_idx,
            )
            print(f"COMPILED job={job['id']!r} -> {job_dir}")
        return 0

    for job in jobs:
        job_id = str(job["id"])
        job_dir = os.path.abspath(os.path.join(args.out_dir, job_id))
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "resolved_scene_order.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "source_order": order_path,
                "resolved_job": job,
                "runtime": order_runtime_settings(job),
            }, fh, indent=2)
        with open(os.path.join(job_dir, "composition_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(composition_metadata(job), fh, indent=2)
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "exposure_render_demo.py"),
            "--scene-order", order_path, "--scene-job", job_id,
            "--integrator", "bdpt", "--backend", "cpp", "--frames", "1",
            "--gpu-resident", "--no-window", "--save-files",
            "--out-dir", job_dir, "--bdpt-native-packages", "1",
        ]
        if args.no_vcm:
            cmd.append("--no-vcm")
        print(f"RENDER job={job_id!r} -> {job_dir}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
