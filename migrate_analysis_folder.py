#!/usr/bin/env python3
"""
migrate_analysis_folder.py — one-shot migration for existing analysis folders.

What it does
------------
1. Moves .tmp_subbands/band_NNNNN.npy  ->  filterbank/band_NN.npy
   (the temp memmap files ARE the data; no recomputation needed)
2. Rewrites filterbank_meta.json so every "file" entry references the new
   band_NN.npy instead of band_NN.wav / band_NN.flac
3. Removes any leftover band_NN.wav / band_NN.flac files (user already
   deleted them, but if any remain they are cleaned up)
4. Writes an itinerary_<hash>.json to mark cqt_saved so the analysis tool
   knows CQT shards are already present and will not re-run the transform
5. Prints a summary of every action taken

Usage
-----
    python migrate_analysis_folder.py <analysis_dir>

Safe to re-run: already-moved files are detected and skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import sys


def _band_idx_from_meta_filename(fname: str) -> int | None:
    """Return 0-based band index from metadata 'file' value like 'band_03.wav'."""
    base = os.path.splitext(os.path.basename(fname))[0]  # band_03
    parts = base.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


def migrate(analysis_dir: str, dry_run: bool = False) -> None:
    analysis_dir = os.path.abspath(analysis_dir)
    if not os.path.isdir(analysis_dir):
        print(f"ERROR: Not a directory: {analysis_dir}")
        sys.exit(1)

    fb_dir = os.path.join(analysis_dir, "filterbank")
    meta_path = os.path.join(fb_dir, "filterbank_meta.json")
    tmp_sub_dir = os.path.join(fb_dir, ".tmp_subbands")
    tag = "[DRY-RUN] " if dry_run else ""

    print(f"Migrating: {analysis_dir}")
    print(f"  Filter bank dir : {fb_dir}")

    # ── 1. Move temp subbands to permanent NPY shards ────────────────────
    moved = 0
    skipped = 0
    if os.path.isdir(tmp_sub_dir):
        tmp_files = sorted(
            f for f in os.listdir(tmp_sub_dir) if f.endswith(".npy"))
        print(f"  Temp subbands   : {len(tmp_files)} files in .tmp_subbands/")
        for fname in tmp_files:
            # fname like band_00042.npy  ->  band_42.npy (zero-pad to 2 digits)
            base = os.path.splitext(fname)[0]  # band_00042
            try:
                idx = int(base.split("_")[-1])
            except ValueError:
                print(f"    SKIP (unrecognised name): {fname}")
                continue
            dst_name = f"band_{idx:02d}.npy"
            src = os.path.join(tmp_sub_dir, fname)
            dst = os.path.join(fb_dir, dst_name)
            if os.path.isfile(dst):
                skipped += 1
                continue
            print(f"  {tag}MOVE  {fname}  ->  {dst_name}")
            if not dry_run:
                shutil.move(src, dst)
            moved += 1

        # Remove the now-empty temp dir.
        if not dry_run and os.path.isdir(tmp_sub_dir) and not os.listdir(tmp_sub_dir):
            os.rmdir(tmp_sub_dir)
            print("  REMOVED empty .tmp_subbands/")
    else:
        print("  Temp subbands   : no .tmp_subbands/ directory found")

    print(f"  Band files moved : {moved}  (skipped already-present: {skipped})")

    # ── 2. Update filterbank_meta.json ───────────────────────────────────
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        changed = 0
        for bm in meta.get("bands", []):
            old_file = bm.get("file", "")
            idx = _band_idx_from_meta_filename(old_file)
            if idx is None:
                continue
            new_file = f"band_{idx:02d}.npy"
            if old_file != new_file:
                bm["file"] = new_file
                bm["format"] = "npy"
                changed += 1

        if changed:
            print(f"  {tag}Updating filterbank_meta.json ({changed} entries)")
            if not dry_run:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        else:
            print("  filterbank_meta.json already up to date")
    else:
        print("  filterbank_meta.json not found — skipping metadata update")

    # ── 3. Remove leftover WAV / FLAC band files ─────────────────────────
    removed_audio = 0
    if os.path.isdir(fb_dir):
        for fname in os.listdir(fb_dir):
            if fname.startswith("band_") and (
                fname.endswith(".wav") or fname.endswith(".flac")
            ):
                fpath = os.path.join(fb_dir, fname)
                print(f"  {tag}DELETE legacy audio: {fname}")
                if not dry_run:
                    os.remove(fpath)
                removed_audio += 1
    print(f"  Legacy audio files removed: {removed_audio}")

    # ── 4. Write itinerary so CQT is not re-run ──────────────────────────
    # Find the settings hash from cqt_data_XXXXXXXX.stream directories.
    hashes: list[str] = []
    for entry in os.listdir(analysis_dir):
        if entry.startswith("cqt_data_") and entry.endswith(".stream"):
            h = entry[len("cqt_data_"):-len(".stream")]
            if len(h) == 8:
                hashes.append(h)

    for h in hashes:
        itin_path = os.path.join(analysis_dir, f"itinerary_{h}.json")
        if os.path.isfile(itin_path):
            with open(itin_path) as f:
                itin = json.load(f)
        else:
            itin = {}

        # Mark CQT as done (stream directory already has complete shards).
        changed_itin = False
        for stage in ("cqt_saved",):
            if not itin.get(stage):
                itin[stage] = True
                changed_itin = True

        if changed_itin or not os.path.isfile(itin_path):
            print(f"  {tag}Writing itinerary_{h}.json  stages: {list(itin)}")
            if not dry_run:
                with open(itin_path, "w") as f:
                    json.dump(itin, f, indent=2)
        else:
            print(f"  itinerary_{h}.json already present")

    if not hashes:
        print("  No CQT stream found — itinerary not written")

    print(f"\nMigration {'(DRY-RUN) ' if dry_run else ''}complete.")
    print("The analysis folder is now compatible with the updated code.")
    print("Resume will skip CQT computation and load existing stream shards.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_dir",
                    help="Path to an existing *_analysis directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without making any changes")
    args = ap.parse_args()
    migrate(args.analysis_dir, dry_run=args.dry_run)
