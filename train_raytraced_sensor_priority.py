"""Train sensor work value from bounded randomized spectral text exposures."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from camera_software import (
    SensorWorkValueNet,
    load_raytraced_states,
    measured_work_examples,
    normalized_camera_rgb,
    sensor_features,
    train_from_raytraced_examples,
)
from live_spectral_text_demo import JOB_ID, build_paragraph_order


WORDS = (
    "actual spectral camera glass light shadow quiet bright amber violet focus "
    "sensor detail letters signal noise photon surface through around between "
    "clarity uncertain resolve evidence texture reflected scattered precise"
).split()


def _save_camera(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    value = np.maximum(np.asarray(rgb, np.float32), 0.0)
    positive = value[value > 0.0]
    white = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    display = np.clip(value / max(white, 1.0e-12), 0.0, 1.0)
    Image.fromarray(np.asarray(display * 255.0, np.uint8), "RGB").save(path)


def _save_heat(path: Path, values: np.ndarray) -> None:
    from PIL import Image
    value = np.maximum(np.asarray(values, np.float32), 0.0)
    value /= max(float(value.max()), 1.0e-12)
    heat = np.stack([value, np.sqrt(value) * 0.35, 1.0 - value], axis=-1)
    Image.fromarray(np.asarray(heat * 255.0, np.uint8), "RGB").save(path)


def _save_training_diagnostics(
    scene_dir: Path, states, example, model_path: str,
) -> None:
    _save_camera(scene_dir / "training_camera_input.png", example.camera_rgb)
    _save_camera(
        scene_dir / "training_later_exposure.png",
        normalized_camera_rgb(states[-1].sensor_sum, states[-1].exposure_weight),
    )
    _save_heat(scene_dir / "measured_work_value.png", example.measured_improvement)
    model = SensorWorkValueNet().cuda()
    model.load_glsl_parameters(np.load(model_path, allow_pickle=False)["parameters"])
    rgb = torch.as_tensor(example.camera_rgb, device="cuda").permute(2, 0, 1)[None]
    exposure = torch.as_tensor(
        example.exposure_weight, device="cuda"
    )[None, None]
    with torch.no_grad():
        prediction = model(sensor_features(rgb, exposure))[0, 0].cpu().numpy()
    _save_heat(scene_dir / "predicted_work_value.png", prediction)


def random_training_text(rng: random.Random) -> str:
    """Produce varied text without leaking a fixed phrase into the scheduler."""
    words = [rng.choice(WORDS) for _ in range(rng.randint(4, 12))]
    if rng.random() < 0.45:
        index = rng.randrange(len(words))
        words[index] = "".join(rng.sample(words[index], len(words[index])))
    if rng.random() < 0.35:
        words.insert(rng.randrange(len(words) + 1), str(rng.randint(10, 9999)))
    if rng.random() < 0.30:
        words[rng.randrange(len(words))] = rng.choice(words).upper()
    return " ".join(words)


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="exposures/raytraced_priority_training")
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--steps-per-layer", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=300)
    parser.add_argument("--targeted-fraction", type=float, default=0.5,
                        help="Keep substantial exploration while collecting labels.")
    parser.add_argument("--resume-model", default="")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(sys.argv[1:] if argv is None else argv)
    if args.scenes <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("scene count and dimensions must be positive")
    if args.min_epochs < 2 or args.max_epochs < args.min_epochs:
        raise SystemExit("epoch interval must satisfy 2 <= min <= max")
    if not 0.0 <= args.targeted_fraction <= 1.0:
        raise SystemExit("--targeted-fraction must be in [0,1]")
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    script_dir = Path(__file__).resolve().parent
    automatic_model = root / "model" / "raytraced_priority_network.npz"
    current_model = (
        str(Path(args.resume_model).resolve()) if args.resume_model
        else (str(automatic_model) if automatic_model.is_file() else "")
    )
    all_examples = []
    if current_model:
        print(f"[trainer] resuming reusable model {current_model}", flush=True)

    for scene_index in range(args.scenes):
        text = random_training_text(rng)
        epochs = rng.randint(args.min_epochs, args.max_epochs)
        scene_dir = root / f"scene_{scene_index:04d}"
        progress_dir = scene_dir / "progress"
        render_dir = scene_dir / "render"
        scene_dir.mkdir(parents=True, exist_ok=True)
        order_path = scene_dir / "scene_order.json"
        order_path.write_text(json.dumps(build_paragraph_order(
            text,
            display_width=args.width,
            display_height=args.height,
            sensor_sweeps=1,
        ), indent=2), encoding="utf-8")
        try:
            existing_states = load_raytraced_states(str(progress_dir))
        except (ValueError, OSError):
            existing_states = []
        if existing_states:
            existing_examples = measured_work_examples(existing_states)
            all_examples.extend(existing_examples)
            print(
                f"[trainer] reusing completed scene={scene_index + 1}/{args.scenes} "
                f"states={len(existing_states)} examples={len(existing_examples)}",
                flush=True,
            )
            if not current_model:
                result = train_from_raytraced_examples(
                    all_examples,
                    str(root / "model"),
                    steps=args.training_steps,
                    seed=args.seed + scene_index,
                )
                current_model = result.model_path
            continue
        command = [
            sys.executable,
            str(script_dir / "exposure_render_demo.py"),
            "--scene-order", str(order_path),
            "--scene-job", JOB_ID,
            "--integrator", "bdpt",
            "--backend", "cpp",
            "--frames", "1",
            "--gpu-resident",
            "--no-window",
            "--save-files",
            "--out-dir", str(render_dir),
            "--bdpt-native-packages", "1",
            "--progress-dir", str(progress_dir),
            "--progress-exposure-id", f"trainer-{scene_index:04d}",
        ]
        environment = os.environ.copy()
        environment["SPECTRAL_SENSOR_MAX_EPOCHS"] = str(epochs)
        environment["SPECTRAL_SENSOR_PERSISTENT_EPOCHS"] = "0"
        environment["SPECTRAL_SENSOR_STEPS_PER_LAYER"] = str(args.steps_per_layer)
        environment["SPECTRAL_SENSOR_TARGETED_FRACTION"] = str(args.targeted_fraction)
        environment["SPECTRAL_PROGRESS_RETAIN_LAYERS"] = "0"
        if current_model:
            environment["SPECTRAL_SENSOR_PRIORITY_MODEL"] = current_model
        else:
            environment.pop("SPECTRAL_SENSOR_PRIORITY_MODEL", None)
        print(
            f"[trainer] scene={scene_index + 1}/{args.scenes} epochs={epochs} "
            f"text={text!r} model={current_model or 'heuristic exploration'}",
            flush=True,
        )
        log_path = scene_dir / "render.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=script_dir,
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
                log.write(line)
                log.flush()
                console_encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(
                    console_encoding, errors="replace"
                ).decode(console_encoding, errors="replace")
                print(safe_line, end="", flush=True)
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        states = load_raytraced_states(str(progress_dir))
        examples = measured_work_examples(states)
        all_examples.extend(examples)
        training_dir = root / "model"
        result = train_from_raytraced_examples(
            all_examples,
            str(training_dir),
            steps=args.training_steps,
            seed=args.seed + scene_index,
            resume_model=current_model,
        )
        current_model = result.model_path
        _save_training_diagnostics(
            scene_dir, states, examples[0], result.model_path
        )
        print(
            f"[trainer] learned examples={result.example_count} "
            f"loss={result.final_loss:.6f} model={result.model_path}",
            flush=True,
        )

    print(f"[trainer] complete reusable_model={current_model}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
