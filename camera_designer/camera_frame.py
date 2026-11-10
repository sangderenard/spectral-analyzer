"""The sensor's look-at frame, shared by ray launch and physical camera pose."""
from __future__ import annotations

import numpy as np


def sensor_camera_basis(camera_eye, camera_target):
    """Return the existing SensorAccumulator (eye, right, up, forward) basis.

    Keep the float32 and vertical-view convention identical to sensor launch.
    CameraPreset's optical +Z is forward, so its rigid local-to-world rotation
    has columns (-right, up, forward), not the left-handed image-plane basis.
    """
    eye = np.asarray(camera_eye, np.float32).ravel()[:3]
    target = np.asarray(camera_target, np.float32).ravel()[:3]
    fwd = target - eye
    fwd_l = float(np.linalg.norm(fwd))
    fwd = (fwd / fwd_l if fwd_l > 1e-6 else np.array([0, 0, 1], np.float32)).astype(np.float32)
    world_up = np.array([0, 1, 0], np.float32)
    if abs(float(np.dot(fwd, world_up))) > 0.97:
        world_up = np.array([0, 0, 1], np.float32)
    right = np.cross(fwd, world_up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, fwd)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    return eye, right, up, fwd


def thin_lens_housing_frame(preset, camera_eye, camera_target):
    """Place native housing behind the analytical entrance-pupil plane.

    The native iris has finite thickness. Its front surface is one authored
    iris thickness behind the launch plane; every forward camera ray therefore
    starts outside the body, even at the pupil edge. No second refractor is
    placed between that plane and the subject. Body dimensions stay in metres.
    """
    eye, right, up, fwd = sensor_camera_basis(camera_eye, camera_target)
    rotation = np.column_stack((-right, up, fwd)).astype(np.float64)
    iris = preset.aperture_stop
    thickness = float(iris.thickness_m)
    front = float(iris.z_pos) + 0.5 * thickness
    translation = eye.astype(np.float64) - (front + thickness) * fwd
    return rotation, translation
