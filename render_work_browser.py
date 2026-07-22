"""Live work/asset browser built on the repository scrolling-list widget."""
from __future__ import annotations

from typing import Any, Iterable

from camera_software.convergence_metrics import (
    artifact_convergence_metric,
    artifact_convergence_velocity,
    clamp01,
)


def artifact_convergence(artifact: Any) -> float:
    """Map retained image evidence to a stable, displayable convergence score."""

    return artifact_convergence_metric(artifact)


def _best_artifact(subtype: Any) -> Any | None:
    artifacts = tuple(getattr(subtype, "artifacts", ()))
    return max(
        artifacts,
        key=lambda item: (
            int(getattr(item, "samples", 0)),
            float(getattr(item, "created_at_s", 0.0)),
        ),
        default=None,
    )


def _checkpoint_passes(artifact: Any) -> int:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    return max(
        1, int(metadata.get("checkpoint_epochs_per_exposure", 1) or 1)
    )


class RenderWorkBrowser:
    """Adapter from bakery jobs/objects to ``ScrollableSubpanelList``."""

    def __init__(self, title: str = "WORK / ASSETS") -> None:
        from bass_viewer import ScrollableSubpanelList
        from camera_software.calibration_modes import CALIBRATION_MODES

        self.widget = ScrollableSubpanelList(
            title, max_height=220, key_prefix="render_work"
        )
        self._signature: tuple[Any, ...] = ()
        self._payloads: dict[str, dict[str, Any]] = {}
        self._surface = None
        self._font = None
        self._work_assets_expanded = False
        # Camera bootstrap is orchestration, not a scene. Spectral lane-table
        # policy is configured independently from these physical scenes.
        self._calibration_modes = tuple(
            mode for mode in CALIBRATION_MODES
            if mode.key != "camera-bootstrap"
        )
        self._calibration_index = next(
            i for i, mode in enumerate(self._calibration_modes)
            if mode.key == "color-science"
        )
        self._calibration_rect = None
        self._calibration_run_rect = None
        self._calibration_auto_rect = None
        self._delete_version_rect = None
        self._calibration_auto = False
        self._focus_distance_m = 3.75
        self._startup_report = None
        self._basic_validation_results: tuple[Any, ...] = ()
        self._startup_exposures: dict[str, dict[str, Any]] = {}
        self._selected_metrics: dict[str, Any] = {
            "convergence": 0.0,
            "convergence_velocity_per_pass": 0.0,
            "convergence_update_passes": 1,
            "priority_share": 0.0,
            "working": False,
            "status": "waiting",
        }

    @staticmethod
    def _artifact_preview(subtype: Any) -> str:
        for artifact in getattr(subtype, "artifacts", ()):
            path = str(
                getattr(artifact, "preview_path", "")
                or getattr(artifact, "linear_path", "")
            )
            if path:
                return path
        return ""

    def sync(
        self,
        pending: Iterable[Any],
        bundles: Iterable[Any],
        interfaces: Iterable[Any] = (),
        *,
        active: tuple[str, str, int, bool] = ("", "", 0, False),
        layout_work: tuple[Any, Any] | None = None,
        resolved_text_images: Iterable[Any] = (),
        calibration_work: Iterable[Any] = (),
    ) -> None:
        """Refresh rows while retaining selection, expansion, and scroll."""

        from bass_viewer import ModularSubpanelSpec

        active_key, active_token, active_pass, is_active = active
        rows: list[tuple[str, str, list[str], tuple[int, int, int], dict[str, Any]]] = []
        pending = tuple(pending)
        bundles = tuple(bundles)
        bundle_by_key = {
            str(getattr(bundle, "object_key", "")): bundle
            for bundle in bundles
        }
        subtype_by_token: dict[str, Any] = {}
        subtype_by_source: dict[str, Any] = {}
        for bundle in bundles:
            for subtype in getattr(bundle, "subtypes", ()):
                subtype_by_token[str(getattr(subtype, "text", ""))] = subtype
                subtype_by_source[str(getattr(subtype, "source_asset_key", ""))] = subtype

        active_was_matched = False
        priority_payloads: list[dict[str, Any]] = []
        for request in pending:
            token_asset = getattr(request, "token_asset", None)
            token = str(getattr(token_asset, "token", "") or "interface")
            target_kind = str(getattr(getattr(request, "target_kind", None), "value", "job"))
            request_key = str(getattr(request, "request_key", token))
            target_key = str(getattr(request, "target_key", ""))
            subtype = subtype_by_source.get(target_key) or subtype_by_token.get(token)
            best_artifact = _best_artifact(subtype)
            convergence = artifact_convergence(best_artifact)
            convergence_velocity = artifact_convergence_velocity(best_artifact)
            working = bool(
                active_token
                and not active_was_matched
                and token == active_token
            )
            active_was_matched = active_was_matched or working
            refinement_pass = int(getattr(request, "refinement_pass", 0))
            payload = {
                "kind": "job",
                "request": request,
                "subtype": subtype,
                "convergence": convergence,
                "convergence_velocity_per_pass": convergence_velocity,
                "convergence_update_passes": _checkpoint_passes(best_artifact),
                "working": working and is_active,
                "status": (
                    "working" if working and is_active
                    else "checkpointed" if working
                    else "queued"
                ),
                # This is the planner's actual score, not a UI approximation.
                "_priority_need": float(getattr(
                    request,
                    "priority_need",
                    max(0.02, 1.0 - convergence)
                    / float(1 + max(0, refinement_pass)),
                )),
            }
            priority_payloads.append(payload)
            rows.append((
                f"job:{request_key}",
                (
                    f"● WORKING  {token}" if working and is_active
                    else f"◐ CHECKPOINT  {token}" if working
                    else f"QUEUED  {token}"
                ),
                [
                    f"kind: {target_kind}",
                    f"resume pass: {refinement_pass}",
                ],
                (235, 156, 55) if working else (95, 135, 210),
                payload,
            ))
        if active_token and not active_was_matched:
            subtype = subtype_by_token.get(active_token)
            best_artifact = _best_artifact(subtype)
            convergence = artifact_convergence(best_artifact)
            convergence_velocity = artifact_convergence_velocity(best_artifact)
            payload = {
                "kind": "active",
                "token": active_token,
                "subtype": subtype,
                "convergence": convergence,
                "convergence_velocity_per_pass": convergence_velocity,
                "convergence_update_passes": _checkpoint_passes(best_artifact),
                "working": bool(is_active),
                "status": "working" if is_active else "checkpointed",
                "_priority_need": max(0.02, 1.0 - convergence)
                / float(1 + max(0, int(active_pass))),
            }
            priority_payloads.insert(0, payload)
            rows.insert(0, (
                f"active:{active_key or active_token}",
                (
                    f"● WORKING  {active_token}" if is_active
                    else f"◐ CHECKPOINT  {active_token}"
                ),
                [
                    f"state: {'integrating' if is_active else 'checkpointed'}",
                    f"pass: {int(active_pass)}",
                ],
                (235, 156, 55),
                payload,
            ))
        priority_total = sum(
            float(payload["_priority_need"]) for payload in priority_payloads
        )
        for payload in priority_payloads:
            payload["priority_share"] = (
                float(payload["_priority_need"]) / priority_total
                if priority_total > 0.0 else 0.0
            )
            payload.pop("_priority_need", None)
        for _key, _title, summary, _accent, payload in rows:
            if payload.get("kind") in {"active", "job"}:
                summary.extend([
                    f"convergence: {100.0 * float(payload['convergence']):5.1f}%",
                    (
                        "convergence velocity: "
                        f"{float(payload.get('convergence_velocity_per_pass', 0.0)):+.3e} pass^-1"
                    ),
                    (
                        "quality checkpoint: every "
                        f"{int(payload.get('convergence_update_passes', 1))} passes"
                    ),
                    f"priority share: {100.0 * float(payload['priority_share']):5.1f}%",
                ])
        for bundle in bundles:
            bundle_name = str(getattr(bundle, "display_name", "render object"))
            for subtype in getattr(bundle, "subtypes", ()):
                subtype_key = str(getattr(subtype, "subtype_key", ""))
                text = str(getattr(subtype, "text", ""))
                kind = str(getattr(subtype, "kind", "asset"))
                complete = bool(getattr(subtype, "complete", False))
                artifacts = tuple(getattr(subtype, "artifacts", ()))
                samples = max(
                    (int(getattr(item, "samples", 0)) for item in artifacts),
                    default=0,
                )
                preview = self._artifact_preview(subtype)
                best_artifact = _best_artifact(subtype)
                convergence = artifact_convergence(best_artifact)
                convergence_velocity = artifact_convergence_velocity(
                    best_artifact
                )
                parametric_scene = kind == "parametric_layout"
                source_scene = next(
                    (
                        str(getattr(scene, "scene_path", ""))
                        for scene in getattr(subtype, "scenes", ())
                        if str(getattr(scene, "scene_path", ""))
                    ),
                    "",
                )
                rows.append((
                    f"asset:{subtype_key}",
                    (
                        f"SCENE READY  {getattr(subtype, 'display_name', kind)}"
                        if parametric_scene else
                        f"{'DONE' if complete else 'PARTIAL'}  {text}"
                    ),
                    [
                        f"{bundle_name} / {kind}",
                        *(
                            ["parameters: role / crop / fill / target / pose"]
                            if parametric_scene else []
                        ),
                        *(
                            [f"source scene: {source_scene}"]
                            if parametric_scene else []
                        ),
                        f"samples: {samples}",
                        f"convergence: {100.0 * convergence:5.1f}%",
                        (
                            "convergence velocity: "
                            f"{convergence_velocity:+.3e} pass^-1"
                        ),
                        (
                            "quality checkpoint: every "
                            f"{_checkpoint_passes(best_artifact)} passes"
                        ),
                        "priority share:   0.0%",
                    ],
                    (
                        (75, 145, 185) if parametric_scene else
                        (70, 165, 105) if complete else (180, 135, 55)
                    ),
                    {
                        "kind": "asset",
                        "bundle": bundle,
                        "subtype": subtype,
                        "preview_path": preview,
                        "convergence": convergence,
                        "convergence_velocity_per_pass": convergence_velocity,
                        "convergence_update_passes": _checkpoint_passes(
                            best_artifact
                        ),
                        "priority_share": 0.0,
                        "working": False,
                        "status": (
                            "scene_ready" if parametric_scene else
                            "done" if complete else "cached"
                        ),
                    },
                ))
        if layout_work is not None:
            layout_manifest, layout_cache = layout_work
            cache_snapshot = layout_cache.snapshot()
            for design_object in getattr(layout_manifest, "objects", ()):
                library_bundle = bundle_by_key.get(
                    str(getattr(design_object, "object_key", ""))
                )
                for subtype in getattr(design_object, "subtypes", ()):
                    animations = tuple(getattr(subtype, "pose_animations", ()))
                    frame_keys = [
                        frame.progress_cache_key
                        for animation in animations for frame in animation.frames
                    ]
                    complete_frames = sum(
                        str(cache_snapshot.get(key, {}).get("status", ""))
                        in {"accepted", "completed"}
                        for key in frame_keys
                    )
                    total_frames = len(frame_keys)
                    replacement = str(layout_cache.replacement_path(subtype))
                    library_subtype = next(
                        (
                            item for item in (
                                () if library_bundle is None
                                else getattr(library_bundle, "subtypes", ())
                            )
                            if str(getattr(item, "source_asset_key", ""))
                            == str(getattr(subtype, "subtype_key", ""))
                        ),
                        None,
                    )
                    convergence = (
                        complete_frames / float(total_frames)
                        if total_frames else 0.0
                    )
                    animation_summary = (
                        "none"
                        if not animations else
                        f"{animations[0].name}; "
                        f"{animations[0].rack_frames}/"
                        f"{animations[0].hold_frames}/"
                        f"{animations[0].rack_frames} frames; "
                        f"{animations[0].focus_location_count} focus locations"
                    )
                    rows.append((
                        f"design:{design_object.object_key}:{subtype.subtype_key}",
                        (
                            f"DESIGN READY  {design_object.display_name}"
                            if replacement else
                            f"DESIGN UNRENDERED  {design_object.display_name}"
                        ),
                        [
                            f"subtype: {subtype.display_name}",
                            f"consumers: {len(subtype.consumers)} panel patches",
                            f"pose animation: {animation_summary}",
                            f"pose cache: {complete_frames}/{total_frames}",
                            (
                                "replacement: live rendered hold pose"
                                if replacement else
                                "replacement: authored color fallback"
                            ),
                            (
                                "library scene: registered; render job not yet leased"
                                if library_subtype is not None else
                                "library scene: missing"
                            ),
                        ],
                        (75, 165, 140) if replacement else (155, 105, 190),
                        {
                            "kind": "design",
                            "design_object": design_object,
                            "subtype": subtype,
                            "preview_path": replacement,
                            "convergence": convergence,
                            "convergence_velocity_per_pass": 0.0,
                            "priority_share": 0.0,
                            "working": False,
                            "status": "done" if replacement else "unrendered",
                        },
                    ))
        for image in resolved_text_images:
            parts = tuple(getattr(image, "parts", ()))
            subtype = next(
                (
                    getattr(part, "subtype", None) for part in parts
                    if getattr(part, "subtype", None) is not None
                ),
                None,
            )
            preview = next(
                (
                    str(getattr(part, "artifact_path", ""))
                    for part in parts
                    if str(getattr(part, "artifact_path", ""))
                ),
                "",
            )
            resolution = str(getattr(image, "resolution", "pending"))
            convergence = float(getattr(image, "convergence", 0.0))
            convergence_velocity = float(
                getattr(image, "convergence_velocity_per_pass", 0.0)
            )
            rows.append((
                f"text-image:{getattr(image, 'owner_id', '')}",
                (
                    f"TEXT EXACT  {getattr(image, 'owner_id', '')}"
                    if resolution == "exact_token" else
                    f"TEXT GLYPHS  {getattr(image, 'owner_id', '')}"
                    if resolution == "glyph_fallback" else
                    f"TEXT PENDING  {getattr(image, 'owner_id', '')}"
                ),
                [
                    f"authored: {getattr(image, 'text', '')}",
                    f"resolution: {resolution}",
                    f"subtype parts: {len(parts)}",
                    (
                        "missing characters: "
                        + "".join(getattr(image, "missing_characters", ()))
                    ),
                    f"convergence: {100.0 * convergence:5.1f}%",
                    (
                        "convergence velocity: "
                        f"{convergence_velocity:+.3e} pass^-1"
                    ),
                    "material: progressive monofont evidence",
                ],
                (
                    (70, 165, 105) if convergence >= 1.0
                    else (90, 140, 205)
                ),
                {
                    "kind": "asset",
                    "resolution_kind": "text_image",
                    "resolved_text_image": image,
                    "subtype": subtype,
                    "preview_path": preview,
                    "convergence": convergence,
                    "convergence_velocity_per_pass": convergence_velocity,
                    "priority_share": 0.0,
                    "working": False,
                    "status": resolution,
                },
            ))
        for assembly in interfaces:
            latest = getattr(assembly, "latest", None)
            if latest is None:
                continue
            key = str(getattr(assembly, "assembly_key", "interface"))
            rows.append((
                f"interface:{key}",
                f"INTERFACE  {getattr(assembly, 'display_name', key)}",
                [
                    f"revision: {int(getattr(latest, 'revision', 0))}",
                    "after-render complete",
                ],
                (135, 95, 190),
                {
                    "kind": "interface",
                    "assembly": assembly,
                    "preview_path": str(getattr(latest, "image_path", "")),
                    "convergence": 1.0,
                    "convergence_velocity_per_pass": 0.0,
                    "priority_share": 0.0,
                    "working": False,
                    "status": "done",
                },
            ))
        for record in calibration_work:
            status = str(getattr(record, "status", "checkpointed"))
            mode_key = str(getattr(record, "mode_key", "calibration"))
            preview_path = str(getattr(record, "preview_path", ""))
            completed_runs = int(getattr(record, "completed_runs", 0))
            launch_count = int(getattr(record, "launch_count", 0))
            rows.append((
                f"calibration-work:{getattr(record, 'work_key', mode_key)}",
                (
                    f"CALIBRATION WORKING  {getattr(record, 'display_name', mode_key)}"
                    if status == "working" else
                    f"CALIBRATION FAILED  {getattr(record, 'display_name', mode_key)}"
                    if status == "failed" else
                    f"CALIBRATION ASSET  {getattr(record, 'display_name', mode_key)}"
                ),
                [
                    f"room: {mode_key}",
                    f"retained runs: {completed_runs}/{launch_count}",
                    (
                        "resume: next launch continues compatible evidence"
                        if completed_runs else "resume: checkpoint home created"
                    ),
                    f"status: {status}",
                ],
                (
                    (235, 156, 55) if status == "working" else
                    (205, 85, 65) if status == "failed" else
                    (85, 165, 125)
                ),
                {
                    "kind": "calibration_asset",
                    "calibration_work": record,
                    "preview_path": preview_path,
                    "convergence": 1.0 if status == "complete" else 0.0,
                    "convergence_velocity_per_pass": 0.0,
                    "priority_share": 0.0,
                    "working": status == "working",
                    "status": status,
                },
            ))
        work_child_rows = rows
        validation_rows: list[
            tuple[str, str, list[str], tuple[int, int, int], dict[str, Any]]
        ] = []
        report = self._startup_report
        if report is None:
            validation_summary = [
                "startup spectral matrix has not run",
                "native CPU evidence pending; GPU is reported separately",
            ]
            validation_status = "pending"
        else:
            expected_exposures = (
                [f"spectral:{int(case.lane_count)}" for case in report.cases]
                + [
                    f"basic:{result.validator_key}"
                    for result in self._basic_validation_results
                ]
            )
            completed_exposures = sum(
                self._startup_exposures.get(key, {}).get("status") == "passed"
                for key in expected_exposures
            )
            passed_count = sum(bool(case.passed) for case in report.cases)
            validation_summary = [
                f"camera exposures: {completed_exposures}/{len(expected_exposures)} complete",
                f"native lane prechecks: {passed_count}/{len(report.cases)} passed",
                f"bounded startup time: {float(report.elapsed_ms):.2f} ms",
            ]
            any_failed = any(
                self._startup_exposures.get(key, {}).get("status") == "failed"
                for key in expected_exposures
            )
            validation_status = (
                "failed" if any_failed or not report.passed else
                "passed" if completed_exposures == len(expected_exposures) else
                "working"
            )
        validation_rows.append((
            "validation-root",
            "VALIDATION / STARTUP",
            validation_summary,
            (70, 165, 105) if validation_status == "passed" else (190, 125, 55),
            {
                "kind": "group",
                "group": "startup_validation",
                "convergence": 1.0 if validation_status == "passed" else 0.0,
                "priority_share": 0.0,
                "working": False,
                "status": validation_status,
            },
        ))
        if report is not None:
            for case in report.cases:
                measurements = dict(case.measurements)
                exposure_key = f"spectral:{int(case.lane_count)}"
                evidence = self._startup_exposures.get(
                    exposure_key, {"status": "queued", "preview_path": ""}
                )
                exposure_status = str(evidence.get("status", "queued"))
                validation_rows.append((
                    f"validation:spectral:{int(case.lane_count)}",
                    (
                        f"  EXPOSURE {exposure_status.upper()}  {int(case.lane_count)} LANES"
                    ),
                    [
                        f"native CPU lane precheck: {case.cpu_status}",
                        f"native GPU camera exposure: {exposure_status}",
                        (
                            "one ray/lane: "
                            f"{int(measurements.get('segments_observed', 0.0))}/"
                            f"{int(measurements.get('rays_requested', case.lane_count))}"
                        ),
                        (
                            "frequency error: "
                            f"{float(measurements.get('maximum_frequency_error_hz', 0.0)):.3e} Hz"
                        ),
                        f"time: {float(case.elapsed_ms):.3f} ms",
                    ],
                    (
                        (70, 165, 105) if exposure_status == "passed" else
                        (205, 85, 65) if exposure_status == "failed" else
                        (190, 125, 55)
                    ),
                    {
                        "kind": "validation",
                        "validation_kind": "spectral_lane_matrix",
                        "case": case,
                        "preview_path": str(evidence.get("preview_path", "")),
                        "convergence": 1.0 if exposure_status == "passed" else 0.0,
                        "priority_share": 0.0,
                        "working": exposure_status == "working",
                        "status": exposure_status,
                    },
                ))
        for result in self._basic_validation_results:
            exposure_key = f"basic:{result.validator_key}"
            evidence = self._startup_exposures.get(
                exposure_key, {"status": "queued", "preview_path": ""}
            )
            exposure_status = str(evidence.get("status", "queued"))
            validation_rows.append((
                f"validation:basic:{result.validator_key}",
                (
                    f"  EXPOSURE {exposure_status.upper()}  {result.validator_key}"
                ),
                [result.detail, f"camera image: {exposure_status}"],
                (
                    (75, 150, 115) if exposure_status == "passed" else
                    (205, 85, 65) if exposure_status == "failed" else
                    (190, 125, 55)
                ),
                {
                    "kind": "validation",
                    "validation_kind": "basic_camera",
                    "result": result,
                    "preview_path": str(evidence.get("preview_path", "")),
                    "convergence": 1.0 if exposure_status == "passed" else 0.0,
                    "priority_share": 0.0,
                    "working": exposure_status == "working",
                    "status": exposure_status,
                },
            ))
        calibration_rows = [(
            "calibration-root", "CALIBRATION ROOMS",
            ["start here; each run becomes retained, resumable Work / Assets"],
            (132, 100, 190),
            {"kind": "group", "group": "calibration_rooms", "status": "ready"},
        )]
        for mode in self._calibration_modes:
            if mode.key == "off":
                continue
            summary = [str(mode.description)]
            if mode.key == "focus-hall":
                summary.insert(0, f"focus distance: {self._focus_distance_m:.2f} m")
            elif mode.key == "double-slit":
                summary.insert(0, "CPU strip + GPU strip; wave solvers only")
            calibration_rows.append((
                f"calibration:{mode.key}",
                f"  {mode.label.upper()} CALIBRATION ROOM",
                summary,
                (90, 150, 205) if mode.key != "double-slit" else (145, 105, 205),
                {"kind": "calibration", "mode": mode.key, "status": "ready"},
            ))
        rows = validation_rows + calibration_rows + [(
            "work-assets-root",
            "WORK / ASSETS",
            [
                f"{len(work_child_rows)} items available",
                "opt-in hierarchy; click header to show jobs and retained assets",
            ],
            (90, 125, 185),
            {
                "kind": "group",
                "group": "work_assets",
                "convergence": 0.0,
                "priority_share": 0.0,
                "working": False,
                "status": "expanded" if self._work_assets_expanded else "hidden",
            },
        )]
        if self._work_assets_expanded:
            rows.extend(
                (key, f"  {title}", summary, accent, payload)
                for key, title, summary, accent, payload in work_child_rows
            )
        signature = tuple(
            (key, title, tuple(summary), tuple(accent), payload.get("preview_path", ""))
            for key, title, summary, accent, payload in rows
        )
        if signature == self._signature:
            return
        expanded = {
            spec.key: bool(spec.expanded) for spec in self.widget.subpanels
        }
        specs = [
            ModularSubpanelSpec(
                key=key,
                title=title,
                summary_lines=summary,
                expanded=(
                    self._work_assets_expanded
                    if key == "work-assets-root"
                    else expanded.get(key, False)
                ),
                accent_rgb=accent,
                payload=payload,
                body_height=(
                    64 if key == "calibration:focus-hall"
                    else 0
                ),
                render_body=(
                    self._render_calibration_control_body
                    if key == "calibration:focus-hall"
                    else None
                ),
            )
            for key, title, summary, accent, payload in rows
        ]
        self._payloads = {spec.key: dict(spec.payload or {}) for spec in specs}
        self.widget.set_subpanels(specs)
        self._signature = signature
        self._refresh_selected_metrics()

    def _refresh_selected_metrics(self) -> None:
        payload = self._payloads.get(self.widget.selected_key or "", {})
        self._selected_metrics = {
            "convergence": clamp01(payload.get("convergence", 0.0)),
            "priority_share": clamp01(payload.get("priority_share", 0.0)),
            "convergence_velocity_per_pass": float(
                payload.get("convergence_velocity_per_pass", 0.0) or 0.0
            ),
            "convergence_update_passes": int(
                payload.get("convergence_update_passes", 1) or 1
            ),
            "working": bool(payload.get("working", False)),
            "status": str(payload.get("status", "waiting")),
        }

    @property
    def selected_payload(self) -> dict[str, Any]:
        payload = dict(self._payloads.get(self.widget.selected_key or "", {}))
        self._refresh_selected_metrics()
        return payload

    @property
    def selected_metrics(self) -> dict[str, Any]:
        self._refresh_selected_metrics()
        return dict(self._selected_metrics)

    @property
    def selected_preview_path(self) -> str:
        return str(self.selected_payload.get("preview_path", ""))

    @property
    def selected_calibration_mode(self) -> Any:
        from camera_software.calibration_modes import calibration_mode

        return calibration_mode(self._calibration_modes[self._calibration_index].key)

    def select_calibration_mode(self, key: str) -> None:
        index = next(
            (i for i, mode in enumerate(self._calibration_modes) if mode.key == str(key)),
            None,
        )
        if index is None:
            raise KeyError(key)
        self._calibration_index = index

    def calibration_parameters(self, key: str | None = None) -> dict[str, Any]:
        mode_key = str(key or self.selected_calibration_mode.key)
        parameters: dict[str, Any] = {}
        if mode_key == "focus-hall":
            parameters["focus_distance_m"] = float(self._focus_distance_m)
        return parameters

    def _render_calibration_control_body(
        self, surface: Any, font: Any, rect: Any, spec: Any,
        _item_map: Any, register_control: Any,
    ) -> None:
        import pygame

        mode = str(dict(spec.payload or {}).get("mode", ""))
        def control_row(
            y: int, label: str, path_name: str, left_text: str = "<",
            right_text: str = ">",
        ) -> None:
            text_surface = font.render(label, True, (210, 216, 226))
            surface.blit(text_surface, (rect.x + 8, y + 3))
            left = pygame.Rect(rect.right - 126, y, 54, 22)
            right = pygame.Rect(rect.right - 64, y, 54, 22)
            for button, text_value in ((left, left_text), (right, right_text)):
                pygame.draw.rect(surface, (44, 49, 62), button)
                pygame.draw.rect(surface, spec.accent_rgb, button, 1)
                rendered = font.render(text_value, True, (232, 234, 240))
                surface.blit(rendered, (
                    button.centerx - rendered.get_width() // 2,
                    button.centery - rendered.get_height() // 2,
                ))
            register_control(
                ("calibration", mode, path_name, "decrement"), left,
                kind="selector", payload={"direction": -1},
            )
            register_control(
                ("calibration", mode, path_name, "increment"), right,
                kind="selector", payload={"direction": 1},
            )

        if mode == "focus-hall":
            control_row(
                rect.y + 20,
                f"Focus distance  {self._focus_distance_m:.2f} m",
                "distance", "-", "+",
            )

    def set_work_assets_expanded(self, expanded: bool) -> None:
        self._work_assets_expanded = bool(expanded)
        self._signature = ()

    def set_startup_validation(
        self, report: Any, basic_results: Iterable[Any] = ()
    ) -> None:
        """Publish startup evidence as the primary, always-visible hierarchy."""

        self._startup_report = report
        self._basic_validation_results = tuple(basic_results)
        self._signature = ()

    def set_startup_exposure(
        self,
        key: str,
        status: str,
        *,
        preview_path: str = "",
        detail: str = "",
    ) -> None:
        """Attach a real camera exposure artifact to one startup test row."""

        state = str(status)
        if state not in {"queued", "working", "passed", "failed"}:
            raise ValueError("startup exposure status is invalid")
        self._startup_exposures[str(key)] = {
            "status": state,
            "preview_path": str(preview_path),
            "detail": str(detail),
        }
        self._signature = ()

    @property
    def calibration_manifest(self) -> dict[str, Any]:
        manifest = self.selected_calibration_mode.work_asset_manifest()
        manifest["parameters"] = self.calibration_parameters(
            self.selected_calibration_mode.key
        )
        return manifest

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        import pygame

        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 12)
            self.widget.TITLE_H = 28
            self.widget.HEADER_H = 28
        if self._surface is None or self._surface.get_size() != (width, height):
            self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._surface.fill((18, 20, 26, 245))
        selector_h = 30
        auto_width = max(58, min(96, width // 6))
        run_width = max(72, min(112, width // 5))
        delete_width = max(72, min(112, width // 5))
        selector_width = max(
            40, width - auto_width - run_width - delete_width - 30
        )
        selector = pygame.Rect(6, 3, selector_width, selector_h - 6)
        run_rect = pygame.Rect(selector.right + 6, 3, run_width, selector_h - 6)
        auto_rect = pygame.Rect(run_rect.right + 6, 3, auto_width, selector_h - 6)
        delete_rect = pygame.Rect(
            auto_rect.right + 6, 3, delete_width, selector_h - 6
        )
        pygame.draw.rect(self._surface, (30, 34, 44), selector)
        pygame.draw.rect(self._surface, (112, 92, 180), selector, 1)
        mode = self.selected_calibration_mode
        label = self._font.render(
            f"CALIBRATION  < {mode.label} >", True, (224, 218, 244)
        )
        previous_clip = self._surface.get_clip()
        self._surface.set_clip(selector)
        self._surface.blit(label, (selector.x + 6, selector.y + 3))
        self._surface.set_clip(previous_clip)
        pygame.draw.rect(self._surface, (42, 72, 50), run_rect)
        pygame.draw.rect(self._surface, (92, 168, 108), run_rect, 1)
        run_label = self._font.render("RUN SELECTED", True, (226, 244, 230))
        self._surface.blit(run_label, (
            run_rect.centerx - run_label.get_width() // 2,
            run_rect.centery - run_label.get_height() // 2,
        ))
        pygame.draw.rect(
            self._surface,
            (58, 78, 54) if self._calibration_auto else (66, 48, 48),
            auto_rect,
        )
        pygame.draw.rect(self._surface, (130, 112, 112), auto_rect, 1)
        auto_label = self._font.render(
            "AUTO: ON" if self._calibration_auto else "AUTO: OFF",
            True, (232, 230, 230),
        )
        self._surface.blit(auto_label, (
            auto_rect.centerx - auto_label.get_width() // 2,
            auto_rect.centery - auto_label.get_height() // 2,
        ))
        pygame.draw.rect(self._surface, (82, 42, 42), delete_rect)
        pygame.draw.rect(self._surface, (188, 92, 92), delete_rect, 1)
        delete_label = self._font.render("DELETE REV", True, (248, 226, 226))
        self._surface.blit(delete_label, (
            delete_rect.centerx - delete_label.get_width() // 2,
            delete_rect.centery - delete_label.get_height() // 2,
        ))
        self._calibration_rect = selector
        self._calibration_run_rect = run_rect
        self._calibration_auto_rect = auto_rect
        self._delete_version_rect = delete_rect
        self.widget.max_height = max(
            60, height - selector_h - self.widget.TITLE_H - 4
        )
        self.widget.render(self._surface, self._font, 0, selector_h, width)
        destination.blit(self._surface, (x, y))

    def handle_event(
        self, event: Any, rect: tuple[int, int, int, int]
    ) -> str | None:
        import pygame

        x, y, width, height = map(int, rect)
        host = pygame.Rect(x, y, width, height)
        if event.type == pygame.MOUSEWHEEL:
            pointer = getattr(event, "pos", pygame.mouse.get_pos())
            if not host.collidepoint(pointer):
                return None
            return "scroll" if self.widget.handle_scroll(-int(event.y)) else ""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not host.collidepoint(event.pos):
                return None
            lx, ly = int(event.pos[0]) - x, int(event.pos[1]) - y
            if (
                self._delete_version_rect is not None
                and self._delete_version_rect.collidepoint(lx, ly)
            ):
                return "delete-selected-version"
            if self._calibration_run_rect is not None and self._calibration_run_rect.collidepoint(lx, ly):
                return f"calibration:{self.selected_calibration_mode.key}"
            if self._calibration_auto_rect is not None and self._calibration_auto_rect.collidepoint(lx, ly):
                self._calibration_auto = not self._calibration_auto
                return "calibration-auto:" + ("on" if self._calibration_auto else "off")
            if self._calibration_rect is not None and self._calibration_rect.collidepoint(lx, ly):
                self._calibration_index = (
                    self._calibration_index + 1
                ) % len(self._calibration_modes)
                action = f"calibration:{self.selected_calibration_mode.key}"
                return action if self._calibration_auto else "calibration-config:" + action.split(":", 1)[1]
            action = self.widget.handle_click(lx, ly)
            if action and action.startswith("control:calibration|"):
                pieces = action.split("|")
                _prefix, mode = pieces[:2]
                control = pieces[2] if len(pieces) == 4 else "distance"
                direction = pieces[-1]
                delta = -1 if direction == "decrement" else 1
                if mode == "focus-hall":
                    self._focus_distance_m = round(
                        max(2.7, min(5.1, self._focus_distance_m + 0.25 * delta)),
                        2,
                    )
                self.select_calibration_mode(mode)
                self._signature = ()
                action = f"calibration:{mode}"
                return action if self._calibration_auto else f"calibration-config:{mode}"
            if action and action.startswith("calibration:"):
                mode = action.split(":", 1)[1]
                self.select_calibration_mode(mode)
                return action if self._calibration_auto else f"calibration-config:{mode}"
            if action == "work-assets-root":
                self._work_assets_expanded = not self._work_assets_expanded
                self._signature = ()
            return action
        return None


__all__ = ["RenderWorkBrowser", "artifact_convergence"]
