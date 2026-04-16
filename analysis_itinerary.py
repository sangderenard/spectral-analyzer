from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias


@dataclass
class AnalysisGlobalSettings:
    channel_mode_idx: int = 0
    analysis_compute_precision_idx: int = 1
    analysis_save_precision_idx: int = 1
    cafls_anchor: float = 30.87
    hybrid_redundancy_octaves: float = 0.0
    resample_engine_idx: int = 0
    resample_interp_idx: int = 0
    resample_taps_idx: int = 3
    resample_precision_idx: int = 2
    region_start: float = 0.0
    region_end: float = 0.0

    def apply_to(self, target: Any) -> None:
        for key, value in asdict(self).items():
            setattr(target, key, value)


@dataclass
class AnalysisTimeRange:
    start_sec: float = 0.0
    end_sec: float = 0.0
    total_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "total_sec": float(self.total_sec),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisTimeRange":
        return cls(
            start_sec=float(data.get("start_sec", 0.0) or 0.0),
            end_sec=float(data.get("end_sec", 0.0) or 0.0),
            total_sec=float(data.get("total_sec", 0.0) or 0.0),
        )

    def clamped(self) -> "AnalysisTimeRange":
        total = max(0.0, float(self.total_sec))
        start = max(0.0, float(self.start_sec))
        end = max(0.0, float(self.end_sec))
        if total > 0.0:
            start = min(start, total)
            end = min(end, total)
        if end > 0.0 and end < start:
            end = start
        return AnalysisTimeRange(start_sec=start, end_sec=end, total_sec=total)

    def effective_end_sec(self) -> float:
        rng = self.clamped()
        if rng.end_sec > 0.0:
            return rng.end_sec
        if rng.total_sec > 0.0:
            return rng.total_sec
        return 0.0

    def as_fraction_span(self) -> tuple[float, float]:
        rng = self.clamped()
        total = max(rng.total_sec, rng.effective_end_sec(), 1e-9)
        lo = max(0.0, min(1.0, rng.start_sec / total))
        hi = max(lo, min(1.0, rng.effective_end_sec() / total))
        return lo, hi


@dataclass
class FFTAnalysisSettings:
    fft_algorithm: str = "librosa"
    cqt_window_idx: int = 0
    hop_length: int = 512
    stft_n_fft: int = 4096
    bins_per_octave: int = 1200
    cqt_filter_scale: float = 1.0
    cqt_fmin: float = 30.87
    cqt_fmax: float = 19912.13
    cqt_bpo_schedule: list[int] | None = None
    cqt_hop_schedule: list[int] | None = None
    cqt_filter_scale_schedule: list[float] | None = None
    # Pre-analysis bandpass filter (applied to audio before the transform).
    # None means that edge is unfiltered.
    prefilter_bp_fmin: float | None = None
    prefilter_bp_fmax: float | None = None
    # Heterodyne carrier (Hz, signed).  Positive: signal is mixed UP by this
    # amount so sub-bass / subsonic / seismic content enters the analysis
    # window.  Negative: signal is mixed DOWN so high-frequency content
    # shifts into a lower analysis range.  0.0 = no heterodyne.
    # Physical frequency of a stored analysis bin f: physical = f - heterodyne_hz
    heterodyne_hz: float = 0.0
    # Post-heterodyne bandpass filter (applied to audio after the heterodyne
    # mix, immediately before the transform).  Useful to reject the mirror
    # image band produced by real-valued DSB mixing.  None = unfiltered.
    postfilter_bp_fmin: float | None = None
    postfilter_bp_fmax: float | None = None
    # Filter implementation: "fir" (freq-domain LR, HQ) or "iir" (Butterworth, fast).
    filter_mode: str = "fir"
    # Spline detrender config (SplineDetrenderConfig serialised as dict).
    # None = detrender disabled.
    spline_detrend: dict | None = None
    # Last-resort resume: when set, bass_analysis receives --shash <value> so it
    # finds the exact stream directory even if computed hash differs.  Set by
    # _run_resume() when no matching settings file is found and the itinerary
    # is synthesised directly from the stream dir's self-describing snapshot.
    # Serialised into analysis_settings.json so the override persists on retry.
    shash_override: str | None = None

    def apply_to(self, target: Any) -> None:
        if hasattr(target, "_CQT_ALGORITHMS"):
            try:
                setattr(
                    target,
                    "cqt_algorithm_idx",
                    list(getattr(target, "_CQT_ALGORITHMS")).index(self.fft_algorithm),
                )
            except Exception:
                setattr(target, "cqt_algorithm_idx", 0)
        else:
            algo_map = {"librosa": 0, "nsgt": 1, "stft": 2}
            setattr(target, "cqt_algorithm_idx", algo_map.get(self.fft_algorithm, 0))
        setattr(target, "cqt_window_idx", self.cqt_window_idx)
        setattr(target, "hop_length", self.hop_length)
        setattr(target, "stft_n_fft", self.stft_n_fft)
        setattr(target, "bins_per_octave", self.bins_per_octave)
        setattr(target, "cqt_filter_scale", self.cqt_filter_scale)
        setattr(target, "cqt_fmin", self.cqt_fmin)
        setattr(target, "cqt_fmax", self.cqt_fmax)
        setattr(
            target, "cqt_bpo_schedule",
            list(self.cqt_bpo_schedule) if self.cqt_bpo_schedule else None,
        )
        setattr(
            target, "cqt_hop_schedule",
            list(self.cqt_hop_schedule) if self.cqt_hop_schedule else None,
        )
        setattr(
            target, "cqt_filter_scale_schedule",
            list(self.cqt_filter_scale_schedule)
            if self.cqt_filter_scale_schedule else None,
        )
        setattr(target, "prefilter_bp_fmin", self.prefilter_bp_fmin)
        setattr(target, "prefilter_bp_fmax", self.prefilter_bp_fmax)
        setattr(target, "heterodyne_hz", float(self.heterodyne_hz))
        setattr(target, "postfilter_bp_fmin", self.postfilter_bp_fmin)
        setattr(target, "postfilter_bp_fmax", self.postfilter_bp_fmax)
        setattr(target, "filter_mode", self.filter_mode)
        setattr(target, "spline_detrend", self.spline_detrend)


@dataclass
class FilterbankAnalysisSettings:
    fb_filter_type_idx: int = 0
    fb_label_mode_idx: int = 1
    fb_hop: int = 1
    fb_config_mode_idx: int = 0
    fb_bpo: int = 12
    fb_banded_width: int = 3
    fb_crossovers: list[float] = field(default_factory=list)
    fb_hybrid: bool = False
    fb_q_norm: bool = False
    prefilter_bp_fmin: float | None = None
    prefilter_bp_fmax: float | None = None
    heterodyne_hz: float = 0.0
    postfilter_bp_fmin: float | None = None
    postfilter_bp_fmax: float | None = None
    filter_mode: str = "fir"
    spline_detrend: dict | None = None

    def apply_to(self, target: Any) -> None:
        for key, value in asdict(self).items():
            setattr(target, key, list(value) if key == "fb_crossovers" else value)


@dataclass
class WaveletAnalysisSettings:
    wavelet_mode_idx: int = 1
    wavelet_family_idx: int = 0
    wavelet_order_idx: int = 0
    wavelet_depth_pct: float = 100.0
    wavelet_depth_abs: int = 5
    wavelet_depth_mode: int = 0
    wavelet_ext_idx: int = 0
    cwt_wavelet_idx: int = 0
    cwt_hop_length: int = 1
    cwt_scales_per_octave: int = 12
    cwt_sigma: float = 6.0
    cwt_epsilon: float = 0.01
    prefilter_bp_fmin: float | None = None
    prefilter_bp_fmax: float | None = None
    heterodyne_hz: float = 0.0
    postfilter_bp_fmin: float | None = None
    postfilter_bp_fmax: float | None = None
    filter_mode: str = "fir"
    spline_detrend: dict | None = None

    def apply_to(self, target: Any) -> None:
        for key, value in asdict(self).items():
            setattr(target, key, value)


RunSettings: TypeAlias = FFTAnalysisSettings | FilterbankAnalysisSettings | WaveletAnalysisSettings


@dataclass
class AnalysisArtifact:
    kind: str
    path: str
    settings_hash: str = ""
    folder: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "settings_hash": self.settings_hash,
            "folder": self.folder,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisArtifact":
        return cls(
            kind=str(data.get("kind", "")),
            path=str(data.get("path", "")),
            settings_hash=str(data.get("settings_hash", "")),
            folder=str(data.get("folder", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AnalysisDatasetRecord:
    dataset_key: str
    engine: str
    algorithm: str = ""
    run_key: str = ""
    settings_hash: str = ""
    folder: str = ""
    status: str = "materialized"
    label: str = ""
    time_range: AnalysisTimeRange = field(default_factory=AnalysisTimeRange)
    settings: dict[str, Any] = field(default_factory=dict)
    artifacts: list[AnalysisArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "engine": self.engine,
            "algorithm": self.algorithm,
            "run_key": self.run_key,
            "settings_hash": self.settings_hash,
            "folder": self.folder,
            "status": self.status,
            "label": self.label,
            "time_range": self.time_range.to_dict(),
            "settings": dict(self.settings),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisDatasetRecord":
        return cls(
            dataset_key=str(data.get("dataset_key", "")),
            engine=str(data.get("engine", "")),
            algorithm=str(data.get("algorithm", "")),
            run_key=str(data.get("run_key", "")),
            settings_hash=str(data.get("settings_hash", "")),
            folder=str(data.get("folder", "")),
            status=str(data.get("status", "materialized")),
            label=str(data.get("label", "")),
            time_range=AnalysisTimeRange.from_dict(dict(data.get("time_range", {}))),
            settings=dict(data.get("settings", {})),
            artifacts=[
                AnalysisArtifact.from_dict(dict(artifact))
                for artifact in data.get("artifacts", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )


def _settings_to_dict(settings: RunSettings) -> dict[str, Any]:
    if isinstance(settings, FFTAnalysisSettings):
        return {"kind": "fft", **asdict(settings)}
    if isinstance(settings, FilterbankAnalysisSettings):
        return {"kind": "fb", **asdict(settings)}
    if isinstance(settings, WaveletAnalysisSettings):
        return {"kind": "wavelet", **asdict(settings)}
    raise TypeError(f"Unsupported settings type: {type(settings)!r}")


def _settings_from_dict(data: dict[str, Any]) -> RunSettings:
    kind = str(data.get("kind", "")).strip().lower()
    payload = dict(data)
    payload.pop("kind", None)
    if kind == "fft":
        return FFTAnalysisSettings(**payload)
    if kind == "fb":
        return FilterbankAnalysisSettings(**payload)
    if kind == "wavelet":
        return WaveletAnalysisSettings(**payload)
    raise ValueError(f"Unsupported run settings kind: {kind!r}")


@dataclass
class AnalysisRun:
    key: str
    engine: str
    settings: RunSettings
    enabled: bool = True
    label: str = ""
    status: str = "planned"
    output_folder: str = ""
    settings_hash: str = ""
    time_range: AnalysisTimeRange = field(default_factory=AnalysisTimeRange)
    deposits: list[AnalysisArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "engine": self.engine,
            "enabled": self.enabled,
            "label": self.label,
            "status": self.status,
            "output_folder": self.output_folder,
            "settings_hash": self.settings_hash,
            "time_range": self.time_range.to_dict(),
            "deposits": [artifact.to_dict() for artifact in self.deposits],
            "metadata": dict(self.metadata),
            "settings": _settings_to_dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRun":
        return cls(
            key=str(data.get("key", "")),
            engine=str(data.get("engine", "")),
            enabled=bool(data.get("enabled", True)),
            label=str(data.get("label", "")),
            status=str(data.get("status", "planned")),
            output_folder=str(data.get("output_folder", "")),
            settings_hash=str(data.get("settings_hash", "")),
            time_range=AnalysisTimeRange.from_dict(dict(data.get("time_range", {}))),
            deposits=[
                AnalysisArtifact.from_dict(dict(artifact))
                for artifact in data.get("deposits", [])
            ],
            metadata=dict(data.get("metadata", {})),
            settings=_settings_from_dict(dict(data.get("settings", {}))),
        )

    def apply_to(self, target: Any) -> None:
        self.settings.apply_to(target)


@dataclass
class AnalysisItinerary:
    globals: AnalysisGlobalSettings = field(default_factory=AnalysisGlobalSettings)
    runs: list[AnalysisRun] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "analysis_itinerary_v1",
            "schema_version": self.schema_version,
            "globals": asdict(self.globals),
            "runs": [run.to_dict() for run in self.runs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisItinerary":
        globals_data = dict(data.get("globals", {}))
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            globals=AnalysisGlobalSettings(**globals_data),
            runs=[AnalysisRun.from_dict(dict(run)) for run in data.get("runs", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "AnalysisItinerary":
        return cls.from_dict(json.loads(raw))

    def add_run(self, run: AnalysisRun) -> None:
        self.runs.append(run)

    def enabled_runs(self) -> list[AnalysisRun]:
        return [run for run in self.runs if run.enabled]

    def get_run(self, key: str) -> AnalysisRun:
        for run in self.runs:
            if run.key == key:
                return run
        raise KeyError(key)

    def apply_run_to(self, target: Any, run: str | AnalysisRun) -> AnalysisRun:
        self.globals.apply_to(target)
        selected = self.get_run(run) if isinstance(run, str) else run
        selected.apply_to(target)
        return selected


@dataclass
class AnalysisInventory:
    datasets: list[AnalysisDatasetRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "analysis_inventory_v2",
            "schema_version": self.schema_version,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisInventory":
        if "datasets" in data:
            return cls(
                schema_version=int(data.get("schema_version", 1)),
                datasets=[
                    AnalysisDatasetRecord.from_dict(dict(dataset))
                    for dataset in data.get("datasets", [])
                ],
                metadata=dict(data.get("metadata", {})),
            )

        # Back-compat with the older engine-grouped inventory.
        datasets: list[AnalysisDatasetRecord] = []
        engines = dict(data.get("engines", {}))
        for engine, entry in engines.items():
            for idx, run in enumerate(entry.get("runs", [])):
                dataset_key = str(run.get("dataset_key", "")) or (
                    f"{engine}:{run.get('run_key', idx)}"
                )
                settings_hash = str(run.get("settings_hash", "")) or str(run.get("run_key", ""))
                artifacts: list[AnalysisArtifact] = []
                for art_key in ("npz", "legacy_npz", "meta", "data", "manifest", "envelopes", "versions_manifest"):
                    if art_key in run:
                        artifacts.append(AnalysisArtifact(
                            kind=art_key,
                            path=str(run[art_key]),
                            settings_hash=settings_hash,
                        ))
                datasets.append(AnalysisDatasetRecord(
                    dataset_key=dataset_key,
                    engine=str(engine),
                    algorithm=str(run.get("algorithm", "")),
                    run_key=str(run.get("run_key", "")),
                    settings_hash=settings_hash,
                    folder=str(run.get("folder", "")),
                    label=str(run.get("label", "")),
                    settings=dict(run.get("settings", {})),
                    artifacts=artifacts,
                    metadata={k: v for k, v in run.items()
                              if k not in {
                                  "dataset_key", "engine", "algorithm", "run_key",
                                  "settings_hash", "folder", "label", "settings",
                                  "npz", "legacy_npz", "meta", "data", "manifest",
                                  "envelopes", "versions_manifest",
                              }},
                ))
        return cls(datasets=datasets, metadata={})

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "AnalysisInventory":
        return cls.from_dict(json.loads(raw))

    def upsert_dataset(self, dataset: AnalysisDatasetRecord) -> None:
        for idx, existing in enumerate(self.datasets):
            if existing.dataset_key == dataset.dataset_key:
                self.datasets[idx] = dataset
                break
        else:
            self.datasets.append(dataset)
        self.datasets.sort(key=lambda d: (d.engine, d.algorithm, d.dataset_key))

    def datasets_for_engine(self, engine: str) -> list[AnalysisDatasetRecord]:
        return [dataset for dataset in self.datasets if dataset.engine == engine]
