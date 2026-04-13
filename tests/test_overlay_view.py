import numpy as np

import bass_viewer


def test_cafls_filterbank_bands_scale_geometrically_around_anchor():
    anchor = 64.0
    bpo = 12
    bands = bass_viewer.FilterBankDecomposition.bands_from_cafls(
        anchor=anchor, bpo=bpo, banded_width=1, sr=4096)

    core = bands[1:-1]
    centers = np.array([band.centre for band in core], dtype=np.float64)
    widths = np.array([band.fmax - band.fmin for band in core], dtype=np.float64)
    step_ratio = 2.0 ** (1.0 / bpo)
    mid = len(core) // 2

    assert np.isclose(core[mid].centre, anchor, rtol=1e-6)
    assert np.allclose(centers[1:] / centers[:-1], step_ratio, rtol=1e-6)
    assert not np.allclose(widths[1:], widths[:-1])
    assert widths[mid - 1] < widths[mid] < widths[mid + 1]


def test_active_field_bindings_exclude_wavelet_from_plain_cqt_tf():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.display_mode = "tf"
    viewer.tf_source = "cqt"
    viewer.field_sources = [
        bass_viewer.FieldSource("mag_left", "Mag L", "cqt_left_polar",
                                0, category="cqt"),
        bass_viewer.FieldSource("fb_mag_left", "FB Mag L", "fb_left_polar",
                                0, category="fb"),
        bass_viewer.FieldSource("wvlt_approx", "Wvlt Approx",
                                "wavelet_coeffs", 0, category="wvlt"),
    ]
    viewer.field_configs = [
        bass_viewer.FieldConfig(target=0),
        bass_viewer.FieldConfig(target=1),
        bass_viewer.FieldConfig(target=2),
    ]

    active = viewer._active_field_bindings()

    assert [fs.name for _, fs, _ in active] == ["mag_left"]


def test_active_field_bindings_include_wavelet_in_hybrid_tf():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.display_mode = "tf"
    viewer.tf_source = "hybrid"
    viewer.field_sources = [
        bass_viewer.FieldSource("mag_left", "Mag L", "cqt_left_polar",
                                0, category="cqt"),
        bass_viewer.FieldSource("fb_mag_left", "FB Mag L", "fb_left_polar",
                                0, category="fb"),
        bass_viewer.FieldSource("wvlt_approx", "Wvlt Approx",
                                "wavelet_coeffs", 0, category="wvlt"),
    ]
    viewer.field_configs = [
        bass_viewer.FieldConfig(target=0),
        bass_viewer.FieldConfig(target=1),
        bass_viewer.FieldConfig(target=2),
    ]

    active = viewer._active_field_bindings()

    assert [fs.name for _, fs, _ in active] == [
        "mag_left", "fb_mag_left", "wvlt_approx",
    ]


def test_overlay_resample_clip_outside_zero_fills_frequency_gaps():
    src = np.array([[1.0], [2.0]], dtype=np.float32)
    out = bass_viewer._overlay_resample_to_grid(
        src,
        src_x=np.array([0.0]),
        src_y=np.array([30.0, 60.0]),
        dst_x=np.array([0.0]),
        dst_y=np.array([10.0, 30.0, 60.0, 70.0]),
        clip_outside=True,
    )

    assert out.shape == (4, 1)
    assert np.allclose(out[:, 0], np.array([0.0, 1.0, 2.0, 0.0],
                                           dtype=np.float32))


def test_project_filterbank_to_overlay_uses_band_frequency_extents():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.t_start = 0.0
    viewer.t_end = 1.0
    viewer._fb_bands = [
        bass_viewer.BandDef(0.0, 10.0, "LP"),
        bass_viewer.BandDef(10.0, 20.0, "Band"),
    ]

    data = np.array([
        [1.0, 1.0],
        [10.0, 10.0],
    ], dtype=np.float32)
    projected = viewer._project_filterbank_to_overlay(
        data,
        dst_x=np.array([0.0, 1.0], dtype=np.float64),
        dst_y=np.array([bass_viewer.CAFLS_SEISMIC_FLOOR, 5.0, 15.0],
                       dtype=np.float64),
    )

    expected = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [10.0, 10.0],
    ], dtype=np.float32)
    assert np.allclose(projected, expected)


def test_overlay_grid_shape_caps_to_gl_texture_limit():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.n_bins = 106
    viewer.n_frames = 38214
    viewer._wv_n_levels = 0
    viewer._wv_n_frames = 0
    viewer._fb_bands = []
    viewer._raw_arrays = {}
    viewer._gl_max_texture_size = 32768

    assert viewer._overlay_grid_shape() == (106, 32768)


def test_overlay_grid_shape_uses_tallest_source_without_crushing_rows():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.n_bins = 64
    viewer.n_frames = 1024
    viewer._wv_n_levels = 96
    viewer._wv_n_frames = 0
    viewer._fb_bands = [object()] * 75
    viewer._raw_arrays = {}
    viewer._gl_max_texture_size = 32768

    assert viewer._overlay_grid_shape() == (96, 1024)


def test_build_field_sources_keeps_fb_fields_gated_by_has_fb():
    sources = bass_viewer._build_field_sources(
        has_onset=False, has_fb=False, has_wvlt=False)
    fb_fields = {s.name: s for s in sources if s.category == "fb"}

    assert "fb_mag_left" in fb_fields
    assert fb_fields["fb_mag_left"].available is False


def test_uses_progressive_scan_includes_hybrid_tf():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.display_mode = "tf"
    viewer.tf_source = "hybrid"

    assert viewer._uses_progressive_scan() is True


def test_hybrid_visual_fb_bands_exclude_lp_hp_gutters():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.hybrid_view = bass_viewer.HybridViewConfig(hide_gutter_bands=True)
    viewer._fb_bands = [
        bass_viewer.BandDef(0.0, 10.0, "LP"),
        bass_viewer.BandDef(10.0, 20.0, "B1"),
        bass_viewer.BandDef(20.0, 40.0, "B2"),
        bass_viewer.BandDef(40.0, 80.0, "HP"),
    ]

    visual = viewer._hybrid_visual_fb_bands()

    assert [idx for idx, _ in visual] == [1, 2]
    assert [band.label for _, band in visual] == ["B1", "B2"]


def test_project_filterbank_to_hybrid_skips_gutters_and_applies_mix():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.hybrid_view = bass_viewer.HybridViewConfig(
        hide_gutter_bands=True, fb_mix=0.5)
    viewer.t_start = 0.0
    viewer.t_end = 1.0
    viewer._fb_bands = [
        bass_viewer.BandDef(0.0, 10.0, "LP"),
        bass_viewer.BandDef(10.0, 20.0, "B1"),
        bass_viewer.BandDef(20.0, 40.0, "B2"),
        bass_viewer.BandDef(40.0, 80.0, "HP"),
    ]

    data = np.array([
        [1.0, 1.0],
        [10.0, 10.0],
        [20.0, 20.0],
        [100.0, 100.0],
    ], dtype=np.float32)

    out = viewer._project_filterbank_to_hybrid(
        data,
        dst_x=np.array([0.0, 1.0], dtype=np.float64),
        dst_y=np.array([5.0, 15.0, 30.0, 60.0], dtype=np.float64),
    )

    expected = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
        [10.0, 10.0],
        [0.0, 0.0],
    ], dtype=np.float32)
    assert np.allclose(out, expected)


def test_hybrid_scale_split_uses_saved_center_and_edges():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer._fb_bands = []
    viewer._fb_center_hz = 0.0
    viewer._freqs = np.array([5.0, 200.0], dtype=np.float64)
    viewer._wv_freqs = np.array([bass_viewer.CAFLS_SEISMIC_FLOOR, 80.0],
                                dtype=np.float64)
    viewer._npz = None
    viewer.synth = None
    viewer._hybrid_meta_loaded = True
    viewer._hybrid_center_hz = 30.0
    viewer._hybrid_cqt_low_edge_hz = 12.0
    viewer._hybrid_cwt_high_edge_hz = 75.0

    assert viewer._hybrid_scale_split_freq() == 30.0
    assert viewer._hybrid_overlap_edges() == (30.0, 12.0, 75.0)


def test_hybrid_scale_split_prefers_filterbank_center_when_meta_missing():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.hybrid_view = bass_viewer.HybridViewConfig()
    viewer._fb_bands = bass_viewer.FilterBankDecomposition.bands_from_cafls(
        anchor=96.0, bpo=12, banded_width=1, sr=4096)
    viewer._fb_center_hz = bass_viewer._infer_fb_center_hz_from_bands(
        viewer._fb_bands)
    viewer._freqs = np.array([], dtype=np.float64)
    viewer._wv_freqs = None
    viewer._npz = None
    viewer.synth = None
    viewer._hybrid_meta_loaded = False
    viewer._hybrid_center_hz = 30.87
    viewer._hybrid_cqt_low_edge_hz = 30.87
    viewer._hybrid_cwt_high_edge_hz = 30.87

    assert np.isclose(viewer._hybrid_scale_split_freq(), 96.0, rtol=1e-6)


def test_hybrid_field_chunk_tapers_wavelet_above_center():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.hybrid_view = bass_viewer.HybridViewConfig(overlap_alpha_floor=0.25)
    viewer.t_start = 0.0
    viewer.t_end = 1.0
    viewer.t_dur = 1.0
    viewer.n_bins = 4
    viewer.n_frames = 4
    viewer._wv_n_levels = 2
    viewer._wv_n_frames = 2
    viewer._fb_bands = []
    viewer._hybrid_meta_loaded = True
    viewer._hybrid_center_hz = 20.0
    viewer._hybrid_cqt_low_edge_hz = 5.0
    viewer._hybrid_cwt_high_edge_hz = 80.0
    viewer._freqs = np.array([5.0, 200.0], dtype=np.float64)
    viewer._wv_freqs = np.array([bass_viewer.CAFLS_SEISMIC_FLOOR, 80.0],
                                dtype=np.float64)
    viewer.times = np.array([], dtype=np.float64)
    viewer._npz = None
    viewer.synth = None
    viewer._raw_arrays = {
        "wvlt_approx": np.ones((2, 2), dtype=np.float32),
    }
    viewer._resolve_field_array = lambda name: viewer._raw_arrays.get(name)

    out = viewer._hybrid_field_chunk(
        "wvlt_approx", 0.0, 4.0, 0.0, 4.0, 0, 1, 1, 16)

    col = out[:, 0]
    assert out.shape == (16, 1)
    assert col[0] > 0.95
    assert np.any((col > 0.0) & (col < 1.0))
    assert np.any(col == 0.0)


def test_hybrid_field_chunk_tapers_cqt_below_center():
    viewer = bass_viewer.SpectrogramViewer.__new__(bass_viewer.SpectrogramViewer)
    viewer.hybrid_view = bass_viewer.HybridViewConfig(overlap_alpha_floor=0.25)
    viewer.t_start = 0.0
    viewer.t_end = 1.0
    viewer.t_dur = 1.0
    viewer.n_bins = 4
    viewer.n_frames = 2
    viewer._wv_n_levels = 2
    viewer._wv_n_frames = 2
    viewer._fb_bands = []
    viewer._hybrid_meta_loaded = True
    viewer._hybrid_center_hz = 20.0
    viewer._hybrid_cqt_low_edge_hz = 5.0
    viewer._hybrid_cwt_high_edge_hz = 80.0
    viewer._freqs = np.array([5.0, 200.0], dtype=np.float64)
    viewer._wv_freqs = np.array([bass_viewer.CAFLS_SEISMIC_FLOOR, 80.0],
                                dtype=np.float64)
    viewer.times = np.array([0.0, 1.0], dtype=np.float64)
    viewer._npz = None
    viewer.synth = None
    viewer._raw_arrays = {
        "mag_left": np.ones((2, 2), dtype=np.float32),
    }
    viewer._resolve_field_array = lambda name: viewer._raw_arrays.get(name)

    out = viewer._hybrid_field_chunk(
        "mag_left", 0.0, 2.0, 0.0, 4.0, 0, 1, 1, 16)

    col = out[:, 0]
    assert out.shape == (16, 1)
    assert col[0] == 0.0
    assert np.any((col > 0.0) & (col < 1.0))
    assert col[-1] > 0.95
