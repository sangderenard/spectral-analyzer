"""Patch AMRGLComputeBackend.__init__, setup_plate, and step() in acoustic_amr.py."""

with open('acoustic_amr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ── Phase 1: Replace __init__ signature and body up through shader compile ─
old_init_start = content.index('    def __init__(self, grid: AcousticAMRGrid, c: float = 343.0, rho_air: float = 1.21):')
# Find end of __init__ by finding 'self._bulk = '
old_init_end_marker = '        # Precomputed step uniforms\n        self._dt_over_rho = float(self._dt / rho_air)\n        self._bulk = float(rho_air * c * c * self._dt)\n'
old_init_end = content.index(old_init_end_marker) + len(old_init_end_marker)
old_init = content[old_init_start:old_init_end]

new_init = '''    def __init__(
        self,
        grid: AcousticAMRGrid,
        c: float = 343.0,
        rho_air: float = 1.21,
        gradient_order: int = 2,
    ):
        if gradient_order not in (2, 8):
            raise ValueError(f"gradient_order must be 2 or 8, got {gradient_order!r}")
        self.gradient_order = int(gradient_order)

        with _amr_step_span(None, 0.0, "amr.gl.init",
                            "create OpenGL AMR backend", f"cells={grid.n_cells} faces={grid.n_faces}"):
            try:
                from OpenGL.GL import GL_VERSION  # noqa: F401 — presence check only
            except ImportError as exc:
                raise RuntimeError(
                    "PyOpenGL is required for AMRGLComputeBackend. "
                    "Install with: pip install PyOpenGL"
                ) from exc

            self.grid = grid
            self.c = float(c)
            self.rho_air = float(rho_air)

        n_cells = grid.n_cells
        n_faces = grid.n_faces

        # ── CFL timestep: matches C++ amr_create convention ──────────────────
        # order 2 → CFL 0.77;  order 8 → CFL 0.10
        cfl = 0.10 if gradient_order == 8 else 0.77
        min_dx = float(2.0 * grid.cell_half_sizes.min())
        self._dt = cfl * min_dx / (c * math.sqrt(3.0))

        # ── Precomputed topology arrays ───────────────────────────────────────
        face_flux_coef = (grid.face_area * grid.face_open_fraction).astype(np.float32)

        cell_inv_denom = np.zeros(n_cells, dtype=np.float32)
        acoustic_mask = ((grid.cell_types == 0) | (grid.cell_types == 3)) & (grid.open_volume_fraction > 0)
        cell_inv_denom[acoustic_mask] = (
            1.0 / (grid.cell_volumes[acoustic_mask] * np.maximum(grid.open_volume_fraction[acoustic_mask], 1e-12))
        ).astype(np.float32)

        # ── CSR per-cell face lists with pre-baked weights (Phase 3) ─────────
        # csr_face_weight[k] = ±face_flux_coef[face] — sign folded in at setup.
        # This eliminates csr_face_sign + face_flux_coef from the hot dispatch.
        csr_starts = np.zeros(n_cells + 1, dtype=np.int32)
        fn = grid.face_cell_neg.astype(np.int32)
        fp = grid.face_cell_pos.astype(np.int32)
        np.add.at(csr_starts[1:], fn, 1)
        np.add.at(csr_starts[1:], fp, 1)
        np.cumsum(csr_starts, out=csr_starts)
        total = int(csr_starts[n_cells])
        csr_face_idx    = np.empty(total, dtype=np.int32)
        csr_face_weight = np.empty(total, dtype=np.float32)
        fill = csr_starts[:-1].copy()
        for f in range(n_faces):
            a, b = int(fn[f]), int(fp[f])
            ka = fill[a]; fill[a] += 1
            csr_face_idx[ka]    = f;  csr_face_weight[ka] = +float(face_flux_coef[f])
            kb = fill[b]; fill[b] += 1
            csr_face_idx[kb]    = f;  csr_face_weight[kb] = -float(face_flux_coef[f])

        # ── Order-2: build per-face neg/pos index and inv-distance arrays ─────
        face_neg_i32  = grid.face_cell_neg.astype(np.int32)
        face_pos_i32  = grid.face_cell_pos.astype(np.int32)
        face_inv_dist = (1.0 / np.maximum(grid.face_distance, 1e-30)).astype(np.float32)

        # ── Order-8 Fornberg stencil: built only when gradient_order == 8 ─────
        if self.gradient_order == 8:
            with _amr_step_span(None, 0.0, "amr.gl.init.stencil",
                                "build order-8 Fornberg stencil", f"faces={n_faces}"):
                s_cells, s_coeff = _build_face_stencil(grid, sw=4)
            self._buf_stencil_cells = _make_ssbo(s_cells, binding=0)
            self._buf_stencil_coeff = _make_ssbo(s_coeff, binding=0)
        else:
            # order 2: stencil not needed; attributes stay None
            self._buf_stencil_cells = None
            self._buf_stencil_coeff = None

        # ── Upload topology SSBOs (persistent, never change) ──────────────────
        self._buf_pressure      = _make_ssbo_zeros(n_cells * 4)
        self._buf_velocity      = _make_ssbo_zeros(n_faces * 4)
        # Phase 2: no intermediate div_flux buffer; fused shader handles it inline.
        self._buf_csr_starts    = _make_ssbo(csr_starts,      binding=0)
        self._buf_csr_idx       = _make_ssbo(csr_face_idx,    binding=0)
        self._buf_csr_weight    = _make_ssbo(csr_face_weight, binding=0)
        self._buf_inv_denom     = _make_ssbo(cell_inv_denom,  binding=0)
        # Phase 1 (order 2): per-face neg/pos cell index and inverse distance
        self._buf_face_neg      = _make_ssbo(face_neg_i32,  binding=0)
        self._buf_face_pos      = _make_ssbo(face_pos_i32,  binding=0)
        self._buf_face_inv_dist = _make_ssbo(face_inv_dist, binding=0)
        # PML damping arrays — initialised to identity (no absorption).
        # Call setup_border() after construction to activate PML.
        self._buf_face_v_damp   = _make_ssbo(np.ones(n_faces, dtype=np.float32), binding=0)
        self._buf_p_damp        = _make_ssbo(np.ones(n_cells, dtype=np.float32), binding=0)
        self._buf_p_src_coeff   = _make_ssbo(np.ones(n_cells, dtype=np.float32), binding=0)

        self._n_cells = n_cells
        self._n_faces = n_faces

        # ── Plate state (populated in setup_plate) ────────────────────────────
        self._plate_active        = False
        self._n_active_plate      = 0
        self._buf_plate_w         = None
        self._buf_plate_wp        = None
        self._buf_plate_wn        = None
        self._buf_ext_force       = None
        self._buf_active_idx      = None
        self._buf_cell_above      = None
        self._buf_cell_below      = None
        self._buf_bc_starts       = None
        self._buf_bc_idx          = None
        self._buf_bc_sign         = None
        self._buf_bc_wgt          = None
        # Phase 5: L4 biharmonic cache buffers (allocated in setup_plate)
        self._buf_plate_L4_prev   = None
        self._buf_plate_L4_curr   = None
        # Phase 6: per-face plate ownership map (allocated in setup_plate)
        self._buf_plate_owner     = None
        self._buf_plate_owner_sign = None
        self._buf_plate_owner_wgt  = None
        self._plate_uniforms: dict = {}

        # ── Mic state (populated in setup_mics) ───────────────────────────────
        self._n_mics = 0
        self._buf_mic_out    = None
        self._buf_mc_idx     = None
        self._buf_mc_wgt     = None
        self._buf_mf_idx     = None
        self._buf_mf_wx      = None
        self._buf_mf_wy      = None
        self._buf_mf_wz      = None
        self._buf_mic_starts = None
        self._mic_ring_capacity = max(1, int(os.environ.get("SPECTRAL_GL_MIC_RING", "256")))
        self._mic_ring_write = 0
        self._mic_ring_pending = 0
        # Phase 7: offline mic output (allocated in setup_offline_mic_output)
        self._offline_mic_steps = 0
        self._offline_mic_write = 0

        # ── Compile / cache shaders (Phase 8) ─────────────────────────────────
        self._prog_vel_order2 = get_or_compile_compute("velocity_order2", _VELOCITY_UPDATE_ORDER2_GLSL)
        self._prog_vel_order8 = get_or_compile_compute("velocity_order8", _VELOCITY_UPDATE_GLSL)
        self._prog_vel        = self._prog_vel_order8 if gradient_order == 8 else self._prog_vel_order2
        self._prog_div_pres   = get_or_compile_compute("divergence_pressure", _DIVERGENCE_PRESSURE_GLSL)
        self._prog_plate      = get_or_compile_compute("plate_step", _PLATE_STEP_GLSL)
        self._prog_plate_commit = get_or_compile_compute("plate_commit", _PLATE_COMMIT_GLSL)
        self._prog_plate_bc   = get_or_compile_compute("plate_bc", _PLATE_BC_GLSL)
        self._prog_mic        = get_or_compile_compute("mic_sample", _MIC_SAMPLE_GLSL)

        # Precomputed step uniforms
        self._dt_over_rho = float(self._dt / rho_air)
        self._bulk = float(rho_air * c * c * self._dt)
'''

assert old_init in content, "Old __init__ not found"
content = content.replace(old_init, new_init, 1)
print("Phase 1+2+3+4+5+6+8 __init__ replaced: OK")

with open('acoustic_amr.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Written.")
