"""Patch setup_plate() and step() in AMRGLComputeBackend."""

with open('acoustic_amr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ── Patch setup_plate: add L4 buffers + per-face owner map (Phase 5+6) ────
OLD_SETUP_PLATE_TAIL = '''        self._buf_bc_starts = _make_ssbo(bc_starts, binding=0)
        self._buf_bc_idx    = _make_ssbo(bc_idx,    binding=0)
        self._buf_bc_sign   = _make_ssbo(bc_sign,   binding=0)
        self._buf_bc_wgt    = _make_ssbo(bc_wgt,    binding=0)

        # Plate uniforms (precomputed, constant after setup)'''

NEW_SETUP_PLATE_TAIL = '''        self._buf_bc_starts = _make_ssbo(bc_starts, binding=0)
        self._buf_bc_idx    = _make_ssbo(bc_idx,    binding=0)
        self._buf_bc_sign   = _make_ssbo(bc_sign,   binding=0)
        self._buf_bc_wgt    = _make_ssbo(bc_wgt,    binding=0)

        # Phase 5: L4 biharmonic cache (one float per plate node, full grid size)
        self._buf_plate_L4_prev = _make_ssbo_zeros(N * 4)
        self._buf_plate_L4_curr = _make_ssbo_zeros(N * 4)

        # Phase 6: build per-face ownership map from merged BC CSR.
        # plate_owner[f] = active-node index that owns face f, or -1 if none.
        n_faces_total = self._n_faces
        plate_owner      = np.full(n_faces_total, -1, dtype=np.int32)
        plate_owner_sign = np.zeros(n_faces_total, dtype=np.float32)
        plate_owner_wgt  = np.zeros(n_faces_total, dtype=np.float32)
        for nd in range(n_active):
            for k in range(int(bc_starts[nd]), int(bc_starts[nd + 1])):
                f = int(bc_idx[k])
                if plate_owner[f] != -1:
                    raise RuntimeError(
                        f"Face {f} has duplicate plate owners: {plate_owner[f]} and {nd}"
                    )
                plate_owner[f]      = nd
                plate_owner_sign[f] = float(bc_sign[k])
                plate_owner_wgt[f]  = float(bc_wgt[k])
        self._buf_plate_owner      = _make_ssbo(plate_owner,      binding=0)
        self._buf_plate_owner_sign = _make_ssbo(plate_owner_sign, binding=0)
        self._buf_plate_owner_wgt  = _make_ssbo(plate_owner_wgt,  binding=0)

        # Plate uniforms (precomputed, constant after setup)'''

assert OLD_SETUP_PLATE_TAIL in content, "setup_plate tail not found"
content = content.replace(OLD_SETUP_PLATE_TAIL, NEW_SETUP_PLATE_TAIL, 1)
print("setup_plate L4+owner map: OK")

# ── Rewrite step() completely ─────────────────────────────────────────────
OLD_STEP = '''    def step(self, n_steps: int = 1) -> None:
        """Advance AMR FDTD by ``n_steps`` steps on the GPU."""
        from OpenGL.GL import (
            glUseProgram, glDispatchCompute, glMemoryBarrier,
            GL_SHADER_STORAGE_BARRIER_BIT,
        )
        n_cells  = self._n_cells
        n_faces  = self._n_faces
        L        = self._LOCAL
        n_active = self._n_active_plate

        with _amr_step_span(None, 0.0, "amr.gl.step",
                            "step OpenGL AMR backend",
                            f"n_steps={int(n_steps)} cells={n_cells} faces={n_faces} active_plate={n_active}"):
         for _ in range(n_steps):
            # ── 1. Velocity update ────────────────────────────────────
            with _amr_step_span(None, 0.0, "amr.gl.step.velocity_update",
                                "dispatch AMR GL velocity update", f"faces={n_faces} groups={_gl_ceil_div(n_faces, L)}"):
                glUseProgram(self._prog_vel)
                self._bind_base(self._buf_pressure,      0)
                self._bind_base(self._buf_velocity,      1)
                self._bind_base(self._buf_stencil_cells, 2)
                self._bind_base(self._buf_stencil_coeff, 3)
                self._bind_base(self._buf_face_v_damp,   4)
                _uniform_f(self._prog_vel, "dt_over_rho", self._dt_over_rho)
                _uniform_i(self._prog_vel, "n_faces", n_faces)
                glDispatchCompute(_gl_ceil_div(n_faces, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 2. Plate BC (if active) ───────────────────────────────
            if self._plate_active:
                with _amr_step_span(None, 0.0, "amr.gl.step.plate_bc",
                                    "dispatch AMR GL plate BC", f"active_nodes={n_active}"):
                    glUseProgram(self._prog_plate_bc)
                    self._bind_base(self._buf_plate_w,  0)
                    self._bind_base(self._buf_plate_wp, 1)
                    self._bind_base(self._buf_velocity, 2)
                    self._bind_base(self._buf_active_idx, 3)
                    self._bind_base(self._buf_bc_starts,  4)
                    self._bind_base(self._buf_bc_idx,     5)
                    self._bind_base(self._buf_bc_sign,    6)
                    self._bind_base(self._buf_bc_wgt,     7)
                    _uniform_i(self._prog_plate_bc, "N_active", n_active)
                    _uniform_f(self._prog_plate_bc, "inv_dt",
                               self._plate_uniforms["inv_dt"])
                    glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 3. Divergence CSR ─────────────────────────────────────
            with _amr_step_span(None, 0.0, "amr.gl.step.divergence_csr",
                                "dispatch AMR GL divergence CSR", f"cells={n_cells} groups={_gl_ceil_div(n_cells, L)}"):
                glUseProgram(self._prog_div)
                self._bind_base(self._buf_velocity,  0)
                self._bind_base(self._buf_div_flux,  1)
                self._bind_base(self._buf_csr_starts, 2)
                self._bind_base(self._buf_csr_idx,   3)
                self._bind_base(self._buf_csr_sign,  4)
                self._bind_base(self._buf_flux_coef, 5)
                _uniform_i(self._prog_div, "n_cells", n_cells)
                glDispatchCompute(_gl_ceil_div(n_cells, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 4. Pressure update (exact CPML) ──────────────────────
            with _amr_step_span(None, 0.0, "amr.gl.step.pressure_update",
                                "dispatch AMR GL pressure update", f"cells={n_cells}"):
                glUseProgram(self._prog_pres)
                self._bind_base(self._buf_pressure,    0)
                self._bind_base(self._buf_div_flux,    1)
                self._bind_base(self._buf_inv_denom,   2)
                self._bind_base(self._buf_p_damp,      3)
                self._bind_base(self._buf_p_src_coeff, 4)
                _uniform_f(self._prog_pres, "bulk", self._bulk)
                _uniform_i(self._prog_pres, "n_cells", n_cells)
                glDispatchCompute(_gl_ceil_div(n_cells, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 5. Plate step (if active) ─────────────────────────────
            if self._plate_active:
                with _amr_step_span(None, 0.0, "amr.gl.step.plate_step",
                                    "dispatch AMR GL plate step", f"active_nodes={n_active}"):
                    pu = self._plate_uniforms
                    glUseProgram(self._prog_plate)
                    self._bind_base(self._buf_plate_w,    0)
                    self._bind_base(self._buf_plate_wp,   1)
                    self._bind_base(self._buf_plate_wn,   2)
                    self._bind_base(self._buf_ext_force,  3)
                    self._bind_base(self._buf_active_idx, 4)
                    self._bind_base(self._buf_pressure,   5)
                    self._bind_base(self._buf_cell_above, 6)
                    self._bind_base(self._buf_cell_below, 7)
                    _uniform_i(self._prog_plate, "N_active", n_active)
                    _uniform_i(self._prog_plate, "Nx",       pu["Nx"])
                    _uniform_i(self._prog_plate, "Ny",       pu["Ny"])
                    _uniform_f(self._prog_plate, "dx2",      pu["dx2"])
                    _uniform_f(self._prog_plate, "dx4",      pu["dx4"])
                    _uniform_f(self._prog_plate, "coeff_D0", pu["coeff_D0"])
                    _uniform_f(self._prog_plate, "coeff_Dp", pu["coeff_Dp"])
                    _uniform_f(self._prog_plate, "damp_fwd", pu["damp_fwd"])
                    _uniform_f(self._prog_plate, "damp_bwd", pu["damp_bwd"])
                    _uniform_f(self._prog_plate, "dt2_inv_rh", pu["dt2_inv_rh"])
                    glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

                # ── 6. Plate commit (w ← w_new, clear ext_force) ─────
                with _amr_step_span(None, 0.0, "amr.gl.step.plate_commit",
                                    "dispatch AMR GL plate commit", f"active_nodes={n_active}"):
                    glUseProgram(self._prog_plate_commit)
                    self._bind_base(self._buf_plate_w,    0)
                    self._bind_base(self._buf_plate_wp,   1)
                    self._bind_base(self._buf_plate_wn,   2)
                    self._bind_base(self._buf_ext_force,  3)
                    self._bind_base(self._buf_active_idx, 4)
                    _uniform_i(self._prog_plate_commit, "N_active", n_active)
                    glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += n_steps'''

NEW_STEP = '''    def step(self, n_steps: int = 1) -> None:
        """Advance AMR FDTD by ``n_steps`` steps on the GPU."""
        from OpenGL.GL import (
            glUseProgram, glDispatchCompute, glMemoryBarrier,
            GL_SHADER_STORAGE_BARRIER_BIT,
        )
        n_cells  = self._n_cells
        n_faces  = self._n_faces
        L        = self._LOCAL
        n_active = self._n_active_plate
        plate_bc_enabled = 1 if (self._plate_active and self._buf_plate_owner is not None) else 0

        for _ in range(n_steps):
            # ── 1. Velocity update (order 2 or 8, plate BC fused in) ──────────
            glUseProgram(self._prog_vel)
            self._bind_base(self._buf_pressure,    0)
            self._bind_base(self._buf_velocity,    1)
            if self.gradient_order == 8:
                self._bind_base(self._buf_stencil_cells, 2)
                self._bind_base(self._buf_stencil_coeff, 3)
                self._bind_base(self._buf_face_v_damp,   4)
            else:
                self._bind_base(self._buf_face_neg,      2)
                self._bind_base(self._buf_face_pos,      3)
                self._bind_base(self._buf_face_inv_dist, 4)
                self._bind_base(self._buf_face_v_damp,   5)
            if plate_bc_enabled:
                self._bind_base(self._buf_plate_owner,      6)
                self._bind_base(self._buf_plate_owner_sign, 7)
                self._bind_base(self._buf_plate_owner_wgt,  8)
                self._bind_base(self._buf_active_idx,       9)
                self._bind_base(self._buf_plate_w,         10)
                self._bind_base(self._buf_plate_wp,        11)
            _uniform_f(self._prog_vel, "dt_over_rho", self._dt_over_rho)
            _uniform_i(self._prog_vel, "n_faces",     n_faces)
            _uniform_i(self._prog_vel, "plate_bc_enabled", plate_bc_enabled)
            if plate_bc_enabled:
                _uniform_f(self._prog_vel, "plate_inv_dt",
                           self._plate_uniforms["inv_dt"])
            glDispatchCompute(_gl_ceil_div(n_faces, L), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 2. Divergence + pressure (fused, Phase 2+3) ───────────────────
            glUseProgram(self._prog_div_pres)
            self._bind_base(self._buf_velocity,    0)
            self._bind_base(self._buf_pressure,    1)
            self._bind_base(self._buf_csr_starts,  2)
            self._bind_base(self._buf_csr_idx,     3)
            self._bind_base(self._buf_csr_weight,  4)
            self._bind_base(self._buf_inv_denom,   5)
            self._bind_base(self._buf_p_damp,      6)
            self._bind_base(self._buf_p_src_coeff, 7)
            _uniform_f(self._prog_div_pres, "bulk",    self._bulk)
            _uniform_i(self._prog_div_pres, "n_cells", n_cells)
            glDispatchCompute(_gl_ceil_div(n_cells, L), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 3. Plate step + commit (if active) ────────────────────────────
            if self._plate_active:
                pu = self._plate_uniforms
                # 3a. Plate step
                glUseProgram(self._prog_plate)
                self._bind_base(self._buf_plate_w,      0)
                self._bind_base(self._buf_plate_wp,     1)
                self._bind_base(self._buf_plate_wn,     2)
                self._bind_base(self._buf_ext_force,    3)
                self._bind_base(self._buf_active_idx,   4)
                self._bind_base(self._buf_pressure,     5)
                self._bind_base(self._buf_cell_above,   6)
                self._bind_base(self._buf_cell_below,   7)
                self._bind_base(self._buf_plate_L4_prev, 8)
                self._bind_base(self._buf_plate_L4_curr, 9)
                _uniform_i(self._prog_plate, "N_active",   n_active)
                _uniform_i(self._prog_plate, "Nx",         pu["Nx"])
                _uniform_i(self._prog_plate, "Ny",         pu["Ny"])
                _uniform_f(self._prog_plate, "dx2",        pu["dx2"])
                _uniform_f(self._prog_plate, "dx4",        pu["dx4"])
                _uniform_f(self._prog_plate, "coeff_D0",   pu["coeff_D0"])
                _uniform_f(self._prog_plate, "coeff_Dp",   pu["coeff_Dp"])
                _uniform_f(self._prog_plate, "damp_fwd",   pu["damp_fwd"])
                _uniform_f(self._prog_plate, "damp_bwd",   pu["damp_bwd"])
                _uniform_f(self._prog_plate, "dt2_inv_rh", pu["dt2_inv_rh"])
                glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

                # 3b. Plate commit (w ← w_new, rotate L4 cache)
                glUseProgram(self._prog_plate_commit)
                self._bind_base(self._buf_plate_w,      0)
                self._bind_base(self._buf_plate_wp,     1)
                self._bind_base(self._buf_plate_wn,     2)
                self._bind_base(self._buf_ext_force,    3)
                self._bind_base(self._buf_active_idx,   4)
                self._bind_base(self._buf_plate_L4_prev, 8)
                self._bind_base(self._buf_plate_L4_curr, 9)
                _uniform_i(self._prog_plate_commit, "N_active", n_active)
                glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += n_steps'''

assert OLD_STEP in content, "Old step() not found"
content = content.replace(OLD_STEP, NEW_STEP, 1)
print("step() rewrite: OK")

with open('acoustic_amr.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Written.")
