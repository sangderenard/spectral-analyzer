/* ─────────────────────────────────────────────────────────────────────────────
 * t5_schedule_connect.cpp.inc  —  drain-when-empty scored dispatch for T5.
 *
 * THIS is the file that actually makes prioritization happen at runtime.
 * It REPLACES the flat nested cbatch/lbatch loop in ray_tracer.cpp's T5 pass
 * with: score every (cam-batch, light-batch) unit, sort units by combined score
 * S descending, then dispatch them in that order.  Nothing is discarded — low-S
 * units simply dispatch last, so a render that is interrupted (cancel / time
 * budget) has already spent its cycles on the highest-yield work, and a render
 * that runs to completion produces the identical image, just reordered.
 *
 * Apply this OR batch_score_host.cpp.inc — not both.  The host harness is the
 * inspection-only variant (dumps scores.bin, changes nothing).  This file is
 * the production path and folds the same scoring dispatch in.
 *
 * ── PREREQUISITES (same one-time setup as batch_score_host.cpp.inc) ──────────
 *   - prog_batch_score loaded (non-shadow compute program) + cached uniform
 *     locations (u_bs_*) listed in batch_score_host.cpp.inc.
 *   - ssbo_batch_score / cap_batch_score declared.
 *   - Tunables in scope (add as cfg fields or constants):
 *       float   T5_SCORE_wA, T5_SCORE_wB, T5_SCORE_wC;   // start 1,0,0
 *       float   T5_SCORE_nA, T5_SCORE_nB, T5_SCORE_nC;   // start 1,1,1
 *       uint32_t T5_SCORE_KSAMPLES;                      // start 256
 *
 * ── WHERE THIS GOES ──────────────────────────────────────────────────────────
 *   Replace the entire original block:
 *
 *       const uint32_t n_cbatches = (n_tile_cam + T5_CAM_BATCH - 1u)/T5_CAM_BATCH;
 *       ... fprintf("[T5-gpu tile...] cam=... cbatches=...") ...
 *       for (uint32_t c = 0; c < n_cbatches && gpu_ok; ++c) {
 *           ...
 *           for (uint32_t b = 0; b < n_batches && gpu_ok && !cancel; ++b) {
 *               ... dispatch ...
 *           }
 *       }
 *
 *   (ray_tracer.cpp ~lines 11374–11436) with the body below.  Everything before
 *   it (tile cam filtering, SSBO grow, cam-vert upload, pixel-accum zero, the
 *   par.tile_* setup) and everything after it (pixel readback / scatter) stays.
 * ───────────────────────────────────────────────────────────────────────────*/

{
    const uint32_t n_cbatches = (n_tile_cam + T5_CAM_BATCH - 1u) / T5_CAM_BATCH;
    const uint32_t n_lbatches = n_batches;
    const size_t   n_units    = (size_t)n_cbatches * n_lbatches;

    /* ── 1. Score every (c,b) unit (cheap prepass; no MIS, no shadow). ──────── */
    std::vector<float> unit_S(n_units, 0.0f);   /* combined score per unit  */
    bool scored = false;

    if (prog_batch_score && n_units > 0) {
        const GLsizeiptr need = (GLsizeiptr)(n_units * 4u * sizeof(float));
        if (need > cap_batch_score) {
            cap_batch_score = need * 2;
            ensure_ssbo(ssbo_batch_score, cap_batch_score);
        }

        glc_UseProgram(prog_batch_score);
        bind_ssbo(ssbo_t5_light,    0);
        bind_ssbo(ssbo_t5_cam,      1);
        bind_ssbo(ssbo_batch_score, 9);

        if (u_bs_ncam   >= 0) glc_Uniform1ui(u_bs_ncam,   n_tile_cam);
        if (u_bs_nlight >= 0) glc_Uniform1ui(u_bs_nlight, n_light);
        if (u_bs_cbsz   >= 0) glc_Uniform1ui(u_bs_cbsz,   T5_CAM_BATCH);
        if (u_bs_lbsz   >= 0) glc_Uniform1ui(u_bs_lbsz,   T5_LIGHT_BATCH);
        if (u_bs_nbands >= 0) glc_Uniform1i (u_bs_nbands, par.n_bands);
        if (u_bs_ksamp  >= 0) glc_Uniform1ui(u_bs_ksamp,  T5_SCORE_KSAMPLES);
        if (u_bs_mingeom>= 0) glc_Uniform1f (u_bs_mingeom, par.min_geom);
        if (u_bs_seed   >= 0) glc_Uniform1ui(u_bs_seed,   (GLuint)(tx*2654435761u ^ ty*40503u ^ 0x1234u));
        if (u_bs_wA >= 0) glc_Uniform1f(u_bs_wA, T5_SCORE_wA);
        if (u_bs_wB >= 0) glc_Uniform1f(u_bs_wB, T5_SCORE_wB);
        if (u_bs_wC >= 0) glc_Uniform1f(u_bs_wC, T5_SCORE_wC);
        if (u_bs_nA >= 0) glc_Uniform1f(u_bs_nA, T5_SCORE_nA);
        if (u_bs_nB >= 0) glc_Uniform1f(u_bs_nB, T5_SCORE_nB);
        if (u_bs_nC >= 0) glc_Uniform1f(u_bs_nC, T5_SCORE_nC);

        glc_DispatchCompute(n_cbatches, n_lbatches, 1u);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glFinish();

        std::vector<float> host_scores(n_units * 4u);
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_batch_score);
        glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                             (GLsizeiptr)(n_units * 4u * sizeof(float)),
                             host_scores.data());
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        for (size_t u = 0; u < n_units; ++u) unit_S[u] = host_scores[u * 4u + 3u];
        scored = true;
    }

    /* ── 2. Build dispatch order. ───────────────────────────────────────────── *
     * order[k] = unit index = c * n_lbatches + b.  Sorted by S DESCENDING when  *
     * scoring succeeded; otherwise falls back to the original row-major order    *
     * so behaviour is unchanged if the score program is missing.                */
    std::vector<uint32_t> order(n_units);
    for (size_t u = 0; u < n_units; ++u) order[u] = (uint32_t)u;
    if (scored) {
        std::stable_sort(order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) { return unit_S[a] > unit_S[b]; });
    }

    /* ── 3. Re-bind the T5 connection program + its SSBOs. ───────────────────── *
     * The scoring pass switched the active program and bound binding 9, so       *
     * restore T5's full binding set before the real connection dispatches.       */
    glc_UseProgram(prog_t5);
    bind_ssbo(ssbo_t5_light,    0);
    bind_ssbo(ssbo_t5_cam,      1);
    bind_ssbo(ssbo_t5_pix,      2);
    bind_ssbo(ssbo_t5_params,   3);
    bind_ssbo(ssbo_t5_spectral, 4);
    bind_ssbo(ssbo_bvh,         5);
    bind_ssbo(ssbo_tri_id,      6);
    bind_ssbo(ssbo_tri_full,    7);
    bind_ssbo(ssbo_mat_band,    8);
    if (uloc_t5_n_mats >= 0)
        glc_Uniform1i(uloc_t5_n_mats, ps.st ? ps.st->mat_n_mats : 0);

    fprintf(stderr,
        "[T5-gpu tile(%d,%d) %d/%d] cam=%u units=%zu (cbatch=%u x lbatch=%u) %s\n",
        tx, ty, tiles_done + 1, n_tiles_total, n_tile_cam, n_units,
        n_cbatches, n_lbatches, scored ? "score-sorted" : "row-major(fallback)");
    fflush(stderr);

    /* ── 4. Dispatch every unit in priority order. NEVER DISCARD. ───────────── *
     * Identical per-unit dispatch to the original nested loop; only the visit    *
     * order changes.  The TDR heartbeat (glFinish every 8 dispatches) and the    *
     * cancel check are preserved.                                                */
    uint32_t done = 0;
    for (size_t k = 0; k < n_units && gpu_ok
                       && !ps.cancel.load(std::memory_order_relaxed); ++k) {
        const uint32_t unit = order[k];
        const uint32_t c = unit / n_lbatches;
        const uint32_t b = unit % n_lbatches;

        const uint32_t cam_batch_actual =
            std::min(T5_CAM_BATCH, n_tile_cam - c * T5_CAM_BATCH);
        par.cam_offset  = c * T5_CAM_BATCH;
        par.n_cam_verts = par.cam_offset + cam_batch_actual;   /* upper bound */
        const GLuint wg_x = (cam_batch_actual + (uint32_t)T5_TILE_C - 1u)
                          / (uint32_t)T5_TILE_C;

        par.light_offset     = b * T5_LIGHT_BATCH;
        par.light_batch_size = std::min(T5_LIGHT_BATCH, n_light - par.light_offset);
        upload(ssbo_t5_params, &par, sizeof(T5GpuParams));
        const GLuint wg_y = (par.light_batch_size + (uint32_t)T5_TILE_L - 1u)
                          / (uint32_t)T5_TILE_L;

        glc_DispatchCompute(wg_x, wg_y, 1u);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glFlush();

        GLenum gl_err = gpu_check_dispatch("T5:connect", wg_x, wg_y, 1u);
        if (gl_err != GL_NO_ERROR) {
            fprintf(stderr,
                "[T5-gpu] GL error 0x%04x at unit %u/%zu (c=%u b=%u) tile(%d,%d) — TDR? aborting\n",
                (unsigned)gl_err, done, n_units, c, b, tx, ty);
            fflush(stderr);
            gpu_ok = false;
            break;
        }

        ++done;
        const bool is_heartbeat = ((done & 7u) == 0u) || (done == n_units);
        if (is_heartbeat) {
            glFinish();
            GLenum finish_err = glGetError();
            if (finish_err != GL_NO_ERROR) {
                fprintf(stderr,
                    "[T5-gpu] GL error 0x%04x after glFinish at unit %u/%zu tile(%d,%d) — device lost? aborting\n",
                    (unsigned)finish_err, done, n_units, tx, ty);
                fflush(stderr);
                gpu_ok = false;
                break;
            }
            const auto now = std::chrono::steady_clock::now();
            const float elapsed = std::chrono::duration<float>(now - tile_t0).count();
            fprintf(stderr, "[T5-gpu tile(%d,%d)] unit %u/%zu  %.1fs\n",
                    tx, ty, done, n_units, elapsed);
            fflush(stderr);
        }
    }
}
