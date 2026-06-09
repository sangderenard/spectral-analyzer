/* ─────────────────────────────────────────────────────────────────────────────
 * batch_score_host.cpp.inc  —  harness to dispatch batch_score.comp.glsl and
 * read the per-(cam-batch, light-batch) scores back to disk for inspection.
 *
 * This is the "scoring shader first" path: it does NOT yet reorder or skip any
 * T5 dispatch. It runs the prepass, dumps scores.bin, and (optionally) prints a
 * now/idle split preview so you can choose wA/wB/wC and a cut point before we
 * wire the drain-when-empty scheduler into the hot loop.
 *
 * ── One-time setup (next to where prog_t5 is created, ~line 9536) ───────────
 *
 *     GLuint prog_batch_score = 0;
 *     // Plain compute program — NO shadow BVH preamble needed (no shadow rays).
 *     if (!load_compute_program("batch_score.comp.glsl", prog_batch_score)) {
 *         fprintf(stderr, "[batch-score] not loaded — triage disabled\n");
 *     }
 *     // Cache uniform locations once:
 *     const GLint u_bs_ncam   = glGetUniformLocation(prog_batch_score, "n_cam_verts");
 *     const GLint u_bs_nlight = glGetUniformLocation(prog_batch_score, "n_light_verts");
 *     const GLint u_bs_cbsz   = glGetUniformLocation(prog_batch_score, "cam_batch_sz");
 *     const GLint u_bs_lbsz   = glGetUniformLocation(prog_batch_score, "light_batch_sz");
 *     const GLint u_bs_nbands = glGetUniformLocation(prog_batch_score, "n_bands");
 *     const GLint u_bs_ksamp  = glGetUniformLocation(prog_batch_score, "k_samples");
 *     const GLint u_bs_mingeom= glGetUniformLocation(prog_batch_score, "min_geom");
 *     const GLint u_bs_seed   = glGetUniformLocation(prog_batch_score, "seed");
 *     const GLint u_bs_wA     = glGetUniformLocation(prog_batch_score, "wA");
 *     const GLint u_bs_wB     = glGetUniformLocation(prog_batch_score, "wB");
 *     const GLint u_bs_wC     = glGetUniformLocation(prog_batch_score, "wC");
 *     const GLint u_bs_nA     = glGetUniformLocation(prog_batch_score, "normA");
 *     const GLint u_bs_nB     = glGetUniformLocation(prog_batch_score, "normB");
 *     const GLint u_bs_nC     = glGetUniformLocation(prog_batch_score, "normC");
 *     GLuint ssbo_batch_score = 0;
 *     GLsizeiptr cap_batch_score = 0;
 *
 * `load_compute_program` is the non-BVH twin of load_with_shadow_bvh; if you
 * don't have one, just compile+link the single stage the way load_with_shadow_bvh
 * does internally but skip the shadow preamble injection.
 *
 * ── Per-tile call: paste INSIDE the tile loop, AFTER n_tile_cam is known and
 *    BEFORE the `for (uint32_t c = 0; ...)` cbatch loop (~line 11374). ────────
 *    Reuses T5_CAM_BATCH / T5_LIGHT_BATCH / n_batches already in scope.
 * ───────────────────────────────────────────────────────────────────────────*/

if (prog_batch_score && n_tile_cam > 0) {
    const uint32_t n_cbatch = (n_tile_cam + T5_CAM_BATCH - 1u) / T5_CAM_BATCH;
    const uint32_t n_lbatch = n_batches;                 /* light batches, already computed */
    const size_t   n_units  = (size_t)n_cbatch * n_lbatch;
    const GLsizeiptr need   = (GLsizeiptr)(n_units * 4u * sizeof(float));

    /* Grow / (re)create the score SSBO. */
    if (need > cap_batch_score) {
        cap_batch_score = need * 2;
        ensure_ssbo(ssbo_batch_score, cap_batch_score);
    }

    glc_UseProgram(prog_batch_score);
    bind_ssbo(ssbo_t5_light,     0);   /* same light verts T5 will connect      */
    bind_ssbo(ssbo_t5_cam,       1);   /* tile-local cam verts already uploaded */
    bind_ssbo(ssbo_batch_score,  9);

    if (u_bs_ncam   >= 0) glc_Uniform1ui(u_bs_ncam,   n_tile_cam);
    if (u_bs_nlight >= 0) glc_Uniform1ui(u_bs_nlight, n_light);
    if (u_bs_cbsz   >= 0) glc_Uniform1ui(u_bs_cbsz,   T5_CAM_BATCH);
    if (u_bs_lbsz   >= 0) glc_Uniform1ui(u_bs_lbsz,   T5_LIGHT_BATCH);
    if (u_bs_nbands >= 0) glc_Uniform1i (u_bs_nbands, par.n_bands);
    if (u_bs_ksamp  >= 0) glc_Uniform1ui(u_bs_ksamp,  256u);          /* tune */
    if (u_bs_mingeom>= 0) glc_Uniform1f (u_bs_mingeom, par.min_geom);
    if (u_bs_seed   >= 0) glc_Uniform1ui(u_bs_seed,   (GLuint)(tx*2654435761u ^ ty*40503u ^ 0x1234u));
    /* Raw A/B/C are always emitted; these only shape the combined column S.
     * Start neutral (all weight on A, no normalisation) and retune from the histogram. */
    if (u_bs_wA >= 0) glc_Uniform1f(u_bs_wA, 1.0f);
    if (u_bs_wB >= 0) glc_Uniform1f(u_bs_wB, 0.0f);
    if (u_bs_wC >= 0) glc_Uniform1f(u_bs_wC, 0.0f);
    if (u_bs_nA >= 0) glc_Uniform1f(u_bs_nA, 1.0f);
    if (u_bs_nB >= 0) glc_Uniform1f(u_bs_nB, 1.0f);
    if (u_bs_nC >= 0) glc_Uniform1f(u_bs_nC, 1.0f);

    /* One work-group per (cam-batch, light-batch) unit. */
    glc_DispatchCompute(n_cbatch, n_lbatch, 1u);
    glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    /* Read back and append to scores.bin (header: c,b counts as uint32). */
    std::vector<float> host_scores(n_units * 4u);
    glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_batch_score);
    glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                         (GLsizeiptr)(n_units * 4u * sizeof(float)),
                         host_scores.data());
    glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    if (FILE* f = std::fopen("scores.bin", "ab")) {
        const uint32_t hdr[4] = { n_cbatch, n_lbatch, (uint32_t)tx, (uint32_t)ty };
        std::fwrite(hdr, sizeof(uint32_t), 4, f);
        std::fwrite(host_scores.data(), sizeof(float), host_scores.size(), f);
        std::fclose(f);
    }

    /* Optional inline preview: how many units clear a trial cut on column S.
     * "Never discard" means this is only a RUN-ORDER split, not a kill. */
    {
        const float trial_cut = 0.0f;     /* set after first histogram pass */
        size_t now_units = 0, idle_units = 0;
        double s_min = 1e300, s_max = -1e300;
        for (size_t u = 0; u < n_units; ++u) {
            const float S = host_scores[u * 4u + 3u];
            s_min = std::min<double>(s_min, S);
            s_max = std::max<double>(s_max, S);
            (S > trial_cut ? now_units : idle_units)++;
        }
        fprintf(stderr,
            "[batch-score tile(%d,%d)] units=%zu  S[min=%.3g max=%.3g]  now=%zu idle=%zu (cut=%.3g)\n",
            tx, ty, n_units, s_min, s_max, now_units, idle_units, trial_cut);
        fflush(stderr);
    }
}

/* ── Next step once the histogram looks right ────────────────────────────────
 * The drain-when-empty scheduler then becomes: build an index array of the
 * n_units work cells, std::stable_sort it by host_scores[u*4+3] DESCENDING, and
 * drive the existing (c,b) dispatch from that order instead of the raw nested
 * loops. Nothing is dropped — low-score cells simply dispatch last, so a long
 * render keeps refining the "nothing" regions only after the high-yield ones
 * are done. The TDR heartbeat (glFinish every 8) carries over unchanged.
 * ───────────────────────────────────────────────────────────────────────────*/
