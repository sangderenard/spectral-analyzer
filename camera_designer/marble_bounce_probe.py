"""Sparse room-depth diagnostic using the canonical forward transport shader.

The pilot observes transport energy, not image error. It never terminates an
ordinary render path and it never deposits pilot rays into the render field.
"""
from __future__ import annotations

import math
import time
import numpy as np

PROBE_DEPTH = 1000
RENDER_CAP = 200
PROBE_GROUPS = 4
RAYS_PER_GROUP = 64
TAIL_WINDOW = 64
TAIL_FRACTION = 0.01


def scatter_sources(bounds, wavelengths, *, seed=73417):
    """Four jittered samples per 4x4x4 room cell, isotropic directions."""
    lo, hi = (np.asarray(v, dtype=np.float64) for v in bounds)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError("probe bounds must have positive finite extent")
    waves = np.asarray(wavelengths, dtype=np.float64)
    if waves.ndim != 1 or not len(waves) or not np.isfinite(waves).all() or np.any(waves <= 0):
        raise ValueError("probe wavelengths must be finite and positive")
    cells = np.array(list(np.ndindex(4, 4, 4)), dtype=np.float64)
    result = np.zeros((PROBE_GROUPS, RAYS_PER_GROUP, 12), np.float32)
    for group in range(PROBE_GROUPS):
        rng = np.random.default_rng(seed + group * 104729)
        points = lo + ((cells + rng.random((64, 3))) / 4) * (hi - lo)
        z = rng.uniform(-1, 1, 64)
        angle = rng.uniform(0, 2 * math.pi, 64)
        radius = np.sqrt(1-z*z)
        result[group, :, :3] = points
        result[group, :, 3] = 1
        result[group, :, 4:7] = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), z))
        result[group, :, 8] = 2.998e14 / waves[(np.arange(64) + group) % len(waves)]
        result[group, :, 10] = 1
    return result


def summarize_probe(samples, *, elapsed_seconds=0.0):
    """Select one global limit from whole observed tails, including late returns.

    samples[g,r,d] is energy arriving at interaction d (zero-based); column
    1000 is energy still propagating after interaction 1000, not a new hit.
    The 95% t spread describes variation between four pilot batches only.
    It is not a confidence bound on unseen caustics or camera-image error.
    """
    energy = np.asarray(samples, dtype=np.float64)
    if energy.shape != (PROBE_GROUPS, RAYS_PER_GROUP, PROBE_DEPTH + 1):
        raise ValueError("probe requires four 64-ray batches through depth 1000")
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("probe returned invalid transport energy")
    hits = (energy[:, :, 0] > 0).sum(axis=1)
    by_group = energy[:, :, :PROBE_DEPTH].sum(axis=1)
    totals = by_group.sum(axis=1)
    tails = np.cumsum(by_group[:, ::-1], axis=1)[:, ::-1]
    ratios = np.divide(tails, totals[:, None], out=np.zeros_like(tails), where=totals[:, None] > 0)
    endpoint = energy[:, :, PROBE_DEPTH]
    censored = int(np.count_nonzero(endpoint > 0))
    coverage = bool(np.all(hits >= 16))
    enough = coverage and not censored
    chosen = RENDER_CAP
    found = None
    for depth in range(8, PROBE_DEPTH - TAIL_WINDOW + 1):
        values = ratios[:, depth]
        spread = float(values.mean() + 3.182446 * values.std(ddof=1) / math.sqrt(PROBE_GROUPS))
        # Every independent batch and its variation must pass. The complete
        # remaining horizon is examined, not an isolated empty bounce bin.
        if enough and max(float(values.max()), spread) <= TAIL_FRACTION:
            found = depth
            chosen = min(depth, RENDER_CAP)
            break
    fractions = ratios[:, chosen]
    cap_survivors = int(np.count_nonzero(energy[:, :, RENDER_CAP] > 0))
    warnings = []
    if not coverage:
        warnings.append("Sparse probe coverage: fewer than 16 first hits in a batch. Using render cap 200.")
    if censored:
        warnings.append(f"Mirror-room warning: {censored} of 256 probe paths still carry energy after 1000 bounces. Their remaining lifetime is unknown; rendering is capped at 200.")
    elif found is None or found > RENDER_CAP:
        warnings.append("Significant or uncertain transport remains beyond the render cap of 200 bounces.")
    return {
        "version": 1, "mode": "auto", "chosen_depth": chosen,
        "render_cap": RENDER_CAP, "probe_depth": PROBE_DEPTH,
        "probe_rays": PROBE_GROUPS * RAYS_PER_GROUP, "probe_batches": PROBE_GROUPS,
        "window_bounces": TAIL_WINDOW, "threshold_fraction": TAIL_FRACTION,
        "estimated_required_depth": found,
        "coverage_hits": hits.astype(int).tolist(),
        "paths_at_render_cap": cap_survivors, "paths_at_probe_limit": censored,
        "energy_at_probe_limit": float(endpoint.sum()),
        "tail_fraction": float(fractions.mean()),
        "tail_fraction_batch_max": float(fractions.max()),
        "tail_fraction_95_batch_upper": float(fractions.mean() + 3.182446 * fractions.std(ddof=1) / 2),
        "tail_fraction_beyond_render_cap": float(ratios[:, RENDER_CAP].mean()),
        "confidence": "limited sparse pilot" if coverage else "insufficient spatial coverage",
        "metric": "remaining forward transport-energy sum through depth 1000; not camera-image error",
        "limitations": "Four spatial pilot batches can miss rare paths and small objects. Existing forward energy cutoff 0.002 remains; sensor roulette and sensor throughput cutoffs differ. No guarantee of image convergence or of zero energy beyond 1000.",
        "minimum_depth": 8, "sampling": "four independent isotropic unit-energy rays per 4x4x4 room cell",
        "warning": " ".join(warnings), "elapsed_seconds": round(float(elapsed_seconds), 3),
    }


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Canonical forward shader changed; Auto probe instrumentation needs review")
    return source.replace(old, new, 1)


def _empty_function(source, name):
    """Remove only a named function body, retaining its exact declaration."""
    marker = "void " + name + "("
    start = source.find(marker)
    if start < 0:
        raise RuntimeError("Canonical forward shader is missing " + name)
    brace = source.find("{", start)
    level, end = 1, brace + 1
    while level and end < len(source):
        level += (source[end] == "{") - (source[end] == "}")
        end += 1
    if level:
        raise RuntimeError("Unbalanced canonical forward shader")
    return source[:brace] + "{ return; }" + source[end:]


def probe_shader(canonical):
    """Instrument real forward interactions, without copying their physics.

    Probe rays do not splat voxels or queue secondaries. The Studio's ordinary
    prebuilt forward pump likewise does not dispatch the secondary queue.
    No new per-path termination rule is introduced here.
    """
    source = canonical
    for name in ("splat_spectral", "splat_activation", "splat_segment_multiscale"):
        source = _empty_function(source, name)
    source = _replace_once(source, "uniform int   uMaxBounces;", """uniform int   uMaxBounces;
layout(std430, binding = 11) buffer BounceProbe { float probe_energy[]; };
""")
    # Restrict instrumentation to the existing main forward body. Other
    # functions have their own bounce loops and must remain untouched.
    start = source.index("void main() {")
    end = source.index("void reactive_main(uint gid, inout uint rng) {", start)
    forward = source[start:end]
    forward = _replace_once(forward,
        "        if (hit < 0) break;",
        "        if (hit < 0) break;\n        probe_energy[ray_id * uint(uMaxBounces + 1) + uint(bounce)] = max(packet.z, 0.0);")
    # The last slot records a path still alive after the full diagnostic
    # horizon. Ordinary, manifold, and explicit glass/air continuations must
    # all be counted; reflection at an explicit boundary also stays alive.
    needle = "if (packet.z < 0.002) break;"
    if forward.count(needle) != 3 or forward.count("if(studio_boundary(tri)) {") != 1:
        raise RuntimeError("Canonical continuation sites changed; Auto probe needs review")
    forward = forward.replace(needle, needle + "\n        if (bounce + 1 == uMaxBounces) probe_energy[ray_id * uint(uMaxBounces + 1) + uint(uMaxBounces)] = max(packet.z, 0.0);")
    return source[:start] + forward + source[end:]


def run_bounce_probe(renderer, state, bounds, wavelengths, *, n_triangles, n_nodes, context_count=0, progress=None):
    """At most four 64-ray dispatches; own buffers/program, existing scene read-only.

    No scene/material/camera mutation and no field writes. A full ray has a
    strict 1000-interaction limit. Host checks a 10-second pilot budget between
    dispatches; each individual GL dispatch still needs the driver's completion.
    """
    from OpenGL import GL as gl
    started = time.monotonic()
    canonical = renderer._GPU_RAY_FIELD_CS
    if state.get('volume_extinction') is not None:
        from camera_designer.studio_volume_extinction import transport_shader
        canonical = transport_shader(renderer._GPU_RAY_FIELD_REGION_CS if state.get('region_count') else canonical)
    source = probe_shader(canonical)
    program = renderer._prog((source, gl.GL_COMPUTE_SHADER))
    buffers = []
    try:
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError("Auto bounce probe shader failed to link")
        sources = scatter_sources(bounds, wavelengths)
        records = np.zeros((PROBE_GROUPS, RAYS_PER_GROUP, PROBE_DEPTH + 1), np.float32)
        buffers = [int(x) for x in gl.glGenBuffers(2)]
        gl.glUseProgram(program)
        if state.get('volume_extinction') is not None:
            from camera_designer.studio_volume_extinction import bind as bind_volume
            bind_volume(state['volume_extinction'], program)
        def integer(name, value):
            gl.glUniform1i(gl.glGetUniformLocation(program, name.encode()), int(value))
        def scalar(name, value):
            gl.glUniform1f(gl.glGetUniformLocation(program, name.encode()), float(value))
        for binding, buffer in enumerate(state["ssbo"]):
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding, buffer)
        integer("uStudioSphereCount", state.get("sphere_count", 0))
        if state.get('volume_extinction') is not None:
            integer("uStudioRegionCount", state.get("region_count", 0))
        integer("uTriCount", n_triangles)
        integer("uNodeCount", n_nodes)
        integer("uSourceCount", RAYS_PER_GROUP)
        integer("uMaxBounces", PROBE_DEPTH)
        integer("uBatchSize", RAYS_PER_GROUP)
        integer("uBatchOffset", 0)
        integer("uTotalRaysPerSource", RAYS_PER_GROUP)
        integer("uMode", 0)
        integer("uDiagnosticsEnabled", 0)
        integer("uSegmentCapture", 0)
        integer("uPendingCap", 0)
        integer("uScaleContextCount", context_count)
        integer("uFixed3Transport", 1 if len(wavelengths) == 3 else 0)
        scalar("uSpeedOfMedium", 2.998e8)
        scalar("uMediumExtinction", .5)
        for group in range(PROBE_GROUPS):
            if time.monotonic() - started > 10:
                raise RuntimeError("Auto bounce probe exceeded its 10-second budget; render cap 200 retained")
            if progress:
                progress(group, PROBE_GROUPS)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffers[0])
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, sources[group].nbytes, sources[group], gl.GL_STATIC_DRAW)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 5, buffers[0])
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffers[1])
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, records[group].nbytes, records[group], gl.GL_DYNAMIC_READ)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 11, buffers[1])
            integer("uSeed", 73417 + group * 104729)
            gl.glDispatchCompute(1, 1, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
            gl.glFinish()
            raw = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, records[group].nbytes)
            records[group] = np.frombuffer(raw, np.float32).reshape(RAYS_PER_GROUP, PROBE_DEPTH + 1)
        return summarize_probe(records, elapsed_seconds=time.monotonic()-started)
    finally:
        gl.glUseProgram(0)
        # Return all shared bindings to their original objects, even on error.
        for binding, buffer in enumerate(state["ssbo"]):
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding, buffer)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 11, 0)
        if buffers:
            gl.glDeleteBuffers(len(buffers), buffers)
        gl.glDeleteProgram(program)
