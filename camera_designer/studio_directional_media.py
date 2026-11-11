"""Record existing refracted segments with their authoritative Beer attenuation.

This changes only the optional directional observer. It neither traces rays nor
changes primary/scalar transport. The installed medium helper is the authority
for material selection and additive dye k; the recorder's rate function is a
mechanically derived view of that exact helper, not another optical model.
"""
from __future__ import annotations

import re


_SIGNATURE = re.compile(
    r"float (studio_(?:legacy_)?medium_transmittance)\(int medium,\s*float frequency,\s*float distance\)\s*\{")


def _medium_functions(source):
    result = []
    for match in _SIGNATURE.finditer(source):
        depth, end = 1, match.end()
        while end < len(source) and depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        if depth:
            raise ValueError("unterminated canonical medium transmittance helper")
        result.append((match.group(1), source[match.end():end - 1]))
    names = [name for name, _ in result]
    if names.count("studio_medium_transmittance") != 1 or len(set(names)) != len(names):
        raise ValueError("canonical medium transmittance helper is missing or ambiguous")
    return result


def medium_rate_glsl(source):
    """Derive sigma from the selected canonical Beer body, retaining its k logic.

    Keep original transmittance declarations and arithmetic entirely untouched.
    Strict seams deliberately refuse a new law rather than infer its inverse
    from an underflowed endpoint transmittance (a dense rod can have T == 0).
    """
    result = []
    for name, body in _medium_functions(source):
        terminal = re.search(r"return exp\(-([^;\n]+)\);\s*$", body)
        if terminal is None or terminal.group(1).count("*max(distance,0.0)") != 1:
            raise ValueError("canonical medium Beer-rate expression changed")
        rate = terminal.group(1).replace("*max(distance,0.0)", "", 1)
        prefix = body[:terminal.start()]
        prefix, air_count = re.subn(r"if\(medium<0\)\s*return 1\.0;",
                                    "if(medium<0) return 0.0;", prefix)
        if air_count != 1:
            raise ValueError("canonical medium air identity changed")
        prefix = prefix.replace("if(medium==-2)return 0.0;", "if(medium==-2)return -1.0;")
        prefix = prefix.replace("studio_legacy_medium_transmittance(medium,frequency,distance)",
                                "studio_record_legacy_medium_sigma(medium,frequency)")
        if "distance" in prefix or "transmittance(" in prefix:
            raise ValueError("unsupported canonical medium attenuation dependency")
        derived = name.replace("studio_", "studio_record_", 1).replace("transmittance", "sigma")
        result.append(f"float {derived}(int medium,float frequency) {{" + prefix +
                      f"return {rate};\n}}\n")
    return "\n".join(result)


INTEGRAL_GLSL = r'''
// Integral from segment distances first to last. sigma is selected by the
// unchanged canonical medium/region/dye helper for this actual traced segment.
float studio_record_medium_integral(float sigma,float first,float last) {
    float span=max(last-first,0.0);
    if(!(span>0.0) || sigma<0.0)return 0.0;
    if(sigma==0.0)return span;
    if(isinf(sigma))return 0.0;
    float tau=sigma*span;
    // Stable evaluation of -expm1(-tau)/sigma, including its zero limit.
    float within=tau<0.001 ? span*(1.0-tau*0.5+tau*tau/6.0) : (1.0-exp(-tau))/sigma;
    return exp(-sigma*max(first,0.0))*within;
}
'''


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("canonical internal-field seam changed: " + old[:90])
    return source.replace(old, new, 1)


def with_internal_medium_recording(source, *, volume_extinction=False):
    """Specialize an already assembled directional forward shader.

    Apply after boundary/region/sphere and directional composition. Recording
    starts before the two canonical Beer updates and receives entry throughput.
    The scalar splat remains under its original exterior-only guard.

    An optional ExtinctionGrid is not silently multiplied with a separate Beer
    integral: the product needs integration over both cell partitions. Until
    that composition is supplied, explicitly refuse this combination.
    """
    if volume_extinction or "studio_volume_path(" in source or "studio_sigma_t[" in source:
        raise ValueError("internal directional recording with ExtinctionGrid requires joint attenuation integration")
    if "studio_record_medium_integral(" in source:
        raise ValueError("internal directional recording is already installed")
    helpers = medium_rate_glsl(source) + INTEGRAL_GLSL
    source = _replace_once(source, "void studio_record_segment(vec3 a,vec3 b,vec4 packet) {",
        helpers + "\nvoid studio_record_segment(vec3 a,vec3 b,vec4 packet,int medium) {\n"
        "    if(medium < -1)return; // invalid ownership is not air or glass\n"
        "    float studio_record_sigma=studio_record_medium_sigma(medium,packet.x);\n"
        "    if(isnan(studio_record_sigma) || studio_record_sigma<0.0)return;")
    source = _replace_once(source, "studio_record_track(q,d,packet,next-cursor);",
        "studio_record_track(q,d,packet,studio_record_medium_integral(studio_record_sigma,cursor,next));")
    # The recorder was deliberately co-located with the old air-only scalar
    # splats. Remove only that observer call; do not relax the scalar guards.
    for call in ("splat_segment_multiscale(ro, hp, packet, path_len);",
                 "splat_segment_spectral(ro, hp, packet);"):
        candidates = ["if(currentMedium" + comparison + ") { studio_record_segment(ro,hp,packet); " + call + " }"
                      for comparison in ("<0", "==-1")]
        found = [item for item in candidates if source.count(item) == 1]
        if len(found) != 1:
            raise ValueError("canonical exterior scalar/recorder guard changed")
        source = source.replace(found[0], found[0].replace("studio_record_segment(ro,hp,packet); ", ""), 1)
    attenuation = "packet.z*=studio_medium_transmittance(currentMedium,packet.x,best);"
    if source.count(attenuation) != 2:
        raise ValueError("canonical forward/reactive Beer update changed")
    return source.replace(attenuation,
        "studio_record_segment(ro,hp,packet,currentMedium);\n        " + attenuation)
