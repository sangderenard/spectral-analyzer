#version 430 core

/*
 * Scientific vehicle graph fragment shader.
 *
 * Component class owns the unstrained base hue.  Signed strain adds a
 * diverging blue (compression) / red (tension) tint while retaining enough
 * of that hue to keep type legible.  Yield and fracture are per-element ABI
 * values, so changing material parameters does not require recompiling GLSL.
 * Bushings and bearings use the same color law as members and are drawn as
 * distinct point glyphs.
 */

in vec3 vNormalV;
in vec3 vPositionV;
in float vAxialStrain;
in float vYieldStrain;
in float vFractureStrain;
flat in int vComponentType;
flat in int vDamageState;
flat in int vJointKind;
in float vLaplacianEnergy;
in float vConservationResidual;

uniform int   uPrimitiveKind = 0;  // 0 surface/line, 1 point glyph
uniform vec3  uLightDirectionV = vec3(0.349, 0.798, 0.478);
uniform float uExposure = 1.0;
uniform float uAlpha = 1.0; // premultiplied weight for normalized batch overlays

layout(location = 0) out vec4 FragColor;

vec3 component_color(int type_id) {
    if (type_id == 0) return vec3(0.48, 0.56, 0.64); // frame / structure
    if (type_id == 1) return vec3(0.18, 0.78, 0.40); // suspension / links
    if (type_id == 2) return vec3(0.96, 0.48, 0.10); // drivetrain / rotating
    if (type_id == 3) return vec3(0.16, 0.56, 0.96); // tyre / contact
    if (type_id == 4) return vec3(0.72, 0.30, 0.94); // external fixture
    if (type_id == 5) return vec3(0.10, 0.84, 0.88); // hydraulic / pneumatic
    if (type_id == 6) return vec3(0.96, 0.78, 0.16); // electrical / controls
    return vec3(0.72, 0.68, 0.58);                  // body / other
}

vec3 strain_color(float strain, float yield_strain, float fracture_strain,
                  vec3 type_color) {
    float magnitude = abs(strain);
    float yield_use = magnitude / max(yield_strain, 1.0e-8);
    float fracture_use = magnitude / max(fracture_strain, yield_strain);
    vec3 signed_tint = strain < 0.0
        ? vec3(0.08, 0.34, 1.00)   // compression
        : vec3(1.00, 0.12, 0.06);  // tension
    float elastic_mix = 0.68 * smoothstep(0.05, 1.0, min(yield_use, 1.0));
    vec3 color = mix(type_color, signed_tint, elastic_mix);
    if (yield_use > 1.0) {
        float plastic = smoothstep(1.0, max(fracture_strain / yield_strain, 1.01), yield_use);
        color = mix(color, vec3(1.00, 0.08, 0.78), 0.35 + 0.45 * plastic);
    }
    if (fracture_use >= 1.0) color = vec3(1.0, 1.0, 1.0);
    return color;
}

float point_glyph_alpha(int joint_kind, vec2 point_coord) {
    vec2 p = point_coord * 2.0 - 1.0;
    float r = length(p);
    if (r > 1.0) discard;
    if (joint_kind == 1) { // bushing: compliant annulus plus solid core
        float annulus = 1.0 - smoothstep(0.08, 0.18, abs(r - 0.72));
        float core = 1.0 - smoothstep(0.18, 0.30, r);
        return max(annulus, core);
    }
    if (joint_kind == 2) { // bearing: double race and axis crosshair
        float outer_race = 1.0 - smoothstep(0.05, 0.13, abs(r - 0.78));
        float inner_race = 1.0 - smoothstep(0.05, 0.13, abs(r - 0.43));
        float crosshair = 1.0 - smoothstep(0.035, 0.075, min(abs(p.x), abs(p.y)));
        return max(max(outer_race, inner_race), crosshair * step(r, 0.62));
    }
    return 1.0 - smoothstep(0.80, 1.0, r);
}

void main() {
    vec3 type_color = component_color(vComponentType);
    vec3 color = strain_color(vAxialStrain, vYieldStrain,
                              vFractureStrain, type_color);
    // These are read-only graph-Laplacian accounting lanes. Stored elastic
    // energy warms toward amber; unexplained positive energy residual flashes
    // toward white. Neither lane feeds back into the mechanical solve.
    color = mix(color, vec3(1.0, 0.62, 0.04), smoothstep(0.2, 1.4, vLaplacianEnergy) * 0.45);
    color = mix(color, vec3(1.0), smoothstep(0.15, 1.2, vConservationResidual) * 0.82);
    if (vDamageState == 1) color = mix(color, vec3(1.0, 0.08, 0.78), 0.35);
    if (vDamageState >= 2) color = vec3(1.0);

    float alpha = 1.0;
    if (uPrimitiveKind == 1) {
        alpha = point_glyph_alpha(vJointKind, gl_PointCoord);
        // Preserve component identity in the core and put strain on the races.
        float core = 1.0 - smoothstep(0.0, 0.28, length(gl_PointCoord * 2.0 - 1.0));
        color = mix(color, type_color, core * 0.72);
    } else {
        vec3 N = normalize(vNormalV);
        float diffuse = 0.34 + 0.66 * abs(dot(N, normalize(uLightDirectionV)));
        vec3 V = normalize(-vPositionV);
        vec3 H = normalize(normalize(uLightDirectionV) + V);
        float specular = pow(max(abs(dot(N, H)), 0.0), 32.0) * 0.22;
        color = color * diffuse + vec3(specular);
    }

    // A failed element is unmistakable without hiding its geometry.
    if (vDamageState >= 2) {
        float hatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) * 0.16));
        color = mix(vec3(0.04), vec3(1.0), hatch);
    }
    FragColor = vec4(color * uExposure * uAlpha, alpha * uAlpha);
}
