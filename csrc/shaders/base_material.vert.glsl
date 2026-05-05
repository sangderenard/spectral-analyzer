#version 430 core
/**
 * base_material.vert.glsl
 *
 * Vertex shader for the base material Phong renderer.
 *
 * Inputs (per vertex):
 *   location 0 — vec3  aPos    : object-space position
 *   location 1 — vec3  aNorm   : object-space normal
 *   location 2 — int   aMatId  : material index into SSBO chunks (flat)
 *
 * Uniforms:
 *   uMVP — 4×4 model-view-projection (clip = uMVP * vec4(aPos,1))
 *   uMV  — 4×4 model-view            (view-space position/normal)
 *
 * Outputs to fragment stage:
 *   vNormV   — view-space normal (not normalised; fragment does it)
 *   vPosV    — view-space position
 *   vMatId   — flat-interpolated material index
 */

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in int  aMatId;

uniform mat4 uMVP;
uniform mat4 uMV;

out vec3 vNormV;
out vec3 vPosV;
flat out int vMatId;

void main() {
    vec4 posV   = uMV  * vec4(aPos, 1.0);
    vPosV       = posV.xyz;
    vNormV      = mat3(uMV) * aNorm;
    vMatId      = aMatId;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
