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
 *   location 3 — vec2  aUv     : UV coords for the emission/depth/remit
 *                                texture stack (Stage 2 of the UV pack
 *                                action plan).  When the VAO does NOT
 *                                bind this attribute, the GL default
 *                                supplies vec4(0,0,0,1), which samples
 *                                the (0,0) texel of the bound 1×1×1
 *                                default texture — i.e. identity.
 *
 * Uniforms:
 *   uMVP — 4×4 model-view-projection (clip = uMVP * vec4(aPos,1))
 *   uMV  — 4×4 model-view            (view-space position/normal)
 *
 * Outputs to fragment stage:
 *   vNormV   — view-space normal (not normalised; fragment does it)
 *   vPosV    — view-space position
 *   vMatId   — flat-interpolated material index
 *   vGroupId — flat-interpolated emitting surface group identity
 *   vUv      — emission/depth/remit UV
 */

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in int  aMatId;
layout(location = 3) in vec2 aUv;
layout(location = 4) in int  aGroupId;
layout(location = 5) in int  aCullImmune;

uniform mat4 uMVP;
uniform mat4 uMV;

out vec3 vNormV;
out vec3 vPosV;
flat out int vMatId;
flat out int vGroupId;
flat out int vCullImmune;
out vec2 vUv;

void main() {
    vec4 posV   = uMV  * vec4(aPos, 1.0);
    vPosV       = posV.xyz;
    vNormV      = mat3(uMV) * aNorm;
    vMatId      = aMatId;
    vGroupId    = aGroupId;
    vCullImmune = aCullImmune;
    vUv         = aUv;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
