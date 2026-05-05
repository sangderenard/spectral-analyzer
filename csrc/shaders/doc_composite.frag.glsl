#version 430 core

/* doc_composite.frag.glsl
 *
 * Blits the composited document-hierarchy RGBA texture onto the framebuffer.
 *
 * Uniforms:
 *   uDocAtlas  — the composited RGBA8 texture produced by dr_composite().
 *   uAlpha     — global opacity multiplier [0,1]; default 1.0.
 *
 * Blend mode:
 *   Caller must enable GL_BLEND and set:
 *     glBlendEquation(GL_FUNC_ADD)
 *     glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
 *                         GL_ONE,       GL_ONE_MINUS_SRC_ALPHA)
 *   This performs standard premult-correct over-compositing.
 *
 * Coordinate convention:
 *   uDocAtlas origin is top-left (matches CPU RGBA buffer row-major order).
 *   GL texture origin is bottom-left, so we flip Y.
 */

in  vec2 vUv;
out vec4 fragColor;

uniform sampler2D uDocAtlas;
uniform float     uAlpha;

void main() {
    /* Flip Y: CPU buffer row 0 = top of screen; GL tex row 0 = bottom */
    vec2 uv      = vec2(vUv.x, 1.0 - vUv.y);
    vec4 texel   = texture(uDocAtlas, uv);

    /* Apply global opacity */
    texel.a     *= uAlpha;

    /* Discard fully transparent fragments to avoid polluting the depth buffer */
    if (texel.a < 0.004) discard;

    fragColor = texel;
}
