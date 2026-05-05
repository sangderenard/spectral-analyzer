#version 430 core

/* doc_composite.vert.glsl
 *
 * Fullscreen triangle trick: renders exactly one triangle that covers the
 * viewport completely.  No VBO required — call glDrawArrays(GL_TRIANGLES, 0, 3)
 * with an empty VAO bound.
 *
 * Outputs:
 *   vUv  — [0,1]² UV coordinates, (0,0) at bottom-left.
 */

out vec2 vUv;

void main() {
    /*  Vertex 0: (-1, -1) → uv (0, 0)
        Vertex 1: ( 3, -1) → uv (2, 0)   (off-screen right)
        Vertex 2: (-1,  3) → uv (0, 2)   (off-screen top)
        The triangle covers the entire NDC [-1,1]² clip square.         */
    vec2 pos = vec2(
        (gl_VertexID & 1) != 0 ?  3.0 : -1.0,
        (gl_VertexID & 2) != 0 ?  3.0 : -1.0
    );
    vUv        = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
