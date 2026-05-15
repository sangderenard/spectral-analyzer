/**
 * gl_compute.h — Minimal WGL headless GL 4.3 compute context + proc loading.
 *
 * Windows-only: creates a hidden HWND, sets a pixel format, upgrades to a
 * GL 4.3 core context via wglCreateContextAttribsARB, and loads all GL 4.3
 * function pointers needed by the ray-pipeline compute shaders.
 *
 * Usage:
 *   GlComputeContext ctx;
 *   gl_compute_create_context(&ctx);      // once per process
 *   gl_compute_load_procs();              // after context is current
 *   // ... GL dispatch calls ...
 *   gl_compute_destroy_context(&ctx);
 *
 * Only one context is supported per process (all pointers are globals).
 */
#pragma once

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <GL/gl.h>          /* base GL 1.1 types + constants from Windows SDK */
#else
/* Non-Windows stub types so the header parses on other platforms. */
typedef unsigned int   GLenum;
typedef unsigned int   GLuint;
typedef int            GLint;
typedef int            GLsizei;
typedef float          GLfloat;
typedef unsigned char  GLboolean;
typedef unsigned int   GLbitfield;
#  define APIENTRY
#  define GLAPIENTRY
#endif

#include <stddef.h>   /* ptrdiff_t, size_t */
#include <stdint.h>

/* ── Extra GL types not in <GL/gl.h> ────────────────────────────────────── */
#ifndef GL_COMPUTE_SHADER
#  define GL_COMPUTE_SHADER               0x91B9
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
#  define GL_SHADER_STORAGE_BUFFER        0x90D2
#endif
#ifndef GL_STATIC_DRAW
#  define GL_STATIC_DRAW                  0x88B4
#endif
#ifndef GL_DYNAMIC_DRAW
#  define GL_DYNAMIC_DRAW                 0x88E8
#endif
#ifndef GL_DYNAMIC_READ
#  define GL_DYNAMIC_READ                 0x88E9
#endif
#ifndef GL_DYNAMIC_COPY
#  define GL_DYNAMIC_COPY                 0x88EA
#endif
#ifndef GL_STREAM_READ
#  define GL_STREAM_READ                  0x88E1
#endif
#ifndef GL_COMPILE_STATUS
#  define GL_COMPILE_STATUS               0x8B81
#endif
#ifndef GL_LINK_STATUS
#  define GL_LINK_STATUS                  0x8B82
#endif
#ifndef GL_INFO_LOG_LENGTH
#  define GL_INFO_LOG_LENGTH              0x8B84
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
#  define GL_SHADER_STORAGE_BARRIER_BIT   0x00002000
#endif
#ifndef GL_ALL_BARRIER_BITS
#  define GL_ALL_BARRIER_BITS             0xFFFFFFFF
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#  define GL_TEXTURE_2D_ARRAY             0x8C1A
#endif
#ifndef GL_READ_WRITE
#  define GL_READ_WRITE                   0x88BA
#endif
#ifndef GL_WRITE_ONLY
#  define GL_WRITE_ONLY                   0x88B9
#endif
#ifndef GL_R32UI
#  define GL_R32UI                        0x8236
#endif
#ifndef GL_R32F
#  define GL_R32F                         0x822E
#endif
#ifndef GL_RED
#  define GL_RED                          0x1903
#endif
#ifndef GL_FLOAT
#  define GL_FLOAT                        0x1406
#endif
#ifndef GL_UNSIGNED_INT
#  define GL_UNSIGNED_INT                 0x1405
#endif

typedef ptrdiff_t  GLintptr;
typedef ptrdiff_t  GLsizeiptr;
typedef char       GLchar;

/* ── Headless context struct ─────────────────────────────────────────────── */

struct GlComputeContext {
#ifdef _WIN32
    HWND    hwnd   = NULL;
    HDC     hdc    = NULL;
    HGLRC   hglrc  = NULL;
    ATOM    wclass = 0;
#endif
    bool    ready  = false;
    char    error[512] = {};
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a WGL headless 4.3 core context on a hidden window.
 * Returns true on success; ctx->error contains a description on failure.
 * Must be called from the thread that will own (and drive) the context.
 */
bool gl_compute_create_context(GlComputeContext* ctx);

/**
 * Load all GL 4.3 extension function pointers.
 * Must be called after gl_compute_create_context (context must be current).
 * Returns true if all required functions were loaded.
 */
bool gl_compute_load_procs(void);

/**
 * Make this context current on the calling thread.
 * All GL calls must come from the same thread.
 */
bool gl_compute_make_current(GlComputeContext* ctx);

/**
 * Release the context from the current thread (make no-context current).
 */
void gl_compute_release_current(void);

/**
 * Destroy the context and associated window.
 */
void gl_compute_destroy_context(GlComputeContext* ctx);

#ifdef __cplusplus
}
#endif

/* ── GL 4.3 function pointer types ─────────────────────────────────────── */
/* We define our own types to avoid depending on <GL/glext.h>. */

typedef void    (APIENTRY* PFNGLGENBUFFERSPROC)(GLsizei n, GLuint* buffers);
typedef void    (APIENTRY* PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint* buffers);
typedef void    (APIENTRY* PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void    (APIENTRY* PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
typedef void    (APIENTRY* PFNGLBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, const void* data);
typedef void    (APIENTRY* PFNGLGETBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, void* data);
typedef void    (APIENTRY* PFNGLBINDBUFFERBASEPROC)(GLenum target, GLuint index, GLuint buffer);
typedef void    (APIENTRY* PFNGLBINDBUFFERRANGEPROC)(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);

typedef GLuint  (APIENTRY* PFNGLCREATESHADERPROC)(GLenum type);
typedef void    (APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
typedef void    (APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void    (APIENTRY* PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint* params);
typedef void    (APIENTRY* PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
typedef void    (APIENTRY* PFNGLDELETESHADERPROC)(GLuint shader);

typedef GLuint  (APIENTRY* PFNGLCREATEPROGRAMPROC)(void);
typedef void    (APIENTRY* PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void    (APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void    (APIENTRY* PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint* params);
typedef void    (APIENTRY* PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
typedef void    (APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void    (APIENTRY* PFNGLDELETEPROGRAMPROC)(GLuint program);

typedef GLint   (APIENTRY* PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar* name);
typedef void    (APIENTRY* PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
typedef void    (APIENTRY* PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
typedef void    (APIENTRY* PFNGLUNIFORM1UIPROC)(GLint location, GLuint v0);
typedef void    (APIENTRY* PFNGLUNIFORM1IVPROC)(GLint location, GLsizei count, const GLint* value);
typedef void    (APIENTRY* PFNGLUNIFORM1FVPROC)(GLint location, GLsizei count, const GLfloat* value);
typedef void    (APIENTRY* PFNGLUNIFORM4FVPROC)(GLint location, GLsizei count, const GLfloat* value);
typedef void    (APIENTRY* PFNGLUNIFORM2IVPROC)(GLint location, GLsizei count, const GLint* value);

typedef void    (APIENTRY* PFNGLDISPATCHCOMPUTEPROC)(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
typedef void    (APIENTRY* PFNGLMEMORYBARRIERPROC)(GLbitfield barriers);

/* GL 4.2 image textures */
typedef void    (APIENTRY* PFNGLBINDIMAGETEXTUREPROC)(GLuint unit, GLuint texture, GLint level,
                                                       GLboolean layered, GLint layer,
                                                       GLenum access, GLenum format);
typedef void    (APIENTRY* PFNGLTEXSTORAGE3DPROC)(GLenum target, GLsizei levels,
                                                    GLenum internalformat,
                                                    GLsizei width, GLsizei height, GLsizei depth);

/* GL 3.0 VAO (needed even for pure-compute to satisfy some drivers) */
typedef void    (APIENTRY* PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint* arrays);
typedef void    (APIENTRY* PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void    (APIENTRY* PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint* arrays);

/* ── Extern declarations for each function pointer ──────────────────────── */

extern PFNGLGENBUFFERSPROC          glc_GenBuffers;
extern PFNGLDELETEBUFFERSPROC       glc_DeleteBuffers;
extern PFNGLBINDBUFFERPROC          glc_BindBuffer;
extern PFNGLBUFFERDATAPROC          glc_BufferData;
extern PFNGLBUFFERSUBDATAPROC       glc_BufferSubData;
extern PFNGLGETBUFFERSUBDATAPROC    glc_GetBufferSubData;
extern PFNGLBINDBUFFERBASEPROC      glc_BindBufferBase;
extern PFNGLBINDBUFFERRANGEPROC     glc_BindBufferRange;

extern PFNGLCREATESHADERPROC        glc_CreateShader;
extern PFNGLSHADERSOURCEPROC        glc_ShaderSource;
extern PFNGLCOMPILESHADERPROC       glc_CompileShader;
extern PFNGLGETSHADERIVPROC         glc_GetShaderiv;
extern PFNGLGETSHADERINFOLOGPROC    glc_GetShaderInfoLog;
extern PFNGLDELETESHADERPROC        glc_DeleteShader;

extern PFNGLCREATEPROGRAMPROC       glc_CreateProgram;
extern PFNGLATTACHSHADERPROC        glc_AttachShader;
extern PFNGLLINKPROGRAMPROC         glc_LinkProgram;
extern PFNGLGETPROGRAMIVPROC        glc_GetProgramiv;
extern PFNGLGETPROGRAMINFOLOGPROC   glc_GetProgramInfoLog;
extern PFNGLUSEPROGRAMPROC          glc_UseProgram;
extern PFNGLDELETEPROGRAMPROC       glc_DeleteProgram;

extern PFNGLGETUNIFORMLOCATIONPROC  glc_GetUniformLocation;
extern PFNGLUNIFORM1IPROC           glc_Uniform1i;
extern PFNGLUNIFORM1FPROC           glc_Uniform1f;
extern PFNGLUNIFORM1UIPROC          glc_Uniform1ui;
extern PFNGLUNIFORM1IVPROC          glc_Uniform1iv;
extern PFNGLUNIFORM1FVPROC          glc_Uniform1fv;
extern PFNGLUNIFORM4FVPROC          glc_Uniform4fv;
extern PFNGLUNIFORM2IVPROC          glc_Uniform2iv;

extern PFNGLDISPATCHCOMPUTEPROC     glc_DispatchCompute;
extern PFNGLMEMORYBARRIERPROC       glc_MemoryBarrier;

extern PFNGLBINDIMAGETEXTUREPROC    glc_BindImageTexture;
extern PFNGLTEXSTORAGE3DPROC        glc_TexStorage3D;

extern PFNGLGENVERTEXARRAYSPROC     glc_GenVertexArrays;
extern PFNGLBINDVERTEXARRAYPROC     glc_BindVertexArray;
extern PFNGLDELETEVERTEXARRAYSPROC  glc_DeleteVertexArrays;

/* ── Convenience aliases matching standard GL names ─────────────────────── */
/* Use the glc_ prefix to avoid symbol conflicts with any other GL headers. */

#define GLC_BUFFER_TARGET_SSBO    GL_SHADER_STORAGE_BUFFER

/**
 * Compile a compute shader from source string.
 * Returns program ID on success, 0 on failure (error written to err_out if non-null).
 */
#ifdef __cplusplus
GLuint gl_compute_build_program(const char* glsl_source,
                                 char* err_out, int err_sz);
#endif
