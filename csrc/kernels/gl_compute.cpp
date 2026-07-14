/**
 * gl_compute.cpp — WGL headless GL 4.3 core context + function proc loading.
 *
 * Windows-only implementation.  A dummy HWND+WGL context is created first
 * (required to call wglCreateContextAttribsARB), then upgraded to a proper
 * GL 4.3 core context.  The dummy context is destroyed before returning.
 *
 * All GL 4.3 function pointers are stored as globals with the glc_ prefix to
 * avoid collisions with headers that pull in <GL/glext.h>.
 */

#include "gl_compute.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

/* ── Global function pointer storage ────────────────────────────────────── */

PFNGLGENBUFFERSPROC         glc_GenBuffers        = nullptr;
PFNGLDELETEBUFFERSPROC      glc_DeleteBuffers      = nullptr;
PFNGLBINDBUFFERPROC         glc_BindBuffer         = nullptr;
PFNGLBUFFERDATAPROC         glc_BufferData         = nullptr;
PFNGLBUFFERSUBDATAPROC      glc_BufferSubData      = nullptr;
PFNGLGETBUFFERSUBDATAPROC   glc_GetBufferSubData   = nullptr;
PFNGLBINDBUFFERBASEPROC     glc_BindBufferBase     = nullptr;
PFNGLBINDBUFFERRANGEPROC    glc_BindBufferRange    = nullptr;

PFNGLCREATESHADERPROC       glc_CreateShader       = nullptr;
PFNGLSHADERSOURCEPROC       glc_ShaderSource       = nullptr;
PFNGLCOMPILESHADERPROC      glc_CompileShader      = nullptr;
PFNGLGETSHADERIVPROC        glc_GetShaderiv        = nullptr;
PFNGLGETSHADERINFOLOGPROC   glc_GetShaderInfoLog   = nullptr;
PFNGLDELETESHADERPROC       glc_DeleteShader       = nullptr;

PFNGLCREATEPROGRAMPROC      glc_CreateProgram      = nullptr;
PFNGLATTACHSHADERPROC       glc_AttachShader       = nullptr;
PFNGLLINKPROGRAMPROC        glc_LinkProgram        = nullptr;
PFNGLGETPROGRAMIVPROC       glc_GetProgramiv       = nullptr;
PFNGLGETPROGRAMINFOLOGPROC  glc_GetProgramInfoLog  = nullptr;
PFNGLUSEPROGRAMPROC         glc_UseProgram         = nullptr;
PFNGLDELETEPROGRAMPROC      glc_DeleteProgram      = nullptr;

PFNGLGETUNIFORMLOCATIONPROC glc_GetUniformLocation = nullptr;
PFNGLUNIFORM1IPROC          glc_Uniform1i          = nullptr;
PFNGLUNIFORM1FPROC          glc_Uniform1f          = nullptr;
PFNGLUNIFORM1UIPROC         glc_Uniform1ui         = nullptr;
PFNGLUNIFORM1IVPROC         glc_Uniform1iv         = nullptr;
PFNGLUNIFORM1FVPROC         glc_Uniform1fv         = nullptr;
PFNGLUNIFORM3FVPROC         glc_Uniform3fv         = nullptr;
PFNGLUNIFORM4FVPROC         glc_Uniform4fv         = nullptr;
PFNGLUNIFORM2IVPROC         glc_Uniform2iv         = nullptr;

PFNGLDISPATCHCOMPUTEPROC         glc_DispatchCompute         = nullptr;
PFNGLDISPATCHCOMPUTEINDIRECTPROC glc_DispatchComputeIndirect = nullptr;
PFNGLMEMORYBARRIERPROC           glc_MemoryBarrier           = nullptr;
PFNGLFENCESYNCPROC          glc_FenceSync          = nullptr;
PFNGLCLIENTWAITSYNCPROC     glc_ClientWaitSync     = nullptr;
PFNGLDELETESYNCPROC         glc_DeleteSync         = nullptr;

PFNGLBINDIMAGETEXTUREPROC   glc_BindImageTexture   = nullptr;
PFNGLTEXSTORAGE3DPROC       glc_TexStorage3D       = nullptr;

PFNGLGENVERTEXARRAYSPROC    glc_GenVertexArrays    = nullptr;
PFNGLBINDVERTEXARRAYPROC    glc_BindVertexArray    = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glc_DeleteVertexArrays = nullptr;

PFNGLGETINTEGERI_VPROC      glc_GetIntegeri_v      = nullptr;

/* ── Platform-specific implementation ───────────────────────────────────── */

#ifdef _WIN32

/* WGL_ARB_create_context constants */
#define WGL_CONTEXT_MAJOR_VERSION_ARB    0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB    0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB     0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#define WGL_CONTEXT_FLAGS_ARB            0x2094
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x0002

typedef HGLRC (WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC hDC, HGLRC hShareContext, const int* attribList);

static PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = nullptr;

/* Lookup wrapper: tries wglGetProcAddress first, then GetProcAddress on opengl32.dll */
static void* gl_get_proc(const char* name) {
    void* p = (void*)wglGetProcAddress(name);
    if (!p) {
        HMODULE h = GetModuleHandleA("opengl32.dll");
        if (h) p = (void*)GetProcAddress(h, name);
    }
    return p;
}

bool gl_compute_create_context(GlComputeContext* ctx, void* hShareContext,
                               void* hShareDC) {
    if (!ctx) return false;

    /* Step 1: Register a window class for the hidden window. */
    WNDCLASSA wc    = {};
    wc.style        = CS_OWNDC;
    wc.lpfnWndProc  = DefWindowProcA;
    wc.hInstance    = GetModuleHandleA(nullptr);
    wc.lpszClassName = "GlComputeHidden_v1";

    ctx->wclass = RegisterClassA(&wc);
    if (!ctx->wclass) {
        /* Class may already be registered from a prior call — that's OK. */
        ctx->wclass = (ATOM)1; /* sentinel: don't unregister */
    }

    /* Step 2: Create a hidden window. */
    ctx->hwnd = CreateWindowExA(
        0, "GlComputeHidden_v1", "GL Compute",
        WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
        0, 0, 1, 1,
        nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
    if (!ctx->hwnd) {
        snprintf(ctx->error, sizeof(ctx->error),
                 "CreateWindowExA failed (err=%lu)", GetLastError());
        return false;
    }

    ctx->hdc = GetDC(ctx->hwnd);
    if (!ctx->hdc) {
        snprintf(ctx->error, sizeof(ctx->error), "GetDC failed");
        return false;
    }

    /* Step 3: Set a pixel format for the dummy context.
     * If the caller provides the display DC, copy its pixel format so that
     * wglCreateContextAttribsARB finds compatible formats on both sides.
     * Pixel-format mismatch is the most common cause of err=0 failures. */
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize    = sizeof(pfd);
    pfd.nVersion = 1;
    int fmt = 0;
    if (hShareDC) {
        int share_fmt = GetPixelFormat((HDC)hShareDC);
        if (share_fmt > 0) {
            DescribePixelFormat((HDC)hShareDC, share_fmt, sizeof(pfd), &pfd);
            if (SetPixelFormat(ctx->hdc, share_fmt, &pfd)) {
                fmt = share_fmt;
                fprintf(stderr,
                        "[gpu-dispatch] WGL share pixel format copied: display=%d hidden=%d flags=0x%08lx color=%u depth=%u\n",
                        share_fmt, GetPixelFormat(ctx->hdc), (unsigned long)pfd.dwFlags,
                        (unsigned)pfd.cColorBits, (unsigned)pfd.cDepthBits);
                fflush(stderr);
            } else {
                DWORD e = GetLastError();
                fprintf(stderr,
                        "[gpu-dispatch] WGL share pixel format copy failed: display=%d err=%lu (0x%08lx); choosing compatible format\n",
                        share_fmt, e, (unsigned long)e);
                fflush(stderr);
                fmt = 0;
            }
        }
    }
    if (!fmt) {
        pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        pfd.cDepthBits = 24;
        pfd.iLayerType = PFD_MAIN_PLANE;
        fmt = ChoosePixelFormat(ctx->hdc, &pfd);
        if (!fmt || !SetPixelFormat(ctx->hdc, fmt, &pfd)) {
            snprintf(ctx->error, sizeof(ctx->error),
                     "SetPixelFormat failed (err=%lu)", GetLastError());
            return false;
        }
    }

    /* Step 4: Create a dummy legacy context to bootstrap wglCreateContextAttribsARB. */
    HGLRC dummy = wglCreateContext(ctx->hdc);
    if (!dummy) {
        snprintf(ctx->error, sizeof(ctx->error),
                 "wglCreateContext (dummy) failed (err=%lu)", GetLastError());
        return false;
    }
    if (!wglMakeCurrent(ctx->hdc, dummy)) {
        wglDeleteContext(dummy);
        snprintf(ctx->error, sizeof(ctx->error), "wglMakeCurrent (dummy) failed");
        return false;
    }

    /* Step 5: Load wglCreateContextAttribsARB from the dummy context. */
    wglCreateContextAttribsARB =
        (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
    if (!wglCreateContextAttribsARB) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(dummy);
        snprintf(ctx->error, sizeof(ctx->error),
                 "wglCreateContextAttribsARB not found — need WGL_ARB_create_context");
        return false;
    }

    /* Step 6: Create a proper GL 4.3 core context. */
    const int attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 3,
        WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        0
    };
    ctx->hglrc = wglCreateContextAttribsARB(ctx->hdc, (HGLRC)hShareContext, attribs);
    if (!ctx->hglrc && hShareContext) {
        /* Sharing failed (e.g. share context is current on another thread, or
         * pixel-format mismatch despite best efforts).  Retry without sharing
         * so at least compute shaders work; GPU-direct UV blit will be skipped
         * but the CPU readback fallback path remains active. */
        DWORD e = GetLastError();
        fprintf(stderr, "[gpu-dispatch] shared context failed (err=%lu / 0x%08lx, display_pf=%d hidden_pf=%d), retrying without share\n",
                e, (unsigned long)e,
                hShareDC ? GetPixelFormat((HDC)hShareDC) : 0,
                GetPixelFormat(ctx->hdc)); fflush(stderr);
        ctx->hglrc = wglCreateContextAttribsARB(ctx->hdc, nullptr, attribs);
        if (ctx->hglrc) {
            /* Mark that sharing is unavailable so callers can skip blit setup. */
            snprintf(ctx->error, sizeof(ctx->error), "no-share");
        }
    }

    /* Step 7: Tear down the dummy context. */
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(dummy);

    if (!ctx->hglrc) {
        snprintf(ctx->error, sizeof(ctx->error),
                 "wglCreateContextAttribsARB (4.3 core) failed (err=%lu)", GetLastError());
        return false;
    }

    /* Step 8: Make the real context current on this thread. */
    if (!wglMakeCurrent(ctx->hdc, ctx->hglrc)) {
        snprintf(ctx->error, sizeof(ctx->error), "wglMakeCurrent (4.3 core) failed");
        return false;
    }

    ctx->ready = true;
    return true;
}

bool gl_compute_make_current(GlComputeContext* ctx) {
    if (!ctx || !ctx->hdc || !ctx->hglrc) return false;
    return wglMakeCurrent(ctx->hdc, ctx->hglrc) == TRUE;
}

void gl_compute_release_current(void) {
    wglMakeCurrent(nullptr, nullptr);
}

void gl_compute_destroy_context(GlComputeContext* ctx) {
    if (!ctx) return;
    wglMakeCurrent(nullptr, nullptr);
    if (ctx->hglrc) { wglDeleteContext(ctx->hglrc); ctx->hglrc = nullptr; }
    if (ctx->hdc && ctx->hwnd) { ReleaseDC(ctx->hwnd, ctx->hdc); ctx->hdc = nullptr; }
    if (ctx->hwnd)  { DestroyWindow(ctx->hwnd);  ctx->hwnd   = nullptr; }
    if (ctx->wclass && ctx->wclass != (ATOM)1) {
        UnregisterClassA("GlComputeHidden_v1", GetModuleHandleA(nullptr));
        ctx->wclass = 0;
    }
    ctx->ready = false;
}

bool gl_compute_load_procs(void) {
#define LOAD(name, type, sym) \
    name = (type)gl_get_proc(#sym); \
    if (!name) { return false; }

    LOAD(glc_GenBuffers,       PFNGLGENBUFFERSPROC,       glGenBuffers)
    LOAD(glc_DeleteBuffers,    PFNGLDELETEBUFFERSPROC,    glDeleteBuffers)
    LOAD(glc_BindBuffer,       PFNGLBINDBUFFERPROC,       glBindBuffer)
    LOAD(glc_BufferData,       PFNGLBUFFERDATAPROC,       glBufferData)
    LOAD(glc_BufferSubData,    PFNGLBUFFERSUBDATAPROC,    glBufferSubData)
    LOAD(glc_GetBufferSubData, PFNGLGETBUFFERSUBDATAPROC, glGetBufferSubData)
    LOAD(glc_BindBufferBase,   PFNGLBINDBUFFERBASEPROC,   glBindBufferBase)
    LOAD(glc_BindBufferRange,  PFNGLBINDBUFFERRANGEPROC,  glBindBufferRange)

    LOAD(glc_CreateShader,     PFNGLCREATESHADERPROC,     glCreateShader)
    LOAD(glc_ShaderSource,     PFNGLSHADERSOURCEPROC,     glShaderSource)
    LOAD(glc_CompileShader,    PFNGLCOMPILESHADERPROC,    glCompileShader)
    LOAD(glc_GetShaderiv,      PFNGLGETSHADERIVPROC,      glGetShaderiv)
    LOAD(glc_GetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog)
    LOAD(glc_DeleteShader,     PFNGLDELETESHADERPROC,     glDeleteShader)

    LOAD(glc_CreateProgram,    PFNGLCREATEPROGRAMPROC,    glCreateProgram)
    LOAD(glc_AttachShader,     PFNGLATTACHSHADERPROC,     glAttachShader)
    LOAD(glc_LinkProgram,      PFNGLLINKPROGRAMPROC,      glLinkProgram)
    LOAD(glc_GetProgramiv,     PFNGLGETPROGRAMIVPROC,     glGetProgramiv)
    LOAD(glc_GetProgramInfoLog,PFNGLGETPROGRAMINFOLOGPROC,glGetProgramInfoLog)
    LOAD(glc_UseProgram,       PFNGLUSEPROGRAMPROC,       glUseProgram)
    LOAD(glc_DeleteProgram,    PFNGLDELETEPROGRAMPROC,    glDeleteProgram)

    LOAD(glc_GetUniformLocation,PFNGLGETUNIFORMLOCATIONPROC,glGetUniformLocation)
    LOAD(glc_Uniform1i,        PFNGLUNIFORM1IPROC,        glUniform1i)
    LOAD(glc_Uniform1f,        PFNGLUNIFORM1FPROC,        glUniform1f)
    LOAD(glc_Uniform1ui,       PFNGLUNIFORM1UIPROC,       glUniform1ui)
    LOAD(glc_Uniform1iv,       PFNGLUNIFORM1IVPROC,       glUniform1iv)
    LOAD(glc_Uniform1fv,       PFNGLUNIFORM1FVPROC,       glUniform1fv)
    LOAD(glc_Uniform3fv,       PFNGLUNIFORM3FVPROC,       glUniform3fv)
    LOAD(glc_Uniform4fv,       PFNGLUNIFORM4FVPROC,       glUniform4fv)
    LOAD(glc_Uniform2iv,       PFNGLUNIFORM2IVPROC,       glUniform2iv)

    LOAD(glc_DispatchCompute,         PFNGLDISPATCHCOMPUTEPROC,         glDispatchCompute)
    LOAD(glc_DispatchComputeIndirect, PFNGLDISPATCHCOMPUTEINDIRECTPROC, glDispatchComputeIndirect)
    LOAD(glc_MemoryBarrier,           PFNGLMEMORYBARRIERPROC,           glMemoryBarrier)
    LOAD(glc_FenceSync,        PFNGLFENCESYNCPROC,         glFenceSync)
    LOAD(glc_ClientWaitSync,   PFNGLCLIENTWAITSYNCPROC,    glClientWaitSync)
    LOAD(glc_DeleteSync,       PFNGLDELETESYNCPROC,        glDeleteSync)

    LOAD(glc_BindImageTexture, PFNGLBINDIMAGETEXTUREPROC, glBindImageTexture)
    LOAD(glc_TexStorage3D,     PFNGLTEXSTORAGE3DPROC,     glTexStorage3D)

    LOAD(glc_GenVertexArrays,  PFNGLGENVERTEXARRAYSPROC,  glGenVertexArrays)
    LOAD(glc_BindVertexArray,  PFNGLBINDVERTEXARRAYPROC,  glBindVertexArray)
    LOAD(glc_DeleteVertexArrays,PFNGLDELETEVERTEXARRAYSPROC,glDeleteVertexArrays)
#undef LOAD
    /* Best-effort (GL 3.0 core; never gate context creation on it). */
    glc_GetIntegeri_v = (PFNGLGETINTEGERI_VPROC)gl_get_proc("glGetIntegeri_v");
    return true;
}

/* ── Shader compiler helper ─────────────────────────────────────────────── */

GLuint gl_compute_build_program(const char* glsl_source, char* err_out, int err_sz) {
    GLuint shader = glc_CreateShader(GL_COMPUTE_SHADER);
    if (!shader) {
        if (err_out) snprintf(err_out, err_sz, "glCreateShader failed");
        return 0;
    }

    const GLchar* src_ptr = glsl_source;
    glc_ShaderSource(shader, 1, &src_ptr, nullptr);
    glc_CompileShader(shader);

    GLint status = 0;
    glc_GetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        if (err_out) {
            glc_GetShaderInfoLog(shader, err_sz, nullptr, err_out);
        }
        glc_DeleteShader(shader);
        return 0;
    }

    GLuint prog = glc_CreateProgram();
    if (!prog) {
        glc_DeleteShader(shader);
        if (err_out) snprintf(err_out, err_sz, "glCreateProgram failed");
        return 0;
    }
    glc_AttachShader(prog, shader);
    glc_LinkProgram(prog);
    glc_DeleteShader(shader);  /* shader can be freed after linking */

    glc_GetProgramiv(prog, GL_LINK_STATUS, &status);
    if (!status) {
        if (err_out) {
            glc_GetProgramInfoLog(prog, err_sz, nullptr, err_out);
        }
        glc_DeleteProgram(prog);
        return 0;
    }
    return prog;
}

/* Two-source variant: splits main_source at the end of its #version line,
 * then passes [version_line, preamble, rest_of_shader] as three source strings
 * so the driver sees one compilation unit with preamble declarations available
 * throughout the main shader body.  If preamble is nullptr, falls back to the
 * single-source path. */
GLuint gl_compute_build_program2(const char* main_source, const char* preamble,
                                  char* err_out, int err_sz) {
    if (!preamble || preamble[0] == '\0')
        return gl_compute_build_program(main_source, err_out, err_sz);

    /* Find end of first line (the #version directive). */
    const char* nl = main_source;
    while (*nl && *nl != '\n') ++nl;
    if (*nl == '\n') ++nl;  /* include the newline in the version string */

    /* version_part = everything up to and including the first newline.
     * rest_part    = everything after it. */
    std::string version_part(main_source, static_cast<size_t>(nl - main_source));
    const char* rest_part = nl;

    GLuint shader = glc_CreateShader(GL_COMPUTE_SHADER);
    if (!shader) {
        if (err_out) snprintf(err_out, err_sz, "glCreateShader failed");
        return 0;
    }

    const GLchar* srcs[3] = {
        version_part.c_str(),
        preamble,
        rest_part
    };
    glc_ShaderSource(shader, 3, srcs, nullptr);
    glc_CompileShader(shader);

    GLint status = 0;
    glc_GetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        if (err_out) glc_GetShaderInfoLog(shader, err_sz, nullptr, err_out);
        glc_DeleteShader(shader);
        return 0;
    }

    GLuint prog = glc_CreateProgram();
    if (!prog) {
        glc_DeleteShader(shader);
        if (err_out) snprintf(err_out, err_sz, "glCreateProgram failed");
        return 0;
    }
    glc_AttachShader(prog, shader);
    glc_LinkProgram(prog);
    glc_DeleteShader(shader);

    glc_GetProgramiv(prog, GL_LINK_STATUS, &status);
    if (!status) {
        if (err_out) glc_GetProgramInfoLog(prog, err_sz, nullptr, err_out);
        glc_DeleteProgram(prog);
        return 0;
    }
    return prog;
}

#else  /* !_WIN32 — stub implementation for non-Windows platforms */

bool gl_compute_create_context(GlComputeContext* ctx, void* /*hShareContext*/) {
    if (ctx) snprintf(ctx->error, sizeof(ctx->error),
                      "gl_compute: not supported on this platform");
    return false;
}
bool gl_compute_make_current(GlComputeContext*) { return false; }
void gl_compute_release_current(void) {}
void gl_compute_destroy_context(GlComputeContext*) {}
bool gl_compute_load_procs(void) { return false; }
GLuint gl_compute_build_program(const char*, char*, int) { return 0; }
GLuint gl_compute_build_program2(const char*, const char*, char*, int) { return 0; }

#endif /* _WIN32 */
