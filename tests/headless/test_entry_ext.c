/*
 * test_entry_ext.c — every entry point of every ADVERTISED extension must
 * resolve through a GL loader (dlsym) and behave.  Regression for the
 * glBlendBarrierKHR NULL-symbol SEGFAULT in dEQP-GL45-ES3 draw_buffers_indexed
 * (2026-10-31).  The full advertised-extension audit is ab/ext_entry_audit.py;
 * GL_ARB_gpu_shader_int64's uniform entry points are the one documented gap.
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include "glcorearb.h"

/* prototypes for the aliases this test exercises (entry_ext_compat.c) */
extern void glBlendBarrierKHR(void);
extern void glBlendBarrierNV(void);
extern void glClearColorIiEXT(GLint r, GLint g, GLint b, GLint a);
extern void glClearColorIuiEXT(GLuint r, GLuint g, GLuint b, GLuint a);
extern void glBlendEquationiARB(GLuint buf, GLenum mode);
extern void glBlendFunciARB(GLuint buf, GLenum src, GLenum dst);
extern void glTexBufferARB(GLenum target, GLenum internalformat, GLuint buffer);
extern void glDepthRangeIndexeddNV(GLuint index, GLdouble n, GLdouble f);
extern void glDebugMessageControlKHR(GLenum source, GLenum type, GLenum severity,
                                     GLsizei count, const GLuint *ids, GLboolean enabled);
extern void glGetnPolygonStippleARB(GLsizei bufSize, GLubyte *pattern);

TEST(blend_barrier_resolves_and_is_silent)
{
    /* dlsym exactly like the glcts loader does */
    void *h = dlopen(NULL, RTLD_NOW);
    EXPECT_TRUE(h != NULL);
    const char *need[] = {
        "glBlendBarrierKHR", "glBlendBarrierNV", "glBlendBarrier",
        "glBlendEquationiARB", "glBlendEquationSeparateiARB", "glBlendFunciARB", "glBlendFuncSeparateiARB",
        "glGenQueriesARB", "glDeleteQueriesARB", "glIsQueryARB", "glBeginQueryARB", "glEndQueryARB",
        "glGetQueryivARB", "glGetQueryObjectivARB", "glGetQueryObjectuivARB",
        "glTexBufferARB", "glDepthRangeArraydvNV", "glDepthRangeIndexeddNV",
        "glTexParameterIivEXT", "glTexParameterIuivEXT", "glGetTexParameterIivEXT",
        "glGetTexParameterIuivEXT", "glClearColorIiEXT", "glClearColorIuiEXT",
        "glDebugMessageControlKHR", "glDebugMessageInsertKHR", "glDebugMessageCallbackKHR",
        "glGetDebugMessageLogKHR", "glPushDebugGroupKHR", "glPopDebugGroupKHR",
        "glObjectLabelKHR", "glGetObjectLabelKHR", "glObjectPtrLabelKHR",
        "glGetObjectPtrLabelKHR", "glGetPointervKHR",
        "glGetGraphicsResetStatusARB", "glReadnPixelsARB", "glGetnTexImageARB",
        "glGetnCompressedTexImageARB", "glGetnUniformfvARB", "glGetnUniformivARB",
        "glGetnUniformuivARB", "glGetnUniformdvARB", "glGetnMapdvARB", "glGetnMapfvARB",
        "glGetnMapivARB", "glGetnPixelMapfvARB", "glGetnPixelMapuivARB",
        "glGetnPixelMapusvARB", "glGetnPolygonStippleARB",
        "glMultiDrawArraysIndirectCountARB", "glMultiDrawElementsIndirectCountARB",
    };
    for (size_t i = 0; i < sizeof need / sizeof need[0]; i++)
    {
        void *p = dlsym(h, need[i]);
        if (!p) MGL_FAIL("dlsym(%s) == NULL (loader would segfault)", need[i]);
        EXPECT_TRUE(p != NULL);
    }
    dlclose(h);

    /* the barrier is a documented no-op (raster-ordered framebuffer fetch):
     * callable repeatedly, no GL error, no state damage */
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glBlendBarrierKHR();
    glBlendBarrierNV();
    EXPECT_NO_GL_ERROR();
    GLfloat cc[4] = { -1, -1, -1, -1 };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, cc);
    EXPECT_NEAR(cc[0], 0.25f, 1e-6);
    EXPECT_NEAR(cc[1], 0.5f, 1e-6);
}

TEST(clear_color_i_ext_clears_integer_fbo_exactly)
{
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8I, 4, 4, 0, GL_RGBA_INTEGER, GL_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* float clear colour must NOT be used for the integer attachment */
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClearColorIiEXT(5, -3, 100, -128);
    EXPECT_NO_GL_ERROR();
    glClear(GL_COLOR_BUFFER_BIT);
    GLint px[4] = { 0, 0, 0, 0 };
    glReadPixels(1, 1, 1, 1, GL_RGBA_INTEGER, GL_INT, px);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(px[0], 5);
    EXPECT_EQ_INT(px[1], -3);
    EXPECT_EQ_INT(px[2], 100);
    EXPECT_EQ_INT(px[3], -128);

    /* unsigned variant: exact bits, on an unsigned attachment (on the signed
     * RGBA8I above, 200 would clamp to 127 per EXT_texture_integer) */
    GLuint utex = 0, ufbo = 0;
    glGenTextures(1, &utex);
    glBindTexture(GL_TEXTURE_2D, utex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, 4, 4, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &ufbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ufbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, utex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glClearColorIuiEXT(200, 7, 255, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint upx[4] = { 0, 0, 0, 0 };
    glReadPixels(1, 1, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, upx);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(upx[0], 200);
    EXPECT_EQ_INT(upx[1], 7);
    EXPECT_EQ_INT(upx[2], 255);
    EXPECT_EQ_INT(upx[3], 1);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &ufbo);
    glDeleteTextures(1, &utex);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

TEST(aliases_forward_to_core_state)
{
    /* ARB_draw_buffers_blend alias reaches the per-buffer blend state */
    GLint eq = 0;
    glDisable(GL_BLEND);
    glBlendEquationiARB(0, GL_FUNC_ADD);
    EXPECT_NO_GL_ERROR();
    glGetIntegeri_v(GL_BLEND_EQUATION_RGB, 0, &eq);
    EXPECT_EQ_INT(eq, GL_FUNC_ADD);
    glBlendEquationiARB(0, GL_FUNC_REVERSE_SUBTRACT);
    glGetIntegeri_v(GL_BLEND_EQUATION_RGB, 0, &eq);
    EXPECT_EQ_INT(eq, GL_FUNC_REVERSE_SUBTRACT);
    glBlendEquationiARB(0, GL_FUNC_ADD);

    /* ARB_texture_buffer_object alias is a real buffer texture */
    GLuint buf = 0, tbo = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, 64, NULL, GL_STATIC_DRAW);
    glGenTextures(1, &tbo);
    glBindTexture(GL_TEXTURE_BUFFER, tbo);
    glTexBufferARB(GL_TEXTURE_BUFFER, GL_R32F, buf);
    EXPECT_NO_GL_ERROR();
    GLint fmt = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_BUFFER, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt);
    EXPECT_EQ_INT(fmt, GL_R32F);
    glDeleteTextures(1, &tbo);
    glDeleteBuffers(1, &buf);

    /* robustness: reset status alias */
    EXPECT_EQ_INT(glGetGraphicsResetStatusARB(), GL_NO_ERROR);

    /* KHR_debug alias routes through the same unsuffixed path */
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageControlKHR(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_FALSE);
    glDisable(GL_DEBUG_OUTPUT);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(blend_barrier_resolves_and_is_silent);
    RUN(clear_color_i_ext_clears_integer_fbo_exactly);
    RUN(aliases_forward_to_core_state);
    return mgl_test_summary();
}
