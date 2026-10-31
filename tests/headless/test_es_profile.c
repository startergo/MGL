/*
 * test_es_profile.c — '#version 310 es' / '320 es' shaders keep ES semantics
 * in a desktop GL 4.6 context (ARB_ES3_1_compatibility / ES3_2_compatibility).
 * Regression for the glslang force-to-core bug: ~3700 of the first 12k
 * dEQP-GL45-ES31 cases failed with "compute shaders require es profile with
 * version 310", "arrays of arrays not supported", "samplerCubeArray Reserved
 * word", "gl_HelperInvocation undeclared", "'binding' not supported" (2026-10-31).
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>

/* compute at #version 310 es (the shared_var repro), layout(binding=) */
TEST(es310_compute)
{
    const char *cs =
        "#version 310 es\n"
        "layout(local_size_x = 8) in;\n"
        "layout(binding = 0) buffer Out { int data[8]; };\n"
        "shared int tmp[8];\n"
        "void main()\n"
        "{\n"
        "    tmp[gl_LocalInvocationID.x] = int(gl_GlobalInvocationID.x) + 1;\n"
        "    barrier();\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < 8; i++) s += tmp[i];\n"
        "    data[gl_GlobalInvocationID.x] = s;\n"
        "}";
    GLuint p = mgl_make_program_cs(cs);
    if (!p) { EXPECT_TRUE(p != 0); return; }
    EXPECT_NO_GL_ERROR();

    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 8 * sizeof(GLint), NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);

    glUseProgram(p);
    glDispatchCompute(8, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    EXPECT_NO_GL_ERROR();

    GLint out[8] = { 0 };
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    /* every lane sees the same sum 1..8 = 36 */
    for (int i = 0; i < 8; i++)
        EXPECT_EQ_INT(out[i], 36);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glDeleteProgram(p);
    glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

/* arrays of arrays + samplerCubeArray at #version 310 es */
TEST(es310_features)
{
    const char *vs =
        "#version 310 es\n"
        "in vec4 pos;\n"
        "void main() { gl_Position = pos; }";
    const char *fs =
        "#version 310 es\n"
        "precision highp float;\n"
        "#extension GL_OES_texture_cube_map_array : enable\n"
        "precision highp samplerCubeArray;\n"
        "uniform samplerCubeArray ca;\n"
        "uniform vec4 c[2][2];   /* arrays of arrays: ES 3.1 */\n"
        "out vec4 o;\n"
        "void main()\n"
        "{\n"
        "    vec4 a = texture(ca, vec4(0.5));\n"
        "    o = c[0][0] + c[1][1] + a * 0.0;\n"
        "}";
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    if (!p) { EXPECT_TRUE(p != 0); return; }
    EXPECT_NO_GL_ERROR();

    GLint loc = glGetUniformLocation(p, "c[0][0]");
    EXPECT_TRUE(loc >= 0);
    glUseProgram(p);
    float vals[16];
    for (int i = 0; i < 16; i++) vals[i] = (float)i * 0.5f;
    glUniform4fv(loc, 4, vals);   /* 2x2 array-of-arrays flattened: 4 vec4s */
    GLint loc11 = glGetUniformLocation(p, "c[1][1]");
    EXPECT_TRUE(loc11 >= 0);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

/* gl_HelperInvocation (ES 3.1) in a #version 310 es fragment shader */
TEST(es310_helper_invocation)
{
    const char *vs =
        "#version 310 es\n"
        "void main() { gl_Position = vec4(0, 0, 0, 1); }";
    const char *fs =
        "#version 310 es\n"
        "precision highp float;\n"
        "out vec4 o;\n"
        "void main()\n"
        "{\n"
        "    if (gl_HelperInvocation) discard;\n"
        "    o = vec4(1.0, 0.0, 1.0, 1.0);\n"
        "}";
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    if (!p) { EXPECT_TRUE(p != 0); return; }
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

/* #version 320 es (ES3_2_compatibility) */
TEST(es320_shader)
{
    const char *cs =
        "#version 320 es\n"
        "layout(binding = 0) writeonly uniform highp image2D img;\n"
        "layout(local_size_x = 4) in;\n"
        "void main() { imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(0.25, 0.5, 0.75, 1.0)); }";
    GLuint p = mgl_make_program_cs(cs);
    if (!p) { EXPECT_TRUE(p != 0); return; }
    EXPECT_NO_GL_ERROR();

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
    EXPECT_NO_GL_ERROR();

    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLubyte px[4] = { 0, 0, 0, 0 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    EXPECT_NEAR(px[0], 64, 2);   /* 0.25 */
    EXPECT_NEAR(px[1], 128, 2);  /* 0.5  */
    EXPECT_NEAR(px[2], 191, 2);  /* 0.75 */
    EXPECT_NEAR(px[3], 255, 2);

    /* NOTE: unbind the image unit before deleting the texture — MGL's
     * context teardown dereferences a texture still referenced by an image
     * unit (pre-existing at 965d88f, desktop shaders too; reported, not
     * fixed here — textures domain). */
    glDeleteProgram(p);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ES 3.0 shaders still compile through the legacy desktop mapping */
TEST(es300_still_works)
{
    const char *vs =
        "#version 300 es\n"
        "in vec4 pos; out vec4 vc;\n"
        "void main() { gl_Position = pos; vc = pos; }";
    const char *fs =
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec4 vc; out vec4 o;\n"
        "void main() { o = vc; }";
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    if (!p) { EXPECT_TRUE(p != 0); return; }
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

/* desktop GLSL is untouched by the ES path */
TEST(desktop_unaffected)
{
    const char *vs = "#version 330 core\nin vec4 pos;\nvoid main(){ gl_Position = pos; }";
    const char *fs = "#version 330 core\nout vec4 o;\nvoid main(){ o = vec4(1,0,0,1); }";
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    if (!p) { EXPECT_TRUE(p != 0); return; }
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

int main(void)
{
    setbuf(stdout, NULL);
    RUN(es320_shader);
    RUN(es310_compute);
    RUN(es310_features);
    RUN(es310_helper_invocation);
    RUN(es300_still_works);
    RUN(desktop_unaffected);
    return mgl_test_summary();
}
