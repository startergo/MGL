/* geom: multisample image sidecar — image2DMS / image2DMSArray load, store,
 * atomics through the SPIR-V lowering pass (mgl_spirv_lower_ms_images), the
 * sidecar seed/flush kernels and the barrier ordering.  Pixel-verified with
 * glReadPixels / texelFetch. */
#include "mgl_test.h"

static const char *TRI_VS =
    "#version 460 core\n"
    "void main(){vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);"
    " gl_Position = vec4(p*2.0-1.0,0,1);}";

static GLuint mk_fbo_rgba8(int w, int h)
{
    GLuint rbo = 0, fbo = 0;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    glViewport(0, 0, w, h);
    return fbo;
}

/* one fragment covers exactly texel (2,2); four stores, one per sample */
TEST(ms_store_per_sample)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, 8, 8, GL_TRUE);

    GLuint fbo = mk_fbo_rgba8(8, 8);
    (void)fbo;
    static const char *sfs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMS img;\n"
        "void main() {\n"
        "  if (ivec2(gl_FragCoord.xy) != ivec2(2)) { o = vec4(0); return; }\n"
        "  imageStore(img, ivec2(2), 0, vec4(10.0, 20.0, 30.0, 40.0));\n"
        "  imageStore(img, ivec2(2), 1, vec4(11.0, 21.0, 31.0, 41.0));\n"
        "  imageStore(img, ivec2(2), 2, vec4(12.0, 22.0, 32.0, 42.0));\n"
        "  imageStore(img, ivec2(2), 3, vec4(13.0, 23.0, 33.0, 43.0));\n"
        "  o = vec4(1);\n"
        "}\n";
    GLuint sp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, sfs);
    EXPECT_TRUE(sp != 0);
    static const char *ffs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 3) uniform sampler2DMS tex;\n"
        "uniform int s;\n"
        "void main() { o = texelFetch(tex, ivec2(2), s) / 100.0; }\n";
    GLuint fp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, ffs);
    EXPECT_TRUE(fp != 0);
    GLint su = glGetUniformLocation(fp, "s");
    EXPECT_TRUE(su >= 0);

    glUseProgram(sp);
    glBindImageTexture(1, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    glFinish();

    glUseProgram(fp);
    glBindTextureUnit(3, tex);
    for (int s = 0; s < 4; s++)
    {
        glUniform1i(su, s);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        float px[4] = { -9, -9, -9, -9 };
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
        EXPECT_NEAR(px[0], (10.0f + s) / 100.0f, 1.0f / 255.0f);
        EXPECT_NEAR(px[1], (20.0f + s) / 100.0f, 1.0f / 255.0f);
        EXPECT_NEAR(px[2], (30.0f + s) / 100.0f, 1.0f / 255.0f);
        EXPECT_NEAR(px[3], (40.0f + s) / 100.0f, 1.0f / 255.0f);
    }
    EXPECT_NO_GL_ERROR();
}

/* stores at FIXED texel coordinates across the image: catches sidecar-layout
 * collisions between (texel, sample) pairs — with the old word formula
 * 4*sample + S*linear, texel(1,4)/s3 and texel(4,4)/s0 share word 144 and
 * the two stores stomp each other.  All fragments write identical values, so
 * there is no write race. */
TEST(ms_store_layout_no_collision)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, 8, 8, GL_TRUE);

    GLuint fbo = mk_fbo_rgba8(8, 8);
    (void)fbo;
    static const char *sfs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMS img;\n"
        "void main() {\n"
        "  imageStore(img, ivec2(0, 0), 0, vec4(1.0, 2.0, 3.0, 4.0));\n"
        "  imageStore(img, ivec2(7, 7), 3, vec4(13.0, 14.0, 15.0, 16.0));\n"
        "  imageStore(img, ivec2(1, 4), 3, vec4(100.0, 101.0, 102.0, 103.0));\n"
        "  imageStore(img, ivec2(4, 4), 0, vec4(50.0, 51.0, 52.0, 53.0));\n"
        "  o = vec4(1);\n"
        "}\n";
    GLuint sp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, sfs);
    EXPECT_TRUE(sp != 0);
    static const char *ffs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 3) uniform sampler2DMS tex;\n"
        "uniform ivec2 tc;\n"
        "uniform int s;\n"
        "void main() { o = texelFetch(tex, tc, s) / 200.0; }\n";
    GLuint fp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, ffs);
    EXPECT_TRUE(fp != 0);
    GLint tcu = glGetUniformLocation(fp, "tc"), su = glGetUniformLocation(fp, "s");
    EXPECT_TRUE(tcu >= 0 && su >= 0);

    glUseProgram(sp);
    glBindImageTexture(1, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    glFinish();

    static const struct { int x, y, s; float v[4]; } cases[] = {
        { 0, 0, 0, { 1, 2, 3, 4 } },
        { 7, 7, 3, { 13, 14, 15, 16 } },
        { 1, 4, 3, { 100, 101, 102, 103 } },
        { 4, 4, 0, { 50, 51, 52, 53 } },
        { 4, 4, 3, { 0, 0, 0, 0 } },             /* never written */
    };
    glUseProgram(fp);
    glBindTextureUnit(3, tex);
    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++)
    {
        glUniform2i(tcu, cases[c].x, cases[c].y);
        glUniform1i(su, cases[c].s);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        float px[4] = { -9, -9, -9, -9 };
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
        for (int k = 0; k < 4; k++)
            EXPECT_NEAR(px[k], cases[c].v[k] / 200.0f, 1.0f / 255.0f);
    }
    EXPECT_NO_GL_ERROR();
}

/* imageLoad of an MS image: the sidecar is seeded from the texture, the read
 * goes through the sidecar */
TEST(ms_image_load)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, 8, 8, GL_TRUE);

    /* put a distinct value per sample by rendering with gl_SampleID */
    {
        GLuint cf = 0;
        glGenFramebuffers(1, &cf);
        glBindFramebuffer(GL_FRAMEBUFFER, cf);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
        EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), (int)GL_FRAMEBUFFER_COMPLETE);
        glViewport(0, 0, 8, 8);
        static const char *cfs =
            "#version 460 core\n"
            "layout(location = 0) out vec4 o;\n"
            "void main() { o = vec4(gl_SampleID) * 0.25 + 0.25; }\n";
        GLuint cp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, cfs);
        EXPECT_TRUE(cp != 0);
        glUseProgram(cp);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
    }

    GLuint fbo = mk_fbo_rgba8(8, 8);
    (void)fbo;
    static const char *lfs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 2, rgba32f) readonly uniform image2DMS img;\n"
        "uniform int s;\n"
        "void main() { o = imageLoad(img, ivec2(5), s); }\n";
    GLuint lp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, lfs);
    EXPECT_TRUE(lp != 0);
    GLint su = glGetUniformLocation(lp, "s");
    EXPECT_TRUE(su >= 0);
    glUseProgram(lp);
    glBindImageTexture(2, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();
    for (int s = 0; s < 4; s++)
    {
        glUniform1i(su, s);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        float px[4] = { -9, -9, -9, -9 };
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
        EXPECT_NEAR(px[0], 0.25f + 0.25f * s, 1.0f / 255.0f);
        EXPECT_NEAR(px[1], 0.25f + 0.25f * s, 1.0f / 255.0f);
        EXPECT_NEAR(px[2], 0.25f + 0.25f * s, 1.0f / 255.0f);
        EXPECT_NEAR(px[3], 0.25f + 0.25f * s, 1.0f / 255.0f);
    }
}

/* imageAtomicAdd on an MS image: two draws accumulate at two samples of one
 * texel; the other samples stay zero */
TEST(ms_image_atomic)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_R32UI, 8, 8, GL_TRUE);

    GLuint fbo = mk_fbo_rgba8(8, 8);
    (void)fbo;
    static const char *afs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 1, r32ui) uniform uimage2DMS img;\n"
        "uniform int s;\n"
        "uniform uint add;\n"
        "void main() {\n"
        "  imageAtomicAdd(img, ivec2(3), s, add);\n"
        "  o = vec4(1);\n"
        "}\n";
    GLuint ap = mgl_make_program5(TRI_VS, NULL, NULL, NULL, afs);
    EXPECT_TRUE(ap != 0);
    if (!ap) return;
    GLint su = glGetUniformLocation(ap, "s"), au = glGetUniformLocation(ap, "add");
    glUseProgram(ap);
    glBindImageTexture(1, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    EXPECT_NO_GL_ERROR();
    glUniform1ui(au, 7);
    glUniform1i(su, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUniform1i(su, 2);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    glFinish();

    static const char *ffs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 3) uniform usampler2DMS tex;\n"
        "uniform int s;\n"
        "void main() { o = vec4(float(texelFetch(tex, ivec2(3), s)) / 14.0); }\n";
    GLuint fp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, ffs);
    EXPECT_TRUE(fp != 0);
    GLint su2 = glGetUniformLocation(fp, "s");
    glUseProgram(fp);
    glBindTextureUnit(3, tex);
    for (int s = 0; s < 4; s++)
    {
        glUniform1i(su2, s);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        float px[4] = { -9, -9, -9, -9 };
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
        EXPECT_NEAR(px[0], s == 1 || s == 2 ? 1.0f : 0.0f, 1.0f / 255.0f);
    }
    EXPECT_NO_GL_ERROR();
}

/* image2DMSArray: stores across layers and samples, read back per (layer, sample) */
TEST(msarray_store)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, tex);
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 4, GL_RGBA32F, 8, 8, 2, GL_TRUE);

    GLuint fbo = mk_fbo_rgba8(8, 8);
    (void)fbo;
    static const char *sfs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMSArray img;\n"
        "void main() {\n"
        "  imageStore(img, ivec3(2, 2, 0), 1, vec4(10.0, 20.0, 30.0, 40.0));\n"
        "  imageStore(img, ivec3(2, 2, 1), 2, vec4(11.0, 21.0, 31.0, 41.0));\n"
        "  o = vec4(1);\n"
        "}\n";
    GLuint sp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, sfs);
    EXPECT_TRUE(sp != 0);
    static const char *ffs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "layout(binding = 3) uniform sampler2DMSArray tex;\n"
        "uniform int layer, s;\n"
        "void main() { o = texelFetch(tex, ivec3(2, 2, layer), s) / 100.0; }\n";
    GLuint fp = mgl_make_program5(TRI_VS, NULL, NULL, NULL, ffs);
    EXPECT_TRUE(fp != 0);
    GLint lu = glGetUniformLocation(fp, "layer"), su = glGetUniformLocation(fp, "s");
    EXPECT_TRUE(lu >= 0 && su >= 0);

    glUseProgram(sp);
    glBindImageTexture(1, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    glFinish();

    glUseProgram(fp);
    glBindTextureUnit(3, tex);
    for (int l = 0; l < 2; l++)
        for (int s = 0; s < 4; s++)
        {
            glUniform1i(lu, l);
            glUniform1i(su, s);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            float px[4] = { -9, -9, -9, -9 };
            glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
            /* only the two stored (layer, sample) combos carry a value; every
             * other combination was never written and reads back as zero */
            int stored = ((l == 0 && s == 1) || (l == 1 && s == 2));
            float base = (l == 0 && s == 1) ? 10.0f : 11.0f;
            for (int k = 0; k < 4; k++)
                EXPECT_NEAR(px[k], stored ? (base + 10.0f * k) / 100.0f : 0.0f, 1.0f / 255.0f);
        }
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(ms_store_per_sample);
    RUN(ms_store_layout_no_collision);
    RUN(ms_image_load);
    RUN(ms_image_atomic);
    RUN(msarray_store);
    return mgl_test_summary();
}
