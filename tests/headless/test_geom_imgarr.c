/* geom: image-ARRAY sidecars — arrays of storage images (plain and multisample)
 * on the COMPUTE and FRAGMENT paths, element seeding, remapped units, and the
 * r82 zero-cost proof counters (mgl_geom_ms_stat).
 *
 *   SPIR-V pass     geometry.c  mgl_spirv_lower_ms_images (array classification)
 *   MSL text pass   geometry.c  mgl_msl_fix_image_atomics (spvMglAIArrMS_*)
 *   renderer        MGLRenderer+Geometry.m (geomImageAtomicBufferArray,
 *                   geomBindImageAtomicBuffers{Render,Compute},
 *                   geomPrepareImageAtomics{Render,Compute}, geomMsFlushView,
 *                   geomSeedImageAtomic)
 */
#include "mgl_test.h"
#include "mgl_geom_diag.h"

static const char *TRI_VS =
    "#version 460 core\n"
    "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);"
    " gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";

static GLuint mk_fbo_rgba8(int w, int h)
{
    GLuint fbo = 0, rbo = 0;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    return fbo;
}

static GLuint mk_ssbo(GLsizeiptr bytes)
{
    GLuint b = 0;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    return b;
}

static GLuint mk_tex2d_r32ui(int w, int h, uint32_t initial)
{
    GLuint t = 0;
    uint32_t *init = malloc((size_t)w * h * 4);
    for (int i = 0; i < w * h; i++) init[i] = initial;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, init);
    free(init);
    return t;
}

/* ---------------------------------------------------------------------- */
/* A fragment-stage image ARRAY atomic: r82 fixed the render binder, which
 * bound N per-unit sidecars at mtl_index+e (buffers the array MSL never
 * reads) so the result reached no texture.  One 1x1 viewport fragment adds 5
 * to g[1] texel (0,0); a second compute program reads it back with +0. */
TEST(fragment_array_atomic_lands)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex[2] = { mk_tex2d_r32ui(4, 4, 0), mk_tex2d_r32ui(4, 4, 0) };
    glBindImageTexture(1, tex[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glBindImageTexture(2, tex[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    EXPECT_NO_GL_ERROR();

    GLuint fbo = mk_fbo_rgba8(1, 1);
    glViewport(0, 0, 1, 1);
    (void)fbo;

    static const char *fs =
        "#version 460 core\n"
        "layout(binding = 1, r32ui) uniform uimage2D g[2];\n"
        "layout(location = 0) out vec4 col;\n"
        "void main() { uint r = imageAtomicAdd(g[1], ivec2(0), 5u);"
        " col = vec4(float(r) / 255.0, 0.0, 1.0, 1.0); }\n";
    GLuint p = mgl_make_program5(TRI_VS, NULL, NULL, NULL, fs);
    EXPECT_TRUE(p != 0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    /* the returned value is the SEED content: 0 (fresh texture) */
    EXPECT_PIXEL(0, 0, 0, 0, 255, 255, 2);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);

    static const char *cs =
        "#version 460 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 1, r32ui) uniform uimage2D g[2];\n"
        "layout(std430, binding = 0) buffer O { uint v; };\n"
        "void main() { v = imageAtomicAdd(g[1], ivec2(0), 0u); }\n";
    GLuint q = mgl_make_program_cs(cs);
    EXPECT_TRUE(q != 0);
    GLuint ssbo = mk_ssbo(16);
    (void)ssbo;
    glUseProgram(q);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t out = 0;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &out);
    EXPECT_EQ_INT(out, 5u);
}

/* a fragment-stage atomic must see the texture's PRE-EXISTING content: the
 * sidecar is seeded at draw time (r82 — the seed used to be gated on MS slots
 * only, so an r32ui atomic on an uploaded texture returned 0) */
TEST(fragment_atomic_sees_texture_content)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex = mk_tex2d_r32ui(4, 4, 9);
    glBindImageTexture(1, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    GLuint fbo = mk_fbo_rgba8(1, 1);
    glViewport(0, 0, 1, 1);
    (void)fbo;

    static const char *fs =
        "#version 460 core\n"
        "layout(binding = 1, r32ui) uniform uimage2D g;\n"
        "layout(location = 0) out vec4 col;\n"
        "void main() { uint r = imageAtomicAdd(g, ivec2(0), 0u);"
        " col = vec4(float(r) / 255.0, 0.0, 1.0, 1.0); }\n";
    GLuint p = mgl_make_program5(TRI_VS, NULL, NULL, NULL, fs);
    EXPECT_TRUE(p != 0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(0, 0, 9, 0, 255, 255, 2);
}

/* a compute image ARRAY atomic persists across dispatches (guards the compute
 * binder's array side buffer) */
TEST(compute_array_atomic_persists)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex[2] = { mk_tex2d_r32ui(4, 4, 0), mk_tex2d_r32ui(4, 4, 0) };
    glBindImageTexture(1, tex[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glBindImageTexture(2, tex[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    mk_ssbo(16);

    static const char *cs1 =
        "#version 460 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 1, r32ui) uniform uimage2D g[2];\n"
        "layout(std430, binding = 0) buffer O { uint v; };\n"
        "void main() { imageAtomicAdd(g[1], ivec2(1), 7u); v = 1u; }\n";
    static const char *cs2 =
        "#version 460 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 1, r32ui) uniform uimage2D g[2];\n"
        "layout(std430, binding = 0) buffer O { uint v; };\n"
        "void main() { v = imageAtomicAdd(g[1], ivec2(1), 0u); }\n";
    GLuint p1 = mgl_make_program_cs(cs1), p2 = mgl_make_program_cs(cs2);
    EXPECT_TRUE(p1 != 0);
    EXPECT_TRUE(p2 != 0);
    glUseProgram(p1);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);
    glUseProgram(p2);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t out = 0;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &out);
    EXPECT_EQ_INT(out, 7u);
}

/* remapped units: g[0] -> unit 5, g[1] -> unit 3 (image-array elements are
 * independent uniforms — KHR-GL46.advanced-sso-simple binds 0,2,4,6); the
 * array sidecar's CPU layout must agree with the MSL helper's runtime offsets */
TEST(compute_array_atomic_remapped_units)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    /* seeded from the texture's contents (7) so the +11 must return 18 — this
     * also proves the SEED walked the remapped element units */
    GLuint tex[2] = { mk_tex2d_r32ui(4, 4, 7), mk_tex2d_r32ui(4, 4, 7) };
    glBindImageTexture(5, tex[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glBindImageTexture(3, tex[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    mk_ssbo(16);

    static const char *cs1 =
        "#version 460 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 1, r32ui) uniform uimage2D g[2];\n"
        "layout(std430, binding = 0) buffer O { uint v; };\n"
        "void main() { v = imageAtomicAdd(g[1], ivec2(1), 11u); }\n";
    static const char *cs2 =
        "#version 460 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 1, r32ui) uniform uimage2D g[2];\n"
        "layout(std430, binding = 0) buffer O { uint v; };\n"
        "void main() { v = imageAtomicAdd(g[1], ivec2(1), 0u); }\n";
    GLuint p1 = mgl_make_program_cs(cs1), p2 = mgl_make_program_cs(cs2);
    EXPECT_TRUE(p1 != 0);
    EXPECT_TRUE(p2 != 0);
    glUseProgram(p1);
    glUniform1i(glGetUniformLocation(p1, "g[0]"), 5);
    glUniform1i(glGetUniformLocation(p1, "g[1]"), 3);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);
    glUseProgram(p2);
    glUniform1i(glGetUniformLocation(p2, "g[0]"), 5);
    glUniform1i(glGetUniformLocation(p2, "g[1]"), 3);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t out = 0;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &out);
    EXPECT_EQ_INT(out, 18u);
    /* the barrier must resolve element 1's region back into ITS texture (unit 3,
     * not unit 2): read the texel straight out of the texture */
    {
        uint32_t texback[16] = { 0 };
        glBindTexture(GL_TEXTURE_2D, tex[1]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, texback);
        EXPECT_EQ_INT(texback[5], 18u);       /* texel (1,1) */
    }
}

/* an ARRAY of multisample images (image2DMS g[2]) through the compute path:
 * the SPIR-V pass lowers the array element's store into the side buffer with
 * the folded per-sample word layout, and the barrier rasterizes each element's
 * region back into its own texture */
TEST(ms_array_store_per_sample)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex[2];
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex[i]);
        glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, 4, 4, GL_TRUE);
    }
    glBindImageTexture(1, tex[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    glBindImageTexture(2, tex[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();
    mk_ssbo(256);

    static const char *cs =
        "#version 460 core\n"
        "layout(local_size_x = 4, local_size_y = 4) in;\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMS g[2];\n"
        "layout(std430, binding = 0) buffer O { vec4 v; };\n"
        "void main() { imageStore(g[1], ivec2(gl_GlobalInvocationID.xy), 1,"
        " vec4(7.0, 8.0, 9.0, 10.0)); v = vec4(1.0); }\n";
    static const char *csr =
        "#version 460 core\n"
        "layout(local_size_x = 4, local_size_y = 4) in;\n"
        "layout(binding = 1, rgba32f) readonly uniform image2DMS g[2];\n"
        "layout(std430, binding = 0) buffer O { vec4 v[16]; };\n"
        "void main() { int i = int(gl_GlobalInvocationID.x) + 4 * int(gl_GlobalInvocationID.y);"
        " v[i] = imageLoad(g[1], ivec2(gl_GlobalInvocationID.xy), 1); }\n";
    GLuint p = mgl_make_program_cs(cs), q = mgl_make_program_cs(csr);
    EXPECT_TRUE(p != 0);
    EXPECT_TRUE(q != 0);
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);
    glUseProgram(q);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    float out[64] = { 0 };
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(out[i * 4 + 0], 7.0f, 1e-6);
        EXPECT_NEAR(out[i * 4 + 1], 8.0f, 1e-6);
        EXPECT_NEAR(out[i * 4 + 2], 9.0f, 1e-6);
        EXPECT_NEAR(out[i * 4 + 3], 10.0f, 1e-6);
    }
}

/* the FRAGMENT-stage MS array: stores land through the render path (the MS
 * seed runs at draw time) and texelFetch reads every sample back */
TEST(ms_array_fragment_store)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex[2];
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex[i]);
        glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, 2, 2, GL_TRUE);
    }
    glBindImageTexture(1, tex[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(2, tex[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();

    GLuint fbo = mk_fbo_rgba8(2, 2);
    glViewport(0, 0, 1, 1);
    (void)fbo;
    static const char *fs =
        "#version 460 core\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMS g[2];\n"
        "layout(location = 0) out vec4 col;\n"
        "void main() { imageStore(g[1], ivec2(0), 3, vec4(11.0, 12.0, 13.0, 14.0));"
        " col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program5(TRI_VS, NULL, NULL, NULL, fs);
    EXPECT_TRUE(p != 0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);

    static const char *cs =
        "#version 460 core\n"
        "layout(local_size_x = 4) in;\n"
        "layout(binding = 2) uniform sampler2DMS t;\n"
        "layout(std430, binding = 0) buffer O { vec4 v[4]; };\n"
        "void main() { int i = int(gl_GlobalInvocationID.x);"
        " v[i] = texelFetch(t, ivec2(i & 1, i >> 1), 3); }\n";
    GLuint q = mgl_make_program_cs(cs);
    EXPECT_TRUE(q != 0);
    glBindTextureUnit(2, tex[1]);
    glUseProgram(q);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    float out[16] = { 0 };
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 64, out);
    EXPECT_NEAR(out[0], 11.0f, 1e-6);
    EXPECT_NEAR(out[1], 12.0f, 1e-6);
    EXPECT_NEAR(out[2], 13.0f, 1e-6);
    EXPECT_NEAR(out[3], 14.0f, 1e-6);
}

/* image2DMSArray g[2]: arrayed MS elements — the folded layout carries the
 * layer and the helper multiplies the element footprint by the array length */
TEST(msarray_array_store_layered)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex[2];
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, tex[i]);
        glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 4, GL_RGBA32F, 2, 2, 4, GL_TRUE);
    }
    glBindImageTexture(1, tex[0], 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F);
    glBindImageTexture(2, tex[1], 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();
    mk_ssbo(64);

    static const char *cs =
        "#version 460 core\n"
        "layout(local_size_x = 4) in;\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMSArray g[2];\n"
        "layout(std430, binding = 0) buffer O { vec4 v; };\n"
        "void main() { imageStore(g[1], ivec3(int(gl_GlobalInvocationID.x) & 1, 0, 2), 1,"
        " vec4(3.0, 4.0, 5.0, 6.0)); v = vec4(1.0); }\n";
    static const char *csr =
        "#version 460 core\n"
        "layout(local_size_x = 4) in;\n"
        "layout(binding = 1, rgba32f) readonly uniform image2DMSArray g[2];\n"
        "layout(std430, binding = 0) buffer O { vec4 v; };\n"
        "void main() { if (gl_GlobalInvocationID.x == 0u)"
        " v = imageLoad(g[1], ivec3(0, 0, 2), 1); }\n";
    GLuint p = mgl_make_program_cs(cs), q = mgl_make_program_cs(csr);
    EXPECT_TRUE(p != 0);
    EXPECT_TRUE(q != 0);
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);
    glUseProgram(q);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    float out[4] = { 0 };
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, out);
    EXPECT_NEAR(out[0], 3.0f, 1e-6);
    EXPECT_NEAR(out[1], 4.0f, 1e-6);
    EXPECT_NEAR(out[2], 5.0f, 1e-6);
    EXPECT_NEAR(out[3], 6.0f, 1e-6);
}

/* the r82 zero-cost proof, structural: a program with NO image atomics must
 * not move a single sidecar byte — 0 seeds, 0 flushes, 0 resolves, 0
 * allocations across draws and barriers.  r83 adds the gate proof: the
 * "any atomic sidecar slot?" flag is computed from the resource lists exactly
 * ONCE per program (the memo), never per draw. */
TEST(sidecar_zero_cost_when_unused)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint fbo = mk_fbo_rgba8(4, 4);
    glViewport(0, 0, 4, 4);
    (void)fbo;
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 col;\n"
        "void main() { col = vec4(1.0, 0.5, 0.25, 1.0); }\n";
    GLuint p = mgl_make_program5(TRI_VS, NULL, NULL, NULL, fs);
    EXPECT_TRUE(p != 0);
    glUseProgram(p);

    uint64_t s0 = mgl_geom_ms_stat(0), f0 = mgl_geom_ms_stat(1),
             r0 = mgl_geom_ms_stat(2), a0 = mgl_geom_ms_stat(3);
    /* the first draw computes the flag once (the memo miss this test pins) */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    uint64_t g1 = mgl_geom_ms_stat(4);
    for (int i = 0; i < 8; i++) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
    }
    EXPECT_EQ_INT(mgl_geom_ms_stat(4) - g1, 0);       /* r83: the gate is memoized — one walk per program, not per draw */
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_TEXTURE_UPDATE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();
    EXPECT_EQ_INT(mgl_geom_ms_stat(0) - s0, 0);      /* no seeds */
    EXPECT_EQ_INT(mgl_geom_ms_stat(1) - f0, 0);      /* no flushes */
    EXPECT_EQ_INT(mgl_geom_ms_stat(2) - r0, 0);      /* no resolves */
    EXPECT_EQ_INT(mgl_geom_ms_stat(3) - a0, 0);      /* no allocations */
}

/* ...and the used path runs each pass exactly once per sidecar: one MS image
 * store + one barrier = one allocation, one seed, one flush */
TEST(sidecar_runs_once_when_used)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, 4, 4, GL_TRUE);
    glBindImageTexture(1, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();

    GLuint fbo = mk_fbo_rgba8(4, 4);
    glViewport(0, 0, 1, 1);
    (void)fbo;
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 col;\n"
        "layout(binding = 1, rgba32f) writeonly uniform image2DMS g;\n"
        "void main() { imageStore(g, ivec2(0), 2, vec4(1.0, 2.0, 3.0, 4.0));"
        " col = vec4(1.0); }\n";
    GLuint p = mgl_make_program5(TRI_VS, NULL, NULL, NULL, fs);
    EXPECT_TRUE(p != 0);
    glUseProgram(p);

    uint64_t a0 = mgl_geom_ms_stat(3), s0 = mgl_geom_ms_stat(0), f0 = mgl_geom_ms_stat(1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    uint64_t allocs = mgl_geom_ms_stat(3) - a0;
    uint64_t seeds = mgl_geom_ms_stat(0) - s0;
    EXPECT_EQ_INT(allocs, 1);                       /* exactly one sidecar */
    EXPECT_EQ_INT(seeds, 1);                        /* seeded once */
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    EXPECT_EQ_INT(mgl_geom_ms_stat(1) - f0, 1);     /* flushed once */
    /* a second draw + barrier re-uses the sidecar: no new allocation, no
     * re-seed, but the flush repeats (the store may have touched it) */
    uint64_t a1 = mgl_geom_ms_stat(3), s1 = mgl_geom_ms_stat(0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    EXPECT_EQ_INT(mgl_geom_ms_stat(3) - a1, 0);
    EXPECT_EQ_INT(mgl_geom_ms_stat(0) - s1, 0);
}

int main(void)
{
    RUN(fragment_array_atomic_lands);
    RUN(fragment_atomic_sees_texture_content);
    RUN(compute_array_atomic_persists);
    RUN(compute_array_atomic_remapped_units);
    RUN(ms_array_store_per_sample);
    RUN(ms_array_fragment_store);
    RUN(msarray_array_store_layered);
    RUN(sidecar_zero_cost_when_unused);
    RUN(sidecar_runs_once_when_used);
    printf("%d checks, %d failures\n", mgl_test_checks, mgl_test_failures);
    return mgl_test_failures ? 1 : 0;
}
