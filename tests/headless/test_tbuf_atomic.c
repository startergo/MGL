/*
 * test_tbuf_atomic — imageAtomicAdd / imageAtomicCompSwap / imageLoad on a
 * uimageBuffer (GL_R32UI buffer texture) in a compute shader, then CPU readback
 * through the buffer (KHR-GL46.texture_buffer texture_buffer_atomic_functions,
 * which does dispatch -> glMemoryBarrier -> mapBufferRange and expects every
 * texel to hold n*(n-1)/2).
 *
 * Two things are under test:
 *  1. the GPU writes reach the CPU view of the buffer (a buffer texture's Metal
 *     storage aliases the GL buffer: the foreign-write wait in
 *     mgl_buffers_prepare_cpu_read covers it);
 *  2. imageLoad of a texel that a same-kernel imageAtomic* wrote is coherent
 *     with it (mgl_msl_fix_image_atomics routes loads/stores of an
 *     atomics-carrying image through the atomic sidecar).
 */
#include "mgl_test.h"

TEST(tbuf_atomic_add_compwap) {
    const char *cs =
        "#version 450 core\n"
        "layout(local_size_x = 64) in;\n"
        "layout(r32ui, binding = 0) coherent uniform highp uimageBuffer ubuf;\n"
        "void main() {\n"
        "   uint value = imageLoad(ubuf, int(gl_LocalInvocationID.x) + 1).x;\n"
        "   imageAtomicAdd(ubuf, 0, value);\n"
        "   memoryBarrier();\n"
        "   barrier();\n"
        "   value = imageLoad(ubuf, 0).x;\n"
        "   imageAtomicCompSwap(ubuf, int(gl_LocalInvocationID.x) + 1,\n"
        "                       gl_LocalInvocationID.x + 1u, value);\n"
        "}\n";
    const GLuint N = 65;                     /* wg size + 1: lanes touch 1..64 */
    const GLuint expected = N * (N - 1) / 2; /* every texel ends here (CTS) */
    GLuint prog = glCreateProgram();
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &cs, NULL);
    glCompileShader(sh);
    char log[2048]; GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(sh, sizeof log, NULL, log);
               fprintf(stderr, "CS compile:\n%s\n", log); }
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(prog, sizeof log, NULL, log);
               fprintf(stderr, "link:\n%s\n", log); }
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glDeleteProgram(prog); glDeleteShader(sh); return; }

    GLuint tbo, tex;
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo);
    GLuint init[65], i;
    for (i = 0; i < N; i++) init[i] = i;
    glBufferData(GL_TEXTURE_BUFFER, N * sizeof(GLuint), init, GL_DYNAMIC_COPY);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, tbo);
    EXPECT_NO_GL_ERROR();

    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glUseProgram(prog);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    /* the CTS reads with glMapBufferRange; exercise both readback paths */
    GLuint out[65];
    glGetBufferSubData(GL_TEXTURE_BUFFER, 0, N * sizeof(GLuint), out);
    for (i = 0; i < N; i++)
        EXPECT_EQ_INT(out[i], expected);

    /* and again through a persistent-free map */
    GLuint *map = (GLuint *)glMapBufferRange(GL_TEXTURE_BUFFER, 0, N * sizeof(GLuint), GL_MAP_READ_BIT);
    if (map) {
        for (i = 0; i < N; i++)
            EXPECT_EQ_INT(map[i], expected);
        glUnmapBuffer(GL_TEXTURE_BUFFER);
    } else {
        EXPECT_TRUE(map != NULL);
    }
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(prog); glDeleteShader(sh); glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &tbo);
    EXPECT_NO_GL_ERROR();
}

/* imageStore alone on a uimageBuffer (the write must reach the CPU bytes) */
TEST(tbuf_store) {
    const char *cs =
        "#version 450 core\n"
        "layout(local_size_x = 8) in;\n"
        "layout(r32ui, binding = 0) coherent uniform highp uimageBuffer ubuf;\n"
        "void main() { imageStore(ubuf, int(gl_GlobalInvocationID.x), uvec4(gl_GlobalInvocationID.x + 100u)); }\n";
    GLuint prog = glCreateProgram(), sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &cs, NULL); glCompileShader(sh);
    char log[2048]; GLint ok = 0;
    glAttachShader(prog, sh); glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glGetProgramInfoLog(prog, sizeof log, NULL, log); fprintf(stderr, "link:\n%s\n", log); return; }
    GLuint tbo, tex; glGenBuffers(1, &tbo);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo);
    GLuint init[64], i;
    for (i = 0; i < 64; i++) init[i] = i;
    glBufferData(GL_TEXTURE_BUFFER, 64 * sizeof(GLuint), init, GL_DYNAMIC_COPY);
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_BUFFER, tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, tbo);
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glUseProgram(prog); glDispatchCompute(8, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    GLuint out[64]; glGetBufferSubData(GL_TEXTURE_BUFFER, 0, sizeof out, out);
    for (i = 0; i < 64; i++) EXPECT_EQ_INT(out[i], i + 100);
    glDeleteProgram(prog); glDeleteShader(sh); glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &tbo);
    EXPECT_NO_GL_ERROR();
}

int main(void) {
    RUN(tbuf_atomic_add_compwap);
    RUN(tbuf_store);
    return mgl_test_summary();
}
