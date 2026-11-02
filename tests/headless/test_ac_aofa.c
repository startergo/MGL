/*
 * test_ac_aofa.c — atomic counters in ARRAYS OF ARRAYS (GL 4.3 / ARB_arrays_of_arrays:
 * `layout(binding=0) uniform atomic_uint ac[2][3];`).
 *
 * geometry.c's atomic-counter MSL argument injection used to rewrite only ONE
 * subscript group: `&ac[i][j]` became `&ac[0+(i)][j]` — a `device atomic_uint*`
 * has no second subscript, so the whole stage failed to compile
 * (dEQP-GL45-ES31 PIQ uniform.atomic_counter_buffer_index.default_block.array.
 * var_array_array).  This file exercises the flattened rewrite in the compute
 * stage, in a vertex+fragment pair, through a helper function (the counter is
 * touched inside a non-entry function), and with a nonzero `offset` qualifier.
 */
#include "mgl_test.h"

TEST(compute_aofa_increment)
{
    const char *cs =
        "#version 450 core\n"
        "layout(local_size_x=6) in;\n"
        "layout(binding=0) uniform atomic_uint ac[2][3];\n"
        "layout(std430, binding=1) writeonly buffer Out { uint o[6]; };\n"
        "void main() {\n"
        "    uint ndx = gl_GlobalInvocationID.x;\n"
        "    uint r = ndx / 3u, c = ndx % 3u;\n"
        "    o[ndx] = atomicCounterIncrement(ac[r][c]);   /* returns the old value */\n"
        "}\n";
    GLuint p = mgl_make_program_cs(cs);
    EXPECT_NE(p, 0);
    GLuint buf = 0, out = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buf);
    uint32_t zeros[8] = {0};
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof zeros, zeros, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, buf);
    glGenBuffers(1, &out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 6 * 4, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out);

    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t *m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 6 * 4, GL_MAP_READ_BIT);
    EXPECT_TRUE(m != NULL);
    /* each element incremented exactly once: the flattened [r][c] is a bijection */
    for (int i = 0; i < 6; i++)
        EXPECT_EQ_INT(m[i], 0u);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    /* every element of ac[2][3] incremented exactly once by the six invocations */
    uint32_t *cm = glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof zeros, GL_MAP_READ_BIT);
    EXPECT_TRUE(cm != NULL);
    for (int i = 0; i < 6; i++)
        EXPECT_EQ_INT(cm[i], 1u);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    glDeleteProgram(p);
    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &out);
    EXPECT_NO_GL_ERROR();
}

TEST(compute_aofa_offset_and_helper)
{
    const char *cs =
        "#version 450 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(binding=1, offset=8) uniform atomic_uint ac[2][2];\n"
        "layout(std430, binding=1) writeonly buffer Out { uint o[4]; };\n"
        "uint bump(uint i, uint j) { return atomicCounterIncrement(ac[i][j]); }\n"
        "void main() {\n"
        "    for (uint i = 0u; i < 2u; i++)\n"
        "        for (uint j = 0u; j < 2u; j++)\n"
        "            o[i * 2u + j] = bump(i, j) + 1u;\n"
        "}\n";
    GLuint p = mgl_make_program_cs(cs);
    EXPECT_NE(p, 0);
    GLuint buf = 0, out = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buf);
    uint32_t zeros[8] = {0};
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof zeros, zeros, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, buf);   /* binding=1 per the qualifier */
    glGenBuffers(1, &out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out);

    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t *m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 16, GL_MAP_READ_BIT);
    EXPECT_TRUE(m != NULL);
    for (int i = 0; i < 4; i++)
        EXPECT_EQ_INT(m[i], 1u);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    /* element (i,j) is word (2+i*2+j): offset 8 = word 2 */
    uint32_t *cm = glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof zeros, GL_MAP_READ_BIT);
    EXPECT_TRUE(cm != NULL);
    EXPECT_EQ_INT(cm[2], 1u);   /* ac[0][0] */
    EXPECT_EQ_INT(cm[3], 1u);   /* ac[0][1] */
    EXPECT_EQ_INT(cm[4], 1u);   /* ac[1][0] */
    EXPECT_EQ_INT(cm[5], 1u);   /* ac[1][1] */
    EXPECT_EQ_INT(cm[0], 0u);
    EXPECT_EQ_INT(cm[7], 0u);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    glDeleteProgram(p);
    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &out);
    EXPECT_NO_GL_ERROR();
}

TEST(render_stage_aofa)
{
    /* a vertex-stage array-of-arrays counter: read back through a SSBO */
    const char *vs =
        "#version 450 core\n"
        "layout(binding=0) uniform atomic_uint ac[2][2];\n"
        "layout(std430, binding=0) buffer Out { uint o[4]; };\n"
        "void main() {\n"
        "    gl_Position = vec4(0.0);\n"
        "    o[gl_VertexID * 4u + 0u] = atomicCounter(ac[0][0]);\n"
        "    o[gl_VertexID * 4u + 1u] = atomicCounter(ac[0][1]);\n"
        "    o[gl_VertexID * 4u + 2u] = atomicCounter(ac[1][0]);\n"
        "    o[gl_VertexID * 4u + 3u] = atomicCounter(ac[1][1]);\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "out vec4 c;\n"
        "void main() { c = vec4(1.0); }\n";
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    EXPECT_NE(p, 0);
    GLuint buf = 0, out = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buf);
    uint32_t init[4] = {3u, 5u, 7u, 11u};
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof init, init, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, buf);
    glGenBuffers(1, &out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, out);

    glUseProgram(p);
    /* a VAO with one enabled attribute (gl_VertexID-only draws are fine in GL, but
     * this mirrors the shape every other vertex-stage AC test uses) */
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float one[1] = {1.0f};
    glBufferData(GL_ARRAY_BUFFER, sizeof one, one, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, (void *)0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    glFinish();
    uint32_t *m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 16, GL_MAP_READ_BIT);
    EXPECT_TRUE(m != NULL);
    EXPECT_EQ_INT(m[0], 3u);
    EXPECT_EQ_INT(m[1], 5u);
    EXPECT_EQ_INT(m[2], 7u);
    EXPECT_EQ_INT(m[3], 11u);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glDeleteProgram(p);
    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &out);
    EXPECT_NO_GL_ERROR();
}

TEST(geometry_stage_aofa)
{
    /* the geometry emulation kernel collects its own atomic-counter set from the
     * GS module (geom_module_ac_vars): array-of-arrays must flatten there too */
    const char *vs =
        "#version 450 core\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "layout(binding=0) uniform atomic_uint ac[2][2];\n"
        "layout(std430, binding=0) buffer Out { uint o[4]; };\n"
        "void main() {\n"
        "    for (uint i = 0u; i < 2u; i++)\n"
        "        for (uint j = 0u; j < 2u; j++)\n"
        "            o[i * 2u + j] = atomicCounterIncrement(ac[i][j]);\n"
        "    EmitVertex();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "out vec4 c;\n"
        "void main() { c = vec4(1.0); }\n";
    GLuint p = mgl_make_program3(vs, gs, fs);
    EXPECT_NE(p, 0);
    GLuint buf = 0, out = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buf);
    uint32_t zeros[4] = {0};
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof zeros, zeros, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, buf);
    glGenBuffers(1, &out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, out);

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    glFinish();

    uint32_t *m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 16, GL_MAP_READ_BIT);
    EXPECT_TRUE(m != NULL);
    for (int i = 0; i < 4; i++)
        EXPECT_EQ_INT(m[i], 0u);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    uint32_t *cm = glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof zeros, GL_MAP_READ_BIT);
    EXPECT_TRUE(cm != NULL);
    for (int i = 0; i < 4; i++)
        EXPECT_EQ_INT(cm[i], 1u);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &out);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(compute_aofa_increment);
    RUN(compute_aofa_offset_and_helper);
    RUN(render_stage_aofa);
    RUN(geometry_stage_aofa);
    return mgl_test_summary();
}
