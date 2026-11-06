/* test_compat_ssbo_barrier — dEQP ES31 compute.basic.ssbo_cmd_barrier_{single,multiple}
 * repro: dispatch0 writes values[i] = base+i into SSBO binding 1, glMemoryBarrier(
 * SHADER_STORAGE), dispatch1 reads values[i] and atomicAdd's them into a coherent
 * uint sum in SSBO binding 0.  Both dispatches must see each other's writes; the
 * sum must equal sum(base+i).
 */
#include "mgl_test.h"

static const char *kCs0 =
    "#version 450\n"
    "layout (local_size_x = 1) in;\n"
    "uniform uint u_baseVal;\n"
    "layout(binding = 1) buffer Output {\n"
    "    uint values[];\n"
    "};\n"
    "void main (void) {\n"
    "    uint offset = gl_NumWorkGroups.x*gl_NumWorkGroups.y*gl_WorkGroupID.z + "
    "gl_NumWorkGroups.x*gl_WorkGroupID.y + gl_WorkGroupID.x;\n"
    "    values[offset] = u_baseVal+offset;\n"
    "}\n";

static const char *kCs1 =
    "#version 450\n"
    "layout (local_size_x = 1) in;\n"
    "uniform uint u_baseVal;\n"
    "layout(binding = 1) buffer Input {\n"
    "    uint values[];\n"
    "};\n"
    "layout(binding = 0) buffer Output {\n"
    "    coherent uint sum;\n"
    "};\n"
    "void main (void) {\n"
    "    uint offset = gl_NumWorkGroups.x*gl_NumWorkGroups.y*gl_WorkGroupID.z + "
    "gl_NumWorkGroups.x*gl_WorkGroupID.y + gl_WorkGroupID.x;\n"
    "    uint value  = values[offset];\n"
    "    atomicAdd(sum, value);\n"
    "}\n";

static GLuint mkcs(const char *src)
{
    GLuint p = glCreateProgram(), s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    return p;
}

static int run_case(int wx, int wy, int wz, const char *tag)
{
    GLuint p0 = mkcs(kCs0), p1 = mkcs(kCs1), temp = 0, out = 0;
    int fails = 0;
    GLuint base = 127;
    int n = wx * wy * wz;

    glGenBuffers(1, &temp);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, temp);
    GLuint zeros[11 * 5 * 7] = {0};
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * 4, NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, temp);

    glGenBuffers(1, &out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4, zeros, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, out);

    glUseProgram(p0);
    glUniform1ui(glGetUniformLocation(p0, "u_baseVal"), base);
    glDispatchCompute(wx, wy, wz);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (getenv("SSBO_DEBUG"))
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, temp);
        GLuint *m0 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, n * 4, GL_MAP_READ_BIT);
        if (m0)
        {
            fprintf(stderr, "DBG after dispatch0: v[0]=%u v[%d]=%u", m0[0], n - 1, m0[n - 1]);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, out);
        GLuint *m1 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 4, GL_MAP_READ_BIT);
        if (m1) { fprintf(stderr, " sum=%u", m1[0]); glUnmapBuffer(GL_SHADER_STORAGE_BUFFER); }
        fprintf(stderr, "\n");
    }
    glUseProgram(p1);
    glDispatchCompute(wx, wy, wz);

    GLuint *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 4, GL_MAP_READ_BIT);
    EXPECT_TRUE(map != NULL);
    GLuint ref = 0;
    for (int i = 0; i < n; i++) ref += base + (GLuint)i;
    if (map && map[0] != ref)
    {
        MGL_FAIL("%s: sum == %u, expected %u", tag, map[0], ref);
        fails++;
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glDeleteProgram(p0);
    glDeleteProgram(p1);
    glDeleteBuffers(1, &temp);
    glDeleteBuffers(1, &out);
    while (glGetError() != GL_NO_ERROR) {}
    return fails;
}

TEST(compat_ssbo_barrier_single)   { EXPECT_EQ_INT(run_case(1, 1, 1, "single"), 0); }
TEST(compat_ssbo_barrier_multiple) { EXPECT_EQ_INT(run_case(11, 5, 7, "multiple"), 0); }

int main(void)
{
    RUN(compat_ssbo_barrier_single);
    RUN(compat_ssbo_barrier_multiple);
    return mgl_test_summary();
}
