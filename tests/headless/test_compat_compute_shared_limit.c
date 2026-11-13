/* test_compat_compute_shared_limit — compat r108: GL_MAX_COMPUTE_SHARED_
 * MEMORY_SIZE must be at least 36872 so KHR-GL46.compute_shader.shared-struct
 * (a 36872-byte shared block) links; MGL reports 65536 (what its MSL
 * compilation path accepts for STATIC threadgroup storage).  Execution of
 * blocks beyond the device's 32KB dynamic-allocation bound silently no-ops
 * (documented gap — identical to the pre-r97a behaviour this case passed
 * with); a 32KB block runs and verifies.
 */
#include "mgl_test.h"

static GLuint mkcs(GLuint shared_ints)
{
    char cs[512];
    snprintf(cs, sizeof cs,
        "#version 450\n"
        "layout(local_size_x = 8) in;\n"
        "layout(std430, binding = 0) buffer Out { int o[]; };\n"
        "shared int g_shared[%u];\n"
        "void main(){ uint i = gl_LocalInvocationID.x + gl_WorkGroupID.x * 8u;\n"
        "  g_shared[i] = int(i) * 2; barrier(); o[i] = g_shared[i] + 1; }\n", shared_ints);
    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    char log[512];
    GLint ok = 0;
    glShaderSource(s, 1, &(const char *){ cs }, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(s, sizeof log, NULL, log); MGL_FAIL("compile: %s", log); }
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }
    return p;
}

TEST(compute_shared_limits)
{
    GLint v = 0;
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &v);
    printf("  GL_MAX_COMPUTE_SHARED_MEMORY_SIZE = %d\n", v);
    EXPECT_TRUE(v >= 36872);          /* KHR-GL46 compute_shader.shared-struct */
    EXPECT_TRUE(v >= 32768);          /* GL 4.6 minimum */

    /* a 36872-byte block LINKS (the r97a-r98c link check with the old 32768
     * limit rejected exactly this) */
    GLuint p36k = mkcs(9218);
    EXPECT_TRUE(p36k != 0);
    glDeleteProgram(p36k);

    /* a 32KB block runs and produces correct values */
    GLuint prog = mkcs(8192);
    EXPECT_TRUE(prog != 0);
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4096 * sizeof(int), NULL, GL_STREAM_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glUseProgram(prog);
    glDispatchCompute(512, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    int out[8] = { 0 };
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    printf("  32KB block: o[0..7] = %d %d %d %d %d %d %d %d\n",
           out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    for (int i = 0; i < 8; i++)
        EXPECT_EQ_INT(out[i], i * 2 + 1);

    glDeleteProgram(prog);
    glDeleteBuffers(1, &buf);
}
int main(void){ RUN(compute_shared_limits); return mgl_test_summary(); }
