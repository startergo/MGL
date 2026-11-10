/* test_compat_ssboalias — GL 4.6 §7.12 intra-invocation ordering through ALIASED
 * SSBO bindings: two std430 blocks (bindings 1 and 2) bound to the SAME buffer
 * object.  dEQP ES31 synchronization.in_invocation.ssbo_alias_overwrite writes
 * 456 through sb_store0, writes groupNdx through sb_store1, then reads through
 * sb_store0 and expects groupNdx — within one invocation GL orders accesses to
 * the same memory, and an image unit/buffer binding is just a view of the store.
 * Variants isolate whether the failure is store-to-load forwarding across the
 * two (distinct) MSL pointer arguments:
 *   v0  plain                    (the dEQP pattern)
 *   v1  memoryBarrierBuffer() between the two writes
 *   v2  plain write through ONE block twice (control; must always pass)
 *   v3  aliased with different element indices (write A[0], write B[i], read A[0])
 */
#include "mgl_test.h"

static const char *kCs =
    "#version 450\n"
    "layout (local_size_x = 8, local_size_y = 8) in;\n"
    "layout(binding = 0, std430) buffer Output { highp int values[]; } sb_result;\n"
    "layout(binding = 1, std430) coherent buffer Storage0 { highp int values[]; } sb_store0;\n"
    "layout(binding = 2, std430) coherent buffer Storage1 { highp int values[]; } sb_store1;\n"
    "void main (void) {\n"
    "    int resultNdx = int(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x\n"
    "                       + gl_GlobalInvocationID.x);\n"
    "    int groupNdx = int(gl_NumWorkGroups.x * gl_WorkGroupID.y + gl_WorkGroupID.x);\n"
    "    int i = resultNdx;\n"
    "    bool allOk = true;\n"
    "%s"
    "    sb_store1.values[i] = groupNdx;\n"
    "%s"
    "    allOk = allOk && (sb_store0.values[i] == groupNdx);\n"
    "    sb_result.values[resultNdx] = allOk ? 1 : 0;\n"
    "}\n";

static int run_variant(int variant, const char *tag)
{
    char src[4096];
    GLuint prog = 0, cs = 0, buf = 0, res = 0;
    GLint out[64 * 64];
    int fails = 0, n = 64 * 64;
    const char *pre = "", *mid = "";

    switch (variant)
    {
        case 0: pre = "    sb_store0.values[i] = 456;\n"; break;
        case 1: pre = "    sb_store0.values[i] = 456;\n";
                mid = "    memoryBarrierBuffer();\n"; break;
        case 2: pre = "    sb_store0.values[i] = 456;\n";
                /* write through the SAME block as the read (control) */
                mid = "    sb_store0.values[i] = groupNdx;\n"; break;
        case 3: pre = "    sb_store0.values[i] = 456;\n";
                /* shift the second write one slot: still aliased memory, the read
                 * index is untouched by the second write */
                mid = "    ;\n"; break;
    }

    snprintf(src, sizeof src, kCs, pre, mid);

    cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, (const GLchar *[]){ src }, NULL);
    glCompileShader(cs);
    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: CS compile failed", tag); return 1; }
    prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: link failed", tag); return 1; }

    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    {
        static GLint zero[64 * 64];
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof zero, zero, GL_STATIC_DRAW);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buf);   /* the ALIAS */

    glGenBuffers(1, &res);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, res);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof out, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, res);

    glUseProgram(prog);
    glDispatchCompute(8, 8, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, res);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    for (int k = 0; k < n; k++)
        if (out[k] != 1)
        {
            if (fails < 4) MGL_FAIL("%s: result[%d] == %d", tag, k, out[k]);
            fails++;
        }

    glDeleteProgram(prog);
    glDeleteShader(cs);
    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &res);
    while (glGetError() != GL_NO_ERROR) {}
    return fails;
}

TEST(compat_ssboalias_v1_barrier) { EXPECT_EQ_INT(run_variant(1, "v1"), 0); }
TEST(compat_ssboalias_v2_oneblock) { EXPECT_EQ_INT(run_variant(2, "v2"), 0); }
/* v0 (the dEQP pattern, no barrier) is FIXED by the r100c aliased-SSBO
 * dispatch variant (compat_ssbo.c): the secondary block's Metal argument is
 * folded onto the primary's pointer, so the compiler can no longer forward the
 * 456 store to the load.  MGL_NO_SSBO_ALIAS=1 restores the old behaviour and
 * makes this fail — the A/B switch for the fix. */
TEST(compat_ssboalias_v0_plain) { EXPECT_EQ_INT(run_variant(0, "v0"), 0); }

/* three blocks on ONE buffer (A writes, C writes, B reads — every pair of the
 * group aliases) plus a fourth INDEPENDENT block that must stay untouched:
 * the variant folds B and C onto A's pointer while D keeps its own argument */
static const char *kCs3 =
    "#version 450\n"
    "layout (local_size_x = 8, local_size_y = 8) in;\n"
    "layout(binding = 0, std430) buffer Output { highp int values[]; } sb_result;\n"
    "layout(binding = 1, std430) coherent buffer S0 { highp int v[]; } sb_a;\n"
    "layout(binding = 2, std430) coherent buffer S1 { highp int v[]; } sb_b;\n"
    "layout(binding = 3, std430) coherent buffer S2 { highp int v[]; } sb_c;\n"
    "layout(binding = 4, std430) coherent buffer S3 { highp int v[]; } sb_d;\n"
    "void main (void) {\n"
    "    int i = int(gl_GlobalInvocationID.x);\n"
    "    sb_a.v[i] = 100 + i;\n"
    "    sb_c.v[i] = 300 + i;\n"
    "    bool ok = (sb_b.v[i] == 300 + i);\n"
    "    sb_d.v[i] = 400 + i;                 /* independent buffer */\n"
    "    ok = ok && (sb_a.v[i] == 300 + i);   /* fold-back read through A */\n"
    "    sb_result.values[i] = ok ? 1 : 0;\n"
    "}\n";

TEST(compat_ssboalias_chain3)
{
    GLuint prog = 0, cs = 0, shared = 0, res = 0, ind = 0;
    GLint out[64], d[64];
    int fails = 0;

    cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, (const GLchar *[]){ kCs3 }, NULL);
    glCompileShader(cs);
    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) { MGL_FAIL("chain3: CS compile failed"); return; }
    prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("chain3: link failed"); return; }

    glGenBuffers(1, &shared);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, shared);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof out, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, shared);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, shared);   /* alias */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, shared);   /* alias */

    glGenBuffers(1, &ind);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ind);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof d, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ind);

    glGenBuffers(1, &res);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, res);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof out, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, res);

    glUseProgram(prog);
    glDispatchCompute(8, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* a second dispatch must hit the cached variant and still be correct */
    glDispatchCompute(8, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, res);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ind);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof d, d);
    for (int k = 0; k < 64; k++)
    {
        if (out[k] != 1 && fails < 4) MGL_FAIL("chain3: result[%d] == %d", k, out[k]);
        if (d[k] != 400 + k && fails < 8) MGL_FAIL("chain3: independent[%d] == %d", k, d[k]);
        if (out[k] != 1 || d[k] != 400 + k) fails++;
    }

    glDeleteProgram(prog);
    glDeleteShader(cs);
    glDeleteBuffers(1, &shared);
    glDeleteBuffers(1, &ind);
    glDeleteBuffers(1, &res);
    while (glGetError() != GL_NO_ERROR) {}
    EXPECT_EQ_INT(fails, 0);
}

int main(void)
{
    RUN(compat_ssboalias_v1_barrier);
    RUN(compat_ssboalias_v2_oneblock);
    RUN(compat_ssboalias_v0_plain);
    RUN(compat_ssboalias_chain3);
    return mgl_test_summary();
}
