#include "mgl_test.h"

/* GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS semantics (r98 fix).
 *
 * The combined limit counts the per-stage block totals SUMMED over the active
 * stages: a block array declared with the same name in every stage counts once
 * per stage (Mesa st_extensions.c does the same for its default, and dEQP's
 * es31fProgramInterfaceDefinitionUtil::getCombinedProgramResourceUsage counts
 * it identically).  MGL reports combined = 5 * per-stage (the five graphics
 * stages; a compute-only program cannot exceed its own per-stage limit), the
 * same relation the uniform blocks have (per-stage 14 / combined 70).
 *
 * Before r98 the combined limit was reported as the binding-point count (8),
 * which made the r97b link check reject every multi-stage block-array program
 * with 9..15 combined instances — dEQP-GL45-ES31
 * program_interface_query.*.referenced_by.*.block_array went NotSupported.
 */

TEST(combined_ssbo_limit_values)
{
    GLint mv = 0, mtcs = 0, mtes = 0, mgs = 0, mf = 0, mc = 0, mosr = 0, mcs = 0;
    glGetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, &mv);
    glGetIntegerv(GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS, &mtcs);
    glGetIntegerv(GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, &mtes);
    glGetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, &mgs);
    glGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &mf);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &mcs);
    glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &mc);
    glGetIntegerv(GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES, &mosr);
    printf("  per-stage ssbo: vs=%d tcs=%d tes=%d gs=%d fs=%d cs=%d combined=%d outres=%d\n",
           mv, mtcs, mtes, mgs, mf, mcs, mc, mosr);
    fflush(stdout);

    /* every graphics stage reports the same per-stage limit (8 binding points) */
    EXPECT_EQ_INT(mv, mtcs);
    EXPECT_EQ_INT(mv, mtes);
    EXPECT_EQ_INT(mv, mgs);
    EXPECT_EQ_INT(mv, mf);
    /* combined = the sum over the graphics stages (>= the GL 4.6 floor of 8,
     * >= any single stage, and >= the compute-only program's maximum) */
    EXPECT_EQ_INT(mc, 5 * mv);
    EXPECT_TRUE(mc >= 8);
    EXPECT_TRUE(mc >= mcs);
    /* MAX_COMBINED_SHADER_OUTPUT_RESOURCES = combined images + combined ssbo
     * blocks + max draw buffers (never smaller than the ssbo term) */
    EXPECT_TRUE(mosr >= mc);
    EXPECT_NO_GL_ERROR();
}

/* the regression: one 3-instance block array declared in VS+TCS+TES+FS is 12
 * combined blocks but only 3 binding points — it must link, and the program's
 * GL_SHADER_STORAGE_BLOCK interface must expose the 3 deduped instances with
 * every stage marked as referencing them */
TEST(block_array_across_stages_links)
{
    const char *vs =
        "#version 450 core\n"
        "layout(std430, binding = 0) buffer TargetInterface { float target; } targetInstance[3];\n"
        "void main() {\n"
        "    vec4 r = vec4(targetInstance[0].target + targetInstance[1].target + targetInstance[2].target);\n"
        "    targetInstance[0].target = r.y;\n"
        "    gl_Position = r;\n"
        "}\n";
    const char *tcs =
        "#version 450 core\n"
        "layout(std430, binding = 0) buffer TargetInterface { float target; } targetInstance[3];\n"
        "layout(vertices = 3) out;\n"
        "void main() {\n"
        "    vec4 r = gl_in[gl_InvocationID].gl_Position + vec4(targetInstance[1].target);\n"
        "    gl_out[gl_InvocationID].gl_Position = r;\n"
        "}\n";
    const char *tes =
        "#version 450 core\n"
        "layout(std430, binding = 0) buffer TargetInterface { float target; } targetInstance[3];\n"
        "layout(triangles, equal_spacing) in;\n"
        "void main() {\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + vec4(targetInstance[2].target);\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(std430, binding = 0) buffer TargetInterface { float target; } targetInstance[3];\n"
        "layout(location = 0) out vec4 o;\n"
        "void main() {\n"
        "    targetInstance[2].target = 1.0;\n"
        "    o = vec4(targetInstance[0].target);\n"
        "}\n";
    GLuint p = mgl_make_program5(vs, tcs, tes, NULL, fs);
    EXPECT_NE(p, 0);
    if (!p) return;

    GLint n = -1;
    glGetProgramInterfaceiv(p, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &n);
    printf("  active shader storage blocks = %d\n", n);
    EXPECT_EQ_INT(n, 3);

    GLuint i0 = glGetProgramResourceIndex(p, GL_SHADER_STORAGE_BLOCK, "TargetInterface[0]");
    GLuint i1 = glGetProgramResourceIndex(p, GL_SHADER_STORAGE_BLOCK, "TargetInterface[1]");
    GLuint i2 = glGetProgramResourceIndex(p, GL_SHADER_STORAGE_BLOCK, "TargetInterface[2]");
    if (i0 == GL_INVALID_INDEX && n > 0)
    {
        /* dump the actual names so the reported block naming stays visible */
        for (int i = 0; i < n && i < 4; i++)
        {
            char nm[128] = "";
            GLsizei l = 0;
            glGetProgramResourceName(p, GL_SHADER_STORAGE_BLOCK, (GLuint)i, sizeof nm, &l, nm);
            printf("  ssbo[%d] name=\"%s\"\n", i, nm);
        }
    }
    EXPECT_NE(i0, GL_INVALID_INDEX);
    EXPECT_NE(i1, GL_INVALID_INDEX);
    EXPECT_NE(i2, GL_INVALID_INDEX);
    /* per-element usage in this program: VS touches [0][1][2], TCS reads [1],
     * TES reads [2], FS reads [0] and writes [2]; no GS stage, no CS stage */
    const int want[3][7] = {
        { 1, 0, 0, 0, 1, 0, 1 },   /* [0]: vs, fs */
        { 1, 1, 0, 0, 0, 0, 1 },   /* [1]: vs, tcs */
        { 1, 0, 1, 0, 1, 0, 1 },   /* [2]: vs, tes, fs */
    };
    const GLuint idx[3] = { i0, i1, i2 };
    for (int e = 0; e < 3; e++)
    {
        if (idx[e] == GL_INVALID_INDEX) continue;
        const GLenum refd[] = {
            GL_REFERENCED_BY_VERTEX_SHADER, GL_REFERENCED_BY_TESS_CONTROL_SHADER,
            GL_REFERENCED_BY_TESS_EVALUATION_SHADER, GL_REFERENCED_BY_GEOMETRY_SHADER,
            GL_REFERENCED_BY_FRAGMENT_SHADER, GL_REFERENCED_BY_COMPUTE_SHADER,
            GL_NUM_ACTIVE_VARIABLES,
        };
        GLint vals[7] = { 0, 0, 0, 0, 0, 0, 0 };
        GLsizei len = 0;
        glGetProgramResourceiv(p, GL_SHADER_STORAGE_BLOCK, idx[e], 7, refd, 7, &len, vals);
        printf("  TargetInterface[%d] refd by vs=%d tcs=%d tes=%d gs=%d fs=%d cs=%d nvars=%d\n",
               e, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]);
        for (int k = 0; k < 7; k++)
            EXPECT_EQ_INT(vals[k], want[e][k]);
    }
    /* the block member is a buffer variable of every stage too; a block-array
     * element's member keeps the "<block>.<member>" spelling (one resource per
     * array element — matches dEQP's program_interface_query model) */
    {
        GLint nbv = -1;
        glGetProgramInterfaceiv(p, GL_BUFFER_VARIABLE, GL_ACTIVE_RESOURCES, &nbv);
        printf("  active buffer variables = %d\n", nbv);
        EXPECT_EQ_INT(nbv, 3);
        char nm[128] = "";
        GLsizei l = 0;
        glGetProgramResourceName(p, GL_BUFFER_VARIABLE, 0, sizeof nm, &l, nm);
        printf("  bv[0] name=\"%s\"\n", nm);
        EXPECT_STREQ(nm, "TargetInterface.target");
    }
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(p);
}

/* per-stage over-limit still fails the link (r97b check): one more instance
 * than GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS in the vertex stage only */
TEST(per_stage_ssbo_over_limit_fails)
{
    GLint mv = 0;
    glGetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, &mv);
    printf("  max vertex shader storage blocks = %d\n", mv);
    if (mv <= 0 || mv > 32) { printf("  (per-stage limit out of test range)\n"); return; }

    char decls[4096] = "", uses[4096] = "";
    int off = 0;
    /* mv+1 single-instance blocks; bindings are cycled mod mv so every declared
     * binding stays inside GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS (the per-stage
     * BLOCK count is the limit under test, not the binding range — a 9-element
     * block array would trip the compile-time binding-range check first) */
    for (int i = 0; i <= mv; i++)
        off += snprintf(decls + off, sizeof decls - off,
                        "layout(std430, binding = %d) buffer B%d { vec4 b; } v%d;\n",
                        i % mv, i, i);
    off = 0;
    for (int i = 0; i <= mv; i++)
        off += snprintf(uses + off, sizeof uses - off, "r += v%d.b;", i);
    char vs[16384];
    snprintf(vs, sizeof vs,
             "#version 450 core\n%s\nlayout(location = 0) in vec4 p;\n"
             "void main() { vec4 r = vec4(0); %s gl_Position = p + r; }\n", decls, uses);
    const char *fs =
        "#version 450 core\nlayout(location = 0) out vec4 o;\n"
        "void main() { o = vec4(1); }\n";
    const char *vsp = vs;
    GLuint prog = glCreateProgram();
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s1, 1, &vsp, NULL); glCompileShader(s1);
    glShaderSource(s2, 1, &fs, NULL); glCompileShader(s2);
    GLint c1 = 0, c2 = 0;
    glGetShaderiv(s1, GL_COMPILE_STATUS, &c1);
    glGetShaderiv(s2, GL_COMPILE_STATUS, &c2);
    printf("  %d instances in vs: compile vs=%d fs=%d\n", mv + 1, c1, c2);
    if (!c1) { char slog[1024] = ""; glGetShaderInfoLog(s1, sizeof slog, NULL, slog); printf("  vs log: %s", slog); }
    EXPECT_EQ_INT(c1, 1);
    EXPECT_EQ_INT(c2, 1);
    glAttachShader(prog, s1); glAttachShader(prog, s2);
    glLinkProgram(prog);
    GLint ls = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ls);
    char log[512] = "";
    glGetProgramInfoLog(prog, sizeof log, NULL, log);
    printf("  link=%d (want 0) log: %s", ls, log);
    EXPECT_EQ_INT(ls, 0);
    EXPECT_TRUE(strstr(log, "stage limit") != NULL);
    glDeleteShader(s1); glDeleteShader(s2); glDeleteProgram(prog);
}

/* the combined check stays armed for implementations whose combined limit is
 * lower than the sum of the per-stage limits; on MGL it is exactly the sum
 * (40 = 5 x 8), so this case reports itself skippable — same as dEQP's
 * max_combined_block_number_limit does on us now */
TEST(combined_ssbo_over_limit_fails_when_reachable)
{
    GLint mv = 0, mtcs = 0, mtes = 0, mgs = 0, mf = 0, mc = 0;
    glGetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, &mv);
    glGetIntegerv(GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS, &mtcs);
    glGetIntegerv(GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, &mtes);
    glGetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, &mgs);
    glGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &mf);
    glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &mc);
    if (mv + mtcs + mtes + mgs + mf <= mc)
    {
        printf("  (combined limit %d >= per-stage sum %d — over-limit unreachable, skipped)\n",
               mc, mv + mtcs + mtes + mgs + mf);
        return;
    }
    /* build per-stage max instances in every graphics stage: must fail to link */
    char vsd[8192] = "", vsu[8192] = "", fsd[8192] = "", fsu[8192] = "";
    int off = 0;
    for (int i = 0; i < mv; i++)
        off += snprintf(vsd + off, sizeof vsd - off,
                        "layout(std430, binding = 0) buffer B%d { vec4 b; } v%d;\n", i, i);
    off = 0;
    for (int i = 0; i < mv; i++)
        off += snprintf(vsu + off, sizeof vsu - off, "r += v%d.b.x;", i);
    off = 0;
    for (int i = 0; i < mf; i++)
        off += snprintf(fsd + off, sizeof fsd - off,
                        "layout(std430, binding = 0) buffer C%d { vec4 c; } f%d;\n", i, i);
    off = 0;
    for (int i = 0; i < mf; i++)
        off += snprintf(fsu + off, sizeof fsu - off, "o += f%d.c.x;", i);
    char vs[16384], fs[16384];
    snprintf(vs, sizeof vs, "#version 450 core\n%s\nlayout(location=0) in vec4 p;\nvoid main() { vec4 r = vec4(0); %s gl_Position = p + r; }\n", vsd, vsu);
    snprintf(fs, sizeof fs, "#version 450 core\n%s\nlayout(location=0) out vec4 o;\nvoid main() { vec4 r = vec4(0); %s o = r; }\n", fsd, fsu);
    const char *vsp = vs, *fsp = fs;
    GLuint prog = glCreateProgram();
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s1, 1, &vsp, NULL); glCompileShader(s1);
    glShaderSource(s2, 1, &fsp, NULL); glCompileShader(s2);
    glAttachShader(prog, s1); glAttachShader(prog, s2);
    glLinkProgram(prog);
    GLint ls = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ls);
    char log[512] = "";
    glGetProgramInfoLog(prog, sizeof log, NULL, log);
    printf("  link=%d (want 0) log: %s", ls, log);
    EXPECT_EQ_INT(ls, 0);
    EXPECT_TRUE(strstr(log, "COMBINED_SHADER_STORAGE_BLOCKS") != NULL);
    glDeleteShader(s1); glDeleteShader(s2); glDeleteProgram(prog);
}

int main(void)
{
    RUN(combined_ssbo_limit_values);
    RUN(block_array_across_stages_links);
    RUN(per_stage_ssbo_over_limit_fails);
    RUN(combined_ssbo_over_limit_fails_when_reachable);
    return mgl_test_summary();
}
