/* test_geom_nonames.c — geom r146: ARB_gl_spirv name independence
 *
 * OpName/OpMemberName are NON-SEMANTIC debug instructions; a module without
 * them is fully valid and must produce identical results.  Before r146 a
 * name-less module SILENTLY MISCOMPILED through the geometry emulation: the
 * generated GLSL pasted variable names into floatBitsToUint(%s) and every
 * expression, and the xfb resolver matched interface-block members by name.
 * spv_parse now synthesizes stable identifiers (_mgl_v<id>, _mgl_in/out<loc>,
 * _mgl_t<id>, m<idx>) for the generators, and transform_feedback reflects a
 * NAMED COPY of the module — while introspection keeps reporting the module's
 * real (empty) names.
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "spv_nonames.h"

/* the GLSL twins of the embedded modules (named) — results must be identical */
static const char *vs_named =
    "#version 450 core\n"
    "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
static const char *gs_named =
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(triangle_strip, max_vertices = 3) out;\n"
    "layout(location=0) out vec4 g_a;\n"
    "layout(location=1) flat out int g_b;\n"
    "layout(location=2) flat out ivec2 g_c;\n"
    "void main() {\n"
    "  g_a = vec4(0.2, 0.4, 0.6, 0.8);\n"
    "  g_b = 7;\n"
    "  g_c = ivec2(3, 5);\n"
    "  gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); EmitVertex();\n"
    "  gl_Position = vec4(3.0, -1.0, 0.0, 1.0); EmitVertex();\n"
    "  gl_Position = vec4(-1.0, 3.0, 0.0, 1.0); EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *fs_named =
    "#version 450 core\n"
    "layout(location=0) in vec4 g_a;\n"
    "layout(location=1) flat in int g_b;\n"
    "layout(location=2) flat in ivec2 g_c;\n"
    "layout(location=0) out vec4 c;\n"
    "void main() { c = vec4(g_a.x + g_b * 0.01 + g_c.x * 0.001, g_a.y, g_a.z, 1.0); }\n";

static GLuint load_spirv_stage(GLenum type, const uint32_t *words, size_t n)
{
    GLuint shader = glCreateShader(type);

    glShaderBinary(1, &shader, GL_SHADER_BINARY_FORMAT_SPIR_V, words, (GLsizei)(n * 4));
    if (glGetError() != GL_NO_ERROR) { glDeleteShader(shader); return 0; }
    glSpecializeShader(shader, "main", 0, NULL, NULL);
    {
        GLint ok = 0;

        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char log[512] = "";

            glGetShaderInfoLog(shader, sizeof log, NULL, log);
            MGL_FAIL("specialize failed: %s", log);
            glDeleteShader(shader);
            return 0;
        }
    }
    return shader;
}

/* render through the NAME-LESS modules and compare against the GLSL twin */
TEST(nonames_render)
{
    GLuint vs = load_spirv_stage(GL_VERTEX_SHADER, spv_mix_vs_nonames,
                                 sizeof spv_mix_vs_nonames / 4);
    GLuint gs = load_spirv_stage(GL_GEOMETRY_SHADER, spv_mix_gs_nonames,
                                 sizeof spv_mix_gs_nonames / 4);
    GLuint fs = load_spirv_stage(GL_FRAGMENT_SHADER, spv_mix_fs_nonames,
                                 sizeof spv_mix_fs_nonames / 4);
    GLuint p, ref;
    GLubyte a[4] = {0}, b[4] = {0};

    if (!vs || !gs || !fs) return;
    p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, gs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(gs);
    glDeleteShader(fs);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
    }

    /* the named reference */
    ref = mgl_make_program5(vs_named, NULL, NULL, gs_named, fs_named);
    if (!ref) { glDeleteProgram(p); return; }

    glViewport(0, 0, 8, 8);
    glUseProgram(p);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, a);

    glUseProgram(ref);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, b);

    /* identical pixels: g_a.x + 7*0.01 + 3*0.001 = 0.273 etc. */
    EXPECT_EQ_INT(a[0], b[0]);
    EXPECT_EQ_INT(a[1], b[1]);
    EXPECT_EQ_INT(a[2], b[2]);
    EXPECT_TRUE(a[0] == 70 && a[1] == 102 && a[2] == 153);   /* 0.2+0.07+0.003, 0.4, 0.6 */
}

/* the xfb fence: the NAME-LESS anonymous-block GS captures byte-identically to its
 * GLSL twin (the resolver's member matching now runs over a NAMED COPY) */
static const char *blk_gs_named =
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "layout(xfb_buffer = 0, xfb_stride = 40) out Blk {\n"
    "   layout(xfb_offset = 0) Inner i;\n"
    "   layout(xfb_offset = 24) float w;\n"
    "};\n"
    "void main() {\n"
    "  gl_Position = gl_in[0].gl_Position;\n"
    "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  i.b = vec2(0.75, 0.5);\n"
    "  w = 0.9;\n"
    "  EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";

static void blk_capture(GLuint p, float *out, size_t bytes, GLint *prims)
{
    GLuint bo = 0, q = 0;

    glUseProgram(p);
    glGenBuffers(1, &bo);
    glGenQueries(1, &q);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, prims);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bytes, out);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
}

TEST(nonames_xfb_block)
{
    const char *vs = "#version 450 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    GLuint ref, svs, sgs, p;
    float a[12], b[12];
    GLint pa = -1, pb = -1;

    /* the named reference (same declared capture), built and captured FIRST —
     * the r146 CPU-pack advances its per-draw state once per draw, so each
     * program's capture is self-consistent regardless of order */
    ref = mgl_make_program5(vs, NULL, NULL, blk_gs_named,
        "#version 450 core\nlayout(location=0) out vec4 c;\nvoid main(){c=vec4(1);}\n");
    if (!ref) return;
    blk_capture(ref, b, sizeof b, &pb);
    /* ref stays alive: program-name recycling hits a stale xfb slot (an xfb-domain
     * table issue, out of scope here) */

    svs = load_spirv_stage(GL_VERTEX_SHADER, spv_blk_vs_nonames,
                           sizeof spv_blk_vs_nonames / 4);
    sgs = load_spirv_stage(GL_GEOMETRY_SHADER, spv_blk_gs_nonames,
                           sizeof spv_blk_gs_nonames / 4);
    if (!svs || !sgs) return;
    p = glCreateProgram();
    glAttachShader(p, svs);
    glAttachShader(p, sgs);
    glLinkProgram(p);
    glDeleteShader(svs);
    glDeleteShader(sgs);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok)
        {
            char log[512] = "";

            glGetProgramInfoLog(p, sizeof log, NULL, log);
            MGL_FAIL("name-less xfb link failed: %s", log);
            /* p stays alive: glDeleteProgram recycles the GL name onto a stale xfb slot (xfb domain) */
            return;
        }
    }
    blk_capture(p, a, sizeof a, &pa);

    EXPECT_EQ_INT(pa, 1);
    EXPECT_EQ_INT(pb, 1);
    for (int i = 0; i < 10; i++)
        EXPECT_NEAR(a[i], b[i], 0.0);   /* byte-identical */
    EXPECT_NEAR(a[0], 1.0, 1e-6);
    EXPECT_NEAR(a[4], 0.75, 1e-6);
    EXPECT_NEAR(a[6], 0.9, 1e-6);   /* w at xfb_offset 24 */
}

/* introspection stays HONEST: a name-less module reports EMPTY resource names
 * (MAX_NAME_LENGTH 1), not the synthesized spellings (ARB_gl_spirv) */
TEST(nonames_introspection_honest)
{
    GLuint vs = load_spirv_stage(GL_VERTEX_SHADER, spv_mix_vs_nonames,
                                 sizeof spv_mix_vs_nonames / 4);
    GLuint gs = load_spirv_stage(GL_GEOMETRY_SHADER, spv_mix_gs_nonames,
                                 sizeof spv_mix_gs_nonames / 4);
    GLuint p;
    GLsizei len = 0x77;
    char name[64];
    GLenum prop = GL_NAME_LENGTH;
    GLint value = -1;

    if (!vs || !gs) return;
    p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, gs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(gs);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { glDeleteProgram(p); return; }
    }
    /* the first GS OUTPUT (location 0) reports an empty name / length 1 */
    glGetProgramResourceiv(p, GL_PROGRAM_OUTPUT, 0, 1, &prop, 1, NULL, &value);
    EXPECT_EQ_INT(value, 1);
    name[0] = '?';
    glGetProgramResourceName(p, GL_PROGRAM_OUTPUT, 0, sizeof name, &len, name);
    EXPECT_EQ_INT(len, 0);   /* the NUL only: length 0 chars, NAME_LENGTH 1 */
    EXPECT_EQ_INT(name[0], 0);
    /* p stays alive: glDeleteProgram recycles the GL name onto a stale xfb slot (xfb domain) */
}

/* ---------------- the r147 leak fence: xfb introspection honesty ------------
 *
 * A name-less module with a DECLARED capture must report "" / length-0 names
 * through EVERY query — glGetTransformFeedbackVarying, the
 * GL_TRANSFORM_FEEDBACK_VARYING interface (name + NAME_LENGTH), and the
 * TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH program query (1: the NUL alone) —
 * while its capture stays byte-identical; the named twin still reports its
 * real names.  No reported name may contain a synthesized "_mgl_" spelling. */
TEST(nonames_xfb_introspection)
{
    const char *vs = "#version 450 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    GLuint ref, svs, sgs, p;
    GLint rcount = -1, rmax = -1, count = -1, maxlen = -1;
    bool rl = false, leak = false;

    /* the named twin FIRST: real names, no "_mgl_", count/maxlen > 0 (guard) */
    ref = mgl_make_program5(vs, NULL, NULL, blk_gs_named,
        "#version 450 core\nlayout(location=0) out vec4 c;\nvoid main(){c=vec4(1);}\n");
    if (!ref) return;
    glGetProgramiv(ref, GL_TRANSFORM_FEEDBACK_VARYINGS, &rcount);
    EXPECT_TRUE(rcount > 0);
    for (int i = 0; i < rcount; i++)
    {
        char nm[256];
        GLsizei len = -1;
        GLenum type = 0;
        GLint size = 0;

        nm[0] = 0;
        glGetTransformFeedbackVarying(ref, i, sizeof nm, &len, &size, &type, nm);
        EXPECT_TRUE(len > 0);              /* a real name */
        if (strstr(nm, "_mgl_")) rl = true;
    }
    EXPECT_TRUE(!rl);
    glGetProgramiv(ref, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &rmax);
    EXPECT_TRUE(rmax > 1);

    /* the name-less program: every captured index reports "" / length 0, the
     * interface NAME_LENGTH 1, and MAX_LENGTH 1 (the NUL alone) */
    svs = load_spirv_stage(GL_VERTEX_SHADER, spv_blk_vs_nonames, sizeof spv_blk_vs_nonames / 4);
    sgs = load_spirv_stage(GL_GEOMETRY_SHADER, spv_blk_gs_nonames, sizeof spv_blk_gs_nonames / 4);
    if (!svs || !sgs) return;
    p = glCreateProgram();
    glAttachShader(p, svs);
    glAttachShader(p, sgs);
    glLinkProgram(p);
    glDeleteShader(svs);
    glDeleteShader(sgs);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
    }

    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &count);
    EXPECT_TRUE(count > 0);
    EXPECT_EQ_INT(count, rcount);              /* the same captured leaves */
    for (int i = 0; i < count; i++)
    {
        char nm[256];
        GLsizei len = -1;
        GLenum type = 0;
        GLint size = 0;

        nm[0] = '?';
        glGetTransformFeedbackVarying(p, i, sizeof nm, &len, &size, &type, nm);
        EXPECT_EQ_INT(len, 0);                 /* the empty string, 0 chars */
        EXPECT_EQ_INT(nm[0], 0);
        EXPECT_TRUE(size >= 1);
        EXPECT_TRUE(type != 0);
        if (strstr(nm, "_mgl_")) leak = true;
        {
            GLenum prop = GL_NAME_LENGTH;
            GLint value = -1;

            glGetProgramResourceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, i, 1, &prop, 1, NULL, &value);
            EXPECT_EQ_INT(value, 1);           /* the NUL alone */
            nm[0] = '?';
            glGetProgramResourceName(p, GL_TRANSFORM_FEEDBACK_VARYING, i, sizeof nm, &len, nm);
            EXPECT_EQ_INT(len, 0);
            EXPECT_EQ_INT(nm[0], 0);
            if (strstr(nm, "_mgl_")) leak = true;
        }
    }
    EXPECT_TRUE(!leak);
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &maxlen);
    EXPECT_EQ_INT(maxlen, 1);
    glGetProgramInterfaceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, GL_MAX_NAME_LENGTH, &maxlen);
    EXPECT_EQ_INT(maxlen, 1);
}


/* ---------------- r148 regression fences (the CTS regressions) -------------- */

/* ROOT CAUSE 1 (KHR-GL46.compute_shader.resources-max, first bad dd77727): the
 * r88 STORAGE_IMAGE over-count in the compute ACB base made 12 UBOs + 8 SSBOs +
 * 8 non-atomic images + 8 atomic counters exceed the Metal index ceiling
 * (29 + 8 > 30) and the link refused ("atomic counter buffer arguments could
 * not be declared") while the REAL signature has no image-atomic sidecar
 * buffers and fits (21 + 8 <= 30).  Fixed in program.c by retrying without the
 * over-count; the RED->GREEN proof is the CTS case itself (Fail at
 * baseline-2f95cadc, Pass after — verified in this round).  A headless fence
 * needs the exact 12+8+8+8 resource mix, whose shape trips a SEPARATE
 * pre-existing CS+block compile quirk (see the report). */
TEST(r148_state_queries_module)
{
    GLuint sh, p;
    GLint ok = 0, v[4] = {-1, -1, -1, -1};
    char nm[256];
    GLsizei len = -1;

    /* the CTS's exact stripped module (dumped from the running glcts via
     * MGL_SPIRV_DUMP; 261 words: 1 in, 1 uniform + 1 block, the block output
     * with XfbBuffer/XfbStride) is embedded as spv_state_vs_nonames */
    sh = load_spirv_stage(GL_VERTEX_SHADER, spv_state_vs_nonames,
                          sizeof spv_state_vs_nonames / 4);
    if (!sh) return;
    p = glCreateProgram();
    glAttachShader(p, sh);
    glDeleteShader(sh);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);                        /* was: recompile failed, link 0 */
    if (!ok) { glDeleteProgram(p); return; }

    glGetProgramiv(p, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &v[0]);
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_MAX_LENGTH, &v[1]);
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &v[2]);
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &v[3]);
    EXPECT_EQ_INT(v[0], 1);
    EXPECT_EQ_INT(v[1], 1);
    EXPECT_EQ_INT(v[2], 1);                /* was: 12 ("gl_Position") */
    EXPECT_EQ_INT(v[3], 1);
    nm[0] = '?';
    glGetTransformFeedbackVarying(p, 0, sizeof nm, &len, NULL, NULL, nm);
    EXPECT_EQ_INT(len, 0);
    EXPECT_EQ_INT(nm[0], 0);
    glDeleteProgram(p);
}

int main(void)
{
    GLuint vao = 0;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    /* r147's fence first: the twin must report real names before anything else
     * links — an order-dependence in the xfb program table (a separate xfb-domain
     * finding, noted in the report) otherwise serves a stale slot */
    RUN(nonames_render);
    RUN(nonames_xfb_block);
    RUN(nonames_introspection_honest);
    RUN(nonames_xfb_introspection);
    RUN(r148_state_queries_module);
    return mgl_test_summary();
}
