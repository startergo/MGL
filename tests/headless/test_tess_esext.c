/* tests/headless/test_tess_esext.c — regressions from the KHR-GL44 ES-EXT
 * tessellation_shader family (esextcTessellationShader*), which exercises
 * shapes the dEQP ES31 / KHR-GL46 suites do not: GL_PATCH_VERTICES=1 draws,
 * xfb capture of gl_TessCoord with rasterizer discard, PRIMITIVES_GENERATED
 * queries on emulated (isolines) paths, extreme tess levels (-1 .. 64), and
 * the two-fresh-programs-per-config build/draw churn of the CTS utils.
 */
#include "mgl_test.h"

static const char *kVs =
    "#version 440\n"
    "void main() { gl_Position = vec4(1.0, 0.0, 0.0, 1.0); }\n";

static const char *kFs =
    "#version 440\n"
    "void main() {}\n";

static const char *kTcs1 =
    "#version 440\n"
    "layout (vertices = 1) out;\n"
    "uniform vec2 inner_tess_level;\n"
    "uniform vec4 outer_tess_level;\n"
    "void main()\n"
    "{\n"
    "    gl_out[gl_InvocationID].gl_Position = gl_in[0].gl_Position;\n"
    "    gl_TessLevelInner[0] = inner_tess_level.x;\n"
    "    gl_TessLevelInner[1] = inner_tess_level.y;\n"
    "    gl_TessLevelOuter[0] = outer_tess_level.x;\n"
    "    gl_TessLevelOuter[1] = outer_tess_level.y;\n"
    "    gl_TessLevelOuter[2] = outer_tess_level.z;\n"
    "    gl_TessLevelOuter[3] = outer_tess_level.w;\n"
    "}\n";

/* builds one counter/data program: fresh TCS/TES shader objects, the VS/FS
 * objects are the caller's (shared, exactly like the CTS utils) */
static GLuint make_prog(GLuint vs, GLuint fs, const char *tcs, const char *tes)
{
    GLuint p = glCreateProgram();
    GLuint tc = glCreateShader(GL_TESS_CONTROL_SHADER);
    GLuint te = glCreateShader(GL_TESS_EVALUATION_SHADER);
    glShaderSource(tc, 1, &tcs, NULL);
    glCompileShader(tc);
    glShaderSource(te, 1, &tes, NULL);
    glCompileShader(te);
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glAttachShader(p, tc);
    glAttachShader(p, te);
    glDeleteShader(tc);
    glDeleteShader(te);
    const char *varyings[] = { "result_uvw" };
    glTransformFeedbackVaryings(p, 1, varyings, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    return p;
}

static GLuint g_vs, g_fs, g_q, g_tfo, g_tfb;

/* CTS TessellationShaderUtils flow for one configuration:
 * counter program + query draw, then data program + xfb draw + readback */
static void cts_one_config(const char *spacing, const float inner[2], const float outer[4],
                           GLuint *prims_out, int *mapped_out)
{
    char tes[512];
    snprintf(tes, sizeof tes,
             "#version 440\n"
             "layout (%s, %s, ccw) in;\n"
             "out vec3 result_uvw;\n"
             "void main()\n"
             "{\n"
             "    gl_Position = gl_in[0].gl_Position;\n"
             "    result_uvw  = gl_TessCoord;\n"
             "}\n", "isolines", spacing);

    /* 1. counter program: query draw with rasterizer discard */
    GLuint po = make_prog(g_vs, g_fs, kTcs1, tes);
    GLint link = 0;
    glGetProgramiv(po, GL_LINK_STATUS, &link);
    if (link != GL_TRUE) { fprintf(stdout, "  LINK FAIL (counter, %s)\n", spacing); glDeleteProgram(po); *prims_out = 0; *mapped_out = 0; return; }
    glUseProgram(po);
    glUniform2fv(glGetUniformLocation(po, "inner_tess_level"), 1, inner);
    glUniform4fv(glGetUniformLocation(po, "outer_tess_level"), 1, outer);

    glBeginQuery(GL_PRIMITIVES_GENERATED, g_q);
    glDrawArrays(GL_PATCHES, 0, 1);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    GLuint prim = 0;
    glGetQueryObjectuiv(g_q, GL_QUERY_RESULT, &prim);
    glDeleteProgram(po);
    *prims_out = prim;

    GLuint nvtx = prim * 2;   /* isolines: 2 vertices per segment */
    if (!nvtx) { *mapped_out = 0; return; }

    /* 2. data program: xfb draw + map/copy */
    GLuint po2 = make_prog(g_vs, g_fs, kTcs1, tes);
    glGetProgramiv(po2, GL_LINK_STATUS, &link);
    if (link != GL_TRUE) { fprintf(stdout, "  LINK FAIL (data, %s)\n", spacing); glDeleteProgram(po2); return; }
    glUseProgram(po2);
    glUniform2fv(glGetUniformLocation(po2, "inner_tess_level"), 1, inner);
    glUniform4fv(glGetUniformLocation(po2, "outer_tess_level"), 1, outer);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, g_tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, g_tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, nvtx * 3 * sizeof(float), NULL, GL_STATIC_DRAW);
    glBeginTransformFeedback(GL_LINES);
    glDrawArrays(GL_PATCHES, 0, 1);
    glEndTransformFeedback();
    float *data = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, nvtx * 3 * sizeof(float), GL_MAP_READ_BIT);
    if (data) {
        for (GLuint i = 0; i < nvtx; i++)
            if (data[i*3] < -1e-4f || data[i*3] > 1.0001f || data[i*3+1] < -1e-4f || data[i*3+1] > 1.0001f) {
                fprintf(stdout, "  uv[%u]=(%g %g) out of range\n", i, data[i*3], data[i*3+1]);
                break;
            }
        *mapped_out = (int)nvtx;
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    }
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteProgram(po2);
    while (glGetError() != GL_NO_ERROR) {}
}

TEST(TessEsExtIsolinesCtsSweep)
{
    GLint max_level = 0;
    glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &max_level);
    EXPECT_EQ_INT(max_level, 64);

    const float lv[] = { -1.0f, 4.0f, (float)max_level * 0.5f, (float)max_level };
    const int nl = 4;
    static const char *spc[3] = { "equal_spacing", "fractional_even_spacing", "fractional_odd_spacing" };

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    g_vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(g_vs, 1, &kVs, NULL);
    glCompileShader(g_vs);
    g_fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(g_fs, 1, &kFs, NULL);
    glCompileShader(g_fs);
    glGenQueries(1, &g_q);
    glGenBuffers(1, &g_tfb);
    glGenTransformFeedbacks(1, &g_tfo);

    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glEnable(GL_RASTERIZER_DISCARD);

    GLuint prims_total = 0, runs = 0;
    int mapped_total = 0;
    for (int s = 0; s < 3; s++)
    for (int o1 = 0; o1 < nl; o1++)
    for (int o2 = 0; o2 < nl; o2++)
    for (int irr = 0; irr < 2; irr++)
    {
        float inner[2] = { lv[irr], lv[irr] };
        float outer[4] = { lv[o1], lv[o2], lv[irr], lv[irr] };
        GLuint prims = 0; int mapped = 0;
        cts_one_config(spc[s], inner, outer, &prims, &mapped);
        prims_total += prims;
        mapped_total += mapped;
        runs++;
    }
    printf("  cts sweep: %u runs, %u prims, %d mapped vertices\n", runs, prims_total, mapped_total);
    EXPECT_TRUE(runs == 3 * 4 * 4 * 2);
    EXPECT_TRUE(prims_total > 0);
    EXPECT_TRUE(mapped_total > 0);

    glDisable(GL_RASTERIZER_DISCARD);
    glDeleteQueries(1, &g_q);
    glDeleteBuffers(1, &g_tfb);
    glDeleteTransformFeedbacks(1, &g_tfo);
    glDeleteShader(g_vs);
    glDeleteShader(g_fs);
    glDeleteVertexArrays(1, &vao);
}

/* the exact first configuration, verified against the GL isolines rule */
TEST(TessEsExtIsolinesOnePatchVertex)
{
    const char *tes =
        "#version 440\n"
        "layout (isolines, equal_spacing, ccw) in;\n"
        "out vec3 result_uvw;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = gl_in[0].gl_Position;\n"
        "    result_uvw  = gl_TessCoord;\n"
        "}\n";

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    g_vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(g_vs, 1, &kVs, NULL);
    glCompileShader(g_vs);
    g_fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(g_fs, 1, &kFs, NULL);
    glCompileShader(g_fs);
    glGenQueries(1, &g_q);
    glGenBuffers(1, &g_tfb);
    glGenTransformFeedbacks(1, &g_tfo);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glEnable(GL_RASTERIZER_DISCARD);

    float inner[2] = { -1.0f, -1.0f };   /* irrelevant for isolines */
    float outer[4] = { 2.0f, 2.0f, -1.0f, -1.0f };
    GLuint prims = 0; int mapped = 0;
    cts_one_config("equal_spacing", inner, outer, &prims, &mapped);
    EXPECT_NO_GL_ERROR();
    printf("  isolines prims=%u (want 4: 2 lines x 2 segments)\n", prims);
    EXPECT_EQ_INT(prims, 4);

    /* capture again and check the first segment: (0,0) -> (0.5,0) */
    GLuint nvtx = prims * 2;
    GLuint po = make_prog(g_vs, g_fs, kTcs1, tes);
    glUseProgram(po);
    glUniform2fv(glGetUniformLocation(po, "inner_tess_level"), 1, inner);
    glUniform4fv(glGetUniformLocation(po, "outer_tess_level"), 1, outer);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, g_tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, g_tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, nvtx * 3 * sizeof(float), NULL, GL_STATIC_DRAW);
    glBeginTransformFeedback(GL_LINES);
    glDrawArrays(GL_PATCHES, 0, 1);
    glEndTransformFeedback();
    float *data = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, nvtx * 3 * sizeof(float), GL_MAP_READ_BIT);
    EXPECT_TRUE(data != NULL);
    if (data)
    {
        printf("  uv0=(%g %g) uv1=(%g %g)\n", data[0], data[1], data[3], data[4]);
        EXPECT_NEAR(data[0], 0.0f, 1e-5f);
        EXPECT_NEAR(data[1], 0.0f, 1e-5f);
        EXPECT_NEAR(data[3], 0.5f, 1e-5f);
        EXPECT_NEAR(data[4], 0.0f, 1e-5f);
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    }
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDisable(GL_RASTERIZER_DISCARD);
    glDeleteProgram(po);
    glDeleteQueries(1, &g_q);
    glDeleteBuffers(1, &g_tfb);
    glDeleteTransformFeedbacks(1, &g_tfo);
    glDeleteShader(g_vs);
    glDeleteShader(g_fs);
    glDeleteVertexArrays(1, &vao);
}

/* r59: dEQP-GL45-ES31 opaque_type_indexing tessellation_* — an ES-profile TCS
 * with an explicit textureLod on a sampler*Shadow (glslang gates the overload
 * behind GL_EXT_texture_shadow_lod + an ES version >= 320; MGL's compat pass
 * enables the extension and presents the shader as the ESSL 3.20 it is
 * equivalent to).  The compare must actually run: LEQUAL against 0.75/0.25.
 * r60: the ES3.1 alias strings live in a separate droppable commit — a
 * '#extension GL_EXT_tessellation_shader' source is only meaningful when that
 * alias is advertised, so the test self-skips (PASS + note) without it. */
static int tess_es_alias_advertised(void)
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (glGetStringi(GL_EXTENSIONS, i) &&
            !strcmp((const char *)glGetStringi(GL_EXTENSIONS, i), "GL_EXT_tessellation_shader"))
            return 1;
    return 0;
}

TEST(TessEsShadowLodCompat)
{
    GLuint tex[2];
    float lvl0[4] = { 0.25f, 0.25f, 0.25f, 0.25f };
    float lvl1[4] = { 0.75f, 0.75f, 0.75f, 0.75f };
    if (!tess_es_alias_advertised())
    {
        printf("SKIP TessEsShadowLodCompat (GL_EXT_tessellation_shader not advertised; droppable alias commit absent)\n");
        return;
    }
    glGenTextures(2, tex);

    const char *tcs =
        "#version 310 es\n"
        "#extension GL_EXT_tessellation_shader : require\n"
        "uniform highp sampler2DArrayShadow s[2];\n"
        "layout (vertices = 1) out;\n"
        "out highp vec2 tc_res[];\n"
        "void main()\n"
        "{\n"
        "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
        "    tc_res[gl_InvocationID] = vec2(textureLod(s[0], vec4(0.5, 0.5, 0.0, 0.5), 0.0),\n"
        "                                    textureLod(s[1], vec4(0.5, 0.5, 0.0, 0.5), 0.0));\n"
        "}\n";
    const char *tes =
        "#version 310 es\n"
        "#extension GL_EXT_tessellation_shader : require\n"
        "layout (triangles) in;\n"
        "in highp vec2 tc_res[];\n"
        "out mediump vec4 col;\n"
        "void main()\n"
        "{\n"
        "    col = vec4(tc_res[0], 0.0, 1.0);\n"
        "    gl_Position = vec4(gl_TessCoord.xy * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 col;\n"
        "layout(location = 0) out vec4 o;\n"
        "void main() { o = col; }\n";
    const char *vs =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(0, 0, 0, 1); }\n";

    /* one-layer depth arrays on units 0 and 1 */
    for (int i = 0; i < 2; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex[i]);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, 2, 2, 1, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, i ? lvl1 : lvl0);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    }

    GLuint p = glCreateProgram();
    GLenum ty[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    const char *src[4] = { vs, tcs, tes, fs };
    for (int i = 0; i < 4; i++)
    {
        GLuint sh = glCreateShader(ty[i]);
        glShaderSource(sh, 1, &src[i], NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint link = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE)
    {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  link log: %s", log);
    }
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);
    glUniform1iv(glGetUniformLocation(p, "s"), 2, (GLint[]){ 0, 1 });

    GLuint fbo, ctex;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &ctex);
    glBindTexture(GL_TEXTURE_2D, ctex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 8, 8);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, ctex, 0);
    glViewport(0, 0, 8, 8);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glDrawArrays(GL_PATCHES, 0, 1);
    EXPECT_NO_GL_ERROR();
    unsigned char px[3] = { 0, 0, 0 };
    glReadPixels(1, 1, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    printf("  shadow px: (%d %d %d) want (0 255 0)\n", px[0], px[1], px[2]);
    EXPECT_EQ_INT(px[0], 0);    /* unit 0: 0.5 <= 0.25 fails */
    EXPECT_EQ_INT(px[1], 255);  /* unit 1: 0.5 <= 0.75 passes */

    while (glGetError() != GL_NO_ERROR) {}
}

/* dEQP-GL45-ES31 opaque_type_indexing sampler.*.compute.*shadow (r63): a
 * shadow sampler ARRAY in a compute kernel occupies [[sampler(b)]]..b+n-1;
 * binding every element at the array's base slot left element 1 on Metal's
 * default sampler, whose missing compare function made sample_compare return
 * the raw depth. */
TEST(Es32ComputeShadowSamplerArray)
{
    GLuint t0, t1, sbo, prog;
    /* 2x2 depth, row-major upload: row 0 = (0.25, 0.75), row 1 = (0.25, 0.75) */
    float d[4] = { 0.25f, 0.75f, 0.25f, 0.75f };
    static const char *cs =
        "#version 320 es\n"
        "uniform highp sampler2DShadow samp[2];\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 0) buffer Out { highp float o[4]; };\n"
        "void main()\n"
        "{\n"
        "    o[0] = texture(samp[0], vec3(0.25, 0.5, 0.5));\n"
        "    o[1] = texture(samp[0], vec3(0.25, 0.5, 0.125));\n"
        "    o[2] = texture(samp[1], vec3(0.75, 0.5, 0.5));\n"
        "    o[3] = texture(samp[1], vec3(0.75, 0.5, 0.9));\n"
        "}\n";

    glGenTextures(1, &t0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 2, 2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, d);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    glGenTextures(1, &t1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, t1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 2, 2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, d);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    prog = mgl_make_program_cs(cs);
    EXPECT_NE(prog, 0);
    if (!prog) return;

    glGenBuffers(1, &sbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo);
    float zero[4] = { -1, -1, -1, -1 };
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof zero, zero, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sbo);

    glUseProgram(prog);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    float *m = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    if (m) {
        /* NEAREST at (0.25,0.5) -> texel(0,1) = 0.25; at (0.75,0.5) -> texel(1,1) = 0.75;
         * GL_LEQUAL: result = (ref <= depth) */
        printf("  shadow array: %g %g %g %g (expect 0 1 1 0)\n", m[0], m[1], m[2], m[3]);
        mgl_test_checks++;
        if (m[0] != 0.0f || m[1] != 1.0f || m[2] != 1.0f || m[3] != 0.0f)
            MGL_FAIL("compute shadow sampler array element 1 lost its compare function");
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    } else {
        MGL_FAIL("map failed");
    }
    glDeleteProgram(prog);
    glDeleteBuffers(1, &sbo);
    glDeleteTextures(1, &t0);
    glDeleteTextures(1, &t1);
}

/* dEQP-GL45-ES31 opaque_type_indexing atomic_counter.*_vertex (r63): an
 * atomic_uint ARRAY is one buffer binding (the elements live inside it), not
 * one Metal argument per element; the per-element expansion filled all 32
 * vertex buffer slots and failed the draw. */
TEST(Es32AtomicCounterArrayVertex)
{
    static const char *vs =
        "#version 320 es\n"
        "layout(binding = 0) uniform atomic_uint counter[1024];\n"
        "in highp vec4 a_position;\n"
        "flat out highp uint r0;\n"
        "flat out highp uint r1;\n"
        "flat out highp uint r2;\n"
        "flat out highp uint r3;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = a_position;\n"
        "    gl_PointSize = 1.0;\n"
        "    r0 = atomicCounterIncrement(counter[2]);\n"
        "    r1 = atomicCounterIncrement(counter[2]);\n"
        "    r2 = atomicCounterIncrement(counter[0]);\n"
        "    r3 = atomicCounterIncrement(counter[0]);\n"
        "}\n";
    static const char *fs =
        "#version 320 es\n"
        "flat in highp uint r0, r1, r2, r3;\n"
        "layout(location = 0) out highp uvec4 o;\n"
        "void main() { o = uvec4(r0, r1, r2, r3); }\n";
    GLuint p = mgl_make_program(vs, fs);

    EXPECT_NE(p, 0);
    if (!p) return;

    GLuint cbo;
    glGenBuffers(1, &cbo);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, cbo);
    GLuint zero[1024] = { 0 };
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof zero, zero, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, cbo);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, 1, 1);
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pos[4] = { 0, 0, 0, 1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    GLint a = glGetAttribLocation(p, "a_position");
    glEnableVertexAttribArray(a);
    glVertexAttribPointer(a, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glViewport(0, 0, 1, 1);
    glUseProgram(p);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLuint px[4];
    glReadPixels(0, 0, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
    printf("  atomic array vertex: %u %u %u %u (expect 0 1 0 1)\n", px[0], px[1], px[2], px[3]);
    mgl_test_checks++;
    if (px[0] != 0 || px[1] != 1 || px[2] != 0 || px[3] != 1)
        MGL_FAIL("vertex atomic counter array return values wrong");

    GLuint cnt[4] = { 0 };
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof cnt, cnt);
    mgl_test_checks++;
    if (cnt[0] != 2 || cnt[2] != 2)
        MGL_FAIL("vertex atomic counter array buffer wrong");

    glDeleteProgram(p);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &cbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(void)
{
    RUN(TessEsExtIsolinesOnePatchVertex);
    RUN(TessEsExtIsolinesCtsSweep);
    RUN(TessEsShadowLodCompat);
    RUN(Es32ComputeShadowSamplerArray);
    RUN(Es32AtomicCounterArrayVertex);
    return mgl_test_summary();
}
