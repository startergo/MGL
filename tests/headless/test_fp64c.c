/*
 * test_fp64c.c — software IEEE fp64, part 3: dvecN vertex attributes
 * (glVertexAttribLPointer raw fetch, GL_DOUBLE conversions, generic current
 * values), flat double varying, dvec3/4 varyings, out-param fns, uniform
 * round trip, xfb capture.  (scalar ops: test_fp64.c; conversions/UBO/SSBO:
 * test_fp64b.c)
 */
#include "mgl_test.h"
#include <stdint.h>
#include <stdbool.h>

/* ------------------------------------------------------------------ fbo -- */

static GLuint g_fbo, g_tex;

static void f32fbo_begin(void)
{
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 4, 4);
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
}

static void f32fbo_end(void)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &g_fbo);
    glDeleteTextures(1, &g_tex);
    g_fbo = g_tex = 0;
}

static const char *g_vs =
    "#version 460 core\n"
    "layout(location=0) in vec2 p;\n"
    "void main(){ gl_Position = vec4(p, 0.0, 1.0); }\n";

static const char *fs_pre =
    "#version 460 core\n"
    "layout(location=0) out vec4 o;\n"
    "uniform double ua, ub, uc;\n"
    "uniform dvec3 uv;\n"
    "uniform dmat3 um;\n"
    "uniform int ui;\n"
    "uniform float uf;\n"
    "void main(){\n"
    "  double r = 0.0lf;\n";

static const char *fs_post =
    "  uvec2 b = unpackDouble2x32(r);\n"
    "  o = vec4(float(b.x & 65535u), float(b.x >> 16u), float(b.y & 65535u), float(b.y >> 16u));\n"
    "}\n";

/* run `body` (which assigns to r) and return the computed double bits */
static uint64_t run_double(const char *body, double a, double b, double c,
                           const double *vec3, const double *mat9, int ui, float uf)
{
    char src[4096];
    snprintf(src, sizeof src, "%s%s%s", fs_pre, body, fs_post);
    GLuint prog = mgl_make_program(g_vs, src);
    if (!prog) return UINT64_MAX;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glUseProgram(prog);
    glUniform1d(glGetUniformLocation(prog, "ua"), a);
    glUniform1d(glGetUniformLocation(prog, "ub"), b);
    glUniform1d(glGetUniformLocation(prog, "uc"), c);
    if (vec3) glUniform3d(glGetUniformLocation(prog, "uv"), vec3[0], vec3[1], vec3[2]);
    if (mat9) glUniformMatrix3dv(glGetUniformLocation(prog, "um"), 1, GL_FALSE, mat9);
    glUniform1i(glGetUniformLocation(prog, "ui"), ui);
    glUniform1f(glGetUniformLocation(prog, "uf"), uf);

    glDrawArrays(GL_TRIANGLES, 0, 3);
    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_FLOAT, px);
    uint64_t bits = ((uint64_t)(uint32_t)(unsigned short)px[0]) |
                    ((uint64_t)(uint32_t)(unsigned short)px[1] << 16) |
                    ((uint64_t)(uint32_t)(unsigned short)px[2] << 32) |
                    ((uint64_t)(uint32_t)(unsigned short)px[3] << 48);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    return bits;
}

typedef void (*dop_t)(double a, double b, double c, int ui, float uf, double *out);

/* any-NaN payload: NaN results are unspecified in both GLSL and IEEE for
 * these ops, so a NaN from the GPU matches a NaN from the CPU ref */
static int gd_nan(uint64_t bits)
{
    return (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL &&
           (bits & 0x000fffffffffffffULL) != 0;
}

static void check_op(const char *name, const char *body, dop_t cpu,
                     double a, double b, double c, int ui, float uf)
{
    uint64_t got = run_double(body, a, b, c, NULL, NULL, ui, uf);
    double want = 0;
    cpu(a, b, c, ui, uf, &want);
    uint64_t wbits;
    memcpy(&wbits, &want, 8);
    mgl_test_checks++;
    if (got != wbits && !(want != want && gd_nan(got))) {   /* any NaN payload is spec-legal */
        double gd;
        memcpy(&gd, &got, 8);
        MGL_FAIL("%s: bits %016llx (%.17g), expected %016llx (%.17g) [a=%.17g b=%.17g c=%.17g]",
                 name, (unsigned long long)got, gd, (unsigned long long)wbits, want, a, b, c);
    }
}

/* ------------------------------------------------------------ cpu refs -- */

static void c_add(double a, double b, double c, int ui, float uf, double *o) { *o = a + b; }
static void c_sub(double a, double b, double c, int ui, float uf, double *o) { *o = a - b; }
static void c_mul(double a, double b, double c, int ui, float uf, double *o) { *o = a * b; }
static void c_div(double a, double b, double c, int ui, float uf, double *o) { *o = a / b; }
static void c_neg(double a, double b, double c, int ui, float uf, double *o) { *o = -a; }
static void c_fma(double a, double b, double c, int ui, float uf, double *o) { *o = fma(a, b, c); }
static void c_sqrt(double a, double b, double c, int ui, float uf, double *o) { *o = sqrt(a); }
static void c_floor(double a, double b, double c, int ui, float uf, double *o) { *o = floor(a); }
static void c_ceil(double a, double b, double c, int ui, float uf, double *o) { *o = ceil(a); }
static void c_trunc(double a, double b, double c, int ui, float uf, double *o) { *o = trunc(a); }
/* GLSL round(): MGL rounds ties away from zero (implementation-defined
 * direction per the spec; matches mgl_fround64).  Exact: a - trunc(a) and the
 * neighbour are representable at these magnitudes; round(-0) is -0. */
static void c_round(double a, double b, double c, int ui, float uf, double *o)
{
    double t = trunc(a), f = fabs(a - t);
    *o = (f >= 0.5) ? t + copysign(1.0, a) : t;
}
static void c_roundeven(double a, double b, double c, int ui, float uf, double *o) { *o = nearbyint(a); }
static void c_fabs(double a, double b, double c, int ui, float uf, double *o) { *o = fabs(a); }
static void c_sign(double a, double b, double c, int ui, float uf, double *o) { *o = (a > 0) - (a < 0); }
static void c_fract(double a, double b, double c, int ui, float uf, double *o) { *o = a - floor(a); }
static void c_min(double a, double b, double c, int ui, float uf, double *o) { *o = a < b ? a : b; }
static void c_max(double a, double b, double c, int ui, float uf, double *o) { *o = a > b ? a : b; }
static void c_mod(double a, double b, double c, int ui, float uf, double *o) { *o = fmod(a, b); }
/* GLSL mod(x, y) = x - y*floor(x/y), rounded per operation; `volatile` keeps
 * the compiler from contracting the expression into an FMA (Apple clang
 * contracts by default), which would not match per-op GPU rounding. */
static void c_glmod(double a, double b, double c, int ui, float uf, double *o)
{
    volatile double q = a / b, f = floor(q), m = b * f;
    *o = a - m;
}
/* the "rem" row evaluates mod(ua, -ub): same GLSL formula with y = -b */
static void c_rem(double a, double b, double c, int ui, float uf, double *o)
{
    volatile double nb = -b;
    volatile double q = a / nb, f = floor(q), m = nb * f;
    *o = a - m;
}
static void c_mix(double a, double b, double c, int ui, float uf, double *o)
{
    /* MGL lowers OpFMix to fma(a, b - x, x): one rounding of the blend */
    volatile double d = b - a;
    *o = fma(c, d, a);
}
static void c_step(double a, double b, double c, int ui, float uf, double *o) { *o = b < a ? 0.0 : 1.0; }
static void c_clamp(double a, double b, double c, int ui, float uf, double *o) { *o = fmin(fmax(a, b), c); }
static void c_smooth(double a, double b, double c, int ui, float uf, double *o)
{
    volatile double d = (c - a) / (b - a);
    double t0 = fmin(fmax((double)d, 0.0), 1.0);
    volatile double t = t0;
    volatile double m = 3.0 - 2.0 * t, t2 = t * t;
    *o = t2 * m;
}
static void c_f2d(double a, double b, double c, int ui, float uf, double *o) { *o = (double)uf; }
static void c_d2f2d(double a, double b, double c, int ui, float uf, double *o) { *o = (double)(float)a; }
static void c_i2d(double a, double b, double c, int ui, float uf, double *o) { *o = (double)ui; }
static void c_d2i2d(double a, double b, double c, int ui, float uf, double *o) { *o = (double)(int)a; }
static void c_ldexp(double a, double b, double c, int ui, float uf, double *o) { *o = ldexp(a, ui); }
static void c_isnan(double a, double b, double c, int ui, float uf, double *o) { *o = isnan(a) ? 1.0 : 0.0; }
static void c_isinf(double a, double b, double c, int ui, float uf, double *o) { *o = isinf(a) ? 1.0 : 0.0; }
static void c_const(double a, double b, double c, int ui, float uf, double *o) { *o = 3.141592653589793238462643383279502884; }
static void c_constf(double a, double b, double c, int ui, float uf, double *o) { *o = 0.5001220703125; }

/* ---------------------------------------------------------------- tests -- */

TEST(fp64_varying_dvec34)
{
    static const struct { const char *ty; int n; double v[4]; } cases[] = {
        { "dvec2", 2, { 0.5, -2.25, 0, 0 } },
        { "dvec3", 3, { 0.5, -2.25, 1.0000000000002287, 0 } },
        { "dvec4", 4, { 0.5, -2.25, 1.0000000000002287, -1.000000000000001e300 } },
    };
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        char vs[512], fs[512], lit[256] = "", cond[512] = "";
        const char *swz[4] = { "x", "y", "z", "w" };
        for (int i = 0; i < cases[k].n; i++) {
            char b[128], c[128];
            snprintf(b, sizeof b, "%s%.17glf", i ? ", " : "", cases[k].v[i]);
            snprintf(c, sizeof c, "%s(v.%s == %.17glf)", i ? " && " : "", swz[i], cases[k].v[i]);
            strcat(lit, b);
            strcat(cond, c);
        }
        snprintf(vs, sizeof vs,
            "#version 430 core\n"
            "in vec2 pos;\n"
            "flat out %s v;\n"
            "void main() { gl_Position = vec4(pos, 0.0, 1.0); v = %s(%s); }\n",
            cases[k].ty, cases[k].ty, lit);
        snprintf(fs, sizeof fs,
            "#version 430 core\n"
            "flat in %s v;\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() { color = (%s) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1); }\n",
            cases[k].ty, cond);
        GLuint p = mgl_make_program(vs, fs);
        EXPECT_TRUE(p != 0);
        if (!p) continue;

        GLuint vao, vbo, fbo, tex;
        static const float v[3][2] = { {-1,-1},{3,-1},{-1,3} };
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
        glEnableVertexAttribArray(0);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glViewport(0, 0, 4, 4);
        glUseProgram(p);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        GLubyte px[3] = {9,9,9};
        glReadPixels(1, 1, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
        fprintf(stderr, "  %s varying: %u %u %u (want 0 255 0)\n", cases[k].ty, px[0], px[1], px[2]);
        EXPECT_TRUE(px[1] == 255 && px[0] == 0);
        glDeleteProgram(p);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    EXPECT_NO_GL_ERROR();
}

TEST(fp64_out_param_fns)
{
    const char *vs =
        "#version 430 core\n"
        "in vec2 pos;\n"
        "uniform double u;\n"
        "flat out double mant;\n"
        "flat out int expo;\n"
        "flat out double mfrac;\n"
        "flat out double mint;\n"
        "void main() {\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  int e; double ip;\n"
        "  mant = frexp(u, e); expo = e;\n"
        "  mfrac = modf(u, ip); mint = ip;\n"
        "}\n";
    GLuint p = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &vs, NULL);
    glCompileShader(sh);
    glAttachShader(p, sh);
    const char *names[4] = { "mant", "expo", "mfrac", "mint" };
    glTransformFeedbackVaryings(p, 4, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) return;

    GLuint vao, vbo, tfbuf;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    static const float v[3][2] = { {-1,-1},{3,-1},{-1,3} };
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &tfbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 256, NULL, GL_STREAM_READ);   /* 3 prims x 32B stride */
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfbuf);
    glUseProgram(p);

    const double inputs[] = { -1023.75, 0.0, 5e-320 };
    for (int i = 0; i < 3; i++) {
        glUniform1d(glGetUniformLocation(p, "u"), inputs[i]);
        glEnable(GL_RASTERIZER_DISCARD);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndTransformFeedback();
        glDisable(GL_RASTERIZER_DISCARD);
        unsigned char raw[40];
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof raw, raw);
        double mant, mfrac, mint;
        int e;
        memcpy(&mant, raw + 0, 8);    /* interleaved: mant@0, expo@8 (4B),
                                       * mfrac@16 (8B aligned), mint@24 */
        memcpy(&e, raw + 8, 4);
        memcpy(&mfrac, raw + 16, 8);
        memcpy(&mint, raw + 24, 8);
        int ce = 0;
        double cm = frexp(inputs[i], &ce);
        double cip, cf = modf(inputs[i], &cip);
        fprintf(stderr, "  %g: frexp=(%.17g,%d) modf=(%.17g,%.17g)\n", inputs[i], mant, e, mfrac, mint);
        EXPECT_TRUE(mant == cm && e == ce && mfrac == cf && mint == cip);
    }
    glDeleteProgram(p);
    glDeleteBuffers(1, &tfbuf);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    EXPECT_NO_GL_ERROR();
}

/* dvec3/dvec4 flat varyings: the interface repr is array[2] of uint4 (two GL
 * locations); every component must survive VS -> FS bit-exactly */
TEST(fp64_uniform_roundtrip)
{
    GLuint p = mgl_make_program(
        "#version 430 core\n"
        "uniform double u;\n"
        "uniform dvec2 v;\n"
        "uniform dmat2 m;\n"
        "void main() { gl_Position = vec4(float(u + v.x + m[0].x), 0.0, 0.0, 1.0); }\n",
        "#version 430 core\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(1); }\n");
    EXPECT_TRUE(p != 0);
    if (!p) return;
    glUseProgram(p);
    GLint lu = glGetUniformLocation(p, "u");
    GLint lv = glGetUniformLocation(p, "v");
    GLint lm = glGetUniformLocation(p, "m");
    EXPECT_TRUE(lu >= 0 && lv >= 0 && lm >= 0);
    double u = 1.0 + 0x1p-51;
    double v[2] = { -0.1e300, 3.141592653589793 };
    double m[4] = { 1.5, -2.5, 5.0e-324, 1e308 };  /* incl. smallest denormal + huge */
    glUniform1d(lu, u);
    glUniform2dv(lv, 1, v);
    glUniformMatrix2dv(lm, 1, GL_FALSE, m);
    double rb[4] = {0};
    glGetUniformdv(p, lu, rb);
    EXPECT_TRUE(rb[0] == u);
    glGetUniformdv(p, lv, rb);
    EXPECT_TRUE(rb[0] == v[0] && rb[1] == v[1]);
    glGetUniformdv(p, lm, rb);
    EXPECT_TRUE(rb[0] == m[0] && rb[1] == m[1] && rb[2] == m[2] && rb[3] == m[3]);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------- double<->float/int conversions --
 * OpFConvert in both directions, from uniforms and stage varyings, value-exact. */
TEST(fp64_xfb_capture)
{
    const char *vs =
        "#version 430 core\n"
        "in vec2 pos;\n"
        "out double dv;\n"
        "out dvec2 dv2;\n"
        "void main() {\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  dv  = 1.0lf + 0.0000000000009094947017729282379150390625lf * double(gl_VertexID + 1);\n"
        "  dv2 = dvec2(0.5lf, -2.0lf);\n"
        "}\n";
    GLuint p = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &vs, NULL);
    glCompileShader(sh);
    glAttachShader(p, sh);
    const char *names[2] = { "dv", "dv2" };
    glTransformFeedbackVaryings(p, 2, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "link failed: %s\n", log);
        EXPECT_TRUE(false);
        return;
    }
    GLint nv = 0;
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &nv);
    EXPECT_TRUE(nv == 2);

    GLuint vao, vbo, tfo, tfbuf;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    static const float v[3][2] = { {-1,-1},{3,-1},{-1,3} };
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &tfbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 3 * 24, NULL, GL_STREAM_READ);
    glGenTransformFeedbacks(1, &tfo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfbuf);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    GLdouble *out = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 3 * 24, GL_MAP_READ_BIT);
    if (!out) { fprintf(stderr, "map failed\n"); EXPECT_TRUE(false); return; }
    for (int i = 0; i < 3; i++) {
        double a = out[i * 3 + 0];
        double x = out[i * 3 + 1], y = out[i * 3 + 2];
        double want = 1.0 + 0x1p-40 * (i + 1);
        fprintf(stderr, "  v%d: dv=%a (want %a) dv2=(%.17g, %.17g)%s\n",
                i, a, want, x, y,
                (a == want && x == 0.5 && y == -2.0) ? "" : "  <-- MISMATCH");
    }
    EXPECT_TRUE(out[0] == 1.0 + 0.0000000000009094947017729282379150390625);
    EXPECT_TRUE(out[3] == 1.0 + 2.0 * 0.0000000000009094947017729282379150390625);
    EXPECT_TRUE(out[4] == 0.5 && out[5] == -2.0);
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------- dvecN vertex attributes -- */
/*
 * ARB_vertex_attrib_64bit on the draw path: dvecN inputs are fetched as raw
 * u32 word slots (two words per component) by the fp64 lowering, dvec3/dvec4
 * spanning locations k and k+1.  Pixel checks: each case draws a triangle and
 * reads the centre pixel (green = the doubles arrived exactly).
 */

static GLuint attr_draw(const char *vs, const void *verts, GLsizei stride, int lptr,
                        int size, GLenum type, int rel, GLuint extra_loc)
{
    GLuint p = mgl_make_program(vs,
        "#version 430 core\n"
        "in float v;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = (v == 1.0) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1); }\n");
    if (!p) return 0;
    GLuint vao, vbo;
    static const float pv[3][2] = { {-1,-1},{3,-1},{-1,3} };
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * stride + (extra_loc ? sizeof pv : 0), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * stride, verts);
    if (extra_loc)
    {
        glBufferSubData(GL_ARRAY_BUFFER, 3 * stride, sizeof pv, pv);
        glVertexAttribPointer(extra_loc, 2, GL_FLOAT, GL_FALSE, 8, (void *)(uintptr_t)(3 * stride));
        glEnableVertexAttribArray(extra_loc);
    }
    if (lptr) glVertexAttribLPointer(0, size, GL_DOUBLE, stride, (void *)(uintptr_t)rel);
    else      glVertexAttribPointer(0, size, type, GL_FALSE, stride, (void *)(uintptr_t)rel);
    glEnableVertexAttribArray(0);
    f32fbo_begin();
    glViewport(0, 0, 4, 4);
    glUseProgram(p);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLfloat px[4] = {9, 9, 9, 9};
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_FLOAT, px);
    glDeleteProgram(p);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    return (px[0] == 0.0f && px[1] == 1.0f && px[2] == 0.0f);
}

TEST(fp64_attributes)
{
    static const char *vs_pos =
        "#version 430 core\n"
        "layout(location = 0) in dvec4 pos;\n"
        "out float v;\n"
        "void main() { gl_Position = vec4(pos.xy / 2.0, 0.0, 1.0); v = float(pos.w == 1.0lf ? 1.0 : 0.0); }\n";

    /* dvec4 via two uint4 slots, direct raw fetch */
    {
        double v[3][4] = { {-2,-2,0,1},{6,-2,0,1},{-2,6,0,1} };
        EXPECT_TRUE(attr_draw(vs_pos, v, 32, true, 4, GL_DOUBLE, 0, 0));
    }
    /* dvec4, misaligned relative offset -> converted (cached) stream */
    {
        unsigned char buf[3 * 40];
        double v[3][4] = { {-2,-2,0,1},{6,-2,0,1},{-2,6,0,1} };
        for (int i = 0; i < 3; i++) memcpy(buf + i * 40 + 3, v[i], 32);
        EXPECT_TRUE(attr_draw(vs_pos, buf, 40, true, 4, GL_DOUBLE, 3, 0));
    }
    /* dvec3 (uint4 @k + uint2 @k+1) */
    {
        static const char *vs =
            "#version 430 core\n"
            "layout(location = 0) in dvec3 pos;\n"
            "layout(location = 2) in vec2 quad;\n"
            "out float v;\n"
            "void main() { gl_Position = vec4(quad * pos.x, 0.0, 1.0); v = float(pos.z == 1.0lf ? 1.0 : 0.0); }\n";
        double v[3][3] = { {1,0,1},{1,0,1},{1,0,1} };
        EXPECT_TRUE(attr_draw(vs, v, 24, true, 3, GL_DOUBLE, 0, 2));
    }
    /* dvec2 */
    {
        static const char *vs =
            "#version 430 core\n"
            "layout(location = 0) in dvec2 pos;\n"
            "out float v;\n"
            "void main() { gl_Position = vec4(pos / 2.0, 0.0, 1.0); v = 1.0; }\n";
        double v[3][2] = { {-2,-2},{6,-2},{-2,6} };
        EXPECT_TRUE(attr_draw(vs, v, 16, true, 2, GL_DOUBLE, 0, 0));
    }
    /* scalar double */
    {
        static const char *vs =
            "#version 430 core\n"
            "layout(location = 0) in double x;\n"
            "layout(location = 1) in vec2 quad;\n"
            "out float v;\n"
            "void main() { gl_Position = vec4(quad, 0.0, 1.0); v = float(x == 1.0lf ? 1.0 : 0.0); }\n";
        double v[3] = { 1, 1, 1 };
        EXPECT_TRUE(attr_draw(vs, v, 8, true, 1, GL_DOUBLE, 0, 1));
    }
    /* glVertexAttribPointer GL_DOUBLE: spec conversion (fp32 round trip) */
    {
        static const char *vs =
            "#version 430 core\n"
            "layout(location = 0) in dvec2 pos;\n"
            "out float v;\n"
            "void main() { gl_Position = vec4(pos / 2.0, 0.0, 1.0); v = 1.0; }\n";
        double v[3][2] = { {-2,-2},{6,-2},{-2,6} };
        EXPECT_TRUE(attr_draw(vs, v, 16, false, 2, GL_DOUBLE, 0, 0));
        /* a value only representable in fp32 conversions correctly: 1 + 2^-24 */
        double w[3][2] = { {-(1 + 0x1p-24) * 2, -2}, {6, -2}, {-2, 6} };
        EXPECT_TRUE(attr_draw(vs, w, 16, false, 2, GL_DOUBLE, 0, 0));
    }
    /* float data feeding a dvec2 input (widened to double) */
    {
        static const char *vs =
            "#version 430 core\n"
            "layout(location = 0) in dvec2 pos;\n"
            "out float v;\n"
            "void main() { gl_Position = vec4(pos / 2.0, 0.0, 1.0); v = 1.0; }\n";
        float v[3][2] = { {-2,-2},{6,-2},{-2,6} };
        EXPECT_TRUE(attr_draw(vs, v, 8, false, 2, GL_FLOAT, 0, 0));
    }
    /* disabled array: the generic current value from glVertexAttribL3d */
    {
        GLuint p = mgl_make_program(
            "#version 430 core\n"
            "layout(location = 0) in dvec3 scale;\n"
            "layout(location = 2) in vec2 pos;\n"
            "out float v;\n"
            "void main() { gl_Position = vec4(pos * scale.x, 0.0, 1.0); v = float(scale.z == 1.0lf ? 1.0 : 0.0); }\n",
            "#version 430 core\n"
            "in float v;\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() { color = (v == 1.0) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1); }\n");
        EXPECT_TRUE(p != 0);
        if (p)
        {
            GLuint vao, vbo;
            static const float pv[3][2] = { {-1,-1},{3,-1},{-1,3} };
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);
            glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof pv, pv, GL_STATIC_DRAW);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8, 0);
            glEnableVertexAttribArray(2);
            f32fbo_begin();
            glViewport(0, 0, 4, 4);
            glUseProgram(p);
            glDisableVertexAttribArray(0);
            glVertexAttribL3d(0, 1.0, 1.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLfloat px[4] = {9, 9, 9, 9};
            glReadPixels(1, 1, 1, 1, GL_RGBA, GL_FLOAT, px);
            EXPECT_TRUE(px[1] == 1.0f && px[0] == 0.0f);
            glDeleteProgram(p);
            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);
        }
    }
    /* dvec3/dvec4 consume two locations: an explicit attribute on the second
     * location must fail to link (location conflict) */
    {
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        GLuint p = glCreateProgram();
        const char *vsrc =
            "#version 430 core\n"
            "layout(location = 0) in dvec4 a;\n"
            "layout(location = 1) in vec2 b;\n"
            "void main() { gl_Position = vec4(b, 0.0, 1.0) + vec4(float(a.x)); }\n";
        const char *fsrc =
            "#version 430 core\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() { color = vec4(1); }\n";
        glShaderSource(vs, 1, &vsrc, NULL);
        glShaderSource(fs, 1, &fsrc, NULL);
        glCompileShader(vs);
        glCompileShader(fs);
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        GLint linked = 1;
        glGetProgramiv(p, GL_LINK_STATUS, &linked);
        EXPECT_TRUE(!linked);
        char log[512] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        EXPECT_TRUE(strstr(log, "conflicting vertex attribute locations") != NULL);
        glDeleteProgram(p);
        glDeleteShader(vs);
        glDeleteShader(fs);
        while (glGetError() != GL_NO_ERROR) {}
    }
    EXPECT_NO_GL_ERROR();
    f32fbo_end();
}


/* --------------------------------------------- doubles across the API -- */
TEST(fp64_flat_varying)
{
    /* color = blue when the double survives the stage boundary exactly */
    GLuint p = mgl_make_program(
        "#version 430 core\n"
        "in vec2 pos;\n"
        "flat out double d;\n"
        "void main() { gl_Position = vec4(pos, 0.0, 1.0); d = 1.00000000000002875lf; }\n",
        "#version 430 core\n"
        "flat in double d;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = (d == 1.00000000000002875lf) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1); }\n");
    EXPECT_TRUE(p != 0);
    if (!p) return;
    GLuint vao, vbo, fbo, tex;
    static const float v[3][2] = { {-1,-1},{3,-1},{-1,3} };
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glEnableVertexAttribArray(0);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 4, 4);
    glUseProgram(p);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLubyte px[3] = {9, 9, 9};
    glReadPixels(1, 1, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "  flat varying pixel: %u %u %u (want 0 255 0)\n", px[0], px[1], px[2]);
    EXPECT_TRUE(px[1] == 255 && px[0] == 0);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(fp64_attributes);
    RUN(fp64_flat_varying);
    RUN(fp64_varying_dvec34);
    RUN(fp64_out_param_fns);
    RUN(fp64_uniform_roundtrip);
    RUN(fp64_xfb_capture);
    return mgl_test_summary();
}
