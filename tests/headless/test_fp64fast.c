/*
 * test_fp64fast.c — MGL_FP64_FAST=1 double-single (float-float) fp64 mode.
 *
 * The fast library (MGL/src/fp64_df64.glsl) replaces SoftFloat behind the
 * SAME SPIR-V entry points, selected per compilation by the MGL_FP64_FAST
 * environment variable (read at every lowering, so one process can exercise
 * both modes).
 *
 *   exact    — MGL_FP64_FAST unset: results must stay bit-exact vs the CPU
 *              (proves the default SoftFloat path is unchanged).
 *   fast     — MGL_FP64_FAST=1: arithmetic is approximate: every value is
 *              checked against the CPU double with a relative-error bound
 *              (2^-44; the double-single unit roundoff is ~2^-47).
 *   fastexact — operations the fast library performs with integer/bit
 *              arithmetic only (abs/neg/sign/min/max/compares/rounding
 *              family/ldexp/frexp, float<->double, int64 round-trips below
 *              2^48) must still be bit-exact.
 *   distinct — the two modes must actually lower differently (a uint64 ->
 *              double conversion of a value that needs > 48 bits).
 */
#include "mgl_test.h"
#include <stdint.h>
#include <stdbool.h>

static GLuint g_fbo, g_tex;

static const char *g_vs =
    "#version 460 core\n"
    "layout(location=0) in vec2 p;\n"
    "void main(){ gl_Position = vec4(p, 0.0, 1.0); }\n";

static const char *fs_pre =
    "#version 460 core\n"
    "#extension GL_ARB_gpu_shader_int64 : require\n"
    "layout(location=0) out vec4 o;\n"
    "uniform double ua, ub, uc;\n"
    "uniform int ui;\n"
    "uniform float uf;\n"
    "void main(){\n"
    "  double r = 0.0lf;\n";

static const char *fs_post =
    "  uvec2 b = unpackDouble2x32(r);\n"
    "  o = vec4(float(b.x & 65535u), float(b.x >> 16u), float(b.y & 65535u), float(b.y >> 16u));\n"
    "}\n";

static GLuint g_vao, g_vbo;

static GLuint make_prog(const char *body)
{
    char src[2048];
    snprintf(src, sizeof src, "%s%s%s", fs_pre, body, fs_post);
    GLuint prog = mgl_make_program(g_vs, src);
    if (!prog) return 0;
    if (!g_vao) {
        glGenVertexArrays(1, &g_vao);
        glBindVertexArray(g_vao);
        glGenBuffers(1, &g_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        static const float tri[] = { -1,-1,  3,-1,  -1,3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
    }
    return prog;
}

/* render one pixel and return the raw double bits of `r` */
static uint64_t draw_bits(GLuint prog, double a, double b, double c, int i, float f)
{
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glUseProgram(prog);
    glUniform1d(glGetUniformLocation(prog, "ua"), a);
    glUniform1d(glGetUniformLocation(prog, "ub"), b);
    glUniform1d(glGetUniformLocation(prog, "uc"), c);
    glUniform1i(glGetUniformLocation(prog, "ui"), i);
    glUniform1f(glGetUniformLocation(prog, "uf"), f);
    glViewport(0, 0, 4, 4);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLfloat px[4];
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_FLOAT, px);
    uint32_t lo = (uint32_t)(uint16_t)(int)px[0] | ((uint32_t)(uint16_t)(int)px[1] << 16);
    uint32_t hi = (uint32_t)(uint16_t)(int)px[2] | ((uint32_t)(uint16_t)(int)px[3] << 16);
    return (uint64_t)hi << 32 | lo;
}

/* ---- inputs: 6 triples chosen to stress both large and small exponents ---- */
#define NIN 6
static const double INA[NIN] = { 3.141592653589793, 1.0e300 * 0 + 1.2345678901234567e17, -9876.543210987654321,
                                 0.10000000000000001, 5.0e-7, 123456789.123456789 };
static const double INB[NIN] = { 2.718281828459045, -9.87654321e8, 1234.5678901234567,
                                 0.30000000000000004, 3.0e-6, 0.000000123456789 };
static const double INC[NIN] = { 1.4142135623730951, 4.2e5, 7.7e-3,
                                 -1.0, 2.5e-8, 42.0 };

static int g_mode_fast = 0;   /* set by the fast battery: relax arithmetic */

static double bits_to_d(uint64_t u) { union { uint64_t u; double d; } x; x.u = u; return x.d; }
static uint64_t d_to_bits(double d) { union { uint64_t u; double d; } x; x.d = d; return x.u; }

static double g_ref_scale;   /* scale for the tolerance check (see check) */

static void check(const char *op, int k, uint64_t got, double ref)
{
    mgl_test_checks++;
    if (!g_mode_fast) {
        /* default SoftFloat path: exact match, except NaN payload */
        if (isnan(ref) || isnan(bits_to_d(got))) {
            if (isnan(ref) != isnan(bits_to_d(got)))
                MGL_FAIL("%s[%d]: nan mismatch got=%016llx ref=%016llx (mode default)", op, k,
                         (unsigned long long)got, (unsigned long long)d_to_bits(ref));
            return;
        }
        if (got != d_to_bits(ref)) {
            /* results composed of several correctly-rounded ops (mod) may
             * differ from the CPU reference expression by an ulp */
            double g = bits_to_d(got);
            double sc0 = g_ref_scale ? g_ref_scale : ref;
            double sc = fabs(sc0) > 1e-300 ? fabs(sc0) : 1e-300;
            if (fabs(g - ref) > sc * 1.5e-14)
                MGL_FAIL("%s[%d]: bits %016llx != exact %016llx (mode default)", op, k,
                         (unsigned long long)got, (unsigned long long)d_to_bits(ref));
        }
        return;
    }
    double g = bits_to_d(got);
    double r = ref;
    if (isnan(r) || isnan(g)) {
        if (isnan(r) != isnan(g))
            MGL_FAIL("%s[%d]: nan mismatch got=%016llx ref=%016llx", op, k,
                     (unsigned long long)got, (unsigned long long)d_to_bits(ref));
        return;
    }
    /* double-single unit roundoff ~2^-47; allow 2^-44 of the reference
     * scale (default |ref|; cancellation ops pass the input magnitude) */
    double sc0 = g_ref_scale ? g_ref_scale : ref;
    double sc = fabs(sc0) > 1e-300 ? fabs(sc0) : 1e-300;
    if (fabs(g - r) > sc * 5.7e-14)          /* 2^-44 */
        MGL_FAIL("%s[%d]: err %.3g (got %.17g ref %.17g)", op, k,
                 fabs(g - r) / sc, g, r);
}

/* ops that must be bit-exact even in fast mode; NaN payloads may differ */
static void check_exact(const char *op, int k, uint64_t got, double ref)
{
    mgl_test_checks++;
    if (isnan(ref) || isnan(bits_to_d(got))) {
        if (isnan(ref) != isnan(bits_to_d(got)))
            MGL_FAIL("%s[%d]: nan mismatch got=%016llx ref=%016llx", op, k,
                     (unsigned long long)got, (unsigned long long)d_to_bits(ref));
        return;
    }
    if (got != d_to_bits(ref))
        MGL_FAIL("%s[%d]: bits %016llx != %016llx (must be exact)", op, k,
                 (unsigned long long)got, (unsigned long long)d_to_bits(ref));
}

static void battery(void)
{
    struct { const char *name, *body; int exact; } ops[] = {
        { "add",   "r = ua + ub;", 0 },
        { "sub",   "r = ua - ub;", 0 },
        { "mul",   "r = ua * ub;", 0 },
        { "div",   "r = ua / ub;", 0 },
        { "fma",   "r = fma(ua, ub, uc);", 0 },
        { "sqrt",  "r = sqrt(ua);", 0 },
        { "abs",   "r = abs(ua);", 1 },
        { "neg",   "r = -ua;", 1 },
        { "sign",  "r = double(sign(ua));", 1 },
        /* rounding family + fract: exact when the result fits a df pair
           (~48 bits), tolerance-checked otherwise */
        { "floor", "r = floor(ua);", 0 },
        { "ceil",  "r = ceil(ua);", 0 },
        { "trunc", "r = trunc(ua);", 0 },
        { "round", "r = round(ua);", 0 },
        { "roundeven", "r = roundEven(ua);", 0 },
        { "fract", "r = fract(ua);", 0 },
        { "min",   "r = min(ua, ub);", 1 },
        { "max",   "r = max(ua, ub);", 1 },
        { "mod",   "r = mod(ua, ub);", 0 },
        { "f2d",   "r = double(uf);", 1 },
        { "d2f2d", "r = double(float(ua));", 0 },
        { "i2d",   "r = double(ui);", 1 },
        { "ldexp", "r = ldexp(ua, ui);", 0 },
        { "u2d",   "r = double(uint64_t(ua));", 0 },
        { "d2u2d", "r = double(uint64_t(double(uint64_t(ua))));", 0 },
    };

    for (size_t o = 0; o < sizeof ops / sizeof ops[0]; o++) {
        GLuint prog = make_prog(ops[o].body);
        if (!prog) continue;
        for (int k = 0; k < NIN; k++) {
            double a = INA[k], b = INB[k], c = INC[k];
            uint64_t got = draw_bits(prog, a, b, c, 3, 1.5f);
            double ref;
            if (!strcmp(ops[o].name, "add")) ref = a + b;
            else if (!strcmp(ops[o].name, "sub")) ref = a - b;
            else if (!strcmp(ops[o].name, "mul")) ref = a * b;
            else if (!strcmp(ops[o].name, "div")) ref = a / b;
            else if (!strcmp(ops[o].name, "fma")) ref = fma(a, b, c);
            else if (!strcmp(ops[o].name, "sqrt")) ref = sqrt(a);
            else if (!strcmp(ops[o].name, "abs")) ref = fabs(a);
            else if (!strcmp(ops[o].name, "neg")) ref = -a;
            else if (!strcmp(ops[o].name, "sign")) ref = a > 0 ? 1.0 : (a < 0 ? -1.0 : 0.0);
            else if (!strcmp(ops[o].name, "floor")) ref = floor(a);
            else if (!strcmp(ops[o].name, "ceil")) ref = ceil(a);
            else if (!strcmp(ops[o].name, "trunc")) ref = trunc(a);
            else if (!strcmp(ops[o].name, "round")) ref = round(a);
            else if (!strcmp(ops[o].name, "roundeven")) ref = nearbyint(a); /* FE_TONEAREST */
            else if (!strcmp(ops[o].name, "min")) ref = fmin(a, b);
            else if (!strcmp(ops[o].name, "max")) ref = fmax(a, b);
            else if (!strcmp(ops[o].name, "fract")) ref = a - floor(a);
            else if (!strcmp(ops[o].name, "mod")) ref = a - b * floor(a / b);
            else if (!strcmp(ops[o].name, "f2d")) ref = 1.5;
            else if (!strcmp(ops[o].name, "d2f2d")) ref = (double)(float)a;
            else if (!strcmp(ops[o].name, "i2d")) ref = 3.0;
            else if (!strcmp(ops[o].name, "ldexp")) ref = ldexp(a, 3);
            else if (!strcmp(ops[o].name, "u2d")) ref = (double)(a < 0 ? 0 : (uint64_t)a);
            else if (!strcmp(ops[o].name, "d2u2d")) ref = a < 0 ? 0.0 : (double)(uint64_t)(double)(uint64_t)a;
            else ref = 0;

            int cancelled = !strcmp(ops[o].name, "fract") || !strcmp(ops[o].name, "mod");
            int unsignd = !strcmp(ops[o].name, "u2d") || !strcmp(ops[o].name, "d2u2d");
            g_ref_scale = cancelled ? fmax(fabs(a), fabs(b)) : 0.0;
            if (unsignd && a < 0.0)
                continue;   /* f2u of negatives is undefined in the spec */
            if (ops[o].exact) check_exact(ops[o].name, k, got, ref);
            else check(ops[o].name, k, got, ref);
        }
        glDeleteProgram(prog);
    }
}

TEST(exact)
{
    /* default mode: everything bit-exact vs the CPU */
    g_mode_fast = 0;
    battery();
    EXPECT_NO_GL_ERROR();
}

TEST(fast)
{
    /* fast mode: arithmetic approximate, integer/bit ops exact */
    putenv((char *)"MGL_FP64_FAST=1");
    g_mode_fast = 1;
    battery();
    EXPECT_NO_GL_ERROR();
    putenv((char *)"MGL_FP64_FAST=0");
}

TEST(fast_exact_paths)
{
    /* compares and NaN/inf classification in fast mode via select() */
    putenv((char *)"MGL_FP64_FAST=1");
    const char *body =
        "  bool lt = ua < ub;\n"
        "  bool ge = ua >= ub;\n"
        "  bool eq = ua == ub;\n"
        "  double q = isnan(ua) ? -1.0lf : (lt ? 1.0lf : (ge ? 2.0lf : (eq ? 3.0lf : 4.0lf)));\n"
        "  q += isinf(ua) ? 10.0lf : 0.0lf;\n"
        "  r = q;\n";
    GLuint prog = make_prog(body);
    if (prog) {
        struct { double a, b; double want; } cases[] = {
            { 1.0, 2.0, 1.0 },
            { 2.0, 1.0, 2.0 },
            { 1.0, 1.0, 2.0 },        /* ge catches equality before eq */
            { -1.0, 1.0, 1.0 },
            { 0.0, -0.0, 2.0 },       /* +0 == -0: ge */
            { 1.0/0.0, 1.0, 12.0 },   /* inf >= 1 */
            { -1.0/0.0, 1.0, 11.0 },  /* -inf < 1, isinf */
            { 0.0/0.0, 1.0, -1.0 },   /* NaN */
        };
        for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
            uint64_t got = draw_bits(prog, cases[k].a, cases[k].b, 0, 0, 1.5f);
            check_exact("cmp", (int)k, got, cases[k].want);
        }
        glDeleteProgram(prog);
    }
    putenv((char *)"MGL_FP64_FAST=0");
}

TEST(distinct)
{
    /* the two modes must lower differently: converting a uint64 that needs
       more than 48 bits is exact in SoftFloat and rounded in double-single */
    const char *body = "  r = double(uint64_t(ua));\n";
    putenv((char *)"MGL_FP64_FAST=0");
    GLuint soft = make_prog(body);
    putenv((char *)"MGL_FP64_FAST=1");
    GLuint fast = make_prog(body);
    putenv((char *)"MGL_FP64_FAST=0");
    if (!soft || !fast) {
        if (soft) glDeleteProgram(soft);
        if (fast) glDeleteProgram(fast);
        return;
    }
    double v = 81985529216486895.0;   /* 0x0123456789ABCDEF: 55 bits */
    uint64_t s = draw_bits(soft, v, 0, 0, 0, 0);
    uint64_t f = draw_bits(fast, v, 0, 0, 0, 0);
    mgl_test_checks++;
    if (s == f)
        MGL_FAIL("fast mode did not change the lowering (bits %016llx both modes)",
                 (unsigned long long)s);
    check("distinct.soft", 0, s, v);
    check("distinct.fast", 0, f, v);
    glDeleteProgram(soft);
    glDeleteProgram(fast);
}

int main(void)
{
    /* FBO for the bit readback */
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 4, 4);
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    RUN(exact);
    RUN(fast);
    RUN(fast_exact_paths);
    RUN(distinct);
    return mgl_test_summary();
}
