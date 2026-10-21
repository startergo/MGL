/*
 * test_fp64.c — software IEEE fp64 (GL_ARB_gpu_shader_fp64 / ARB_vertex_attrib_64bit).
 *
 * fp64_scalar_ops: the fragment shader computes a double expression and writes
 * the raw 64 bits of the result to an RGBA32F target as four 16-bit halves, so
 * the readback is bit-exact and can be compared against the same expression
 * computed on the CPU with hardware doubles.  One program per operator (32
 * compiles); the 13 input triples only change uniforms.
 * (attributes / flat varying live in test_fp64b.c)
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

/* One program per `body`: the 13 input triples only change uniforms, so the
 * program (glslang -> SPIR-V -> fp64 lowering -> MSL -> Metal pipeline) is
 * compiled once and reused for every case — 32 compiles instead of 416. */
static GLuint g_dprog_vao, g_dprog_vbo;

static GLuint make_double_prog(const char *body)
{
    char src[4096];
    snprintf(src, sizeof src, "%s%s%s", fs_pre, body, fs_post);
    GLuint prog = mgl_make_program(g_vs, src);
    if (!prog) return 0;
    if (!g_dprog_vao) {
        glGenVertexArrays(1, &g_dprog_vao);
        glBindVertexArray(g_dprog_vao);
        glGenBuffers(1, &g_dprog_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, g_dprog_vbo);
        static const float tri[] = { -1,-1,  3,-1,  -1,3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
    }
    return prog;
}

/* draw one case with an already-compiled program and read the result bits */
static uint64_t eval_double(GLuint prog, double a, double b, double c,
                            const double *vec3, const double *mat9, int ui, float uf)
{
    glBindVertexArray(g_dprog_vao);
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
    return ((uint64_t)(uint32_t)(unsigned short)px[0]) |
           ((uint64_t)(uint32_t)(unsigned short)px[1] << 16) |
           ((uint64_t)(uint32_t)(unsigned short)px[2] << 32) |
           ((uint64_t)(uint32_t)(unsigned short)px[3] << 48);
}

typedef void (*dop_t)(double a, double b, double c, int ui, float uf, double *out);

/* any-NaN payload: NaN results are unspecified in both GLSL and IEEE for
 * these ops, so a NaN from the GPU matches a NaN from the CPU ref */
static int gd_nan(uint64_t bits)
{
    return (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL &&
           (bits & 0x000fffffffffffffULL) != 0;
}

static void check_op(const char *name, GLuint prog, dop_t cpu,
                     double a, double b, double c, int ui, float uf)
{
    uint64_t got = prog ? eval_double(prog, a, b, c, NULL, NULL, ui, uf) : UINT64_MAX;
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

TEST(fp64_scalar_ops)
{
    f32fbo_begin();
    struct { const char *name, *body; dop_t cpu; } t[] = {
        { "add",   "r = ua + ub;", c_add },
        { "sub",   "r = ua - ub;", c_sub },
        { "mul",   "r = ua * ub;", c_mul },
        { "div",   "r = ua / ub;", c_div },
        { "neg",   "r = -ua;", c_neg },
        { "fma",   "r = fma(ua, ub, uc);", c_fma },
        { "sqrt",  "r = sqrt(ua);", c_sqrt },
        { "floor", "r = floor(ua);", c_floor },
        { "ceil",  "r = ceil(ua);", c_ceil },
        { "trunc", "r = trunc(ua);", c_trunc },
        { "round", "r = round(ua);", c_round },
        { "roundeven", "r = roundEven(ua);", c_roundeven },  /* ties to even */
        { "abs",   "r = abs(ua);", c_fabs },
        { "sign",  "r = sign(ua);", c_sign },
        { "fract", "r = fract(ua);", c_fract },
        { "min",   "r = min(ua, ub);", c_min },
        { "max",   "r = max(ua, ub);", c_max },
        { "mod",   "r = mod(ua, ub);", c_glmod },
        { "rem",   "r = mod(ua, -ub);", c_rem },
        { "mix",   "r = mix(ua, ub, uc);", c_mix },
        { "step",  "r = step(ua, ub);", c_step },
        { "clamp", "r = clamp(ua, ub, uc);", c_clamp },
        { "smooth", "r = smoothstep(ua, ub, uc);", c_smooth },
        { "f2d",   "r = double(uf);", c_f2d },
        { "d2f2d", "r = double(float(ua));", c_d2f2d },
        { "i2d",   "r = double(ui);", c_i2d },
        { "d2i2d", "r = double(int(ua));", c_d2i2d },
        { "ldexp", "r = ldexp(ua, ui);", c_ldexp },
        { "isnan", "r = isnan(ua) ? 1.0lf : 0.0lf;", c_isnan },
        { "isinf", "r = isinf(ua) ? 1.0lf : 0.0lf;", c_isinf },
        { "const", "r = 3.141592653589793238462643383279502884lf;", c_const },
        { "constf", "r = double(0.5001220703125f);", c_constf },  /* float literal widens exactly */
    };
    double cases[][3] = {
        { 1.0, 2.0, 0.5 },
        { 0.3333333333333333, -7.125, 0.1 },
        { 1e300, 1e-300, 0.25 },
        { -0.0, 4.0, 0.75 },
        { 123456789.123456789, 0.0000123456789, 0.9999999999 },
        { 1023.9999999999999, 3.0, 0.5 },
        { 4503599627370497.0, 3.0, 0.5 },        /* odd mantissa */
        { 2.2250738585072014e-308, 2.0, 0.5 },   /* smallest normal */
        { 4.9406564584124654e-324, 2.0, 0.5 },   /* smallest denormal */
        { 1.7976931348623157e308, 2.0, 0.5 },    /* max double */
        { -510.5, -510.5, 0.5 },                 /* ties: step(edge==x)=1,
                                                    roundEven to even */
        { 2.5, 3.5, 0.5 },
        { 4503599627370495.5, 0.5, 0.5 },        /* tie just below 2^52 */
    };
    int iu[3] = { 3, -40, 1074 };
    for (size_t k = 0; k < sizeof t / sizeof t[0]; k++) {
        /* one program per operator; the 13 input triples only change uniforms */
        GLuint prog = make_double_prog(t[k].body);
        for (size_t v = 0; v < sizeof cases / sizeof cases[0]; v++) {
            check_op(t[k].name, prog, t[k].cpu, cases[v][0], cases[v][1], cases[v][2],
                     iu[v % 3], (float)cases[v][1]);
        }
        if (prog) glDeleteProgram(prog);
    }
    if (g_dprog_vbo) { glDeleteBuffers(1, &g_dprog_vbo); g_dprog_vbo = 0; }
    if (g_dprog_vao) { glDeleteVertexArrays(1, &g_dprog_vao); g_dprog_vao = 0; }
    f32fbo_end();
}

int main(void)
{
    RUN(fp64_scalar_ops);
    return mgl_test_summary();
}
