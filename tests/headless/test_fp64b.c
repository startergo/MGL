/*
 * test_fp64b.c — part 2 of test_fp64.c (see there). — software IEEE fp64 (GL_ARB_gpu_shader_fp64 / ARB_vertex_attrib_64bit).
 *
 * The fragment shader computes a double expression and writes the raw 64 bits
 * of the result to an RGBA32F target as four 16-bit halves, so the readback is
 * bit-exact and can be compared against the same expression computed on the
 * CPU with hardware doubles.
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

TEST(fp64_conversions)
{
    f32fbo_begin();
    /* double -> float truncation of a uniform, then a fragment output: the
     * 24-bit mantissa of 1.0 + 2^-40 truncates to exactly 1.0f */
    GLuint p = mgl_make_program(g_vs,
        "#version 460 core\n"
        "layout(location=0) out vec4 o;\n"
        "uniform double ua;\n"
        "void main(){ o = vec4(float(ua), float(ua*2.0lf), float(ua-1.0lf), float(-ua)); }\n");
    EXPECT_TRUE(p != 0);
    glUseProgram(p);
    glUniform1d(glGetUniformLocation(p, "ua"), 1.0 + 0x1p-40);
    static const float quad[3][2] = { {-1,-1},{3,-1},{-1,3} };
    GLuint vao, buf; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &buf); glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    float px[4] = {-9,-9,-9,-9};
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NEAR(px[0], 1.0f, 1e-6);
    EXPECT_NEAR(px[1], 2.0f, 1e-6);
    EXPECT_NEAR(px[2], 0.0f, 1e-6);
    EXPECT_NEAR(px[3], -1.0f, 1e-6);
    glDeleteProgram(p);

    /* float -> double -> float round trip plus int/double mixing */
    p = mgl_make_program(g_vs,
        "#version 460 core\n"
        "layout(location=0) out vec4 o;\n"
        "uniform float fa;\n"
        "uniform int ib;\n"
        "void main(){ double d = double(fa) * double(ib);\n"
        "            o = vec4(float(d), float(d/3.0lf), float(double(ib)), float(sqrt(d))); }\n");
    EXPECT_TRUE(p != 0);
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "fa"), 0.5f);
    glUniform1i(glGetUniformLocation(p, "ib"), 3);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(2, 2, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NEAR(px[0], 1.5f, 1e-6);
    EXPECT_NEAR(px[1], 0.5f, 1e-6);
    EXPECT_NEAR(px[2], 3.0f, 1e-6);
    EXPECT_NEAR(px[3], 1.224744871f, 1e-5);
    glDeleteProgram(p);

    /* dvecN -> vecN component-wise */
    p = mgl_make_program(g_vs,
        "#version 460 core\n"
        "layout(location=0) out vec4 o;\n"
        "uniform dvec4 v;\n"
        "void main(){ o = vec4(v) * 0.5f; }\n");
    EXPECT_TRUE(p != 0);
    glUseProgram(p);
    double dv[4] = {0.25, 0.5, 1.0, 2.0};
    glUniform4dv(glGetUniformLocation(p, "v"), 1, dv);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(2, 2, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NEAR(px[0], 0.125f, 1e-6);
    EXPECT_NEAR(px[1], 0.25f, 1e-6);
    EXPECT_NEAR(px[2], 0.5f, 1e-6);
    EXPECT_NEAR(px[3], 1.0f, 1e-6);
    glDeleteProgram(p);
    glDeleteBuffers(1, &buf); glDeleteVertexArrays(1, &vao);
    f32fbo_end();
}

/* -------------------------------- std140 uniform blocks with double members --
 * every member kind (scalar, dvec2/3/4, scalar-double arrays with the 16-byte
 * std140 stride, dvecN arrays, dmatN) must read the exact double the app wrote
 * at the introspected offset. */
static const char *ubo_fs =
    "#version 460 core\n"
    "flat in double o; layout(location=0) out vec4 c;\n"
    "void main(){ c = vec4(float(o),1,0,1); }\n";

static GLuint ubo_p;
static char ubo_buf[512];

static void ubo_begin(const char *decl, const char *sum)
{
    char vs[1024];
    snprintf(vs, sizeof vs,
        "#version 460 core\nlayout(std140) uniform Blk { %s } b;\n"
        "layout(location=0) in vec2 pos;\nflat out double o;\n"
        "void main(){ gl_Position = vec4(pos,0,1); o = %s; }", decl, sum);
    ubo_p = mgl_make_program(vs, ubo_fs);
    EXPECT_TRUE(ubo_p != 0);
    if (ubo_p) glUseProgram(ubo_p);
}

static void ubo_put(int off, const double *v, int n)
{
    for (int i = 0; i < n; i++) memcpy(ubo_buf + off + 8*i, &v[i], 8);
}

static int ubo_off(const char *name)
{
    GLuint ui = 0; GLint off = -1;
    glGetUniformIndices(ubo_p, 1, &name, &ui);
    if (ui == 0xffffffffu) return -1;
    glGetActiveUniformsiv(ubo_p, 1, &ui, GL_UNIFORM_OFFSET, &off);
    return off;
}

static void ubo_draw(const char *tag, int dsz)
{
    GLint idx = glGetUniformBlockIndex(ubo_p, "Blk");
    EXPECT_TRUE(idx >= 0);
    GLuint ubo, vao, buf;
    glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, dsz, ubo_buf, GL_STATIC_DRAW);
    glUniformBlockBinding(ubo_p, idx, 3);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, ubo);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &buf); glBindBuffer(GL_ARRAY_BUFFER, buf);
    float tri[3][2] = { {-1,-1},{3,-1},{-1,3} };
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3); glFinish();
    GLubyte px[4]; glReadPixels(1,1,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);
    if (px[0] < 254) MGL_FAIL("%s: %.3f (expected 1.0)", tag, px[0]/255.0f);
    glDeleteBuffers(1, &ubo); glDeleteBuffers(1, &buf); glDeleteVertexArrays(1, &vao);
    glDeleteProgram(ubo_p); ubo_p = 0;
    memset(ubo_buf, 0, sizeof ubo_buf);
}

TEST(fp64_ubo_layout)
{
    f32fbo_begin();
    { ubo_begin("double s;", "b.s;"); int o = ubo_off("Blk.s");
      double d[] = {1.0}; ubo_put(o, d, 1); ubo_draw("scalar", 8); }
    { ubo_begin("dvec2 v;", "b.v.y;"); int o = ubo_off("Blk.v");
      double d[] = {0.5, 1.0}; ubo_put(o, d, 2); ubo_draw("dvec2.y", 16); }
    { ubo_begin("dvec3 v;", "b.v.z;"); int o = ubo_off("Blk.v");
      double d[] = {0.5, 0.5, 1.0}; ubo_put(o, d, 3); ubo_draw("dvec3.z", 32); }
    { ubo_begin("dvec4 v;", "b.v.w;"); int o = ubo_off("Blk.v");
      double d[] = {0.5, 0.5, 0.5, 1.0}; ubo_put(o, d, 4); ubo_draw("dvec4.w", 32); }
    /* the classic std140 minefield: offsets 0/16/32/64 with the dvec3->dvec4 gap */
    { ubo_begin("double s; dvec2 v2; dvec3 v3; dvec4 v4; double arr[2];",
                "b.s + b.v2.x + b.v3.x + b.v4.x + b.arr[0] + b.arr[1];");
      double v = 1.0/6.0;
      ubo_put(0,&v,1); ubo_put(16,&v,1); ubo_put(32,&v,1);
      ubo_put(64,&v,1); ubo_put(96,&v,1); ubo_put(112,&v,1);
      ubo_draw("full-mix", 128); }
    { ubo_begin("double a[2];", "b.a[1];"); int o = ubo_off("Blk.a");
      double d[] = {0.5, 1.0}; ubo_put(o, d, 1); ubo_put(o+16, d+1, 1); ubo_draw("darr[1]", 32); }
    { ubo_begin("dvec2 a[2];", "b.a[1].y;"); int o = ubo_off("Blk.a");
      double d[] = {0.5, 0.5, 0.5, 1.0}; ubo_put(o, d, 2); ubo_put(o+16, d+2, 2); ubo_draw("d2arr[1].y", 32); }
    { ubo_begin("dvec3 a[2];", "b.a[1].z;"); int o = ubo_off("Blk.a");
      double d[] = {0.5,0.5,0.5, 0.5,0.5,1.0}; ubo_put(o, d, 3); ubo_put(o+32, d+3, 3); ubo_draw("d3arr[1].z", 96); }
    { ubo_begin("dvec4 a[2];", "b.a[1].w;"); int o = ubo_off("Blk.a");
      double d[] = {0.5,0.5,0.5,0.5, 0.5,0.5,0.5,1.0};
      ubo_put(o, d, 4); ubo_put(o+32, d+4, 4); ubo_draw("d4arr[1].w", 128); }
    { ubo_begin("dmat2 m;", "b.m[1].y;"); int o = ubo_off("Blk.m");
      double d[] = {0.5,0.5, 0.5,1.0}; ubo_put(o, d, 4); ubo_draw("dmat2[1].y", 64); }
    { ubo_begin("dmat3x2 m;", "b.m[2].y;"); int o = ubo_off("Blk.m");
      double d[] = {0.5,0.5, 0.5,0.5, 0.5,1.0};
      ubo_put(o, d, 2); ubo_put(o+16, d+2, 2); ubo_put(o+32, d+4, 2); ubo_draw("dmat3x2[2].y", 64); }
    { ubo_begin("dmat4 m;", "b.m[3].w;"); int o = ubo_off("Blk.m");
      double d[] = {0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,1.0};
      for (int c = 0; c < 4; c++) ubo_put(o + 32*c, d + 4*c, 4);
      ubo_draw("dmat4[3].w", 128); }
    f32fbo_end();
}

/* ------------------ SSBO double loads and stores from a compute shader ------
 * std430 (natural strides, unwidened) loads plus stores into a widened std140
 * scalar-double array; results are compared bit-exactly. */
TEST(fp64_ssbo_compute)
{
    GLuint p = glCreateProgram();
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    const char *src =
        "#version 460 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430) buffer In { double s; dvec2 v2; dvec3 v3; dvec4 v4; double arr[3]; } i;\n"
        "layout(std140) buffer Out { double o[8]; } o1;\n"
        "void main(){\n"
        "  o1.o[0] = i.s;\n"
        "  o1.o[1] = i.v2.y;\n"
        "  o1.o[2] = i.v3.z;\n"
        "  o1.o[3] = i.v4.w;\n"
        "  o1.o[4] = i.arr[0] + i.arr[1] + i.arr[2];\n"
        "  o1.o[5] = i.arr[2] * 2.0lf;\n"
        "  o1.o[6] = sqrt(i.v2.x);\n"
        "  o1.o[7] = i.s / i.v2.x;\n"
        "}\n";
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, sh); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char l[2048]; glGetProgramInfoLog(p, sizeof l, NULL, l); MGL_FAIL("link: %s", l); return; }
    glUseProgram(p);

    const GLenum props[1] = { GL_OFFSET };
    const char *names[5] = {"In.s","In.v2","In.v3","In.v4","In.arr"};
    GLint offs[5] = {-1,-1,-1,-1,-1};
    for (int k = 0; k < 5; k++) {
        GLuint idx = glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, names[k]);
        EXPECT_TRUE(idx != GL_INVALID_INDEX);
        GLsizei len = 0;
        glGetProgramResourceiv(p, GL_BUFFER_VARIABLE, idx, 1, props, 1, &len, &offs[k]);
    }
    char inb[512]; memset(inb, 0, sizeof inb);
    double s = 1.5, v2y = 4.0, v3z = 9.0, v4w = 16.0;
    double arr[3] = {2.0, 3.0, 4.0};
    memcpy(inb+offs[0], &s, 8);
    memcpy(inb+offs[1], &arr[0], 8);        /* v2.x = 2.0 */
    memcpy(inb+offs[1]+8, &v2y, 8);         /* v2.y */
    memcpy(inb+offs[2]+16, &v3z, 8);        /* v3.z */
    memcpy(inb+offs[3]+24, &v4w, 8);        /* v4.w */
    memcpy(inb+offs[4], arr, 24);           /* arr[0..2], stride 8 */

    GLuint bufs[2]; glGenBuffers(2, bufs);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof inb, inb, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufs[1]);
    char init[256]; memset(init, 0, sizeof init);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256, init, GL_DYNAMIC_COPY);

    glDispatchCompute(1,1,1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[1]);
    char rb[256]; glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 256, rb);
    const double expect[8] = {1.5, 4.0, 9.0, 16.0, 9.0, 8.0, sqrt(2.0), 0.75};
    for (int k = 0; k < 8; k++) {
        double r; memcpy(&r, rb + 16*k, 8);   /* std140 array stride 16 */
        if (memcmp(&r, &expect[k], 8) != 0) MGL_FAIL("o[%d] = %.17g, expected %.17g", k, r, expect[k]);
    }
    EXPECT_NO_GL_ERROR();
    glDeleteBuffers(2, bufs);
    glDeleteShader(sh);
    glDeleteProgram(p);
}

/* ------------------------------------------------- extension advertisement --
 * both fp64 extensions must be listed and their entry points must work */
TEST(fp64_extensions)
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    bool fp64 = false, va64 = false;
    for (GLint i = 0; i < n; i++) {
        const char *e = (const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (!e) continue;
        if (!strcmp(e, "GL_ARB_gpu_shader_fp64")) fp64 = true;
        if (!strcmp(e, "GL_ARB_vertex_attrib_64bit")) va64 = true;
    }
    EXPECT_TRUE(fp64);
    EXPECT_TRUE(va64);

    /* ARB_vertex_attrib_64bit entry points */
    GLuint b, vao;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    double d[3] = {1.5, -2.5, 0.25};
    glBufferData(GL_ARRAY_BUFFER, sizeof d, d, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glVertexAttribLPointer(3, 1, GL_DOUBLE, 0, (void *)0);
    glVertexAttribL1d(5, 1.25);
    glVertexAttribL2d(6, 1.0, 2.0);
    glVertexAttribL3d(7, 1.0, 2.0, 4.0);
    glVertexAttribL4d(8, 1.0, 2.0, 4.0, 8.0);
    double q[4] = {-1,-1,-1,-1};
    glGetVertexAttribLdv(5, GL_CURRENT_VERTEX_ATTRIB, q);
    EXPECT_TRUE(q[0] == 1.25 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 1.0);
    glGetVertexAttribLdv(8, GL_CURRENT_VERTEX_ATTRIB, q);
    EXPECT_TRUE(q[0] == 1.0 && q[1] == 2.0 && q[2] == 4.0 && q[3] == 8.0);
    GLint sz = 0, tp = 0;
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_SIZE, &sz);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_TYPE, &tp);
    EXPECT_TRUE(sz == 1 && tp == GL_DOUBLE);
    EXPECT_NO_GL_ERROR();
    glDeleteBuffers(1, &b);
    glDeleteVertexArrays(1, &vao);
}

int main(void)
{
    RUN(fp64_conversions);
    RUN(fp64_ubo_layout);
    RUN(fp64_ssbo_compute);
    RUN(fp64_extensions);
    return mgl_test_summary();
}
