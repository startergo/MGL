/* test_uniforms.c — default-block uniform packing + glProgramUniform* (core profile)
 *
 * Two reports from the compat domain, checked here on a core-profile context:
 *   1. "individual default-block float uniforms next to vec4s are clobbered by
 *      MGL's UBO packing" -> mixed float/vec2/vec3/vec4/mat/int/uint/bool/array
 *      uniforms, set forward, in reverse, and with single re-writes; verified
 *      via glGetUniform* readback AND rendered pixels (fragment stage), plus a
 *      vertex-stage program with the same sandwich pattern.
 *   2. "glProgramUniform1f* silently no-ops" -> every glProgramUniform* entry
 *      point is called on a linked program that is NOT the current program,
 *      then the program is bound and the values are checked in readback and
 *      in rendered pixels.
 */
#include "mgl_test.h"

#define W 100
#define H 100

static const char *VS =
    "#version 460 core\n"
    "layout(location=0) in vec3 pos;\n"
    "void main(){ gl_Position = vec4(pos, 1.0); }\n";

static const char *VS_UNI =
    "#version 460 core\n"
    "layout(location=0) in vec3 pos;\n"
    "layout(location=0) out float vo;\n"
    "uniform vec4 a0;\n"
    "uniform float s1;\n"
    "uniform vec4 a1;\n"
    "uniform float s2;\n"
    "uniform vec3 t2;\n"
    "void main(){ vo = (a0.x + s1 + a1.w + s2 + t2.y) * 0.5; gl_Position = vec4(pos, 1.0); }\n";

/* same uniforms consumed by BOTH stages (per-stage block packing) */
static const char *VS_SHARED =
    "#version 460 core\n"
    "layout(location=0) in vec3 pos;\n"
    "layout(location=0) out float vo;\n"
    "uniform vec4 a0;\n"
    "uniform float s1;\n"
    "uniform vec4 a1;\n"
    "uniform float s2;\n"
    "void main(){ vo = a0.y + s1 * 0.5f + a1.y + s2 * 0.25f; gl_Position = vec4(pos, 1.0); }\n";

static const char *FS_OUT =
    "#version 460 core\n"
    "layout(location=0) in float vo;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vec4(vo, 0.25, 0.5, 1.0); }\n";

/* mixed default-block uniforms: floats, vecs, matrices, ints, arrays */
static const char *FS =
    "#version 460 core\n"
    "layout(location=0) out vec4 color;\n"
    "uniform int sel;\n"
    "uniform vec4 a0;\n"
    "uniform float s1;\n"
    "uniform vec4 a1;\n"
    "uniform float s2;\n"
    "uniform vec2 t1;\n"
    "uniform float s3;\n"
    "uniform vec3 t2;\n"
    "uniform float s4;\n"
    "uniform mat4 m4;\n"
    "uniform mat3 m3;\n"
    "uniform int i1;\n"
    "uniform uint u1;\n"
    "uniform bool b1;\n"
    "uniform float arr[3];\n"
    "uniform vec4 varr[2];\n"
    "void main(){\n"
    "  float x = 0.0;\n"
    "  if (sel == 0) x = s1;\n"
    "  else if (sel == 1) x = a0.x; else if (sel == 2) x = a0.w;\n"
    "  else if (sel == 3) x = s2; else if (sel == 4) x = a1.x; else if (sel == 5) x = a1.w;\n"
    "  else if (sel == 6) x = t1.x; else if (sel == 7) x = t1.y;\n"
    "  else if (sel == 8) x = s3; else if (sel == 9) x = t2.x; else if (sel == 10) x = t2.z;\n"
    "  else if (sel == 11) x = s4;\n"
    "  else if (sel == 12) x = m4[0].x; else if (sel == 13) x = m4[2].z; else if (sel == 14) x = m4[3].w;\n"
    "  else if (sel == 15) x = m3[1].y;\n"
    "  else if (sel == 16) x = float(i1) / 255.0;\n"
    "  else if (sel == 17) x = float(u1) / 255.0;\n"
    "  else if (sel == 18) x = b1 ? 1.0 : 0.0;\n"
    "  else if (sel == 19) x = arr[0]; else if (sel == 20) x = arr[2];\n"
    "  else if (sel == 21) x = varr[0].x; else if (sel == 22) x = varr[1].w;\n"
    "  color = vec4(x, 0.25, 0.5, 1.0);\n"
    "}\n";

static GLuint g_vao, g_vbo;

static void setup_geom(void)
{
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    float v[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,-1,0,  1,1,0,  -1,1,0 };
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void *)0);
    glEnableVertexAttribArray(0);
}

static void reset_state(void)
{
    glDisable(GL_BLEND); glDisable(GL_CULL_FACE); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST);
    glColorMask(1, 1, 1, 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, W, H);
    while (glGetError() != GL_NO_ERROR) {}
}

/* --- expected values (all in [0,1], separated by >= 0.05) --- */
static const float E_s1 = 0.20f, E_s2 = 0.40f, E_s3 = 0.60f, E_s4 = 0.80f;
static const float E_a0[4]  = { 0.10f, 0.21f, 0.22f, 0.23f };
static const float E_a1[4]  = { 0.30f, 0.31f, 0.32f, 0.33f };
static const float E_t1[2]  = { 0.50f, 0.51f };
static const float E_t2[3]  = { 0.70f, 0.71f, 0.72f };
static const float E_arr[3] = { 0.11f, 0.12f, 0.13f };
static const float E_varr[8]= { 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.24f, 0.25f };
static const GLint  E_i1 = 80;
static const GLuint E_u1 = 100;

/* column-major: m4[0].x=0.60, m4[2].z=0.61, m4[3].w=0.62; m3[1].y=0.90 */
static const float E_m4[16] = { 0.60f,0.02f,0.02f,0.02f,  0.02f,0.02f,0.02f,0.02f,
                                0.02f,0.02f,0.61f,0.02f,  0.02f,0.02f,0.02f,0.62f };
static const float E_m3[9]  = { 0.03f,0.03f,0.03f,  0.03f,0.90f,0.03f,  0.03f,0.03f,0.03f };

typedef struct { int sel; float x; const char *what; } Case;
static const Case CASES[] = {
    { 0,  0.20f, "s1" },     { 1,  0.10f, "a0.x" },    { 2,  0.23f, "a0.w" },
    { 3,  0.40f, "s2" },     { 4,  0.30f, "a1.x" },    { 5,  0.33f, "a1.w" },
    { 6,  0.50f, "t1.x" },   { 7,  0.51f, "t1.y" },
    { 8,  0.60f, "s3" },     { 9,  0.70f, "t2.x" },    { 10, 0.72f, "t2.z" },
    { 11, 0.80f, "s4" },
    { 12, 0.60f, "m4[0].x" },{ 13, 0.61f, "m4[2].z" },{ 14, 0.62f, "m4[3].w" },
    { 15, 0.90f, "m3[1].y" },
    { 16, (float)80/255.0f, "i1" }, { 17, (float)100/255.0f, "u1" }, { 18, 1.0f, "b1" },
    { 19, 0.11f, "arr[0]" }, { 20, 0.13f, "arr[2]" },
    { 21, 0.14f, "varr[0].x" }, { 22, 0.25f, "varr[1].w" },
};

typedef struct {
    GLint a0, s1, a1, s2, t1, s3, t2, s4, m4, m3, i1, u1, b1, arr, varr, sel;
} Locs;

static void get_locs(GLuint prog, Locs *l)
{
    l->a0 = glGetUniformLocation(prog, "a0");
    l->s1 = glGetUniformLocation(prog, "s1");
    l->a1 = glGetUniformLocation(prog, "a1");
    l->s2 = glGetUniformLocation(prog, "s2");
    l->t1 = glGetUniformLocation(prog, "t1");
    l->s3 = glGetUniformLocation(prog, "s3");
    l->t2 = glGetUniformLocation(prog, "t2");
    l->s4 = glGetUniformLocation(prog, "s4");
    l->m4 = glGetUniformLocation(prog, "m4");
    l->m3 = glGetUniformLocation(prog, "m3");
    l->i1 = glGetUniformLocation(prog, "i1");
    l->u1 = glGetUniformLocation(prog, "u1");
    l->b1 = glGetUniformLocation(prog, "b1");
    l->arr = glGetUniformLocation(prog, "arr[0]");
    l->varr = glGetUniformLocation(prog, "varr[0]");
    l->sel = glGetUniformLocation(prog, "sel");
    EXPECT_TRUE(l->a0 >= 0 && l->s1 >= 0 && l->a1 >= 0 && l->s2 >= 0);
    EXPECT_TRUE(l->t1 >= 0 && l->s3 >= 0 && l->t2 >= 0 && l->s4 >= 0);
    EXPECT_TRUE(l->m4 >= 0 && l->m3 >= 0 && l->i1 >= 0 && l->u1 >= 0 && l->b1 >= 0);
    EXPECT_TRUE(l->arr >= 0 && l->varr >= 0 && l->sel >= 0);
}

/* set every uniform through glUniform*, forward or reverse order */
static void set_all_gl(GLuint prog, const Locs *l, int reverse)
{
    if (!reverse) {
        glUniform4fv(l->a0, 1, E_a0); glUniform1f(l->s1, E_s1);
        glUniform4fv(l->a1, 1, E_a1); glUniform1f(l->s2, E_s2);
        glUniform2fv(l->t1, 1, E_t1); glUniform1f(l->s3, E_s3);
        glUniform3fv(l->t2, 1, E_t2); glUniform1f(l->s4, E_s4);
        glUniformMatrix4fv(l->m4, 1, GL_FALSE, E_m4);
        glUniformMatrix3fv(l->m3, 1, GL_FALSE, E_m3);
        glUniform1i(l->i1, E_i1); glUniform1ui(l->u1, E_u1); glUniform1i(l->b1, GL_TRUE);
        glUniform1fv(l->arr, 3, E_arr); glUniform4fv(l->varr, 2, E_varr);
    } else {
        glUniform4fv(l->varr, 2, E_varr); glUniform1fv(l->arr, 3, E_arr);
        glUniform1i(l->b1, GL_TRUE); glUniform1ui(l->u1, E_u1); glUniform1i(l->i1, E_i1);
        glUniformMatrix3fv(l->m3, 1, GL_FALSE, E_m3);
        glUniformMatrix4fv(l->m4, 1, GL_FALSE, E_m4);
        glUniform1f(l->s4, E_s4); glUniform3fv(l->t2, 1, E_t2); glUniform1f(l->s3, E_s3);
        glUniform2fv(l->t1, 1, E_t1); glUniform1f(l->s2, E_s2);
        glUniform4fv(l->a1, 1, E_a1); glUniform1f(l->s1, E_s1); glUniform4fv(l->a0, 1, E_a0);
    }
}

/* set every uniform through glProgramUniform* */
static void set_all_program(GLuint prog, const Locs *l)
{
    glProgramUniform4fv(prog, l->a0, 1, E_a0);
    glProgramUniform1f(prog, l->s1, E_s1);
    glProgramUniform4f(prog, l->a1, E_a1[0], E_a1[1], E_a1[2], E_a1[3]);
    glProgramUniform1f(prog, l->s2, E_s2);
    glProgramUniform2fv(prog, l->t1, 1, E_t1);
    glProgramUniform1f(prog, l->s3, E_s3);
    glProgramUniform3fv(prog, l->t2, 1, E_t2);
    glProgramUniform1f(prog, l->s4, E_s4);
    glProgramUniformMatrix4fv(prog, l->m4, 1, GL_FALSE, E_m4);
    glProgramUniformMatrix3fv(prog, l->m3, 1, GL_FALSE, E_m3);
    glProgramUniform1i(prog, l->i1, E_i1);
    glProgramUniform1ui(prog, l->u1, E_u1);
    glProgramUniform1i(prog, l->b1, GL_TRUE);
    glProgramUniform1fv(prog, l->arr, 3, E_arr);
    glProgramUniform4fv(prog, l->varr, 2, E_varr);
}

static void check_readback(GLuint prog, const Locs *l)
{
    float f[16];
    GLint iv[4];
    GLuint uv[4];

    EXPECT_NEAR((glGetUniformfv(prog, l->s1, f), f[0]), E_s1, 1e-6);
    EXPECT_NEAR((glGetUniformfv(prog, l->s2, f), f[0]), E_s2, 1e-6);
    EXPECT_NEAR((glGetUniformfv(prog, l->s3, f), f[0]), E_s3, 1e-6);
    EXPECT_NEAR((glGetUniformfv(prog, l->s4, f), f[0]), E_s4, 1e-6);
    glGetUniformfv(prog, l->a0, f);
    EXPECT_NEAR(f[0], E_a0[0], 1e-6); EXPECT_NEAR(f[3], E_a0[3], 1e-6);
    glGetUniformfv(prog, l->a1, f);
    EXPECT_NEAR(f[0], E_a1[0], 1e-6); EXPECT_NEAR(f[3], E_a1[3], 1e-6);
    glGetUniformfv(prog, l->t1, f);
    EXPECT_NEAR(f[0], E_t1[0], 1e-6); EXPECT_NEAR(f[1], E_t1[1], 1e-6);
    glGetUniformfv(prog, l->t2, f);
    EXPECT_NEAR(f[0], E_t2[0], 1e-6); EXPECT_NEAR(f[2], E_t2[2], 1e-6);
    glGetUniformfv(prog, l->m4, f);
    EXPECT_NEAR(f[0], E_m4[0], 1e-6); EXPECT_NEAR(f[10], E_m4[10], 1e-6); EXPECT_NEAR(f[15], E_m4[15], 1e-6);
    glGetUniformfv(prog, l->m3, f);
    EXPECT_NEAR(f[4], E_m3[4], 1e-6);
    glGetUniformiv(prog, l->i1, iv);   EXPECT_EQ_INT(iv[0], E_i1);
    glGetUniformuiv(prog, l->u1, uv);  EXPECT_EQ_INT(uv[0], E_u1);
    glGetUniformiv(prog, l->b1, iv);   EXPECT_EQ_INT(iv[0], 1);
    glGetUniformfv(prog, l->arr + 1, f);   EXPECT_NEAR(f[0], E_arr[1], 1e-6);
    glGetUniformfv(prog, l->varr, f);      EXPECT_NEAR(f[0], E_varr[0], 1e-6);
    glGetUniformfv(prog, l->varr + 1, f);  EXPECT_NEAR(f[0], E_varr[4], 1e-6);
    EXPECT_NEAR(f[3], E_varr[7], 1e-6);
}

static void render_cases(GLuint prog, const Locs *l)
{
    glBindVertexArray(g_vao);
    for (unsigned i = 0; i < sizeof CASES / sizeof CASES[0]; i++) {
        glUniform1i(l->sel, CASES[i].sel);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        int expect = (int)(CASES[i].x * 255.0f + 0.5f);
        EXPECT_PIXEL(W / 2, H / 2, expect, 64, 128, 255, 2);
    }
    EXPECT_NO_GL_ERROR();
}

/* 1. bound-program mixed packing: forward, reverse, and vec4-only rewrites */
TEST(default_block_mixed_packing)
{
    setup_geom(); reset_state();
    GLuint prog = mgl_make_program(VS, FS);
    EXPECT_TRUE(prog != 0);
    if (!prog) return;
    Locs l; get_locs(prog, &l);
    glUseProgram(prog);

    set_all_gl(prog, &l, 0);
    EXPECT_NO_GL_ERROR();
    check_readback(prog, &l);
    render_cases(prog, &l);

    set_all_gl(prog, &l, 1);   /* reverse order */
    EXPECT_NO_GL_ERROR();
    check_readback(prog, &l);
    render_cases(prog, &l);

    /* re-write only the two vec4s (and one float): all neighbours must survive */
    glUniform4fv(l.a0, 1, E_a0);
    glUniform4fv(l.a1, 1, E_a1);
    glUniform1f(l.s2, E_s2);
    EXPECT_NO_GL_ERROR();
    check_readback(prog, &l);
    render_cases(prog, &l);

    glDeleteProgram(prog);
}

/* 1b. same sandwich pattern in the VERTEX stage */
TEST(default_block_mixed_packing_vs)
{
    setup_geom(); reset_state();
    GLuint prog = mgl_make_program(VS_UNI, FS_OUT);
    EXPECT_TRUE(prog != 0);
    if (!prog) return;

    GLint a0 = glGetUniformLocation(prog, "a0"), s1 = glGetUniformLocation(prog, "s1");
    GLint a1 = glGetUniformLocation(prog, "a1"), s2 = glGetUniformLocation(prog, "s2");
    GLint t2 = glGetUniformLocation(prog, "t2");
    EXPECT_TRUE(a0 >= 0 && s1 >= 0 && a1 >= 0 && s2 >= 0 && t2 >= 0);

    const float want = (E_a0[0] + E_s1 + E_a1[3] + E_s2 + E_t2[1]) * 0.5f;
    glBindVertexArray(g_vao);
    glUseProgram(prog);
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 0) {   /* forward */
            glUniform4fv(a0, 1, E_a0); glUniform1f(s1, E_s1);
            glUniform4fv(a1, 1, E_a1); glUniform1f(s2, E_s2); glUniform3fv(t2, 1, E_t2);
        } else {           /* reverse */
            glUniform3fv(t2, 1, E_t2); glUniform1f(s2, E_s2);
            glUniform4fv(a1, 1, E_a1); glUniform1f(s1, E_s1); glUniform4fv(a0, 1, E_a0);
        }
        EXPECT_NO_GL_ERROR();
        float f[4];
        glGetUniformfv(prog, s1, f); EXPECT_NEAR(f[0], E_s1, 1e-6);
        glGetUniformfv(prog, s2, f); EXPECT_NEAR(f[0], E_s2, 1e-6);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        EXPECT_PIXEL(W / 2, H / 2, (int)(want * 255.0f + 0.5f), 64, 128, 255, 2);
        EXPECT_NO_GL_ERROR();
    }
    glDeleteProgram(prog);
}

/* 1c. the same mixed uniforms consumed by BOTH stages */
TEST(default_block_shared_across_stages)
{
    setup_geom(); reset_state();
    GLuint prog = mgl_make_program(VS_SHARED, FS);
    EXPECT_TRUE(prog != 0);
    if (!prog) return;
    Locs l; get_locs(prog, &l);
    glUseProgram(prog);

    set_all_gl(prog, &l, 0);
    EXPECT_NO_GL_ERROR();
    check_readback(prog, &l);
    render_cases(prog, &l);        /* FS-side view */

    /* VS-side view of the shared uniforms */
    const float want = (E_a0[1] + E_s1 * 0.5f + E_a1[1] + E_s2 * 0.25f);
    GLuint pvs = mgl_make_program(VS_SHARED, FS_OUT);
    EXPECT_TRUE(pvs != 0);
    if (pvs) {
        glUseProgram(pvs);
        GLint a0 = glGetUniformLocation(pvs, "a0"), s1 = glGetUniformLocation(pvs, "s1");
        GLint a1 = glGetUniformLocation(pvs, "a1"), s2 = glGetUniformLocation(pvs, "s2");
        glUniform4fv(a0, 1, E_a0); glUniform1f(s1, E_s1);
        glUniform4fv(a1, 1, E_a1); glUniform1f(s2, E_s2);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(g_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        EXPECT_PIXEL(W / 2, H / 2, (int)(want * 255.0f + 0.5f), 64, 128, 255, 2);
        EXPECT_NO_GL_ERROR();
        glDeleteProgram(pvs);
    }
    glDeleteProgram(prog);
}

/* 2. glProgramUniform* on a program that is NOT the current program */
TEST(program_uniform_not_bound)
{
    setup_geom(); reset_state();
    GLuint prog = mgl_make_program(VS, FS);          /* target: stays unbound */
    GLuint other = mgl_make_program(
        "#version 460 core\nlayout(location=0) in vec3 pos;\nvoid main(){gl_Position=vec4(pos,1.0);}\n",
        "#version 460 core\nlayout(location=0) out vec4 c;\nuniform vec4 oc;\nvoid main(){c=oc;}\n");
    EXPECT_TRUE(prog && other);
    if (!prog || !other) return;
    Locs l; get_locs(prog, &l);

    glUseProgram(other);   /* 'prog' is linked but NOT current */
    glUniform4f(glGetUniformLocation(other, "oc"), 0, 1, 0, 1);

    set_all_program(prog, &l);
    EXPECT_NO_GL_ERROR();
    check_readback(prog, &l);       /* readback through the still-unbound program */
    EXPECT_NO_GL_ERROR();

    /* bind and render: the glProgramUniform values must arrive on the GPU */
    glUseProgram(prog);
    render_cases(prog, &l);

    /* overwrite singles from outside again, re-render */
    glUseProgram(other);
    glProgramUniform1f(prog, l.s1, E_s1);
    glProgramUniform1f(prog, l.s2, E_s2);
    glProgramUniform4f(prog, l.a0, E_a0[0], E_a0[1], E_a0[2], E_a0[3]);
    EXPECT_NO_GL_ERROR();
    glUseProgram(prog);
    render_cases(prog, &l);

    /* negative: location outside the program still reports INVALID_OPERATION */
    glUseProgram(other);
    glProgramUniform1f(prog, 9999, 1.0f);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUseProgram(prog);

    glDeleteProgram(prog);
    glDeleteProgram(other);
}

int main(void)
{
    RUN(default_block_mixed_packing);
    RUN(default_block_mixed_packing_vs);
    RUN(default_block_shared_across_stages);
    RUN(program_uniform_not_bound);
    return mgl_test_summary();
}
