/* test_rstate.c — render state ('rstate' domain) headless tests */
#include "mgl_test.h"

/* rstate helpers exported by libmgl (MGL/src/mgl_ext_rstate.h) */
typedef struct GLMContextRec_t *GLMContext;
extern GLMContext MGLgetCurrentContext(void);
extern _Bool mgl_rstate_index_rewrite_needed(GLMContext ctx, GLenum mode, GLenum type, const void *indices, GLsizei count);
extern void *mgl_rstate_rewrite_indices(GLMContext ctx, GLenum mode, GLenum type, const void *indices,
                                        GLsizei count, GLenum *out_mode, GLenum *out_type, GLsizei *out_count);
extern _Bool mgl_rstate_clamp_read_color(GLMContext ctx, _Bool read_fixed_point);

#define W 100
#define H 100

static GLuint g_prog, g_vao, g_vbo;

static const char *VS =
    "#version 450 core\n"
    "layout(location=0) in vec3 pos;\n"
    "layout(location=1) in vec4 col;\n"
    "layout(location=0) out vec4 vcol;\n"
    "void main(){ vcol = col; gl_Position = vec4(pos, 1.0); }\n";
static const char *FS =
    "#version 450 core\n"
    "layout(location=0) in vec4 vcol;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vcol; }\n";

static void setup(void)
{
    if (g_prog) return;
    g_prog = mgl_make_program(VS, FS);
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * 64, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
}

/* draw triangles from (x,y,z) vertices with one color */
static void draw_tris(const float *xyz, int nverts, float r, float g, float b, float a)
{
    float buf[7 * 64];
    for (int i = 0; i < nverts; i++) {
        buf[7*i+0] = xyz[3*i+0]; buf[7*i+1] = xyz[3*i+1]; buf[7*i+2] = xyz[3*i+2];
        buf[7*i+3] = r; buf[7*i+4] = g; buf[7*i+5] = b; buf[7*i+6] = a;
    }
    glUseProgram(g_prog);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * nverts, buf, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, nverts);
}

/* axis-aligned rect in NDC at depth z, CCW */
static void draw_rect(float x0, float y0, float x1, float y1, float z, float r, float g, float b, float a)
{
    float v[] = { x0,y0,z, x1,y0,z, x1,y1,z,  x0,y0,z, x1,y1,z, x0,y1,z };
    draw_tris(v, 6, r, g, b, a);
}

static void full(float z, float r, float g, float b, float a) { draw_rect(-1, -1, 1, 1, z, r, g, b, a); }

static void clear_black(void)
{
    glDisable(GL_SCISSOR_TEST);
    glColorMask(1, 1, 1, 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void reset_state(void)
{
    glDisable(GL_BLEND); glDisable(GL_CULL_FACE); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST);
    glDisable(GL_RASTERIZER_DISCARD); glDisable(GL_DEPTH_CLAMP); glDisable(GL_POLYGON_OFFSET_FILL);
    glColorMask(1, 1, 1, 1); glDepthMask(GL_TRUE); glDepthFunc(GL_LESS); glDepthRange(0, 1);
    glFrontFace(GL_CCW); glCullFace(GL_BACK);
    glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
    glBlendFunc(GL_ONE, GL_ZERO); glBlendEquation(GL_FUNC_ADD);
    glViewport(0, 0, W, H); glScissor(0, 0, W, H);
    while (glGetError() != GL_NO_ERROR) {}
}

/* ---------------------------------------------------------------- orientation */

TEST(orientation_lower_left)
{
    setup(); reset_state(); clear_black();
    /* NDC lower half -> GL window rows 0..49 */
    draw_rect(-1, -1, 1, 0, 0, 1, 0, 0, 1);
    EXPECT_PIXEL(50, 10, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(50, 90, 0, 0, 0, 255, 2);
    /* NDC left half -> x 0..49 */
    clear_black();
    draw_rect(-1, -1, 0, 1, 0, 0, 1, 0, 1);
    EXPECT_PIXEL(10, 50, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(90, 50, 0, 0, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

TEST(clip_control_upper_left)
{
    setup(); reset_state(); clear_black();
    glClipControl(GL_UPPER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
    EXPECT_NO_GL_ERROR();
    /* UPPER_LEFT negates NDC y: the NDC lower half lands at the top of the window */
    draw_rect(-1, -1, 1, 0, 0, 1, 0, 0, 1);
    EXPECT_PIXEL(50, 90, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(50, 10, 0, 0, 0, 255, 2);
    reset_state();
}

TEST(clip_control_depth_mode)
{
    setup(); reset_state(); clear_black();
    /* NEGATIVE_ONE_TO_ONE keeps z in [-1,1]: z=-0.5 is visible */
    draw_rect(-1, -1, 0, 1, -0.5f, 1, 0, 0, 1);
    EXPECT_PIXEL(25, 50, 255, 0, 0, 255, 2);
    /* ZERO_TO_ONE clips z < 0 */
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    draw_rect(0, -1, 1, 1, -0.5f, 0, 1, 0, 1);
    EXPECT_PIXEL(75, 50, 0, 0, 0, 255, 2);
    draw_rect(0, -1, 1, 1, 0.5f, 0, 1, 0, 1);
    EXPECT_PIXEL(75, 50, 0, 255, 0, 255, 2);

    /* depth mapping: z=0 in N1T1 -> window 0.5; z=0.45 in ZERO_TO_ONE -> 0.45 (passes LESS),
     * the same z in N1T1 -> 0.725 (fails) */
    reset_state(); clear_black();
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    full(0.0f, 1, 0, 0, 1);
    draw_rect(-1, -1, 0, 1, 0.45f, 0, 0, 1, 1);          /* N1T1: fails */
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    draw_rect(0, -1, 1, 1, 0.45f, 0, 1, 0, 1);           /* Z01: passes */
    EXPECT_PIXEL(25, 50, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(75, 50, 0, 255, 0, 255, 2);
    GLint v[2] = { 0, 0 };
    glGetIntegerv(GL_CLIP_ORIGIN, &v[0]);
    glGetIntegerv(GL_CLIP_DEPTH_MODE, &v[1]);
    EXPECT_EQ_INT(v[0], GL_LOWER_LEFT);
    EXPECT_EQ_INT(v[1], GL_ZERO_TO_ONE);
    glClipControl(GL_LOWER_LEFT + 7, GL_ZERO_TO_ONE);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glClipControl(GL_LOWER_LEFT, GL_ZERO);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    reset_state();
}

TEST(blend_modes)
{
    setup(); reset_state();
    glClearColor(0.25f, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    full(0, 0.5f, 0, 0, 0);
    EXPECT_PIXEL(50, 50, 191, 0, 0, 255, 3);

    /* constant color */
    clear_black();
    glBlendColor(0.0f, 0.5f, 0.25f, 1.0f);
    glBlendFunc(GL_CONSTANT_COLOR, GL_ZERO);
    full(0, 1, 1, 1, 1);
    EXPECT_PIXEL(50, 50, 0, 128, 64, 255, 3);

    /* reverse subtract: dst - src */
    glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
    full(0, 0.25f, 0.5f, 0.75f, 0);
    EXPECT_PIXEL(50, 50, 191, 128, 64, 255, 3);

    /* separate alpha: rgb = src*alpha + dst*(1-alpha), alpha = zero*src + one*dst */
    glClearColor(0, 0, 1, 0.5f); glClear(GL_COLOR_BUFFER_BIT);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    full(0, 1, 0, 0, 0.5f);
    EXPECT_PIXEL(50, 50, 128, 0, 128, 128, 3);

    /* MIN on rgb */
    glClearColor(0.5f, 0.5f, 0.5f, 1); glClear(GL_COLOR_BUFFER_BIT);
    glBlendEquationSeparate(GL_MIN, GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ZERO);
    full(0, 1, 0.25f, 0.75f, 1);
    EXPECT_PIXEL(50, 50, 128, 64, 128, 255, 3);

    /* disabling blend mid-pass replaces */
    glDisable(GL_BLEND);
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 255, 0, 255, 2);

    GLint i = 0;
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
    glGetIntegerv(GL_BLEND_SRC_RGB, &i);   EXPECT_EQ_INT(i, GL_SRC_ALPHA);
    glGetIntegerv(GL_BLEND_DST_RGB, &i);   EXPECT_EQ_INT(i, GL_ONE_MINUS_SRC_ALPHA);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &i); EXPECT_EQ_INT(i, GL_ONE);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &i); EXPECT_EQ_INT(i, GL_ZERO);
    glBlendEquationSeparate(GL_FUNC_SUBTRACT, GL_MAX);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &i);   EXPECT_EQ_INT(i, GL_FUNC_SUBTRACT);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &i); EXPECT_EQ_INT(i, GL_MAX);
    GLfloat bc[4];
    glGetFloatv(GL_BLEND_COLOR, bc);
    EXPECT_NEAR(bc[1], 0.5, 1e-6);
    EXPECT_NEAR(bc[2], 0.25, 1e-6);
    glBlendFunc(GL_ONE, GL_ZERO + 2);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBlendEquation(GL_ZERO);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBlendFunci(8, GL_ONE, GL_ONE);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBlendEquationi(100, GL_FUNC_ADD);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    reset_state();
}

TEST(blend_indexed_state)
{
    reset_state();
    glEnablei(GL_BLEND, 2);
    EXPECT_TRUE(glIsEnabledi(GL_BLEND, 2));
    EXPECT_TRUE(!glIsEnabledi(GL_BLEND, 1));
    EXPECT_TRUE(!glIsEnabled(GL_BLEND));
    glBlendFuncSeparatei(2, GL_DST_COLOR, GL_SRC_COLOR, GL_ONE, GL_ONE);
    glBlendEquationSeparatei(2, GL_FUNC_SUBTRACT, GL_MIN);
    GLint v = 0;
    glGetIntegeri_v(GL_BLEND_SRC_RGB, 2, &v);   EXPECT_EQ_INT(v, GL_DST_COLOR);
    glGetIntegeri_v(GL_BLEND_DST_RGB, 2, &v);   EXPECT_EQ_INT(v, GL_SRC_COLOR);
    glGetIntegeri_v(GL_BLEND_EQUATION_RGB, 2, &v);   EXPECT_EQ_INT(v, GL_FUNC_SUBTRACT);
    glGetIntegeri_v(GL_BLEND_EQUATION_ALPHA, 2, &v); EXPECT_EQ_INT(v, GL_MIN);
    glGetIntegeri_v(GL_BLEND_SRC_RGB, 0, &v);   EXPECT_EQ_INT(v, GL_ONE);
    GLboolean b = GL_FALSE;
    glGetBooleani_v(GL_BLEND, 2, &b); EXPECT_EQ_INT(b, GL_TRUE);
    glColorMaski(3, GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
    GLboolean m[4];
    glGetBooleani_v(GL_COLOR_WRITEMASK, 3, m);
    EXPECT_EQ_INT(m[0], 1); EXPECT_EQ_INT(m[1], 0); EXPECT_EQ_INT(m[2], 1); EXPECT_EQ_INT(m[3], 0);
    glGetBooleani_v(GL_COLOR_WRITEMASK, 0, m);
    EXPECT_EQ_INT(m[1], 1);
    glEnable(GL_BLEND);
    EXPECT_TRUE(glIsEnabledi(GL_BLEND, 7));
    glDisable(GL_BLEND);
    EXPECT_TRUE(!glIsEnabledi(GL_BLEND, 2));
    glEnablei(GL_BLEND, 8);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glEnablei(GL_DEPTH_TEST, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glColorMaski(8, 1, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    reset_state();
}

/* two color attachments, blending only attachment 0, color mask only on attachment 1 */
TEST(blend_per_attachment)
{
    static const char *fs2 =
        "#version 450 core\n"
        "layout(location=0) in vec4 vcol;\n"
        "layout(location=0) out vec4 c0;\n"
        "layout(location=1) out vec4 c1;\n"
        "void main(){ c0 = vcol; c1 = vcol; }\n";
    setup(); reset_state();
    GLuint prog = mgl_make_program(VS, fs2);
    if (!prog) return;
    GLuint fbo, tex[2];
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
    }
    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, bufs);
    EXPECT_TRUE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 32, 32);
    glClearColor(0.5f, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glEnablei(GL_BLEND, 0);
    glBlendFunci(0, GL_ONE, GL_ONE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE);
    GLuint saved = g_prog; g_prog = prog;
    full(0, 0.25f, 0.5f, 1.0f, 1);
    g_prog = saved;
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_PIXEL(16, 16, 191, 128, 255, 255, 3);       /* blended */
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_PIXEL(16, 16, 64, 128, 0, 255, 3);          /* replaced, blue masked off */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadBuffer(GL_FRONT);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(2, tex);
    EXPECT_NO_GL_ERROR();
    reset_state();
}

TEST(color_mask)
{
    setup(); reset_state(); clear_black();
    glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_TRUE);
    full(0, 1, 1, 1, 1);
    EXPECT_PIXEL(50, 50, 255, 0, 255, 255, 2);
    GLboolean m[4];
    glGetBooleanv(GL_COLOR_WRITEMASK, m);
    EXPECT_EQ_INT(m[0], 1); EXPECT_EQ_INT(m[1], 0); EXPECT_EQ_INT(m[2], 1); EXPECT_EQ_INT(m[3], 1);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 50, 255, 0, 255, 255, 2);
    reset_state();
}

TEST(polygon_offset)
{
    setup(); reset_state(); clear_black();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    full(0.2f, 1, 0, 0, 1);
    /* same depth fails LESS */
    draw_rect(-1, -1, 0, 1, 0.2f, 0, 0, 1, 1);
    EXPECT_PIXEL(25, 50, 255, 0, 0, 255, 2);
    /* pulled towards the viewer by the offset */
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(0.0f, -4.0f);
    draw_rect(0, -1, 1, 1, 0.2f, 0, 1, 0, 1);
    EXPECT_PIXEL(75, 50, 0, 255, 0, 255, 2);
    /* pushed away: fails even though slightly nearer */
    glPolygonOffset(0.0f, 64.0f);
    draw_rect(-1, -1, 0, 1, 0.1999995f, 0, 0, 1, 1);
    EXPECT_PIXEL(25, 50, 255, 0, 0, 255, 2);
    GLfloat f = 0;
    glPolygonOffsetClamp(1.5f, 2.5f, 0.125f);
    glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &f); EXPECT_NEAR(f, 1.5, 1e-6);
    glGetFloatv(GL_POLYGON_OFFSET_UNITS, &f);  EXPECT_NEAR(f, 2.5, 1e-6);
    glGetFloatv(GL_POLYGON_OFFSET_CLAMP, &f);  EXPECT_NEAR(f, 0.125, 1e-6);
    EXPECT_TRUE(glIsEnabled(GL_POLYGON_OFFSET_FILL));
    EXPECT_NO_GL_ERROR();
    reset_state();
}

TEST(viewport_scissor)
{
    setup(); reset_state(); clear_black();
    glViewport(50, 0, 50, 50);                 /* bottom-right quadrant */
    full(0, 1, 0, 0, 1);
    EXPECT_PIXEL(75, 25, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(25, 25, 0, 0, 0, 255, 2);
    EXPECT_PIXEL(75, 75, 0, 0, 0, 255, 2);
    glViewport(0, 0, W, H);
    clear_black();
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 50, 50, 50);                  /* top-left quadrant */
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(25, 75, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(75, 75, 0, 0, 0, 255, 2);
    EXPECT_PIXEL(25, 25, 0, 0, 0, 255, 2);
    /* mid-pass disable must reach the live encoder */
    glDisable(GL_SCISSOR_TEST);
    full(0, 0, 0, 1, 1);
    EXPECT_PIXEL(75, 25, 0, 0, 255, 255, 2);
    /* scissor partially outside the target is clamped */
    glEnable(GL_SCISSOR_TEST);
    glScissor(-20, 90, 500, 500);
    full(0, 1, 1, 0, 1);
    EXPECT_PIXEL(5, 95, 255, 255, 0, 255, 2);
    EXPECT_PIXEL(5, 85, 0, 0, 255, 255, 2);
    GLint sb[4];
    glGetIntegerv(GL_SCISSOR_BOX, sb);
    EXPECT_EQ_INT(sb[0], -20); EXPECT_EQ_INT(sb[1], 90); EXPECT_EQ_INT(sb[2], 500);
    GLint vp[4];
    glViewport(-5, 3, 40, 41);
    glGetIntegerv(GL_VIEWPORT, vp);
    EXPECT_EQ_INT(vp[0], -5); EXPECT_EQ_INT(vp[1], 3); EXPECT_EQ_INT(vp[2], 40); EXPECT_EQ_INT(vp[3], 41);
    glViewport(0, 0, -1, 10);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glScissor(0, 0, 10, -1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    reset_state();
}

TEST(viewport_scissor_arrays)
{
    static const char *vs_vp =
        "#version 450 core\n"
        "#extension GL_ARB_shader_viewport_layer_array : require\n"
        "layout(location=0) in vec3 pos;\n"
        "layout(location=0) out vec4 vcol;\n"
        "void main(){\n"
        "  vec4 c[4] = vec4[4](vec4(1,0,0,1), vec4(0,1,0,1), vec4(0,0,1,1), vec4(1,1,1,1));\n"
        "  vcol = c[gl_InstanceID];\n"
        "  gl_ViewportIndex = gl_InstanceID;\n"
        "  gl_Position = vec4(pos, 1.0);\n"
        "}\n";
    setup(); reset_state(); clear_black();
    GLuint prog = mgl_make_program(vs_vp, FS);
    if (!prog) return;
    /* quadrants: 0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right */
    const GLfloat vps[16] = { 0,0,50,50,  50,0,50,50,  0,50,50,50,  50,50,50,50 };
    glViewportArrayv(0, 4, vps);
    /* scissor quadrant 3 to its left half */
    glScissorIndexed(3, 50, 50, 25, 50);
    glEnablei(GL_SCISSOR_TEST, 3);
    float v[] = { -1,-1,0, 1,-1,0, 1,1,0,  -1,-1,0, 1,1,0, -1,1,0 };
    float buf[7 * 6];
    for (int i = 0; i < 6; i++) { buf[7*i] = v[3*i]; buf[7*i+1] = v[3*i+1]; buf[7*i+2] = v[3*i+2];
                                  buf[7*i+3] = buf[7*i+4] = buf[7*i+5] = buf[7*i+6] = 1; }
    glUseProgram(prog);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof buf, buf, GL_DYNAMIC_DRAW);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 4);
    EXPECT_PIXEL(25, 25, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(75, 25, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(25, 75, 0, 0, 255, 255, 2);
    EXPECT_PIXEL(60, 75, 255, 255, 255, 255, 2);
    EXPECT_PIXEL(90, 75, 0, 0, 0, 255, 2);           /* scissored */

    GLfloat f[4]; GLint iv[4]; GLdouble d[2];
    glGetFloati_v(GL_VIEWPORT, 2, f);
    EXPECT_NEAR(f[0], 0, 0); EXPECT_NEAR(f[1], 50, 0); EXPECT_NEAR(f[2], 50, 0);
    glViewportIndexedf(5, 1.5f, 2.5f, 10.0f, 20.0f);
    glGetFloati_v(GL_VIEWPORT, 5, f);
    EXPECT_NEAR(f[0], 1.5, 1e-6); EXPECT_NEAR(f[3], 20, 1e-6);
    const GLfloat vf[4] = { 4, 5, 6, 7 };
    glViewportIndexedfv(6, vf);
    glGetIntegeri_v(GL_VIEWPORT, 6, iv); EXPECT_EQ_INT(iv[2], 6);
    glGetIntegeri_v(GL_SCISSOR_BOX, 3, iv);
    EXPECT_EQ_INT(iv[0], 50); EXPECT_EQ_INT(iv[2], 25);
    const GLint sv[4] = { 1, 2, 3, 4 };
    glScissorIndexedv(7, sv);
    glScissorArrayv(8, 1, sv);
    glGetIntegeri_v(GL_SCISSOR_BOX, 8, iv); EXPECT_EQ_INT(iv[3], 4);
    EXPECT_TRUE(glIsEnabledi(GL_SCISSOR_TEST, 3));
    EXPECT_TRUE(!glIsEnabledi(GL_SCISSOR_TEST, 0));
    glDepthRangeIndexed(4, 0.25, 0.75);
    const GLdouble dr[4] = { 0.1, 0.2, -1.0, 2.0 };
    glDepthRangeArrayv(9, 2, dr);
    glGetDoublei_v(GL_DEPTH_RANGE, 4, d); EXPECT_NEAR(d[0], 0.25, 1e-9); EXPECT_NEAR(d[1], 0.75, 1e-9);
    glGetDoublei_v(GL_DEPTH_RANGE, 10, d); EXPECT_NEAR(d[0], 0.0, 1e-9); EXPECT_NEAR(d[1], 1.0, 1e-9);
    glDepthRangef(0.125f, 0.5f);
    glGetDoublei_v(GL_DEPTH_RANGE, 15, d); EXPECT_NEAR(d[0], 0.125, 1e-7);
    GLint maxvp = 0;
    glGetIntegerv(GL_MAX_VIEWPORTS, &maxvp);
    EXPECT_EQ_INT(maxvp, 16);
    EXPECT_NO_GL_ERROR();
    glViewportIndexedf(16, 0, 0, 1, 1);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glViewportArrayv(15, 2, vps);               EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glViewportIndexedf(0, 0, 0, -1, 1);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glScissorIndexed(16, 0, 0, 1, 1);           EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glScissorIndexed(0, 0, 0, -1, 1);           EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glScissorArrayv(0, -1, sv);                 EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDepthRangeIndexed(16, 0, 1);              EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glEnablei(GL_SCISSOR_TEST, 16);             EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetFloati_v(GL_VIEWPORT, 16, f);          EXPECT_GL_ERROR(GL_INVALID_VALUE);
    reset_state();
}

TEST(cull_front_face)
{
    setup(); reset_state();
    glEnable(GL_CULL_FACE);
    clear_black(); glCullFace(GL_BACK);
    full(0, 1, 0, 0, 1);                       /* CCW == front: drawn */
    EXPECT_PIXEL(50, 50, 255, 0, 0, 255, 2);
    clear_black(); glCullFace(GL_FRONT);
    full(0, 1, 0, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    clear_black(); glFrontFace(GL_CW);          /* CCW now back, culling front: drawn */
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 255, 0, 255, 2);
    clear_black(); glCullFace(GL_BACK);
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    /* ARB_clip_control: with UPPER_LEFT the window-space area sign is reversed, so facing
     * stays the NDC winding (CCW front stays front) */
    clear_black(); glFrontFace(GL_CCW); glCullFace(GL_BACK);
    glClipControl(GL_UPPER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
    full(0, 0, 0, 1, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 255, 255, 2);
    clear_black(); glCullFace(GL_FRONT);
    full(0, 0, 0, 1, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    glCullFace(GL_BACK);
    glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
    clear_black(); glCullFace(GL_FRONT_AND_BACK);
    full(0, 1, 1, 1, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    glDisable(GL_CULL_FACE);
    full(0, 1, 1, 1, 1);
    EXPECT_PIXEL(50, 50, 255, 255, 255, 255, 2);
    GLint v;
    glGetIntegerv(GL_CULL_FACE_MODE, &v); EXPECT_EQ_INT(v, GL_FRONT_AND_BACK);
    glFrontFace(GL_CW); glGetIntegerv(GL_FRONT_FACE, &v); EXPECT_EQ_INT(v, GL_CW);
    glCullFace(GL_LEFT);  EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glFrontFace(GL_LEFT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    reset_state();
}

TEST(rasterizer_discard)
{
    setup(); reset_state(); clear_black();
    glEnable(GL_RASTERIZER_DISCARD);
    EXPECT_TRUE(glIsEnabled(GL_RASTERIZER_DISCARD));
    full(0, 1, 0, 0, 1);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

TEST(depth_clamp)
{
    setup(); reset_state(); clear_black();
    /* beyond the far plane: clipped */
    full(1.5f, 1, 0, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    glEnable(GL_DEPTH_CLAMP);
    full(1.5f, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 255, 0, 255, 2);
    /* clamped depth == far: fails LESS against a cleared 1.0, passes LEQUAL */
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    full(1.5f, 1, 0, 0, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 2);
    glDepthFunc(GL_LEQUAL);
    full(1.5f, 0, 0, 1, 1);
    EXPECT_PIXEL(50, 50, 0, 0, 255, 255, 2);
    EXPECT_TRUE(glIsEnabled(GL_DEPTH_CLAMP));
    reset_state();
}

TEST(depth_func_mask_range)
{
    setup(); reset_state(); clear_black();
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    full(0.0f, 1, 0, 0, 1);                             /* depth 0.5 */
    glDepthFunc(GL_GREATER);
    draw_rect(-1, -1, 0, 1, 0.5f, 0, 1, 0, 1);         /* 0.75: passes GREATER */
    EXPECT_PIXEL(25, 50, 0, 255, 0, 255, 2);
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LESS);
    draw_rect(-1, -1, 0, 1, 0.25f, 0, 0, 1, 1);        /* 0.625 < 0.75: passes, no write */
    draw_rect(-1, -1, 0, 1, 0.4f, 1, 1, 1, 1);         /* 0.7 < 0.75 still: passes */
    EXPECT_PIXEL(25, 50, 255, 255, 255, 255, 2);
    /* depth range squeezes z into [0.9, 1] */
    glDepthMask(GL_TRUE);
    glDepthRange(0.9, 1.0);
    draw_rect(0, -1, 1, 1, -1.0f, 0, 0, 1, 1);         /* window z 0.9 > 0.5: fails */
    EXPECT_PIXEL(75, 50, 255, 0, 0, 255, 2);
    GLboolean b; glGetBooleanv(GL_DEPTH_WRITEMASK, &b); EXPECT_EQ_INT(b, 1);
    GLint f; glGetIntegerv(GL_DEPTH_FUNC, &f); EXPECT_EQ_INT(f, GL_LESS);
    GLdouble dr[2]; glGetDoublev(GL_DEPTH_RANGE, dr); EXPECT_NEAR(dr[0], 0.9, 1e-6);
    glDepthFunc(GL_ONE); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    reset_state();
}

TEST(frag_coord_and_derivatives)
{
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(gl_FragCoord.y / 100.0, dFdy(gl_FragCoord.y) > 0.0 ? 1.0 : 0.0,\n"
        "                          gl_FragCoord.x / 100.0, 1.0); }\n";
    setup(); reset_state(); clear_black();
    GLuint prog = mgl_make_program(VS, fs);
    if (!prog) return;
    GLuint saved = g_prog; g_prog = prog;
    full(0, 0, 0, 0, 1);
    g_prog = saved;
    EXPECT_PIXEL(30, 10, 27, 255, 78, 255, 2);    /* (10.5/100)*255, dFdy > 0, (30.5/100)*255 */
    EXPECT_PIXEL(30, 90, 232, 255, 78, 255, 2);
}

TEST(point_size_and_coord)
{
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec3 pos;\n"
        "void main(){ gl_Position = vec4(pos, 1.0); }\n";
    static const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(gl_PointCoord.x, gl_PointCoord.y, 0.0, 1.0); }\n";
    setup(); reset_state(); clear_black();
    GLuint prog = mgl_make_program(vs, fs);
    if (!prog) return;
    glPointSize(20.0f);
    float pt[7] = { 0, 0, 0, 1, 1, 1, 1 };
    glUseProgram(prog);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pt, pt, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, 1);
    /* point covers [40,60): default origin UPPER_LEFT -> t grows downwards */
    GLubyte top[4], bot[4];
    mgl_read_rgba(50, 58, top);
    mgl_read_rgba(50, 41, bot);
    EXPECT_TRUE(top[1] < 64 && bot[1] > 192);
    EXPECT_PIXEL(30, 50, 0, 0, 0, 255, 2);
    clear_black();
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
    glDrawArrays(GL_POINTS, 0, 1);
    mgl_read_rgba(50, 58, top);
    mgl_read_rgba(50, 41, bot);
    EXPECT_TRUE(top[1] > 192 && bot[1] < 64);
    GLfloat f; GLint i;
    glGetFloatv(GL_POINT_SIZE, &f); EXPECT_NEAR(f, 20, 0);
    glGetIntegerv(GL_POINT_SPRITE_COORD_ORIGIN, &i); EXPECT_EQ_INT(i, GL_LOWER_LEFT);
    glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 3.0f);
    glGetFloatv(GL_POINT_FADE_THRESHOLD_SIZE, &f); EXPECT_NEAR(f, 3, 0);
    GLfloat fv = 2.0f; glPointParameterfv(GL_POINT_FADE_THRESHOLD_SIZE, &fv);
    GLint iv = GL_UPPER_LEFT; glPointParameteriv(GL_POINT_SPRITE_COORD_ORIGIN, &iv);
    glGetIntegerv(GL_POINT_SPRITE_COORD_ORIGIN, &i); EXPECT_EQ_INT(i, GL_UPPER_LEFT);
    EXPECT_NO_GL_ERROR();
    glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, -1.0f); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_FRONT); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPointParameteri(GL_POINT_SIZE, 1);                       EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glPointSize(0.0f);                                         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPointSize(1.0f);
    reset_state();
}

TEST(polygon_mode_line)
{
    setup(); reset_state(); clear_black();
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    draw_rect(-0.5f, -0.5f, 0.5f, 0.5f, 0, 1, 1, 1, 1);
    EXPECT_PIXEL(50, 60, 0, 0, 0, 255, 2);           /* interior not filled */
    GLint m[2]; glGetIntegerv(GL_POLYGON_MODE, m); EXPECT_EQ_INT(m[0], GL_LINE);
    glPolygonMode(GL_FRONT, GL_LINE);  EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FRONT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    full(0, 0, 1, 0, 1);
    EXPECT_PIXEL(50, 60, 0, 255, 0, 255, 2);
    reset_state();
}

TEST(primitive_restart_rewrite)
{
    GLMContext ctx = MGLgetCurrentContext();
    GLenum om, ot; GLsizei oc;
    reset_state();
    glPrimitiveRestartIndex(7);
    glEnable(GL_PRIMITIVE_RESTART);
    GLint ri = 0; glGetIntegerv(GL_PRIMITIVE_RESTART_INDEX, &ri); EXPECT_EQ_INT(ri, 7);
    EXPECT_TRUE(glIsEnabled(GL_PRIMITIVE_RESTART));

    /* strip with custom restart index -> Metal 0xFFFF */
    const GLushort s[] = { 0, 1, 2, 3, 7, 4, 5, 6 };
    EXPECT_TRUE(mgl_rstate_index_rewrite_needed(ctx, GL_TRIANGLE_STRIP, GL_UNSIGNED_SHORT, s, 8));
    GLushort *o = mgl_rstate_rewrite_indices(ctx, GL_TRIANGLE_STRIP, GL_UNSIGNED_SHORT, s, 8, &om, &ot, &oc);
    EXPECT_TRUE(o != NULL);
    EXPECT_EQ_INT(om, GL_TRIANGLE_STRIP); EXPECT_EQ_INT(ot, GL_UNSIGNED_SHORT); EXPECT_EQ_INT(oc, 8);
    if (o) { EXPECT_EQ_INT(o[4], 0xFFFF); EXPECT_EQ_INT(o[5], 4); free(o); }

    /* triangle list: restart drops the partial triangle */
    const GLuint t[] = { 0, 1, 2, 3, 4, 7, 5, 6, 8 };
    o = mgl_rstate_rewrite_indices(ctx, GL_TRIANGLES, GL_UNSIGNED_INT, t, 9, &om, &ot, &oc);
    EXPECT_EQ_INT(om, GL_TRIANGLES); EXPECT_EQ_INT(oc, 6);
    if (o) { EXPECT_EQ_INT(o[3], 5); EXPECT_EQ_INT(o[5], 8); free(o); }

    /* restart disabled: 0xFFFF in a strip must not restart -> widened to 32-bit */
    glDisable(GL_PRIMITIVE_RESTART);
    const GLushort s2[] = { 0, 1, 0xFFFF, 2 };
    EXPECT_TRUE(mgl_rstate_index_rewrite_needed(ctx, GL_LINE_STRIP, GL_UNSIGNED_SHORT, s2, 4));
    GLuint *o32 = mgl_rstate_rewrite_indices(ctx, GL_LINE_STRIP, GL_UNSIGNED_SHORT, s2, 4, &om, &ot, &oc);
    EXPECT_EQ_INT(ot, GL_UNSIGNED_INT); EXPECT_EQ_INT(oc, 4);
    if (o32) { EXPECT_EQ_INT(o32[2], 0xFFFF); free(o32); }
    EXPECT_TRUE(!mgl_rstate_index_rewrite_needed(ctx, GL_TRIANGLES, GL_UNSIGNED_SHORT, s2, 4));

    /* fixed index (0xFF for bytes) with a fan -> triangle list */
    glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    const GLubyte b[] = { 0, 1, 2, 3, 0xFF, 4, 5, 6 };
    o = mgl_rstate_rewrite_indices(ctx, GL_TRIANGLE_FAN, GL_UNSIGNED_BYTE, b, 8, &om, &ot, &oc);
    EXPECT_EQ_INT(om, GL_TRIANGLES); EXPECT_EQ_INT(ot, GL_UNSIGNED_SHORT); EXPECT_EQ_INT(oc, 9);
    if (o) { EXPECT_EQ_INT(o[3], 0); EXPECT_EQ_INT(o[4], 2); EXPECT_EQ_INT(o[5], 3);
             EXPECT_EQ_INT(o[6], 4); EXPECT_EQ_INT(o[8], 6); free(o); }
    glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    EXPECT_NO_GL_ERROR();
}

TEST(sample_state)
{
    reset_state();
    GLfloat f; GLint i; GLboolean b;
    glSampleCoverage(0.5f, GL_TRUE);
    glGetFloatv(GL_SAMPLE_COVERAGE_VALUE, &f); EXPECT_NEAR(f, 0.5, 1e-6);
    glGetBooleanv(GL_SAMPLE_COVERAGE_INVERT, &b); EXPECT_EQ_INT(b, GL_TRUE);
    glSampleCoverage(2.0f, GL_FALSE);
    glGetFloatv(GL_SAMPLE_COVERAGE_VALUE, &f); EXPECT_NEAR(f, 1.0, 1e-6);
    glSampleMaski(0, 0x5u);
    glGetIntegeri_v(GL_SAMPLE_MASK_VALUE, 0, &i); EXPECT_EQ_INT(i, 5);
    glSampleMaski(1, 0x5u); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMinSampleShading(0.75f);
    glGetFloatv(GL_MIN_SAMPLE_SHADING_VALUE, &f); EXPECT_NEAR(f, 0.75, 1e-6);
    const GLenum caps[] = { GL_SAMPLE_ALPHA_TO_COVERAGE, GL_SAMPLE_ALPHA_TO_ONE, GL_SAMPLE_COVERAGE,
                            GL_SAMPLE_MASK, GL_SAMPLE_SHADING, GL_MULTISAMPLE };
    for (int k = 0; k < 6; k++) {
        glEnable(caps[k]);  EXPECT_TRUE(glIsEnabled(caps[k]));
        glGetBooleanv(caps[k], &b); EXPECT_EQ_INT(b, GL_TRUE);
        glDisable(caps[k]); EXPECT_TRUE(!glIsEnabled(caps[k]));
    }
    glEnable(GL_MULTISAMPLE);
    EXPECT_NO_GL_ERROR();
}

TEST(misc_state_getters)
{
    GLint i; GLfloat f;
    reset_state();
    glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
    glGetIntegerv(GL_PROVOKING_VERTEX, &i); EXPECT_EQ_INT(i, GL_FIRST_VERTEX_CONVENTION);
    glProvokingVertex(GL_LAST_VERTEX_CONVENTION);
    glProvokingVertex(GL_FRONT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glGetIntegerv(GL_PATCH_VERTICES, &i); EXPECT_EQ_INT(i, 4);
    glPatchParameteri(GL_PATCH_VERTICES, 0); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPatchParameteri(GL_PATCH_VERTICES, 33); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPatchParameteri(GL_PATCH_DEFAULT_INNER_LEVEL, 3); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    const GLfloat outer[4] = { 2, 3, 4, 5 }, inner[2] = { 6, 7 };
    glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
    glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
    GLfloat o4[4], i2[2];
    glGetFloatv(GL_PATCH_DEFAULT_OUTER_LEVEL, o4); EXPECT_NEAR(o4[3], 5, 0);
    glGetFloatv(GL_PATCH_DEFAULT_INNER_LEVEL, i2); EXPECT_NEAR(i2[1], 7, 0);
    glPatchParameterfv(GL_PATCH_VERTICES, outer); EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
    glGetIntegerv(GL_CLAMP_READ_COLOR, &i); EXPECT_EQ_INT(i, GL_FALSE);
    EXPECT_TRUE(!mgl_rstate_clamp_read_color(MGLgetCurrentContext(), 1));
    glClampColor(GL_CLAMP_READ_COLOR, GL_FIXED_ONLY);
    EXPECT_TRUE(mgl_rstate_clamp_read_color(MGLgetCurrentContext(), 1));
    EXPECT_TRUE(!mgl_rstate_clamp_read_color(MGLgetCurrentContext(), 0));
    glClampColor(GL_CLAMP_READ_COLOR, GL_RED); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glClampColor(GL_RED, GL_TRUE);             EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glLineWidth(2.0f);
    glGetFloatv(GL_LINE_WIDTH, &f); EXPECT_NEAR(f, 2.0, 0);
    glLineWidth(0.0f); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glLineWidth(1.0f);

    glLogicOp(GL_XOR);
    glGetIntegerv(GL_LOGIC_OP_MODE, &i); EXPECT_EQ_INT(i, GL_XOR);
    glLogicOp(GL_FRONT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glLogicOp(GL_COPY);
    glEnable(GL_COLOR_LOGIC_OP); EXPECT_TRUE(glIsEnabled(GL_COLOR_LOGIC_OP)); glDisable(GL_COLOR_LOGIC_OP);

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glGetIntegerv(GL_LINE_SMOOTH_HINT, &i); EXPECT_EQ_INT(i, GL_NICEST);
    glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_FASTEST);
    glGetIntegerv(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, &i); EXPECT_EQ_INT(i, GL_FASTEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_LINE); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glHint(GL_POINT_SIZE, GL_NICEST);     EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* stencil front/back */
    glStencilFuncSeparate(GL_BACK, GL_GREATER, 3, 0x0F);
    glStencilOpSeparate(GL_FRONT, GL_ZERO, GL_INCR_WRAP, GL_INVERT);
    glStencilMaskSeparate(GL_BACK, 0x33);
    glGetIntegerv(GL_STENCIL_BACK_FUNC, &i); EXPECT_EQ_INT(i, GL_GREATER);
    glGetIntegerv(GL_STENCIL_BACK_REF, &i); EXPECT_EQ_INT(i, 3);
    glGetIntegerv(GL_STENCIL_BACK_VALUE_MASK, &i); EXPECT_EQ_INT(i, 0x0F);
    glGetIntegerv(GL_STENCIL_FUNC, &i); EXPECT_EQ_INT(i, GL_ALWAYS);
    glGetIntegerv(GL_STENCIL_FAIL, &i); EXPECT_EQ_INT(i, GL_ZERO);
    glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &i); EXPECT_EQ_INT(i, GL_INCR_WRAP);
    glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &i); EXPECT_EQ_INT(i, GL_INVERT);
    glGetIntegerv(GL_STENCIL_BACK_FAIL, &i); EXPECT_EQ_INT(i, GL_KEEP);
    glGetIntegerv(GL_STENCIL_BACK_WRITEMASK, &i); EXPECT_EQ_INT(i, 0x33);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glGetIntegerv(GL_STENCIL_VALUE_MASK, &i); EXPECT_EQ_INT(i, 0xFF);
    glGetIntegerv(GL_STENCIL_BACK_FUNC, &i); EXPECT_EQ_INT(i, GL_EQUAL);
    glStencilMask(0xF0);
    glGetIntegerv(GL_STENCIL_WRITEMASK, &i); EXPECT_EQ_INT(i, 0xF0);
    glStencilFuncSeparate(GL_LEFT, GL_EQUAL, 0, 0); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_ONE); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glStencilMaskSeparate(GL_LEFT, 0); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glStencilFunc(GL_ALWAYS, 0, 0xFFFFFFFF); glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP); glStencilMask(0xFFFFFFFF);

    /* every core enable cap round-trips through Enable / IsEnabled / GetBooleanv */
    const GLenum caps[] = { GL_BLEND, GL_CULL_FACE, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_DITHER, GL_SCISSOR_TEST,
        GL_COLOR_LOGIC_OP, GL_POLYGON_OFFSET_POINT, GL_POLYGON_OFFSET_LINE, GL_POLYGON_OFFSET_FILL,
        GL_RASTERIZER_DISCARD, GL_FRAMEBUFFER_SRGB, GL_PRIMITIVE_RESTART, GL_PRIMITIVE_RESTART_FIXED_INDEX,
        GL_DEPTH_CLAMP, GL_TEXTURE_CUBE_MAP_SEAMLESS, GL_PROGRAM_POINT_SIZE, GL_LINE_SMOOTH, GL_POLYGON_SMOOTH,
        GL_CLIP_DISTANCE0 + 3 };
    for (unsigned k = 0; k < sizeof caps / sizeof caps[0]; k++) {
        GLboolean was = glIsEnabled(caps[k]), b2 = 0;
        glEnable(caps[k]);
        EXPECT_TRUE(glIsEnabled(caps[k]));
        if (caps[k] != GL_CLIP_DISTANCE0 + 3) { glGetBooleanv(caps[k], &b2); EXPECT_EQ_INT(b2, 1); }
        glDisable(caps[k]);
        EXPECT_TRUE(!glIsEnabled(caps[k]));
        if (was) glEnable(caps[k]);
    }
    EXPECT_NO_GL_ERROR();
    glEnable(GL_FRONT);    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDisable(GL_FRONT);   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glIsEnabled(GL_FRONT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    reset_state();
}

int main(int argc, char **argv)
{
    /* optional: run a single case by name */
    const char *only = argc > 1 ? argv[1] : NULL;
#define RUN_IF(n) do { if (!only || !strcmp(only, #n)) RUN(n); } while (0)
    RUN_IF(orientation_lower_left);
    RUN_IF(clip_control_upper_left);
    RUN_IF(clip_control_depth_mode);
    RUN_IF(blend_modes);
    RUN_IF(blend_indexed_state);
    RUN_IF(blend_per_attachment);
    RUN_IF(color_mask);
    RUN_IF(polygon_offset);
    RUN_IF(viewport_scissor);
    RUN_IF(viewport_scissor_arrays);
    RUN_IF(cull_front_face);
    RUN_IF(rasterizer_discard);
    RUN_IF(depth_clamp);
    RUN_IF(depth_func_mask_range);
    RUN_IF(frag_coord_and_derivatives);
    RUN_IF(point_size_and_coord);
    RUN_IF(polygon_mode_line);
    RUN_IF(primitive_restart_rewrite);
    RUN_IF(sample_state);
    RUN_IF(misc_state_getters);
    return mgl_test_summary();
}
