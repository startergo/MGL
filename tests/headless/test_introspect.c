/* test_introspect.c — program interface, uniforms, blocks, bindings, subroutines */
#include "mgl_test.h"

static GLuint g_vao, g_vbo;
static void setup_tri(void)
{
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    if (!g_vao) {
        glGenVertexArrays(1, &g_vao); glBindVertexArray(g_vao);
        glGenBuffers(1, &g_vbo); glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    }
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    for (int i = 0; i < 8; i++) glDisableVertexAttribArray(i);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static void draw_clear(void)
{
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static const char *VS_PASS = "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";

static GLuint make_cs(const char *src)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL); glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log); MGL_FAIL("cs compile: %s", log); return 0; }
    GLuint p = glCreateProgram(); glAttachShader(p, s); glLinkProgram(p);
    return p;
}

static int find_uniform(GLuint p, const char *name, GLint *size, GLenum *type)
{
    GLint n = 0; glGetProgramInterfaceiv(p, GL_UNIFORM, GL_ACTIVE_RESOURCES, &n);
    for (GLint i = 0; i < n; i++) {
        char nm[128]; GLsizei len; GLint sz; GLenum ty;
        glGetActiveUniform(p, (GLuint)i, sizeof nm, &len, &sz, &ty, nm);
        if (!strcmp(nm, name)) { if (size) *size = sz; if (type) *type = ty; return i; }
    }
    return -1;
}

/* ---------------------------------------------------------------- default block */

TEST(draw_uniforms_vs_fs)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "uniform float k;\n"
        "uniform vec2 offs;\n"
        "void main(){ gl_Position = vec4(pos * k + offs, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 tint;\n"
        "uniform float scale;\n"
        "void main(){ color = tint * scale; }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    setup_tri();
    glUseProgram(p);
    GLint lk = glGetUniformLocation(p, "k"), lo = glGetUniformLocation(p, "offs");
    GLint lt = glGetUniformLocation(p, "tint"), ls = glGetUniformLocation(p, "scale");
    EXPECT_TRUE(lk >= 0); EXPECT_TRUE(lo >= 0); EXPECT_TRUE(lt >= 0); EXPECT_TRUE(ls >= 0);
    glUniform1f(lk, 1.0f); glUniform2f(lo, 0.0f, 0.0f);
    glUniform4f(lt, 0.0f, 1.0f, 0.5f, 1.0f); glUniform1f(ls, 1.0f);
    EXPECT_NO_GL_ERROR();
    draw_clear();
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(10, 10, 0, 255, 128, 255, 3);
    glUniform4f(lt, 1.0f, 0.0f, 0.0f, 1.0f);
    draw_clear();
    EXPECT_PIXEL(10, 10, 255, 0, 0, 255, 3);
    glUniform1f(lk, 0.0f); /* vertex-stage uniform collapses the triangle */
    draw_clear();
    EXPECT_PIXEL(10, 10, 0, 0, 0, 255, 3);
    EXPECT_NO_GL_ERROR();
}

/* explicit default-block uniform locations in the VERTEX stage (the former draw bug) */
TEST(draw_explicit_uniform_locations_vs)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=7) uniform float k;\n"
        "uniform vec2 offs;\n"
        "void main(){ gl_Position = vec4(pos * k + offs, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "layout(location=3) uniform vec4 tint;\n"
        "void main(){ color = tint; }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    setup_tri();
    glUseProgram(p);
    GLint lk = glGetUniformLocation(p, "k");
    GLint lo = glGetUniformLocation(p, "offs");
    GLint lt = glGetUniformLocation(p, "tint");
    EXPECT_EQ_INT(lk, 7);
    EXPECT_EQ_INT(lt, 3);
    EXPECT_TRUE(lo >= 0 && lo != 3 && lo != 7); /* auto location stays clear of the explicit ones */
    glUniform1f(lk, 1.0f); glUniform2f(lo, 0.0f, 0.0f); glUniform4f(lt, 0, 0, 1, 1);
    EXPECT_NO_GL_ERROR();
    draw_clear();
    EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 3);
    /* a location nobody uses is silently ignored by ProgramUniform* */
    glProgramUniform1f(p, -1, 1.0f);
    glProgramUniform4f(p, 1000, 1, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    draw_clear();
    EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 3);
}

/* bool/bvec default-block uniforms read by the GPU (stored as ints in the Metal block) */
TEST(draw_bool_uniforms)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform bool on;\n"
        "uniform bvec2 sel;\n"
        "uniform float shade;\n"
        "void main(){ color = vec4(on ? 1.0 : 0.0, sel.x ? 0.5 : 0.0, sel.y ? 0.25 : 0.0, shade); }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    setup_tri();
    glUseProgram(p);
    GLint lon = glGetUniformLocation(p, "on");
    GLint lsel = glGetUniformLocation(p, "sel");
    GLint lshade = glGetUniformLocation(p, "shade");
    EXPECT_TRUE(lon >= 0); EXPECT_TRUE(lsel >= 0); EXPECT_TRUE(lshade >= 0);
    GLint ty = 0, sz = 0;
    find_uniform(p, "on", &sz, &ty); EXPECT_EQ_INT(ty, GL_BOOL);
    find_uniform(p, "sel", &sz, &ty); EXPECT_EQ_INT(ty, GL_BOOL_VEC2);
    GLint b = 1, selv[2] = { 1, 0 }; float shade = 1.0f;
    EXPECT_NO_GL_ERROR();
    glUniform1iv(lon, 1, &b);
    glUniform2iv(lsel, 1, selv);
    glUniform1f(lshade, 1.0f);
    EXPECT_NO_GL_ERROR();
    GLint back = 0;
    glGetUniformiv(p, lon, &back);
    EXPECT_EQ_INT(back, 1); /* GL bool reads back as 0/1 */
    draw_clear();
    EXPECT_PIXEL(10, 10, 255, 128, 0, 255, 3);
    b = 0; selv[0] = 0; selv[1] = 1;
    glUniform1iv(lon, 1, &b);
    glUniform2iv(lsel, 1, selv);
    draw_clear();
    EXPECT_PIXEL(10, 10, 0, 0, 64, 255, 3);
    EXPECT_NO_GL_ERROR();
}

/* glGetnUniform*: bufSize too small for the returned data */
TEST(n_uniform_bufsize)
{
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 v;\n"
        "uniform double d;\n"
        "void main(){ color = v + float(d); }\n";
    GLuint p = mgl_make_program(VS_PASS, fs);
    if (!p) return;
    glUseProgram(p);
    GLint lv = glGetUniformLocation(p, "v"), ld = glGetUniformLocation(p, "d");
    EXPECT_TRUE(lv >= 0); EXPECT_TRUE(ld >= 0);
    glUniform4f(lv, 1, 2, 3, 4);
    glProgramUniform1d(p, ld, 0.5);
    EXPECT_NO_GL_ERROR();
    GLfloat fv[4] = { 0 };
    glGetnUniformfv(p, lv, 0, fv);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetnUniformfv(p, lv, (GLsizei)(3 * sizeof(GLfloat)), fv);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetnUniformfv(p, lv, (GLsizei)(4 * sizeof(GLfloat)), fv);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(fv[0], 1); EXPECT_EQ_INT(fv[3], 4);
    GLdouble dv = 0;
    glGetnUniformdv(p, ld, 4, &dv);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetnUniformdv(p, ld, 8, &dv);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(dv == 0.5);
}

static const char *FS_TYPES = "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "struct S { vec2 a; int b; };\n"
    "uniform float uf;\n"
    "uniform vec3 uv3;\n"
    "uniform ivec2 ui2;\n"
    "uniform uvec4 uu4;\n"
    "uniform bool ub;\n"
    "uniform mat3 um3;\n"
    "uniform mat2x4 um24;\n"
    "uniform vec4 arr[3];\n"
    "uniform S s;\n"
    "uniform S sarr[2];\n"
    "layout(location = 20) uniform float explicit_loc;\n"
    "void main(){\n"
    "  vec4 c = vec4(uf) + vec4(uv3, 1.0) + vec4(ui2, 0, 0) + vec4(uu4) + (ub ? vec4(1) : vec4(0));\n"
    "  c += vec4(um3[1], 0.0) + um24[1] + arr[2] + vec4(s.a, s.b, 0) + vec4(sarr[1].a, sarr[0].b, 0) + vec4(explicit_loc);\n"
    "  color = c * 0.0001;\n"
    "}\n";

TEST(default_block_reflection)
{
    GLuint p = mgl_make_program(VS_PASS, FS_TYPES);
    if (!p) return;
    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_UNIFORM, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 15);
    GLint sz = 0; GLenum ty = 0;
    EXPECT_TRUE(find_uniform(p, "uf", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_FLOAT); EXPECT_EQ_INT(sz, 1);
    EXPECT_TRUE(find_uniform(p, "uv3", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_FLOAT_VEC3);
    EXPECT_TRUE(find_uniform(p, "ui2", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_INT_VEC2);
    EXPECT_TRUE(find_uniform(p, "uu4", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_UNSIGNED_INT_VEC4);
    EXPECT_TRUE(find_uniform(p, "ub", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_BOOL);
    EXPECT_TRUE(find_uniform(p, "um3", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_FLOAT_MAT3);
    EXPECT_TRUE(find_uniform(p, "um24", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_FLOAT_MAT2x4);
    EXPECT_TRUE(find_uniform(p, "arr[0]", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_FLOAT_VEC4); EXPECT_EQ_INT(sz, 3);
    EXPECT_TRUE(find_uniform(p, "s.a", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_FLOAT_VEC2);
    EXPECT_TRUE(find_uniform(p, "s.b", &sz, &ty) >= 0); EXPECT_EQ_INT(ty, GL_INT);
    EXPECT_TRUE(find_uniform(p, "sarr[1].a", &sz, &ty) >= 0);
    EXPECT_TRUE(find_uniform(p, "sarr[0].b", &sz, &ty) >= 0);

    /* locations: arrays, struct members, explicit layout */
    GLint la = glGetUniformLocation(p, "arr");
    EXPECT_TRUE(la >= 0);
    EXPECT_EQ_INT(glGetUniformLocation(p, "arr[0]"), la);
    EXPECT_EQ_INT(glGetUniformLocation(p, "arr[2]"), la + 2);
    EXPECT_EQ_INT(glGetUniformLocation(p, "arr[3]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "explicit_loc"), 20);
    EXPECT_TRUE(glGetUniformLocation(p, "s.a") >= 0);
    EXPECT_EQ_INT(glGetUniformLocation(p, "nope"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "gl_FragCoord"), -1);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_UNIFORM, "arr[1]"), la + 1);

    /* GetActiveUniformsiv on the default block */
    GLuint idx[2]; const char *names[2] = { "um3", "arr" };
    glGetUniformIndices(p, 2, names, idx);
    EXPECT_TRUE(idx[0] != GL_INVALID_INDEX && idx[1] != GL_INVALID_INDEX);
    GLint v[2];
    glGetActiveUniformsiv(p, 2, idx, GL_UNIFORM_BLOCK_INDEX, v); EXPECT_EQ_INT(v[0], -1); EXPECT_EQ_INT(v[1], -1);
    glGetActiveUniformsiv(p, 2, idx, GL_UNIFORM_OFFSET, v); EXPECT_EQ_INT(v[0], -1);
    glGetActiveUniformsiv(p, 2, idx, GL_UNIFORM_SIZE, v); EXPECT_EQ_INT(v[0], 1); EXPECT_EQ_INT(v[1], 3);
    glGetActiveUniformsiv(p, 2, idx, GL_UNIFORM_NAME_LENGTH, v); EXPECT_EQ_INT(v[0], 4); EXPECT_EQ_INT(v[1], 7);
    char nm[64]; GLsizei len = 0;
    glGetActiveUniformName(p, idx[1], sizeof nm, &len, nm); EXPECT_STREQ(nm, "arr[0]"); EXPECT_EQ_INT(len, 6);
    GLint ml = 0; glGetProgramInterfaceiv(p, GL_UNIFORM, GL_MAX_NAME_LENGTH, &ml); EXPECT_EQ_INT(ml, 13); /* explicit_loc */
    EXPECT_NO_GL_ERROR();
    /* errors */
    glGetActiveUniformsiv(p, 1, idx, GL_UNIFORM_BLOCK_BINDING, v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    GLuint bad = 999; glGetActiveUniformsiv(p, 1, &bad, GL_UNIFORM_SIZE, v); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetActiveUniformName(p, 999, sizeof nm, &len, nm); EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(uniform_round_trip)
{
    GLuint p = mgl_make_program(VS_PASS, FS_TYPES);
    if (!p) return;
    glUseProgram(p);
    GLint l;
    float f = 0;
    l = glGetUniformLocation(p, "uf"); glUniform1f(l, 2.5f); glGetUniformfv(p, l, &f); EXPECT_NEAR(f, 2.5, 1e-6);
    GLint iv[4] = { 0 };
    glGetUniformiv(p, l, iv); EXPECT_EQ_INT(iv[0], 3); /* float -> int rounds */
    l = glGetUniformLocation(p, "uv3"); glUniform3f(l, 1, 2, 3);
    float f3[3] = { 0 }; glGetUniformfv(p, l, f3); EXPECT_NEAR(f3[2], 3, 1e-6);
    l = glGetUniformLocation(p, "ui2"); glUniform2i(l, -7, 9); glGetUniformiv(p, l, iv); EXPECT_EQ_INT(iv[0], -7); EXPECT_EQ_INT(iv[1], 9);
    l = glGetUniformLocation(p, "uu4"); glUniform4ui(l, 1, 2, 3, 0xFFFFFFFFu);
    GLuint uv[4] = { 0 }; glGetUniformuiv(p, l, uv); EXPECT_EQ_INT(uv[3], 0xFFFFFFFFu);
    l = glGetUniformLocation(p, "ub"); glUniform1f(l, 0.5f); glGetUniformiv(p, l, iv); EXPECT_EQ_INT(iv[0], 1);
    glUniform1i(l, 0); glGetUniformiv(p, l, iv); EXPECT_EQ_INT(iv[0], 0);
    glUniform1ui(l, 4); glGetUniformfv(p, l, &f); EXPECT_NEAR(f, 1.0, 0);
    /* mat3 with transpose */
    const float m[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    l = glGetUniformLocation(p, "um3"); glUniformMatrix3fv(l, 1, GL_TRUE, m);
    float r[9] = { 0 }; glGetUniformfv(p, l, r);
    EXPECT_NEAR(r[0], 1, 0); EXPECT_NEAR(r[1], 4, 0); EXPECT_NEAR(r[3], 2, 0); EXPECT_NEAR(r[8], 9, 0);
    const float m24[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    l = glGetUniformLocation(p, "um24"); glUniformMatrix2x4fv(l, 1, GL_FALSE, m24);
    float r24[8] = { 0 }; glGetUniformfv(p, l, r24); EXPECT_NEAR(r24[5], 6, 0);
    /* arrays: partial update from an element location, clamped count */
    const float a[12] = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
    l = glGetUniformLocation(p, "arr"); glUniform4fv(l, 3, a);
    glUniform4fv(l + 1, 5, a + 8); /* writes elements 1,2 from {3,..} */
    float e[4]; glGetUniformfv(p, l + 1, e); EXPECT_NEAR(e[0], 3, 0);
    glGetUniformfv(p, l, e); EXPECT_NEAR(e[0], 1, 0);
    /* struct members */
    l = glGetUniformLocation(p, "s.b"); glUniform1i(l, 42); glGetUniformiv(p, l, iv); EXPECT_EQ_INT(iv[0], 42);
    l = glGetUniformLocation(p, "sarr[1].a"); glUniform2f(l, 5, 6); float f2[2]; glGetUniformfv(p, l, f2); EXPECT_NEAR(f2[1], 6, 0);
    EXPECT_NO_GL_ERROR();
    /* GetnUniform */
    l = glGetUniformLocation(p, "uv3");
    glGetnUniformfv(p, l, 8, f3); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetnUniformfv(p, l, 12, f3); EXPECT_NO_GL_ERROR(); EXPECT_NEAR(f3[1], 2, 0);
    GLint i3[3]; glGetnUniformiv(p, l, 12, i3); EXPECT_EQ_INT(i3[1], 2);
    GLuint u3[3]; glGetnUniformuiv(p, l, 12, u3); EXPECT_EQ_INT(u3[2], 3);
    GLdouble d3[3]; glGetnUniformdv(p, l, 24, d3); EXPECT_NEAR(d3[0], 1, 0);
    glGetnUniformdv(p, l, 16, d3); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetUniformdv(p, l, d3); EXPECT_NEAR(d3[2], 3, 0);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
}

TEST(uniform_errors)
{
    GLuint p = mgl_make_program(VS_PASS, FS_TYPES);
    if (!p) return;
    glUseProgram(0);
    glUniform1f(0, 1.0f); EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* no current program */
    glUseProgram(p);
    GLint lf = glGetUniformLocation(p, "uf"), lv = glGetUniformLocation(p, "uv3"), li = glGetUniformLocation(p, "ui2");
    glUniform1i(lf, 1); EXPECT_GL_ERROR(GL_INVALID_OPERATION);   /* type mismatch */
    glUniform2f(lv, 1, 2); EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* size mismatch */
    glUniform2ui(li, 1, 2); EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* int vs uint */
    float two[2] = { 1, 2 };
    glUniform1fv(lf, 2, two); EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* count > 1 on non-array */
    glUniform1fv(lf, -1, two); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUniform1f(-1, 5.0f); EXPECT_NO_GL_ERROR();                  /* -1 silently ignored */
    glUniform1f(5000, 5.0f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    const float m[4] = { 0 };
    glUniformMatrix2fv(glGetUniformLocation(p, "um3"), 1, GL_FALSE, m); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUniform1d(lf, 1.0); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    float f; glGetUniformfv(p, 5000, &f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetUniformfv(12345, 0, &f); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLuint unlinked = glCreateProgram();
    glGetUniformfv(unlinked, 0, &f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_EQ_INT(glGetUniformLocation(unlinked, "uf"), -1); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glProgramUniform1f(unlinked, 0, 1.0f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glProgramUniform1f(12345, 0, 1.0f); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glProgramUniform1f(sh, 0, 1.0f); EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* shader name */
    glUseProgram(0);
}

TEST(program_uniform_non_current)
{
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 c;\n"
        "uniform mat2 m;\n"
        "void main(){ color = c + vec4(m[0], m[1]) * 0.0; }\n";
    GLuint a = mgl_make_program(VS_PASS, fs), b = mgl_make_program(VS_PASS, fs);
    if (!a || !b) return;
    setup_tri();
    glUseProgram(b);
    GLint la = glGetUniformLocation(a, "c");
    glProgramUniform4f(a, la, 0.0f, 0.0f, 1.0f, 1.0f);     /* a is not current */
    glProgramUniform4f(b, glGetUniformLocation(b, "c"), 1.0f, 1.0f, 0.0f, 1.0f);
    const float m[4] = { 1, 2, 3, 4 };
    glProgramUniformMatrix2fv(a, glGetUniformLocation(a, "m"), 1, GL_TRUE, m);
    EXPECT_NO_GL_ERROR();
    float r[4]; glGetUniformfv(a, la, r); EXPECT_NEAR(r[2], 1.0, 0);
    glGetUniformfv(a, glGetUniformLocation(a, "m"), r); EXPECT_NEAR(r[1], 3, 0);
    draw_clear(); EXPECT_PIXEL(10, 10, 255, 255, 0, 255, 2);
    glUseProgram(a);
    draw_clear(); EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 2);
    /* glProgramUniform on the current program behaves like glUniform */
    glProgramUniform4f(a, la, 0.0f, 1.0f, 0.0f, 1.0f);
    draw_clear(); EXPECT_PIXEL(10, 10, 0, 255, 0, 255, 2);
    glProgramUniform1f(a, -1, 1.0f); EXPECT_NO_GL_ERROR();
    glProgramUniform3f(a, la, 1, 2, 3); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    /* relink resets uniform values */
    glLinkProgram(a);
    glGetUniformfv(a, glGetUniformLocation(a, "c"), r); EXPECT_NEAR(r[1], 0.0, 0);
    glUseProgram(0);
}

TEST(uniform_initializer)
{
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 c = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "uniform float k = 2.0;\n"
        "void main(){ color = c * (k * 0.5); }\n";
    GLuint p = mgl_make_program(VS_PASS, fs);
    if (!p) return;
    float r[4]; glGetUniformfv(p, glGetUniformLocation(p, "c"), r); EXPECT_NEAR(r[1], 1.0, 0);
    glGetUniformfv(p, glGetUniformLocation(p, "k"), r); EXPECT_NEAR(r[0], 2.0, 0);
    setup_tri(); glUseProgram(p);
    draw_clear(); EXPECT_PIXEL(10, 10, 0, 255, 0, 255, 2);
    glUseProgram(0);
}

/* ---------------------------------------------------------------- uniform blocks */

static const char *FS_UBO = "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "layout(std140, binding = 2) uniform Colors { vec4 col; mat4 xf; float w[2]; } colors;\n"
    "layout(std140) uniform Other { vec2 o2; vec3 o3; };\n"
    "void main(){ color = colors.col + vec4(o2, 0, 0) * 0.0 + vec4(o3, 0) * 0.0 + colors.xf[0] * 0.0 + vec4(colors.w[1]) * 0.0; }\n";

TEST(ubo_reflection)
{
    GLuint p = mgl_make_program(VS_PASS, FS_UBO);
    if (!p) return;
    GLint n = 0; glGetProgramInterfaceiv(p, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &n); EXPECT_EQ_INT(n, 2);
    GLuint bi = glGetUniformBlockIndex(p, "Colors");
    GLuint bo = glGetUniformBlockIndex(p, "Other");
    EXPECT_TRUE(bi != GL_INVALID_INDEX); EXPECT_TRUE(bo != GL_INVALID_INDEX);
    EXPECT_EQ_INT(glGetUniformBlockIndex(p, "colors"), GL_INVALID_INDEX);
    GLint v = 0;
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_BINDING, &v); EXPECT_EQ_INT(v, 2);
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_DATA_SIZE, &v); EXPECT_EQ_INT(v, 112);
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &v); EXPECT_EQ_INT(v, 3);
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_NAME_LENGTH, &v); EXPECT_EQ_INT(v, 7);
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER, &v); EXPECT_EQ_INT(v, 1);
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER, &v); EXPECT_EQ_INT(v, 0);
    glGetActiveUniformBlockiv(p, bo, GL_UNIFORM_BLOCK_DATA_SIZE, &v); EXPECT_EQ_INT(v, 32);
    GLint ids[3] = { -1, -1, -1 };
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, ids);
    char nm[64]; GLsizei len;
    glGetActiveUniformBlockName(p, bi, sizeof nm, &len, nm); EXPECT_STREQ(nm, "Colors");
    /* member reflection (named instance -> "Colors.member"; anonymous block -> plain) */
    const char *names[4] = { "Colors.col", "Colors.xf", "Colors.w", "o3" };
    GLuint mi[4]; glGetUniformIndices(p, 4, names, mi);
    for (int i = 0; i < 4; i++) EXPECT_TRUE(mi[i] != GL_INVALID_INDEX);
    GLint off[4], as[4], ms[4], bx[4], rm[4], ty[4];
    glGetActiveUniformsiv(p, 4, mi, GL_UNIFORM_OFFSET, off);
    glGetActiveUniformsiv(p, 4, mi, GL_UNIFORM_ARRAY_STRIDE, as);
    glGetActiveUniformsiv(p, 4, mi, GL_UNIFORM_MATRIX_STRIDE, ms);
    glGetActiveUniformsiv(p, 4, mi, GL_UNIFORM_BLOCK_INDEX, bx);
    glGetActiveUniformsiv(p, 4, mi, GL_UNIFORM_IS_ROW_MAJOR, rm);
    glGetActiveUniformsiv(p, 4, mi, GL_UNIFORM_TYPE, ty);
    EXPECT_EQ_INT(off[0], 0); EXPECT_EQ_INT(off[1], 16); EXPECT_EQ_INT(off[2], 80); EXPECT_EQ_INT(off[3], 16);
    EXPECT_EQ_INT(as[0], 0); EXPECT_EQ_INT(as[2], 16);
    EXPECT_EQ_INT(ms[1], 16); EXPECT_EQ_INT(ms[0], 0);
    EXPECT_EQ_INT(bx[0], (GLint)bi); EXPECT_EQ_INT(bx[3], (GLint)bo);
    EXPECT_EQ_INT(rm[1], 0);
    EXPECT_EQ_INT(ty[1], GL_FLOAT_MAT4); EXPECT_EQ_INT(ty[2], GL_FLOAT);
    int has_col = 0; for (int i = 0; i < 3; i++) if (ids[i] == (GLint)mi[0]) has_col = 1;
    EXPECT_TRUE(has_col);
    /* the same through GL 4.3 resource queries */
    GLenum props[4] = { GL_BUFFER_BINDING, GL_BUFFER_DATA_SIZE, GL_NUM_ACTIVE_VARIABLES, GL_REFERENCED_BY_FRAGMENT_SHADER };
    GLint pv[4] = { 0 }; GLsizei pl = 0;
    glGetProgramResourceiv(p, GL_UNIFORM_BLOCK, bi, 4, props, 4, &pl, pv);
    EXPECT_EQ_INT(pl, 4); EXPECT_EQ_INT(pv[0], 2); EXPECT_EQ_INT(pv[1], 112); EXPECT_EQ_INT(pv[2], 3); EXPECT_EQ_INT(pv[3], 1);
    EXPECT_EQ_INT(glGetProgramResourceIndex(p, GL_UNIFORM_BLOCK, "Other"), bo);
    GLint loc = 0; GLenum lp = GL_LOCATION;
    glGetProgramResourceiv(p, GL_UNIFORM, mi[0], 1, &lp, 1, NULL, &loc); EXPECT_EQ_INT(loc, -1);
    EXPECT_NO_GL_ERROR();
    glGetActiveUniformBlockiv(p, 99, GL_UNIFORM_BLOCK_BINDING, &v); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_TYPE, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glUniformBlockBinding(p, 99, 0); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUniformBlockBinding(p, bi, 100000); EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(uniform_block_binding_draw)
{
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "layout(std140) uniform Col { vec4 c; };\n"
        "void main(){ color = c; }\n";
    GLuint p = mgl_make_program(VS_PASS, fs);
    if (!p) return;
    setup_tri();
    const float red[4] = { 1, 0, 0, 1 }, blue[4] = { 0, 0, 1, 1 };
    GLuint ub[2]; glGenBuffers(2, ub);
    glBindBuffer(GL_UNIFORM_BUFFER, ub[0]); glBufferData(GL_UNIFORM_BUFFER, 16, red, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, ub[1]); glBufferData(GL_UNIFORM_BUFFER, 16, blue, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, ub[0]);
    glBindBufferBase(GL_UNIFORM_BUFFER, 5, ub[1]);
    GLuint bi = glGetUniformBlockIndex(p, "Col");
    glUniformBlockBinding(p, bi, 3);
    GLint v; glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_BINDING, &v); EXPECT_EQ_INT(v, 3);
    glUseProgram(p);
    draw_clear(); EXPECT_PIXEL(10, 10, 255, 0, 0, 255, 2);
    glUniformBlockBinding(p, bi, 5);
    draw_clear(); EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 2);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
}

/* ---------------------------------------------------------------- storage blocks */

TEST(ssbo_reflection_and_binding)
{
    const char *cs = "#version 450 core\n"
        "layout(local_size_x = 1) in;\n"
        "struct P { vec4 pos; float m; };\n"
        "layout(std430, binding = 0) buffer Out { uint head; vec2 v2; P parts[2]; uint vals[]; } outb;\n"
        "layout(location = 0) uniform uint k;\n"
        "void main(){ outb.head = k; outb.vals[1] = k * 2u; outb.parts[1].m = 1.0; outb.v2 = vec2(1); }\n";
    GLuint p = make_cs(cs);
    if (!p) return;
    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &n); EXPECT_EQ_INT(n, 1);
    GLuint bi = glGetProgramResourceIndex(p, GL_SHADER_STORAGE_BLOCK, "Out");
    EXPECT_EQ_INT(bi, 0);
    GLenum bp[3] = { GL_BUFFER_BINDING, GL_NUM_ACTIVE_VARIABLES, GL_REFERENCED_BY_COMPUTE_SHADER };
    GLint bv[3] = { 0 };
    glGetProgramResourceiv(p, GL_SHADER_STORAGE_BLOCK, bi, 3, bp, 3, NULL, bv);
    EXPECT_EQ_INT(bv[0], 0); EXPECT_EQ_INT(bv[1], 5); EXPECT_EQ_INT(bv[2], 1);
    GLuint vi = glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, "Out.vals[0]");
    GLuint pi = glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, "Out.parts[0].m");
    GLuint hi = glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, "Out.head");
    EXPECT_TRUE(vi != GL_INVALID_INDEX); EXPECT_TRUE(pi != GL_INVALID_INDEX); EXPECT_TRUE(hi != GL_INVALID_INDEX);
    GLenum vp[6] = { GL_OFFSET, GL_ARRAY_SIZE, GL_ARRAY_STRIDE, GL_TOP_LEVEL_ARRAY_SIZE, GL_TOP_LEVEL_ARRAY_STRIDE, GL_BLOCK_INDEX };
    GLint vv[6];
    glGetProgramResourceiv(p, GL_BUFFER_VARIABLE, vi, 6, vp, 6, NULL, vv);
    EXPECT_EQ_INT(vv[0], 80); EXPECT_EQ_INT(vv[1], 0); EXPECT_EQ_INT(vv[2], 4); EXPECT_EQ_INT(vv[3], 1); EXPECT_EQ_INT(vv[5], 0);
    glGetProgramResourceiv(p, GL_BUFFER_VARIABLE, pi, 6, vp, 6, NULL, vv);
    EXPECT_EQ_INT(vv[0], 32); EXPECT_EQ_INT(vv[3], 2); EXPECT_EQ_INT(vv[4], 32);
    EXPECT_NO_GL_ERROR();

    /* ShaderStorageBlockBinding redirects the writes */
    GLuint sb[2]; glGenBuffers(2, sb);
    GLuint zero[32] = { 0 };
    for (int i = 0; i < 2; i++) { glBindBuffer(GL_SHADER_STORAGE_BUFFER, sb[i]); glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof zero, zero, GL_DYNAMIC_COPY); }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sb[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, sb[1]);
    glShaderStorageBlockBinding(p, bi, 4);
    glGetProgramResourceiv(p, GL_SHADER_STORAGE_BLOCK, bi, 1, bp, 1, NULL, bv); EXPECT_EQ_INT(bv[0], 4);
    glUseProgram(p);
    glUniform1ui(0, 21u);
    glDispatchCompute(1, 1, 1); glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish();
    GLuint out[32] = { 0 };
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sb[1]); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 21); EXPECT_EQ_INT(out[21], 42);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sb[0]); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 0);
    glShaderStorageBlockBinding(p, bi, 1);
    glUniform1ui(0, 5u);
    glDispatchCompute(1, 1, 1); glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sb[0]); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 5); EXPECT_EQ_INT(out[21], 10);
    EXPECT_NO_GL_ERROR();
    glShaderStorageBlockBinding(p, 7, 0); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUseProgram(0);
}

/* ---------------------------------------------------------------- attributes / outputs */

TEST(bind_attrib_location)
{
    const char *vs = "#version 450 core\n"
        "in vec2 pos;\n"
        "in vec4 col;\n"
        "out vec4 vcol;\n"
        "void main(){ vcol = col; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "in vec4 vcol;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vcol; }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    GLint n = 0; glGetProgramInterfaceiv(p, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &n); EXPECT_EQ_INT(n, 2);
    glBindAttribLocation(p, 3, "pos");
    glBindAttribLocation(p, 1, "col");
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(glGetAttribLocation(p, "pos") != 3); /* not effective before relink */
    glLinkProgram(p);
    EXPECT_EQ_INT(glGetAttribLocation(p, "pos"), 3);
    EXPECT_EQ_INT(glGetAttribLocation(p, "col"), 1);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "col"), 1);
    EXPECT_EQ_INT(glGetAttribLocation(p, "nope"), -1);
    char nm[32]; GLint sz; GLenum ty; GLsizei len;
    GLint na = 0; glGetProgramiv(p, GL_ACTIVE_ATTRIBUTES, &na);
    glGetActiveAttrib(p, 0, sizeof nm, &len, &sz, &ty, nm);
    EXPECT_TRUE(!strcmp(nm, "pos") || !strcmp(nm, "col"));
    EXPECT_TRUE(ty == GL_FLOAT_VEC2 || ty == GL_FLOAT_VEC4);
    glGetActiveAttrib(p, 5, sizeof nm, &len, &sz, &ty, nm); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindAttribLocation(p, 1000, "pos"); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindAttribLocation(p, 0, "gl_Vertex"); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    /* draw: position from attribute 3, colour from attribute 1 */
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    static const float cols[] = { 0,1,0,1, 0,1,0,1, 0,1,0,1 };
    GLuint vao, vb[2];
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(2, vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb[0]); glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(3);
    glBindBuffer(GL_ARRAY_BUFFER, vb[1]); glBufferData(GL_ARRAY_BUFFER, sizeof cols, cols, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(1);
    glUseProgram(p);
    draw_clear();
    EXPECT_PIXEL(10, 10, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
    glBindVertexArray(0);
}

TEST(bind_frag_data_location)
{
    const char *fs = "#version 450 core\n"
        "out vec4 colA;\n"
        "out vec4 colB;\n"
        "void main(){ colA = vec4(1,0,0,1); colB = vec4(0,1,0,1); }\n";
    GLuint p = mgl_make_program(VS_PASS, fs);
    if (!p) return;
    glBindFragDataLocation(p, 1, "colA");
    glBindFragDataLocation(p, 0, "colB");
    glLinkProgram(p);
    EXPECT_EQ_INT(glGetFragDataLocation(p, "colA"), 1);
    EXPECT_EQ_INT(glGetFragDataLocation(p, "colB"), 0);
    EXPECT_EQ_INT(glGetFragDataIndex(p, "colB"), 0);
    EXPECT_EQ_INT(glGetFragDataLocation(p, "nope"), -1);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_OUTPUT, "colA"), 1);
    EXPECT_EQ_INT(glGetProgramResourceLocationIndex(p, GL_PROGRAM_OUTPUT, "colA"), 0);
    setup_tri();
    glUseProgram(p);
    draw_clear(); /* default framebuffer has one colour buffer: location 0 = colB = green */
    EXPECT_PIXEL(10, 10, 0, 255, 0, 255, 2);
    glUseProgram(0);
    /* dual-source index */
    glBindFragDataLocationIndexed(p, 0, 0, "colA");
    glBindFragDataLocationIndexed(p, 0, 1, "colB");
    glLinkProgram(p);
    EXPECT_EQ_INT(glGetFragDataIndex(p, "colB"), 1);
    EXPECT_EQ_INT(glGetFragDataIndex(p, "colA"), 0);
    EXPECT_EQ_INT(glGetFragDataLocation(p, "colB"), 0);
    GLuint oi = glGetProgramResourceIndex(p, GL_PROGRAM_OUTPUT, "colB");
    GLenum pp[2] = { GL_LOCATION, GL_LOCATION_INDEX }; GLint pv[2];
    glGetProgramResourceiv(p, GL_PROGRAM_OUTPUT, oi, 2, pp, 2, NULL, pv); EXPECT_EQ_INT(pv[0], 0); EXPECT_EQ_INT(pv[1], 1);
    EXPECT_NO_GL_ERROR();
    glBindFragDataLocationIndexed(p, 0, 2, "colB"); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindFragDataLocation(p, 1000, "colB"); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindFragDataLocation(p, 0, "gl_FragColor"); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
}

/* ---------------------------------------------------------------- program interface */

TEST(resource_queries)
{
    const char *vs = "#version 450 core\n"
        "layout(location=2) in vec3 a;\n"
        "layout(location=0) in vec2 pos;\n"
        "out vec3 v;\n"
        "uniform mat4 mvp;\n"
        "void main(){ v = a; gl_Position = mvp * vec4(pos, 0, 1); }\n";
    const char *fs = "#version 450 core\n"
        "in vec3 v;\n"
        "layout(location=0) out vec4 color;\n"
        "uniform sampler2D tex;\n"
        "void main(){ color = vec4(v, 1) + texture(tex, v.xy); }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &n); EXPECT_EQ_INT(n, 2);
    glGetProgramInterfaceiv(p, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &n); EXPECT_EQ_INT(n, 1);
    glGetProgramInterfaceiv(p, GL_PROGRAM_INPUT, GL_MAX_NAME_LENGTH, &n); EXPECT_EQ_INT(n, 4);
    GLuint ai = glGetProgramResourceIndex(p, GL_PROGRAM_INPUT, "a");
    EXPECT_TRUE(ai != GL_INVALID_INDEX);
    GLenum pp[5] = { GL_TYPE, GL_ARRAY_SIZE, GL_LOCATION, GL_REFERENCED_BY_VERTEX_SHADER, GL_NAME_LENGTH };
    GLint pv[5] = { 0 }; GLsizei len = 0;
    glGetProgramResourceiv(p, GL_PROGRAM_INPUT, ai, 5, pp, 5, &len, pv);
    EXPECT_EQ_INT(len, 5); EXPECT_EQ_INT(pv[0], GL_FLOAT_VEC3); EXPECT_EQ_INT(pv[1], 1); EXPECT_EQ_INT(pv[2], 2);
    EXPECT_EQ_INT(pv[3], 1); EXPECT_EQ_INT(pv[4], 2);
    /* count limits the written values */
    glGetProgramResourceiv(p, GL_PROGRAM_INPUT, ai, 5, pp, 2, &len, pv); EXPECT_EQ_INT(len, 2);
    char nm[32];
    glGetProgramResourceName(p, GL_PROGRAM_INPUT, ai, sizeof nm, &len, nm); EXPECT_STREQ(nm, "a"); EXPECT_EQ_INT(len, 1);
    GLuint ui = glGetProgramResourceIndex(p, GL_UNIFORM, "mvp");
    GLenum up[3] = { GL_TYPE, GL_LOCATION, GL_REFERENCED_BY_VERTEX_SHADER }; GLint uv[3];
    glGetProgramResourceiv(p, GL_UNIFORM, ui, 3, up, 3, NULL, uv);
    EXPECT_EQ_INT(uv[0], GL_FLOAT_MAT4); EXPECT_EQ_INT(uv[1], glGetUniformLocation(p, "mvp")); EXPECT_EQ_INT(uv[2], 1);
    GLuint ti = glGetProgramResourceIndex(p, GL_UNIFORM, "tex");
    glGetProgramResourceiv(p, GL_UNIFORM, ti, 1, up, 1, NULL, uv); EXPECT_EQ_INT(uv[0], GL_SAMPLER_2D);
    GLint act = 0; glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &act);
    EXPECT_NO_GL_ERROR();
    /* error cases */
    glGetProgramInterfaceiv(p, GL_TEXTURE_2D, GL_ACTIVE_RESOURCES, &n); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetProgramInterfaceiv(p, GL_UNIFORM, GL_TEXTURE_2D, &n); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetProgramInterfaceiv(p, GL_ATOMIC_COUNTER_BUFFER, GL_MAX_NAME_LENGTH, &n); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetProgramInterfaceiv(p, GL_UNIFORM, GL_MAX_NUM_ACTIVE_VARIABLES, &n); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_EQ_INT(glGetProgramResourceIndex(p, GL_ATOMIC_COUNTER_BUFFER, "x"), GL_INVALID_INDEX); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_EQ_INT(glGetProgramResourceIndex(p, GL_UNIFORM, "nope"), GL_INVALID_INDEX); EXPECT_NO_GL_ERROR();
    glGetProgramResourceName(p, GL_UNIFORM, 999, sizeof nm, &len, nm); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLenum bad = GL_BUFFER_BINDING;
    glGetProgramResourceiv(p, GL_UNIFORM, ui, 1, &bad, 1, NULL, uv); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    bad = GL_TEXTURE_2D;
    glGetProgramResourceiv(p, GL_UNIFORM, ui, 1, &bad, 1, NULL, uv); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetProgramResourceiv(p, GL_UNIFORM, ui, 0, up, 1, NULL, uv); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_UNIFORM_BLOCK, "x"), -1); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_EQ_INT(glGetProgramResourceLocationIndex(p, GL_PROGRAM_INPUT, "a"), -1); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    GLuint sh[4] = { 0 }; GLsizei cnt = 0;
    glGetAttachedShaders(p, 4, &cnt, sh); EXPECT_EQ_INT(cnt, 2); EXPECT_TRUE(sh[0] && sh[1]);
    glGetAttachedShaders(p, 1, &cnt, sh); EXPECT_EQ_INT(cnt, 1);
    glGetAttachedShaders(p, -1, &cnt, sh); EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(sampler_unit_uniform)
{
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform sampler2D tex;\n"
        "void main(){ color = texture(tex, vec2(0.5)); }\n";
    GLuint p = mgl_make_program(VS_PASS, fs);
    if (!p) return;
    const GLubyte red[4] = { 255, 0, 0, 255 }, blue[4] = { 0, 0, 255, 255 };
    GLuint t[2]; glGenTextures(2, t);
    for (int i = 0; i < 2; i++) {
        glActiveTexture(GL_TEXTURE0 + i * 3);
        glBindTexture(GL_TEXTURE_2D, t[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, i ? blue : red);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glActiveTexture(GL_TEXTURE0);
    setup_tri(); glUseProgram(p);
    GLint l = glGetUniformLocation(p, "tex");
    glUniform1i(l, 0); draw_clear(); EXPECT_PIXEL(10, 10, 255, 0, 0, 255, 2);
    glUniform1i(l, 3); draw_clear(); EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 2);
    GLint v = -1; glGetUniformiv(p, l, &v); EXPECT_EQ_INT(v, 3);
    glUniform1f(l, 1.0f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUniform1i(l, -1); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUseProgram(0);
}

/* ---------------------------------------------------------------- subroutines */

TEST(subroutines)
{
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "subroutine vec4 ColorFn(vec2 uv);\n"
        "subroutine float ScaleFn();\n"
        "subroutine(ColorFn) vec4 red(vec2 uv) { return vec4(1, 0, 0, 1); }\n"
        "layout(index = 5) subroutine(ColorFn) vec4 blue(vec2 uv) { return vec4(0, 0, 1, 1); }\n"
        "subroutine(ScaleFn) float one() { return 1.0; }\n"
        "subroutine(ScaleFn) float half_() { return 0.5; }\n"
        "subroutine uniform ColorFn pick;\n"
        "subroutine uniform ScaleFn scale;\n"
        "void main(){ color = pick(vec2(0)) * scale(); color.a = 1.0; }\n";
    GLuint p = mgl_make_program(VS_PASS, fs);
    if (!p) return;
    GLint v = 0;
    glGetProgramStageiv(p, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINES, &v); EXPECT_EQ_INT(v, 4);
    glGetProgramStageiv(p, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORMS, &v); EXPECT_EQ_INT(v, 2);
    glGetProgramStageiv(p, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &v); EXPECT_EQ_INT(v, 2);
    glGetProgramStageiv(p, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_MAX_LENGTH, &v); EXPECT_EQ_INT(v, 6);
    glGetProgramStageiv(p, GL_VERTEX_SHADER, GL_ACTIVE_SUBROUTINES, &v); EXPECT_EQ_INT(v, 0);
    GLuint ired = glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "red");
    GLuint iblue = glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "blue");
    GLuint ione = glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "one");
    GLuint ihalf = glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "half_");
    EXPECT_EQ_INT(iblue, 5);
    EXPECT_TRUE(ired != GL_INVALID_INDEX && ione != GL_INVALID_INDEX && ihalf != GL_INVALID_INDEX);
    EXPECT_EQ_INT(glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "nope"), GL_INVALID_INDEX);
    GLint lp = glGetSubroutineUniformLocation(p, GL_FRAGMENT_SHADER, "pick");
    GLint ls = glGetSubroutineUniformLocation(p, GL_FRAGMENT_SHADER, "scale");
    EXPECT_TRUE(lp >= 0 && ls >= 0 && lp != ls);
    char nm[32]; GLsizei len;
    glGetActiveSubroutineName(p, GL_FRAGMENT_SHADER, iblue, sizeof nm, &len, nm); EXPECT_STREQ(nm, "blue");
    glGetActiveSubroutineUniformName(p, GL_FRAGMENT_SHADER, 0, sizeof nm, &len, nm); EXPECT_STREQ(nm, "pick");
    GLint nc = 0, comp[4] = { -1, -1, -1, -1 };
    glGetActiveSubroutineUniformiv(p, GL_FRAGMENT_SHADER, 0, GL_NUM_COMPATIBLE_SUBROUTINES, &nc); EXPECT_EQ_INT(nc, 2);
    glGetActiveSubroutineUniformiv(p, GL_FRAGMENT_SHADER, 0, GL_COMPATIBLE_SUBROUTINES, comp);
    EXPECT_TRUE((comp[0] == (GLint)ired && comp[1] == (GLint)iblue) || (comp[1] == (GLint)ired && comp[0] == (GLint)iblue));
    GLint n = 0; glGetProgramInterfaceiv(p, GL_FRAGMENT_SUBROUTINE_UNIFORM, GL_ACTIVE_RESOURCES, &n); EXPECT_EQ_INT(n, 2);
    glGetProgramInterfaceiv(p, GL_FRAGMENT_SUBROUTINE_UNIFORM, GL_MAX_NUM_COMPATIBLE_SUBROUTINES, &n); EXPECT_EQ_INT(n, 2);
    EXPECT_EQ_INT(glGetProgramResourceIndex(p, GL_FRAGMENT_SUBROUTINE, "blue"), 5);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_FRAGMENT_SUBROUTINE_UNIFORM, "scale"), ls);
    /* hidden selector uniforms are not reported */
    GLint act = 0; glGetProgramInterfaceiv(p, GL_UNIFORM, GL_ACTIVE_RESOURCES, &act); EXPECT_EQ_INT(act, 0);
    EXPECT_NO_GL_ERROR();

    setup_tri(); glUseProgram(p);
    GLuint sel[2];
    sel[lp] = ired; sel[ls] = ione;
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 2, sel); EXPECT_NO_GL_ERROR();
    draw_clear(); EXPECT_PIXEL(10, 10, 255, 0, 0, 255, 2);
    sel[lp] = iblue; sel[ls] = ihalf;
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 2, sel);
    draw_clear(); EXPECT_PIXEL(10, 10, 0, 0, 128, 255, 3);
    GLuint got = 0; glGetUniformSubroutineuiv(GL_FRAGMENT_SHADER, lp, &got); EXPECT_EQ_INT(got, iblue);
    /* errors */
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, sel); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    sel[lp] = ione; glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 2, sel); EXPECT_GL_ERROR(GL_INVALID_VALUE); /* incompatible */
    glGetUniformSubroutineuiv(GL_FRAGMENT_SHADER, 7, &got); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetProgramStageiv(p, GL_TEXTURE_2D, GL_ACTIVE_SUBROUTINES, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetActiveSubroutineName(p, GL_FRAGMENT_SHADER, 77, sizeof nm, &len, nm); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUseProgram(0);
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 2, sel); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
}

/* ---------------------------------------------------------------- doubles */

TEST(double_uniforms)
{
    const char *cs = "#version 450 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer O { float r; } o;\n"
        "uniform double d;\n"
        "uniform dvec2 d2;\n"
        "uniform dmat2 dm;\n"
        "void main(){ o.r = float(d + d2.y + dm[1][0]); }\n";
    GLuint p = make_cs(cs);
    if (!p) return;
    GLint ld = glGetUniformLocation(p, "d"), ld2 = glGetUniformLocation(p, "d2"), ldm = glGetUniformLocation(p, "dm");
    EXPECT_TRUE(ld >= 0 && ld2 >= 0 && ldm >= 0);
    GLint sz; GLenum ty;
    find_uniform(p, "dm", &sz, &ty); EXPECT_EQ_INT(ty, GL_DOUBLE_MAT2);
    glProgramUniform1d(p, ld, 1.0 / 3.0);
    glProgramUniform2d(p, ld2, 1e300, -2.0);
    const double m[4] = { 1, 2, 3, 4 };
    glProgramUniformMatrix2dv(p, ldm, 1, GL_TRUE, m);
    EXPECT_NO_GL_ERROR();
    GLdouble r[4] = { 0 };
    glGetUniformdv(p, ld, r); EXPECT_TRUE(r[0] == 1.0 / 3.0); /* full double precision kept */
    glGetUniformdv(p, ld2, r); EXPECT_TRUE(r[0] == 1e300);
    glGetUniformdv(p, ldm, r); EXPECT_NEAR(r[1], 3, 0); EXPECT_NEAR(r[2], 2, 0);
    float f[2]; glGetUniformfv(p, ld2, f); EXPECT_NEAR(f[1], -2.0, 0);
    glProgramUniform1f(p, ld, 1.0f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* the GPU stores fp32 (Metal has no fp64): a dispatch must see the converted values */
    GLuint cbo;
    glGenBuffers(1, &cbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_DYNAMIC_COPY);
    glUseProgram(p);
    glProgramUniform1d(p, ld, 0.25);
    glProgramUniform2d(p, ld2, 2.0, 4.0);
    const double m2[4] = { 1, 2, 3, 4 };
    glProgramUniformMatrix2dv(p, ldm, 1, GL_TRUE, m2);
    EXPECT_NO_GL_ERROR();
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    float out = 0;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &out);
    /* d + d2.y + dm[1][0]: transpose=TRUE lays dm out as rows (1,2)/(3,4), so dm[1][0] = 2 */
    EXPECT_NEAR(out, 0.25f + 4.0f + 2.0f, 0.0005);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(draw_uniforms_vs_fs);
    RUN(draw_explicit_uniform_locations_vs);
    RUN(draw_bool_uniforms);
    RUN(n_uniform_bufsize);
    RUN(default_block_reflection);
    RUN(uniform_round_trip);
    RUN(uniform_errors);
    RUN(program_uniform_non_current);
    RUN(uniform_initializer);
    RUN(ubo_reflection);
    RUN(uniform_block_binding_draw);
    RUN(ssbo_reflection_and_binding);
    RUN(bind_attrib_location);
    RUN(bind_frag_data_location);
    RUN(resource_queries);
    RUN(sampler_unit_uniform);
    RUN(subroutines);
    RUN(double_uniforms);
    return mgl_test_summary();
}
