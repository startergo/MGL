#include "mgl_test.h"

static const char *fs_src =
    "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vec4(1.0); }\n";

static const char *vs_src =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out vec3 vcolor;\n"
    "layout(location=1) out float vid;\n"
    "void main(){\n"
    "  gl_Position = vec4(pos, 0.25, 1.0);\n"
    "  vcolor = vec3(pos.x * 2.0, pos.y + 10.0, float(gl_VertexID));\n"
    "  vid = float(gl_VertexID) + 100.0;\n"
    "}\n";

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

/* link VS(+FS) after setting varyings */
static GLuint make_xfb_program(const char *vs, const char *fs, int n, const char **vars, GLenum mode)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    if (fs) glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    if (n) glTransformFeedbackVaryings(p, n, vars, mode);
    glLinkProgram(p);
    return p;
}

static GLuint vao, vbo;
static const float verts[] = { -1.0f, -1.0f,   1.0f, -1.0f,   -1.0f, 1.0f,   1.0f, 1.0f,   0.5f, 0.5f };

static void setup_geometry(void)
{
    if (vao) { glBindVertexArray(vao); return; }
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static GLuint make_buffer(size_t size)
{
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, b);
    float *junk = malloc(size);
    for (size_t i = 0; i < size / 4; i++) junk[i] = -12345.0f;
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, size, junk, GL_DYNAMIC_READ);
    free(junk);
    return b;
}

static void read_buffer(GLuint b, size_t size, float *out)
{
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, b);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, size, out);
}

TEST(interleaved_capture)
{
    const char *vars[] = { "gl_Position", "vcolor" };
    GLuint p = make_xfb_program(vs_src, fs_src, 2, vars, GL_INTERLEAVED_ATTRIBS);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_TRUE);
    GLint n = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n); EXPECT_EQ_INT(n, 2);
    GLint mode = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &mode); EXPECT_EQ_INT(mode, GL_INTERLEAVED_ATTRIBS);
    GLint maxlen = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &maxlen); EXPECT_EQ_INT(maxlen, 12);

    char name[64]; GLsizei len = 0, size = 0; GLenum type = 0;
    glGetTransformFeedbackVarying(p, 1, sizeof name, &len, &size, &type, name);
    EXPECT_STREQ(name, "vcolor"); EXPECT_EQ_INT(len, 6); EXPECT_EQ_INT(size, 1); EXPECT_EQ_INT(type, GL_FLOAT_VEC3);
    glGetTransformFeedbackVarying(p, 0, sizeof name, &len, &size, &type, name);
    EXPECT_STREQ(name, "gl_Position"); EXPECT_EQ_INT(type, GL_FLOAT_VEC4);
    EXPECT_NO_GL_ERROR();

    setup_geometry();
    GLuint tfb = make_buffer(3 * 7 * 4);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float out[21];
    read_buffer(tfb, sizeof out, out);
    for (int v = 0; v < 3; v++) {
        const float *r = out + v * 7;
        EXPECT_NEAR(r[0], verts[v * 2], 1e-6); EXPECT_NEAR(r[1], verts[v * 2 + 1], 1e-6);
        EXPECT_NEAR(r[2], 0.25, 1e-6); EXPECT_NEAR(r[3], 1.0, 1e-6);
        EXPECT_NEAR(r[4], verts[v * 2] * 2.0, 1e-6); EXPECT_NEAR(r[5], verts[v * 2 + 1] + 10.0, 1e-6);
        EXPECT_NEAR(r[6], v, 1e-6);
    }
    EXPECT_NO_GL_ERROR();
}


/* simple program: position + vertex id (+ instance id) */
static const char *vs_id_src =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out float vid;\n"
    "layout(location=1) out int iid;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); vid = float(gl_VertexID); iid = gl_InstanceID * 10 - 3; }\n";

static GLuint id_prog(void)
{
    static GLuint p;
    if (!p) { const char *v[] = { "vid" }; p = make_xfb_program(vs_id_src, fs_src, 1, v, GL_INTERLEAVED_ATTRIBS); }
    return p;
}

static void capture_vids(GLenum tfmode, GLenum mode, int count, float *out, int nout)
{
    GLuint tfb = make_buffer(nout * 4);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glUseProgram(id_prog());
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(tfmode);
    glDrawArrays(mode, 0, count);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    read_buffer(tfb, nout * 4, out);
}

TEST(triangle_strip_expansion)
{
    setup_geometry();
    float out[10];
    capture_vids(GL_TRIANGLES, GL_TRIANGLE_STRIP, 5, out, 10);
    const float exp[9] = { 0,1,2,  2,1,3,  2,3,4 };
    for (int i = 0; i < 9; i++) EXPECT_NEAR(out[i], exp[i], 0);
    EXPECT_NEAR(out[9], -12345.0, 0);
    EXPECT_NO_GL_ERROR();
}

TEST(triangle_fan_and_line_loop)
{
    setup_geometry();
    float out[10];
    capture_vids(GL_TRIANGLES, GL_TRIANGLE_FAN, 5, out, 10);
    const float fan[9] = { 0,1,2,  0,2,3,  0,3,4 };
    for (int i = 0; i < 9; i++) EXPECT_NEAR(out[i], fan[i], 0);

    float lo[7];
    capture_vids(GL_LINES, GL_LINE_LOOP, 3, lo, 7);
    const float loop[6] = { 0,1, 1,2, 2,0 };
    for (int i = 0; i < 6; i++) EXPECT_NEAR(lo[i], loop[i], 0);
    EXPECT_NEAR(lo[6], -12345.0, 0);

    float ls[4];
    capture_vids(GL_LINES, GL_LINE_STRIP, 3, ls, 4);
    const float strip[4] = { 0,1, 1,2 };
    for (int i = 0; i < 4; i++) EXPECT_NEAR(ls[i], strip[i], 0);
    EXPECT_NO_GL_ERROR();
}

TEST(separate_attribs)
{
    const char *vars[] = { "gl_Position", "vid" };
    GLuint p = make_xfb_program(vs_src, fs_src, 2, vars, GL_SEPARATE_ATTRIBS);
    GLint mode = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &mode); EXPECT_EQ_INT(mode, GL_SEPARATE_ATTRIBS);
    setup_geometry();
    GLuint b0 = make_buffer(3 * 16), b1 = make_buffer(3 * 4 + 4);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b0);
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 1, b1, 4, 12);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    float o0[12], o1[4];
    read_buffer(b0, sizeof o0, o0);
    read_buffer(b1, sizeof o1, o1);
    for (int v = 0; v < 3; v++) {
        EXPECT_NEAR(o0[v * 4 + 0], verts[v * 2], 0);
        EXPECT_NEAR(o0[v * 4 + 1], verts[v * 2 + 1], 0);
        EXPECT_NEAR(o0[v * 4 + 2], 0.25, 0);
        EXPECT_NEAR(o0[v * 4 + 3], 1.0, 0);
    }
    EXPECT_NEAR(o1[0], -12345.0, 0);     /* before the bound range */
    EXPECT_NEAR(o1[1], 100.0, 0);
    EXPECT_NEAR(o1[2], 101.0, 0);
    EXPECT_NEAR(o1[3], 102.0, 0);
}

TEST(skip_components_and_next_buffer)
{
    const char *vars[] = { "vid", "gl_SkipComponents2", "gl_NextBuffer", "vcolor" };
    GLuint p = make_xfb_program(vs_src, fs_src, 4, vars, GL_INTERLEAVED_ATTRIBS);
    GLint n = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n); EXPECT_EQ_INT(n, 4);
    char name[64]; GLsizei size = -1; GLenum type = 1;
    glGetTransformFeedbackVarying(p, 1, sizeof name, NULL, &size, &type, name);
    EXPECT_STREQ(name, "gl_SkipComponents2"); EXPECT_EQ_INT(size, 2); EXPECT_EQ_INT(type, GL_NONE);
    glGetTransformFeedbackVarying(p, 2, sizeof name, NULL, &size, &type, name);
    EXPECT_STREQ(name, "gl_NextBuffer"); EXPECT_EQ_INT(size, 0); EXPECT_EQ_INT(type, GL_NONE);

    setup_geometry();
    GLuint b0 = make_buffer(2 * 12), b1 = make_buffer(2 * 12);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, b1);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 1, 2);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    float o0[6], o1[6];
    read_buffer(b0, sizeof o0, o0);
    read_buffer(b1, sizeof o1, o1);
    EXPECT_NEAR(o0[0], 101.0, 0); EXPECT_NEAR(o0[1], -12345.0, 0); EXPECT_NEAR(o0[2], -12345.0, 0);
    EXPECT_NEAR(o0[3], 102.0, 0);
    EXPECT_NEAR(o1[0], 2.0, 0); EXPECT_NEAR(o1[1], 9.0, 0); EXPECT_NEAR(o1[2], 1.0, 0);
    EXPECT_NEAR(o1[3], -2.0, 0); EXPECT_NEAR(o1[4], 11.0, 0); EXPECT_NEAR(o1[5], 2.0, 0);
}

TEST(shader_xfb_qualifiers)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0, xfb_buffer=0, xfb_offset=4, xfb_stride=20) out vec2 p2;\n"
        "layout(location=1, xfb_buffer=0, xfb_offset=12) out float k;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); p2 = pos * 3.0; k = 7.0; }\n";
    GLuint p = make_xfb_program(vs, fs_src, 0, NULL, GL_INTERLEAVED_ATTRIBS);
    GLint n = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n); EXPECT_EQ_INT(n, 2);
    setup_geometry();
    GLuint b = make_buffer(2 * 20);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 2);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    float o[10];
    read_buffer(b, sizeof o, o);
    EXPECT_NEAR(o[0], -12345.0, 0); EXPECT_NEAR(o[1], -3.0, 0); EXPECT_NEAR(o[2], -3.0, 0); EXPECT_NEAR(o[3], 7.0, 0);
    EXPECT_NEAR(o[4], -12345.0, 0);
    EXPECT_NEAR(o[6], 3.0, 0); EXPECT_NEAR(o[7], -3.0, 0); EXPECT_NEAR(o[8], 7.0, 0);
}

TEST(indexed_instanced_and_restart)
{
    const char *vars[] = { "vid", "iid" };
    GLuint p = make_xfb_program(vs_id_src, fs_src, 2, vars, GL_INTERLEAVED_ATTRIBS);
    char name[16]; GLenum type = 0; GLsizei size = 0;
    glGetTransformFeedbackVarying(p, 1, sizeof name, NULL, &size, &type, name);
    EXPECT_EQ_INT(type, GL_INT);
    setup_geometry();
    static const GLushort idx[] = { 4, 2, 0xFFFF, 1, 3, 0 };
    GLuint ebo; glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glVertexArrayElementBuffer(vao, ebo);  /* BindBuffer(ELEMENT_ARRAY) does not reach the VAO yet */
    GLuint b = make_buffer(16 * 8);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    glBeginTransformFeedback(GL_LINES);
    glDrawElementsInstanced(GL_LINE_STRIP, 6, GL_UNSIGNED_SHORT, 0, 2);
    glEndTransformFeedback();
    glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    /* strips {4,2} and {1,3,0}: lines (4,2) (1,3) (3,0), per instance */
    struct { float vid; int iid; } o[16];
    read_buffer(b, sizeof o, (float *)o);
    const float ev[6] = { 4,2, 1,3, 3,0 };
    for (int inst = 0; inst < 2; inst++)
        for (int i = 0; i < 6; i++) {
            EXPECT_NEAR(o[inst * 6 + i].vid, ev[i], 0);
            EXPECT_EQ_INT(o[inst * 6 + i].iid, inst * 10 - 3);
        }
    EXPECT_NEAR(o[12].vid, -12345.0, 0);
    glBindVertexArray(0);
}

TEST(pause_resume)
{
    setup_geometry();
    GLuint tfo; glGenTransformFeedbacks(1, &tfo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    GLuint b = make_buffer(8 * 4);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    glUseProgram(id_prog());
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 2);
    glPauseTransformFeedback();
    GLint v = -1;
    glGetTransformFeedbackiv(tfo, GL_TRANSFORM_FEEDBACK_PAUSED, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glGetIntegerv(GL_TRANSFORM_FEEDBACK_PAUSED, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glDrawArrays(GL_POINTS, 2, 2);          /* not captured */
    glDrawArrays(GL_TRIANGLES, 0, 3);       /* mode mismatch is fine while paused */
    EXPECT_NO_GL_ERROR();
    glResumeTransformFeedback();
    glDrawArrays(GL_POINTS, 4, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    float o[4];
    read_buffer(b, sizeof o, o);
    EXPECT_NEAR(o[0], 0, 0); EXPECT_NEAR(o[1], 1, 0); EXPECT_NEAR(o[2], 4, 0); EXPECT_NEAR(o[3], -12345.0, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfo);
    EXPECT_NO_GL_ERROR();
}

TEST(overflow_and_draw_transform_feedback)
{
    const char *pv[] = { "gl_Position" };
    GLuint p1 = make_xfb_program(vs_src, fs_src, 1, pv, GL_INTERLEAVED_ATTRIBS);
    setup_geometry();

    GLuint tfo[2]; glGenTransformFeedbacks(2, tfo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo[0]);
    /* room for 1.5 triangles of vec4: only the first triangle fits */
    GLuint a = make_buffer(4 * 16 + 8);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, a);
    glUseProgram(p1);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  /* 2 triangles needed */
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    float oa[18];
    read_buffer(a, sizeof oa, oa);
    for (int v = 0; v < 3; v++) { EXPECT_NEAR(oa[v * 4], verts[v * 2], 0); EXPECT_NEAR(oa[v * 4 + 1], verts[v * 2 + 1], 0); }
    EXPECT_NEAR(oa[12], -12345.0, 0);       /* second triangle not written */

    /* re-draw what tfo[0] captured (3 vertices) through a second capture */
    const char *vs2 =
        "#version 450 core\n"
        "layout(location=0) in vec4 p;\n"
        "layout(location=0) out vec4 q;\n"
        "void main(){ gl_Position = p; q = p * 2.0 + vec4(0.0, 0.0, 0.0, float(gl_VertexID)); }\n";
    const char *qv[] = { "q" };
    GLuint p2 = make_xfb_program(vs2, fs_src, 1, qv, GL_INTERLEAVED_ATTRIBS);
    GLuint vao2; glGenVertexArrays(1, &vao2); glBindVertexArray(vao2);
    glBindBuffer(GL_ARRAY_BUFFER, a);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(0);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo[1]);
    GLuint b = make_buffer(4 * 16);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    glUseProgram(p2);
    glBeginTransformFeedback(GL_POINTS);
    glDrawTransformFeedback(GL_POINTS, tfo[0]);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    float ob[16];
    read_buffer(b, sizeof ob, ob);
    for (int v = 0; v < 3; v++) {
        EXPECT_NEAR(ob[v * 4 + 0], verts[v * 2] * 2.0, 0);
        EXPECT_NEAR(ob[v * 4 + 1], verts[v * 2 + 1] * 2.0, 0);
        EXPECT_NEAR(ob[v * 4 + 3], 2.0 + v, 0);
    }
    EXPECT_NEAR(ob[12], -12345.0, 0);       /* exactly 3 vertices drawn */

    /* instanced + stream variants (stream 1 captured nothing: no draw) */
    glDrawTransformFeedbackStream(GL_POINTS, tfo[0], 1);
    glDrawTransformFeedbackInstanced(GL_POINTS, tfo[0], 2);
    glDrawTransformFeedbackStreamInstanced(GL_POINTS, tfo[0], 0, 1);
    EXPECT_NO_GL_ERROR();
    glDrawTransformFeedbackStream(GL_POINTS, tfo[0], 4);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glBindVertexArray(vao);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(2, tfo);
    EXPECT_NO_GL_ERROR();
}

TEST(uniforms_and_rasterization_on)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(std140, binding=2) uniform Params { vec4 scale; };\n"
        "layout(location=0) out vec2 s;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); s = pos * scale.xy; }\n";
    const char *sv[] = { "s" };
    GLuint p = make_xfb_program(vs, fs_src, 1, sv, GL_INTERLEAVED_ATTRIBS);
    setup_geometry();
    static const float scale[4] = { 3.0f, 5.0f, 0, 0 };
    GLuint ubo; glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof scale, scale, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, ubo);
    GLuint b = make_buffer(3 * 8);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    glUseProgram(p);
    /* rasterizer discard off: the normal draw still happens after the capture */
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glFinish();
    EXPECT_NO_GL_ERROR();
    float o[6];
    read_buffer(b, sizeof o, o);
    for (int v = 0; v < 3; v++) {
        EXPECT_NEAR(o[v * 2], verts[v * 2] * 3.0, 0);
        EXPECT_NEAR(o[v * 2 + 1], verts[v * 2 + 1] * 5.0, 0);
    }
}

TEST(object_state_queries)
{
    GLuint t[2];
    glGenTransformFeedbacks(2, t);
    EXPECT_TRUE(glIsTransformFeedback(t[0]) == GL_FALSE);   /* not bound yet */
    GLuint c; glCreateTransformFeedbacks(1, &c);
    EXPECT_TRUE(glIsTransformFeedback(c) == GL_TRUE);
    EXPECT_TRUE(glIsTransformFeedback(0) == GL_FALSE);

    GLint v = -1;
    glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &v); EXPECT_EQ_INT(v, 0);
    GLuint b1 = make_buffer(64), b2 = make_buffer(64);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, t[0]);
    EXPECT_TRUE(glIsTransformFeedback(t[0]) == GL_TRUE);
    glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &v); EXPECT_EQ_INT(v, t[0]);
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 1, b1, 8, 32);
    glGetIntegerv(GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, &v); EXPECT_EQ_INT(v, b1);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, b2);
    glTransformFeedbackBufferRange(t[0], 2, b2, 16, 16);      /* DSA on an unbound object */
    glTransformFeedbackBufferBase(c, 0, b1);
    EXPECT_NO_GL_ERROR();

    glGetTransformFeedbacki_v(t[0], GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, b1);
    glGetTransformFeedbacki_v(t[0], GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 2, &v); EXPECT_EQ_INT(v, b2);
    glGetTransformFeedbacki_v(0, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, b2);
    glGetTransformFeedbacki_v(c, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 0, &v); EXPECT_EQ_INT(v, b1);
    GLint64 s = -1;
    glGetTransformFeedbacki64_v(t[0], GL_TRANSFORM_FEEDBACK_BUFFER_START, 1, &s); EXPECT_EQ_INT(s, 8);
    glGetTransformFeedbacki64_v(t[0], GL_TRANSFORM_FEEDBACK_BUFFER_SIZE, 1, &s); EXPECT_EQ_INT(s, 32);
    glGetTransformFeedbacki64_v(t[0], GL_TRANSFORM_FEEDBACK_BUFFER_START, 2, &s); EXPECT_EQ_INT(s, 16);
    glGetTransformFeedbacki64_v(0, GL_TRANSFORM_FEEDBACK_BUFFER_SIZE, 1, &s); EXPECT_EQ_INT(s, 0);   /* Base binding */
    glGetTransformFeedbackiv(t[0], GL_TRANSFORM_FEEDBACK_ACTIVE, &v); EXPECT_EQ_INT(v, GL_FALSE);
    EXPECT_NO_GL_ERROR();

    /* bindings follow the object */
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, t[0]);
    GLint64 s2 = -1;
    glGetTransformFeedbacki64_v(0, GL_TRANSFORM_FEEDBACK_BUFFER_SIZE, 1, &s2);
    glGetTransformFeedbacki_v(t[0], GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, b1);
    glGetTransformFeedbacki_v(0, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, b2);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_BUFFERS, &v); EXPECT_EQ_INT(v, 4);
    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS, &v); EXPECT_TRUE(v >= 64);
    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS, &v); EXPECT_EQ_INT(v, 4);
    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS, &v); EXPECT_EQ_INT(v, 4);
    glGetIntegerv(GL_MAX_VERTEX_STREAMS, &v); EXPECT_EQ_INT(v, 4);
    glGetIntegerv(GL_TRANSFORM_FEEDBACK_ACTIVE, &v); EXPECT_EQ_INT(v, GL_FALSE);
    EXPECT_NO_GL_ERROR();

    glDeleteTransformFeedbacks(2, t);
    glDeleteTransformFeedbacks(1, &c);
    EXPECT_TRUE(glIsTransformFeedback(t[0]) == GL_FALSE);
    EXPECT_NO_GL_ERROR();
}

TEST(error_cases)
{
    /* TransformFeedbackVaryings */
    GLuint p = glCreateProgram();
    const char *v1[] = { "vcolor" };
    glTransformFeedbackVaryings(p, 1, v1, GL_RGBA);             EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTransformFeedbackVaryings(p, -1, v1, GL_INTERLEAVED_ATTRIBS); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    const char *v5[] = { "a", "b", "c", "d", "e" };
    glTransformFeedbackVaryings(p, 5, v5, GL_SEPARATE_ATTRIBS); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTransformFeedbackVaryings(p + 1000, 1, v1, GL_INTERLEAVED_ATTRIBS); EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* unknown varying fails the link */
    const char *bad[] = { "does_not_exist" };
    GLuint pb = make_xfb_program(vs_src, fs_src, 1, bad, GL_INTERLEAVED_ATTRIBS);
    GLint ok = 1; glGetProgramiv(pb, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_FALSE);
    const char *dup[] = { "vid", "vid" };
    GLuint pd = make_xfb_program(vs_src, fs_src, 2, dup, GL_INTERLEAVED_ATTRIBS);
    glGetProgramiv(pd, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_FALSE);
    const char *sep[] = { "vid", "gl_NextBuffer" };
    GLuint ps = make_xfb_program(vs_src, fs_src, 2, sep, GL_SEPARATE_ATTRIBS);
    glGetProgramiv(ps, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_FALSE);
    glGetTransformFeedbackVarying(id_prog(), 5, 0, NULL, NULL, NULL, NULL); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    EXPECT_NO_GL_ERROR();

    /* objects */
    glBindTransformFeedback(GL_ARRAY_BUFFER, 0);                EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 4242);       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGenTransformFeedbacks(-1, NULL);                          EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLint iv;
    glGetTransformFeedbackiv(4242, GL_TRANSFORM_FEEDBACK_ACTIVE, &iv); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetTransformFeedbackiv(0, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, &iv); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetTransformFeedbacki_v(0, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, 4, &iv); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLuint never; glCreateTransformFeedbacks(1, &never);
    glDrawTransformFeedback(GL_POINTS, never);                  EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawTransformFeedback(GL_POINTS, 4242);                   EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* begin / pause / resume / end */
    glPauseTransformFeedback();                                 EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glResumeTransformFeedback();                                EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndTransformFeedback();                                   EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUseProgram(id_prog());
    glBeginTransformFeedback(GL_LINE_STRIP);                    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    GLuint t; glCreateTransformFeedbacks(1, &t);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, t);
    glBeginTransformFeedback(GL_POINTS);                        EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* no buffer */
    GLuint b = make_buffer(64);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 4, b);       EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b, 2, 16); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    EXPECT_NO_GL_ERROR();

    GLuint plain = mgl_make_program(vs_id_src, fs_src);
    glUseProgram(plain);
    glBeginTransformFeedback(GL_POINTS);                        EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* no varyings */
    glUseProgram(id_prog());

    setup_geometry();
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_NO_GL_ERROR();
    glBeginTransformFeedback(GL_POINTS);                        EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);          EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTransformFeedbackBufferBase(t, 0, b);                     EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawArrays(GL_TRIANGLES, 0, 3);                           EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* mode mismatch */
    glDeleteTransformFeedbacks(1, &t);                          EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTransformFeedbackVaryings(p, 1, v1, GL_INTERLEAVED_ATTRIBS); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glPauseTransformFeedback();
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);          /* allowed while paused */
    EXPECT_NO_GL_ERROR();
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, t);
    glUseProgram(plain);
    glResumeTransformFeedback();                                EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* program changed */
    glUseProgram(id_prog());
    glResumeTransformFeedback();
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &t);
    EXPECT_NO_GL_ERROR();
}

TEST(map_buffer_readback)
{
    GLuint b = make_buffer(3 * 4);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);
    setup_geometry();
    glUseProgram(id_prog());
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 2, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, b);
    const float *m = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 12, GL_MAP_READ_BIT);
    EXPECT_TRUE(m != NULL);
    if (m) { EXPECT_NEAR(m[0], 2, 0); EXPECT_NEAR(m[1], 3, 0); EXPECT_NEAR(m[2], 4, 0); }
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(interleaved_capture);
    RUN(triangle_strip_expansion);
    RUN(triangle_fan_and_line_loop);
    RUN(separate_attribs);
    RUN(skip_components_and_next_buffer);
    RUN(shader_xfb_qualifiers);
    RUN(indexed_instanced_and_restart);
    RUN(pause_resume);
    RUN(overflow_and_draw_transform_feedback);
    RUN(uniforms_and_rasterization_on);
    RUN(object_state_queries);
    RUN(error_cases);
    RUN(map_buffer_readback);
    return mgl_test_summary();
}
