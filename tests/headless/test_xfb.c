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

int main(void)
{
    RUN(interleaved_capture);
    return mgl_test_summary();
}
