/* geom r107 probes, round 2: GS output struct with aom member written through
 * a helper; xfb of an array-of-struct with an aom member. */
#include "mgl_test.h"

static GLuint mk_prog3(const char *vs, const char *gs, const char *fs, const char *tag)
{
    GLuint prog = glCreateProgram();
    struct { GLenum t; const char *s; } sh[3] = {
        { GL_VERTEX_SHADER, vs }, { GL_GEOMETRY_SHADER, gs }, { GL_FRAGMENT_SHADER, fs } };
    for (int i = 0; i < 3; i++)
    {
        GLuint h = glCreateShader(sh[i].t);
        glShaderSource(h, 1, &sh[i].s, NULL);
        glCompileShader(h);
        GLint ok = 0;
        glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
        if (ok != 1)
        {
            char log[2048]; GLsizei len = 0;
            glGetShaderInfoLog(h, sizeof log, &len, log);
            fprintf(stderr, "  [%s] stage %d compile failed:\n%s\n", tag, i, log);
        }
        glAttachShader(prog, h);
        glDeleteShader(h);
    }
    glLinkProgram(prog);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [%s] LINK failed:\n%s\n", tag, log);
    }
    return prog;
}

TEST(gs_aom_struct_func)
{
    static const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "    vec2 p = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));\n"
        "    gl_Position = vec4(p * 4.0 - 2.0, 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "struct Out { mat2x3 m[2]; float f; };\n"
        "layout(location = 0) flat out Out vo;\n"
        "void emit_at(vec2 c, vec4 pos) {\n"
        "    gl_Position = pos;\n"
        "    EmitVertex();\n"
        "}\n"
        "void fill(out Out o, float s) {\n"
        "    o.m[0] = mat2x3(1,2,3,4,5,6) * s;\n"
        "    o.m[1] = mat2x3(7,8,9,10,11,12) * s;\n"
        "    o.f = s;\n"
        "}\n"
        "void main() {\n"
        "    fill(vo, 0.5);\n"
        "    emit_at(vec2(0), vec4(-1,-1,0,1));\n"
        "    emit_at(vec2(1,0), vec4(1,-1,0,1));\n"
        "    emit_at(vec2(0,1), vec4(-1,1,0,1));\n"
        "    emit_at(vec2(1,1), vec4(1,1,0,1));\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Out { mat2x3 m[2]; float f; };\n"
        "layout(location = 0) flat in Out i;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(i.f, i.m[0][0].x, i.m[1][1].y * 0.1, 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "gs_aom_struct_func");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1) { glDeleteProgram(prog); EXPECT_EQ_INT(lc, 1); return; }

    glUseProgram(prog);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    GLubyte px[4];
    mgl_read_rgba(8, 8, px);
    /* fill(vo, 0.5): f = 0.5, m[0] col0.x = 0.5, m[1] col1.y = 5.5 -> 0.55 */
    const GLubyte expect[4] = { 128, 128, 140, 255 };
    for (int c = 0; c < 4; c++)
    {
        mgl_test_checks++;
        if (abs((int)px[c] - (int)expect[c]) > 2)
            MGL_FAIL("gs_aom_struct_func: channel %d == %d, expected ~%d", c, px[c], expect[c]);
    }
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

TEST(aos_xfb)
{
    static const char *vs =
        "#version 460 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(xfb_buffer=0, xfb_offset=0, location=0) flat out Inner o[2];\n"
        "void fill(out Inner x, float s) {\n"
        "    x.m[0] = mat2x3(1,2,3,4,5,6) * s;\n"
        "    x.m[1] = mat2x3(7,8,9,10,11,12) * s;\n"
        "    x.f = s;\n"
        "}\n"
        "void main() {\n"
        "    gl_Position = vec4(float(gl_VertexID & 1) * 4.0 - 1.0, float(gl_VertexID >> 1) * 4.0 - 1.0, 0.0, 1.0);\n"
        "    fill(o[0], 1.0);\n"
        "    fill(o[1], 0.5);\n"
        "}\n";
    GLuint prog = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &vs, NULL);
    glCompileShader(sh);
    glAttachShader(prog, sh);
    glDeleteShader(sh);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (ok != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [aos_xfb] LINK failed:\n%s\n", log);
        glDeleteProgram(prog);
        EXPECT_EQ_INT(ok, 1);
        return;
    }
    GLint stride = 0;
    {
        GLenum props[1] = { GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE };
        glGetProgramResourceiv(prog, GL_TRANSFORM_FEEDBACK_BUFFER, 0, 1, props, 1, NULL, &stride);
        /* two elements x 52 bytes (4 packed vec3 columns + f), the same
         * packing GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE reports */
        EXPECT_EQ_INT(stride, 104);
    }
    GLuint vao = 0, buf = 0, tfb = 0, q = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4 * 256, NULL, GL_STREAM_READ);
    glGenTransformFeedbacks(1, &tfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glGenQueries(1, &q);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 2);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    /* each record: element 0 = fill(o[0],1.0) -> m[0]=(1..6) m[1]=(7..12) f=1;
     * element 1 = fill(o[1],0.5) -> (0.5..3) (3.5..6) f=0.5; the element
     * members pack back to back (12 bytes per vec3 column, f at +48) */
    static const float expect[26] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1,
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 0.5f
    };
    float got[64];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof got, got);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    for (int v = 0; v < 2; v++)
        for (int w = 0; w < 26; w++)
        {
            mgl_test_checks++;
            if (fabsf(got[v * 26 + w] - expect[w]) > 0.002f)
                MGL_FAIL("aos_xfb: vertex %d word %d == %f, expected %f",
                         v, w, got[v * 26 + w], expect[w]);
        }
    glDeleteTransformFeedbacks(1, &tfb);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &buf);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(gs_aom_struct_func);
    RUN(aos_xfb);
    return mgl_test_summary();
}
