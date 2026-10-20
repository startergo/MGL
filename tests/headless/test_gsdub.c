/* Default-block uniform ARRAYS in the geometry stage (CTS geometry_shader.limits.
 * max_uniform_components shape): the kernel re-declares the stage's default uniforms
 * as a std430 block laid out from introspect's member offsets, so every element must
 * arrive (a shadowed slot or a wrong stride reads 0) — verified through transform
 * feedback, and once through the raster path. */
#include <stdio.h>
#include <stdlib.h>
#include "glcorearb.h"
#include "mgl_test.h"

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";

static const char *kGsFmt =
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices=1) out;\n"
    "uniform ivec4 uni_array[%d];\n"
    "flat out uint gs_out_sum;\n"
    "void main(){\n"
    "    gs_out_sum = 0u;\n"
    "    for (int i = 0; i < %d; ++i) {\n"
    "        gs_out_sum += uint(uni_array[i].x);\n"
    "        gs_out_sum += uint(uni_array[i].y);\n"
    "        gs_out_sum += uint(uni_array[i].z);\n"
    "        gs_out_sum += uint(uni_array[i].w);\n"
    "    }\n"
    "    EmitVertex();\n"
    "    EndPrimitive();\n"
    "}\n";

static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ c0 = vec4(0); }\n";

/* scalar + array mixed: the array suffix parse and the DUB offsets both matter */
static const char *kGsMixed =
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(triangle_strip, max_vertices=4) out;\n"
    "uniform ivec4 uni_sv;\n"
    "uniform ivec4 uni_arr[4];\n"
    "layout(location=0) out float g_s;\n"
    "void main(){\n"
    "    int s = uni_sv.x + uni_sv.y + uni_sv.z + uni_sv.w;\n"
    "    for (int i = 0; i < 4; i++) s += uni_arr[i].x;\n"
    "    g_s = float(s);\n"
    "    gl_Position = vec4(-0.5, -0.5, 0, 1); EmitVertex();\n"
    "    gl_Position = vec4( 0.5, -0.5, 0, 1); EmitVertex();\n"
    "    gl_Position = vec4(-0.5,  0.5, 0, 1); EmitVertex();\n"
    "    gl_Position = vec4( 0.5,  0.5, 0, 1); EmitVertex();\n"
    "    EndPrimitive();\n"
    "}\n";

static const char *kFsPassthru =
    "#version 450 core\n"
    "layout(location=0) in float g_s;\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ c0 = vec4(g_s / 40.0, 0, 0, 1); }\n";

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

static GLuint link(GLuint vs, GLuint gs, GLuint fs, const char *xfb, char *log, size_t logsz)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, gs);
    glAttachShader(p, fs);
    if (xfb)
    {
        const char *vars[2] = { xfb, NULL };
        glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok && log && logsz) glGetProgramInfoLog(p, (GLsizei)logsz, NULL, log);
    return p;
}

static GLuint make_xfb_draw_state(GLuint *vao, GLuint *vbo, GLuint *tbuf)
{
    glGenVertexArrays(1, vao);
    glGenBuffers(1, vbo);
    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    float pt[2] = { 0, 0 };
    glBufferData(GL_ARRAY_BUFFER, sizeof pt, pt, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, *tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(GLuint) * 4, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *tbuf);
    return 0;
}

static void run_case(int nv)
{
    char gs[4096], log[4096];
    snprintf(gs, sizeof gs, kGsFmt, nv, nv);

    GLuint p = link(compile(GL_VERTEX_SHADER, kVs),
                    compile(GL_GEOMETRY_SHADER, gs),
                    compile(GL_FRAGMENT_SHADER, kFs),
                    "gs_out_sum", log, sizeof log);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("NV=%d link: %s", nv, log); return; }

    glUseProgram(p);
    GLint loc = glGetUniformLocation(p, "uni_array");
    EXPECT_EQ_INT(loc, 0);
    GLint want = 0;
    GLint *data = calloc((size_t)nv * 4, sizeof(GLint));
    for (int i = 0; i < nv * 4; i++) { data[i] = i + 1; want += i + 1; }
    glUniform4iv(loc, nv, data);

    GLuint vao, vbo, tbuf;
    make_xfb_draw_state(&vao, &vbo, &tbuf);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    GLuint got = 0;
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 4, &got);
    EXPECT_TRUE(got == (GLuint)want);
    glDeleteProgram(p);
    glDeleteBuffers(1, &tbuf);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    free(data);
}

TEST(GsUniformArrayXfb)
{
    run_case(1);
    run_case(2);
    run_case(8);
    run_case(256);     /* CTS geometry_shader.limits.max_uniform_components */
}

TEST(GsUniformMixedRender)
{
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 64, 64);

    char log[4096];
    GLuint p = link(compile(GL_VERTEX_SHADER, kVs),
                    compile(GL_GEOMETRY_SHADER, kGsMixed),
                    compile(GL_FRAGMENT_SHADER, kFsPassthru),
                    NULL, log, sizeof log);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("link: %s", log); return; }

    glUseProgram(p);
    GLint l1 = glGetUniformLocation(p, "uni_sv");
    GLint l2 = glGetUniformLocation(p, "uni_arr");
    GLint l3 = glGetUniformLocation(p, "uni_arr[2]");
    EXPECT_EQ_INT(l1, 0);
    EXPECT_EQ_INT(l2, 1);
    EXPECT_EQ_INT(l3, 3);
    GLint sv[4] = { 1, 2, 3, 4 };
    glUniform4iv(l1, 1, sv);
    GLint arr[16];
    for (int i = 0; i < 16; i++) arr[i] = i + 1;
    glUniform4iv(l2, 4, arr);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pt[2] = { 0, 0 };
    glBufferData(GL_ARRAY_BUFFER, sizeof pt, pt, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLubyte c[4];
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    /* sum 38 of a maximum 40 */
    EXPECT_TRUE(c[0] > 240);
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

/* macro-sized array (CTS uses "#define NUMBER_OF_UNIFORMS 256u"): the kernel's
 * re-declared block must come after the leading preprocessor lines, and the strip
 * must not swallow the uniform declaration into the #define line */
static const char *kGsMacro[4] = {
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices=1) out;\n",
    "#define NUMBER_OF_UNIFORMS ",
    "256",
    "u\n"
    "uniform ivec4 uni_array[NUMBER_OF_UNIFORMS];\n"
    "flat out uint gs_out_sum;\n"
    "void main()\n"
    "{\n"
    "    gs_out_sum = 0u;\n"
    "    for (uint i = 0u; i < NUMBER_OF_UNIFORMS; ++i)\n"
    "    {\n"
    "        gs_out_sum += uint(uni_array[i].x);\n"
    "        gs_out_sum += uint(uni_array[i].y);\n"
    "        gs_out_sum += uint(uni_array[i].z);\n"
    "        gs_out_sum += uint(uni_array[i].w);\n"
    "    }\n"
    "    EmitVertex();\n"
    "    EndPrimitive();\n"
    "}\n"
};

TEST(GsUniformMacroArrayXfb)
{
    GLuint gs = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(gs, 4, kGsMacro, NULL);
    glCompileShader(gs);
    GLint ok = 0;
    glGetShaderiv(gs, GL_COMPILE_STATUS, &ok);
    if (!ok) { char l[4096]; glGetShaderInfoLog(gs, sizeof l, NULL, l); MGL_FAIL("gs: %s", l); return; }

    char log[4096];
    GLuint p = link(compile(GL_VERTEX_SHADER, kVs), gs, compile(GL_FRAGMENT_SHADER, kFs),
                    "gs_out_sum", log, sizeof log);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("link: %s", log); return; }

    glUseProgram(p);
    GLint loc = glGetUniformLocation(p, "uni_array");
    EXPECT_EQ_INT(loc, 0);
    GLint want = 0;
    GLint data[1024];
    for (int i = 0; i < 1024; i++) { data[i] = i + 1; want += i + 1; }
    glUniform4iv(loc, 256, data);

    GLuint vao, vbo, tbuf;
    make_xfb_draw_state(&vao, &vbo, &tbuf);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    GLuint got = 0;
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 4, &got);
    EXPECT_TRUE(got == (GLuint)want);
    glDeleteProgram(p);
    glDeleteBuffers(1, &tbuf);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

int main(void)
{
    RUN(GsUniformArrayXfb);
    RUN(GsUniformMixedRender);
    RUN(GsUniformMacroArrayXfb);
    return mgl_test_summary();
}
