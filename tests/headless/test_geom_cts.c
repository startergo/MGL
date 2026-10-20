#include "mgl_test.h"
/* regression repros for KHR-GL46.geometry_shader cases found by the CTS; each TEST builds
 * the offending shader pair and checks the link (the CTS expects links to succeed) */

static GLuint try_link(const char *vs, const char *gs, const char *fs, const char *what)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint g = gs ? glCreateShader(GL_GEOMETRY_SHADER) : 0;
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint p = glCreateProgram();
    const char *src[3];
    GLint ok = 0;

    src[0] = vs; glShaderSource(v, 1, src, NULL); glCompileShader(v);
    if (gs) { src[0] = gs; glShaderSource(g, 1, src, NULL); glCompileShader(g); }
    src[0] = fs; glShaderSource(f, 1, src, NULL); glCompileShader(f);

    glAttachShader(p, v);
    if (gs) glAttachShader(p, g);
    glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[8192];
        GLsizei len = 0;
        log[0] = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        printf("%s: link FAILED:\n%s\n", what, log);
    }
    else printf("%s: linked\n", what);
    EXPECT_NO_GL_ERROR();
    return p;
}

/* KHR-GL46.geometry_shader.rendering.rendering.lines_input_triangle_strip_output_* :
 * a "#define" block directly ahead of an interface declaration made the transform's
 * statement copier swallow the declaration ("flat in int instance_id[N];" survived into
 * the compute kernel, where storage qualifiers are illegal) */
TEST(cts_gs_define_then_iface)
{
    static const char *vs =
        "#version 440\n"
        "out vec4 vs_gs_color;\n"
        "flat out int instance_id;\n"
        "void main(){ vs_gs_color = vec4(1);\n"
        "  instance_id = gl_InstanceID; gl_Position = vec4(gl_VertexID, 0, 0, 1); }\n";
    static const char *gs =
        "#version 440\n"
        "layout(lines) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "\n"
        "#define N_VERTEX0     (0)\n"
        "#define N_VERTEX1     (1)\n"
        "#define N_VERTICES_IN (2)\n"
        "flat in   int instance_id[N_VERTICES_IN];\n"
        "     in  vec4 vs_gs_color[N_VERTICES_IN];\n"
        "     out vec4 gs_fs_color;\n"
        "uniform ivec2 renderingTargetSize;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = gl_in[N_VERTEX0].gl_Position;\n"
        "    gs_fs_color = vs_gs_color[N_VERTEX0];\n"
        "    EmitVertex();\n"
        "    gl_Position = gl_in[N_VERTEX1].gl_Position;\n"
        "    gs_fs_color = vs_gs_color[N_VERTEX1];\n"
        "    EmitVertex();\n"
        "    gl_Position = vec4(0.0, -0.5 + float(instance_id[0]) * 0.5, 0.0, 1.0);\n"
        "    gs_fs_color = vs_gs_color[N_VERTEX0];\n"
        "    EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 440\n"
        "precision highp float;\n"
        "in vec4 gs_fs_color;\n"
        "layout(location=0) out vec4 fs_out_color;\n"
        "void main(){ fs_out_color = gs_fs_color; }\n";
    GLuint p = try_link(vs, gs, fs, "define_then_iface");
    GLint linked = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    EXPECT_TRUE(linked == GL_TRUE);
    glDeleteProgram(p);
}

TEST(cts_gs_limits_uniform_array)
{
    /* KHR-GL46.geometry_shader.limits.max_uniform_components: a "#define" between the
     * preamble and the declarations, then a plain uniform array and a "flat out" that
     * must not survive into the compute kernel */
    static const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec2 p;\n"
        "void main(){ gl_Position = vec4(p, 0.0, 1.0); }\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "// definition of NUMBER_OF_UNIFORMS goes here\n"
        "#define NUMBER_OF_UNIFORMS 256u\n"
        "uniform ivec4 uni_array[NUMBER_OF_UNIFORMS];\n"
        "flat out uint gs_out_sum;\n"
        "void main()\n"
        "{\n"
        "    gs_out_sum = 0u;\n"
        "    for (uint n = 0u; n < 1u; ++n)\n"
        "    {\n"
        "        gs_out_sum += uint(uni_array[0][0]);\n"
        "    }\n"
        "    EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    {
        GLuint p = glCreateProgram();
        GLuint v = glCreateShader(GL_VERTEX_SHADER), g = glCreateShader(GL_GEOMETRY_SHADER),
              f = glCreateShader(GL_FRAGMENT_SHADER);
        GLint ok = 0, loc = -1;
        GLuint vao, vbo, tex, fbo;
        static const float one[2] = { 0.0f, 0.0f };
        int data[4] = { 7, 0, 0, 0 };

        glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
        glShaderSource(g, 1, &gs, NULL); glCompileShader(g);
        glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
        glAttachShader(p, v); glAttachShader(p, g); glAttachShader(p, f);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[4096]; GLsizei len = 0; log[0] = 0;
            glGetProgramInfoLog(p, sizeof log, &len, log);
            printf("limits_uniform_array: link FAILED:\n%s\n", log);
        }
        EXPECT_TRUE(ok == GL_TRUE);
        if (!ok) return;
        glUseProgram(p);
        loc = glGetUniformLocation(p, "uni_array[0]");
        EXPECT_TRUE(loc >= 0);
        if (loc >= 0) glProgramUniform4iv(p, loc, 1, data);

        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof one, one, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 200, 200, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glViewport(0, 0, 200, 200);
        glDrawArrays(GL_POINTS, 0, 1);
        EXPECT_NO_GL_ERROR();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(p);
        EXPECT_NO_GL_ERROR();
    }
}
/* KHR-GL46.geometry_shader.input.gl_in_array_contents: "#ifdef"-guarded unsized input
 * arrays must both transform (the unsized [] gets the input primitive size) */
TEST(cts_gs_ifdef_unsized_arrays)
{
    static const char *vs =
        "#version 440\n"
        "out vec2 vs_gs_a;\n"
        "flat out ivec4 vs_gs_b;\n"
        "void main(){ vs_gs_a = vec2(gl_VertexID, 0); vs_gs_b = ivec4(0, gl_VertexID, 0, 1); }\n";
    static const char *gs_preamble =
        "#version 440\n"
        "layout(triangles)                      in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "\n";
    static const char *gs_unsized =
        "#ifdef USE_UNSIZED_ARRAYS\n"
        "         in  vec2 vs_gs_a[];\n"
        "    flat in ivec4 vs_gs_b[];\n"
        "#else\n"
        "         in  vec2 vs_gs_a[3];\n"
        "    flat in ivec4 vs_gs_b[3];\n"
        "#endif\n"
        "\n"
        "     out  vec2 gs_fs_a;\n"
        "flat out ivec4 gs_fs_b;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(-1, -1, 0, 1);\n"
        "    gs_fs_a = vs_gs_a[0]; gs_fs_b = vs_gs_b[0]; EmitVertex();\n"
        "    gl_Position = vec4(-1,  1, 0, 1);\n"
        "    gs_fs_a = vs_gs_a[1]; gs_fs_b = vs_gs_b[1]; EmitVertex();\n"
        "    gl_Position = vec4( 1,  1, 0, 1);\n"
        "    gs_fs_a = vs_gs_a[2]; gs_fs_b = vs_gs_b[2]; EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 440\n"
        "precision highp float;\n"
        "in vec2 gs_fs_a; flat in ivec4 gs_fs_b;\n"
        "layout(location=0) out vec4 fs_out_color;\n"
        "void main(){ fs_out_color = vec4(1,1,1,1); }\n";
    static const char *parts[2];
    GLuint p1, p2;
    GLint linked = 0;

    parts[0] = gs_preamble; parts[1] = gs_unsized;
    p1 = glCreateProgram();
    {
        GLuint v = glCreateShader(GL_VERTEX_SHADER), g = glCreateShader(GL_GEOMETRY_SHADER),
              f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
        glShaderSource(g, 2, parts, NULL); glCompileShader(g);
        glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
        glAttachShader(p1, v); glAttachShader(p1, g); glAttachShader(p1, f);
        glLinkProgram(p1);
    }
    glGetProgramiv(p1, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        char log[8192]; GLsizei len = 0; log[0] = 0;
        glGetProgramInfoLog(p1, sizeof log, &len, log);
        printf("unsized arrays: link FAILED:\n%s\n", log);
    }
    EXPECT_TRUE(linked == GL_TRUE);
    glDeleteProgram(p1);
    (void)p2; (void)try_link;
}

int main(void)
{
    RUN(cts_gs_limits_uniform_array);
    RUN(cts_gs_define_then_iface);
    RUN(cts_gs_ifdef_unsized_arrays);
    return mgl_test_summary();
}
