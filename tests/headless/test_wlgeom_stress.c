/* r71: the geometry-stage wide-line expansion under ring pressure.  A burst of
 * un-synchronised draws exhausts the tess ring's slots; the reserve (before the
 * draw's first encoder opens) then walks tessRingAcquire's pressure path — end
 * the render encoding, commit the shared command buffer — which is exactly the
 * path that wedged the ring when the acquire ran with the geometry kernel's
 * compute encoder open (the r70 "joined encoder deadlock").  After the burst a
 * widened draw must still render exactly `width' rows and the native 1-px path
 * must still work.  The second test repeats the burst through the
 * tess->geometry path (the emit then joins the tess import encoder that geomDraw
 * keeps open across its kernel section). */
#include "mgl_test.h"

static GLuint stages_prog(const GLenum *types, const char **srcs, int n, int *link)
{
    GLuint p = glCreateProgram();
    for (int i = 0; i < n; i++)
    {
        GLuint so = glCreateShader(types[i]);
        glShaderSource(so, 1, &srcs[i], NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, link);
    return p;
}

static void setup_vao(void)
{
    static const float pos[4] = { -0.5f, 0.0f, 0.5f, 0.0f };   /* two vec2 */
    GLuint vao, b;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
}

static int rows_lit(int min)
{
    float c[4];
    int rows = 0;
    for (int y = 0; y < 200; y++)
    {
        int n = 0;
        for (int x = 0; x < 200; x++)
        {
            glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, c);
            if (c[1] > 0.5f) n++;
        }
        if (n >= min) rows++;
    }
    return rows;
}

TEST(TessWideLineGeomBurst)
{
    const char *vs = "#version 460 core\nlayout(location=0) in vec2 a_pos;\nvoid main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
    const char *gs = "#version 460 core\nlayout(lines) in;\nlayout(line_strip, max_vertices = 2) out;\n"
        "void main(){ gl_Position = gl_in[0].gl_Position; EmitVertex(); gl_Position = gl_in[1].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    const char *fs = "#version 460 core\nlayout(location=0) out vec4 o_col;\nvoid main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[3] = { vs, gs, fs };
    GLuint p;
    int link = 0;

    p = stages_prog(types, srcs, 3, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    setup_vao();
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glLineWidth(5.0f);

    /* 400 draws, no sync between them: the ring's slots all go busy on the
     * shared command buffer and the pressure path must commit it (with no
     * encoder open — the reserve runs before the draw's first encoder) */
    for (int i = 0; i < 400; i++)
        glDrawArrays(GL_LINES, 0, 2);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(rows_lit(90), 5);

    /* the native 1-px path after the burst */
    glLineWidth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    {
        int r = rows_lit(90);
        EXPECT_TRUE(r >= 1 && r <= 3);
    }
    glDeleteProgram(p);
}

TEST(TessWideLineTessGeomBurst)
{
    const char *vs = "#version 460 core\nlayout(location=0) in vec2 a_pos;\nvoid main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
    const char *tcs = "#version 460 core\nlayout(vertices=2) out;\nvoid main(){\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; }\n}\n";
    const char *tes = "#version 460 core\nlayout(isolines) in;\nvoid main(){\n"
        "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n}\n";
    const char *gs = "#version 460 core\nlayout(lines) in;\nlayout(line_strip, max_vertices = 2) out;\n"
        "void main(){ gl_Position = gl_in[0].gl_Position; EmitVertex(); gl_Position = gl_in[1].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    const char *fs = "#version 460 core\nlayout(location=0) out vec4 o_col;\nvoid main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER,
                              GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[5] = { vs, tcs, tes, gs, fs };
    GLuint p;
    int link = 0;

    p = stages_prog(types, srcs, 5, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    setup_vao();
    glPatchParameteri(GL_PATCH_VERTICES, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glLineWidth(5.0f);

    /* the same burst through the tessellation producer: every draw also takes a
     * tess ring block (the TES capture) before the geometry pass runs */
    for (int i = 0; i < 200; i++)
        glDrawArrays(GL_PATCHES, 0, 2);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(rows_lit(90), 5);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(TessWideLineGeomBurst);
    RUN(TessWideLineTessGeomBurst);
    return mgl_test_summary();
}
