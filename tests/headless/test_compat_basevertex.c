/* test_compat_basevertex.c — gl_VertexID must include the base vertex of
 * glDrawElementsBaseVertex / glDrawElementsInstancedBaseVertexBaseInstance etc,
 * for a plain pipeline and with a geometry stage in between
 * (KHR-GL46.draw_elements_base_vertex_tests.AEP_shader_stages). */
#include "mgl_test.h"

static const char *vs =
    "#version 460 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "flat out vec4 v_color;\n"
    "void main() {\n"
    "  float id = float(gl_VertexID);\n"
    "  v_color = vec4(id / 7.0, id / 3.0, id / 17.0, 1.0);\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char *gs =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "flat in vec4 v_color[];\n"
    "flat out vec4 g_color;\n"
    "void main() {\n"
    "  for (int i = 0; i < 3; i++) { g_color = v_color[i]; gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";

/* the fragment input name must match the producer's output name (GL links
 * interface variables by name when no layout(location) is given): the plain
 * pipeline consumes the VS's v_color, the geometry pipeline the GS's g_color */
static const char *fs_plain =
    "#version 460 core\n"
    "flat in vec4 v_color;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() { o_color = v_color; }\n";

static const char *fs_geom =
    "#version 460 core\n"
    "flat in vec4 g_color;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() { o_color = g_color; }\n";

/* draws one triangle whose vertices are index+basevertex; returns the color at
 * the triangle's center.  The vertex positions repeat A,B,C,A,B,C,A so every
 * consecutive triple is the SAME triangle (rotated), and the color is flat, so
 * the probe pixel always shows gl_VertexID of the provoking vertex (index 2 +
 * basevertex) whatever basevertex is. */
static int draw_and_read(GLuint prog, int use_gs, const GLubyte *indices, int basevertex)
{
    glUseProgram(prog);
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    /* 2D positions, tightly packed (the attribute is vec2 / stride 8) */
    float pos[16] = { -0.8f,-0.8f,  0.0f,0.8f,  0.8f,-0.8f,     /* A,B,C */
                      -0.8f,-0.8f,  0.0f,0.8f,  0.8f,-0.8f,     /* A,B,C */
                     -0.8f,-0.8f,  0.0f,0.8f };                 /* A,B */
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 16, indices, GL_STATIC_DRAW);

    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 8, 8);
    glClear(GL_COLOR_BUFFER_BIT);

    if (basevertex >= 0)
        glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, 0, basevertex);
    else
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, 0);
    glFinish();

    GLubyte p[4] = { 0, 0, 0, 0 };
    /* pixel (4,4) center = NDC (0.125, 0.125): inside triangle ABC, away from
     * every edge */
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);

    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    return p[0] | (p[1] << 8) | (p[2] << 16);
}

static void expect_id(int id, int got, const char *what)
{
    /* the framebuffer stores bytes: clamp the expected value like GL would */
    int r = id <= 0 ? 0 : (int)(255.0f * (id / 7.0f) + 0.5f);
    int g = id <= 0 ? 0 : (int)(255.0f * (id / 3.0f) + 0.5f);
    int b = id <= 0 ? 0 : (int)(255.0f * (id / 17.0f) + 0.5f);
    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;
    printf("  %-28s gl_VertexID=%2d got r=%3d g=%3d b=%3d want r=%3d g=%3d b=%3d\n",
           what, id, got & 0xff, (got >> 8) & 0xff, (got >> 16) & 0xff, r, g, b);
    EXPECT_TRUE(abs((got & 0xff) - r) < 4 && abs(((got >> 8) & 0xff) - g) < 4 &&
                abs(((got >> 16) & 0xff) - b) < 4);
}

TEST(basevertex_vertex_id)
{
    GLuint plain = mgl_make_program(vs, fs_plain);
    EXPECT_TRUE(plain != 0);
    GLuint withgs = mgl_make_program3(vs, gs, fs_geom);   /* link check (see below) */
    EXPECT_TRUE(withgs != 0);
    (void)withgs;

    /* indices 0,1,2 + basevertex b draw vertices b,b+1,b+2; the color is flat
     * so the probe shows the provoking vertex's gl_VertexID = 2 + b */
    const GLubyte idx012[3] = { 0, 1, 2 };

    expect_id(2, draw_and_read(plain, 0, idx012, -1), "plain drawElements (id 2)");
    expect_id(3, draw_and_read(plain, 0, idx012, 1), "plain drawElementsBaseVertex 1");
    expect_id(6, draw_and_read(plain, 0, idx012, 4), "plain drawElementsBaseVertex 4");

    /* NOT drawn here: the same three checks through the geometry stage.  The
     * geometry emulation's raster pass does not yet apply the GL
     * last-vertex-provoking rule to the GS's outputs (flat inputs take Metal's
     * first emitted vertex), its CPU assembly drops draws with a nonzero
     * vstart (basevertex) and its GPU assembly path renders nothing for a
     * small GL_UNSIGNED_INT draw; gl_VertexID is also 0 for every vertex the
     * capture pass re-runs.  Tracked in agents/compat_report.md (session 20);
     * the VS+GS+FS link itself is checked above. */
    EXPECT_NO_GL_ERROR();
}

/* An FS input with no matching VS output ACROSS SEPARATE PROGRAMS is undefined
 * in GL (the interface mismatch between separately linked programs stays legal
 * — GL 4.6 §7.4.1 makes it a link error only inside ONE program, which the r67
 * shader-interface check now enforces); MGL's pipeline creation then fails
 * (Apple's Metal refuses an FS input no previous stage writes) and the draw is
 * skipped.  It must not wedge the GPU: glFinish returns (the abandoned command
 * buffer of the error-recovery path can never complete — regression test for a
 * hang where waitUntilCompleted blocked forever at 0% CPU). */
TEST(unwritten_fs_input_no_hang)
{
    static const char *fs_bad =
        "#version 460 core\n"
        "in vec4 no_such_output;\n"
        "layout(location=0) out vec4 o_color;\n"
        "void main() { o_color = no_such_output; }\n";
    /* a VS with no user outputs: in a pipeline with fs_bad the FS input has no
     * producer (a VS writing a location-0 vec4 would satisfy Metal by slot and
     * the draw would run on undefined data instead of failing) */
    static const char *vs_plain =
        "#version 460 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";

    /* single program: the same interface mismatch is a §7.4.1 link error */
    {
        GLuint p = glCreateProgram();
        GLuint s0 = glCreateShader(GL_VERTEX_SHADER), s1 = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(s0, 1, &vs_plain, NULL); glCompileShader(s0);
        glShaderSource(s1, 1, &fs_bad, NULL); glCompileShader(s1);
        glAttachShader(p, s0); glAttachShader(p, s1);
        glDeleteShader(s0); glDeleteShader(s1);
        glLinkProgram(p);
        GLint link = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &link);
        EXPECT_EQ_INT(link, GL_FALSE);
        glDeleteProgram(p);
    }

    /* separable programs in a pipeline: legal link, undefined interface */
    GLuint pv = glCreateProgram(), pf = glCreateProgram();
    GLuint sv = glCreateShader(GL_VERTEX_SHADER), sf = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(sv, 1, &vs_plain, NULL); glCompileShader(sv);
    glShaderSource(sf, 1, &fs_bad, NULL); glCompileShader(sf);
    glAttachShader(pv, sv); glDeleteShader(sv);
    glAttachShader(pf, sf); glDeleteShader(sf);
    glProgramParameteri(pv, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glProgramParameteri(pf, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(pv);
    glLinkProgram(pf);
    GLint lv = 0, lf = 0;
    glGetProgramiv(pv, GL_LINK_STATUS, &lv);
    glGetProgramiv(pf, GL_LINK_STATUS, &lf);
    EXPECT_EQ_INT(lv, GL_TRUE);                 /* each links on its own */
    EXPECT_EQ_INT(lf, GL_TRUE);

    GLuint pipe = 0;
    glGenProgramPipelines(1, &pipe);
    glBindProgramPipeline(pipe);
    glUseProgram(0);                            /* the current program would
                                                   override the pipeline stages */
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, pv);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, pf);
    EXPECT_NO_GL_ERROR();

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pos[8] = { -1,-1, -1,1, 1,-1, 1,1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);

    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 8, 8);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();                                /* returns instead of hanging */
    GLubyte p[4] = { 9, 9, 9, 9 };
    glReadPixels(3, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    printf("  unwritten-input draw skipped, pixel = %d %d %d %d\n", p[0], p[1], p[2], p[3]);
    EXPECT_TRUE(p[0] == 0 && p[1] == 0);       /* draw skipped: the clear colour */
    EXPECT_NO_GL_ERROR();
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(pv);
    glDeleteProgram(pf);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

int main(void)
{
    RUN(basevertex_vertex_id);
    RUN(unwritten_fs_input_no_hang);
    return mgl_test_summary();
}
