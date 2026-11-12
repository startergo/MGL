/* test_compat_glsl_plain_legacy — the compat-as-superset audit regression
 * (compat r106): KHR-GL46.transform_feedback was 21/21 on a core context but
 * 8/21 on a compatibility context.  Its shaders are core-style sources that
 * merely carry a legacy "#version 130"/"#version 150" — zero compat tokens.
 * mgl_compat_translate_shader used to run them through the fixed-function
 * translation anyway (version gate), upgrading them to 460 core and injecting
 * the mgl* varyings, which broke the multi-stage link
 * ("vertex block member gl_CullDistance has no corresponding member in
 * geometry block").  A source with no compat token must be left untouched
 * whatever its #version; the shared core path then upgrades it exactly like
 * on a core context.
 */
#include "mgl_test.h"
#include "mgl_compat.h"

static GLuint link(const char *vs_src, const char *gs_src, const char *fs_src)
{
    char log[512];
    GLint ok = 0;
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint g = gs_src ? glCreateShader(GL_GEOMETRY_SHADER) : 0;
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs_src, NULL); glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(v, sizeof log, NULL, log); MGL_FAIL("vs: %s", log); }
    if (g)
    {
        glShaderSource(g, 1, &gs_src, NULL); glCompileShader(g);
        glGetShaderiv(g, GL_COMPILE_STATUS, &ok);
        if (!ok) { glGetShaderInfoLog(g, sizeof log, NULL, log); MGL_FAIL("gs: %s", log); }
    }
    glShaderSource(f, 1, &fs_src, NULL); glCompileShader(f);
    glAttachShader(p, v); if (g) glAttachShader(p, g); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link: %s", log);
    }
    return p;
}

TEST(plain_legacy_version_links_and_renders_on_compat_context)
{
    void *cc = mglCompatCreateContext();
    EXPECT_TRUE(cc != NULL);
    EXPECT_TRUE(mglCurrentContextIsCompat());

    /* #version 130 VS + #version 150 GS + #version 130 FS, no compat tokens:
     * must compile, link and render exactly like on a core context */
    GLuint p = link(
        "#version 130\n"
        "out vec4 v_xfb;\n"
        "void main(){ v_xfb = vec4(0.5); gl_Position = vec4(0.0, 0.0, 0.0, 1.0); gl_PointSize = 8.0; }\n",
        "#version 150\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "in vec4 v_xfb[];\n"
        "out vec4 g_xfb;\n"
        "void main(){ g_xfb = v_xfb[0] * 2.0; gl_Position = gl_in[0].gl_Position; gl_PointSize = 8.0; EmitVertex(); EndPrimitive(); }\n",
        "#version 130\n"
        "out vec4 o;\n"
        "void main(){ o = vec4(0, 1, 0, 1); }\n");
    EXPECT_TRUE(p != 0);

    GLuint vao, fbo, rb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 32, 32);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 32, 32);
    glUseProgram(p);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    GLubyte px[4] = { 0, 0, 0, 0 };
    glReadPixels(16, 16, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(px[1], 255);   /* green through the GS */

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteProgram(p);
}

TEST(real_legacy_still_gets_compat_plumbing)
{
    /* the guard must NOT fire for real legacy shaders: this one uses
     * gl_ModelViewProjectionMatrix / ftransform / gl_FragColor and must keep
     * working on the compat context */
    void *cc = mglCompatCreateContext();
    EXPECT_TRUE(cc != NULL);

    GLuint p = link(
        "#version 120\n"
        "void main(){ gl_Position = ftransform(); gl_PointSize = 8.0; }\n",
        NULL,
        "#version 120\n"
        "void main(){ gl_FragColor = vec4(1.0, 0.5, 0.25, 1.0); }\n");
    EXPECT_TRUE(p != 0);

    GLuint vao, fbo, rb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 32, 32);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    glViewport(0, 0, 32, 32);
    glUseProgram(p);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    GLubyte px[4] = { 0, 0, 0, 0 };
    glReadPixels(16, 16, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_EQ_INT(px[0], 255);
    EXPECT_NEAR(px[1], 128, 1);
    EXPECT_NEAR(px[2], 64, 1);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(plain_legacy_version_links_and_renders_on_compat_context);
    RUN(real_legacy_still_gets_compat_plumbing);
    return mgl_test_summary();
}
