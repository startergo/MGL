/* test_cts.c — regressions for Khronos GL CTS (KHR-GL46) failure clusters fixed in MGL.
 * Each test mirrors one CTS case (named in a comment) that used to fail.
 */
#include "mgl_test.h"

static GLuint make_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";
        glGetShaderInfoLog(s, sizeof log - 1, NULL, log);
        fprintf(stderr, "  [shader compile log] %s", log);
    }
    return s;
}

static GLuint link(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    if (vs) glAttachShader(p, vs);
    if (fs) glAttachShader(p, fs);
    glLinkProgram(p);
    return p;
}

static GLint link_ok(GLuint p)
{
    GLint ok = 0, len = 0;
    char log[2048] = "";
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
    if (len > 0) glGetProgramInfoLog(p, sizeof log - 1, NULL, log);
    if (!ok) fprintf(stderr, "  [link log] %s", log);
    return ok;
}

/* ------------------------------------------------------------------ cluster #1: MSL reserved words */

/* KHR-GL46.pixelstoragemodes.teximage2d.r11g11b10f.0_0 — the CTS fragment shader declares
 * `uniform highp sampler2D sampler;`: `sampler` is an MSL type name, so the SPIRV-Cross
 * MSL output ("sampler samplerSmplr") did not compile in Metal ("must use 'struct' tag"). */
static const char *psm_vs =
    "#version 330 core\n"
    "in vec2 position;\n"
    "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";

static const char *psm_fs_sampler =
    "#version 330 core\n"
    "layout(location=0) out vec4 colour;\n"
    "uniform highp sampler2D sampler;\n"
    "uniform vec2 texcoords;\n"
    "void main() { colour = texelFetch(sampler, ivec2(texcoords), 0); }\n";

TEST(cts_msl_reserved_sampler_uniform)
{
    GLuint vs = make_shader(GL_VERTEX_SHADER, psm_vs);
    GLuint fs = make_shader(GL_FRAGMENT_SHADER, psm_fs_sampler);
    GLuint p = link(vs, fs);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);

    /* introspection keeps the GLSL name */
    GLint loc = glGetUniformLocation(p, "sampler");
    EXPECT_TRUE(loc >= 0);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteShader(vs); glDeleteShader(fs); glDeleteProgram(p);
}

/* MSL keywords / built-in type names used as struct members, block members and functions */
static const char *reserved_fs =
    "#version 450 core\n"
    "layout(location=0) out vec4 colour;\n"
    "uniform sampler2D tex;\n"
    "uniform float kernel;\n"                              /* MSL kernel qualifier    */
    "layout(std140) uniform Blk { float device; float thread; float texture; float sampler; } blk;\n"
    "float saturate(float x) { return x < 0.0 ? 0.0 : x; }\n" /* metal std lib function */
    "struct float4 { float x; };\n"                          /* MSL matrix type name   */
    "void main() {\n"
    "    float4 f; f.x = 0.25;\n"
    "    colour = texelFetch(tex, ivec2(f.x, blk.texture), 0)\n"
    "            + vec4(kernel, blk.device, blk.thread, saturate(blk.sampler));\n"
    "}\n";

TEST(cts_msl_reserved_words_everywhere)
{
    GLuint vs = make_shader(GL_VERTEX_SHADER, psm_vs);
    GLuint fs = make_shader(GL_FRAGMENT_SHADER, reserved_fs);
    GLuint p = link(vs, fs);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);

    EXPECT_TRUE(glGetUniformLocation(p, "kernel") >= 0);
    /* members of a *named* uniform block are introspectable via glGetActiveUniform but
     * have no location (GL 4.6 §7.3.1.1: names associated with a named block get -1) */
    EXPECT_EQ_INT(glGetUniformLocation(p, "Blk.device"), -1);

    GLint count = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &count);
    int seen_blk[4] = {0};
    static const char *const want[] = { "Blk.device", "Blk.thread", "Blk.texture", "Blk.sampler" };
    for (int i = 0; i < count; i++)
    {
        char name[256] = "";
        GLsizei len = 0; GLint size = 0; GLenum type = 0;
        glGetActiveUniform(p, i, sizeof name - 1, &len, &size, &type, name);
        for (int k = 0; k < 4; k++)
            if (!strcmp(name, want[k])) { seen_blk[k] = 1; EXPECT_EQ_INT(type, GL_FLOAT); }
    }
    for (int k = 0; k < 4; k++) EXPECT_EQ_INT(seen_blk[k], 1);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteShader(vs); glDeleteShader(fs); glDeleteProgram(p);
}

int main(void)
{
    RUN(cts_msl_reserved_sampler_uniform);
    RUN(cts_msl_reserved_words_everywhere);
    return mgl_test_summary();
}
