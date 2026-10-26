/* test_xctx6.c — compat profile: the real glmark2 jellyfish shaders; uniform
 * locations must be distinct and hold their values (brightness difference) */
#include "mgl_test.h"
#include <stdlib.h>
#include <string.h>

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    char *s;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    s = (char *)calloc(1, n + 1);
    if (fread(s, 1, n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    fclose(f);
    return s;
}

static GLuint compile(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok = 0;
    char log[8192];
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "compile %x: %s\n", type, log); }
    EXPECT_TRUE(ok);
    return sh;
}

TEST(compat_jellyfish_uniforms)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    char *vs, *fs;
    GLuint p, v, f;
    GLint ok = 0;
    static const char *names[] = {
        "uWorld", "uWorldViewProj", "uWorldInvTranspose", "uLightPos", "uLightRadius",
        "uLightCol", "uAmbientCol", "uFresnelPower", "uFresnelCol", "uCurrentTime"
    };
    GLfloat vals[4];

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    vs = slurp("/opt/homebrew/share/glmark2/shaders/jellyfish.vert");
    fs = slurp("/opt/homebrew/share/glmark2/shaders/jellyfish.frag");
    if (!vs || !fs) { fprintf(stderr, "  note: shaders not found, skip\n"); return; }

    v = compile(GL_VERTEX_SHADER, vs);
    f = compile(GL_FRAGMENT_SHADER, fs);
    p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    glUseProgram(p);
    while (glGetError() != GL_NO_ERROR) {}

    /* every uniform must get its own slot; write distinct tagged values and
     * read them back */
    int collisions = 0, missing = 0;
    for (int i = 0; i < (int)(sizeof names / sizeof names[0]); i++)
    {
        GLint loc = glGetUniformLocation(p, names[i]);
        if (loc < 0) { fprintf(stderr, "  note: uniform %s: location %d\n", names[i], loc); missing++; continue; }

        if (strstr(names[i], "mat") || strstr(names[i], "World"))
        {
            GLfloat m[16];
            for (int k = 0; k < 16; k++) m[k] = (GLfloat)(100 + i * 10 + k);
            glUniformMatrix4fv(loc, 1, GL_FALSE, m);
            memset(m, 0, sizeof m);
            glGetUniformfv(p, loc, m);
            for (int k = 0; k < 16; k++)
                if (m[k] != (GLfloat)(100 + i * 10 + k)) { collisions++; break; }
        }
        else if (strcmp(names[i], "uLightPos") == 0)
        {
            GLfloat w[3] = { 203, 204, 205 };
            glUniform3fv(loc, 1, w);
            glGetUniformfv(p, loc, vals);
            if (vals[0] != w[0] || vals[1] != w[1] || vals[2] != w[2])
            { fprintf(stderr, "  note: uLightPos readback %g,%g,%g\n", vals[0], vals[1], vals[2]); collisions++; }
        }
        else if (strcmp(names[i], "uLightCol") == 0 || strcmp(names[i], "uAmbientCol") == 0 ||
                 strcmp(names[i], "uFresnelCol") == 0)
        {
            GLfloat w[4] = { (GLfloat)(200 + i), (GLfloat)(201 + i), (GLfloat)(202 + i), (GLfloat)(203 + i) };
            glUniform4fv(loc, 1, w);
            glGetUniformfv(p, loc, vals);
            if (vals[0] != w[0] || vals[1] != w[1] || vals[2] != w[2] || vals[3] != w[3])
            { fprintf(stderr, "  note: %s readback %g,%g,%g,%g\n", names[i], vals[0], vals[1], vals[2], vals[3]); collisions++; }
        }
        else   /* uLightRadius, uFresnelPower, uCurrentTime: floats */
        {
            GLfloat w = (GLfloat)(200 + i);
            glUniform1f(loc, w);
            glGetUniformfv(p, loc, vals);
            if (vals[0] != w)
            { fprintf(stderr, "  note: %s readback %g wrote %g\n", names[i], vals[0], w); collisions++; }
        }
    }
    /* attributes too */
    static const char *attrs[] = { "aVertexPosition", "aVertexNormal", "aVertexColor", "aTextureCoord" };
    for (int i = 0; i < 4; i++)
    {
        GLint loc = glGetAttribLocation(p, attrs[i]);
        fprintf(stderr, "  note: attribute %s -> %d\n", attrs[i], loc);
        if (loc < 0) missing++;
        for (int j = 0; j < i; j++)
        {
            GLint l2 = glGetAttribLocation(p, attrs[j]);
            if (loc == l2) { fprintf(stderr, "  note: attribute collision %s == %s\n", attrs[i], attrs[j]); collisions++; }
        }
    }

    EXPECT_EQ_INT(collisions, 0);
    EXPECT_EQ_INT(missing, 0);
    fprintf(stderr, "PHASE cleanup\n");

    fprintf(stderr, "PHASE delprog\n");
    glDeleteProgram(p);
    fprintf(stderr, "PHASE free\n");
    free(vs); free(fs);
    fprintf(stderr, "PHASE destroy\n");
    destroyGLMContext(c);
    fprintf(stderr, "PHASE unset\n");
    MGLsetCurrentContext(NULL);
    fprintf(stderr, "PHASE end\n");
}

int main(void)
{
    RUN(compat_jellyfish_uniforms);
    return mgl_test_summary();
}
