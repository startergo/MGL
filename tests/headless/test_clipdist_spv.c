/* test_clipdist_spv — the SPIR-V-binary iteration of test_clipdist: the same
 * vertex stage (gl_ClipDistance[0] = ±1) loaded via glShaderBinary must clip
 * exactly like the GLSL one (KHR-GL46.gl_spirv.spirv_validation_builtin_
 * variable_decorations validation 1, clipNegative).  The module is the
 * prepared SPIR-V of the GLSL test (MGL_PREPARED_DUMP), i.e. the same opcodes
 * the CTS's glslang emits for the same source. */
#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>

static const char *fs =
    "#version 450\n"
    "layout (location = 1) in vec4 vColor;\n"
    "layout (location = 0) out vec4 fColor;\n"
    "void main() { fColor = vColor; }\n";

static unsigned char *load_spv(const char *path, GLsizei *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(n);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (GLsizei)n;
    return b;
}

static int draw_and_read(void)
{
    GLsizei len = 0, flen = 0;
    unsigned char *spv = load_spv("/tmp/clip_vs.spv", &len);
    unsigned char *fspv = load_spv("/tmp/clip_fs.spv", &flen);
    EXPECT_TRUE(spv != NULL && fspv != NULL);
    if (!spv || !fspv) return -1;

    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    glGetError(); /* drain */
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, spv, len);
    EXPECT_NO_GL_ERROR();
    glSpecializeShader(s, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
               MGL_FAIL("vs specialize: %.600s", log); return -1; }

    GLuint fsid = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderBinary(1, &fsid, GL_SHADER_BINARY_FORMAT_SPIR_V, fspv, flen);
    EXPECT_NO_GL_ERROR();
    glSpecializeShader(fsid, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();
    glGetShaderiv(fsid, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(fsid, sizeof log, NULL, log);
               MGL_FAIL("fs specialize: %.600s", log); return -1; }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glAttachShader(p, fsid);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("link: %.600s", log); return -1; }
    glUseProgram(p);

    float v[] = { -1, -1, 0, 1, -1, 0, 0, 1, 0 };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_CLIP_DISTANCE0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_CLIP_DISTANCE0);
    GLubyte px[4];
    mgl_read_rgba(32, 32, px);
    glDeleteProgram(p);
    glDeleteShader(s);
    glDeleteShader(fsid);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    free(spv);
    free(fspv);
    return px[0];
}

/* the dumped module was built with gl_ClipDistance[0] = -1.0: it must cull */
TEST(clip_negative_spv_culls)
{
    /* control in the same process: the same shader from GLSL must cull */
    const char *gvs =
        "#version 450\n"
        "layout (location = 0) in vec3 position;\n"
        "layout (location = 1) out vec4 vColor;\n"
        "void main()\n"
        "{\n"
        "    gl_PointSize = 10.0f;\n"
        "    gl_Position = vec4(position.x, position.y + 0.3 * gl_InstanceID, position.z, 1.0);\n"
        "    gl_ClipDistance[0] = -1.0;\n"
        "    gl_CullDistance[0] = 1.0;\n"
        "    vColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "}\n";
    GLuint gp = mgl_make_program(gvs, fs);
    EXPECT_TRUE(gp != 0);
    if (gp)
    {
        glUseProgram(gp);
        float v[] = { -1, -1, 0, 1, -1, 0, 0, 1, 0 };
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glViewport(0, 0, 64, 64);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_CLIP_DISTANCE0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_CLIP_DISTANCE0);
        GLubyte px[4];
        mgl_read_rgba(32, 32, px);
        EXPECT_EQ_INT(px[0], 0);
        glDeleteProgram(gp);
    }
    int r1 = draw_and_read();
    /* sequence probe: same GLSL control again after the binary draw */
    GLuint gp2 = mgl_make_program(gvs, fs);
    if (gp2)
    {
        glUseProgram(gp2);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_CLIP_DISTANCE0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_CLIP_DISTANCE0);
        GLubyte px[4];
        mgl_read_rgba(32, 32, px);
        printf("glsl-after r=%d\n", px[0]);
        glDeleteProgram(gp2);
    }
    /* and the binary program once more */
    int r2 = draw_and_read();
    EXPECT_EQ_INT(r2, 0);
}

int main(void)
{
    RUN(clip_negative_spv_culls);
    return mgl_test_summary();
}
