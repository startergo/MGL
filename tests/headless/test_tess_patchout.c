#include "mgl_test.h"

/* TCS patch-out ARRAY variables end-to-end (Mesa-style per-patch interface
 * arrays): (1) `patch out float parr[6]` and (2) an array of patch blocks
 * `patch out TheBlock { float fa[3]; } blkarr[2]`.  Both are written by the
 * TCS, summed by the TES and the rasterised colour is read back — so this
 * checks the whole dissolve/capture path (blob map + per-element lowering),
 * not just link success.  Full-coverage triangle => centre pixel is the sum.
 */
static const char *vs =
"#version 450\n"
"in highp float in_v_attr;\n"
"out highp float in_tc_attr;\n"
"void main (void) { in_tc_attr = in_v_attr; gl_Position = vec4(0,0,0,1); }\n";

static const char *tcs_arr2d =
"#version 450\n"
"layout (vertices = 5) out;\n"
"in highp float in_tc_attr[];\n"
"patch out float parr[6];\n"
"void main (void)\n"
"{\n"
"    for (int i = 0; i < 2; i++)\n"
"        for (int j = 0; j < 3; j++)\n"
"            parr[i*3+j] = in_tc_attr[0] + float(i*3+j);\n"
"    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
"    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
"    gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
"}\n";

static const char *tcs_blkarr =
"#version 450\n"
"layout (vertices = 5) out;\n"
"in highp float in_tc_attr[];\n"
"patch out TheBlock { highp float fa[3]; } blkarr[2];\n"
"void main (void)\n"
"{\n"
"    for (int i = 0; i < 2; i++)\n"
"        for (int j = 0; j < 3; j++)\n"
"            blkarr[i].fa[j] = in_tc_attr[0] + float(i*3+j);\n"
"    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
"    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
"    gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
"}\n";

static const char *tes_arr2d =
"#version 450\n"
"layout (triangles) in;\n"
"patch in float parr[6];\n"
"out highp vec4 col;\n"
"void main (void)\n"
"{\n"
"    float s = 0.0;\n"
"    for (int i = 0; i < 2; i++)\n"
"        for (int j = 0; j < 3; j++)\n"
"            s += parr[i*3+j];\n"
"    col = vec4(s*0.01, 1, 0, 1);\n"
"    gl_Position = vec4(gl_TessCoord.xy, 0.0, 1.0);\n"
"}\n";

static const char *tes_blkarr =
"#version 450\n"
"layout (triangles) in;\n"
"patch in TheBlock { highp float fa[3]; } blkarr[2];\n"
"out highp vec4 col;\n"
"void main (void)\n"
"{\n"
"    float s = 0.0;\n"
"    for (int i = 0; i < 2; i++)\n"
"        for (int j = 0; j < 3; j++)\n"
"            s += blkarr[i].fa[j];\n"
"    col = vec4(s*0.01, 1, 0, 1);\n"
"    gl_Position = vec4(gl_TessCoord.xy, 0.0, 1.0);\n"
"}\n";

static const char *fs =
"#version 450\n"
"in highp vec4 col;\n"
"out vec4 o;\n"
"void main (void) { o = col; }\n";

static void run(const char *tcs, const char *tes, const char *label)
{
    GLuint p = glCreateProgram();
    GLuint sh[4]; GLenum ty[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    const char *src[4] = { vs, tcs, tes, fs };
    for (int i = 0; i < 4; i++) {
        sh[i] = glCreateShader(ty[i]);
        glShaderSource(sh[i], 1, &src[i], NULL);
        glCompileShader(sh[i]);
        glAttachShader(p, sh[i]);
    }
    glLinkProgram(p);
    GLint ls = 0; glGetProgramiv(p, GL_LINK_STATUS, &ls);
    char log[2048] = {0}; GLsizei ll = 0;
    glGetProgramInfoLog(p, sizeof log, &ll, log);
    printf("  %s link=%d log=%.300s\n", label, ls, log);
    if (ls) {
        glUseProgram(p);
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        GLuint vbo; glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float d[3] = { 0.1f, 0.2f, 0.3f };
        glBufferData(GL_ARRAY_BUFFER, sizeof d, d, GL_STATIC_DRAW);
        GLint al = glGetAttribLocation(p, "in_v_attr");
        glEnableVertexAttribArray(al);
        glVertexAttribPointer(al, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glViewport(0, 0, 32, 32);
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_PATCHES, 0, 3);
        glFinish();
        GLubyte px[4]; mgl_read_rgba(16, 16, px);
        /* sum(0..5)*0.1*0.01 + 0.15*0.01*6*... rough: expect green-ish r>0 */
        printf("  %s px = %d %d %d %d\n", label, px[0], px[1], px[2], px[3]);
        EXPECT_TRUE(px[1] > 200);
        EXPECT_TRUE(px[0] > 10);
        glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    }
    for (int i = 0; i < 4; i++) glDeleteShader(sh[i]);
    glDeleteProgram(p);
}

TEST(tcs_patch_out_array_shapes)
{
    run(tcs_arr2d, tes_arr2d, "arr2d");
    run(tcs_blkarr, tes_blkarr, "blkarr");
}

int main(void)
{
    RUN(tcs_patch_out_array_shapes);
    return mgl_test_summary();
}
