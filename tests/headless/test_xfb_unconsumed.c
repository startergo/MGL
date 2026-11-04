/* tests/headless/test_xfb_unconsumed.c — GL 4.6 13.2 + 11.1.2.1: a shader
 * output the next stage does not declare is INACTIVE — it is given no
 * interpolator storage and must not count against the producer's location
 * cap, however many such outputs transform feedback captures ("captured
 * outputs need not be inputs of the subsequent stage", 13.2).
 *
 * Shape from KHR-GL46.transform_feedback.get_xfb_varying (CheckGetXFBVarying):
 * MGL reports MAX_VARYING_COMPONENTS=124, MAX_VARYING_VECTORS=31,
 * MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS=128, so the CTS derives
 * 30 mat2 vertex outputs (30 <= 31 vectors, 120 <= 124-4 components,
 * 120 <= 128 interleaved) = 60 locations against a fragment stage that
 * consumes nothing — the link must succeed, every one of the 30 must come
 * back from glGetTransformFeedbackVarying, and the capture draw must deliver
 * them.  The CONSUMED location cap (16..31) must keep failing as before.
 */
#include "mgl_test.h"

#define N_OUT 30          /* mat2 outputs: result_0 .. result_29 (60 locations) */

static GLuint build_mat2_program(const char *fs_body, const char **varyings, int nv, GLenum mode)
{
    GLuint p = glCreateProgram(), vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
    char src[8192], fs_src[1024];
    int o = snprintf(src, sizeof src, "#version 460 core\n");
    for (int i = 0; i < N_OUT; i++) o += snprintf(src + o, sizeof src - o, "out mat2 result_%d;\n", i);
    o += snprintf(src + o, sizeof src - o,
        "void main() {\n"
        "    vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);\n"
        "    gl_Position = vec4(pos, 0.0, 1.0);\n");
    for (int i = 0; i < N_OUT; i++)
        o += snprintf(src + o, sizeof src - o, "    result_%d = mat2(float(gl_VertexID) + 1.0 + %d / 100.0);\n", i, i);
    o += snprintf(src + o, sizeof src - o, "}\n");

    snprintf(fs_src, sizeof fs_src, "#version 460 core\n%s", fs_body);

    glShaderSource(vs, 1, (const char *[]){ src }, NULL);
    glCompileShader(vs);
    glShaderSource(fs, 1, (const char *[]){ fs_src }, NULL);
    glCompileShader(fs);
    glAttachShader(p, vs); glAttachShader(p, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    if (varyings) glTransformFeedbackVaryings(p, nv, varyings, mode);
    glLinkProgram(p);
    return p;
}

/* the whole CTS shape: link + query + capture */
TEST(UnconsumedMat2LinkAndCapture)
{
    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);

    const char *varyings[N_OUT];
    char names[N_OUT][16];
    for (int i = 0; i < N_OUT; i++) { snprintf(names[i], 16, "result_%d", i); varyings[i] = names[i]; }

    GLuint p = build_mat2_program("out vec4 color;\nvoid main() { color = vec4(1); }\n",
                                  varyings, N_OUT, GL_INTERLEAVED_ATTRIBS);
    GLint link = -1; glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE)
    {
        char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  link log: %s", log);
    }
    EXPECT_EQ_INT(link, GL_TRUE);

    /* every one of the 30 must be queryable with the right name/type/size */
    GLint nxfb = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &nxfb);
    EXPECT_EQ_INT(nxfb, N_OUT);
    int queried = 0;
    for (int i = 0; i < N_OUT; i++)
    {
        char nm[32]; GLsizei len = 0; GLint size = 0; GLenum type = 0;
        glGetTransformFeedbackVarying(p, i, sizeof nm, &len, &size, &type, nm);
        char want[16]; snprintf(want, sizeof want, "result_%d", i);
        if (len > 0 && !strcmp(nm, want) && size == 1 && type == GL_FLOAT_MAT2) queried++;
    }
    printf("  xfb varyings queried: %d/%d\n", queried, N_OUT);
    EXPECT_EQ_INT(queried, N_OUT);

    /* capture draw: 3 POINTS, rasterizer discarded, 30*mat2 = 480 B/vertex */
    GLuint tfo, tfb; glGenTransformFeedbacks(1, &tfo); glGenBuffers(1, &tfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 3 * N_OUT * 16, NULL, GL_STATIC_READ);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float *data = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 3 * N_OUT * 16, GL_MAP_READ_BIT);
    int bad = 0;
    if (data)
    {
        for (int v = 0; v < 3 && !bad; v++)
            for (int i = 0; i < N_OUT && !bad; i++)
            {
                float want = (float)v + 1.0f + (float)i / 100.0f;
                const float *m = data + (v * N_OUT + i) * 4;
                /* mat2(w) column-major = { w, 0, 0, w } */
                for (int k = 0; k < 4; k++)
                {
                    float w = (k == 1 || k == 2) ? 0.0f : want;
                    float d = m[k] - w; if (d < 0) d = -d;
                    if (d > 1e-5f) { printf("  v%d m%d[%d] = %g want %g\n", v, i, k, m[k], w); bad++; break; }
                }
            }
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    }
    else { printf("  capture map failed\n"); bad++; }
    printf("  capture: %s\n", bad ? "MISMATCH" : "3 vertices x 30 mat2 exact");
    EXPECT_EQ_INT(bad, 0);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteBuffers(1, &tfb); glDeleteTransformFeedbacks(1, &tfo);
    glDeleteProgram(p); glDeleteVertexArrays(1, &vao);
    while (glGetError() != GL_NO_ERROR) {}
}

/* the unconsumed outputs ride along while the FS consumes ONE of them —
 * a real (rasterizing) draw proves the pipeline builds with unconsumed
 * user locations far past the Metal interpolator budget */
TEST(UnconsumedDrawWithFsConsumer)
{
    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    GLuint p = build_mat2_program(
        "in mat2 result_0;\nout vec4 color;\n"
        "void main() { color = vec4(result_0[0], 0.25, 0.5); }\n",
        NULL, 0, 0);
    GLint link = -1; glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE)
    {
        char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  link log: %s", log);
    }
    EXPECT_EQ_INT(link, GL_TRUE);

    glUseProgram(p);
    glViewport(0, 0, 64, 64);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    unsigned char px[3] = { 0 };
    glReadPixels(32, 32, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    printf("  center px after draw: (%d %d %d)\n", px[0], px[1], px[2]);
    /* result_0 = mat2(gl_VertexID+1): column 0 = (v, 0) for the covered
     * fragments' provoking... interpolated across the triangle the first
     * component lands in [1..3] -> clipped to 1.0 => r=255; column 0's y=0,
     * then the literal 0.25 / 0.5 fill g/b */
    EXPECT_EQ_INT(px[0], 255);
    EXPECT_EQ_INT(px[1], 0);
    EXPECT_EQ_INT(px[2], 64);

    glDeleteProgram(p); glDeleteVertexArrays(1, &vao);
    while (glGetError() != GL_NO_ERROR) {}
}

/* guard: CONSUMED outputs keep the location cap (33rd vec4 location fails) */
TEST(ConsumedLocationCapStillFails)
{
    GLuint p = glCreateProgram(), vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs, 1, (const char *[]){ "#version 460 core\n"
        "layout(location=32) out vec4 x;\nvoid main() { x = vec4(1); gl_Position = vec4(0); }\n" }, NULL);
    glCompileShader(vs);
    glShaderSource(fs, 1, (const char *[]){ "#version 460 core\n"
        "layout(location=32) in vec4 x;\nout vec4 c;\nvoid main() { c = x; }\n" }, NULL);
    glCompileShader(fs);
    glAttachShader(p, vs); glAttachShader(p, fs); glDeleteShader(vs); glDeleteShader(fs);
    glLinkProgram(p);
    GLint link = -1; glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_FALSE);       /* consumed: the cap applies */
    glDeleteProgram(p);
    while (glGetError() != GL_NO_ERROR) {}
}

int main(void)
{
    RUN(UnconsumedMat2LinkAndCapture);
    RUN(UnconsumedDrawWithFsConsumer);
    RUN(ConsumedLocationCapStillFails);
    return mgl_test_summary();
}
