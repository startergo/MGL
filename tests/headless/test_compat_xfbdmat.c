/* compat r97c: transform feedback of double matrices with ODD row counts
 * (dmat2x3 / dmat3 / dmat4x3) from a tessellation evaluation shader.
 *
 * The fp64 lowering keeps every matrix column in its own slot group, so a
 * 3-double column leaves a pad half-slot after it; the tess xfb scatter kernel
 * now copies per column (chunk = rows*2 words, gap = pad) into the tightly
 * packed GL transform-feedback record (KHR-GL46.enhanced_layouts.
 * xfb_stride_dmat{2x3,3,4x3}_tessellation_evaluation).
 */
#include "mgl_test.h"

static const char *vs =
"#version 450 core\n"
"layout(location = 0) in float in_attr;\n"
"out float vs_tc;\n"
"void main() { vs_tc = in_attr; gl_Position = vec4(0,0,0,1); }\n";

static const char *tcs =
"#version 450 core\n"
"layout(vertices = 4) out;\n"
"in float vs_tc[];\n"
"out float tc_te[];\n"
"void main() {\n"
"    tc_te[gl_InvocationID] = vs_tc[gl_InvocationID];\n"
"    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
"    gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
"    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
"}\n";

static const char *fs =
"#version 450 core\n"
"out vec4 fs_out;\n"
"void main() { fs_out = vec4(0,1,0,1); }\n";

static const struct { const char *gltype; int cols, rows; } mats[] = {
    { "dmat2x3", 2, 3 },
    { "dmat3",   3, 3 },
    { "dmat4x3", 4, 3 },
    { "dmat2x4", 2, 4 },   /* even rows: no pad, regression control */
    { "dmat2",   2, 2 },
};

static void run_case(int mi)
{
    char tes[1024];
    int cols = mats[mi].cols, rows = mats[mi].rows;
    int stride = cols * rows * 8;
    snprintf(tes, sizeof tes,
        "#version 450 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout(triangles) in;\n"
        "in float tc_te[];\n"
        "layout(xfb_buffer = 0, xfb_stride = %d, xfb_offset = 0) out %s out_te;\n"
        "void main() {\n"
        "    out_te = %s(%s);\n"
        "    gl_Position = vec4(gl_TessCoord.xy * float(tc_te[0]) * 0.0, 0.0, 1.0);\n"
        "}\n",
        stride, mats[mi].gltype, mats[mi].gltype,
        rows == 4 ? (cols == 2 ? "dvec4(1,2,3,4), dvec4(5,6,7,8)"
                                : cols == 3 ? "dvec4(1,2,3,4), dvec4(5,6,7,8), dvec4(9,10,11,12)"
                                            : "dvec4(1,2,3,4), dvec4(5,6,7,8), dvec4(9,10,11,12), dvec4(13,14,15,16)")
                  : (cols == 2 ? "dvec3(1.0,2.0,3.0), dvec3(4.0,5.0,6.0)"
                               : cols == 3 ? "dvec3(1.0,2.0,3.0), dvec3(4.0,5.0,6.0), dvec3(7.0,8.0,9.0)"
                                           : "dvec3(1.0,2.0,3.0), dvec3(4.0,5.0,6.0), dvec3(7.0,8.0,9.0), dvec3(10.0,11.0,12.0)"));

    GLuint p = glCreateProgram();
    GLuint sh[4]; GLenum ty[4] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER};
    const char *src[4] = {vs, tcs, tes, fs};
    for (int i = 0; i < 4; i++) { sh[i] = glCreateShader(ty[i]); glShaderSource(sh[i], 1, &src[i], NULL); glCompileShader(sh[i]); glAttachShader(p, sh[i]); }
    glLinkProgram(p);
    GLint ls = 0; glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, 1);
    if (!ls) { printf("  %s: link failed\n", mats[mi].gltype); goto out; }

    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    {
        GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        GLuint vbo; glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float d[4] = {1, 2, 3, 4};
        glBufferData(GL_ARRAY_BUFFER, sizeof d, d, GL_STATIC_DRAW);
        GLint al = glGetAttribLocation(p, "in_attr");
        glEnableVertexAttribArray(al);
        glVertexAttribPointer(al, 1, GL_FLOAT, GL_FALSE, 0, 0);

        GLuint tfo, tfb;
        glGenTransformFeedbacks(1, &tfo);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
        glGenBuffers(1, &tfb);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 65536, NULL, GL_STREAM_READ);

        glEnable(GL_RASTERIZER_DISCARD);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_PATCHES, 0, 4);
        glEndTransformFeedback();
        glDisable(GL_RASTERIZER_DISCARD);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

        double want[16], got[16] = {0};
        int n = cols * rows, k = 0;
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++)
                want[k++] = (double)(c * rows + r + 1);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
        void *m = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, n * 8, GL_MAP_READ_BIT);
        if (m) { memcpy(got, m, n * 8); glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER); }
        int bad = 0;
        for (int i = 0; i < n; i++)
            if (got[i] != want[i]) { bad = 1; break; }
        printf("  %s: %s (", mats[mi].gltype, bad ? "MISMATCH" : "ok");
        for (int i = 0; i < n && i < 6; i++) printf("%g ", got[i]);
        printf(")\n");
        EXPECT_EQ_INT(bad, 0);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
        glDeleteBuffers(1, &tfb); glDeleteTransformFeedbacks(1, &tfo);
        glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    }
out:
    for (int i = 0; i < 4; i++) glDeleteShader(sh[i]);
    glDeleteProgram(p);
}

TEST(tes_dmat_xfb_capture)
{
    for (unsigned i = 0; i < sizeof mats / sizeof mats[0]; i++)
        run_case((int)i);
}

int main(void)
{
    RUN(tes_dmat_xfb_capture);
    return mgl_test_summary();
}
