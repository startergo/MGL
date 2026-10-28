/* test_sptf_geom — KHR-GL46.separable_programs_tf.geometry_active: separable
 * per-stage programs bound in a program pipeline; when the pipeline contains a
 * GEOMETRY stage, transform feedback must capture the geometry stage's
 * varyings (o_geom == 3) and TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN must be 1. */
#include "mgl_test.h"

static const char *vs =
    "#version 450\n"
    "flat out int o_vert;\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "void main() { o_vert = 1; gl_Position = vec4(1, 0, 0, 1); }\n";
static const char *tcs =
    "#version 450\n"
    "layout(vertices = 1) out;\n"
    "flat in int o_vert[];\n"
    "out gl_PerVertex { vec4 gl_Position; } gl_out[];\n"
    "void main() {\n"
    "    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
    "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
    "    gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
    "}\n";
static const char *tes =
    "#version 450\n"
    "layout (triangles, point_mode) in;\n"
    "flat out int o_tess;\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "void main() { o_tess = 2; gl_Position = vec4(gl_TessCoord.xy*2.0 - 1.0, 0.0, 1.0); }\n";
static const char *gs =
    "#version 450\n"
    "layout (points) in;\n"
    "layout (points, max_vertices = 1) out;\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "flat in int o_tess[];\n"
    "flat out int o_geom;\n"
    "void main() {\n"
    "    o_geom = 3;\n"
    "    gl_Position = vec4(-1, -1, 0, 1);\n"
    "    EmitVertex();\n"
    "}\n";
static const char *fs =
    "#version 450\n"
    "flat in int o_geom;\n"
    "out vec4 o_color;\n"
    "void main() { o_color = vec4(1.0); }\n";

static GLuint mksep(GLenum type, const char *src, const char **vars, int nvars)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
               printf("  [%s] compile: %.600s\n", glGetString(GL_SHADER_TYPE) ? "" : "", log); glDeleteShader(s); return 0; }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    if (nvars) glTransformFeedbackVaryings(p, nvars, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               printf("  link(type %d): %.600s\n", type, log); glDeleteProgram(p); glDeleteShader(s); return 0; }
    glDeleteShader(s);
    return p;
}

TEST(geometry_stage_tf)
{
    /* drop the tessellation stages first (env knob): the geometry TF path alone */
    const int no_tess = getenv("MGL_TEST_NO_TESS") != NULL;
    const char *vvs[] = { "o_vert" }, *vtes[] = { "o_tess" }, *vgs[] = { "o_geom" };
    GLuint pvs = mksep(GL_VERTEX_SHADER, vs, vvs, 1);
    GLuint ptcs = no_tess ? 0 : mksep(GL_TESS_CONTROL_SHADER, tcs, NULL, 0);
    GLuint ptes = no_tess ? 0 : mksep(GL_TESS_EVALUATION_SHADER, tes, vtes, 1);
    GLuint pgs = mksep(GL_GEOMETRY_SHADER, gs, vgs, 1);
    GLuint pfs = mksep(GL_FRAGMENT_SHADER, fs, NULL, 0);
    EXPECT_TRUE(pvs && (!ptcs || no_tess == 0) && (!ptes || no_tess == 0) && pgs && pfs);
    if (!pgs) return;

    GLuint pipe;
    glGenProgramPipelines(1, &pipe);
    glBindProgramPipeline(pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, pvs);
    if (ptcs) glUseProgramStages(pipe, GL_TESS_CONTROL_SHADER_BIT, ptcs);
    if (ptes) glUseProgramStages(pipe, GL_TESS_EVALUATION_SHADER_BIT, ptes);
    glUseProgramStages(pipe, GL_GEOMETRY_SHADER_BIT, pgs);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, pfs);
    EXPECT_NO_GL_ERROR();

    glValidateProgramPipeline(pipe);
    GLint vst = 0;
    glGetProgramPipelineiv(pipe, GL_VALIDATE_STATUS, &vst);
    EXPECT_EQ_INT(vst, GL_TRUE);

    GLuint tfo;
    glGenBuffers(1, &tfo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 400, NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfo);

    GLuint q;
    glGenQueries(1, &q);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    printf("err-begin-tf=%04x\n", glGetError());
    glDrawArrays(no_tess ? GL_POINTS : GL_PATCHES, 0, 1);
    printf("err-draw=%04x\n", glGetError());
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    GLuint written = 0;
    glGetQueryObjectuiv(q, GL_QUERY_RESULT, &written);
    /* with tessellation: triangles+point_mode at level 1 = 3 points, one GS vertex
     * each; without: the single drawn point, one GS vertex out */
    EXPECT_EQ_INT(written, no_tess ? 1 : 3);

    GLint *fb = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 400, GL_MAP_READ_BIT);
    EXPECT_TRUE(fb != NULL);
    if (fb) {
        EXPECT_EQ_INT(fb[0], 3);   /* o_geom: TF comes from the geometry stage */
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &tfo);
    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(pvs);
    if (ptcs) glDeleteProgram(ptcs);
    if (ptes) glDeleteProgram(ptes);
    glDeleteProgram(pgs);
    glDeleteProgram(pfs);
}

/* the tessellation_active sibling first (TF from the TES), then geometry_active:
 * KHR-GL46.separable_programs_tf runs them in this order in one context */
TEST(tess_then_geometry_stage_tf)
{
    const char *vvs[] = { "o_vert" }, *vtes[] = { "o_tess" }, *vgs[] = { "o_geom" };
    GLuint pvs = mksep(GL_VERTEX_SHADER, vs, vvs, 1);
    GLuint ptcs = mksep(GL_TESS_CONTROL_SHADER, tcs, NULL, 0);
    GLuint ptes = mksep(GL_TESS_EVALUATION_SHADER, tes, vtes, 1);
    GLuint pgs = mksep(GL_GEOMETRY_SHADER, gs, vgs, 1);
    GLuint pfs = mksep(GL_FRAGMENT_SHADER, fs, NULL, 0);
    EXPECT_TRUE(pvs && ptcs && ptes && pgs && pfs);
    if (!pgs || !ptes) return;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 1);

    /* --- tessellation_active: no GS in the pipeline, TF expects o_tess == 2 --- */
    GLuint pipe1;
    glGenProgramPipelines(1, &pipe1);
    glBindProgramPipeline(pipe1);
    glUseProgramStages(pipe1, GL_VERTEX_SHADER_BIT, pvs);
    glUseProgramStages(pipe1, GL_TESS_CONTROL_SHADER_BIT, ptcs);
    glUseProgramStages(pipe1, GL_TESS_EVALUATION_SHADER_BIT, ptes);
    glUseProgramStages(pipe1, GL_FRAGMENT_SHADER_BIT, pfs);
    GLuint tfo1, q1;
    glGenBuffers(1, &tfo1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfo1);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 400, NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfo1);
    glGenQueries(1, &q1);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q1);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 1);
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    GLuint w1 = 0;
    glGetQueryObjectuiv(q1, GL_QUERY_RESULT, &w1);
    EXPECT_TRUE(w1 > 0);
    GLint *fb1 = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 400, GL_MAP_READ_BIT);
    if (fb1) { EXPECT_EQ_INT(fb1[0], 2); glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER); }
    glDeleteQueries(1, &q1);
    glDeleteBuffers(1, &tfo1);
    glDeleteProgramPipelines(1, &pipe1);

    /* --- geometry_active: same context, GS added, TF expects o_geom == 3 --- */
    GLuint pipe2;
    glGenProgramPipelines(1, &pipe2);
    glBindProgramPipeline(pipe2);
    glUseProgramStages(pipe2, GL_VERTEX_SHADER_BIT, pvs);
    glUseProgramStages(pipe2, GL_TESS_CONTROL_SHADER_BIT, ptcs);
    glUseProgramStages(pipe2, GL_TESS_EVALUATION_SHADER_BIT, ptes);
    glUseProgramStages(pipe2, GL_GEOMETRY_SHADER_BIT, pgs);
    glUseProgramStages(pipe2, GL_FRAGMENT_SHADER_BIT, pfs);
    GLuint tfo2, q2;
    glGenBuffers(1, &tfo2);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfo2);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 400, NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfo2);
    glGenQueries(1, &q2);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q2);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 1);
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    GLuint w2 = 0;
    glGetQueryObjectuiv(q2, GL_QUERY_RESULT, &w2);
    EXPECT_TRUE(w2 > 0);
    GLint *fb2 = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 400, GL_MAP_READ_BIT);
    if (fb2) { EXPECT_EQ_INT(fb2[0], 3); glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER); }
    glDeleteQueries(1, &q2);
    glDeleteBuffers(1, &tfo2);
    glDeleteProgramPipelines(1, &pipe2);
    glBindProgramPipeline(0);
    glDeleteProgram(pvs);
    glDeleteProgram(ptcs);
    glDeleteProgram(ptes);
    glDeleteProgram(pgs);
    glDeleteProgram(pfs);
}

int main(void)
{
    if (!getenv("MGL_TEST_SKIP_FIRST"))
        RUN(geometry_stage_tf);
    RUN(tess_then_geometry_stage_tf);
    return mgl_test_summary();
}
