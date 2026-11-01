/*
 * test_sepshader86.c — separate-shader-object varying matching vs a monolithic program
 * (dEQP GL45-ES31 separate_shader.random: implicit+explicit varyings of mixed types,
 * including matrices, must render identically through a pipeline)
 */
#include "mgl_test.h"

static const char *VS = "#version 310 es\n"
    "const highp float vtxScale = 0.5;\n"
    "smooth out highp mat4x3 var0;\n"
    "flat out highp uvec4 var1;\n"
    "layout(location=5) smooth out highp vec3 vtxVar2;\n"
    "layout(location=6) flat out highp vec4 vtxVar3;\n"
    "const vec2 triangle[3] = vec2[3](vec2( 0.8750, 0.5625), vec2(-0.6875, 0.3125), vec2( 0.2500,-0.8125));\n"
    "const mat4x3 var0Inits[3] = mat4x3[3](\n"
    "  mat4x3( 0.1250, 0.3125, 0.1875, 0.6250, 0.0625, 0.3125, 0.6875, 0.3125, 0.9375, 0.9375, 0.6250, 1.0),\n"
    "  mat4x3( 0.8125, 1.0,    0.9375, 0.1875, 0.25,   0.75,   0.1250, 0.3750, 0.8125, 0.8125, 0.5625, 0.0625),\n"
    "  mat4x3( 0.8750, 0.9375, 0.4375, 0.8125, 0.1875, 0.6250, 0.3750, 1.0,    0.5,    0.6875, 0.3125, 0.1875));\n"
    "const uvec4 var1Inits[3] = uvec4[3](uvec4(8,174,21,65), uvec4(214,150,184,229), uvec4(24,130,8,215));\n"
    "const vec3 vtxVar2Inits[3] = vec3[3](vec3(0.6875,0.4375,0.5), vec3(0.5625,0.0,0.125), vec3(0.5625,0.375,0.1875));\n"
    "const vec4 vtxVar3Inits[3] = vec4[3](vec4(0.6875,0.0,0.5,0.5625), vec4(0.75,0.75,0.125,0.9375), vec4(0.0625,0.0625,0.125,0.5));\n"
    "void main (void) {\n"
    "  gl_Position = vec4(vtxScale * triangle[gl_VertexID], 0.0, 1.0);\n"
    "  var0 = var0Inits[gl_VertexID]; var1 = var1Inits[gl_VertexID];\n"
    "  vtxVar2 = vtxVar2Inits[gl_VertexID]; vtxVar3 = vtxVar3Inits[gl_VertexID];\n"
    "}\n";

static const char *FS = "#version 310 es\n"
    "precision highp float;\n"
    "out vec4 fragColor;\n"
    "const highp float frgScale = 0.875;\n"
    "smooth in highp mat4x3 var0;\n"
    "flat in highp uvec4 var1;\n"
    "layout(location=5) in highp vec3 frgVar2;\n"
    "layout(location=6) flat in highp vec4 frgVar3;\n"
    "void main (void) {\n"
    "  vec4 c = vec4(var0[0] + var0[1] + var0[2], var1.x);\n"
    "  c += vec4(frgVar2, 1.0) + frgVar3;\n"
    "  fragColor = c * 0.05;\n"
    "}\n";

static GLuint compile_one(GLenum type, const char *src)
{
    char log[2048]; GLint ok = 0;
    GLuint s = glCreateShader(type), p = glCreateProgram();
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { GLsizei n = 0; glGetShaderInfoLog(s, sizeof log, &n, log); printf("   shader: %.*s\n", n, log); }
    EXPECT_EQ_INT(ok, GL_TRUE);
    glAttachShader(p, s);
    /* separable, like every program dEQP's separate_shader tests put in a pipeline
     * (ProgramSeparable(true) / glCreateShaderProgramv).  A NON-separable program is
     * rejected by glUseProgramStages with INVALID_OPERATION no matter which stage bits
     * are selected — that is what KHR-GL46.sepshaderobjs.UseProgStagesApi verifies
     * (and MGL passes); do not relax mglUseProgramStages for it. */
    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { GLsizei n = 0; glGetProgramInfoLog(p, sizeof log, &n, log); printf("   link: %.*s\n", n, log); }
    glDeleteShader(s);
    EXPECT_EQ_INT(ok, GL_TRUE);
    return p;
}

TEST(pipeline_varying_match)
{
    float mono[4], pipe[4];
    GLuint vs_only = compile_one(GL_VERTEX_SHADER, VS);
    GLuint fs_only = compile_one(GL_FRAGMENT_SHADER, FS);
    GLuint monolithic = glCreateProgram();
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
    GLint ok = 0;
    char log[2048];
    GLuint pipe_obj = 0, vao = 0;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0,0,64,64); glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glShaderSource(s1, 1, &VS, NULL); glCompileShader(s1);
    glShaderSource(s2, 1, &FS, NULL); glCompileShader(s2);
    glAttachShader(monolithic, s1); glAttachShader(monolithic, s2);
    glLinkProgram(monolithic);
    glGetProgramiv(monolithic, GL_LINK_STATUS, &ok);
    if (!ok) { GLsizei n = 0; glGetProgramInfoLog(monolithic, sizeof log, &n, log); printf("   mono link: %.*s\n", n, log); }
    EXPECT_EQ_INT(ok, GL_TRUE);

    glUseProgram(monolithic);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_FLOAT, mono);

    glGenProgramPipelines(1, &pipe_obj);
    glUseProgramStages(pipe_obj, GL_VERTEX_SHADER_BIT, vs_only);
    glUseProgramStages(pipe_obj, GL_FRAGMENT_SHADER_BIT, fs_only);
    glBindProgramPipeline(pipe_obj);
    EXPECT_NO_GL_ERROR();
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_FLOAT, pipe);
    glBindProgramPipeline(0);
    glUseProgram(0);
    EXPECT_NO_GL_ERROR();

    printf("   mono=(%g,%g,%g,%g) pipe=(%g,%g,%g,%g)\n", mono[0], mono[1], mono[2], mono[3],
           pipe[0], pipe[1], pipe[2], pipe[3]);
    for (int i = 0; i < 4; i++)
        if (mono[i] != pipe[i]) MGL_FAIL("channel %d: pipeline %g != monolithic %g\n", i, pipe[i], mono[i]);

    glDeleteProgram(monolithic); glDeleteProgram(vs_only); glDeleteProgram(fs_only);
    glDeleteProgramPipelines(1, &pipe_obj);
    glDeleteShader(s1); glDeleteShader(s2);
}

TEST(pipeline_flat_varying_provoking)
{
    /* Metal's flat interpolation takes the FIRST vertex of a primitive; GL takes the
     * LAST.  The rstate domain rotates the index list when the fragment stage has a
     * flat input — but its probe reads ctx->state.program, which is NULL while a
     * program pipeline draws, so flat varyings through a pipeline took vertex 0's
     * value (dEQP GL45-ES31 separate_shader.random.111). */
    static const char *fvs = "#version 310 es\n"
        "uniform highp float scale;\n"
        "flat out highp vec2 var0;\n"
        "const vec2 triangle[3] = vec2[3](\n"
        "    vec2( 0.9375,  0.3125),\n"
        "    vec2(-0.4375,  0.9375),\n"
        "    vec2( 0.3750, -0.6250)\n"
        ");\n"
        "const vec2 var0Inits[3] = vec2[3](\n"
        "    vec2( 0.5000,  0.5625),\n"
        "    vec2( 0.8750,  0.1875),\n"
        "    vec2( 0.3125,  0.3125));\n"
        "void main (void)\n"
        "{\n"
        "    gl_Position = vec4(scale * triangle[gl_VertexID], 0.0, 1.0);\n"
        "    var0 = var0Inits[gl_VertexID];\n"
        "}\n";
    static const char *ffs = "#version 310 es\n"
        "precision highp float;\n"
        "out vec4 fragColor;\n"
        "uniform highp float scale;\n"
        "flat in highp vec2 var0;\n"
        "void main (void)\n"
        "{\n"
        "    fragColor = vec4(vec3(scale), 1.0) * vec4(vec3(var0, 1.0), 1.0);\n"
        "}\n";
    float mono[4], pipe[4];
    GLuint vao = 0, pl = 0, vp = compile_one(GL_VERTEX_SHADER, fvs), fp = compile_one(GL_FRAGMENT_SHADER, ffs);
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER), p = glCreateProgram();

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    glShaderSource(s1, 1, &fvs, NULL); glCompileShader(s1);
    glShaderSource(s2, 1, &ffs, NULL); glCompileShader(s2);
    glAttachShader(p, s1); glAttachShader(p, s2);
    glLinkProgram(p);
    {
        GLint lk = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &lk);
        EXPECT_EQ_INT(lk, GL_TRUE);
    }
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "scale"), 0.5625f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(37, 36, 1, 1, GL_RGBA, GL_FLOAT, mono);   /* inside the triangle, near v0 */

    glUseProgram(0);
    glProgramUniform1f(vp, glGetUniformLocation(vp, "scale"), 0.5625f);
    glProgramUniform1f(fp, glGetUniformLocation(fp, "scale"), 0.5625f);
    glGenProgramPipelines(1, &pl);
    glUseProgramStages(pl, GL_VERTEX_SHADER_BIT, vp);
    glUseProgramStages(pl, GL_FRAGMENT_SHADER_BIT, fp);
    glBindProgramPipeline(pl);
    EXPECT_NO_GL_ERROR();
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(37, 36, 1, 1, GL_RGBA, GL_FLOAT, pipe);
    glBindProgramPipeline(0);
    glUseProgram(0);
    EXPECT_NO_GL_ERROR();

    printf("   mono=(%g,%g,%g,%g) pipe=(%g,%g,%g,%g)\n", mono[0], mono[1], mono[2], mono[3],
           pipe[0], pipe[1], pipe[2], pipe[3]);
    /* flat must read the LAST vertex's value (0.3125, 0.3125): 0.5625 * 0.3125 */
    EXPECT_NEAR(mono[0], 0.5625f * 0.3125f, 1e-3);
    for (int i = 0; i < 4; i++)
        if (mono[i] != pipe[i]) MGL_FAIL("channel %d: pipeline %g != monolithic %g\n", i, pipe[i], mono[i]);

    glDeleteProgram(p); glDeleteShader(s1); glDeleteShader(s2);
    glDeleteProgramPipelines(1, &pl);
}

int main(void)
{
    RUN(pipeline_varying_match);
    RUN(pipeline_flat_varying_provoking);
    return mgl_test_summary();
}
