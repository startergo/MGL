/* test_program.c — program / shader / pipeline / binary / SPIR-V domain tests */
#include "mgl_test.h"
#include <stdint.h>
#include "spirv_spec_comp.h"

#define SSBO_WORDS 2048   /* >= 4 KiB so the renderer backs it with an MTLBuffer */

static GLuint make_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

static GLuint make_ssbo(GLuint binding)
{
    static GLuint zero[SSBO_WORDS];
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof zero, zero, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
    return buf;
}

static void read_ssbo(GLuint buf, GLuint *out, int n)
{
    glFinish();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(GLuint), out);
}

static const char *cs_src =
    "#version 450 core\n"
    "layout(local_size_x=4) in;\n"
    "layout(std430, binding=0) buffer B { uint v[]; };\n"
    "void main(){ v[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 3u + 1u; }\n";

static const char *vs_src =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=1) in vec2 extra;\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "void main(){ gl_Position = vec4(pos + extra * 0.0, 0.0, 1.0); }\n";

static const char *fs_src =
    "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "layout(binding=0) uniform sampler2D tex;\n"
    "layout(std140, binding=1) uniform Params { vec4 tint; float scale; } params;\n"
    "void main(){ color = texture(tex, vec2(0.5)) * params.tint * params.scale; }\n";

TEST(compute_link_and_dispatch)
{
    GLuint s = make_shader(GL_COMPUTE_SHADER, cs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    GLint wg[3] = {0}; glGetProgramiv(p, GL_COMPUTE_WORK_GROUP_SIZE, wg);
    EXPECT_EQ_INT(wg[0], 4); EXPECT_EQ_INT(wg[1], 1); EXPECT_EQ_INT(wg[2], 1);
    GLuint buf = make_ssbo(0);
    glUseProgram(p);
    GLint cur = 0; glGetIntegerv(GL_CURRENT_PROGRAM, &cur);
    EXPECT_EQ_INT(cur, p);
    glDispatchCompute(2, 1, 1);
    GLuint out[9]; read_ssbo(buf, out, 9);
    for (int i = 0; i < 8; i++) EXPECT_EQ_INT(out[i], i * 3 + 1);
    EXPECT_EQ_INT(out[8], 0);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
    glDeleteProgram(p); glDeleteShader(s); glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

TEST(shader_queries)
{
    GLuint s = make_shader(GL_FRAGMENT_SHADER, fs_src);
    GLint v = -1;
    glGetShaderiv(s, GL_SHADER_TYPE, &v);        EXPECT_EQ_INT(v, GL_FRAGMENT_SHADER);
    glGetShaderiv(s, GL_COMPILE_STATUS, &v);     EXPECT_EQ_INT(v, GL_TRUE);
    glGetShaderiv(s, GL_DELETE_STATUS, &v);      EXPECT_EQ_INT(v, GL_FALSE);
    glGetShaderiv(s, GL_SPIR_V_BINARY, &v);      EXPECT_EQ_INT(v, GL_FALSE);
    glGetShaderiv(s, GL_SHADER_SOURCE_LENGTH, &v); EXPECT_EQ_INT(v, (GLint)strlen(fs_src) + 1);
    char src[1024]; GLsizei len = 0;
    glGetShaderSource(s, sizeof src, &len, src);
    EXPECT_EQ_INT(len, (GLint)strlen(fs_src));
    EXPECT_STREQ(src, fs_src);
    glGetShaderSource(s, 8, &len, src);          /* truncation: 7 chars + NUL */
    EXPECT_EQ_INT(len, 7);
    EXPECT_EQ_INT(src[7], 0);
    EXPECT_NO_GL_ERROR();

    glGetShaderiv(s, GL_LINK_STATUS, &v);        EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetShaderiv(987654, GL_SHADER_TYPE, &v);   EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLuint p = glCreateProgram();
    glGetShaderiv(p, GL_SHADER_TYPE, &v);        EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetProgramiv(s, GL_LINK_STATUS, &v);       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glCreateShader(GL_RGBA);                     EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* compile error: real glslang diagnostics in the info log */
    GLuint bad = make_shader(GL_VERTEX_SHADER, "#version 450 core\nvoid main(){ undefined_fn(); }\n");
    glGetShaderiv(bad, GL_COMPILE_STATUS, &v);   EXPECT_EQ_INT(v, GL_FALSE);
    GLint loglen = 0; glGetShaderiv(bad, GL_INFO_LOG_LENGTH, &loglen);
    EXPECT_TRUE(loglen > 1);
    char log[2048] = ""; glGetShaderInfoLog(bad, sizeof log, &len, log);
    EXPECT_EQ_INT(len, loglen - 1);
    EXPECT_TRUE(strstr(log, "undefined_fn") != NULL);
    glGetShaderInfoLog(bad, -1, &len, log);      EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDeleteShader(bad); glDeleteShader(s); glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

TEST(program_queries_and_link_errors)
{
    GLuint p = glCreateProgram();
    GLint v = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &v);       EXPECT_EQ_INT(v, GL_FALSE);
    glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &v); EXPECT_EQ_INT(v, 0);
    glGetProgramiv(p, GL_COMPUTE_WORK_GROUP_SIZE, &v); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUseProgram(p);                             EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* empty link fails with a log */
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &v);       EXPECT_EQ_INT(v, GL_FALSE);
    glGetProgramiv(p, GL_INFO_LOG_LENGTH, &v);   EXPECT_TRUE(v > 1);

    GLuint vs = make_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = make_shader(GL_FRAGMENT_SHADER, fs_src);
    glAttachShader(p, vs); glAttachShader(p, fs);
    glAttachShader(p, vs);                       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetProgramiv(p, GL_ATTACHED_SHADERS, &v);  EXPECT_EQ_INT(v, 2);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &v);       EXPECT_EQ_INT(v, GL_TRUE);
    glGetProgramiv(p, GL_ACTIVE_ATTRIBUTES, &v); EXPECT_EQ_INT(v, 2);
    glGetProgramiv(p, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &v); EXPECT_EQ_INT(v, 6);
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCKS, &v); EXPECT_EQ_INT(v, 1);
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &v); EXPECT_EQ_INT(v, 7);
    glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &v);   EXPECT_EQ_INT(v, 3);   /* tex, Params.tint, Params.scale */
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_MAX_LENGTH, &v); EXPECT_EQ_INT(v, 13);
    glGetProgramiv(p, GL_PROGRAM_SEPARABLE, &v); EXPECT_EQ_INT(v, GL_FALSE);
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &v); EXPECT_EQ_INT(v, GL_INTERLEAVED_ATTRIBS);
    EXPECT_EQ_INT(glGetAttribLocation(p, "extra"), 1);
    EXPECT_EQ_INT(glGetAttribLocation(p, "nope"), -1);
    glGetProgramiv(p, GL_GEOMETRY_VERTICES_OUT, &v); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetProgramiv(p, GL_SHADER_TYPE, &v);       EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* detaching after link keeps the executable */
    glDetachShader(p, vs);
    glDetachShader(p, vs);                       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetProgramiv(p, GL_ATTACHED_SHADERS, &v);  EXPECT_EQ_INT(v, 1);
    glGetProgramiv(p, GL_LINK_STATUS, &v);       EXPECT_EQ_INT(v, GL_TRUE);
    glUseProgram(p);                             EXPECT_NO_GL_ERROR();
    glUseProgram(0);

    /* a failing relink reports LINK_STATUS FALSE with the compile problem in the log */
    GLuint badfs = make_shader(GL_FRAGMENT_SHADER, "#version 450 core\nvoid main(){ x = 1; }\n");
    GLuint p2 = glCreateProgram();
    glAttachShader(p2, vs); glAttachShader(p2, badfs);
    glLinkProgram(p2);
    glGetProgramiv(p2, GL_LINK_STATUS, &v);      EXPECT_EQ_INT(v, GL_FALSE);
    char log[1024] = ""; glGetProgramInfoLog(p2, sizeof log, NULL, log);
    EXPECT_TRUE(strstr(log, "not successfully compiled") != NULL);

    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, 5); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glProgramParameteri(p, GL_LINK_STATUS, 1);   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glProgramParameteri(p, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    glGetProgramiv(p, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, &v); EXPECT_EQ_INT(v, GL_TRUE);

    glDeleteProgram(p); glDeleteProgram(p2);
    glDeleteShader(vs); glDeleteShader(fs); glDeleteShader(badfs);
    glDeleteProgram(0); glDeleteShader(0);       EXPECT_NO_GL_ERROR();
    glDeleteProgram(123456);                     EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(delete_refcount_semantics)
{
    GLuint cs = make_shader(GL_COMPUTE_SHADER, cs_src);
    GLuint p = glCreateProgram();
    GLint v = -1;
    glAttachShader(p, cs);
    glDeleteShader(cs);                          /* attached: flagged only */
    EXPECT_EQ_INT(glIsShader(cs), GL_TRUE);
    glGetShaderiv(cs, GL_DELETE_STATUS, &v);     EXPECT_EQ_INT(v, GL_TRUE);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &v);       EXPECT_EQ_INT(v, GL_TRUE);
    glDetachShader(p, cs);                       /* last reference: freed */
    EXPECT_EQ_INT(glIsShader(cs), GL_FALSE);

    glUseProgram(p);
    glDeleteProgram(p);                          /* current: flagged only */
    EXPECT_EQ_INT(glIsProgram(p), GL_TRUE);
    glGetProgramiv(p, GL_DELETE_STATUS, &v);     EXPECT_EQ_INT(v, GL_TRUE);
    GLuint buf = make_ssbo(0);
    glDispatchCompute(1, 1, 1);                  /* still usable */
    GLuint out[4]; read_ssbo(buf, out, 4);
    EXPECT_EQ_INT(out[3], 10);
    glUseProgram(0);
    EXPECT_EQ_INT(glIsProgram(p), GL_FALSE);
    glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

TEST(pipeline_objects)
{
    GLuint pipe = 0, v2;
    GLint v = -1;
    glGenProgramPipelines(1, &pipe);
    EXPECT_EQ_INT(glIsProgramPipeline(pipe), GL_FALSE);   /* not an object until bound */
    glBindProgramPipeline(pipe);
    EXPECT_EQ_INT(glIsProgramPipeline(pipe), GL_TRUE);
    glGetIntegerv(GL_PROGRAM_PIPELINE_BINDING, &v); EXPECT_EQ_INT(v, pipe);
    glBindProgramPipeline(424242);               EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glCreateProgramPipelines(1, &v2);
    EXPECT_EQ_INT(glIsProgramPipeline(v2), GL_TRUE);

    GLuint vsp = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vs_src);
    GLuint fsp = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fs_src);
    glGetProgramiv(vsp, GL_PROGRAM_SEPARABLE, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glGetProgramiv(vsp, GL_LINK_STATUS, &v);       EXPECT_EQ_INT(v, GL_TRUE);
    glGetProgramiv(vsp, GL_ATTACHED_SHADERS, &v);  EXPECT_EQ_INT(v, 0);

    /* non-separable program cannot be installed */
    GLuint ns = glCreateProgram();
    GLuint vs = make_shader(GL_VERTEX_SHADER, vs_src);
    glAttachShader(ns, vs); glLinkProgram(ns);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, ns); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUseProgramStages(pipe, 0x80000000u >> 1, vsp);    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUseProgramStages(777777, GL_VERTEX_SHADER_BIT, vsp); EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    EXPECT_NO_GL_ERROR();
    glGetProgramPipelineiv(pipe, GL_VERTEX_SHADER, &v);   EXPECT_EQ_INT(v, vsp);
    glGetProgramPipelineiv(pipe, GL_FRAGMENT_SHADER, &v); EXPECT_EQ_INT(v, 0);

    /* vertex without fragment: validation fails with a log */
    glValidateProgramPipeline(pipe);
    glGetProgramPipelineiv(pipe, GL_VALIDATE_STATUS, &v); EXPECT_EQ_INT(v, GL_FALSE);
    glGetProgramPipelineiv(pipe, GL_INFO_LOG_LENGTH, &v); EXPECT_TRUE(v > 1);

    /* stages the program has no executable for are cleared */
    glUseProgramStages(pipe, GL_ALL_SHADER_BITS, fsp);
    glGetProgramPipelineiv(pipe, GL_VERTEX_SHADER, &v);   EXPECT_EQ_INT(v, 0);
    glGetProgramPipelineiv(pipe, GL_FRAGMENT_SHADER, &v); EXPECT_EQ_INT(v, fsp);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    glValidateProgramPipeline(pipe);
    glGetProgramPipelineiv(pipe, GL_VALIDATE_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);
    char log[512] = ""; GLsizei len = 0;
    glGetProgramPipelineInfoLog(pipe, sizeof log, &len, log);
    EXPECT_TRUE(len > 0);

    glActiveShaderProgram(pipe, fsp);
    glGetProgramPipelineiv(pipe, GL_ACTIVE_PROGRAM, &v);  EXPECT_EQ_INT(v, fsp);
    glGetProgramPipelineiv(pipe, GL_LINK_STATUS, &v);     EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* programs referenced by a pipeline survive deletion until released */
    glDeleteProgram(vsp);
    EXPECT_EQ_INT(glIsProgram(vsp), GL_TRUE);
    glDeleteProgramPipelines(1, &pipe);
    EXPECT_EQ_INT(glIsProgram(vsp), GL_FALSE);
    EXPECT_EQ_INT(glIsProgramPipeline(pipe), GL_FALSE);
    glGetIntegerv(GL_PROGRAM_PIPELINE_BINDING, &v); EXPECT_EQ_INT(v, 0);
    glDeleteProgramPipelines(1, &v2);
    glDeleteProgram(fsp); glDeleteProgram(ns); glDeleteShader(vs);
    EXPECT_NO_GL_ERROR();
}

TEST(pipeline_compute_dispatch)
{
    GLuint csp = glCreateShaderProgramv(GL_COMPUTE_SHADER, 1, &cs_src);
    GLuint pipe;
    glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_COMPUTE_SHADER_BIT, csp);
    glUseProgram(0);
    glBindProgramPipeline(pipe);
    GLuint buf = make_ssbo(0);
    glDispatchCompute(3, 1, 1);
    GLuint out[13]; read_ssbo(buf, out, 13);
    for (int i = 0; i < 12; i++) EXPECT_EQ_INT(out[i], i * 3 + 1);
    EXPECT_EQ_INT(out[12], 0);
    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe); glDeleteProgram(csp); glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

TEST(pipeline_draw_separable)
{
    /* separate VS + FS programs; the FS reports through an SSBO so the result does not
     * depend on framebuffer readback */
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 uv;\n"
        "out gl_PerVertex { vec4 gl_Position; };\n"
        "void main(){ uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) in vec2 uv;\n"
        "layout(location=0) out vec4 color;\n"
        "layout(std430, binding=2) buffer Out { uint hits; uint marker; };\n"
        "void main(){ atomicAdd(hits, 1u); marker = 0xC0FFEEu; color = vec4(uv, 0.0, 1.0); }\n";
    GLuint vsp = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vs);
    GLuint fsp = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fs);
    GLint v = 0;
    glGetProgramiv(vsp, GL_LINK_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glGetProgramiv(fsp, GL_LINK_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);

    GLuint pipe;
    glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fsp);
    glBindProgramPipeline(pipe);
    glValidateProgramPipeline(pipe);
    glGetProgramPipelineiv(pipe, GL_VALIDATE_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);

    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    GLuint buf = make_ssbo(2);

    glUseProgram(0);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLuint out[2]; read_ssbo(buf, out, 2);
    EXPECT_EQ_INT(out[1], 0xC0FFEE);
    EXPECT_TRUE(out[0] > 0);                 /* fragments ran */

    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(vsp); glDeleteProgram(fsp);
    glDeleteBuffers(1, &buf); glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    EXPECT_NO_GL_ERROR();
}

TEST(program_binary_roundtrip)
{
    GLint n = 0, fmt = 0;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &n);  EXPECT_EQ_INT(n, 1);
    glGetIntegerv(GL_PROGRAM_BINARY_FORMATS, &fmt);
    EXPECT_TRUE(fmt != 0);

    GLuint src = glCreateShaderProgramv(GL_COMPUTE_SHADER, 1, &cs_src);
    GLint len = 0; glGetProgramiv(src, GL_PROGRAM_BINARY_LENGTH, &len);
    EXPECT_TRUE(len > 24);
    void *bin = malloc(len);
    GLsizei got = 0; GLenum format = 0;
    glGetProgramBinary(src, len - 1, &got, &format, bin);  EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetProgramBinary(src, len, &got, &format, bin);
    EXPECT_EQ_INT(got, len);
    EXPECT_EQ_INT(format, fmt);
    EXPECT_NO_GL_ERROR();

    GLuint dst = glCreateProgram();
    glProgramBinary(dst, format, bin, got);
    GLint v = 0;
    glGetProgramiv(dst, GL_LINK_STATUS, &v);   EXPECT_EQ_INT(v, GL_TRUE);
    glGetProgramiv(dst, GL_PROGRAM_SEPARABLE, &v); EXPECT_EQ_INT(v, GL_TRUE);
    GLint wg[3] = {0}; glGetProgramiv(dst, GL_COMPUTE_WORK_GROUP_SIZE, wg); EXPECT_EQ_INT(wg[0], 4);
    EXPECT_NO_GL_ERROR();

    /* identical output from the original and the restored program */
    GLuint out_a[8], out_b[8];
    GLuint buf = make_ssbo(0);
    glUseProgram(src); glDispatchCompute(2, 1, 1); read_ssbo(buf, out_a, 8);
    glDeleteBuffers(1, &buf); buf = make_ssbo(0);
    glUseProgram(dst); glDispatchCompute(2, 1, 1); read_ssbo(buf, out_b, 8);
    for (int i = 0; i < 8; i++) { EXPECT_EQ_INT(out_b[i], out_a[i]); EXPECT_EQ_INT(out_b[i], i * 3 + 1); }
    glUseProgram(0);

    /* corrupt binary: LINK_STATUS FALSE, no GL error, no crash */
    ((unsigned char *)bin)[len / 2] ^= 0x5A;
    glProgramBinary(dst, format, bin, got);
    EXPECT_NO_GL_ERROR();
    glGetProgramiv(dst, GL_LINK_STATUS, &v);   EXPECT_EQ_INT(v, GL_FALSE);
    char log[256] = ""; glGetProgramInfoLog(dst, sizeof log, NULL, log);
    EXPECT_TRUE(strstr(log, "binary") != NULL);
    glProgramBinary(dst, format, bin, 7);      /* truncated */
    glGetProgramiv(dst, GL_LINK_STATUS, &v);   EXPECT_EQ_INT(v, GL_FALSE);
    EXPECT_NO_GL_ERROR();
    glProgramBinary(dst, 0x1234, bin, got);    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetProgramBinary(dst, len, &got, &format, bin); EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    free(bin);
    glDeleteProgram(src); glDeleteProgram(dst); glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

TEST(spirv_shader_binary_specialize)
{
    GLint n = 0, f = 0;
    glGetIntegerv(GL_NUM_SHADER_BINARY_FORMATS, &n);  EXPECT_EQ_INT(n, 1);
    glGetIntegerv(GL_SHADER_BINARY_FORMATS, &f);      EXPECT_EQ_INT(f, GL_SHADER_BINARY_FORMAT_SPIR_V);
    glGetIntegerv(GL_NUM_SPIR_V_EXTENSIONS, &n);      EXPECT_TRUE(n > 0);
    const GLubyte *ext = glGetStringi(GL_SPIR_V_EXTENSIONS, 0);
    EXPECT_TRUE(ext && !strncmp((const char *)ext, "SPV_", 4));
    EXPECT_NO_GL_ERROR();

    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &s, 0x1234, spec_comp_spv, sizeof spec_comp_spv); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, spec_comp_spv, sizeof spec_comp_spv);
    EXPECT_NO_GL_ERROR();
    GLint v = 0;
    glGetShaderiv(s, GL_SPIR_V_BINARY, &v);    EXPECT_EQ_INT(v, GL_TRUE);
    glGetShaderiv(s, GL_COMPILE_STATUS, &v);   EXPECT_EQ_INT(v, GL_FALSE);

    GLuint idx[2] = { 7, 9 }, val[2] = { 5, 1 };
    GLuint badidx = 3;
    glSpecializeShader(s, "nope", 0, NULL, NULL);       EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glSpecializeShader(s, "main", 1, &badidx, val);     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glSpecializeShader(s, "main", 2, idx, val);
    EXPECT_NO_GL_ERROR();
    glGetShaderiv(s, GL_COMPILE_STATUS, &v);   EXPECT_EQ_INT(v, GL_TRUE);
    glSpecializeShader(s, "main", 2, idx, val);         EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &v);     EXPECT_EQ_INT(v, GL_TRUE);
    GLuint buf = make_ssbo(0);
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    GLuint out[5]; read_ssbo(buf, out, 5);
    for (int i = 0; i < 4; i++) EXPECT_EQ_INT(out[i], 500 + i + 50);
    EXPECT_EQ_INT(out[4], 0);
    glUseProgram(0);

    /* SPIR-V program survives a binary round trip with its specialization */
    GLint len = 0; glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &len);
    void *bin = malloc(len); GLenum fmt; GLsizei got;
    glGetProgramBinary(p, len, &got, &fmt, bin);
    GLuint p2 = glCreateProgram();
    glProgramBinary(p2, fmt, bin, got);
    glGetProgramiv(p2, GL_LINK_STATUS, &v);    EXPECT_EQ_INT(v, GL_TRUE);
    free(bin);

    /* unspecialized SPIR-V and GLSL+SPIR-V mixes do not link */
    GLuint s2 = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &s2, GL_SHADER_BINARY_FORMAT_SPIR_V, spec_comp_spv, sizeof spec_comp_spv);
    GLuint p3 = glCreateProgram();
    glAttachShader(p3, s2); glLinkProgram(p3);
    glGetProgramiv(p3, GL_LINK_STATUS, &v);    EXPECT_EQ_INT(v, GL_FALSE);
    GLuint vs = make_shader(GL_VERTEX_SHADER, vs_src);
    glAttachShader(p3, vs);
    glSpecializeShader(s2, "main", 0, NULL, NULL);
    glLinkProgram(p3);
    glGetProgramiv(p3, GL_LINK_STATUS, &v);    EXPECT_EQ_INT(v, GL_FALSE);
    char log[512] = ""; glGetProgramInfoLog(p3, sizeof log, NULL, log);
    EXPECT_TRUE(strstr(log, "SPIR-V") != NULL);

    /* source replaces the module */
    glShaderSource(s2, 1, &cs_src, NULL);
    glGetShaderiv(s2, GL_SPIR_V_BINARY, &v);   EXPECT_EQ_INT(v, GL_FALSE);
    glCompileShader(s2);
    glGetShaderiv(s2, GL_COMPILE_STATUS, &v);  EXPECT_EQ_INT(v, GL_TRUE);

    glDeleteProgram(p); glDeleteProgram(p2); glDeleteProgram(p3);
    glDeleteShader(s); glDeleteShader(s2); glDeleteShader(vs); glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

TEST(validate_program_samplers)
{
    const char *fs_ok = "#version 450 core\n"
        "layout(location=0) out vec4 c;\n"
        "layout(binding=0) uniform sampler2D a;\n"
        "layout(binding=1) uniform sampler3D b;\n"
        "void main(){ c = texture(a, vec2(0.5)) + texture(b, vec3(0.5)); }\n";
    const char *fs_bad = "#version 450 core\n"
        "layout(location=0) out vec4 c;\n"
        "layout(binding=0) uniform sampler2D a;\n"
        "layout(binding=0) uniform sampler3D b;\n"
        "void main(){ c = texture(a, vec2(0.5)) + texture(b, vec3(0.5)); }\n";
    GLuint good = mgl_make_program(vs_src, fs_ok);
    GLuint bad = mgl_make_program(vs_src, fs_bad);
    if (!good || !bad) return;
    GLint v = -1;
    glGetProgramiv(good, GL_VALIDATE_STATUS, &v); EXPECT_EQ_INT(v, GL_FALSE);   /* not validated yet */
    glValidateProgram(good);
    glGetProgramiv(good, GL_VALIDATE_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glValidateProgram(bad);
    glGetProgramiv(bad, GL_VALIDATE_STATUS, &v);  EXPECT_EQ_INT(v, GL_FALSE);
    char log[512] = ""; glGetProgramInfoLog(bad, sizeof log, NULL, log);
    EXPECT_TRUE(strstr(log, "texture image unit 0") != NULL);
    glValidateProgram(31337);                     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDeleteProgram(good); glDeleteProgram(bad);
    EXPECT_NO_GL_ERROR();
}

TEST(unsupported_stages_fail_link)
{
    const char *gs = "#version 450 core\n"
        "layout(triangles) in; layout(triangle_strip, max_vertices=3) out;\n"
        "void main(){ for(int i=0;i<3;i++){ gl_Position = gl_in[i].gl_Position; EmitVertex(); } EndPrimitive(); }\n";
    GLuint vs = make_shader(GL_VERTEX_SHADER, vs_src);
    GLuint g = make_shader(GL_GEOMETRY_SHADER, gs);
    GLint v = 0;
    glGetShaderiv(g, GL_COMPILE_STATUS, &v);   EXPECT_EQ_INT(v, GL_TRUE);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, g);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &v);     EXPECT_EQ_INT(v, GL_FALSE);
    char log[512] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
    EXPECT_TRUE(strstr(log, "geometry shaders are not supported") != NULL);

    GLuint cs = make_shader(GL_COMPUTE_SHADER, cs_src);
    GLuint p2 = glCreateProgram();
    glAttachShader(p2, vs); glAttachShader(p2, cs);
    glLinkProgram(p2);
    glGetProgramiv(p2, GL_LINK_STATUS, &v);    EXPECT_EQ_INT(v, GL_FALSE);
    glDeleteProgram(p); glDeleteProgram(p2);
    glDeleteShader(vs); glDeleteShader(g); glDeleteShader(cs);
    EXPECT_NO_GL_ERROR();
}

TEST(release_compiler_and_precision)
{
    glReleaseShaderCompiler();
    EXPECT_NO_GL_ERROR();
    GLuint p = glCreateShaderProgramv(GL_COMPUTE_SHADER, 1, &cs_src);
    GLint v = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &v);     EXPECT_EQ_INT(v, GL_TRUE);
    glDeleteProgram(p);
    glReleaseShaderCompiler();
    GLuint s = make_shader(GL_FRAGMENT_SHADER, fs_src);
    glGetShaderiv(s, GL_COMPILE_STATUS, &v);   EXPECT_EQ_INT(v, GL_TRUE);
    /* release with a live compiled shader is deferred until it is deleted */
    glReleaseShaderCompiler();
    GLuint legacy_src_shader = make_shader(GL_VERTEX_SHADER,
        "#version 140\nin vec2 pos;\nvoid main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n");
    glGetShaderiv(legacy_src_shader, GL_COMPILE_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glReleaseShaderCompiler();
    glDeleteShader(s);
    glDeleteShader(legacy_src_shader);           /* last live shader: compiler state torn down here */
    /* everything still compiles and links afterwards, including legacy-version upgrades */
    GLuint again = make_shader(GL_VERTEX_SHADER,
        "#version 140\nin vec2 pos;\nvoid main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n");
    glGetShaderiv(again, GL_COMPILE_STATUS, &v); EXPECT_EQ_INT(v, GL_TRUE);
    GLuint fs2 = make_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint p2 = glCreateProgram();
    glAttachShader(p2, again); glAttachShader(p2, fs2); glLinkProgram(p2);
    glGetProgramiv(p2, GL_LINK_STATUS, &v);    EXPECT_EQ_INT(v, GL_TRUE);
    glDeleteProgram(p2); glDeleteShader(again); glDeleteShader(fs2);

    GLint range[2] = {0}, prec = -1;
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_HIGH_FLOAT, range, &prec);
    EXPECT_EQ_INT(range[0], 127); EXPECT_EQ_INT(range[1], 127); EXPECT_EQ_INT(prec, 23);
    glGetShaderPrecisionFormat(GL_VERTEX_SHADER, GL_LOW_INT, range, &prec);
    EXPECT_EQ_INT(range[0], 31); EXPECT_EQ_INT(range[1], 30); EXPECT_EQ_INT(prec, 0);
    glGetShaderPrecisionFormat(GL_COMPUTE_SHADER, GL_HIGH_FLOAT, range, &prec); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetShaderPrecisionFormat(GL_VERTEX_SHADER, GL_RGBA, range, &prec);        EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* CreateShaderProgramv with a broken shader: program exists, link fails, log carries the error */
    const char *broken = "#version 450 core\nvoid main(){ nope(); }\n";
    p = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &broken);
    EXPECT_TRUE(p != 0);
    glGetProgramiv(p, GL_LINK_STATUS, &v);     EXPECT_EQ_INT(v, GL_FALSE);
    char log[1024] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
    EXPECT_TRUE(strstr(log, "nope") != NULL);
    glDeleteProgram(p);
    glCreateShaderProgramv(GL_RGBA, 1, &broken); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_NO_GL_ERROR();
}

TEST(instance_and_vertex_id_bases)
{
    /* GL: gl_InstanceID excludes baseinstance, gl_VertexID includes basevertex */
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 ids;\n"
        "void main(){ ids = vec2(gl_InstanceID == 0 ? 1.0 : 0.0, gl_VertexID >= 3 ? 1.0 : 0.0);\n"
        "  gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) in vec2 ids;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(ids, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    /* verts 0..2 are off screen, 3..5 cover the viewport */
    static const float v[] = { 5,5, 6,5, 5,6,   -1,-1, 3,-1, -1,3 };
    static const GLuint idx[] = { 0, 1, 2 };
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glUseProgram(p);

    glClearColor(0, 0, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
    glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 3, 3, 1, 5);
    EXPECT_PIXEL(10, 10, 255, 255, 0, 255, 2);      /* instance id 0 despite baseinstance 5 */

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0, 3);
    EXPECT_PIXEL(10, 10, 255, 255, 0, 255, 2);      /* vertex ids 3..5 */
    EXPECT_NO_GL_ERROR();

    glUseProgram(0);
    glDeleteProgram(p); glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &ebo); glDeleteVertexArrays(1, &vao);
}

TEST(pipeline_and_binary_draw_pixels)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "out gl_PerVertex { vec4 gl_Position; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0, 0.0, 1.0, 1.0); }\n";
    static const float tri[] = { -1,-1, 3,-1, -1,3 };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    /* separable stages through a pipeline */
    GLuint vsp = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vs);
    GLuint fsp = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fs);
    GLuint pipe; glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fsp);
    glUseProgram(0);
    glBindProgramPipeline(pipe);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(10, 10, 255, 0, 255, 255, 2);
    glBindProgramPipeline(0);

    /* a program restored from its binary draws the same image */
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    GLint len = 0; glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &len);
    void *bin = malloc(len); GLenum fmt = 0; GLsizei got = 0;
    glGetProgramBinary(p, len, &got, &fmt, bin);
    GLuint q = glCreateProgram();
    glProgramBinary(q, fmt, bin, got);
    free(bin);
    GLint ok = 0; glGetProgramiv(q, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_TRUE);
    glUseProgram(p);
    glClear(GL_COLOR_BUFFER_BIT); glDrawArrays(GL_TRIANGLES, 0, 3);
    GLubyte a[4]; mgl_read_rgba(10, 10, a);
    glUseProgram(q);
    glClear(GL_COLOR_BUFFER_BIT); glDrawArrays(GL_TRIANGLES, 0, 3);
    GLubyte b[4]; mgl_read_rgba(10, 10, b);
    EXPECT_TRUE(!memcmp(a, b, 4));
    EXPECT_PIXEL(10, 10, 255, 0, 255, 255, 2);
    EXPECT_NO_GL_ERROR();

    glUseProgram(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(vsp); glDeleteProgram(fsp); glDeleteProgram(p); glDeleteProgram(q);
    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
}

int main(void)
{
    RUN(compute_link_and_dispatch);
    RUN(shader_queries);
    RUN(program_queries_and_link_errors);
    RUN(delete_refcount_semantics);
    RUN(pipeline_objects);
    RUN(pipeline_compute_dispatch);
    RUN(pipeline_draw_separable);
    RUN(program_binary_roundtrip);
    RUN(spirv_shader_binary_specialize);
    RUN(validate_program_samplers);
    RUN(unsupported_stages_fail_link);
    RUN(release_compiler_and_precision);
    RUN(instance_and_vertex_id_bases);
    RUN(pipeline_and_binary_draw_pixels);
    return mgl_test_summary();
}
