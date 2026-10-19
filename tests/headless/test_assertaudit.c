/*
 * test_assertaudit — audit probe: legal GL 4.6 core calls must not raise errors,
 * no-op silently, or return dummy values (removed assert(0) sites).
 * Each case exercises input the spec says is legal and checks for spurious
 * GL errors / wrong query results.
 */
#include "mgl_test.h"

/* ---------- multisample texture storage (TexImage*Multisample / TextureStorage*Multisample) --- */
TEST(ms_texture_storage_legal)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, 32, 32, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    GLint samples = 0, ictx = 0;
    glGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA8, GL_SAMPLES, 1, &samples);
    EXPECT_NO_GL_ERROR();
    glGetTexLevelParameteriv(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_SAMPLES, &samples);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(samples, 4);
    glGetTexLevelParameteriv(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_FIXED_SAMPLE_LOCATIONS, &ictx);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(ictx, 1);

    glTextureStorage2DMultisample(tex, 4, GL_RGBA8, 16, 16, GL_FALSE);
    EXPECT_NO_GL_ERROR();
    glGetTextureLevelParameteriv(tex, 0, GL_TEXTURE_SAMPLES, &samples);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(samples, 4);

    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

TEST(ms_texture_storage_3d)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, tex);
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 4, GL_RGBA8, 8, 8, 4, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    GLint samples = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0, GL_TEXTURE_SAMPLES, &samples);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(samples, 4);

    glTextureStorage3DMultisample(tex, 4, GL_RGBA8, 8, 8, 2, GL_FALSE);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ---------- texture clears ---------- */
TEST(clear_tex_image_legal)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_NO_GL_ERROR();
    const GLfloat color[4] = { 0.25f, 0.5f, 0.75f, 1.0f };
    glClearTexImage(tex, 0, GL_RGBA, GL_FLOAT, color);
    EXPECT_NO_GL_ERROR();
    GLubyte px[4 * 16];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(px[0], 64);
    EXPECT_EQ_INT(px[1], 128);
    EXPECT_EQ_INT(px[2], 191);
    EXPECT_EQ_INT(px[3], 255);
    glClearTexSubImage(tex, 0, 1, 1, 0, 2, 2, 1, GL_RGBA, GL_FLOAT, color);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ---------- texture views ---------- */
TEST(texture_view_legal)
{
    GLuint tex = 0, view = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 3, GL_RGBA8, 8, 8);
    EXPECT_NO_GL_ERROR();
    glGenTextures(1, &view);
    glTextureView(view, GL_TEXTURE_2D, tex, GL_RGBA8, 0, 3, 0, 1);
    EXPECT_NO_GL_ERROR();
    GLint levels = 0, fmt = 0;
    glGetTextureLevelParameteriv(view, 0, GL_TEXTURE_WIDTH, &levels);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(levels, 8);
    glGetTextureParameteriv(view, GL_TEXTURE_IMMUTABLE_FORMAT, &fmt);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(fmt, GL_TRUE);
    glDeleteTextures(1, &view);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ---------- texture buffer ---------- */
TEST(texture_buffer_legal)
{
    GLuint tex = 0, buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    GLfloat data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    glBufferData(GL_TEXTURE_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    EXPECT_NO_GL_ERROR();
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, buf);
    EXPECT_NO_GL_ERROR();
    GLint sz = 0, ifmt = 0;
    glGetTextureLevelParameteriv(tex, 0, GL_TEXTURE_BUFFER_SIZE, &sz);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(sz, (int)sizeof(data));
    glGetTextureLevelParameteriv(tex, 0, GL_TEXTURE_INTERNAL_FORMAT, &ifmt);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(ifmt, GL_RGBA32F);

    glTextureBufferRange(tex, GL_R32F, buf, 16, 16);
    EXPECT_NO_GL_ERROR();
    glGetTextureLevelParameteriv(tex, 0, GL_TEXTURE_BUFFER_SIZE, &sz);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(sz, 16);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

/* ---------- get texture sub image ---------- */
TEST(get_texture_subimage_legal)
{
    GLuint tex = 0;
    GLubyte img[4 * 4 * 4], out[2 * 2 * 4];
    for (int i = 0; i < 4 * 4 * 4; i++) img[i] = (GLubyte)(i * 7);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    EXPECT_NO_GL_ERROR();
    
    glGetTextureSubImage(tex, 0, 1, 1, 0, 2, 2, 1, GL_RGBA, GL_UNSIGNED_BYTE, (GLsizei)sizeof(out), out);
    EXPECT_NO_GL_ERROR();
    
    /* row 0 of the sub-image = texels (1,1),(2,1) */
    EXPECT_EQ_INT(out[0], img[(1 * 4 + 1) * 4 + 0]);
    EXPECT_EQ_INT(out[4], img[(1 * 4 + 2) * 4 + 0]);
    /* row 1 = texels (1,2),(2,2) */
    EXPECT_EQ_INT(out[8], img[(2 * 4 + 1) * 4 + 0]);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ---------- invalidation hints are legal ---------- */
TEST(invalidate_legal)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glInvalidateTexImage(tex, 0);
    EXPECT_NO_GL_ERROR();
    glInvalidateTexSubImage(tex, 0, 0, 0, 0, 2, 2, 1);
    EXPECT_NO_GL_ERROR();
    glInvalidateBufferData(0); /* 0 is legal for tex? no - buffer only */
    while (glGetError() != GL_NO_ERROR) {}
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, 64, NULL, GL_STATIC_DRAW);
    glInvalidateBufferData(buf);
    EXPECT_NO_GL_ERROR();
    glInvalidateBufferSubData(buf, 8, 16);
    EXPECT_NO_GL_ERROR();
    glDeleteBuffers(1, &buf);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ---------- image units ---------- */
TEST(bind_image_textures_legal)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    EXPECT_NO_GL_ERROR();
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
    EXPECT_NO_GL_ERROR();
    GLint v = 0;
    glGetIntegeri_v(GL_IMAGE_BINDING_NAME, 0, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT((GLuint)v, tex);
    glBindImageTextures(1, 1, &tex);   /* bind to unit 1 via multi-bind */
    EXPECT_NO_GL_ERROR();
    glGetIntegeri_v(GL_IMAGE_BINDING_NAME, 1, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT((GLuint)v, tex);
    GLuint zero = 0;
    glBindImageTextures(0, 2, &zero);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* ---------- sampler integer params ---------- */
TEST(sampler_iparams_legal)
{
    GLuint s = 0;
    glGenSamplers(1, &s);
    const GLint iv[4] = { 1, 2, 3, 4 };
    glSamplerParameterIiv(s, GL_TEXTURE_BORDER_COLOR, iv);
    EXPECT_NO_GL_ERROR();
    GLint qiv[4] = { 0 };
    glGetSamplerParameterIiv(s, GL_TEXTURE_BORDER_COLOR, qiv);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(qiv[0], 1);
    EXPECT_EQ_INT(qiv[3], 4);
    const GLuint uv[4] = { 5, 6, 7, 8 };
    glSamplerParameterIuiv(s, GL_TEXTURE_BORDER_COLOR, uv);
    EXPECT_NO_GL_ERROR();
    GLuint quv[4] = { 0 };
    glGetSamplerParameterIuiv(s, GL_TEXTURE_BORDER_COLOR, quv);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(quv[0], 5);
    EXPECT_EQ_INT(quv[3], 8);
    /* the integer and unsigned views see the same raw bits */
    glGetSamplerParameterIiv(s, GL_TEXTURE_BORDER_COLOR, qiv);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(qiv[0], 5);
    glDeleteSamplers(1, &s);
    EXPECT_NO_GL_ERROR();
}

/* ---------- point parameters ---------- */
TEST(point_params_legal)
{
    glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 0.5f);
    EXPECT_NO_GL_ERROR();
    GLfloat v = 0;
    glGetFloatv(GL_POINT_FADE_THRESHOLD_SIZE, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(v, 0.5f, 1e-6);
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
    EXPECT_NO_GL_ERROR();
    GLint o = 0;
    glGetIntegerv(GL_POINT_SPRITE_COORD_ORIGIN, &o);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(o, GL_LOWER_LEFT);
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_UPPER_LEFT);
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_POINT_SPRITE_COORD_ORIGIN, &o);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(o, GL_UPPER_LEFT);
}

/* ---------- sync objects ---------- */
TEST(sync_legal)
{
    GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    EXPECT_TRUE(s != NULL);
    EXPECT_NO_GL_ERROR();
    GLint v = 0, len = 0;
    glGetSynciv(s, GL_OBJECT_TYPE, 1, &len, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_SYNC_FENCE);
    glGetSynciv(s, GL_SYNC_STATUS, 1, &len, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(v == GL_UNSIGNALED || v == GL_SIGNALED);
    glGetSynciv(s, GL_SYNC_CONDITION, 1, &len, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_SYNC_GPU_COMMANDS_COMPLETE);
    glGetSynciv(s, GL_SYNC_FLAGS, 1, &len, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, 0);
    GLenum res = glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
    EXPECT_TRUE(res == GL_ALREADY_SIGNALED || res == GL_TIMEOUT_EXPIRED || res == GL_CONDITION_SATISFIED);
    EXPECT_NO_GL_ERROR();
    glWaitSync(s, 0, GL_TIMEOUT_IGNORED);
    EXPECT_NO_GL_ERROR();
    glDeleteSync(s);
    EXPECT_NO_GL_ERROR();
    glDeleteSync(NULL);   /* silently ignored */
    EXPECT_NO_GL_ERROR();
}

/* ---------- indirect compute dispatch ---------- */
TEST(dispatch_compute_indirect_legal)
{
    GLuint prog = glCreateProgram();
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    const char *src = "#version 450 core\nlayout(local_size_x=4) in;\n"
                      "layout(binding=0) buffer Out { float o[]; };\n"
                      "void main() { o[gl_GlobalInvocationID.x] = 1.0; }\n";
    glShaderSource(cs, 1, &src, NULL);
    glCompileShader(cs);
    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glDeleteProgram(prog); glDeleteShader(cs); return; }
    glUseProgram(prog);

    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buf);
    GLuint cmd[3] = { 1, 1, 1 };
    glBufferData(GL_DISPATCH_INDIRECT_BUFFER, sizeof(cmd), cmd, GL_STATIC_DRAW);
    GLuint ssbo = 0;
    glGenBuffers(1, &ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glNamedBufferData(ssbo, 64 * 1024, NULL, GL_STATIC_DRAW);
    EXPECT_NO_GL_ERROR();
    glDispatchComputeIndirect(0);
    EXPECT_NO_GL_ERROR();
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glDeleteBuffers(1, &ssbo);
    glDeleteBuffers(1, &buf);
    glDeleteProgram(prog);
    glDeleteShader(cs);
    EXPECT_NO_GL_ERROR();
}

/* ---------- internalformat queries ---------- */
TEST(internalformat_query_legal)
{
    GLint v = 0;
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_PREFERRED, 1, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_RGBA8);
    GLint samples[16] = { 0 };
    glGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA8, GL_SAMPLES, 8, samples);
    EXPECT_NO_GL_ERROR();
    GLint n = 0;
    glGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA8, GL_NUM_SAMPLE_COUNTS, 1, &n);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(n >= 1);
    GLint64 v64 = 0;
    glGetInternalformati64v(GL_TEXTURE_2D, GL_RGBA32F, GL_INTERNALFORMAT_PREFERRED, 1, &v64);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v64, GL_RGBA32F);
}

/* ---------- 64-bit indexed gets ---------- */
TEST(get_integer64i_v_legal)
{
    GLint64 v = 0xdeadbeef;
    glGetInteger64i_v(GL_UNIFORM_BUFFER_BINDING, 3, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, 0);   /* nothing bound at index 3 */
    GLint i32 = 1234;
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 3, &i32);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(i32, 0);
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, buf);
    EXPECT_NO_GL_ERROR();
    glGetInteger64i_v(GL_UNIFORM_BUFFER_BINDING, 3, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, (GLint64)buf);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 3, &i32);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(i32, (GLint)buf);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, 0);
    glDeleteBuffers(1, &buf);
    EXPECT_NO_GL_ERROR();
}

/* ---------- unbound texture / image units in compute shaders are legal ---------- */
TEST(unbound_compute_textures_legal)
{
    const char *cs = "#version 450 core\n"
                     "layout(local_size_x=4) in;\n"
                     "layout(binding=0) uniform sampler2D tex;\n"       /* never bound */
                     "layout(binding=1, rgba32f) uniform image2D img;\n" /* never bound */
                     "layout(binding=2) buffer Out { float o[]; };\n"
                     "void main() { float v = texelFetch(tex, ivec2(0), 0).r;"
                     "             float w = imageLoad(img, ivec2(0)).r;"
                     "             o[gl_GlobalInvocationID.x] = 7.0 + v * 0.0 + w * 0.0; }\n";
    GLuint prog = glCreateProgram();
    GLuint c = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(c, 1, &cs, NULL);
    glCompileShader(c);
    GLint ok = 0;
    glGetShaderiv(c, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    glAttachShader(prog, c);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (ok)
    {
        glUseProgram(prog);
        GLuint ssbo = 0;
        glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        const GLfloat zero[4] = { 0, 0, 0, 0 };
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), zero, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);

        glDispatchCompute(1, 1, 1);
        EXPECT_NO_GL_ERROR();                 /* was: dispatch silently dropped */

        GLfloat *m = (GLfloat *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        EXPECT_TRUE(m != NULL);
        if (m)
        {
            EXPECT_NEAR(m[0], 7.0f, 1e-6);     /* kernel must have run */
            EXPECT_NEAR(m[3], 7.0f, 1e-6);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
        glDeleteBuffers(1, &ssbo);
        glUseProgram(0);
    }
    glDeleteProgram(prog);
    glDeleteShader(c);
}

/* ---------- unbound shader-declared buffer blocks are legal ---------- */
TEST(unbound_shader_buffers_legal)
{
    /* compute: declared SSBO with nothing bound — the dispatch must still run */
    const char *cs = "#version 450 core\nlayout(local_size_x=4) in;\n"
                     "layout(binding=3) buffer Out { float o[]; };\n"
                     "void main() { o[gl_GlobalInvocationID.x] = 1.0; }\n";
    GLuint prog = glCreateProgram();
    GLuint c = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(c, 1, &cs, NULL);
    glCompileShader(c);
    GLint ok = 0;
    glGetShaderiv(c, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    glAttachShader(prog, c);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (ok)
    {
        glUseProgram(prog);
        glDispatchCompute(1, 1, 1);
        EXPECT_NO_GL_ERROR();                 /* was GL_INVALID_OPERATION */
        GLuint ind = 0;
        glGenBuffers(1, &ind);
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, ind);
        GLuint cmd[3] = { 1, 1, 1 };
        glBufferData(GL_DISPATCH_INDIRECT_BUFFER, sizeof(cmd), cmd, GL_STATIC_DRAW);
        glDispatchComputeIndirect(0);
        EXPECT_NO_GL_ERROR();                 /* was GL_INVALID_OPERATION */
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
        glUseProgram(0);
        glDeleteBuffers(1, &ind);
    }
    glDeleteProgram(prog);
    glDeleteShader(c);

    /* draw: declared UBO + SSBO with nothing bound — the draw must still happen */
    const char *vs = "#version 450 core\n"
                     "layout(location=0) in vec4 pos;\n"
                     "layout(binding=0) uniform U { vec4 c; };\n"
                     "layout(binding=1) buffer B { vec4 b; };\n"
                     "void main() { gl_Position = pos + c * 0.0 + b * 0.0; }\n";
    const char *fs = "#version 450 core\nout vec4 col;\nvoid main() { col = vec4(0, 1, 0, 1); }\n";
    GLuint dp = mgl_make_program(vs, fs);
    EXPECT_TRUE(dp != 0);
    if (dp)
    {
        GLuint vao = 0, vbo = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        const float tri[9] = { -1, -1, 0, 3, -1, 0, -1, 3, 0 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glUseProgram(dp);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();                 /* was GL_INVALID_OPERATION */
        EXPECT_PIXEL(0, 0, 0, 255, 0, 255, 0);
        glDeleteProgram(dp);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        glBindVertexArray(0);
    }
}

int main(void)
{
    RUN(ms_texture_storage_legal);
    RUN(ms_texture_storage_3d);
    RUN(clear_tex_image_legal);
    RUN(texture_view_legal);
    RUN(texture_buffer_legal);
    RUN(get_texture_subimage_legal);
    RUN(invalidate_legal);
    RUN(bind_image_textures_legal);
    RUN(sampler_iparams_legal);
    RUN(point_params_legal);
    RUN(sync_legal);
    RUN(dispatch_compute_indirect_legal);
    RUN(internalformat_query_legal);
    RUN(get_integer64i_v_legal);
    RUN(unbound_compute_textures_legal);
    RUN(unbound_shader_buffers_legal);
    return mgl_test_summary();
}
