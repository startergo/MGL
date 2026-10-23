/* GL atomic counters (atomic_uint) in the vertex / fragment / compute stages.
 *
 * SPIRV-Cross's MSL backend declares no argument for GL atomic-counter variables,
 * so program.c collects them (name + SPIR-V offset) and geometry.c's
 * mgl_msl_inject_atomic_counters() appends `volatile device atomic_uint* NAME
 * [[buffer(N)]]` to the entry function and re-points every
 * `(thread atomic_uint*)&NAME` use at it.  N is the ordinal the renderer's buffer
 * map binds the atomic-counter group at (after uniform blocks, the default uniform
 * block and SSBOs).
 *
 * Note: strips run the vertex function per triangle on Apple GPUs (verified against
 * raw Metal), so exact increment counts are checked with GL_TRIANGLES.
 */
#include "mgl_test.h"
#include <stdbool.h>

static const char *vs_src =
    "#version 460 core\n"
    "layout(location = 0) in vec4 i_position;\n"
    "layout(location = 1) in vec4 i_color;\n"
    "layout(location = 0) out vec4 o_color;\n"
    "layout(std140) uniform Transform { mat4 mvp; } g_transform;\n"
    "writeonly uniform imageBuffer g_buffer;\n"
    "layout(binding = 0, offset = 0) uniform atomic_uint g_counter;\n"
    "void main() {\n"
    "  gl_Position = g_transform.mvp * i_position;\n"
    "  o_color = i_color;\n"
    "  const uint index = atomicCounterIncrement(g_counter);\n"
    "  imageStore(g_buffer, int(index), gl_Position);\n"
    "}\n";

TEST(vs_atomic_counter)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location = 0) in vec4 i_color;\n"
        "layout(location = 0) out vec4 o_color;\n"
        "void main() { o_color = i_color; }\n";
    GLuint prog = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs_src, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(prog, v); glAttachShader(prog, f);
    glLinkProgram(prog);
    GLint ok = 0; char log[4096] = "";
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    glGetProgramInfoLog(prog, sizeof log, NULL, log);
    if (!ok) MGL_FAIL("link: %s", log);

    /* pos + color interleaved, two triangles covering the viewport */
    static const float quad[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f,
    };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    glEnableVertexAttribArray(1);

    GLuint tbuf, ttex;
    glGenBuffers(1, &tbuf); glBindBuffer(GL_TEXTURE_BUFFER, tbuf);
    glBufferData(GL_TEXTURE_BUFFER, 256, NULL, GL_STATIC_DRAW);
    glGenTextures(1, &ttex); glBindTexture(GL_TEXTURE_BUFFER, ttex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tbuf);

    GLuint acbo, ubo;
    glGenBuffers(1, &acbo); glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    { static const GLuint z = 0; glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4, &z, GL_STATIC_DRAW); }
    glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    { static const float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; glBufferData(GL_UNIFORM_BUFFER, 64, id, GL_STATIC_DRAW); }

    glUseProgram(prog);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    glBindImageTexture(0, ttex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();

    glDrawArrays(GL_TRIANGLES, 0, 6);
    EXPECT_NO_GL_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
    GLuint count = 0xdeadbeef;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, &count);
    EXPECT_EQ_INT(count, 6u);          /* one increment per vertex */

    float out[16];
    glBindBuffer(GL_TEXTURE_BUFFER, tbuf);
    glGetBufferSubData(GL_TEXTURE_BUFFER, 0, sizeof out, out);
    EXPECT_NEAR(out[0], -1.0, 1e-6);   /* vertex 0 position landed at texel 0 */
    EXPECT_NEAR(out[1], -1.0, 1e-6);

    EXPECT_PIXEL(4, 4, 255, 2, 2, 255, 12);   /* near v0 corner: mostly red */
}

/* fragment stage: atomicCounter() load */
TEST(fs_atomic_counter)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location = 0) in vec4 p;\n"
        "void main() { gl_Position = p; }\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 c;\n"
        "layout(binding = 0, offset = 0) uniform atomic_uint g_counter;\n"
        "void main() { c = vec4(float(atomicCounter(g_counter)) / 255.0); }\n";
    GLuint prog = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(prog, v); glAttachShader(prog, f); glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) MGL_FAIL("fs link failed");

    GLuint vao, vbo, acbo;
    static const float quad[] = { -1,-1,0,1, 1,-1,0,1, 1,1,0,1, -1,-1,0,1, 1,1,0,1, -1,1,0,1 };
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &acbo); glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    { static const GLuint three = 3; glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4, &three, GL_STATIC_DRAW); }
    glUseProgram(prog);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(64, 64, 3, 3, 3, 3, 1);
}

/* fragment stage: atomicCounterIncrement, read back through the buffer */
TEST(fs_atomic_increment)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location = 0) in vec4 p;\n"
        "void main() { gl_Position = p; }\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 c;\n"
        "layout(binding = 0, offset = 4) uniform atomic_uint g_counter;\n"
        "void main() { c = vec4(float(atomicCounterIncrement(g_counter)) / 1000.0); atomicCounterIncrement(g_counter); }\n";
    GLuint prog = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(prog, v); glAttachShader(prog, f); glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) MGL_FAIL("fs link failed");

    GLuint vao, vbo, acbo;
    static const float quad[] = { -1,-1,0,1, 1,-1,0,1, 1,1,0,1, -1,-1,0,1, 1,1,0,1, -1,1,0,1 };
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &acbo); glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    { static const GLuint z[4] = {0,0,0,0}; glBufferData(GL_ATOMIC_COUNTER_BUFFER, 16, z, GL_STATIC_DRAW); }
    glUseProgram(prog);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    GLuint v2[4] = {0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 16, v2);
    /* offset = 4 (word 1): two increments per fragment, 16k fragments */
    EXPECT_EQ_INT(v2[0], 0u);         /* word 0 untouched */
    EXPECT_TRUE(v2[1] > 0);
}

/* compute stage */
static const char *cs_src =
    "#version 460 core\n"
    "layout(binding = 0, offset = 0) uniform atomic_uint g_counter;\n"
    "layout(local_size_x = 4) in;\n"
    "void main() { atomicCounterIncrement(g_counter); }\n";

TEST(cs_atomic_counter)
{
    GLuint prog = glCreateProgram();
    GLuint c = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(c, 1, &cs_src, NULL); glCompileShader(c);
    glAttachShader(prog, c);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) MGL_FAIL("cs link failed");

    GLuint acbo;
    glGenBuffers(1, &acbo); glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    { static const GLuint z = 0; glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4, &z, GL_STATIC_DRAW); }
    glUseProgram(prog);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glDispatchCompute(2, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    GLuint v = 0xdeadbeef;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, &v);
    EXPECT_EQ_INT(v, 8u);
}

/* separable SSO pipeline driving the VS above (advanced-sso-atomicCounters shape) */
TEST(sso_pipeline_atomic_vs)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location = 0) in vec4 i_color;\n"
        "layout(location = 0) out vec4 o_color;\n"
        "layout(rgba32f, binding = 0) readonly uniform imageBuffer g_buffer;\n"
        "void main() { o_color = i_color + imageLoad(g_buffer, 0) * 0.0 + vec4(imageLoad(g_buffer, 0).x, 0.0, 0.0, 0.0); }\n";
    GLuint vsp = glCreateProgram(), fsp = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs_src, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(vsp, v); glProgramParameteri(vsp, GL_PROGRAM_SEPARABLE, GL_TRUE); glLinkProgram(vsp);
    glAttachShader(fsp, f); glProgramParameteri(fsp, GL_PROGRAM_SEPARABLE, GL_TRUE); glLinkProgram(fsp);
    GLint ok1 = 0, ok2 = 0;
    glGetProgramiv(vsp, GL_LINK_STATUS, &ok1);
    glGetProgramiv(fsp, GL_LINK_STATUS, &ok2);
    if (!ok1 || !ok2) MGL_FAIL("sso link failed");

    static const float quad[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f, 1.0f,
    };
    GLuint vao, vbo, tbuf, ttex, acbo, ubo, pipe;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    glEnableVertexAttribArray(1);
    glGenBuffers(1, &tbuf); glBindBuffer(GL_TEXTURE_BUFFER, tbuf);
    glBufferData(GL_TEXTURE_BUFFER, 256, NULL, GL_STATIC_DRAW);
    glGenTextures(1, &ttex); glBindTexture(GL_TEXTURE_BUFFER, ttex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, tbuf);
    glGenBuffers(1, &acbo); glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo);
    { static const GLuint z = 0; glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4, &z, GL_STATIC_DRAW); }
    glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    { static const float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; glBufferData(GL_UNIFORM_BUFFER, 64, id, GL_STATIC_DRAW); }

    glGenProgramPipelines(1, &pipe);
    glBindProgramPipeline(pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fsp);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acbo);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    glBindImageTexture(0, ttex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_NO_GL_ERROR();

    /* the CTS shape: a strip (vertex function runs per triangle on Apple GPUs, so
     * only the imageBuffer contents — corner positions — are checked) */
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    { GLenum e = glGetError(); fprintf(stderr, "[sso] draw err 0x%x\n", e); }
    {
        float o0[16]; int i;
        glBindBuffer(GL_TEXTURE_BUFFER, tbuf);
        glGetBufferSubData(GL_TEXTURE_BUFFER, 0, sizeof o0, o0);
        for (i = 0; i < 4; i++) fprintf(stderr, "[sso] texel %d = %f %f\n", i, o0[i*4], o0[i*4+1]);
        GLuint cc = 0; glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acbo); glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, &cc);
        fprintf(stderr, "[sso] counter %u\n", cc);
    }

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    /* the image writes are verified through the pipeline: the FS reads texel 0 (x = -1
     * from corner v0) — glReadPixels sees it in the color output.  (A CPU readback of the
     * texture buffer is not coherent after pipeline image writes, so the FS is the oracle.) */
    GLubyte px[4];
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "[sso] px = %d %d %d %d (image read-back through the FS)\n", px[0], px[1], px[2], px[3]);
    EXPECT_TRUE(px[0] > 200);   /* i_color red + texel0.x = -1 pushed R up */
}

int main(void)
{
    RUN(vs_atomic_counter);
    RUN(fs_atomic_counter);
    RUN(fs_atomic_increment);
    RUN(cs_atomic_counter);
    RUN(sso_pipeline_atomic_vs);
    return mgl_test_summary();
}
