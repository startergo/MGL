/* test_compat_imgcast2 — KHR-GL46.es_31_compatibility.shader_image_load_store.
 * advanced-allStages-oneImage repro: one 4x4 r32ui image on unit 3; the VERTEX
 * shader imageAtomicCompSwap's (0, gl_VertexID) 0->100, the FRAGMENT shader
 * imageAtomicAdd's (0, fragCoord.x & 3) by 1 per pixel (256x256 viewport => w/4*h
 * = 16384 adds per word), then a compute shader imageLoads (0, localID).  Every
 * record must be 16384 + 100.
 */
#include "mgl_test.h"

static const char *kVs =
    "#version 460 core\n"
    "layout(location = 0) in vec4 i_position;\n"
    "layout(r32ui, binding = 0) coherent uniform uimage2D g_image;\n"
    "void main() {\n"
    "  gl_Position = i_position;\n"
    "  imageAtomicCompSwap(g_image, ivec2(0, gl_VertexID), 0u, 100u);\n"
    "}\n";

static const char *kFs =
    "#version 460 core\n"
    "layout(r32ui, binding = 0) coherent uniform uimage2D g_image;\n"
    "layout(location = 0) out vec4 frag_color;\n"
    "void main() {\n"
    "  imageAtomicAdd(g_image, ivec2(0, int(gl_FragCoord.x) & 0x03), 0x1u);\n"
    "  frag_color = vec4(1.0);\n"
    "}\n";

static const char *kCs =
    "#version 460 core\n"
    "#define KSIZE 4\n"
    "layout (local_size_x = KSIZE) in;\n"
    "layout(r32ui, binding = 0) uniform uimage2D g_image;\n"
    "layout(std430) buffer out_data {\n"
    "  uvec4 data[KSIZE];\n"
    "};\n"
    "void main() {\n"
    "  uvec4 v = imageLoad(g_image, ivec2(0, gl_LocalInvocationID.x));\n"
    "  data[gl_LocalInvocationIndex] = v;\n"
    "}\n";

TEST(compat_imgcast_allstages)
{
    GLuint tex = 0, buf = 0, vao = 0, vbo = 0, prog = 0, cprog = 0, rfbo = 0, rtex = 0;
    GLuint init16[16] = {0};

    static const float quad[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,  1.0f,  1.0f, 0.0f, 1.0f,
    };

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 4, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RED_INTEGER, GL_UNSIGNED_INT, init16);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);

    for (int st = 0; st < 3; st++)
    {
        GLuint p = glCreateProgram(), s = glCreateShader(st == 0 ? GL_VERTEX_SHADER : st == 1 ? GL_FRAGMENT_SHADER : GL_COMPUTE_SHADER);
        const char *src = st == 0 ? kVs : st == 1 ? kFs : kCs;
        if (st == 1)
        {
            GLuint v = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(v, 1, &(const char *){ "#version 460 core\nlayout(location=0) in vec4 p;\nvoid main(){gl_Position=p;}\n" }, NULL);
            glCompileShader(v);
            glAttachShader(p, v);
            glDeleteShader(v);
        }
        glShaderSource(s, 1, &src, NULL);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        EXPECT_TRUE(ok);
        glAttachShader(p, s);
        glDeleteShader(s);
        glLinkProgram(p);
        if (st == 0) prog = p; else if (st == 1) prog = p; else cprog = p;
        (void)ok;
    }
    /* rebuild the raster program as VS+FS (the loop above links each stage alone
     * only for compile checks); do it properly: */
    glDeleteProgram(prog);
    {
        GLuint p = glCreateProgram();
        GLuint vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vs, 1, &kVs, NULL); glCompileShader(vs);
        glShaderSource(fs, 1, &kFs, NULL); glCompileShader(fs);
        glAttachShader(p, vs); glAttachShader(p, fs);
        glLinkProgram(p);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        glDeleteShader(vs); glDeleteShader(fs);
        prog = p;
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, 0);

    /* a 256x256 target so the add count matches the CTS surface math */
    glGenTextures(1, &rtex);
    glBindTexture(GL_TEXTURE_2D, rtex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 256, 256);
    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtex, 0);

    glGenBuffers(1, &buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * 16, init16, GL_STATIC_DRAW);

    glUseProgram(prog);
    glViewport(0, 0, 256, 256);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    EXPECT_NO_GL_ERROR();
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    if (getenv("IMGCAST2_DEBUG"))
    {
        GLubyte px[4] = {0};
        glReadPixels(128, 128, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        fprintf(stderr, "DBG color(128,128) = %u %u %u %u\n", px[0], px[1], px[2], px[3]);
    }
    glUseProgram(cprog);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    GLuint *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 4 * 16, GL_MAP_READ_BIT);
    EXPECT_TRUE(map != NULL);
    int bad = 0;
    if (map)
    {
        GLuint want = (256 / 4) * 256 + 100;   /* 16384 adds + the VS swap */
        for (int i = 0; i < 4; i++)
            if (map[i * 4] != want)
            {
                MGL_FAIL("record[%d] = %u, expected %u", i, map[i * 4], want);
                bad++;
            }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_EQ_INT(bad, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &rfbo);
    glDeleteTextures(1, &rtex);
    glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    glDeleteProgram(cprog);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &buf);
    glDeleteVertexArrays(1, &vao);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(compat_imgcast_allstages);
    return mgl_test_summary();
}
