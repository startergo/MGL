/* test_compat_imgcast — KHR-GL46.es_31_compatibility.shader_image_load_store.
 * advanced-cast-fs repro: two RGBA8 textures bound as image units with the
 * CAST formats r32i / r32ui; a full-viewport FS quad does imageAtomicAdd
 * (+2,-1 into the r32i image, +1u,+2u into the r32ui one), glMemoryBarrier,
 * then a compute shader imageLoads both images and writes an ivec4 record;
 * the record must read back (0,1,0,3) semantics: r32i .x = 1, r32ui .x = 3,
 * other components 0.
 */
#include "mgl_test.h"

static const char *kFs =
    "#version 460 core\n"
    "layout(r32i, binding = 0) coherent uniform iimage2D g_image0;\n"
    "layout(r32ui, binding = 1) coherent uniform uimage2D g_image1;\n"
    "void main() {\n"
    "  ivec2 coord = ivec2(gl_FragCoord.xy);\n"
    "  imageAtomicAdd(g_image0, coord, 2);\n"
    "  imageAtomicAdd(g_image0, coord, -1);\n"
    "  imageAtomicAdd(g_image1, coord, 1u);\n"
    "  imageAtomicAdd(g_image1, coord, 2u);\n"
    "}\n";

static const char *kCs =
    "#version 460 core\n"
    "#define KSIZE 11\n"
    "layout (local_size_x = KSIZE, local_size_y = KSIZE) in;\n"
    "layout(r32i, binding = 0) uniform iimage2D gi_image;\n"
    "layout(r32ui, binding = 1) uniform uimage2D gu_image;\n"
    "layout(std430) buffer out_data {\n"
    "  ivec4 data[KSIZE*KSIZE];\n"
    "};\n"
    "void main() {\n"
    "  ivec2 coord = ivec2(gl_LocalInvocationID);\n"
    "  data[gl_LocalInvocationIndex].yx = imageLoad(gi_image, coord).xy;\n"
    "  data[gl_LocalInvocationIndex].wz = ivec2(imageLoad(gu_image, coord).xz);\n"
    "}\n";

static const char *kVs =
    "#version 460 core\n"
    "layout(location = 0) in vec4 i_position;\n"
    "void main() { gl_Position = i_position; }\n";

#define KSIZE 11

static GLuint mk_prog(int stage, const char *src, const char *vs)
{
    GLuint p = glCreateProgram(), s = glCreateShader(stage);
    if (vs)
    {
        GLuint v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vs, NULL);
        glCompileShader(v);
        GLint ok2 = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &ok2);
        if (!ok2) { MGL_FAIL("vs compile"); return 0; }
        glAttachShader(p, v);
        glDeleteShader(v);
    }
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        MGL_FAIL("stage %d compile: %s", stage, log);
        return 0;
    }
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("stage %d link: %s", stage, log);
    }
    glDeleteShader(s);
    return p;
}

TEST(compat_imgcast_fs)
{
    GLuint tex[2] = {0}, buf = 0, vao = 0, vbo = 0, ebo = 0;
    GLuint prog = 0, cprog = 0;
    GLubyte zeros[KSIZE * KSIZE * 4] = {0};

    static const float quad[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,  1.0f,  1.0f, 0.0f, 1.0f,
    };
    static const GLushort idx[] = { 0, 1, 2, 3 };

    for (int t = 0; t < 2; t++)
    {
        glGenTextures(1, &tex[t]);
        glActiveTexture(GL_TEXTURE11 + t);
        glBindTexture(GL_TEXTURE_2D, tex[t]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, KSIZE, KSIZE);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, KSIZE, KSIZE, GL_RGBA, GL_UNSIGNED_BYTE, zeros);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glActiveTexture(GL_TEXTURE0);

    glBindImageTexture(0, tex[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(1, tex[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);

    glGenBuffers(1, &buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, KSIZE * KSIZE * 16, zeros, GL_STATIC_DRAW);

    prog = mk_prog(GL_FRAGMENT_SHADER, kFs, kVs);
    cprog = mk_prog(GL_COMPUTE_SHADER, kCs, NULL);

    /* an 11x11 render target: the FS must rasterize every texel the CS reads
     * (the headless default framebuffer is smaller than 11x11) */
    GLuint rtex = 0, rfbo = 0;
    glGenTextures(1, &rtex);
    glBindTexture(GL_TEXTURE_2D, rtex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, KSIZE, KSIZE);
    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtex, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, 0);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    glUseProgram(prog);
    glViewport(0, 0, KSIZE, KSIZE);
    glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    if (getenv("IMGCAST_DEBUG"))
    {
        /* what did the FS atomics leave in the texture? (RGBA8 storage: the
         * packed r32ui word 3 = bytes 3,0,0,0) */
        GLubyte rb[KSIZE * KSIZE * 4];
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_2D, tex[1]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rb);
        fprintf(stderr, "FS-ATOMIC tex[1] byte0 map (3=hit):\n");
        for (int y = 0; y < KSIZE; y++)
        {
            for (int x = 0; x < KSIZE; x++)
                fprintf(stderr, "%d", rb[(y * KSIZE + x) * 4] ? 1 : 0);
            fprintf(stderr, "\n");
        }
        glActiveTexture(GL_TEXTURE0);
    }

    glUseProgram(cprog);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    GLint *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, KSIZE * KSIZE * 16, GL_MAP_READ_BIT);
    EXPECT_TRUE(map != NULL);
    int bad = 0;
    if (map)
    {
        for (int i = 0; i < KSIZE * KSIZE; i++)
        {
            int x = map[i * 4 + 0], y = map[i * 4 + 1], z = map[i * 4 + 2], w = map[i * 4 + 3];
            /* data.yx = i.xy (y=1), data.wz = u.xz (w=3): record = (0,1,0,3) */
            if (x != 0 || y != 1 || z != 0 || w != 3)
            {
                if (bad < 200)
                    fprintf(stderr, "REC %d %d -> (%d,%d,%d,%d)\n", i / 11, i % 11, x, y, z, w);
                bad++;
            }
        }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_EQ_INT(bad, 0);

    glViewport(0, 0, 256, 256);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &rfbo);
    glDeleteTextures(1, &rtex);
    glDeleteProgram(prog);
    glDeleteProgram(cprog);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &buf);
    glDeleteTextures(2, tex);
    glDeleteVertexArrays(1, &vao);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(compat_imgcast_fs);
    return mgl_test_summary();
}
