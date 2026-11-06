/* test_compat_sync_multi — dEQP ES31 synchronization.inter_call.with_memory_barrier.
 * image_atomic_multiple_write_read repro: two r32i images, two writers that
 * imageAtomicAdd 1 (one image each), a barrier, then a verifier that imageLoads
 * both images and records 1 where both equal 1.
 */
#include "mgl_test.h"

static const char *kWriter =
    "#version 450\n"
    "layout (local_size_x = 1) in;\n"
    "layout(r32i, binding = 0) coherent uniform iimage2D image;\n"
    "void main (void) {\n"
    "    imageAtomicAdd(image, ivec2(gl_GlobalInvocationID.xy), 1);\n"
    "}\n";

static const char *kVerifier =
    "#version 450\n"
    "layout (local_size_x = 1) in;\n"
    "layout(r32i, binding = 0) coherent uniform iimage2D image1;\n"
    "layout(r32i, binding = 1) coherent uniform iimage2D image2;\n"
    "layout(std430) buffer out_data { int data[]; };\n"
    "void main (void) {\n"
    "    ivec2 c = ivec2(gl_GlobalInvocationID.xy);\n"
    "    int a = imageLoad(image1, c).x;\n"
    "    int b = imageLoad(image2, c).x;\n"
    "    data[c.y * 64 + c.x] = (a == 1 && b == 1) ? 1 : 0;\n"
    "}\n";

static GLuint mkcs(const char *src)
{
    GLuint p = glCreateProgram(), s = glCreateShader(GL_COMPUTE_SHADER);
    GLint ok = 0;
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    return p;
}

TEST(compat_sync_multi)
{
    GLuint img[2] = {0}, res = 0;
    GLuint w1 = mkcs(kWriter), w2 = mkcs(kWriter), ver = mkcs(kVerifier);
    int *zeros = calloc(64 * 128, sizeof(int));

    for (int i = 0; i < 2; i++)
    {
        glGenTextures(1, &img[i]);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, img[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, 64, 128);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 128, GL_RED_INTEGER, GL_INT, zeros);
    }
    glActiveTexture(GL_TEXTURE0);

    glGenBuffers(1, &res);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, res);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 64 * 64 * 4, zeros, GL_STREAM_READ);

    /* writer 1 -> image 1 (unit 0) */
    glBindImageTexture(0, img[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glUseProgram(w1);
    glDispatchCompute(64, 64, 1);

    /* writer 2 -> image 2 (unit 0: rebound) */
    glBindImageTexture(0, img[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glUseProgram(w2);
    glDispatchCompute(64, 64, 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    /* verifier: both images at once */
    glBindImageTexture(0, img[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32I);
    glBindImageTexture(1, img[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32I);
    glUseProgram(ver);
    glDispatchCompute(64, 64, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    int *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 64 * 64 * 4, GL_MAP_READ_BIT);
    EXPECT_TRUE(map != NULL);
    int bad = 0;
    if (map)
    {
        for (int i = 0; i < 64 * 64; i++)
            if (map[i] != 1)
            {
                if (bad < 4) MGL_FAIL("data[%d] = %d, expected 1", i, map[i]);
                bad++;
            }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_EQ_INT(bad, 0);

    glDeleteProgram(w1);
    glDeleteProgram(w2);
    glDeleteProgram(ver);
    glDeleteTextures(2, img);
    glDeleteBuffers(1, &res);
    free(zeros);
    while (glGetError() != GL_NO_ERROR) {}
}

int main(void)
{
    RUN(compat_sync_multi);
    return mgl_test_summary();
}
