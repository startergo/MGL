/* test_bufmap2.c — ARB_buffer_storage mapping/sub-data interaction rules
 * (KHR-GL46.buffer_storage.map_persistent_buffer_sub_data /
 *  map_persistent_read_pixels regression) */
#include "mgl_test.h"
#include <string.h>

/* A persistent mapping never blocks BufferSubData/NamedBufferSubData; a
 * non-persistent mapping blocks exactly the ranges it covers. */
TEST(subdata_vs_mapping)
{
    static const GLintptr map_off = 16;
    static const GLsizeiptr map_len = 16;
    /* {offset, size, crosses the mapped region?} */
    static const struct { GLintptr off; GLsizeiptr len; int cross; } cases[] = {
        { 0, 16, 0 }, { 32, 16, 0 }, { 8, 16, 1 }, { 24, 16, 1 }, { 12, 8, 1 },
    };
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));

    GLubyte data[64];
    for (int i = 0; i < 64; i++) data[i] = (GLubyte)i;

    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferStorage(GL_ARRAY_BUFFER, 64, NULL,
                    GL_DYNAMIC_STORAGE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 64, data);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* persistent mapping: every range is legal (GL 4.6 table: only a
     * non-persistent mapping produces the "buffer is mapped" error) */
    GLubyte *m = (GLubyte *)glMapBufferRange(GL_ARRAY_BUFFER, map_off, map_len,
                                             GL_MAP_PERSISTENT_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_TRUE(m != NULL);
    if (m)
        for (int i = 0; i < map_len; i++)
            EXPECT_EQ_INT(m[i], data[map_off + i]);
    for (int i = 0; i < n; i++)
    {
        glBufferSubData(GL_ARRAY_BUFFER, cases[i].off, cases[i].len, data);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        glNamedBufferSubData(buf, cases[i].off, cases[i].len, data);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* non-persistent mapping: disjoint ranges still legal, overlapping ones
     * GL_INVALID_OPERATION (BufferSubData and NamedBufferSubData) */
    m = (GLubyte *)glMapBufferRange(GL_ARRAY_BUFFER, map_off, map_len,
                                    GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_TRUE(m != NULL);
    for (int i = 0; i < n; i++)
    {
        GLenum expect = cases[i].cross ? GL_INVALID_OPERATION : GL_NO_ERROR;
        glBufferSubData(GL_ARRAY_BUFFER, cases[i].off, cases[i].len, data);
        EXPECT_EQ_INT(glGetError(), expect);
        glNamedBufferSubData(buf, cases[i].off, cases[i].len, data);
        EXPECT_EQ_INT(glGetError(), expect);
        /* ClearBufferSubData follows the same range rule */
        glClearBufferSubData(GL_ARRAY_BUFFER, GL_R8UI, cases[i].off, cases[i].len,
                             GL_RED_INTEGER, GL_UNSIGNED_BYTE, data);
        EXPECT_EQ_INT(glGetError(), expect);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    /* unmapped again: everything legal */
    glBufferSubData(GL_ARRAY_BUFFER, 8, 16, data);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glClearBufferSubData(GL_ARRAY_BUFFER, GL_R8UI, 8, 16, GL_RED_INTEGER, GL_UNSIGNED_BYTE, data);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteBuffers(1, &buf);
}

/* ReadPixels into a persistently mapped PIXEL_PACK_BUFFER is legal and the
 * mapped client address observes the new contents after the barrier. */
TEST(readpixels_persistent_pack_pbo)
{
    const GLuint w = 8, h = 8;
    const size_t data_size = w * h;
    GLubyte init[64], upd[64];
    for (size_t i = 0; i < data_size; i++) { init[i] = (GLubyte)i; upd[i] = (GLubyte)(data_size - i); }

    GLuint tex = 0, fbo = 0, buf = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8UI, w, h);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_BYTE, init);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glGenBuffers(1, &buf);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, buf);
    glBufferStorage(GL_PIXEL_PACK_BUFFER, data_size, NULL,
                    GL_MAP_PERSISTENT_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* unmapped read first: must work */
    glReadPixels(0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_BYTE, (void *)0);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_BYTE, upd);

    GLubyte *m = (GLubyte *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, data_size,
                                             GL_MAP_PERSISTENT_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_TRUE(m != NULL);
    if (m)
    {
        glReadPixels(0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_BYTE, (void *)0);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        glFinish();
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        int bad = 0;
        for (size_t i = 0; i < data_size; i++) if (m[i] != upd[i]) bad++;
        EXPECT_EQ_INT(bad, 0);
    }
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &buf);
}

int main(void)
{
    RUN(subdata_vs_mapping);
    RUN(readpixels_persistent_pack_pbo);
    return mgl_test_summary();
}
