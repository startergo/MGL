/* test_mapbuf.c — glMapBufferRange flag combinations (CTS constant_expressions bounce buffer) */
#include "mgl_test.h"
#include <string.h>

static GLuint make_cs(void)
{
    static const char *src =
        "#version 460 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430) buffer Out { float o; };\n"
        "void main() { o = 42.0; }\n";
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

/* the CTS BufferIoExecutor pattern: glBufferData(NULL) -> map WRITE -> unmap -> use ->
 * glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT) -> map READ */
TEST(map_read_after_write_static_draw)
{
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256, NULL, GL_STATIC_DRAW);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    void *p = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 256, GL_MAP_WRITE_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_TRUE(p != NULL);
    if (p) { ((float *)p)[0] = 1.0f; EXPECT_EQ_INT(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER), GL_TRUE); }
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* compute writes it, then the CTS syncs and maps READ */
    GLuint prog = glCreateProgram();
    GLuint cs = make_cs();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glUseProgram(prog);
    glDispatchCompute(1, 1, 1);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    float *q = (float *)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 256, GL_MAP_READ_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_TRUE(q != NULL);
    if (q) EXPECT_EQ_INT((int)q[0], 42);
    if (q) glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteProgram(prog); glDeleteShader(cs); glDeleteBuffers(1, &buf);
}

/* every legal flag combination on a glBufferData (mutable) buffer must map */
TEST(map_flag_combinations_mutable)
{
    struct { GLbitfield flags; const char *name; } combos[] = {
        { GL_MAP_READ_BIT, "READ" },
        { GL_MAP_WRITE_BIT, "WRITE" },
        { GL_MAP_READ_BIT | GL_MAP_WRITE_BIT, "R|W" },
        { GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT, "W|INVAL_BUF" },
        { GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT, "W|FLUSH" },
        { GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT, "W|UNSYNC" },
    };
    /* GL 4.6 §6.3: READ together with INVALIDATE_RANGE/INVALIDATE_BUFFER/UNSYNCHRONIZED and
     * PERSISTENT/COHERENT without matching buffer-storage flags are INVALID_OPERATION —
     * verify these too, since the CTS probes them */
    struct { GLbitfield flags; GLenum err; const char *name; } bad[] = {
        { GL_MAP_READ_BIT | GL_MAP_INVALIDATE_RANGE_BIT, GL_INVALID_OPERATION, "R|INVAL_RANGE" },
        { GL_MAP_READ_BIT | GL_MAP_INVALIDATE_BUFFER_BIT, GL_INVALID_OPERATION, "R|INVAL_BUF" },
        { GL_MAP_READ_BIT | GL_MAP_UNSYNCHRONIZED_BIT, GL_INVALID_OPERATION, "R|UNSYNC" },
        { GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT, GL_NO_ERROR, "W|FLUSH|INVAL_BUF" },
        { GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT, GL_INVALID_OPERATION, "R|W|PERSIST" },
        { GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT, GL_INVALID_OPERATION, "R|W|COHERENT" },
        { 0, GL_INVALID_OPERATION, "NONE" },
    };
    for (unsigned i = 0; i < sizeof combos / sizeof combos[0]; i++)
    {
        GLuint buf = 0;
        glGenBuffers(1, &buf);
        glBindBuffer(GL_ARRAY_BUFFER, buf);
        glBufferData(GL_ARRAY_BUFFER, 64, NULL, GL_STREAM_DRAW);
        void *p = glMapBufferRange(GL_ARRAY_BUFFER, 0, 64, combos[i].flags);
        GLenum err = glGetError();
        if (err != GL_NO_ERROR || !p)
            fprintf(stderr, "  [%s] flags=0x%x -> err 0x%x ptr %p\n", combos[i].name, combos[i].flags, err, p);
        EXPECT_EQ_INT(err, GL_NO_ERROR);
        EXPECT_TRUE(p != NULL);
        if (p)
        {
            if (combos[i].flags & GL_MAP_FLUSH_EXPLICIT_BIT)
                glFlushMappedBufferRange(GL_ARRAY_BUFFER, 0, 32);
                EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        glDeleteBuffers(1, &buf);
    }
    for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; i++)
    {
        GLuint buf = 0;
        glGenBuffers(1, &buf);
        glBindBuffer(GL_ARRAY_BUFFER, buf);
        glBufferData(GL_ARRAY_BUFFER, 64, NULL, GL_STREAM_DRAW);
        void *p = glMapBufferRange(GL_ARRAY_BUFFER, 0, 64, bad[i].flags);
        GLenum err = glGetError();
        if (err != bad[i].err)
            fprintf(stderr, "  [%s] flags=0x%x -> err 0x%x want 0x%x\n", bad[i].name, bad[i].flags, err, bad[i].err);
        EXPECT_EQ_INT(err, bad[i].err);
        if (p) glUnmapBuffer(GL_ARRAY_BUFFER);
        glGetError();
        glDeleteBuffers(1, &buf);
    }
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
}

/* immutable storage: only the flags granted at creation (GL 4.6 §6.3) */
TEST(map_flag_combinations_immutable)
{
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferStorage(GL_ARRAY_BUFFER, 64, NULL, GL_MAP_READ_BIT);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_TRUE(glMapBufferRange(GL_ARRAY_BUFFER, 0, 64, GL_MAP_READ_BIT) != NULL);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    /* write map not granted -> GL_INVALID_OPERATION */
    EXPECT_TRUE(glMapBufferRange(GL_ARRAY_BUFFER, 0, 64, GL_MAP_WRITE_BIT) == NULL);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    /* persistent without the flag -> GL_INVALID_OPERATION */
    EXPECT_TRUE(glMapBufferRange(GL_ARRAY_BUFFER, 0, 64, GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT) == NULL);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    glDeleteBuffers(1, &buf);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
}

int main(void)
{
    RUN(map_read_after_write_static_draw);
    RUN(map_flag_combinations_mutable);
    RUN(map_flag_combinations_immutable);
    return mgl_test_summary();
}
