/* test_buffers.c — buffer objects, mapping, copies/clears, compute + barriers, indirect
 * draws (incl. GL 4.6 *IndirectCount), emulated primitive types, fences, draw errors. */
#include "mgl_test.h"
#include <stdint.h>

typedef struct { GLuint count, instanceCount, first, baseInstance; } DrawArraysIndirectCommand;
typedef struct { GLuint count, instanceCount, firstIndex; GLint baseVertex; GLuint baseInstance; } DrawElementsIndirectCommand;

static const char *k_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); gl_PointSize = 1.0; }\n";

/* FS: counts fragments (binding 2) and fragments at pixel (0,0) (second word) */
static const char *k_fs_count =
    "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "layout(std430, binding=2) buffer Count { uint frags; uint at_origin; };\n"
    "void main(){ atomicAdd(frags, 1u); if (ivec2(gl_FragCoord.xy) == ivec2(0)) atomicAdd(at_origin, 1u);\n"
    "             color = vec4(0.0, 1.0, 0.0, 1.0); }\n";

static GLuint g_vao, g_vbo;

static void geometry(const float *verts, int n)
{
    if (!g_vao) { glGenVertexArrays(1, &g_vao); glGenBuffers(1, &g_vbo); }
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, n * 2 * sizeof(float), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static const float k_fullscreen[] = { -1,-1,  3,-1,  -1,3 };

static GLuint make_ssbo(GLuint binding, GLsizeiptr size)
{
    GLuint buf;
    void *zero = calloc(1, size);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, zero, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
    free(zero);
    return buf;
}

static void read_buf(GLuint buf, GLintptr off, GLsizeiptr size, void *out)
{
    glGetNamedBufferSubData(buf, off, size, out);
}

static GLuint compute_program(const char *src)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log); MGL_FAIL("cs compile: %s", log); return 0; }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("cs link failed"); return 0; }
    return p;
}

static const char *k_cs_write =
    "#version 450 core\n"
    "layout(local_size_x=4) in;\n"
    "layout(std430, binding=0) buffer B { uint v[]; };\n"
    "void main(){ v[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 2u + 1u; }\n";

/* ------------------------------------------------------------------------------------ */

TEST(buffer_data_and_params)
{
    GLuint b;
    glGenBuffers(1, &b);
    EXPECT_TRUE(!glIsBuffer(b));               /* object created at first bind */
    glBindBuffer(GL_ARRAY_BUFFER, b);
    EXPECT_TRUE(glIsBuffer(b));

    GLint v = -1;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_USAGE, &v);  EXPECT_EQ_INT(v, GL_STATIC_DRAW);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_ACCESS, &v); EXPECT_EQ_INT(v, GL_READ_WRITE);

    uint8_t data[100];
    for (int i = 0; i < 100; i++) data[i] = (uint8_t)i;
    glBufferData(GL_ARRAY_BUFFER, 100, data, GL_DYNAMIC_COPY);
    EXPECT_NO_GL_ERROR();

    GLint64 v64 = 0;
    glGetBufferParameteri64v(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &v64); EXPECT_EQ_INT(v64, 100);
    glGetNamedBufferParameteriv(b, GL_BUFFER_USAGE, &v);             EXPECT_EQ_INT(v, GL_DYNAMIC_COPY);
    glGetNamedBufferParameteri64v(b, GL_BUFFER_IMMUTABLE_STORAGE, &v64); EXPECT_EQ_INT(v64, GL_FALSE);
    glGetNamedBufferParameteriv(b, GL_BUFFER_STORAGE_FLAGS, &v);
    EXPECT_EQ_INT(v, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
    EXPECT_NO_GL_ERROR();

    uint8_t patch[4] = { 200, 201, 202, 203 };
    glBufferSubData(GL_ARRAY_BUFFER, 10, 4, patch);
    uint8_t out[100];
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, 100, out);
    EXPECT_EQ_INT(out[9], 9); EXPECT_EQ_INT(out[10], 200); EXPECT_EQ_INT(out[13], 203); EXPECT_EQ_INT(out[14], 14);
    EXPECT_NO_GL_ERROR();

    /* errors */
    glBufferData(GL_ARRAY_BUFFER, -1, NULL, GL_STATIC_DRAW);      EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBufferData(GL_ARRAY_BUFFER, 4, NULL, GL_RGBA);              EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBufferSubData(GL_ARRAY_BUFFER, 98, 4, patch);               EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBufferSubData(GL_ARRAY_BUFFER, -1, 1, patch);               EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_RGBA, &v);         EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBufferData(GL_ARRAY_BUFFER, 4, NULL, GL_STATIC_DRAW);       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &v);  EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindBuffer(0x1234, b);                                      EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBindBuffer(GL_ARRAY_BUFFER, 0x7fffff);                      EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glNamedBufferData(0x7fffff, 4, NULL, GL_STATIC_DRAW);         EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glDeleteBuffers(1, &b);
    EXPECT_TRUE(!glIsBuffer(b));
    EXPECT_NO_GL_ERROR();
}

TEST(buffer_storage_flags)
{
    GLuint b;
    glCreateBuffers(1, &b);
    EXPECT_TRUE(glIsBuffer(b));

    glNamedBufferStorage(b, 0, NULL, 0);                                         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glNamedBufferStorage(b, 64, NULL, 0x80000000u);                               EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glNamedBufferStorage(b, 64, NULL, GL_MAP_PERSISTENT_BIT);                     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glNamedBufferStorage(b, 64, NULL, GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT);    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    uint32_t init[16];
    for (int i = 0; i < 16; i++) init[i] = 0x100 + i;
    glNamedBufferStorage(b, 64, init, GL_MAP_WRITE_BIT);
    EXPECT_NO_GL_ERROR();

    GLint v = 0;
    glGetNamedBufferParameteriv(b, GL_BUFFER_IMMUTABLE_STORAGE, &v); EXPECT_EQ_INT(v, GL_TRUE);
    glGetNamedBufferParameteriv(b, GL_BUFFER_STORAGE_FLAGS, &v);     EXPECT_EQ_INT(v, GL_MAP_WRITE_BIT);

    glNamedBufferStorage(b, 64, NULL, 0);                          EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* already immutable */
    glNamedBufferData(b, 64, NULL, GL_STATIC_DRAW);                EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glNamedBufferSubData(b, 0, 4, init);                           EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* no DYNAMIC_STORAGE */
    EXPECT_TRUE(glMapNamedBufferRange(b, 0, 4, GL_MAP_READ_BIT) == NULL); EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    uint32_t *p = glMapNamedBufferRange(b, 4, 8, GL_MAP_WRITE_BIT);
    EXPECT_TRUE(p != NULL);
    if (p) { p[0] = 0xAAAA; p[1] = 0xBBBB; }
    EXPECT_TRUE(glUnmapNamedBuffer(b));
    uint32_t out[16];
    read_buf(b, 0, 64, out);
    EXPECT_EQ_INT(out[0], 0x100); EXPECT_EQ_INT(out[1], 0xAAAA); EXPECT_EQ_INT(out[2], 0xBBBB); EXPECT_EQ_INT(out[3], 0x103);
    EXPECT_NO_GL_ERROR();

    /* dynamic storage allows SubData */
    GLuint d;
    glCreateBuffers(1, &d);
    glNamedBufferStorage(d, 16, NULL, GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT);
    uint32_t four[4] = { 1, 2, 3, 4 };
    glNamedBufferSubData(d, 0, 16, four);
    read_buf(d, 0, 16, out);
    EXPECT_EQ_INT(out[3], 4);
    EXPECT_NO_GL_ERROR();

    glDeleteBuffers(1, &b);
    glDeleteBuffers(1, &d);
}

TEST(map_flag_combinations)
{
    GLuint b;
    uint8_t data[256], out[256];
    for (int i = 0; i < 256; i++) data[i] = (uint8_t)i;
    glGenBuffers(1, &b);
    glBindBuffer(GL_COPY_WRITE_BUFFER, b);
    glBufferData(GL_COPY_WRITE_BUFFER, 256, data, GL_STREAM_DRAW);

    /* read mapping sees the data */
    const uint8_t *r = glMapBufferRange(GL_COPY_WRITE_BUFFER, 16, 32, GL_MAP_READ_BIT);
    EXPECT_TRUE(r && r[0] == 16 && r[31] == 47);
    GLint v; GLint64 v64; void *ptr = NULL;
    glGetBufferParameteriv(GL_COPY_WRITE_BUFFER, GL_BUFFER_MAPPED, &v);       EXPECT_EQ_INT(v, GL_TRUE);
    glGetBufferParameteriv(GL_COPY_WRITE_BUFFER, GL_BUFFER_ACCESS, &v);       EXPECT_EQ_INT(v, GL_READ_ONLY);
    glGetBufferParameteriv(GL_COPY_WRITE_BUFFER, GL_BUFFER_ACCESS_FLAGS, &v); EXPECT_EQ_INT(v, GL_MAP_READ_BIT);
    glGetBufferParameteri64v(GL_COPY_WRITE_BUFFER, GL_BUFFER_MAP_OFFSET, &v64); EXPECT_EQ_INT(v64, 16);
    glGetBufferParameteri64v(GL_COPY_WRITE_BUFFER, GL_BUFFER_MAP_LENGTH, &v64); EXPECT_EQ_INT(v64, 32);
    glGetBufferPointerv(GL_COPY_WRITE_BUFFER, GL_BUFFER_MAP_POINTER, &ptr);     EXPECT_TRUE(ptr == r);
    /* mapped: second map, SubData, GetSubData and FlushMapped (no FLUSH_EXPLICIT) fail */
    EXPECT_TRUE(glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 4, GL_MAP_READ_BIT) == NULL); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBufferSubData(GL_COPY_WRITE_BUFFER, 0, 4, data);                  EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, 4, out);                EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glFlushMappedBufferRange(GL_COPY_WRITE_BUFFER, 0, 4);               EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_TRUE(glUnmapBuffer(GL_COPY_WRITE_BUFFER));
    glGetBufferPointerv(GL_COPY_WRITE_BUFFER, GL_BUFFER_MAP_POINTER, &ptr); EXPECT_TRUE(ptr == NULL);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);                                EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* write + invalidate range */
    uint8_t *w = glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 8, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT);
    EXPECT_TRUE(w != NULL);
    if (w) memset(w, 0xEE, 8);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, 256, out);
    EXPECT_EQ_INT(out[7], 0xEE); EXPECT_EQ_INT(out[8], 8);

    /* write + flush explicit */
    w = glMapBufferRange(GL_COPY_WRITE_BUFFER, 64, 64, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
    EXPECT_TRUE(w != NULL);
    if (w) { w[0] = 0x11; w[63] = 0x22; }
    glFlushMappedBufferRange(GL_COPY_WRITE_BUFFER, 0, 1);
    glFlushMappedBufferRange(GL_COPY_WRITE_BUFFER, 63, 1);
    EXPECT_NO_GL_ERROR();
    glFlushMappedBufferRange(GL_COPY_WRITE_BUFFER, 60, 8);              EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, 256, out);
    EXPECT_EQ_INT(out[64], 0x11); EXPECT_EQ_INT(out[127], 0x22);

    /* read/write, unsynchronized write, invalidate buffer */
    w = glMapBufferRange(GL_COPY_WRITE_BUFFER, 200, 4, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    EXPECT_TRUE(w && w[0] == 200);
    if (w) w[0] = 0x55;
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    w = glMapBufferRange(GL_COPY_WRITE_BUFFER, 201, 1, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (w) w[0] = 0x66;
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 200, 2, out);
    EXPECT_EQ_INT(out[0], 0x55); EXPECT_EQ_INT(out[1], 0x66);
    w = glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 256, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if (w) memset(w, 7, 256);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, 256, out);
    EXPECT_EQ_INT(out[0], 7); EXPECT_EQ_INT(out[255], 7);

    /* legacy MapBuffer */
    w = glMapBuffer(GL_COPY_WRITE_BUFFER, GL_READ_WRITE);
    EXPECT_TRUE(w && w[10] == 7);
    glGetBufferParameteriv(GL_COPY_WRITE_BUFFER, GL_BUFFER_ACCESS, &v); EXPECT_EQ_INT(v, GL_READ_WRITE);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    glMapBuffer(GL_COPY_WRITE_BUFFER, GL_RGBA);                         EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_NO_GL_ERROR();

    /* map errors */
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 0, GL_MAP_READ_BIT);      EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glMapBufferRange(GL_COPY_WRITE_BUFFER, -1, 4, GL_MAP_READ_BIT);     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 250, 8, GL_MAP_READ_BIT);    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 4, 0);                    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 4, GL_MAP_READ_BIT | GL_MAP_INVALIDATE_RANGE_BIT); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 4, GL_MAP_READ_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);   EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 4, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);     EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* mutable */
    glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 4, GL_MAP_READ_BIT | 0x1000);                      EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* DSA variants */
    w = glMapNamedBuffer(b, GL_WRITE_ONLY);
    if (w) w[3] = 99;
    EXPECT_TRUE(glUnmapNamedBuffer(b));
    w = glMapNamedBufferRange(b, 0, 16, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
    if (w) w[4] = 98;
    glFlushMappedNamedBufferRange(b, 4, 1);
    glGetNamedBufferPointerv(b, GL_BUFFER_MAP_POINTER, &ptr); EXPECT_TRUE(ptr == w);
    glUnmapNamedBuffer(b);
    read_buf(b, 0, 8, out);
    EXPECT_EQ_INT(out[3], 99); EXPECT_EQ_INT(out[4], 98);
    EXPECT_NO_GL_ERROR();

    glDeleteBuffers(1, &b);
}

TEST(copy_and_clear)
{
    GLuint a, b;
    uint32_t src[64], out[64];
    for (int i = 0; i < 64; i++) src[i] = 1000 + i;
    glCreateBuffers(1, &a); glCreateBuffers(1, &b);
    glNamedBufferData(a, sizeof src, src, GL_STATIC_DRAW);
    glNamedBufferData(b, sizeof src, NULL, GL_STATIC_DRAW);

    glCopyNamedBufferSubData(a, b, 8, 16, 32);
    read_buf(b, 0, sizeof out, out);
    EXPECT_EQ_INT(out[4], 1002); EXPECT_EQ_INT(out[11], 1009); EXPECT_EQ_INT(out[12], 0);

    glBindBuffer(GL_COPY_READ_BUFFER, a); glBindBuffer(GL_COPY_WRITE_BUFFER, a);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 128, 64);   /* same buffer, disjoint */
    read_buf(a, 128, 16, out);
    EXPECT_EQ_INT(out[0], 1000); EXPECT_EQ_INT(out[3], 1003);
    EXPECT_NO_GL_ERROR();
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 32, 64);    EXPECT_GL_ERROR(GL_INVALID_VALUE); /* overlap */
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 4, 0);      EXPECT_NO_GL_ERROR();
    glCopyNamedBufferSubData(a, b, 250, 0, 8);                                    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glCopyNamedBufferSubData(a, b, -4, 0, 8);                                     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMapNamedBufferRange(a, 0, 4, GL_MAP_READ_BIT);
    glCopyNamedBufferSubData(a, b, 0, 0, 4);                                      EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUnmapNamedBuffer(a);
    glCopyNamedBufferSubData(a, 0, 0, 0, 4);                                      EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* clears with conversion */
    float rgba[4] = { 1.0f, 0.5f, 0.0f, 0.25f };
    glClearNamedBufferSubData(b, GL_RGBA8, 0, 16, GL_RGBA, GL_FLOAT, rgba);
    uint8_t bytes[16];
    read_buf(b, 0, 16, bytes);
    EXPECT_EQ_INT(bytes[0], 255); EXPECT_EQ_INT(bytes[1], 128); EXPECT_EQ_INT(bytes[2], 0); EXPECT_EQ_INT(bytes[3], 64);
    EXPECT_EQ_INT(bytes[12], 255); EXPECT_EQ_INT(bytes[15], 64);

    GLuint red = 0xDEADBEEF;
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glClearBufferData(GL_ARRAY_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &red);
    read_buf(b, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 0xDEADBEEF); EXPECT_EQ_INT(out[63], 0xDEADBEEF);

    uint8_t bgra[4] = { 10, 20, 30, 40 };
    glClearBufferSubData(GL_ARRAY_BUFFER, GL_RGBA8UI, 4, 4, GL_BGRA_INTEGER, GL_UNSIGNED_BYTE, bgra);
    read_buf(b, 4, 4, bytes);
    EXPECT_EQ_INT(bytes[0], 30); EXPECT_EQ_INT(bytes[1], 20); EXPECT_EQ_INT(bytes[2], 10); EXPECT_EQ_INT(bytes[3], 40);

    float half_src = 2.0f;
    glClearNamedBufferSubData(b, GL_R16F, 8, 2, GL_RED, GL_FLOAT, &half_src);
    uint16_t h = 0; read_buf(b, 8, 2, &h);
    EXPECT_EQ_INT(h, 0x4000);

    glClearNamedBufferData(b, GL_R32F, GL_RED, GL_FLOAT, NULL);   /* NULL -> zero */
    read_buf(b, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 0); EXPECT_EQ_INT(out[63], 0);
    EXPECT_NO_GL_ERROR();

    glClearNamedBufferData(b, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, bytes);              EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glClearNamedBufferSubData(b, GL_RGBA8, 2, 4, GL_RGBA, GL_UNSIGNED_BYTE, bytes);    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glClearNamedBufferSubData(b, GL_RGBA8, 0, 1024, GL_RGBA, GL_UNSIGNED_BYTE, bytes); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glClearNamedBufferData(b, GL_R32UI, GL_RED, GL_FLOAT, rgba);                       EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* invalidate */
    glInvalidateBufferData(b);                                     EXPECT_NO_GL_ERROR();
    glInvalidateBufferSubData(b, 0, 16);                           EXPECT_NO_GL_ERROR();
    glInvalidateBufferSubData(b, 250, 16);                         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glInvalidateBufferData(0x7fffff);                              EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteBuffers(1, &a); glDeleteBuffers(1, &b);
}

TEST(indexed_bindings)
{
    GLuint b[3];
    glGenBuffers(3, b);
    for (int i = 0; i < 3; i++) { glBindBuffer(GL_UNIFORM_BUFFER, b[i]); glBufferData(GL_UNIFORM_BUFFER, 1024, NULL, GL_DYNAMIC_DRAW); }

    glBindBufferRange(GL_UNIFORM_BUFFER, 1, b[1], 256, 512);
    GLint v = 0; GLint64 v64 = 0;
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 1, &v);      EXPECT_EQ_INT(v, b[1]);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_START, 1, &v64);    EXPECT_EQ_INT(v64, 256);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, 1, &v64);     EXPECT_EQ_INT(v64, 512);
    glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &v);           EXPECT_EQ_INT(v, b[1]);   /* generic binding too */
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, b[2]);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, 2, &v64);     EXPECT_EQ_INT(v64, 0);    /* base binding */
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, 0);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 2, &v);      EXPECT_EQ_INT(v, 0);
    EXPECT_NO_GL_ERROR();

    GLintptr offs[3] = { 0, 256, 512 };
    GLsizeiptr sizes[3] = { 64, 64, 64 };
    glBindBuffersRange(GL_SHADER_STORAGE_BUFFER, 0, 3, b, offs, sizes);
    glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 2, &v); EXPECT_EQ_INT(v, b[2]);
    glGetInteger64i_v(GL_SHADER_STORAGE_BUFFER_START, 2, &v64); EXPECT_EQ_INT(v64, 512);
    glBindBuffersBase(GL_SHADER_STORAGE_BUFFER, 0, 3, NULL);
    glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, 0);
    glBindBuffersBase(GL_ATOMIC_COUNTER_BUFFER, 0, 2, b);
    glGetIntegeri_v(GL_ATOMIC_COUNTER_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, b[1]);
    EXPECT_NO_GL_ERROR();

    glBindBufferBase(GL_UNIFORM_BUFFER, 1000, b[0]);              EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBufferBase(GL_ARRAY_BUFFER, 0, b[0]);                   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, b[0], 0, 0);          EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, b[0], 3, 16);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBuffersBase(GL_UNIFORM_BUFFER, 14, 8, b);               EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 5000, &v);         EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* generic bindings of every target */
    glBindBuffer(GL_COPY_READ_BUFFER, b[0]);      glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &v);      EXPECT_EQ_INT(v, b[0]);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, b[1]);  glGetIntegerv(GL_DRAW_INDIRECT_BUFFER_BINDING, &v);  EXPECT_EQ_INT(v, b[1]);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, b[2]); glGetIntegerv(GL_DISPATCH_INDIRECT_BUFFER_BINDING, &v); EXPECT_EQ_INT(v, b[2]);
    glBindBuffer(GL_PARAMETER_BUFFER, b[0]);      glGetIntegerv(GL_PARAMETER_BUFFER_BINDING, &v);      EXPECT_EQ_INT(v, b[0]);
    glBindBuffer(GL_QUERY_BUFFER, b[1]);          glGetIntegerv(GL_QUERY_BUFFER_BINDING, &v);          EXPECT_EQ_INT(v, b[1]);
    glBindBuffer(GL_ARRAY_BUFFER, b[2]);          glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &v);          EXPECT_EQ_INT(v, b[2]);
    glBindBuffer(GL_PARAMETER_BUFFER, 0); glBindBuffer(GL_QUERY_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0); glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    EXPECT_NO_GL_ERROR();

    /* deleting unbinds everywhere */
    glDeleteBuffers(3, b);
    glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &v);       EXPECT_EQ_INT(v, 0);
    glGetIntegeri_v(GL_ATOMIC_COUNTER_BUFFER_BINDING, 1, &v); EXPECT_EQ_INT(v, 0);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------------------------ */

TEST(compute_ssbo_barrier_readback)
{
    GLuint p = compute_program(k_cs_write);
    if (!p) return;
    glUseProgram(p);

    /* small (setBytes-sized) and large SSBOs both receive the writes */
    GLuint small = make_ssbo(0, 64);
    glDispatchCompute(4, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    uint32_t out[16];
    glGetNamedBufferSubData(small, 0, sizeof out, out);
    for (int i = 0; i < 16; i++) EXPECT_EQ_INT(out[i], i * 2 + 1);

    GLuint big = make_ssbo(0, 64 * 1024);
    glDispatchCompute(1024, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    uint32_t last[2];
    glGetNamedBufferSubData(big, (4096 - 2) * 4, sizeof last, last);
    EXPECT_EQ_INT(last[0], (4096 - 2) * 2 + 1);
    EXPECT_EQ_INT(last[1], (4096 - 1) * 2 + 1);

    /* GPU-written data then copied on the GPU and mapped for reading */
    GLuint dst; glCreateBuffers(1, &dst);
    glNamedBufferData(dst, 64, NULL, GL_STATIC_READ);
    glCopyNamedBufferSubData(big, dst, 0, 0, 64);
    const uint32_t *m = glMapNamedBufferRange(dst, 0, 64, GL_MAP_READ_BIT);
    EXPECT_TRUE(m && m[0] == 1 && m[15] == 31);
    glUnmapNamedBuffer(dst);

    /* a CPU write after the dispatch must not disturb the queued GPU work */
    glDispatchCompute(4, 1, 1);
    uint32_t junk[16] = { 0 };
    glNamedBufferSubData(small, 0, sizeof junk, junk);      /* ordered after the dispatch */
    glGetNamedBufferSubData(small, 0, sizeof out, out);
    EXPECT_EQ_INT(out[5], 0);

    glDispatchCompute(0, 1, 1);                                   EXPECT_NO_GL_ERROR();  /* empty grid */
    glDispatchCompute(1u << 30, 1, 1);                            EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMemoryBarrier(0x80000000u);                                 EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMemoryBarrierByRegion(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);  EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMemoryBarrierByRegion(GL_SHADER_STORAGE_BARRIER_BIT);       EXPECT_NO_GL_ERROR();
    glUseProgram(0);
    glDispatchCompute(1, 1, 1);                                   EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glDeleteBuffers(1, &small); glDeleteBuffers(1, &big); glDeleteBuffers(1, &dst);
    glDeleteProgram(p);
}

TEST(dispatch_compute_indirect)
{
    GLuint p = compute_program(k_cs_write);
    if (!p) return;
    glUseProgram(p);
    GLuint ssbo = make_ssbo(0, 256);

    GLuint cmd[6] = { 0xFFFF, 3, 1, 1, 0, 0 };    /* command at byte offset 4: 3 groups of 4 */
    GLuint ind; glGenBuffers(1, &ind);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, ind);
    glBufferData(GL_DISPATCH_INDIRECT_BUFFER, sizeof cmd, cmd, GL_STATIC_DRAW);
    glDispatchComputeIndirect(4);
    EXPECT_NO_GL_ERROR();

    uint32_t out[16];
    glGetNamedBufferSubData(ssbo, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 1); EXPECT_EQ_INT(out[11], 23); EXPECT_EQ_INT(out[12], 0);

    glDispatchComputeIndirect(2);                                 EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDispatchComputeIndirect(-4);                                EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDispatchComputeIndirect(16);                                EXPECT_GL_ERROR(GL_INVALID_OPERATION); /* 16+12 > 24 */
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    glDispatchComputeIndirect(0);                                 EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glUseProgram(0);
    glDeleteBuffers(1, &ssbo); glDeleteBuffers(1, &ind); glDeleteProgram(p);
}

/* fragments counted by the FS through the SSBO at binding 2 */
static GLuint g_count_prog, g_count_buf;

static void count_reset(void)
{
    if (!g_count_prog)
        g_count_prog = mgl_make_program(k_vs, k_fs_count);
    if (!g_count_buf)
        g_count_buf = make_ssbo(2, 4096);
    uint32_t z[2] = { 0, 0 };
    glNamedBufferSubData(g_count_buf, 0, sizeof z, z);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_count_buf);
    glUseProgram(g_count_prog);
}

static void count_read(uint32_t *frags, uint32_t *origin)
{
    uint32_t v[2];
    glGetNamedBufferSubData(g_count_buf, 0, sizeof v, v);
    *frags = v[0]; *origin = v[1];
}

TEST(multi_draw_indirect_count)
{
    count_reset();
    if (!g_count_prog) return;
    geometry(k_fullscreen, 3);

    DrawArraysIndirectCommand cmds[3] = { { 3, 1, 0, 0 }, { 3, 2, 0, 0 }, { 3, 1, 0, 0 } };
    GLuint ind; glGenBuffers(1, &ind);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ind);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);

    GLuint params[2] = { 2, 7 };
    GLuint pb; glGenBuffers(1, &pb);
    glBindBuffer(GL_PARAMETER_BUFFER, pb);
    glBufferData(GL_PARAMETER_BUFFER, sizeof params, params, GL_STATIC_DRAW);

    uint32_t frags, origin;
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, 0, 0, 3, 0);    /* count 2: 1 + 2 instances */
    EXPECT_NO_GL_ERROR();
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 3);

    count_reset();
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, 0, 4, 3, 0);    /* count 7 clamped to 3 */
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 4);

    count_reset();
    glMultiDrawArraysIndirect(GL_TRIANGLES, (const void *)(uintptr_t)sizeof(DrawArraysIndirectCommand), 2, 0);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 3);

    count_reset();
    glDrawArraysIndirect(GL_TRIANGLES, (const void *)(uintptr_t)sizeof(DrawArraysIndirectCommand));
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 2);

    /* elements variant (fan indices through the emulation path as well) */
    GLuint idx[3] = { 0, 1, 2 };
    GLuint eb; glGenBuffers(1, &eb);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    DrawElementsIndirectCommand ecmds[2] = { { 3, 1, 0, 0, 0 }, { 3, 3, 0, 0, 0 } };
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof ecmds, ecmds, GL_STATIC_DRAW);
    count_reset();
    glMultiDrawElementsIndirectCount(GL_TRIANGLES, GL_UNSIGNED_INT, 0, 0, 2, 0);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 4);   /* params[0] == 2 -> both commands: 1 + 3 instances */
    count_reset();
    glMultiDrawElementsIndirectCount(GL_TRIANGLE_FAN, GL_UNSIGNED_INT, 0, 0, 2, 0);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 4);
    count_reset();
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (const void *)(uintptr_t)sizeof(DrawElementsIndirectCommand));
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 3);
    EXPECT_NO_GL_ERROR();

    /* errors */
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, 0, 2, 3, 0);              EXPECT_GL_ERROR(GL_INVALID_VALUE);    /* drawcount offset */
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, 0, 8, 3, 0);              EXPECT_GL_ERROR(GL_INVALID_OPERATION);/* beyond param buffer */
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, 0, 0, -1, 0);             EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMultiDrawArraysIndirect(GL_TRIANGLES, 0, 2, 3);                      EXPECT_GL_ERROR(GL_INVALID_VALUE);    /* stride */
    glMultiDrawArraysIndirect(GL_TRIANGLES, 0, 8, 0);                      EXPECT_GL_ERROR(GL_INVALID_OPERATION);/* out of range */
    glDrawArraysIndirect(GL_TRIANGLES, (const void *)2);                   EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBuffer(GL_PARAMETER_BUFFER, 0);
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, 0, 0, 3, 0);              EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glDrawArraysIndirect(GL_TRIANGLES, 0);                                 EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glDeleteBuffers(1, &ind); glDeleteBuffers(1, &pb); glDeleteBuffers(1, &eb);
}

TEST(emulated_primitive_types)
{
    count_reset();
    if (!g_count_prog) return;
    GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp);

    /* a fan over the whole viewport covers every pixel exactly once */
    static const float quad[] = { -1,-1,  1,-1,  1,1,  -1,1 };
    geometry(quad, 4);
    uint32_t frags, origin;
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(frags, (uint32_t)(vp[2] * vp[3]));
    EXPECT_EQ_INT(origin, 1);

    /* indexed fan with GL_UNSIGNED_BYTE indices */
    GLubyte ub[4] = { 0, 1, 2, 3 };
    GLuint eb; glGenBuffers(1, &eb);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ub, ub, GL_STATIC_DRAW);
    count_reset();
    glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_BYTE, 0);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(frags, (uint32_t)(vp[2] * vp[3]));

    /* a 2-vertex line loop draws the segment twice, a line strip once */
    static const float seg[] = { -0.5f, 0.1f,  0.5f, 0.1f };
    geometry(seg, 2);
    uint32_t strip_frags, loop_frags;
    count_reset(); glDrawArrays(GL_LINE_STRIP, 0, 2); count_read(&strip_frags, &origin);
    count_reset(); glDrawArrays(GL_LINE_LOOP, 0, 2);  count_read(&loop_frags, &origin);
    EXPECT_TRUE(strip_frags > 0);
    EXPECT_EQ_INT(loop_frags, strip_frags * 2);

    /* adjacency without a geometry shader ignores the adjacent vertices */
    static const float adj[] = { -1,-1,  -1,-1,  1,-1,  1,-1,  -1,1,  -1,1 };  /* tri (v0,v2,v4) */
    geometry(adj, 6);
    count_reset(); glDrawArrays(GL_TRIANGLES_ADJACENCY, 0, 6); count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 1);
    EXPECT_TRUE(frags > (uint32_t)(vp[2] * vp[3] / 3));

    /* base vertex: indices 0..2 shifted onto the fullscreen triangle stored at 3..5 */
    static const float shifted[] = { 5,5, 5,6, 6,5,   -1,-1, 3,-1, -1,3 };
    geometry(shifted, 6);
    GLuint ui[3] = { 0, 1, 2 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ui, ui, GL_STATIC_DRAW);
    count_reset();
    glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0, 2, 3, 5);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 2);
    count_reset();
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);   /* without base vertex: offscreen */
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 0);
    count_reset();
    GLsizei counts[2] = { 3, 3 }; const void *offs[2] = { 0, 0 }; GLint bvs[2] = { 3, 0 };
    glMultiDrawElementsBaseVertex(GL_TRIANGLES, counts, GL_UNSIGNED_INT, offs, 2, bvs);
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 1);
    EXPECT_NO_GL_ERROR();

    glDeleteBuffers(1, &eb);
}

TEST(persistent_coherent_between_draws)
{
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "layout(std430, binding=1) readonly buffer In { uint v; };\n"
        "layout(std430, binding=2) buffer Count { uint frags; uint at_origin; };\n"
        "void main(){ if (ivec2(gl_FragCoord.xy) == ivec2(0)) atomicAdd(at_origin, v); color = vec4(1.0); }\n";
    GLuint prog = mgl_make_program(k_vs, fs);
    if (!prog) return;
    count_reset();
    glUseProgram(prog);
    geometry(k_fullscreen, 3);

    GLuint pb; glCreateBuffers(1, &pb);
    GLbitfield f = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    glNamedBufferStorage(pb, 16, NULL, f);
    volatile uint32_t *p = glMapNamedBufferRange(pb, 0, 16, f);
    EXPECT_TRUE(p != NULL);
    if (!p) return;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pb);

    p[0] = 5;
    glDrawArrays(GL_TRIANGLES, 0, 3);           /* drawing while persistently mapped is legal */
    EXPECT_NO_GL_ERROR();
    GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    GLenum r = glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 2000000000ull);
    EXPECT_TRUE(r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
    glDeleteSync(s);

    p[0] = 7;                                   /* coherent: visible to the next draw */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    uint32_t frags, origin;
    count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 12);

    GLint mapped = 0;
    glGetNamedBufferParameteriv(pb, GL_BUFFER_MAPPED, &mapped); EXPECT_EQ_INT(mapped, GL_TRUE);
    glUnmapNamedBuffer(pb);

    /* a non-persistent mapping blocks draws that source the buffer */
    geometry(k_fullscreen, 3);
    glMapNamedBufferRange(g_vbo, 0, 8, GL_MAP_READ_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);           EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glUnmapNamedBuffer(g_vbo);

    glDeleteBuffers(1, &pb);
    glDeleteProgram(prog);
    EXPECT_NO_GL_ERROR();
}

TEST(fences)
{
    count_reset();
    geometry(k_fullscreen, 3);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    EXPECT_TRUE(s != NULL && glIsSync(s));
    GLint v = 0; GLsizei len = -1;
    glGetSynciv(s, GL_OBJECT_TYPE, 1, &len, &v);     EXPECT_EQ_INT(v, GL_SYNC_FENCE); EXPECT_EQ_INT(len, 1);
    glGetSynciv(s, GL_SYNC_CONDITION, 1, NULL, &v);  EXPECT_EQ_INT(v, GL_SYNC_GPU_COMMANDS_COMPLETE);
    glGetSynciv(s, GL_SYNC_FLAGS, 1, NULL, &v);      EXPECT_EQ_INT(v, 0);
    glGetSynciv(s, GL_SYNC_STATUS, 0, &len, &v);     EXPECT_EQ_INT(len, 0);

    /* not flushed yet: a zero timeout can't succeed */
    GLenum r = glClientWaitSync(s, 0, 0);
    EXPECT_TRUE(r == GL_TIMEOUT_EXPIRED || r == GL_ALREADY_SIGNALED);
    r = glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000000ull);
    EXPECT_TRUE(r == GL_CONDITION_SATISFIED || r == GL_ALREADY_SIGNALED);
    glGetSynciv(s, GL_SYNC_STATUS, 1, NULL, &v);     EXPECT_EQ_INT(v, GL_SIGNALED);
    r = glClientWaitSync(s, 0, 0);                   EXPECT_EQ_INT(r, GL_ALREADY_SIGNALED);
    glWaitSync(s, 0, GL_TIMEOUT_IGNORED);
    EXPECT_NO_GL_ERROR();

    /* server wait on a pending fence, then rendering continues */
    GLsync s2 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glWaitSync(s2, 0, GL_TIMEOUT_IGNORED);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glGetSynciv(s2, GL_SYNC_STATUS, 1, NULL, &v);    EXPECT_EQ_INT(v, GL_SIGNALED);
    uint32_t frags, origin; count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 2);
    EXPECT_NO_GL_ERROR();

    /* errors */
    glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 1);   EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glFenceSync(0x1234, 0);                          EXPECT_GL_ERROR(GL_INVALID_ENUM);
    r = glClientWaitSync(s, 0x10, 0);                EXPECT_EQ_INT(r, GL_WAIT_FAILED); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glWaitSync(s, 0, 10);                            EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glWaitSync(s, 1, GL_TIMEOUT_IGNORED);            EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetSynciv(s, 0x1234, 1, NULL, &v);             EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetSynciv(s, GL_SYNC_STATUS, -1, NULL, &v);    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteSync(s); glDeleteSync(s2);
    EXPECT_TRUE(!glIsSync(s));
    glDeleteSync(NULL);                              EXPECT_NO_GL_ERROR();
    r = glClientWaitSync(s, 0, 0);                   EXPECT_EQ_INT(r, GL_WAIT_FAILED); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDeleteSync(s);                                 EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(draw_errors)
{
    count_reset();
    geometry(k_fullscreen, 3);

    glDrawArrays(0x1234, 0, 3);                                   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDrawArrays(GL_TRIANGLES, 0, -1);                            EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDrawArrays(GL_TRIANGLES, -1, 3);                            EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, -1);                EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDrawElements(GL_TRIANGLES, 3, GL_FLOAT, 0);                 EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDrawElements(GL_TRIANGLES, -3, GL_UNSIGNED_INT, 0);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDrawRangeElements(GL_TRIANGLES, 5, 1, 3, GL_UNSIGNED_INT, 0); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);          EXPECT_GL_ERROR(GL_INVALID_OPERATION);  /* no element buffer */
    glDrawArrays(GL_PATCHES, 0, 3);                               EXPECT_GL_ERROR(GL_INVALID_OPERATION);  /* no tessellation */
    glMultiDrawArrays(GL_TRIANGLES, NULL, NULL, -1);              EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDrawArrays(GL_TRIANGLES, 0, 0);                             EXPECT_NO_GL_ERROR();

    GLuint dummy; glGenVertexArrays(1, &dummy);
    glBindVertexArray(0);
    glDrawArrays(GL_TRIANGLES, 0, 3);                             EXPECT_GL_ERROR(GL_INVALID_OPERATION);  /* no VAO */
    glBindVertexArray(g_vao);
    glDeleteVertexArrays(1, &dummy);

    /* nothing was drawn by the failing calls */
    uint32_t frags, origin; count_read(&frags, &origin);
    EXPECT_EQ_INT(origin, 0);
}

int main(void)
{
    RUN(buffer_data_and_params);
    RUN(buffer_storage_flags);
    RUN(map_flag_combinations);
    RUN(copy_and_clear);
    RUN(indexed_bindings);
    RUN(compute_ssbo_barrier_readback);
    RUN(dispatch_compute_indirect);
    RUN(multi_draw_indirect_count);
    RUN(emulated_primitive_types);
    RUN(persistent_coherent_between_draws);
    RUN(fences);
    RUN(draw_errors);
    return mgl_test_summary();
}
