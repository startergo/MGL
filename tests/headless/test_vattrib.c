/*
 * test_vattrib.c — generic vertex attributes, vertex arrays, vertex buffer bindings.
 *
 * Attribute fetch is verified without rasterization: the vertex shader writes the value it
 * received into an SSBO (index gl_VertexID + 64 * gl_InstanceID), read back with
 * glGetBufferSubData.  glFinish currently commits without waiting for completion, so
 * gpu_wait() sleeps briefly after it.
 */
#include "mgl_test.h"
#include <unistd.h>
#include <limits.h>

#define CAP_SLOTS 256

static void gpu_wait(void) { glFinish(); usleep(200000); }

/*
 * The renderer cannot yet record more draws after a glFinish in the same process (the committed
 * command buffer is not replaced), so every draw writes to its own SSBO and all fetched values are
 * checked once, after a single glFinish, by verify_captures().
 */
static GLuint g_ssbo;

typedef struct { const char *test; int line; GLuint ssbo; int slot; int is_int; double exp[4]; double eps; } Capture;
static Capture g_caps[512];
static int g_ncaps;

static void cap_expect(int line, int slot, int is_int, double a, double b, double c, double d, double eps)
{
    if (g_ncaps >= 512) return;
    Capture *k = &g_caps[g_ncaps++];
    k->test = mgl_test_current; k->line = line; k->ssbo = g_ssbo; k->slot = slot; k->is_int = is_int;
    k->exp[0] = a; k->exp[1] = b; k->exp[2] = c; k->exp[3] = d; k->eps = eps;
}
#define CAP_VEC4(slot, a, b, c, d, eps) cap_expect(__LINE__, slot, 0, a, b, c, d, eps)
#define CAP_IVEC4(slot, a, b, c, d)     cap_expect(__LINE__, slot, 1, a, b, c, d, 0)

/* program whose VS stores <expr> (of type <otype>) for each vertex */
static GLuint cap_program(const char *decls, const char *otype, const char *expr)
{
    char vs[4096];
    snprintf(vs, sizeof vs,
        "#version 450 core\n%s\n"
        "layout(std430, binding=0) buffer Out { %s v[]; } o;\n"
        "void main(){ o.v[gl_VertexID + gl_InstanceID * 64] = %s; gl_Position = vec4(0.0, 0.0, 0.0, 1.0); gl_PointSize = 1.0; }\n",
        decls, otype, expr);
    const char *fs = "#version 450 core\nlayout(location=0) out vec4 c;\nvoid main(){ c = vec4(1.0); }\n";
    return mgl_make_program(vs, fs);
}

/* a fresh capture buffer for the next draw */
static void cap_reset(void)
{
    static GLuint zero[CAP_SLOTS * 4];
    glGenBuffers(1, &g_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof zero, zero, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_ssbo);
}

static void verify_captures(void)
{
    gpu_wait();
    const char *cur = NULL;
    int fails_before = 0;
    for (int i = 0; i <= g_ncaps; i++)
    {
        if (i == g_ncaps || (cur && strcmp(cur, g_caps[i].test)))
        {
            printf("%s %s.fetch\n", mgl_test_failures == fails_before ? "PASS" : "FAIL", cur);
            if (i == g_ncaps) break;
        }
        if (!cur || strcmp(cur, g_caps[i].test)) { cur = g_caps[i].test; fails_before = mgl_test_failures; }
        Capture *k = &g_caps[i];
        union { float f[4]; GLint i[4]; } v;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, k->ssbo);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, k->slot * 16, 16, &v);
        for (int c = 0; c < 4; c++)
        {
            double got = k->is_int ? (double)v.i[c] : (double)v.f[c];
            mgl_test_checks++;
            if (!(fabs(got - k->exp[c]) <= k->eps))
            {
                mgl_test_failures++;
                fprintf(stderr, "  [%s] test_vattrib.c:%d: slot %d comp %d == %f, expected %f\n",
                        k->test, k->line, k->slot, c, got, k->exp[c]);
            }
        }
    }
}

static GLuint make_vbo(const void *data, GLsizeiptr size)
{
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    return b;
}

static GLuint make_vao(void)
{
    GLuint v;
    glGenVertexArrays(1, &v);
    glBindVertexArray(v);
    return v;
}

#define EXPECT_VEC4(p, a, b, c, d, eps) do { EXPECT_NEAR((p)[0], a, eps); EXPECT_NEAR((p)[1], b, eps); \
        EXPECT_NEAR((p)[2], c, eps); EXPECT_NEAR((p)[3], d, eps); } while (0)
#define EXPECT_IVEC4(p, a, b, c, d) do { EXPECT_EQ_INT((p)[0], a); EXPECT_EQ_INT((p)[1], b); \
        EXPECT_EQ_INT((p)[2], c); EXPECT_EQ_INT((p)[3], d); } while (0)

/* ------------------------------------------------------------------------------------------ */

TEST(current_value_setters)
{
    GLfloat f[4]; GLint i[4]; GLuint u[4]; GLdouble d[4];

    glVertexAttrib1f(1, 2.5f);
    glGetVertexAttribfv(1, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 2.5, 0, 0, 1, 0);
    glVertexAttrib2s(1, -3, 4);
    glGetVertexAttribfv(1, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -3, 4, 0, 1, 0);
    glVertexAttrib3d(1, 1.5, 2.5, 3.5);
    glGetVertexAttribdv(1, GL_CURRENT_VERTEX_ATTRIB, d); EXPECT_VEC4(d, 1.5, 2.5, 3.5, 1, 0);
    { const GLfloat v[4] = { 1, 2, 3, 4 }; glVertexAttrib4fv(2, v); }
    glGetVertexAttribfv(2, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1, 2, 3, 4, 0);
    { const GLshort v[4] = { 5, -6, 7, -8 }; glVertexAttrib4sv(2, v); }
    glGetVertexAttribiv(2, GL_CURRENT_VERTEX_ATTRIB, i); EXPECT_IVEC4(i, 5, -6, 7, -8);
    { const GLdouble v[2] = { 0.25, 0.75 }; glVertexAttrib2dv(3, v); }
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 0.25, 0.75, 0, 1, 0);

    /* unnormalized 4{b,ub,us,ui,i}v */
    { const GLbyte v[4] = { -1, 2, -3, 4 }; glVertexAttrib4bv(3, v); }
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -1, 2, -3, 4, 0);
    { const GLubyte v[4] = { 255, 0, 1, 2 }; glVertexAttrib4ubv(3, v); }
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 255, 0, 1, 2, 0);
    { const GLuint v[4] = { 100000, 1, 2, 3 }; glVertexAttrib4uiv(3, v); }
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 100000, 1, 2, 3, 0);

    /* normalized 4N* edge values (signed: max(c / (2^(b-1)-1), -1)) */
    { const GLbyte v[4] = { -128, -127, 127, 0 }; glVertexAttrib4Nbv(4, v); }
    glGetVertexAttribfv(4, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -1, -1, 1, 0, 1e-6);
    { const GLshort v[4] = { -32768, 32767, 16384, 0 }; glVertexAttrib4Nsv(4, v); }
    glGetVertexAttribfv(4, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -1, 1, 16384.0 / 32767.0, 0, 1e-6);
    { const GLint v[4] = { INT_MIN, INT_MAX, 0, 0 }; glVertexAttrib4Niv(4, v); }
    glGetVertexAttribfv(4, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -1, 1, 0, 0, 1e-6);
    glVertexAttrib4Nub(4, 255, 0, 51, 255);
    glGetVertexAttribfv(4, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1, 0, 0.2, 1, 1e-6);
    { const GLushort v[4] = { 65535, 0, 0, 65535 }; glVertexAttrib4Nusv(4, v); }
    glGetVertexAttribfv(4, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1, 0, 0, 1, 1e-6);
    { const GLuint v[4] = { 0xFFFFFFFFu, 0, 0, 0 }; glVertexAttrib4Nuiv(4, v); }
    glGetVertexAttribfv(4, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1, 0, 0, 0, 1e-6);

    /* pure integer */
    glVertexAttribI1i(5, -7);
    glGetVertexAttribIiv(5, GL_CURRENT_VERTEX_ATTRIB, i); EXPECT_IVEC4(i, -7, 0, 0, 1);
    glVertexAttribI4ui(5, 0xFFFFFFF0u, 1, 2, 3);
    glGetVertexAttribIuiv(5, GL_CURRENT_VERTEX_ATTRIB, u);
    EXPECT_EQ_INT(u[0], 0xFFFFFFF0u); EXPECT_EQ_INT(u[3], 3);
    { const GLbyte v[4] = { -1, -2, -3, -4 }; glVertexAttribI4bv(5, v); }
    glGetVertexAttribIiv(5, GL_CURRENT_VERTEX_ATTRIB, i); EXPECT_IVEC4(i, -1, -2, -3, -4);
    { const GLushort v[4] = { 65535, 1, 2, 3 }; glVertexAttribI4usv(5, v); }
    glGetVertexAttribIuiv(5, GL_CURRENT_VERTEX_ATTRIB, u); EXPECT_EQ_INT(u[0], 65535);
    { const GLuint v[3] = { 9, 8, 7 }; glVertexAttribI3uiv(5, v); }
    glGetVertexAttribIuiv(5, GL_CURRENT_VERTEX_ATTRIB, u); EXPECT_IVEC4(u, 9, 8, 7, 1);

    /* 64 bit */
    glVertexAttribL4d(6, 1.0 / 3.0, 1e300, -2.0, 0.5);
    glGetVertexAttribLdv(6, GL_CURRENT_VERTEX_ATTRIB, d);
    EXPECT_TRUE(d[0] == 1.0 / 3.0); EXPECT_TRUE(d[1] == 1e300); EXPECT_TRUE(d[2] == -2.0);
    glVertexAttribL1d(6, 4.0);
    glGetVertexAttribLdv(6, GL_CURRENT_VERTEX_ATTRIB, d); EXPECT_VEC4(d, 4, 0, 0, 1, 0);

    EXPECT_NO_GL_ERROR();
}

TEST(packed_setters)
{
    GLfloat f[4];
    /* INT_2_10_10_10_REV: x=-512 y=511 z=-1 w=-2 */
    GLuint v = (0x200u) | (0x1FFu << 10) | (0x3FFu << 20) | (0x2u << 30);
    glVertexAttribP4ui(1, GL_INT_2_10_10_10_REV, GL_TRUE, v);
    glGetVertexAttribfv(1, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -1, 1, -1.0 / 511.0, -1, 1e-6);
    glVertexAttribP4ui(1, GL_INT_2_10_10_10_REV, GL_FALSE, v);
    glGetVertexAttribfv(1, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, -512, 511, -1, -2, 0);
    /* w = 1 normalizes to 1 */
    glVertexAttribP4uiv(1, GL_INT_2_10_10_10_REV, GL_TRUE, &(GLuint){ 1u << 30 });
    glGetVertexAttribfv(1, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 0, 0, 0, 1, 1e-6);

    GLuint uv = 1023u | (0u << 10) | (512u << 20) | (3u << 30);
    glVertexAttribP4ui(2, GL_UNSIGNED_INT_2_10_10_10_REV, GL_TRUE, uv);
    glGetVertexAttribfv(2, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1, 0, 512.0 / 1023.0, 1, 1e-6);
    glVertexAttribP3ui(2, GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, uv);
    glGetVertexAttribfv(2, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1023, 0, 512, 1, 0);
    glVertexAttribP2ui(2, GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, uv);
    glGetVertexAttribfv(2, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1023, 0, 0, 1, 0);
    glVertexAttribP1uiv(2, GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, &uv);
    glGetVertexAttribfv(2, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1023, 0, 0, 1, 0);

    /* 10F_11F_11F: r = 1.0 (e=15), g = 2.0 (e=16), b = 0.5 (e=14, 10 bit) */
    GLuint fv = (15u << 6) | ((16u << 6) << 11) | ((14u << 5) << 22);
    glVertexAttribP3ui(3, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FALSE, fv);
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_VEC4(f, 1, 2, 0.5, 1, 0);
    /* r with mantissa: 1.5 = e15 m32 */
    glVertexAttribP3uiv(3, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_TRUE, &(GLuint){ (15u << 6) | 32u });
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_NEAR(f[0], 1.5, 0);

    glVertexAttribP4ui(3, GL_FLOAT, GL_FALSE, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glVertexAttribP1ui(3, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FALSE, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_NO_GL_ERROR();
}

TEST(setter_errors)
{
    GLint max = 0;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max);
    EXPECT_TRUE(max >= 16);
    glVertexAttrib4f(max, 0, 0, 0, 0);          EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribI4i(max, 0, 0, 0, 0);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribL1d(max, 0);                  EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribP4ui(max, GL_INT_2_10_10_10_REV, GL_FALSE, 0); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLfloat f[4];
    glGetVertexAttribfv(max, GL_CURRENT_VERTEX_ATTRIB, f); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindVertexArray(0);
    glGetVertexAttribfv(0, GL_VERTEX_ATTRIB_ARRAY_SIZE, f); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetVertexAttribfv(0, GL_CURRENT_VERTEX_ATTRIB, f);   EXPECT_NO_GL_ERROR();
}

TEST(limits)
{
    GLint v = 0;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIB_BINDINGS, &v);        EXPECT_EQ_INT(v, 16);
    glGetIntegerv(GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET, &v); EXPECT_EQ_INT(v, 2047);
    glGetIntegerv(GL_MAX_VERTEX_ATTRIB_STRIDE, &v);          EXPECT_EQ_INT(v, 2048);
    EXPECT_NO_GL_ERROR();
}

TEST(pointer_errors)
{
    GLuint vbo = make_vbo(NULL, 64);
    glBindVertexArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEnableVertexAttribArray(0);                                EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribFormat(0, 4, GL_FLOAT, GL_FALSE, 0);           EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindVertexBuffer(0, vbo, 0, 16);                           EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribDivisor(0, 1);                                 EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    make_vao();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void *)16); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);       EXPECT_NO_GL_ERROR();
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 5, GL_FLOAT, GL_FALSE, 0, 0);          EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribPointer(0, 4, GL_RGBA, GL_FALSE, 0, 0);           EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, -1, 0);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4096, 0);       EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribPointer(16, 4, GL_FLOAT, GL_FALSE, 0, 0);         EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribPointer(0, GL_BGRA, GL_FLOAT, GL_TRUE, 0, 0);     EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribPointer(0, GL_BGRA, GL_UNSIGNED_BYTE, GL_FALSE, 0, 0); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribPointer(0, 3, GL_INT_2_10_10_10_REV, GL_TRUE, 0, 0);   EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribPointer(0, 4, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FALSE, 0, 0); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexAttribIPointer(0, 4, GL_FLOAT, 0, 0);                   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glVertexAttribIPointer(0, GL_BGRA, GL_UNSIGNED_BYTE, 0, 0);     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribLPointer(0, 4, GL_FLOAT, 0, 0);                   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glVertexAttribFormat(0, 4, GL_FLOAT, GL_FALSE, 2048);           EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribIFormat(0, 4, GL_HALF_FLOAT, 0);                  EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glVertexAttribLFormat(0, 2, GL_DOUBLE, 0);                      EXPECT_NO_GL_ERROR();
    glVertexAttribBinding(0, 16);                                   EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexAttribBinding(16, 0);                                   EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glVertexBindingDivisor(16, 1);                                  EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindVertexBuffer(16, vbo, 0, 16);                             EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindVertexBuffer(0, vbo, -4, 16);                             EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindVertexBuffer(0, 12345, 0, 16);                            EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    {
        GLuint b[2] = { vbo, vbo }; GLintptr o[2] = { 0, 0 }; GLsizei s[2] = { 16, 16 };
        glBindVertexBuffers(15, 2, b, o, s);                        EXPECT_GL_ERROR(GL_INVALID_OPERATION);
        glBindVertexBuffers(0, 2, b, o, s);                         EXPECT_NO_GL_ERROR();
        glBindVertexBuffers(0, 2, NULL, NULL, NULL);                EXPECT_NO_GL_ERROR();
        GLint name = -1; glGetIntegeri_v(GL_VERTEX_BINDING_BUFFER, 1, &name);
        while (glGetError() != GL_NO_ERROR) {}
    }
    glEnableVertexAttribArray(16);                                  EXPECT_GL_ERROR(GL_INVALID_VALUE);
    void *p;
    glGetVertexAttribPointerv(0, GL_VERTEX_ATTRIB_ARRAY_SIZE, &p);  EXPECT_GL_ERROR(GL_INVALID_ENUM);
    GLint iv;
    glGetVertexAttribiv(0, GL_TEXTURE_2D, &iv);                     EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glBindVertexArray(54321);                                       EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGenVertexArrays(-1, NULL);                                    EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(attrib_queries)
{
    GLuint vao = make_vao();
    GLuint vbo = make_vbo(NULL, 256);
    GLint v;
    void *p;

    /* defaults */
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_SIZE, &v);       EXPECT_EQ_INT(v, 4);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_TYPE, &v);       EXPECT_EQ_INT(v, GL_FLOAT);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_BINDING, &v);          EXPECT_EQ_INT(v, 3);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &v);    EXPECT_EQ_INT(v, 0);

    glVertexAttribPointer(3, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 20, (void *)8);
    glEnableVertexAttribArray(3);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_SIZE, &v);       EXPECT_EQ_INT(v, GL_BGRA);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_TYPE, &v);       EXPECT_EQ_INT(v, GL_UNSIGNED_BYTE);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &v);     EXPECT_EQ_INT(v, 20);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &v); EXPECT_EQ_INT(v, 1);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &v);    EXPECT_EQ_INT(v, 0);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &v);    EXPECT_EQ_INT(v, 1);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &v); EXPECT_EQ_INT(v, vbo);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_RELATIVE_OFFSET, &v);  EXPECT_EQ_INT(v, 0);
    glGetVertexAttribPointerv(3, GL_VERTEX_ATTRIB_ARRAY_POINTER, &p); EXPECT_TRUE(p == (void *)8);
    glGetIntegeri_v(GL_VERTEX_BINDING_OFFSET, 3, &v);
    if (glGetError() == GL_NO_ERROR) { EXPECT_EQ_INT(v, 8); } /* indexed get hook needs the debug domain's get.c wiring */

    glVertexAttribIPointer(4, 2, GL_SHORT, 0, (void *)4);
    glGetVertexAttribIiv(4, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &v);   EXPECT_EQ_INT(v, 1);
    glGetVertexAttribiv(4, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &v);     EXPECT_EQ_INT(v, 0);
    GLuint uv; glGetVertexAttribIuiv(4, GL_VERTEX_ATTRIB_ARRAY_SIZE, &uv); EXPECT_EQ_INT(uv, 2);

    glVertexAttribLPointer(5, 3, GL_DOUBLE, 48, 0);
    GLdouble dv; glGetVertexAttribLdv(5, GL_VERTEX_ATTRIB_ARRAY_LONG, &dv); EXPECT_NEAR(dv, 1, 0);
    GLfloat fv; glGetVertexAttribfv(5, GL_VERTEX_ATTRIB_ARRAY_TYPE, &fv);  EXPECT_NEAR(fv, GL_DOUBLE, 0);

    glVertexAttribDivisor(5, 3);
    glGetVertexAttribiv(5, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &v);    EXPECT_EQ_INT(v, 3);

    /* separate format / binding */
    glVertexAttribFormat(6, 3, GL_SHORT, GL_TRUE, 12);
    glVertexAttribBinding(6, 9);
    glBindVertexBuffer(9, vbo, 32, 24);
    glVertexBindingDivisor(9, 2);
    glGetVertexAttribiv(6, GL_VERTEX_ATTRIB_RELATIVE_OFFSET, &v);  EXPECT_EQ_INT(v, 12);
    glGetVertexAttribiv(6, GL_VERTEX_ATTRIB_BINDING, &v);          EXPECT_EQ_INT(v, 9);
    glGetVertexAttribiv(6, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &v);    EXPECT_EQ_INT(v, 2);
    glGetVertexAttribiv(6, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &v); EXPECT_EQ_INT(v, vbo);

    GLint bound; glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &bound);    EXPECT_EQ_INT(bound, vao);
    EXPECT_TRUE(glIsVertexArray(vao));
    glDeleteVertexArrays(1, &vao);
    EXPECT_TRUE(!glIsVertexArray(vao));
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &bound);                 EXPECT_EQ_INT(bound, 0);
    EXPECT_NO_GL_ERROR();
}

TEST(dsa_vao)
{
    GLuint vao, gen;
    GLint v;
    GLint64 v64;
    GLuint vbo = make_vbo(NULL, 128);
    GLuint ebo; glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBindVertexArray(0);

    glCreateVertexArrays(1, &vao);
    EXPECT_TRUE(glIsVertexArray(vao));
    glGenVertexArrays(1, &gen);
    EXPECT_TRUE(!glIsVertexArray(gen));
    glVertexArrayAttribFormat(gen, 0, 4, GL_FLOAT, GL_FALSE, 0);   EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEnableVertexArrayAttrib(777, 0);                             EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetVertexArrayiv(777, GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);  EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glVertexArrayVertexBuffer(vao, 2, vbo, 16, 32);
    glVertexArrayAttribIFormat(vao, 1, 3, GL_UNSIGNED_SHORT, 4);
    glVertexArrayAttribBinding(vao, 1, 2);
    glVertexArrayBindingDivisor(vao, 2, 5);
    glEnableVertexArrayAttrib(vao, 1);
    glVertexArrayElementBuffer(vao, ebo);
    EXPECT_NO_GL_ERROR();

    glGetVertexArrayiv(vao, GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);  EXPECT_EQ_INT(v, ebo);
    glGetVertexArrayiv(vao, GL_VERTEX_ARRAY_BINDING, &v);          EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &v);  EXPECT_EQ_INT(v, 1);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_SIZE, &v);     EXPECT_EQ_INT(v, 3);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_TYPE, &v);     EXPECT_EQ_INT(v, GL_UNSIGNED_SHORT);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &v);  EXPECT_EQ_INT(v, 1);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_LONG, &v);     EXPECT_EQ_INT(v, 0);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &v);  EXPECT_EQ_INT(v, 5);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_RELATIVE_OFFSET, &v); EXPECT_EQ_INT(v, 4);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &v); EXPECT_EQ_INT(v, 0);
    glGetVertexArrayIndexediv(vao, 16, GL_VERTEX_ATTRIB_ARRAY_SIZE, &v);    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetVertexArrayIndexediv(vao, 1, GL_VERTEX_BINDING_OFFSET, &v);        EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetVertexArrayIndexed64iv(vao, 2, GL_VERTEX_BINDING_OFFSET, &v64);    EXPECT_EQ_INT(v64, 16);
    glGetVertexArrayIndexed64iv(vao, 2, GL_VERTEX_ATTRIB_ARRAY_SIZE, &v64); EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* element array binding is VAO state */
    glBindVertexArray(vao);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);            EXPECT_EQ_INT(v, ebo);
    glBindVertexArray(gen);   /* first bind creates the object */
    EXPECT_TRUE(glIsVertexArray(gen));
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);            EXPECT_EQ_INT(v, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glGetVertexArrayiv(gen, GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);  EXPECT_EQ_INT(v, ebo);
    glVertexArrayElementBuffer(gen, 0);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);            EXPECT_EQ_INT(v, 0);
    glVertexArrayElementBuffer(gen, 4242);                         EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexArrayVertexBuffer(gen, 0, 4242, 0, 16);                EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    {
        GLuint b[2] = { vbo, 4242 }; GLintptr o[2] = { 4, 8 }; GLsizei s[2] = { 8, 8 };
        glVertexArrayVertexBuffers(gen, 3, 2, b, o, s);            EXPECT_GL_ERROR(GL_INVALID_OPERATION);
        glGetVertexArrayIndexed64iv(gen, 3, GL_VERTEX_BINDING_OFFSET, &v64); EXPECT_EQ_INT(v64, 4);
    }
    EXPECT_NO_GL_ERROR();
}

/* ---------------------------------- attribute fetch on Metal -------------------------------- */

TEST(constant_attribute)
{
    GLuint prog = cap_program("layout(location=0) in vec2 pos;\nlayout(location=1) in vec4 color;",
                              "vec4", "color + vec4(pos, 0.0, 0.0) * 0.0");
    if (!prog) return;
    static const float pos[] = { 0, 0, 0, 0, 0, 0 };
    make_vao();
    make_vbo(pos, sizeof pos);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glUseProgram(prog);
    cap_reset();
    glVertexAttrib4f(1, 0.25f, 0.5f, 0.75f, 1.0f);
    glDrawArrays(GL_POINTS, 0, 3);
    CAP_VEC4(0, 0.25, 0.5, 0.75, 1.0, 1e-6);
    CAP_VEC4(2, 0.25, 0.5, 0.75, 1.0, 1e-6);

    /* the current value is re-read at each draw */
    cap_reset();
    glVertexAttrib3f(1, 9, 8, 7);
    glDrawArrays(GL_POINTS, 0, 2);
    CAP_VEC4(1, 9, 8, 7, 1, 1e-6);
    EXPECT_NO_GL_ERROR();
}

TEST(constant_int_attributes)
{
    GLuint prog = cap_program("layout(location=0) in vec2 pos;\nlayout(location=2) in ivec4 iv;\nlayout(location=3) in uvec4 uv;",
                              "ivec4", "(gl_VertexID == 0) ? iv : ivec4(uv) + ivec4(int(pos.x))");
    if (!prog) return;
    static const float pos[] = { 0, 0, 0, 0 };
    make_vao();
    make_vbo(pos, sizeof pos);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    cap_reset();
    glVertexAttribI4i(2, -100000, 7, -3, 2000000000);
    glVertexAttribI4ui(3, 0x7FFFFFF0u, 1, 2, 3);
    glDrawArrays(GL_POINTS, 0, 2);
    CAP_IVEC4(0, -100000, 7, -3, 2000000000);
    CAP_IVEC4(1, 0x7FFFFFF0, 1, 2, 3);
    EXPECT_NO_GL_ERROR();
}

TEST(integer_arrays)
{
    GLuint prog = cap_program("layout(location=0) in ivec4 a;\nlayout(location=1) in uvec4 b;",
                              "ivec4", "(gl_InstanceID == 0) ? a : ivec4(b)");
    if (!prog) return;
    static const GLint ai[] = { 100000, -5, 7, -2000000000,   1, 2, 3, 4 };
    static const GLubyte bu[] = { 255, 0, 128, 7,   1, 2, 3, 4 };
    make_vao();
    make_vbo(ai, sizeof ai);
    glVertexAttribIPointer(0, 4, GL_INT, 0, 0);
    glEnableVertexAttribArray(0);
    make_vbo(bu, sizeof bu);
    glVertexAttribIPointer(1, 4, GL_UNSIGNED_BYTE, 0, 0);
    glEnableVertexAttribArray(1);
    glUseProgram(prog);
    cap_reset();
    glDrawArraysInstanced(GL_POINTS, 0, 2, 2);
    CAP_IVEC4(0, 100000, -5, 7, -2000000000);
    CAP_IVEC4(1, 1, 2, 3, 4);
    CAP_IVEC4(64, 255, 0, 128, 7);
    CAP_IVEC4(65, 1, 2, 3, 4);

    /* signedness mismatch (GL_SHORT data into a uvec4) goes through the CPU converted stream */
    static const GLshort sh[] = { 3, 4, 5, 6,   -1, 0, 0, 0 };
    make_vbo(sh, sizeof sh);
    glVertexAttribIPointer(1, 3, GL_SHORT, 8, 0);
    cap_reset();
    glDrawArraysInstanced(GL_POINTS, 0, 1, 2);
    CAP_IVEC4(64, 3, 4, 5, 1);
    EXPECT_NO_GL_ERROR();
}

static void fetch_one(GLint size, GLenum type, GLboolean norm, GLsizei stride, const void *data, GLsizeiptr bytes,
                      int verts)
{
    static GLuint prog;
    if (!prog) prog = cap_program("layout(location=0) in vec4 a;", "vec4", "a");
    make_vao();
    make_vbo(data, bytes);
    glVertexAttribPointer(0, size, type, norm, stride, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    cap_reset();
    glDrawArrays(GL_POINTS, 0, verts);
}

TEST(float_conversions)
{

    { static const GLubyte d[] = { 255, 0, 51, 255,  0, 255, 0, 0 };
      fetch_one(4, GL_UNSIGNED_BYTE, GL_TRUE, 0, d, sizeof d, 2);
      CAP_VEC4(0, 1, 0, 0.2, 1, 1e-3); CAP_VEC4(1, 0, 1, 0, 0, 1e-3); }

    { static const GLbyte d[] = { -128, 127, 0, 64 };
      fetch_one(4, GL_BYTE, GL_TRUE, 0, d, sizeof d, 1);
      CAP_VEC4(0, -1, 1, 0, 64.0 / 127.0, 1e-3); }

    /* GL_SHORT, size 1, normalized (used to map to UShortNormalized) */
    { static const GLshort d[] = { -32767, 0, 16384, 0 };
      fetch_one(1, GL_SHORT, GL_TRUE, 4, d, sizeof d, 2);
      CAP_VEC4(0, -1, 0, 0, 1, 1e-3); CAP_VEC4(1, 16384.0 / 32767.0, 0, 0, 1, 1e-3); }

    /* GL_SHORT, size 1, unnormalized float input (CPU converted) */
    { static const GLshort d[] = { -300, 0, 5, 0 };
      fetch_one(1, GL_SHORT, GL_FALSE, 4, d, sizeof d, 2);
      CAP_VEC4(0, -300, 0, 0, 1, 0); CAP_VEC4(1, 5, 0, 0, 1, 0); }

    /* unnormalized ubyte -> float */
    { static const GLubyte d[] = { 200, 1, 2, 3 };
      fetch_one(4, GL_UNSIGNED_BYTE, GL_FALSE, 0, d, sizeof d, 1);
      CAP_VEC4(0, 200, 1, 2, 3, 0); }

    /* ubyte3 with a 3 byte stride (not 4 byte aligned -> CPU converted) */
    { static const GLubyte d[] = { 255, 0, 0,  0, 255, 0,  0, 0, 255, 0 };
      fetch_one(3, GL_UNSIGNED_BYTE, GL_TRUE, 3, d, sizeof d, 3);
      CAP_VEC4(0, 1, 0, 0, 1, 1e-3); CAP_VEC4(1, 0, 1, 0, 1, 1e-3); CAP_VEC4(2, 0, 0, 1, 1, 1e-3); }

    /* BGRA */
    { static const GLubyte d[] = { 255, 128, 0, 64 };   /* B G R A */
      fetch_one(GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 0, d, sizeof d, 1);
      CAP_VEC4(0, 0, 128.0 / 255.0, 1, 64.0 / 255.0, 1e-3); }

    /* half float */
    { static const GLushort d[] = { 0x3C00, 0xC000, 0x3800, 0x0000 };  /* 1, -2, 0.5, 0 */
      fetch_one(4, GL_HALF_FLOAT, GL_FALSE, 0, d, sizeof d, 1);
      CAP_VEC4(0, 1, -2, 0.5, 0, 0); }

    /* fixed 16.16 */
    { static const GLint d[] = { 0x18000, -0x10000, 0x4000, 0 };
      fetch_one(3, GL_FIXED, GL_FALSE, 16, d, sizeof d, 1);
      CAP_VEC4(0, 1.5, -1, 0.25, 1, 1e-6); }

    /* doubles via VertexAttribPointer */
    { static const GLdouble d[] = { 1.25, -2.5, 1e10, 0.5 };
      fetch_one(2, GL_DOUBLE, GL_FALSE, 32, d, sizeof d, 1);
      CAP_VEC4(0, 1.25, -2.5, 0, 1, 0); }

    /* 32 bit normalized int (CPU converted) */
    { static const GLint d[] = { INT_MAX, INT_MIN, 0, 0 };
      fetch_one(2, GL_INT, GL_TRUE, 16, d, sizeof d, 1);
      CAP_VEC4(0, 1, -1, 0, 1, 1e-6); }

    /* packed 2_10_10_10 */
    { static const GLuint d[] = { (0x200u) | (0x1FFu << 10) | (0u << 20) | (1u << 30),  1023u | (3u << 30) };
      fetch_one(4, GL_INT_2_10_10_10_REV, GL_TRUE, 0, d, 4, 1);
      CAP_VEC4(0, -1, 1, 0, 1, 1e-3);
      fetch_one(4, GL_INT_2_10_10_10_REV, GL_FALSE, 0, d, 4, 1);
      CAP_VEC4(0, -512, 511, 0, 1, 0);
      fetch_one(4, GL_UNSIGNED_INT_2_10_10_10_REV, GL_TRUE, 0, &d[1], 4, 1);
      CAP_VEC4(0, 1, 0, 0, 1, 1e-3);
      fetch_one(GL_BGRA, GL_UNSIGNED_INT_2_10_10_10_REV, GL_TRUE, 0, &d[1], 4, 1);
      CAP_VEC4(0, 0, 0, 1, 1, 1e-3); }

    /* 10F_11F_11F */
    { static const GLuint d[] = { (15u << 6) | ((16u << 6) << 11) | ((14u << 5) << 22) };
      fetch_one(3, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FALSE, 0, d, 4, 1);
      CAP_VEC4(0, 1, 2, 0.5, 1, 1e-3); }

    EXPECT_NO_GL_ERROR();
}

TEST(long_attribute)
{
    /* Metal has no fp64 vertex inputs: L arrays are delivered to float shader inputs */
    GLuint prog = cap_program("layout(location=0) in vec4 a;", "vec4", "a");
    if (!prog) return;
    static const GLdouble d[] = { 0.5, 1.5, 2.5, 3.5,  -1, -2, -3, -4 };
    make_vao();
    make_vbo(d, sizeof d);
    glVertexAttribLPointer(0, 4, GL_DOUBLE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    cap_reset();
    glDrawArrays(GL_POINTS, 0, 2);
    CAP_VEC4(0, 0.5, 1.5, 2.5, 3.5, 0);
    CAP_VEC4(1, -1, -2, -3, -4, 0);
    EXPECT_NO_GL_ERROR();
}

TEST(separate_format_and_offsets)
{
    GLuint prog = cap_program("layout(location=0) in vec2 a;\nlayout(location=1) in vec2 b;", "vec4", "vec4(a, b)");
    if (!prog) return;
    /* interleaved {a.x a.y b.x b.y} with a 16 byte header the binding offset skips */
    static const float d[] = { 99, 99, 99, 99,   1, 2, 3, 4,   5, 6, 7, 8 };
    make_vao();
    GLuint vbo = make_vbo(d, sizeof d);
    glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 8);
    glVertexAttribBinding(0, 5);
    glVertexAttribBinding(1, 5);
    glBindVertexBuffer(5, vbo, 16, 16);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glUseProgram(prog);
    cap_reset();
    glDrawArrays(GL_POINTS, 0, 2);
    CAP_VEC4(0, 1, 2, 3, 4, 0);
    CAP_VEC4(1, 5, 6, 7, 8, 0);

    /* rebinding with a different offset is picked up */
    glBindVertexBuffer(5, vbo, 32, 16);
    cap_reset();
    glDrawArrays(GL_POINTS, 0, 1);
    CAP_VEC4(0, 5, 6, 7, 8, 0);
    EXPECT_NO_GL_ERROR();
}

TEST(instanced_divisors)
{
    GLuint prog = cap_program("layout(location=0) in float v;\nlayout(location=1) in float i1;\nlayout(location=2) in float i2;",
                              "vec4", "vec4(v, i1, i2, float(gl_InstanceID))");
    if (!prog) return;
    static const float pv[] = { 10, 20 };
    static const float p1[] = { 100, 200, 300, 400, 500 };
    static const float p2[] = { 1000, 2000, 3000 };
    make_vao();
    make_vbo(pv, sizeof pv); glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);
    make_vbo(p1, sizeof p1); glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(1);
    make_vbo(p2, sizeof p2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(2);
    glVertexAttribDivisor(1, 1);
    glVertexAttribDivisor(2, 2);
    glUseProgram(prog);
    cap_reset();
    glDrawArraysInstanced(GL_POINTS, 0, 2, 4);
    for (int inst = 0; inst < 4; inst++)
        for (int vtx = 0; vtx < 2; vtx++)
        {
            CAP_VEC4(inst * 64 + vtx, pv[vtx], p1[inst], p2[inst / 2], inst, 0);
        }

    /* base instance: instanced element = floor(instance / divisor) + baseinstance.  The capture
     * slot is derived from the fetched value so this does not depend on gl_InstanceID. */
    GLuint prog2 = cap_program("layout(location=0) in float v;\nlayout(location=1) in float i1;\nlayout(location=2) in float i2;",
                               "vec4", "vec4(v, i1, i2, 0.0); o.v[int(i1) / 100 + 8] = vec4(v, i1, i2, 1.0)");
    if (!prog2) return;
    glUseProgram(prog2);
    cap_reset();
    glDrawArraysInstancedBaseInstance(GL_POINTS, 0, 1, 2, 1);
    CAP_VEC4(8 + 2, 10, 200, 2000, 1, 0);   /* instance 0 -> p1[1], p2[0 / 2 + 1] */
    CAP_VEC4(8 + 3, 10, 300, 2000, 1, 0);   /* instance 1 -> p1[2], p2[1 / 2 + 1] */
    EXPECT_NO_GL_ERROR();
}

TEST(dsa_draw_elements)
{
    GLuint prog = cap_program("layout(location=3) in vec3 p;", "vec4", "vec4(p, 1.0)");
    if (!prog) return;
    static const float d[] = { 1, 1, 1,  2, 2, 2,  3, 3, 3 };
    static const GLuint idx[] = { 2, 0, 1 };
    GLuint vbo, ebo, vao;
    glCreateBuffers(1, &vbo); glNamedBufferData(vbo, sizeof d, d, GL_STATIC_DRAW);
    glCreateBuffers(1, &ebo); glNamedBufferData(ebo, sizeof idx, idx, GL_STATIC_DRAW);
    glCreateVertexArrays(1, &vao);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 12);
    glVertexArrayAttribFormat(vao, 3, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 3, 0);
    glEnableVertexArrayAttrib(vao, 3);
    glVertexArrayElementBuffer(vao, ebo);
    glBindVertexArray(vao);
    glUseProgram(prog);
    cap_reset();
    glDrawElements(GL_POINTS, 3, GL_UNSIGNED_INT, 0);
    /* gl_VertexID is the index value */
    CAP_VEC4(0, 1, 1, 1, 1, 0);
    CAP_VEC4(1, 2, 2, 2, 1, 0);
    CAP_VEC4(2, 3, 3, 3, 1, 0);
    EXPECT_NO_GL_ERROR();
}

TEST(stride_zero_binding)
{
    GLuint prog = cap_program("layout(location=0) in vec4 a;", "vec4", "a");
    if (!prog) return;
    static const float d[] = { 4, 3, 2, 1,  9, 9, 9, 9 };
    make_vao();
    GLuint vbo = make_vbo(d, sizeof d);
    glVertexAttribFormat(0, 4, GL_FLOAT, GL_FALSE, 0);
    glBindVertexBuffer(0, vbo, 0, 0);   /* every vertex reads element 0 */
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    cap_reset();
    glDrawArrays(GL_POINTS, 0, 3);
    CAP_VEC4(0, 4, 3, 2, 1, 0);
    CAP_VEC4(2, 4, 3, 2, 1, 0);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(current_value_setters);
    RUN(packed_setters);
    RUN(setter_errors);
    RUN(limits);
    RUN(pointer_errors);
    RUN(attrib_queries);
    RUN(dsa_vao);
    RUN(constant_attribute);
    RUN(constant_int_attributes);
    RUN(integer_arrays);
    RUN(float_conversions);
    RUN(long_attribute);
    RUN(separate_format_and_offsets);
    RUN(instanced_divisors);
    RUN(dsa_draw_elements);
    RUN(stride_zero_binding);
    verify_captures();
    return mgl_test_summary();
}
