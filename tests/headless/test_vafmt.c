/* test_vafmt.c — vertex attribute formats through converted (shadow) streams,
 * regression for the round-40 heap-corruption bug: the vattrib draw plan did not
 * reset dbl[]/shadow_stride[], so after a program with double inputs the next
 * program's int/float shadow slots were converted with mgl_va_fetch_dbl_words
 * (writes 8 bytes per component) into 16-byte elements -> heap overflow, and
 * stale values made XFB results wrong.
 *
 * Mirrors KHR-GL46.direct_state_access.vertex_arrays_attribute_format: two
 * vertex bindings into ONE buffer with offset/stride, results read back via
 * transform feedback.
 */
#include "mgl_test.h"

static const char *kVSHead = "#version 450\n\n";
static const char *kVSBody = "\nvoid main()\n{\n    gl_Position = vec4(1.0);\n    result = a_0 + a_1;"
                             "}\n";
static const char *kFS = "#version 450\n\nout vec4 color;\n\nvoid main()\n{ color = vec4(1.0); }\n";

static const char *decl_float[4] = {
    "in float a_0; in float a_1; out float result;\n",
    "in vec2 a_0;  in vec2 a_1;  out vec2 result;\n",
    "in vec3 a_0;  in vec3 a_1;  out vec3 result;\n",
    "in vec4 a_0;  in vec4 a_1;  out vec4 result;\n",
};
static const char *decl_double[4] = {
    "in double a_0; in double a_1; out double result;\n",
    "in dvec2 a_0;  in dvec2 a_1;  out dvec2 result;\n",
    "in dvec3 a_0;  in dvec3 a_1;  out dvec3 result;\n",
    "in dvec4 a_0;  in dvec4 a_1;  out dvec4 result;\n",
};
static const char *decl_int[4] = {
    "in int a_0;   in int a_1;   out int result;\n",
    "in ivec2 a_0; in ivec2 a_1; out ivec2 result;\n",
    "in ivec3 a_0; in ivec3 a_1; out ivec3 result;\n",
    "in ivec4 a_0; in ivec4 a_1; out ivec4 result;\n",
};

typedef enum { FN_FLOAT = 0, FN_DOUBLE, FN_INT } FnKind;

static void fill_data(GLenum type, int norm, void *out)
{
    for (int i = 0; i < 16; i++)
    {
        switch (type) {
        case GL_FLOAT:         ((GLfloat *)out)[i] = (GLfloat)i; break;
        case GL_DOUBLE:        ((GLdouble *)out)[i] = i; break;
        case GL_BYTE:          ((GLbyte *)out)[i] = (GLbyte)i; break;
        case GL_UNSIGNED_BYTE: ((GLubyte *)out)[i] = (GLubyte)(norm ? (i * 255.0 / 15.0 + 0.5) : i); break;
        case GL_SHORT:         ((GLshort *)out)[i] = (GLshort)(norm ? (i * 32767.0 / 15.0 + 0.5) : i); break;
        case GL_UNSIGNED_SHORT:((GLushort *)out)[i]= (GLushort)(norm ? (i * 65535.0 / 15.0 + 0.5) : i); break;
        case GL_INT:           ((GLint *)out)[i] = (GLint)(norm ? (long long)(i * 2147483647.0 / 15.0) : i); break;
        case GL_UNSIGNED_INT:  ((GLuint *)out)[i] = (GLuint)(norm ? (unsigned long long)(i * 4294967295.0 / 15.0) : i); break;
        default: break;
        }
    }
}

static double elem_d(GLenum type, const void *base, int idx)
{
    const GLubyte *p = (const GLubyte *)base;
    switch (type) {
    case GL_FLOAT:          { GLfloat  v; memcpy(&v, p + idx * 4, 4); return v; }
    case GL_DOUBLE:         { GLdouble v; memcpy(&v, p + idx * 8, 8); return v; }
    case GL_BYTE:           { GLbyte   v; memcpy(&v, p + idx, 1);     return v; }
    case GL_UNSIGNED_BYTE:  { GLubyte  v; memcpy(&v, p + idx, 1);     return v; }
    case GL_SHORT:          { GLshort  v; memcpy(&v, p + idx * 2, 2); return v; }
    case GL_UNSIGNED_SHORT: { GLushort v; memcpy(&v, p + idx * 2, 2); return v; }
    case GL_INT:            { GLint    v; memcpy(&v, p + idx * 4, 4); return v; }
    case GL_UNSIGNED_INT:   { GLuint   v; memcpy(&v, p + idx * 4, 4); return v; }
    default: return 0;
    }
}

static int elem_size(GLenum type)
{
    switch (type) {
    case GL_FLOAT: case GL_INT: case GL_UNSIGNED_INT: return 4;
    case GL_DOUBLE: return 8;
    case GL_BYTE: case GL_UNSIGNED_BYTE: return 1;
    case GL_SHORT: case GL_UNSIGNED_SHORT: return 2;
    default: return 4;
    }
}

static double norm_scale(GLenum type)
{
    switch (type) {
    case GL_UNSIGNED_BYTE:  return 1.0 / 255.0;
    case GL_BYTE:           return 1.0 / 127.0;
    case GL_UNSIGNED_SHORT: return 1.0 / 65535.0;
    case GL_SHORT:          return 1.0 / 32767.0;
    case GL_UNSIGNED_INT:   return 1.0 / 4294967295.0;
    case GL_INT:            return 1.0 / 2147483647.0;
    default: return 1.0;
    }
}

/* one prepare+draw+check: returns 1 when the XFB results match */
static int run_one(int fn, int size, GLenum type, int norm, const char *label)
{
    const char *decl = fn == FN_FLOAT ? decl_float[size - 1] : fn == FN_DOUBLE ? decl_double[size - 1] : decl_int[size - 1];
    char vs[1024];
    snprintf(vs, sizeof vs, "%s%s%s", kVSHead, decl, kVSBody);

    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, (const char *[]){vs}, NULL);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
        MGL_FAIL("compile %s: %s", label, log); glDeleteShader(sh); return 0; }

    GLuint po = glCreateProgram();
    glAttachShader(po, sh);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, (const char *[]){kFS}, NULL);
    glCompileShader(fs);
    glAttachShader(po, fs);
    const char *varying = "result";
    glTransformFeedbackVaryings(po, 1, &varying, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(po);
    glDeleteShader(sh); glDeleteShader(fs);
    glGetProgramiv(po, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(po, sizeof log, NULL, log);
        MGL_FAIL("link %s: %s", label, log); glDeleteProgram(po); return 0; }
    glUseProgram(po);

    GLint loc0 = glGetAttribLocation(po, "a_0"), loc1 = glGetAttribLocation(po, "a_1");

    GLubyte data[16 * 8];
    memset(data, 0, sizeof data);
    fill_data(type, norm, data);
    int tsz = elem_size(type);

    GLuint vao, bo, boxfb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &bo);
    glBindBuffer(GL_ARRAY_BUFFER, bo);
    glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);

    glVertexAttribBinding(loc0, 0);
    glVertexAttribBinding(loc1, 1);
    if (fn == FN_FLOAT)
    {
        glVertexAttribFormat(loc0, size, type, norm, 0);
        glVertexAttribFormat(loc1, size, type, norm, 0);
    }
    else if (fn == FN_DOUBLE)
    {
        glVertexAttribLFormat(loc0, size, type, 0);
        glVertexAttribLFormat(loc1, size, type, 0);
    }
    else
    {
        glVertexAttribIFormat(loc0, size, type, 0);
        glVertexAttribIFormat(loc1, size, type, 0);
    }
    GLsizei stride = (GLsizei)(tsz * size * 2);
    glBindVertexBuffer(0, bo, 0, stride);
    glBindVertexBuffer(1, bo, (GLintptr)(tsz * size), stride);
    glEnableVertexAttribArray(loc0);
    glEnableVertexAttribArray(loc1);

    glGenBuffers(1, &boxfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, boxfb);
    glBufferStorage(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof data, NULL, GL_MAP_READ_BIT);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, boxfb);

    glUseProgram(po);
    glBindVertexArray(vao);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 2);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();

    int good = 1;
    double *res = (double *)glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
    if (!res) { MGL_FAIL("map xfb %s", label); good = 0; }
    else
    {
        double ns = norm_scale(type);
        for (int i = 0; i < 2 && good; i++)
            for (int j = 0; j < size; j++)
            {
                int idx = i * size * 2 + j, idx2 = i * size * 2 + j + size;
                double ref = elem_d(type, data, idx) + elem_d(type, data, idx2);
                if (fn == FN_FLOAT && norm) ref *= ns;

                double got = 0;
                if (fn == FN_DOUBLE) got = res[i * size + j];
                else if (fn == FN_FLOAT) got = ((float *)res)[i * size + j];
                else got = (double)((int *)res)[i * size + j];

                double eps = (fn == FN_INT) ? 0.0 : (norm ? 0.02 : 1e-5);
                if (fabs(got - ref) > eps)
                {
                    MGL_FAIL("%s [%d][%d] got %.9g expected %.9g", label, i, j, got, ref);
                    good = 0;
                    break;
                }
            }
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    }

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glBindVertexArray(0);
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &boxfb);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(po);
    mgl_test_checks++;
    return good;
}

/* double programs followed by converted int/float programs: the stale-dbl
 * overflow scenario (int size 3/4 wrote up to 32 bytes per element) */
TEST(vafmt_double_then_int)
{
    run_one(FN_DOUBLE, 1, GL_DOUBLE, 0, "double size 1");
    run_one(FN_DOUBLE, 2, GL_DOUBLE, 0, "double size 2");
    run_one(FN_INT, 1, GL_BYTE, 0, "int byte size 1");
    run_one(FN_INT, 2, GL_BYTE, 0, "int byte size 2");
    run_one(FN_INT, 3, GL_BYTE, 0, "int byte size 3");
    run_one(FN_INT, 4, GL_BYTE, 0, "int byte size 4");
}

TEST(vafmt_double_then_float)
{
    run_one(FN_DOUBLE, 2, GL_DOUBLE, 0, "double size 2 (2)");
    run_one(FN_FLOAT, 1, GL_BYTE, 0, "float byte size 1");
    run_one(FN_FLOAT, 3, GL_UNSIGNED_BYTE, 1, "float norm ubyte size 3");
    run_one(FN_FLOAT, 4, GL_SHORT, 0, "float short size 4");
}

TEST(vafmt_int_all_types)
{
    GLenum types[] = { GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT };
    const char *tn[] = { "byte", "ubyte", "short", "ushort", "int", "uint" };
    for (int t = 0; t < 6; t++)
        for (int size = 1; size <= 4; size++)
        {
            char label[64]; snprintf(label, sizeof label, "int %s size %d", tn[t], size);
            run_one(FN_INT, size, types[t], 0, label);
        }
}

TEST(vafmt_float_norm)
{
    GLenum types[] = { GL_UNSIGNED_BYTE, GL_SHORT };
    const char *tn[] = { "ubyte", "short" };
    for (int t = 0; t < 2; t++)
        for (int size = 1; size <= 4; size++)
        {
            char label[64]; snprintf(label, sizeof label, "float norm %s size %d", tn[t], size);
            run_one(FN_FLOAT, size, types[t], 1, label);
        }
}

int main(void)
{
    RUN(vafmt_double_then_int);
    RUN(vafmt_double_then_float);
    RUN(vafmt_int_all_types);
    RUN(vafmt_float_norm);
    return mgl_test_summary();
}
