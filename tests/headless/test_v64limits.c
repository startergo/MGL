/* KHR-GL46.vertex_attrib_64bit.limits_test / .vao regression tests.
 *
 * Covers the r25 fixes:
 *  - link: 16 internal double slots at explicit/auto GL locations (the old
 *    "prefer the GL location" allocator fragmented [0,MAX_ATTRIBS) and a full
 *    16-slot set failed the link with "too many vertex attribute components")
 *  - fp64: whole-matrix reads (in dmat2 a; a != dmat2(...)), whole arrays
 *    (in double a[8]) and matrix elements of array-of-matrix attributes
 *    (in dmat2 a[8]; a[i]) — the fp64 slot lowering used to emit
 *    as_type<ulong2>(spvUnsafeArray<uint4,2>) or refuse the element access
 *  - draw: dmatNxM columns are double-word attributes (dbl_slot), matrix
 *    array elements have N GL locations per element, double attributes honor
 *    glVertexAttribDivisor (Metal stepRate), interleaved xfb layout is packed
 *    (no 8-byte alignment between varyings)
 */
#include "mgl_test.h"
#include <stdbool.h>
#include <string.h>

typedef struct { int n_elements, n_rows; const char *tyname; } Elem;
typedef struct { int per_group; int n_types; Elem el[2]; const char *name; } Cfg;

static const Cfg cfgs[] = {
    { 8, 2, { {1,1,"double"}, {2,2,"dvec2"} }, "DOUBLE_DVEC2" },
    { 4, 2, { {3,3,"dvec3"}, {4,4,"dvec4"} }, "DVEC3_DVEC4" },
    { 8, 1, { {4,2,"dmat2"}, {0,0,""} }, "DMAT2" },
    { 2, 2, { {6,2,"dmat3x2"}, {8,2,"dmat4x2"} }, "DMAT3X2_DMAT4X2" },
    { 2, 2, { {6,3,"dmat2x3"}, {8,4,"dmat2x4"} }, "DMAT2X3_DMAT2X4" },
    { 1, 2, { {9,3,"dmat3"}, {12,4,"dmat3x4"} }, "DMAT3_DMAT3X4" },
    { 1, 2, { {12,3,"dmat4x3"}, {16,4,"dmat4"} }, "DMAT4X3_DMAT4" },
};

/* the CTS calculateAttributeLocation (attribute = -1 for arrays) */
static int calc_loc(const Cfg *c, int attribute, int n_type)
{
    int result = 0;
    for (int i = c->n_types - 1; i > n_type; --i)
        result += (c->el[i].n_elements / c->el[i].n_rows) * c->per_group;
    if (attribute >= 0)
        result += (c->el[n_type].n_elements / c->el[n_type].n_rows) * (c->per_group - 1 - attribute);
    return result;
}

/* ---------- link: every configuration must link (2 variants each) -------- */
static void link_cfg(const Cfg *c, bool use_arrays, bool use_locations, bool *ok)
{
    static char src[65536];
    int n = 0;
    n += snprintf(src + n, sizeof src - n, "#version 400\n"
        "#extension GL_ARB_vertex_attrib_64bit : require\n\nprecision highp float;\n\n");
    for (int t = 0; t < c->n_types; t++) {
        const char *tn = c->el[t].tyname;
        if (!use_arrays) {
            for (int a = 0; a < c->per_group; a++) {
                if (use_locations)
                    n += snprintf(src + n, sizeof src - n, "layout(location = %d) ",
                                  calc_loc(c, a, t));
                n += snprintf(src + n, sizeof src - n, "in %s attribute_%d;\n", tn,
                              a + t * c->per_group);
            }
        } else {
            if (use_locations)
                n += snprintf(src + n, sizeof src - n, "layout(location = %d) ",
                              calc_loc(c, -1, t));
            n += snprintf(src + n, sizeof src - n, "in %s attribute_%d[%d];\n", tn, t, c->per_group);
        }
    }
    n += snprintf(src + n, sizeof src - n, "out int vs_output_value;\n\nvoid main()\n{\n");
    for (int t = 0; t < c->n_types; t++) {
        const char *tn = c->el[t].tyname;
        int n_elements = c->el[t].n_elements;
        for (int a = 0; a < c->per_group; a++) {
            int ordinal = a + t * c->per_group;
            n += snprintf(src + n, sizeof src - n, "    %s (attribute_",
                          ordinal == 0 ? "if" : "else if");
            if (!use_arrays) n += snprintf(src + n, sizeof src - n, "%d", ordinal);
            else n += snprintf(src + n, sizeof src - n, "%d[%d]", t, a);
            n += snprintf(src + n, sizeof src - n, " != %s(", tn);
            for (int e = 0; e < n_elements; e++)
                n += snprintf(src + n, sizeof src - n, "%s%d + gl_VertexID * 16 + %d",
                              e ? ", " : "", ordinal, e);
            n += snprintf(src + n, sizeof src - n, "))\n    {\n        vs_output_value = 0;\n    }\n");
        }
    }
    n += snprintf(src + n, sizeof src - n,
                  "    else\n    {\n        vs_output_value = 1;\n    }\n}\n\n");

    const char *fs = "#version 400\nout vec4 c;\nvoid main(){ c = vec4(1.0); }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, (const char *[]){ src }, NULL);
    glCompileShader(v);
    GLint sc = 0; glGetShaderiv(v, GL_COMPILE_STATUS, &sc);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ls = 0; glGetProgramiv(p, GL_LINK_STATUS, &ls);
    if (!sc || !ls) {
        char log[2048] = "";
        if (!sc) glGetShaderInfoLog(v, sizeof log, NULL, log);
        else glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  %s arrays=%d locs=%d: %s", c->name, use_arrays, use_locations, log);
        *ok = false;
    }
    glDeleteShader(v); glDeleteShader(f); glDeleteProgram(p);
}

TEST(link_limits)
{
    bool ok = true;
    for (int i = 0; i < (int)(sizeof cfgs / sizeof cfgs[0]); i++)
        for (int a = 0; a < 2; a++)
            for (int l = 0; l < 2; l++)
                link_cfg(&cfgs[i], a, l, &ok);
    EXPECT_TRUE(ok);   /* all 28 configurations link */
}

/* ---------- draw: all types feed the right doubles -------------------- */
static void draw_cfg(const Cfg *c, bool use_arrays, bool *okp)
{
    int nattrs = c->n_types * c->per_group;
    int vlen = 0;
    int colbase[16], ncols[16], nrows[16], nword[16];
    for (int t = 0; t < c->n_types; t++) {
        int ne = c->el[t].n_elements, nr = c->el[t].n_rows, cols = ne / nr;
        for (int a = 0; a < c->per_group; a++) {
            int o = a + t * c->per_group;
            colbase[o] = vlen; ncols[o] = cols; nrows[o] = nr; nword[o] = ne;
            vlen += ne;
        }
    }
    int nvert = 4;
    double *vbo = malloc(sizeof(double) * vlen * nvert);
    for (int v = 0; v < nvert; v++)
        for (int o = 0; o < nattrs; o++)
            for (int e = 0; e < nword[o]; e++)
                vbo[v * vlen + colbase[o] + e] = o + v * 16 + e;

    char src[32768];
    int n = 0;
    n += snprintf(src + n, sizeof src - n, "#version 400\n"
        "#extension GL_ARB_vertex_attrib_64bit : require\n");
    for (int t = 0; t < c->n_types; t++) {
        if (!use_arrays) {
            for (int a = 0; a < c->per_group; a++)
                n += snprintf(src + n, sizeof src - n, "in %s attribute_%d;\n",
                              c->el[t].tyname, a + t * c->per_group);
        } else {
            n += snprintf(src + n, sizeof src - n, "in %s attribute_%d[%d];\n",
                          c->el[t].tyname, t, c->per_group);
        }
    }
    n += snprintf(src + n, sizeof src - n, "out vec4 col;\nvoid main(){\n int allok=1;\n");
    for (int t = 0; t < c->n_types; t++) {
        const char *tn = c->el[t].tyname;
        int ne = c->el[t].n_elements;
        for (int a = 0; a < c->per_group; a++) {
            int o = a + t * c->per_group;
            n += snprintf(src + n, sizeof src - n, "  if (");
            if (!use_arrays) n += snprintf(src + n, sizeof src - n, "attribute_%d", o);
            else n += snprintf(src + n, sizeof src - n, "attribute_%d[%d]", t, a);
            n += snprintf(src + n, sizeof src - n, " != %s(", tn);
            for (int e = 0; e < ne; e++)
                n += snprintf(src + n, sizeof src - n, "%sdouble(%d + gl_VertexID*16 + %d)",
                              e ? ", " : "", o, e);
            n += snprintf(src + n, sizeof src - n, ")) allok = 0;\n");
        }
    }
    n += snprintf(src + n, sizeof src - n,
                  "  col = (allok==1) ? vec4(0.0,1.0,0.0,1.0) : vec4(1.0,0.0,0.0,1.0);\n"
                  "  gl_Position = vec4(4.0*(gl_VertexID==0?0.0:1.0)-1.0,"
                  " 4.0*(gl_VertexID==2?1.0:0.0)-1.0, 0.0, 1.0);\n}\n");

    const char *fs = "#version 400\nin vec4 col; out vec4 o;\nvoid main(){ o = col; }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, (const char *[]){ src }, NULL);
    glCompileShader(v);
    GLint ok = 0; glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024] = {}; glGetProgramInfoLog(p, 1024, NULL, log);
        printf("  %s arrays=%d LINK FAIL: %s", c->name, use_arrays, log); *okp = false; goto out; }

    glUseProgram(p);
    {
        GLuint vao, buf;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &buf);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, buf);
        glBufferData(GL_ARRAY_BUFFER, sizeof(double) * vlen * nvert, vbo, GL_STATIC_DRAW);
        GLsizei stride = (GLsizei)(sizeof(double) * vlen);
    {
        bool bad = false;
        int base = 0;
        for (int t = 0; t < c->n_types; t++) {
            int cols_of_type = (c->el[t].n_elements / c->el[t].n_rows) * c->per_group;
            for (int a = 0; a < c->per_group && !bad; a++) {
                int o = a + t * c->per_group;
                char nm[64];
                if (!use_arrays) {
                    snprintf(nm, sizeof nm, "attribute_%d", o);
                    /* separate attributes: locations follow the columns used so far */
                    int loc = glGetAttribLocation(p, nm);
                    if (loc != base) {
                        printf("  %s arrays=0: %s at %d, expected %d\n", c->name, nm, loc, base);
                        bad = true; *okp = false;
                    }
                    base += ncols[o];
                } else {
                    snprintf(nm, sizeof nm, "attribute_%d[%d]", t, a);
                    int loc = glGetAttribLocation(p, nm);
                    int want = base + a * ncols[o];
                    if (loc != want) {
                        printf("  %s arrays=1: %s at %d, expected %d\n", c->name, nm, loc, want);
                        bad = true; *okp = false;
                    }
                }
                if (!bad) {
                    int loc = glGetAttribLocation(p, nm);
                    for (int cc = 0; cc < ncols[o]; cc++) {
                        GLintptr offs = (GLintptr)(sizeof(double) * (colbase[o] + cc * nrows[o]));
                        glVertexAttribLPointer(loc + cc, nrows[o], GL_DOUBLE, stride, (void *)offs);
                        glEnableVertexAttribArray(loc + cc);
                    }
                }
            }
            if (use_arrays) base += cols_of_type;
        }
        if (!bad) {
            glViewport(0, 0, 4, 4);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLubyte px[4] = {0, 0, 0, 0};
            glFinish(); glReadPixels(2, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            if (!(px[1] > 200 && px[0] < 50)) {
                printf("  %s arrays=%d: values wrong (px %d,%d,%d)\n",
                       c->name, use_arrays, px[0], px[1], px[2]);
                *okp = false;
            }
        }
        glDeleteBuffers(1, &buf);
        glDeleteVertexArrays(1, &vao);
    }
    }
out:
    glDeleteShader(v); glDeleteProgram(p);
    free(vbo);
}

TEST(draw_limits)
{
    bool ok = true;
    for (int i = 0; i < (int)(sizeof cfgs / sizeof cfgs[0]); i++)
        for (int a = 0; a < 2; a++)
            draw_cfg(&cfgs[i], a, &ok);
    EXPECT_TRUE(ok);   /* all 14 configurations draw the right values */
}

/* ---------- xfb: packed interleaved layout + divisor 2 (the vao test) --- */
TEST(xfb_double_attrs)
{
    const char *vs =
        "#version 400\n#extension GL_ARB_vertex_attrib_64bit : require\n"
        "in dmat3  a;\n"
        "in double d;\n"
        "in dvec2  v;\n"
        "out dmat3  oa;\n"
        "out double od;\n"
        "out dvec2  ov;\n"
        "void main(){ oa = a; od = d; ov = v; }\n";
    const char *xfb[] = { "oa", "od", "ov" };
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glCompileShader(v);
    GLint ok = 0; glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    glAttachShader(p, v);
    glTransformFeedbackVaryings(p, 3, xfb, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);   /* xfb program links */
    if (!ok) return;
    glUseProgram(p);

    int nb = 8, batch = 96, ninst = 4;
    unsigned char *bo = calloc(1, batch * nb);
    for (int b = 0; b < nb; b++) {
        double m[9];
        for (int e = 0; e < 9; e++) m[e] = b * 9 + e + 1;
        memcpy(bo + b * batch, m, sizeof m);
        double d2[2] = { 2.0 * b, 2.0 * b + 1 };
        memcpy(bo + b * batch + 75, d2, sizeof d2);
    }
    GLuint vao, vbo, xb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, batch * nb, bo, GL_STATIC_DRAW);

    GLint la = glGetAttribLocation(p, "a");
    GLint ld = glGetAttribLocation(p, "d");
    GLint lv = glGetAttribLocation(p, "v");
    for (int c = 0; c < 3; c++) {
        glVertexAttribLPointer(la + c, 3, GL_DOUBLE, 96, (void *)(size_t)(24 * c));
        glEnableVertexAttribArray(la + c);
        glVertexAttribDivisor(la + c, 2);
    }
    /* offset 0 / stride 48 / divisor 2 like the CTS double attribute */
    glVertexAttribLPointer(ld, 1, GL_DOUBLE, 48, (void *)0);
    glEnableVertexAttribArray(ld);
    glVertexAttribDivisor(ld, 2);
    glVertexAttribLPointer(lv, 2, GL_DOUBLE, 96, (void *)75);
    glEnableVertexAttribArray(lv);
    glVertexAttribDivisor(lv, 2);

    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 1 << 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

    glBeginTransformFeedback(GL_POINTS);
    glDrawArraysInstanced(GL_POINTS, 0, nb, ninst);
    glEndTransformFeedback();

    glBindBuffer(GL_ARRAY_BUFFER, xb);
    double *out = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (!out) { printf("  xfb map failed 0x%x\n", glGetError()); return; }
    /* packed layout: dmat3 (72) then double (8) then dvec2 (16) = 96/record */
    bool good = true;
    for (int inst = 0; inst < ninst; inst++) {
        int elem = inst / 2;
        double *r = out + inst * nb * 12;
        for (int k = 0; k < 9; k++)
            if (r[k] != (double)(elem * 9 + k + 1)) good = false;
        if (r[9] != *(double *)(bo + 48 * elem)) good = false;
        if (r[10] != 2.0 * elem || r[11] != 2.0 * elem + 1) good = false;
    }
    EXPECT_TRUE(good);  /* packed layout + divisor-2 double attributes */
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &xb); glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    glDeleteShader(v); glDeleteProgram(p);
    free(bo);
}

int main(void)
{
    RUN(link_limits);
    RUN(draw_limits);
    RUN(xfb_double_attrs);
    return 0;
}
