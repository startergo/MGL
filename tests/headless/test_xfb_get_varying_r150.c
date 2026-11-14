/* r150: replica of KHR-GL46.transform_feedback.get_xfb_varying (GS attached first, interleaved/separate x VS/GS x 15 types). Fences the r149 regression: spv_parse left SpvName.syn uninitialized for REAL OpNames, so a random real varying (result_N) read as synthesized and was blanked from glGetTransformFeedbackVarying. */
#include "mgl_test.h"
#include <string.h>
#include <stdio.h>

static const struct { GLenum type; const char *name; int comps; int isf; } T[] = {
    {GL_FLOAT,"float",1,1},{GL_FLOAT_VEC2,"vec2",2,1},{GL_FLOAT_VEC3,"vec3",3,1},{GL_FLOAT_VEC4,"vec4",4,1},
    {GL_INT,"int",1,0},{GL_INT_VEC2,"ivec2",2,0},{GL_INT_VEC3,"ivec3",3,0},{GL_INT_VEC4,"ivec4",4,0},
    {GL_UNSIGNED_INT,"uint",1,0},{GL_UNSIGNED_INT_VEC2,"uvec2",2,0},{GL_UNSIGNED_INT_VEC3,"uvec3",3,0},
    {GL_UNSIGNED_INT_VEC4,"uvec4",4,0},{GL_FLOAT_MAT2,"mat2",4,1},{GL_FLOAT_MAT3,"mat3",9,1},{GL_FLOAT_MAT4,"mat4",16,1}};
static const GLenum WAYS[2] = {GL_INTERLEAVED_ATTRIBS, GL_SEPARATE_ATTRIBS};
static GLint mvc, mgtoc, mxic, mxsc, mxsa, mvv;

static unsigned nattr(int w, int gs, int k)
{
    unsigned maxtot = (unsigned)((gs ? mgtoc : mvc) - 4);
    unsigned ac = (unsigned)T[k].comps;
    unsigned mx = WAYS[w] == GL_INTERLEAVED_ATTRIBS ? (unsigned)mxic : ac * (unsigned)mxsc;
    if (WAYS[w] == GL_SEPARATE_ATTRIBS && ac > (unsigned)mxsc) return 0;
    unsigned n = mx / ac;
    if (WAYS[w] == GL_SEPARATE_ATTRIBS && n > (unsigned)mxsa) n = (unsigned)mxsa;
    if (n * ac > maxtot) n = maxtot / ac;
    if (ac <= 4) { if (n > (unsigned)mvv) n = (unsigned)mvv; }
    else if (n > (unsigned)mvv / 4) n = (unsigned)mvv / 4;
    return n;
}

static char decl[65536], sets[65536], src[140000];
static const char *fs = "#version 130\n\nout vec4 color;\n\nvoid main()\n{\n    color = vec4(1.0);\n}\n";

int main(void)
{
    GLuint vao = 0; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGetIntegerv(GL_MAX_VARYING_COMPONENTS, &mvc);
    glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS, &mgtoc);
    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS, &mxic);
    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS, &mxsc);
    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS, &mxsa);
    glGetIntegerv(GL_MAX_VARYING_VECTORS, &mvv);
    int bad = 0;
    for (int w = 0; w < 2; w++) for (int gs = 0; gs < 2; gs++) for (int k = 0; k < 15; k++)
    {
        unsigned n = nattr(w, gs, k);
        if (!n) continue;
        static char nm[512][24]; static const char *names[512];
        size_t d = 0, s = 0;
        for (unsigned i = 0; i < n; i++) {
            snprintf(nm[i], sizeof nm[i], "result_%u", i); names[i] = nm[i];
            d += snprintf(decl + d, sizeof decl - d, "out %s %s;\n", T[k].name, nm[i]);
            s += snprintf(sets + s, sizeof sets - s, "    %s = %s(2%s);\n", nm[i], T[k].name, T[k].isf ? ".0" : "");
        }
        const char *vs, *gsrc = NULL; char vsb[] = "#version 130\n\nvoid main()\n{\n}\n";
        if (!gs) { snprintf(src, sizeof src, "#version 130\n\n%s\nvoid main()\n{\n    gl_Position = vec4(1.0);\n%s}\n", decl, sets); vs = src; }
        else { vs = vsb; snprintf(src, sizeof src, "#version 150\n\nlayout(points) in;\nlayout(points, max_vertices = 1) out;\n\n%s\nvoid main()\n{\n    gl_Position = vec4(1.0);\n%s    EmitVertex();\n}\n", decl, sets); gsrc = src; }
        GLuint p = glCreateProgram();
        const char *ss[3] = {gsrc, vs, fs}; GLenum st[3] = {GL_GEOMETRY_SHADER, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
        GLuint ids[3] = {0, 0, 0};
        for (int i = 0; i < 3; i++) { if (!ss[i]) continue; GLuint sh = glCreateShader(st[i]); ids[i] = sh;
            glAttachShader(p, sh);
            glShaderSource(sh, 1, &ss[i], NULL); glCompileShader(sh); GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
            if (!ok) { char l[512] = ""; glGetShaderInfoLog(sh, 512, NULL, l); printf("COMPILE FAIL w%d gs%d %s: %s\n", w, gs, T[k].name, l); } }
        glTransformFeedbackVaryings(p, (GLsizei)n, names, WAYS[w]);
        glLinkProgram(p);
        for (int i = 0; i < 3; i++) if (ids[i]) { glDetachShader(p, ids[i]); glDeleteShader(ids[i]); }
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { char l[512] = ""; glGetProgramInfoLog(p, 512, NULL, l); printf("LINK FAIL w%d gs%d %s n=%u: %s\n", w, gs, T[k].name, n, l); bad++; glDeleteProgram(p); continue; }
        int casebad = 0; unsigned maxlen = 0;
        for (unsigned i = 0; i < n && !casebad; i++) {
            GLsizei len = 0, size = 0; GLenum ty = 0; char name[18] = {0};
            glGetTransformFeedbackVarying(p, i, 18, &len, &size, &ty, name);
            GLenum e = glGetError();
            if ((unsigned)len > maxlen) maxlen = (unsigned)len;
            if (e || !len || strcmp(name, nm[i]) || size != 1 || ty != T[k].type) {
                printf("BAD w%d gs%d %s n=%u i=%u: err=0x%x len=%d name='%s' size=%d type=0x%x (want 0x%x)\n",
                       w, gs, T[k].name, n, i, e, len, name, size, ty, T[k].type); casebad = 1; }
        }
        if (casebad && k == 0) { printf("  all:"); for (unsigned i = 0; i < n; i++) { GLsizei len = 0, size = 0; GLenum ty = 0; char name[64] = {0};
            glGetTransformFeedbackVarying(p, i, 64, &len, &size, &ty, name); printf(" [%u]'%s'", i, name); } printf("\n"); }
        GLint xv = 0, xm = 0, xl = 0;
        glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &xv);
        glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &xm);
        glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &xl);
        if (!casebad && ((unsigned)xv != n || (GLenum)xm != WAYS[w] || (unsigned)xl < maxlen)) {
            printf("BADQ w%d gs%d %s n=%u: varyings=%d mode=0x%x maxlen=%d (want >=%u)\n", w, gs, T[k].name, n, xv, xm, xl, maxlen); casebad = 1; }
        if (casebad) bad++;
        glDeleteProgram(p);
    }
    EXPECT_EQ_INT(bad, 0);
    return mgl_test_summary();
}
