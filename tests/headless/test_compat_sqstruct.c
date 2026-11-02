/*
 * test_compat_sqstruct.c — GLSL 4.60 §4.1.10 / 4.4: plain structure members
 * accept no qualifiers at all in desktop GLSL except precision (which has no
 * effect there); storage/interpolation/layout/packing qualifiers on struct
 * members must be a compile error (KHR-GL46.shaders.negative.
 * non_precision_qualifiers_in_struct_members).  ES keeps member precision.
 */
#include "mgl_test.h"
#include "gl_compat.h"

static int compiles(const char *src)
{
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    glDeleteShader(s);
    return ok;
}

static const char *vsrc(const char *qual)
{
    static char buf[512];
    snprintf(buf, sizeof buf,
             "#version 460 core\n"
             "struct S { %s float x; };\n"
             "out float ov;\n"
             "void main() { S s; s.x = 1.0; ov = s.x; gl_Position = vec4(0.0); }\n",
             qual);
    return buf;
}

static int compiles2(const char *src)
{
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    glDeleteShader(s);
    return ok;
}

/* the CTS sources put the member on its own line: the struct header and body
 * span several lines */
static const char *vsrc_ml(const char *qual)
{
    static char buf[512];
    snprintf(buf, sizeof buf,
             "#version 460 core\n"
             "struct S\n"
             "{\n"
             "    %s float x;\n"
             "};\n"
             "out float ov;\n"
             "void main()\n"
             "{\n"
             "    S s;\n"
             "    s.x = 1.0;\n"
             "    ov = s.x;\n"
             "    gl_Position = vec4(0.0);\n"
             "}\n",
             qual);
    return buf;
}

/* word 'layout'/'precise' inside the struct body but NOT as a qualifier on a
 * member (comment / nested member name) must not be rejected */
static const char *vsrc_benign(const char *snippet)
{
    static char buf[512];
    snprintf(buf, sizeof buf,
             "#version 460 core\n"
             "struct S\n{\n%s\n};\n"
             "out float ov;\n"
             "void main() { S s; s.x = 1.0; ov = s.x; gl_Position = vec4(0.0); }\n",
             snippet);
    return buf;
}

TEST(struct_member_qualifiers_rejected)
{
    static const char *bad[] = {
        "const", "in", "out", "uniform", "flat", "smooth", "noperspective",
        "centroid", "patch", "sample", "invariant", "shared", "packed",
        "std140", "std430", "coherent", "volatile", "restrict", "readonly",
        "writeonly", "layout(shared)", "layout(location=1)",
        "layout(offset=4)", "layout(binding=1)", "layout(column_major)",
        "precise", "highp layout(shared)", "layout(shared) precise",
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++)
    {
        mgl_test_checks++;
        if (compiles(vsrc(bad[i])))
            MGL_FAIL("struct member qualifier '%s' compiled (must be rejected)", bad[i]);
        mgl_test_checks++;
        if (compiles2(vsrc_ml(bad[i])))
            MGL_FAIL("multiline struct member qualifier '%s' compiled (must be rejected)", bad[i]);
    }
}

TEST(struct_member_benign_words)
{
    /* words that look like qualifiers but are not member qualifiers */
    static const char *ok[] = {
        "float x;  // layout(shared) precise mentioned in a comment",
        "/* layout(std140) */ float x;",
        "float layoutish; float precisex;",
        "float x;\n    float precise_not_a_qualifier_is_ident(constant in C is fine) = 0;",  /* placeholder */
    };
    static const char *ok2[] = {
        "float x;  // layout(shared) precise mentioned in a comment",
        "/* layout(std140) */ float x;",
        "float layoutish; float precisex;",
    };
    (void)ok;
    for (size_t i = 0; i < sizeof ok2 / sizeof ok2[0]; i++)
    {
        mgl_test_checks++;
        if (!compiles2(vsrc_benign(ok2[i])))
            MGL_FAIL("benign struct body '%s' rejected", ok2[i]);
    }
    /* a nested struct inside the body still only rejects member qualifiers */
    static const char bad_nested[] =
        "struct Inner { layout(shared) float y; };\n"
        "float x;";
    mgl_test_checks++;
    if (compiles2(vsrc_benign(bad_nested)))
        MGL_FAIL("nested struct member layout qualifier compiled");
}

TEST(struct_member_precision_allowed_desktop)
{
    /* GLSL 4.60 4.7: precision qualifiers are accepted (no effect) on desktop,
     * including on struct members — matching Mesa's "only precision qualifiers
     * may be applied to structure members" diagnostic. */
    const char *q[] = { "highp", "mediump", "lowp", "" };
    for (size_t i = 0; i < sizeof q / sizeof q[0]; i++)
    {
        mgl_test_checks++;
        if (!compiles(vsrc(q[i])))
            MGL_FAIL("struct member precision '%s' rejected on desktop GLSL", q[i]);
    }
}

TEST(struct_member_qualifiers_es)
{
    /* ES: precision on members legal, storage qualifiers still illegal */
    static const char ok_es[] =
        "#version 320 es\n"
        "struct S { highp float x; };\n"
        "void main() { S s; s.x = 1.0; gl_Position = vec4(s.x); }\n";
    static const char bad_es[] =
        "#version 320 es\n"
        "struct S { uniform float x; };\n"
        "void main() { S s; s.x = 1.0; gl_Position = vec4(s.x); }\n";
    mgl_test_checks++;
    if (!compiles(ok_es)) MGL_FAIL("ES struct member precision rejected");
    mgl_test_checks++;
    if (compiles(bad_es)) MGL_FAIL("ES struct member uniform accepted");
}

int main(void)
{
    RUN(struct_member_qualifiers_rejected);
    RUN(struct_member_precision_allowed_desktop);
    RUN(struct_member_qualifiers_es);
    return mgl_test_summary();
}
