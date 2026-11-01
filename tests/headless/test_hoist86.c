/*
 * test_hoist86.c — nested struct/array samplers in the default uniform block:
 * the GL names ("u.i[1].a[0]") of members hoisted out of MSL-unfriendly structs
 */
#include "mgl_test.h"

static void list_uniforms(GLuint p)
{
    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_UNIFORM, GL_ACTIVE_RESOURCES, &n);
    for (int i = 0; i < n; i++) {
        char nm[256];
        GLsizei w = 0;
        glGetProgramResourceName(p, GL_UNIFORM, i, sizeof nm, &w, nm);
        printf("   uniform[%d] = %.*s\n", i, w, nm);
    }
}

static GLuint build2(const char *vs, const char *fs, GLint *linked)
{
    char log[4096];
    GLint ok = 0;
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint p = glCreateProgram();
    glShaderSource(s1, 1, &vs, NULL); glCompileShader(s1);
    glGetShaderiv(s1, GL_COMPILE_STATUS, &ok);
    if (!ok) { GLsizei n = 0; glGetShaderInfoLog(s1, sizeof log, &n, log); printf("   VS: %.*s\n", n, log); }
    glShaderSource(s2, 1, &fs, NULL); glCompileShader(s2);
    glGetShaderiv(s2, GL_COMPILE_STATUS, &ok);
    if (!ok) { GLsizei n = 0; glGetShaderInfoLog(s2, sizeof log, &n, log); printf("   FS: %.*s\n", n, log); }
    glAttachShader(p, s1); glAttachShader(p, s2); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { GLsizei n = 0; glGetProgramInfoLog(p, sizeof log, &n, log); printf("   link: %.*s\n", n, log); }
    glDeleteShader(s1); glDeleteShader(s2);
    if (linked) { *linked = ok; return p; }
    EXPECT_EQ_INT(ok, GL_TRUE);
    return p;
}

static GLuint build(const char *vs, const char *fs) { return build2(vs, fs, NULL); }

TEST(nested_struct_array_samplers)
{
    const char *vs = "#version 310 es\n"
        "struct Inner { highp sampler2D a[2]; };\n"
        "struct S { Inner i[2]; };\n"
        "uniform S u;\n"
        "uniform vec4 unusedZero;\n"
        "void main() {\n"
        "  gl_Position = vec4(float(textureSize(u.i[0].a[0], 0).x + textureSize(u.i[1].a[1], 0).x)) + unusedZero; }\n";
    const char *fs = "#version 310 es\n"
        "precision highp float;\n"
        "out vec4 c; void main() { c = vec4(1); }\n";
    /* a sampler array inside an array of structs hoists to an array of arrays of
     * textures, which MSL does not support: the link must fail (documented gap) */
    GLint linked = 0;
    GLuint p = build2(vs, fs, &linked);
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    EXPECT_EQ_INT(linked, GL_FALSE);
    if (!linked) { glDeleteProgram(p); return; }
    list_uniforms(p);
    const char *names[4] = { "u.i[0].a[0]", "u.i[0].a[1]", "u.i[1].a[0]", "u.i[1].a[1]" };
    for (int i = 0; i < 4; i++) {
        GLint idx = glGetProgramResourceIndex(p, GL_UNIFORM, names[i]);
        EXPECT_NE(idx, -1);
        GLint loc = glGetUniformLocation(p, names[i]);
        EXPECT_NE(loc, -1);
    }
    glDeleteProgram(p);
}

TEST(struct_array_of_samplers)
{
    const char *vs = "#version 310 es\n"
        "struct S { highp sampler2D a; highp samplerCube b; };\n"
        "uniform S u[3];\n"
        "uniform vec4 unusedZero;\n"
        "void main() {\n"
        "  gl_Position = vec4(float(textureSize(u[0].a, 0).x + int(texture(u[1].b, vec3(0.5)).x))) + unusedZero; }\n";
    const char *fs = "#version 310 es\n"
        "precision highp float;\n"
        "out vec4 c; void main() { c = vec4(1); }\n";
    GLuint p = build(vs, fs);
    list_uniforms(p);
    const char *names[6] = { "u[0].a", "u[0].b", "u[1].a", "u[1].b", "u[2].a", "u[2].b" };
    for (int i = 0; i < 6; i++) {
        EXPECT_NE((int)glGetProgramResourceIndex(p, GL_UNIFORM, names[i]), -1);
        EXPECT_NE(glGetUniformLocation(p, names[i]), -1);
    }
    glDeleteProgram(p);
}

int main(void)
{
    RUN(nested_struct_array_samplers);
    RUN(struct_array_of_samplers);
    return mgl_test_summary();
}
