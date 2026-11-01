/*
 * test_introspect86.c — introspection round 86: names GL must report for
 * default-block uniforms whose struct members were hoisted (samplers cannot
 * live in an MSL buffer-backed struct), SSBO member enumeration for nested
 * structs/arrays (GL 4.6 §7.3.1), and the geometry capture variant's plain
 * uniform stripping with a leading precision qualifier.
 */
#include "mgl_test.h"

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof log, &n, log);
        printf("   shader %u compile failed: %.*s\n", s, n, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    return s;
}

static void link(GLuint p)
{
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof log, &n, log);
        printf("   link failed: %.*s\n", n, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
}

/* GL name of a hoisted sampler: struct { sampler2D s; } u; -> "u.s" */
TEST(hoisted_sampler_name)
{
    const char *vs = "#version 450 core\n"
        "struct S { sampler2D s; };\n"
        "uniform S u;\n"
        "uniform vec4 unusedZero;\n"
        "void main() { gl_Position = vec4(textureSize(u.s, 0).x) + unusedZero; }\n";
    const char *fs = "#version 450 core\n"
        "out vec4 c; void main() { c = vec4(1); }\n";
    GLuint p = glCreateProgram(), v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f); link(p);

    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_UNIFORM, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 2);
    GLint idx = glGetProgramResourceIndex(p, GL_UNIFORM, "u.s");
    EXPECT_NE(idx, -1);
    if (idx != 0 && idx != 1) { MGL_FAIL("bad index for u.s: %d\n", idx); }
    GLenum locp = GL_LOCATION;
    GLint loc = -1;
    glGetProgramResourceiv(p, GL_UNIFORM, idx, 1, &locp, 1, NULL, &loc);
    EXPECT_EQ_INT(glGetUniformLocation(p, "u.s"), loc);
    GLenum tp = GL_TYPE;
    GLint ty = 0;
    glGetProgramResourceiv(p, GL_UNIFORM, idx, 1, &tp, 1, NULL, &ty);
    EXPECT_EQ_INT(ty, GL_SAMPLER_2D);
    glDeleteShader(v); glDeleteShader(f); glDeleteProgram(p);
}

/* "S u[3];" with a sampler member: three resources "u[0].s".."u[2].s" */
TEST(hoisted_sampler_array_of_struct)
{
    const char *vs = "#version 450 core\n"
        "struct S { sampler2D s; };\n"
        "uniform S u[3];\n"
        "uniform vec4 unusedZero;\n"
        "void main() { gl_Position = vec4(textureSize(u[0].s, 0).x) + unusedZero\n"
        "                              + textureSize(u[1].s, 0).x + textureSize(u[2].s, 0).x; }\n";
    const char *fs = "#version 450 core\n"
        "out vec4 c; void main() { c = vec4(1); }\n";
    GLuint p = glCreateProgram(), v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f); link(p);

    for (int i = 0; i < 3; i++) {
        char nm[32];
        snprintf(nm, sizeof nm, "u[%d].s", i);
        GLint idx = glGetProgramResourceIndex(p, GL_UNIFORM, nm);
        GLenum props[2] = { GL_ARRAY_SIZE, GL_TYPE };
        GLint vals[2] = { 0, 0 };
        if (idx >= 0) glGetProgramResourceiv(p, GL_UNIFORM, idx, 2, props, 2, NULL, vals);
        EXPECT_EQ_INT(vals[0], 1);
        EXPECT_EQ_INT(vals[1], GL_SAMPLER_2D);
        GLint loc = glGetUniformLocation(p, nm);
        EXPECT_NE(loc, -1);
        /* distinct consecutive locations */
        if (i && loc != glGetUniformLocation(p, "u[0].s") + i)
            MGL_FAIL("u[%d].s location %d not consecutive\n", i, loc);
    }
    glDeleteShader(v); glDeleteShader(f); glDeleteProgram(p);
}

/* a sampler array inside the struct: one resource "u.s[0]" of size 3 */
TEST(hoisted_sampler_member_array)
{
    const char *vs = "#version 450 core\n"
        "struct S { sampler2D s[3]; };\n"
        "uniform S u;\n"
        "uniform vec4 unusedZero;\n"
        "void main() { gl_Position = vec4(textureSize(u.s[2], 0).x) + unusedZero; }\n";
    const char *fs = "#version 450 core\n"
        "out vec4 c; void main() { c = vec4(1); }\n";
    GLuint p = glCreateProgram(), v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f); link(p);

    GLint idx = glGetProgramResourceIndex(p, GL_UNIFORM, "u.s[0]");
    EXPECT_NE(idx, -1);
    GLenum props[2] = { GL_ARRAY_SIZE, GL_TYPE };
    GLint vals[2] = { 0, 0 };
    if (idx >= 0) glGetProgramResourceiv(p, GL_UNIFORM, idx, 2, props, 2, NULL, vals);
    EXPECT_EQ_INT(vals[0], 3);
    EXPECT_EQ_INT(vals[1], GL_SAMPLER_2D);
    EXPECT_NE(glGetUniformLocation(p, "u.s[2]"), -1);
    glDeleteShader(v); glDeleteShader(f); glDeleteProgram(p);
}

/* SSBO members: only the block member's OWN array collapses to name[0]; arrays
 * inside a struct enumerate every element (GL 4.6 §7.3.1) */
TEST(ssbo_nested_array_enum)
{
    const char *cs = "#version 450 core\n"
        "layout(local_size_x = 1) in;\n"
        "struct Inner { vec4 t; };\n"
        "struct Outer { Inner arr[3]; };\n"
        "writeonly buffer B { Outer m; } b;\n"
        "void main() { b.m.arr[1].t = vec4(1); b.m.arr[2].t = vec4(2); }\n";
    GLuint p = glCreateProgram(), s = compile(GL_COMPUTE_SHADER, cs);
    glAttachShader(p, s); link(p);

    const char *names[3] = { "B.m.arr[0].t", "B.m.arr[1].t", "B.m.arr[2].t" };
    for (int i = 0; i < 3; i++)
        EXPECT_NE((int)glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, names[i]), -1);
    /* the block member itself is an array -> collapses to name[0] */
    const char *cs2 = "#version 450 core\n"
        "layout(local_size_x = 1) in;\n"
        "writeonly buffer B { vec4 v[4]; } b;\n"
        "void main() { b.v[3] = vec4(1); }\n";
    GLuint p2 = glCreateProgram(), s2 = compile(GL_COMPUTE_SHADER, cs2);
    glAttachShader(p2, s2); link(p2);
    EXPECT_NE((int)glGetProgramResourceIndex(p2, GL_BUFFER_VARIABLE, "B.v[0]"), -1);
    GLint n = 0;
    glGetProgramInterfaceiv(p2, GL_BUFFER_VARIABLE, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 1);
    glDeleteShader(s); glDeleteProgram(p); glDeleteShader(s2); glDeleteProgram(p2);
}

/* a top-level array of structs in an SSBO keeps one resource per the CTS rule */
TEST(ssbo_toplevel_array_of_struct)
{
    const char *cs = "#version 450 core\n"
        "layout(local_size_x = 1) in;\n"
        "struct S { vec4 t; };\n"
        "writeonly buffer B { S arr[2]; } b;\n"
        "void main() { b.arr[1].t = vec4(1); }\n";
    GLuint p = glCreateProgram(), s = compile(GL_COMPUTE_SHADER, cs);
    glAttachShader(p, s); link(p);
    EXPECT_NE((int)glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, "B.arr[0].t"), -1);
    GLenum props[3] = { GL_TOP_LEVEL_ARRAY_SIZE, GL_TOP_LEVEL_ARRAY_STRIDE, GL_TYPE };
    GLint vals[3] = { 0, 0, 0 };
    GLuint idx = glGetProgramResourceIndex(p, GL_BUFFER_VARIABLE, "B.arr[0].t");
    if (idx != GL_INVALID_INDEX) {
        GLsizei len = 0;
        glGetProgramResourceiv(p, GL_BUFFER_VARIABLE, idx, 3, props, 3, &len, vals);
        EXPECT_EQ_INT(vals[0], 2);          /* TOP_LEVEL_ARRAY_SIZE */
        EXPECT_EQ_INT(vals[1], 16);         /* TOP_LEVEL_ARRAY_STRIDE */
        EXPECT_EQ_INT(vals[2], GL_FLOAT_VEC4);
    } else MGL_FAIL("b.arr[0].t not found\n");
    glDeleteShader(s); glDeleteProgram(p);
}

/* the geometry capture variant strips plain uniforms even when a precision
 * qualifier precedes "uniform" (dEQP's unusedZero spelling) */
TEST(geom_highp_uniform_link)
{
    const char *vs = "#version 450 core\n"
        "uniform TargetInterface { bvec3 target; } targetInstance;\n"
        "layout(location = 0) in int noOpt;\n"
        "highp uniform vec4 unusedZero;\n"
        "void main() { gl_Position = noOpt == 0 ? unusedZero : vec4(targetInstance.target.xyxy); }\n";
    const char *gs = "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "void main() { for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); } EndPrimitive(); }\n";
    const char *fs = "#version 450 core\n"
        "out vec4 c; void main() { c = vec4(1); }\n";
    GLuint p = glCreateProgram();
    GLuint v = compile(GL_VERTEX_SHADER, vs), g = compile(GL_GEOMETRY_SHADER, gs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, g); glAttachShader(p, f); link(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) printf("   (link failed: see above)\n");
    glDeleteShader(v); glDeleteShader(g); glDeleteShader(f); glDeleteProgram(p);
}

int main(void)
{
    RUN(hoisted_sampler_name);
    RUN(hoisted_sampler_array_of_struct);
    RUN(hoisted_sampler_member_array);
    RUN(ssbo_nested_array_enum);
    RUN(ssbo_toplevel_array_of_struct);
    RUN(geom_highp_uniform_link);
    return mgl_test_summary();
}
