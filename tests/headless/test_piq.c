/* test_piq — regressions for KHR-GL46.program_interface_query (r57):
 *   1. gl_PerVertex built-in block members as program inputs/outputs of
 *      separable VS/GS/TCS/TES programs (separate-programs-*)
 *   2. fragment built-in array outputs (gl_SampleMask[0]) (output-built-in)
 *   3. transform-feedback placeholders enumerated but not name-addressable
 *      (transform-feedback-built-in)
 *   4. strict uniform array-index names (array-names)
 *   5. a failed link drops active resources (link-failure / relink-failure)
 *   6. per-element REFERENCED_BY_*_SHADER for uniform-block arrays
 *      (uniform-block-types)
 *   7. subroutine uniform arrays as "b[0]" (subroutines-compute)
 *   8. arrays-of-matrices vertex inputs link + draw (input-types / input-layout)
 */
#include "mgl_test.h"
#include <string.h>

/* --- 1. gl_PerVertex built-in blocks ------------------------------------ */

static const char *kVs =
    "#version 430 core\n"
    "layout(location = 0) in vec4 in_vertex;\n"
    "out Color { float r, g, b; vec4 iLikePie; } vs_color;\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "uniform float u; uniform vec4 v;\n"
    "void main() { gl_Position = in_vertex; vs_color.r = u; vs_color.g = 0.0; vs_color.b = 0.0; vs_color.iLikePie = v; }\n";

static const char *kGs =
    "#version 430 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 4) out;\n"
    "out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[]; };\n"
    "in gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[]; } gl_in[];\n"
    "void main() {\n"
    "   gl_Position = vec4(-1, 1, 0, 1);    EmitVertex();\n"
    "   gl_Position = vec4(-1, -1, 0, 1);   EmitVertex();\n"
    "   gl_Position = vec4(1, 1, 0, 1);     EmitVertex();\n"
    "   gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
    "   EndPrimitive();\n"
    "}\n";

static const char *kTcs =
    "#version 430 core\n"
    "layout(vertices = 3) out;\n"
    "layout(location = 0) patch out vec4 data;\n"
    "out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[]; } gl_out[];\n"
    "in Color { float r, g, b; vec4 iLikePie; } vs_color[];\n"
    "void main() {\n"
    "   data = vec4(1);\n"
    "   gl_out[gl_InvocationID].gl_Position = vec4(vs_color[0].r, vs_color[0].g, vs_color[0].b, 1.0);\n"
    "   gl_TessLevelInner[0] = 1.0; gl_TessLevelOuter[0] = 1.0;\n"
    "}\n";

static const char *kTes =
    "#version 430 core\n"
    "layout(triangles, equal_spacing) in;\n"
    "out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[]; };\n"
    "in gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[]; } gl_in[];\n"
    "void main() { gl_Position = gl_in[0].gl_Position; }\n";

static GLuint sep_program(GLenum type, const char *src)
{
    GLuint p = glCreateProgram(), s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    glAttachShader(p, s);
    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(p);
    glDeleteShader(s);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    return ok ? p : 0;
}

TEST(pervertex_blocks)
{
    GLuint vs = sep_program(GL_VERTEX_SHADER, kVs);
    GLuint gs = sep_program(GL_GEOMETRY_SHADER, kGs);
    GLuint tcs = sep_program(GL_TESS_CONTROL_SHADER, kTcs);
    GLuint tes = sep_program(GL_TESS_EVALUATION_SHADER, kTes);
    EXPECT_TRUE(vs && gs && tcs && tes);

    GLint n = 0;
    glGetProgramInterfaceiv(vs, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 5);    /* gl_Position + 4 Color members */
    EXPECT_TRUE(glGetProgramResourceIndex(vs, GL_PROGRAM_OUTPUT, "gl_Position") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(vs, GL_PROGRAM_OUTPUT, "Color.r") != GL_INVALID_INDEX);

    EXPECT_TRUE(glGetProgramResourceIndex(gs, GL_PROGRAM_INPUT, "gl_PerVertex.gl_Position") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(gs, GL_PROGRAM_OUTPUT, "gl_Position") != GL_INVALID_INDEX);

    EXPECT_TRUE(glGetProgramResourceIndex(tcs, GL_PROGRAM_OUTPUT, "gl_PerVertex.gl_Position") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(tcs, GL_PROGRAM_OUTPUT, "data") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(tcs, GL_PROGRAM_INPUT, "Color.iLikePie") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(tcs, GL_PROGRAM_INPUT, "gl_InvocationID") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(tcs, GL_PROGRAM_INPUT, "gl_PerVertex.gl_Position") == GL_INVALID_INDEX); /* gl_in never read */

    EXPECT_TRUE(glGetProgramResourceIndex(tes, GL_PROGRAM_INPUT, "gl_PerVertex.gl_Position") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(tes, GL_PROGRAM_OUTPUT, "gl_Position") != GL_INVALID_INDEX);

    /* CTS property vector for the GS input member */
    GLuint i = glGetProgramResourceIndex(gs, GL_PROGRAM_INPUT, "gl_PerVertex.gl_Position");
    GLenum props[4] = { GL_NAME_LENGTH, GL_TYPE, GL_ARRAY_SIZE, GL_IS_PER_PATCH };
    GLint vals[4] = { -1, -1, -1, -1 };
    GLsizei len = 0;
    glGetProgramResourceiv(gs, GL_PROGRAM_INPUT, i, 4, props, 4, &len, vals);
    EXPECT_EQ_INT(vals[0], 25);
    EXPECT_EQ_INT(vals[1], GL_FLOAT_VEC4);
    EXPECT_EQ_INT(vals[2], 1);
    EXPECT_EQ_INT(vals[3], 0);

    glDeleteProgram(vs); glDeleteProgram(gs); glDeleteProgram(tcs); glDeleteProgram(tes);
}

/* --- 2. fragment built-in array outputs --------------------------------- */

TEST(fs_builtin_array_output)
{
    const char *fs = "#version 430 core\n"
                     "layout(location = 0) out vec4 c;\n"
                     "void main() { c = vec4(1); gl_FragDepth = 0.1; gl_SampleMask[0] = 1; }\n";
    GLuint p = mgl_make_program(kVs, fs);
    EXPECT_TRUE(p != 0);
    GLint n = 0, mnl = 0;
    glGetProgramInterfaceiv(p, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &n);
    glGetProgramInterfaceiv(p, GL_PROGRAM_OUTPUT, GL_MAX_NAME_LENGTH, &mnl);
    EXPECT_EQ_INT(n, 3);        /* c + gl_FragDepth + gl_SampleMask[0] */
    EXPECT_EQ_INT(mnl, 17);     /* "gl_SampleMask[0]\0" */
    EXPECT_TRUE(glGetProgramResourceIndex(p, GL_PROGRAM_OUTPUT, "gl_SampleMask[0]") != GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(p, GL_PROGRAM_OUTPUT, "gl_FragDepth") != GL_INVALID_INDEX);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_OUTPUT, "gl_SampleMask"), -1);
    glDeleteProgram(p);
}

/* --- 3. transform-feedback placeholders --------------------------------- */

static const char *kXfbVs =
    "#version 430 core\n"
    "layout(location = 0) in vec4 p;\n"
    "out vec4 color;\n"
    "out vec4 texcoord;\n"
    "out vec4 normal;\n"
    "void main() {\n"
    "   gl_Position = p; color = vec4(0.5); texcoord = vec4(1); normal = vec4(0.5);\n"
    "}\n";

TEST(xfb_placeholders)
{
    GLuint p = sep_program(GL_VERTEX_SHADER, kXfbVs);
    EXPECT_TRUE(p != 0);
    const char *varyings[6] = { "color", "gl_SkipComponents1", "texcoord", "gl_NextBuffer",
                                "gl_SkipComponents4", "normal" };
    glTransformFeedbackVaryings(p, 6, varyings, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    GLint n = 0, mnl = 0;
    glGetProgramInterfaceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, GL_ACTIVE_RESOURCES, &n);
    glGetProgramInterfaceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, GL_MAX_NAME_LENGTH, &mnl);
    EXPECT_EQ_INT(n, 6);        /* placeholders stay enumerable */
    EXPECT_EQ_INT(mnl, 19);     /* "gl_SkipComponents1\0" */
    /* ...but are not addressable by name */
    EXPECT_TRUE(glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "gl_NextBuffer") == GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "gl_SkipComponents4") == GL_INVALID_INDEX);
    EXPECT_TRUE(glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "color") != GL_INVALID_INDEX);
    /* enumerated placeholder entry reports TYPE=GL_NONE, ARRAY_SIZE = skip count */
    GLuint i = GL_INVALID_INDEX;
    for (int q = 0; q < n; q++) {
        char nm[64]; GLsizei len = 0;
        glGetProgramResourceName(p, GL_TRANSFORM_FEEDBACK_VARYING, q, sizeof nm, &len, nm);
        if (!strcmp(nm, "gl_SkipComponents4")) i = q;
    }
    EXPECT_TRUE(i != GL_INVALID_INDEX);
    if (i != GL_INVALID_INDEX) {
        GLenum props[3] = { GL_NAME_LENGTH, GL_TYPE, GL_ARRAY_SIZE };
        GLint vals[3] = { -1, -1, -1 };
        GLsizei len = 0;
        glGetProgramResourceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, i, 3, props, 3, &len, vals);
        EXPECT_EQ_INT(vals[0], 19);
        EXPECT_EQ_INT(vals[1], GL_NONE);
        EXPECT_EQ_INT(vals[2], 4);
    }
    glDeleteProgram(p);
}

/* --- 4. strict uniform array-index names -------------------------------- */

TEST(strict_uniform_names)
{
    const char *fs = "#version 430 core\n"
                     "layout(location = 0) out vec4 c;\n"
                     "uniform vec4 a[4];\n"
                     "void main() { c = a[0] + a[1] + a[2] + a[3]; }\n";
    GLuint p = mgl_make_program(kVs, fs);
    EXPECT_TRUE(p != 0);
    EXPECT_TRUE(glGetUniformLocation(p, "a") >= 0);
    EXPECT_TRUE(glGetUniformLocation(p, "a[0]") >= 0);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[ 0]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[0 ]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[\n0]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[\t0]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[01]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[00]"), -1);
    EXPECT_EQ_INT(glGetUniformLocation(p, "a[0+0]"), -1);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_UNIFORM, "a[ 0]"), -1);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_UNIFORM, "a[01]"), -1);
    glDeleteProgram(p);
}

/* --- 5. failed link drops active resources ------------------------------ */

TEST(link_failure_clears)
{
    const char *vs = "#version 430 core\n"
                     "layout(location = 0) in vec4 in_vertex;\n"
                     "layout(location = 0) out vec4 vs_color;\n"
                     "void main() { gl_Position = in_vertex; vs_color = in_vertex; }\n";
    const char *fs = "#version 430 core\n"
                     "layout(location = 0) in vec4 vs_color;\n"
                     "layout(location = 0) out vec4 c;\n"
                     "void main() { c = vs_color; }\n";
    GLuint p = mgl_make_program(vs, fs);
    EXPECT_TRUE(p != 0);
    EXPECT_TRUE(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "in_vertex") >= 0);
    /* now make the link fail: capture a varying that is not a VS output */
    const char *varyings[1] = { "q" };
    glTransformFeedbackVaryings(p, 1, varyings, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 1; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "in_vertex"), -1);
    EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_OUTPUT, "vs_color"), -1);
    GLint n = 1;
    glGetProgramInterfaceiv(p, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 0);
    glDeleteProgram(p);
}

/* --- 6. per-element block usage ------------------------------------------ */

TEST(ubo_array_element_usage)
{
    const char *fs = "#version 430 core\n"
                     "layout(location = 0) out vec4 c;\n"
                     "uniform TrickyBlock { mat4 a; float b; vec2 c; } e[2];\n"
                     "void main() { c = e[0].a[0] + vec4(e[0].b) + vec4(e[0].c, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program(kVs, fs);
    EXPECT_TRUE(p != 0);
    GLuint i0 = glGetProgramResourceIndex(p, GL_UNIFORM_BLOCK, "TrickyBlock[0]");
    GLuint i1 = glGetProgramResourceIndex(p, GL_UNIFORM_BLOCK, "TrickyBlock[1]");
    EXPECT_TRUE(i0 != GL_INVALID_INDEX);
    EXPECT_TRUE(i1 != GL_INVALID_INDEX);
    GLenum prop = GL_REFERENCED_BY_FRAGMENT_SHADER;
    GLint v0 = -1, v1 = -1;
    GLsizei len = 0;
    glGetProgramResourceiv(p, GL_UNIFORM_BLOCK, i0, 1, &prop, 1, &len, &v0);
    glGetProgramResourceiv(p, GL_UNIFORM_BLOCK, i1, 1, &prop, 1, &len, &v1);
    EXPECT_EQ_INT(v0, 1);
    EXPECT_EQ_INT(v1, 0);       /* e[1] never referenced */
    glDeleteProgram(p);
}

/* --- 7. subroutine uniform arrays --------------------------------------- */

TEST(subroutine_uniform_array)
{
    const char *cs =
        "#version 430 core\n"
        "layout(local_size_x = 1, local_size_y = 1) in;\n"
        "layout(std430) buffer Output { vec4 data; } g_out;\n"
        "subroutine vec4 a_t();\n"
        "subroutine uniform a_t a;\n"
        "subroutine(a_t) vec4 ax() { return vec4(1); }\n"
        "subroutine(a_t) vec4 ay() { return vec4(0); }\n"
        "subroutine vec4 b_t();\n"
        "subroutine uniform b_t b[3];\n"
        "subroutine(b_t) vec4 bx() { return vec4(1); }\n"
        "subroutine(b_t) vec4 by() { return vec4(0); }\n"
        "subroutine(b_t) vec4 bz() { return vec4(-1); }\n"
        "void main() { g_out.data = a() + b[0]() + b[1]() + b[2](); }\n";
    GLuint p = sep_program(GL_COMPUTE_SHADER, cs);
    EXPECT_TRUE(p != 0);
    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_COMPUTE_SUBROUTINE, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 5);
    glGetProgramInterfaceiv(p, GL_COMPUTE_SUBROUTINE_UNIFORM, GL_ACTIVE_RESOURCES, &n);
    EXPECT_EQ_INT(n, 2);
    GLuint bi = glGetProgramResourceIndex(p, GL_COMPUTE_SUBROUTINE_UNIFORM, "b[0]");
    EXPECT_TRUE(bi != GL_INVALID_INDEX);
    if (bi != GL_INVALID_INDEX) {
        GLenum props[4] = { GL_NAME_LENGTH, GL_ARRAY_SIZE, GL_NUM_COMPATIBLE_SUBROUTINES, GL_LOCATION };
        GLint vals[4] = { -1, -1, -1, -1 };
        GLsizei len = 0;
        glGetProgramResourceiv(p, GL_COMPUTE_SUBROUTINE_UNIFORM, bi, 4, props, 4, &len, vals);
        EXPECT_EQ_INT(vals[0], 5);
        EXPECT_EQ_INT(vals[1], 3);
        EXPECT_EQ_INT(vals[2], 3);
        EXPECT_EQ_INT(vals[3], 1);
    }
    glDeleteProgram(p);
}

/* --- 8. arrays-of-matrices vertex inputs --------------------------------- */

TEST(aom_vertex_input)
{
    const char *vs =
        "#version 430 core\n"
        "in mat4 a;           /* locations 0-3 */\n"
        "in ivec4 b;          /* location 4 */\n"
        "in float c[2];       /* locations 5-6 */\n"
        "in mat2x3 d[2];      /* locations 7-10 */\n"
        "in uvec2 e;          /* location 11 */\n"
        "in uint f;          /* location 12 */\n"
        "in vec3 g[2];        /* locations 13-14 */\n"
        "in int h;            /* location 15 */\n"
        "void main() {\n"
        "   int i = h % 2;\n"
        "   mat2x3 dm = d[0] + d[1] + d[i];\n"
        "   float k = c[0] + c[1] + c[i] + float(h);\n"
        "   gl_Position = vec4(a[0].xyz + dm[0] + dm[1] + g[0] + g[i] + vec3(float(f), e.x, k), 1.0);\n"
        "}\n";
    const char *fs = "#version 430 core\n"
                     "layout(location = 0) out vec4 color;\n"
                     "void main() { color = vec4(1, 0, 0, 1); }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    /* the CTS binds the attribute names to exactly these locations */
    const struct { const char *n; GLuint l; } binds[] = {
        { "a", 0 }, { "b", 4 }, { "c", 5 }, { "d", 7 }, { "e", 11 }, { "f", 12 }, { "g", 13 }, { "h", 15 } };
    for (size_t i = 0; i < sizeof binds / sizeof binds[0]; i++) glBindAttribLocation(p, binds[i].l, binds[i].n);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  link: %s", log); }
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (ok) {
        EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "d[0]"), 7);
        EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "a"), 0);
        EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "c[0]"), 5);
        EXPECT_EQ_INT(glGetProgramResourceLocation(p, GL_PROGRAM_INPUT, "g[1]"), 14);
        /* d occupies locations 7..10 (2 elements x 2 columns) */
        GLuint i = glGetProgramResourceIndex(p, GL_PROGRAM_INPUT, "d[0]");
        GLenum props[3] = { GL_NAME_LENGTH, GL_TYPE, GL_ARRAY_SIZE };
        GLint vals[3] = { -1, -1, -1 };
        GLsizei len = 0;
        glGetProgramResourceiv(p, GL_PROGRAM_INPUT, i, 3, props, 3, &len, vals);
        EXPECT_EQ_INT(vals[0], 5);
        EXPECT_EQ_INT(vals[1], GL_FLOAT_MAT2x3);
        EXPECT_EQ_INT(vals[2], 2);
        /* and it draws */
        glUseProgram(p);
        GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        EXPECT_NO_GL_ERROR();
        GLfloat data[64];
        for (int q = 0; q < 64; q++) data[q] = 0.25f * q;
        GLuint vbo; glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        for (int loc = 0; loc < 16; loc++) {
            glEnableVertexAttribArray(loc);
            if (loc == 4) glVertexAttribIPointer(loc, 4, GL_INT, 0, (void *)(uintptr_t)(8 * loc));
            else if (loc == 11) glVertexAttribIPointer(loc, 2, GL_UNSIGNED_INT, 0, (void *)(uintptr_t)(8 * loc));
            else if (loc == 12) glVertexAttribIPointer(loc, 1, GL_UNSIGNED_INT, 0, (void *)(uintptr_t)(8 * loc));
            else if (loc == 15) glVertexAttribIPointer(loc, 1, GL_INT, 0, (void *)(uintptr_t)(8 * loc));
            else glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)(8 * loc));
        }
        EXPECT_NO_GL_ERROR();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteProgram(p);
}

int main(void)
{
    RUN(pervertex_blocks);
    RUN(fs_builtin_array_output);
    RUN(xfb_placeholders);
    RUN(strict_uniform_names);
    RUN(link_failure_clears);
    RUN(ubo_array_element_usage);
    RUN(subroutine_uniform_array);
    RUN(aom_vertex_input);
    return mgl_test_summary();
}
