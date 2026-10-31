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

/* --- 9..15: GL 4.6 §7.3.1 struct members of interface blocks (r85) -----
 * dEQP-GL45-ES31.program_interface_query program_input/program_output
 * interface_blocks.*.struct crashed the CTS: struct members were reported as
 * one resource of GL_TYPE GL_NONE and leaf names ("Block.member.field") were
 * missing. */

static int piq_dump(GLuint prog, GLenum iface, char names[][64], GLenum *types, GLint *sizes, GLint *locs, int cap)
{
    GLint n = 0;
    glGetProgramInterfaceiv(prog, iface, GL_ACTIVE_RESOURCES, &n);
    for (int i = 0; i < n && i < cap; i++) {
        GLsizei len = 0;
        glGetProgramResourceName(prog, iface, i, 64, &len, names[i]);
        GLenum props[3] = { GL_TYPE, GL_ARRAY_SIZE, GL_LOCATION };
        GLint vals[3]; GLsizei pl = 0;
        glGetProgramResourceiv(prog, iface, i, 3, props, 3, &pl, vals);
        if (types) types[i] = (GLenum)vals[0];
        if (sizes) sizes[i] = vals[1];
        if (locs) locs[i] = vals[2];
    }
    return n;
}

static GLuint piq_link(GLuint prog, const char *label)
{
    glLinkProgram(prog);
    GLint st = 0; glGetProgramiv(prog, GL_LINK_STATUS, &st);
    if (!st) {
        GLchar log[4096]; glGetProgramInfoLog(prog, sizeof log, NULL, log);
        printf("  %s link failed: %s\n", label, log);
    }
    return prog;
}

TEST(struct_members_of_named_blocks)
{
    /* FS-only program (like the CTS io-block cases): struct member of a named
     * input block is enumerated per leaf with the leaf type */
    const char *fs =
        "#version 450\n"
        "struct StructType0 { highp float target; };\n"
        "in TargetInterface { flat StructType0 target; } targetInstance;\n"
        "void main() { gl_FragDepth = targetInstance.target.target; }\n";
    const char *vs =
        "#version 450\n"
        "struct StructType0 { highp float target; };\n"
        "out TargetInterface { flat StructType0 target; } targetInstance;\n"
        "void main() { targetInstance.target.target = 1.0; gl_Position = vec4(0,0,0,1); }\n";
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER), v = glCreateShader(GL_VERTEX_SHADER);
    GLuint pf = glCreateProgram(), pv = glCreateProgram();
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f); glAttachShader(pf, f);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v); glAttachShader(pv, v);
    piq_link(pf, "fs"); piq_link(pv, "vs");
    char names[8][64]; GLenum types[8]; GLint sizes[8], locs[8];
    int n = piq_dump(pf, GL_PROGRAM_INPUT, names, types, sizes, locs, 8);
    EXPECT_EQ_INT(n, 1);
    EXPECT_STREQ(names[0], "TargetInterface.target.target");
    EXPECT_EQ_INT(types[0], GL_FLOAT);
    EXPECT_EQ_INT(sizes[0], 1);
    EXPECT_EQ_INT(locs[0], -1);           /* no layout(location): §11.1.1 */
    EXPECT_TRUE(glGetProgramResourceIndex(pf, GL_PROGRAM_INPUT, "TargetInterface.target.target") == 0);
    n = piq_dump(pv, GL_PROGRAM_OUTPUT, names, types, sizes, locs, 8);
    EXPECT_EQ_INT(n, 2);                  /* block member leaf + gl_Position */
    EXPECT_STREQ(names[0], "TargetInterface.target.target");
    EXPECT_EQ_INT(types[0], GL_FLOAT);
    EXPECT_EQ_INT(locs[0], -1);           /* VS output, not fragment output */
    glDeleteProgram(pf); glDeleteProgram(pv); glDeleteShader(f); glDeleteShader(v);
}

TEST(anonymous_block_names)
{
    /* a block without an instance name does NOT prefix its members */
    const char *fs =
        "#version 450\n"
        "struct S { vec4 a; float b; };\n"
        "layout(location = 0) in Blk { S s; vec4 v; };\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = s.a + v + vec4(s.b); }\n";
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint pf = glCreateProgram(); glAttachShader(pf, f); piq_link(pf, "fs");
    char names[8][64]; GLint locs[8];
    int n = piq_dump(pf, GL_PROGRAM_INPUT, names, NULL, NULL, locs, 8);
    EXPECT_EQ_INT(n, 3);
    EXPECT_STREQ(names[0], "s.a");  EXPECT_EQ_INT(locs[0], 0);
    EXPECT_STREQ(names[1], "s.b");  EXPECT_EQ_INT(locs[1], 1);
    EXPECT_STREQ(names[2], "v");    EXPECT_EQ_INT(locs[2], 2);
    glDeleteProgram(pf); glDeleteShader(f);
}

TEST(nested_structs_and_arrays)
{
    /* nested structs: dotted leaves; arrays of basic types: one resource
     * "m[0]" with ARRAY_SIZE = count; arrays of aggregates: per element */
    const char *fs =
        "#version 450\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "struct Outer { Inner i; vec4 arr[2]; };\n"
        "layout(location = 3) in Blk { Outer o; Inner inst[2]; } blk;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = blk.o.i.a + vec4(blk.o.i.b.x) + blk.o.arr[1] + blk.inst[1].b.xxyy; }\n";
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint pf = glCreateProgram(); glAttachShader(pf, f); piq_link(pf, "fs");
    char names[12][64]; GLenum types[12]; GLint sizes[12], locs[12];
    int n = piq_dump(pf, GL_PROGRAM_INPUT, names, types, sizes, locs, 12);
    EXPECT_EQ_INT(n, 7);
    EXPECT_STREQ(names[0], "Blk.o.i.a");     EXPECT_EQ_INT(types[0], GL_FLOAT_VEC4); EXPECT_EQ_INT(locs[0], 3);
    EXPECT_STREQ(names[1], "Blk.o.i.b");     EXPECT_EQ_INT(types[1], GL_FLOAT_VEC2); EXPECT_EQ_INT(locs[1], 4);
    EXPECT_STREQ(names[2], "Blk.o.arr[0]");  EXPECT_EQ_INT(types[2], GL_FLOAT_VEC4); EXPECT_EQ_INT(sizes[2], 2);
    EXPECT_EQ_INT(locs[2], 5);
    EXPECT_STREQ(names[3], "Blk.inst[0].a"); EXPECT_EQ_INT(types[3], GL_FLOAT_VEC4); EXPECT_EQ_INT(locs[3], 7);
    EXPECT_STREQ(names[4], "Blk.inst[0].b"); EXPECT_EQ_INT(types[4], GL_FLOAT_VEC2); EXPECT_EQ_INT(locs[4], 8);
    EXPECT_STREQ(names[5], "Blk.inst[1].a"); EXPECT_EQ_INT(types[5], GL_FLOAT_VEC4); EXPECT_EQ_INT(locs[5], 9);
    EXPECT_STREQ(names[6], "Blk.inst[1].b"); EXPECT_EQ_INT(types[6], GL_FLOAT_VEC2); EXPECT_EQ_INT(locs[6], 10);
    glDeleteProgram(pf); glDeleteShader(f);
}

TEST(non_vs_inputs_location_minus_one)
{
    /* §11.1.1: inputs of stages other than the VS and outputs of stages other
     * than the FS report -1 unless declared with layout(location) */
    const char *fs0 =
        "#version 450\n"
        "in TargetInterface { flat highp vec4 target; } targetInstance;\n"
        "in highp vec4 plain;\n"
        "void main() { gl_FragDepth = targetInstance.target.x + plain.y; }\n";
    const char *fs1 =
        "#version 450\n"
        "layout(location = 5) in TargetInterface { flat highp vec4 target; } targetInstance;\n"
        "layout(location = 9) in highp vec4 plain;\n"
        "void main() { gl_FragDepth = targetInstance.target.x + plain.y; }\n";
    const char *vs0 =
        "#version 450\n"
        "out highp vec4 target;\n"
        "void main() { target = vec4(1); gl_Position = vec4(0,0,0,1); }\n";
    const struct { const char *src; int expl; int vs; } cases[] = { { fs0, 0, 0 }, { fs1, 1, 0 }, { vs0, 0, 1 } };
    for (int k = 0; k < 3; k++) {
        GLenum st = cases[k].vs ? GL_VERTEX_SHADER : GL_FRAGMENT_SHADER;
        GLuint s = glCreateShader(st);
        glShaderSource(s, 1, &cases[k].src, NULL); glCompileShader(s);
        GLuint p = glCreateProgram(); glAttachShader(p, s); piq_link(p, "sep");
        char names[8][64]; GLint locs[8];
        GLenum iface = (st == GL_VERTEX_SHADER) ? GL_PROGRAM_OUTPUT : GL_PROGRAM_INPUT;
        int n = piq_dump(p, iface, names, NULL, NULL, locs, 8);
        int want = cases[k].expl;
        for (int i = 0; i < n; i++) {
            if (!strcmp(names[i], "gl_FragDepth") || !strcmp(names[i], "gl_Position")) continue;
            printf("  %s loc=%d (explicit=%d)\n", names[i], locs[i], want);
            EXPECT_EQ_INT(locs[i] >= 0, want ? 1 : 0);
            if (want) EXPECT_EQ_INT(locs[i], (i == 0) ? 5 : 9);
        }
        glDeleteProgram(p); glDeleteShader(s);
    }
}

TEST(patch_block_is_per_patch)
{
    /* glslang marks `patch` blocks on their members, not the variable */
    const char *tes =
        "#version 450\n"
        "layout(triangles) in;\n"
        "patch in TargetInterface { highp vec4 target; } targetInstance;\n"
        "patch in highp vec4 patchPlain;\n"
        "void main() { gl_Position = targetInstance.target + patchPlain; }\n";
    GLuint s = glCreateShader(GL_TESS_EVALUATION_SHADER);
    glShaderSource(s, 1, &tes, NULL); glCompileShader(s);
    GLuint p = glCreateShaderProgramv(GL_TESS_EVALUATION_SHADER, 1, &tes);
    glDeleteShader(s);
    GLint st = 0; glGetProgramiv(p, GL_LINK_STATUS, &st);
    EXPECT_EQ_INT(st, GL_TRUE);
    char names[8][64]; GLint locs[8];
    int n = piq_dump(p, GL_PROGRAM_INPUT, names, NULL, NULL, locs, 8);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strcmp(names[i], "TargetInterface.target") && strcmp(names[i], "patchPlain")) continue;
        GLenum pp = GL_IS_PER_PATCH; GLint v = 0; GLsizei l = 0;
        glGetProgramResourceiv(p, GL_PROGRAM_INPUT, i, 1, &pp, 1, &l, &v);
        printf("  %s per_patch=%d loc=%d\n", names[i], v, locs[i]);
        EXPECT_EQ_INT(v, 1);
        EXPECT_EQ_INT(locs[i], -1);
        found++;
    }
    EXPECT_EQ_INT(found, 2);
    glDeleteProgram(p);
}

TEST(arrayed_tcs_io_name_and_size)
{
    /* TCS arrayed io stays "name[0]" with ARRAY_SIZE = max patch vertices */
    const char *tcs =
        "#version 450\n"
        "layout(vertices = 1) out;\n"
        "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
        "in highp vec4 target[];\n"
        "void main() {\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[0].gl_Position + target[0];\n"
        "    gl_TessLevelOuter[0] = 1.0;\n"
        "}\n";
    GLuint p = glCreateShaderProgramv(GL_TESS_CONTROL_SHADER, 1, &tcs);
    GLint st = 0; glGetProgramiv(p, GL_LINK_STATUS, &st);
    EXPECT_EQ_INT(st, GL_TRUE);
    GLint mpv = 0; glGetIntegerv(GL_MAX_PATCH_VERTICES, &mpv);
    char names[8][64]; GLint sizes[8], locs[8];
    int n = piq_dump(p, GL_PROGRAM_INPUT, names, NULL, sizes, locs, 8);
    int hit = 0;
    for (int i = 0; i < n; i++)
        if (!strcmp(names[i], "target[0]")) {
            printf("  %s size=%d loc=%d (max patch = %d)\n", names[i], sizes[i], locs[i], mpv);
            EXPECT_EQ_INT(sizes[i], mpv);
            EXPECT_EQ_INT(locs[i], -1);
            hit++;
        }
    EXPECT_EQ_INT(hit, 1);
    EXPECT_TRUE(glGetProgramResourceIndex(p, GL_PROGRAM_INPUT, "target[0]") != GL_INVALID_INDEX);
    glDeleteProgram(p);
}

TEST(struct_block_varying_pipeline)
{
    /* a struct-typed io block varying links and rasterizes through the pipe */
    const char *vs =
        "#version 450\n"
        "struct S { vec4 a; float b; };\n"
        "out Blk { S s; vec4 v; } o;\n"
        "void main() { o.s.a = vec4(float(gl_VertexID)); o.s.b = 1.0; o.v = vec4(1); gl_Position = vec4(0,0,0,1); }\n";
    const char *fs =
        "#version 450\n"
        "struct S { vec4 a; float b; };\n"
        "in Blk { S s; vec4 v; } i;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = i.s.a + i.v + vec4(i.s.b); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); piq_link(p, "pipe");
    GLint st = 0; glGetProgramiv(p, GL_LINK_STATUS, &st);
    EXPECT_EQ_INT(st, GL_TRUE);
    GLuint vao = 0; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glBindVertexArray(0); glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p); glDeleteShader(v); glDeleteShader(f);
}

TEST(block_member_explicit_location)
{
    /* with per-member layout(location) qualifiers (all members must carry one,
     * ARB_enhanced_layouts) every member reports its explicit location */
    const char *fs =
        "#version 450\n"
        "in Blk { layout(location = 6) highp vec4 a; layout(location = 7) highp vec4 b; } inst;\n"
        "layout(location = 0) out vec4 o;\n"
        "void main() { o = inst.a + inst.b; }\n";
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint pf = glCreateProgram(); glAttachShader(pf, f); piq_link(pf, "fs");
    char names[8][64]; GLint locs[8];
    int n = piq_dump(pf, GL_PROGRAM_INPUT, names, NULL, NULL, locs, 8);
    EXPECT_EQ_INT(n, 2);
    EXPECT_STREQ(names[0], "Blk.a"); EXPECT_EQ_INT(locs[0], 6);
    EXPECT_STREQ(names[1], "Blk.b"); EXPECT_EQ_INT(locs[1], 7);
    glDeleteProgram(pf); glDeleteShader(f);
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
    RUN(struct_members_of_named_blocks);
    RUN(anonymous_block_names);
    RUN(nested_structs_and_arrays);
    RUN(non_vs_inputs_location_minus_one);
    RUN(patch_block_is_per_patch);
    RUN(arrayed_tcs_io_name_and_size);
    RUN(struct_block_varying_pipeline);
    RUN(block_member_explicit_location);
    return mgl_test_summary();
}
