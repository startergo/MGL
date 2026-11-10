/*
 * test_geom_stageio.c — geometry r111: nested STRUCTURE members of stage-IO
 * interface blocks (GS in/out, VS out feeding a GS) and ARRAY-OF-STRUCT
 * members of plain struct outputs / blocks: render, xfb capture names,
 * packed offsets, locations.
 *
 * Every case here was a link-time REFUSAL before r111 ("unsupported interface
 * block member type" / "array-of-structure member ... is not supported").
 */
#include "mgl_test.h"
#include <stdbool.h>

/* ---------------------------------------------------------------- helpers */

/* compile a VS/GS/FS program; returns 0 and reports the info log on failure */
static GLuint make_vgf(const char *vs, const char *gs, const char *fs)
{
    GLuint p = glCreateProgram();
    const char *srcs[3] = { vs, gs, fs };
    GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };

    for (int i = 0; i < 3; i++)
    {
        GLuint s;

        if (!srcs[i]) continue;
        s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], NULL);
        glCompileShader(s);
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[2048] = "";

            glGetProgramInfoLog(p, sizeof log, NULL, log);
            MGL_FAIL("link failed: %s", log);
            glDeleteProgram(p);
            return 0;
        }
    }
    return p;
}

/* a whole-triangle fullscreen geometry shader (covers an 8x8 viewport) */
#define GS_TRI_QUAD(WRITEV)                                                   \
    "#version 460 core\n"                                                     \
    "layout(triangles) in;\n"                                                 \
    "layout(triangle_strip, max_vertices = 3) out;\n"                         \
    WRITEV                                                                    \
    "void main() {\n"                                                         \
    "  for (int k = 0; k < 3; k++) {\n"                                       \
    "    gl_Position = gl_in[k].gl_Position;\n"                               \
    "    flat int kk = k;\n"                                                  \
    WRITE_BODY                                                                \
    "    EmitVertex();\n"                                                     \
    "  }\n"                                                                   \
    "  EndPrimitive();\n"                                                     \
    "}\n"

/* ------------------------------------------------------------ nested block */

static const char *vs_out_blk_nested =
    "#version 460 core\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "out Blk { Inner i; vec4 t; };\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  i.b = vec2(0.75, 0.5);\n"
    "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "}\n";

/* the GS reads the nested members and passes them on as a plain varying */
TEST(gs_in_block_nested_render)
{
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "in Blk { Inner i; vec4 t; } vs_gs_in[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].i.a.xy + vs_gs_in[k].i.b, vs_gs_in[k].t.z, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = make_vgf(vs_out_blk_nested, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    /* 1.0+0.75, 0.5+0.5, 0.3 */
    EXPECT_PIXEL(4, 4, 255, 255, 77, 255, 8);
    glDeleteProgram(p);
}

/* GS OUTPUT interface block with a nested struct member, consumed by the FS */
TEST(gs_out_block_nested_render)
{
    const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "out Blk { Inner i; vec4 t; } goku;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    goku.i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "    goku.i.b = vec2(0.75, 0.5);\n"
        "    goku.t = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "in Blk { Inner i; vec4 t; } goku;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = vec4(goku.i.a.xy + goku.i.b, goku.t.z, 1.0); }\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 255, 255, 77, 255, 8);
    glDeleteProgram(p);
}

/* plain STRUCT GS output with a nested struct member, consumed by the FS */
TEST(gs_out_plain_nested_render)
{
    const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "struct Goku { Inner i; float w; };\n"
        "out Goku goku;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    goku.i.a = vec4(0.2, 0.4, 0.6, 0.8);\n"
        "    goku.i.b = vec2(0.8, 0.6);\n"
        "    goku.w = 0.4;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "struct Goku { Inner i; float w; };\n"
        "in Goku goku;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = vec4(goku.i.a.x + goku.i.b.x, goku.w, goku.i.b.y, 1.0); }\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 255, 102, 153, 255, 8);
    glDeleteProgram(p);
}

/* ------------------------------------------------------- array of structs */

static const char *gs_out_aos_src =
    "#version 460 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "struct Leaf { vec4 v; };\n"
    "struct Goku { Leaf l[2]; float w; };\n"
    "out Goku goku;\n"
    "void main() {\n"
    "  gl_Position = gl_in[0].gl_Position;\n"
    "  goku.l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
    "  goku.l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
    "  goku.w = 0.9;\n"
    "  EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *vs_point_src =
    "#version 460 core\n"
    "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";

/* run one POINTS xfb capture of the gs_out_aos_src program; `varyings` is the
 * API name list.  Returns the program (bound) and fills out[] / the primitive
 * count. */
static GLuint aos_capture(const char **varyings, int nvarying, GLenum mode,
                          float *out, size_t out_bytes, GLint *prims, float *out2)
{
    GLuint p = glCreateProgram();
    GLuint q = 0, bo = 0, bo2 = 0;

    {
        const char *srcs[2] = { vs_point_src, gs_out_aos_src };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint s = glCreateShader(types[i]);

            glShaderSource(s, 1, &srcs[i], NULL);
            glCompileShader(s);
            glAttachShader(p, s);
            glDeleteShader(s);
        }
    }
    glTransformFeedbackVaryings(p, nvarying, varyings, mode);
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[2048] = "";

            glGetProgramInfoLog(p, sizeof log, NULL, log);
            MGL_FAIL("link failed: %s", log);
            glDeleteProgram(p);
            return 0;
        }
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo);
    glGenBuffers(1, &bo2);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo2);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    if (mode == GL_SEPARATE_ATTRIBS)
    {
        /* distinct buffers per binding point (the overlapping-base case is
         * undefined territory GL does not need to make work) */
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, bo2);
    }
    else
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, prims);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out);
    if (mode == GL_SEPARATE_ATTRIBS && out2)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo2);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out2);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &bo2);
    return p;
}

TEST(gs_aos_out_xfb_whole)
{
    const char *names[1] = { "goku" };
    float out[9];
    GLint prims = 0;
    GLuint p = aos_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

    if (!p) return;
    EXPECT_EQ_INT(prims, 1);
    /* 13.2.2 packing: l[0].v (4 words), l[1].v (4 words), w (1 word) */
    EXPECT_NEAR(out[0], 0.25, 1e-5);
    EXPECT_NEAR(out[2], 0.75, 1e-5);
    EXPECT_NEAR(out[4], 0.125, 1e-5);
    EXPECT_NEAR(out[5], 0.375, 1e-5);
    EXPECT_NEAR(out[8], 0.9, 1e-5);
    glDeleteProgram(p);
}

TEST(gs_aos_out_xfb_member_names)
{
    /* "goku.l" selects the whole array-of-struct member (both leaves) */
    {
        const char *names[1] = { "goku.l" };
        float out[8];
        GLint prims = 0;
        GLuint p = aos_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.25, 1e-5);
        EXPECT_NEAR(out[4], 0.125, 1e-5);
        glDeleteProgram(p);
    }
    /* "goku.l[1]" selects one element of the array-of-struct member */
    {
        const char *names[1] = { "goku.l[1]" };
        float out[4];
        GLint prims = 0;
        GLuint p = aos_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.125, 1e-5);
        EXPECT_NEAR(out[2], 0.625, 1e-5);
        glDeleteProgram(p);
    }
    /* "goku.w" is the plain member after the array */
    {
        const char *names[1] = { "goku.w" };
        float out[1];
        GLint prims = 0;
        GLuint p = aos_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.9, 1e-5);
        glDeleteProgram(p);
    }
    /* SEPARATE attribs: "goku.l" and "goku.w" each land in their own binding */
    {
        const char *names[2] = { "goku.l", "goku.w" };
        float out[10], out2[10];
        GLint prims = 0;
        GLuint p = aos_capture(names, 2, GL_SEPARATE_ATTRIBS, out, sizeof out, &prims, out2);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.25, 1e-5);   /* binding 0: the array leaves */
        EXPECT_NEAR(out[4], 0.125, 1e-5);
        EXPECT_NEAR(out2[0], 0.9, 1e-5);   /* binding 1: w */
        glDeleteProgram(p);
    }
}

/* GS OUTPUT array-of-struct member consumed by the fragment stage */
TEST(gs_out_aos_render)
{
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Leaf { vec4 v; };\n"
        "struct Goku { Leaf l[2]; float w; };\n"
        "out Goku goku;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    goku.l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "    goku.l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
        "    goku.w = 0.9;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "struct Leaf { vec4 v; };\n"
        "struct Goku { Leaf l[2]; float w; };\n"
        "in Goku goku;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = vec4(goku.l[0].v.x, goku.l[1].v.y, goku.w, 1.0); }\n";
    const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 64, 96, 230, 255, 8);
    glDeleteProgram(p);
}

/* GS INPUT block with an array-of-struct member (named instance) */
TEST(gs_in_block_aos_render)
{
    const char *vs =
        "#version 460 core\n"
        "struct Leaf { vec4 v; };\n"
        "out Blk { Leaf l[2]; float w; };\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "  l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
        "  w = 0.9;\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Leaf { vec4 v; };\n"
        "in Blk { Leaf l[2]; float w; } vs_gs_in[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].l[0].v.x, vs_gs_in[k].l[1].v.y, vs_gs_in[k].w, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 64, 96, 230, 255, 8);
    glDeleteProgram(p);
}

/* ------------------------------------------------- nested-block xfb names */

static const char *gs_out_blk_nested_pts =
    "#version 460 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "out Blk { Inner i; vec4 t; };\n"
    "void main() {\n"
    "  gl_Position = gl_in[0].gl_Position;\n"
    "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  i.b = vec2(0.75, 0.5);\n"
    "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "  EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";

static GLuint blk_capture(const char **varyings, int nvarying, GLenum mode,
                          float *out, size_t out_bytes, GLint *prims, float *out2)
{
    GLuint p = glCreateProgram();
    GLuint q = 0, bo = 0, bo2 = 0;

    {
        const char *srcs[2] = { vs_point_src, gs_out_blk_nested_pts };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint s = glCreateShader(types[i]);

            glShaderSource(s, 1, &srcs[i], NULL);
            glCompileShader(s);
            glAttachShader(p, s);
            glDeleteShader(s);
        }
    }
    glTransformFeedbackVaryings(p, nvarying, varyings, mode);
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[2048] = "";

            glGetProgramInfoLog(p, sizeof log, NULL, log);
            MGL_FAIL("link failed: %s", log);
            glDeleteProgram(p);
            return 0;
        }
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo);
    glGenBuffers(1, &bo2);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo2);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    if (mode == GL_SEPARATE_ATTRIBS)
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, bo2);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, prims);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out);
    if (mode == GL_SEPARATE_ATTRIBS && out2)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo2);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out2);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &bo2);
    return p;
}

TEST(gs_blk_nested_xfb_names)
{
    /* the whole BLOCK by name */
    {
        const char *names[1] = { "Blk" };
        float out[10];
        GLint prims = 0;
        GLuint p = blk_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[4], 0.75, 1e-5);
        EXPECT_NEAR(out[6], 0.1, 1e-5);   /* i (6 words) then t */
        glDeleteProgram(p);
    }
    /* the whole STRUCTURE member "Blk.i" */
    {
        const char *names[1] = { "Blk.i" };
        float out[6];
        GLint prims = 0;
        GLuint p = blk_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[4], 0.75, 1e-5);
        glDeleteProgram(p);
    }
    /* single leaves */
    {
        const char *names[2] = { "Blk.i.a", "Blk.i.b" };
        float out[6];
        GLint prims = 0;
        GLuint p = blk_capture(names, 2, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[4], 0.75, 1e-5);
        glDeleteProgram(p);
    }
    /* a member spelled with the instance name */
    {
        const char *names[1] = { "i.b" };
        float out[2];
        GLint prims = 0;
        GLuint p = blk_capture(names, 1, GL_INTERLEAVED_ATTRIBS, out, sizeof out, &prims, NULL);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.75, 1e-5);
        EXPECT_NEAR(out[1], 0.5, 1e-5);
        glDeleteProgram(p);
    }
    /* SEPARATE attribs over the nested leaves */
    {
        const char *names[2] = { "Blk.i", "Blk.t" };
        float out[10], out2[10];
        GLint prims = 0;
        GLuint p = blk_capture(names, 2, GL_SEPARATE_ATTRIBS, out, sizeof out, &prims, out2);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out2[0], 0.1, 1e-5);   /* binding 1: t */
        glDeleteProgram(p);
    }
    /* an unknown name still fails the link (GL_INVALID_OPERATION at link) */
    {
        const char *names[1] = { "Blk.nope" };
        GLuint p = glCreateProgram();
        GLint ok = 1;

        {
            const char *srcs[2] = { vs_point_src, gs_out_blk_nested_pts };
            GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

            for (int i = 0; i < 2; i++)
            {
                GLuint s = glCreateShader(types[i]);

                glShaderSource(s, 1, &srcs[i], NULL);
                glCompileShader(s);
                glAttachShader(p, s);
                glDeleteShader(s);
            }
        }
        glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 0);
        glDeleteProgram(p);
    }
}

/* ------------------------------------------------- deeper / exotic shapes */

/* struct-in-struct + matrix + int + double leaves, captured whole */
TEST(gs_deep_struct_xfb)
{
    const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "struct D { dvec4 d; };\n"
        "struct C { ivec4 iv; D d; mat2 m; };\n"
        "struct B { vec4 v; C c; };\n"
        "struct A { B b[2]; float w; };\n"
        "out A a;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  a.b[0].v = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "  a.b[0].c.iv = ivec4(1, 2, 3, 4);\n"
        "  a.b[0].c.d.d = dvec4(0.5, 0.6, 0.7, 0.8);\n"
        "  a.b[0].c.m = mat2(0.9, 0.8, 0.7, 0.6);\n"
        "  a.b[1].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "  a.b[1].c.iv = ivec4(5, 6, 7, 8);\n"
        "  a.b[1].c.d.d = dvec4(-0.5, -0.6, -0.7, -0.8);\n"
        "  a.b[1].c.m = mat2(-0.9, -0.8, -0.7, -0.6);\n"
        "  a.w = 0.125;\n"
        "  EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *names[1] = { "a" };
    GLuint p = 0, q = 0, bo = 0;
    GLint prims = 0, ok = 0;
    /* packed 13.2.2 layout in words:
     *   b[0].v        4 floats   @0
     *   b[0].c.iv     4 ints     @4
     *   b[0].c.d.d    4 doubles   @8   (aligned 8)
     *   b[0].c.m      4 floats   @16
     *   b[1].*        same,      @20
     *   w             1 float    @36 */
    float out[40];
    double dout[4];

    p = glCreateProgram();
    {
        const char *srcs[2] = { vs_point_src, gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint s = glCreateShader(types[i]);

            glShaderSource(s, 1, &srcs[i], NULL);
            glCompileShader(s);
            glAttachShader(p, s);
            glDeleteShader(s);
        }
    }
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";

        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link failed: %s", log);
        glDeleteProgram(p);
        return;
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &prims);
    EXPECT_EQ_INT(prims, 1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 32, sizeof dout, dout);
    {
        GLint iv[4];
        double d1[4];
        float w = 0.0f;

        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 96, sizeof iv, iv);
        EXPECT_EQ_INT(iv[0], 5);            /* b[1].c.iv @ byte 96 */
        EXPECT_EQ_INT(iv[3], 8);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 112, sizeof d1, d1);
        EXPECT_NEAR(d1[0], -0.5, 1e-9);     /* b[1].c.d.d.x @ byte 112 */
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 160, sizeof w, &w);
        EXPECT_NEAR(w, 0.125, 1e-6);        /* w @ byte 160 */
    }
    EXPECT_NEAR(out[0], 0.1, 1e-5);         /* b[0].v.x  @ byte 0 */
    EXPECT_NEAR(out[16], 0.9, 1e-5);        /* b[0].c.m.x @ byte 64 */
    EXPECT_NEAR(out[20], 0.25, 1e-5);       /* b[1].v.x  @ byte 80 */
    EXPECT_NEAR(out[36], -0.9, 1e-5);       /* b[1].c.m.x @ byte 144 */
    EXPECT_NEAR(dout[0], 0.5, 1e-9);        /* b[0].c.d.d.x @ byte 32 */
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    glDeleteProgram(p);
}

/* arrays-of-arrays of leaves inside a captured struct */
TEST(gs_aoa_leaf_struct_xfb)
{
    const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "struct S { vec4 v[2][2]; float w; };\n"
        "out S s;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  s.v[0][0] = vec4(0.0);\n"
        "  s.v[0][1] = vec4(0.1);\n"
        "  s.v[1][0] = vec4(0.2);\n"
        "  s.v[1][1] = vec4(0.3);\n"
        "  s.w = 0.4;\n"
        "  EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *names[1] = { "s" };
    GLuint p = 0, q = 0, bo = 0;
    GLint prims = 0, ok = 0;
    float out[17];

    p = glCreateProgram();
    {
        const char *srcs[2] = { vs_point_src, gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint s = glCreateShader(types[i]);

            glShaderSource(s, 1, &srcs[i], NULL);
            glCompileShader(s);
            glAttachShader(p, s);
            glDeleteShader(s);
        }
    }
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";

        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link failed: %s", log);
        glDeleteProgram(p);
        return;
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &prims);
    EXPECT_EQ_INT(prims, 1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    EXPECT_NEAR(out[0], 0.0, 1e-5);
    EXPECT_NEAR(out[5], 0.1, 1e-5);
    EXPECT_NEAR(out[9], 0.2, 1e-5);
    EXPECT_NEAR(out[13], 0.3, 1e-5);
    EXPECT_NEAR(out[16], 0.4, 1e-5);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    glDeleteProgram(p);
}

/* explicit member locations AROUND a nested struct member */
TEST(gs_nested_locations_render)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location = 3) out vec4 pre;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(location = 5) out Blk { Inner i; };\n"
        "layout(location = 8) out vec4 post;\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  pre = vec4(0.1);\n"
        "  i.a = vec4(0.2, 0.4, 0.6, 0.8);\n"
        "  i.b = vec2(0.8, 0.6);\n"
        "  post = vec4(0.9);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 3) in vec4 pre[];\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(location = 5) in Blk { Inner i; } vs_gs_in[];\n"
        "layout(location = 8) in vec4 post[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].i.a.x + vs_gs_in[k].i.b.x, pre[k].x, post[k].x, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 255, 26, 230, 255, 8);
    glDeleteProgram(p);
}

/* the VS output plain struct with nested members feeds a GS struct input */
TEST(vs_plain_struct_to_gs_render)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "struct Goku { Inner i; float w; };\n"
        "out Goku goku;\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  goku.i.a = vec4(0.2, 0.4, 0.6, 0.8);\n"
        "  goku.i.b = vec2(0.8, 0.6);\n"
        "  goku.w = 0.4;\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "struct Goku { Inner i; float w; };\n"
        "in Goku goku[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(goku[k].i.a.x + goku[k].i.b.x, goku[k].w, goku[k].i.b.y, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 255, 102, 153, 255, 8);
    glDeleteProgram(p);
}

/* instanced + indexed draws with nested-struct stage IO */
TEST(gs_nested_instanced_indexed)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "out Blk { Inner i; };\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  i.a = vec4(0.5 + 0.25 * float(gl_InstanceID), 0.25, 0.75, 1.0);\n"
        "  i.b = vec2(0.125, 0.375);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "in Blk { Inner i; } vs_gs_in[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].i.a.x + vs_gs_in[k].i.b.x, vs_gs_in[k].i.a.y, vs_gs_in[k].i.b.y, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = make_vgf(vs, gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });

    /* instanced: instance 0 -> 0.5+0.125 = 0.625 (159) */
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 1);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 159, 64, 96, 255, 8);

    /* indexed with an index buffer */
    {
        GLuint vao = 0, ibo = 0;
        const GLuint idx[3] = { 0, 1, 2 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(4, 4, 159, 64, 96, 255, 8);
        glDeleteBuffers(1, &ibo);
        glDeleteVertexArrays(1, &vao);
        /* deleting the bound VAO reverts MGL to the "no VAO bound" state, whose
         * draws are rejected (see the r112 report: the core-profile default VAO
         * is a buffers-domain finding) — leave a bound VAO behind for the tests
         * that follow */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    glDeleteProgram(p);
}


/* ---------------------------------------------- VS-only xfb capture names (r112) */

/* the same nested-struct / array-of-struct shapes through a VERTEX-ONLY program:
 * the xfb resolver (transform_feedback.c) flattens them exactly like the geometry
 * resolver does, so a program links the same whether or not a GS is attached */

static const char *vs_xfb_blk_anon =
    "#version 460 core\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "out Blk { Inner i; vec4 t; };\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  i.b = vec2(0.75, 0.5);\n"
    "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "}\n";
static const char *vs_xfb_blk_named =
    "#version 460 core\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "out Blk { Inner i; vec4 t; } b;\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  b.i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  b.i.b = vec2(0.75, 0.5);\n"
    "  b.t = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "}\n";
static const char *vs_xfb_aos =
    "#version 460 core\n"
    "struct Leaf { vec4 v; };\n"
    "struct Goku { Leaf l[2]; float w; };\n"
    "out Goku goku;\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  goku.l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
    "  goku.l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
    "  goku.w = 0.9;\n"
    "}\n";

/* capture one name from a vertex-only program; the packed layout of the first
 * primitive is returned in out[] (words).  Returns 0 (and fails) on link error. */
static GLuint vs_xfb_capture(const char *vs, const char *varying, GLenum mode,
                             float *out, size_t out_bytes, float *out2, GLint *prims)
{
    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    GLuint q = 0, bo = 0, bo2 = 0;
    GLint ok = 0;

    glShaderSource(s, 1, &vs, NULL);
    glCompileShader(s);
    glAttachShader(p, s);
    glDeleteShader(s);
    glTransformFeedbackVaryings(p, 1, &varying, mode);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";

        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("capture '%s': link failed: %s", varying, log);
        glDeleteProgram(p);
        return 0;
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo);
    glGenBuffers(1, &bo2);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo2);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    if (mode == GL_SEPARATE_ATTRIBS)
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, bo2);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, prims);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out);
    if (mode == GL_SEPARATE_ATTRIBS && out2)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo2);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out2);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &bo2);
    return p;
}

TEST(vs_xfb_nested_blk_names)
{
    GLint prims = 0;
    float out[16];

    /* the whole ANONYMOUS block: i.a (4), i.b (2), t (4) = 10 words */
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_anon, "Blk", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);           /* 3 vertices of TRIANGLES = 1 primitive */
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[4], 0.75, 1e-5);
        EXPECT_NEAR(out[6], 0.1, 1e-5);
        glDeleteProgram(p);
    }
    /* the whole structure member "Blk.i" */
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_anon, "Blk.i", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[5], 0.5, 1e-5);
        EXPECT_NEAR(out[6], 1.0, 1e-5);   /* vertex 2 starts right after (6 words) */
        glDeleteProgram(p);
    }
    /* leaves through the block spelling and the bare member spelling */
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_anon, "Blk.i.a", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[1], 0.5, 1e-5);
        EXPECT_NEAR(out[4], 1.0, 1e-5);    /* vertex 2 at word 4 */
        glDeleteProgram(p);
    }
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_anon, "i.b", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.75, 1e-5);
        EXPECT_NEAR(out[2], 0.75, 1e-5);
        glDeleteProgram(p);
    }
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_anon, "Blk.t", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[2], 0.3, 1e-5);
        EXPECT_NEAR(out[4], 0.1, 1e-5);
        glDeleteProgram(p);
    }
    /* a NAMED instance: the structure member "Blk.i.b" (2 words per vertex) */
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_named, "Blk.i.b", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[1], 0.5, 1e-5);
        EXPECT_NEAR(out[2], 0.75, 1e-5);   /* vertex 2 at word 2 */
        glDeleteProgram(p);
    }
    {
        /* r114: the INSTANCE spelling ("b.t") resolves too — the instance root is
         * registered alongside the block name (both are accepted spellings) */
        GLuint p = vs_xfb_capture(vs_xfb_blk_named, "b.t", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[3], 0.4, 1e-5);
        glDeleteProgram(p);
    }
    /* the whole NAMED block */
    {
        GLuint p = vs_xfb_capture(vs_xfb_blk_named, "Blk", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[6], 0.1, 1e-5);
        glDeleteProgram(p);
    }
}

TEST(vs_xfb_plain_struct_aos)
{
    GLint prims = 0;
    float out[12], out2[12];

    /* whole struct: l[0] (4), l[1] (4), w (1) — tight */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aos, "goku", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.25, 1e-5);
        EXPECT_NEAR(out[4], 0.125, 1e-5);
        EXPECT_NEAR(out[8], 0.9, 1e-5);
        EXPECT_NEAR(out[9], 0.25, 1e-5);   /* vertex 2 at word 9 */
        glDeleteProgram(p);
    }
    /* the whole array-of-struct member and one element */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aos, "goku.l", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.25, 1e-5);
        EXPECT_NEAR(out[4], 0.125, 1e-5);
        EXPECT_NEAR(out[8], 0.25, 1e-5);   /* 8-word span per vertex */
        glDeleteProgram(p);
    }
    {
        GLuint p = vs_xfb_capture(vs_xfb_aos, "goku.l[1]", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.125, 1e-5);
        EXPECT_NEAR(out[2], 0.625, 1e-5);
        EXPECT_NEAR(out[4], 0.125, 1e-5);
        glDeleteProgram(p);
    }
    /* SEPARATE attribs: a single leaf fits the 4-component limit; the whole
     * struct (9 components) exceeds it and fails the link */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aos, "goku.l[0]", GL_SEPARATE_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.25, 1e-5);
        EXPECT_NEAR(out[3], 1.0, 1e-5);
        EXPECT_NEAR(out[4], 0.25, 1e-5);   /* vertex 2 at word 4 */
        glDeleteProgram(p);
    }
    {
        /* the whole struct (9 components) exceeds the 4-component SEPARATE limit
         * and fails the link — check it directly, without the capture helper */
        GLuint prog = glCreateProgram();
        GLuint sh = glCreateShader(GL_VERTEX_SHADER);
        GLint ok = 1;
        const char *nm = "goku";

        glShaderSource(sh, 1, &vs_xfb_aos, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
        glTransformFeedbackVaryings(prog, 1, &nm, GL_SEPARATE_ATTRIBS);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 0);
        glDeleteProgram(prog);
    }
    /* an unknown member name still fails the link */
    {
        GLuint p = glCreateProgram();
        GLuint s = glCreateShader(GL_VERTEX_SHADER);
        GLint ok = 1;
        const char *bad = "goku.nope";

        glShaderSource(s, 1, &vs_xfb_aos, NULL);
        glCompileShader(s);
        glAttachShader(p, s);
        glDeleteShader(s);
        glTransformFeedbackVaryings(p, 1, &bad, GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 0);
        glDeleteProgram(p);
    }
}


/* --------------------------------------------- shader-DECLARED capture (r113) */

/* layout(xfb_buffer/xfb_offset/xfb_stride) on the r111/r112 shapes — VS-only and
 * through a GS.  Every case here failed to link before r113 ("a structure cannot
 * be captured as a whole") or produced a capture variant that did not compile
 * (leaves named without their struct path). */

static GLuint declared_capture(const char *const *srcs, const GLenum *types, int n,
                               GLenum mode, GLuint stride, float *out, size_t out_bytes,
                               GLint *prims)
{
    GLuint p = glCreateProgram();
    GLuint q = 0, bo = 0;
    GLint ok = 0;

    for (int i = 0; i < n; i++)
    {
        GLuint sh = glCreateShader(types[i]);

        glShaderSource(sh, 1, &srcs[i], NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";

        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("declared capture: link failed: %s", log);
        glDeleteProgram(p);
        return 0;
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(mode);
    glDrawArrays(mode, 0, mode == GL_TRIANGLES ? 3 : 1);   /* a complete primitive */
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, prims);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, out_bytes, out);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    return p;
}

/* a VS-only declared capture of an anonymous block with a nested struct member:
 * i at 0 (a@0 4w, b@16 2w), t at 24 */
TEST(vs_decl_blk_nested)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(xfb_buffer = 0, xfb_stride = 40) out Blk {\n"
        "   layout(xfb_offset = 0) Inner i;\n"
        "   layout(xfb_offset = 24) vec4 t;\n"
        "};\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  i.b = vec2(0.75, 0.5);\n"
        "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "}\n";
    const char *srcs[1] = { vs };
    const GLenum types[1] = { GL_VERTEX_SHADER };
    GLint prims = 0;
    float out[12];
    GLuint p = declared_capture(srcs, types, 1, GL_TRIANGLES, 40, out, sizeof out, &prims);

    if (!p) return;
    EXPECT_EQ_INT(prims, 1);
    EXPECT_NEAR(out[0], 1.0, 1e-5);
    EXPECT_NEAR(out[4], 0.75, 1e-5);
    EXPECT_NEAR(out[6], 0.1, 1e-5);
    glDeleteProgram(p);
}

/* a NAMED instance with a nonzero struct-member offset (i at 8) */
TEST(vs_decl_blk_nested_offset)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(xfb_buffer = 0, xfb_stride = 48) out Blk {\n"
        "   layout(xfb_offset = 8) Inner i;\n"
        "   layout(xfb_offset = 32) vec4 t;\n"
        "} b;\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  b.i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  b.i.b = vec2(0.75, 0.5);\n"
        "  b.t = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "}\n";
    const char *srcs[1] = { vs };
    const GLenum types[1] = { GL_VERTEX_SHADER };
    GLint prims = 0;
    float out[12];
    GLuint p = declared_capture(srcs, types, 1, GL_TRIANGLES, 48, out, sizeof out, &prims);

    if (!p) return;
    EXPECT_EQ_INT(prims, 1);
    EXPECT_NEAR(out[0], 0.0, 1e-6);        /* the 8-byte hole */
    EXPECT_NEAR(out[2], 1.0, 1e-5);        /* i.a.x at word 2 */
    EXPECT_NEAR(out[6], 0.75, 1e-5);       /* i.b.x at word 6 */
    EXPECT_NEAR(out[8], 0.1, 1e-5);        /* t.x at word 8 */
    glDeleteProgram(p);
}

/* an ARRAY-OF-STRUCT member of a declared block: l[0]@0, l[1]@16, w@32 */
TEST(vs_decl_blk_aos)
{
    const char *vs =
        "#version 460 core\n"
        "struct Leaf { vec4 v; };\n"
        "layout(xfb_buffer = 0, xfb_stride = 36) out Blk {\n"
        "   layout(xfb_offset = 0) Leaf l[2];\n"
        "   layout(xfb_offset = 32) float w;\n"
        "};\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "  l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
        "  w = 0.9;\n"
        "}\n";
    const char *srcs[1] = { vs };
    const GLenum types[1] = { GL_VERTEX_SHADER };
    GLint prims = 0;
    float out[10];
    GLuint p = declared_capture(srcs, types, 1, GL_TRIANGLES, 36, out, sizeof out, &prims);

    if (!p) return;
    EXPECT_EQ_INT(prims, 1);
    EXPECT_NEAR(out[0], 0.25, 1e-5);
    EXPECT_NEAR(out[4], 0.125, 1e-5);
    EXPECT_NEAR(out[8], 0.9, 1e-5);
    glDeleteProgram(p);
}

/* the same shapes declared on the GEOMETRY stage output */
TEST(gs_decl_blk_shapes)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    GLint prims = 0;
    float out[12];

    /* nested struct member + t */
    {
        const char *gs =
            "#version 460 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices = 1) out;\n"
            "struct Inner { vec4 a; vec2 b; };\n"
            "layout(xfb_buffer = 0, xfb_stride = 40) out Blk {\n"
            "   layout(xfb_offset = 0) Inner i;\n"
            "   layout(xfb_offset = 24) vec4 t;\n"
            "};\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
            "  i.b = vec2(0.75, 0.5);\n"
            "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "}\n";
        const char *srcs[2] = { vs, gs };
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };
        GLuint p = declared_capture(srcs, types, 2, GL_POINTS, 40, out, sizeof out, &prims);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[4], 0.75, 1e-5);
        EXPECT_NEAR(out[6], 0.1, 1e-5);
        glDeleteProgram(p);
    }
    /* array-of-struct member + w */
    {
        const char *gs =
            "#version 460 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices = 1) out;\n"
            "struct Leaf { vec4 v; };\n"
            "layout(xfb_buffer = 0, xfb_stride = 36) out Blk {\n"
            "   layout(xfb_offset = 0) Leaf l[2];\n"
            "   layout(xfb_offset = 32) float w;\n"
            "};\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
            "  l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
            "  w = 0.9;\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "}\n";
        const char *srcs[2] = { vs, gs };
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };
        GLuint p = declared_capture(srcs, types, 2, GL_POINTS, 36, out, sizeof out, &prims);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 0.25, 1e-5);
        EXPECT_NEAR(out[4], 0.125, 1e-5);
        EXPECT_NEAR(out[8], 0.9, 1e-5);
        glDeleteProgram(p);
    }
    /* a stride too small for the declared layout still fails the link */
    {
        const char *gs =
            "#version 460 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices = 1) out;\n"
            "struct Inner { vec4 a; vec2 b; };\n"
            "layout(xfb_buffer = 0, xfb_stride = 28) out Blk {\n"
            "   layout(xfb_offset = 0) Inner i;\n"
            "   layout(xfb_offset = 24) vec4 t;\n"
            "};\n"
            "void main() { EmitVertex(); EndPrimitive(); }\n";
        const char *srcs[2] = { vs, gs };
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };
        GLuint p = glCreateProgram();
        GLint ok = 1;

        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);

            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 0);
        glDeleteProgram(p);
    }
}


/* ------------------------------------------------- array-of-blocks xfb (r114) */

/* "out Blk { Inner i; vec4 t; } b[2];" — the instance name, the block name, one
 * element, a structure member of one element, and a leaf of one element all name
 * captures; elements stride by the std430 block size (48 bytes here: i.a@0, i.b@16,
 * t@32) while the members INSIDE an element pack per 13.2.2 (i.a@0, i.b@16, t@24).
 * Every name failed to resolve before r114. */
static const char *vs_xfb_aob =
    "#version 460 core\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "out Blk { Inner i; vec4 t; } b[2];\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  b[0].i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  b[0].i.b = vec2(0.75, 0.5);\n"
    "  b[0].t = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "  b[1].i.a = vec4(0.9, 0.8, 0.7, 0.6);\n"
    "  b[1].i.b = vec2(0.3, 0.2);\n"
    "  b[1].t = vec4(0.05, 0.15, 0.25, 0.35);\n"
    "}\n";

TEST(vs_xfb_array_of_blocks)
{
    GLint prims = 0;
    float out[32];

    /* the whole instance array: element 0 (words 0-9), element 1 at word 12
     * (std430 stride 48), vertex 2 at word 24 */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aob, "b", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_EQ_INT(prims, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);      /* b[0].i.a.x */
        EXPECT_NEAR(out[4], 0.75, 1e-5);     /* b[0].i.b.x @16 */
        EXPECT_NEAR(out[6], 0.1, 1e-5);      /* b[0].t.x @24 */
        EXPECT_NEAR(out[12], 0.9, 1e-5);     /* b[1].i.a.x @48 */
        EXPECT_NEAR(out[16], 0.3, 1e-5);     /* b[1].i.b.x @64 */
        EXPECT_NEAR(out[18], 0.05, 1e-5);    /* b[1].t.x @72 */
        EXPECT_NEAR(out[22], 1.0, 1e-5);     /* vertex 2's record starts @88 (the
                                                captured extent: element 0's 40 bytes
                                                after the std430 element stride 48) */
        glDeleteProgram(p);
    }
    /* the BLOCK NAME captures the same array */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aob, "Blk", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        EXPECT_NEAR(out[12], 0.9, 1e-5);
        EXPECT_NEAR(out[22], 1.0, 1e-5);
        glDeleteProgram(p);
    }
    /* one element */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aob, "b[1]", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.9, 1e-5);
        EXPECT_NEAR(out[4], 0.3, 1e-5);
        EXPECT_NEAR(out[10], 0.9, 1e-5);     /* vertex 2 at word 10 */
        glDeleteProgram(p);
    }
    /* the structure member of one element */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aob, "b[1].i", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.9, 1e-5);
        EXPECT_NEAR(out[5], 0.2, 1e-5);
        EXPECT_NEAR(out[6], 0.9, 1e-5);      /* 6-word span */
        glDeleteProgram(p);
    }
    /* a leaf of one element */
    {
        GLuint p = vs_xfb_capture(vs_xfb_aob, "b[0].i.a", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[1], 0.5, 1e-5);
        EXPECT_NEAR(out[4], 1.0, 1e-5);      /* vertex 2's i.a.x (16-byte stride) */
        glDeleteProgram(p);
    }
    {
        GLuint p = vs_xfb_capture(vs_xfb_aob, "b[1].t", GL_INTERLEAVED_ATTRIBS, out, sizeof out, NULL, &prims);

        if (!p) return;
        EXPECT_NEAR(out[0], 0.05, 1e-5);
        EXPECT_NEAR(out[4], 0.05, 1e-5);
        glDeleteProgram(p);
    }
    /* an out-of-range element subscript still fails the link */
    {
        GLuint prog = glCreateProgram();
        GLuint sh = glCreateShader(GL_VERTEX_SHADER);
        GLint ok = 1;
        const char *bad = "b[2]";

        glShaderSource(sh, 1, &vs_xfb_aob, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
        glTransformFeedbackVaryings(prog, 1, &bad, GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 0);
        glDeleteProgram(prog);
    }
}


/* --------------------------------------- array-of-blocks VS out -> GS in (r115) */

/* glslang's cross-stage link and I/O mapper both reject an ARRAY-OF-BLOCKS
 * instance feeding a geometry block input ("INTERNAL ERROR: Invalid In/Out
 * variable type") even when the sizes are legal — r115 accepts that false
 * positive (and skips the mapper for it), keeps the instance array in the
 * rewritten capture/kernel sources, and the emulation runs.  A GENUINE size
 * mismatch (2 elements for a 3-vertex triangle input) still fails the link. */
static const char *aob_gs_vs =
    "#version 460 core\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "out Blk { Inner i; vec4 t; } b[3];\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  for (int e = 0; e < 3; e++) {\n"
    "    b[e].i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "    b[e].i.b = vec2(0.75, 0.5);\n"
    "    b[e].t = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "  }\n"
    "}\n";
static const char *aob_gs_gs =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 3) out;\n"
    "struct Inner { vec4 a; vec2 b; };\n"
    "in Blk { Inner i; vec4 t; } vs_gs_in[];\n"
    "out vec4 c_out;\n"
    "void main() {\n"
    "  for (int k = 0; k < 3; k++) {\n"
    "    gl_Position = gl_in[k].gl_Position;\n"
    "    c_out = vec4(vs_gs_in[k].i.a.xy + vs_gs_in[k].i.b, vs_gs_in[k].t.z, 1.0);\n"
    "    EmitVertex();\n"
    "  }\n"
    "  EndPrimitive();\n"
    "}\n";

TEST(gs_in_array_of_blocks_render)
{
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = make_vgf(aob_gs_vs, aob_gs_gs, fs);

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    /* 1.0+0.75, 0.5+0.5, 0.3 */
    EXPECT_PIXEL(4, 4, 255, 255, 77, 255, 8);
    glDeleteProgram(p);
}

TEST(gs_in_array_of_blocks_size_error)
{
    /* two elements cannot feed a 3-vertex triangle input: the link must fail */
    const char *vs2 =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "out Blk { Inner i; vec4 t; } b[2];\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  b[0].i.a = vec4(1.0);\n"
        "  b[1].t = vec4(0.1);\n"
        "}\n";
    GLuint p = glCreateProgram();
    GLint ok = 1;

    {
        const char *srcs[2] = { vs2, aob_gs_gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint s = glCreateShader(types[i]);

            glShaderSource(s, 1, &srcs[i], NULL);
            glCompileShader(s);
            glAttachShader(p, s);
            glDeleteShader(s);
        }
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 0);
    glDeleteProgram(p);
}


/* ---------------------------------- interactions of the new shapes (r116 probes) */

/* glGetTransformFeedbackVarying reports the r112 LEAF expansion (name/type/size
 * of "goku.l[0].v") and the max-length query fits the longest leaf name */
TEST(tfv_introspection_of_leaves)
{
    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    GLint n = -1, maxlen = -1;
    const char *names[1] = { "goku" };
    int seen_l1 = -1, seen_w = -1;

    glShaderSource(s, 1, &vs_xfb_aos, NULL);
    glCompileShader(s);
    glAttachShader(p, s);
    glDeleteShader(s);
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
    }
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
    EXPECT_EQ_INT(n, 3);                       /* l[0].v, l[1].v, w */
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &maxlen);
    EXPECT_TRUE(maxlen >= 12);                 /* "goku.l[0].v" + NUL */
    for (int i = 0; i < n; i++)
    {
        char nm[256];
        GLsizei len = 0;
        GLenum type = 0;
        GLint size = 0;

        nm[0] = 0;
        glGetTransformFeedbackVarying(p, i, sizeof nm, &len, &size, &type, nm);
        if (!strcmp(nm, "goku.l[1].v"))
        {
            seen_l1 = i;
            EXPECT_EQ_INT(type, GL_FLOAT_VEC4);
            EXPECT_EQ_INT(size, 1);
        }
        if (!strcmp(nm, "goku.w"))
        {
            seen_w = i;
            EXPECT_EQ_INT(type, GL_FLOAT);
        }
    }
    EXPECT_TRUE(seen_l1 >= 0);
    EXPECT_TRUE(seen_w >= 0);
    glDeleteProgram(p);
}

/* a separable GS program + separable FS program on one pipeline, the r111
 * nested-block shape between them */
TEST(separable_pipeline_nested_block)
{
    const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "out Blk { Inner i; } goku;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    goku.i.a = vec4(0.2, 0.4, 0.6, 0.8);\n"
        "    goku.i.b = vec2(0.6, 0.4);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "in Blk { Inner i; } goku;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = vec4(goku.i.a.x + goku.i.b.x, goku.i.a.y, goku.i.b.y, 1.0); }\n";
    GLuint vsp = glCreateProgram(), gsp = glCreateProgram(), fsp = glCreateProgram();
    GLuint pipe = 0;

    {
        const char *srcs[3] = { vs, gs, fs };
        GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
        GLuint progs[3] = { vsp, gsp, fsp };

        for (int i = 0; i < 3; i++)
        {
            GLuint sh = glCreateShader(types[i]);

            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            glAttachShader(progs[i], sh);
            glDeleteShader(sh);
            glProgramParameteri(progs[i], GL_PROGRAM_SEPARABLE, GL_TRUE);
            glLinkProgram(progs[i]);
            {
                GLint ok = 0;

                glGetProgramiv(progs[i], GL_LINK_STATUS, &ok);
                EXPECT_TRUE(ok);
            }
        }
    }
    glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    glUseProgramStages(pipe, GL_GEOMETRY_SHADER_BIT, gsp);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fsp);
    glUseProgram(0);               /* an active program would override the pipeline */
    glBindProgramPipeline(pipe);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 204, 102, 102, 255, 8);   /* 0.2+0.6, 0.4, 0.4 */
    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(vsp);
    glDeleteProgram(gsp);
    glDeleteProgram(fsp);
}

/* stream 1 carries a nested-struct block; PRIMITIVES_GENERATED counts it */
TEST(stream1_nested_block_xfb)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 2) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(stream = 0) out vec4 s0;\n"
        "layout(stream = 1) out Blk1 { Inner i; };\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  s0 = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "  EmitStreamVertex(0);\n"
        "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  i.b = vec2(0.75, 0.5);\n"
        "  EmitStreamVertex(1);\n"
        "}\n";
    const char *names[2] = { "s0", "Blk1" };
    GLuint p = glCreateProgram();
    GLuint q = 0, bo0 = 0, bo1 = 0;
    GLint prims1 = 0;
    float o0[8], o1[8];

    {
        const char *srcs[2] = { vs, gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);

            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glTransformFeedbackVaryings(p, 2, names, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
    }
    glUseProgram(p);
    glGenQueries(1, &q);
    glGenBuffers(1, &bo0);
    glGenBuffers(1, &bo1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo0);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo1);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, bo1);
    glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 1, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glEndQueryIndexed(GL_PRIMITIVES_GENERATED, 1);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &prims1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo0);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof o0, o0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo1);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof o1, o1);
    EXPECT_NEAR(o0[0], 0.1, 1e-5);
    EXPECT_NEAR(o1[0], 1.0, 1e-5);        /* stream 1: i.a */
    EXPECT_NEAR(o1[4], 0.75, 1e-5);       /* stream 1: i.b */
    EXPECT_EQ_INT(prims1, 1);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo0);
    glDeleteBuffers(1, &bo1);
    glDeleteProgram(p);
}

/* glDrawArraysIndirect with the nested-struct GS input shape */
TEST(indirect_nested_block)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "out Blk { Inner i; };\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  i.b = vec2(0.75, 0.5);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "in Blk { Inner i; } vs_gs_in[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].i.a.xy + vs_gs_in[k].i.b, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";
    GLuint p = mgl_make_program5(vs, NULL, NULL, gs, fs);
    GLuint ib = 0;
    const GLuint args[4] = { 3, 1, 0, 0 };

    if (!p) return;
    glUseProgram(p);
    glGenBuffers(1, &ib);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ib);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof args, args, GL_STATIC_DRAW);
    glViewport(0, 0, 8, 8);
    glDrawArraysIndirect(GL_TRIANGLES, NULL);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 255, 255, 0, 255, 8);
    glDeleteBuffers(1, &ib);
    glDeleteProgram(p);
}


/* --------------------------------- adjacency + capture equivalence (r117) */

/* adjacency GS inputs with the nested-block shape: the LAST adjacency vertex
 * (gl_in[3] of lines_adjacency, gl_in[5] of triangles_adjacency) must reach the
 * block member; and POINTS into a lines_adjacency GS is INVALID_OPERATION */
static const char *adj_vs =
    "#version 460 core\n"
    "struct Inner { vec4 a; };\n"
    "out Blk { Inner i; };\n"
    "void main() {\n"
    "  vec2 p = vec2(float((gl_VertexID << 1) & 2) - 1.0, float(gl_VertexID & 2) - 1.0);\n"
    "  gl_Position = vec4(p, 0.0, 1.0);\n"
    "  i.a = vec4(0.0, 1.0, 0.0, 1.0);\n"
    "}\n";

TEST(adjacency_nested_block)
{
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 c;\n"
        "void main() { c = c_out; }\n";

    {
        /* lines_adjacency: 4 vertices, the last one paints the triangle */
        const char *gs =
            "#version 460 core\n"
            "layout(lines_adjacency) in;\n"
            "layout(triangle_strip, max_vertices = 3) out;\n"
            "struct Inner { vec4 a; };\n"
            "in Blk { Inner i; } vs_gs_in[];\n"
            "out vec4 c_out;\n"
            "void main() {\n"
            "  c_out = vs_gs_in[3].i.a;\n"
            "  gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); EmitVertex();\n"
            "  gl_Position = vec4(3.0, -1.0, 0.0, 1.0); EmitVertex();\n"
            "  gl_Position = vec4(-1.0, 3.0, 0.0, 1.0); EmitVertex();\n"
            "  EndPrimitive();\n"
            "}\n";
        GLuint p = mgl_make_program5(adj_vs, NULL, NULL, gs, fs);

        if (!p) return;
        glUseProgram(p);
        glViewport(0, 0, 8, 8);
        glDrawArrays(GL_LINES_ADJACENCY, 0, 8);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 8);
        glDeleteProgram(p);
    }
    {
        /* triangles_adjacency: 6 vertices */
        const char *gs =
            "#version 460 core\n"
            "layout(triangles_adjacency) in;\n"
            "layout(triangle_strip, max_vertices = 3) out;\n"
            "struct Inner { vec4 a; };\n"
            "in Blk { Inner i; } vs_gs_in[];\n"
            "out vec4 c_out;\n"
            "void main() {\n"
            "  c_out = vs_gs_in[5].i.a;\n"
            "  gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); EmitVertex();\n"
            "  gl_Position = vec4(3.0, -1.0, 0.0, 1.0); EmitVertex();\n"
            "  gl_Position = vec4(-1.0, 3.0, 0.0, 1.0); EmitVertex();\n"
            "  EndPrimitive();\n"
            "}\n";
        GLuint p = mgl_make_program5(adj_vs, NULL, NULL, gs, fs);

        if (!p) return;
        glUseProgram(p);
        glViewport(0, 0, 8, 8);
        glDrawArrays(GL_TRIANGLES_ADJACENCY, 0, 6);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 8);
        glDeleteProgram(p);
    }
    {
        /* POINTS into a lines_adjacency GS: INVALID_OPERATION */
        const char *gs =
            "#version 460 core\n"
            "layout(lines_adjacency) in;\n"
            "layout(points, max_vertices = 1) out;\n"
            "struct Inner { vec4 a; };\n"
            "in Blk { Inner i; } vs_gs_in[];\n"
            "void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n";
        GLuint p = mgl_make_program5(adj_vs, NULL, NULL, gs,
            "#version 460 core\nlayout(location=0) out vec4 c;\nvoid main(){c=vec4(1);}\n");

        if (!p) return;
        glUseProgram(p);
        glViewport(0, 0, 8, 8);
        while (glGetError() != GL_NO_ERROR) {}
        glDrawArrays(GL_POINTS, 0, 4);
        EXPECT_GL_ERROR(GL_INVALID_OPERATION);
        glDeleteProgram(p);
    }
}

/* capture equivalence (the r112 invariant): the SAME vertex outputs capture
 * byte-identical layouts with and without a geometry shader attached */
TEST(capture_equivalence_with_gs)
{
    const char *vs =
        "#version 460 core\n"
        "struct Leaf { vec4 v; };\n"
        "struct Goku { Leaf l[2]; float w; };\n"
        "out Goku goku;\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  goku.l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "  goku.l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
        "  goku.w = 0.9;\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Leaf { vec4 v; };\n"
        "struct Goku { Leaf l[2]; float w; };\n"
        "in Goku goku[];\n"
        "out Goku gout;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; gout = goku[k]; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *names_vs[1] = { "goku" };
    const char *names_gs[1] = { "gout" };
    float a[9] = {0}, b[9] = {0};

    for (int with_gs = 0; with_gs < 2; with_gs++)
    {
        GLuint p = glCreateProgram();
        GLuint bo = 0;

        {
            const char *srcs[2] = { vs, gs };
            GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };
            int n = with_gs ? 2 : 1;

            for (int i = 0; i < n; i++)
            {
                GLuint sh = glCreateShader(types[i]);

                glShaderSource(sh, 1, &srcs[i], NULL);
                glCompileShader(sh);
                glAttachShader(p, sh);
                glDeleteShader(sh);
            }
        }
        glTransformFeedbackVaryings(p, 1, with_gs ? names_gs : names_vs, GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(p);
        {
            GLint ok = 0;

            glGetProgramiv(p, GL_LINK_STATUS, &ok);
            if (!ok)
            {
                char log[512] = "";

                glGetProgramInfoLog(p, sizeof log, NULL, log);
                MGL_FAIL("with_gs=%d link failed: %s", with_gs, log);
                glDeleteProgram(p);
                return;
            }
        }
        glUseProgram(p);
        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof a, with_gs ? b : a);
        glDeleteBuffers(1, &bo);
        glDeleteProgram(p);
    }
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(b[i], a[i], 0.0);   /* byte-identical */
}


/* --------------------------------------- leaf-count capacity (r118) */

/* a stage whose flattened LEAF count exceeds the old 64-entry reflection table
 * (3 x vec4[23] = 69 leaves) captures EVERY leaf — the old cap silently
 * truncated at 64; and 300 struct leaves (S[300]) refuse the link cleanly. */
TEST(leaf_capacity)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    const char *big_gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "struct Big { vec4 v[23]; };\n"
        "out Blk { Big a; Big b; Big c; };\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  for (int i = 0; i < 23; i++) { a.v[i] = vec4(float(i)); b.v[i] = vec4(1.0); c.v[i] = vec4(2.0); }\n"
        "  EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *names[1] = { "Blk" };
    GLuint p = glCreateProgram();
    GLuint bo = 0;
    GLint ok = 0;
    float out[300];

    {
        const char *srcs[2] = { vs, big_gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);

            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) { glDeleteProgram(p); return; }
    glUseProgram(p);
    glGenBuffers(1, &bo);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    EXPECT_NEAR(out[88], 22.0, 1e-5);    /* a.v[22].x */
    EXPECT_NEAR(out[92], 1.0, 1e-5);     /* b.v[0].x */
    EXPECT_NEAR(out[272], 2.0, 1e-5);    /* c.v[22].x — the leaf that the old cap dropped */
    glDeleteBuffers(1, &bo);
    glDeleteProgram(p);

    /* 300 struct leaves: clean link refusal, not truncation */
    {
        const char *huge_gs =
            "#version 460 core\n"
            "layout(points) in;\n"
            "layout(points, max_vertices = 1) out;\n"
            "struct S { vec4 a; };\n"
            "out Blk { S s[300]; };\n"
            "void main() {\n"
            "  gl_Position = gl_in[0].gl_Position;\n"
            "  EmitVertex();\n"
            "  EndPrimitive();\n"
            "}\n";
        GLuint hp = glCreateProgram();
        GLint hok = 1;

        {
            const char *srcs[2] = { vs, huge_gs };
            GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

            for (int i = 0; i < 2; i++)
            {
                GLuint sh = glCreateShader(types[i]);

                glShaderSource(sh, 1, &srcs[i], NULL);
                glCompileShader(sh);
                glAttachShader(hp, sh);
                glDeleteShader(sh);
            }
        }
        glLinkProgram(hp);
        glGetProgramiv(hp, GL_LINK_STATUS, &hok);
        EXPECT_EQ_INT(hok, 0);
        glDeleteProgram(hp);
    }
}


/* ------------------------------------------ GS output limits (r119) */

/* the ENFORCED budget: MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS (2048, also what the
 * query reports) counts gl_Position + the user outputs x max_vertices; 2044
 * components link, 2084 refuse; max_vertices 256 links, 257 refuses.  The
 * 128-component PER-VERTEX sub-limit (GL_MAX_GEOMETRY_OUTPUT_COMPONENTS) is
 * deliberately NOT enforced: MGL's compute record holds the whole 2048 budget
 * per vertex, the CTS geometry_shader group passed without the check (r83), and
 * reporting the smaller limit is the safe direction for real apps. */
static int gs_link_case(const char *decl, const char *write, const char *layout)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    char gs[4096];
    GLuint p;
    GLint ok = 0;

    snprintf(gs, sizeof gs,
             "#version 460 core\n"
             "layout(points) in;\n"
             "%s\n"
             "%s\n"
             "void main() {\n"
             "  gl_Position = gl_in[0].gl_Position;\n"
             "%s"
             "  EmitVertex();\n"
             "  EndPrimitive();\n"
             "}\n", layout ? layout : "layout(points, max_vertices = 1) out;", decl, write);
    {
        const char *srcs[2] = { vs, gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        p = glCreateProgram();
        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);

            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glDeleteProgram(p);
    return ok;
}

TEST(gs_output_limits)
{
    GLint v = -1;

    glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS, &v);
    EXPECT_EQ_INT(v, 2048);
    {
        char decl[64], write[256];
        int n = 510;

        snprintf(decl, sizeof decl, "out vec4 u[%d];", n);
        snprintf(write, sizeof write, "  u[0] = vec4(1.0);\n  u[%d] = vec4(1.0);\n", n - 1);
        EXPECT_TRUE(gs_link_case(decl, write, NULL));       /* 2044 <= 2048 */
    }
    {
        char decl[64], write[256];
        int n = 520;

        snprintf(decl, sizeof decl, "out vec4 u[%d];", n);
        snprintf(write, sizeof write, "  u[0] = vec4(1.0);\n");
        EXPECT_TRUE(!gs_link_case(decl, write, NULL));      /* 2084 > 2048 */
    }
    EXPECT_TRUE(gs_link_case("out vec4 u;", "  u = vec4(1.0);\n",
                             "layout(points, max_vertices = 256) out;"));
    EXPECT_TRUE(!gs_link_case("out vec4 u;", "  u = vec4(1.0);\n",
                              "layout(points, max_vertices = 257) out;"));
}


/* --------------------------------------- xfb stride limit consistency (r120) */

/* xfb_stride is bounded by MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS (128,
 * also what the query reports): 512 bytes (128 components) links, 516 refuses
 * — glslang enforces exactly the reported limit */
static int stride_case(int stride_bytes, int n_floats)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    char gs[8192];
    int off = 0;
    GLuint p;
    GLint ok = 0;

    off += snprintf(gs + off, sizeof gs - off,
                    "#version 460 core\n"
                    "layout(points) in;\n"
                    "layout(points, max_vertices = 1) out;\n"
                    "layout(xfb_buffer = 0, xfb_stride = %d) out;\n", stride_bytes);
    for (int i = 0; i < n_floats; i++)
        off += snprintf(gs + off, sizeof gs - off,
                        "layout(xfb_buffer = 0, xfb_offset = %d) out float f%d;\n", i * 4, i);
    off += snprintf(gs + off, sizeof gs - off,
                    "void main() {\n"
                    "  gl_Position = gl_in[0].gl_Position;\n");
    for (int i = 0; i < n_floats; i++)
        off += snprintf(gs + off, sizeof gs - off, "  f%d = %d.5;\n", i, i);
    off += snprintf(gs + off, sizeof gs - off,
                    "  EmitVertex();\n"
                    "  EndPrimitive();\n"
                    "}\n");
    {
        const char *srcs[2] = { vs, gs };
        GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        p = glCreateProgram();
        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            const char *sp = srcs[i];

            glShaderSource(sh, 1, &sp, NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glDeleteProgram(p);
    return ok;
}

TEST(xfb_stride_limit)
{
    GLint v = -1;

    glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS, &v);
    EXPECT_EQ_INT(v, 128);
    EXPECT_TRUE(stride_case(512, 32));       /* 32 floats = 128 components: at the cap */
    EXPECT_TRUE(!stride_case(516, 32));      /* one byte over the component budget */
}


/* ------------------------------------ malformed capture names (r123) */

/* every malformed capture name refuses the link with a SPECIFIC error (no
 * crash, no GL error), and the block-name + subscript spelling "Blk[0].t"
 * resolves like the instance spelling */
static const char *mal_vs =
    "#version 460 core\n"
    "struct Leaf { vec4 v; };\n"
    "struct Goku { Leaf l[2]; float w; };\n"
    "out Goku goku;\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  goku.l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
    "  goku.l[1].v = vec4(0.125, 0.375, 0.625, 0.875);\n"
    "  goku.w = 0.9;\n"
    "}\n";
static const char *mal_vs_blk =
    "#version 460 core\n"
    "out Blk { vec4 t; vec4 u; } b[2];\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "  b[0].t = vec4(0.1); b[1].u = vec4(0.2);\n"
    "}\n";

static int mal_link(const char *src, const char *name)
{
    GLuint p = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    const char *sp = src;
    GLint ok = 0;

    glShaderSource(sh, 1, &sp, NULL);
    glCompileShader(sh);
    glAttachShader(p, sh);
    glDeleteShader(sh);
    glTransformFeedbackVaryings(p, 1, &name, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glDeleteProgram(p);
    return ok;
}

TEST(malformed_capture_names)
{
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l[5]"));                   /* out of range */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l[-1]"));                  /* negative */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l[99999999999999999999]"));/* overflow */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l[1"));                    /* unterminated */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l[x]"));                   /* non-numeric */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l[]"));                    /* empty */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.w[0]"));                   /* non-array */
    EXPECT_TRUE(!mal_link(mal_vs, "goku.l..v"));                   /* double dot */
    EXPECT_TRUE(!mal_link(mal_vs, ".goku"));                       /* leading dot */
    EXPECT_TRUE(!mal_link(mal_vs, "goku."));                       /* trailing dot */
    EXPECT_TRUE(!mal_link(mal_vs, "goku[0]"));                     /* non-array */
    EXPECT_TRUE(!mal_link(mal_vs_blk, "b[2]"));
    EXPECT_TRUE(!mal_link(mal_vs_blk, "b[-1]"));
    EXPECT_TRUE(!mal_link(mal_vs_blk, "b[]x"));
    EXPECT_TRUE(!mal_link(mal_vs_blk, "b[0].t[1]"));
    EXPECT_TRUE(!mal_link(mal_vs_blk, "b[1][0]"));
    /* the block-name spelling with a subscript resolves like the instance one */
    {
        GLuint p = glCreateProgram();
        GLuint sh = glCreateShader(GL_VERTEX_SHADER);
        const char *sp = mal_vs_blk;
        const char *name = "Blk[0].t";
        GLint ok = 0;
        GLuint bo = 0;
        float out[4] = {0};

        glShaderSource(sh, 1, &sp, NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
        glTransformFeedbackVaryings(p, 1, &name, GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
        glUseProgram(p);
        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
        EXPECT_NEAR(out[0], 0.1, 1e-5);
        glDeleteBuffers(1, &bo);
        glDeleteProgram(p);
    }
    EXPECT_NO_GL_ERROR();
}


/* --------------------------------- program-binary reload (r124) */

/* glProgramBinary of a program whose GS consumes an ANONYMOUS block with a
 * nested struct member: the decompiler (SPIRV-Cross) used to synthesize an
 * instance name ("_46") for the anonymous block, so the regenerated capture
 * variant failed with "'i' : undeclared identifier".  r124 strips synthesized
 * instances back to anonymous blocks; the reloaded program must relink and
 * render identically. */
TEST(binary_reload_nested_block)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "out Blk { Inner i; };\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  i.b = vec2(0.75, 0.5);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "in Blk { Inner i; } vs_gs_in[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].i.a.xy + vs_gs_in[k].i.b, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = c_out; }\n";
    GLuint p = make_vgf(vs, gs, fs);
    GLint size = 0;
    GLenum fmt = 0;
    void *buf;

    if (!p) return;
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(4, 4, 255, 255, 0, 255, 8);

    glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &size);
    EXPECT_TRUE(size > 0);
    if (size <= 0) { glDeleteProgram(p); return; }
    buf = malloc((size_t)size);
    glGetProgramBinary(p, size, NULL, &fmt, buf);
    EXPECT_NO_GL_ERROR();
    {
        GLuint p2 = glCreateProgram();

        glProgramBinary(p2, fmt, buf, size);
        {
            GLint ok = 0;

            glGetProgramiv(p2, GL_LINK_STATUS, &ok);
            EXPECT_TRUE(ok);
            if (!ok)
            {
                char log[512] = "";

                glGetProgramInfoLog(p2, sizeof log, NULL, log);
                MGL_FAIL("relink failed: %s", log);
                glDeleteProgram(p2);
                free(buf);
                glDeleteProgram(p);
                return;
            }
        }
        glUseProgram(p2);
        glViewport(0, 0, 8, 8);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(4, 4, 255, 255, 0, 255, 8);   /* identical to the source build */
        glDeleteProgram(p2);
    }
    free(buf);
    glDeleteProgram(p);
}


/* ------------------------- binary reload of the xfb state (r125) */

/* glProgramBinary round-trip must preserve the TRANSFORM_FEEDBACK_VARYINGS state
 * (GL 4.6 7.3.1) — the capture names are serialized in the binary, and the
 * reloaded program's xfb link decompiles the vertex stage's SPIR-V (no GLSL
 * source exists).  The reloaded capture must produce the same bytes. */
TEST(binary_reload_xfb_state)
{
    const char *vs =
        "#version 460 core\n"
        "struct Leaf { vec4 v; };\n"
        "struct Goku { Leaf l[2]; float w; };\n"
        "out Goku goku;\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  goku.l[0].v = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "  goku.l[1].v = vec4(0.125);\n"
        "  goku.w = 0.9;\n"
        "}\n";
    const char *fs = "#version 460 core\nlayout(location=0) out vec4 c;\nvoid main(){ c = vec4(1); }\n";
    const char *name = "goku";
    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    const char *sp = vs;
    GLint size = 0;
    GLenum fmt = 0;
    void *buf;
    float a[9], b[9];

    glShaderSource(s, 1, &sp, NULL);
    glCompileShader(s);
    glAttachShader(p, s);
    glDeleteShader(s);
    glTransformFeedbackVaryings(p, 1, &name, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
    }
    /* capture with the source program */
    glUseProgram(p);
    {
        GLuint bo = 0;

        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof a, a);
        glDeleteBuffers(1, &bo);
    }
    glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &size);
    buf = malloc((size_t)size);
    glGetProgramBinary(p, size, NULL, &fmt, buf);
    {
        GLuint p2 = glCreateProgram();
        GLint ok = 0, n = -1;

        glProgramBinary(p2, fmt, buf, size);
        glGetProgramiv(p2, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        glGetProgramiv(p2, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
        EXPECT_EQ_INT(n, 3);                    /* l[0].v, l[1].v, w survived */
        if (ok && n == 3)
        {
            glUseProgram(p2);
            {
                GLuint bo = 0;

                glGenBuffers(1, &bo);
                glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
                glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
                glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
                glBeginTransformFeedback(GL_TRIANGLES);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glEndTransformFeedback();
                glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
                glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof b, b);
                glDeleteBuffers(1, &bo);
            }
            for (int i = 0; i < 9; i++)
                EXPECT_NEAR(b[i], a[i], 0.0);   /* byte-identical */
        }
        glDeleteProgram(p2);
    }
    free(buf);
    glDeleteProgram(p);
}


/* --------------- binary reload of a DECLARED (xfb_offset) capture (r126) */

/* a shader-declared capture (layout(xfb_buffer/xfb_offset)) round-trips through
 * glProgramBinary: the reloaded program has no GLSL source, so the xfb link
 * decompiles the vertex stage's SPIR-V and rebuilds the capture — the bytes
 * must be identical. */
TEST(binary_reload_declared_capture)
{
    const char *vs =
        "#version 460 core\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(xfb_buffer = 0, xfb_stride = 40) out Blk {\n"
        "   layout(xfb_offset = 0) Inner i;\n"
        "   layout(xfb_offset = 24) vec4 t;\n"
        "};\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  i.b = vec2(0.75, 0.5);\n"
        "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "}\n";
    const char *fs = "#version 460 core\nlayout(location=0) out vec4 c;\nvoid main(){ c = vec4(1); }\n";
    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    const char *sp = vs;
    GLint size = 0;
    GLenum fmt = 0;
    void *buf;
    float a[10] = {0}, b[10] = {0};

    glShaderSource(s, 1, &sp, NULL);
    glCompileShader(s);
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    {
        GLint ok = 0;

        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (!ok) { glDeleteProgram(p); return; }
    }
    glUseProgram(p);
    {
        GLuint bo = 0;

        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof a, a);
        glDeleteBuffers(1, &bo);
    }
    glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &size);
    buf = malloc((size_t)size);
    glGetProgramBinary(p, size, NULL, &fmt, buf);
    {
        GLuint p2 = glCreateProgram();
        GLint ok = 0;

        glProgramBinary(p2, fmt, buf, size);
        glGetProgramiv(p2, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (ok)
        {
            glUseProgram(p2);
            {
                GLuint bo = 0;

                glGenBuffers(1, &bo);
                glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
                glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
                glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
                glBeginTransformFeedback(GL_TRIANGLES);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glEndTransformFeedback();
                glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
                glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof b, b);
                glDeleteBuffers(1, &bo);
            }
            for (int i = 0; i < 10; i++)
                EXPECT_NEAR(b[i], a[i], 0.0);   /* byte-identical */
        }
        glDeleteProgram(p2);
    }
    free(buf);
    glDeleteProgram(p);
}


/* ------------- binary reload of a GS DECLARED capture (r127) --------------- */

/* a shader-declared capture on the GEOMETRY stage's outputs must survive
 * glProgramBinary: the last stage has no source after the reload, so the
 * declared capture is detected from that stage's SPIR-V decorations (the
 * resolver state used to be wiped by the link's re-run and TFV reported 0).
 * DrawTransformFeedback with the reloaded program must also work. */
TEST(binary_reload_gs_declared_capture)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "struct Inner { vec4 a; vec2 b; };\n"
        "layout(xfb_buffer = 0, xfb_stride = 40) out Blk {\n"
        "   layout(xfb_offset = 0) Inner i;\n"
        "   layout(xfb_offset = 24) vec4 t;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  i.a = vec4(1.0, 0.5, 0.25, 0.125);\n"
        "  i.b = vec2(0.75, 0.5);\n"
        "  t = vec4(0.1, 0.2, 0.3, 0.4);\n"
        "  EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    GLuint p = glCreateProgram();
    GLint ok = 0, size = 0;
    GLenum fmt = 0;
    void *buf;
    float a[10] = {0}, b[10] = {0};
    const char *srcs[2] = { vs, gs };
    const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

    for (int i = 0; i < 2; i++)
    {
        GLuint sh = glCreateShader(types[i]);
        const char *sp = srcs[i];

        glShaderSource(sh, 1, &sp, NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) { glDeleteProgram(p); return; }
    glUseProgram(p);
    {
        GLuint bo = 0;

        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 2);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof a, a);
        glDeleteBuffers(1, &bo);
    }
    glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &size);
    buf = malloc((size_t)size);
    glGetProgramBinary(p, size, NULL, &fmt, buf);
    {
        GLuint p2 = glCreateProgram();

        glProgramBinary(p2, fmt, buf, size);
        glGetProgramiv(p2, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (ok)
        {
            GLint n = -1;

            glGetProgramiv(p2, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
            EXPECT_EQ_INT(n, 3);
            glUseProgram(p2);
            {
                GLuint bo = 0;

                glGenBuffers(1, &bo);
                glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
                glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
                glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
                glBeginTransformFeedback(GL_POINTS);
                glDrawArrays(GL_POINTS, 0, 2);
                glEndTransformFeedback();
                glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
                glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof b, b);
                glDeleteBuffers(1, &bo);
            }
            for (int i = 0; i < 10; i++)
                EXPECT_NEAR(b[i], a[i], 0.0);   /* byte-identical */
        }
        glDeleteProgram(p2);
    }
    free(buf);
    glDeleteProgram(p);
}


/* -------- separable pipeline + layered rendering with a reloaded GS (r128) -- */

/* the full reload stack: the GS program (gl_Layer = 1, nested-block input) is
 * reloaded from glProgramBinary, put on a PIPELINE with separate VS/FS
 * programs, and must still paint layer 1 of a texture array */
TEST(binary_reload_pipeline_layered)
{
    const char *vs_feed =
        "#version 460 core\n"
        "struct Inner { vec4 a; };\n"
        "out Blk { Inner i; };\n"
        "void main() {\n"
        "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "  i.a = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";
    const char *gs_layer =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec4 a; };\n"
        "in Blk { Inner i; } vs_gs_in[];\n"
        "out vec4 c_out;\n"
        "void main() {\n"
        "  gl_Layer = 1;\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    gl_Position = gl_in[k].gl_Position;\n"
        "    c_out = vec4(vs_gs_in[k].i.a.xy, 0.0, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 c_out;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = c_out; }\n";
    GLuint vsp = glCreateProgram(), gsp = glCreateProgram(), fsp = glCreateProgram();
    GLuint pipe = 0, tex = 0, fbo = 0;
    GLint ok = 0, size = 0;
    GLenum fmt = 0;
    void *buf;

    {
        const char *srcs[3] = { vs_feed, gs_layer, fs };
        const GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
        GLuint progs[3] = { vsp, gsp, fsp };

        for (int i = 0; i < 3; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            const char *sp = srcs[i];

            glShaderSource(sh, 1, &sp, NULL);
            glCompileShader(sh);
            glAttachShader(progs[i], sh);
            glDeleteShader(sh);
            glProgramParameteri(progs[i], GL_PROGRAM_SEPARABLE, GL_TRUE);
            glLinkProgram(progs[i]);
        }
    }
    glGetProgramiv(gsp, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) goto out;

    glGetProgramiv(gsp, GL_PROGRAM_BINARY_LENGTH, &size);
    buf = malloc((size_t)size);
    glGetProgramBinary(gsp, size, NULL, &fmt, buf);
    {
        GLuint g2 = glCreateProgram();

        glProgramBinary(g2, fmt, buf, size);
        glGetProgramiv(g2, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        if (ok) { glDeleteProgram(gsp); gsp = g2; }
        else glDeleteProgram(g2);
    }
    free(buf);
    if (!ok) goto out;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 8, 8, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, 1);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vsp);
    glUseProgramStages(pipe, GL_GEOMETRY_SHADER_BIT, gsp);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fsp);
    glUseProgram(0);
    glBindProgramPipeline(pipe);
    glViewport(0, 0, 8, 8);
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 8);   /* layer 1 painted */

    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
out:
    glDeleteProgram(vsp);
    glDeleteProgram(gsp);
    glDeleteProgram(fsp);
}


/* ------------------------------ primitive restart + GS (r129) --------------- */

/* a restart index splitting a TRIANGLE_STRIP: both segments must reach the GS
 * with CONTINUOUS gl_PrimitiveIDIn (0 and 1 — it does not reset at the restart).
 * The two segments paint DISJOINT regions: overlapping geometry within one draw
 * has implementation-defined raster order (several of my earlier probes
 * mis-read that as a pid bug).  Both restart flavors are covered. */
static const char *rst_vs =
    "#version 460 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *rst_gs =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 3) out;\n"
    "flat out int pid;\n"
    "void main() {\n"
    "  pid = gl_PrimitiveIDIn;\n"
    "  for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *rst_fs =
    "#version 460 core\n"
    "flat in int pid;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = (pid == 0) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0); }\n";

TEST(primitive_restart_gs)
{
    GLuint p = mgl_make_program5(rst_vs, NULL, NULL, rst_gs, rst_fs);
    GLuint vao = 0, vbo = 0, ibo = 0;

    if (!p) return;
    {
        /* v0 v1 v2 = left half; v3 v4 v5 = right half */
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,
        };
        const GLuint idx[7] = { 0, 1, 2, 0xFFFFFFFFu, 3, 4, 5 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

        /* GL_PRIMITIVE_RESTART + glPrimitiveRestartIndex */
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(0xFFFFFFFFu);
        glDrawElements(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_INT, NULL);
        EXPECT_NO_GL_ERROR();
        glDisable(GL_PRIMITIVE_RESTART);
        EXPECT_PIXEL(3, 8, 255, 0, 0, 255, 8);    /* segment 1, pid 0 */
        EXPECT_PIXEL(12, 8, 0, 255, 0, 255, 8);   /* segment 2, pid 1 — continuous id */

        /* GL_PRIMITIVE_RESTART_FIXED_INDEX */
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        glDrawElements(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_INT, NULL);
        EXPECT_NO_GL_ERROR();
        glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        EXPECT_PIXEL(3, 8, 255, 0, 0, 255, 8);
        EXPECT_PIXEL(12, 8, 0, 255, 0, 255, 8);
    }
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}


/* ------------------ restart + adjacency input + instancing (r130) ---------- */

/* a restart index splitting TRIANGLES_ADJACENCY groups: both groups reach the
 * GS with their full 6-vertex adjacency (gl_in[5] read) and CONTINUOUS ids;
 * and restart + glDrawElementsInstanced resets gl_PrimitiveIDIn per INSTANCE
 * (GL 4.6 11.1.3.6) — exact ids asserted via bit coloring. */
static const char *radj_vs =
    "#version 460 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *radj_gs =
    "#version 460 core\n"
    "layout(triangles_adjacency) in;\n"
    "layout(triangle_strip, max_vertices = 3) out;\n"
    "flat out int pid;\n"
    "void main() {\n"
    "  pid = gl_PrimitiveIDIn * 10 + int(gl_in[5].gl_Position.x * 8.0) + 4;\n"
    "  for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *radj_fs =
    "#version 460 core\n"
    "flat in int pid;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = (pid < 10) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0); }\n";

TEST(restart_adjacency_gs)
{
    GLuint p = mgl_make_program5(radj_vs, NULL, NULL, radj_gs, radj_fs);
    GLuint vao = 0, vbo = 0, ibo = 0;

    if (!p) return;
    {
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,    /* left triangle */
             0.3f, 0.6f,    0.6f, 0.6f,    0.5f, 1.0f,    /* adjacency of group 1 */
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,    /* right triangle */
             0.6f, 0.6f,    0.9f, 0.6f,    1.5f, 1.0f,    /* adjacency of group 2 */
        };
        const GLuint idx[13] = { 0, 1, 2, 3, 4, 5, 0xFFFFFFFFu, 6, 7, 8, 9, 10, 11 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(0xFFFFFFFFu);
        glDrawElements(GL_TRIANGLES_ADJACENCY, 13, GL_UNSIGNED_INT, NULL);
        EXPECT_NO_GL_ERROR();
        glDisable(GL_PRIMITIVE_RESTART);
        EXPECT_PIXEL(3, 8, 255, 0, 0, 255, 8);    /* group 1: pid < 10 */
        EXPECT_PIXEL(12, 8, 0, 255, 0, 255, 8);   /* group 2: pid >= 10, gl_in[5] intact */
    }
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

TEST(restart_instanced_gs)
{
    /* both instances draw the same two restart-separated triangles; each
     * instance's ids restart at 0, so the LAST instance (which lands on top)
     * shows pid 0 (left, r=0 g=0) and pid 1 (right, r=1 g=0) */
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "flat out int pid;\n"
        "void main() {\n"
        "  pid = gl_PrimitiveIDIn;\n"
        "  for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "flat in int pid;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(float(pid & 1), 0.0, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program5(radj_vs, NULL, NULL, gs, fs);
    GLuint vao = 0, vbo = 0, ibo = 0;

    if (!p) return;
    {
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,
        };
        const GLuint idx[7] = { 0, 1, 2, 0xFFFFFFFFu, 3, 4, 5 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(0xFFFFFFFFu);
        glDrawElementsInstanced(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_INT, NULL, 2);
        EXPECT_NO_GL_ERROR();
        glDisable(GL_PRIMITIVE_RESTART);
        EXPECT_PIXEL(3, 8, 0, 0, 0, 255, 8);      /* pid 0 */
        EXPECT_PIXEL(12, 8, 255, 0, 0, 255, 8);   /* pid 1 — per-instance reset */
    }
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}


/* --------------------- line inputs + restart + xfb ids (r131) --------------- */

/* a LINES-input GS fed by a GL_LINE_STRIP draw split by a restart: both runs
 * reach the GS with continuous ids (the fat line is 0.3 NDC thick so pixel
 * centres always sample it — an earlier 0.1-thick probe was below the sampling
 * grid and looked like nothing rendered). */
TEST(restart_lines_input_gs)
{
    const char *gs =
        "#version 460 core\n"
        "layout(lines) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "flat out int pid;\n"
        "void main() {\n"
        "  pid = gl_PrimitiveIDIn;\n"
        "  vec2 a = gl_in[0].gl_Position.xy, b = gl_in[1].gl_Position.xy;\n"
        "  vec2 d = normalize(b - a);\n"
        "  vec2 n = vec2(-d.y, d.x) * 0.15;\n"
        "  gl_Position = vec4(a - n, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(b - n, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(a + n, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(b + n, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "flat in int pid;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = (pid == 0) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program5(radj_vs, NULL, NULL, gs, fs);
    GLuint vao = 0, vbo = 0, ibo = 0;

    if (!p) return;
    {
        const float verts[] = {
            -0.8f, 0.6f,   0.8f, 0.6f,
            -0.8f, -0.6f,  0.8f, -0.6f,
        };
        const GLuint idx[5] = { 0, 1, 0xFFFFFFFFu, 2, 3 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(0xFFFFFFFFu);
        glDrawElements(GL_LINE_STRIP, 5, GL_UNSIGNED_INT, NULL);
        EXPECT_NO_GL_ERROR();
        glDisable(GL_PRIMITIVE_RESTART);
        EXPECT_PIXEL(8, 12, 255, 0, 0, 255, 2);   /* run 1: pid 0 */
        EXPECT_PIXEL(8, 3, 0, 255, 0, 255, 2);    /* run 2: pid 1 */
    }
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

/* restart + xfb: the captured gl_PrimitiveIDIn values are CONTINUOUS across the
 * restart (0 and 1) — matching the raster-mode behaviour pinned in r129 */
TEST(restart_xfb_pid_continuity)
{
    const char *vs = "#version 460 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "flat out int pid;\n"
        "void main() { pid = gl_PrimitiveIDIn; gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    const char *names[1] = { "pid" };
    GLuint p = glCreateProgram();
    GLuint vao = 0, ibo = 0, bo = 0, q = 0;
    GLint ok = 0, gen = -1;
    int out[4] = {0};
    const GLuint idx[7] = { 0, 1, 2, 0xFFFFFFFFu, 3, 4, 5 };

    {
        const char *srcs[2] = { vs, gs };
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            const char *sp = srcs[i];

            glShaderSource(sh, 1, &sp, NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) { glDeleteProgram(p); return; }
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &ibo);
    glGenBuffers(1, &bo);
    glGenQueries(1, &q);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glUseProgram(p);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glBeginTransformFeedback(GL_POINTS);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);
    glDrawElements(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_INT, NULL);
    glDisable(GL_PRIMITIVE_RESTART);
    glEndTransformFeedback();
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &gen);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    EXPECT_EQ_INT(gen, 2);
    EXPECT_EQ_INT(out[0], 0);
    EXPECT_EQ_INT(out[1], 1);        /* continuous across the restart */
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &ibo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}


/* ---------------------------- multi-draw indirect + GS (r132) --------------- */

/* glMultiDrawElementsIndirect with a geometry shader: two 5-word element
 * commands, ids reset PER DRAW (both pid 0 -> black), and a single
 * glDrawElementsIndirect honours firstIndex.  (baseVertex + GS is a known
 * open bug — see the r132 report; these tests use baseVertex = 0.) */
TEST(mdi_elements_gs)
{
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "flat out int pid;\n"
        "void main() {\n"
        "  pid = gl_PrimitiveIDIn;\n"
        "  for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "flat in int pid;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(float(pid & 1), float((pid >> 1) & 1), 0.0, 1.0); }\n";
    GLuint p = mgl_make_program5(radj_vs, NULL, NULL, gs, fs);
    GLuint vao = 0, vbo = 0, ibo = 0, dib = 0;

    if (!p) return;
    {
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,     /* 0-2: left */
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,     /* 3-5: right */
        };
        const GLuint idx[6] = { 0, 1, 2, 3, 4, 5 };
        /* count, instanceCount, firstIndex, baseVertex, baseInstance */
        const GLuint cmds[10] = {
            3, 1, 0, 0, 0,
            3, 1, 3, 0, 0,
        };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glGenBuffers(1, &dib);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, dib);
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL, 2, 0);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(3, 8, 0, 0, 0, 255, 8);    /* draw 1: pid 0 */
        EXPECT_PIXEL(12, 8, 0, 0, 0, 255, 8);   /* draw 2: pid 0 again — per-draw reset */

        /* single glDrawElementsIndirect with firstIndex = 3 selects the right half */
        {
            const GLuint one[5] = { 3, 1, 3, 0, 0 };

            glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof one, one, GL_STATIC_DRAW);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL);
            EXPECT_NO_GL_ERROR();
            EXPECT_PIXEL(12, 8, 0, 0, 0, 255, 8);
            EXPECT_PIXEL(3, 8, 255, 255, 255, 255, 8);   /* left untouched */
        }
    }
    glDeleteBuffers(1, &dib);
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}


/* ---------------- element-indirect baseVertex through a GS (r133) ---------- */

/* baseVertex on an element-INDIRECT draw must shift the GS's inputs exactly
 * once: the prep kernels' primitive/capture lists are ABSOLUTE (index +
 * baseVertex) and the indirect draw arguments publish baseVertex = 0 — before
 * r133 both indirect draws added baseVertex AGAIN (every record fetch shifted
 * by baseVertex; a 6-vertex buffer with baseVertex 3 drew NOTHING). */
TEST(indirect_basevertex_gs)
{
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "void main() { for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; EmitVertex(); } EndPrimitive(); }\n";
    GLuint p = mgl_make_program5(radj_vs, NULL, NULL, gs,
        "#version 460 core\nlayout(location=0) out vec4 color;\nvoid main(){ color = vec4(1.0, 0.0, 0.0, 1.0); }\n");
    GLuint vao = 0, vbo = 0, ibo = 0, dib = 0;

    if (!p) return;
    {
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,     /* 0-2: left */
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,     /* 3-5: right */
        };
        const GLuint idx[6] = { 0, 1, 2, 3, 4, 5 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glGenBuffers(1, &dib);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, dib);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

        /* baseVertex = 3 over indices {0,1,2} -> vertices 3,4,5 (right half) */
        {
            const GLuint cmds[5] = { 3, 1, 0, 3, 0 };

            glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL);
            EXPECT_NO_GL_ERROR();
            EXPECT_PIXEL(12, 8, 255, 0, 0, 255, 8);   /* right half */
            EXPECT_PIXEL(3, 8, 0, 0, 255, 255, 8);    /* left untouched */
        }
        /* the same through glMultiDrawElementsIndirect */
        {
            const GLuint cmds[10] = { 3, 1, 0, 3, 0,  3, 1, 0, 3, 0 };

            glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);
            glClear(GL_COLOR_BUFFER_BIT);
            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL, 2, 0);
            EXPECT_NO_GL_ERROR();
            EXPECT_PIXEL(12, 8, 255, 0, 0, 255, 8);
            EXPECT_PIXEL(3, 8, 0, 0, 255, 255, 8);
        }
        /* and baseVertex = 0 still selects the left half */
        {
            const GLuint cmds[5] = { 3, 1, 0, 0, 0 };

            glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL);
            EXPECT_NO_GL_ERROR();
            EXPECT_PIXEL(3, 8, 255, 0, 0, 255, 8);
            EXPECT_PIXEL(12, 8, 0, 0, 255, 255, 8);
        }
    }
    glDeleteBuffers(1, &dib);
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

/* ------------------ baseVertex on DIRECT element draws (r135) --------------- */

/* glDrawElementsBaseVertex through a GS: baseVertex 3 over indices {0,1,2}
 * selects vertices 3-5 AND the VS's gl_VertexID is the ABSOLUTE id (3), both
 * rasterized and captured through xfb.  The r133 fix covers the direct path
 * too (same prep chain); this pins it. */
static const char *bv_vs =
    "#version 460 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "flat out int vvid;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); vvid = gl_VertexID; }\n";

TEST(basevertex_direct_gs)
{
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "flat in int vvid[];\n"
        "flat out int gvid;\n"
        "void main() {\n"
        "  gvid = vvid[0];\n"
        "  for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "flat in int gvid;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = (gvid >= 3) ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program5(bv_vs, NULL, NULL, gs, fs);
    GLuint vao = 0, vbo = 0, ibo = 0;

    if (!p) return;
    {
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,     /* 0-2: left */
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,     /* 3-5: right */
        };
        const GLuint idx[3] = { 0, 1, 2 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glUseProgram(p);
        glViewport(0, 0, 16, 16);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL, 3);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(12, 8, 0, 255, 0, 255, 8);    /* right, gl_VertexID 3 -> green */
        EXPECT_PIXEL(3, 8, 0, 0, 255, 255, 8);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL, 0);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(3, 8, 255, 0, 0, 255, 8);     /* left, id 0 -> red */
        EXPECT_PIXEL(12, 8, 0, 0, 255, 255, 8);
    }
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

TEST(basevertex_xfb_absolute_vid)
{
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "flat in int vvid[];\n"
        "flat out int gv;\n"
        "void main() { gv = vvid[0]; gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    const char *names[1] = { "gv" };
    GLuint p = glCreateProgram();
    GLuint vao = 0, vbo = 0, ibo = 0, bo = 0;
    GLint ok = 0;
    int out[2] = {0};

    {
        const char *srcs[2] = { bv_vs, gs };
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            const char *sp = srcs[i];

            glShaderSource(sh, 1, &sp, NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) { glDeleteProgram(p); return; }
    {
        const float verts[] = {
            -1.0f, -1.0f,   0.0f, -1.0f,  -1.0f, 3.0f,
             0.0f, -1.0f,   1.0f, -1.0f,   0.0f, 3.0f,
        };
        const GLuint idx[3] = { 0, 1, 2 };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glGenBuffers(1, &bo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glUseProgram(p);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_POINTS);
        glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL, 3);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
        EXPECT_EQ_INT(out[0], 3);       /* the ABSOLUTE gl_VertexID, not the raw index */
    }
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}


/* ------------------------ baseInstance + GS (r137) ------------------------- */

/* baseInstance on indirect draws: gl_InstanceID seen by the VS (and passed
 * through the GS) is ABSOLUTE — baseInstance 2 with instanceCount 2 gives ids
 * 2,3 on the arrays-indirect path AND the element-indirect path; baseInstance 0
 * gives 0,1.  Asserted through an xfb capture of the GS's point output. */
static void baseinstance_case(const GLuint *cmd, int cmd_words, bool elements)
{
    const char *vs =
        "#version 460 core\n"
        "flat out int ids;\n"
        "void main() { ids = gl_InstanceID * 10 + gl_VertexID; }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "flat in int ids[];\n"
        "flat out int gv;\n"
        "void main() { gv = ids[0]; gl_Position = vec4(0.0, 0.0, 0.0, 1.0); EmitVertex(); EndPrimitive(); }\n";
    const char *names[1] = { "gv" };
    GLuint p = glCreateProgram();
    GLuint vao = 0, ibo = 0, bo = 0, dib = 0;
    GLint ok = 0;
    int out[4] = {0};

    {
        const char *srcs[2] = { vs, gs };
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER };

        for (int i = 0; i < 2; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            const char *sp = srcs[i];

            glShaderSource(sh, 1, &sp, NULL);
            glCompileShader(sh);
            glAttachShader(p, sh);
            glDeleteShader(sh);
        }
    }
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) { glDeleteProgram(p); return; }
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &ibo);
    glGenBuffers(1, &bo);
    glGenBuffers(1, &dib);
    glBindVertexArray(vao);
    if (elements)
    {
        const GLuint idx[3] = { 0, 1, 2 };

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, dib);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, (size_t)cmd_words * 4, cmd, GL_STATIC_DRAW);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
    glUseProgram(p);
    glBeginTransformFeedback(GL_POINTS);
    if (elements)
        glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL);
    else
        glDrawArraysIndirect(GL_TRIANGLES, NULL);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    glDeleteBuffers(1, &bo);
    glDeleteBuffers(1, &dib);
    glDeleteBuffers(1, &ibo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    EXPECT_EQ_INT(out[2], 0);
}

TEST(baseinstance_indirect_gs)
{
    /* arrays: count=3, instanceCount=2, first=0, baseInstance=2 */
    const GLuint arr[4] = { 3, 2, 0, 2 };
    /* elements: count=3, instanceCount=2, firstIndex=0, baseVertex=0, baseInstance=2 */
    const GLuint el[5] = { 3, 2, 0, 0, 2 };

    baseinstance_case(arr, 4, false);
    baseinstance_case(el, 5, true);
}

int main(void)
{
    GLuint vao = 0;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    RUN(gs_in_block_nested_render);
    RUN(gs_out_block_nested_render);
    RUN(gs_out_plain_nested_render);
    RUN(gs_aos_out_xfb_whole);
    RUN(gs_aos_out_xfb_member_names);
    RUN(gs_out_aos_render);
    RUN(gs_in_block_aos_render);
    RUN(gs_blk_nested_xfb_names);
    RUN(gs_deep_struct_xfb);
    RUN(gs_aoa_leaf_struct_xfb);
    RUN(gs_nested_locations_render);
    RUN(vs_plain_struct_to_gs_render);
    RUN(gs_nested_instanced_indexed);
    RUN(vs_xfb_nested_blk_names);
    RUN(vs_xfb_plain_struct_aos);
    RUN(vs_decl_blk_nested);
    RUN(vs_decl_blk_nested_offset);
    RUN(vs_decl_blk_aos);
    RUN(gs_decl_blk_shapes);
    RUN(vs_xfb_array_of_blocks);
    RUN(gs_in_array_of_blocks_render);
    RUN(gs_in_array_of_blocks_size_error);
    RUN(tfv_introspection_of_leaves);
    RUN(separable_pipeline_nested_block);
    RUN(stream1_nested_block_xfb);
    RUN(indirect_nested_block);
    RUN(adjacency_nested_block);
    RUN(capture_equivalence_with_gs);
    RUN(leaf_capacity);
    RUN(gs_output_limits);
    RUN(xfb_stride_limit);
    RUN(malformed_capture_names);
    RUN(binary_reload_nested_block);
    RUN(binary_reload_xfb_state);
    RUN(binary_reload_declared_capture);
    RUN(binary_reload_gs_declared_capture);
    RUN(binary_reload_pipeline_layered);
    RUN(primitive_restart_gs);
    RUN(restart_adjacency_gs);
    RUN(restart_instanced_gs);
    RUN(restart_lines_input_gs);
    RUN(restart_xfb_pid_continuity);
    RUN(mdi_elements_gs);
    RUN(indirect_basevertex_gs);
    RUN(basevertex_direct_gs);
    RUN(basevertex_xfb_absolute_vid);
    RUN(baseinstance_indirect_gs);
    return mgl_test_summary();
}
