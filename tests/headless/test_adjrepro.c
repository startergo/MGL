/* local repro: LINES_ADJACENCY, 4 segments, GS echoes every gl_in slot to xfb */
#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out vec2 v_pos;\n"
    "void main(){ v_pos = pos; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *kGs =
    "#version 450 core\n"
    "layout(lines_adjacency) in;\n"
    "layout(location=0) in vec2 v_pos[];\n"
    "layout(line_strip, max_vertices=2) out;\n"
    "layout(location=0) out vec4 g_adj;\n"
    "layout(location=1) out vec4 g_geom;\n"
    "void main(){\n"
    "  g_adj = vec4(v_pos[0], 0.0, 1.0); g_geom = vec4(v_pos[1], 1.0, 1.0); EmitVertex();\n"
    "  g_adj = vec4(v_pos[3], 3.0, 1.0); g_geom = vec4(v_pos[2], 2.0, 1.0); EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kGsTri =
    "#version 450 core\n"
    "layout(triangles_adjacency) in;\n"
    "layout(location=0) in vec2 v_pos[];\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "layout(location=0) out vec4 g_adj;\n"
    "layout(location=1) out vec4 g_geom;\n"
    "void main(){\n"
    "  for (int i = 0; i < 3; i++) {\n"
    "    g_adj = vec4(v_pos[2*i+1], float(2*i+1), 1.0); g_geom = vec4(v_pos[2*i], float(2*i), 1.0); EmitVertex();\n"
    "  }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) out vec4 c;\n"
    "void main(){ c = vec4(1); }\n";

TEST(TriAdjacency2)
{
    /* 2 triangles x 6 vertices, distinct x per stream vertex: the GS echoes every
     * gl_in slot (even = triangle vertex, odd = adjacency) to two xfb buffers */
    GLuint p = glCreateProgram();
    const char *s[3] = { kVs, kGsTri, kFs };
    GLenum t[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++)
    {
        GLuint sh = glCreateShader(t[i]);
        glShaderSource(sh, 1, &s[i], NULL);
        glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                   MGL_FAIL("compile: %s", log); return; }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    const char *vars[2] = { "g_adj", "g_geom" };
    glTransformFeedbackVaryings(p, 2, vars, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("link: %s", log); return; }

    float pts[12][2];
    for (int i = 0; i < 12; i++) { pts[i][0] = (float)i; pts[i][1] = 1.0f; }
    GLuint vao, vbo, xfb, xfb2;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pts, pts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &xfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 6 * 4 * sizeof(float), NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb);
    glGenBuffers(1, &xfb2);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb2);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 6 * 4 * sizeof(float), NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, xfb2);

    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES_ADJACENCY, 0, 12);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float out[6 * 4], out2[6 * 4];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb2);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out2, out2);
    int bad = 0;
    for (int prim = 0; prim < 2; prim++)
        for (int v = 0; v < 3; v++)
        {
            int sa = 2 * v + 1, sg = 2 * v;      /* slots echoed this vertex */
            float x = out[(prim * 3 + v) * 4], tag = out[(prim * 3 + v) * 4 + 2];
            float x2 = out2[(prim * 3 + v) * 4], tag2 = out2[(prim * 3 + v) * 4 + 2];
            printf("tri %d v%d adj gl_in[%d] x=%g tag=%g | geom gl_in[%d] x=%g tag=%g\n",
                   prim, v, sa, x, tag, sg, x2, tag2);
            if (x != prim * 6 + sa || tag != sa) bad++;
            if (x2 != prim * 6 + sg || tag2 != sg) bad++;
        }
    printf("triangles_adjacency 2-triangle capture: %d wrong of 12\n", bad);
    EXPECT_TRUE(bad == 0);
}

TEST(LinesAdjacency4)
{
    GLint rb = 0, fbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
    (void)fbo;
    GLuint p = glCreateProgram();
    const char *s[3] = { kVs, kGs, kFs };
    GLenum t[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++)
    {
        GLuint sh = glCreateShader(t[i]);
        glShaderSource(sh, 1, &s[i], NULL);
        glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                   MGL_FAIL("compile: %s", log); return; }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    const char *vars[2] = { "g_adj", "g_geom" };
    glTransformFeedbackVaryings(p, 2, vars, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("link: %s", log); return; }

    /* 16 stream vertices, x = index so every slot is identifiable */
    float pts[16][2];
    for (int i = 0; i < 16; i++) { pts[i][0] = (float)i; pts[i][1] = 1.0f; }
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pts, pts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    GLuint xfb, xfb2;
    glGenBuffers(1, &xfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8 * 4 * sizeof(float), NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb);
    glGenBuffers(1, &xfb2);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb2);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8 * 4 * sizeof(float), NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, xfb2);

    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_LINES);
    glDrawArrays(GL_LINES_ADJACENCY, 0, 16);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float out[8 * 4], out2[8 * 4];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb2);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out2, out2);
    int bad = 0;
    for (int prim = 0; prim < 4; prim++)
        for (int v = 0; v < 2; v++)
        {
            /* adj buffer: [gl_in[0], gl_in[3]]; geom buffer: [gl_in[1], gl_in[2]] */
            int slot = (v == 0) ? 0 : 3;
            float x = out[(prim * 2 + v) * 4 + 0];
            float tag = out[(prim * 2 + v) * 4 + 2];
            float x2 = out2[(prim * 2 + v) * 4 + 0];
            float tag2 = out2[(prim * 2 + v) * 4 + 2];
            int slot2 = (v == 0) ? 1 : 2;
            printf("prim %d adj gl_in[%d] = x=%g tag=%g | geom gl_in[%d] = x=%g tag=%g\n",
                   prim, slot, x, tag, slot2, x2, tag2);
            if (x != prim * 4 + slot || tag != slot) bad++;
            if (x2 != prim * 4 + slot2 || tag2 != slot2) bad++;
        }
    printf("lines_adjacency 4-segment two-buffer capture: %d wrong of 16\n", bad);
    EXPECT_TRUE(bad == 0);
}

int main(void)
{
    RUN(LinesAdjacency4);
    RUN(TriAdjacency2);
    return mgl_test_summary();
}
