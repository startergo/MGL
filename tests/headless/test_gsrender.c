#include "mgl_test.h"
#include <string.h>

/* KHR-GL46.geometry_shader.rendering.rendering.points_input_points_output (draw_arrays
 * subtest) repro: 8 points in, GS emits a 3x3 grid of points around each, one pixel per
 * point; the test samples pixel (5*n+1, 1) and expects the vertex color of point n. */
static const char *VS =
    "#version 440\n"
    "in      vec4  position;\n"
    "uniform ivec2 renderingTargetSize;\n"
    "out     vec4  vs_gs_color;\n"
    "void main()\n"
    "{\n"
    "    gl_PointSize = 1.0;\n"
    "    if (renderingTargetSize.y == 3)\n"
    "    {\n"
    "        gl_Position = position;\n"
    "    }\n"
    "    else\n"
    "    {\n"
    "        gl_Position = vec4(position.x,\n"
    "                           -1.0 + (1.5 + float(5 * gl_InstanceID)) / float(renderingTargetSize.y) * 2.0,\n"
    "                           position.zw);\n"
    "    }\n"
    "    switch(gl_VertexID)\n"
    "    {\n"
    "        case 0: vs_gs_color = vec4(1, 0, 0, 0); break;\n"
    "        case 1: vs_gs_color = vec4(0, 1, 0, 0); break;\n"
    "        case 2: vs_gs_color = vec4(0, 0, 1, 0); break;\n"
    "        case 3: vs_gs_color = vec4(0, 0, 0, 1); break;\n"
    "        case 4: vs_gs_color = vec4(1, 1, 0, 0); break;\n"
    "        case 5: vs_gs_color = vec4(1, 0, 1, 0); break;\n"
    "        case 6: vs_gs_color = vec4(1, 0, 0, 1); break;\n"
    "        case 7: vs_gs_color = vec4(1, 1, 1, 0); break;\n"
    "        default: vs_gs_color = vec4(1, 1, 0, 1); break;\n"
    "    }\n"
    "}\n";

static const char *GS =
    "#version 440\n"
    "layout(points)                 in;\n"
    "layout(points, max_vertices=9) out;\n"
    "in  vec4 vs_gs_color[1];\n"
    "out vec4 gs_fs_color;\n"
    "uniform ivec2 renderingTargetSize;\n"
    "void main()\n"
    "{\n"
    "    float dx = 2.0 / float(renderingTargetSize.x);\n"
    "    float dy = 2.0 / float(renderingTargetSize.y);\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(-dx, -dy, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(0, -dy, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(dx, -dy, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(-dx, 0, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position;\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(dx, 0, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(-dx, dy, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(0, dy, 0, 0);\n"
    "    EmitVertex();\n"
    "    gs_fs_color  = vs_gs_color[0];\n"
    "    gl_Position  = gl_in[0].gl_Position + vec4(dx, dy, 0, 0);\n"
    "    EmitVertex();\n"
    "    EndPrimitive();\n"
    "}\n";

static const char *FS =
    "#version 440\n"
    "precision highp float;\n"
    "in  vec4 gs_fs_color;\n"
    "out vec4 result;\n"
    "void main(){ result = gs_fs_color; }\n";

TEST(cts_rendering_points)
{
    const int W = 38, H = 3;
    float pos[8 * 4];
    static const unsigned char expect[8][4] = {
        {255,0,0,0},{0,255,0,0},{0,0,255,0},{0,0,0,255},{255,255,0,0},{255,0,255,0},{255,0,0,255},{255,255,255,0}
    };
    GLuint v, g, f, p, vao, vbo, fbo, tex;
    GLint loc, ok = 0;
    unsigned char px[4];

    for (int n = 0; n < 8; n++)
    {
        pos[n*4+0] = -1.0f + ((float)(1 + 5*n) + 0.5f) / W * 2.0f;
        pos[n*4+1] = -1.0f + (1.5f + 0.5f) / H * 2.0f;
        pos[n*4+2] = 0.0f;
        pos[n*4+3] = 1.0f;
    }

    v = glCreateShader(GL_VERTEX_SHADER);
    g = glCreateShader(GL_GEOMETRY_SHADER);
    f = glCreateShader(GL_FRAGMENT_SHADER);
    p = glCreateProgram();
    glShaderSource(v, 1, &VS, NULL); glCompileShader(v);
    glShaderSource(g, 1, &GS, NULL); glCompileShader(g);
    glShaderSource(f, 1, &FS, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, g); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[8192]; GLsizei len = 0; log[0] = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "LINK FAILED:\n%s\n", log);
    }
    EXPECT_TRUE(ok == GL_TRUE);
    if (!ok) return;
    glUseProgram(p);
    loc = glGetUniformLocation(p, "renderingTargetSize");
    glUniform2i(loc, W, H);

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    {
        GLint aloc = glGetAttribLocation(p, "position");
        glEnableVertexAttribArray(aloc);
        glVertexAttribPointer(aloc, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, W, H);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    glDrawArrays(GL_POINTS, 0, 8);
    EXPECT_NO_GL_ERROR();


    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    for (int n = 0; n < 8; n++)
    {
        int x = n * 5 + 1, y = 1;
        glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        if (memcmp(px, expect[n], 4) != 0)
            printf("point %d at (%d,%d): got [%d %d %d %d] want [%d %d %d %d]\n",
                   n, x, y, px[0], px[1], px[2], px[3],
                   expect[n][0], expect[n][1], expect[n][2], expect[n][3]);
        EXPECT_TRUE(memcmp(px, expect[n], 4) == 0);
    }
    /* dump the whole framebuffer for diagnosis */
    {
        static unsigned char row[38 * 4];
        for (int y = 0; y < H; y++)
        {
            glReadPixels(0, y, W, 1, GL_RGBA, GL_UNSIGNED_BYTE, row);
            printf("row %d:", y);
            for (int x = 0; x < W; x++)
            {
                int c = (row[x*4] ? 1 : 0) | (row[x*4+1] ? 2 : 0) | (row[x*4+2] ? 4 : 0) | (row[x*4+3] ? 8 : 0);
                printf("%x", c);
            }
            printf("\n");
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}



/* ---------------------------------------------------------------------- *
 * KHR-GL46.geometry_shader.rendering.rendering.triangles_input_triangle_strip_
 * output_triangle_strip_drawcall replica: a 7-vertex GL_TRIANGLE_STRIP "BACDAEB"
 * (two degenerate-ish middle triangles) through the CTS triangles-GS that emits two
 * sub-triangles per input primitive; the four quadrant samples must carry the CTS's
 * expected vertex colors.  Exercises both the GPU assembly route (drawElements with a
 * reversed UNSIGNED_BYTE index list over a reversed array) and the CPU route
 * (drawArrays).  Regression test for the emitted-strip ordering and the capture-list /
 * primitive-slot buffer layout. */

#define RT 29

static const char *T_VS =
    "#version 460\n"
    "in  vec4 position;\n"
    "out vec4 vs_gs_color;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = position;\n"
    "    int vertex_id = gl_VertexID + 1;\n"
    "    vs_gs_color = vec4(float(vertex_id) / 48.0, float(vertex_id % 3) / 2.0,\n"
    "                       float(vertex_id % 4) / 3.0, float(vertex_id % 5) / 4.0);\n"
    "}\n";

static const char *T_GS =
    "#version 460\n"
    "layout(triangles)                      in;\n"
    "layout(triangle_strip, max_vertices=6) out;\n"
    "uniform ivec2 renderingTargetSize;\n"
    "in  vec4 vs_gs_color[3];\n"
    "out vec4 gs_fs_color;\n"
    "void main()\n"
    "{\n"
    "    float dx = 2.0 / float(renderingTargetSize.x);\n"
    "    float dy = 2.0 / float(renderingTargetSize.y);\n"
    "    vec4 v0 = gl_in[0].gl_Position;\n"
    "    vec4 v1 = gl_in[1].gl_Position;\n"
    "    vec4 v2 = gl_in[2].gl_Position;\n"
    "    vec4 a = v0, b, c, d;\n"
    "    if (abs(v1.w) >= 1.0) { a = v1; if (abs(v0.x - a.x) < dx) { b = v0; c = v2; } else { b = v2; c = v0; } }\n"
    "    else if (abs(v2.w) >= 1.0) { a = v2; if (abs(v0.x - a.x) < dx) { b = v0; c = v1; } else { b = v1; c = v0; } }\n"
    "    else { if (abs(v1.x - a.x) < dx) { b = v1; c = v2; } else { b = v2; c = v1; } }\n"
    "    d = (b + c) * vec4(0.5);\n"
    "    gl_Position = vec4(a.xyz, 1); gs_fs_color = vs_gs_color[0]; EmitVertex();\n"
    "    gl_Position = vec4(b.xyz, 1); gs_fs_color = vs_gs_color[0]; EmitVertex();\n"
    "    gl_Position = vec4(d.xyz, 1); gs_fs_color = vs_gs_color[0]; EmitVertex();\n"
    "    EndPrimitive();\n"
    "    gl_Position = vec4(a.xyz, 1); gs_fs_color = vec4(2.0, 1.0, 1.0, 1.0) * vs_gs_color[2]; EmitVertex();\n"
    "    gl_Position = vec4(c.xyz, 1); gs_fs_color = vec4(2.0, 1.0, 1.0, 1.0) * vs_gs_color[2]; EmitVertex();\n"
    "    gl_Position = vec4(d.xyz, 1); gs_fs_color = vec4(2.0, 1.0, 1.0, 1.0) * vs_gs_color[2]; EmitVertex();\n"
    "    EndPrimitive();\n"
    "}\n";

static const char *T_FS =
    "#version 460\n"
    "in  vec4 gs_fs_color;\n"
    "out vec4 result;\n"
    "void main() { result = gs_fs_color; }\n";

/* data_coordinates: middle/left/top/bottom/right in the CTS naming (its "top" sits at
 * NDC y=-1); the strip order spells BACDAEB */
static const float T_QUAD[5][4] = {
    { 0.0f,  0.0f, 0, 1 },
    {-1.0f,  0.0f, 0, 0 },
    { 0.0f, -1.0f, 0, 0 },
    { 1.0f,  0.0f, 0, 0 },
    { 0.0f,  1.0f, 0, 0 },
};
static const int T_STRIP[7] = { 1, 0, 2, 3, 0, 4, 1 };

static float t_color(int id, int comp)
{
    int v = id + 1;
    switch (comp) {
        case 0: return (float)v / 48.0f;
        case 1: return (float)(v % 3) / 2.0f;
        case 2: return (float)(v % 4) / 3.0f;
        default: return (float)(v % 5) / 4.0f;
    }
}

static void t_run_variant(GLuint prog, GLuint vbo, int indexed, const char *tag)
{
    GLubyte px[RT][RT][4];
    GLubyte idx[7];
    GLuint ebo = 0;

    glViewport(0, 0, RT, RT);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glUniform2i(glGetUniformLocation(prog, "renderingTargetSize"), RT, RT);

    if (indexed)
    {
        for (int i = 0; i < 7; i++) idx[i] = (GLubyte)(6 - i);
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glDrawElements(GL_TRIANGLE_STRIP, 7, GL_UNSIGNED_BYTE, NULL);
    }
    else
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 7);
    }
    glFinish();
    glReadPixels(0, 0, RT, RT, GL_RGBA, GL_UNSIGNED_BYTE, px);

    /* the CTS sample points: (a,b,d) and (a,c,d) centroids of its four quadrant models,
     * in model coordinates (y = row index); the expected vertex id doubles for the
     * second sub-triangle's red channel and mirrors for the elements draw */
    struct { float x, y; int id; const char *name; } s[8] = {
        { 5.0f / 12, 1.0f / 4,  0, "TL1" }, { 1.0f / 4,  5.0f / 12,  2, "TL2" },
        { 7.0f / 12, 1.0f / 4,  2, "TR1" }, { 3.0f / 4,  5.0f / 12,  4, "TR2" },
        { 7.0f / 12, 3.0f / 4,  4, "BR1" }, { 3.0f / 4,  7.0f / 12,  5, "BR2" },
        { 5.0f / 12, 3.0f / 4,  4, "BL1" }, { 1.0f / 4,  7.0f / 12,  6, "BL2" },
    };
    int bad = 0;
    for (int i = 0; i < 8; i++)
    {
        int col = (int)(s[i].x * (RT - 1) + 0.5f);
        int row = (int)(s[i].y * (RT - 1) + 0.5f);
        int eid = indexed ? 6 - s[i].id : s[i].id;
        float e[4];
        for (int k = 0; k < 4; k++) e[k] = t_color(eid, k);
        if (i % 2 == 1) e[0] *= 2.0f;
        for (int k = 0; k < 4; k++)
        {
            float got = px[row][col][k] / 255.0f;
            if (got < e[k] - 1.0f / 256.0f || got > e[k] + 1.0f / 256.0f)
            {
                printf("  %s %s px(%d,%d) component %d = %.3f want %.3f\n",
                       tag, s[i].name, col, row, k, got, e[k]);
                bad++;
            }
        }
    }
    if (ebo) glDeleteBuffers(1, &ebo);
    EXPECT_TRUE(bad == 0);
}

TEST(cts_tri_strip_output)
{
    GLuint p = glCreateProgram(), tex, fbo, vbo, vao;
    float verts[7][4];

    {
        const char *srcs[3] = { T_VS, T_GS, T_FS };
        GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
        for (int i = 0; i < 3; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
            if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                       MGL_FAIL("compile: %s", log); return; }
            glAttachShader(p, sh);
        }
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
                   MGL_FAIL("link: %s", log); return; }
    }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, RT, RT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);

    /* drawArrays: the raw vertex order; drawElements: the CTS stores the array reversed
     * and reverses the index list, so the pipeline still sees the raw order while
     * gl_VertexID is mirrored */
    for (int i = 0; i < 7; i++) memcpy(verts[i], T_QUAD[T_STRIP[i]], sizeof verts[i]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    t_run_variant(p, vbo, 0, "arrays");

    for (int i = 0; i < 7; i++) memcpy(verts[i], T_QUAD[T_STRIP[6 - i]], sizeof verts[i]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    t_run_variant(p, vbo, 1, "elements");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}


/* ---------------------------------------------------------------------- *
 * GL_LINE_LOOP feeding a lines-input GS: the closing segment (n-1, 0) must reach
 * gl_in[] - the CPU assembly route dropped it (the loop stopped at i = n - 2).  Each
 * input segment emits two points colored by gl_PrimitiveIDIn; the pixel at each loop
 * corner shows the last segment that used it. */

#define LW 64
static const char *L_VS =
    "#version 460\n"
    "in vec4 position;\n"
    "void main() { gl_Position = position; }\n";
static const char *L_GS =
    "#version 460\n"
    "layout(lines) in;\n"
    "layout(points, max_vertices=2) out;\n"
    "flat out int seg;\n"
    "void main()\n"
    "{\n"
    "    for (int v = 0; v < 2; v++) {\n"
    "        seg = gl_PrimitiveIDIn;\n"
    "        gl_Position = gl_in[v].gl_Position;\n"
    "        gl_PointSize = 3.0;\n"
    "        EmitVertex();\n"
    "    }\n"
    "    EndPrimitive();\n"
    "}\n";
static const char *L_FS =
    "#version 460\n"
    "flat in int seg;\n"
    "out vec4 o;\n"
    "void main() { o = vec4(float(seg + 1) / 5.0, 0.5, 0.5, 1); }\n";

/* run the line-loop corner check; variant selects the draw: 0 = arrays, 1 = uint16
 * elements at offset 0 (GPU assembly), 2 = uint16 elements at an odd byte offset (the
 * alignment check pushes the draw onto the CPU assembly route) */
static void l_run(GLuint p, GLuint vbo, int variant, const char *tag)
{
    GLuint tex, fbo;
    int px[4] = { 8, 52, 52, 8 }, py[4] = { 8, 8, 52, 52 };   /* corner pixels (y = row) */
    int bad = 0;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, LW, LW, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, LW, LW);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    if (variant == 0)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDrawArrays(GL_LINE_LOOP, 0, 4);
    }
    else
    {
        /* reversed index list: the drawn loop is (v3 v2 v1 v0); variant 2 puts the
         * indices at an odd byte offset, which the GPU prep kernel cannot read (its
         * index words must be aligned) - that draw takes the CPU assembly route */
        GLushort idx[4] = { 3, 2, 1, 0 };
        GLubyte blob[1 + sizeof idx];
        GLuint ebo;
        size_t bytes = sizeof idx;
        const void *off = 0;
        if (variant == 2) { blob[0] = 0; memcpy(blob + 1, idx, sizeof idx); off = (const void *)1; }
        else memcpy(blob, idx, sizeof idx);
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytes + (variant == 2 ? 1 : 0), blob, GL_STATIC_DRAW);
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, off);
        glDeleteBuffers(1, &ebo);
    }
    glFinish();

    /* every corner is shared by two segments and the later one wins, so the corner
     * colors identify the segments: corner 0 can only come from the loop's closing
     * segment (3, 0) - a red for segment 0 means it was dropped */
    /* every corner is shared by two segments and the later one wins, so the corner
     * colors identify the segments: corner 0 can only come from the loop's closing
     * segment (3, 0) - segment 0's color there means the closing segment was dropped */
    /* the reversed loop's segments are (3,2),(2,1),(1,0),(0,3), so the last segment
     * through each corner differs from the arrays order */
    int last[4];
    if (variant == 0) { last[0] = 3; last[1] = 1; last[2] = 2; last[3] = 3; }
    else              { last[0] = 3; last[1] = 2; last[2] = 1; last[3] = 3; }
    for (int i = 0; i < 4; i++)
    {
        GLubyte c[4];
        glReadPixels(px[i], py[i], 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
        int seg = (c[0] + 25) / 51 - 1;   /* 51 = 255/5 per segment */
        if (seg != last[i]) { printf("  %s: corner %d shows segment %d, want %d (raw %d)\n", tag, i, seg, last[i], c[0]); bad++; }
    }
    EXPECT_TRUE(bad == 0);
}

TEST(line_loop_segments)
{
    GLuint p = glCreateProgram(), tex, fbo, vbo, vao;
    float v[4][4];
    int px[4] = { 8, 52, 52, 8 }, py[4] = { 8, 8, 52, 52 };   /* corner pixels (y = row) */

    for (int i = 0; i < 4; i++)
    {
        v[i][0] = -1.0f + (2.0f * px[i] + 1.0f) / LW;
        v[i][1] = -1.0f + (2.0f * py[i] + 1.0f) / LW;
        v[i][2] = 0; v[i][3] = 1;
    }

    {
        const char *srcs[3] = { L_VS, L_GS, L_FS };
        GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
        for (int i = 0; i < 3; i++)
        {
            GLuint sh = glCreateShader(types[i]);
            glShaderSource(sh, 1, &srcs[i], NULL);
            glCompileShader(sh);
            GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
            if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                       MGL_FAIL("compile: %s", log); return; }
            glAttachShader(p, sh);
        }
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
                   MGL_FAIL("link: %s", log); return; }
    }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, LW, LW, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);

    l_run(p, vbo, 0, "arrays");
    l_run(p, vbo, 1, "elements");
    l_run(p, vbo, 2, "elements odd offset (CPU assembly)");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(cts_rendering_points);
    RUN(cts_tri_strip_output);
    RUN(line_loop_segments);
    return mgl_test_summary();
}
