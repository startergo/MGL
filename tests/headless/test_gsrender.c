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

int main(void)
{
    RUN(cts_rendering_points);
    return mgl_test_summary();
}
