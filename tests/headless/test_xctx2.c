/* test_xctx2.c — focused: name 31/32/33 texture behavior in ONE context */
#include "mgl_test.h"

static const char *k_vs =
    "#version 330\n"
    "layout(location=0) in vec2 pos;\n"
    "out vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs =
    "#version 330\n"
    "uniform sampler2D tex;\n"
    "in vec2 uv;\n"
    "out vec4 o;\n"
    "void main() { o = texture(tex, uv); }\n";

static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };

TEST(names_at_table_boundary)
{
    GLuint tex[40] = {0};
    GLubyte img[4*4*4];
    GLuint prog, vao, vbo, loc;

    for (int i = 0; i < 16; i++) { img[i*4]=0; img[i*4+1]=255; img[i*4+2]=0; img[i*4+3]=255; }

    glGenTextures(40, tex);
    for (int i = 0; i < 40; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    while (glGetError() != GL_NO_ERROR) {}

    /* every bound texture must be a texture */
    for (int i = 0; i < 40; i++) {
        if (!glIsTexture(tex[i]))
            fprintf(stderr, "  note: glIsTexture(%u) == false (i=%d)\n", tex[i], i);
    }
    EXPECT_TRUE(glIsTexture(tex[30]));
    EXPECT_TRUE(glIsTexture(tex[31]));   /* name == initial table size */
    EXPECT_TRUE(glIsTexture(tex[32]));
    EXPECT_TRUE(glIsTexture(tex[39]));

    prog = mgl_make_program(k_vs, k_fs);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);
    loc = glGetUniformLocation(prog, "tex");
    if (loc >= 0) glUniform1i(loc, 0);
    glViewport(0, 0, 4, 4);

    int bad = 0;
    for (int i = 0; i < 40; i++) {
        GLubyte px[4];
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        mgl_read_rgba(1, 1, px);
        if (px[1] < 250) { bad++; fprintf(stderr, "  note: sample tex%u(name %u): px=%d,%d,%d,%d\n",
                                          (unsigned)i, tex[i], px[0],px[1],px[2],px[3]); }
        while (glGetError() != GL_NO_ERROR) {}
    }
    EXPECT_EQ_INT(bad, 0);
}

int main(void)
{
    RUN(names_at_table_boundary);
    return mgl_test_summary();
}
