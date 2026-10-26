#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"

/* minimal: VS texture() sample captured via xfb, ONE draw, one link. */

static const char *kVs =
    "#version 450 core\n"
    "uniform sampler2D samp;\n"
    "layout(location = 0) in vec2 posAttr;\n"
    "layout(location = 1) in vec2 tcAttr;\n"
    "out vec4 col;\n"
    "void main()\n"
    "{\n"
    "    col = texture(samp, tcAttr);\n"
    "    gl_Position = vec4(posAttr, 0.0, 1.0);\n"
    "}\n";
static const char *kFs =
    "#version 450 core\n"
    "in vec4 col;\n"
    "out vec4 o;\n"
    "void main() { o = col; }\n";
static const char *kVaryings[] = { "col" };

TEST(XfbVsSampleMinimal)
{
    GLuint prog, vs, fs, tex, vao, vbo, xbuf;
    unsigned char fdata[16 * 16 * 4];
    float out[4];
    const float vb[] = {
        -1, -1,  1, -1,  -1, 1,   /* pos, 3 verts */
         0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  /* tc = (0.5,0.5) -> texel(8,8)=(128,128,200,255) */
    };
    const int tc_off = 6 * sizeof(float);

    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++)
        { int i = y*16+x; fdata[i*4]= x*16; fdata[i*4+1]=y*16; fdata[i*4+2]=200; fdata[i*4+3]=255; }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, fdata);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &kVs, NULL); glCompileShader(vs);
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &kFs, NULL); glCompileShader(fs);
    prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glTransformFeedbackVaryings(prog, 1, kVaryings, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    { GLint ok=0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
      if (!ok) { char log[2048]; glGetProgramInfoLog(prog, sizeof log, NULL, log); MGL_FAIL("link: %s", log); } }
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "samp"), 0);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof vb, vb, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)tc_off);

    glGenBuffers(1, &xbuf);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);

    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    memcpy(out, glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, GL_MAP_READ_BIT), sizeof out);
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    printf("xfb col: %.3f %.3f %.3f %.3f (want 0.502 0.502 0.784 1.0)\n", out[0], out[1], out[2], out[3]);
    EXPECT_TRUE(out[0] > 0.4f && out[1] > 0.4f && out[2] > 0.7f);

    glDeleteBuffers(1, &xbuf); glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &tex); glDeleteProgram(prog); glDeleteShader(vs); glDeleteShader(fs);
}

int main(void) { RUN(XfbVsSampleMinimal); return mgl_test_summary(); }
