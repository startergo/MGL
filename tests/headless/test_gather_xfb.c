#include <stdio.h>
#include <string.h>
#include "glcorearb.h"
#include "mgl_test.h"

/* gpu_shader5 texture_gather_offset_*: VERTEX-stage textureGather on an
 * isampler2D captured through transform feedback — all-zero results. */

static const char *kVs =
    "#version 450 core\n"
    "uniform isampler2D samp;\n"
    "in vec2 texCoords;\n"
    "flat out ivec4 g0;\n"
    "flat out ivec4 g1;\n"
    "flat out ivec4 g2;\n"
    "flat out ivec4 r0;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(texCoords * 2.0 - 1.0, 0.0, 1.0);\n"
    "    g0 = textureGather     (samp, texCoords, 0);\n"
    "    g1 = textureGather     (samp, texCoords, 1);\n"
    "    g2 = ivec4(texture(samp, texCoords).xy, 99, 42);\n"
    "    r0 = ivec4(7, 8, 9, 10);\n"
    "}\n";

TEST(VertexGatherXfb)
{
    GLuint prog, vs, fs, tfo, buf, tex, vao, vbo;
    const char *varyings[] = {"g0", "g1", "g2", "r0"};
    const float coords[] = {0.25f, 0.25f, 0.75f, 0.25f, 0.25f, 0.75f, 0.75f, 0.75f};
    int data[16 * 16 * 4];
    int out[4][4 * 4];

    for (int i = 0; i < 16 * 16; i++)
    {
        data[i * 4 + 0] = i % 16;
        data[i * 4 + 1] = i / 16;
        data[i * 4 + 2] = 100 + (i % 16);
        data[i * 4 + 3] = 200 + (i / 16);
    }

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &kVs, NULL);
    glCompileShader(vs);
    { GLint ok = 0; glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
      if (!ok) { char log[4096]; glGetShaderInfoLog(vs, sizeof log, NULL, log);
                 MGL_FAIL("VS compile: %s", log); } }
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    { const char *f = "#version 450 core\nout vec4 c; void main(){c=vec4(1);}";
      glShaderSource(fs, 1, &f, NULL); glCompileShader(fs); }
    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glTransformFeedbackVaryings(prog, 4, varyings, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    { GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
      if (!ok) { char log[4096]; glGetProgramInfoLog(prog, sizeof log, NULL, log);
                 MGL_FAIL("link: %s", log); } }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    int floattex = getenv("MGL_FLOAT_TEX") != NULL;
    if (floattex)
    {
        unsigned char fdata[16 * 16 * 4];
        for (int i = 0; i < 16 * 16; i++)
        { fdata[i*4]=i%16*16; fdata[i*4+1]=i/16*16; fdata[i*4+2]=(i%16)*4; fdata[i*4+3]=255; }
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, fdata);
    }
    else
    {
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, 16, 16);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA_INTEGER, GL_INT, data);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "samp"), 0);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof coords, coords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &buf);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof out, NULL, GL_STREAM_READ);

    if (!getenv("MGL_NO_DISCARD")) glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 4);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    memcpy(out, glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, GL_MAP_READ_BIT), sizeof out);
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);

    for (int v = 0; v < 4; v++)
    {
        printf("v%d tex=(%.2f,%.2f) r0=(%d %d %d %d) g2=(%d %d %d %d) g0=(%d %d %d %d) g1=(%d %d %d %d)\n",
               v, coords[v * 2], coords[v * 2 + 1],
               out[v][0], out[v][1], out[v][2], out[v][3],
               out[v][4], out[v][5], out[v][6], out[v][7],
               out[v][8], out[v][9], out[v][10], out[v][11],
               out[v][12], out[v][13], out[v][14], out[v][15]);
    }
    /* interleaved order g0,g1,g2,r0: r0 is a constant in words 12..15 */
    EXPECT_EQ_INT(7, out[0][12]);
    EXPECT_EQ_INT(10, out[0][15]);
    /* g2 packs texture().xy + markers in words 8..11 */
    EXPECT_EQ_INT(99, out[0][10]);
    EXPECT_EQ_INT(42, out[0][11]);
    /* g0 (words 0..3) must hold gather texel values: vertex 0 tc=(0.25,0.25) */
    EXPECT_EQ_INT(3, out[0][0]);
    EXPECT_EQ_INT(4, out[0][1]);

    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

int main(void) { RUN(VertexGatherXfb); return mgl_test_summary(); }
