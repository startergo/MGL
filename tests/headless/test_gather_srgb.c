#include <stdio.h>
#include <string.h>
#include "glcorearb.h"
#include "mgl_test.h"

/* KHR-GL46.texture_gather plain-gather-float-2d-srgb regression: a FRAGMENT-stage
 * textureGather on an SRGB8_ALPHA8 texture must return sRGB-DECODED texels.
 * MGL returned the raw (encoded) values. */

static const char *kVs =
    "#version 460 core\n"
    "layout(location=0) in vec4 p;\n"
    "out vec2 uv;\n"
    "void main() { uv = p.xy * 0.5 + 0.5; gl_Position = p; }\n";
static const char *kFs =
    "#version 460 core\n"
    "uniform sampler2D my_sampler;\n"
    "in vec2 uv;\n"
    "layout(location=0) out vec4 o;\n"
    "void main() { o = textureGather(my_sampler, uv); }\n";
static const char *kVsG =
    "#version 460 core\n"
    "uniform sampler2D my_sampler;\n"
    "layout(location=0) in vec4 p;\n"
    "out vec4 col;\n"
    "void main() { col = textureGather(my_sampler, p.xy * 0.5 + 0.5); gl_Position = p; }\n";
static const char *kFsG =
    "#version 460 core\n"
    "in vec4 col;\n"
    "layout(location=0) out vec4 o;\n"
    "void main() { o = col; }\n";

TEST(FragmentGatherSrgb)
{
    GLuint prog, vs, fs, tex, vao, vbo;
    const float quad[] = { -1,-1,0,1, 1,-1,0,1, -1,1,0,1, 1,1,0,1 };
    float fdata[4 * 4 * 4];
    unsigned char data[4 * 4 * 4];  /* every texel = (64,128,192,255) */
    float px[4];

    memset(data, 0, sizeof data);
    for (int i = 0; i < 16; i++)
    { data[i*4+0] = 64; data[i*4+1] = 128; data[i*4+2] = 192; data[i*4+3] = 255;
      fdata[i*4+0] = 64/255.0f; fdata[i*4+1] = 128/255.0f; fdata[i*4+2] = 192/255.0f; fdata[i*4+3] = 1.0f; }

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, getenv("MGL_VS_GATHER") ? &kVsG : &kVs, NULL); glCompileShader(vs);
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    if (getenv("MGL_VS_GATHER"))
        glShaderSource(fs, 1, &kFsG, NULL);
    else
        glShaderSource(fs, 1, &kFs, NULL);
    glCompileShader(fs);
    prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs); glLinkProgram(prog);
    { GLint ok=0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
      if (!ok) { char log[2048]; glGetProgramInfoLog(prog, sizeof log, NULL, log); MGL_FAIL("link: %s", log); } }
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "my_sampler"), 0);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    if (getenv("MGL_MUTABLE"))
    {
        /* the CTS way: glTexImage2D (mutable) + glTexSubImage2D + mipmaps */
        int size = 32;
        for (int i = 0; size > 0; i++, size /= 2)
            glTexImage2D(GL_TEXTURE_2D, i, GL_SRGB8_ALPHA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClearTexImage(tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, getenv("MGL_FLOAT_UP") ? GL_FLOAT : GL_UNSIGNED_BYTE, getenv("MGL_FLOAT_UP") ? (void *)fdata : (void *)data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, 4, 4);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, getenv("MGL_FLOAT_UP") ? GL_FLOAT : GL_UNSIGNED_BYTE, getenv("MGL_FLOAT_UP") ? (void *)fdata : (void *)data);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glViewport(0, 0, 4, 4);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    glReadPixels(2, 2, 1, 1, GL_RGBA, GL_FLOAT, px);
    if (getenv("MGL_MUTABLE"))
    {
        printf("cts-like gather on sRGB: %.4f %.4f %.4f %.4f (want 0 0.0513 0.2158 0.5195)\n",
               px[0], px[1], px[2], px[3]);
        EXPECT_NEAR(px[0], 0.0f, 0.01);
        EXPECT_NEAR(px[1], 0.0513f, 0.01);
        EXPECT_NEAR(px[2], 0.2158f, 0.01);
        EXPECT_NEAR(px[3], 0.5195f, 0.02);
    }
    else
    {
    printf("fragment gather on sRGB: %.4f %.4f %.4f %.4f (want ~0.0513 0.2158 0.5195 1.0)\n",
           px[1], px[1], px[1], px[1]);
    EXPECT_NEAR(px[0], 0.0513, 0.01);
    }
    (void)data;

    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &tex); glDeleteProgram(prog); glDeleteShader(vs); glDeleteShader(fs);
}

int main(void) { RUN(FragmentGatherSrgb); return mgl_test_summary(); }
