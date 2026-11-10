/*
 * test_compat_srgbdecode.c — GL_EXT_texture_sRGB_decode (compat domain).
 *
 * GL_TEXTURE_SRGB_DECODE_EXT = GL_DECODE_EXT (default) / GL_SKIP_DECODE_EXT on
 * textures AND sampler objects; SKIP_DECODE sampling must return the RAW
 * (non-linear) stored values.  Metal has no sampler flag for this, so MGL binds
 * a plain (non-sRGB) view of the texture's storage at the sampled-image bind
 * sites (compatSRGBDecodeViewForTex, MGLRenderer+Compat.m).
 *
 * State checks mirror the CTS state_query/negative_coverage slices: default
 * value, every setter family, query-back on both objects, GL_INVALID_ENUM for
 * out-of-range values, plus a pixel-exact sampling check for decode/skip
 * through both the texture parameter and a bound sampler object.
 */
#include "mgl_test.h"
#include <string.h>

#ifndef GL_TEXTURE_SRGB_DECODE_EXT
#define GL_TEXTURE_SRGB_DECODE_EXT 0x8A48
#endif
#ifndef GL_DECODE_EXT
#define GL_DECODE_EXT 0x8A49
#endif
#ifndef GL_SKIP_DECODE_EXT
#define GL_SKIP_DECODE_EXT 0x8A4A
#endif

#define W 4
#define H 4

/* sRGB decode of 128/255 = 0.2159..; on RGBA8 readback the SKIP sample is the
 * raw byte (128), the DECODE sample of the same byte is 128*srgb -> 55 */
#define RAW_BYTE 128
#define DECODE_BYTE 55

static GLuint make_prog(void)
{
    static const char *vs =
        "#version 150\n"
        "void main() {\n"
        "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 150\n"
        "uniform sampler2D uTex;\n"
        "layout (location = 0) out vec4 o;\n"
        "void main() { o = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

/* upload a uniform raw byte through an sRGB texture */
static GLuint make_srgb_tex(void)
{
    GLuint tex;
    GLubyte data[W * H * 4];
    memset(data, RAW_BYTE, sizeof data);
    for (int i = 0; i < W * H; i++) data[i * 4 + 3] = 255;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    return tex;
}

static int sample_byte(GLuint prog)
{
    GLubyte c[4];
    glFinish();
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    return c[0];
}

static void draw(void)
{
    static GLuint vao = 0;
    if (!vao) glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, W, H);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

TEST(texparam_state)
{
    GLuint tex;
    GLint i = -1;
    GLfloat f = -1.0f;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    /* initial value: DECODE */
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_DECODE_EXT);

    /* every setter family round-trips */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
    EXPECT_NO_GL_ERROR();
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_SKIP_DECODE_EXT);

    i = GL_DECODE_EXT;
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_NO_GL_ERROR();
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_DECODE_EXT);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLfloat)GL_SKIP_DECODE_EXT);
    EXPECT_NO_GL_ERROR();
    glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &f);
    EXPECT_EQ_INT((int)f, GL_SKIP_DECODE_EXT);

    f = (GLfloat)GL_DECODE_EXT;
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &f);
    EXPECT_NO_GL_ERROR();
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_DECODE_EXT);

    /* pure-integer setters */
    glTexParameterIiv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_SKIP_DECODE_EXT});
    EXPECT_NO_GL_ERROR();
    glGetTexParameterIiv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_SKIP_DECODE_EXT);
    glTexParameterIuiv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLuint[]){GL_DECODE_EXT});
    EXPECT_NO_GL_ERROR();
    {
        GLuint u = 0;
        glGetTexParameterIuiv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &u);
        EXPECT_EQ_INT((int)u, GL_DECODE_EXT);
    }

    /* INVALID_ENUM for any other value (all six entry points) */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, -1);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, -1.0f);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_LINEAR});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLfloat[]){GL_NEAREST});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameterIiv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_DONT_CARE});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameterIuiv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, (GLuint[]){GL_DONT_CARE});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* the failed sets must not change the value */
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_DECODE_EXT);

    glDeleteTextures(1, &tex);
}

TEST(sampler_state)
{
    GLuint s;
    GLint i = -1;

    glGenSamplers(1, &s);

    /* initial value + round trip */
    glGetSamplerParameteriv(s, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_DECODE_EXT);
    glSamplerParameteri(s, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
    EXPECT_NO_GL_ERROR();
    glGetSamplerParameteriv(s, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_SKIP_DECODE_EXT);
    glSamplerParameterf(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLfloat)GL_DECODE_EXT);
    EXPECT_NO_GL_ERROR();
    {
        GLfloat f = -1.0f;
        glGetSamplerParameterfv(s, GL_TEXTURE_SRGB_DECODE_EXT, &f);
        EXPECT_EQ_INT((int)f, GL_DECODE_EXT);
    }
    glSamplerParameteriv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_SKIP_DECODE_EXT});
    EXPECT_NO_GL_ERROR();
    glSamplerParameterIiv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_DECODE_EXT});
    EXPECT_NO_GL_ERROR();
    glSamplerParameterIuiv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLuint[]){GL_SKIP_DECODE_EXT});
    EXPECT_NO_GL_ERROR();
    glGetSamplerParameteriv(s, GL_TEXTURE_SRGB_DECODE_EXT, &i);
    EXPECT_EQ_INT(i, GL_SKIP_DECODE_EXT);

    /* INVALID_ENUM for other values */
    glSamplerParameteri(s, GL_TEXTURE_SRGB_DECODE_EXT, -1);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameterf(s, GL_TEXTURE_SRGB_DECODE_EXT, -1.0f);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameteriv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_LINEAR});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameterfv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLfloat[]){GL_NEAREST});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameterIiv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLint[]){GL_DONT_CARE});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameterIuiv(s, GL_TEXTURE_SRGB_DECODE_EXT, (GLuint[]){GL_DONT_CARE});
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glDeleteSamplers(1, &s);
}

TEST(sample_decode_skip_texture_param)
{
    GLuint tex = make_srgb_tex();
    GLuint prog = make_prog();

    glBindTexture(GL_TEXTURE_2D, tex);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    /* default DECODE: sRGB -> linear */
    draw();
    EXPECT_EQ_INT(sample_byte(prog), DECODE_BYTE);

    /* texture-level SKIP_DECODE: raw bytes */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
    draw();
    EXPECT_EQ_INT(sample_byte(prog), RAW_BYTE);

    /* back to DECODE */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, GL_DECODE_EXT);
    draw();
    EXPECT_EQ_INT(sample_byte(prog), DECODE_BYTE);

    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);
}

TEST(sample_skip_sampler_object)
{
    GLuint tex = make_srgb_tex();
    GLuint prog = make_prog();
    GLuint smp;

    glGenSamplers(1, &smp);
    glBindSampler(0, smp);
    glSamplerParameteri(smp, GL_TEXTURE_MIN_FILTER, GL_NEAREST);    /* else the
     * default NEAREST_MIPMAP_LINEAR makes the single-mip texture incomplete */
    glSamplerParameteri(smp, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    /* the sampler object's state replaces the texture's: SKIP on the sampler
     * skips even though the texture says DECODE */
    glSamplerParameteri(smp, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
    draw();
    EXPECT_EQ_INT(sample_byte(prog), RAW_BYTE);

    /* sampler says DECODE again */
    glSamplerParameteri(smp, GL_TEXTURE_SRGB_DECODE_EXT, GL_DECODE_EXT);
    draw();
    EXPECT_EQ_INT(sample_byte(prog), DECODE_BYTE);

    /* the texture's own SKIP state is replaced by the bound sampler's DECODE */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
    draw();
    EXPECT_EQ_INT(sample_byte(prog), DECODE_BYTE);

    /* unbinding the sampler object restores the texture's own state */
    glBindSampler(0, 0);
    draw();
    EXPECT_EQ_INT(sample_byte(prog), RAW_BYTE);

    glDeleteProgram(prog);
    glDeleteSamplers(1, &smp);
    glDeleteTextures(1, &tex);
}

TEST(sample_two_units_both_modes)
{
    /* the same texture on two units with different decode settings: the
     * per-unit resolve must not let one unit's view win for the other */
    GLuint tex = make_srgb_tex();
    GLuint tex2 = make_srgb_tex();      /* a second sRGB texture (always DECODE) */
    GLuint prog;
    static const char *vs =
        "#version 150\n"
        "void main() {\n"
        "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 150\n"
        "uniform sampler2D uSkip;\n"
        "uniform sampler2D uDecode;\n"
        "layout (location = 0) out vec4 o;\n"
        "void main() { o = texelFetch(uSkip, ivec2(gl_FragCoord.xy), 0) +\n"
        "                    texelFetch(uDecode, ivec2(gl_FragCoord.xy), 0); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    prog = glCreateProgram();
    glAttachShader(prog, v); glAttachShader(prog, f);
    glLinkProgram(prog);
    glDeleteShader(v); glDeleteShader(f);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex2);     /* stays DECODE */

    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uSkip"), 0);
    glUniform1i(glGetUniformLocation(prog, "uDecode"), 1);
    draw();

    /* raw 128 + decoded 55 -> clamp of 183 */
    EXPECT_EQ_INT(sample_byte(prog), 183);

    glActiveTexture(GL_TEXTURE0);
    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &tex2);
}

int main(void)
{
    RUN(texparam_state);
    RUN(sampler_state);
    RUN(sample_decode_skip_texture_param);
    RUN(sample_skip_sampler_object);
    RUN(sample_two_units_both_modes);
    return mgl_test_summary();
}
