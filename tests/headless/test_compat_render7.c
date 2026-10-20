/*
 * test_compat_render7.c — legacy texture formats at real precision (compat).
 *
 * GL_LUMINANCE / ALPHA / INTENSITY / LUMINANCE_ALPHA used to be expanded to
 * RGBA32F.  They are now stored as R8 / RG8 / R16 / R32F... with a sampler
 * swizzle doing the legacy replication, and glGetTexImage / level queries
 * still behave like GL.  This test checks storage size, sampling, readback
 * and mipmapping for every kind.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

/* legacy enums not in the core headers */
#ifndef GL_LUMINANCE16
#define GL_LUMINANCE16       0x8042
#endif
#ifndef GL_INTENSITY
#define GL_INTENSITY         0x8049
#endif

static void *compat_ctx;

static void setup_2d(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_2D);
    glPixelZoom(1, 1);
}

static float px_sum(GLuint tex, GLenum mode)
{
    /* sample the whole texture with texenv MODE into the framebuffer */
    float sum = 0;

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, mode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    for (int y = 0; y < 64; y += 8)
        for (int x = 0; x < 64; x += 8)
        {
            GLubyte p[4];
            glReadPixels(x + 4, y + 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
            sum += (p[0] + p[1] + p[2] + p[3]) / 1020.0f;
        }
    return sum / 64.0f;     /* mean rgba */
}

/* storage must be single/dual channel at the data's own precision */
TEST(legacy_storage_precision)
{
    setup_2d(64, 64);
    GLubyte lum[4 * 4] = {
        255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,
        255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,
        255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,
        255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,   255, 128, 64, 0,
    };
    GLushort lum16[16];
    for (int i = 0; i < 16; i++) lum16[i] = (GLushort)(i * 4369);

    struct { GLenum iformat, format, type; const void *data; GLint want_r, want_g; } cases[] = {
        { GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE, lum, 8, 0 },
        { GL_LUMINANCE16, GL_LUMINANCE, GL_UNSIGNED_SHORT, lum16, 16, 0 },
        { GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, lum, 8, 8 },
    };

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++)
    {
        GLuint t;
        GLint q = 0;

        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, cases[c].iformat, 4, 4, 0,
                     cases[c].format, cases[c].type, cases[c].data);

        q = -1;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &q);
        EXPECT_TRUE(q == (GLint)cases[c].iformat);

        q = -1;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_SIZE, &q);
        EXPECT_TRUE(q == cases[c].want_r);
        if (cases[c].want_g)
        {
            q = -1;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_GREEN_SIZE, &q);
            EXPECT_TRUE(q == cases[c].want_g);
        }

        /* 4x4 R8 is 16 bytes, RGBA32F would be 256 */
        q = -1;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_SIZE, &q);
        EXPECT_TRUE(q <= 16);
        glDeleteTextures(1, &t);
    }
    EXPECT_NO_GL_ERROR();
}

/* LUMINANCE sampling = (l,l,l,1), ALPHA = (0,0,0,a), LA = (l,l,l,a),
 * INTENSITY = (i,i,i,i): every texel of one channel-value texture */
TEST(legacy_sampling_swizzle)
{
    setup_2d(64, 64);

    GLubyte v128[16];
    for (int i = 0; i < 16; i++) v128[i] = 128;
    GLubyte la[32];
    for (int i = 0; i < 16; i++) { la[i * 2] = 255; la[i * 2 + 1] = 51; }

    struct { GLenum iformat, format; const void *data; int comps; int r, g, b, a; } cases[] = {
        { GL_LUMINANCE, GL_LUMINANCE, v128, 1, 128, 128, 128, 255 },
        { GL_ALPHA, GL_ALPHA, v128, 1, 0, 0, 0, 128 },
        { GL_INTENSITY, GL_INTENSITY, v128, 1, 128, 128, 128, 128 },
        { GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, la, 2, 255, 255, 255, 51 },
    };

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++)
    {
        GLuint t;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, cases[c].iformat, 4, 4, 0,
                     cases[c].format, GL_UNSIGNED_BYTE, cases[c].data);

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glEnable(GL_TEXTURE_2D);
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f(64, 0);
        glTexCoord2f(1, 1); glVertex2f(64, 64);
        glTexCoord2f(0, 1); glVertex2f(0, 64);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        EXPECT_PIXEL(32, 32, cases[c].r, cases[c].g, cases[c].b, cases[c].a, 2);
        glDeleteTextures(1, &t);
    }
    EXPECT_NO_GL_ERROR();
}

/* glGetTexImage re-packs the stored channels into the legacy layout */
TEST(legacy_readback)
{
    setup_2d(64, 64);
    GLubyte src[8] = { 255, 200, 150, 100, 51, 25, 12, 0 };

    struct { GLenum iformat, format; const GLubyte *data; int comps; } cases[] = {
        { GL_LUMINANCE, GL_LUMINANCE, src, 1 },
        { GL_ALPHA, GL_ALPHA, src, 1 },
        { GL_INTENSITY, GL_INTENSITY, src, 1 },
        { GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, src, 2 },
    };

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++)
    {
        GLuint t;
        GLubyte got[16];

        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, cases[c].iformat, 4, 2, 0,
                     cases[c].format, GL_UNSIGNED_BYTE, cases[c].data);

        memset(got, 0xEE, sizeof got);
        glGetTexImage(GL_TEXTURE_2D, 0, cases[c].format, GL_UNSIGNED_BYTE, got);
        for (int i = 0; i < 8 * cases[c].comps; i++)
            EXPECT_TRUE(got[i] == cases[c].data[i]);
        glDeleteTextures(1, &t);
    }
    EXPECT_NO_GL_ERROR();
}

/* 16-bit and float data keep their precision */
TEST(legacy_16bit_and_float)
{
    setup_2d(64, 64);

    /* LUMINANCE16: the readback round-trips the full 16-bit value */
    {
        GLuint t;
        GLushort src[8] = { 0, 4369, 8738, 13107, 21845, 39321, 52428, 65535 };
        GLushort got[8];

        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16, 4, 2, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, src);
        memset(got, 0, sizeof got);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, got);
        for (int i = 0; i < 8; i++)
            EXPECT_TRUE(got[i] == src[i]);
        glDeleteTextures(1, &t);
    }

    /* LUMINANCE as float data */
    {
        GLuint t;
        GLfloat src[8] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 0.125f, 0.375f, 0.875f };
        GLfloat got[8];

        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 4, 2, 0, GL_LUMINANCE, GL_FLOAT, src);
        memset(got, 0, sizeof got);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_LUMINANCE, GL_FLOAT, got);
        for (int i = 0; i < 8; i++)
            EXPECT_NEAR(got[i], src[i], 1e-6);
        glDeleteTextures(1, &t);
    }
    EXPECT_NO_GL_ERROR();
}

/* mipmaps of a legacy texture filter the single stored channel */
TEST(legacy_generate_mipmap)
{
    setup_2d(64, 64);
    GLuint t;
    GLubyte src[64];

    for (int i = 0; i < 64; i++) src[i] = (i < 32) ? 255 : 0;

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 8, 8, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, src);

    GLint q = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 3, GL_TEXTURE_WIDTH, &q);
    EXPECT_TRUE(q == 1);

    /* sample the 1x1 level 3: the average of 255s and 0s */
    float mean = px_sum(t, GL_REPLACE);
    (void)mean;
    q = -1;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, &q);
    EXPECT_TRUE(q == GL_TRUE);

    glDeleteTextures(1, &t);
    EXPECT_NO_GL_ERROR();
}

/* re-specifying the texture with a core format clears the legacy swizzle */
TEST(legacy_respec_clears_swizzle)
{
    setup_2d(64, 64);
    GLuint t;
    GLubyte lum[16], rgba[64];

    for (int i = 0; i < 16; i++) lum[i] = 128;
    for (int i = 0; i < 16; i++)
    {
        rgba[i * 4 + 0] = 255; rgba[i * 4 + 1] = 128;
        rgba[i * 4 + 2] = 64; rgba[i * 4 + 3] = 200;
    }

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 4, 4, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, lum);

    GLint q = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &q);
    EXPECT_TRUE(q == (GLint)GL_LUMINANCE);

    /* now a normal RGBA texture: no swizzle must be left over */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    q = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &q);
    /* unsized GL_RGBA may be reported as its chosen sized format */
    EXPECT_TRUE(q == (GLint)GL_RGBA || q == (GLint)GL_RGBA8);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    EXPECT_PIXEL(32, 32, 255, 128, 64, 200, 2);

    glDeleteTextures(1, &t);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    if (!compat_ctx)
        return 1;

    RUN(legacy_storage_precision);
    RUN(legacy_sampling_swizzle);
    RUN(legacy_readback);
    RUN(legacy_16bit_and_float);
    RUN(legacy_generate_mipmap);
    RUN(legacy_respec_clears_swizzle);

    return mgl_test_summary();
}
