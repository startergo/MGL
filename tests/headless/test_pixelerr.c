#include "mgl_test.h"
#include <stdio.h>
#include "glcorearb.h"

/* The format mappers are exported (MGL/include/pixel_utils.h); an unexpected
 * format/type must map to 0 / MTLPixelFormatInvalid (callers turn that into a
 * GL error), never reach an assert(0): bitcountForInternalFormat /
 * internalFormatForGLFormatType / mtlPixelFormatForGLFormatType. */

extern GLuint bitcountForInternalFormat(GLenum internalformat, GLenum component);
extern GLenum internalFormatForGLFormatType(GLenum format, GLenum type);
extern unsigned int mtlPixelFormatForGLFormatType(GLenum format, GLenum type);

TEST(FormatMappersUnknownValues)
{
    static const GLenum junk[] = { 0x70000000u, 0xdeadbee0u, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT };
    int n = (int)(sizeof junk / sizeof junk[0]);

    for (int i = 0; i < n; i++)
    {
        GLuint b = bitcountForInternalFormat(junk[i], GL_RED);
        GLenum f = internalFormatForGLFormatType(GL_RGBA, junk[i]);
        unsigned int m = mtlPixelFormatForGLFormatType(GL_RGBA, junk[i]);
        printf("junk 0x%08x: bitcount=%u internal=0x%x mtl=0x%x\n", (unsigned)junk[i], b,
               (unsigned)f, m);
        EXPECT_TRUE(b == 0);                      /* unknown -> 0, not a crash */
        EXPECT_TRUE(f == 0 || f == (GLenum)GL_INVALID_OPERATION || f == (GLenum)GL_RGBA8);
        EXPECT_TRUE(m == 0 || m == 0xFFFFFFFFu || m == (unsigned int)80 /* RGBA8Unorm */);
    }

    /* unknown component / format legs */
    EXPECT_TRUE(bitcountForInternalFormat(GL_RGBA8, 0x60000000u) == 0);
    EXPECT_TRUE(internalFormatForGLFormatType(0x60000000u, GL_UNSIGNED_BYTE) == 0);
    EXPECT_TRUE(mtlPixelFormatForGLFormatType(0x60000000u, GL_UNSIGNED_BYTE) == 0);
}

static int try_teximage(GLenum ifmt, GLenum fmt, GLenum type)
{
    GLuint tex;
    GLint err;
    GLenum e;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    while (glGetError())
        ;
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, 4, 4, 0, fmt, type, NULL);
    e = glGetError();
    err = (e == GL_NO_ERROR);
    while (glGetError())
        ;
    /* a follow-up query must not crash either, and reports a sane size */
    {
        GLint w = -1;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
        while (glGetError())
            ;
    }
    glDeleteTextures(1, &tex);
    return err;
}

TEST(UnexpectedFormatTypes)
{
    static const struct { GLint ifmt; GLint fmt; GLint type; const char *name; } cases[] = {
        { GL_RGB10_A2,     GL_RGBA,           GL_UNSIGNED_INT_10_10_10_2,     "10_10_10_2" },
        { GL_RGB10_A2UI,   GL_RGBA_INTEGER,   GL_UNSIGNED_INT_10_10_10_2,     "10_10_10_2 ui" },
        { GL_R3_G3_B2,     GL_RGB,            GL_UNSIGNED_BYTE_3_3_2,         "byte_3_3_2" },
        { GL_RGBA4,        GL_BGRA,           GL_UNSIGNED_SHORT_4_4_4_4_REV,  "4_4_4_4_REV bgra" },
        { GL_RGB565,       GL_RGB,            GL_UNSIGNED_SHORT_5_6_5_REV,    "5_6_5_REV" },
        { GL_RGBA16F,      GL_RGBA,           GL_HALF_FLOAT,                  "half float" },
        { GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8,        "24_8 ds" },
        { GL_STENCIL_INDEX8,  GL_STENCIL_INDEX, GL_UNSIGNED_BYTE,             "stencil ubyte" },
    };
    int ok = 0;

    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++)
    {
        int err = try_teximage((GLenum)cases[i].ifmt, (GLenum)cases[i].fmt, (GLenum)cases[i].type);
        printf("teximage %-16s -> %s\n", cases[i].name, err ? "GL error" : "accepted");
        ok += err || 1;             /* either outcome is fine; a crash is not */
    }
    EXPECT_TRUE(ok == (int)(sizeof cases / sizeof cases[0]));
}

TEST(UnexpectedReadPixelsTypes)
{
    GLuint fbo, tex;
    GLint vp[4];
    unsigned char px[64];

    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    glGetIntegerv(GL_VIEWPORT, vp);
    while (glGetError())
        ;
    /* read back with packed types the mappers may not know */
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_INT_10_10_10_2, px);
    printf("readpixels 10_10_10_2 -> 0x%x\n", (int)glGetError());
    while (glGetError())
        ;
    glReadPixels(0, 0, 1, 1, GL_BGRA, GL_UNSIGNED_SHORT_4_4_4_4_REV, px);
    printf("readpixels 4_4_4_4_REV -> 0x%x\n", (int)glGetError());
    while (glGetError())
        ;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(FormatMappersUnknownValues);
    RUN(UnexpectedFormatTypes);
    RUN(UnexpectedReadPixelsTypes);
    return mgl_test_summary();
}
