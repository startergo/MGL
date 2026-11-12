#include "mgl_test.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <dlfcn.h>

/* mglTextureIOSurface2D: a GL_TEXTURE_2D gets single-level storage on an
 * IOSurface (host presentation).  Checked:
 *  - same-shape conversion keeps the content, whether it is still a pending
 *    CPU upload or already GPU resident (rendered);
 *  - GL draws into the texture land in the surface (also through an FBO that
 *    was attached before the conversion);
 *  - unsupported formats/targets return NULL and leave the texture untouched;
 *  - a shape change redefines the storage (immutable, one level).
 * The IOSurface API is resolved with dlsym so the test needs no extra
 * framework flags. */

extern void *mglTextureIOSurface2D(GLuint texture, GLenum internalformat, GLsizei width, GLsizei height);

typedef int (*lock_fn)(void *, uint32_t, uint32_t *);
typedef void *(*base_fn)(void *);
typedef size_t (*size_fn)(void *);
typedef void (*rel_fn)(const void *);
static lock_fn s_lock, s_unlock;
static base_fn s_base;
static size_fn s_bpr, s_w, s_h;
static rel_fn s_release;

static bool load_iosurface(void)
{
    dlopen("/System/Library/Frameworks/IOSurface.framework/IOSurface", RTLD_NOW);
    s_lock = (lock_fn)dlsym(RTLD_DEFAULT, "IOSurfaceLock");
    s_unlock = (lock_fn)dlsym(RTLD_DEFAULT, "IOSurfaceUnlock");
    s_base = (base_fn)dlsym(RTLD_DEFAULT, "IOSurfaceGetBaseAddress");
    s_bpr = (size_fn)dlsym(RTLD_DEFAULT, "IOSurfaceGetBytesPerRow");
    s_w = (size_fn)dlsym(RTLD_DEFAULT, "IOSurfaceGetWidth");
    s_h = (size_fn)dlsym(RTLD_DEFAULT, "IOSurfaceGetHeight");
    s_release = (rel_fn)dlsym(RTLD_DEFAULT, "CFRelease");
    return s_lock && s_unlock && s_base && s_bpr && s_w && s_h && s_release;
}

/* texel (x,y) of the surface, RGBA byte order ('RGBA' surface) */
static void surf_px(void *surf, int x, int y, uint8_t out[4])
{
    s_lock(surf, 1 /* kIOSurfaceLockReadOnly */, NULL);
    const uint8_t *row = (const uint8_t *)s_base(surf) + (size_t)y * s_bpr(surf);
    for (int i = 0; i < 4; i++) out[i] = row[x * 4 + i];
    s_unlock(surf, 1, NULL);
}

#define EXPECT_SURF(surf, x, y, r, g, b, a) do { uint8_t _p[4]; surf_px(surf, x, y, _p); mgl_test_checks++; \
        if (abs(_p[0]-(r)) > 1 || abs(_p[1]-(g)) > 1 || abs(_p[2]-(b)) > 1 || abs(_p[3]-(a)) > 1) \
            MGL_FAIL("surface(%d,%d) = %d,%d,%d,%d expected %d,%d,%d,%d", x, y, _p[0], _p[1], _p[2], _p[3], r, g, b, a); \
    } while (0)

static GLuint new_tex_pattern(GLsizei w, GLsizei h)
{
    GLubyte px[16 * 16 * 4];
    for (int i = 0; i < w * h; i++) {
        px[i * 4 + 0] = (GLubyte)(i * 13);
        px[i * 4 + 1] = (GLubyte)(255 - i * 7);
        px[i * 4 + 2] = 7;
        px[i * 4 + 3] = 255;
    }
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

static GLuint fbo_for(GLuint tex)
{
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return f;
}

TEST(Setup) { EXPECT_TRUE(load_iosurface()); }

/* the texture was only uploaded (CPU shadow pending): content carries */
TEST(KeepsPendingUpload)
{
    GLuint t = new_tex_pattern(4, 4);
    void *s = mglTextureIOSurface2D(t, GL_RGBA8, 4, 4);
    EXPECT_TRUE(s != NULL);
    if (!s) return;
    glFinish();
    EXPECT_EQ_INT(s_w(s), 4);
    EXPECT_EQ_INT(s_h(s), 4);
    EXPECT_SURF(s, 0, 0, 0, 255, 7, 255);
    EXPECT_SURF(s, 3, 0, 39, 234, 7, 255);
    EXPECT_SURF(s, 1, 2, 117, 192, 7, 255);   /* i = 9 */
    EXPECT_SURF(s, 3, 3, 195, 150, 7, 255);   /* i = 15 */
    GLint imm = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT, &imm);
    EXPECT_EQ_INT(imm, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    s_release(s);
    glDeleteTextures(1, &t);
}

/* the texture was rendered (GPU resident, mipmapped): level 0 carries,
 * the other levels go */
TEST(KeepsRenderedContent)
{
    GLuint t = new_tex_pattern(4, 4);
    glGenerateMipmap(GL_TEXTURE_2D);
    GLuint f = fbo_for(t);
    glViewport(0, 0, 4, 4);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    void *s = mglTextureIOSurface2D(t, GL_RGBA8, 4, 4);
    EXPECT_TRUE(s != NULL);
    if (!s) return;
    glFinish();
    EXPECT_SURF(s, 0, 0, 0, 0, 255, 255);
    EXPECT_SURF(s, 3, 3, 0, 0, 255, 255);
    GLint w1 = -1;
    glBindTexture(GL_TEXTURE_2D, t);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH, &w1);
    EXPECT_EQ_INT(w1, 0);

    /* draws through the FBO attached before the conversion reach the surface */
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    EXPECT_SURF(s, 2, 1, 255, 0, 0, 255);
    EXPECT_PIXEL(2, 1, 255, 0, 0, 255, 1);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &f);
    s_release(s);
    glDeleteTextures(1, &t);
}

/* a new FBO on the converted texture: clears and scissored clears land in the surface,
 * and glGetTexImage reads the same bytes */
TEST(DrawsReachSurface)
{
    GLuint t = new_tex_pattern(8, 8);
    void *s = mglTextureIOSurface2D(t, GL_RGBA8, 8, 8);
    EXPECT_TRUE(s != NULL);
    if (!s) return;
    GLuint f = fbo_for(t);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 8, 8);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 4, 4);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    glFinish();
    EXPECT_SURF(s, 1, 1, 255, 255, 255, 255);
    EXPECT_SURF(s, 6, 6, 0, 255, 0, 255);
    GLubyte img[8 * 8 * 4];
    glBindTexture(GL_TEXTURE_2D, t);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    mgl_test_checks++;
    if (img[(6 * 8 + 6) * 4 + 1] != 255 || img[(1 * 8 + 1) * 4 + 0] != 255)
        MGL_FAIL("glGetTexImage disagrees with the surface");
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &f);
    s_release(s);
    glDeleteTextures(1, &t);
}

/* unsupported shapes: NULL, and the texture keeps its storage and content */
TEST(RefusesAndLeavesTextureAlone)
{
    GLuint t = new_tex_pattern(4, 4);
    EXPECT_TRUE(mglTextureIOSurface2D(t, GL_DEPTH_COMPONENT32F, 4, 4) == NULL);
    EXPECT_TRUE(mglTextureIOSurface2D(t, GL_SRGB8_ALPHA8, 4, 4) == NULL);
    EXPECT_TRUE(mglTextureIOSurface2D(t, GL_RGBA16F, 4, 4) == NULL);
    EXPECT_TRUE(mglTextureIOSurface2D(t, GL_RGBA8, 0, 4) == NULL);
    GLint w = 0, fmt = 0, imm = 1;
    glBindTexture(GL_TEXTURE_2D, t);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT, &imm);
    EXPECT_EQ_INT(w, 4);
    EXPECT_EQ_INT(fmt, GL_RGBA8);
    EXPECT_EQ_INT(imm, GL_FALSE);
    GLubyte img[4 * 4 * 4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    mgl_test_checks++;
    if (img[9 * 4 + 0] != 117 || img[9 * 4 + 1] != 192)
        MGL_FAIL("content changed by a refused conversion: %d,%d", img[36], img[37]);

    GLuint cube;
    glGenTextures(1, &cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA8, 4, 4);
    EXPECT_TRUE(mglTextureIOSurface2D(cube, GL_RGBA8, 4, 4) == NULL);
    EXPECT_TRUE(mglTextureIOSurface2D(987654, GL_RGBA8, 4, 4) == NULL);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &cube);
    glDeleteTextures(1, &t);
}

/* shape change: new single-level immutable storage of the requested size */
TEST(ShapeChangeRedefines)
{
    GLuint t = new_tex_pattern(8, 8);
    glGenerateMipmap(GL_TEXTURE_2D);
    void *s = mglTextureIOSurface2D(t, GL_RGBA8, 16, 8);
    EXPECT_TRUE(s != NULL);
    if (!s) return;
    EXPECT_EQ_INT(s_w(s), 16);
    EXPECT_EQ_INT(s_h(s), 8);
    GLint w = 0, h = 0, w1 = -1, imm = 0;
    glBindTexture(GL_TEXTURE_2D, t);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH, &w1);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT, &imm);
    EXPECT_EQ_INT(w, 16);
    EXPECT_EQ_INT(h, 8);
    EXPECT_EQ_INT(w1, 0);
    EXPECT_EQ_INT(imm, GL_TRUE);
    GLuint f = fbo_for(t);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 16, 8);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    EXPECT_SURF(s, 15, 7, 0, 0, 255, 255);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &f);
    s_release(s);
    glDeleteTextures(1, &t);
}

int main(void)
{
    RUN(Setup);
    RUN(KeepsPendingUpload);
    RUN(KeepsRenderedContent);
    RUN(DrawsReachSurface);
    RUN(RefusesAndLeavesTextureAlone);
    RUN(ShapeChangeRedefines);
    return mgl_test_summary();
}
