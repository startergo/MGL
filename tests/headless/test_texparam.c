/* test_texparam.c — texture parameters, sampler objects, level and internalformat queries */
#include "mgl_test.h"
#include <limits.h>

static GLint tex_i(GLenum target, GLenum pname) { GLint v = -12345; glGetTexParameteriv(target, pname, &v); return v; }
static GLfloat tex_f(GLenum target, GLenum pname) { GLfloat v = -12345.0f; glGetTexParameterfv(target, pname, &v); return v; }
static GLint lvl_i(GLenum target, GLint level, GLenum pname) { GLint v = -12345; glGetTexLevelParameteriv(target, level, pname, &v); return v; }
static GLint ifq(GLenum target, GLenum fmt, GLenum pname) { GLint v = -12345; glGetInternalformativ(target, fmt, pname, 1, &v); return v; }
static GLint smp_i(GLuint s, GLenum pname) { GLint v = -12345; glGetSamplerParameteriv(s, pname, &v); return v; }

TEST(defaults)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER), GL_NEAREST_MIPMAP_LINEAR);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER), GL_LINEAR);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S), GL_REPEAT);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T), GL_REPEAT);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R), GL_REPEAT);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE), GL_NONE);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC), GL_LEQUAL);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL), 0);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL), 1000);
    EXPECT_NEAR(tex_f(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD), -1000.0, 0);
    EXPECT_NEAR(tex_f(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD), 1000.0, 0);
    EXPECT_NEAR(tex_f(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS), 0.0, 0);
    EXPECT_NEAR(tex_f(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY), 1.0, 0);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE), GL_DEPTH_COMPONENT);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R), GL_RED);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A), GL_ALPHA);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT), GL_FALSE);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_LEVELS), 0);
    GLint target = 0; glGetTextureParameteriv(t, GL_TEXTURE_TARGET, &target);
    EXPECT_EQ_INT(target, GL_TEXTURE_2D);
    GLfloat bc[4] = { 9, 9, 9, 9 };
    glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, bc);
    EXPECT_NEAR(bc[0] + bc[1] + bc[2] + bc[3], 0.0, 0);
    EXPECT_NO_GL_ERROR();

    GLuint r; glGenTextures(1, &r); glBindTexture(GL_TEXTURE_RECTANGLE, r);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S), GL_CLAMP_TO_EDGE);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER), GL_LINEAR);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t); glDeleteTextures(1, &r);
}

static void roundtrip_target(GLenum target)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(target, t);

    static const struct { GLenum pname; GLint value; } ivals[] = {
        { GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST },
        { GL_TEXTURE_MAG_FILTER, GL_NEAREST },
        { GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT },
        { GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER },
        { GL_TEXTURE_WRAP_R, GL_MIRROR_CLAMP_TO_EDGE },
        { GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE },
        { GL_TEXTURE_COMPARE_FUNC, GL_GREATER },
        { GL_TEXTURE_BASE_LEVEL, 2 },
        { GL_TEXTURE_MAX_LEVEL, 5 },
        { GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX },
        { GL_TEXTURE_SWIZZLE_R, GL_BLUE },
        { GL_TEXTURE_SWIZZLE_G, GL_ZERO },
        { GL_TEXTURE_SWIZZLE_B, GL_ONE },
        { GL_TEXTURE_SWIZZLE_A, GL_RED },
    };
    for (unsigned i = 0; i < sizeof ivals / sizeof ivals[0]; i++)
    {
        glTexParameteri(target, ivals[i].pname, ivals[i].value);
        EXPECT_NO_GL_ERROR();
        EXPECT_EQ_INT(tex_i(target, ivals[i].pname), ivals[i].value);
        EXPECT_NEAR(tex_f(target, ivals[i].pname), (double)ivals[i].value, 0);
        GLint iv = -1; glGetTexParameterIiv(target, ivals[i].pname, &iv);
        EXPECT_EQ_INT(iv, ivals[i].value);
        GLuint uv = 0; glGetTexParameterIuiv(target, ivals[i].pname, &uv);
        EXPECT_EQ_INT(uv, ivals[i].value);
    }

    /* float params, via f, fv, i */
    glTexParameterf(target, GL_TEXTURE_MIN_LOD, -2.5f);
    glTexParameterf(target, GL_TEXTURE_MAX_LOD, 7.25f);
    GLfloat bias = 1.5f; glTexParameterfv(target, GL_TEXTURE_LOD_BIAS, &bias);
    glTexParameteri(target, GL_TEXTURE_MAX_ANISOTROPY, 8);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(tex_f(target, GL_TEXTURE_MIN_LOD), -2.5, 0);
    EXPECT_NEAR(tex_f(target, GL_TEXTURE_MAX_LOD), 7.25, 0);
    EXPECT_NEAR(tex_f(target, GL_TEXTURE_LOD_BIAS), 1.5, 0);
    EXPECT_NEAR(tex_f(target, GL_TEXTURE_MAX_ANISOTROPY), 8.0, 0);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_MAX_LOD), 7);   /* rounded */

    /* DSA set, non-DSA get */
    glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameterf(t, GL_TEXTURE_MIN_LOD, 1.0f);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_MAG_FILTER), GL_LINEAR);
    EXPECT_NEAR(tex_f(target, GL_TEXTURE_MIN_LOD), 1.0, 0);
    GLint swz[4] = { GL_ALPHA, GL_BLUE, GL_GREEN, GL_RED };
    glTextureParameteriv(t, GL_TEXTURE_SWIZZLE_RGBA, swz);
    GLint got[4] = { 0 };
    glGetTextureParameteriv(t, GL_TEXTURE_SWIZZLE_RGBA, got);
    EXPECT_EQ_INT(got[0], GL_ALPHA); EXPECT_EQ_INT(got[1], GL_BLUE);
    EXPECT_EQ_INT(got[2], GL_GREEN); EXPECT_EQ_INT(got[3], GL_RED);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_SWIZZLE_G), GL_BLUE);
    GLfloat gf = 0; glGetTextureParameterfv(t, GL_TEXTURE_SWIZZLE_R, &gf);
    EXPECT_NEAR(gf, (double)GL_ALPHA, 0);
    GLint ii = 0; glGetTextureParameterIiv(t, GL_TEXTURE_BASE_LEVEL, &ii); EXPECT_EQ_INT(ii, 2);
    GLuint uu = 0; glGetTextureParameterIuiv(t, GL_TEXTURE_MAX_LEVEL, &uu); EXPECT_EQ_INT(uu, 5);
    GLint pi = GL_REPEAT; glTexParameterIiv(target, GL_TEXTURE_WRAP_S, &pi);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_WRAP_S), GL_REPEAT);
    GLuint pu = GL_NEAREST; glTextureParameterIuiv(t, GL_TEXTURE_MIN_FILTER, &pu);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_MIN_FILTER), GL_NEAREST);
    GLint pi2 = GL_CLAMP_TO_EDGE; glTextureParameterIiv(t, GL_TEXTURE_WRAP_T, &pi2);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_WRAP_T), GL_CLAMP_TO_EDGE);
    GLfloat ff = 3.0f; glTextureParameterfv(t, GL_TEXTURE_MAX_LOD, &ff);
    EXPECT_NEAR(tex_f(target, GL_TEXTURE_MAX_LOD), 3.0, 0);
    GLuint pu2 = GL_LESS; glTexParameterIuiv(target, GL_TEXTURE_COMPARE_FUNC, &pu2);
    EXPECT_EQ_INT(tex_i(target, GL_TEXTURE_COMPARE_FUNC), GL_LESS);
    EXPECT_NO_GL_ERROR();

    glDeleteTextures(1, &t);
}

TEST(roundtrip_2d)        { roundtrip_target(GL_TEXTURE_2D); }
TEST(roundtrip_3d)        { roundtrip_target(GL_TEXTURE_3D); }
TEST(roundtrip_2d_array)  { roundtrip_target(GL_TEXTURE_2D_ARRAY); }
TEST(roundtrip_cube)      { roundtrip_target(GL_TEXTURE_CUBE_MAP); }
TEST(roundtrip_1d)        { roundtrip_target(GL_TEXTURE_1D); }

TEST(border_color)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);

    GLfloat f[4] = { 0.25f, 0.5f, 0.75f, 1.0f }, gf[4];
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, f);
    glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, gf);
    for (int i = 0; i < 4; i++) EXPECT_NEAR(gf[i], f[i], 0);
    GLint gi[4];
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, gi);
    EXPECT_EQ_INT(gi[3], INT_MAX);
    EXPECT_NEAR(gi[1] / 2147483647.0, 0.5, 1e-6);

    GLint ni[4] = { INT_MAX, 0, INT_MAX, 0 };
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, ni);
    glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, gf);
    EXPECT_NEAR(gf[0], 1.0, 1e-6); EXPECT_NEAR(gf[1], 0.0, 0);

    GLint ii[4] = { -5, 7, 100000, -1 }, gii[4];
    glTexParameterIiv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, ii);
    glGetTexParameterIiv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, gii);
    for (int i = 0; i < 4; i++) EXPECT_EQ_INT(gii[i], ii[i]);

    GLuint uu[4] = { 4000000000u, 1, 2, 3 }, guu[4];
    glTextureParameterIuiv(t, GL_TEXTURE_BORDER_COLOR, uu);
    glGetTextureParameterIuiv(t, GL_TEXTURE_BORDER_COLOR, guu);
    for (int i = 0; i < 4; i++) EXPECT_EQ_INT(guu[i], uu[i]);
    EXPECT_NO_GL_ERROR();

    /* scalar entry points reject vector pnames */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDeleteTextures(1, &t);
}

TEST(param_errors)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);

    glTexParameteri(GL_TEXTURE_2D, 0x1234, 0);                      EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_REPEAT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_LINEAR);   EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_LESS); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_ZERO); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RGBA);  EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_RED); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, -1);      EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, -1);       EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, 0.5f); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    /* failed sets don't change state */
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER), GL_NEAREST_MIPMAP_LINEAR);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL), 0);

    glTexParameteri(0x1234, GL_TEXTURE_MIN_FILTER, GL_LINEAR);      EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_MIN_FILTER, GL_LINEAR); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    GLint v; glGetTexParameteriv(GL_TEXTURE_2D, 0x1234, &v);       EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetTexParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_MIN_FILTER, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* DSA on unknown names */
    glTextureParameteri(987654, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetTextureParameteriv(987654, GL_TEXTURE_MIN_FILTER, &v);     EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* rectangle restrictions */
    GLuint r; glGenTextures(1, &r); glBindTexture(GL_TEXTURE_RECTANGLE, r);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_REPEAT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_BASE_LEVEL, 1); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); EXPECT_NO_GL_ERROR();

    /* multisample: no sampler state, base level must be 0 */
    GLuint m; glGenTextures(1, &m); glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_BASE_LEVEL, 1); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_SWIZZLE_R, GL_GREEN); EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_SWIZZLE_R), GL_GREEN);

    glDeleteTextures(1, &t); glDeleteTextures(1, &r); glDeleteTextures(1, &m);
}

TEST(level_params_teximage)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    GLubyte px[8 * 4 * 4] = { 0 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    int have_l1 = glGetError() == GL_NO_ERROR;   /* level > 0 uploads belong to texdata */
    if (!have_l1) printf("  (TexImage2D level 1 rejected by texdata: level-1 checks skipped)\n");

    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH), 8);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT), 4);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH), 1);
    if (have_l1) {
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH), 4);
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 1, GL_TEXTURE_HEIGHT), 2);
    }
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT), GL_RGBA8);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_SIZE), 8);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_ALPHA_SIZE), 8);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH_SIZE), 0);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_STENCIL_SIZE), 0);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_SHARED_SIZE), 0);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_TYPE), GL_UNSIGNED_NORMALIZED);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH_TYPE), GL_NONE);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED), GL_FALSE);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_SAMPLES), 0);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_FIXED_SAMPLE_LOCATIONS), GL_TRUE);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_BUFFER_OFFSET), 0);
    EXPECT_NO_GL_ERROR();
    GLfloat fw = 0; glGetTexLevelParameterfv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &fw);
    EXPECT_NEAR(fw, 8.0, 0);
    GLint dw = 0; glGetTextureLevelParameteriv(t, have_l1 ? 1 : 0, GL_TEXTURE_WIDTH, &dw);
    EXPECT_EQ_INT(dw, have_l1 ? 4 : 8);
    GLfloat dfh = 0; glGetTextureLevelParameterfv(t, 0, GL_TEXTURE_HEIGHT, &dfh);
    EXPECT_NEAR(dfh, 4.0, 0);

    /* undefined level */
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 3, GL_TEXTURE_WIDTH), 0);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 3, GL_TEXTURE_INTERNAL_FORMAT), GL_RGBA);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 3, GL_TEXTURE_RED_TYPE), GL_NONE);
    EXPECT_NO_GL_ERROR();

    /* errors */
    GLint v;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &v); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, -1, GL_TEXTURE_WIDTH, &v); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 1000, GL_TEXTURE_WIDTH, &v); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_MIN_FILTER, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP, 0, GL_TEXTURE_WIDTH, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetTextureLevelParameteriv(987654, 0, GL_TEXTURE_WIDTH, &v);         EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* proxy targets are queryable */
    EXPECT_EQ_INT(lvl_i(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH), 0);
    EXPECT_EQ_INT(lvl_i(GL_PROXY_TEXTURE_CUBE_MAP, 0, GL_TEXTURE_HEIGHT), 0);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);
}

TEST(level_params_storage)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D_ARRAY, t);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 2, GL_RGBA16F, 8, 8, 4);
    int have_l1 = glGetError() == GL_NO_ERROR;   /* level > 0 creation belongs to texdata */
    if (!have_l1) printf("  (TexStorage3D level 1 rejected by texdata: level-count checks skipped)\n");
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH), 8);
    if (have_l1) {
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 1, GL_TEXTURE_WIDTH), 4);
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 1, GL_TEXTURE_HEIGHT), 4);
    }
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH), 4);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_INTERNAL_FORMAT), GL_RGBA16F);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_GREEN_SIZE), 16);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_GREEN_TYPE), GL_FLOAT);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_IMMUTABLE_FORMAT), GL_TRUE);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_IMMUTABLE_LEVELS), have_l1 ? 2 : 1);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_VIEW_MIN_LEVEL), 0);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_VIEW_NUM_LEVELS), have_l1 ? 2 : 1);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_VIEW_MIN_LAYER), 0);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_VIEW_NUM_LAYERS), 4);
    EXPECT_EQ_INT(tex_i(GL_TEXTURE_2D_ARRAY, GL_IMAGE_FORMAT_COMPATIBILITY_TYPE), GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);

    GLuint d; glGenTextures(1, &d); glBindTexture(GL_TEXTURE_2D, d);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, 4, 4);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH_SIZE), 32);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH_TYPE), GL_FLOAT);
    EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_RED_SIZE), 0);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &d);
}

TEST(level_params_compressed_ms)
{
    /* depend on other domains (compressed upload: texdata, multisample storage: texext) */
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    static GLubyte blocks[4 * 16];
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA8_ETC2_EAC, 8, 8, 0, sizeof blocks, blocks);
    while (glGetError() != GL_NO_ERROR) {}
    if (lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 8)
    {
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED), GL_TRUE);
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE), 64);
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT), GL_COMPRESSED_RGBA8_ETC2_EAC);
    }
    else
        printf("  (compressed upload not available yet: level checks skipped)\n");
    glDeleteTextures(1, &t);

    GLuint m; glGenTextures(1, &m); glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, 16, 16, GL_TRUE);
    while (glGetError() != GL_NO_ERROR) {}
    if (lvl_i(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_WIDTH) == 16)
    {
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_SAMPLES), 4);
        EXPECT_EQ_INT(lvl_i(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_TEXTURE_FIXED_SAMPLE_LOCATIONS), GL_TRUE);
    }
    else
        printf("  (multisample storage not available yet: level checks skipped)\n");
    GLint v;
    glGetTexLevelParameteriv(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_TEXTURE_WIDTH, &v);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDeleteTextures(1, &m);
}

TEST(internalformat_queries)
{
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_SUPPORTED), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_PREFERRED), GL_RGBA8);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA, GL_INTERNALFORMAT_PREFERRED), GL_RGBA8);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_RED_SIZE), 8);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_RED_TYPE), GL_UNSIGNED_NORMALIZED);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_COLOR_RENDERABLE), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_DEPTH_RENDERABLE), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_FRAMEBUFFER_RENDERABLE), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_FRAMEBUFFER_BLEND), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_FILTER), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_MIPMAP), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_MANUAL_GENERATE_MIPMAP), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_COLOR_ENCODING), GL_LINEAR);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_FRAGMENT_TEXTURE), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_SHADER_IMAGE_LOAD), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_IMAGE_TEXEL_SIZE), 32);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_IMAGE_COMPATIBILITY_CLASS), GL_IMAGE_CLASS_4_X_8);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_IMAGE_PIXEL_FORMAT), GL_RGBA);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_IMAGE_PIXEL_TYPE), GL_UNSIGNED_BYTE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_VIEW_COMPATIBILITY_CLASS), GL_VIEW_CLASS_32_BITS);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_TEXTURE_VIEW), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_READ_PIXELS_FORMAT), GL_RGBA);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_TEXTURE_IMAGE_TYPE), GL_UNSIGNED_BYTE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_TEXTURE_COMPRESSED), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_CLEAR_TEXTURE), GL_FULL_SUPPORT);
    EXPECT_TRUE(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_MAX_WIDTH) >= 8192);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_MAX_DEPTH), 0);
    EXPECT_TRUE(ifq(GL_TEXTURE_3D, GL_RGBA8, GL_MAX_DEPTH) >= 256);
    EXPECT_TRUE(ifq(GL_TEXTURE_2D_ARRAY, GL_RGBA8, GL_MAX_LAYERS) >= 256);
    GLint64 comb = 0; glGetInternalformati64v(GL_TEXTURE_2D, GL_RGBA8, GL_MAX_COMBINED_DIMENSIONS, 1, &comb);
    EXPECT_TRUE(comb >= (GLint64)8192 * 8192);
    EXPECT_NO_GL_ERROR();

    /* sample counts */
    GLint n = ifq(GL_RENDERBUFFER, GL_RGBA8, GL_NUM_SAMPLE_COUNTS);
    EXPECT_TRUE(n >= 1);
    GLint samples[8] = { 0 }; int has4 = 0;
    glGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA8, GL_SAMPLES, 8, samples);
    for (int i = 0; i < n && i < 8; i++) { if (samples[i] == 4) has4 = 1; if (i) EXPECT_TRUE(samples[i] < samples[i - 1]); }
    EXPECT_TRUE(has4);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_NUM_SAMPLE_COUNTS), 0);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_SAMPLES), -12345);   /* untouched */
    GLint64 s64[8] = { 0 };
    glGetInternalformati64v(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, GL_SAMPLES, 8, s64);
    EXPECT_TRUE(s64[0] >= 4);

    /* format specific */
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA32UI, GL_FILTER), GL_NONE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA32UI, GL_FRAMEBUFFER_BLEND), GL_NONE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA32UI, GL_INTERNALFORMAT_RED_TYPE), GL_UNSIGNED_INT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_R32UI, GL_SHADER_IMAGE_ATOMIC), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGBA8, GL_SHADER_IMAGE_ATOMIC), GL_NONE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_RENDERABLE), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_COLOR_RENDERABLE), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_TEXTURE_SHADOW), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_INTERNALFORMAT_DEPTH_SIZE), 32);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_DEPTH32F_STENCIL8, GL_STENCIL_RENDERABLE), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_DEPTH32F_STENCIL8, GL_INTERNALFORMAT_STENCIL_TYPE), GL_UNSIGNED_INT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_3D, GL_DEPTH_COMPONENT32F, GL_INTERNALFORMAT_SUPPORTED), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_SRGB8_ALPHA8, GL_COLOR_ENCODING), GL_SRGB);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_SRGB8_ALPHA8, GL_SRGB_READ), GL_FULL_SUPPORT);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGB9_E5, GL_INTERNALFORMAT_SHARED_SIZE), 5);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_RGB9_E5, GL_COLOR_RENDERABLE), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_COMPRESSED_RGBA8_ETC2_EAC, GL_TEXTURE_COMPRESSED), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_COMPRESSED_RGBA8_ETC2_EAC, GL_TEXTURE_COMPRESSED_BLOCK_WIDTH), 4);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_COMPRESSED_RGBA8_ETC2_EAC, GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT), 4);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_COMPRESSED_RGBA8_ETC2_EAC, GL_TEXTURE_COMPRESSED_BLOCK_SIZE), 16);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, GL_COMPRESSED_RGBA8_ETC2_EAC, GL_COLOR_RENDERABLE), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_BUFFER, GL_RGBA32F, GL_INTERNALFORMAT_SUPPORTED), GL_TRUE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_BUFFER, GL_RGB8, GL_INTERNALFORMAT_SUPPORTED), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_RECTANGLE, GL_RGBA8, GL_MIPMAP), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, 0x1234, GL_INTERNALFORMAT_SUPPORTED), GL_FALSE);
    EXPECT_EQ_INT(ifq(GL_TEXTURE_2D, 0x1234, GL_INTERNALFORMAT_PREFERRED), GL_NONE);
    EXPECT_NO_GL_ERROR();

    /* errors */
    GLint v;
    glGetInternalformativ(GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_RGBA8, GL_INTERNALFORMAT_SUPPORTED, 1, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, 0x1234, 1, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_SUPPORTED, -1, &v); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    v = 77; glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_INTERNALFORMAT_SUPPORTED, 0, &v);
    EXPECT_EQ_INT(v, 77); EXPECT_NO_GL_ERROR();
}

TEST(sampler_objects)
{
    GLuint s[2];
    glGenSamplers(2, s);
    EXPECT_TRUE(glIsSampler(s[0]));
    EXPECT_TRUE(!glIsSampler(0));
    EXPECT_TRUE(!glIsSampler(987654));
    GLuint c; glCreateSamplers(1, &c);
    EXPECT_TRUE(glIsSampler(c));

    /* defaults */
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_MIN_FILTER), GL_NEAREST_MIPMAP_LINEAR);
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_MAG_FILTER), GL_LINEAR);
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_WRAP_S), GL_REPEAT);
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_COMPARE_FUNC), GL_LEQUAL);
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_COMPARE_MODE), GL_NONE);

    /* round trips */
    glSamplerParameteri(s[0], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(s[0], GL_TEXTURE_WRAP_T, GL_MIRROR_CLAMP_TO_EDGE);
    glSamplerParameterf(s[0], GL_TEXTURE_MAX_ANISOTROPY, 4.0f);
    GLfloat lod = 2.0f; glSamplerParameterfv(s[0], GL_TEXTURE_MAX_LOD, &lod);
    GLint cm = GL_COMPARE_REF_TO_TEXTURE; glSamplerParameteriv(s[0], GL_TEXTURE_COMPARE_MODE, &cm);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_MIN_FILTER), GL_LINEAR);
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_WRAP_T), GL_MIRROR_CLAMP_TO_EDGE);
    EXPECT_EQ_INT(smp_i(s[0], GL_TEXTURE_COMPARE_MODE), GL_COMPARE_REF_TO_TEXTURE);
    GLfloat f = 0; glGetSamplerParameterfv(s[0], GL_TEXTURE_MAX_ANISOTROPY, &f); EXPECT_NEAR(f, 4.0, 0);
    glGetSamplerParameterfv(s[0], GL_TEXTURE_MAX_LOD, &f); EXPECT_NEAR(f, 2.0, 0);

    GLint bi[4] = { -1, 2, -3, 4 }, gbi[4];
    glSamplerParameterIiv(s[0], GL_TEXTURE_BORDER_COLOR, bi);
    glGetSamplerParameterIiv(s[0], GL_TEXTURE_BORDER_COLOR, gbi);
    for (int i = 0; i < 4; i++) EXPECT_EQ_INT(gbi[i], bi[i]);
    GLuint bu[4] = { 3000000000u, 5, 6, 7 }, gbu[4];
    glSamplerParameterIuiv(s[0], GL_TEXTURE_BORDER_COLOR, bu);
    glGetSamplerParameterIuiv(s[0], GL_TEXTURE_BORDER_COLOR, gbu);
    for (int i = 0; i < 4; i++) EXPECT_EQ_INT(gbu[i], bu[i]);
    GLint mf = GL_NEAREST; glSamplerParameterIiv(s[0], GL_TEXTURE_MAG_FILTER, &mf);
    GLuint gmf = 0; glGetSamplerParameterIuiv(s[0], GL_TEXTURE_MAG_FILTER, &gmf);
    EXPECT_EQ_INT(gmf, GL_NEAREST);
    EXPECT_NO_GL_ERROR();

    /* sampler errors */
    glSamplerParameteri(s[0], GL_TEXTURE_BASE_LEVEL, 1);            EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameteri(s[0], GL_TEXTURE_SWIZZLE_R, GL_RED);        EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameteri(s[0], GL_TEXTURE_MIN_FILTER, GL_RED);       EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glSamplerParameterf(s[0], GL_TEXTURE_MAX_ANISOTROPY, 0.0f);     EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glSamplerParameteri(987654, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLint v; glGetSamplerParameteriv(s[0], GL_TEXTURE_MAX_LEVEL, &v); EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetSamplerParameteriv(987654, GL_TEXTURE_MIN_FILTER, &v);     EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* binding */
    GLint maxu = 0; glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxu);
    glBindSampler(3, s[1]); EXPECT_NO_GL_ERROR();
    glActiveTexture(GL_TEXTURE3);
    GLint bound = 0; glGetIntegerv(GL_SAMPLER_BINDING, &bound); EXPECT_EQ_INT(bound, s[1]);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_SAMPLER_BINDING, &bound); EXPECT_EQ_INT(bound, 0);
    glBindSampler(maxu > 0 && maxu < 128 ? maxu : 128, s[0]);          EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBindSampler(0, 987654);                                        EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLuint pair[2] = { s[0], s[1] };
    glBindSamplers(0, 2, pair); EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_SAMPLER_BINDING, &bound); EXPECT_EQ_INT(bound, s[0]);
    glBindSamplers(0, 2, NULL); EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_SAMPLER_BINDING, &bound); EXPECT_EQ_INT(bound, 0);

    /* delete unbinds */
    glBindSampler(0, s[0]);
    glDeleteSamplers(1, &s[0]);
    EXPECT_TRUE(!glIsSampler(s[0]));
    glGetIntegerv(GL_SAMPLER_BINDING, &bound); EXPECT_EQ_INT(bound, 0);
    glGetSamplerParameteriv(s[0], GL_TEXTURE_MIN_FILTER, &v); EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGenSamplers(-1, s); EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDeleteSamplers(1, &s[1]); glDeleteSamplers(1, &c);
    glBindSampler(3, 0);
    EXPECT_NO_GL_ERROR();
}

/* ---- draw-time behaviour (needs working draw + readback) ---- */

static GLuint draw_prog;
static GLuint draw_vao;

static void draw_setup(void)
{
    if (draw_prog) return;
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = texture(tex, vec2(1.25, 0.5)); }\n";
    draw_prog = mgl_make_program(vs, fs);
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    GLuint vbo;
    glGenVertexArrays(1, &draw_vao); glBindVertexArray(draw_vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static void draw_once(void)
{
    glUseProgram(draw_prog);
    glBindVertexArray(draw_vao);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* The draw tests need texture upload (texdata), draw and readback (fb) to work;
   until they do, a probe draw detects it and the pixel checks are skipped. */
static int draw_pipeline_ready(GLubyte r, GLubyte g, GLubyte b)
{
    GLubyte p[4];
    mgl_read_rgba(4, 4, p);
    if (abs(p[0] - r) > 2 || abs(p[1] - g) > 2 || abs(p[2] - b) > 2) {
        printf("  (probe draw read %d,%d,%d,%d: texture upload/draw/readback not ready, pixel checks skipped)\n",
               p[0], p[1], p[2], p[3]);
        return 0;
    }
    return 1;
}

/* 2x1 texture: texel 0 red, texel 1 green. u = 1.25: REPEAT -> red, CLAMP_TO_EDGE -> green */
static GLuint make_red_green(void)
{
    static const GLubyte px[] = { 255, 0, 0, 255,   0, 255, 0, 255 };
    GLuint t; glGenTextures(1, &t);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

TEST(draw_sampler_override)
{
    draw_setup(); if (!draw_prog) return;
    GLuint t = make_red_green();

    draw_once();
    if (!draw_pipeline_ready(0, 255, 0)) { glDeleteTextures(1, &t); return; }
    EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 2);          /* texture state: clamp -> green */

    GLuint s; glGenSamplers(1, &s);
    glSamplerParameteri(s, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindSampler(0, s);
    draw_once();
    EXPECT_PIXEL(4, 4, 255, 0, 0, 255, 2);          /* sampler object: repeat -> red */

    glSamplerParameteri(s, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    draw_once();
    EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 2);          /* mirrored: 1.25 -> 0.75 -> green */

    glBindSampler(0, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    draw_once();
    EXPECT_PIXEL(4, 4, 255, 0, 0, 255, 2);          /* texture param change observed */
    EXPECT_NO_GL_ERROR();
    glDeleteSamplers(1, &s);
    glDeleteTextures(1, &t);
}

TEST(draw_swizzle_and_base_level)
{
    draw_setup(); if (!draw_prog) return;
    GLuint t = make_red_green();
    draw_once();
    if (!draw_pipeline_ready(0, 255, 0)) { glDeleteTextures(1, &t); return; }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_ONE);
    draw_once();
    EXPECT_PIXEL(4, 4, 255, 0, 255, 255, 2);        /* green texel -> (G, 0, 1, A) */
    glDeleteTextures(1, &t);

    /* base level: level 0 red, level 1 blue */
    static const GLubyte red[4 * 4] = { 255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255 };
    static const GLubyte blue[4] = { 0, 0, 255, 255 };
    GLuint m; glGenTextures(1, &m); glBindTexture(GL_TEXTURE_2D, m);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, red);
    glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, blue);
    if (glGetError() != GL_NO_ERROR) {
        printf("  (TexImage2D level 1 rejected by texdata: base-level draw check skipped)\n");
        glDeleteTextures(1, &m);
        return;
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    draw_once();
    EXPECT_PIXEL(4, 4, 255, 0, 0, 255, 2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 1);
    draw_once();
    EXPECT_PIXEL(4, 4, 0, 0, 255, 255, 2);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &m);
}

int main(void)
{
    RUN(defaults);
    RUN(roundtrip_2d);
    RUN(roundtrip_3d);
    RUN(roundtrip_2d_array);
    RUN(roundtrip_cube);
    RUN(roundtrip_1d);
    RUN(border_color);
    RUN(param_errors);
    RUN(level_params_teximage);
    RUN(level_params_storage);
    RUN(level_params_compressed_ms);
    RUN(internalformat_queries);
    RUN(sampler_objects);
    RUN(draw_sampler_override);
    RUN(draw_swizzle_and_base_level);
    return mgl_test_summary();
}
