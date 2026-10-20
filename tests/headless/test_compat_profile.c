/*
 * test_compat_profile.c — compatibility-profile context plumbing (compat domain).
 *
 * A compat context must report the compatibility profile, the compat
 * extension, accept the fixed-function entry points, and answer every legacy
 * state query; a core context must keep rejecting exactly those.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static void *compat_ctx = NULL;

TEST(compat_context_creation)
{
    compat_ctx = mglCompatCreateContext();

    EXPECT_TRUE(compat_ctx != NULL);
    EXPECT_TRUE(mglCurrentContextIsCompat());
    EXPECT_NO_GL_ERROR();
}

TEST(version_and_profile)
{
    GLint mask = 0;

    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    EXPECT_EQ_INT(mask, GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    EXPECT_NO_GL_ERROR();

    EXPECT_TRUE(strstr((const char *)glGetString(GL_VERSION), "4.6") != NULL);
    EXPECT_NO_GL_ERROR();

    /* GL_ARB_compatibility is extension 0 in a compat context */
    const GLubyte *ext = glGetStringi(GL_EXTENSIONS, 0);
    EXPECT_STREQ((const char *)ext, "GL_ARB_compatibility");

    GLint nexts = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &nexts);
    EXPECT_TRUE(nexts > 1);
}

TEST(fixed_function_state_queries)
{
    GLint i = 0;
    GLfloat f = 0, v4[4];

    glGetIntegerv(GL_LIGHTING, &i);
    EXPECT_EQ_INT(i, GL_FALSE);

    glEnable(GL_LIGHTING);
    EXPECT_EQ_INT(glIsEnabled(GL_LIGHTING), GL_TRUE);
    glGetIntegerv(GL_LIGHTING, &i);
    EXPECT_EQ_INT(i, GL_TRUE);
    glDisable(GL_LIGHTING);
    EXPECT_EQ_INT(glIsEnabled(GL_LIGHTING), GL_FALSE);

    glGetIntegerv(GL_MAX_LIGHTS, &i);
    EXPECT_TRUE(i >= 8);
    glGetIntegerv(GL_MATRIX_MODE, &i);
    EXPECT_EQ_INT(i, GL_MODELVIEW);

    glGetFloatv(GL_CURRENT_COLOR, v4);
    EXPECT_NEAR(v4[0], 1.0, 1e-6);
    EXPECT_NEAR(v4[3], 1.0, 1e-6);

    glGetFloatv(GL_SHADE_MODEL, &f);
    EXPECT_EQ_INT((int)f, GL_SMOOTH);
    (void)f;

    glGetFloatv(GL_ZOOM_X, &f);
    EXPECT_NEAR(f, 1.0, 1e-6);

    EXPECT_NO_GL_ERROR();
}

TEST(matrix_stack)
{
    GLfloat m[16];
    GLint depth = 0;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glGetFloatv(GL_PROJECTION_MATRIX, m);
    EXPECT_NEAR(m[0], 1.0, 1e-6);
    EXPECT_NEAR(m[5], 1.0, 1e-6);
    EXPECT_NEAR(m[10], 1.0, 1e-6);
    EXPECT_NEAR(m[15], 1.0, 1e-6);
    EXPECT_NEAR(m[12], 0.0, 1e-6);

    glPushMatrix();
    glGetIntegerv(GL_PROJECTION_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 2);

    glTranslated(1.0, 2.0, 3.0);
    glGetFloatv(GL_PROJECTION_MATRIX, m);
    EXPECT_NEAR(m[12], 1.0, 1e-6);
    EXPECT_NEAR(m[13], 2.0, 1e-6);
    EXPECT_NEAR(m[14], 3.0, 1e-6);

    glPopMatrix();
    glGetFloatv(GL_PROJECTION_MATRIX, m);
    EXPECT_NEAR(m[12], 0.0, 1e-6);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* underflow must raise GL_STACK_UNDERFLOW */
    glPopMatrix();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    EXPECT_NEAR(m[0], 1.0, 1e-6);
    EXPECT_NEAR(m[12], 0.0, 1e-6);
}

TEST(light_and_material_state)
{
    GLfloat v[4];

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();   /* light positions transform through the modelview */

    const GLfloat pos[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    glGetLightfv(GL_LIGHT0, GL_POSITION, v);
    EXPECT_NEAR(v[2], 1.0, 1e-6);

    const GLfloat diffuse[4] = { 0.25f, 0.5f, 0.75f, 1.0f };
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse);
    glGetMaterialfv(GL_FRONT, GL_DIFFUSE, v);
    EXPECT_NEAR(v[1], 0.5, 1e-6);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat[4]){ 0.3f, 0.3f, 0.3f, 1.0f });
    glGetFloatv(GL_LIGHT_MODEL_AMBIENT, v);
    EXPECT_NEAR(v[0], 0.3f, 1e-6);

    /* invalid light enumerant */
    glLightf(GL_LIGHT0 + 9, GL_SPOT_EXPONENT, 1.0f);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    EXPECT_NO_GL_ERROR();
}

TEST(texenv_texgen_fog_alpha)
{
    GLfloat v[4];
    GLint i = 0;

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &i);
    EXPECT_EQ_INT(i, GL_REPLACE);

    glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, (GLfloat[4]){ 0.1f, 0.2f, 0.3f, 0.4f });
    glGetTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, v);
    EXPECT_NEAR(v[2], 0.3f, 1e-6);

    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
    glGetTexGeniv(GL_S, GL_TEXTURE_GEN_MODE, &i);
    EXPECT_EQ_INT(i, GL_SPHERE_MAP);

    glFogi(GL_FOG_MODE, GL_EXP2);
    {
        GLint mode = 0;
        glGetIntegerv(0x0B65, &mode);
        EXPECT_EQ_INT(mode, GL_EXP2);
    }

    glAlphaFunc(GL_LESS, 0.5f);
    {
        GLfloat ref = 1.0f;
        glGetFloatv(GL_ALPHA_TEST_REF, &ref);
        EXPECT_NEAR(ref, 0.5f, 1e-6);
    }

    EXPECT_NO_GL_ERROR();
}

TEST(display_list_state)
{
    GLuint list = glGenLists(1);

    EXPECT_TRUE(list > 0);

    glNewList(list, GL_COMPILE);
    EXPECT_NO_GL_ERROR();

    GLint mode = 0, index = 0;
    glGetIntegerv(GL_LIST_MODE, &mode);
    EXPECT_EQ_INT(mode, GL_COMPILE);
    glGetIntegerv(GL_LIST_INDEX, &index);
    EXPECT_EQ_INT(index, (GLint)list);

    glEndList();
    EXPECT_NO_GL_ERROR();

    EXPECT_EQ_INT(glIsList(list), GL_TRUE);
    EXPECT_EQ_INT(glIsList(list + 1), GL_FALSE);

    glDeleteLists(list, 1);
    EXPECT_EQ_INT(glIsList(list), GL_FALSE);
}

TEST(core_context_still_rejects_compat)
{
    /* switch back to the core context: the compat entry points must fail */
    mglMakeContextCurrent(mglCoreContext());

    EXPECT_TRUE(!mglCurrentContextIsCompat());

    GLint mask = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    EXPECT_EQ_INT(mask, GL_CONTEXT_CORE_PROFILE_BIT);

    glBegin(GL_TRIANGLES);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glMatrixMode(GL_MODELVIEW);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glPushMatrix();
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat[4]){ 0, 0, 1, 0 });
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    GLint n = 0;
    glGetIntegerv(GL_LIGHTING, &n);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* restore the compat context for the remaining tests */
    mglMakeContextCurrent(compat_ctx);
    EXPECT_TRUE(mglCurrentContextIsCompat());
}

int main(void)
{
    RUN(compat_context_creation);
    RUN(version_and_profile);
    RUN(fixed_function_state_queries);
    RUN(matrix_stack);
    RUN(light_and_material_state);
    RUN(texenv_texgen_fog_alpha);
    RUN(display_list_state);
    RUN(core_context_still_rejects_compat);
    return mgl_test_summary();
}
