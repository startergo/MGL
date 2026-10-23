/*
 * test_glx_config.c — headless unit test of the GLX shim's pure logic
 * (MGL/glx/glx_config.c): the synthesized fbconfig table, the attribute
 * matching rules of glXChooseVisual/glXChooseFBConfig, the readout used by
 * glXGetConfig/glXGetFBConfigAttrib, the sort order and the client strings.
 * No X11 connection is needed (glx_types.h is X-free on purpose).
 */
#include "mgl_test.h"
#include "glx_types.h"

/* pull the implementation in directly: run.sh builds one file per test */
#include "glx_config.c"

static MGLXFBConfigRec cfgs[MGLX_NCONFIGS];

TEST(glx_config_table)
{
    mglx_build_configs(cfgs, MGLX_NCONFIGS, 0, 0x2a, 24, 0x101);
    EXPECT_EQ_INT(cfgs[0].fbconfig_id, 0x101);
    EXPECT_EQ_INT((int)cfgs[0].visualid, 0x2a);
    EXPECT_EQ_INT(cfgs[0].doublebuffer, 1);
    EXPECT_EQ_INT(cfgs[0].depth_size, 24);
    EXPECT_EQ_INT(cfgs[0].stencil_size, 8);
    EXPECT_EQ_INT(cfgs[0].alpha_size, 8);
    EXPECT_EQ_INT(cfgs[0].x_visual_type, GLX_TRUE_COLOR);
    EXPECT_EQ_INT(cfgs[0].render_type, GLX_RGBA_BIT);
    EXPECT_EQ_INT(cfgs[0].drawable_type,
                  GLX_WINDOW_BIT | GLX_PIXMAP_BIT | GLX_PBUFFER_BIT);

    /* unique fbconfig ids */
    for (int i = 1; i < MGLX_NCONFIGS; i++)
    {
        EXPECT_TRUE(cfgs[i].fbconfig_id != cfgs[i - 1].fbconfig_id);
        EXPECT_EQ_INT((int)cfgs[i].visualid, 0x2a);
    }
    /* the table is generated best-first (see mglx_config_compare) */
    for (int i = 1; i < MGLX_NCONFIGS; i++)
        EXPECT_TRUE(mglx_config_compare(&cfgs[i - 1], &cfgs[i]) <= 0);
}

TEST(glx_choose_matching)
{
    mglx_build_configs(cfgs, MGLX_NCONFIGS, 0, 0x2a, 24, 0x101);

    /* the classic request: RGBA + double buffer + depth */
    int want1[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 1, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want1, 0, NULL) == 1);

    /* min-size semantics: asking for depth 32 must fail (we have 24) */
    int want2[] = { GLX_DEPTH_SIZE, 32, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want2, 0, NULL) == 0);

    /* boolean tokens are value-less: {GLX_DOUBLEBUFFER} means "want double";
     * omitting the token entirely is how single-buffered configs are chosen */
    int want3[] = { GLX_RGBA, GLX_DOUBLEBUFFER, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want3, 0, NULL) == 1);
    int n_double = 0, n_single = 0;
    for (int i = 0; i < MGLX_NCONFIGS; i++)
        if (mglx_config_matches(&cfgs[i], want3, 0, NULL) == 1)
            n_double++;
    for (int i = 0; i < MGLX_NCONFIGS; i++)
        if (cfgs[i].doublebuffer == 0)
            n_single++;
    EXPECT_TRUE(n_double >= 1);
    EXPECT_TRUE(n_single >= 1);
    EXPECT_TRUE(n_double + n_single == MGLX_NCONFIGS);
    EXPECT_TRUE(mglx_config_attrib(&cfgs[0], GLX_DOUBLEBUFFER) == 1);

    /* GLX_DONT_CARE ignores an attribute (pair style with DONT_CARE value) */
    int want4[] = { GLX_DOUBLEBUFFER, 0xFFFFFFFF, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want4, 0, NULL) == 1);

    /* an accumulation config exists and matches a 16-bit request */
    int want5[] = { GLX_ACCUM_RED_SIZE, 16, 0 };
    int found_accum = 0;
    for (int i = 0; i < MGLX_NCONFIGS; i++)
        if (mglx_config_matches(&cfgs[i], want5, 1, NULL) == 1)
            found_accum = 1;
    EXPECT_TRUE(found_accum);

    /* unknown attribute -> error (mglx_config_matches returns < 0) */
    const char *bad = NULL;
    int want6[] = { 0xdeadbeef, 1, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want6, 1, &bad) < 0);
    EXPECT_STREQ(bad, "unknown attribute");

    /* fbconfig-only attributes are rejected in glXChooseVisual */
    int want7[] = { GLX_RENDER_TYPE, GLX_RGBA_BIT, 0 };
    bad = NULL;
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want7, 0, &bad) < 0);
    EXPECT_STREQ(bad, "fbconfig-only attribute in glXChooseVisual");
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want7, 1, &bad) == 1);

    /* drawable-type set membership */
    int want8[] = { GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want8, 1, NULL) == 1);
    int want9[] = { GLX_DRAWABLE_TYPE, 0x01000000, 0 };   /* a bit we lack */
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want9, 1, NULL) == 0);

    /* no transparency available */
    int want10[] = { GLX_TRANSPARENT_TYPE, GLX_TRANSPARENT_RGB, 0 };
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], want10, 1, NULL) == 0);

    /* NULL attrib list matches everything */
    EXPECT_TRUE(mglx_config_matches(&cfgs[0], NULL, 0, NULL) == 1);
}

TEST(glx_config_attrib_readout)
{
    mglx_build_configs(cfgs, MGLX_NCONFIGS, 0, 0x2a, 24, 0x101);

    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_DOUBLEBUFFER), 1);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_RGBA), 1);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_STEREO), 0);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_RED_SIZE), 8);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_ALPHA_SIZE), 8);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_DEPTH_SIZE), 24);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_STENCIL_SIZE), 8);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_BUFFER_SIZE), 32);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_SAMPLE_BUFFERS), 0);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_SAMPLES), 0);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_TRANSPARENT_TYPE), GLX_NONE);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_SWAP_INTERVAL_EXT), 1);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], GLX_MAX_SWAP_INTERVAL_EXT), 1);
    EXPECT_EQ_INT((int)mglx_config_attrib(&cfgs[0], GLX_VISUAL_ID), 0x2a);

    /* an MSAA config reports samples */
    int msaa = -1;
    for (int i = 0; i < MGLX_NCONFIGS; i++)
        if (mglx_config_attrib(&cfgs[i], GLX_SAMPLES) == 4)
            msaa = i;
    EXPECT_TRUE(msaa >= 0);
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[msaa], GLX_SAMPLE_BUFFERS), 1);

    /* not a GLX attribute */
    EXPECT_EQ_INT(mglx_config_attrib(&cfgs[0], 0x1234), GLX_NONE);
}

TEST(glx_client_strings)
{
    EXPECT_STREQ(mglx_client_string(GLX_VENDOR), "MGL (Metal)");
    EXPECT_STREQ(mglx_client_string(GLX_VERSION), "1.4 MGL");
    const char *ext = mglx_extensions_string();
    EXPECT_TRUE(strstr(ext, "GLX_ARB_create_context") != NULL);
    EXPECT_TRUE(strstr(ext, "GLX_ARB_create_context_profile") != NULL);
    EXPECT_TRUE(strstr(ext, "GLX_SGIX_fbconfig") != NULL);
    EXPECT_TRUE(strstr(ext, "GLX_EXT_swap_control") != NULL);
    EXPECT_TRUE(mglx_client_string(999) == NULL);
}

int main(void)
{
    RUN(glx_config_table);
    RUN(glx_choose_matching);
    RUN(glx_config_attrib_readout);
    RUN(glx_client_strings);
    return mgl_test_summary();
}
