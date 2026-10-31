/*
 * test_compat_eval.c — evaluator conformance for the compatibility profile.
 *
 * Reference behaviour: Mesa (MIT) main/eval.c, vbo/vbo_exec_eval.c,
 * math/m_eval.c and the GL 2.1/4.6-compat spec section 12.1:
 *   - glMap1/glMap2 stride is counted in ELEMENTS (floats/doubles) between
 *     control points, never bytes; stride < k (values per control point) and
 *     u1==u2 / v1==v2 are GL_INVALID_VALUE, as is order 0 or > 30;
 *   - Map1/Map2 on a texture unit other than 0 is GL_INVALID_OPERATION
 *     (evaluators generate texture coordinates for unit 0 only);
 *   - glMap1/glMap2 only DEFINE a map; glEnable(GL_MAPx_*) enables it, so an
 *     eval of a disabled map produces nothing and an enabled-but-never-defined
 *     map evaluates to the Mesa defaults (vertex (0,0,0,1), color (1,1,1,1),
 *     normal (0,0,1), index 1, texcoord (0,0,0,1));
 *   - glEvalCoord1/2 EXTRAPOLATE outside [u1,u2] (Mesa's horner/de Casteljau
 *     never clamps the parameter);
 *   - GL_AUTO_NORMAL + a 2-D vertex map generates the analytic surface normal
 *     from the de Casteljau partial derivatives (cross of du,dv, normalised,
 *     homogeneous correction for MAP2_VERTEX_4) and OVERRIDES the normal map;
 *     it has no effect on glEvalCoord1;
 *   - GL_MAP1_INDEX/GL_MAP2_INDEX evaluation updates GL_CURRENT_INDEX;
 *   - glEvalMesh1/2 with no vertex map enabled is a complete no-op (not even
 *     the colour map is evaluated).
 *
 * Vertex positions are observed through GL_FEEDBACK (GL_3D) with identity
 * modelview/projection:  win_x = vp.x + vp.w * (x/2 + 0.5).
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static void *compat_ctx;

static void eval_begin(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
}

/* enter feedback with a big buffer; returns the feedback float count on exit */
static GLint feedback_run(GLenum type, GLfloat *buf, GLsizei n, void (*fn)(void))
{
    GLint count;

    while (glGetError() != GL_NO_ERROR) {}
    glFeedbackBuffer(n, type, buf);
    glRenderMode(GL_FEEDBACK);
    fn();
    count = glRenderMode(GL_RENDER);
    while (glGetError() != GL_NO_ERROR) {}
    return count;
}

/* ------------------------------------------------------------------ */
/* stride is counted in elements                                       */

static GLfloat stride_pts[12] = {
    0, 0, 0, 777,    /* control point 0 + filler */
    1, 0, 0, 777,    /* control point 1 + filler */
    2, 0, 0, 777,    /* control point 2 + filler */
};

TEST(map1_stride_is_element_units)
{
    GLfloat coeff[12] = { 0 };

    eval_begin();                          /* create the compat context first */
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 4, 3, stride_pts);
    EXPECT_NO_GL_ERROR();
    glGetMapfv(GL_MAP1_VERTEX_3, GL_COEFF, coeff);
    EXPECT_NEAR(coeff[0], 0, 1e-6);
    EXPECT_NEAR(coeff[3], 1, 1e-6);
    EXPECT_NEAR(coeff[6], 2, 1e-6);
    EXPECT_NEAR(coeff[1], 0, 1e-6);
    EXPECT_NEAR(coeff[11], 0, 1e-6);
    /* the filler column must not leak into any component */
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(coeff[i], (i % 3 == 0) ? (GLfloat)(i / 3) : 0.0f, 1e-6);
}

TEST(map2_stride_is_element_units)
{
    /* [u][v] layout, vstride 4 elements, ustride 8 elements, filler 777 */
    GLfloat pts[16] = {
        0, 0, 0, 777,   0, 2, 0, 777,     /* u = 0 row: P00, P01 */
        4, 0, 0, 777,   4, 2, 0, 777,     /* u = 1 row: P10, P11 */
    };
    GLfloat coeff[12] = { 0 };

    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 8, 2, 0, 1, 4, 2, pts);
    EXPECT_NO_GL_ERROR();
    glGetMapfv(GL_MAP2_VERTEX_3, GL_COEFF, coeff);
    EXPECT_NEAR(coeff[0], 0, 1e-6); EXPECT_NEAR(coeff[1], 0, 1e-6); EXPECT_NEAR(coeff[2], 0, 1e-6);
    EXPECT_NEAR(coeff[3], 0, 1e-6); EXPECT_NEAR(coeff[4], 2, 1e-6); EXPECT_NEAR(coeff[5], 0, 1e-6);
    EXPECT_NEAR(coeff[6], 4, 1e-6); EXPECT_NEAR(coeff[7], 0, 1e-6); EXPECT_NEAR(coeff[8], 0, 1e-6);
    EXPECT_NEAR(coeff[9], 4, 1e-6); EXPECT_NEAR(coeff[10], 2, 1e-6); EXPECT_NEAR(coeff[11], 0, 1e-6);

    glGetMapfv(GL_MAP2_VERTEX_3, GL_ORDER, coeff);
    EXPECT_EQ_INT((int)coeff[0], 2);
    EXPECT_EQ_INT((int)coeff[1], 2);
    glGetMapfv(GL_MAP2_VERTEX_3, GL_DOMAIN, coeff);
    EXPECT_NEAR(coeff[0], 0, 1e-6); EXPECT_NEAR(coeff[1], 1, 1e-6);
    EXPECT_NEAR(coeff[2], 0, 1e-6); EXPECT_NEAR(coeff[3], 1, 1e-6);
}

/* ------------------------------------------------------------------ */
/* error surface                                                       */

TEST(map_error_surface)
{
    GLfloat ok[6] = { 0, 0, 0, 1, 1, 1 };

    glMap1f(GL_MAP1_VERTEX_3, 1.0f, 1.0f, 3, 2, ok);       /* u1 == u2 */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 0, ok);       /* order < 1 */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 31, ok);      /* order > 30 */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 2, 2, ok);       /* stride < k=3 */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 0, 2, ok);       /* stride 0 < k */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 2, NULL);     /* no points */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap1f(0x1F00, 0.0f, 1.0f, 3, 2, ok);                 /* bad target */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 2, ok);       /* valid again */
    EXPECT_NO_GL_ERROR();

    /* Map1/Map2 generate texture coordinates for unit 0 only */
    glActiveTexture(GL_TEXTURE1);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 2, ok);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 6, 2, 0, 1, 3, 2, ok);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glActiveTexture(GL_TEXTURE0);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 3, 2, ok);
    EXPECT_NO_GL_ERROR();

    glMap2f(GL_MAP2_VERTEX_3, 1, 1, 6, 2, 0, 1, 3, 2, ok); /* u1 == u2 */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 6, 2, 1, 1, 3, 2, ok); /* v1 == v2 */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 2, 2, 0, 1, 3, 2, ok); /* ustride < k */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 6, 2, 0, 1, 2, 2, ok); /* vstride < k */
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 6, 2, 0, 1, 3, 2, ok);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* enable semantics                                                    */

static GLfloat line_pts[6] = { 0, 0, 0,  0.5f, 0, 0 };   /* linear: x = u/2 */
static GLint vp[4];

static void mesh5(void)
{
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, line_pts);
    glMapGrid1f(4, 0, 1);
    glEvalMesh1(GL_POINT, 0, 4);          /* 5 points at u = 0,0.25,..,1 */
}

TEST(map_enable_semantics)
{
    GLfloat fb[256];
    GLint n;

    eval_begin();
    glViewport(0, 0, 64, 64);
    glGetIntegerv(GL_VIEWPORT, vp);
    EXPECT_EQ_INT(vp[2], 64);

    /* defined but NOT enabled: nothing is drawn */
    n = feedback_run(GL_3D, fb, 256, mesh5);
    EXPECT_EQ_INT(n, 0);

    /* enabled: 5 points x (token + 3 coords) */
    glEnable(GL_MAP1_VERTEX_3);
    EXPECT_EQ_INT(glIsEnabled(GL_MAP1_VERTEX_3), GL_TRUE);
    n = feedback_run(GL_3D, fb, 256, mesh5);
    EXPECT_EQ_INT(n, 5 * 4);
    /* first point is u=0 -> object x=0 -> window x = vp.w/2 */
    EXPECT_NEAR(fb[0], (GLfloat)GL_POINT_TOKEN, 0);
    EXPECT_NEAR(fb[1], 32.0f, 0.01f);
    EXPECT_NEAR(fb[2], 32.0f, 0.01f);
    EXPECT_NEAR(fb[3], 0.5f, 0.01f);
    /* third point u=0.5 -> object x=0.25 -> window x = 64*(0.625) = 40 */
    EXPECT_NEAR(fb[4 * 2 + 1], 40.0f, 0.01f);

    /* a map stays enabled across re-definition */
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, line_pts);
    n = feedback_run(GL_3D, fb, 256, mesh5);
    EXPECT_EQ_INT(n, 5 * 4);

    glDisable(GL_MAP1_VERTEX_3);
    EXPECT_EQ_INT(glIsEnabled(GL_MAP1_VERTEX_3), GL_FALSE);
    n = feedback_run(GL_3D, fb, 256, mesh5);
    EXPECT_EQ_INT(n, 0);
}

TEST(default_map_values)
{
    GLfloat v[4] = { 9, 9, 9, 9 };

    /* never-defined maps (this binary never maps TEXTURE_COORD_3): Mesa
     * defaults (order 1, one control point, domain [0,1]) */
    glGetMapfv(GL_MAP1_TEXTURE_COORD_3, GL_ORDER, v);
    EXPECT_EQ_INT((int)v[0], 1);
    glGetMapfv(GL_MAP1_TEXTURE_COORD_3, GL_DOMAIN, v);
    EXPECT_NEAR(v[0], 0, 1e-6);
    EXPECT_NEAR(v[1], 1, 1e-6);

    /* enabled but undefined colour map: current colour becomes (1,1,1,1) */
    glColor3f(0.1f, 0.1f, 0.1f);
    glEnable(GL_MAP1_COLOR_4);
    glEvalCoord1f(0.5f);
    EXPECT_NO_GL_ERROR();
    glGetFloatv(GL_CURRENT_COLOR, v);
    EXPECT_NEAR(v[0], 1, 1e-5);
    EXPECT_NEAR(v[1], 1, 1e-5);
    EXPECT_NEAR(v[2], 1, 1e-5);
    EXPECT_NEAR(v[3], 1, 1e-5);
    glDisable(GL_MAP1_COLOR_4);

}

static void eval_default_vertex(void) { glBegin(GL_POINTS); glEvalCoord1f(0.5f); glEnd(); }

TEST(default_map_vertex_position)
{
    GLfloat fb[64];
    GLint n;

    glEnable(GL_MAP1_VERTEX_4);           /* undefined -> (0,0,0,1) */
    n = feedback_run(GL_3D, fb, 64, eval_default_vertex);
    EXPECT_EQ_INT(n, 4);
    EXPECT_NEAR(fb[0], (GLfloat)GL_POINT_TOKEN, 0);
    EXPECT_NEAR(fb[1], 32.0f, 0.01f);     /* vp.w * (0*0.5+0.5) */
    EXPECT_NEAR(fb[2], 32.0f, 0.01f);
    EXPECT_NEAR(fb[3], 0.5f, 0.01f);
    glDisable(GL_MAP1_VERTEX_4);
}

/* ------------------------------------------------------------------ */
/* extrapolation outside the domain (no clamping)                      */

static GLfloat half_line[6] = { 0, 0, 0,  0.5f, 0, 0 };  /* x(u) = u/2, u in [0,1] */

/* a lone EvalCoord outside glBegin/glEnd is ignored like any vertex, so every
 * emission helper wraps it in a one-point glBegin/glEnd pair */
static void eval_at_025(void) { glBegin(GL_POINTS); glEvalCoord1f(0.25f); glEnd(); }
static void eval_at_m05(void)  { glBegin(GL_POINTS); glEvalCoord1f(-0.5f); glEnd(); }

TEST(evalcoord_extrapolates_outside_domain)
{
    GLfloat fb[64];
    GLint n;

    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, half_line);
    glEnable(GL_MAP1_VERTEX_3);

    n = feedback_run(GL_3D, fb, 64, eval_at_025);
    EXPECT_EQ_INT(n, 4);
    EXPECT_NEAR(fb[1], 64.0f * (0.125f * 0.5f + 0.5f), 0.01f);

    /* u = -0.5 extrapolates the curve to t=-0.5, x=-0.25 (NOT clamped to u1) */
    n = feedback_run(GL_3D, fb, 64, eval_at_m05);
    EXPECT_EQ_INT(n, 4);
    EXPECT_NEAR(fb[1], 64.0f * (-0.25f * 0.5f + 0.5f), 0.01f);

    glDisable(GL_MAP1_VERTEX_3);
}

/* ------------------------------------------------------------------ */
/* auto-normal generation                                              */

static GLfloat flat_patch[12] = {
    0, 0, 0,   0, 1, 0,      /* u = 0: P00, P01 */
    1, 0, 0,   1, 1, 0,      /* u = 1: P10, P11 */
};
static GLfloat saddle_patch[12] = {
    0, 0, 0,   0, 1, 0,
    1, 0, 0,   1, 1, 1,      /* z = u*v */
};
static GLfloat neg_normal[12] = {
    0, 0, -1,  0, 0, -1,
    0, 0, -1,  0, 0, -1,
};

static void auto_setup(GLfloat *patch)
{
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 6, 2, 0, 1, 3, 2, patch);
    glEnable(GL_MAP2_VERTEX_3);
}

TEST(auto_normal_flat_patch)
{
    GLfloat n[4] = { 9, 9, 9, 9 };

    auto_setup(flat_patch);
    glNormal3f(1, 0, 0);
    glEnable(GL_AUTO_NORMAL);
    EXPECT_EQ_INT(glIsEnabled(GL_AUTO_NORMAL), GL_TRUE);
    glEvalCoord2f(0.5f, 0.5f);
    EXPECT_NO_GL_ERROR();
    glGetFloatv(GL_CURRENT_NORMAL, n);
    EXPECT_NEAR(n[0], 0, 1e-5);           /* du = (1,0,0), dv = (0,1,0) */
    EXPECT_NEAR(n[1], 0, 1e-5);
    EXPECT_NEAR(n[2], 1, 1e-5);
    glDisable(GL_AUTO_NORMAL);
    glDisable(GL_MAP2_VERTEX_3);
}

TEST(auto_normal_saddle_patch)
{
    double ex = -0.5, ey = -0.5, ez = 1.0, len = sqrt(ex * ex + ey * ey + ez * ez);
    GLfloat n[4] = { 9, 9, 9, 9 };

    auto_setup(saddle_patch);
    glEnable(GL_AUTO_NORMAL);
    glEvalCoord2f(0.5f, 0.5f);            /* du=(1,0,.5), dv=(0,1,.5) */
    glGetFloatv(GL_CURRENT_NORMAL, n);
    EXPECT_NEAR(n[0], (GLfloat)(ex / len), 1e-4);
    EXPECT_NEAR(n[1], (GLfloat)(ey / len), 1e-4);
    EXPECT_NEAR(n[2], (GLfloat)(ez / len), 1e-4);

    /* normals vary over the patch: at u=0.25,v=0.5, du=(1,0,0.5), dv=(0,1,0.25) */
    {
        double du[3] = { 1, 0, 0.5 }, dv[3] = { 0, 1, 0.25 };
        double cx = du[1] * dv[2] - du[2] * dv[1];
        double cy = du[2] * dv[0] - du[0] * dv[2];
        double cz = du[0] * dv[1] - du[1] * dv[0];
        double l2 = sqrt(cx * cx + cy * cy + cz * cz);

        glEvalCoord2f(0.25f, 0.5f);
        glGetFloatv(GL_CURRENT_NORMAL, n);
        EXPECT_NEAR(n[0], (GLfloat)(cx / l2), 1e-4);
        EXPECT_NEAR(n[1], (GLfloat)(cy / l2), 1e-4);
        EXPECT_NEAR(n[2], (GLfloat)(cz / l2), 1e-4);
    }
    glDisable(GL_AUTO_NORMAL);
    glDisable(GL_MAP2_VERTEX_3);
}

TEST(auto_normal_overrides_normal_map)
{
    GLfloat n[4] = { 9, 9, 9, 9 };

    auto_setup(flat_patch);
    glMap2f(GL_MAP2_NORMAL, 0, 1, 6, 2, 0, 1, 3, 2, neg_normal);
    glEnable(GL_MAP2_NORMAL);

    glDisable(GL_AUTO_NORMAL);
    glEvalCoord2f(0.5f, 0.5f);            /* normal map wins */
    glGetFloatv(GL_CURRENT_NORMAL, n);
    EXPECT_NEAR(n[0], 0, 1e-5);
    EXPECT_NEAR(n[1], 0, 1e-5);
    EXPECT_NEAR(n[2], -1, 1e-5);

    glEnable(GL_AUTO_NORMAL);
    glEvalCoord2f(0.5f, 0.5f);            /* analytic normal wins */
    glGetFloatv(GL_CURRENT_NORMAL, n);
    EXPECT_NEAR(n[0], 0, 1e-5);
    EXPECT_NEAR(n[1], 0, 1e-5);
    EXPECT_NEAR(n[2], 1, 1e-5);

    glDisable(GL_AUTO_NORMAL);
    glDisable(GL_MAP2_NORMAL);
    glDisable(GL_MAP2_VERTEX_3);
}

TEST(auto_normal_homogeneous_patch)
{
    /* w(u) = 1+u: projected x = u/(1+u); the analytic normal of the
     * projected surface is +z (with the w^2 homogeneous correction) */
    GLfloat pts[16] = {
        0, 0, 0, 1,   0, 1, 0, 1,
        1, 0, 0, 2,   1, 1, 0, 2,
    };
    GLfloat n[4] = { 9, 9, 9, 9 };

    glMap2f(GL_MAP2_VERTEX_4, 0, 1, 8, 2, 0, 1, 4, 2, pts);
    glEnable(GL_MAP2_VERTEX_4);
    glEnable(GL_AUTO_NORMAL);
    glEvalCoord2f(0.5f, 0.5f);
    EXPECT_NO_GL_ERROR();
    glGetFloatv(GL_CURRENT_NORMAL, n);
    EXPECT_NEAR(n[0], 0, 1e-4);
    EXPECT_NEAR(n[1], 0, 1e-4);
    EXPECT_NEAR(n[2], 1, 1e-4);
    glDisable(GL_AUTO_NORMAL);
    glDisable(GL_MAP2_VERTEX_4);
}

static GLfloat any_line[6] = { 0, 0, 0, 1, 0, 0 };

TEST(auto_normal_ignores_evalcoord1)
{
    GLfloat n[4] = { 9, 9, 9, 9 };

    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, any_line);
    glEnable(GL_MAP1_VERTEX_3);
    glNormal3f(0, 1, 0);
    glEnable(GL_AUTO_NORMAL);
    glEvalCoord1f(0.5f);
    EXPECT_NO_GL_ERROR();
    glGetFloatv(GL_CURRENT_NORMAL, n);
    EXPECT_NEAR(n[0], 0, 1e-6);           /* untouched: no 1-D auto normal */
    EXPECT_NEAR(n[1], 1, 1e-6);
    EXPECT_NEAR(n[2], 0, 1e-6);
    glDisable(GL_AUTO_NORMAL);
    glDisable(GL_MAP1_VERTEX_3);
}

/* ------------------------------------------------------------------ */
/* index maps                                                          */

static GLfloat idx_pts[2] = { 5, 9 };

TEST(index_map_sets_current_index)
{
    GLfloat idx = 0;

    glIndexi(1);
    glEnable(GL_MAP1_INDEX);
    glMap1f(GL_MAP1_INDEX, 0, 1, 1, 2, idx_pts);
    glEvalCoord1f(0.5f);
    EXPECT_NO_GL_ERROR();
    glGetFloatv(GL_CURRENT_INDEX, &idx);
    EXPECT_NEAR(idx, 7.0f, 1e-4);         /* (5+9)/2 */

    glDisable(GL_MAP1_INDEX);
    glIndexi(2);
    glEvalCoord1f(0.5f);
    glGetFloatv(GL_CURRENT_INDEX, &idx);
    EXPECT_NEAR(idx, 2.0f, 1e-4);         /* map disabled: untouched */
}

/* ------------------------------------------------------------------ */
/* EvalMesh is a no-op without a vertex map (even for colour maps)     */

static GLfloat colour_ramp[8] = { 1, 0, 0, 1,  0, 1, 0, 1 };
static GLfloat ramp_line[6] = { 0, 0, 0, 1, 0, 0 };

static void mesh_colour_only(void)
{
    glColor3f(0, 0, 0);
    glMap1f(GL_MAP1_COLOR_4, 0, 1, 4, 2, colour_ramp);
    glEnable(GL_MAP1_COLOR_4);
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, ramp_line);
    glMapGrid1f(2, 0, 1);
    glEvalMesh1(GL_POINT, 0, 2);          /* vertex map still disabled */
}

TEST(evalmesh_noop_without_vertex_map)
{
    GLfloat fb[128];
    GLfloat c[4] = { 9, 9, 9, 9 };
    GLint n;

    n = feedback_run(GL_3D, fb, 128, mesh_colour_only);
    EXPECT_EQ_INT(n, 0);                  /* no geometry at all */
    glGetFloatv(GL_CURRENT_COLOR, c);
    EXPECT_NEAR(c[0], 0, 1e-5);           /* and the colour map was not
                                           * evaluated either (Mesa returns
                                           * before any evaluation) */
    EXPECT_NEAR(c[1], 0, 1e-5);

    glEnable(GL_MAP1_VERTEX_3);
    n = feedback_run(GL_3D, fb, 128, mesh_colour_only);
    EXPECT_EQ_INT(n, 3 * 4);
    glGetFloatv(GL_CURRENT_COLOR, c);
    EXPECT_NEAR(c[0], 0, 1e-5);           /* last grid point u=1 -> green */
    EXPECT_NEAR(c[1], 1, 1e-5);

    glDisable(GL_MAP1_VERTEX_3);
    glDisable(GL_MAP1_COLOR_4);
}

/* ------------------------------------------------------------------ */
/* glEvalPoint* and the grid                                           */

static void evalpoint0(void) { glBegin(GL_POINTS); glEvalPoint1(0); glEnd(); }
static void evalpoint1(void) { glBegin(GL_POINTS); glEvalPoint1(1); glEnd(); }

TEST(evalpoint_uses_grid)
{
    GLfloat fb[64];
    GLint n;

    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, half_line);
    glEnable(GL_MAP1_VERTEX_3);

    /* default grid n=1, u1=0, u2=1: EvalPoint1(0) evaluates u=0 */
    n = feedback_run(GL_3D, fb, 64, evalpoint0);
    EXPECT_EQ_INT(n, 4);
    EXPECT_NEAR(fb[1], 32.0f, 0.01f);

    glMapGrid1f(2, 0, 1);                 /* u(1) = 0.5 -> x = 0.25 */
    n = feedback_run(GL_3D, fb, 64, evalpoint1);
    EXPECT_EQ_INT(n, 4);
    EXPECT_NEAR(fb[1], 40.0f, 0.01f);

    glDisable(GL_MAP1_VERTEX_3);
}

/* ------------------------------------------------------------------ */
/* glPushAttrib/glPopAttrib must deep-copy the evaluator control points:
 * a glMap1 between push and pop frees them, so a struct restore would
 * resurrect dangling pointers (and re-defining later would double-free)  */

static GLfloat curve3[9] = { 0, 0, 0,  0.5f, 0, 0,  1, 0, 0 };
static GLfloat curve2[6] = { 5, 5, 5,  6, 6, 6 };
static void eval_third(void) { glBegin(GL_POINTS); glEvalCoord1f(1.0f / 3.0f); glEnd(); }

TEST(attrib_snapshot_restores_maps_deeply)
{
    GLfloat v[16] = { 0 };
    GLfloat fb[64];
    GLint n;

    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 3, curve3);
    glEnable(GL_MAP1_VERTEX_3);

    glPushAttrib(GL_EVAL_BIT);

    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, curve2);   /* different order: the
                                                     * pushed control points
                                                     * are freed for real */
    glGetMapfv(GL_MAP1_VERTEX_3, GL_ORDER, v);
    EXPECT_EQ_INT((int)v[0], 2);

    glPopAttrib();

    glGetMapfv(GL_MAP1_VERTEX_3, GL_ORDER, v);
    EXPECT_EQ_INT((int)v[0], 3);                     /* order restored */
    glGetMapfv(GL_MAP1_VERTEX_3, GL_COEFF, v);
    EXPECT_NEAR(v[3], 0.5f, 1e-6);
    EXPECT_NEAR(v[6], 1.0f, 1e-6);
    EXPECT_NEAR(v[8], 0.0f, 1e-6);
    EXPECT_EQ_INT(glIsEnabled(GL_MAP1_VERTEX_3), GL_TRUE);

    /* evaluation through the restored map: B(1/3) = 1/3 -> win x = 42.667 */
    n = feedback_run(GL_3D, fb, 64, eval_third);
    EXPECT_EQ_INT(n, 4);
    EXPECT_NEAR(fb[1], 64.0f * (1.0f / 3.0f * 0.5f + 0.5f), 0.05f);

    /* a re-definition after the pop must not double-free anything; the new
     * curve lives far outside the view volume so the point is clipped and
     * emits no feedback token — but nothing may crash or error */
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, curve2);
    EXPECT_NO_GL_ERROR();
    n = feedback_run(GL_3D, fb, 64, eval_third);
    EXPECT_EQ_INT(n, 0);
    glDisable(GL_MAP1_VERTEX_3);
}

/* ------------------------------------------------------------------ */
/* evaluator commands compile into display lists (Mesa dlist.c): nothing
 * executes at compile time and the replay re-evaluates with the maps that
 * are current AT CALL TIME (the list itself re-defines its own map)       */

static GLfloat dbl_line[6] = { 0, 0, 0,  1, 0, 0 };    /* x(u) = u */
static GLuint eval_list;

static void call_eval_list(void) { glCallList(eval_list); }

TEST(display_lists_compile_evaluators)
{
    GLfloat v[4] = { 9, 9, 9, 9 };
    GLfloat fb[256];
    GLint n;

    eval_begin();
    glViewport(0, 0, 64, 64);
    eval_list = glGenLists(1);

    glNewList(eval_list, GL_COMPILE);
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, half_line);   /* compiled only */
    glMapGrid1f(4, 0, 1);
    glBegin(GL_POINTS);
    glEvalCoord1f(0.5f);
    glEnd();
    glEvalMesh1(GL_POINT, 0, 1);
    glBegin(GL_POINTS);
    glEvalPoint1(1);
    glEnd();
    glEndList();
    EXPECT_NO_GL_ERROR();

    /* compilation must not have defined the map (still the default order 1),
     * enabled anything, or drawn anything */
    glGetMapfv(GL_MAP1_VERTEX_3, GL_ORDER, v);
    EXPECT_EQ_INT((int)v[0], 1);
    EXPECT_EQ_INT(glIsEnabled(GL_MAP1_VERTEX_3), GL_FALSE);
    n = feedback_run(GL_3D, fb, 256, call_eval_list);
    EXPECT_EQ_INT(n, 0);                               /* map still disabled */

    /* enable a DIFFERENT map (x(u) = u) and call the list: the list replays
     * its own map definition first (x(u) = u/2), then evaluates with it */
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, dbl_line);
    glEnable(GL_MAP1_VERTEX_3);
    n = feedback_run(GL_3D, fb, 256, call_eval_list);
    EXPECT_EQ_INT(n, 4 * 4);                           /* coord + 2 mesh + point */

    EXPECT_NEAR(fb[0], (GLfloat)GL_POINT_TOKEN, 0);
    EXPECT_NEAR(fb[1], 40.0f, 0.01f);                  /* EvalCoord(0.5): x=.25 */
    EXPECT_NEAR(fb[4 * 1 + 1], 32.0f, 0.01f);          /* mesh i=0: u=0 */
    EXPECT_NEAR(fb[4 * 2 + 1], 36.0f, 0.01f);          /* mesh i=1: u=0.25 */
    EXPECT_NEAR(fb[4 * 3 + 1], 36.0f, 0.01f);          /* EvalPoint1(1): u=0.25 */

    /* the map lives at call time only: after the call the state still holds
     * the list's definition (half_line), not the outer dbl_line */
    glGetMapfv(GL_MAP1_VERTEX_3, GL_COEFF, v);
    EXPECT_NEAR(v[3], 0.5f, 1e-6);

    glDisable(GL_MAP1_VERTEX_3);
}

int main(void)
{
    RUN(map1_stride_is_element_units);
    RUN(map2_stride_is_element_units);
    RUN(map_error_surface);
    RUN(map_enable_semantics);
    RUN(default_map_values);
    RUN(default_map_vertex_position);
    RUN(evalcoord_extrapolates_outside_domain);
    RUN(auto_normal_flat_patch);
    RUN(auto_normal_saddle_patch);
    RUN(auto_normal_overrides_normal_map);
    RUN(auto_normal_homogeneous_patch);
    RUN(auto_normal_ignores_evalcoord1);
    RUN(index_map_sets_current_index);
    RUN(evalmesh_noop_without_vertex_map);
    RUN(evalpoint_uses_grid);
    RUN(attrib_snapshot_restores_maps_deeply);
    RUN(display_lists_compile_evaluators);
    return mgl_test_summary();
}
