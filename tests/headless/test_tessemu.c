#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "glcorearb.h"
#include "mgl_test.h"

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec4 a_pos;\n"
    "layout(location=1) in vec4 a_col;\n"
    "layout(location=0) out vec4 v_col;\n"
    "void main() { gl_Position = a_pos; v_col = a_col; }\n";
static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) in vec4 f_col;\n"
    "layout(location=0) out vec4 o_col;\n"
    "void main() { o_col = f_col; }\n";
static const char *kTcs =
    "#version 450 core\n"
    "layout(vertices=2) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 8.0; }\n"
    "}\n";
static const char *kTcsQuad =
    "#version 450 core\n"
    "layout(vertices=4) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0]=1.0; gl_TessLevelOuter[1]=1.0; gl_TessLevelOuter[2]=1.0; gl_TessLevelOuter[3]=1.0;\n"
    "    gl_TessLevelInner[0]=1.0; gl_TessLevelInner[1]=1.0; }\n"
    "}\n";

/* isolines: control points 0->1 along u, lines at v = (i+0.5)/n0 */
static const char *kTesIso =
    "#version 450 core\n"
    "layout(isolines, equal_spacing) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
    "  gl_Position.y = mix(-0.75, 0.75, gl_TessCoord.y);\n"
    "  f_col = mix(tc_col[0], tc_col[1], gl_TessCoord.x);\n"
    "}\n";
/* quads with point mode: emit the corners as points */
static const char *kTesQuadPt =
    "#version 450 core\n"
    "layout(quads, equal_spacing, ccw, point_mode) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = mix(mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),\n"
    "                    mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x), gl_TessCoord.y);\n"
    "  f_col = tc_col[0];\n"
    "}\n";
static const char *kTcsTri =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0]=1.0; gl_TessLevelOuter[1]=1.0; gl_TessLevelOuter[2]=1.0;\n"
    "    gl_TessLevelInner[0]=1.0; }\n"
    "}\n";
/* triangles with point mode, barycentric evaluation: the 3 corner vertices of the
 * level-1 triangle land exactly on the 3 control points */
static const char *kTesTriPt =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw, point_mode) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = tc_col[0];\n"
    "}\n";

static GLuint prog2(const char *tcs, const char *tes);
static GLuint prog(const char *tes) { return prog2(kTcs, tes); }
static GLuint prog2(const char *tcs, const char *tes)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[4] = {
        { GL_VERTEX_SHADER, kVs }, { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes }, { GL_FRAGMENT_SHADER, kFs } };
    for (int i = 0; i < 4; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL); glCompileShader(so); glAttachShader(p, so); glDeleteShader(so);
    }
    glLinkProgram(p);
    return p;
}

static const char *kVsPlain =
    "#version 450 core\n"
    "layout(location=0) in vec4 a_pos;\n"
    "layout(location=0) out vec4 v_col;\n"
    "void main() { gl_Position = a_pos; v_col = vec4(1); }\n";
static const char *kFsWhite =
    "#version 450 core\n"
    "layout(location=0) in vec4 f_col;\n"
    "layout(location=0) out vec4 o_col;\n"
    "void main() { o_col = f_col; }\n";

static void setup(const float pos[][4], const float col[][4], int n, int patch_verts)
{
    GLuint vao, b[2];
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]); glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * n, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL); glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]); glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * n, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL); glEnableVertexAttribArray(1);
    glPatchParameteri(GL_PATCH_VERTICES, patch_verts);
}

/* pixel-exact reference: draw the first `n` attributes of the bound VAO as plain GL_POINTS
 * and diff the lit pixels against `tess_px`. Returns true when the sets are identical with
 * exactly `expect` lit pixels. */
static bool point_mode_matches_plain_points(const unsigned char *tess_px, int w, int h, int n, int expect)
{
    unsigned char *plain_px = calloc((size_t)w * h, 4);
    GLuint q = glCreateProgram();
    GLuint vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs, 1, &kVsPlain, NULL); glCompileShader(vs); glAttachShader(q, vs); glDeleteShader(vs);
    glShaderSource(fs, 1, &kFsWhite, NULL); glCompileShader(fs); glAttachShader(q, fs); glDeleteShader(fs);
    glLinkProgram(q);
    GLint lq = 0; glGetProgramiv(q, GL_LINK_STATUS, &lq);
    if (lq != GL_TRUE) { printf("plain GL_POINTS program failed to link\n"); glDeleteProgram(q); free(plain_px); return false; }
    glUseProgram(q);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, n);
    if (glGetError() != GL_NO_ERROR) printf("plain GL_POINTS draw error\n");
    glFinish();
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, plain_px);
    glDeleteProgram(q);

    int only_tess = 0, only_plain = 0, both = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int a = tess_px[(y * w + x) * 4] > 200, b = plain_px[(y * w + x) * 4] > 200;
            if (a && b) both++;
            else if (a) only_tess++;
            else if (b) only_plain++;
        }
    printf("point_mode vs GL_POINTS: both=%d tess-only=%d plain-only=%d\n", both, only_tess, only_plain);
    free(plain_px);
    return both == expect && only_tess == 0 && only_plain == 0;
}

TEST(IsolinesGpu)
{
    GLuint p = prog(kTesIso);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    /* two control points spanning the screen horizontally; 3 isolines (outer0=3),
       8 segments each (outer1=8). The TES maps v to y in [-0.75, 0.75], so the lines
       sit at NDC y = -0.5, 0, +0.5 and interpolate red->green along x. */
    const float pos[2][4] = { { -0.9f, 0, 0, 1 }, { 0.9f, 0, 0, 1 } };
    const float col[2][4] = { { 1,0,0,1 }, { 0,1,0,1 } };
    setup(pos, col, 2, 2);

    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    float w = (float)vp[2], h = (float)vp[3];
    float c[4];

    /* each lit row must be one of the 3 isolines; count rows with non-black pixels */
    int lit_rows = 0;
    for (int y = 0; y < (int)h; y++)
    {
        int lit = 0;
        for (int x = (int)(w * 0.3f); x < (int)(w * 0.7f); x += 2)
        {
            glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, c);
            if (c[0] + c[1] + c[2] > 0.3f) { lit = 1; break; }
        }
        lit_rows += lit;
    }
    /* 3 thin lines in a 200-row window: allow a small band per line */
    EXPECT_TRUE(lit_rows >= 3 && lit_rows <= 24);

    /* the left end of a line is red-dominant, the right end green-dominant */
    int found = 0;
    for (int y = 0; y < (int)h && !found; y++)
    {
        glReadPixels((int)(w * 0.35f), y, 1, 1, GL_RGBA, GL_FLOAT, c);
        if (c[0] > 0.5f && c[1] < 0.5f)
        {
            found = 1;
            float c2[4];
            glReadPixels((int)(w * 0.65f), y, 1, 1, GL_RGBA, GL_FLOAT, c2);
            EXPECT_TRUE(c2[1] > 0.5f);   /* green-dominant at the right end */
        }
    }
    EXPECT_TRUE(found);
    glDeleteProgram(p);
}

TEST(QuadPointModeGpu)
{
    GLuint p = prog2(kTcsQuad, kTesQuadPt);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    /* a 2x2 inner grid (levels 1) emits 4 points at the corners of the viewport quad */
    const float pos[4][4] = { { -0.5f,-0.5f,0,1 }, { 0.5f,-0.5f,0,1 }, { 0.5f,0.5f,0,1 }, { -0.5f,0.5f,0,1 } };
    const float col[4][4] = { { 1,1,1,1 }, { 1,1,1,1 }, { 1,1,1,1 }, { 1,1,1,1 } };
    setup(pos, col, 4, 4);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int w = vp[2], h = vp[3];
    unsigned char *tess_px = calloc((size_t)w * h, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 4);
    EXPECT_NO_GL_ERROR();
    glFinish();
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, tess_px);

    int points = 0, xs[64], ys[64], npts = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (tess_px[(y * w + x) * 4] + tess_px[(y * w + x) * 4 + 1] + tess_px[(y * w + x) * 4 + 2] > 200)
            { points++; if (npts < 64) { xs[npts] = x; ys[npts] = y; npts++; } }
    printf("point_mode lit pixels:");
    for (int k = 0; k < npts; k++) printf(" (%d,%d)", xs[k], ys[k]);
    printf("\n");
    /* 4 points of 1 pixel each (gl_PointSize is 1) */
    EXPECT_TRUE(points >= 4 && points <= 16);

    /* the 4 points sit at the quad corners: NDC (+-0.5, +-0.5) -> window (0.25/0.75 of the
     * viewport). Point rasterization rounds to the pixel whose centre is closest; a window
     * coordinate exactly on a pixel edge goes to the LOWER pixel on this hardware (verified
     * against Apple's native GL), so accept the projected position +-1 px. */
    int corners = 0;
    int cx[4] = { (int)(w * 0.25f), (int)(w * 0.75f), (int)(w * 0.75f), (int)(w * 0.25f) };
    int cy[4] = { (int)(h * 0.25f), (int)(h * 0.25f), (int)(h * 0.75f), (int)(h * 0.75f) };
    for (int k = 0; k < 4; k++)
    {
        int lit = 0;
        for (int dy = -1; dy <= 1 && !lit; dy++)
            for (int dx = -1; dx <= 1 && !lit; dx++)
            {
                int x = cx[k] + dx, y = cy[k] + dy;
                if (x < 0 || y < 0 || x >= w || y >= h) continue;
                if (tess_px[(y * w + x) * 4] + tess_px[(y * w + x) * 4 + 1] + tess_px[(y * w + x) * 4 + 2] > 200)
                    lit = 1;
            }
        corners += lit;
    }
    EXPECT_TRUE(corners == 4);

    /* pixel-exact reference: the same 4 NDC positions drawn as plain GL_POINTS must
     * rasterize to exactly the same pixels as the tessellated point_mode output */
    EXPECT_TRUE(point_mode_matches_plain_points(tess_px, w, h, 4, 4));
    free(tess_px);
    glDeleteProgram(p);
}

TEST(TrianglePointModeGpu)
{
    GLuint p = prog2(kTcsTri, kTesTriPt);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    /* a level-1 triangle patch emits exactly its 3 control points */
    const float pos[3][4] = { { -0.5f,-0.5f,0,1 }, { 0.5f,-0.5f,0,1 }, { 0.0f,0.5f,0,1 } };
    const float col[3][4] = { { 1,1,1,1 }, { 1,1,1,1 }, { 1,1,1,1 } };
    setup(pos, col, 3, 3);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int w = vp[2], h = vp[3];
    unsigned char *tess_px = calloc((size_t)w * h, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glFinish();
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, tess_px);

    int points = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (tess_px[(y * w + x) * 4] + tess_px[(y * w + x) * 4 + 1] + tess_px[(y * w + x) * 4 + 2] > 200)
                points++;
    EXPECT_TRUE(points >= 3 && points <= 12);

    EXPECT_TRUE(point_mode_matches_plain_points(tess_px, w, h, 3, 3));
    free(tess_px);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(IsolinesGpu);
    RUN(QuadPointModeGpu);
    RUN(TrianglePointModeGpu);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}
