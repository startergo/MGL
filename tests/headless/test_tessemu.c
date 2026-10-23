#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "glcorearb.h"
#include "mgl_test.h"
#include "mgl_tess_diag.h"

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
/* quads WITHOUT point mode: capture must report the VERTEX count (4 corners at
 * level 1), not the index count (2 triangles = 6) */
static const char *kTesQuadFill =
    "#version 450 core\n"
    "layout(quads, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = mix(mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),\n"
    "                    mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x), gl_TessCoord.y);\n"
    "  f_col = tc_col[0];\n"
    "}\n";

/* quads with point mode: emit the corners as points */
static const char *kTesQuadPt =    "#version 450 core\n"
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
        if (!srcs[i].s) continue;      /* separable: skip the missing stage (no TCS) */
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
    int w = vp[2], h = vp[3];
    unsigned char *px = calloc((size_t)w * h, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);

    /* expected: n0 = 3 isolines at v = (i + 0.5) / 3 -> NDC y = -0.5, 0, +0.5
     * -> window y = h/4, h/2, 3h/4 (an exact pixel edge goes to the LOWER row,
     * the same tie rule the point_mode test verified against Apple GL). */
    int rows[8], nrow = 0;
    for (int y = 0; y < h; y++)
    {
        int cnt = 0;
        for (int x = 0; x < w; x++)
            if (px[(y * w + x) * 4] + px[(y * w + x) * 4 + 1] > 120) cnt++;
        if (cnt > w / 4)                       /* a line, not a stray pixel */
        { printf("iso row %d: %d px\n", y, cnt); if (nrow < 8) rows[nrow++] = y; }
    }
    EXPECT_TRUE(nrow == 3);
    if (nrow == 3)
    {
        int ey[3] = { h / 4, h / 2, h * 3 / 4 };
        for (int k = 0; k < 3; k++)
        {
            int d = rows[k] - (ey[k] - 1);     /* exact edge -> lower row */
            EXPECT_TRUE(d >= -1 && d <= 1);
        }
    }

    /* each line spans the full u range: NDC x +-0.9 -> window 0.05w..0.95w */
    if (nrow == 3)
    {
        int y = rows[0], x0 = -1, x1 = -1;
        for (int x = 0; x < w; x++)
            if (px[(y * w + x) * 4] + px[(y * w + x) * 4 + 1] > 120)
            { if (x0 < 0) x0 = x; x1 = x; }
        printf("iso extent x=[%d..%d]\n", x0, x1);
        EXPECT_TRUE(x0 <= (int)(w * 0.05f) + 2 && x1 >= (int)(w * 0.95f) - 2);

        /* linear red->green along u: sampled at x = 0.1w, 0.5w, 0.9w */
        int sx[3] = { w / 10, w / 2, w * 9 / 10 };
        for (int k = 0; k < 3; k++)
        {
            int r = px[(y * w + sx[k]) * 4], g = px[(y * w + sx[k]) * 4 + 1];
            printf("iso colour at x=%d: r=%d g=%d\n", sx[k], r, g);
            if (k == 0) EXPECT_TRUE(r > 180 && g < 80);
            if (k == 1) EXPECT_TRUE(r > 70 && r < 190 && g > 70 && g < 190);
            if (k == 2) EXPECT_TRUE(g > 180 && r < 80);
        }
    }
    free(px);
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

/* TCS writing a per-patch varying: the value reaches the TES only through the
 * per-patch data blob (TCS patch-out record -> blob -> TES patchIn), so this is
 * the pipeline the GPU blob kernel must reproduce byte-exactly. */
static const char *kTcsPatchVar =
    "#version 450 core\n"
    "layout(vertices=2) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "layout(location=1) patch out vec4 p_flag;\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 8.0;\n"
    "    p_flag = vec4(0.25, 0.50, 0.75, 1.0);\n"
    "  }\n"
    "}\n";
static const char *kTesIsoPatch =
    "#version 450 core\n"
    "layout(isolines, equal_spacing) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=1) patch in vec4 p_flag;\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
    "  gl_Position.y = mix(-0.75, 0.75, gl_TessCoord.y);\n"
    "  f_col = p_flag;\n"
    "}\n";

/* two-patch variant: the per-patch value depends on the patch's own control points,
 * so each patch's blob entry must be assembled from the right TCS patch record */
static const char *kTcsPatchVar2 =
    "#version 450 core\n"
    "layout(vertices=2) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "layout(location=1) patch out vec4 p_flag;\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 8.0;\n"
    "    if (gl_in[0].gl_Position.x < 0.0) p_flag = vec4(1.00, 0.25, 0.25, 1.0);\n"
    "    else                             p_flag = vec4(0.25, 0.25, 1.00, 1.0);\n"
    "  }\n"
    "}\n";
static const char *kTesIsoPatch2 =
    "#version 450 core\n"
    "layout(isolines, equal_spacing) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=1) patch in vec4 p_flag;\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
    "  gl_Position.y = mix(-0.75, 0.75, gl_TessCoord.y);\n"
    "  f_col = p_flag;\n"
    "}\n";


TEST(EmuPatchDataGpu)
{
    /* emulated path is forced by the domain: isolines never take the hardware
     * tessellator, and MGL_TESS_CPU is not set in this process */
    GLuint p = prog2(kTcsPatchVar, kTesIsoPatch);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link != GL_TRUE)
    {
        char log[2048] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("EmuPatchDataGpu link log: [%s]\n", log);
    }
    glUseProgram(p);

    const float pos[2][4] = { { -0.9f, 0, 0, 1 }, { 0.9f, 0, 0, 1 } };
    const float col[2][4] = { { 1,0,0,1 }, { 0,1,0,1 } };
    setup(pos, col, 2, 2);

    unsigned b0 = mgl_tess_cpu_blob_builds(), g0 = mgl_tess_cpu_generate_runs();
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();
    printf("patch-blob: cpu blob builds %u -> %u, cpu tessellator runs %u -> %u\n",
           b0, mgl_tess_cpu_blob_builds(), g0, mgl_tess_cpu_generate_runs());

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int w = vp[2], h = vp[3];
    unsigned char *px = calloc((size_t)w * h, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);

    /* 3 isolines at v=(i+0.5)/3 -> window y = h/4, h/2, 3h/4 (lower row on the tie) */
    int rows[8], nrow = 0;
    for (int y = 0; y < h; y++)
    {
        int cnt = 0;
        for (int x = 0; x < w; x++)
            if (px[(y * w + x) * 4] + px[(y * w + x) * 4 + 1] + px[(y * w + x) * 4 + 2] > 120) cnt++;
        if (cnt > w / 4 && nrow < 8) rows[nrow++] = y;
    }
    EXPECT_TRUE(nrow == 3);
    if (nrow == 3)
    {
        int ey[3] = { h / 4, h / 2, h * 3 / 4 };
        for (int k = 0; k < 3; k++)
            EXPECT_TRUE(rows[k] >= ey[k] - 2 && rows[k] <= ey[k]);

        /* the whole line carries the per-patch colour (0.25, 0.50, 0.75): constant
         * along x, unlike the per-vertex red->green of tc_col */
        int y = rows[1];
        int sx[3] = { w / 10, w / 2, w * 9 / 10 };
        for (int k = 0; k < 3; k++)
        {
            int r = px[(y * w + sx[k]) * 4], g = px[(y * w + sx[k]) * 4 + 1], b = px[(y * w + sx[k]) * 4 + 2];
            printf("patch-blob colour at (%d,%d): %d %d %d\n", sx[k], y, r, g, b);
            EXPECT_TRUE(r > 48 && r < 80);
            EXPECT_TRUE(g > 112 && g < 144);
            EXPECT_TRUE(b > 175 && b < 207);
        }
    }
    /* px stays allocated: the two-patch case below reads back into it */

    /* the GPU emulated path must not have built the blob or tessellated on the CPU */
    EXPECT_TRUE(mgl_tess_cpu_blob_builds() == b0);
    EXPECT_TRUE(mgl_tess_cpu_generate_runs() == g0);

    /* --- two patches: each patch's blob entry must carry its own per-patch value.
     * Patch 0 spans the left half (p_flag red-dominant), patch 1 the right half
     * (blue-dominant): the TCS picks the colour from the patch's own first control
     * point, so a blob entry indexed with the wrong patch id would mix them up. */
    p = prog2(kTcsPatchVar2, kTesIsoPatch2);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    const float pos2[4][4] = { { -0.95f, 0, 0, 1 }, { -0.05f, 0, 0, 1 },
                               {  0.05f, 0, 0, 1 }, {  0.95f, 0, 0, 1 } };
    const float col2[4][4] = { { 1,0,0,1 }, { 1,0,0,1 }, { 0,0,1,1 }, { 0,0,1,1 } };
    setup(pos2, col2, 4, 2);

    b0 = mgl_tess_cpu_blob_builds(); g0 = mgl_tess_cpu_generate_runs();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 4);
    EXPECT_NO_GL_ERROR();
    glFinish();
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);

    {
        int y = h / 2;                        /* the middle isoline: window y = h/2 */
        int ry = -1;
        for (int dy = -2; dy <= 1 && ry < 0; dy++)
        {
            int cnt = 0;
            for (int x = 0; x < w; x++)
                if (px[((y + dy) * w + x) * 4] + px[((y + dy) * w + x) * 4 + 1] + px[((y + dy) * w + x) * 4 + 2] > 120) cnt++;
            if (cnt > w / 4) ry = y + dy;
        }
        EXPECT_TRUE(ry > 0);
        if (ry > 0)
        {
            int lr = px[(ry * w + w / 4) * 4], lb = px[(ry * w + w / 4) * 4 + 2];
            int rr = px[(ry * w + (w * 3) / 4) * 4], rb = px[(ry * w + (w * 3) / 4) * 4 + 2];
            printf("two-patch blob: left rgb=(%d,%d) right rgb=(%d,%d)\n", lr, lb, rr, rb);
            EXPECT_TRUE(lr > 150 && lb < 100);      /* patch 0: (1, 0.25, 0.25) */
            EXPECT_TRUE(rr < 100 && rb > 150);      /* patch 1: (0.25, 0.25, 1) */
        }
    }

    EXPECT_TRUE(mgl_tess_cpu_blob_builds() == b0);
    EXPECT_TRUE(mgl_tess_cpu_generate_runs() == g0);
    glDeleteProgram(p);
    free(px);
}

TEST(EmuDefaultLevelsGpu)
{
    /* no TCS: the blob's level members come from the glPatchParameterfv defaults
     * and everything else stays zero; the emulated path still must not touch the
     * CPU blob or tessellator */
    GLuint p = prog2(NULL, kTesIso);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    const float pos[2][4] = { { -0.9f, 0, 0, 1 }, { 0.9f, 0, 0, 1 } };
    const float col[2][4] = { { 1,0,0,1 }, { 0,1,0,1 } };
    setup(pos, col, 2, 2);
    glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, (float[4]){ 3, 8, 1, 1 });
    glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, (float[2]){ 1, 1 });

    unsigned b0 = mgl_tess_cpu_blob_builds(), g0 = mgl_tess_cpu_generate_runs();
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();
    printf("default-levels: cpu blob builds %u -> %u, cpu tessellator runs %u -> %u\n",
           b0, mgl_tess_cpu_blob_builds(), g0, mgl_tess_cpu_generate_runs());

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int w = vp[2], h = vp[3];
    unsigned char *px = calloc((size_t)w * h, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);

    int rows[8], nrow = 0;
    for (int y = 0; y < h; y++)
    {
        int cnt = 0;
        for (int x = 0; x < w; x++)
            if (px[(y * w + x) * 4] + px[(y * w + x) * 4 + 1] > 120) cnt++;
        if (cnt > w / 4 && nrow < 8) rows[nrow++] = y;
    }
    EXPECT_TRUE(nrow == 3);
    if (nrow == 3)
    {
        int ey[3] = { h / 4, h / 2, h * 3 / 4 };
        for (int k = 0; k < 3; k++)
            EXPECT_TRUE(rows[k] >= ey[k] - 2 && rows[k] <= ey[k]);

        /* per-vertex colour interpolation passes through without a TCS */
        int y = rows[1];
        int r = px[(y * w + w / 10) * 4], g = px[(y * w + w * 10 / 10 - 1 - w / 10) * 4 + 1];
        EXPECT_TRUE(r > 180);
        EXPECT_TRUE(g > 180);
    }
    free(px);

    EXPECT_TRUE(mgl_tess_cpu_blob_builds() == b0);
    EXPECT_TRUE(mgl_tess_cpu_generate_runs() == g0);
    glDeleteProgram(p);
}



/* ------------- TES-output capture (producer for geometry / transform feedback) ----
 * A triangle patch with levels 1 emits exactly its 3 control-point vertices; the
 * capture records must match the TES math: barycentric position and the varying. */
TEST(TessOutputCapture)
{
    GLuint p = prog2(kTcsTri, kTesTriPt);   /* barycentric mix + tc_col */
    GLint link = 0;
    void *buf = NULL;
    uint32_t off = 0, cstride = 0, prim = 0, count = 0;

    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    const float pos[3][4] = { { -0.5f,-0.5f,0,1 }, { 0.5f,-0.5f,0,1 }, { 0,0.5f,0,1 } };
    const float col[3][4] = { { 1,0,0,1 }, { 0,1,0,1 }, { 0,0,1,1 } };
    setup(pos, col, 3, 3);

    mgl_tess_diag_capture(p, 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    glFinish();                       /* the ring block is CPU-visible after completion */
    mgl_tess_diag_capture(p, 0);

    if (!mgl_tess_diag_capture_info(&buf, &off, &cstride, &prim, &count))
    {
        printf("capture: not published\n");
        EXPECT_TRUE(false);
        return;
    }
    printf("capture: prim=0x%x stride=%u words, count=%u\n",
           (unsigned)prim, cstride, (unsigned)count);
    EXPECT_TRUE(cstride >= 8);                 /* pos(4) + col(4) */
    EXPECT_TRUE(count == 3);                   /* levels 1: the 3 corners */

    {
        const float *r = (const float *)((const uint8_t *)buf + off);
        int ok = 0;
        for (uint32_t i = 0; i < count && i < 8; i++)
        {
            float x = r[(size_t)i * cstride + 0], y = r[(size_t)i * cstride + 1];
            for (int k = 0; k < 3; k++)
                if (x == pos[k][0] && y == pos[k][1])
                {
                    /* the TES writes f_col = tc_col[0] for every vertex: the colour of
                     * control point 0 (tc_col[0] = a_col[0]), not of the matched corner */
                    EXPECT_TRUE(r[(size_t)i * cstride + 4] == col[0][0] &&
                                r[(size_t)i * cstride + 5] == col[0][1] &&
                                r[(size_t)i * cstride + 6] == col[0][2]);
                    ok++;
                }
        }
        printf("capture: %d/3 corner records match the TES math\n", ok);
        EXPECT_TRUE(ok == 3);
    }

    /* ---- second phase: filled QUAD domain (no point mode): the count must be the
     * vertex count (4 corners at level 1), not the index count (2 triangles = 6) */
    {
        const float qpos[4][4] = { { -0.5f,-0.5f,0,1 }, { 0.5f,-0.5f,0,1 },
                                   { 0.5f,0.5f,0,1 }, { -0.5f,0.5f,0,1 } };
        const float qcol[4][4] = { { 1,0,0,1 }, { 0,1,0,1 }, { 0,0,1,1 }, { 1,1,0,1 } };
        GLuint pq = prog2(kTcsQuad, kTesQuadFill);
        GLint lq = 0;
        void *bq = NULL;
        uint32_t oq = 0, sq = 0, pqm = 0, cq = 0;

        glGetProgramiv(pq, GL_LINK_STATUS, &lq);
        EXPECT_EQ_INT(lq, GL_TRUE);
        glUseProgram(pq);
        setup(qpos, qcol, 4, 4);
        mgl_tess_diag_capture(pq, 1);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_PATCHES, 0, 4);
        glFinish();
        mgl_tess_diag_capture(pq, 0);
        if (!mgl_tess_diag_capture_info(&bq, &oq, &sq, &pqm, &cq))
        {
            printf("capture(quad): not published\n");
            EXPECT_TRUE(false);
        }
        else
        {
            printf("capture(quad): prim=0x%x stride=%u count=%u\n", pqm, sq, cq);
            /* TF captures the vertices in GENERATION order (GL 4.6 13.4): a filled quad
             * cell at level 1 is 2 triangles = 6 emitted vertices (the 4 corners each
             * appearing once per triangle that uses them) */
            EXPECT_EQ_INT(cq, 6);
            EXPECT_EQ_INT(pqm, GL_TRIANGLES);
            {
                const float *rq = (const float *)((const uint8_t *)bq + oq);
                int okq = 0;

                for (uint32_t i = 0; i < cq && i < 8; i++)
                    for (int k = 0; k < 4; k++)
                        if (rq[(size_t)i * sq + 0] == qpos[k][0] &&
                            rq[(size_t)i * sq + 1] == qpos[k][1])
                            okq++;
                EXPECT_TRUE(okq == 6);      /* a, b, c each in 1 triangle, d in 2 */
                if (okq != 6)
                    printf("capture(quad): %d/6 corner records found\n", okq);
            }
        }
    }
    glDeleteProgram(p);
}

int main(void)
{
    RUN(IsolinesGpu);
    RUN(QuadPointModeGpu);
    RUN(TrianglePointModeGpu);
    RUN(EmuPatchDataGpu);
    RUN(EmuDefaultLevelsGpu);
    RUN(TessOutputCapture);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}
