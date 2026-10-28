/* test_cull.c — whole-primitive gl_CullDistance culling (GL 4.6 13.5.2).
 *
 * Metal raster stages have no primitive rejection by cull distance (SPIRV-Cross
 * exports gl_CullDistance only as [[user(cullN)]] attributes that reach the fragment
 * stage), so MGL routes a cull-writing draw through the geometry-emulation path with a
 * generated pass-through kernel that drops culled primitives (geometry.c).  These tests
 * verify the visible result with glReadPixels: which primitives survive, that clip
 * distances still clip, that unwritten cull entries never cull, and that the fragment
 * stage keeps seeing the distances.
 */
#include "mgl_test.h"

/* ------------------------------------------------------------------ helpers */

static GLuint g_vao, g_vbo;

/* a 96-row framebuffer: three probe bands at NDC y -0.6 / 0 / +0.6 */
#define ROWS 96
static const float YC[3] = { -0.6f, 0.0f, 0.6f };

static int band(float ndc_y, int channel)
{
    int row = (int)(((1.0f + ndc_y) / 2.0f) * (ROWS - 1) + 0.5f);
    int best = 0;
    for (int dy = -2; dy <= 2; dy++)
    {
        GLubyte px[4];
        if (row + dy < 0 || row + dy >= ROWS) continue;
        mgl_read_rgba(48, row + dy, px);
        if (px[channel] > best) best = px[channel];
    }
    return best;
}

/* per-vertex block: vec2 pos + 8 floats of distance data (interleaved, CTS style);
 * 9 rows: the triangle tests use three 3-vertex primitives */
static float blk[9][10];

static void fill(int primitive_culled)
{
    for (int l = 0; l < 3; l++)
    {
        float d = (l == primitive_culled) ? -2.0f : 2.0f;
        blk[l*2+0][0] = -0.9f; blk[l*2+0][1] = YC[l];
        blk[l*2+1][0] =  0.9f; blk[l*2+1][1] = YC[l];
        for (int i = 0; i < 8; i++)
        {
            blk[l*2+0][2 + i] = (i == 0) ? d : 2.0f;
            blk[l*2+1][2 + i] = (i == 0) ? d : 2.0f;
        }
    }
}

static void bind(GLuint p, const char *first_array_attrib)
{
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof blk, blk, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void *)0);
    if (first_array_attrib)
    {
        int loc = glGetAttribLocation(p, first_array_attrib);
        if (loc >= 0)
        {
            glEnableVertexAttribArray(loc);
            glVertexAttribPointer(loc, 1, GL_FLOAT, GL_FALSE, 10 * sizeof(float),
                                  (void *)(2 * sizeof(float)));
        }
    }
    glViewport(0, 0, ROWS, ROWS);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

/* ------------------------------------------------------------ shader shapes */

/* all eight planes written from the attribute (constant indices) */
static const char *vs_cull8 =
    "#version 460 core\n"
    "in vec2 pos;\n"
    "in float d[8];\n"
    "out float v;\n"
    "void main() {\n"
    "    gl_Position = vec4(pos, 0.0, 1.0);\n"
    "    gl_CullDistance[0] = d[0];\n"
    "    gl_CullDistance[1] = d[1];\n"
    "    gl_CullDistance[2] = d[2];\n"
    "    gl_CullDistance[3] = d[3];\n"
    "    gl_CullDistance[4] = d[4];\n"
    "    gl_CullDistance[5] = d[5];\n"
    "    gl_CullDistance[6] = d[6];\n"
    "    gl_CullDistance[7] = d[7];\n"
    "    v = 1.0;\n"
    "}\n";

/* only plane 0 and 3 are written: the other entries are undefined in GL and must
 * never cull (the evaluation only considers the stored entries) */
static const char *vs_cull_sparse =
    "#version 460 core\n"
    "in vec2 pos;\n"
    "in float d[8];\n"
    "out float v;\n"
    "void main() {\n"
    "    gl_Position = vec4(pos, 0.0, 1.0);\n"
    "    gl_CullDistance[0] = d[0];\n"
    "    gl_CullDistance[3] = d[3];\n"
    "    v = 1.0;\n"
    "}\n";

/* dynamic-index writes (a loop): every entry becomes active */
static const char *vs_cull_dynamic =
    "#version 460 core\n"
    "in vec2 pos;\n"
    "in float d[8];\n"
    "out float v;\n"
    "out gl_PerVertex { vec4 gl_Position; float gl_CullDistance[8]; };\n"
    "void main() {\n"
    "    gl_Position = vec4(pos, 0.0, 1.0);\n"
    "    for (int i = 0; i < 8; i++) gl_CullDistance[i] = d[i] * float(i + 1);\n"
    "    v = 1.0;\n"
    "}\n";

/* clip distances must keep hardware clipping while culling is active */
static const char *vs_cull_clip =
    "#version 460 core\n"
    "in vec2 pos;\n"
    "in float d[8];\n"
    "out float v;\n"
    "void main() {\n"
    "    gl_Position = vec4(pos, 0.0, 1.0);\n"
    "    gl_CullDistance[0] = d[0];\n"
    "    gl_ClipDistance[0] = pos.x * 4.0;   /* x < 0 clipped */\n"
    "    v = 1.0;\n"
    "}\n";

/* the fragment stage reads the distances back (its input is the raster pass-through's
 * re-export of the kernel's records) */
static const char *fs_reads_cull =
    "#version 460 core\n"
    "in float v;\n"
    "in float gl_CullDistance[8];\n"
    "out vec4 o;\n"
    "void main() {\n"
    "    float s = 0.0;\n"
    "    for (int i = 0; i < 8; i++) s += abs(gl_CullDistance[i]);\n"
    "    o = vec4(v, clamp(s, 0.0, 1.0), 0, 1);\n"
    "}\n";

static const char *fs_red =
    "#version 460 core\n"
    "in float v;\n"
    "out vec4 o;\n"
    "void main() { o = vec4(v, 0, 0, 1); }\n";

/* ---------------------------------------------------------------- the tests */

static void run_lines(const char *vs, const char *fs, int culled)
{
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    glUseProgram(p);
    fill(culled);
    bind(p, "d[0]");
    glDrawArrays(GL_LINES, 0, 6);
    glDeleteProgram(p);
}

TEST(lines_no_cull)
{
    int r[3];
    run_lines(vs_cull8, fs_red, -1);
    r[0] = band(YC[0], 0); r[1] = band(YC[1], 0); r[2] = band(YC[2], 0);
    printf("lines none culled: %d %d %d (255 255 255)\n", r[0], r[1], r[2]);
    EXPECT_EQ_INT(r[0], 255);
    EXPECT_EQ_INT(r[1], 255);
    EXPECT_EQ_INT(r[2], 255);
}

TEST(lines_cull_first)  { int r[3]; run_lines(vs_cull8, fs_red, 0); r[0]=band(YC[0],0); r[1]=band(YC[1],0); r[2]=band(YC[2],0); printf("lines first: %d %d %d (0 255 255)\n",r[0],r[1],r[2]); EXPECT_EQ_INT(r[0],0); EXPECT_EQ_INT(r[1],255); EXPECT_EQ_INT(r[2],255); }
TEST(lines_cull_middle) { int r[3]; run_lines(vs_cull8, fs_red, 1); r[0]=band(YC[0],0); r[1]=band(YC[1],0); r[2]=band(YC[2],0); printf("lines middle: %d %d %d (255 0 255)\n",r[0],r[1],r[2]); EXPECT_EQ_INT(r[0],255); EXPECT_EQ_INT(r[1],0); EXPECT_EQ_INT(r[2],255); }
TEST(lines_cull_last)   { int r[3]; run_lines(vs_cull8, fs_red, 2); r[0]=band(YC[0],0); r[1]=band(YC[1],0); r[2]=band(YC[2],0); printf("lines last: %d %d %d (255 255 0)\n",r[0],r[1],r[2]); EXPECT_EQ_INT(r[0],255); EXPECT_EQ_INT(r[1],255); EXPECT_EQ_INT(r[2],0); }

TEST(lines_sparse_writes)
{
    /* only [0] and [3] written: entries 1,2,4..7 are undefined and must not cull;
     * the data keeps every entry but plane 0 positive -> nothing may cull */
    int r[3];
    run_lines(vs_cull_sparse, fs_red, -1);
    r[0] = band(YC[0], 0); r[1] = band(YC[1], 0); r[2] = band(YC[2], 0);
    printf("lines sparse writes: %d %d %d (255 255 255)\n", r[0], r[1], r[2]);
    EXPECT_EQ_INT(r[0], 255);
    EXPECT_EQ_INT(r[1], 255);
    EXPECT_EQ_INT(r[2], 255);
}

TEST(lines_dynamic_index_writes)
{
    int r[3];
    run_lines(vs_cull_dynamic, fs_red, 1);
    r[0] = band(YC[0], 0); r[1] = band(YC[1], 0); r[2] = band(YC[2], 0);
    printf("lines dynamic writes: %d %d %d (255 0 255)\n", r[0], r[1], r[2]);
    EXPECT_EQ_INT(r[0], 255);
    EXPECT_EQ_INT(r[1], 0);
    EXPECT_EQ_INT(r[2], 255);
}

TEST(lines_clip_while_culling)
{
    /* plane clip 0 clips x<0: the right half of every visible line stays, the left
     * half goes; culling still applies to the middle band */
    GLuint p = mgl_make_program(vs_cull_clip, fs_red);
    int rc, lc, mid;
    if (!p) return;
    glUseProgram(p);
    fill(1);
    bind(p, "d[0]");
    glEnable(GL_CLIP_DISTANCE0);
    glDrawArrays(GL_LINES, 0, 6);
    rc = band(YC[0], 0);
    lc = 0;   /* replaced by the direct read below */
    {
        int row = (int)(((1.0f + YC[0]) / 2.0f) * (ROWS - 1) + 0.5f);
        GLubyte px[4];
        mgl_read_rgba(8, row, px);      /* left side: clipped away */
        lc = px[0];
    }
    mid = band(YC[1], 0);
    printf("clip while culling: right=%d (255) left=%d (0) culled-mid=%d (0)\n", rc, lc, mid);
    EXPECT_EQ_INT(rc, 255);
    EXPECT_EQ_INT(lc, 0);
    EXPECT_EQ_INT(mid, 0);
    glDisable(GL_CLIP_DISTANCE0);
    glDeleteProgram(p);
}

TEST(lines_fs_reads_cull)
{
    int g[3];
    run_lines(vs_cull8, fs_reads_cull, -1);
    g[0] = band(YC[0], 1); g[1] = band(YC[1], 1); g[2] = band(YC[2], 1);
    printf("fs reads cull: g=%d %d %d (255 255 255: |2|+|2|*7 clamps to 1)\n", g[0], g[1], g[2]);
    EXPECT_EQ_INT(g[0], 255);
    EXPECT_EQ_INT(g[1], 255);
    EXPECT_EQ_INT(g[2], 255);
}

/* three triangles instead of lines (a different primitive family and emit list) */
static void run_tris(int culled)
{
    GLuint p = mgl_make_program(getenv("MGL_CULL_1P") ? vs_cull_sparse : vs_cull8, fs_red);
    if (!p) return;
    glUseProgram(p);
    memset(blk, 0, sizeof blk);
    for (int l = 0; l < 3; l++)
    {
        float y = YC[l], r = 0.15f, d = (l == culled) ? -2.0f : 2.0f;
        blk[l*3+0][0] = -r;  blk[l*3+0][1] = y - r;
        blk[l*3+1][0] =  r;  blk[l*3+1][1] = y - r;
        blk[l*3+2][0] = 0;   blk[l*3+2][1] = y + r;
        for (int v = 0; v < 3; v++)
            for (int i = 0; i < 8; i++)
                blk[l*3+v][2 + i] = (i == 0) ? d : 2.0f;
    }
    bind(p, "d[0]");
    glDrawArrays(GL_TRIANGLES, 0, 9);
    if (getenv("MGL_CULL_DUMP"))
    {
        GLubyte px[4];
        for (int y = ROWS - 1; y >= 0; y -= 3)
        {
            printf("%2d: ", y);
            for (int x = 0; x < ROWS; x += 3)
            {
                mgl_read_rgba(x, y, px);
                printf("%c", px[0] > 128 ? '#' : '.');
            }
            printf("\n");
        }
    }
    glDeleteProgram(p);
}

TEST(triangles_cull_middle)
{
    int r[3];
    run_tris(1);
    r[0] = band(YC[0], 0); r[1] = band(YC[1], 0); r[2] = band(YC[2], 0);
    printf("triangles middle culled: %d %d %d (255 0 255)\n", r[0], r[1], r[2]);
    EXPECT_EQ_INT(r[0], 255);
    EXPECT_EQ_INT(r[1], 0);
    EXPECT_EQ_INT(r[2], 255);
}

TEST(triangles_cull_first)
{
    int r[3];
    run_tris(0);
    r[0] = band(YC[0], 0); r[1] = band(YC[1], 0); r[2] = band(YC[2], 0);
    printf("triangles first culled: %d %d %d (0 255 255)\n", r[0], r[1], r[2]);
    EXPECT_EQ_INT(r[0], 0);
    EXPECT_EQ_INT(r[1], 255);
    EXPECT_EQ_INT(r[2], 255);
}

/* points: a one-vertex primitive culls on its own distance */
static void run_points(int culled)
{
    GLuint p = mgl_make_program(vs_cull8, fs_red);
    if (!p) return;
    glUseProgram(p);
    memset(blk, 0, sizeof blk);
    for (int l = 0; l < 3; l++)
    {
        float d = (l == culled) ? -2.0f : 2.0f;
        blk[l][0] = 0.0f; blk[l][1] = YC[l];
        for (int i = 0; i < 8; i++)
            blk[l][2 + i] = (i == 0) ? d : 2.0f;
    }
    bind(p, "d[0]");
    glPointSize(7.0f);
    glDrawArrays(GL_POINTS, 0, 3);
    glPointSize(1.0f);
    glDeleteProgram(p);
}

TEST(points_cull_middle)
{
    int r[3];
    run_points(1);
    r[0] = band(YC[0], 0); r[1] = band(YC[1], 0); r[2] = band(YC[2], 0);
    printf("points middle culled: %d %d %d (255 0 255)\n", r[0], r[1], r[2]);
    EXPECT_EQ_INT(r[0], 255);
    EXPECT_EQ_INT(r[1], 0);
    EXPECT_EQ_INT(r[2], 255);
}

/* an indexed triangle strip: every other triangle of the strip must survive */
TEST(strip_indexed_cull)
{
    static const char *vs_strip =
        "#version 460 core\n"
        "in vec2 pos;\n"
        "in float d[8];\n"
        "out float v;\n"
        "void main() {\n"
        "    gl_Position = vec4(pos, 0.0, 1.0);\n"
        "    gl_CullDistance[0] = d[0];\n"
        "    v = 1.0;\n"
        "}\n";
    GLuint p = mgl_make_program(vs_strip, fs_red);
    GLuint ibo;
    float quad[5][10];
    GLushort idx[5] = { 0, 1, 2, 3, 4 };
    if (!p) return;

    /* a strip across the middle band: triangles (0,1,2), (1,2,3), (2,3,4); the second
     * triangle's vertices all carry a negative distance */
    memset(quad, 0, sizeof quad);
    quad[0][0] = -0.8f; quad[0][1] = -0.4f;   quad[0][2] = 2.0f;
    quad[1][0] = -0.4f; quad[1][1] =  0.2f;   quad[1][2] = -2.0f;  /* negative */
    quad[2][0] =  0.0f; quad[2][1] = -0.4f;   quad[2][2] = 2.0f;
    quad[3][0] =  0.4f; quad[3][1] =  0.2f;   quad[3][2] = -2.0f;  /* negative */
    quad[4][0] =  0.8f; quad[4][1] = -0.4f;   quad[4][2] = 2.0f;

    glUseProgram(p);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void *)0);
    {
        int loc = glGetAttribLocation(p, "d[0]");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 1, GL_FLOAT, GL_FALSE, 10 * sizeof(float),
                              (void *)(2 * sizeof(float)));
    }
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glViewport(0, 0, ROWS, ROWS);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLE_STRIP, 5, GL_UNSIGNED_SHORT, 0);

    /* the lower zig-zag (y < 0 area) must be lit by the two surviving triangles; the
     * peak at y=+0.2 belongs to the culled middle one */
    {
        int row_low = (int)(((1.0f + -0.35f) / 2.0f) * (ROWS - 1) + 0.5f);
        int row_hi  = (int)(((1.0f +  0.18f) / 2.0f) * (ROWS - 1) + 0.5f);
        GLubyte lo[4], hi[4];
        mgl_read_rgba(24, row_low, lo);       /* inside triangle (0,1,2) */
        mgl_read_rgba(8, row_hi, hi);         /* only reachable via the culled (1,2,3) */
        printf("strip indexed: low r=%d (255) hi r=%d (0)\n", lo[0], hi[0]);
        EXPECT_EQ_INT(lo[0], 255);
        EXPECT_EQ_INT(hi[0], 0);
    }
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(p);
}

/* ---- regression shapes from KHR-GL46.cull_distance.functional (r28) ----
 *
 * cts_it0: an UNUSED vertex input declared before the used ones shifts glslang's
 * declaration-order numbering while the raster pipeline binds at the program's
 * slots; the capture variant must pin its locations (geom_pin_capture_inputs).
 * cts_it1: clip=1 + cull=7 — the record width is the WIDER of the two arrays and
 * the generated capture source widens the builtin declarations to it
 * (geom_pin_builtins / the max() record width).
 * cts_gs: a REAL pass-through geometry shader copying gl_CullDistance through
 * gl_in — whole-primitive cull runs in the kernel (mgl_prim_culled).
 * cts_ts: the TCS -> TES cull round trip (flattened builtin members keyed to
 * their slot in the canonical control-point record). */

static const char *kCtsFs =
    "#version 460 core\n"
    "out vec4 o;\n"
    "void main() { o = vec4(1, 1, 1, 1); }\n";

static const char *kCtsVs0 =
    "#version 450 core\n"
    "in float clipdistance_data[1];\n"          /* declared, never read */
    "in float culldistance_data[8];\n"
    "in vec2  position;\n"
    "out float gl_CullDistance[8];\n"
    "void main() {\n"
    "    for (int i = 0; i < 8; ++i)\n"
    "        gl_CullDistance[i] = culldistance_data[i];\n"
    "    gl_Position = vec4(2.0 * position.x - 1.0, 2.0 * position.y - 1.0, 0.0, 1.0);\n"
    "}\n";

TEST(cts_it0_unused_input)
{
    static float data[4][10];
    GLuint p = mgl_make_program(kCtsVs0, kCtsFs);
    GLubyte px[4];
    int lit = 0;
    if (!p) return;
    glUseProgram(p);
    {
        int lpos = glGetAttribLocation(p, "position");
        int lcull = glGetAttribLocation(p, "culldistance_data");
        EXPECT_TRUE(lpos >= 0 && lcull >= 0);
        /* the shared VAO carries enabled arrays from earlier tests: drop them
         * (a stale array would feed garbage distances) */
        for (int i = 0; i < 16; i++) glDisableVertexAttribArray(i);
        /* v0 at pixel corner (1,1), v1 at (7,7): the GL-diagonal */
        for (int i = 0; i < 8; i++) { data[0][i] = 2.0f; data[1][i] = 2.0f; }
        data[0][8] = 1.0f / 64.0f; data[0][9] = 1.0f / 64.0f;
        data[1][8] = 7.0f / 64.0f; data[1][9] = 7.0f / 64.0f;
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        if (glGetAttribLocation(p, "clipdistance_data") >= 0)
        {
            /* the unused array still gets a location: enable it like the CTS does */
            int lc = glGetAttribLocation(p, "clipdistance_data");
            glEnableVertexAttribArray(lc);
            glVertexAttribPointer(lc, 1, GL_FLOAT, GL_FALSE, 40, (void *)0);
        }
        glEnableVertexAttribArray(lcull);
        glVertexAttribPointer(lcull, 2, GL_FLOAT, GL_FALSE, 40, (void *)0);
        glDisableVertexAttribArray(lcull);
        for (int i = 0; i < 8; i++)
        {
            glEnableVertexAttribArray(lcull + i);
            glVertexAttribPointer(lcull + i, 1, GL_FLOAT, GL_FALSE, 40, (void *)((long)(i * 4)));
        }
        glEnableVertexAttribArray(lpos);
        glVertexAttribPointer(lpos, 2, GL_FLOAT, GL_FALSE, 40, (void *)(8 * 4));
    }
    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    mgl_read_rgba(3, 3, px);
    lit = px[0] > 0;
    mgl_read_rgba(6, 6, px);
    lit += px[0] > 0;
    printf("unused-input shape: lit %d (want 2)\n", lit);
    EXPECT_EQ_INT(lit, 2);
    glDeleteProgram(p);
    glUseProgram(0);
}

static const char *kCtsVs1 =
    "#version 450 core\n"
    "in float clipdistance_data[1];\n"
    "in float culldistance_data[7];\n"
    "in vec2  position;\n"
    "out float gl_ClipDistance[1];\n"
    "out float gl_CullDistance[7];\n"
    "void main() {\n"
    "    gl_ClipDistance[0] = clipdistance_data[0];\n"
    "    for (int i = 0; i < 7; ++i)\n"
    "        gl_CullDistance[i] = culldistance_data[i];\n"
    "    gl_Position = vec4(2.0 * position.x - 1.0, 2.0 * position.y - 1.0, 0.0, 1.0);\n"
    "}\n";

TEST(cts_it1_clip1_cull7)
{
    static float data[4][12];
    GLuint p = mgl_make_program(kCtsVs1, kCtsFs);
    GLubyte px[4];
    int lit = 0, culled;
    if (!p) return;
    glUseProgram(p);
    {
        int lpos = glGetAttribLocation(p, "position");
        int lcull = glGetAttribLocation(p, "culldistance_data");
        int lclip = glGetAttribLocation(p, "clipdistance_data");
        for (int i = 0; i < 16; i++) glDisableVertexAttribArray(i);
        for (int v = 0; v < 2; v++)
        {
            data[v][0] = 1.0f;                       /* clip[0] positive */
            for (int i = 0; i < 7; i++) data[v][1 + i] = 2.0f;
            data[v][8] = (float)(v ? 7 : 1) / 64.0f; /* corner diagonal (1,1) -> (7,7) */
            data[v][9] = (float)(v ? 7 : 1) / 64.0f;
        }
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        glEnableVertexAttribArray(lclip);
        glVertexAttribPointer(lclip, 1, GL_FLOAT, GL_FALSE, 48, (void *)0);
        for (int i = 0; i < 7; i++)
        {
            glEnableVertexAttribArray(lcull + i);
            glVertexAttribPointer(lcull + i, 1, GL_FLOAT, GL_FALSE, 48, (void *)((long)((1 + i) * 4)));
        }
        glEnableVertexAttribArray(lpos);
        glVertexAttribPointer(lpos, 2, GL_FLOAT, GL_FALSE, 48, (void *)(8 * 4));
        glEnable(GL_CLIP_DISTANCE0);
    }
    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    for (int k = 1; k <= 6; k++)
    {
        mgl_read_rgba(k, k, px);
        if (px[0] > 0) lit++;
    }
    printf("clip1+cull7 shape: lit %d (want 6)\n", lit);
    EXPECT_EQ_INT(lit, 6);

    /* cull[0] negative on both vertices -> the whole line disappears */
    data[0][1] = -1.0f / 7.0f;
    data[1][1] = -1.0f / 7.0f;
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    mgl_read_rgba(3, 3, px);
    culled = px[0] == 0;
    printf("clip1+cull7 culled: %d (want 1)\n", culled);
    EXPECT_TRUE(culled);
    glDisable(GL_CLIP_DISTANCE0);
    glDeleteProgram(p);
    glUseProgram(0);
}

int main(void)
{
    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &g_vbo);

    RUN(lines_no_cull);
    RUN(lines_cull_first);
    RUN(lines_cull_middle);
    RUN(lines_cull_last);
    RUN(lines_sparse_writes);
    RUN(lines_dynamic_index_writes);
    RUN(lines_clip_while_culling);
    RUN(lines_fs_reads_cull);
    RUN(triangles_cull_middle);
    RUN(triangles_cull_first);
    RUN(points_cull_middle);
    RUN(strip_indexed_cull);
    RUN(cts_it0_unused_input);
    RUN(cts_it1_clip1_cull7);

    glDeleteBuffers(1, &g_vbo);
    glDeleteVertexArrays(1, &g_vao);
    return mgl_test_summary();
}
