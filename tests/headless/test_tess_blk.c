/* test_tess_blk.c — TCS->TES interface-block and struct varying crossing.
 *
 * Exercised shapes (all render green when every leaf value crosses):
 *  - block with plain members            (BlkV { vec4 a; vec4 b; })
 *  - block with a struct member          (S s; S = { int x; vec4 y; float z[2]; })
 *  - block with an array member          (float fa[3])
 *  - block with an array-of-struct       (S sa[2])   — per-leaf dissolve vars
 *  - block with all of the above mixed
 *  - plain (non-block) struct varyings   (out S vb_s[]; out float vb_fa[][3];)
 *
 * Whole-struct loads (S s0 = vb[0].s;) are NOT supported by SPIRV-Cross on
 * Metal: MGL declines them with a clean LINK error instead of crashing
 * (TessBlkWholeStructLoadDeclined).
 */
#include "mgl_test.h"
#include <string.h>

static const char *k_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out vec4 vs_pos;\n"
    "void main(){ gl_Position = vec4(pos,0,1); vs_pos = vec4(pos,0,1); }\n";

static const char *k_fs =
    "#version 450 core\n"
    "in vec4 col; out vec4 o;\n"
    "void main(){ o = col; }\n";

#define TCS_HEAD "#version 450 core\nlayout(vertices=3) out;\nin vec4 vs_pos[];\n" \
                 "struct S { int x; vec4 y; float z[2]; };\n"
#define TCS_TAIL "  if (gl_InvocationID == 0) {\n" \
                 "    gl_TessLevelOuter[0]=3.0; gl_TessLevelOuter[1]=3.0; gl_TessLevelOuter[2]=3.0;\n" \
                 "    gl_TessLevelInner[0]=3.0;\n  }\n}\n"
#define TES_HEAD "#version 450 core\nlayout(triangles, equal_spacing, ccw) in;\nin vec4 tc_pos[];\n" \
                 "struct S { int x; vec4 y; float z[2]; };\n"
#define TES_TAIL "  vec3 p = gl_TessCoord.x*tc_pos[0].xyz + gl_TessCoord.y*tc_pos[1].xyz" \
                 " + gl_TessCoord.z*tc_pos[2].xyz;\n" \
                 "  gl_Position = vec4(p.xy*0.5, 0, 1);\n" \
                 "  col = ok ? vec4(0,1,0,1) : vec4(1,0,0,1);\n}\n"

static int draw_and_expect_green(const char *tcs, const char *tes)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[] = {
        { GL_VERTEX_SHADER, k_vs },
        { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes },
        { GL_FRAGMENT_SHADER, k_fs },
    };
    for (unsigned i = 0; i < 4; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL);
        glCompileShader(so);
        GLint cok = 0; glGetShaderiv(so, GL_COMPILE_STATUS, &cok);
        if (!cok) { char log[2048]; glGetShaderInfoLog(so, sizeof log, NULL, log); fprintf(stderr, "compile: %s\n", log); return 2; }
        glAttachShader(p, so);
    }
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (!lk) { char log[4096]; glGetProgramInfoLog(p, sizeof log, NULL, log); fprintf(stderr, "link: %s\n", log); return 3; }

    static const float pos[3][2] = { { -1, 1 }, { -1, -1 }, { 1, -1 } };
    GLuint vao, buf;
    glViewport(0, 0, 512, 512);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glUseProgram(p);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    /* the TES scales positions by 0.5: the triangle covers the viewport's
     * centre quarter (128..384); scan it */
    float px[4] = { -1, -1, -1, -1 };
    int any = 0;
    for (int yy = 128; yy < 384 && !any; yy += 4)
        for (int xx = 128; xx < 384 && !any; xx += 4)
        {
            float q[4];
            glReadPixels(xx, yy, 1, 1, GL_RGBA, GL_FLOAT, q);
            if (q[0] > 0.01f || q[1] > 0.01f || q[2] > 0.01f) { memcpy(px, q, sizeof px); any = 1; }
        }
    printf("pixel = (%.2f %.2f %.2f %.2f)%s\n", px[0], px[1], px[2], px[3], any ? "" : " (nothing)");
    EXPECT_TRUE(any);
    EXPECT_NEAR(px[1], 1.0f, 0.05f);   /* green when every leaf crosses */
    EXPECT_NEAR(px[0], 0.0f, 0.05f);
    return px[1] > 0.9f ? 0 : 1;
}

/* --- per-vertex interface blocks ---------------------------------------- */

TEST(TessBlkPlainMembers)
{
    static const char *tcs =
        TCS_HEAD
        "out BlkV { vec4 a; vec4 b; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].a = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].b = vec4(1,0,0,1);\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { vec4 a; vec4 b; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  bool ok = vb[0].b.x == 1.0 && vb[2].a.x > 0.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

TEST(TessBlkStructMember)
{
    static const char *tcs =
        TCS_HEAD
        "out BlkV { S s; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].s.x = 1; vb[gl_InvocationID].s.y = vec4(1,0,0,1);\n"
        "  vb[gl_InvocationID].s.z[0] = 0.5; vb[gl_InvocationID].s.z[1] = 0.25;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { S s; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  bool ok = vb[0].s.x==1 && vb[0].s.z[1]==0.25 && vb[1].s.y.x==1.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

TEST(TessBlkArrayMember)
{
    /* OPT-IN (MGL_TESS_ARRAY_LOWER=1): lowering everything regresses nine
     * dEQP per_patch_block cases and aborts two others (r51 CTS batch), so
     * the pass stays gated until the patch-record path handles it */
    setenv("MGL_TESS_ARRAY_LOWER", "1", 1);
    static const char *tcs =
        TCS_HEAD
        "out BlkV { float fa[3]; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].fa[0]=1.0; vb[gl_InvocationID].fa[1]=2.0; vb[gl_InvocationID].fa[2]=3.0;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { float fa[3]; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  bool ok = vb[0].fa[2]==3.0 && vb[2].fa[0]==1.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

TEST(TessBlkStructArrayMember)
{
    /* S sa[2]: the member dissolves to one variable per (element, field) */
    static const char *tcs =
        TCS_HEAD
        "out BlkV { S sa[2]; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].sa[0].x=2; vb[gl_InvocationID].sa[0].y=vec4(0,1,0,1);\n"
        "  vb[gl_InvocationID].sa[0].z[0]=0.125;\n"
        "  vb[gl_InvocationID].sa[1].x=3; vb[gl_InvocationID].sa[1].y=vec4(0,0,1,1);\n"
        "  vb[gl_InvocationID].sa[1].z[1]=0.75;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { S sa[2]; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  bool ok = vb[2].sa[0].x==2 && vb[2].sa[0].y.y==1.0 &&\n"
        "            vb[0].sa[1].x==3 && vb[0].sa[1].z[1]==0.75;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

TEST(TessBlkMixedMembers)
{
    /* the dEQP user_defined_io per-vertex block shape: struct, array,
     * array-of-struct and scalar members in one block */
    static const char *tcs =
        TCS_HEAD
        "out BlkV { S s; float fa[3]; S sa[2]; float f; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].s.x = 1; vb[gl_InvocationID].s.y = vec4(1,0,0,1);\n"
        "  vb[gl_InvocationID].s.z[0] = 0.5; vb[gl_InvocationID].s.z[1] = 0.25;\n"
        "  vb[gl_InvocationID].fa[0] = 1.0; vb[gl_InvocationID].fa[1] = 2.0; vb[gl_InvocationID].fa[2] = 3.0;\n"
        "  vb[gl_InvocationID].sa[0].x = 2; vb[gl_InvocationID].sa[0].y = vec4(0,1,0,1);\n"
        "  vb[gl_InvocationID].sa[0].z[0] = 0.125; vb[gl_InvocationID].sa[0].z[1] = 0.25;\n"
        "  vb[gl_InvocationID].sa[1].x = 3; vb[gl_InvocationID].sa[1].y = vec4(0,0,1,1);\n"
        "  vb[gl_InvocationID].sa[1].z[0] = 0.5; vb[gl_InvocationID].sa[1].z[1] = 0.75;\n"
        "  vb[gl_InvocationID].f = 4.0;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { S s; float fa[3]; S sa[2]; float f; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  bool ok = vb[0].s.x==1 && vb[0].s.z[1]==0.25 && vb[1].fa[2]==3.0 && vb[2].sa[0].x==2;\n"
        "  ok = ok && vb[0].sa[0].z[0]==0.125 && vb[1].sa[1].z[1]==0.75 && vb[2].f==4.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

/* --- plain (non-block) struct varyings ----------------------------------- */

TEST(TessPlainStructVaryings)
{
    static const char *tcs =
        TCS_HEAD
        "out S vb_s[];\nout float vb_fa[][3];\nout S vb_sa[];\nout float vb_f[];\nout vec4 tc_pos[];\n"
        "void main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb_s[gl_InvocationID].x = 1; vb_s[gl_InvocationID].y = vec4(1,0,0,1);\n"
        "  vb_s[gl_InvocationID].z[0] = 0.5; vb_s[gl_InvocationID].z[1] = 0.25;\n"
        "  vb_fa[gl_InvocationID][0] = 1.0; vb_fa[gl_InvocationID][1] = 2.0; vb_fa[gl_InvocationID][2] = 3.0;\n"
        "  vb_sa[gl_InvocationID].x = 2; vb_sa[gl_InvocationID].y = vec4(0,1,0,1);\n"
        "  vb_sa[gl_InvocationID].z[0] = 0.125; vb_sa[gl_InvocationID].z[1] = 0.25;\n"
        "  vb_f[gl_InvocationID] = 7.0;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in S vb_s[];\nin float vb_fa[][3];\nin S vb_sa[];\nin float vb_f[];\nout vec4 col;\nvoid main() {\n"
        "  bool ok = vb_s[0].x==1 && vb_s[0].z[1]==0.25 && vb_fa[1][2]==3.0;\n"
        "  ok = ok && vb_sa[1].x==2 && vb_sa[1].z[1]==0.25 && vb_f[2]==7.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

/* --- known limitation: whole-struct loads decline with a clean link error */

TEST(TessBlkWholeStructLoadDeclined)
{
    /* S s0 = vb[0].s; cannot be expressed with Metal tessellation attributes:
     * MGL declines the module (clean link error) instead of crashing
     * SPIRV-Cross (which aborts in emit_tessellation_access_chain) */
    static const char *tcs =
        TCS_HEAD
        "out BlkV { S s; float f; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].s.x = 1; vb[gl_InvocationID].f = 7.0;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { S s; float f; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  S s0 = vb[0].s;\n"
        "  bool ok = s0.x == 1;\n" TES_TAIL;
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[] = {
        { GL_VERTEX_SHADER, k_vs },
        { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes },
        { GL_FRAGMENT_SHADER, k_fs },
    };
    for (unsigned i = 0; i < 4; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL);
        glCompileShader(so);
        glAttachShader(p, so);
    }
    glLinkProgram(p);
    GLint lk = 1; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    EXPECT_EQ_INT(lk, 0);   /* declined: link error, no crash */
}

TEST(TessBlkDynamicArrayIndex)
{
    /* a DYNAMIC index into an array member: mgl_tess_lower_dynamic_io copies
     * the per-vertex elements into a function-local array (attribute arrays
     * cannot be dynamically indexed on Metal) */
    static const char *tcs =
        TCS_HEAD
        "out BlkV { float fa[3]; } vb[];\nout vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].fa[0]=1.0; vb[gl_InvocationID].fa[1]=2.0; vb[gl_InvocationID].fa[2]=3.0;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { float fa[3]; } vb[];\nout vec4 col;\nvoid main() {\n"
        "  float acc = 0.0;\n"
        "  for (int i = 0; i < 3; i++) acc += vb[1].fa[i] * float(i + 1);\n"
        "  bool ok = acc == 14.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

/* --- patch block ARRAY + per-vertex block in ONE program (r53) -----------
 * exercises: the per-vertex dissolve skipping member-Patch blocks, the
 * canon-body matrix-merge NOT firing on per-element vec4 leaves, the
 * ||-short-circuit fix in the body flatten, and per-element canon slots */
TEST(TessBlkMixedWithPatchArray)
{
    static const char *tcs =
        TCS_HEAD
        "out BlkV { S s; } vb[];\n"
        "patch out PB { S ps; float w[2]; } pb[2];\n"
        "out vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  vb[gl_InvocationID].s.x = 1; vb[gl_InvocationID].s.y = vec4(1,0,0,1);\n"
        "  vb[gl_InvocationID].s.z[0] = 0.5; vb[gl_InvocationID].s.z[1] = 0.25;\n"
        "  pb[0].ps.x = 7; pb[0].ps.y = vec4(2,3,4,5); pb[0].ps.z[0] = 0.75; pb[0].ps.z[1] = 0.5;\n"
        "  pb[0].w[0] = 1.5; pb[0].w[1] = 2.5;\n"
        "  pb[1].ps.x = 8; pb[1].ps.y = vec4(6,7,8,9); pb[1].ps.z[0] = 0.125; pb[1].ps.z[1] = 0.375;\n"
        "  pb[1].w[0] = 3.5; pb[1].w[1] = 4.5;\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "in BlkV { S s; } vb[];\n"
        "patch in PB { S ps; float w[2]; } pb[2];\n"
        "out vec4 col;\nvoid main() {\n"
        "  bool ok = vb[0].s.x==1 && vb[0].s.z[1]==0.25 && vb[1].s.y.x==1.0;\n"
        "  int e = int(gl_TessCoord.x); if (e > 1) e = 1;\n"
        "  ok = ok && pb[e].ps.x == 7 + e && pb[e].w[1] == 2.5 + float(e)\n"
        "       && pb[e].ps.z[0] == 0.75 - 0.625*float(e) && pb[e].ps.y.z == 4.0 + float(e);\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

TEST(TessBlkPatchArrayLen1)
{
    static const char *tcs =
        TCS_HEAD
        "patch out PB1 { float q; vec4 v; } pb1[1];\n"
        "out BlkA { float fa[3]; } va[];\n"
        "out vec4 tc_pos[];\nvoid main() {\n"
        "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
        "  va[gl_InvocationID].fa[0] = 0.25; va[gl_InvocationID].fa[1] = 0.5; va[gl_InvocationID].fa[2] = 0.75;\n"
        "  pb1[0].q = 1.25; pb1[0].v = vec4(9,8,7,6);\n" TCS_TAIL;
    static const char *tes =
        TES_HEAD
        "patch in PB1 { float q; vec4 v; } pb1[1];\n"
        "in BlkA { float fa[3]; } va[];\n"
        "out vec4 col;\nvoid main() {\n"
        "  int i = int(gl_TessCoord.y * 2.0); if (i > 2) i = 2;\n"
        "  bool ok = va[0].fa[1] == 0.5 && va[2].fa[i] >= 0.25 && pb1[0].q == 1.25 && pb1[0].v.z == 7.0;\n" TES_TAIL;
    EXPECT_EQ_INT(draw_and_expect_green(tcs, tes), 0);
}

int main(void)
{
    RUN(TessBlkPlainMembers);
    RUN(TessBlkStructMember);
    RUN(TessBlkArrayMember);
    RUN(TessBlkStructArrayMember);
    RUN(TessBlkMixedMembers);
    RUN(TessPlainStructVaryings);
    RUN(TessBlkWholeStructLoadDeclined);
    RUN(TessBlkDynamicArrayIndex);
    RUN(TessBlkMixedWithPatchArray);
    RUN(TessBlkPatchArrayLen1);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_summary();
}
