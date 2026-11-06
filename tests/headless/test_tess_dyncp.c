/* test_tess_dyncp.c — per-vertex interface blocks read with DYNAMIC control-point
 * and DYNAMIC array-member indices in the TES (the dEQP
 * user_defined_io.per_vertex_block.vertex_io_array_size_* shapes).
 *
 *  - block with a struct member          (S blockS = { int x; vec4 y; float z[2]; })
 *  - block with an array member          (float blockFa[3])
 *  - block with an array-of-struct       (S blockSa[2])
 *  - whole loop over gl_PatchVerticesIn (dynamic loop bound) with dynamic
 *    member indices inside (i1/i2)
 *  - the same shapes as a per-patch block ARRAY (patch in SomeBlock pb[2])
 *
 * Metal stage-in attributes can be neither structs nor arrays, so MGL must
 * lower these accesses to constant-indexed leaf copies in function-local
 * storage (or read the control points from the record buffer).
 */
#include "mgl_test.h"
#include <string.h>

static const char *k_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=1) in float attr;\n"
    "layout(location=0) out vec4 vs_pos;\n"
    "layout(location=1) out float vs_attr;\n"
    "void main(){ gl_Position = vec4(pos,0,1); vs_pos = vec4(pos,0,1); vs_attr = attr; }\n";

static const char *k_fs =
    "#version 450 core\n"
    "in vec4 col; out vec4 o;\n"
    "void main(){ o = col; }\n";

#define TCS_HEAD "#version 450 core\nlayout(vertices=5) out;\n" \
                 "in vec4 vs_pos[]; in float vs_attr[]; out vec4 tc_pos[];\n" \
                 "struct S { int x; vec4 y; float z[2]; };\n"
#define TCS_TAIL "  if (gl_InvocationID == 0) {\n" \
                 "    gl_TessLevelOuter[0]=3.0; gl_TessLevelOuter[1]=3.0; gl_TessLevelOuter[2]=3.0;\n" \
                 "    gl_TessLevelInner[0]=3.0; gl_TessLevelInner[1]=3.0;\n  }\n}\n"
#define TES_HEAD "#version 450 core\nlayout(triangles, equal_spacing, ccw) in;\n" \
                 "in vec4 tc_pos[];\nstruct S { int x; vec4 y; float z[2]; };\n"
#define TES_TAIL "  vec3 p = gl_TessCoord.x*tc_pos[0].xyz + gl_TessCoord.y*tc_pos[1].xyz" \
                 " + gl_TessCoord.z*tc_pos[2].xyz;\n" \
                 "  gl_Position = vec4(p.xy, 0, 1);\n" \
                 "  col = ok ? vec4(0,1,0,1) : vec4(1,0,0,1);\n}\n"

/* --- per-vertex block: TCS writes with gl_InvocationID, TES reads with
 *     dynamic i0 (loop over gl_PatchVerticesIn) and dynamic i1/i2 --------- */
static const char *k_tcs_dyn =
    TCS_HEAD
    "out TheBlock { S blockS; float blockFa[3]; S blockSa[2]; float blockF; } tcBlock[];\n"
    "void main(){\n"
    "  tc_pos[gl_InvocationID] = vs_pos[gl_InvocationID];\n"
    "  float v = 1.3 + float(gl_InvocationID)*6.4;\n"
    "  tcBlock[gl_InvocationID].blockS.x = int(v); v += 0.4;\n"
    "  tcBlock[gl_InvocationID].blockS.y = vec4(v, v+0.8, v+1.6, v+2.4); v += 0.4;\n"
    "  for (int i0 = 0; i0 < 2; i0++) { tcBlock[gl_InvocationID].blockS.z[i0] = v; v += 0.4; }\n"
    "  for (int i0 = 0; i0 < 3; i0++) { tcBlock[gl_InvocationID].blockFa[i0] = v; v += 0.4; }\n"
    "  for (int i0 = 0; i0 < 2; i0++) {\n"
    "    tcBlock[gl_InvocationID].blockSa[i0].x = int(v); v += 0.4;\n"
    "    tcBlock[gl_InvocationID].blockSa[i0].y = vec4(v, v+0.8, v+1.6, v+2.4); v += 0.4;\n"
    "    for (int i1 = 0; i1 < 2; i1++) { tcBlock[gl_InvocationID].blockSa[i0].z[i1] = v; v += 0.4; }\n"
    "  }\n"
    "  tcBlock[gl_InvocationID].blockF = v; v += 0.4;\n"
    TCS_TAIL;

static const char *k_tes_dyn =
    TES_HEAD
    "in TheBlock { S blockS; float blockFa[3]; S blockSa[2]; float blockF; } teBlock[];\n"
    "out vec4 col;\n"
    "void main(){\n"
    "  bool ok = true;\n"
    "  for (int i0 = 0; i0 < gl_PatchVerticesIn; i0++) {\n"
    "    float w = 1.3 + float(i0)*6.4;\n"
    "    ok = ok && teBlock[i0].blockS.x == int(w); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockS.y.x - w) < 0.01) && (abs(teBlock[i0].blockS.y.y - (w+0.8)) < 0.01) && (abs(teBlock[i0].blockS.y.z - (w+1.6)) < 0.01) && (abs(teBlock[i0].blockS.y.w - (w+2.4)) < 0.01); w += 0.4;\n"
    "    for (int i1 = 0; i1 < 2; i1++) { ok = ok && teBlock[i0].blockS.z[i1] == w; w += 0.4; }\n"
    "    for (int i1 = 0; i1 < 3; i1++) { ok = ok && teBlock[i0].blockFa[i1] == w; w += 0.4; }\n"
    "    for (int i1 = 0; i1 < 2; i1++) {\n"
    "      ok = ok && teBlock[i0].blockSa[i1].x == int(w); w += 0.4;\n"
    "      ok = ok && (abs(teBlock[i0].blockSa[i1].y.x - w) < 0.01) && (abs(teBlock[i0].blockSa[i1].y.y - (w+0.8)) < 0.01) && (abs(teBlock[i0].blockSa[i1].y.z - (w+1.6)) < 0.01) && (abs(teBlock[i0].blockSa[i1].y.w - (w+2.4)) < 0.01); w += 0.4;\n"
    "      for (int i2 = 0; i2 < 2; i2++) { ok = ok && teBlock[i0].blockSa[i1].z[i2] == w; w += 0.4; }\n"
    "    }\n"
    "    ok = ok && teBlock[i0].blockF == w; w += 0.4;\n"
    "  }\n"
    TES_TAIL;

/* --- same block, but every member index is CONSTANT in the TES (only the
 *     control point index is dynamic): must render green through the
 *     attribute path without any lowering ---------------------------------- */
static const char *k_tes_const_members =
    TES_HEAD
    "in TheBlock { S blockS; float blockFa[3]; S blockSa[2]; float blockF; } teBlock[];\n"
    "out vec4 col;\n"
    "void main(){\n"
    "  bool ok = true;\n"
    "  for (int i0 = 0; i0 < gl_PatchVerticesIn; i0++) {\n"
    "    float w = 1.3 + float(i0)*6.4;\n"
    "    ok = ok && teBlock[i0].blockS.x == int(w); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockS.y.x - w) < 0.1) && (abs(teBlock[i0].blockS.y.y - (w+0.8)) < 0.1) && (abs(teBlock[i0].blockS.y.z - (w+1.6)) < 0.1) && (abs(teBlock[i0].blockS.y.w - (w+2.4)) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockS.z[0] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockS.z[1] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockFa[0] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockFa[1] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockFa[2] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && teBlock[i0].blockSa[0].x == int(w); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockSa[0].y.x - w) < 0.1) && (abs(teBlock[i0].blockSa[0].y.y - (w+0.8)) < 0.1) && (abs(teBlock[i0].blockSa[0].y.z - (w+1.6)) < 0.1) && (abs(teBlock[i0].blockSa[0].y.w - (w+2.4)) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockSa[0].z[0] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockSa[0].z[1] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && teBlock[i0].blockSa[1].x == int(w); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockSa[1].y.x - w) < 0.1) && (abs(teBlock[i0].blockSa[1].y.y - (w+0.8)) < 0.1) && (abs(teBlock[i0].blockSa[1].y.z - (w+1.6)) < 0.1) && (abs(teBlock[i0].blockSa[1].y.w - (w+2.4)) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockSa[1].z[0] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockSa[1].z[1] - w) < 0.1); w += 0.4;\n"
    "    ok = ok && (abs(teBlock[i0].blockF - w) < 0.1); w += 0.4;\n"
    "  }\n"
    TES_TAIL;

/* --- per-patch block ARRAY (patch in BlkP pb[2]) with dynamic index ------ */
static const char *k_tcs_pparr =
    TCS_HEAD
    "patch out BlkP { S s; float fa[3]; } ppBlock[2];\n"
    "void main(){\n"
    "  float v = 0.5 + float(gl_InvocationID);\n"
    "  if (gl_InvocationID < 2) {\n"
    "    ppBlock[gl_InvocationID].s.x = int(v); v += 0.25;\n"
    "    ppBlock[gl_InvocationID].s.y = vec4(v, v+0.5, v+1.0, v+1.5); v += 0.25;\n"
    "    for (int i = 0; i < 2; i++) { ppBlock[gl_InvocationID].s.z[i] = v; v += 0.25; }\n"
    "    for (int i = 0; i < 3; i++) { ppBlock[gl_InvocationID].fa[i] = v; v += 0.25; }\n"
    "  }\n"
    TCS_TAIL;

static const char *k_tes_pparr =
    TES_HEAD
    "patch in BlkP { S s; float fa[3]; } ppBlock[2];\n"
    "out vec4 col;\n"
    "void main(){\n"
    "  bool ok = true;\n"
    "  for (int i0 = 0; i0 < 2; i0++) {\n"
    "    float w = 0.5 + float(i0);\n"
    "    ok = ok && ppBlock[i0].s.x == int(w); w += 0.25;\n"
    "    ok = ok && ppBlock[i0].s.y == vec4(w, w+0.5, w+1.0, w+1.5); w += 0.25;\n"
    "    for (int i1 = 0; i1 < 2; i1++) { ok = ok && ppBlock[i0].s.z[i1] == w; w += 0.25; }\n"
    "    for (int i1 = 0; i1 < 3; i1++) { ok = ok && ppBlock[i0].fa[i1] == w; w += 0.25; }\n"
    "  }\n"
    TES_TAIL;

/* --- single per-patch block (SPIRV-Cross native path; must keep passing
 * ---- after the dissolve rewires blocks with explicit locations) --------- */
static const char *k_tcs_psingle =
    TCS_HEAD
    "layout(location=4) patch out BlkS { S s; float fa[3]; } pb;\n"
    "void main(){\n"
    "  float v = 0.5;\n"
    "  pb.s.x = int(v); v += 0.25;\n"
    "  pb.s.y = vec4(v, v+0.5, v+1.0, v+1.5); v += 0.25;\n"
    "  for (int i = 0; i < 2; i++) { pb.s.z[i] = v; v += 0.25; }\n"
    "  for (int i = 0; i < 3; i++) { pb.fa[i] = v; v += 0.25; }\n"
    TCS_TAIL;

static const char *k_tes_psingle =
    TES_HEAD
    "layout(location=4) patch in BlkS { S s; float fa[3]; } pb;\n"
    "out vec4 col;\n"
    "void main(){\n"
    "  bool ok = true;\n"
    "  float w = 0.5;\n"
    "  ok = ok && pb.s.x == int(w); w += 0.25;\n"
    "  ok = ok && pb.s.y == vec4(w, w+0.5, w+1.0, w+1.5); w += 0.25;\n"
    "  for (int i = 0; i < 2; i++) { ok = ok && pb.s.z[i] == w; w += 0.25; }\n"
    "  for (int i = 0; i < 3; i++) { ok = ok && pb.fa[i] == w; w += 0.25; }\n"
    TES_TAIL;

static int g_link_should_fail;
static void run_green_case(const char *tcs, const char *tes)
{
    const char *name = mgl_test_current;
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
        if (!cok) { char log[2048]; glGetShaderInfoLog(so, sizeof log, NULL, log);
            fprintf(stderr, "  [%s] compile(%d): %s\n", name, i, log); EXPECT_TRUE(cok); return; }
        glAttachShader(p, so);
    }
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (!lk) { char log[4096]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "  [%s] link: %s\n", name, log); EXPECT_TRUE(lk); return; }

    static const float pos[5][2] = { { -1, 1 }, { -1, -1 }, { 1, -1 }, { 0.5, 0.9 }, { -0.4, 0.6 } };
    static const float attr[5] = { 3, 3, 3, 3, 3 };
    GLuint vao, buf;
    glViewport(0, 0, 512, 512);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    {
        float data[5][3];
        for (int i = 0; i < 5; i++) { data[i][0] = pos[i][0]; data[i][1] = pos[i][1]; data[i][2] = attr[i]; }
        glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 12, NULL);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 12, (void *)8);
    glPatchParameteri(GL_PATCH_VERTICES, 5);
    glUseProgram(p);
    glDrawArrays(GL_PATCHES, 0, 5);
    glFinish();
    glDeleteBuffers(1, &buf);
    glDeleteVertexArrays(1, &vao);

    float px[4] = { -1, -1, -1, -1 };
    int any = 0;
    for (int yy = 200; yy < 300 && !any; yy += 3)
        for (int xx = 200; xx < 300 && !any; xx += 3) {
            glReadPixels(xx, yy, 1, 1, GL_RGBA, GL_FLOAT, px);
            if (px[0] > 0.5f || px[1] > 0.5f) any = 1;
        }
    (void)any;
    EXPECT_TRUE(px[1] > 0.9f && px[0] < 0.1f);
}

static int run_cts_shaders(int use_xfb);

/* ------------------------------------------------------------------
 * Sources of dEQP-GL45-ES31.functional.tessellation.user_defined_io.
 * per_vertex_block.vertex_io_array_size_implicit.triangles, embedded
 * verbatim (extracted from cts-tess61/tess.qpa). Previously this test
 * read them from /tmp/pvb_*.glsl, a scratch dir wiped on reboot.
 * ------------------------------------------------------------------ */

static const char cts_pvb_vs[] =
    "#version 450\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "\n"
    "\n"
    "in highp float in_v_attr;\n"
    "out highp float in_tc_attr;\n"
    "\n"
    "void main (void)\n"
    "{\n"
    "    in_tc_attr = in_v_attr;\n"
    "}\n"
    ;

static const char cts_pvb_fs[] =
    "#version 450\n"
    "\n"
    "layout (location = 0) out mediump vec4 o_color;\n"
    "\n"
    "in highp vec4 in_f_color;\n"
    "\n"
    "void main (void)\n"
    "{\n"
    "    o_color = in_f_color;\n"
    "}\n"
    ;

static const char cts_pvb_tcs[] =
    "#version 450\n"
    "\n"
    "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
    "\n"
    "out gl_PerVertex { vec4 gl_Position; } gl_out[];\n"
    "\n"
    "\n"
    "layout (vertices = 5) out;\n"
    "\n"
    "in highp float in_tc_attr[];\n"
    "struct S\n"
    "{\n"
    "	highp int x;\n"
    "	highp vec4 y;\n"
    "	highp float z[2];\n"
    "};\n"
    "out TheBlock\n"
    "{\n"
    "	S blockS;\n"
    "	highp float blockFa[3];\n"
    "	S blockSa[2];\n"
    "	highp float blockF;\n"
    "} tcBlock[];\n"
    "\n"
    "patch out highp vec2 in_te_positionScale;\n"
    "patch out highp vec2 in_te_positionOffset;\n"
    "\n"
    "void main (void)\n"
    "{\n"
    "	{\n"
    "		highp float v = 1.3;\n"
    "		v += float(gl_InvocationID)*6.4;\n"
    "\n"
    "		// Assign values to output tcBlock\n"
    "		tcBlock[gl_InvocationID].blockS.x = int(v);\n"
    "		v += 0.4;\n"
    "		tcBlock[gl_InvocationID].blockS.y = vec4(v, v+0.8, v+1.6, v+2.4);\n"
    "		v += 0.4;\n"
    "		for (int i0 = 0; i0 < 2; i0++)\n"
    "		{\n"
    "			tcBlock[gl_InvocationID].blockS.z[i0] = v;\n"
    "			v += 0.4;\n"
    "		}\n"
    "		for (int i0 = 0; i0 < 3; i0++)\n"
    "		{\n"
    "			tcBlock[gl_InvocationID].blockFa[i0] = v;\n"
    "			v += 0.4;\n"
    "		}\n"
    "		for (int i0 = 0; i0 < 2; i0++)\n"
    "		{\n"
    "			tcBlock[gl_InvocationID].blockSa[i0].x = int(v);\n"
    "			v += 0.4;\n"
    "			tcBlock[gl_InvocationID].blockSa[i0].y = vec4(v, v+0.8, v+1.6, v+2.4);\n"
    "			v += 0.4;\n"
    "			for (int i1 = 0; i1 < 2; i1++)\n"
    "			{\n"
    "				tcBlock[gl_InvocationID].blockSa[i0].z[i1] = v;\n"
    "				v += 0.4;\n"
    "			}\n"
    "		}\n"
    "		tcBlock[gl_InvocationID].blockF = v;\n"
    "		v += 0.4;\n"
    "		v += float(5-gl_InvocationID-1)*6.4;\n"
    "	}\n"
    "\n"
    "    in_te_positionScale  = vec2(in_tc_attr[6], in_tc_attr[7]);\n"
    "    in_te_positionOffset = vec2(in_tc_attr[8], in_tc_attr[9]);\n"
    "\n"
    "    gl_TessLevelInner[0] = in_tc_attr[0];\n"
    "    gl_TessLevelInner[1] = in_tc_attr[1];\n"
    "\n"
    "    gl_TessLevelOuter[0] = in_tc_attr[2];\n"
    "    gl_TessLevelOuter[1] = in_tc_attr[3];\n"
    "    gl_TessLevelOuter[2] = in_tc_attr[4];\n"
    "    gl_TessLevelOuter[3] = in_tc_attr[5];\n"
    "}\n"
    ;

static const char cts_pvb_tes[] =
    "#version 450\n"
    "\n"
    "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
    "\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "\n"
    "\n"
    "layout (triangles) in;\n"
    "\n"
    "struct S\n"
    "{\n"
    "	highp int x;\n"
    "	highp vec4 y;\n"
    "	highp float z[2];\n"
    "};\n"
    "in TheBlock\n"
    "{\n"
    "	S blockS;\n"
    "	highp float blockFa[3];\n"
    "	S blockSa[2];\n"
    "	highp float blockF;\n"
    "} teBlock[];\n"
    "\n"
    "patch in highp vec2 in_te_positionScale;\n"
    "patch in highp vec2 in_te_positionOffset;\n"
    "\n"
    "out highp vec4 in_f_color;\n"
    "// Will contain the index of the first incorrect input,\n"
    "// or the number of inputs if all are correct\n"
    "flat out highp uint out_te_firstFailedInputIndex;\n"
    "\n"
    "bool compare_int   (int   a, int   b) { return a == b; }\n"
    "bool compare_float (float a, float b) { return abs(a - b) < 0.01f; }\n"
    "bool compare_vec4  (vec4  a, vec4  b) { return all(lessThan(abs(a - b), vec4(0.01f))); }\n"
    "\n"
    "void main (void)\n"
    "{\n"
    "	bool allOk = true;\n"
    "	highp uint firstFailedInputIndex = 0u;\n"
    "	{\n"
    "		highp float v = 1.3;\n"
    "\n"
    "		// Check values in input teBlock\n"
    "		for (int i0 = 0; i0 < 5; i0++)\n"
    "		{\n"
    "			allOk = allOk && compare_int(teBlock[i0].blockS.x, int(v));\n"
    "			v += 0.4;\n"
    "			if (allOk) firstFailedInputIndex++;\n"
    "			allOk = allOk && compare_vec4(teBlock[i0].blockS.y, vec4(v, v+0.8, v+1.6, v+2.4));\n"
    "			v += 0.4;\n"
    "			if (allOk) firstFailedInputIndex++;\n"
    "			for (int i1 = 0; i1 < 2; i1++)\n"
    "			{\n"
    "				allOk = allOk && compare_float(teBlock[i0].blockS.z[i1], v);\n"
    "				v += 0.4;\n"
    "				if (allOk) firstFailedInputIndex++;\n"
    "			}\n"
    "			for (int i1 = 0; i1 < 3; i1++)\n"
    "			{\n"
    "				allOk = allOk && compare_float(teBlock[i0].blockFa[i1], v);\n"
    "				v += 0.4;\n"
    "				if (allOk) firstFailedInputIndex++;\n"
    "			}\n"
    "			for (int i1 = 0; i1 < 2; i1++)\n"
    "			{\n"
    "				allOk = allOk && compare_int(teBlock[i0].blockSa[i1].x, int(v));\n"
    "				v += 0.4;\n"
    "				if (allOk) firstFailedInputIndex++;\n"
    "				allOk = allOk && compare_vec4(teBlock[i0].blockSa[i1].y, vec4(v, v+0.8, v+1.6, v+2.4));\n"
    "				v += 0.4;\n"
    "				if (allOk) firstFailedInputIndex++;\n"
    "				for (int i2 = 0; i2 < 2; i2++)\n"
    "				{\n"
    "					allOk = allOk && compare_float(teBlock[i0].blockSa[i1].z[i2], v);\n"
    "					v += 0.4;\n"
    "					if (allOk) firstFailedInputIndex++;\n"
    "				}\n"
    "			}\n"
    "			allOk = allOk && compare_float(teBlock[i0].blockF, v);\n"
    "			v += 0.4;\n"
    "			if (allOk) firstFailedInputIndex++;\n"
    "		}\n"
    "	}\n"
    "\n"
    "    gl_Position = vec4(gl_TessCoord.xy*in_te_positionScale + in_te_positionOffset, 0.0, 1.0);\n"
    "    in_f_color = allOk ? vec4(0.0, 1.0, 0.0, 1.0)\n"
    "                       : vec4(1.0, 0.0, 0.0, 1.0);\n"
    "    out_te_firstFailedInputIndex = firstFailedInputIndex;\n"
    "}\n"
    ;

TEST(TessDynCpBlockDynamicMembers) { run_green_case(k_tcs_dyn, k_tes_dyn); }
TEST(TessDynCpBlockConstMembers) { run_green_case(k_tcs_dyn, k_tes_const_members); }
TEST(TessDynPatchBlockArray)  { run_green_case(k_tcs_pparr, k_tes_pparr); }
TEST(TessDynPatchBlockSingle) { run_green_case(k_tcs_psingle, k_tes_psingle); }
TEST(TessCtsVerbatim)      { run_cts_shaders(0); }
TEST(TessCtsVerbatimXfb)   { run_cts_shaders(1); }

static int run_cts_shaders(int use_xfb)
{
    const char *name = use_xfb ? "TessCtsVerbatimXfb" : "TessCtsVerbatim";
    const char *vs = cts_pvb_vs, *fs = cts_pvb_fs, *tcs = cts_pvb_tcs, *tes = cts_pvb_tes;

    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[] = {
        { GL_VERTEX_SHADER, vs },
        { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes },
        { GL_FRAGMENT_SHADER, fs },
    };
    for (unsigned i = 0; i < 4; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL);
        glCompileShader(so);
        GLint cok = 0; glGetShaderiv(so, GL_COMPILE_STATUS, &cok);
        if (!cok) { char log[2048]; glGetShaderInfoLog(so, sizeof log, NULL, log);
            fprintf(stderr, "  [%s] compile(%d): %s\n", name, i, log); return 0; }
        glAttachShader(p, so);
    }
    if (use_xfb) {
        const char *xb[] = { "out_te_firstFailedInputIndex" };
        glTransformFeedbackVaryings(p, 1, xb, GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (!lk) { char log[4096]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "  [%s] link: %s\n", name, log); EXPECT_TRUE(0); return 0; }
    EXPECT_TRUE(lk);
    return 0;
}

int main(void)
{
    RUN(TessDynCpBlockDynamicMembers);
    RUN(TessDynCpBlockConstMembers);
    RUN(TessDynPatchBlockArray);
    RUN(TessDynPatchBlockSingle);
    RUN(TessCtsVerbatim);
    RUN(TessCtsVerbatimXfb);
    return 0;
}
