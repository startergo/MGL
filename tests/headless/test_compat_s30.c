/* compat s30 stage 3: EXACT replica of the CTS case
 * ext_texture_shadow_lod.texture.sampler2darrayshadow_vertex — verbatim CTS
 * shaders (vec4 a_position with z=0.5, mediump v_color), single VBO with the
 * two attributes at their queried locations, uint16 indices, 256x256 default
 * framebuffer.  Reference = per-vertex eval + triangle interp. */
#include "mgl_test.h"

#define W 128
#define LAYERS 4
#define CELL 32
#define G 64
#define VP 256

static float depth_at(int layer, int x, int y)
{
    float fA = (float)layer / (float)LAYERS;
    int mx = (x / CELL) % 2, my = (y / CELL) % 2;
    return (mx ^ my) ? (1.0f - fA) : fA;
}

static float ref_lookup(float s, float t, float r, float ref)
{
    int layer = (int)floorf(r + 0.5f);
    if (layer < 0) layer = 0;
    if (layer > LAYERS - 1) layer = LAYERS - 1;
    int i = (int)floorf(s * W);
    int j = (int)floorf(t * W);
    i = ((i % W) + W) % W;
    j = ((j % W) + W) % W;
    float d = depth_at(layer, i, j);
    /* the comparison is against the STORED D16 texel, not the idealised depth:
     * the upload rounds to n = round(d*65535) and a GL DEPTH_COMPONENT16 texel
     * reads back n/65535 (tess r38 stores D16 in D32F exactly this way, so the
     * GPU compares in float against n/65535 — CTS sampler2darrayshadow_vertex
     * hits the ref 0.5 vs texel 0x8000 boundary; d == 0.5 rounds UP to
     * 0x8000 = 0.5000076 > 0.5, so those vertices are LIT).  The pre-r38
     * Depth16Unorm storage compared against n/65536 == d exactly and returned
     * unlit, which this test used to lock in — that was the CTS Fail. */
    unsigned short n = (unsigned short)(int)(d * 65535.0f + 0.5f);   /* same rounding as the upload */
    float dq = (float)((double)n / 65535.0);
    if (ref < 0.0f) ref = 0.0f;
    if (ref > 1.0f) ref = 1.0f;
    return ref < dq ? 1.0f : 0.0f;
}

static const char *vs =
    "#version 460\n"
    "#extension GL_EXT_texture_shadow_lod : require\n\n"
    "in highp vec4 a_position;\n"
    "in highp vec4 a_in0;\n"
    "out mediump vec4 v_color;\n"
    "uniform highp sampler2DArrayShadow u_sampler;\n"
    "uniform highp vec4 u_scale;\n"
    "uniform highp vec4 u_bias;\n\n"
    "void main()\n"
    "{\n"
    "\tgl_Position = a_position;\n"
    "\tv_color = vec4(texture(u_sampler, a_in0), 0.0, 0.0, 1.0);\n"
    "}\n";
static const char *fs =
    "#version 460\n"
    "#extension GL_EXT_texture_shadow_lod : require\n\n"
    "out mediump vec4 o_color;\n"
    "in mediump vec4 v_color;\n\n"
    "void main()\n"
    "{\n"
    "\to_color = v_color;\n"
    "}\n";

TEST(shadowlod_vs_grid)
{
    GLuint tex, vbo, ibo, prog, vao;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    {   /* CTS glu::Texture2DArray: one glTexImage3D alloc+upload, UNSIGNED_SHORT, all layers */
        static unsigned short all[LAYERS * W * W];
        for (int layer = 0; layer < LAYERS; layer++)
            for (int y = 0; y < W; y++)
                for (int x = 0; x < W; x++)
                    all[(layer * W + y) * W + x] = (unsigned short)(int)(depth_at(layer, x, y) * 65535.0f + 0.5f);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT16, W, W, LAYERS, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, all);
    }
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
    EXPECT_NO_GL_ERROR();

    prog = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    EXPECT_TRUE(prog != 0);
    if (!prog) return;
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "u_sampler"), 0);
    int locP = glGetAttribLocation(prog, "a_position");
    int locI = glGetAttribLocation(prog, "a_in0");
    fprintf(stderr, "  [%s] locs a_position=%d a_in0=%d a_coords=%d a_unitCoords=%d a_one=%d\n", mgl_test_current,
            locP, locI, glGetAttribLocation(prog, "a_coords"), glGetAttribLocation(prog, "a_unitCoords"), glGetAttribLocation(prog, "a_one"));

    /* CTS baseCoordTrans */
    float s0[4] = {2.7f, 2.7f, 4.0f, 1.0f}, b0[4] = {-1.2f, -0.4f, -0.5f, 0.0f};
    float pos[(G + 1) * (G + 1)][4];
    float in0[(G + 1) * (G + 1)][4];
    GLushort idx[G * G * 6];
    for (int y = 0; y <= G; y++)
        for (int x = 0; x <= G; x++)
        {
            float sx = (float)x / G, sy = (float)y / G;
            int k = y * (G + 1) + x;
            pos[k][0] = 2.0f * sx - 1.0f;
            pos[k][1] = 2.0f * sy - 1.0f;
            pos[k][2] = 0.5f;
            pos[k][3] = 1.0f;
            in0[k][0] = b0[0] + s0[0] * sx;
            in0[k][1] = b0[1] + s0[1] * sy;
            in0[k][2] = b0[2] + s0[2] / 2.0f + s0[2] / 2.0f * sx - s0[2] / 2.0f * sy;
            in0[k][3] = b0[3] + s0[3] / 2.0f - s0[3] / 2.0f * sx + s0[3] / 2.0f * sy;
        }
    for (int y = 0; y < G; y++)
        for (int x = 0; x < G; x++)
        {
            int stride = G + 1;
            int v00 = y * stride + x, v01 = y * stride + x + 1;
            int v10 = (y + 1) * stride + x, v11 = (y + 1) * stride + x + 1;
            int b = (y * G + x) * 6;
            idx[b + 0] = v10; idx[b + 1] = v00; idx[b + 2] = v01;
            idx[b + 3] = v10; idx[b + 4] = v01; idx[b + 5] = v11;
        }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pos) + sizeof(in0), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(pos), pos);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(pos), sizeof(in0), in0);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(locP);
    glVertexAttribPointer(locP, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
    glEnableVertexAttribArray(locI);
    glVertexAttribPointer(locI, 4, GL_FLOAT, GL_FALSE, 16, (void *)(uintptr_t)sizeof(pos));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glDisable(GL_DITHER);
    glViewport(0, 0, VP, VP);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, G * G * 6, GL_UNSIGNED_SHORT, (void *)0);
    EXPECT_NO_GL_ERROR();
    GLubyte px[VP * VP][4];
    glFinish();
    glReadPixels(0, 0, VP, VP, GL_RGBA, GL_UNSIGNED_BYTE, px);

    { FILE *f = fopen("/tmp/mgl_local.raw", "wb"); fwrite(px, 4, VP * VP, f); fclose(f); }
    float vcol[(G + 1) * (G + 1)];
    for (int y = 0; y <= G; y++)
        for (int x = 0; x <= G; x++)
        {
            int k = y * (G + 1) + x;
            vcol[k] = ref_lookup(in0[k][0], in0[k][1], in0[k][2], in0[k][3]);
        }
    int bad = 0, shown = 0;
    int bandbox[4] = {VP, VP, 0, 0};
    for (int iy = 0; iy < VP; iy++)
        for (int ix = 0; ix < VP; ix++)
        {
            float sfx = ix + 0.5f, sfy = iy + 0.5f;
            int cx = (int)(sfx * G / VP), cy = (int)(sfy * G / VP);
            if (cx >= G) cx = G - 1;
            if (cy >= G) cy = G - 1;
            float fx1 = sfx * G / VP - cx, fy1 = sfy * G / VP - cy;
            int stride = G + 1;
            int v00 = cy * stride + cx, v01 = cy * stride + cx + 1;
            int v10 = (cy + 1) * stride + cx, v11 = (cy + 1) * stride + cx + 1;
            int tri = (fx1 + fy1 <= 1.0f);
            float tx = tri ? fx1 : (1.0f - fx1);
            float ty = tri ? fy1 : (1.0f - fy1);
            float c0 = tri ? vcol[v00] : vcol[v11];
            float c1 = tri ? vcol[v01] : vcol[v10];
            float c2 = tri ? vcol[v10] : vcol[v01];
            float want = c0 + (c1 - c0) * tx + (c2 - c0) * ty;
            float got = px[iy * VP + ix][0] / 255.0f;
            if (fabsf(got - want) > 0.5f)
            {
                if (shown < 14)
                    fprintf(stderr, "  [%s] px(%d,%d) grid(%d,%d) f=(%.2f,%.2f) got=%.3f want=%.3f [v00=%.0f v01=%.0f v10=%.0f v11=%.0f]\n",
                            mgl_test_current, ix, iy, cx, cy, fx1, fy1, got, want,
                            vcol[v00], vcol[v01], vcol[v10], vcol[v11]);
                shown++;
                bad++;
                if (ix < bandbox[0]) bandbox[0] = ix;
                if (iy < bandbox[1]) bandbox[1] = iy;
                if (ix > bandbox[2]) bandbox[2] = ix;
                if (iy > bandbox[3]) bandbox[3] = iy;
            }
        }
    fprintf(stderr, "  [%s] %d/%d pixel mismatches, bbox=(%d,%d)-(%d,%d)\n",
            mgl_test_current, bad, VP * VP, bandbox[0], bandbox[1], bandbox[2], bandbox[3]);
    EXPECT_EQ_INT(bad, 0);
    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);
    glDeleteVertexArrays(1, &vao);
}

int main(void)
{
    RUN(shadowlod_vs_grid);
    return mgl_test_summary();
}
