/* r180 probe (rsvga: D3D9/DX10 BC4/BC5 = ATI1/ATI2, BC6H, BC7): does MGL take
 * the GL RGTC / BPTC formats end to end on this GPU? Per format:
 *   a. glTextureStorage2D + glCompressedTextureSubImage2D (DSA, rsvga's calls),
 *      and the classic glCompressedTexImage2D, without a GL error
 *   b. glGetCompressedTextureImage hands back the uploaded bytes
 *   c. texelFetch in a shader decodes to the block's known value, read back
 *      from an RGBA32F target
 * Blocks are hand-built so their decode is exact: BC4/BC5 with every index 0
 * (texel = endpoint 0), BC7 mode 6 with every index 0 (texel = endpoint 0 with
 * its p-bit), BC6H the all-zero block (mode 0, endpoints 0: texel 0) plus a
 * byte round trip. Each texture is 8x4: two blocks, left and right differ. */
#include "mgl_test.h"
#include <stdint.h>

static GLuint vao, prog_f, fbo, rt;

static GLuint sh(GLenum ty, const char *src)
{
    GLuint s = glCreateShader(ty); glShaderSource(s, 1, &src, NULL); glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char l[1024]; glGetShaderInfoLog(s, sizeof l, NULL, l); printf("  [compile] %s\n", l); }
    return s;
}
static GLuint link_fs(const char *fs)
{
    static GLuint vs;
    if (!vs) vs = sh(GL_VERTEX_SHADER, "#version 450\nvoid main(){ vec2 p = vec2((gl_VertexID&1)*4-1, (gl_VertexID&2)*2-1); gl_Position = vec4(p,0,1); }\n");
    GLuint p = glCreateProgram(); glAttachShader(p, vs); glAttachShader(p, sh(GL_FRAGMENT_SHADER, fs)); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char l[1024]; glGetProgramInfoLog(p, sizeof l, NULL, l); printf("  [link] %s\n", l); }
    return p;
}

/* LSB-first bit writer for BC7 */
static void putbits(uint8_t *b, int *pos, uint32_t v, int n)
{
    for (int i = 0; i < n; i++, (*pos)++)
        if (v >> i & 1) b[*pos >> 3] |= (uint8_t)(1u << (*pos & 7));
}
/* BC7 mode 6, every index 0: texel = (E0 << 1 | P0) per channel */
static void bc7_mode6(uint8_t *b, int r7, int g7, int b7, int a7, int p0)
{
    memset(b, 0, 16); int pos = 0;
    putbits(b, &pos, 1u << 6, 7);                        /* mode 6 */
    int e[8] = { r7, r7, g7, g7, b7, b7, a7, a7 };
    for (int i = 0; i < 8; i++) putbits(b, &pos, (uint32_t)e[i], 7);
    putbits(b, &pos, (uint32_t)p0, 1); putbits(b, &pos, (uint32_t)p0, 1);
    /* indices stay 0 (63 bits) */
}
/* BC4 block with endpoints r0, r1 and every index 0 */
static void bc4(uint8_t *b, int r0, int r1) { memset(b, 0, 8); b[0] = (uint8_t)r0; b[1] = (uint8_t)r1; }

static int near(float a, float b, float tol) { return fabsf(a - b) <= tol; }

static int check_format(const char *name, GLenum fmt, const uint8_t *data, int bytes, const float want[2][4])
{
    int bad = 0;
    while (glGetError()) {}
    /* a. DSA upload */
    GLuint t; glCreateTextures(GL_TEXTURE_2D, 1, &t);
    glTextureStorage2D(t, 1, fmt, 8, 4);
    glCompressedTextureSubImage2D(t, 0, 0, 0, 8, 4, fmt, bytes, data);
    GLenum e = glGetError();
    if (e) { printf("  %-28s DSA upload: GL error %#x  FAIL\n", name, e); return 1; }
    /* a'. classic upload */
    GLuint t2; glGenTextures(1, &t2); glBindTexture(GL_TEXTURE_2D, t2);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, fmt, 8, 4, 0, bytes, data);
    e = glGetError();
    if (e) { printf("  %-28s glCompressedTexImage2D: GL error %#x  FAIL\n", name, e); bad++; }
    /* b. compressed readback */
    uint8_t back[64]; memset(back, 0xAA, sizeof back);
    glGetCompressedTextureImage(t, 0, sizeof back, back);
    e = glGetError();
    int same = !e && memcmp(back, data, (size_t)bytes) == 0;
    if (!same) { printf("  %-28s compressed readback differs (GL error %#x)  FAIL\n", name, e); bad++; }
    /* c. decode on the GPU */
    glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, 8, 4);
    glClearColor(-9, -9, -9, -9); glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog_f); glBindTextureUnit(0, t);
    glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, 3);
    float px[4 * 8 * 4]; glFinish();
    glReadPixels(0, 0, 8, 4, GL_RGBA, GL_FLOAT, px);
    for (int half = 0; half < 2; half++) {
        const float *p = &px[4 * (1 * 8 + (half ? 6 : 1))];
        int ok = near(p[0], want[half][0], 0.01f) && near(p[1], want[half][1], 0.01f) &&
                 near(p[2], want[half][2], 0.01f) && near(p[3], want[half][3], 0.01f);
        printf("  %-28s block %d decodes %7.4f %7.4f %7.4f %7.4f  want %7.4f %7.4f %7.4f %7.4f  %s\n",
               name, half, p[0], p[1], p[2], p[3], want[half][0], want[half][1], want[half][2], want[half][3],
               ok ? "ok" : "FAIL");
        bad += !ok;
    }
    if (!bad) printf("  %-28s upload (DSA + classic), byte readback: ok\n", name);
    glDeleteTextures(1, &t); glDeleteTextures(1, &t2);
    return bad;
}

int main(void)
{
    int bad = 0;
    glGenVertexArrays(1, &vao);
    prog_f = link_fs("#version 450\nlayout(binding=0) uniform sampler2D t;\nlayout(location=0) out vec4 o;\n"
                     "void main(){ o = texelFetch(t, ivec2(gl_FragCoord.xy), 0); }\n");
    glCreateTextures(GL_TEXTURE_2D, 1, &rt); glTextureStorage2D(rt, 1, GL_RGBA32F, 8, 4);
    glCreateFramebuffers(1, &fbo); glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, rt, 0);

    /* what MGL advertises */
    GLint n = 0; glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &n);
    GLint fm[64] = { 0 }; if (n > 64) n = 64; glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, fm);
    const GLenum need[] = { GL_COMPRESSED_RED_RGTC1, GL_COMPRESSED_SIGNED_RED_RGTC1, GL_COMPRESSED_RG_RGTC2,
        GL_COMPRESSED_SIGNED_RG_RGTC2, GL_COMPRESSED_RGBA_BPTC_UNORM, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
        GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT };
    int listed = 0;
    for (unsigned i = 0; i < sizeof need / sizeof need[0]; i++)
        for (int j = 0; j < n; j++) if ((GLenum)fm[j] == need[i]) { listed++; break; }
    printf("GL_COMPRESSED_TEXTURE_FORMATS: %d of 8 RGTC/BPTC formats listed  %s\n", listed, listed == 8 ? "ok" : "FAIL");
    bad += listed != 8;
    GLint ne = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &ne); int rgtc = 0, bptc = 0;
    for (int i = 0; i < ne; i++) {
        const char *x = (const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (!strcmp(x, "GL_ARB_texture_compression_rgtc")) rgtc = 1;
        if (!strcmp(x, "GL_ARB_texture_compression_bptc")) bptc = 1;
    }
    printf("extensions: ARB_texture_compression_rgtc %s, ARB_texture_compression_bptc %s\n",
           rgtc ? "yes" : "NO", bptc ? "yes" : "NO");
    bad += !rgtc + !bptc;

    uint8_t d[32];
    /* BC4 unorm: 200/255 and 40/255 */
    bc4(d, 200, 10); bc4(d + 8, 40, 250);
    { float w[2][4] = { { 200/255.f, 0, 0, 1 }, { 40/255.f, 0, 0, 1 } };
      bad += check_format("RGTC1 (BC4 unorm, ATI1)", GL_COMPRESSED_RED_RGTC1, d, 16, w); }
    /* BC4 snorm: endpoints are signed bytes: 64 -> 64/127, -96 -> -96/127 */
    bc4(d, 64, 0); bc4(d + 8, (uint8_t)-96, 0);
    { float w[2][4] = { { 64/127.f, 0, 0, 1 }, { -96/127.f, 0, 0, 1 } };
      bad += check_format("RGTC1 signed (BC4 snorm)", GL_COMPRESSED_SIGNED_RED_RGTC1, d, 16, w); }
    /* BC5 unorm: R block then G block, per 4x4 block */
    bc4(d, 200, 0); bc4(d + 8, 100, 0); bc4(d + 16, 30, 0); bc4(d + 24, 250, 0);
    { float w[2][4] = { { 200/255.f, 100/255.f, 0, 1 }, { 30/255.f, 250/255.f, 0, 1 } };
      bad += check_format("RGTC2 (BC5 unorm, ATI2)", GL_COMPRESSED_RG_RGTC2, d, 32, w); }
    /* BC5 snorm */
    bc4(d, 127, 0); bc4(d + 8, (uint8_t)-127, 0); bc4(d + 16, 0, 0); bc4(d + 24, 32, 0);
    { float w[2][4] = { { 1, -1, 0, 1 }, { 0, 32/127.f, 0, 1 } };
      bad += check_format("RGTC2 signed (BC5 snorm)", GL_COMPRESSED_SIGNED_RG_RGTC2, d, 32, w); }
    /* BC7 mode 6: (E<<1|P) -> 8 bits */
    bc7_mode6(d, 100, 50, 25, 127, 1); bc7_mode6(d + 16, 10, 120, 64, 32, 0);
    { float w[2][4] = { { 201/255.f, 101/255.f, 51/255.f, 255/255.f }, { 20/255.f, 240/255.f, 128/255.f, 64/255.f } };
      bad += check_format("BPTC unorm (BC7)", GL_COMPRESSED_RGBA_BPTC_UNORM, d, 32, w); }
    /* BC7 sRGB: the same bytes, decoded through sRGB -> linear */
    bc7_mode6(d, 100, 50, 25, 127, 1); bc7_mode6(d + 16, 127, 127, 127, 127, 1);
    { float w[2][4] = { { 0.5840f, 0.1301f, 0.0331f, 1 }, { 1, 1, 1, 1 } };
      bad += check_format("BPTC sRGB (BC7 sRGB)", GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, d, 32, w); }
    /* BC6H: the all-zero block decodes to 0 (mode 0, endpoints 0); alpha reads 1 */
    memset(d, 0, 32);
    { float w[2][4] = { { 0, 0, 0, 1 }, { 0, 0, 0, 1 } };
      bad += check_format("BPTC ufloat (BC6H)", GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, d, 32, w);
      bad += check_format("BPTC sfloat (BC6H signed)", GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, d, 32, w); }

    printf("%s: %d failure(s)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
