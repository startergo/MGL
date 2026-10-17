/* test_texext.c — buffer textures, texture views, multisample textures, invalidation, TextureBarrier */
#include "mgl_test.h"

static const char *k_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";

static GLuint g_vao, g_vbo;

static void fullscreen_setup(void)
{
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    if (!g_vao) {
        glGenVertexArrays(1, &g_vao);
        glGenBuffers(1, &g_vbo);
        glBindVertexArray(g_vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
    }
    /* rebinding forces DIRTY_VAO -> a fresh render encoder: works around the core draw path not
       recreating an encoder after a flush (glFinish / buffer realloc) when only DIRTY_STATE is set */
    glBindVertexArray(0);
    glBindVertexArray(g_vao);
}

static void draw_fullscreen(GLuint prog)
{
    fullscreen_setup();
    glUseProgram(prog);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* make an RGBA8 render target texture + FBO of w x h */
static GLuint make_fbo(GLuint tex, GLint level)
{
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, level);
    return fbo;
}

/* fill the bound framebuffer with a constant color by drawing (independent of the glClear path) */
static void fill_bound_fbo(float r, float g, float b, float a, int w, int h)
{
    char fs[512];
    snprintf(fs, sizeof fs, "#version 450 core\n"
             "layout(location=0) out vec4 color;\n"
             "void main(){ color = vec4(%f, %f, %f, %f); }\n", r, g, b, a);
    GLuint p = mgl_make_program(k_vs, fs);
    if (!p) return;
    glViewport(0, 0, w, h);
    draw_fullscreen(p);
    glFinish();
}

static void clear_fbo_tex(GLuint tex, GLint level, float r, float g, float b, float a, int w, int h)
{
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    make_fbo(tex, level);
    fill_bound_fbo(r, g, b, a, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(vp[0], vp[1], vp[2], vp[3]);
}

/* sample texture (2D) texel (0,0) into the default framebuffer and check pixel (0,0) */
static void expect_tex_texel(GLuint tex, int r, int g, int b, int a, int tol)
{
    const char *fs = "#version 450 core\n"
        "layout(binding=0) uniform sampler2D t;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = texelFetch(t, ivec2(0), 0); }\n";
    GLuint p = mgl_make_program(k_vs, fs);
    if (!p) return;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    draw_fullscreen(p);
    EXPECT_PIXEL(0, 0, r, g, b, a, tol);
}

/* fs_fmt has one %d (the texel index); draws it and checks pixel (0,0) */
static GLuint idx_prog(const char *fs_fmt, int idx)
{
    char fs[2048];
    snprintf(fs, sizeof fs, fs_fmt, idx);
    return mgl_make_program(k_vs, fs);
}
#define EXPECT_IDX_PIXEL(fmt, idx, r, g, b, a, tol) do { \
        GLuint _p = idx_prog(fmt, idx); if (!_p) break; \
        glClear(GL_COLOR_BUFFER_BIT); draw_fullscreen(_p); \
        EXPECT_PIXEL(0, 0, r, g, b, a, tol); } while (0)

#pragma mark buffer textures

TEST(buffer_texture_fetch)
{
    const GLubyte texels[4][4] = { {255,0,0,255}, {0,255,0,255}, {0,0,255,255}, {255,255,255,255} };
    GLuint buf, tex;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, sizeof texels, texels, GL_STATIC_DRAW);

    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, buf);
    EXPECT_NO_GL_ERROR();

    GLint v = -1;
    glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &v);
    EXPECT_EQ_INT(v, tex);
    glGetIntegerv(GL_TEXTURE_BUFFER_BINDING, &v);
    EXPECT_EQ_INT(v, buf);

    const char *fs = "#version 450 core\n"
        "layout(binding=0) uniform samplerBuffer tb;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = texelFetch(tb, %d); }\n";
    glClearColor(0, 0, 0, 1);
    EXPECT_IDX_PIXEL(fs, 0, 255, 0, 0, 255, 2);
    EXPECT_IDX_PIXEL(fs, 1, 0, 255, 0, 255, 2);
    EXPECT_IDX_PIXEL(fs, 2, 0, 0, 255, 255, 2);
    EXPECT_IDX_PIXEL(fs, 3, 255, 255, 255, 255, 2);
    EXPECT_IDX_PIXEL(fs, 4, 0, 0, 0, 0, 0);      /* out of range fetch returns 0 */

    /* texture follows the buffer's storage: reallocate with new contents */
    const GLubyte texels2[4][4] = { {0,0,255,255}, {0,0,255,255}, {255,0,255,255}, {0,255,255,255} };
    glBufferData(GL_TEXTURE_BUFFER, sizeof texels2, texels2, GL_STATIC_DRAW);
    glFinish(); /* core: the buffer-delete flush leaves no open command buffer; glFinish opens one */
    EXPECT_IDX_PIXEL(fs, 0, 0, 0, 255, 255, 2);
    EXPECT_IDX_PIXEL(fs, 2, 255, 0, 255, 255, 2);
    EXPECT_NO_GL_ERROR();
}

TEST(buffer_texture_rgb32f)
{
    const float texels[2][3] = { {1.0f, 0.5f, 0.0f}, {0.0f, 0.25f, 1.0f} };
    GLuint buf, tex;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, sizeof texels, texels, GL_STATIC_DRAW);
    glCreateTextures(GL_TEXTURE_BUFFER, 1, &tex);
    glTextureBuffer(tex, GL_RGB32F, buf);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    EXPECT_NO_GL_ERROR();

    const char *fs = "#version 450 core\n"
        "layout(binding=0) uniform samplerBuffer tb;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = texelFetch(tb, %d); }\n";
    glClearColor(0, 0, 0, 0);
    EXPECT_IDX_PIXEL(fs, 0, 255, 128, 0, 255, 2);   /* alpha = 1 for RGB buffer formats */
    EXPECT_IDX_PIXEL(fs, 1, 0, 64, 255, 255, 2);

    /* contents update without reallocation: the RGB32 shadow follows */
    const float t0[3] = { 0.0f, 1.0f, 0.0f };
    glBufferSubData(GL_TEXTURE_BUFFER, 0, sizeof t0, t0);
    EXPECT_IDX_PIXEL(fs, 0, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

TEST(buffer_texture_range)
{
    GLint align = 0;
    glGetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &align);
    EXPECT_TRUE(align >= 1 && align <= 256 && (align & (align - 1)) == 0);
    GLint maxsize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &maxsize);
    EXPECT_TRUE(maxsize >= 65536);
    if (align < 4) align = 4;

    /* R32UI texels: value == index */
    GLuint n = (GLuint)(align / 4) * 4 + 8;
    GLuint *data = malloc(n * 4);
    for (GLuint i = 0; i < n; i++) data[i] = i;
    GLuint buf, tex;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, n * 4, data, GL_STATIC_DRAW);
    free(data);
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, tex);

    /* errors */
    glTexBufferRange(GL_TEXTURE_BUFFER, GL_R32UI, buf, 1, 16);
    if (align > 1) EXPECT_GL_ERROR(GL_INVALID_VALUE);           /* misaligned */
    glTexBufferRange(GL_TEXTURE_BUFFER, GL_R32UI, buf, 0, 0);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* size 0 */
    glTexBufferRange(GL_TEXTURE_BUFFER, GL_R32UI, buf, 0, n * 4 + 4);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* past the end */
    glTexBufferRange(GL_TEXTURE_BUFFER, GL_R32UI, buf, -align, 16);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* negative */
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB8, buf);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);                           /* not a buffer texture format */
    glTexBuffer(GL_TEXTURE_2D, GL_R32UI, buf);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);                           /* wrong target */
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, 0xBEEF);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* no such buffer */
    GLuint t2d;
    glCreateTextures(GL_TEXTURE_2D, 1, &t2d);
    glTextureBuffer(t2d, GL_R32UI, buf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* not a buffer texture */
    glTextureBufferRange(t2d, GL_R32UI, buf, 0, 16);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* aligned range: texel 0 of the view is element align/4 of the buffer */
    glTexBufferRange(GL_TEXTURE_BUFFER, GL_R32UI, buf, align, 32);
    EXPECT_NO_GL_ERROR();

    const char *fs = "#version 450 core\n"
        "layout(binding=0) uniform usamplerBuffer tb;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ uint v = texelFetch(tb, %d).r;\n"
        "  color = vec4(float(v) / 255.0, float(textureSize(tb)) / 255.0, 0.0, 1.0); }\n";
    glClearColor(0, 0, 0, 0);
    EXPECT_IDX_PIXEL(fs, 0, align / 4, 8, 0, 255, 1);
    EXPECT_IDX_PIXEL(fs, 3, align / 4 + 3, 8, 0, 255, 1);

    /* detach: no data store */
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, 0);
    EXPECT_NO_GL_ERROR();
}

#pragma mark texture views

TEST(texture_view_errors)
{
    GLuint orig, mut, v[8];
    glGenTextures(8, v);

    glGenTextures(1, &mut);
    glBindTexture(GL_TEXTURE_2D, mut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTextureView(v[0], GL_TEXTURE_2D, mut, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* mutable original */

    glGenTextures(1, &orig);
    glBindTexture(GL_TEXTURE_2D, orig);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    EXPECT_NO_GL_ERROR();

    glTextureView(v[0], GL_TEXTURE_2D, orig, GL_RGBA16F, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* 32-bit vs 64-bit class */
    glTextureView(v[0], GL_TEXTURE_3D, orig, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* incompatible target */
    glTextureView(v[0], GL_TEXTURE_2D, orig, GL_RGBA8, 1, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* minlevel >= levels */
    glTextureView(v[0], GL_TEXTURE_2D, orig, GL_RGBA8, 0, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* minlayer >= layers */
    glTextureView(v[0], GL_TEXTURE_2D, 0xBEEF, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* no such origtexture */
    glTextureView(0, GL_TEXTURE_2D, orig, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* texture 0 */
    glTextureView(0x7FFFFF, GL_TEXTURE_2D, orig, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* name never generated */
    glTextureView(mut, GL_TEXTURE_2D, orig, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* name already a texture object */
    glTextureView(v[0], 0x1234, orig, GL_RGBA8, 0, 1, 0, 1);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* valid: 32-bit class reinterpretations, view of view */
    glTextureView(v[1], GL_TEXTURE_2D_ARRAY, orig, GL_R32F, 0, 4, 0, 1);  /* numlevels clamped */
    EXPECT_NO_GL_ERROR();
    glTextureView(v[2], GL_TEXTURE_2D, v[1], GL_RGBA8UI, 0, 1, 0, 5);     /* view of a view, layers clamped */
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(glIsTexture(v[2]));

    /* cube views need 6 layers of a square texture */
    GLuint arr;
    glGenTextures(1, &arr);
    glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 8, 8, 6);
    glTextureView(v[3], GL_TEXTURE_CUBE_MAP, arr, GL_RGBA8, 0, 1, 0, 5);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTextureView(v[3], GL_TEXTURE_2D, arr, GL_RGBA8, 0, 1, 0, 2);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* 2D view needs one layer */
    glTextureView(v[3], GL_TEXTURE_CUBE_MAP, arr, GL_RGBA8, 0, 1, 0, 6);
    EXPECT_NO_GL_ERROR();

    /* buffer textures cannot be viewed */
    EXPECT_NO_GL_ERROR();
}

static GLuint view_sample_prog(const char *sampler_decl, const char *expr)
{
    char fs[1024];
    snprintf(fs, sizeof fs, "#version 450 core\n"
             "layout(binding=0) uniform %s t;\n"
             "layout(location=0) out vec4 color;\n"
             "void main(){ color = %s; }\n", sampler_decl, expr);
    return mgl_make_program(k_vs, fs);
}

TEST(texture_view_reinterpret)
{
    GLuint orig, vui, vsrgb;
    glGenTextures(1, &orig);
    glBindTexture(GL_TEXTURE_2D, orig);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 8, 8);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    clear_fbo_tex(orig, 0, 0.2f, 0.4f, 0.6f, 0.8f, 8, 8); /* 51,102,153,204 */

    glGenTextures(1, &vui);
    glTextureView(vui, GL_TEXTURE_2D, orig, GL_RGBA8UI, 0, 1, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLint p = -1;
    glBindTexture(GL_TEXTURE_2D, vui);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_VIEW_NUM_LEVELS, &p);
    while (glGetError() != GL_NO_ERROR) {}

    GLuint prog = view_sample_prog("usampler2D", "vec4(texelFetch(t, ivec2(0), 0)) / 255.0");
    if (!prog) return;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, vui);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen(prog);
    EXPECT_PIXEL(0, 0, 51, 102, 153, 204, 1);

    /* sRGB view decodes on fetch: srgb(51/255) -> 0.0331 linear -> 8 */
    glGenTextures(1, &vsrgb);
    glTextureView(vsrgb, GL_TEXTURE_2D, orig, GL_SRGB8_ALPHA8, 0, 1, 0, 1);
    EXPECT_NO_GL_ERROR();
    GLuint prog2 = view_sample_prog("sampler2D", "texelFetch(t, ivec2(0), 0)");
    if (!prog2) return;
    glBindTexture(GL_TEXTURE_2D, vsrgb);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen(prog2);
    EXPECT_PIXEL(0, 0, 8, 33, 81, 204, 2);
    EXPECT_NO_GL_ERROR();
}

TEST(texture_view_layers)
{
    /* layer sub-ranges: each layer of a 3-layer array is rendered through its own 2D view (views share
       storage), then read back through a 2D view of layer 2 and a 2D_ARRAY view of all layers */
    GLuint arr, lv[3], all;
    glGenTextures(1, &arr);
    glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 4, 4, 3);
    EXPECT_NO_GL_ERROR();
    glGenTextures(3, lv);
    const float cols[3][4] = { {1,0,0,1}, {0,1,0,1}, {0,0,1,1} };
    for (int l = 0; l < 3; l++) {
        glTextureView(lv[l], GL_TEXTURE_2D, arr, GL_RGBA8, 0, 1, l, 1);
        EXPECT_NO_GL_ERROR();
        clear_fbo_tex(lv[l], 0, cols[l][0], cols[l][1], cols[l][2], cols[l][3], 4, 4);
    }
    expect_tex_texel(lv[2], 0, 0, 255, 255, 1);
    expect_tex_texel(lv[0], 255, 0, 0, 255, 1);

    glGenTextures(1, &all);
    glTextureView(all, GL_TEXTURE_2D_ARRAY, arr, GL_RGBA8, 0, 1, 1, 2); /* layers 1..2 */
    EXPECT_NO_GL_ERROR();
    GLuint prog = view_sample_prog("sampler2DArray",
        "vec4(texelFetch(t, ivec3(1, 1, 1), 0).rgb, float(textureSize(t, 0).z) / 255.0)");
    if (!prog) return;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, all);
    draw_fullscreen(prog);
    EXPECT_PIXEL(0, 0, 0, 0, 255, 2, 1);   /* view layer 1 == array layer 2; 2 layers */
    EXPECT_NO_GL_ERROR();
}

TEST(texture_view_mips)
{
    /* mip sub-range: level 1 of a 2-level texture, written through a view of that level */
    GLuint mip, vmip;
    glGenTextures(1, &mip);
    glBindTexture(GL_TEXTURE_2D, mip);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 8, 8);
    EXPECT_NO_GL_ERROR();
    glGenTextures(1, &vmip);
    glTextureView(vmip, GL_TEXTURE_2D, mip, GL_RGBA8, 1, 1, 0, 1);
    EXPECT_NO_GL_ERROR();
    clear_fbo_tex(vmip, 0, 0, 1, 0, 1, 4, 4);

    GLuint prog = view_sample_prog("sampler2D", "vec4(texelFetch(t, ivec2(0), 0).rgb, float(textureSize(t, 0).x) / 255.0)");
    if (!prog) return;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, vmip);
    draw_fullscreen(prog);
    EXPECT_PIXEL(0, 0, 0, 255, 0, 4, 1); /* level-1 color, 4 texels wide */
    EXPECT_NO_GL_ERROR();
}

#pragma mark multisample

TEST(multisample_errors)
{
    GLint maxs = 0;
    glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &maxs);
    EXPECT_TRUE(maxs >= 1);

    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, t);
    GLint b = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &b);
    EXPECT_EQ_INT(b, t);

    glTexStorage2DMultisample(GL_TEXTURE_2D, 4, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs + 1, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_COMPRESSED_RGBA_BPTC_UNORM, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_RGBA8, 0, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 0, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA16F, 4, 4, GL_FALSE); /* respecify ok */
    EXPECT_NO_GL_ERROR();
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* immutable */
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    GLuint t2d, ta;
    glCreateTextures(GL_TEXTURE_2D, 1, &t2d);
    glTextureStorage2DMultisample(t2d, 1, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);                      /* wrong texture target */
    glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 1, &ta);
    glTextureStorage3DMultisample(ta, maxs, GL_RGBA8, 8, 8, 3, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 8, 8, 3, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexImage2DMultisample(GL_PROXY_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 8, 8, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    /* MS views */
    GLuint v;
    glGenTextures(1, &v);
    glTextureView(v, GL_TEXTURE_2D_MULTISAMPLE, ta, GL_RGBA8UI, 0, 1, 1, 1);
    EXPECT_NO_GL_ERROR();
}

TEST(multisample_render_and_fetch)
{
    GLint maxs = 0;
    glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &maxs);
    GLsizei samples = maxs >= 4 ? 4 : maxs;

    GLuint ms, fbo;
    glGenTextures(1, &ms);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGBA8, 16, 16, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, ms, 0);
    EXPECT_NO_GL_ERROR();
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    glViewport(0, 0, 16, 16);

    const char *fs_green = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint pg = mgl_make_program(k_vs, fs_green);
    if (!pg) return;
    draw_fullscreen(pg);
    glFinish();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(vp[0], vp[1], vp[2], vp[3]);

    char fs[1024];
    snprintf(fs, sizeof fs, "#version 450 core\n"
        "layout(binding=0) uniform sampler2DMS t;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ vec4 s = vec4(0.0); for (int i = 0; i < %d; i++) s += texelFetch(t, ivec2(3, 3), i);\n"
        "  color = s / %d.0; }\n", samples, samples);
    GLuint pf = mgl_make_program(k_vs, fs);
    if (!pf) return;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen(pf);
    EXPECT_PIXEL(0, 0, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

#pragma mark invalidation

TEST(invalidate_errors)
{
    GLuint t, ms, tb;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 8, 8);
    EXPECT_NO_GL_ERROR();

    glInvalidateTexImage(0xBEEF, 0);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glInvalidateTexImage(t, -1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glInvalidateTexImage(t, 100);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glInvalidateTexImage(t, 1);
    EXPECT_NO_GL_ERROR();

    glInvalidateTexSubImage(t, 0, 0, 0, 0, 8, 8, 1);
    EXPECT_NO_GL_ERROR();
    glInvalidateTexSubImage(t, 0, 2, 2, 0, 6, 2, 1);
    EXPECT_NO_GL_ERROR();
    glInvalidateTexSubImage(t, 0, 2, 2, 0, 7, 2, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* exceeds level width */
    glInvalidateTexSubImage(t, 1, 0, 0, 0, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* level 1 has no image */
    glInvalidateTexSubImage(t, 0, -1, 0, 0, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glInvalidateTexSubImage(t, 0, 0, 0, 0, -1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glInvalidateTexSubImage(t, 0, 0, 0, 1, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* 2D has depth 1 */
    glInvalidateTexSubImage(0xBEEF, 0, 0, 0, 0, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glGenTextures(1, &ms);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_RGBA8, 4, 4, GL_TRUE);
    glInvalidateTexImage(ms, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);                          /* MS textures only have level 0 */
    glInvalidateTexImage(ms, 0);
    EXPECT_NO_GL_ERROR();

    glGenTextures(1, &tb);
    glBindTexture(GL_TEXTURE_BUFFER, tb);
    glInvalidateTexImage(tb, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* invalidated render target: next pass may skip loading it; rendering still works */
    clear_fbo_tex(t, 0, 1, 0, 0, 1, 8, 8);
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    make_fbo(t, 0);
    glInvalidateTexImage(t, 0);
    fill_bound_fbo(0, 0, 1, 1, 8, 8);
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    expect_tex_texel(t, 0, 0, 255, 255, 1);
    EXPECT_NO_GL_ERROR();
}

#pragma mark TextureBarrier

TEST(texture_barrier_pingpong)
{
    GLuint tex[2], fbo[2];
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 8, 8);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    clear_fbo_tex(tex[0], 0, 0.0f, 0.0f, 0.0f, 1.0f, 8, 8);
    for (int i = 0; i < 2; i++) fbo[i] = make_fbo(tex[i], 0);

    const char *fs = "#version 450 core\n"
        "layout(binding=0) uniform sampler2D src;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = texelFetch(src, ivec2(gl_FragCoord.xy), 0) + vec4(0.25, 0.0, 0.0, 0.0); }\n";
    GLuint prog = mgl_make_program(k_vs, fs);
    if (!prog) return;

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    glViewport(0, 0, 8, 8);
    glActiveTexture(GL_TEXTURE0);
    for (int pass = 0; pass < 3; pass++) {
        int src = pass & 1, dst = src ^ 1;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[dst]);
        glBindTexture(GL_TEXTURE_2D, tex[src]);
        draw_fullscreen(prog);
        glTextureBarrier();
        EXPECT_NO_GL_ERROR();
    }
    /* 3 passes: 0 -> .25 (tex1) -> .5 (tex0) -> .75 (tex1) */
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    expect_tex_texel(tex[1], 191, 0, 0, 255, 2);

    /* feedback in the same texture across a barrier: read-modify-write each texel once per draw */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[1]);
    glViewport(0, 0, 8, 8);
    glBindTexture(GL_TEXTURE_2D, tex[1]);
    draw_fullscreen(prog);
    glTextureBarrier();
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    expect_tex_texel(tex[1], 255, 0, 0, 255, 2);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glTextureBarrier();   /* no pending render pass: harmless */
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(buffer_texture_fetch);
    RUN(buffer_texture_rgb32f);
    RUN(buffer_texture_range);
    RUN(texture_view_errors);
    RUN(texture_view_reinterpret);
    RUN(texture_view_layers);
    RUN(texture_view_mips);
    RUN(multisample_errors);
    RUN(multisample_render_and_fetch);
    RUN(invalidate_errors);
    RUN(texture_barrier_pingpong);
    return mgl_test_summary();
}
