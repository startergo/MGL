/* r180 probe: 3DMark06's post passes under rsvga stay black on the M4. The scene
 * target is Direct3D's X8R8G8B8, kept by rsvga as GL RGBA8 with
 * TEXTURE_SWIZZLE_A = ONE (the X byte reads as one); 3DMark's lighting writes
 * alpha 0 into it, and the post passes blend SRCALPHA / INVSRCALPHA. If a
 * sample of that texture returns its STORED alpha (0) instead of one, the blend
 * keeps the destination: black. Each section here renders alpha 0 into a GPU
 * target with the alpha swizzle set, then samples it in a full-screen pass:
 *   1. separate image + sampler (SPIR-V, rsvga's form), sampler object, blended
 *   2. the same, drawn three times (the draw fast path's repeat)
 *   3. combined sampler2D (GLSL), sampler object, blended
 *   4. separate, blend off: the written alpha must be 255 (the swizzle's one)
 *   5. swizzle set AFTER the texture was sampled once without it (black first,
 *      then the scene: the per-texture resolve cache must see the change)
 *   6. no sampler object (the texture's own parameters), separate, blended
 * Expected everywhere: the scene colour (204, 102, 51). */
#include "mgl_test.h"
#include <stdint.h>
#include "spv_x8post.h"

#define W 64
static const float SC[4] = { 0.8f, 0.4f, 0.2f, 0.0f };   /* alpha 0, like 3DMark's lighting */

static GLuint vao, vbo, scene_prog, post_sep, post_comb, dst_fbo, dst_tex;

static GLuint compile(GLenum ty, const char *src)
{
    GLuint s = glCreateShader(ty);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, sizeof log, NULL, log); printf("  [compile] %s\n", log); }
    return s;
}
static GLuint spv(GLenum ty, const unsigned char *b, size_t n)
{
    GLuint s = glCreateShader(ty);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, b, (GLsizei)n);
    glSpecializeShader(s, "main", 0, NULL, NULL);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, sizeof log, NULL, log); printf("  [specialize] %s\n", log); }
    return s;
}
static GLuint link2(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [link] %s\n", log); }
    return p;
}

/* a GPU-rendered scene target: RGBA8, alpha swizzled to ONE (unless swizzle_now = 0),
 * filled by a DRAW that writes (0.8, 0.4, 0.2, 0) */
static GLuint make_scene(int swizzle_now)
{
    GLuint t, f;
    glCreateTextures(GL_TEXTURE_2D, 1, &t);
    glTextureStorage2D(t, 1, GL_RGBA8, W, W);
    glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    if (swizzle_now)
        glTextureParameteri(t, GL_TEXTURE_SWIZZLE_A, GL_ONE);
    glCreateFramebuffers(1, &f);
    glNamedFramebufferTexture(f, GL_COLOR_ATTACHMENT0, t, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glViewport(0, 0, W, W);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(scene_prog);
    glUniform4fv(glGetUniformLocation(scene_prog, "c"), 1, SC);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    /* the stored bytes: alpha must really be 0 (else this probe proves nothing) */
    unsigned char px[4] = { 9, 9, 9, 9 };
    glFinish();
    glReadPixels(W / 2, W / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    if (px[3] != 0 || abs(px[0] - 204) > 2)
        printf("  [setup] scene stored %d %d %d %d, want 204 102 51 0\n", px[0], px[1], px[2], px[3]);
    return t;
}

static void post(GLuint prog, GLuint tex, GLuint sampler, int blend)
{
    glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
    glViewport(0, 0, W, W);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    if (blend) { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); }
    glUseProgram(prog);
    glBindTextureUnit(2, tex);
    glBindSampler(2, sampler);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisable(GL_BLEND);
}

static int expect(const char *what, int r, int g, int b, int a)
{
    unsigned char px[4];
    glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
    glFinish();
    glReadPixels(W / 2, W / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    int ok = abs(px[0] - r) <= 2 && abs(px[1] - g) <= 2 && abs(px[2] - b) <= 2 && (a < 0 || abs(px[3] - a) <= 2);
    printf("  %-58s %3d %3d %3d %3d  %s\n", what, px[0], px[1], px[2], px[3], ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main(void)
{
    int bad = 0;
    static const float quad[8] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint vs = compile(GL_VERTEX_SHADER,
        "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) out vec2 uv;\n"
        "void main(){ uv = p*0.5+0.5; gl_Position = vec4(p,0.5,1.0); }\n");
    scene_prog = link2(vs, compile(GL_FRAGMENT_SHADER,
        "#version 450\nuniform vec4 c;\nlayout(location=0) out vec4 o;\nvoid main(){ o = c; }\n"));
    post_comb = link2(vs, compile(GL_FRAGMENT_SHADER,
        "#version 450\nlayout(binding=2) uniform sampler2D t;\nlayout(location=0) in vec2 uv;\n"
        "layout(location=0) out vec4 o;\nvoid main(){ o = texture(t, uv); }\n"));
    post_sep = link2(spv(GL_VERTEX_SHADER, spv_post_vert, sizeof spv_post_vert),
                     spv(GL_FRAGMENT_SHADER, spv_post_frag, sizeof spv_post_frag));

    glCreateTextures(GL_TEXTURE_2D, 1, &dst_tex);
    glTextureStorage2D(dst_tex, 1, GL_RGBA8, W, W);
    glCreateFramebuffers(1, &dst_fbo);
    glNamedFramebufferTexture(dst_fbo, GL_COLOR_ATTACHMENT0, dst_tex, 0);

    GLuint smp; glCreateSamplers(1, &smp);
    glSamplerParameteri(smp, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(smp, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(smp, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(smp, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLuint scene = make_scene(1);
    printf("section 1: separate image+sampler, sampler object, SRCALPHA/INVSRCALPHA\n");
    post(post_sep, scene, smp, 1);
    bad += expect("separate, blended", 204, 102, 51, -1);

    printf("section 2: the same draw three times\n");
    for (int i = 0; i < 3; i++) {
        post(post_sep, scene, smp, 1);
        char w[64]; snprintf(w, sizeof w, "separate, blended, repeat %d", i + 1);
        bad += expect(w, 204, 102, 51, -1);
    }

    printf("section 3: combined sampler2D, sampler object, blended\n");
    post(post_comb, scene, smp, 1);
    bad += expect("combined, blended", 204, 102, 51, -1);

    printf("section 4: separate, blend off: the sampled alpha itself\n");
    post(post_sep, scene, smp, 0);
    bad += expect("separate, unblended (alpha must read 255)", 204, 102, 51, 255);
    post(post_comb, scene, smp, 0);
    bad += expect("combined, unblended (alpha must read 255)", 204, 102, 51, 255);

    printf("section 5: swizzle set after a first sample without it\n");
    GLuint late = make_scene(0);
    post(post_sep, late, smp, 1);
    bad += expect("before the swizzle (stored alpha 0: black)", 0, 0, 0, -1);
    glTextureParameteri(late, GL_TEXTURE_SWIZZLE_A, GL_ONE);
    post(post_sep, late, smp, 1);
    bad += expect("after the swizzle, separate, blended", 204, 102, 51, -1);
    post(post_comb, late, smp, 1);
    bad += expect("after the swizzle, combined, blended", 204, 102, 51, -1);

    printf("section 6: no sampler object\n");
    post(post_sep, scene, 0, 1);
    bad += expect("separate, texture's own sampler state, blended", 204, 102, 51, -1);

    GLenum e = glGetError();
    if (e) { printf("  GL error %#x\n", e); bad++; }
    printf("%s: %d failure(s)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
