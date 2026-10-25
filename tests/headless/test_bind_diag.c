/* 420pack layout(binding=N) diagnostics: which stage's sampling fails (round 19 scratch) */
#include "mgl_test.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

static GLuint mksh(GLenum t, const char *src)
{
    GLuint s = glCreateShader(t);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
               fprintf(stderr, "compile %x log:\n%s\n", t, log); }
    return s;
}

static GLuint mkprog(int n, ...)
{
    va_list ap; va_start(ap, n);
    GLuint p = glCreateProgram();
    for (int i = 0; i < n; i++) { GLuint s = va_arg(ap, GLuint); glAttachShader(p, s); }
    va_end(ap);
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (lk != GL_TRUE) { char log[4096] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               fprintf(stderr, "link log:\n%s\n", log); }
    return p;
}

/* ---------- CTS binding_sampler_single shape, 5 stages, one stage's check armed ---------- */
static int stage_sel(const char *which)
{
    const char *e = getenv("STAGE");
    return !e || !strcmp(e, which);
}
static const char *CHK(const char *w)
{
    return stage_sel(w) ? "    if (vec4(1, 0, 0, 0) != goku_color) { result = vec4(1, 0, 0, 1); }\n" : "";
}

static char VS_B[2048], TC_B[2048], TE_B[2048], GS_B[2048], FS_B[2048];

static void build_cts5_sources(int vs_binding)
{
    snprintf(VS_B, sizeof VS_B,
        "#version 420\n"
        "%s"
        "out vec4 vs_tcs_result;\n"
        "void main()\n{\n"
        "    vec4 result = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "    vec4 goku_color = texture(goku, vec2(0, 0));\n"
        "%s"
        "    vs_tcs_result = result;\n"
        "    gl_Position = vec4(-1.0 + float(gl_VertexID) * 2.0, -1.0 + float(gl_VertexID & 1) * 4.0, 0.0, 1.0);\n"
        "}\n",
        vs_binding ? "layout(binding = 2) uniform sampler2D goku;\n" : "uniform sampler2D goku;\n",
        CHK("vs"));
    snprintf(TC_B, sizeof TC_B,
        "#version 420\n"
        "layout(vertices = 3) out;\n"
        "uniform sampler2D goku;\n"
        "in vec4 vs_tcs_result[];\n"
        "out vec4 tcs_tes_result[];\n"
        "void main()\n{\n"
        "    vec4 result = vs_tcs_result[gl_InvocationID];\n"
        "    vec4 goku_color = texture(goku, vec2(0, 0));\n"
        "%s"
        "    tcs_tes_result[gl_InvocationID] = result;\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "    if (gl_InvocationID == 0) {\n"
        "        gl_TessLevelInner[0] = 1.0; gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
        "    }\n"
        "}\n", CHK("tcs"));
    snprintf(TE_B, sizeof TE_B,
        "#version 420\n"
        "layout(triangles) in;\n"
        "uniform sampler2D goku;\n"
        "in vec4 tcs_tes_result[];\n"
        "out vec4 te_gs_result;\n"
        "void main()\n{\n"
        "    vec4 result = tcs_tes_result[0];\n"
        "    vec4 goku_color = texture(goku, vec2(0, 0));\n"
        "%s"
        "    te_gs_result = result;\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "}\n", CHK("tes"));
    snprintf(GS_B, sizeof GS_B,
        "#version 420\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "uniform sampler2D goku;\n"
        "in vec4 te_gs_result[];\n"
        "out vec4 gs_fs_result;\n"
        "void main()\n{\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    vec4 result = te_gs_result[i];\n"
        "    vec4 goku_color = texture(goku, vec2(0, 0));\n"
        "%s"
        "    gs_fs_result = result;\n"
        "    gl_Position = gl_in[i].gl_Position; EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n", CHK("gs"));
    snprintf(FS_B, sizeof FS_B,
        "#version 420\n"
        "uniform sampler2D goku;\n"
        "layout(location = 0) out vec4 fs_out_result;\n"
        "in vec4 gs_fs_result;\n"
        "void main()\n{\n"
        "    vec4 result = gs_fs_result;\n"
        "    vec4 goku_color = texture(goku, vec2(0, 0));\n"
        "%s"
        "    fs_out_result = result;\n"
        "}\n", CHK("fs"));
}

/* full-screen triangle helper shared by the stages */
static const char *VS_PLAIN_TRI =
    "#version 420\n"
    "out vec4 vs_fs;\n"
    "void main()\n{\n"
    "    vs_fs = vec4(1.0);\n"
    "    gl_Position = vec4(-1.0 + float(gl_VertexID) * 2.0, -1.0 + float(gl_VertexID & 1) * 4.0, 0.0, 1.0);\n"
    "}\n";

TEST(cts_binding_sampler5)
{
    build_cts5_sources(1);
    GLuint vs = mksh(GL_VERTEX_SHADER, VS_B);
    GLuint tc = mksh(GL_TESS_CONTROL_SHADER, TC_B);
    GLuint te = mksh(GL_TESS_EVALUATION_SHADER, TE_B);
    GLuint gs = mksh(GL_GEOMETRY_SHADER, GS_B);
    GLuint fs = mksh(GL_FRAGMENT_SHADER, FS_B);
    GLuint p = mkprog(5, vs, tc, te, gs, fs);
    { GLint loc = glGetUniformLocation(p, "goku");
      GLint val = -1;
      if (loc >= 0) glGetUniformiv(p, loc, &val);
      fprintf(stderr, "5-stage goku uniform loc=%d value=%d (expect 2 per GL 4.20 4.4.5)\n", loc, val); }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    unsigned int data[16 * 16];
    for (int i = 0; i < 16 * 16; i++) data[i] = 0x000000ff;   /* (1,0,0,0) */
    GLuint fb = 0, out = 0, vao = 0;
    glGenFramebuffers(1, &fb); glGenTextures(1, &out); glGenVertexArrays(1, &vao);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    /* CTS order: goku bound at unit 0 (during upload) and unit 2 (explicit) AFTER the fbo texture */
    glBindTexture(GL_TEXTURE_2D, tex);                        /* unit 0 */
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex);                        /* unit 2 */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    glBindVertexArray(vao);
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    GLenum err = glGetError();
    if (err) fprintf(stderr, "GL error after draw: %x\n", err);
    unsigned char px[4] = { 0 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "cts_binding_sampler5 px = %u %u %u %u (want 0 255 0 255; red=stage check failed)\n",
            px[0], px[1], px[2], px[3]);
    EXPECT_EQ_INT(px[1], 255);
    EXPECT_EQ_INT(px[0], 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &out); glDeleteTextures(1, &tex); glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    glDeleteShader(vs); glDeleteShader(tc); glDeleteShader(te); glDeleteShader(gs); glDeleteShader(fs);
}

/* FS-only sampler at unit 0 (no layout binding) */
static const char *FS_UNIT0 =
    "#version 420\n"
    "uniform sampler2D goku;\n"
    "layout(location = 0) out vec4 fs_out;\n"
    "void main() { fs_out = texture(goku, vec2(0, 0)); }\n";

TEST(fs_sampler_unit0)
{
    GLuint vs = mksh(GL_VERTEX_SHADER, VS_PLAIN_TRI);
    GLuint fs = mksh(GL_FRAGMENT_SHADER, FS_UNIT0);
    GLuint p = mkprog(2, vs, fs);
    GLuint tex = 0, fb = 0, out = 0, vao = 0;
    unsigned int data[16 * 16];
    for (int i = 0; i < 16 * 16; i++) data[i] = 0x000000ff;   /* (1,0,0,0) */

    glGenTextures(1, &tex); glGenFramebuffers(1, &fb);
    glGenTextures(1, &out); glGenVertexArrays(1, &vao);
    /* CTS order: framebuffer color texture first (bound at unit 0), then the
     * source texture takes unit 0 (upload) and unit 2 (explicit) */
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);

    glBindTexture(GL_TEXTURE_2D, tex);                        /* unit 0: goku */
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, data);
    if (getenv("NEAREST")) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex);                        /* unit 2: goku */
    glActiveTexture(GL_TEXTURE0);

    { GLint b0 = 0, b2 = 0;
      glGetIntegerv(GL_TEXTURE_BINDING_2D, &b0);
      glActiveTexture(GL_TEXTURE2); glGetIntegerv(GL_TEXTURE_BINDING_2D, &b2);
      glActiveTexture(GL_TEXTURE0);
      fprintf(stderr, "unit0 binding=%d (goku=%d out=%d) unit2 binding=%d\n", b0, (int)tex, (int)out, b2); }
    glBindVertexArray(vao);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    unsigned char px[4] = { 0 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "fs_sampler_unit0 px = %u %u %u %u (want 255 0 0 0)\n", px[0], px[1], px[2], px[3]);
    EXPECT_EQ_INT(px[0], 255);
    EXPECT_EQ_INT(px[1], 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &out); glDeleteTextures(1, &tex); glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p); glDeleteShader(vs); glDeleteShader(fs);
}

static char MVS[2048], MTC[2048], MTE[2048], MFS[2048];

TEST(minimal_vs_sampler)
{
    snprintf(MVS, sizeof MVS,
        "#version 420\n"
        "layout(binding = 2) uniform sampler2D goku;\n"
        "out vec4 vs_fs;\n"
        "void main()\n{\n"
        "    vs_fs = texture(goku, vec2(0, 0));\n"
        "    gl_Position = vec4(-1.0 + float(gl_VertexID) * 2.0, -1.0 + float(gl_VertexID & 1) * 4.0, 0.0, 1.0);\n"
        "}\n");
    snprintf(MTC, sizeof MTC,
        "#version 420\n"
        "layout (vertices = 3) out;\n"
        "in vec4 vs_fs[];\n"
        "out vec4 tc_te[];\n"
        "void main() {\n"
        "  tc_te[gl_InvocationID] = vs_fs[gl_InvocationID];\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelInner[0] = 1.0; gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
        "  }\n"
        "}\n");
    snprintf(MTE, sizeof MTE,
        "#version 420\n"
        "layout (triangles) in;\n"
        "in vec4 tc_te[];\n"
        "out vec4 te_fs;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  te_fs = tc_te[0];\n"
        "}\n");
    snprintf(MFS, sizeof MFS,
        "#version 420\n"
        "layout(location = 0) out vec4 fs_out;\n"
        "in vec4 te_fs;\n"
        "void main() { fs_out = te_fs; }\n");
    GLuint vs = mksh(GL_VERTEX_SHADER, MVS);
    GLuint tc = mksh(GL_TESS_CONTROL_SHADER, MTC);
    GLuint te = mksh(GL_TESS_EVALUATION_SHADER, MTE);
    GLuint fs = mksh(GL_FRAGMENT_SHADER, MFS);
    GLuint p = mkprog(4, vs, tc, te, fs);
    GLuint tex = 0;
    unsigned int data[16 * 16];
    for (int i = 0; i < 16 * 16; i++) data[i] = getenv("GREEN") ? 0xff00ff00 : 0x000000ff;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex);
    if (getenv("TEXIMAGE")) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    else { glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
           glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, data); }
    if (getenv("NEAREST")) { glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); }
    glActiveTexture(GL_TEXTURE0);
    GLuint fb = 0, out = 0, vao = 0;
    glGenFramebuffers(1, &fb); glGenTextures(1, &out); glGenVertexArrays(1, &vao);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    glBindVertexArray(vao);
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    unsigned char px[4] = { 0 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "minimal_vs_sampler px = %u %u %u %u (want 255 0 0 0)\n", px[0], px[1], px[2], px[3]);
    EXPECT_EQ_INT(px[0], 255);
    EXPECT_EQ_INT(px[1], 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &out); glDeleteTextures(1, &tex); glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p); glDeleteShader(vs); glDeleteShader(tc); glDeleteShader(te); glDeleteShader(fs);
}

int main(void)
{
    RUN(minimal_vs_sampler);
    if (getenv("RUN5")) RUN(cts_binding_sampler5);
    RUN(fs_sampler_unit0);
    return mgl_test_summary();
}
