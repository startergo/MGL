/* TCS/TES storage images (glBindImageTexture in the tessellation stages).
 *
 * The TCS runs as a compute kernel and the TES as the post-tessellation vertex
 * function; both bind image units now (tessBindStageTextures' storage-image loop for
 * the kernel, the render path's loop — which must use the VERTEX texture table for
 * the TES — for the raster pass).  GL_MAX_TESS_*_IMAGE_UNIFORMS is 8 (was 0: the
 * CTS shader_image_size basic-*-tcs/tes tests were NotSupported).
 */
#include "mgl_test.h"
#include <stdbool.h>

static const char *vs =
    "#version 460 core\n"
    "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
static const char *tcs =
    "#version 460 core\n"
    "layout(vertices = 3) out;\n"
    "layout(binding = 3, rgba32f) readonly uniform image2D g_image_2d;\n"
    "layout(binding = 0, rgba32i) writeonly uniform iimage2D g_result;\n"
    "void main() {\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
    "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "    imageStore(g_result, ivec2(0, gl_PrimitiveID), ivec4(imageSize(g_image_2d), 0, 0));\n"
    "  }\n"
    "}\n";
static const char *tes =
    "#version 460 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(binding = 4, rgba32f) readonly uniform imageCube g_image_cube;\n"
    "layout(binding = 1, rgba32i) writeonly uniform iimage2D g_result2;\n"
    "void main() {\n"
    "  gl_Position = vec4(gl_TessCoord, 1.0);\n"
    "  imageStore(g_result2, ivec2(0), ivec4(imageSize(g_image_cube), 0, 0));\n"
    "}\n";
static const char *fs =
    "#version 460 core\n"
    "out vec4 c; void main() { c = vec4(1.0, 0.0, 1.0, 1.0); }\n";

static GLuint link_tess(void)
{
    GLuint p = glCreateProgram();
    const char *src[4] = { vs, tcs, tes, fs };
    const GLenum type[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                             GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 4; i++)
    {
        GLuint s = glCreateShader(type[i]);
        glShaderSource(s, 1, &src[i], NULL);
        glCompileShader(s);
        { GLint c = 0; char l[2048] = ""; glGetShaderiv(s, GL_COMPILE_STATUS, &c); glGetShaderInfoLog(s, sizeof l, NULL, l);
          if (!c) fprintf(stderr, "[stage %d] %s\n", i, l); }
        glAttachShader(p, s);
    }
    glLinkProgram(p);
    return p;
}

static bool check(GLint *tcs_out, GLint *tes_out)
{
    bool ok = true;
    if (tcs_out[0] != 256 || tcs_out[1] != 64 || tcs_out[4] != 256 || tcs_out[5] != 64)
    { ok = false; fprintf(stderr, "tcs image: %d %d / %d %d (want 256 64 / 256 64)\n", tcs_out[0], tcs_out[1], tcs_out[4], tcs_out[5]); }
    /* the cube image is bound at level 1 of a 16x16 cube: 8x8.  Only texel 0 is
     * written: the TES runs as the post-tessellation vertex function, where
     * gl_PrimitiveID is the instance id of the emulated patch draw (all patches
     * share instance 0 unless the draw is per-patch instanced). */
    if (tes_out[0] != 8 || tes_out[1] != 8)
    { ok = false; fprintf(stderr, "tes image: %d %d / %d %d (want 8 8 at patch 0)\n",
                          tes_out[0], tes_out[1], tes_out[4], tes_out[5]); }
    return ok;
}

TEST(tess_image_uniforms)
{
    /* follows GL_MAX_IMAGE_UNITS (16): KHR-GL46.gl_spirv.spirv_validation_capabilities
     * links a compute shader with 10 active image uniforms and expects success */
    GLint v = -1;
    glGetIntegerv(GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS, &v);
    EXPECT_EQ_INT(v, 16);
    glGetIntegerv(GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS, &v);
    EXPECT_EQ_INT(v, 16);
}

static void setup(GLuint *res_tcs, GLuint *res_tes, GLuint *src2d, GLuint *srccube)
{
    glGenTextures(1, res_tcs);
    glBindTexture(GL_TEXTURE_2D, *res_tcs);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, 1, 3);
    {
        GLint data[12];
        for (int i = 0; i < 12; i++) data[i] = -7;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 3, GL_RGBA_INTEGER, GL_INT, data);
    }
    glGenTextures(1, res_tes);
    glBindTexture(GL_TEXTURE_2D, *res_tes);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, 1, 3);
    {
        GLint data[12];
        for (int i = 0; i < 12; i++) data[i] = -7;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 3, GL_RGBA_INTEGER, GL_INT, data);
    }
    glGenTextures(1, src2d);
    glBindTexture(GL_TEXTURE_2D, *src2d);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA32F, 512, 128);
    glGenTextures(1, srccube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, *srccube);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 2, GL_RGBA32F, 16, 16);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindImageTexture(0, *res_tcs, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glBindImageTexture(1, *res_tes, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glBindImageTexture(3, *src2d, 1, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(4, *srccube, 1, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
}

TEST(tess_image_draw)
{
    GLuint p = link_tess();
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }

    GLuint res_tcs, res_tes, src2d, srccube, vao;
    setup(&res_tcs, &res_tes, &src2d, &srccube);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 6);          /* 2 patches */
    EXPECT_NO_GL_ERROR();

    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
    GLint tc[12], te[12];
    memset(tc, 0, sizeof tc); memset(te, 0, sizeof te);
    glBindTexture(GL_TEXTURE_2D, res_tcs);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_INT, tc);
    glBindTexture(GL_TEXTURE_2D, res_tes);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_INT, te);
    EXPECT_TRUE(check(tc, te));
}

/* the same draw with rasterizer discard: the tessellation stages still run and their
 * image writes must still land */
TEST(tess_image_discard)
{
    GLuint p = link_tess();
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }

    GLuint res_tcs, res_tes, src2d, srccube, vao;
    setup(&res_tcs, &res_tes, &src2d, &srccube);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_RASTERIZER_DISCARD);
    glDrawArrays(GL_PATCHES, 0, 6);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
    GLint tc[12], te[12];
    memset(tc, 0, sizeof tc); memset(te, 0, sizeof te);
    glBindTexture(GL_TEXTURE_2D, res_tcs);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_INT, tc);
    glBindTexture(GL_TEXTURE_2D, res_tes);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_INT, te);
    EXPECT_TRUE(check(tc, te));
}

int main(void)
{
    RUN(tess_image_uniforms);
    RUN(tess_image_draw);
    RUN(tess_image_discard);
    return mgl_test_summary();
}
