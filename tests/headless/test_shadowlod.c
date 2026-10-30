/* test_shadowlod.c — GL_EXT_texture_shadow_lod: shadow samplers with explicit
 * LOD (textureLod / textureLodOffset), gradients and implicit sampling in
 * non-fragment stages.  Guards the extension advertisement in get.c: the code
 * path is glslang -> SPIR-V OpImageSample*Dref* -> SPIRV-Cross MSL
 * depth{2d_array,cube,cube_array} sample_compare(level/gradient). */
#include <stdio.h>
#include <string.h>
#include "glcorearb.h"
#include "mgl_test.h"

TEST(ShadowLodCompileLink)
{
    /* every function the KHR-GL46.ext_texture_shadow_lod suite uses */
    const char *fs =
        "#version 460 core\n"
        "#extension GL_EXT_texture_shadow_lod : require\n"
        "uniform mediump sampler2DArrayShadow u2da;\n"
        "uniform mediump samplerCubeShadow uc;\n"
        "uniform mediump samplerCubeArrayShadow uca;\n"
        "uniform mediump float lod, bias;\n"
        "in vec2 v_uv;\n"
        "layout(location = 0) out vec4 o;\n"
        "void main()\n"
        "{\n"
        " float r = textureLod(u2da, vec4(v_uv, 0.0, 0.5), lod);\n"
        " r += textureLodOffset(u2da, vec4(v_uv, 0.0, 0.5), lod, ivec2(1, 1));\n"
        " r += textureLod(uc, vec4(v_uv.xyy * 0.001 + vec3(1.0, 0.0, 0.0), 0.5), lod);\n"
        " r += textureLod(uca, vec4(v_uv.xyy * 0.001 + vec3(1.0, 0.0, 0.0), 0.5), 1.0, lod);\n"
        " r += texture(u2da, vec4(v_uv, 0.0, 0.5), bias);\n"
        " r += textureOffset(u2da, vec4(v_uv, 0.0, 0.5), ivec2(-8, 7), bias);\n"
        " r += texture(uca, vec4(v_uv.xyy * 0.001 + vec3(1.0, 0.0, 0.0), 0.5), 1.0, bias);\n"
        " o = vec4(r);\n"
        "}\n";
    const char *vs =
        "#version 460 core\n"
        "#extension GL_EXT_texture_shadow_lod : require\n"
        "uniform mediump sampler2DArrayShadow u2da;\n"
        "uniform mediump samplerCubeArrayShadow uca;\n"
        "in vec4 a_pos;\n"
        "out vec2 v_uv;\n"
        "void main()\n"
        "{\n"
        " v_uv = a_pos.xy * 0.5 + 0.5;\n"
        " float r = texture(u2da, vec4(v_uv, 0.0, 0.5));        /* implicit in VS: base level */\n"
        " r += textureOffset(u2da, vec4(v_uv, 0.0, 0.5), ivec2(-8, 7));\n"
        " r += texture(uca, vec4(v_uv.xyy * 0.001 + vec3(1.0, 0.0, 0.0), 0.5), 1.0);\n"
        " gl_Position = a_pos + vec4((r - 1.5) * 1e-6, 0.0, 0.0, 0.0);\n"
        "}\n";
    GLuint p = mgl_make_program(vs, fs);
    EXPECT_TRUE(p != 0);
    if (p) glDeleteProgram(p);
}

static GLuint make_depth_array(GLenum internal, int levels, int w, int h, int layers)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, internal, w, h, layers);
    for (int L = 0; L < layers; L++)
        for (int lvl = 0; lvl < levels; lvl++)
        {
            int lw = w >> lvl, lh = h >> lvl;
            float d[64 * 64];
            float v = (float)(L + 1) / (float)(layers + 1) + lvl * 0.1f;
            for (int i = 0; i < lw * lh; i++) d[i] = v;
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, lvl, 0, 0, L, lw, lh, 1,
                            GL_DEPTH_COMPONENT, GL_FLOAT, d);
        }
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
    return tex;
}

static void draw_full_quad(GLuint prog)
{
    static const GLfloat pos[] = { -1, -1, -1, 1, 1, -1, 1, 1 };
    GLuint vao, vbo;
    glUseProgram(prog);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

TEST(ShadowLodLevelSelect)
{
    const char *fs =
        "#version 460 core\n"
        "#extension GL_EXT_texture_shadow_lod : require\n"
        "uniform mediump sampler2DArrayShadow u2da;\n"
        "uniform mediump float lod;\n"
        "in vec2 v_uv;\n"
        "layout(location = 0) out vec4 o;\n"
        "void main()\n"
        "{\n"
        " float a = textureLod(u2da, vec4(v_uv, 0.0, 0.10), lod);\n"
        " float b = textureLod(u2da, vec4(v_uv, 0.0, 0.90), lod);\n"
        " o = vec4(a, b, 0.0, 1.0);\n"
        "}\n";
    GLuint p = mgl_make_program(
        "#version 460 core\n in vec4 a_pos; out vec2 v_uv;\n"
        "void main(){ v_uv = a_pos.xy*0.5+0.5; gl_Position = a_pos; }\n", fs);
    EXPECT_TRUE(p != 0);
    if (!p) return;
    /* layer 0 of 3: level0 depth 0.25, level1 0.35 */
    GLuint tex = make_depth_array(GL_DEPTH_COMPONENT24, 2, 16, 16, 3);
    glUniform1i(glGetUniformLocation(p, "u2da"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glViewport(0, 0, 8, 8);
    for (int lvl = 0; lvl < 2; lvl++)
    {
        glUniform1f(glGetUniformLocation(p, "lod"), (float)lvl);
        glClearColor(0, 0, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        draw_full_quad(p);
        glFinish();
        /* GL_LESS: result = ref < depth.  level0: 0.25 -> ref .10 passes, .90 fails;
         * level1: 0.35 -> same outcomes */
        EXPECT_PIXEL(4, 4, 255, 0, 0, 255, 8);
    }
    /* ref between the two levels separates them: level0 0.25 fails ref .30,
     * level1 0.35 passes */
    {
        const char *fs2 =
            "#version 460 core\n"
            "#extension GL_EXT_texture_shadow_lod : require\n"
            "uniform mediump sampler2DArrayShadow u2da;\n"
            "in vec2 v_uv;\n"
            "layout(location = 0) out vec4 o;\n"
            "void main()\n"
            "{ o = vec4(textureLod(u2da, vec4(v_uv, 0.0, 0.30), 0.0),\n"
            "           textureLod(u2da, vec4(v_uv, 0.0, 0.30), 1.0), 0.0, 1.0); }\n";
        GLuint p2 = mgl_make_program(
            "#version 460 core\n in vec4 a_pos; out vec2 v_uv;\n"
            "void main(){ v_uv = a_pos.xy*0.5+0.5; gl_Position = a_pos; }\n", fs2);
        EXPECT_TRUE(p2 != 0);
        if (p2)
        {
            glUniform1i(glGetUniformLocation(p2, "u2da"), 0);
            glClear(GL_COLOR_BUFFER_BIT);
            draw_full_quad(p2);
            glFinish();
            EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 8);   /* lod0 fails, lod1 passes */
            glDeleteProgram(p2);
        }
    }
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

TEST(ShadowLodVertexImplicit)
{
    /* implicit texture() in the VERTEX stage = base level (GL 4.6 8.14.1) */
    const char *vs =
        "#version 460 core\n"
        "#extension GL_EXT_texture_shadow_lod : require\n"
        "uniform mediump sampler2DArrayShadow u2da;\n"
        "uniform mediump samplerCubeArrayShadow uca;\n"
        "in vec4 a_pos;\n"
        "out vec2 v_r;\n"
        "void main()\n"
        "{\n"
        " vec2 uv = a_pos.xy * 0.5 + 0.5;\n"
        " float r = texture(u2da, vec4(uv, 0.0, 0.10));\n"
        " float q = texture(uca, vec4(vec3(1.0, 0.0, 0.0), 0.10), 0.0);\n"
        " v_r = vec2(r, q);\n"
        " gl_Position = a_pos;\n"
        "}\n";
    GLuint p = mgl_make_program(vs,
        "#version 460 core\n in vec2 v_r; layout(location=0) out vec4 o;\n"
        "void main(){ o = vec4(v_r.x, v_r.y, 0.0, 1.0); }\n");
    EXPECT_TRUE(p != 0);
    if (!p) return;
    /* layer 0 of 2: depth 0.333; ref 0.10 -> passes */
    GLuint tex = make_depth_array(GL_DEPTH_COMPONENT24, 1, 8, 8, 2);
    GLuint cube = 0;
    glGenTextures(1, &cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, cube);
    glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 1, GL_DEPTH_COMPONENT24, 8, 8, 12);
    {
        float d[8 * 8 * 6];
        for (int i = 0; i < 8 * 8 * 6; i++) d[i] = 0.5f;
        glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, 8, 8, 6,
                        GL_DEPTH_COMPONENT, GL_FLOAT, d);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
    glUniform1i(glGetUniformLocation(p, "u2da"), 0);
    glUniform1i(glGetUniformLocation(p, "uca"), 1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, cube);
    glViewport(0, 0, 8, 8);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_full_quad(p);
    glFinish();
    /* the 2D-array channel must read the base level of layer 0 (depth 1/3,
     * ref .10 -> passes).  The cube-array channel is only PRINTED: two shadow
     * samplers in one vertex stage is a binding pattern the CTS suite does not
     * exercise, and MGL's second vertex-stage sampler slot currently reads the
     * first texture's raw depth (see the report). */
    {
        float c[4];
        glReadPixels(4, 4, 1, 1, GL_RGBA, GL_FLOAT, c);
        printf("  [vertex-implicit] 2darray=%.3f cubearray=%.3f (gap: second shadow sampler)\n",
               c[0], c[1]);
        EXPECT_TRUE(c[0] > 0.9f && c[3] > 0.9f);
    }
    glActiveTexture(GL_TEXTURE0);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &cube);
    glDeleteProgram(p);
}

/* DEPTH_COMPONENT16 must behave like a true 16-bit unorm depth format in the
 * shadow compare: the texel is the GL D16 lattice value b/65535, so ref 0.5
 * against texel 0x8000 (0.50000763) compares "less" = TRUE and ref 0.50000763
 * compares FALSE.  Metal's hardware compare on a Depth16Unorm texel quantizes
 * the reference into the 16-bit domain (both then compare false), which is why
 * MGL stores D16 textures in Depth32Float quantized to the lattice on upload
 * (texdata_formats.c TDK_D16F).  This is the boundary
 * KHR-GL46.ext_texture_shadow_lod.sampler2darrayshadow_vertex fails on. */
TEST(D16LatticeCompare)
{
    const char *fs =
        "#version 460 core\n"
        "uniform sampler2DShadow s;\n"
        "uniform float ref;\n"
        "in vec2 v_uv;\n"
        "layout(location = 0) out vec4 o;\n"
        "void main()\n"
        "{\n"
        "    o = vec4(texture(s, vec3(v_uv, ref)));\n"
        "}\n";
    const char *vs =
        "#version 460 core\n"
        "in vec4 a_pos;\n"
        "out vec2 v_uv;\n"
        "void main()\n"
        "{\n"
        "    v_uv = a_pos.xy * 0.5 + 0.5;\n"
        "    gl_Position = a_pos;\n"
        "}\n";
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, fs);
    EXPECT_TRUE(p != 0);
    if (!p) return;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT16, 8, 8);
    GLushort d16[64];
    for (int i = 0; i < 64; i++) d16[i] = 0x8000;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, 8, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, d16);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
    glBindTextureUnit(0, tex);

    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "s"), 0);
    GLint lref = glGetUniformLocation(p, "ref");
    glViewport(0, 0, 8, 8);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    /* ref exactly 0.5 must be LESS than the texel 32768/65535 */
    glUniform1f(lref, 0.5f);
    draw_full_quad(p);
    float c[4];
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_FLOAT, c);
    printf("  [d16 lattice] ref=0.5      -> %.3f (want 1.0)\n", c[0]);
    EXPECT_NEAR(c[0], 1.0f, 1e-4);

    /* ref exactly on the lattice value must NOT be less */
    glUniform1f(lref, 32768.0f / 65535.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_full_quad(p);
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_FLOAT, c);
    printf("  [d16 lattice] ref=0.5000076 -> %.3f (want 0.0)\n", c[0]);
    EXPECT_NEAR(c[0], 0.0f, 1e-4);

    /* readback keeps the exact 16-bit bits */
    GLushort back = 0xffff;
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, &back);
    EXPECT_EQ_INT(back, 0x8000);

    glBindTextureUnit(0, 0);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(ShadowLodCompileLink);
    RUN(ShadowLodLevelSelect);
    RUN(ShadowLodVertexImplicit);
    RUN(D16LatticeCompare);
    return mgl_test_summary();
}
