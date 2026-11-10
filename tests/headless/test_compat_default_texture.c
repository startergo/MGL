/* test_compat_default_texture — GL 4.6 core-context semantics of texture object
 * ZERO (Mesa parity, texobj.c bind_tex_obj):
 *
 *  - glBindTexture(GL_TEXTURE_2D, 0) binds the target's DEFAULT texture object,
 *    not "nothing": TexImage/TexParam into it are legal (only immutable storage
 *    on name 0 is GL_INVALID_OPERATION) and the uploaded image can be sampled.
 *    dEQP-GL45-ES31.functional.shaders.framebuffer_fetch.basic.texel_fetch
 *    (the ES31 suite run on a desktop core context) uploads its sampler grid
 *    into texture 0, re-attaches the framebuffer texture, re-binds 0 and
 *    samples the grid through it — MGL used to clear the slot binding instead,
 *    so the fetch read black and the case failed 2/2 once
 *    GL_EXT_shader_framebuffer_fetch was advertised.
 *  - GL_TEXTURE_BINDING_2D still queries 0 (the default object carries a
 *    reserved marker name).
 *  - glTexStorage2D on name 0 stays GL_INVALID_OPERATION (GL 4.6 8.1 / Mesa
 *    _mesa_texstorage: immutable storage cannot live on the default texture).
 *  - re-binding 0 after a named texture reverts the unit to the default
 *    object: a later upload into name 0 is what the unit samples.
 */
#include "mgl_test.h"

static GLuint mkprog(const char *fs)
{
    const char *vs = "#version 450\n"
                     "void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2);"
                     " gl_Position = vec4(p*2.0-1.0, 0.0, 1.0); }";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    GLint ok = 0;
    glShaderSource(v, 1, (const GLchar *[]){ vs }, NULL);
    glCompileShader(v);
    glShaderSource(f, 1, (const GLchar *[]){ fs }, NULL);
    glCompileShader(f);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    return ok ? p : 0;
}

static GLuint mk_target(void)
{
    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fbo;
}

static void upload_grid_to_name0(void)
{
    unsigned char img[8 * 8 * 4];
    for (int i = 0; i < 8 * 8; i++)
    {
        img[i * 4 + 0] = (unsigned char)((i % 8) < 4 ? 255 : 0);
        img[i * 4 + 1] = 0;
        img[i * 4 + 2] = (unsigned char)((i % 8) < 4 ? 0 : 255);
        img[i * 4 + 3] = 255;
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
}

static int sample_center(void)
{
    unsigned char px[4];
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    /* texel (4,4): cell x=4..7 is blue */
    return (px[2] == 255 && px[0] == 0 && px[1] == 0 && px[3] == 255);
}

TEST(default_texture_binds_and_samples)
{
    GLuint fbo, vao, prog;
    GLint bound = -1;

    fbo = mk_target();
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    EXPECT_NO_GL_ERROR();
    upload_grid_to_name0();
    EXPECT_NO_GL_ERROR();       /* TexImage/TexParam on the default texture are legal */
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
    EXPECT_EQ_INT(bound, 0);    /* the marker name still queries as 0 */

    /* immutable storage cannot live on the default texture (Mesa parity) */
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* texelFetch through unit 1 must read the grid uploaded into name 0 */
    prog = mkprog("#version 450\n"
                  "layout(location=0) out vec4 o;\n"
                  "uniform sampler2D s;\n"
                  "void main(){ o = texelFetch(s, ivec2(gl_FragCoord.xy), 0); }\n");
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "s"), 1);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(sample_center());

    /* re-binding 0 after a named texture reverts to the default object:
     * the grid is still what the unit samples */
    {
        GLuint nt;
        glGenTextures(1, &nt);
        glBindTexture(GL_TEXTURE_2D, nt);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
        glClearTexImage(nt, 0, GL_RGBA, GL_UNSIGNED_BYTE, (int[4]){ 0, 0, 0, 0 });
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        EXPECT_TRUE(sample_center());
        glDeleteTextures(1, &nt);
    }

    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
    glDeleteFramebuffers(1, &fbo);
}

TEST(bind_texture_unit_zero_reverts_to_default)
{
    GLuint fbo, vao, prog, nt;
    fbo = mk_target();

    /* put the grid in the default texture at unit 3, then break every binding
     * of the unit with glBindTextureUnit(unit, 0) — the slot reverts to the
     * DEFAULT texture, not "no texture", so the grid still samples */
    glActiveTexture(GL_TEXTURE3);
    upload_grid_to_name0();
    glGenTextures(1, &nt);
    glBindTexture(GL_TEXTURE_2D, nt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTextureUnit(3, 0);
    EXPECT_NO_GL_ERROR();

    prog = mkprog("#version 450\n"
                  "layout(location=0) out vec4 o;\n"
                  "uniform sampler2D s;\n"
                  "void main(){ o = texelFetch(s, ivec2(gl_FragCoord.xy), 0); }\n");
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "s"), 3);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(sample_center());

    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &nt);
    glDeleteFramebuffers(1, &fbo);
}

TEST(multibind_indexed_binding_queries)
{
    /* KHR-GL46.multi_bind.functional_bind_textures regression (compat r103/r104):
     * the ARB_multi_bind indexed getters glGetIntegeri_v(GL_TEXTURE_BINDING_*,
     * unit) returned the default texture object's reserved marker name
     * (0xcafebeef) for every target whose slot holds the default object —
     * the CTS read -889274641 where it expected 0.  The marker must be mapped
     * back to 0 in BOTH query forms. */
    GLuint ta = 0, list[3];
    GLint v = -1;

    glGenTextures(1, &ta);
    glBindTexture(GL_TEXTURE_1D_ARRAY, ta);
    /* no storage needed: the test only exercises binding queries (MGL's
     * glTexImage3D/glTexStorage3D reject GL_TEXTURE_1D_ARRAY — a pre-existing
     * textures-domain deviation, see agents/compat_report.md r104) */
    EXPECT_NO_GL_ERROR();
    glBindTexture(GL_TEXTURE_1D_ARRAY, 0);

    /* bind ta to unit 1, revert unit 2 (nothing bound -> stays default) */
    list[0] = 0; list[1] = ta; list[2] = 0;
    glBindTextures(0, 3, list);
    EXPECT_NO_GL_ERROR();

    /* every target of every touched unit must report a NAME or 0, never the
     * 0xcafebeef marker (checked the way the CTS checks it: all targets) */
    static const GLenum pnames[] = {
        GL_TEXTURE_BINDING_1D, GL_TEXTURE_BINDING_2D, GL_TEXTURE_BINDING_3D,
        GL_TEXTURE_BINDING_1D_ARRAY, GL_TEXTURE_BINDING_2D_ARRAY,
        GL_TEXTURE_BINDING_RECTANGLE, GL_TEXTURE_BINDING_BUFFER,
        GL_TEXTURE_BINDING_CUBE_MAP, GL_TEXTURE_BINDING_CUBE_MAP_ARRAY,
        GL_TEXTURE_BINDING_2D_MULTISAMPLE, GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY,
    };
    for (int u = 0; u < 3; u++)
        for (size_t k = 0; k < sizeof pnames / sizeof pnames[0]; k++)
        {
            v = -1;
            glGetIntegeri_v(pnames[k], (GLuint)u, &v);
            EXPECT_NO_GL_ERROR();
            EXPECT_TRUE(v == 0 || (v > 0 && v < 0x40000000));
        }

    glGetIntegeri_v(GL_TEXTURE_BINDING_1D_ARRAY, 1, &v);
    EXPECT_EQ_INT(v, (int)ta);
    glGetIntegeri_v(GL_TEXTURE_BINDING_1D_ARRAY, 0, &v);
    EXPECT_EQ_INT(v, 0);

    /* non-indexed form on the same unit agrees */
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &v);
    EXPECT_EQ_INT(v, (int)ta);
    glActiveTexture(GL_TEXTURE0);
    EXPECT_NO_GL_ERROR();

    glBindTextures(1, 1, (const GLuint[]){ 0 });
    glGetIntegeri_v(GL_TEXTURE_BINDING_1D_ARRAY, 1, &v);
    EXPECT_EQ_INT(v, 0);
    glDeleteTextures(1, &ta);
}

int main(void)
{
    RUN(default_texture_binds_and_samples);
    RUN(bind_texture_unit_zero_reverts_to_default);
    RUN(multibind_indexed_binding_queries);
    return mgl_test_summary();
}
