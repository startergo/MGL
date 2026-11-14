/* test_compat_rsvga_sm3_depth — r109 Task B (rsvga upstream-candidate bug #5):
 * an SM1-3-translated vertex shader (D3D9 fixed-function translation delivered
 * via GL_ARB_gl_spirv) draws NOTHING whenever a depth attachment is bound to
 * the FBO — even with GL_DEPTH_TEST disabled.  Without the attachment the
 * same draw covers 15/16 pixels of a 4x4 target.  The SPIR-V pair below is
 * rsvga's failing server pair (vs_3f5828e0ecfdb701 / ps_c13e35429294b09f):
 * the VS reads a float4[308] uniform block (set 0 binding 0; c303..c306 =
 * matrix rows, c307 = colour) and outputs locations 0/1/2/19 + position;
 * the twin data draws a quad.
 */
#include "mgl_test.h"

static const unsigned kVS[464] = {
    0x07230203u, 0x00010300u, 0x00000000u, 0x0000004du, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x000c000fu, 0x00000000u, 0x00000009u, 0x6e69616du, 0x00000000u, 0x00000015u, 0x00000017u, 0x00000019u,
    0x0000001bu, 0x00000022u, 0x00000043u, 0x00000045u, 0x00040047u, 0x0000000cu, 0x00000006u, 0x00000010u,
    0x00030047u, 0x0000000du, 0x00000002u, 0x00050048u, 0x0000000du, 0x00000000u, 0x00000023u, 0x00000000u,
    0x00040047u, 0x0000000fu, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000000fu, 0x00000021u, 0x00000000u,
    0x00040047u, 0x00000015u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x00000017u, 0x0000000bu, 0x00000000u,
    0x00040047u, 0x00000019u, 0x0000001eu, 0x00000013u, 0x00040047u, 0x0000001bu, 0x0000001eu, 0x00000002u,
    0x00040047u, 0x00000022u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x00000043u, 0x0000001eu, 0x00000001u,
    0x00040047u, 0x00000045u, 0x0000001eu, 0x00000002u, 0x00030016u, 0x00000002u, 0x00000020u, 0x00040017u,
    0x00000003u, 0x00000002u, 0x00000004u, 0x00040015u, 0x00000004u, 0x00000020u, 0x00000001u, 0x00040017u,
    0x00000005u, 0x00000004u, 0x00000004u, 0x00020014u, 0x00000006u, 0x00040017u, 0x00000007u, 0x00000006u,
    0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u, 0x0000000bu,
    0x00000134u, 0x0004001cu, 0x0000000cu, 0x00000003u, 0x0000000bu, 0x0003001eu, 0x0000000du, 0x0000000cu,
    0x00040020u, 0x0000000eu, 0x00000002u, 0x0000000du, 0x0004003bu, 0x0000000eu, 0x0000000fu, 0x00000002u,
    0x00040020u, 0x00000010u, 0x00000007u, 0x00000003u, 0x0004002bu, 0x00000002u, 0x00000012u, 0x00000000u,
    0x0007002cu, 0x00000003u, 0x00000013u, 0x00000012u, 0x00000012u, 0x00000012u, 0x00000012u, 0x00040020u,
    0x00000014u, 0x00000003u, 0x00000003u, 0x0004003bu, 0x00000014u, 0x00000015u, 0x00000003u, 0x0004003bu,
    0x00000014u, 0x00000017u, 0x00000003u, 0x0004003bu, 0x00000014u, 0x00000019u, 0x00000003u, 0x0004003bu,
    0x00000014u, 0x0000001bu, 0x00000003u, 0x0004002bu, 0x00000008u, 0x0000001cu, 0x00000133u, 0x0004002bu,
    0x00000008u, 0x0000001du, 0x00000000u, 0x00040020u, 0x0000001eu, 0x00000002u, 0x00000003u, 0x00040020u,
    0x00000021u, 0x00000001u, 0x00000003u, 0x0004003bu, 0x00000021u, 0x00000022u, 0x00000001u, 0x0004002bu,
    0x00000008u, 0x00000024u, 0x0000012fu, 0x0004002bu, 0x00000008u, 0x0000002cu, 0x00000130u, 0x0004002bu,
    0x00000008u, 0x00000034u, 0x00000131u, 0x0004002bu, 0x00000008u, 0x0000003cu, 0x00000132u, 0x0004003bu,
    0x00000021u, 0x00000043u, 0x00000001u, 0x0004003bu, 0x00000021u, 0x00000045u, 0x00000001u, 0x00020013u,
    0x0000004bu, 0x00030021u, 0x0000004cu, 0x0000004bu, 0x00050036u, 0x0000004bu, 0x00000009u, 0x00000000u,
    0x0000004cu, 0x000200f8u, 0x0000000au, 0x0004003bu, 0x00000010u, 0x00000011u, 0x00000007u, 0x0004003bu,
    0x00000010u, 0x00000016u, 0x00000007u, 0x0004003bu, 0x00000010u, 0x00000018u, 0x00000007u, 0x0004003bu,
    0x00000010u, 0x0000001au, 0x00000007u, 0x0003003eu, 0x00000011u, 0x00000013u, 0x0003003eu, 0x00000016u,
    0x00000013u, 0x0003003eu, 0x00000018u, 0x00000013u, 0x0003003eu, 0x0000001au, 0x00000013u, 0x00060041u,
    0x0000001eu, 0x0000001fu, 0x0000000fu, 0x0000001du, 0x0000001cu, 0x0004003du, 0x00000003u, 0x00000020u,
    0x0000001fu, 0x0003003eu, 0x00000011u, 0x00000020u, 0x0004003du, 0x00000003u, 0x00000023u, 0x00000022u,
    0x00060041u, 0x0000001eu, 0x00000025u, 0x0000000fu, 0x0000001du, 0x00000024u, 0x0004003du, 0x00000003u,
    0x00000026u, 0x00000025u, 0x00050094u, 0x00000002u, 0x00000027u, 0x00000023u, 0x00000026u, 0x00070050u,
    0x00000003u, 0x00000028u, 0x00000027u, 0x00000027u, 0x00000027u, 0x00000027u, 0x0004003du, 0x00000003u,
    0x00000029u, 0x00000016u, 0x0009004fu, 0x00000003u, 0x0000002au, 0x00000029u, 0x00000028u, 0x00000004u,
    0x00000001u, 0x00000002u, 0x00000003u, 0x0003003eu, 0x00000016u, 0x0000002au, 0x0004003du, 0x00000003u,
    0x0000002bu, 0x00000022u, 0x00060041u, 0x0000001eu, 0x0000002du, 0x0000000fu, 0x0000001du, 0x0000002cu,
    0x0004003du, 0x00000003u, 0x0000002eu, 0x0000002du, 0x00050094u, 0x00000002u, 0x0000002fu, 0x0000002bu,
    0x0000002eu, 0x00070050u, 0x00000003u, 0x00000030u, 0x0000002fu, 0x0000002fu, 0x0000002fu, 0x0000002fu,
    0x0004003du, 0x00000003u, 0x00000031u, 0x00000016u, 0x0009004fu, 0x00000003u, 0x00000032u, 0x00000031u,
    0x00000030u, 0x00000000u, 0x00000005u, 0x00000002u, 0x00000003u, 0x0003003eu, 0x00000016u, 0x00000032u,
    0x0004003du, 0x00000003u, 0x00000033u, 0x00000022u, 0x00060041u, 0x0000001eu, 0x00000035u, 0x0000000fu,
    0x0000001du, 0x00000034u, 0x0004003du, 0x00000003u, 0x00000036u, 0x00000035u, 0x00050094u, 0x00000002u,
    0x00000037u, 0x00000033u, 0x00000036u, 0x00070050u, 0x00000003u, 0x00000038u, 0x00000037u, 0x00000037u,
    0x00000037u, 0x00000037u, 0x0004003du, 0x00000003u, 0x00000039u, 0x00000016u, 0x0009004fu, 0x00000003u,
    0x0000003au, 0x00000039u, 0x00000038u, 0x00000000u, 0x00000001u, 0x00000006u, 0x00000003u, 0x0003003eu,
    0x00000016u, 0x0000003au, 0x0004003du, 0x00000003u, 0x0000003bu, 0x00000022u, 0x00060041u, 0x0000001eu,
    0x0000003du, 0x0000000fu, 0x0000001du, 0x0000003cu, 0x0004003du, 0x00000003u, 0x0000003eu, 0x0000003du,
    0x00050094u, 0x00000002u, 0x0000003fu, 0x0000003bu, 0x0000003eu, 0x00070050u, 0x00000003u, 0x00000040u,
    0x0000003fu, 0x0000003fu, 0x0000003fu, 0x0000003fu, 0x0004003du, 0x00000003u, 0x00000041u, 0x00000016u,
    0x0009004fu, 0x00000003u, 0x00000042u, 0x00000041u, 0x00000040u, 0x00000000u, 0x00000001u, 0x00000002u,
    0x00000007u, 0x0003003eu, 0x00000016u, 0x00000042u, 0x0004003du, 0x00000003u, 0x00000044u, 0x00000043u,
    0x0003003eu, 0x00000018u, 0x00000044u, 0x0004003du, 0x00000003u, 0x00000046u, 0x00000045u, 0x0003003eu,
    0x0000001au, 0x00000046u, 0x0004003du, 0x00000003u, 0x00000047u, 0x00000011u, 0x0003003eu, 0x00000015u,
    0x00000047u, 0x0004003du, 0x00000003u, 0x00000048u, 0x00000016u, 0x0003003eu, 0x00000017u, 0x00000048u,
    0x0004003du, 0x00000003u, 0x00000049u, 0x00000018u, 0x0003003eu, 0x00000019u, 0x00000049u, 0x0004003du,
    0x00000003u, 0x0000004au, 0x0000001au, 0x0003003eu, 0x0000001bu, 0x0000004au, 0x000100fdu, 0x00010038u,
};
static const unsigned kFS[296] = {
    0x07230203u, 0x00010300u, 0x00000000u, 0x00000031u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000004u, 0x00000009u, 0x6e69616du, 0x00000000u, 0x00000018u, 0x0000001au, 0x00030010u,
    0x00000009u, 0x00000007u, 0x00040047u, 0x0000000cu, 0x00000006u, 0x00000010u, 0x00030047u, 0x0000000du,
    0x00000002u, 0x00050048u, 0x0000000du, 0x00000000u, 0x00000023u, 0x00000000u, 0x00040047u, 0x0000000fu,
    0x00000022u, 0x00000000u, 0x00040047u, 0x0000000fu, 0x00000021u, 0x00000100u, 0x00040047u, 0x00000018u,
    0x0000001eu, 0x00000000u, 0x00040047u, 0x0000001au, 0x0000001eu, 0x00000000u, 0x00030016u, 0x00000002u,
    0x00000020u, 0x00040017u, 0x00000003u, 0x00000002u, 0x00000004u, 0x00040015u, 0x00000004u, 0x00000020u,
    0x00000001u, 0x00040017u, 0x00000005u, 0x00000004u, 0x00000004u, 0x00020014u, 0x00000006u, 0x00040017u,
    0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u, 0x0004002bu,
    0x00000008u, 0x0000000bu, 0x0000001cu, 0x0004001cu, 0x0000000cu, 0x00000003u, 0x0000000bu, 0x0003001eu,
    0x0000000du, 0x0000000cu, 0x00040020u, 0x0000000eu, 0x00000002u, 0x0000000du, 0x0004003bu, 0x0000000eu,
    0x0000000fu, 0x00000002u, 0x00040020u, 0x00000010u, 0x00000007u, 0x00000003u, 0x0004002bu, 0x00000002u,
    0x00000012u, 0x00000000u, 0x0007002cu, 0x00000003u, 0x00000013u, 0x00000012u, 0x00000012u, 0x00000012u,
    0x00000012u, 0x00040020u, 0x00000017u, 0x00000003u, 0x00000003u, 0x0004003bu, 0x00000017u, 0x00000018u,
    0x00000003u, 0x00040020u, 0x00000019u, 0x00000001u, 0x00000003u, 0x0004003bu, 0x00000019u, 0x0000001au,
    0x00000001u, 0x0004002bu, 0x00000002u, 0x0000001du, 0x437f0000u, 0x0004002bu, 0x00000002u, 0x0000001eu,
    0x3f000000u, 0x0004002bu, 0x00000002u, 0x0000001fu, 0x3b808081u, 0x0007002cu, 0x00000003u, 0x00000020u,
    0x0000001du, 0x0000001eu, 0x0000001fu, 0x0000001fu, 0x00020013u, 0x0000002fu, 0x00030021u, 0x00000030u,
    0x0000002fu, 0x00050036u, 0x0000002fu, 0x00000009u, 0x00000000u, 0x00000030u, 0x000200f8u, 0x0000000au,
    0x0004003bu, 0x00000010u, 0x00000011u, 0x00000007u, 0x0004003bu, 0x00000010u, 0x00000014u, 0x00000007u,
    0x0004003bu, 0x00000010u, 0x00000015u, 0x00000007u, 0x0004003bu, 0x00000010u, 0x00000016u, 0x00000007u,
    0x0003003eu, 0x00000011u, 0x00000013u, 0x0003003eu, 0x00000014u, 0x00000013u, 0x0003003eu, 0x00000015u,
    0x00000013u, 0x0003003eu, 0x00000016u, 0x00000013u, 0x0004003du, 0x00000003u, 0x0000001bu, 0x0000001au,
    0x0003003eu, 0x00000011u, 0x0000001bu, 0x0004003du, 0x00000003u, 0x0000001cu, 0x00000011u, 0x0009004fu,
    0x00000003u, 0x00000021u, 0x00000020u, 0x00000020u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00050085u, 0x00000003u, 0x00000022u, 0x0000001cu, 0x00000021u, 0x0003003eu, 0x00000014u, 0x00000022u,
    0x0004003du, 0x00000003u, 0x00000023u, 0x00000014u, 0x0009004fu, 0x00000003u, 0x00000024u, 0x00000020u,
    0x00000020u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00050081u, 0x00000003u, 0x00000025u,
    0x00000023u, 0x00000024u, 0x0003003eu, 0x00000014u, 0x00000025u, 0x0004003du, 0x00000003u, 0x00000026u,
    0x00000014u, 0x0006000cu, 0x00000003u, 0x00000027u, 0x00000001u, 0x0000000au, 0x00000026u, 0x0003003eu,
    0x00000015u, 0x00000027u, 0x0004003du, 0x00000003u, 0x00000028u, 0x00000014u, 0x0004003du, 0x00000003u,
    0x00000029u, 0x00000015u, 0x00050083u, 0x00000003u, 0x0000002au, 0x00000028u, 0x00000029u, 0x0003003eu,
    0x00000014u, 0x0000002au, 0x0004003du, 0x00000003u, 0x0000002bu, 0x00000014u, 0x0009004fu, 0x00000003u,
    0x0000002cu, 0x00000020u, 0x00000020u, 0x00000002u, 0x00000002u, 0x00000002u, 0x00000002u, 0x00050085u,
    0x00000003u, 0x0000002du, 0x0000002bu, 0x0000002cu, 0x0003003eu, 0x00000016u, 0x0000002du, 0x0004003du,
    0x00000003u, 0x0000002eu, 0x00000016u, 0x0003003eu, 0x00000018u, 0x0000002eu, 0x000100fdu, 0x00010038u,
};

static int covered(GLuint fbo)
{
    GLubyte px[16 * 4];
    int n = 0;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px);
    for (int i = 0; i < 16; i++)
        if (px[i * 4] > 128 || px[i * 4 + 1] > 128) n++;
    return n;
}

static GLuint ubo_fs_out;
static GLuint mksetup(GLuint *vbo_out, GLuint *ubo_out)
{
    /* the twin data: positions (±1,±1,0.5,1) */
    static const float pos[4][4] = {
        { -1, -1, 0.5f, 1 }, { 1, -1, 0.5f, 1 }, { 1, 1, 0.5f, 1 }, { -1, 1, 0.5f, 1 },
    };
    float c[308][4] = { 0 };
    c[303][0] = 2; c[304][1] = -2; c[305][2] = 1;
    c[306][0] = -1; c[306][1] = 1; c[306][3] = 1;
    c[307][0] = 1;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *) 0);

    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof c, c, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

    /* the fragment stage's block sits at SPIR-V binding 256 (D3D9 constant
     * space); the twin data leaves it zeroed */
    static const float z28[28][4] = { 0 };
    GLuint ubo2;
    glGenBuffers(1, &ubo2);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo2);
    glBufferData(GL_UNIFORM_BUFFER, sizeof z28, z28, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);   /* keep the VS block's buffer current */

    /* the fragment block sits at SPIR-V binding 256 (D3D9 constant space);
     * rsvga binds it with glBindBufferBase — MGL accepts indices beyond the
     * GL table through the overflow table (buffers.c) */
    while (glGetError()) {}
    glBindBufferBase(GL_UNIFORM_BUFFER, 256, ubo2);
    printf("  bind 256 err=0x%x\n", glGetError());
    *vbo_out = vbo;
    *ubo_out = ubo;
    ubo_fs_out = ubo2;
    return vao;
}

static GLuint mkprog(void)
{
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderBinary(1, &vs, GL_SHADER_BINARY_FORMAT_SPIR_V, kVS, sizeof kVS);
    glSpecializeShader(vs, "main", 0, NULL, NULL);
    glShaderBinary(1, &fs, GL_SHADER_BINARY_FORMAT_SPIR_V, kFS, sizeof kFS);
    glSpecializeShader(fs, "main", 0, NULL, NULL);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[512] = { 0 };
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link: %s", log);
    }
    return p;
}

TEST(rsvga_sm3_draws_with_depth_attachment)
{
    GLuint vbo, ubo, vao = mksetup(&vbo, &ubo);
    GLuint prog = mkprog();
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);

    /* bind both UBOs through the program's block indices (the FS block's
     * SPIR-V binding is 256, beyond the GL table — bind by index) */
    GLint nblocks = 0;
    glGetProgramiv(prog, GL_ACTIVE_UNIFORM_BLOCKS, &nblocks);
    printf("  active uniform blocks: %d\n", nblocks);
    for (int b = 0; b < nblocks; b++)
    {
        char name[64] = { 0 };
        GLint vs_stage = 0, binding = -1;
        glGetActiveUniformBlockiv(prog, b, GL_UNIFORM_BLOCK_NAME_LENGTH, &vs_stage);
        glGetActiveUniformBlockName(prog, b, sizeof name, NULL, name);
        glGetActiveUniformBlockiv(prog, b, GL_UNIFORM_BLOCK_BINDING, &binding);
        printf("  block %d (%s) binding=%d\n", b, name, binding);
    }

    GLuint fbo, col;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &col);
    glBindTexture(GL_TEXTURE_2D, col);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);

    /* 1. no depth attachment */
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    int n0 = covered(fbo);
    printf("  no depth: covered %d/16\n", n0);
    EXPECT_TRUE(n0 >= 13);          /* the quad rasterizes (13-15 px depending on the matrix edge) */

    /* 2. GL_DEPTH_COMPONENT16 attachment, depth test OFF */
    GLuint d16;
    glGenTextures(1, &d16);
    glBindTexture(GL_TEXTURE_2D, d16);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 4, 4, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, d16, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    int n1 = covered(fbo);
    printf("  D16:      covered %d/16\n", n1);
    EXPECT_EQ_INT(n1, n0);           /* RED pre-fix: 0 — the rsvga bug */

    /* 3. GL_DEPTH24_STENCIL8 attachment, depth test OFF */
    GLuint ds;
    glGenTextures(1, &ds);
    glBindTexture(GL_TEXTURE_2D, ds);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, 4, 4, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, ds, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    int n2 = covered(fbo);
    printf("  D24S8:    covered %d/16\n", n2);
    EXPECT_EQ_INT(n2, n0);           /* RED pre-fix: 0 — the rsvga bug */

    glDeleteTextures(1, &ds);
    glDeleteTextures(1, &d16);
    glDeleteTextures(1, &col);
    glDeleteFramebuffers(1, &fbo);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &ubo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}
int main(void){ RUN(rsvga_sm3_draws_with_depth_attachment); return mgl_test_summary(); }
