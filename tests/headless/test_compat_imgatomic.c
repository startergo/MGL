/* test_compat_imgatomic — dEQP ES31 compute.basic.image_atomic_op_* repro:
 * a compute shader does imageStore(0) then imageAtomicAdd(value) on an r32ui
 * image (per work group, coordinates = gl_WorkGroupID), with barrier() between
 * (local_size 1 and 8 = the two failing dEQP cases).  The result is read back
 * BOTH through an FBO glReadPixels (the dEQP path: "Comparison failed for
 * pixel 0") and glGetTexImage, and must equal the stored value everywhere.
 */
#include "mgl_test.h"

static const char *kCs =
    "#version 450\n"
    "layout (local_size_x = %d) in;\n"
    "layout(r32ui, binding = 1) uniform highp uimage2D u_dstImg;\n"
    "buffer Input {\n"
    "    uint values[32768];\n"
    "} sb_in;\n"
    "void main (void) {\n"
    "    uint stride = gl_NumWorkGroups.x*gl_WorkGroupSize.x;\n"
    "    uint value  = sb_in.values[gl_GlobalInvocationID.y*stride + gl_GlobalInvocationID.x];\n"
    "    if (gl_LocalInvocationIndex == 0u)\n"
    "        imageStore(u_dstImg, ivec2(gl_WorkGroupID.xy), uvec4(0));\n"
    "    barrier();\n"
    "    imageAtomicAdd(u_dstImg, ivec2(gl_WorkGroupID.xy), value);\n"
    "}\n";

static GLuint values[32768];

static int run_case(int local_size, const char *tag)
{
    char src[2048];
    GLuint prog = 0, cs = 0, img = 0, ssbo = 0, fbo = 0;
    GLuint back[64 * 64];
    int fails = 0;

    snprintf(src, sizeof src, kCs, local_size);

    cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, (const GLchar *[]){ src }, NULL);
    glCompileShader(cs);
    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: CS compile failed", tag); return 1; }
    prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: link failed", tag); return 1; }

    for (int i = 0; i < 32768; i++)
        values[i] = (uint32_t)(0x9E3779B9u * (uint32_t)i + 0x12345678u);   /* full 32-bit */

    glGenTextures(1, &img);
    glBindTexture(GL_TEXTURE_2D, img);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 64, 64);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindImageTexture(1, img, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof values, values, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glUseProgram(prog);
    glDispatchCompute(64, 64, 1);
    if (!getenv("IMGAT_NOBARRIER"))
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT);

    /* readback path A: attach to an FBO and glReadPixels (dEQP's path) */
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, img, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    for (int y = 0; y < 64; y++)
    {
        GLuint rgba[64 * 4];
        glReadPixels(0, y, 64, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, rgba);
        for (int x = 0; x < 64; x++)
        {
            uint32_t want = 0;
            for (int o = 0; o < local_size; o++)
                want += values[y * 64 * local_size + x * local_size + o];
            if (rgba[x * 4] != want)
            {
                if (fails < 8)
                    MGL_FAIL("%s: FBO readback pixel (%d,%d) == %u, expected %u",
                             tag, x, y, rgba[x * 4], want);
                if (getenv("IMGAT_DEBUG") && fails < 12)
                    fprintf(stderr, "DBG %s px(%d,%d) got=%u want=%u idx0=%d\n",
                            tag, x, y, rgba[x * 4], want, y*64*local_size + x*local_size);
                fails++;
            }
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    /* readback path B: glGetTexImage */
    glBindTexture(GL_TEXTURE_2D, img);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, back);
    for (int y = 0; y < 64 && fails < 4; y++)
        for (int x = 0; x < 64; x++)
        {
            uint32_t want = 0;
            for (int o = 0; o < local_size; o++)
                want += values[y * 64 * local_size + x * local_size + o];
            if (back[y * 64 + x] != want)
            {
                MGL_FAIL("%s: GetTexImage pixel (%d,%d) == %u, expected %u",
                         tag, x, y, back[y * 64 + x], want);
                fails++;
                break;
            }
        }

    glDeleteProgram(prog);
    glDeleteShader(cs);
    glDeleteTextures(1, &img);
    glDeleteBuffers(1, &ssbo);
    glDeleteFramebuffers(1, &fbo);
    while (glGetError() != GL_NO_ERROR) {}
    return fails;
}

TEST(compat_imgatomic_local1) { EXPECT_EQ_INT(run_case(1, "local1"), 0); }
TEST(compat_imgatomic_local8) { EXPECT_EQ_INT(run_case(8, "local8"), 0); }

int main(void)
{
    RUN(compat_imgatomic_local1);
    RUN(compat_imgatomic_local8);
    return mgl_test_summary();
}
