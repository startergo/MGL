/* test_compat_imgalias — dEQP ES31 synchronization.{in,inter}_invocation.
 * image_atomic_alias_{write,overwrite} repro: TWO image units (1 and 2) bound to
 * the SAME r32i texture level.  GL 4.6 §8.26 makes an image unit a view of the
 * texture, so imageAtomic* through either name must address the same memory:
 * the alias_write case exchanges through u_image1 and reads back through
 * u_image0; alias_overwrite first adds through u_image0.  Before the compat
 * r100 shared sidecar each unit had its own zeroed sidecar, so every read but
 * group 0 returned the wrong value.  Also checks the final texture contents
 * (the read exchanges 123 in) and a two-dispatch variant where unit 2 adopts
 * unit 1's dirty sidecar with a glMemoryBarrier in between.
 */
#include "mgl_test.h"

static const char *kCs =
    "#version 450\n"
    "layout (local_size_x = 8, local_size_y = 8) in;\n"
    "layout(r32i, binding = 1) coherent uniform highp iimage2D u_image0;\n"
    "layout(r32i, binding = 2) coherent uniform highp iimage2D u_image1;\n"
    "layout(binding = 0, std430) buffer Output { highp int values[]; } sb_result;\n"
    "void main (void) {\n"
    "    ivec2 c = ivec2(gl_GlobalInvocationID.xy);\n"
    "    int resultNdx = int(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x\n"
    "                       + gl_GlobalInvocationID.x);\n"
    "    int groupNdx = int(gl_NumWorkGroups.x * gl_WorkGroupID.y + gl_WorkGroupID.x);\n"
    "    bool allOk = true;\n"
    "%s"                                   /* the overwrite phase (456 add), optional */
    "    imageAtomicExchange(u_image1, c, groupNdx);\n"
    "    memoryBarrierImage();\n"
    "    barrier();\n"
    "    allOk = allOk && (imageAtomicExchange(u_image0, c, 123) == groupNdx);\n"
    "    sb_result.values[resultNdx] = allOk ? 1 : 0;\n"
    "}\n";

/* writer for the two-dispatch case: unit 1 only */
static const char *kCsW =
    "#version 450\n"
    "layout (local_size_x = 8, local_size_y = 8) in;\n"
    "layout(r32i, binding = 1) coherent uniform highp iimage2D u_image0;\n"
    "void main (void) {\n"
    "    imageAtomicExchange(u_image0, ivec2(gl_GlobalInvocationID.xy),\n"
    "                        int(gl_GlobalInvocationID.y * 64u + gl_GlobalInvocationID.x) + 1000);\n"
    "}\n";

/* reader for the two-dispatch case: unit 2 only */
static const char *kCsR =
    "#version 450\n"
    "layout (local_size_x = 8, local_size_y = 8) in;\n"
    "layout(r32i, binding = 2) coherent uniform highp iimage2D u_image1;\n"
    "layout(binding = 0, std430) buffer Output { highp int values[]; } sb_result;\n"
    "void main (void) {\n"
    "    int resultNdx = int(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x\n"
    "                       + gl_GlobalInvocationID.x);\n"
    "    int v = imageAtomicExchange(u_image1, ivec2(gl_GlobalInvocationID.xy), 7);\n"
    "    sb_result.values[resultNdx] = (v == resultNdx + 1000) ? 1 : 0;\n"
    "}\n";

static int run_alias(int overwrite, const char *tag)
{
    char src[4096];
    GLuint prog = 0, cs = 0, img = 0, ssbo = 0;
    GLint res[64 * 64], tex[64 * 64];
    int fails = 0;

    snprintf(src, sizeof src, kCs,
             overwrite ? "    imageAtomicAdd(u_image0, c, 456);\n"
                         "    memoryBarrierImage();\n"
                         "    barrier();\n" : "");

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

    glGenTextures(1, &img);
    glBindTexture(GL_TEXTURE_2D, img);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, 64, 64);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    {
        static GLint zero[64 * 64];
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 64, GL_RED_INTEGER, GL_INT, zero);
    }
    /* the two ALIASING units */
    glBindImageTexture(1, img, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(2, img, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof res, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glUseProgram(prog);
    glDispatchCompute(8, 8, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof res, res);
    for (int i = 0; i < 64 * 64; i++)
        if (res[i] != 1)
        {
            if (fails < 4)
                MGL_FAIL("%s: result[%d] == %d (expected 1)", tag, i, res[i]);
            fails++;
        }

    /* every texel now holds 123 (the read exchange) */
    glBindTexture(GL_TEXTURE_2D, img);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_INT, tex);
    for (int i = 0; i < 64 * 64 && fails < 8; i++)
        if (tex[i] != 123)
        {
            MGL_FAIL("%s: texel %d == %d (expected 123)", tag, i, tex[i]);
            fails++;
        }

    glDeleteProgram(prog);
    glDeleteShader(cs);
    glDeleteTextures(1, &img);
    glDeleteBuffers(1, &ssbo);
    while (glGetError() != GL_NO_ERROR) {}
    return fails;
}

/* dispatch A writes through unit 1, glMemoryBarrier, dispatch B reads through
 * unit 2 (which adopts unit 1's dirty-then-resolved sidecar) */
static int run_two_dispatch(const char *tag)
{
    GLuint pw = 0, csw = 0, pr = 0, csr = 0, img = 0, ssbo = 0;
    GLint res[64 * 64];
    int fails = 0;
    GLint ok;

    csw = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(csw, 1, (const GLchar *[]){ kCsW }, NULL);
    glCompileShader(csw);
    glGetShaderiv(csw, GL_COMPILE_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: writer CS compile failed", tag); return 1; }
    pw = glCreateProgram();
    glAttachShader(pw, csw);
    glLinkProgram(pw);
    glGetProgramiv(pw, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: writer link failed", tag); return 1; }

    csr = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(csr, 1, (const GLchar *[]){ kCsR }, NULL);
    glCompileShader(csr);
    glGetShaderiv(csr, GL_COMPILE_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: reader CS compile failed", tag); return 1; }
    pr = glCreateProgram();
    glAttachShader(pr, csr);
    glLinkProgram(pr);
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("%s: reader link failed", tag); return 1; }

    glGenTextures(1, &img);
    glBindTexture(GL_TEXTURE_2D, img);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, 64, 64);
    glBindImageTexture(1, img, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(2, img, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof res, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glUseProgram(pw);
    glDispatchCompute(8, 8, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glUseProgram(pr);
    glDispatchCompute(8, 8, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof res, res);
    for (int i = 0; i < 64 * 64; i++)
        if (res[i] != 1)
        {
            if (fails < 4)
                MGL_FAIL("%s: result[%d] == %d (expected 1)", tag, i, res[i]);
            fails++;
        }

    glDeleteProgram(pw); glDeleteProgram(pr);
    glDeleteShader(csw); glDeleteShader(csr);
    glDeleteTextures(1, &img);
    glDeleteBuffers(1, &ssbo);
    while (glGetError() != GL_NO_ERROR) {}
    return fails;
}

TEST(compat_imgalias_write) { EXPECT_EQ_INT(run_alias(0, "alias_write"), 0); }
TEST(compat_imgalias_overwrite) { EXPECT_EQ_INT(run_alias(1, "alias_overwrite"), 0); }
TEST(compat_imgalias_two_dispatch) { EXPECT_EQ_INT(run_two_dispatch("two_dispatch"), 0); }

int main(void)
{
    RUN(compat_imgalias_write);
    RUN(compat_imgalias_overwrite);
    RUN(compat_imgalias_two_dispatch);
    return mgl_test_summary();
}
