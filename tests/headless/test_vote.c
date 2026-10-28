/* test_vote — GL_ARB_shader_group_vote (KHR-GL46.shader_group_vote)
 *
 * The CTS specialises its compute shaders for a 4.6 core context to the
 * extension-less spellings (allInvocations/anyInvocation/allInvocationsEqual).
 * cts_glsl_compat.c renames them to the ARB forms glslang knows, glslang
 * emits OpSubgroupAll/Any/AllEqualKHR, and tess_spv_lower.c rewrites those to
 * the SPIR-V 1.3 OpGroupNonUniform* that SPIRV-Cross lowers to
 * simd_all/simd_any/spvSubgroupAllEqual.  End-to-end: dispatch + image
 * readback of every CTS predicate shape (uniform true/false, mixed parity,
 * workgroup-uniform). */
#include "mgl_test.h"

static const char *cs_avail = R"(#version 450
layout(rgba8, binding = 2) writeonly uniform highp image2D destImage;
layout(local_size_x = 16, local_size_y = 16) in;
void main (void)
{
    vec4 outColor = vec4(0.0);
    outColor.r = allInvocations(true) ? 1.0 : 0.0;
    outColor.g = anyInvocation(true) ? 1.0 : 0.0;
    outColor.b = allInvocationsEqual(true) ? 1.0 : 0.0;
    imageStore(destImage, ivec2(gl_GlobalInvocationID.xy), outColor);
}
)";

static const char *cs_all = R"(#version 450
layout(rgba8, binding = 2) writeonly uniform highp image2D destImage;
layout(local_size_x = 16, local_size_y = 16) in;
void main (void)
{
    bool result = allInvocations(true);
    vec4 outColor = vec4(vec3(result ? 1.0 : 0.0), 1.0);
    imageStore(destImage, ivec2(gl_GlobalInvocationID.xy), outColor);
}
)";

static const char *cs_any = R"(#version 450
layout(rgba8, binding = 2) writeonly uniform highp image2D destImage;
layout(local_size_x = 16, local_size_y = 16) in;
void main (void)
{
    bool result = anyInvocation(false);
    vec4 outColor = vec4(vec3(result ? 1.0 : 0.0), 1.0);
    imageStore(destImage, ivec2(gl_GlobalInvocationID.xy), outColor);
}
)";

static const char *cs_v1 = R"(#version 450
layout(rgba8, binding = 2) writeonly uniform highp image2D destImage;
layout(local_size_x = 16, local_size_y = 16) in;
void main (void)
{
    bool result = allInvocations((gl_LocalInvocationIndex % 2u) == 1u) && anyInvocation((gl_LocalInvocationIndex % 2u) == 0u) && anyInvocation((gl_LocalInvocationIndex % 2u) == 1u);
    vec4 outColor = vec4(vec3(result ? 1.0 : 0.0), 1.0);
    imageStore(destImage, ivec2(gl_GlobalInvocationID.xy), outColor);
}
)";

static const char *cs_v2 = R"(#version 450
layout(rgba8, binding = 2) writeonly uniform highp image2D destImage;
layout(local_size_x = 16, local_size_y = 16) in;
void main (void)
{
    bool result = anyInvocation(gl_LocalInvocationIndex < 256u);
    vec4 outColor = vec4(vec3(result ? 1.0 : 0.0), 1.0);
    imageStore(destImage, ivec2(gl_GlobalInvocationID.xy), outColor);
}
)";

static const char *cs_v3 = R"(#version 450
layout(rgba8, binding = 2) writeonly uniform highp image2D destImage;
layout(local_size_x = 16, local_size_y = 16) in;
void main (void)
{
    bool result = allInvocationsEqual(gl_WorkGroupID.x == 0u);
    vec4 outColor = vec4(vec3(result ? 1.0 : 0.0), 1.0);
    imageStore(destImage, ivec2(gl_GlobalInvocationID.xy), outColor);
}
)";

/* run `cs`, dispatch 1x1, read the 16x16 image back; every texel must be rgb */
static int run_vote(const char *cs, int r, int g, int b)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log); printf("  [vote] compile fail: %.400s\n", log); glDeleteShader(s); return 0; }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [vote] link fail: %.400s\n", log); glDeleteShader(s); glDeleteProgram(p); return 0; }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    EXPECT_NO_GL_ERROR();

    glUseProgram(p);
    glBindImageTexture(2, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    EXPECT_NO_GL_ERROR();
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    unsigned char px[16 * 16 * 4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    int bad = 0;
    for (int i = 0; i < 16 * 16; i++)
    {
        if (px[i*4+0] != r || px[i*4+1] != g || px[i*4+2] != b)
        {
            if (bad < 3) printf("  [vote] texel %d = %d,%d,%d,%d (want %d,%d,%d)\n",
                                i, px[i*4], px[i*4+1], px[i*4+2], px[i*4+3], r, g, b);
            bad++;
        }
    }
    glDeleteTextures(1, &tex);
    glDeleteShader(s);
    glDeleteProgram(p);
    return bad == 0;
}

TEST(availability)      { EXPECT_TRUE(run_vote(cs_avail, 255, 255, 255)); }
TEST(all_invocations)   { EXPECT_TRUE(run_vote(cs_all, 255, 255, 255)); }
TEST(any_invocation)    { EXPECT_TRUE(run_vote(cs_any, 0, 0, 0)); }
TEST(vars_expr1_black)  { EXPECT_TRUE(run_vote(cs_v1, 0, 0, 0)); }
TEST(vars_expr2_white)  { EXPECT_TRUE(run_vote(cs_v2, 255, 255, 255)); }
TEST(vars_expr3_white)  { EXPECT_TRUE(run_vote(cs_v3, 255, 255, 255)); }

int main(void)
{
    RUN(availability);
    RUN(all_invocations);
    RUN(any_invocation);
    RUN(vars_expr1_black);
    RUN(vars_expr2_white);
    RUN(vars_expr3_white);
    return 0;
}
