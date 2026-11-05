#include "mgl_test.h"

/* dEQP-GL45-ES31.functional.compute.basic.atomic_counter_*: the compute shader
 * declares "layout(binding = 0, offset = 0) uniform atomic_uint u_count;" —
 * the atomic-counter offset qualifier (GLSL 4.4.6), which must NOT be poisoned
 * by the ARB_enhanced_layouts member-offset rejection (mgl_needs_std_offset). */

static const char *cs =
    "#version 450\n"
    "layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(binding = 0) buffer Output {\n"
    "    uint values[1];\n"
    "} sb_out;\n"
    "\n"
    "layout(binding = 0, offset = 0) uniform atomic_uint u_count;\n"
    "\n"
    "void main (void) {\n"
    "    uint localSize  = gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z;\n"
    "    uint globalNdx  = gl_NumWorkGroups.x*gl_NumWorkGroups.y*gl_WorkGroupID.z + gl_NumWorkGroups.x*gl_WorkGroupID.y + gl_WorkGroupID.x;\n"
    "    uint globalOffs = localSize*globalNdx;\n"
    "    uint localOffs  = gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_LocalInvocationID.z + gl_WorkGroupSize.x*gl_LocalInvocationID.y + gl_LocalInvocationID.x;\n"
    "\n"
    "    uint oldVal = atomicCounterIncrement(u_count);\n"
    "    sb_out.values[globalOffs+localOffs] = oldVal;\n"
    "}\n";

TEST(atomic_counter_offset_qualifier_compiles)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    const char *p = cs;
    glShaderSource(s, 1, &p, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    char log[512] = "";
    glGetShaderInfoLog(s, sizeof log, NULL, log);
    printf("  compile=%d log: %s", ok, log);
    EXPECT_EQ_INT(ok, 1);
    glDeleteShader(s);
}

/* a member offset/align on a default-packed buffer block must still be
 * rejected (the poison path must keep working) */
TEST(member_offset_default_block_still_rejected)
{
    const char *cs2 =
        "#version 450\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 0) buffer B { layout(offset = 8) float a; } b;\n"
        "void main() { b.a = 1.0; }\n";
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    const char *p = cs2;
    glShaderSource(s, 1, &p, NULL);
    glCompileShader(s);
    GLint ok = 2;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    printf("  compile=%d (want 0)\n", ok);
    EXPECT_EQ_INT(ok, 0);
    glDeleteShader(s);
}

int main(void)
{
    RUN(atomic_counter_offset_qualifier_compiles);
    RUN(member_offset_default_block_still_rejected);
    return mgl_test_summary();
}
