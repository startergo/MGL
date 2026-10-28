/* CTS KHR-GL46.gl_spirv.spirv_validation_capabilities compute stage (the
 * image/sampler type zoo, transcribed verbatim from gl4cGlSpirvTests.cpp
 * SpirvValidationCapabilitiesTest::init — an earlier draft added calls that
 * are not in the CTS source, e.g. texelFetch on sampler2DRect, which glslang
 * rejects and which the real test never makes).  The CTS assembles this GLSL
 * to SPIR-V and links a compute program; the headless form checks the same
 * thing through the GLSL path: compile + link must succeed (10 active image
 * uniforms in one stage — needs GL_MAX_IMAGE_UNITS >= 10; writable MS images
 * are emulated, documented gap: the sample index is dropped). */
#include "mgl_test.h"

static const char *cs =
    "#version 450\n"
    "\n"
    "layout (local_size_x = 1, local_size_y = 2, local_size_z = 1) in;\n"
    "\n"
    "layout (location = 0, rgba8) uniform image2DMS img0;\n"
    "layout (location = 1, rgba8) uniform image2DMSArray img1;\n"
    "layout (location = 2, rgba8) uniform image2DRect img2;\n"
    "layout (location = 3, rgba8) uniform imageCube img3;\n"
    "layout (location = 4, rgba8) uniform imageCubeArray img4;\n"
    "layout (location = 5, rgba8) uniform imageBuffer img5;\n"
    "layout (location = 6, rgba8) uniform image2D img6;\n"
    "layout (location = 7, rgba8) uniform image1D img7;\n"
    "layout (location = 8) uniform writeonly image1D img8;\n"
    "layout (location = 9, rg32f) uniform image1D img9;\n"
    "layout (location = 10) uniform sampler2DRect img10;\n"
    "layout (location = 11) uniform samplerCubeArray img11;\n"
    "layout (location = 12) uniform samplerBuffer img12;\n"
    "layout (location = 13) uniform sampler1D img13;\n"
    "layout (location = 14) uniform sampler2D img14;\n"
    "\n"
    "layout (binding = 0) uniform atomic_uint atCounter;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    ivec2 size = imageSize(img6);\n"
    "    ivec3 point = ivec3(gl_GlobalInvocationID);\n"
    "    imageStore(img0, point.xy, 0, vec4(0));\n"
    "    imageStore(img1, point, 0, vec4(0));\n"
    "    imageStore(img2, point.xy, vec4(0));\n"
    "    imageStore(img3, point, vec4(0));\n"
    "    imageStore(img4, point, vec4(0));\n"
    "    imageStore(img5, point.x, vec4(0));\n"
    "    imageStore(img6, point.xy, vec4(0));\n"
    "    imageStore(img7, point.x, vec4(0));\n"
    "    imageStore(img8, point.x, vec4(0));\n"
    "\n"
    "    vec3 coord = vec3(0);\n"
    "    ivec2 offset = ivec2(gl_GlobalInvocationID.xy);\n"
    "    vec4 color;\n"
    "    color = textureGather(img10, coord.xy);\n"
    "    color = textureGather(img11, vec4(0));\n"
    "    color = texelFetch(img12, point.x);\n"
    "    color = textureGatherOffset(img14, coord.xy, offset);\n"
    "    memoryBarrier();\n"
    "}\n";

TEST(caps_zoo_compile_link)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) {
        char log[4096] = "";
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        MGL_FAIL("compile failed: %.2000s", log);
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) {
        char log[8192] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link failed: %.4000s", log);
    }
    glDeleteProgram(p);
    glDeleteShader(s);
}

int main(void) {
    RUN(caps_zoo_compile_link);
    return mgl_test_summary();
}
