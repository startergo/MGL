#include <stdio.h>
#include <string.h>
#include "glcorearb.h"
#include "mgl_test.h"

/* gpu_shader5 texture_gather_offset_depth_repeat: textureGatherOffset with a
 * sampler2DShadow (refZ arg) and a RUNTIME offset in the vertex stage. */

static const char *kVsDepth =
    "#version 460 core\n"
    "precision highp float;\n"
    "precision highp sampler2DShadow;\n"
    "uniform sampler2DShadow sampler;\n"
    "in ivec2 offsets;\n"
    "in vec2 texCoords;\n"
    "flat out vec4 without_offset_0;\n"
    "flat out vec4 with_offset_0;\n"
    "flat out vec4 without_offset_1;\n"
    "flat out vec4 with_offset_1;\n"
    "void main()\n"
    "{\n"
    "    without_offset_0 = textureGather      (sampler, texCoords, 0.5);\n"
    "    with_offset_0    = textureGatherOffset(sampler, texCoords, 0.5, offsets);\n"
    "    without_offset_1 = textureGather      (sampler, texCoords, 0.5, 1);\n"
    "    with_offset_1    = textureGatherOffset(sampler, texCoords, 0.5, offsets, 1);\n"
    "    gl_Position = vec4(texCoords * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static const char *kVsColorClamp =
    "#version 460 core\n"
    "precision highp float;\n"
    "precision highp isampler2D;\n"
    "uniform isampler2D sampler;\n"
    "in ivec2 offsets;\n"
    "in vec2 texCoords;\n"
    "flat out ivec4 without_offset_0;\n"
    "flat out ivec4 with_offset_0;\n"
    "void main()\n"
    "{\n"
    "    without_offset_0 = textureGather      (sampler, texCoords, 0);\n"
    "    with_offset_0    = textureGatherOffset(sampler, texCoords, offsets, 0);\n"
    "    gl_Position = vec4(texCoords * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static int compile(const char *label, const char *src, GLenum type)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    printf("%s: %s\n", label, ok ? "COMPILES" : "FAILS");
    if (!ok)
    {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        printf("   %s", log);
    }
    glDeleteShader(sh);
    return ok ? 0 : 1;
}

TEST(GatherOffsetShadowForms)
{
    int fails = 0;
    fails += compile("depth 4-arg (S,P,ref,off)", kVsDepth, GL_VERTEX_SHADER);
    fails += compile("color isampler 4-arg", kVsColorClamp, GL_VERTEX_SHADER);
    EXPECT_EQ_INT(0, fails);
}

int main(void) { RUN(GatherOffsetShadowForms); return mgl_test_summary(); }
