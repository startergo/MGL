/* test_compat_spirv_dynamic_indexing — supervisor r108 regression (first-bad
 * a095741, the r97a-r98c window): KHR-GL46.gl_spirv.spirv_validation_
 * capabilities_test failed with "Program build failed [2]" and an EMPTY info
 * log — pipeline 2 is a compute shader that dynamically indexes a UBO array,
 * an SSBO array, a sampler2D array and an image2D array with a loop counter
 * (the SPIR-V DynamicIndexing capabilities).  Loading the exact module via
 * glShaderBinary + glSpecializeShader must link.
 */
#include "mgl_test.h"
#include <sys/stat.h>

static const char *cs =
    "#version 450\n"
    "\n"
    "layout (local_size_x = 1) in;\n"
    "layout (location = 0) uniform sampler2D uniSamp[10];\n"
    "layout (location = 10, rgba8) uniform image2D uniImg[10];\n"
    "layout (binding = 5) uniform UniData\n"
    "{\n"
    "   int a[10];\n"
    "} uniBuff[10];\n"
    "layout (binding = 5) buffer StorageData\n"
    "{\n"
    "   int a[10];\n"
    "} storageBuff[10];\n"
    "\n"
    "void main()\n"
    "{\n"
    "    vec2 coord = vec2(0.0);\n"
    "    ivec2 point = ivec2(0);\n"
    "\n"
    "    int ret = 0;\n"
    "    for (int i = 0; i < 10; ++i)"
    "    {\n"
    "        ret = ret + uniBuff[i].a[i] + storageBuff[i].a[i];\n"
    "        textureGather(uniSamp[i], coord);\n"
    "        imageLoad(uniImg[i], point);\n"
    "    }\n"
    "    memoryBarrier();\n"
    "}\n";

static unsigned *load_spv(const char *path, GLsizei *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *len = (GLsizei)n;
    return buf;
}

TEST(spirv_binary_dynamic_indexing_links)
{
    /* 1. produce the module: compile the same source through MGL's GLSL path
     *    with MGL_PREPARED_DUMP capturing the SPIR-V */
    mkdir("/tmp/dispirv", 0755);
    setenv("MGL_PREPARED_DUMP", "/tmp/dispirv", 1);
    GLuint p0 = mgl_make_program_cs(cs);
    unsetenv("MGL_PREPARED_DUMP");
    EXPECT_TRUE(p0 != 0);          /* the GLSL path works */
    if (!p0) return;
    glDeleteProgram(p0);

    /* the dump for the compute stage */
    GLsizei len = 0;
    unsigned *spv = load_spv("/tmp/dispirv/stage5.spv", &len);
    if (!spv) { /* the dump slot number can change; take the newest file */ }

    if (!spv)
    {
        /* name variants: look for whatever got dumped */
        EXPECT_TRUE(spv != NULL);
        return;
    }
    EXPECT_TRUE(len > 4 && spv[0] == 0x07230203);

    /* 2. load it as a binary (the CTS path) */
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &sh, GL_SHADER_BINARY_FORMAT_SPIR_V, spv, len);
    EXPECT_NO_GL_ERROR();
    glSpecializeShader(sh, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();

    GLuint p = glCreateProgram();
    glAttachShader(p, sh);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[512] = { 0 };
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  link log: %s\n", log);
    }
    EXPECT_TRUE(ok);               /* RED pre-fix: link failed with an empty log */

    free(spv);
}
int main(void){ RUN(spirv_binary_dynamic_indexing_links); return mgl_test_summary(); }
