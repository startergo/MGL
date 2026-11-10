/* test_compat_ssbo_sizes — SSBO runtime-array .length() (spvBufferSizeConstants)
 * must be bound for EVERY program that reads it, including a program allocated
 * at addresses a previous (deleted) program used.
 *
 * KHR-GL46.es_31_compatibility.shader_storage_buffer_object.advanced-
 * unsizedArrayLength-fs-* flakes in whole-group runs (all lengths read back 0,
 * isolated runs pass): the "does this stage's MSL read spvBufferSizeConstants"
 * memo was a static cache keyed on (Program*, msl_str*) plus a 32-byte head
 * memcmp — but EVERY SPIRV-Cross MSL starts with the same ~60 bytes, so the
 * head guard never distinguishes two shaders, and a program created at a
 * recycled address pair inherited the dead program's answer (needed=false ->
 * slot 25 never bound -> .length() reads the zero page -> all lengths 0).
 *
 * This test reproduces that sequence deliberately: for a sweep of padding
 * widths it links and draws program A (an SSBO shader whose stage does NOT read
 * .length()), deletes it, then links and draws program B (the CTS
 * advanced-unsizedArrayLength-fs-int / -vs-int shader) and checks the lengths.
 * The pad sweep walks the malloc size classes so B's msl_str lands on A's freed
 * chunk for at least one width; with the address-keyed memo that iteration drew
 * with no size constants (lengths 0).
 */
#include "mgl_test.h"

static const char *kVsPlain =
    "#version 450\n"
    "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";

static const char *kFsPlain =
    "#version 450\n"
    "layout(location = 0) out vec4 o_color;\n"
    "void main() { o_color = vec4(0.0, 1.0, 0.0, 1.0); }\n";

static const char *kBodySizes =
    "  g_length[0] = int(g_input0.length());\n"
    "  g_length[1] = int(g_input23[0].data.length());\n"
    "  g_length[2] = int(g_input23[1].data.length());\n"
    "  g_length2 = int(g_length.length());\n";

static const char *kBodyConst =
    "  g_length2 = 41;\n";

/* blocks(k): the CTS block layout.  A's k extra dead members go into Input0
 * BEFORE its unsized member (an unsized array must be last) and pad the
 * generated MSL ~13 bytes each, walking the allocator size classes; A never
 * reads Input0, and B (the checked program) always uses k = 0. */
static void blocks(char *out, size_t cap, int k)
{
    size_t n = 0;
    n += (size_t)snprintf(out + n, cap - n,
        "#version 450\n"
        "layout(std430, binding = 0) readonly buffer Input0 {");
    for (int i = 0; i < k; i++)
        n += (size_t)snprintf(out + n, cap - n, " int pad%d;", i);
    n += (size_t)snprintf(out + n, cap - n,
        " uint g_input0[]; };\n"
        "layout(std430, binding = 1) readonly buffer Input23 { float data[]; } g_input23[2];\n"
        "layout(std430, binding = 3) buffer Output { int g_length2; int g_length[]; };\n");
}

/* fragment stage with the blocks (CTS fs-int: kFsPlain is swapped in for A) */
static void fs_sizes(char *out, size_t cap, int k, int body_const)
{
    char b[2048];
    blocks(b, sizeof b, k);
    snprintf(out, cap,
        "%s"
        "layout(location = 0) out vec4 o_color;\n"
        "void main() {\n"
        "  o_color = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "%s"
        "}\n", b, body_const ? kBodyConst : kBodySizes);
}

/* vertex stage with the blocks (CTS vs-int: gl_Position + the body) */
static void vs_sizes(char *out, size_t cap, int k, int body_const)
{
    char b[2048];
    blocks(b, sizeof b, k);
    snprintf(out, cap,
        "%s"
        "void main() {\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "%s"
        "}\n", b, body_const ? kBodyConst : kBodySizes);
}

static GLuint link(const char *vs, const char *fs)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    GLint ok = 0;
    glShaderSource(v, 1, (const GLchar *[]){ vs }, NULL);
    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if (!ok) return 0;
    glShaderSource(f, 1, (const GLchar *[]){ fs }, NULL);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok) return 0;
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    return ok ? p : 0;
}

/* one CTS advanced-unsizedArrayLength round with (vs, fs): when check is set
 * the program is B and the four lengths must read back correct; A (check = 0)
 * is drawn only to poison the memo, its writes are not asserted */
static int draw_and_read(const char *vs, const char *fs, const char *tag, int check)
{
    GLuint p, vao, buf[4];
    const int sizes[4] = { 7, 3, 4, 5 };
    int data = 41;
    int bad = 0;

    p = link(vs, fs);
    if (!p) { MGL_FAIL("%s: link failed", tag); return 1; }
    glUseProgram(p);

    glGenBuffers(4, buf);
    for (int i = 0; i < 4; i++)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, buf[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizes[i] * 4, &data, GL_DYNAMIC_COPY);
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf[3]);
    int *out = (int *)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 4 * sizes[3], GL_MAP_READ_BIT);
    if (!out) { MGL_FAIL("%s: map failed", tag); bad = 1; }
    else
    {
        if (check)
        {
            if (out[0] != sizes[3] - 1)
            { MGL_FAIL("%s: g_length.length() == %d, expected %d (size constants not bound?)", tag, out[0], sizes[3] - 1); bad = 1; }
            for (int i = 0; i < 3; i++)
                if (out[i + 1] != sizes[i]) { MGL_FAIL("%s: length[%d] == %d, expected %d", tag, i, out[i + 1], sizes[i]); bad = 1; }
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    glDeleteProgram(p);
    glDeleteBuffers(4, buf);
    glDeleteVertexArrays(1, &vao);
    return bad;
}

TEST(fs_lengths_survive_recycled_addresses)
{
    char a[4096], b[4096];
    int fails = 0;

    /* plain correct case first (no prior program) */
    fs_sizes(b, sizeof b, 0, 0);
    fails += draw_and_read(kVsPlain, b, "fs plain", 1);

    /* pad sweep: A (no .length()) linked, drawn, deleted, then B */
    for (int k = 0; k <= 28 && fails < 3; k++)
    {
        fs_sizes(a, sizeof a, k, 1);
        if (draw_and_read(kVsPlain, a, "fs A", 0)) MGL_FAIL("poison program A(k=%d) failed", k);
        fs_sizes(b, sizeof b, 0, 0);
        if (draw_and_read(kVsPlain, b, "fs B", 1))
        { MGL_FAIL("B after A(k=%d) drew wrong lengths (stale size-constant memo)", k); fails++; }
    }
    EXPECT_TRUE(fails == 0);
}

TEST(vs_lengths_survive_recycled_addresses)
{
    char a[4096], b[4096];
    int fails = 0;

    vs_sizes(b, sizeof b, 0, 0);
    fails += draw_and_read(b, kFsPlain, "vs plain", 1);

    for (int k = 0; k <= 28 && fails < 3; k++)
    {
        vs_sizes(a, sizeof a, k, 1);
        if (draw_and_read(a, kFsPlain, "vs A", 0)) MGL_FAIL("poison program A(k=%d) failed", k);
        vs_sizes(b, sizeof b, 0, 0);
        if (draw_and_read(b, kFsPlain, "vs B", 1))
        { MGL_FAIL("B after A(k=%d) drew wrong lengths (stale size-constant memo)", k); fails++; }
    }
    EXPECT_TRUE(fails == 0);
}

/* direct proof of the alias hole: the exported probe memoized on the string
 * pointer with a 32-byte head guard — every generated MSL starts with the
 * same ~60 bytes, so text rewritten in place at the same address inherited
 * the previous answer */
extern int mgl_msl_has_size_constants(const char *msl);

static const char *kMslHead =
    "#include <metal_stdlib>\n#include <simd/simd.h>\n\nusing namespace metal;\n\n";

TEST(msl_probe_alias)
{
    char *buf = (char *)malloc(128 * 1024);

    /* stage A: MSL that reads spvBufferSizeConstants */
    snprintf(buf, 128 * 1024,
        "%s"
        "struct Output { int g_length2; int g_length[]; };\n"
        "fragment void main(device Output& sb [[buffer(3)]],\n"
        "                  constant uint* spvBufferSizeConstants [[buffer(25)]])\n"
        "{ sb.g_length2 = int((spvBufferSizeConstants[3] - 4) / 4); }\n",
        kMslHead);
    EXPECT_EQ_INT(mgl_msl_has_size_constants(buf), 1);

    /* stage B: same address, same head, no size constants */
    snprintf(buf, 128 * 1024,
        "%s"
        "struct Output { int g_length2; int g_length[]; };\n"
        "fragment void main(device Output& sb [[buffer(3)]])\n"
        "{ sb.g_length2 = 41; }\n",
        kMslHead);
    EXPECT_EQ_INT(mgl_msl_has_size_constants(buf), 0);

    /* and back again (the reverse poison direction) */
    snprintf(buf, 128 * 1024,
        "%s"
        "struct Output { int g_length2; int g_length[]; };\n"
        "fragment void main(device Output& sb [[buffer(3)]],\n"
        "                  constant uint* spvBufferSizeConstants [[buffer(25)]])\n"
        "{ sb.g_length2 = int((spvBufferSizeConstants[3] - 4) / 4); }\n",
        kMslHead);
    EXPECT_EQ_INT(mgl_msl_has_size_constants(buf), 1);

    free(buf);
}

int main(void)
{
    RUN(msl_probe_alias);
    RUN(fs_lengths_survive_recycled_addresses);
    RUN(vs_lengths_survive_recycled_addresses);
    return mgl_test_summary();
}
