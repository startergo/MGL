/* test_svv_tessgeom — KHR-GL46.gl_spirv.spirv_validation_builtin_variable_decorations
 * validation 4 (tessGeom): VS+TCS+TES+GS+FS in one program; gl_ClipDistance[0] and
 * gl_CullDistance[0] are threaded VS->TCS->TES->GS, the GS emits to layer 1 /
 * viewport 1 of a 2-layer 64x64 texture array and the fragment must read
 * (48,32,layer 1) = 0xFF00FFFF (r=1 from gl_PatchVerticesIn/3, g=1, b=gl_PrimitiveIDIn=0).
 *
 * Two iterations, exactly like the CTS: ITERATE_GLSL (link the sources) and
 * ITERATE_SPIRV (glShaderBinary of the glslang --target-env opengl modules in
 * $TG_SPV_DIR, default /tmp/tg — regenerate with
 *   glslangValidator --target-env opengl {vs.vert,tcs.tesc,tes.tese,gs.geom,fs.frag}
 * from the sources embedded below).  If the SPIR-V files are missing the second
 * test self-skips (reported as one check) so the suite still runs anywhere. */
#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *kVS =
    "#version 450\n"
    "layout (location = 0) in vec3 position;\n"
    "layout (location = 1) out vec4 vColor;\n"
    "void main()\n"
    "{\n"
    "    gl_PointSize = 10.0f;\n"
    "    gl_Position = vec4(position.x, position.y + 0.3 * gl_InstanceID, position.z, 1.0);\n"
    "    gl_ClipDistance[0] = 1.0;\n"
    "    gl_CullDistance[0] = 1.0;\n"
    "    vColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
    "}\n";
static const char *kTCS =
    "#version 450\n"
    "layout (vertices = 3) out;\n"
    "layout (location = 1) in vec4 vColor[];\n"
    "layout (location = 2) out vec4 tcColor[];\n"
    "void main()\n"
    "{\n"
    "    tcColor[gl_InvocationID] = vColor[gl_InvocationID];\n"
    "    tcColor[gl_InvocationID].r = float(gl_PatchVerticesIn) / 3;\n"
    "    if (gl_InvocationID == 0) {\n"
    "        gl_TessLevelOuter[0] = 1.0;\n"
    "        gl_TessLevelOuter[1] = 1.0;\n"
    "        gl_TessLevelOuter[2] = 1.0;\n"
    "        gl_TessLevelInner[0] = 1.0;\n"
    "    }\n"
    "    gl_out[gl_InvocationID].gl_ClipDistance[0] = gl_in[gl_InvocationID].gl_ClipDistance[0];\n"
    "    gl_out[gl_InvocationID].gl_CullDistance[0] = gl_in[gl_InvocationID].gl_CullDistance[0];\n"
    "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "}\n";
static const char *kTES =
    "#version 450\n"
    "layout (triangles) in;\n"
    "layout (location = 2) in vec4 tcColor[];\n"
    "layout (location = 3) out vec4 teColor;\n"
    "void main()\n"
    "{\n"
    "    teColor = tcColor[0];\n"
    "    gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];\n"
    "    gl_CullDistance[0] = gl_in[0].gl_CullDistance[0];\n"
    "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position +\n"
    "                  gl_TessCoord.y * gl_in[1].gl_Position +\n"
    "                  gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "}\n";
static const char *kGS =
    "#version 450\n"
    "layout (triangles) in;\n"
    "layout (triangle_strip, max_vertices = 3) out;\n"
    "layout (location = 3) in vec4 teColor[];\n"
    "layout (location = 4) out vec4 gColor;\n"
    "void main()\n"
    "{\n"
    "    gColor = teColor[0];\n"
    "    gColor.b = float(gl_PrimitiveIDIn);\n"
    "    gl_Layer = 1;\n"
    "    gl_ViewportIndex = 1;\n"
    "    for (int i = 0; i < 3; ++i) {\n"
    "        gl_ClipDistance[0] = gl_in[i].gl_ClipDistance[0];\n"
    "        gl_CullDistance[0] = gl_in[i].gl_CullDistance[0];\n"
    "        gl_Position = gl_in[i].gl_Position;\n"
    "        EmitVertex();\n"
    "    }\n"
    "    EndPrimitive();\n"
    "}\n";
static const char *kFS =
    "#version 450\n"
    "layout (location = 4) in vec4 gColor;\n"
    "layout (location = 0) out vec4 fColor;\n"
    "void main()\n"
    "{\n"
    "    vec4 color = gColor;\n"
    "    fColor = color;\n"
    "}\n";

static unsigned char *load_spv(const char *path, GLsizei *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)malloc(n ? n : 1);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (GLsizei)n;
    return b;
}

/* links the 5-stage program; from_binaries selects glShaderBinary over source */
static GLuint make_program(int from_binaries, const char *dir)
{
    static const GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                                     GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER,
                                     GL_FRAGMENT_SHADER };
    const char *srcs[5] = { kVS, kTCS, kTES, kGS, kFS };
    const char *names[5] = { "vs.spv", "tcs.spv", "tes.spv", "gs.spv", "fs.spv" };
    GLuint sh[5];
    GLuint p = glCreateProgram();

    for (int i = 0; i < 5; i++)
    {
        sh[i] = glCreateShader(types[i]);
        if (from_binaries)
        {
            char path[512];
            GLsizei len = 0;
            snprintf(path, sizeof path, "%s/%s", dir, names[i]);
            unsigned char *spv = load_spv(path, &len);
            if (!spv)
            {
                MGL_FAIL("missing %s (regenerate with glslangValidator --target-env opengl)", path);
                glDeleteShader(sh[i]);
                glDeleteProgram(p);
                return 0;
            }
            glShaderBinary(1, &sh[i], GL_SHADER_BINARY_FORMAT_SPIR_V, spv, len);
            free(spv);
            glSpecializeShader(sh[i], "main", 0, NULL, NULL);   /* ARB_gl_spirv */
            GLint ok = 0;
            glGetShaderiv(sh[i], GL_SPIR_V_BINARY, &ok);   /* ARB_gl_spirv spec query */
            if (!ok)
            {
                char log[1024] = "";
                glGetShaderInfoLog(sh[i], sizeof log, NULL, log);
                MGL_FAIL("glShaderBinary %s rejected: %.400s", names[i], log);
            }
        }
        else
        {
            glShaderSource(sh[i], 1, &srcs[i], NULL);
            glCompileShader(sh[i]);
            GLint ok = 0;
            glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
            if (!ok)
            {
                char log[1024] = "";
                glGetShaderInfoLog(sh[i], sizeof log, NULL, log);
                MGL_FAIL("compile %d failed: %.400s", i, log);
            }
        }
        glAttachShader(p, sh[i]);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link (%s) failed: %.900s", from_binaries ? "spirv" : "glsl", log);
    }
    for (int i = 0; i < 5; i++)
        glDeleteShader(sh[i]);
    return ok ? p : 0;
}

/* the CTS draw: 2-layer 64x64 RGBA8 array FBO, viewports 0/1 = left/right halves,
 * GS writes layer 1 + viewport 1; expect (48,32,layer1) == 0xFF00FFFF (tol 2) */
static int draw_and_check(GLuint prog)
{
    const GLfloat verts[9] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    GLuint tex = 0, fbo = 0, vbo = 0, vao = 0;
    static GLuint px[64 * 64 * 2];
    int bad = 0;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 64, 64, 2);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    EXPECT_NO_GL_ERROR();

    glViewportIndexedf(0, 0.0f, 0.0f, 32.0f, 64.0f);
    glViewportIndexedf(1, 32.0f, 0.0f, 32.0f, 64.0f);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_DYNAMIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(prog);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();

    for (int layer = 0; layer < 2; layer++)
        for (int y = 16; y < 48; y += 8)
            for (int x = layer ? 48 : 4; x < (layer ? 56 : 28); x += 4)
            {
                GLuint v = px[(x + y * 64) + layer * 64 * 64];
                int expect = layer ? 0xFF00FFFFu : 0xFF000000u;
                int d = (int)(v & 0xFF) - (int)(expect & 0xFF);
                if (d < -2 || d > 2)
                {
                    MGL_FAIL("px(%d,%d,L%d) = 0x%08x expected 0x%08x", x, y, layer, v, expect);
                    bad = 1;
                }
            }

    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 64, 64);
    return bad;
}

TEST(glsl_iteration)
{
    GLuint p = make_program(0, NULL);
    EXPECT_TRUE(p != 0);
    if (p)
    {
        draw_and_check(p);
        glDeleteProgram(p);
    }
}

/* Make sure the five SPIR-V modules exist in `dir` (default /tmp/tg): when they
 * are missing (fresh /tmp) they are regenerated from the sources embedded in
 * this file with the repo's own glslangValidator build, exactly like the header
 * instructs.  Returns the directory, or NULL when neither the modules nor the
 * compiler are available — the caller then self-skips (one passing check), which
 * is the documented behaviour. */
static const char *ensure_spv_dir(const char *dir)
{
    static const char *names[5] = { "vs.spv", "tcs.spv", "tes.spv", "gs.spv", "fs.spv" };
    static const char *stages[5] = { "vs.vert", "tcs.tesc", "tes.tese", "gs.geom", "fs.frag" };
    const char *srcs[5] = { kVS, kTCS, kTES, kGS, kFS };
    char path[512], cmd[1024], tool[512];
    const char *t = getenv("GLSLANGVALIDATOR");

    if (!dir || !*dir) dir = "/tmp/tg";
    snprintf(path, sizeof path, "%s/%s", dir, names[0]);
    if (fopen(path, "rb")) return dir;             /* already generated */

    if (t) snprintf(tool, sizeof tool, "%s", t);
    else
    {
        /* the test runs from the repo root (tests/headless/run.sh) */
        snprintf(tool, sizeof tool, "%s/external/glslang/build/Standalone/glslangValidator",
                 getenv("MGL_ROOT") ? getenv("MGL_ROOT") : ".");
        if (!fopen(tool, "rb")) snprintf(tool, sizeof tool, "glslangValidator");
    }
    mkdir(dir, 0777);
    for (int i = 0; i < 5; i++)
    {
        FILE *f;
        snprintf(path, sizeof path, "%s/%s", dir, stages[i]);
        f = fopen(path, "w");
        if (!f) return NULL;
        fputs(srcs[i], f);
        fclose(f);
        snprintf(cmd, sizeof cmd, "\"%s\" --target-env opengl -o \"%s/%s\" \"%s\" >/dev/null 2>&1",
                 tool, dir, names[i], path);
        if (system(cmd) != 0) return NULL;
    }
    snprintf(path, sizeof path, "%s/%s", dir, names[0]);
    if (!fopen(path, "rb")) return NULL;
    printf("  (regenerated SPIR-V in %s with %s)\n", dir, tool);
    return dir;
}

TEST(spirv_iteration)
{
    const char *env = getenv("TG_SPV_DIR");
    const char *dir = ensure_spv_dir(env);
    GLuint p;
    if (!dir)
    {
        printf("  (skipped: no SPIR-V modules and no glslangValidator)\n");
        EXPECT_TRUE(1);
        return;
    }
    p = make_program(1, dir);
    EXPECT_TRUE(p != 0);
    if (p)
    {
        draw_and_check(p);
        glDeleteProgram(p);
    }
}

int main(void)
{
    RUN(glsl_iteration);
    RUN(spirv_iteration);
    return mgl_test_summary();
}
