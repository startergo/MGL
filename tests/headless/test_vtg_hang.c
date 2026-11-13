/* r74: the tess+geometry vtg hang (official CTS run, baseline-1cbdd1ce).
 *
 * A GL texture object that was bound to a sampler unit but never given any
 * storage (no glTexImage/glTexStorage) is realised as the opaque-black
 * fallback MTLTexture.  bindMTLTexture's fallback path did NOT clear the
 * texture's dirty bits -- unlike the primary path -- so every subsequent
 * tess draw (whose processGLState re-binds the texture units each draw;
 * plain draws use the per-slot resolution cache and never re-enter)
 * released and re-created the fallback: a per-draw texture-recreation loop
 * that flooded the log with the "emergency fallback" trio (2280 creations in
 * two wide_lines vtg cases here) and hung the official CTS runner's 7 vtg
 * cases for its whole 90-minute alarm.
 *
 * The regression check redirects the test's own stderr (where MGL's NSLogs
 * go) and counts the fallback-creation lines across 300 tess+GS draws:
 * fixed = 1 creation total, pre-fix = one per draw.
 */
#include "mgl_test.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *kVs =
    "#version 460 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
static const char *kTcs =
    "#version 460 core\n"
    "layout(vertices=3) out;\n"
    "void main(){\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "  gl_TessLevelInner[0] = 1.0;\n"
    "}\n";
static const char *kTes =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "void main(){\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "}\n";
static const char *kGs =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "void main(){\n"
    "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFs =
    "#version 460 core\n"
    "layout(location=0) out vec4 o;\n"
    "uniform sampler2DMSArray u_tex;\n"
    "void main(){ o = vec4(texelFetch(u_tex, ivec3(0), 0).r * 0.001, 1.0, 0.0, 1.0); }\n";

static int count_log_lines(const char *path, const char *needle)
{
    FILE *f = fopen(path, "r");
    char buf[4096];
    int n = 0;
    if (!f) return -1;
    while (fgets(buf, sizeof buf, f))
        if (strstr(buf, needle)) n++;
    fclose(f);
    return n;
}

TEST(TessVtgUnallocatedTextureLoop)
{
    const GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[5] = { kVs, kTcs, kTes, kGs, kFs };
    static const float pos[6] = { -0.7f, -0.7f, 0.7f, -0.7f, 0.0f, 0.7f };
    const char *LOG = "/tmp/mgl_vtg_hang_stderr.log";
    GLuint p, vao, b;
    GLint link = 0;
    float c[4];
    int lit = 0, created;

    p = glCreateProgram();
    for (int i = 0; i < 5; i++)
    {
        GLuint so = glCreateShader(types[i]);
        glShaderSource(so, 1, &srcs[i], NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);

    /* the CTS case's state (decoded from the official run's log): the
     * DEFAULT object of GL_TEXTURE_2D_MULTISAMPLE_ARRAY (the object texture
     * 0 selects on that target, name 0xcafebeef) after an ES-semantics
     * upload into texture 0 — one level-complete entry with every dimension
     * zero (num_levels=1, w=h=d=0).  The tess draw path re-walks the texture
     * units every draw; the object resolves (the num_levels!=0 skip does not
     * apply), its primary Metal creation (a 0x0x0x0 descriptor) returns
     * NULL, and it realises as the emergency fallback.  Pre-fix the fallback
     * path never cleared the dirty bits, so every draw released and
     * re-created it -- the per-draw loop that hung the official CTS run's 7
     * vtg cases for its 90-minute alarm. */
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0);
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 1, GL_RGBA8, 0, 0, 0, GL_FALSE);

    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);

    /* capture our own stderr (MGL's NSLogs) for the batch */
    fflush(NULL);
    int saved = dup(2);
    int logfd = open(LOG, O_RDWR | O_CREAT | O_TRUNC, 0600);
    EXPECT_TRUE(logfd >= 0);
    if (logfd < 0) return;
    dup2(logfd, 2);

    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 300; i++)
        glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();

    fflush(NULL);
    dup2(saved, 2);
    close(saved);
    close(logfd);

    created = count_log_lines(LOG, "Fallback texture created successfully");
    printf("  fallback texture creations across 300 tess+GS draws: %d\n", created);
    /* fixed: exactly one (the first realisation).  pre-fix: one per draw
     * (~300, plus 300 Metal texture release/create pairs and 900 log lines) */
    EXPECT_TRUE(created >= 0 && created <= 2);

    /* the render check: the green channel dominates (the unbound unit samples
     * black/opaque) */
    for (int y = 0; y < 200; y++) for (int x = 0; x < 200; x++)
    {
        glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, c);
        if (c[1] > 0.5f) lit++;
    }
    EXPECT_TRUE(lit > 500);

    glDeleteProgram(p);
}

int main(void) { RUN(TessVtgUnallocatedTextureLoop); return mgl_test_summary(); }
