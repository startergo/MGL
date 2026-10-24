/* glmark2 gl41-geometry replica — per-frame cost attribution for the geometry-shader
 * emulation (compat perf session 7). NOT part of the suite (run.sh only globs test_*).
 *
 * The scene is one glDrawArrays(GL_POINTS, 0, 65536) per frame through a GS that expands
 * every point into a 4-vertex triangle strip (uSize/uTime uniforms change per frame).
 *
 * Reports, per case:
 *   app   — draw + glFinish every frame (what the benchmark sees)
 *   gpu   — GL_TIME_ELAPSED over a batch of draws (no per-frame finish)
 *   submit— the same batch WITHOUT any finish, with the geom pools uncapped
 *           (MGL_GEOM_POOL_CAP/BUDGET raised at start): the CPU cost of encoding the
 *           emulation (pools big enough that no waitUntilCompleted happens)
 * Env knobs (read at the first draw, so they must be set before this binary starts or
 * by the harness below): MGL_GEOM_TILED_MIN (serial prep), MGL_GEOM_POOL_*.
 */
#include "mgl_test.h"
#include <time.h>

#define POINTS 65536
#define FRAMES 60

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const char *kVS =
    "#version 150\n"
    "in vec2 position;\n"
    "out float vId;\n"
    "void main() {\n"
    "  vId = position.x * 0.5 + position.y * 0.5;\n"
    "  gl_Position = vec4(position, 0.0, 1.0);\n"
    "}\n";
static const char *kGS =
    "#version 150\n"
    "layout(points) in;\n"
    "layout(triangle_strip, max_vertices=4) out;\n"
    "uniform vec2 uSize;\n"
    "uniform float uTime;\n"
    "out vec2 gUV;\n"
    "void emit(vec2 off, vec2 uv) {\n"
    "  gl_Position = gl_in[0].gl_Position + vec4(off, 0.0, 0.0);\n"
    "  gUV = uv;\n"
    "  EmitVertex();\n"
    "}\n"
    "void main() {\n"
    "  vec2 s = uSize * (0.6 + 0.4 * abs(sin(uTime)));\n"
    "  emit(vec2(-s.x, -s.y), vec2(0.0, 0.0));\n"
    "  emit(vec2( s.x, -s.y), vec2(1.0, 0.0));\n"
    "  emit(vec2(-s.x,  s.y), vec2(0.0, 1.0));\n"
    "  emit(vec2( s.x,  s.y), vec2(1.0, 1.0));\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFS =
    "#version 150\n"
    "in vec2 gUV;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "  float c = smoothstep(0.0, 0.15, min(min(gUV.x, 1.0-gUV.x), min(gUV.y, 1.0-gUV.y)));\n"
    "  fragColor = vec4(0.1 + gUV.x, 0.2 + gUV.y, c, 1.0);\n"
    "}\n";

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

static int g_points = POINTS;
static void frame(GLuint prog, GLint utime, float t)
{
    glUniform1f(utime, t);
    glDrawArrays(GL_POINTS, 0, g_points);
}


int main(int argc, char **argv)
{
    /* the harness runs this binary directly; allow knobs from the command line */
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], "MGL_", 4) == 0)
        {
            char *eq = strchr(argv[i], '=');
            if (!eq) continue;
            *eq = 0;
            setenv(argv[i], eq + 1, 1);
            *eq = '=';
        }

    /* uncap the pools for the submit-only measurement (read at the first draw) */
    if (!getenv("MGL_GEOM_POOL_CAP")) setenv("MGL_GEOM_POOL_CAP", "536870912", 1);
    if (!getenv("MGL_GEOM_POOL_BUDGET")) setenv("MGL_GEOM_POOL_BUDGET", "4294967296", 1);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, compile(GL_VERTEX_SHADER, kVS));
    glAttachShader(prog, compile(GL_GEOMETRY_SHADER, kGS));
    glAttachShader(prog, compile(GL_FRAGMENT_SHADER, kFS));
    glLinkProgram(prog);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    if (!link)
    {
        char log[4096] = "";
        glGetProgramInfoLog(prog, sizeof log, NULL, log);
        printf("LINK FAILED: %s\n", log);
        return 1;
    }
    glUseProgram(prog);
    GLint usize = glGetUniformLocation(prog, "uSize");
    GLint utime = glGetUniformLocation(prog, "uTime");
    {
        float sz = 0.012f;
        const char *e = getenv("G41_SIZE");
        if (e) sz = (float)atof(e);
        glUniform2f(usize, sz, sz);
        printf("uSize = %g\n", sz);
    }

    int npoints = POINTS;
    {
        const char *e = getenv("G41_POINTS");
        if (e) { int v = atoi(e); if (v > 0 && v <= POINTS) npoints = v; }
        g_points = npoints;
        printf("points = %d\n", npoints);
    }
    float *pts = malloc(sizeof(float) * 2 * POINTS);
    const unsigned side = (unsigned)(ceil(sqrt((double)POINTS)) + 0.5);
    for (unsigned i = 0; i < POINTS; i++)
    {
        unsigned x = i % side, y = i / side;
        pts[i * 2 + 0] = (side > 1) ? (-1.0f + 2.0f * ((float)x / (side - 1))) : 0.0f;
        pts[i * 2 + 1] = (side > 1) ? (-1.0f + 2.0f * ((float)y / (side - 1))) : 0.0f;
    }
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * POINTS, pts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glViewport(0, 0, 800, 600);
    glFinish();

    double t0, t1;
    /* 1. app frame: draw + finish per frame (draw and finish timed separately) */
    double t_draw = 0, t_fin = 0;
    for (int f = 0; f < FRAMES; f++)
    {
        t0 = now_s();
        frame(prog, utime, (float)f * 0.01f);
        t1 = now_s();
        glFinish();
        double t3 = now_s();
        t_draw += t1 - t0;
        t_fin += t3 - t1;
    }
    printf("app   (draw+finish/frame): %8.1f us/frame (draw %.1f + finish %.1f)\n",
           (t_draw + t_fin) * 1e6 / FRAMES, t_draw * 1e6 / FRAMES, t_fin * 1e6 / FRAMES);


    /* 2. GPU time per frame: TIME_ELAPSED around ONE draw, finished before the read —
     *    the GPU is never starved, so the elapsed time is the draw's own GPU cost
     *    (a batch without finishes measures CPU-bound starvation, not GPU work) */
    {
        GLuint q = 0;
        GLuint64 ns = 0, tot = 0;
        glGenQueries(1, &q);
        for (int f = 0; f < FRAMES; f++)
        {
            glBeginQuery(GL_TIME_ELAPSED, q);
            frame(prog, utime, (float)f * 0.02f);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            tot += ns;
        }
        glDeleteQueries(1, &q);
        printf("gpu   (per-frame finish): %8.1f us/frame\n", (double)tot / 1000.0 / FRAMES);
    }

    /* 3. submit only: no finish inside the loop (pools uncapped above) */
    glFinish();
    t0 = now_s();
    for (int f = 0; f < FRAMES; f++)
        frame(prog, utime, (float)f * 0.03f);
    t1 = now_s();
    double t2 = now_s();
    glFinish();
    double t3 = now_s();
    printf("submit(no finish)       : %8.1f us/frame (drain %.1f ms)\n",
           (t1 - t0) * 1e6 / FRAMES, (t3 - t2) * 1e3);

    /* 4. submitted-but-not-finished steady state: the app frame without the finish wait */
    t0 = now_s();
    for (int f = 0; f < FRAMES; f++)
        frame(prog, utime, (float)f * 0.04f);
    t1 = now_s();
    glFinish();
    printf("submit-no-wait          : %8.1f us/frame\n", (t1 - t0) * 1e6 / FRAMES);

    /* 5. control: the same points drawn WITHOUT a geometry shader (VS+FS only):
     *    baseline submit + raster + finish cost in this process */
    {
        GLuint c = glCreateProgram();
        const char *vsc =
            "#version 150\n"
            "in vec2 position;\n"
            "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
        const char *fsc =
            "#version 150\n"
            "out vec4 fragColor;\n"
            "void main() { fragColor = vec4(0.1, 0.2, 0.3, 1.0); }\n";
        glAttachShader(c, compile(GL_VERTEX_SHADER, vsc));
        glAttachShader(c, compile(GL_FRAGMENT_SHADER, fsc));
        glLinkProgram(c);
        glUseProgram(c);
        glFinish();
        t0 = now_s();
        for (int f = 0; f < FRAMES; f++)
            glDrawArrays(GL_POINTS, 0, g_points);
        t1 = now_s();
        glFinish();
        double t4 = now_s();
        printf("control points (no GS)   : %8.1f us/frame (submit %.1f + finish-wait %.1f)\n",
               (t4 - t0) * 1e6 / FRAMES, (t1 - t0) * 1e6 / FRAMES, (t4 - t1) * 1e6 / FRAMES);
        glDeleteProgram(c);
    }

    free(pts);
    return 0;
}

