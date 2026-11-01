/* link_bench86.c — link/introspection latency bench for the introspect r86 paths
 * (sampler-struct hoisting, ES precision defaults, GS capture variants, the
 * mixed-stage relink retry).  Not part of the suite; built by ab/abbench_link86.sh
 * against two library trees and compared with the usual min / pair-ratio medians.
 */
#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* sampler-struct + ES precision shape (r86a/b/e: hoisting, highp pad) */
static const char *HVS =
    "#version 310 es\n"
    "precision highp float;\n"
    "struct S { sampler2D a; sampler2D b; };\n"
    "uniform S u[2];\n"
    "uniform highp float w;\n"
    "out highp vec4 c;\n"
    "void main() {\n"
    "  vec4 t = texture(u[0].a, vec2(0.5)) + texture(u[1].b, vec2(0.5));\n"
    "  gl_Position = vec4(%.*f, %.*f, 0.0, 1.0) * t;\n"
    "  c = t * w;\n"
    "}\n";
static const char *HFS =
    "#version 310 es\n"
    "precision mediump float;\n"
    "in highp vec4 c;\n"
    "out vec4 f;\n"
    "void main() { f = c * vec4(%.*f); }\n";

/* explicit uniform locations, same name both stages (r86d: io-mapper / retry) */
static const char *LVS =
    "#version 310 es\n"
    "layout(location = 3) uniform vec4 a;\n"
    "layout(location = 5) uniform vec4 b;\n"
    "void main() { gl_Position = a * %.*f + b; }\n";
static const char *LFS =
    "#version 310 es\n"
    "precision highp float;\n"
    "layout(location = 3) uniform vec4 a;\n"
    "layout(location = 6) uniform vec4 c;\n"
    "out vec4 f;\n"
    "void main() { f = a * c * %.*f; }\n";

/* geometry program (r86a/b: capture + kernel variants of the same source) */
static const char *GVS =
    "#version 330\n"
    "layout(location=0) in vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char *GGS =
    "#version 330\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 4) out;\n"
    "uniform highp float s;\n"
    "void main() {\n"
    "  for (int i = 0; i < 4; i++) {\n"
    "    vec4 q = gl_in[0].gl_Position * s;\n"
    "    q.x += %.*f * float(i);\n"
    "    gl_Position = q;\n"
    "    EmitVertex();\n"
    "  }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *GFS =
    "#version 330\n"
    "out vec4 f;\n"
    "void main() { f = vec4(%.*f, 0.4, 0.6, 1.0); }\n";

static double bench(const char *tag, const char *vs, const char *fs, const char *gs,
                    int n, int distinct)
{
    char src[4096];
    double t0 = now_s();
    for (int i = 0; i < n; i++)
    {
        int k = distinct ? (i % 16) : 0;
        GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
        GLuint p = glCreateProgram();
        GLint ok = 0;
        snprintf(src, sizeof src, vs, 6, 0.125f + k * 0.01f);
        glShaderSource(s1, 1, (const char *[]){ src }, NULL);
        glCompileShader(s1);
        snprintf(src, sizeof src, fs, 6, 0.25f + k * 0.01f);
        glShaderSource(s2, 1, (const char *[]){ src }, NULL);
        glCompileShader(s2);
        glAttachShader(p, s1);
        glAttachShader(p, s2);
        if (gs)
        {
            GLuint s3 = glCreateShader(GL_GEOMETRY_SHADER);
            snprintf(src, sizeof src, gs, 6, 0.375f + k * 0.01f);
            glShaderSource(s3, 1, (const char *[]){ src }, NULL);
            glCompileShader(s3);
            glAttachShader(p, s3);
            glDeleteShader(s3);
        }
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { printf("   LINK FAIL %s i=%d\n", tag, i); break; }
        /* introspection walks the r86 naming paths */
        GLuint idx = glGetProgramResourceIndex(p, GL_PROGRAM_INPUT, "c");
        (void)idx;
        GLint loc = glGetUniformLocation(p, "u[1].b");
        if (loc < 0 && gs == NULL && vs == HVS) loc = glGetUniformLocation(p, "b");
        glDeleteShader(s1);
        glDeleteShader(s2);
        glDeleteProgram(p);
    }
    double dt = now_s() - t0;
    printf("RESULT %-28s n=%d %s %10.3f\n", tag, n, distinct ? "distinct" : "cached   ", dt);
    return dt;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 40;
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 8, 8);

    bench("hoist-sampler-struct distinct", HVS, HFS, NULL, n, 1);
    bench("hoist-sampler-struct cached", HVS, HFS, NULL, n, 0);
    bench("explicit-loc distinct", LVS, LFS, NULL, n, 1);
    bench("explicit-loc cached", LVS, LFS, NULL, n, 0);
    bench("gs-program distinct", GVS, GFS, GGS, n, 1);
    bench("gs-program cached", GVS, GFS, GGS, n, 0);
    return 0;
}
