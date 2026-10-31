/* tess-domain draw-path microbench: GL_PATCHES submit cost vs drawArrays.
 * build: cc -I tests/headless -I MGL/include -o /tmp/ab_patches .agent/ab_patches.c build/libmgl.dylib  (rpath)
 * run:   /tmp/ab_patches [iters]
 * prints RESULT lines: per-draw ms for (a) drawArrays baseline, (b) patches no-TCS
 * (default levels), (c) patches with TCS+TES, submit-only and with glFinish. */
#include "mgl_test.h"
#include <dlfcn.h>
#include <time.h>

static GLuint mgl_make_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
               fprintf(stderr, "compile failed: %s\n", log); exit(1); }
    return s;
}

static GLuint mgl_link4(GLuint a, GLuint b, GLuint c, GLuint d)
{
    GLuint pr = glCreateProgram();
    if (a) glAttachShader(pr, a);
    if (b) glAttachShader(pr, b);
    if (c) glAttachShader(pr, c);
    if (d) glAttachShader(pr, d);
    glLinkProgram(pr);
    GLint ok = 0; glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(pr, sizeof log, NULL, log);
               fprintf(stderr, "LINK failed: %s\n", log); exit(1); }
    return pr;
}

static double now_s(void)
{
    struct timespec ts;
    if (getenv("MGL_AB_CPU"))
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    else
        clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const char *kVS =
    "#version 400 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *kFS =
    "#version 400 core\n"
    "out vec4 col;\n"
    "void main() { col = vec4(0.2, 0.6, 1.0, 1.0); }\n";
static const char *kTES =
    "#version 400 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "in vec2 tcpos[];\n"
    "void main() { gl_Position = vec4(gl_TessCoord.x * tcpos[0].x + gl_TessCoord.y * tcpos[1].x + gl_TessCoord.z * tcpos[2].x,"
    " gl_TessCoord.x * tcpos[0].y + gl_TessCoord.y * tcpos[1].y + gl_TessCoord.z * tcpos[2].y, 0.0, 1.0); }\n";
static const char *kTCS =
    "#version 400 core\n"
    "layout(vertices=3) out;\n"
    "in vec2 vpos[]; out vec2 tcpos[];\n"
    "void main() { tcpos[gl_InvocationID] = vpos[gl_InvocationID];"
    " if (gl_InvocationID == 0) { gl_TessLevelOuter[0]=4.0; gl_TessLevelOuter[1]=4.0; gl_TessLevelOuter[2]=4.0;"
    " gl_TessLevelInner[0]=4.0; } }\n";


int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 2000;
    GLuint vs = mgl_make_shader(GL_VERTEX_SHADER, kVS);
    GLuint fs = mgl_make_shader(GL_FRAGMENT_SHADER, kFS);

    /* plain program */
    GLuint p0 = mgl_link4(vs, fs, 0, 0);

    /* TES-only program (no TCS: default levels) */
    GLuint tes = mgl_make_shader(GL_TESS_EVALUATION_SHADER, kTES);
    GLuint p1 = mgl_link4(vs, tes, fs, 0);

    /* TCS+TES program */
    const char *vsrc3 =
        "#version 400 core\n"
        "layout(location=0) in vec2 pos;\n"
        "out vec2 vpos;\n"
        "void main() { vpos = pos; gl_Position = vec4(pos,0.0,1.0); }\n";
    (void)kTCS;
    GLuint vs3 = mgl_make_shader(GL_VERTEX_SHADER, vsrc3);
    GLuint tcs = mgl_make_shader(GL_TESS_CONTROL_SHADER, kTCS);
    GLuint p2 = mgl_link4(vs3, tcs, tes, fs);

    float quad[] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f };
    GLuint vbo, vao;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    double t;
    struct { int submit; } dummy = { 0 };
    (void)dummy;

    /* 0: drawArrays baseline */
    glUseProgram(p0);
    for (int i = 0; i < 100; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    t = now_s();
    for (int i = 0; i < iters; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    printf("RESULT drawArrays-tris x%d %.4f\n", iters, (now_s() - t) * 1e3);
    t = now_s();
    for (int i = 0; i < iters; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    printf("RESULT drawArrays-tris submit x%d %.4f\n", iters, (now_s() - t) * 1e3);
    glFinish();

    /* 1: GL_PATCHES, no TCS (3 verts/patch, default levels 4) */
    glUseProgram(p1);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    for (int i = 0; i < 100; i++) glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    t = now_s();
    for (int i = 0; i < iters; i++) glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    printf("RESULT patches-noTCS x%d %.4f\n", iters, (now_s() - t) * 1e3);
    t = now_s();
    for (int i = 0; i < iters; i++) glDrawArrays(GL_PATCHES, 0, 3);
    printf("RESULT patches-noTCS submit x%d %.4f\n", iters, (now_s() - t) * 1e3);
    glFinish();

    /* 2: GL_PATCHES with TCS+TES */
    glUseProgram(p2);
    for (int i = 0; i < 100; i++) glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    {
        double times[iters];
        struct timespec c0, c1;
        t = now_s();
        for (int i = 0; i < iters; i++) { double a = now_s(); clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0); glDrawArrays(GL_PATCHES, 0, 3); clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1); times[i] = now_s() - a;
            if (times[i] > 0.010) printf("SLOWDRAW i=%d %.1f ms cpu=%.1f ms\n", i, times[i]*1e3,
                (c1.tv_sec-c0.tv_sec)*1e3 + (c1.tv_nsec-c0.tv_nsec)/1e6); }
        printf("RESULT patches-TCS-TES submit x%d %.4f\n", iters, (now_s() - t) * 1e3);
        /* percentiles */
        for (int i = 1; i < iters; i++) { double v = times[i]; int j = i - 1;
            while (j >= 0 && times[j] > v) { times[j+1] = times[j]; j--; } times[j+1] = v; }
        int idx[5] = {0, iters/2, iters*9/10, iters*99/100, iters-1};
        printf("HIST p50=%.4f p90=%.4f p99=%.4f max=%.4f (ms per draw)\n",
               times[idx[1]]*1e3, times[idx[2]]*1e3, times[idx[3]]*1e3, times[idx[4]]*1e3);
        double sum = 0; for (int i = 0; i < iters; i++) sum += times[i];
        printf("HIST sum(draw)=%.1f ms loop=%.1f ms\n", sum * 1e3, (now_s() - t) * 1e3);
    }
    glFinish();

    /* 3: bigger patch draws: 64 patches per draw (192 verts) */
    float big[64 * 3 * 2];
    for (int i = 0; i < 64 * 3; i++) { big[i*2] = quad[(i%3)*2]; big[i*2+1] = quad[(i%3)*2+1]; }
    GLuint vbo2;
    glGenBuffers(1, &vbo2);
    glBindBuffer(GL_ARRAY_BUFFER, vbo2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(big), big, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    t = now_s();
    for (int i = 0; i < iters; i++) glDrawArrays(GL_PATCHES, 0, 192);
    glFinish();
    printf("RESULT patches-TCS-TES-64 x%d %.4f\n", iters, (now_s() - t) * 1e3);
    {
        void (*dump)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "mgl_profile_dump");
        int (*den)(int) = (int (*)(int))dlsym(RTLD_DEFAULT, "mgl_tess_diag_enable");
        unsigned (*spins)(uint64_t *, uint64_t *) = (unsigned (*)(uint64_t *, uint64_t *))dlsym(RTLD_DEFAULT, "mgl_tess_ring_spins");
        double (*gus)(void) = (double (*)(void))dlsym(RTLD_DEFAULT, "mgl_tess_gpu_time_us");
        unsigned (*cc)(void) = (unsigned (*)(void))dlsym(RTLD_DEFAULT, "mgl_tess_cbs_completed");
        double (*gmax)(void) = (double (*)(void))dlsym(RTLD_DEFAULT, "mgl_tess_gpu_time_max");
        void (*rst)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "mgl_tess_gpu_time_reset");
        if (den) { rst(); den(1); }
        /* phase window for the noTCS shape too (CPU-bound: the GPU keeps pace) */
        {
            void (*prst)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "mgl_tess_phase_reset");
            double (*pus)(int) = (double (*)(int))dlsym(RTLD_DEFAULT, "mgl_tess_phase_us");
            uint64_t (*pcnt)(int) = (uint64_t (*)(int))dlsym(RTLD_DEFAULT, "mgl_tess_phase_cnt");
            static const char *phn[12] = { "setup", "index", "ring", "vsbind", "gather",
                                           "tcs", "notcs", "blob", "order", "render", "post", "TOTAL" };
            glUseProgram(p1);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
            if (prst) prst();
            for (int i = 0; i < 200; i++) glDrawArrays(GL_PATCHES, 0, 3);
            glFinish();
            if (pus && pcnt)
                for (int i = 0; i < 12; i++)
                    if (pcnt(i))
                        printf("PHASE1 %-7s %8.1f us total, %6.2f us/draw (n=%llu)\n",
                               phn[i], pus(i), pus(i) / (double)pcnt(i),
                               (unsigned long long)pcnt(i));
            glUseProgram(p2);
            glBindBuffer(GL_ARRAY_BUFFER, vbo2);
            glBufferData(GL_ARRAY_BUFFER, sizeof(big), big, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        }
        {
            void (*prst)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "mgl_tess_phase_reset");
            double (*pus)(int) = (double (*)(int))dlsym(RTLD_DEFAULT, "mgl_tess_phase_us");
            uint64_t (*pcnt)(int) = (uint64_t (*)(int))dlsym(RTLD_DEFAULT, "mgl_tess_phase_cnt");
            static const char *phn[12] = { "setup", "index", "ring", "vsbind", "gather",
                                           "tcs", "notcs", "blob", "order", "render", "post", "TOTAL" };
            if (prst) prst();
            for (int i = 0; i < 200; i++) glDrawArrays(GL_PATCHES, 0, 192);
            glFinish();
            if (den) den(0);
            if (pus && pcnt)
                for (int i = 0; i < 12; i++)
                    if (pcnt(i))
                        printf("PHASE %-7s %8.1f us total, %6.2f us/draw (n=%llu)\n",
                               phn[i], pus(i), pus(i) / (double)pcnt(i),
                               (unsigned long long)pcnt(i));
        }
        {
            uint64_t mx = 0, sl = 0;
            unsigned n = spins ? spins(&mx, &sl) : 0;
            printf("RING-DIAG sleeps=%u max_sleep_us=%llu total_slept_ms=%llu\n", n,
                   (unsigned long long)mx, (unsigned long long)sl / 1000);
        }
        printf("GPU-DIAG gpu_us_per_cb=%.1f cbs=%u gpu_us_max=%.1f\n",
               gus ? gus() / (cc ? cc() : 1) : -1.0, cc ? cc() : 0, gmax ? gmax() : -1.0);
        if (dump) dump();
    }
    /* 4: frame-like: 10 patch draws + glFinish per frame */
    {
        double ft[600];
        int nf = 60;
        double t0 = now_s();
        for (int f = 0; f < nf; f++)
        {
            double a = now_s();
            for (int i = 0; i < 10; i++) glDrawArrays(GL_PATCHES, 0, 192);
            glFinish();
            ft[f] = now_s() - a;
        }
        double tot = now_s() - t0;
        double mx = 0, sum = 0;
        for (int f = 0; f < nf; f++) { sum += ft[f]; if (ft[f] > mx) mx = ft[f]; }
        printf("RESULT frame10 x%d %.4f (max frame %.4f ms, avg %.4f ms)\n", nf, tot * 1e3, mx * 1e3, sum / nf * 1e3);
    }
    return 0;
}
