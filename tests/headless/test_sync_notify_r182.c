/* r182: mglSyncNotify - a fence completion callback. rsvga's fence watcher
 * blocked in glClientWaitSync while holding MGL's single GL context, and the rest
 * of its renderer queued behind those waits (glmark2 590 with the watcher on, 985
 * off). mglSyncNotify(sync, fn, arg) instead runs fn(arg) once, off the GL thread,
 * when the GPU passes the fence.
 *
 *   1. pending fence: fn has not run when mglSyncNotify returns (the GPU is still
 *      busy), then runs exactly once, on another thread, with the sync signalled;
 *      no glFlush is needed (the fence's commands are submitted by the call)
 *   2. already-signalled fence: fn still runs once, and not on the calling thread
 *   3. the sync deleted before it signals: fn still runs exactly once
 *   4. an invalid sync or a NULL fn: GL_FALSE, GL_INVALID_VALUE, fn never runs
 *   5. 64 fences interleaved with draws: each fn runs exactly once
 *   6. the GL thread is free while a notify is pending: GL calls return at once */
#include "mgl_test.h"
#include "MGLContext.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

typedef struct { _Atomic int count; pthread_t thread; } Slot;
static void on_fence(void *arg)
{
    Slot *s = (Slot *)arg;
    s->thread = pthread_self();
    atomic_fetch_add(&s->count, 1);
}
static uint64_t now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1000000; }
static int wait_count(Slot *s, int want, int ms)
{
    uint64_t end = now_ms() + ms;
    while (atomic_load(&s->count) < want && now_ms() < end) usleep(1000);
    return atomic_load(&s->count);
}

#define W 1024
static GLuint heavy_prog, fbo, rt, vao;
static GLint loc_n;
static void heavy_draw(int n)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, W);
    glBindVertexArray(vao); glUseProgram(heavy_prog);
    glUniform1i(loc_n, n);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
static GLint sync_status(GLsync s)
{
    GLint v = 0; GLsizei len = 0;
    glGetSynciv(s, GL_SYNC_STATUS, 1, &len, &v);
    return v;
}

int main(void)
{
    int bad = 0;
    const char *vs = "#version 450\nvoid main(){ vec2 q = vec2((gl_VertexID&1)*4-1, (gl_VertexID&2)*2-1); gl_Position = vec4(q,0,1); }\n";
    const char *fs = "#version 450\nuniform int n;\nlayout(location=0) out vec4 o;\n"
                     "void main(){ vec4 a = gl_FragCoord; for (int i = 0; i < n; i++) a = sin(a * 1.0001 + 0.37); o = a; }\n";
    heavy_prog = mgl_make_program(vs, fs);
    loc_n = glGetUniformLocation(heavy_prog, "n");
    glGenVertexArrays(1, &vao);
    glCreateTextures(GL_TEXTURE_2D, 1, &rt); glTextureStorage2D(rt, 1, GL_RGBA32F, W, W);
    glCreateFramebuffers(1, &fbo); glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, rt, 0);
    glFinish();
    while (glGetError() != GL_NO_ERROR) {}
    pthread_t me = pthread_self();

    /* 1 */
    {
        Slot s = { 0 };
        heavy_draw(3000);
        GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        uint64_t t0 = now_ms();
        GLboolean ok = mglSyncNotify(f, on_fence, &s);
        int at_return = atomic_load(&s.count);
        GLint st_return = sync_status(f);
        /* 6: the GL thread is not held while the notify is pending */
        GLint maxtex = 0; uint64_t t1 = now_ms();
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtex);
        uint64_t gl_ms = now_ms() - t1;
        int n = wait_count(&s, 1, 10000);
        uint64_t fired_ms = now_ms() - t0;
        usleep(200000);
        int later = atomic_load(&s.count);
        GLint st_after = sync_status(f);
        printf("case 1: registered %d, at return count %d (sync %s), fired after %llu ms, count %d then %d, %s thread; case 6: a GL query took %llu ms\n",
               ok, at_return, st_return == GL_SIGNALED ? "signalled" : "unsignalled", (unsigned long long)fired_ms,
               n, later, pthread_equal(s.thread, me) ? "the calling" : "another", (unsigned long long)gl_ms);
        if (!ok) { printf("  FAIL: not registered\n"); bad++; }
        if (st_return == GL_UNSIGNALED && at_return != 0) { printf("  FAIL: fn ran before the fence signalled\n"); bad++; }
        if (st_return == GL_SIGNALED) printf("  note: the GPU finished before the check; the not-before check is vacuous this run\n");
        if (n != 1 || later != 1) { printf("  FAIL: fn did not run exactly once\n"); bad++; }
        if (pthread_equal(s.thread, me)) { printf("  FAIL: fn ran on the calling thread\n"); bad++; }
        if (st_after != GL_SIGNALED) { printf("  FAIL: the sync is not signalled after fn ran\n"); bad++; }
        if (gl_ms > 50 || maxtex <= 0) { printf("  FAIL: the GL thread was held\n"); bad++; }
        glDeleteSync(f);
    }
    /* 2 */
    {
        Slot s = { 0 };
        heavy_draw(10);
        GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFinish();
        GLint st = sync_status(f);
        GLboolean ok = mglSyncNotify(f, on_fence, &s);
        int n = wait_count(&s, 1, 2000);
        usleep(200000);
        int later = atomic_load(&s.count);
        printf("case 2: sync %s before the call, registered %d, count %d then %d, %s thread\n",
               st == GL_SIGNALED ? "signalled" : "unsignalled", ok, n, later, pthread_equal(s.thread, me) ? "the calling" : "another");
        if (!ok || n != 1 || later != 1 || pthread_equal(s.thread, me)) { printf("  FAIL\n"); bad++; }
        glDeleteSync(f);
    }
    /* 3 */
    {
        Slot s = { 0 };
        heavy_draw(2000);
        GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        GLboolean ok = mglSyncNotify(f, on_fence, &s);
        glDeleteSync(f);
        GLenum e = glGetError();
        int n = wait_count(&s, 1, 10000);
        usleep(200000);
        int later = atomic_load(&s.count);
        printf("case 3: registered %d, deleted (error 0x%x), count %d then %d\n", ok, e, n, later);
        if (!ok || e != GL_NO_ERROR || n != 1 || later != 1) { printf("  FAIL\n"); bad++; }
    }
    /* 4 */
    {
        Slot s = { 0 };
        GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        GLboolean a = mglSyncNotify((GLsync)(uintptr_t)0x1234, on_fence, &s);
        GLenum ea = glGetError();
        GLboolean b = mglSyncNotify(f, NULL, &s);
        GLenum eb = glGetError();
        glFinish();
        usleep(200000);
        printf("case 4: bogus sync -> %d (0x%x), NULL fn -> %d (0x%x), count %d\n", a, ea, b, eb, atomic_load(&s.count));
        if (a || b || ea != GL_INVALID_VALUE || eb != GL_INVALID_VALUE || atomic_load(&s.count)) { printf("  FAIL\n"); bad++; }
        glDeleteSync(f);
    }
    /* 5 */
    {
        static Slot s[64];
        GLsync f[64];
        for (int i = 0; i < 64; i++) {
            heavy_draw(20);
            f[i] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            if (!mglSyncNotify(f[i], on_fence, &s[i])) { printf("  FAIL: fence %d not registered\n", i); bad++; }
        }
        uint64_t end = now_ms() + 10000;
        int done = 0;
        while (now_ms() < end) {
            done = 0;
            for (int i = 0; i < 64; i++) done += atomic_load(&s[i].count) >= 1;
            if (done == 64) break;
            usleep(1000);
        }
        usleep(200000);
        int once = 0;
        for (int i = 0; i < 64; i++) once += atomic_load(&s[i].count) == 1;
        printf("case 5: %d of 64 fired, %d exactly once\n", done, once);
        if (done != 64 || once != 64) { printf("  FAIL\n"); bad++; }
        for (int i = 0; i < 64; i++) glDeleteSync(f[i]);
    }
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) { printf("FAIL: stray GL error 0x%x\n", e); bad++; }
    printf("%s: %d failure(s)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
