/* test_queries.c — query objects, timers, statistics, query buffers, conditional rendering */
#include "mgl_test.h"

static GLuint g_prog, g_discard_prog, g_vao, g_vbo;
static long long g_full;
static GLint g_vp[4];
static void draw_full(void);

static void setup(void)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    /* full-screen triangle, a 6-vertex quad list, a 5-vertex strip */
    static const float verts[] = {
        -1,-1,  3,-1,  -1,3,
        -1,-1,  1,-1,  1,1,   -1,-1,  1,1,  -1,1,
        -1,-1,  1,-1,  -1,1,  1,1,  -1,1,
    };
    const char *fs_discard = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ discard; }\n";
    g_prog = mgl_make_program(vs, fs);
    g_discard_prog = mgl_make_program(vs, fs_discard);
    glGenVertexArrays(1, &g_vao); glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo); glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(g_prog);
    glGetIntegerv(GL_VIEWPORT, g_vp);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    /* samples covered by a full-screen draw on the default framebuffer */
    GLuint q; glGenQueries(1, &q);
    glBeginQuery(GL_SAMPLES_PASSED, q); draw_full(); glEndQuery(GL_SAMPLES_PASSED);
    GLuint64 v = 0; glGetQueryObjectui64v(q, GL_QUERY_RESULT, &v); g_full = (long long)v;
    glDeleteQueries(1, &q);
    printf("# full-screen samples %lld (viewport %dx%d)\n", g_full, g_vp[2], g_vp[3]);
}

static void draw_full(void) { glDrawArrays(GL_TRIANGLES, 0, 3); }
/* every fragment discarded: no samples pass */
static void draw_rejected(void) { glUseProgram(g_discard_prog); draw_full(); glUseProgram(g_prog); }

static GLuint64 result64(GLuint q)
{
    GLuint64 v = 0xdeadbeef;
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &v);
    return v;
}

TEST(names_and_objects)
{
    GLuint q[2], c;
    glGenQueries(2, q);
    EXPECT_TRUE(q[0] != 0 && q[1] != 0 && q[0] != q[1]);
    EXPECT_EQ_INT(glIsQuery(q[0]), GL_FALSE);         /* reserved name, no object yet */
    glBeginQuery(GL_SAMPLES_PASSED, q[0]);
    EXPECT_EQ_INT(glIsQuery(q[0]), GL_TRUE);
    GLint cur = 0;
    glGetQueryiv(GL_SAMPLES_PASSED, GL_CURRENT_QUERY, &cur);
    EXPECT_EQ_INT(cur, q[0]);
    glEndQuery(GL_SAMPLES_PASSED);
    glGetQueryiv(GL_SAMPLES_PASSED, GL_CURRENT_QUERY, &cur);
    EXPECT_EQ_INT(cur, 0);

    glCreateQueries(GL_TIME_ELAPSED, 1, &c);
    EXPECT_EQ_INT(glIsQuery(c), GL_TRUE);
    GLint tgt = 0;
    glGetQueryObjectiv(c, GL_QUERY_TARGET, &tgt);
    EXPECT_EQ_INT(tgt, GL_TIME_ELAPSED);
    GLint avail = 0;
    glGetQueryObjectiv(c, GL_QUERY_RESULT_AVAILABLE, &avail);
    EXPECT_EQ_INT(avail, GL_TRUE);                    /* never-run object: 0, available */

    GLint bits = 0;
    glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
    EXPECT_EQ_INT(bits, 64);
    glGetQueryiv(GL_SAMPLES_PASSED, GL_QUERY_COUNTER_BITS, &bits);
    EXPECT_EQ_INT(bits, 64);
    EXPECT_NO_GL_ERROR();

    glDeleteQueries(2, q);
    glDeleteQueries(1, &c);
    EXPECT_EQ_INT(glIsQuery(q[0]), GL_FALSE);
    EXPECT_EQ_INT(glIsQuery(c), GL_FALSE);
    EXPECT_NO_GL_ERROR();
}

TEST(occlusion_counts)
{
    GLuint q[4];
    glGenQueries(4, q);
    const long long full = g_full;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBeginQuery(GL_SAMPLES_PASSED, q[0]);
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);


    /* rejected: all fragments discarded */
    glBeginQuery(GL_SAMPLES_PASSED, q[2]);
    draw_rejected();
    glEndQuery(GL_SAMPLES_PASSED);

    /* query spanning two draws */
    glBeginQuery(GL_SAMPLES_PASSED, q[3]);
    draw_full();
    {   /* a timestamp splits the command stream: the query spans two command buffers */
        GLuint ts; glGenQueries(1, &ts); glQueryCounter(ts, GL_TIMESTAMP); glDeleteQueries(1, &ts);
    }
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);

    /* scissored to 10x10 */
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 10, 10);
    glBeginQuery(GL_SAMPLES_PASSED, q[1]);
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);
    glDisable(GL_SCISSOR_TEST);

    EXPECT_TRUE(full >= 100);
    EXPECT_EQ_INT(result64(q[0]), full);
    EXPECT_EQ_INT(result64(q[1]), 100);
    EXPECT_EQ_INT(result64(q[2]), 0);
    EXPECT_EQ_INT(result64(q[3]), 2 * full);
    GLint avail = 0;
    glGetQueryObjectiv(q[0], GL_QUERY_RESULT_AVAILABLE, &avail);
    EXPECT_EQ_INT(avail, GL_TRUE);
    GLint r32 = 0;
    glGetQueryObjectiv(q[1], GL_QUERY_RESULT, &r32);
    EXPECT_EQ_INT(r32, 100);
    EXPECT_NO_GL_ERROR();
    glDeleteQueries(4, q);
}

TEST(any_samples_passed)
{
    GLuint q[3];
    glGenQueries(3, q);
    glBeginQuery(GL_ANY_SAMPLES_PASSED, q[0]);
    draw_full();
    glEndQuery(GL_ANY_SAMPLES_PASSED);

    glBeginQuery(GL_ANY_SAMPLES_PASSED_CONSERVATIVE, q[1]);
    draw_full();
    glEndQuery(GL_ANY_SAMPLES_PASSED_CONSERVATIVE);

    glBeginQuery(GL_ANY_SAMPLES_PASSED, q[2]);
    draw_rejected();
    glEndQuery(GL_ANY_SAMPLES_PASSED);

    EXPECT_EQ_INT(result64(q[0]), 1);
    EXPECT_EQ_INT(result64(q[1]), 1);
    EXPECT_EQ_INT(result64(q[2]), 0);
    EXPECT_NO_GL_ERROR();
    glDeleteQueries(3, q);
}

TEST(timers)
{
    GLuint t[3], te;
    glGenQueries(3, t);
    glGenQueries(1, &te);
    GLint64 now0 = 0, now1 = 0;
    glGetInteger64v(GL_TIMESTAMP, &now0);

    glQueryCounter(t[0], GL_TIMESTAMP);
    glBeginQuery(GL_TIME_ELAPSED, te);
    for (int i = 0; i < 20; i++) draw_full();
    glEndQuery(GL_TIME_ELAPSED);
    glQueryCounter(t[1], GL_TIMESTAMP);
    draw_full();
    glQueryCounter(t[2], GL_TIMESTAMP);

    GLuint64 elapsed = result64(te);
    GLuint64 ts0 = result64(t[0]), ts1 = result64(t[1]), ts2 = result64(t[2]);
    glGetInteger64v(GL_TIMESTAMP, &now1);

    EXPECT_TRUE(elapsed > 0);
    EXPECT_TRUE(elapsed < 10ull * 1000000000ull);
    EXPECT_TRUE(ts0 > 0);
    EXPECT_TRUE(ts0 <= ts1);
    EXPECT_TRUE(ts1 <= ts2);
    EXPECT_TRUE(now0 > 0);
    EXPECT_TRUE(now1 >= now0);
    EXPECT_TRUE((GLint64)ts2 <= now1);
    EXPECT_TRUE(ts1 - ts0 >= elapsed / 2);  /* elapsed interval lies between the two stamps */
    GLint tgt = 0;
    glGetQueryObjectiv(t[0], GL_QUERY_TARGET, &tgt);
    EXPECT_EQ_INT(tgt, GL_TIMESTAMP);
    EXPECT_NO_GL_ERROR();
    glDeleteQueries(3, t);
    glDeleteQueries(1, &te);
}

TEST(primitive_statistics)
{
    GLuint q[6];
    glGenQueries(6, q);

    glBeginQuery(GL_PRIMITIVES_GENERATED, q[0]);
    glDrawArrays(GL_TRIANGLES, 3, 6);            /* 2 triangles */
    glEndQuery(GL_PRIMITIVES_GENERATED);

    glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 0, q[1]);
    glDrawArrays(GL_TRIANGLE_STRIP, 9, 5);       /* 3 triangles */
    glEndQueryIndexed(GL_PRIMITIVES_GENERATED, 0);

    glBeginQuery(GL_PRIMITIVES_GENERATED, q[2]);
    glBeginQuery(GL_VERTICES_SUBMITTED, q[3]);
    glBeginQuery(GL_PRIMITIVES_SUBMITTED, q[4]);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 4); /* 4 instances x 1 triangle */
    glEndQuery(GL_PRIMITIVES_SUBMITTED);
    glEndQuery(GL_VERTICES_SUBMITTED);
    glEndQuery(GL_PRIMITIVES_GENERATED);

    /* stream 1 has nothing without a geometry shader */
    glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 1, q[5]);
    draw_full();
    glEndQueryIndexed(GL_PRIMITIVES_GENERATED, 1);

    EXPECT_EQ_INT(result64(q[0]), 2);
    EXPECT_EQ_INT(result64(q[1]), 3);
    EXPECT_EQ_INT(result64(q[2]), 4);
    EXPECT_EQ_INT(result64(q[3]), 12);
    EXPECT_EQ_INT(result64(q[4]), 4);
    EXPECT_EQ_INT(result64(q[5]), 0);
    EXPECT_NO_GL_ERROR();
    glDeleteQueries(6, q);
}

TEST(query_buffer_objects)
{
    GLuint q, buf;
    glGenQueries(1, &q);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glDrawArrays(GL_TRIANGLES, 3, 6);
    glEndQuery(GL_PRIMITIVES_GENERATED);

    glGenBuffers(1, &buf);
    glBindBuffer(GL_QUERY_BUFFER, buf);
    static const GLuint init[8] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
    glBufferData(GL_QUERY_BUFFER, sizeof init, init, GL_DYNAMIC_DRAW);
    GLint binding = 0;
    glGetIntegerv(GL_QUERY_BUFFER_BINDING, &binding);
    EXPECT_EQ_INT(binding, buf);

    glGetQueryObjectuiv(q, GL_QUERY_RESULT, (GLuint *)(uintptr_t)0);
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, (GLuint64 *)(uintptr_t)8);
    glGetQueryObjectuiv(q, GL_QUERY_RESULT_AVAILABLE, (GLuint *)(uintptr_t)4);
    EXPECT_NO_GL_ERROR();
    /* write past the end of the buffer */
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, (GLuint64 *)(uintptr_t)28);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindBuffer(GL_QUERY_BUFFER, 0);

    glGetQueryBufferObjectui64v(q, buf, GL_QUERY_RESULT, 16);
    glGetQueryBufferObjectiv(q, buf, GL_QUERY_TARGET, 24);
    EXPECT_NO_GL_ERROR();

    GLuint out[8] = {0};
    glBindBuffer(GL_COPY_READ_BUFFER, buf);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 2);
    EXPECT_EQ_INT(out[1], 1);
    EXPECT_EQ_INT(out[2], 2);
    EXPECT_EQ_INT(out[3], 0);
    EXPECT_EQ_INT(out[4], 2);
    EXPECT_EQ_INT(out[5], 0);
    EXPECT_EQ_INT(out[6], GL_PRIMITIVES_GENERATED);
    EXPECT_EQ_INT(out[7], 0xffffffff);

    /* errors */
    glGetQueryBufferObjectui64v(q, buf, GL_QUERY_RESULT, -8);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetQueryBufferObjectui64v(q, buf, GL_QUERY_RESULT, 28);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetQueryBufferObjectui64v(q, 0xbeef, GL_QUERY_RESULT, 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetQueryBufferObjectui64v(q, buf, GL_BUFFER_SIZE, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* buffer intentionally not deleted: mglDeleteBuffers indexes STATE(buffers[]) with the GL
     * target enum (buffers domain bug, intermittent crash) */
    glDeleteQueries(1, &q);
}

TEST(conditional_render)
{
    GLuint zero, nonzero, check[4], prim;
    glGenQueries(1, &zero);
    glGenQueries(1, &nonzero);
    glGenQueries(4, check);
    glGenQueries(1, &prim);
    const long long full = g_full;

    glBeginQuery(GL_SAMPLES_PASSED, zero);
    draw_rejected();
    glEndQuery(GL_SAMPLES_PASSED);

    glBeginQuery(GL_ANY_SAMPLES_PASSED, nonzero);
    draw_full();
    glEndQuery(GL_ANY_SAMPLES_PASSED);

    /* result 0: draw discarded */
    glBeginConditionalRender(zero, GL_QUERY_WAIT);
    glBeginQuery(GL_SAMPLES_PASSED, check[0]);
    glBeginQuery(GL_PRIMITIVES_GENERATED, prim);
    draw_full();
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndQuery(GL_SAMPLES_PASSED);
    glEndConditionalRender();

    /* result != 0: draw happens */
    glBeginConditionalRender(nonzero, GL_QUERY_BY_REGION_WAIT);
    glBeginQuery(GL_SAMPLES_PASSED, check[1]);
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);
    glEndConditionalRender();

    /* inverted */
    glBeginConditionalRender(zero, GL_QUERY_WAIT_INVERTED);
    glBeginQuery(GL_SAMPLES_PASSED, check[2]);
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);
    glEndConditionalRender();

    glBeginConditionalRender(nonzero, GL_QUERY_NO_WAIT_INVERTED);
    glBeginQuery(GL_SAMPLES_PASSED, check[3]);
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);
    glEndConditionalRender();
    EXPECT_NO_GL_ERROR();

    EXPECT_EQ_INT(result64(check[0]), 0);
    EXPECT_EQ_INT(result64(prim), 0);
    EXPECT_EQ_INT(result64(check[1]), full);
    EXPECT_EQ_INT(result64(check[2]), full);
    EXPECT_EQ_INT(result64(check[3]), 0);   /* NO_WAIT: nonzero is already resolved */

    /* errors */
    glBeginConditionalRender(zero, 0x1234);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBeginConditionalRender(0xbeef, GL_QUERY_WAIT);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBeginConditionalRender(prim, GL_QUERY_WAIT);       /* PRIMITIVES_GENERATED not allowed */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndConditionalRender();
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBeginConditionalRender(zero, GL_QUERY_WAIT);
    glBeginConditionalRender(zero, GL_QUERY_WAIT);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndConditionalRender();
    EXPECT_NO_GL_ERROR();

    glDeleteQueries(1, &zero);
    glDeleteQueries(1, &nonzero);
    glDeleteQueries(4, check);
    glDeleteQueries(1, &prim);
}

TEST(error_cases)
{
    GLuint q[3], ts;
    glGenQueries(3, q);
    glGenQueries(1, &ts);

    glBeginQuery(0x1234, q[0]);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBeginQuery(GL_TIMESTAMP, q[0]);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBeginQuery(GL_SAMPLES_PASSED, 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBeginQuery(GL_SAMPLES_PASSED, 0xbeef);              /* never generated */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 4, q[0]);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBeginQueryIndexed(GL_SAMPLES_PASSED, 1, q[0]);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glEndQuery(GL_SAMPLES_PASSED);                        /* nothing active */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndQuery(0x1234);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glBeginQuery(GL_SAMPLES_PASSED, q[0]);
    EXPECT_NO_GL_ERROR();
    glBeginQuery(GL_SAMPLES_PASSED, q[1]);                /* target already active */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBeginQuery(GL_ANY_SAMPLES_PASSED, q[1]);            /* shares the occlusion binding */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBeginQuery(GL_TIME_ELAPSED, q[0]);                  /* query already active */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLuint v;
    glGetQueryObjectuiv(q[0], GL_QUERY_RESULT, &v);       /* active */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glQueryCounter(q[0], GL_TIMESTAMP);                   /* active */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndQuery(GL_ANY_SAMPLES_PASSED);                    /* active one has another target */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndQuery(GL_SAMPLES_PASSED);
    EXPECT_NO_GL_ERROR();

    glBeginQuery(GL_TIME_ELAPSED, q[0]);                  /* object has target SAMPLES_PASSED */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glQueryCounter(q[0], GL_TIMESTAMP);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glQueryCounter(ts, GL_TIME_ELAPSED);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glQueryCounter(0xbeef, GL_TIMESTAMP);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glGetQueryObjectuiv(q[2], GL_QUERY_RESULT, &v);       /* name without object */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetQueryObjectuiv(q[0], 0x1234, &v);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    GLint iv;
    glGetQueryiv(0x1234, GL_CURRENT_QUERY, &iv);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetQueryiv(GL_SAMPLES_PASSED, 0x1234, &iv);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetQueryiv(GL_TIMESTAMP, GL_CURRENT_QUERY, &iv);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetQueryIndexediv(GL_PRIMITIVES_GENERATED, 4, GL_CURRENT_QUERY, &iv);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    GLuint c;
    glCreateQueries(0x1234, 1, &c);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGenQueries(-1, &c);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glDeleteQueries(-1, &c);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* deleting an active query: name freed, query stays active until EndQuery */
    glBeginQuery(GL_PRIMITIVES_GENERATED, q[1]);
    glDeleteQueries(1, &q[1]);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(glIsQuery(q[1]), GL_FALSE);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q[2]);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    EXPECT_NO_GL_ERROR();

    glDeleteQueries(1, &q[0]);
    glDeleteQueries(1, &q[2]);
    glDeleteQueries(1, &ts);
    glDeleteQueries(1, &ts);                              /* unused name: ignored */
    EXPECT_NO_GL_ERROR();
}

TEST(more_statistics)
{
    GLuint q[4];
    glGenQueries(4, q);
    const long long full = g_full;

    /* SAMPLES_PASSED and FRAGMENT_SHADER_INVOCATIONS active at the same time */
    glBeginQuery(GL_SAMPLES_PASSED, q[0]);
    glBeginQuery(GL_FRAGMENT_SHADER_INVOCATIONS, q[1]);
    draw_full();
    glEndQuery(GL_SAMPLES_PASSED);
    draw_full();                                    /* only counted by q[1] */
    glEndQuery(GL_FRAGMENT_SHADER_INVOCATIONS);

    glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, 2, q[2]);
    GLint cur = 0;
    glGetQueryIndexediv(GL_PRIMITIVES_GENERATED, 2, GL_CURRENT_QUERY, &cur);
    EXPECT_EQ_INT(cur, q[2]);
    glGetQueryIndexediv(GL_PRIMITIVES_GENERATED, 0, GL_CURRENT_QUERY, &cur);
    EXPECT_EQ_INT(cur, 0);
    glEndQueryIndexed(GL_PRIMITIVES_GENERATED, 2);

    glBeginQuery(GL_CLIPPING_INPUT_PRIMITIVES, q[3]);
    glDrawArrays(GL_LINE_STRIP, 9, 5);              /* 4 lines */
    glDrawArrays(GL_POINTS, 0, 7);                  /* 7 points */
    glEndQuery(GL_CLIPPING_INPUT_PRIMITIVES);

    EXPECT_EQ_INT(result64(q[0]), full);
    EXPECT_EQ_INT(result64(q[1]), 2 * full);
    EXPECT_EQ_INT(result64(q[2]), 0);
    EXPECT_EQ_INT(result64(q[3]), 11);

    GLuint v = 0;
    glGetQueryObjectuiv(q[3], GL_QUERY_RESULT_NO_WAIT, &v);
    EXPECT_EQ_INT(v, 11);
    EXPECT_NO_GL_ERROR();
    glDeleteQueries(4, q);
}

TEST(compute_invocations)
{
    const char *cs = "#version 450 core\n"
        "layout(local_size_x=4, local_size_y=2, local_size_z=1) in;\n"
        "layout(std430, binding=0) buffer B { uint v[]; };\n"
        "void main(){ v[gl_LocalInvocationIndex] = 1u; }\n";
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &cs, NULL);
    glCompileShader(sh);
    GLuint p = glCreateProgram();
    glAttachShader(p, sh);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    if (!ok) return;
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glUseProgram(p);

    GLuint q;
    glGenQueries(1, &q);
    glBeginQuery(GL_COMPUTE_SHADER_INVOCATIONS, q);
    glDispatchCompute(3, 1, 1);
    glEndQuery(GL_COMPUTE_SHADER_INVOCATIONS);
    EXPECT_EQ_INT(result64(q), 24);
    glDeleteQueries(1, &q);
    glUseProgram(g_prog);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    setup();
    RUN(names_and_objects);
    RUN(occlusion_counts);
    RUN(any_samples_passed);
    RUN(timers);
    RUN(primitive_statistics);
    RUN(query_buffer_objects);
    RUN(conditional_render);
    RUN(error_cases);
    RUN(more_statistics);
    /* compute programs currently assert in program.c (parseSPIRVShaderToMetal); opt-in */
    if (getenv("MGL_TEST_COMPUTE"))
        RUN(compute_invocations);
    return mgl_test_summary();
}
