/* test_debug.c — KHR_debug, object labels, glGet* family, strings, legacy entry points. */
#include "mgl_test.h"

/* compatibility-profile entry points (not in glcorearb.h); MGL exports them */
extern void glBegin(GLenum mode);
extern void glEnd(void);
extern void glNewList(GLuint list, GLenum mode);
extern GLuint glGenLists(GLsizei range);
extern GLboolean glIsList(GLuint list);
extern GLint glRenderMode(GLenum mode);
extern void glColor3f(GLfloat r, GLfloat g, GLfloat b);
extern void glMatrixMode(GLenum mode);
extern void glLoadIdentity(void);
extern GLboolean glAreTexturesResident(GLsizei n, const GLuint *textures, GLboolean *residences);
extern void glClientActiveTexture(GLenum texture);
extern void glGetnMapdv(GLenum target, GLenum query, GLsizei bufSize, GLdouble *v);
extern void glVertexP2ui(GLenum type, GLuint value);
extern void glColorP4ui(GLenum type, GLuint color);

/* ---- callback capture ---- */
#define MAX_CAPTURED 128
typedef struct { GLenum source, type, severity; GLuint id; GLsizei length; char msg[256]; } Captured;
static Captured cap[MAX_CAPTURED];
static int ncap;
static int user_tag = 1234;
static const void *last_user;

static void APIENTRY debug_cb(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                              const GLchar *message, const void *userParam)
{
    last_user = userParam;
    if (ncap >= MAX_CAPTURED) return;
    Captured *c = &cap[ncap++];
    c->source = source; c->type = type; c->id = id; c->severity = severity; c->length = length;
    snprintf(c->msg, sizeof c->msg, "%s", message);
}

static void reset_debug(void)
{
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(debug_cb, &user_tag);
    /* back to defaults: everything on except LOW */
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0, NULL, GL_FALSE);
    ncap = 0;
}

TEST(callback_receives_insert)
{
    reset_debug();
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 42,
                         GL_DEBUG_SEVERITY_NOTIFICATION, -1, "hello debug");
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_EQ_INT(cap[0].source, GL_DEBUG_SOURCE_APPLICATION);
    EXPECT_EQ_INT(cap[0].type, GL_DEBUG_TYPE_MARKER);
    EXPECT_EQ_INT(cap[0].id, 42);
    EXPECT_EQ_INT(cap[0].severity, GL_DEBUG_SEVERITY_NOTIFICATION);
    EXPECT_EQ_INT(cap[0].length, 11);
    EXPECT_STREQ(cap[0].msg, "hello debug");
    EXPECT_TRUE(last_user == &user_tag);

    /* explicit length, third-party source */
    ncap = 0;
    glDebugMessageInsert(GL_DEBUG_SOURCE_THIRD_PARTY, GL_DEBUG_TYPE_PERFORMANCE, 7,
                         GL_DEBUG_SEVERITY_HIGH, 3, "abcdef");
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_STREQ(cap[0].msg, "abc");
    EXPECT_EQ_INT(cap[0].length, 3);

    /* invalid inserts */
    ncap = 0;
    glDebugMessageInsert(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, 1, GL_DEBUG_SEVERITY_HIGH, -1, "x");
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, 1, GL_DONT_CARE, -1, "x");
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    char big[2048]; memset(big, 'a', sizeof big); big[sizeof big - 1] = 0;
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, 1, GL_DEBUG_SEVERITY_HIGH, -1, big);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

TEST(callback_receives_gl_errors)
{
    reset_debug();
    const GLubyte *s = glGetString(0xDEAD);
    EXPECT_TRUE(s == NULL);
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_EQ_INT(cap[0].source, GL_DEBUG_SOURCE_API);
    EXPECT_EQ_INT(cap[0].type, GL_DEBUG_TYPE_ERROR);
    EXPECT_EQ_INT(cap[0].severity, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(cap[0].id, GL_INVALID_ENUM);
    EXPECT_TRUE(strstr(cap[0].msg, "glGetString") != NULL);
    EXPECT_TRUE(strstr(cap[0].msg, "GL_INVALID_ENUM") != NULL);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* a second error while the first is still pending still produces a message */
    ncap = 0;
    glPopDebugGroup();                       /* STACK_UNDERFLOW */
    glPushDebugGroup(0x1234, 1, -1, "x");    /* INVALID_ENUM */
    EXPECT_EQ_INT(ncap, 2);
    EXPECT_EQ_INT(cap[0].id, GL_STACK_UNDERFLOW);
    EXPECT_EQ_INT(cap[1].id, GL_INVALID_ENUM);
    EXPECT_TRUE(strstr(cap[1].msg, "glPushDebugGroup") != NULL);
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);     /* first error wins */

    /* DEBUG_OUTPUT disabled: no messages */
    ncap = 0;
    glDisable(GL_DEBUG_OUTPUT);
    GLboolean b = GL_TRUE; glGetBooleanv(GL_DEBUG_OUTPUT, &b);
    EXPECT_EQ_INT(b, GL_FALSE);
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 1, GL_DEBUG_SEVERITY_HIGH, -1, "x");
    glGetString(0xDEAD);
    EXPECT_EQ_INT(ncap, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glEnable(GL_DEBUG_OUTPUT);
    glGetBooleanv(GL_DEBUG_OUTPUT, &b);
    EXPECT_EQ_INT(b, GL_TRUE);
}

static void insert(GLenum type, GLuint id, GLenum sev)
{
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, type, id, sev, -1, "m");
}

TEST(message_control)
{
    reset_debug();

    /* LOW is disabled by default */
    insert(GL_DEBUG_TYPE_OTHER, 1, GL_DEBUG_SEVERITY_LOW);
    insert(GL_DEBUG_TYPE_OTHER, 2, GL_DEBUG_SEVERITY_MEDIUM);
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_EQ_INT(cap[0].id, 2);

    /* enable LOW */
    ncap = 0;
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0, NULL, GL_TRUE);
    insert(GL_DEBUG_TYPE_OTHER, 3, GL_DEBUG_SEVERITY_LOW);
    EXPECT_EQ_INT(ncap, 1);

    /* disable a type for one source */
    ncap = 0;
    glDebugMessageControl(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, GL_DONT_CARE, 0, NULL, GL_FALSE);
    insert(GL_DEBUG_TYPE_MARKER, 4, GL_DEBUG_SEVERITY_HIGH);
    glDebugMessageInsert(GL_DEBUG_SOURCE_THIRD_PARTY, GL_DEBUG_TYPE_MARKER, 5, GL_DEBUG_SEVERITY_HIGH, -1, "m");
    insert(GL_DEBUG_TYPE_OTHER, 6, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(ncap, 2);
    EXPECT_EQ_INT(cap[0].id, 5);
    EXPECT_EQ_INT(cap[1].id, 6);

    /* id list: re-enable id 10 only */
    ncap = 0;
    GLuint ids[] = { 10 };
    glDebugMessageControl(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, GL_DONT_CARE, 1, ids, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    insert(GL_DEBUG_TYPE_MARKER, 10, GL_DEBUG_SEVERITY_LOW);   /* id rules ignore severity */
    insert(GL_DEBUG_TYPE_MARKER, 11, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_EQ_INT(cap[0].id, 10);

    /* disable an id */
    ncap = 0;
    GLuint ids2[] = { 6 };
    glDebugMessageControl(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, ids2, GL_FALSE);
    insert(GL_DEBUG_TYPE_OTHER, 6, GL_DEBUG_SEVERITY_HIGH);
    insert(GL_DEBUG_TYPE_OTHER, 7, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_EQ_INT(cap[0].id, 7);

    /* errors */
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, ids, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDebugMessageControl(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_HIGH, 1, ids, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDebugMessageControl(0x1234, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, -1, NULL, GL_TRUE);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* disabling everything also silences GL errors */
    ncap = 0;
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_FALSE);
    glGetString(0xDEAD);
    EXPECT_EQ_INT(ncap, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    reset_debug();
}

TEST(message_log_without_callback)
{
    reset_debug();
    glDebugMessageCallback(NULL, NULL);
    GLint n = -1;

    /* drain anything already logged */
    while (glGetDebugMessageLog(16, 0, NULL, NULL, NULL, NULL, NULL, NULL) > 0) {}
    glGetIntegerv(GL_DEBUG_LOGGED_MESSAGES, &n);
    EXPECT_EQ_INT(n, 0);
    glGetIntegerv(GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH, &n);
    EXPECT_EQ_INT(n, 0);

    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 100, GL_DEBUG_SEVERITY_HIGH, -1, "first");
    glDebugMessageInsert(GL_DEBUG_SOURCE_THIRD_PARTY, GL_DEBUG_TYPE_OTHER, 101, GL_DEBUG_SEVERITY_MEDIUM, -1, "second!");
    glGetString(0xDEAD);  /* error -> logged */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_EQ_INT(ncap, 0);

    glGetIntegerv(GL_DEBUG_LOGGED_MESSAGES, &n);
    EXPECT_EQ_INT(n, 3);
    glGetIntegerv(GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH, &n);
    EXPECT_EQ_INT(n, 6);

    /* too small a buffer: nothing returned, nothing removed */
    char buf[256];
    EXPECT_EQ_INT(glGetDebugMessageLog(1, 3, NULL, NULL, NULL, NULL, NULL, buf), 0);
    glGetIntegerv(GL_DEBUG_LOGGED_MESSAGES, &n);
    EXPECT_EQ_INT(n, 3);

    GLenum srcs[4], types[4], sevs[4]; GLuint ids[4]; GLsizei lens[4];
    GLuint got = glGetDebugMessageLog(2, sizeof buf, srcs, types, ids, sevs, lens, buf);
    EXPECT_EQ_INT(got, 2);
    EXPECT_EQ_INT(srcs[0], GL_DEBUG_SOURCE_APPLICATION);
    EXPECT_EQ_INT(types[0], GL_DEBUG_TYPE_MARKER);
    EXPECT_EQ_INT(ids[0], 100);
    EXPECT_EQ_INT(sevs[0], GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(lens[0], 6);
    EXPECT_STREQ(buf, "first");
    EXPECT_EQ_INT(srcs[1], GL_DEBUG_SOURCE_THIRD_PARTY);
    EXPECT_EQ_INT(ids[1], 101);
    EXPECT_EQ_INT(lens[1], 8);
    EXPECT_STREQ(buf + 6, "second!");

    got = glGetDebugMessageLog(4, sizeof buf, srcs, types, ids, sevs, lens, buf);
    EXPECT_EQ_INT(got, 1);
    EXPECT_EQ_INT(types[0], GL_DEBUG_TYPE_ERROR);
    EXPECT_EQ_INT(srcs[0], GL_DEBUG_SOURCE_API);
    EXPECT_EQ_INT(ids[0], GL_INVALID_ENUM);
    glGetIntegerv(GL_DEBUG_LOGGED_MESSAGES, &n);
    EXPECT_EQ_INT(n, 0);

    /* the log is bounded by GL_MAX_DEBUG_LOGGED_MESSAGES */
    GLint maxlog = 0;
    glGetIntegerv(GL_MAX_DEBUG_LOGGED_MESSAGES, &maxlog);
    EXPECT_TRUE(maxlog >= 1);
    for (int i = 0; i < maxlog + 10; i++)
        insert(GL_DEBUG_TYPE_OTHER, i, GL_DEBUG_SEVERITY_HIGH);
    glGetIntegerv(GL_DEBUG_LOGGED_MESSAGES, &n);
    EXPECT_EQ_INT(n, maxlog);
    /* oldest kept, newest discarded */
    got = glGetDebugMessageLog(1, 0, NULL, NULL, ids, NULL, NULL, NULL);
    EXPECT_EQ_INT(got, 1);
    EXPECT_EQ_INT(ids[0], 0);
    while (glGetDebugMessageLog(64, 0, NULL, NULL, NULL, NULL, NULL, NULL) > 0) {}

    glGetDebugMessageLog(1, -1, NULL, NULL, NULL, NULL, NULL, buf);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    GLint maxlen = 0;
    glGetIntegerv(GL_MAX_DEBUG_MESSAGE_LENGTH, &maxlen);
    EXPECT_TRUE(maxlen >= 1);
    reset_debug();
}

TEST(debug_groups)
{
    reset_debug();
    GLint depth = 0, maxdepth = 0;
    glGetIntegerv(GL_DEBUG_GROUP_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);
    glGetIntegerv(GL_MAX_DEBUG_GROUP_STACK_DEPTH, &maxdepth);
    EXPECT_TRUE(maxdepth >= 64);

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 77, -1, "frame");
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_DEBUG_GROUP_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 2);
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_EQ_INT(cap[0].type, GL_DEBUG_TYPE_PUSH_GROUP);
    EXPECT_EQ_INT(cap[0].source, GL_DEBUG_SOURCE_APPLICATION);
    EXPECT_EQ_INT(cap[0].id, 77);
    EXPECT_EQ_INT(cap[0].severity, GL_DEBUG_SEVERITY_NOTIFICATION);
    EXPECT_STREQ(cap[0].msg, "frame");

    /* per-group control: disable markers inside the group only */
    glDebugMessageControl(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, GL_DONT_CARE, 0, NULL, GL_FALSE);
    ncap = 0;
    insert(GL_DEBUG_TYPE_MARKER, 1, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(ncap, 0);

    /* nested group inherits the disabled state */
    glPushDebugGroup(GL_DEBUG_SOURCE_THIRD_PARTY, 78, 4, "pass-ignored");
    insert(GL_DEBUG_TYPE_MARKER, 2, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(ncap, 1);  /* only the push message */
    EXPECT_STREQ(cap[0].msg, "pass");
    glPopDebugGroup();
    EXPECT_EQ_INT(ncap, 2);
    EXPECT_EQ_INT(cap[1].type, GL_DEBUG_TYPE_POP_GROUP);
    EXPECT_EQ_INT(cap[1].source, GL_DEBUG_SOURCE_THIRD_PARTY);
    EXPECT_EQ_INT(cap[1].id, 78);
    EXPECT_STREQ(cap[1].msg, "pass");

    glPopDebugGroup();
    EXPECT_EQ_INT(ncap, 3);
    EXPECT_EQ_INT(cap[2].type, GL_DEBUG_TYPE_POP_GROUP);
    EXPECT_STREQ(cap[2].msg, "frame");

    /* outer group restored: markers enabled again */
    ncap = 0;
    insert(GL_DEBUG_TYPE_MARKER, 3, GL_DEBUG_SEVERITY_HIGH);
    EXPECT_EQ_INT(ncap, 1);
    EXPECT_NO_GL_ERROR();

    /* underflow / overflow */
    glPopDebugGroup();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);
    for (int i = 1; i < maxdepth; i++)
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, i, -1, "g");
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_DEBUG_GROUP_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, maxdepth);
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 999, -1, "too deep");
    EXPECT_GL_ERROR(GL_STACK_OVERFLOW);
    for (int i = 1; i < maxdepth; i++)
        glPopDebugGroup();
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_DEBUG_GROUP_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);

    glPushDebugGroup(GL_DEBUG_SOURCE_API, 1, -1, "x");
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    /* groups around real rendering work (mirrored onto the Metal encoder) */
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 5, -1, "clear pass");
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 9, GL_DEBUG_SEVERITY_NOTIFICATION, -1, "marker");
    glPopDebugGroup();
    glFinish();
    EXPECT_PIXEL(0, 0, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

static void check_label_roundtrip(GLenum identifier, GLuint name, const char *what)
{
    char buf[64]; GLsizei len = -1;
    glObjectLabel(identifier, name, -1, what);
    EXPECT_NO_GL_ERROR();
    glGetObjectLabel(identifier, name, sizeof buf, &len, buf);
    EXPECT_NO_GL_ERROR();
    EXPECT_STREQ(buf, what);
    EXPECT_EQ_INT(len, (long long)strlen(what));
}

TEST(object_labels)
{
    char buf[64]; GLsizei len = -1;

    GLuint buffer; glGenBuffers(1, &buffer); glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 64, NULL, GL_STATIC_DRAW);
    check_label_roundtrip(GL_BUFFER, buffer, "vertex buffer");

    /* length query, truncation, explicit length */
    glGetObjectLabel(GL_BUFFER, buffer, 0, &len, NULL);
    EXPECT_EQ_INT(len, 13);
    glGetObjectLabel(GL_BUFFER, buffer, 4, &len, buf);
    EXPECT_STREQ(buf, "ver");
    EXPECT_EQ_INT(len, 3);
    glObjectLabel(GL_BUFFER, buffer, 3, "abcdef");
    glGetObjectLabel(GL_BUFFER, buffer, sizeof buf, &len, buf);
    EXPECT_STREQ(buf, "abc");

    /* removal */
    glObjectLabel(GL_BUFFER, buffer, 0, NULL);
    glGetObjectLabel(GL_BUFFER, buffer, sizeof buf, &len, buf);
    EXPECT_EQ_INT(len, 0);
    EXPECT_STREQ(buf, "");
    EXPECT_NO_GL_ERROR();

    GLuint tex; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    check_label_roundtrip(GL_TEXTURE, tex, "albedo");

    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    check_label_roundtrip(GL_VERTEX_ARRAY, vao, "vao");
    glBindVertexArray(0);

    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    check_label_roundtrip(GL_SHADER, sh, "vs");
    GLuint prog = glCreateProgram();
    check_label_roundtrip(GL_PROGRAM, prog, "main program");

    GLuint fbo; glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    check_label_roundtrip(GL_FRAMEBUFFER, fbo, "gbuffer");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint rbo; glGenRenderbuffers(1, &rbo); glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    check_label_roundtrip(GL_RENDERBUFFER, rbo, "depth rb");
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    /* labels are per (identifier, name) */
    glGetObjectLabel(GL_TEXTURE, tex, sizeof buf, &len, buf);
    EXPECT_STREQ(buf, "albedo");

    /* errors */
    glObjectLabel(GL_BUFFER, 987654, -1, "nope");
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetObjectLabel(GL_TEXTURE, 987654, sizeof buf, &len, buf);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glObjectLabel(GL_QUERY, 987654, -1, "nope");
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glObjectLabel(0x1234, buffer, -1, "nope");
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetObjectLabel(GL_BUFFER, buffer, -1, &len, buf);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLint maxlabel = 0;
    glGetIntegerv(GL_MAX_LABEL_LENGTH, &maxlabel);
    EXPECT_TRUE(maxlabel >= 256);
    char *longlabel = malloc(maxlabel + 1);
    memset(longlabel, 'x', maxlabel); longlabel[maxlabel] = 0;
    glObjectLabel(GL_BUFFER, buffer, -1, longlabel);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    free(longlabel);

    /* a deleted name loses its label; a recreated object does not inherit it
       (buffers are not deleted here: mglDeleteBuffers currently writes out of
       bounds — buffers domain) */
    GLuint vao2; glGenVertexArrays(1, &vao2); glBindVertexArray(vao2);
    check_label_roundtrip(GL_VERTEX_ARRAY, vao2, "old");
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao2);
    glObjectLabel(GL_VERTEX_ARRAY, vao2, -1, "x");
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    /* drawing with labelled objects still works (labels pushed to Metal at commit) */
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    EXPECT_NO_GL_ERROR();

    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &vao);
    glDeleteFramebuffers(1, &fbo);
    glDeleteShader(sh);
    glDeleteProgram(prog);
    while (glGetError() != GL_NO_ERROR) {}
}

TEST(get_pointerv_and_reset)
{
    void *p = NULL;
    glDebugMessageCallback(debug_cb, &user_tag);
    glGetPointerv(GL_DEBUG_CALLBACK_FUNCTION, &p);
    EXPECT_TRUE(p == (void *)debug_cb);
    glGetPointerv(GL_DEBUG_CALLBACK_USER_PARAM, &p);
    EXPECT_TRUE(p == (void *)&user_tag);
    glGetPointerv(0x1234, &p);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDebugMessageCallback(NULL, NULL);
    glGetPointerv(GL_DEBUG_CALLBACK_FUNCTION, &p);
    EXPECT_TRUE(p == NULL);

    EXPECT_EQ_INT(glGetGraphicsResetStatus(), GL_NO_ERROR);
    GLint strat = 0;
    glGetIntegerv(GL_RESET_NOTIFICATION_STRATEGY, &strat);
    EXPECT_EQ_INT(strat, GL_NO_RESET_NOTIFICATION);
    EXPECT_NO_GL_ERROR();
    reset_debug();
}

TEST(strings)
{
    const char *version = (const char *)glGetString(GL_VERSION);
    EXPECT_TRUE(version && strncmp(version, "4.6", 3) == 0);
    EXPECT_STREQ(glGetString(GL_SHADING_LANGUAGE_VERSION), "4.60");
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    EXPECT_TRUE(renderer && strstr(renderer, "Apple") != NULL);   /* hardware name, Apple style */
    EXPECT_TRUE(glGetString(GL_VENDOR) != NULL);
    printf("  renderer: %s / %s / %s\n", glGetString(GL_VENDOR), renderer, version);
    EXPECT_NO_GL_ERROR();

    EXPECT_TRUE(glGetString(GL_EXTENSIONS) == NULL);   /* core profile: GetStringi only */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    EXPECT_TRUE(n > 10);
    int found_debug = 0, all_ok = 1;
    for (int i = 0; i < n; i++) {
        const char *e = (const char *)glGetStringi(GL_EXTENSIONS, i);
        if (!e || strncmp(e, "GL_", 3) != 0) all_ok = 0;
        else if (!strcmp(e, "GL_KHR_debug")) found_debug = 1;
    }
    EXPECT_TRUE(all_ok);
    EXPECT_TRUE(found_debug);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(glGetStringi(GL_EXTENSIONS, n) == NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glGetIntegerv(GL_NUM_SHADING_LANGUAGE_VERSIONS, &n);
    EXPECT_TRUE(n >= 1);
    EXPECT_STREQ(glGetStringi(GL_SHADING_LANGUAGE_VERSION, 0), "460 core");
    EXPECT_TRUE(glGetStringi(GL_VENDOR, 0) == NULL);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    GLint major = 0, minor = 0, mask = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    EXPECT_EQ_INT(major, 4);
    EXPECT_EQ_INT(minor, 6);
    EXPECT_EQ_INT(mask, GL_CONTEXT_CORE_PROFILE_BIT);
}

TEST(get_live_state)
{
    GLint i = -1; GLboolean b; GLfloat f[4]; GLdouble d[4]; GLint64 i64 = -1;

    GLuint bufs[2]; glGenBuffers(2, bufs);
    glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &i);
    EXPECT_EQ_INT(i, bufs[0]);
    glBindBuffer(GL_COPY_READ_BUFFER, bufs[1]);
    glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &i);
    EXPECT_EQ_INT(i, bufs[1]);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &i);
    EXPECT_EQ_INT(i, 0);

    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &i);
    EXPECT_EQ_INT(i, vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1]);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &i);
    EXPECT_EQ_INT(i, bufs[1]);
    glBindVertexArray(0);

    GLuint tex[2]; glGenTextures(2, tex);
    glActiveTexture(GL_TEXTURE3);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &i);
    EXPECT_EQ_INT(i, GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex[0]);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &i);
    EXPECT_EQ_INT(i, tex[0]);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &i);
    EXPECT_TRUE(i != (GLint)tex[0]);
    glBindTexture(GL_TEXTURE_3D, tex[1]);
    glGetIntegerv(GL_TEXTURE_BINDING_3D, &i);
    EXPECT_EQ_INT(i, tex[1]);

    GLuint fbo; glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &i);
    EXPECT_EQ_INT(i, fbo);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &i);
    EXPECT_EQ_INT(i, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    const char *vs = "#version 450 core\nlayout(location=0) in vec2 p; void main(){ gl_Position = vec4(p,0,1); }\n";
    const char *fs = "#version 450 core\nlayout(location=0) out vec4 c; void main(){ c = vec4(1); }\n";
    GLuint prog = mgl_make_program(vs, fs);
    if (prog) {
        glUseProgram(prog);
        glGetIntegerv(GL_CURRENT_PROGRAM, &i);
        EXPECT_EQ_INT(i, prog);
        glUseProgram(0);
        glGetIntegerv(GL_CURRENT_PROGRAM, &i);
        EXPECT_EQ_INT(i, 0);
    }

    /* colour state: float exact, int uses the normalized mapping, bool non-zero */
    glClearColor(0.25f, 0.5f, 0.0f, 1.0f);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, f);
    EXPECT_NEAR(f[0], 0.25, 1e-6); EXPECT_NEAR(f[1], 0.5, 1e-6); EXPECT_NEAR(f[3], 1.0, 1e-6);
    GLint ci[4]; glGetIntegerv(GL_COLOR_CLEAR_VALUE, ci);
    EXPECT_EQ_INT(ci[3], 2147483647);
    EXPECT_EQ_INT(ci[2], 0);
    EXPECT_NEAR((double)ci[1], 1073741824.0, 2.0);
    GLboolean cb[4]; glGetBooleanv(GL_COLOR_CLEAR_VALUE, cb);
    EXPECT_EQ_INT(cb[0], GL_TRUE); EXPECT_EQ_INT(cb[2], GL_FALSE);
    glGetDoublev(GL_COLOR_CLEAR_VALUE, d);
    EXPECT_NEAR(d[0], 0.25, 1e-6);

    glDepthFunc(GL_GEQUAL);
    glGetIntegerv(GL_DEPTH_FUNC, &i);
    EXPECT_EQ_INT(i, GL_GEQUAL);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_FALSE);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &b);
    EXPECT_EQ_INT(b, GL_FALSE);
    glDepthMask(GL_TRUE);

    glEnable(GL_DEPTH_TEST);
    glGetBooleanv(GL_DEPTH_TEST, &b);
    EXPECT_EQ_INT(b, GL_TRUE);
    glGetIntegerv(GL_DEPTH_TEST, &i);
    EXPECT_EQ_INT(i, 1);
    glDisable(GL_DEPTH_TEST);
    glGetBooleanv(GL_DEPTH_TEST, &b);
    EXPECT_EQ_INT(b, GL_FALSE);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glGetIntegerv(GL_BLEND_SRC_RGB, &i);
    EXPECT_EQ_INT(i, GL_SRC_ALPHA);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &i);
    EXPECT_EQ_INT(i, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFunc(GL_ONE, GL_ZERO);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &i);
    EXPECT_EQ_INT(i, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 17);
    glGetIntegerv(GL_PACK_ROW_LENGTH, &i);
    EXPECT_EQ_INT(i, 17);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);

    glGetIntegerv(GL_STENCIL_WRITEMASK, &i);
    EXPECT_EQ_INT(i, -1);       /* 0xFFFFFFFF as GLint */

    glGetFloatv(GL_LINE_WIDTH, f);   /* default 1.0 */
    EXPECT_NEAR(f[0], 1.0, 1e-6);

    glGetInteger64v(GL_MAX_TEXTURE_SIZE, &i64);
    EXPECT_TRUE(i64 >= 16384);
    glGetInteger64v(GL_MAX_SERVER_WAIT_TIMEOUT, &i64);
    EXPECT_TRUE(i64 > 0);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &i);
    EXPECT_TRUE(i >= 8);
    glGetIntegerv(GL_MAX_UNIFORM_LOCATIONS, &i);
    EXPECT_TRUE(i >= 1024);
    glGetIntegerv(GL_MAX_DEBUG_GROUP_STACK_DEPTH, &i);
    EXPECT_TRUE(i >= 64);
    GLfloat range[2] = {0, 0};
    glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, range);
    EXPECT_NEAR(range[0], 1.0, 1e-6);
    GLint dims[2]; glGetIntegerv(GL_MAX_VIEWPORT_DIMS, dims);
    EXPECT_TRUE(dims[0] >= 16384 && dims[1] >= 16384);

    GLint nfmt = 0; glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &nfmt);
    EXPECT_TRUE(nfmt >= 18);
    GLint fmts[64]; glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, fmts);
    EXPECT_EQ_INT(fmts[0], GL_COMPRESSED_RED_RGTC1);
    EXPECT_NO_GL_ERROR();

    GLint64 t0 = 0, t1 = 0;
    glGetInteger64v(GL_TIMESTAMP, &t0);
    glGetInteger64v(GL_TIMESTAMP, &t1);
    EXPECT_TRUE(t0 > 0 && t1 >= t0);

    /* unknown and compatibility-only pnames */
    glGetIntegerv(0xDEAD, &i);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetIntegerv(0x0B00 /* GL_CURRENT_COLOR */, &i);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetIntegerv(GL_UNIFORM_BUFFER_START, &i);   /* indexed-only */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(2, tex);
    glDeleteFramebuffers(1, &fbo);
    while (glGetError() != GL_NO_ERROR) {}
}

TEST(indexed_getters)
{
    GLint i = -1; GLint64 i64 = -1; GLboolean b4[4]; GLfloat f = -1; GLdouble d = -1;

    GLuint ubo; glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 1024, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 2, ubo, 256, 512);
    EXPECT_NO_GL_ERROR();
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 2, &i);
    EXPECT_EQ_INT(i, ubo);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_START, 2, &i64);
    EXPECT_EQ_INT(i64, 256);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, 2, &i64);
    EXPECT_EQ_INT(i64, 512);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 3, &i);
    EXPECT_EQ_INT(i, 0);
    glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &i);   /* generic binding point */
    EXPECT_EQ_INT(i, ubo);
    glGetFloati_v(GL_UNIFORM_BUFFER_START, 2, &f);
    EXPECT_NEAR(f, 256.0, 1e-6);
    glGetDoublei_v(GL_UNIFORM_BUFFER_SIZE, 2, &d);
    EXPECT_NEAR(d, 512.0, 1e-6);

    GLint maxubo = 0; glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &maxubo);
    EXPECT_TRUE(maxubo >= 1);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, maxubo, &i);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetIntegeri_v(GL_DEPTH_FUNC, 0, &i);     /* not indexed state */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &i);
    EXPECT_TRUE(i >= 1024);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &i);
    EXPECT_TRUE(i >= 65535);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 3, &i);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glGetIntegeri_v(GL_BLEND_SRC_RGB, 0, &i);
    EXPECT_EQ_INT(i, GL_SRC_ALPHA);
    glGetIntegeri_v(GL_BLEND_DST_RGB, 1, &i);
    EXPECT_EQ_INT(i, GL_ONE);
    glBlendFunc(GL_ONE, GL_ZERO);

    glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
    glGetBooleani_v(GL_COLOR_WRITEMASK, 0, b4);
    EXPECT_EQ_INT(b4[0], GL_TRUE); EXPECT_EQ_INT(b4[1], GL_FALSE);
    EXPECT_EQ_INT(b4[2], GL_TRUE); EXPECT_EQ_INT(b4[3], GL_FALSE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    GLint vp[4]; glGetIntegeri_v(GL_VIEWPORT, 0, vp);
    GLint vp0[4]; glGetIntegerv(GL_VIEWPORT, vp0);
    EXPECT_EQ_INT(vp[2], vp0[2]);
    EXPECT_NO_GL_ERROR();

    glGetIntegeri_v(GL_IMAGE_BINDING_NAME, 0, &i);
    EXPECT_EQ_INT(i, 0);
    glGetIntegeri_v(GL_IMAGE_BINDING_FORMAT, 0, &i);
    EXPECT_EQ_INT(i, GL_R8);
    EXPECT_NO_GL_ERROR();

    while (glGetError() != GL_NO_ERROR) {}
}

TEST(legacy_entry_points)
{
    reset_debug();
    glBegin(GL_TRIANGLES);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_TRUE(ncap >= 1 && cap[0].id == GL_INVALID_OPERATION && strstr(cap[0].msg, "glBegin"));
    glColor3f(1, 0, 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEnd();
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_EQ_INT(glGenLists(4), 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_EQ_INT(glIsList(1), GL_FALSE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glNewList(1, 0x1300 /* GL_COMPILE */);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    EXPECT_EQ_INT(glRenderMode(0x1C00 /* GL_RENDER */), 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glMatrixMode(0x1700);
    glLoadIdentity();
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLuint t = 1; GLboolean r = GL_TRUE;
    EXPECT_EQ_INT(glAreTexturesResident(1, &t, &r), GL_FALSE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glVertexP2ui(GL_INT_2_10_10_10_REV, 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glColorP4ui(GL_UNSIGNED_INT_2_10_10_10_REV, 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glClientActiveTexture(GL_TEXTURE0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLdouble v[4];
    glGetnMapdv(0x0D90, 0x0A00, sizeof v, v);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* context still healthy afterwards (state queries; rendering after this point in
       the full suite is affected by a renderer command-buffer bug unrelated to legacy calls) */
    GLint vao = -1;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
    EXPECT_EQ_INT(vao, 0);
    EXPECT_NO_GL_ERROR();

    /* getter errors name the entry point */
    ncap = 0;
    glGetIntegerv(0xDEAD, &vao);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_TRUE(ncap == 1 && strstr(cap[0].msg, "glGetIntegerv") != NULL);
    ncap = 0;
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 100000, &vao);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    EXPECT_TRUE(ncap == 1 && strstr(cap[0].msg, "glGetIntegeri_v") != NULL);
}

/* run last: MGL's fence implementation currently encodes a GPU event wait that
   can stall later command buffers (fence domain), so keep it away from pixel tests */
TEST(sync_object_labels)
{
    char buf[64]; GLsizei len = -1;
    /* sync objects via ObjectPtrLabel */
    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    EXPECT_TRUE(sync != NULL);
    glObjectPtrLabel(sync, -1, "fence");
    glGetObjectPtrLabel(sync, sizeof buf, &len, buf);
    EXPECT_STREQ(buf, "fence");
    EXPECT_EQ_INT(len, 5);
    EXPECT_NO_GL_ERROR();
    glObjectPtrLabel((void *)&len, -1, "bogus");
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glGetObjectPtrLabel(sync, -1, &len, buf);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteSync(sync);
    while (glGetError() != GL_NO_ERROR) {}
}

int main(void)
{
    RUN(callback_receives_insert);
    RUN(callback_receives_gl_errors);
    RUN(message_control);
    RUN(message_log_without_callback);
    RUN(debug_groups);
    RUN(object_labels);
    RUN(get_pointerv_and_reset);
    RUN(strings);
    RUN(get_live_state);
    RUN(indexed_getters);
    RUN(legacy_entry_points);
    RUN(sync_object_labels);
    return mgl_test_summary();
}
