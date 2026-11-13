/* test_compat_default_texture_degenerate — supervisor r107 regression:
 *
 * The CTS framework's between-cases state reset (gluStateReset.cpp
 * resetStateGLCore) binds texture name 0 to every target and legally writes
 * DEGENERATE storage into the target's default texture object (compat r103
 * Mesa parity makes name 0 a real object):
 *   glTexImage2D(GL_TEXTURE_RECTANGLE/2D/..., 0, GL_RGBA, 0, 0, ...)  (empties)
 *   glTexImage2DMultisample(..., GL_RGBA8, 0, 0, GL_TRUE)
 *   glTexBuffer(GL_TEXTURE_BUFFER, GL_R8, 0)
 * Afterwards the default objects of several targets carry a level with ZERO
 * extent (num_levels 1, dims 0x0x0).  Every following draw then re-attempted
 * Metal texture creation on them: createMTLTextureFromGLTexture returns NULL
 * ("Primary texture creation returned NULL" + "emergency fallback texture
 * (size: 0x0x0)" ~400k lines/case, 13-60 s instead of 2 s — the
 * texture_swizzle api_errors -> smoke_access snowball).
 *
 * After the fix: a degenerate default object is never realized; sampling it
 * stays incomplete-black (GL 8.17), no GL errors, and a draw burst stays
 * FAST (the per-draw Metal attempt + NSLog was ~100x slower).
 */
#include "mgl_test.h"
#include <time.h>

static GLuint mkprog_sampler_cubemap(void)
{
    const char *vs = "#version 450\n"
                     "const vec2 q[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));\n"
                     "void main(){ gl_Position = vec4(q[gl_VertexID],0,1); }\n";
    const char *fs = "#version 450\n"
                     "layout(location=0) out vec4 o;\n"
                     "uniform sampler2DMSArray s;\n"
                     "void main(){ o = texelFetch(s, ivec3(0, 0, 0), 0); }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    GLint ok = 0; char log[512];
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }
    return p;
}

TEST(degenerate_default_object_draws_fast_and_black)
{
    /* 1. the state-reset sequence on the default objects (GL-legal on a
     *    desktop core context: name 0 is the target's default object).
     *    glTexImage{2,3}DMultisample with ZERO extent is what
     *    gluStateReset.cpp actually issues between test cases. */
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glTexImage2D(GL_TEXTURE_CUBE_MAP, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    while (glGetError()) {}
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_RGBA8, 0, 0, GL_TRUE);
    while (glGetError()) {}
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 1, GL_RGBA8, 0, 0, 0, GL_TRUE);
    while (glGetError()) {}
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R8, 0);
    while (glGetError()) {}

    GLint w = 12345;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 0);          /* the level exists, emptied (GL 8.6) */

    /* 2. draw with a samplerCube reading the degenerate default object */
    GLuint prog = mkprog_sampler_cubemap();
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "s"), 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0);   /* the degenerate default object */

    GLuint vao, fbo, rb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 32, 32);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 32, 32);

    /* 3. RED (pre-fix): the burst re-attempts Metal texture creation on every
     *    draw — hundreds of NSLog lines and failed creations per draw make
     *    500 draws take tens of seconds; post-fix this is microseconds each.
     *    Budget: 5 s for 500 draws (generous for a debug build). */
    clock_t t0 = clock();
    for (int i = 0; i < 500; i++)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    EXPECT_NO_GL_ERROR();
    printf("  500 degenerate-default draws took %.3f s\n", secs);
    EXPECT_TRUE(secs < 5.0);

    /* 4. the sample stays incomplete-black (0,0,0,1) */
    GLubyte px[4] = { 9, 9, 9, 9 };
    glReadPixels(16, 16, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    printf("  sampled rgba(%d,%d,%d,%d)\n", px[0], px[1], px[2], px[3]);
    EXPECT_EQ_INT(px[0], 0);
    EXPECT_EQ_INT(px[1], 0);
    EXPECT_EQ_INT(px[2], 0);
    /* RED pre-fix: the emergency 0x0x0 texture binds and Metal samples
     * (0,0,0,0); the incomplete fallback is GL 8.17's (0,0,0,1) */
    EXPECT_EQ_INT(px[3], 255);

    glActiveTexture(GL_TEXTURE0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(void)
{
    RUN(degenerate_default_object_draws_fast_and_black);
    return mgl_test_summary();
}
