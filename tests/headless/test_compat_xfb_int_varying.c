/* test_compat_xfb_int_varying — supervisor r108 regression (first-bad 4c3bb48,
 * the r86-r95 window): KHR-GL46.direct_state_access.queries_functional went
 * InternalError because its XFB program's captured-varying count came back 0:
 * the test links "#version 450 / out int xfb_result;" +
 * glTransformFeedbackVaryings("xfb_result", INTERLEAVED) and calls
 * glBeginTransformFeedback — which must succeed (GL 13.4: the program has an
 * output to capture).  MGL rejected it with GL_INVALID_OPERATION
 * ("count=0 mask=0"), glu::checkError threw and the case died.
 */
#include "mgl_test.h"

TEST(xfb_int_output_captures)
{
    const char *vs = "#version 450\n"
                     "out int xfb_result;\n"
                     "void main()\n"
                     "{\n"
                     "    switch(gl_VertexID)\n"
                     "    {\n"
                     "        case 0: gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); break;\n"
                     "        case 1: gl_Position = vec4( 1.0, 1.0, 0.0, 1.0); break;\n"
                     "        case 2: gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); break;\n"
                     "        case 3: gl_Position = vec4( 1.0,-1.0, 0.0, 1.0); break;\n"
                     "    }\n"
                     "    xfb_result = gl_VertexID;\n"
                     "}\n";
    const char *fs = "#version 450\nlayout(location=0) out vec4 o;\nvoid main(){ o = vec4(1); }\n";
    char log[512];
    GLint ok = 0;

    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    const char *names[] = { "xfb_result" };
    glTransformFeedbackVaryings(p, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }
    EXPECT_TRUE(ok);

    /* the varying must be reported as a program output with a location */
    GLuint tfo, buf, vao;
    glGenTransformFeedbacks(1, &tfo);
    glGenBuffers(1, &buf);
    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    glUseProgram(p);
    glBindVertexArray(vao);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    while (glGetError()) {}

    /* RED pre-fix: MGL rejected this with GL_INVALID_OPERATION (count=0) */
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDrawArrays(GL_POINTS, 0, 4);   /* POINTS: one capture record per vertex, independent of the (xfb-domain) per-primitive strip capture quirk */
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();

    /* the captured int values are the vertex ids */
    int out[16]; memset(out, 0x5a, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    printf("  xfb_result = %d %d %d %d\n", out[0], out[1], out[2], out[3]);
    EXPECT_EQ_INT(out[0], 0);
    EXPECT_EQ_INT(out[1], 1);
    EXPECT_EQ_INT(out[2], 2);
    EXPECT_EQ_INT(out[3], 3);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
}
int main(void){ RUN(xfb_int_output_captures); return mgl_test_summary(); }
