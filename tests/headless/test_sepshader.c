#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"

/* KHR-GL46.sepshaderobjs.StateInteraction — the failed-relink part:
 * a pipeline-referenced program that fails to relink must keep drawing with
 * its last successful binary (and keep the previously-set uniforms). */

static const char *kVs =
    "#version 450 core\n"
    "in highp vec4 a_position;\n"
    "uniform highp vec4 u_color;\n"
    "void main (void) { gl_Position = a_position; }\n";

static const char *kFsFrag =
    "#version 450 core\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "layout(location = 0) out mediump vec4 o_color;\n"
    "uniform highp vec4 u_color;\n"
    "void main (void) { o_color = u_color; }\n";

static void readcenter(unsigned char *px)
{
    glReadPixels(8, 8, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
}

TEST(StateInteractionFailedRelink)
{
    GLuint vao, buf, ibuf, pipe, progA, progB, vsh;
    const float pos[] = {-1,-1,0,1, 1,-1,0,1, -1,1,0,1, 1,1,0,1};
    const unsigned short idx[] = {0,1,2, 2,1,3};
    unsigned char px[4];

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &ibuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glViewport(0, 0, 16, 16);
    glClearColor(0, 0, 0, 0);

    progA = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &kVs);
    progB = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &kFsFrag);
    if (!progB) { MGL_FAIL("no programB"); goto done; }
    {
        char nm[] = "u_color";
        GLint loc = glGetUniformLocation(progB, nm);
        printf("u_color loc=%d\n", loc);
        glProgramUniform4f(progB, loc, 0.0f, 1.0f, 0.0f, 1.0f); /* green */
    }

    glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, progA);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, progB);
    glBindProgramPipeline(pipe);
    glUseProgram(0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
    glFinish();
    readcenter(px);
    printf("after first draw: %d %d %d %d (want green 0 255 0)\n", px[0], px[1], px[2], px[3]);
    EXPECT_TRUE(px[1] > 200 && px[0] < 50 && px[2] < 50);

    /* white */
    glProgramUniform4f(progB, glGetUniformLocation(progB, "u_color"), 1, 1, 1, 1);

    /* attach a fragment shader whose entry point is misspelled; relink fails */
    vsh = glCreateShader(GL_FRAGMENT_SHADER);
    {
        const char *bad = "#version 450 core\n"
                          "precision highp float;\n"
                          "layout(location=0) out vec4 o_color;\n"
                          "uniform highp vec4 u_color;\n"
                          "void niaM (void) { o_color = u_color; }\n";
        glShaderSource(vsh, 1, &bad, NULL);
        glCompileShader(vsh);
        glAttachShader(progB, vsh);
    }
    glLinkProgram(progB);
    {
        GLint st = 2;
        glGetProgramiv(progB, GL_LINK_STATUS, &st);
        printf("relink LINK_STATUS=%d\n", st);
        EXPECT_EQ_INT(GL_FALSE, st);
        GLenum e = glGetError();
        EXPECT_EQ_INT(GL_NO_ERROR, e);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
    glFinish();
    readcenter(px);
    printf("after failed relink draw: %d %d %d %d (want white)\n", px[0], px[1], px[2], px[3]);
    EXPECT_TRUE(px[0] > 200 && px[1] > 200 && px[2] > 200);

    glDeleteShader(vsh);
done:
    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(progA);
    glDeleteProgram(progB);
    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &ibuf);
    glDeleteVertexArrays(1, &vao);
}

int main(void) { RUN(StateInteractionFailedRelink); return mgl_test_summary(); }
