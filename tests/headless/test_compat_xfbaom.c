/* compat r96: xfb capture of an array-of-matrix output with an explicit
 * xfb_offset (KHR-GL46.enhanced_layouts.xfb_explicit_location_mat2_*): the
 * captured columns must be packed by the std430-style rules the CTS expects
 * (mat2 = 2 x vec2 = 16 bytes, array element stride = element size), not the
 * std140-padded layout glslang emits for the uniform block. */
#include "mgl_test.h"

TEST(xfb_aom_mat2)
{
    GLuint prog, vao, tfb, query, buf;
    const char *vs =
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout (location = 0, xfb_offset = 0) out mat2 goku[3];\n"
        "layout(std140, binding = 0) uniform Goku { mat2 uni_goku[3]; };\n"
        "void main() { goku = uni_goku; }\n";
    GLint linked = 0, stride = 0;
    float uniform_data[24];
    float expect[24];
    float got[24];

    /* std140 uniform block: mat2 members padded to 32 bytes each */
    for (int i = 0; i < 6; i++)
    {
        uniform_data[i * 4 + 0] = (float)(i * 4 + 1);
        uniform_data[i * 4 + 1] = (float)(i * 4 + 2);
        uniform_data[i * 4 + 2] = 0.0f;
        uniform_data[i * 4 + 3] = 0.0f;
    }
    /* packed xfb expectation: mat2 columns back to back */
    for (int i = 0; i < 6; i++)
    {
        expect[i * 2 + 0] = uniform_data[i * 4 + 0];
        expect[i * 2 + 1] = uniform_data[i * 4 + 1];
    }

    prog = glCreateProgram();
    {
        GLuint sh = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(sh, 1, &vs, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
    }
    glProgramParameteri(prog, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    EXPECT_TRUE(linked == 1);
    if (linked != 1) return;


    {
        GLenum props[1] = { GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE };
        GLint rstride = -1;
        glGetProgramResourceiv(prog, GL_TRANSFORM_FEEDBACK_BUFFER, 0, 1, props, 1, NULL, &rstride);
        EXPECT_EQ_INT(rstride, 48);
    }
    {
        static const char *fs_src = "#version 430 core\nout vec4 color;\nvoid main() { color = vec4(1.0); }\n";
        GLuint pipe = 0, fs;
        glUseProgram(0);   /* a current glUseProgram program overrides the pipeline */
        GLint fl = 0;
        fs = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fs_src);
        fprintf(stderr, "[t] prog=%u fs=%u useprogram=%u pipeline=%u\n", prog, fs, 0, pipe);
        glGetProgramiv(fs, GL_LINK_STATUS, &fl);
        EXPECT_EQ_INT(fl, 1);
        glGenProgramPipelines(1, &pipe);
        glBindProgramPipeline(pipe);
        glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, prog);
        glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fs);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof got, expect, GL_DYNAMIC_COPY);

    /* fill the std140 uniform block */
    {
        GLuint ubo;
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof uniform_data, uniform_data, GL_STATIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    }

    glGenQueries(1, &query);
    glGenTransformFeedbacks(1, &tfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glDrawArrays(GL_POINTS, 0, 2);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glEndTransformFeedback();
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof got, got);

    for (int v = 0; v < 2; v++)
    for (int i = 0; i < 12; i++)
    {
        int idx = v * 12 + i;
        if (got[idx] != expect[i])
        {
            mgl_test_checks++;
            MGL_FAIL("xfb v%d[%d]: got %.1f expect %.1f", v, i, got[idx], expect[i]);
        }
    }
    if (0)
    for (int i = 0; i < 24; i++)
    {
        if (got[i] != expect[i])
        {
            mgl_test_checks++;
            MGL_FAIL("xfb[%d]: got %.1f expect %.1f", i, got[i], expect[i]);
            break;
        }
    }
}

TEST(xfb_simple_float)
{
    GLuint prog, vao, tfb, buf;
    const char *vs =
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout (location = 0, xfb_offset = 0) out float goku;\n"
        "layout(std140, binding = 0) uniform Goku { float uni_goku; };\n"
        "void main() { goku = uni_goku; }\n";
    GLint linked = 0;
    float got[4] = {0};

    prog = glCreateProgram();
    {
        GLuint sh = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(sh, 1, &vs, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    EXPECT_EQ_INT(linked, 1);
    if (linked != 1) return;

    glUseProgram(prog);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    {
        GLuint ubo;
        float uni = 7.5f;
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, 16, &uni, GL_STATIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    }
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof got, got, GL_DYNAMIC_COPY);
    glGenTransformFeedbacks(1, &tfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glDrawArrays(GL_POINTS, 0, 2);
    glEndTransformFeedback();
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof got, got);
    EXPECT_TRUE(got[0] == 7.5f);
    EXPECT_TRUE(got[1] == 7.5f);
}

int main(void)
{
    RUN(xfb_simple_float);
    RUN(xfb_aom_mat2);
    return 0;
}
