/* test_clipdist — KHR-GL46.gl_spirv.spirv_validation_builtin_variable_decorations
 * validation 1 (clipNegative): a triangle with gl_ClipDistance[0] = -1.0 and
 * GL_CLIP_DISTANCE0 enabled must be fully clipped (black); +1.0 must render. */
#include "mgl_test.h"

static const char *vs =
    "#version 450\n"
    "layout (location = 0) in vec3 position;\n"
    "layout (location = 1) out vec4 vColor;\n"
    "void main()\n"
    "{\n"
    "    gl_PointSize = 10.0f;\n"
    "    gl_Position = vec4(position.x, position.y + 0.3 * gl_InstanceID, position.z, 1.0);\n"
    "    gl_ClipDistance[0] = %s;\n"
    "    gl_CullDistance[0] = 1.0;\n"
    "    vColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
    "}\n";
static const char *fs =
    "#version 450\n"
    "layout (location = 1) in vec4 vColor;\n"
    "layout (location = 0) out vec4 fColor;\n"
    "void main() { fColor = vColor; }\n";

static int draw_and_read(const char *dist)
{
    char src[1024];
    snprintf(src, sizeof src, vs, dist);
    GLuint p = mgl_make_program(src, fs);
    if (!p) return -1;
    glUseProgram(p);
    float v[] = { -1, -1, 0, 1, -1, 0, 0, 1, 0 };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_CLIP_DISTANCE0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_CLIP_DISTANCE0);
    GLubyte px[4];
    mgl_read_rgba(32, 32, px);
    glDeleteProgram(p);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    return px[0];
}

TEST(clip_negative_culls)
{
    int r = draw_and_read("-1.0");
    EXPECT_EQ_INT(r, 0);
}

TEST(clip_positive_renders)
{
    int r = draw_and_read("1.0");
    EXPECT_EQ_INT(r, 255);
}

int main(void)
{
    RUN(clip_positive_renders);
    RUN(clip_negative_culls);   /* last: its SPIR-V is what MGL_PREPARED_DUMP captures */
    return mgl_test_summary();
}
