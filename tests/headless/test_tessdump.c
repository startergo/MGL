#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"
static const char *kVs =
    "#version 450 core\nlayout(location=0) in vec4 a_pos;\nlayout(location=1) in vec4 a_col;\n"
    "layout(location=0) out vec4 v_col;\nvoid main() { gl_Position = a_pos; v_col = a_col; }\n";
static const char *kFs =
    "#version 450 core\nlayout(location=0) in vec4 f_col;\nlayout(location=0) out vec4 o_col;\n"
    "void main() { o_col = f_col; }\n";
static const char *kTcs =
    "#version 450 core\nlayout(vertices=4) out;\nlayout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0]=2; gl_TessLevelOuter[1]=2; gl_TessLevelOuter[2]=2; gl_TessLevelOuter[3]=2; gl_TessLevelInner[0]=2; gl_TessLevelInner[1]=2; }\n}\n";
static const char *kTes =
    "#version 450 core\nlayout(quads, equal_spacing, ccw) in;\nlayout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x*gl_in[0].gl_Position + gl_TessCoord.y*gl_in[1].gl_Position + gl_TessCoord.z*gl_in[2].gl_Position;\n"
    "  f_col = gl_TessCoord.x*tc_col[0] + gl_TessCoord.y*tc_col[1] + gl_TessCoord.z*tc_col[2]; }\n";
TEST(Dump)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[4] = {
        { GL_VERTEX_SHADER, kVs }, { GL_TESS_CONTROL_SHADER, kTcs },
        { GL_TESS_EVALUATION_SHADER, kTes }, { GL_FRAGMENT_SHADER, kFs } };
    for (int i = 0; i < 4; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL); glCompileShader(so); glAttachShader(p, so); glDeleteShader(so);
    }
    glLinkProgram(p);
    GLint link = 0; glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);
    static const float pos[4][4] = { { -1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, -1, 0, 1 }, { -1, -1, 0, 1 } };
    static const float col[4][4] = { { 1,0,0,1 }, { 0,1,0,1 }, { 0,0,1,1 }, { 1,1,0,1 } };
    GLuint vao, b[2];
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]); glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL); glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]); glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL); glEnableVertexAttribArray(1);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 4);
    EXPECT_NO_GL_ERROR();
    glFinish();
    for (int y = 15; y >= 0; y--) {
        for (int x = 0; x < 16; x++) {
            unsigned char px[4];
            glReadPixels(x * 8 + 4, y * 8 + 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            char c = px[0] > 128 ? 'R' : px[1] > 128 ? 'G' : px[2] > 128 ? 'B' : (px[0]+px[1]+px[2] > 60 ? 'm' : '.');
            putchar(c);
        }
        putchar('\n');
    }
}
int main(void)
{
    RUN(Dump);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}
