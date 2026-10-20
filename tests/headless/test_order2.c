#include "mgl_test.h"
TEST(order2)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(std140, binding = 1) uniform U { vec4 u; };\n"
        "layout(std430, binding = 2) readonly buffer S { float s[]; };\n"
        "uniform sampler2D tex;\n"
        "uniform float f;\n"
        "out vec4 c;\n"
        "void main(){ gl_Position = vec4(pos,0,1); c = u + vec4(f) + texture(tex,pos) + vec4(s[0]); }\n";
    static const char *fs =
        "#version 460 core\n"
        "in vec4 c;\n"
        "layout(location=0) out vec4 o;\n"
        "void main(){ o = c; }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint vao, vbo, tex;
    const float quad[] = { -1, -1, 3, -1, -1, 3 };
    GLint ok = 0;
    char udata[16] = {0}, sdata[16] = {0};
    GLuint ubo, ssbo;
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    { char lg[4096] = {0}; GLsizei n=0; glGetShaderInfoLog(v, 4096, &n, lg); if (lg[0]) printf("VS LOG: %s\n", lg); }
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    { char lg[4096] = {0}; GLsizei n=0; glGetProgramInfoLog(p, 4096, &n, lg); if (lg[0]) printf("LINK LOG: %s\n", lg); }
    EXPECT_TRUE(ok == GL_TRUE);
    glUseProgram(p);
    glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 16, udata, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo);
    glGenBuffers(1, &ssbo); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, sdata, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, udata);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
}
int main(void) { RUN(order2); return mgl_test_summary(); }
