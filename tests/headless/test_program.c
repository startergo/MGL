#include "mgl_test.h"
TEST(compute_ssbo)
{
    const char *cs = "#version 450 core\n"
        "layout(local_size_x=4) in;\n"
        "layout(std430, binding=0) buffer B { uint v[]; };\n"
        "void main(){ v[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 3u + 1u; }\n";
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, NULL); glCompileShader(s);
    GLuint p = glCreateProgram(); glAttachShader(p, s); glLinkProgram(p);
    GLuint buf; glGenBuffers(1, &buf); glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    static GLuint z[2048]; glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof z, z, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glUseProgram(p);
    glDispatchCompute(2,1,1);
    glFinish();
    GLuint out[9]; glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof out, out);
    for (int i=0;i<8;i++) EXPECT_EQ_INT(out[i], i*3+1);
    EXPECT_EQ_INT(out[8], 0);
}
int main(void){ RUN(compute_ssbo); return mgl_test_summary(); }
