/* r76 Task B guard: compute shaders with uniform blocks / SSBOs compile and
 * link in plain harness programs.  geom r148 reported "too many storage
 * qualifiers" from glslang for a CS declaring a uniform block or SSBO outside
 * the CTS harness (pre-existing at their merge tip, with the ACB-base
 * over-count retry unfixed).  On the r76 tip every spelling compiles — the
 * sweep below keeps it that way. */
#include "mgl_test.h"
#include <string.h>

struct csvar { const char *tag; const char *src; };

static const struct csvar kVariants[] = {
    { "ubo",
      "#version 460 core\nlayout(local_size_x=4) in;\n"
      "layout(std140, binding=0) uniform U { vec4 a; };\n"
      "layout(std430, binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "ssbo-pair",
      "#version 460 core\nlayout(local_size_x=4) in;\n"
      "layout(std430, binding=0) readonly buffer In { float i[]; };\n"
      "layout(std430, binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = i[gl_GlobalInvocationID.x] * 2.0; }\n" },
    { "anon-ubo",
      "#version 460 core\nlayout(local_size_x=4) in;\nlayout(std140, binding=0) uniform { vec4 a; };\n"
      "layout(std430, binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "nobind",
      "#version 460 core\nlayout(local_size_x=4) in;\nlayout(std140) uniform U { vec4 a; };\n"
      "layout(std430) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "v450",
      "#version 450 core\nlayout(local_size_x=4) in;\nlayout(std140, binding=0) uniform U { vec4 a; };\n"
      "layout(std430, binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "shared-layout",
      "#version 460 core\nlayout(local_size_x=4) in;\nlayout(shared, binding=0) uniform U { vec4 a; };\n"
      "layout(std430, binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "packed-layout",
      "#version 460 core\nlayout(local_size_x=4) in;\nlayout(packed, binding=0) uniform U { vec4 a; };\n"
      "layout(std430, binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "default-layout",
      "#version 460 core\nlayout(local_size_x=4) in;\nlayout(binding=0) uniform U { vec4 a; };\n"
      "layout(binding=1) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "no-layout-qual",
      "#version 460 core\nlayout(local_size_x=4) in;\nuniform U { vec4 a; };\nbuffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "bind-first",
      "#version 460 core\nlayout(local_size_x=4) in;\nlayout(binding=0, std140) uniform U { vec4 a; };\n"
      "layout(binding=1, std430) writeonly buffer Out { float o[]; };\n"
      "void main(){ o[gl_GlobalInvocationID.x] = a.x; }\n" },
    { "image-only",
      "#version 460 core\nlayout(local_size_x=4) in;\n"
      "layout(binding=0, rgba32f) writeonly uniform image2D img;\n"
      "void main(){ imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(1)); }\n" },
};

TEST(ComputeBlockDeclarations)
{
    for (unsigned k = 0; k < sizeof kVariants / sizeof kVariants[0]; k++)
    {
        GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
        GLint ok = 0;
        char log[512] = {0};
        GLsizei n = 0;
        while (glGetError() != GL_NO_ERROR) {}
        glShaderSource(cs, 1, &kVariants[k].src, NULL);
        glCompileShader(cs);
        glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            glGetShaderInfoLog(cs, sizeof log - 1, &n, log);
            printf("  %-15s FAIL: %.150s\n", kVariants[k].tag, log);
        }
        EXPECT_EQ_INT(ok, GL_TRUE);
        glDeleteShader(cs);
    }

    /* the resources-max fence shape: 12 SSBO + 8 UBO + 8 image + 8 ACB in one
     * compute shader -- the mix geom r148 could not compile in a harness */
    {
        static char big[8192];
        int w = 0;
        GLuint cs, p;
        GLint link = 0;
        w += snprintf(big + w, sizeof big - w, "#version 460 core\nlayout(local_size_x=1) in;\n");
        for (int i = 0; i < 8; i++)
            w += snprintf(big + w, sizeof big - w,
                          "layout(std140, binding=%d) uniform U%d { vec4 u%d; };\n", i, i, i);
        for (int i = 0; i < 12; i++)
            w += snprintf(big + w, sizeof big - w,
                          "layout(std430, binding=%d) buffer B%d { vec4 b%d; };\n", i, i, i);
        for (int i = 0; i < 8; i++)
            w += snprintf(big + w, sizeof big - w,
                          "layout(binding=%d, rgba32f) writeonly uniform image2D t%d;\n", i, i);
        for (int i = 0; i < 8; i++)
            w += snprintf(big + w, sizeof big - w,
                          "layout(binding=%d) uniform atomic_uint a%d;\n", i, i);
        w += snprintf(big + w, sizeof big - w,
                      "void main(){ b0.x = u0.x; imageStore(t0, ivec2(0), vec4(atomicCounter(a0))); }\n");
        cs = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(cs, 1, &(const char *){ big }, NULL);
        glCompileShader(cs);
        {
            GLint ok = 0;
            glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
            EXPECT_EQ_INT(ok, GL_TRUE);
        }
        p = glCreateProgram();
        glAttachShader(p, cs);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &link);
        EXPECT_EQ_INT(link, GL_TRUE);
        glDeleteProgram(p);
        glDeleteShader(cs);
    }

    /* and an SSBO pipeline actually computes in a plain harness program */
    {
        const char *src = kVariants[1].src;
        GLuint cs = glCreateShader(GL_COMPUTE_SHADER), p;
        GLint link = 0, ok = 0;
        glShaderSource(cs, 1, &src, NULL);
        glCompileShader(cs);
        glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        if (!ok) return;
        p = glCreateProgram();
        glAttachShader(p, cs);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &link);
        EXPECT_EQ_INT(link, GL_TRUE);
        if (!link) { glDeleteProgram(p); glDeleteShader(cs); return; }
        glUseProgram(p);
        {
            GLuint inb, outb;
            static const float in_data[4] = { 1, 2, 3, 4 };
            float out_data[4] = { 0 };
            glGenBuffers(1, &inb);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, inb);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof in_data, in_data, GL_STATIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inb);
            glGenBuffers(1, &outb);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, outb);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof out_data, NULL, GL_STATIC_READ);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outb);
            glDispatchCompute(1, 1, 1);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            glFinish();
            glBindBuffer(GL_COPY_READ_BUFFER, outb);
            glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof out_data, out_data);
            EXPECT_NEAR(out_data[0], 2.0f, 1e-5);
            EXPECT_NEAR(out_data[3], 8.0f, 1e-5);
            glDeleteBuffers(1, &inb);
            glDeleteBuffers(1, &outb);
        }
        glDeleteProgram(p);
        glDeleteShader(cs);
    }
}

int main(void) { RUN(ComputeBlockDeclarations); return mgl_test_summary(); }
