/* test_piq88 — r88 PIQ regressions: arrays of per-patch interface blocks
 * (dEQP-GL45-ES31 ...program_output.*.interface_blocks.patch_out.block_array.*)
 * and mat3x2 array leaf types (…type.…array.mat3x2).
 */
#include "mgl_test.h"
#include <string.h>

static int dump(GLuint prog, GLenum iface, char names[][80], GLint *sizes, GLint *locs, int cap)
{
    GLint n = 0;
    glGetProgramInterfaceiv(prog, iface, GL_ACTIVE_RESOURCES, &n);
    for (int i = 0; i < n && i < cap; i++) {
        GLsizei len = 0;
        glGetProgramResourceName(prog, iface, i, 80, &len, names[i]);
        GLenum props[3] = { GL_TYPE, GL_ARRAY_SIZE, GL_LOCATION };
        GLint vals[3]; GLsizei pl = 0;
        glGetProgramResourceiv(prog, iface, i, 3, props, 3, &pl, vals);
        if (sizes) sizes[i] = vals[1];
        if (locs) locs[i] = vals[2];
    }
    return n;
}

/* the exact CTS shader (resource_list.interface_blocks.patch_out.block_array.var):
 * a TCS with an ARRAY of per-patch output blocks — must link (SPIRV-Cross cannot
 * emit it as-is) and enumerate targetInstance[0..2].target as outputs. */
TEST(tcs_patch_block_array)
{
    const char *tcs =
        "#version 450\n"
        "layout(vertices=3) out;\n"
        "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
        "out gl_PerVertex { vec4 gl_Position; } gl_out[];\n"
        "patch out TargetInterface\n"
        "{\n"
        "    highp vec4 target;\n"
        "} targetInstance[3];\n"
        "void main() {\n"
        "    targetInstance[0].target = vec4(gl_in[0].gl_Position.y);\n"
        "    targetInstance[1].target = vec4(gl_in[0].gl_Position.y);\n"
        "    targetInstance[2].target = vec4(gl_in[0].gl_Position.y);\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "    gl_TessLevelOuter[0] = 2.8; gl_TessLevelOuter[1] = 2.8;\n"
        "    gl_TessLevelOuter[2] = 2.8; gl_TessLevelOuter[3] = 2.8;\n"
        "    gl_TessLevelInner[0] = 2.8; gl_TessLevelInner[1] = 2.8;\n"
        "}\n";
    GLuint p = glCreateShaderProgramv(GL_TESS_CONTROL_SHADER, 1, &tcs);
    GLint st = 0; glGetProgramiv(p, GL_LINK_STATUS, &st);
    if (!st) {
        GLchar log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  tcs link: %s", log);
    }
    EXPECT_EQ_INT(st, GL_TRUE);
    char names[16][80]; GLint sizes[16], locs[16];
    int n = dump(p, GL_PROGRAM_OUTPUT, names, sizes, locs, 16);
    int hits = 0;
    for (int i = 0; i < n; i++) {
        printf("  out[%d] %-30s size=%d loc=%d\n", i, names[i], sizes[i], locs[i]);
        /* block arrays have ONE resource per member, named <BlockName>.<member>
         * (no array index in instance names) */
        if (!strcmp(names[i], "TargetInterface.target")) { hits++; EXPECT_EQ_INT(sizes[i], 1); }
    }
    EXPECT_EQ_INT(hits, 1);
    glDeleteProgram(p);
}

/* matching consumer (TES, patch in block array) — the flattened interface must
 * still line up across a real pipeline */
TEST(patch_block_array_pipeline)
{
    const char *vs =
        "#version 450\n"
        "layout(location=0) in vec4 pos;\n"
        "out gl_PerVertex { vec4 gl_Position; };\n"
        "void main() { gl_Position = pos; }\n";
    const char *tcs =
        "#version 450\n"
        "layout(vertices=3) out;\n"
        "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
        "out gl_PerVertex { vec4 gl_Position; } gl_out[];\n"
        "patch out TargetInterface { highp vec4 target; } targetInstance[3];\n"
        "void main() {\n"
        "    targetInstance[0].target = vec4(0.25);\n"
        "    targetInstance[1].target = vec4(0.5);\n"
        "    targetInstance[2].target = vec4(0.75);\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
        "}\n";
    const char *tes =
        "#version 450\n"
        "layout(triangles, equal_spacing) in;\n"
        "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
        "patch in TargetInterface { highp vec4 target; } targetInstance[3];\n"
        "layout(location=0) out vec4 col;\n"
        "void main() {\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "    col = vec4(targetInstance[0].target.r, targetInstance[1].target.r, targetInstance[2].target.r, 1.0);\n"
        "}\n";
    const char *fs =
        "#version 450\n"
        "layout(location=0) in vec4 col;\n"
        "layout(location=0) out vec4 o;\n"
        "void main() { o = col; }\n";
    GLuint p = glCreateProgram();
    GLuint sv = glCreateShader(GL_VERTEX_SHADER), sc = glCreateShader(GL_TESS_CONTROL_SHADER),
           se = glCreateShader(GL_TESS_EVALUATION_SHADER);
    glShaderSource(sv, 1, &vs, NULL); glCompileShader(sv);
    glShaderSource(sc, 1, &tcs, NULL); glCompileShader(sc);
    glShaderSource(se, 1, &tes, NULL); glCompileShader(se);
    GLuint sf = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(sf, 1, &fs, NULL); glCompileShader(sf);
    glAttachShader(p, sv); glAttachShader(p, sc); glAttachShader(p, se); glAttachShader(p, sf);
    GLint st = 0;
    glLinkProgram(p); glGetProgramiv(p, GL_LINK_STATUS, &st);
    if (!st) { GLchar log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  link: %s", log); }
    EXPECT_EQ_INT(st, GL_TRUE);
    glDeleteShader(sv); glDeleteShader(sc); glDeleteShader(se); glDeleteShader(sf);
    if (st) {
        float quad[] = { -1,-1,0,1, 3,-1,0,1, -1,3,0,1 };
        GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(0);
        glUseProgram(p);
        glViewport(0, 0, 200, 200);
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_PATCHES, 0, 3);
        glFinish();
        float px[4] = {-2,-2,-2,-2}; glReadPixels(8, 8, 1, 1, GL_RGBA, GL_FLOAT, px);
        printf("  col=(%f,%f,%f,%f) want (0.25,0.5,0.75,1)\n", px[0], px[1], px[2], px[3]);
        EXPECT_NEAR(px[0], 0.25f, 0.005f);
        EXPECT_NEAR(px[1], 0.50f, 0.005f);
        EXPECT_NEAR(px[2], 0.75f, 0.005f);
        EXPECT_NEAR(px[3], 1.00f, 0.005f);
        glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    }
    glDeleteProgram(p);
}

/* …type.separable_tess_eval.array.mat3x2: TES output array of mat3x2 must
 * report GL_FLOAT_MAT3X2 (per element) */
TEST(tes_out_mat3x2_array_type)
{
    const char *tes =
        "#version 450\n"
        "layout(triangles, equal_spacing) in;\n"
        "in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
        "layout(location=0) out highp mat3x2 target[2];\n"
        "void main() {\n"
        "    gl_Position = gl_in[0].gl_Position;\n"
        "    target[0] = mat3x2(1.0);\n"
        "    target[1] = mat3x2(2.0);\n"
        "}\n";
    GLuint p = glCreateShaderProgramv(GL_TESS_EVALUATION_SHADER, 1, &tes);
    GLint st = 0; glGetProgramiv(p, GL_LINK_STATUS, &st);
    if (!st) { GLchar log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  tes link: %s", log); }
    EXPECT_EQ_INT(st, GL_TRUE);
    char names[8][80]; GLint sizes[8], locs[8];
    int n = dump(p, GL_PROGRAM_OUTPUT, names, sizes, locs, 8);
    int hit = 0;
    for (int i = 0; i < n; i++) {
        printf("  out[%d] %-20s size=%d loc=%d\n", i, names[i], sizes[i], locs[i]);
        if (!strcmp(names[i], "target[0]")) {
            GLenum t = 0; GLsizei l = 0;
            glGetProgramResourceiv(p, GL_PROGRAM_OUTPUT, i, 1, (GLenum[]){GL_TYPE}, 1, &l, (GLint*)&t);
            printf("  type=0x%x (MAT3X2=0x%x)\n", t, GL_FLOAT_MAT3x2);
            EXPECT_EQ_INT(t, GL_FLOAT_MAT3x2);
            EXPECT_EQ_INT(sizes[i], 2);
            hit++;
        }
    }
    EXPECT_EQ_INT(hit, 1);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(tcs_patch_block_array);
    RUN(patch_block_array_pipeline);
    RUN(tes_out_mat3x2_array_type);
    return 0;
}
