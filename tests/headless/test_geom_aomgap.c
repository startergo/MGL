/* geom r107: stage-IO structures with array-of-matrix members that are written
 * or read through a helper function (glslang lowers an `out`/`inout` parameter
 * to a copy-out temporary, so the member accesses live inside the callee).
 *
 * introspect_spirv.c's arrays-of-matrices flattening rewrote the STRUCT TYPE
 * (member m[2] -> m_0, m_1) but never rewrote the callee's access chains: the
 * `pf` marking loop skipped OpFunctionParameter entirely (its `wc < 4` guard
 * vs the instruction's word count 3), so the chains kept the element index and
 * the shifted member index -> broken MSL and a failed link.
 *
 * Each case links, renders, and is verified with glReadPixels; the xfb cases
 * verify the captured bytes of a struct written through a function.
 */
#include "mgl_test.h"

static GLuint mk_prog(const char *vs_src, const char *fs_src, const char *tag)
{
    GLuint prog = glCreateProgram();
    GLuint sh[2] = { glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER) };
    const char *src[2] = { vs_src, fs_src };
    for (int i = 0; i < 2; i++)
    {
        glShaderSource(sh[i], 1, &src[i], NULL);
        glCompileShader(sh[i]);
        GLint ok = 0;
        glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
        if (ok != 1)
        {
            char log[2048]; GLsizei len = 0;
            glGetShaderInfoLog(sh[i], sizeof log, &len, log);
            fprintf(stderr, "  [%s] %s compile failed:\n%s\n", tag, i ? "FS" : "VS", log);
        }
        glAttachShader(prog, sh[i]);
        glDeleteShader(sh[i]);
    }
    glLinkProgram(prog);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [%s] LINK failed:\n%s\n", tag, log);
    }
    return prog;
}

/* draw one big triangle covering the viewport; the flat varying carries the
 * same value everywhere, so every pixel must equal `expect` */
static void draw_full_triangle(GLuint prog, const GLubyte expect[4], const char *tag)
{
    glUseProgram(prog);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 32, 32);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte px[4];
    mgl_read_rgba(8, 8, px);
    for (int c = 0; c < 4; c++)
    {
        mgl_test_checks++;
        if (abs((int)px[c] - (int)expect[c]) > 2)
            MGL_FAIL("%s: channel %d == %d, expected ~%d", tag, c, px[c], expect[c]);
    }
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

/* out-struct written through a helper function: mat member m[2] split into
 * m_0/m_1 by the flattening, callee chains must follow */
TEST(aom_func_out_render)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat out Inner o;\n"
        "void fill(out Inner x, float s) {\n"
        "    x.m[0] = mat2x3(1,2,3,4,5,6) * s;\n"
        "    x.m[1] = mat2x3(7,8,9,10,11,12) * s;\n"
        "    x.f = s;\n"
        "}\n"
        "void main() {\n"
        "    vec2 p = vec2(float((gl_VertexID << 1) & 2) - 1.0, float(gl_VertexID & 2) - 1.0);\n"
        "    gl_Position = vec4(p, 0.0, 1.0);\n"
        "    fill(o, 0.5);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat in Inner i;\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = vec4(i.f, i.m[0][0].x, i.m[1][1].y * 0.1, 1.0); }\n";
    /* fill(o, 0.5): m[0] = col0 (0.5,1,1.5) col1 (2,2.5,3); m[1] col1 = (5,5.5,6);
     * f = 0.5 -> color = (0.5, 0.5, 0.55, 1.0) */
    GLubyte expect[4] = { 128, 128, 140, 255 };
    draw_full_triangle(mk_prog(vs, fs, "aom_func_out_render"), expect, "aom_func_out_render");
}

/* same shape, but the callee is reached through a second function and reads the
 * surviving member after the flattened group (f shifted to member index 2) */
TEST(aom_func_out_nested_render)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat out Inner o;\n"
        "void inner_fill(out Inner x, float s) {\n"
        "    x.m[0] = mat2x3(2,4,6,8,10,12) * s;\n"
        "    x.m[1] = mat2x3(1,3,5,7,9,11) * s;\n"
        "    x.f = 0.25;\n"
        "}\n"
        "void outer_fill(out Inner x, float s) { inner_fill(x, s); }\n"
        "void main() {\n"
        "    vec2 p = vec2(float((gl_VertexID << 1) & 2) - 1.0, float(gl_VertexID & 2) - 1.0);\n"
        "    gl_Position = vec4(p, 0.0, 1.0);\n"
        "    outer_fill(o, 0.5);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat in Inner i;\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = vec4(i.f, i.m[0][1].y * 0.1, i.m[1][0].z * 0.1, 1.0); }\n";
    /* m[0] = (1,2,3|2,4,6) * ... s=0.5: col1 = (4,5,6) -> .y=5 -> 0.5;
     * m[1] col0 = (0.5,1.5,2.5) -> .z = 2.5 -> 0.25; f = 0.25 */
    GLubyte expect[4] = { 64, 128, 64, 255 };
    draw_full_triangle(mk_prog(vs, fs, "aom_func_out_nested_render"), expect, "aom_func_out_nested_render");
}

/* the callee also READS the struct (inout): the copy-in loads the flattened
 * value, the callee's read chains must follow too */
TEST(aom_func_inout_render)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat out Inner o;\n"
        "float scale_of(Inner x) { return x.m[1][0].z; }\n"
        "void seed(out Inner x, float z) {\n"
        "    x.m[0] = mat2x3(0);\n"
        "    x.m[1] = mat2x3(0,0,z,0,0,0);\n"
        "    x.f = 0.0;\n"
        "}\n"
        "void fill(inout Inner x) {\n"
        "    x.m[0] = mat2x3(1,2,3,4,5,6) * scale_of(x);\n"
        "    x.m[1] = mat2x3(7,8,9,10,11,12);\n"
        "    x.f = x.m[1][1].x;\n"
        "}\n"
        "void main() {\n"
        "    vec2 p = vec2(float((gl_VertexID << 1) & 2) - 1.0, float(gl_VertexID & 2) - 1.0);\n"
        "    gl_Position = vec4(p, 0.0, 1.0);\n"
        "    Inner tmp;\n"
        "    seed(tmp, 9.0);\n"
        "    o = tmp;\n"
        "    fill(o);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat in Inner i;\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = vec4(i.m[0][0].x * 0.1, i.f * 0.1, i.m[1][1].y * 0.1, 1.0); }\n";
    /* scale_of reads m[1][0].z = 9 -> m[0] = (9,18,27|36,45,54): col0.x = 9 -> 0.9;
     * m[1] col1 = (10,11,12) -> .y = 11 -> 1.1 clamps to 1.0; f = m[1][1].x = 10 -> 1.0 */
    GLubyte expect[4] = { 230, 255, 255, 255 };
    draw_full_triangle(mk_prog(vs, fs, "aom_func_inout_render"), expect, "aom_func_inout_render");
}

/* array-of-struct output whose struct carries the array-of-matrix member */
TEST(aom_array_of_struct_render)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat out Inner o[2];\n"
        "void main() {\n"
        "    vec2 p = vec2(float((gl_VertexID << 1) & 2) - 1.0, float(gl_VertexID & 2) - 1.0);\n"
        "    gl_Position = vec4(p, 0.0, 1.0);\n"
        "    o[0].m[0] = mat2x3(1,2,3,4,5,6);\n"
        "    o[0].m[1] = mat2x3(7,8,9,10,11,12);\n"
        "    o[0].f = 0.5;\n"
        "    o[1].m[0] = mat2x3(13,14,15,16,17,18);\n"
        "    o[1].m[1] = mat2x3(19,20,21,22,23,24);\n"
        "    o[1].f = 0.125;\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location=0) flat in Inner i[2];\n"
        "layout(location=0) out vec4 color;\n"
        "void main() {\n"
        "    vec3 c = i[1].m[0] * vec2(0.25, 0.5);\n"
        "    color = vec4(i[0].f, i[1].f, c.x + c.y + c.z, 1.0);\n"
        "}\n";
    /* i[1].m[0] col0=(13,14,15) col1=(16,17,18); * vec2(0.25,0.5) = 13*.25+16*.5 = 11.75 -> clamp 1.0 */
    GLubyte expect[4] = { 128, 32, 255, 255 };
    draw_full_triangle(mk_prog(vs, fs, "aom_array_of_struct_render"), expect, "aom_array_of_struct_render");
}

/* transform feedback of a struct written through a function: the captured
 * column of m[0] and the trailing f must carry the written values */
TEST(aom_func_out_xfb)
{
    static const char *vs =
        "#version 460 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(xfb_buffer=0, xfb_offset=0, location=0) flat out Inner o;\n"
        "void fill(out Inner x, float s) {\n"
        "    x.m[0] = mat2x3(1,2,3,4,5,6) * s;\n"
        "    x.m[1] = mat2x3(7,8,9,10,11,12) * s;\n"
        "    x.f = 0.5;\n"
        "}\n"
        "void main() {\n"
        "    gl_Position = vec4(float(gl_VertexID & 1) * 4.0 - 1.0, float(gl_VertexID >> 1) * 4.0 - 1.0, 0.0, 1.0);\n"
        "    fill(o, 1.0);\n"
        "}\n";
    GLuint prog = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &vs, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    glAttachShader(prog, sh);
    glDeleteShader(sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (ok != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [aom_func_out_xfb] LINK failed:\n%s\n", log);
        glDeleteProgram(prog);
        EXPECT_EQ_INT(ok, 1);
        return;
    }

    /* where did f land?  the xfb packing of `Inner { mat2x3 m[2]; float f; }`
     * packs the matrix columns back to back (12 bytes each, 4 columns = 48
     * bytes, matching GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE = 52) */
    GLint f_off = 48;
    GLint stride = 0;
    {
        GLenum props[1] = { GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE };
        glGetProgramResourceiv(prog, GL_TRANSFORM_FEEDBACK_BUFFER, 0, 1, props, 1, NULL, &stride);
        EXPECT_EQ_INT(stride, 52);
    }

    GLuint vao = 0, buf = 0, tfb = 0, q = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4 * (size_t)(stride ? stride : 128), NULL, GL_STREAM_READ);
    glGenTransformFeedbacks(1, &tfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glGenQueries(1, &q);

    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 4);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    GLint n = 0;
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &n);
    EXPECT_EQ_INT(n, 4);

    float *got = malloc(4 * (size_t)(stride ? stride : 128));
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 4 * (size_t)(stride ? stride : 128), got);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    /* vertex 0: fill(o, 1.0) -> m[0] col0 = (1,2,3) at bytes 0,4,8; f at 2*32 */
    for (int c = 0; c < 3; c++)
    {
        mgl_test_checks++;
        if (fabsf(got[c] - (float)(c + 1)) > 0.002f)
            MGL_FAIL("aom_func_out_xfb: m[0] col0[%d] == %f, expected %d", c, got[c], c + 1);
    }
    mgl_test_checks++;
    if (fabsf(got[f_off / 4] - 0.5f) > 0.002f)
        MGL_FAIL("aom_func_out_xfb: f == %f, expected 0.5 (offset %d)", got[f_off / 4], f_off);
    free(got);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfb);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &buf);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(aom_func_out_render);
    RUN(aom_func_out_nested_render);
    RUN(aom_func_inout_render);
    RUN(aom_array_of_struct_render);
    RUN(aom_func_out_xfb);
    return mgl_test_summary();
}
