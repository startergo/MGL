/*
 * test_compat_gpiv.c — TRANSFORM_FEEDBACK_VARYING queries (glGetProgramResourceiv)
 * for API-selected matrix varyings captured from a TES (geom/tess resolver path).
 * Mirrors KHR-Single-GL46.enhanced_layouts.xfb_get_program_resource_api_<mat*>:
 * INTERLEAVED with gl_SkipComponents/gl_NextBuffer; expected offsets/strides per
 * GL 4.6 13.4 (a matCxR takes C*R components, packed column by column).
 */
#include "mgl_test.h"

static const char *vs =
    "#version 430 core\n"
    "void main() { gl_Position = vec4(0.0); }\n";

static const char *gs =
    "#version 430 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n";

static const char *tcs =
    "#version 430 core\n"
    "layout(vertices = 1) out;\n"
    "void main() { gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; }\n";

static void check_case(const char *type, int cols, int rows, int scalar)
{
    char src[2048];
    int sz = cols * rows * scalar;   /* matrix byte size */
    snprintf(src, sizeof src,
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout(triangles, equal_spacing) in;\n"
        "const uint type_size = %d;\n"
        "layout (xfb_buffer = 1, xfb_stride = 4 * type_size) out;\n"
        "layout (xfb_buffer = 0, xfb_offset = 0 * type_size) out %s b0_v0;\n"
        "layout (xfb_buffer = 0, xfb_offset = 1 * type_size) out %s b0_v1;\n"
        "layout (xfb_buffer = 0, xfb_offset = 3 * type_size) out %s b0_v3;\n"
        "layout (xfb_buffer = 1, xfb_offset = 1 * type_size) out %s b1_v1;\n"
        "void main() { gl_Position = gl_in[0].gl_Position;\n"
        "  b0_v0 = %s(1.0); b0_v1 = %s(2.0); b0_v3 = %s(3.0); b1_v1 = %s(4.0); }\n",
        sz, type, type, type, type, type, type, type, type);

    GLuint v = glCreateShader(GL_VERTEX_SHADER), t = glCreateShader(GL_TESS_EVALUATION_SHADER);
    GLuint g = glCreateShader(GL_GEOMETRY_SHADER), c = glCreateShader(GL_TESS_CONTROL_SHADER);
    const char *ts = src, *gs_ = gs, *tcs_ = tcs;
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(t, 1, &ts, NULL); glCompileShader(t);
    glShaderSource(g, 1, &gs_, NULL); glCompileShader(g);
    glShaderSource(c, 1, &tcs_, NULL); glCompileShader(c);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, c); glAttachShader(p, t); glAttachShader(p, g);
    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    const char *vars[] = { "b0_v0", "b0_v1", "gl_SkipComponents4", "gl_SkipComponents4",
                           "b0_v3", "gl_NextBuffer", "gl_SkipComponents4", "gl_SkipComponents4",
                           "b1_v1", "gl_SkipComponents4", "gl_SkipComponents4", "gl_SkipComponents4",
                           "gl_SkipComponents4" };
    glTransformFeedbackVaryings(p, 13, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (!lk)
    {
        char log[4096]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link %s: %.300s", type, log);
        mgl_test_checks++;
        glDeleteProgram(p); glDeleteShader(v); glDeleteShader(t); glDeleteShader(g); glDeleteShader(c);
        return;
    }

    {
        struct { const char *name; int off, buf; } want[4] = {
            { "b0_v0", 0, 0 }, { "b0_v1", sz, 0 }, { "b0_v3", 3 * sz, 0 }, { "b1_v1", sz, 1 }
        };
        for (int i = 0; i < 4; i++)
        {
            GLuint idx = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, want[i].name);
            mgl_test_checks++;
            if (idx == GL_INVALID_INDEX) { MGL_FAIL("%s: no TF index (%s)", want[i].name, type); continue; }
            GLint off = -1, bidx = -1;
            glGetProgramResourceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, idx, 1,
                                   (GLenum[]){GL_OFFSET}, 1, NULL, &off);
            glGetProgramResourceiv(p, GL_TRANSFORM_FEEDBACK_VARYING, idx, 1,
                                   (GLenum[]){GL_TRANSFORM_FEEDBACK_BUFFER_INDEX}, 1, NULL, &bidx);
            if (off != want[i].off) { mgl_test_checks++; MGL_FAIL("%s offset == %d, expected %d (%s)",
                                                 want[i].name, off, want[i].off, type); }
            if (bidx != want[i].buf) { mgl_test_checks++; MGL_FAIL("%s buffer == %d, expected %d (%s)",
                                                  want[i].name, bidx, want[i].buf, type); }
        }
        {
            GLint s0 = -1, s1 = -1;
            GLuint b0 = 0, b1 = 1;
            glGetProgramResourceiv(p, GL_TRANSFORM_FEEDBACK_BUFFER, b0, 1,
                                   (GLenum[]){GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE}, 1, NULL, &s0);
            glGetProgramResourceiv(p, GL_TRANSFORM_FEEDBACK_BUFFER, b1, 1,
                                   (GLenum[]){GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE}, 1, NULL, &s1);
            if (s0 != 4 * sz) { mgl_test_checks++; MGL_FAIL("buffer0 stride == %d, expected %d (%s)", s0, 4 * sz, type); }
            if (s1 != 4 * sz) { mgl_test_checks++; MGL_FAIL("buffer1 stride == %d, expected %d (%s)", s1, 4 * sz, type); }
        }
    }
out:
    glDeleteProgram(p); glDeleteShader(v); glDeleteShader(t); glDeleteShader(g); glDeleteShader(c);
}

TEST(gpiv_matrices)
{
    check_case("mat2", 2, 2, 4);
    check_case("mat3", 3, 3, 4);
    check_case("mat4", 4, 4, 4);
    check_case("mat2x4", 2, 4, 4);
    check_case("mat4x2", 4, 2, 4);
    check_case("dmat2", 2, 2, 8);
    check_case("dmat3", 3, 3, 8);
    check_case("dmat4", 4, 4, 8);
    check_case("dmat2x4", 2, 4, 8);
    check_case("dmat4x2", 4, 2, 8);
}

int main(void)
{
    RUN(gpiv_matrices);
    return 0;
}
