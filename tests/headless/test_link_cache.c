#include <stdio.h>
#include <string.h>
#include "glcorearb.h"
#include "mgl_test.h"

/* The glslang compile cache shares one compiled glslang_shader_t between GL
 * shader objects compiled from the same source.  glslang's LINKER writes
 * program-wide uniform locations back into the stage intermediate, so when a
 * SECOND program borrows the SAME cached stage (fresh FS compile of the same
 * source) together with a different partner, glslang reports
 * "INTERNAL ERROR: Uniform location overlaps across stages"
 * (KHR-GL46.shaders.indexing.varying_array.* regressions). */

static const char *kVsA =
    "#version 330\n"
    "in highp vec4 a_position;\n"
    "uniform mediump int ui_zero, ui_one;\n"
    "out mediump float var[2];\n"
    "void main()\n"
    "{\n"
    "    gl_Position = a_position;\n"
    "    var[ui_zero] = 0.25;\n"
    "    var[ui_one] = 0.5;\n"
    "}\n";
static const char *kVsB =
    "#version 330\n"
    "in highp vec4 a_position;\n"
    "in highp vec4 a_coords;\n"
    "uniform mediump int ui_two, ui_three;\n"
    "out mediump float var[2];\n"
    "void main()\n"
    "{\n"
    "    gl_Position = a_position;\n"
    "    var[ui_two]   = float(a_coords);\n"
    "    var[ui_three] = float(a_coords) * 0.5;\n"
    "}\n";
static const char *kFs =
    "#version 330\n"
    "precision mediump int;\n"
    "layout(location = 0) out mediump vec4 o_color;\n"
    "uniform int ui_four;\n"
    "in mediump float var[2];\n"
    "void main()\n"
    "{\n"
    "   mediump float res = 0.0;\n"
    "   for (int i = 0; i < ui_four; i++)\n"
    "       res += var[i];\n"
    "   o_color = vec4(res, 0.0, 0.0, 1.0);\n"
    "}\n";

static GLuint mk(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log); MGL_FAIL("compile: %s", log); }
    return sh;
}

static GLuint link(GLuint vs, GLuint fs, const char *label)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    char log[2048] = {0};
    glGetProgramInfoLog(p, sizeof log - 1, NULL, log);
    printf("%s: link=%d %s\n", label, ok, log);
    return p;
}

TEST(GlslangCacheSharedStageRelink)
{
    GLuint vsA = mk(GL_VERTEX_SHADER, kVsA);
    GLuint vsB = mk(GL_VERTEX_SHADER, kVsB);
    GLuint fs1 = mk(GL_FRAGMENT_SHADER, kFs);

    GLuint p1 = link(vsA, fs1, "prog1 (vsA+fs1)");
    { GLint ok = 0; glGetProgramiv(p1, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(GL_TRUE, ok); }

    /* same FS source compiled again: cache hit -> the SAME glslang intermediate,
     * already location-mapped by prog1's link */
    GLuint fs2 = mk(GL_FRAGMENT_SHADER, kFs);
    GLuint p2 = link(vsB, fs2, "prog2 (vsB + fs2 [cache-shared after prog1 link])");
    { GLint ok = 0; glGetProgramiv(p2, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(GL_TRUE, ok); }

    /* and once more with the first VS */
    GLuint fs3 = mk(GL_FRAGMENT_SHADER, kFs);
    GLuint p3 = link(vsA, fs3, "prog3 (vsA + fs3 [cache-shared])");
    { GLint ok = 0; glGetProgramiv(p3, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(GL_TRUE, ok); }

    glDeleteProgram(p1); glDeleteProgram(p2); glDeleteProgram(p3);
    glDeleteShader(vsA); glDeleteShader(vsB);
    glDeleteShader(fs1); glDeleteShader(fs2); glDeleteShader(fs3);
}

int main(void) { RUN(GlslangCacheSharedStageRelink); return mgl_test_summary(); }
