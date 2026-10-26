/*
 * test_reserved.c — GLSL 4.60 §3.7 reserved words / keywords as declared names.
 *
 * KHR-GL46.CommonBugs.CommonBug_ReservedNames compiles thousands of shaders
 * that declare an identifier with a reserved word or keyword in every legal
 * declaration slot and requires every one to be REJECTED.  MGL's rejection is
 * the union of glslang's own (it rejects nearly all natively) and the targeted
 * checks for the slots where glslang is lenient (cts_glsl_compat.c:
 * reject_reserved_declarators — subroutine type/uniform names, 'invariant' as
 * a name, 'subroutine' as a block name).  This test pins the UNION: every
 * word x slot below must fail to compile, and a set of legal keyword uses
 * (constructors, loops, qualifiers, subroutine declarations, legacy GLSL)
 * must still compile — the guard against false positives.
 */
#include "mgl_test.h"
#include "mgl_compat.h"

/* compile `src`; returns 1 when the compile was REJECTED */
static int compile_rejected(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    glDeleteShader(sh);
    while (glGetError() != GL_NO_ERROR) {}
    return ok ? 0 : 1;
}

/* declaration templates: %s is the offending identifier */
static const char *const kSlots[] = {
    /* 0 global uniform */
    "#version 460 core\nuniform float %s;\nvoid main(){}\n",
    /* 1 function parameter */
    "#version 460 core\nvoid f(float %s){}\nvoid main(){}\n",
    /* 2 struct field */
    "#version 460 core\nstruct S { int %s; };\nvoid main(){}\n",
    /* 3 struct type name */
    "#version 460 core\nstruct %s { int a; };\nvoid main(){}\n",
    /* 4 interface block name */
    "#version 460 core\nlayout(std140) uniform %s { int a; } inst;\nvoid main(){}\n",
    /* 5 interface block instance */
    "#version 460 core\nlayout(std140) uniform B { int a; } %s;\nvoid main(){}\n",
    /* 6 subroutine TYPE name */
    "#version 460 core\nsubroutine void %s(inout vec4);\nvoid main(){}\n",
    /* 7 subroutine uniform name */
    "#version 460 core\nsubroutine void t1();\nsubroutine uniform t1 %s;\nvoid main(){}\n",
    /* 8 subroutine function name */
    "#version 460 core\nsubroutine void t1();\nsubroutine(t1) void %s() {}\nvoid main(){}\n",
    /* 9 local variable */
    "#version 460 core\nvoid main(){ int %s = 1; }\n",
    /* 10 vertex out (varying) */
    "#version 460 core\nout vec4 %s;\nvoid main(){}\n",
};

/* GLSL 4.60 reserved words (never legal identifiers) */
static const char *const kReserved[] = {
    "common", "partition", "active", "asm", "class", "union", "enum", "typedef",
    "template", "this", "resource", "goto", "inline", "noinline", "public",
    "static", "extern", "external", "interface", "long", "short", "half",
    "fixed", "unsigned", "superp", "input", "output", "hvec2", "hvec3", "hvec4",
    "fvec2", "fvec3", "fvec4", "sampler3DRect", "filter", "sizeof", "cast",
    "namespace", "using",
};

/* GLSL 4.60 keywords (legal tokens, never legal identifiers) */
static const char *const kKeywords[] = {
    "attribute", "const", "uniform", "varying", "buffer", "shared", "coherent",
    "volatile", "restrict", "readonly", "writeonly", "atomic_uint", "layout",
    "centroid", "flat", "smooth", "noperspective", "patch", "sample", "break",
    "continue", "do", "for", "while", "switch", "case", "default", "if", "else",
    "subroutine", "in", "out", "inout", "float", "double", "int", "void", "bool",
    "true", "false", "invariant", "precise", "discard", "return",
    "mat2", "mat3", "mat4", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
    "bvec2", "bvec3", "bvec4", "uvec2", "uvec3", "uvec4", "dvec2", "dvec3",
    "dvec4", "uint", "lowp", "mediump", "highp", "precision",
    "sampler1D", "sampler2D", "sampler3D", "samplerCube", "sampler2DShadow",
    "samplerCubeShadow", "sampler2DArray", "isampler2D", "usampler2D",
    "sampler2DRect", "sampler2DRectShadow", "samplerBuffer", "sampler2DMS",
    "sampler2DMSArray", "samplerCubeArray", "samplerCubeArrayShadow",
    "image1D", "image2D", "image3D", "imageCube", "image2DArray", "imageBuffer",
    "image2DMS", "imageCubeArray", "struct",
};

static char kSrc[512];

TEST(reserved_words_rejected)
{
    int nslots = (int)(sizeof kSlots / sizeof kSlots[0]);
    for (unsigned w = 0; w < sizeof kReserved / sizeof kReserved[0]; w++)
    {
        /* every reserved word in every slot (the CTS covers them all) */
        for (int s = 0; s < nslots; s++)
        {
            snprintf(kSrc, sizeof kSrc, kSlots[s], kReserved[w]);
            if (!compile_rejected(GL_VERTEX_SHADER, kSrc))
                MGL_FAIL("'%s' accepted in slot %d", kReserved[w], s);
        }
    }
    while (glGetError() != GL_NO_ERROR) {}
}

TEST(keywords_rejected)
{
    int nslots = (int)(sizeof kSlots / sizeof kSlots[0]);
    for (unsigned w = 0; w < sizeof kKeywords / sizeof kKeywords[0]; w++)
    {
        for (int s = 0; s < nslots; s++)
        {
            snprintf(kSrc, sizeof kSrc, kSlots[s], kKeywords[w]);
            if (!compile_rejected(GL_VERTEX_SHADER, kSrc))
                MGL_FAIL("'%s' accepted in slot %d", kKeywords[w], s);
        }
    }
    while (glGetError() != GL_NO_ERROR) {}
}

/* the exact slots where glslang is still lenient — the ones cts_glsl_compat.c
 * must catch on its own (subroutine type/uniform names, 'invariant' as a
 * declarator/argument name, 'subroutine' as an interface-block name) */
TEST(lenient_slots_rejected)
{
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\n"
        "subroutine void attribute(inout vec4);\nvoid main(){}\n"));
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\n"
        "subroutine void t1();\nsubroutine uniform void while;\nvoid main(){}\n"));
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\n"
        "subroutine (t1) while;\nvoid main(){}\n"));
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\n"
        "vec4 invariant;\nvoid main(){}\n"));
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\n"
        "void f(vec4 invariant){}\nvoid main(){}\n"));
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\n"
        "uniform subroutine { int a; } inst;\nvoid main(){}\n"));
}

/* legal keyword USES must keep compiling — the false-positive guard
 * (the compat scanner once rejected a constructor call after an identifier
 * operand in a real glmark2 vertex shader) */
TEST(legal_keyword_uses_accepted)
{
    /* constructor call after an identifier operand (jellyfish.vert shape) */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nuniform mat4 uWorldInvTranspose;\n"
        "void main(){ gl_Position = uWorldInvTranspose * vec4(1.0, 2.0, 3.0, 1.0); }\n"));
    /* keyword flow control */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nvoid main(){ int i = 0; while (i < 4) { i++; if (i == 2) break; else continue; } }\n"));
    /* qualifiers in their qualifier position */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nlayout(location = 0) in vec4 p; flat out vec4 v;\n"
        "centroid out vec2 c; noperspective out float f;\n"
        "void main(){ v = p; c = p.xy; f = p.x; gl_Position = p; }\n"));
    /* sample / patch qualifiers (4.2+) */
    EXPECT_TRUE(!compile_rejected(GL_FRAGMENT_SHADER,
        "#version 460 core\nsample in vec4 v;\nout vec4 o;\n"
        "void main(){ o = v; }\n"));
    /* subroutine declaration used correctly */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nsubroutine void t1();\nsubroutine uniform t1 su;\n"
        "subroutine(t1) void impl() { }\nvoid main(){ su(); }\n"));
    /* uniform block + struct */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nstruct S { int a; };\nlayout(std140) uniform B { S s; } inst;\n"
        "void main(){ gl_Position = vec4(inst.s.a); }\n"));
    /* invariant qualifier (not a name here) */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nout vec4 c;\ninvariant c;\nvoid main(){ c = vec4(1.0); }\n"));
    /* mediump/precision legal at 460 */
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 460 core\nprecision mediump float;\nuniform lowp float x;\n"
        "void main(){ gl_Position = vec4(x); }\n"));
}

/* legacy GLSL (compat profile) keeps its own keywords: attribute/varying are
 * ACTIVE there, so a #version 120 shader must still compile on a compat
 * context — the reserved scan must not fire on pre-400 sources */
TEST(legacy_gls_unaffected)
{
    void *ctx = mglCompatCreateContext();
    EXPECT_TRUE(ctx != NULL);
    if (!ctx)
        return;
    EXPECT_TRUE(!compile_rejected(GL_VERTEX_SHADER,
        "#version 120\nattribute vec2 a; varying vec4 v; uniform sampler2D t;\n"
        "void main(){ v = vec4(a, 0.0, 1.0); gl_Position = v; }\n"));
    EXPECT_TRUE(!compile_rejected(GL_FRAGMENT_SHADER,
        "#version 120\nvarying vec4 v; void main(){ gl_FragColor = v; }\n"));
    /* NB 'sample'/'patch'/'centroid' as identifiers in legacy GLSL: legal per
     * GLSL 1.20, but MGL's compat translator re-emits the body at #version 460
     * where those words are keywords — documented gap (glmark2 does not use
     * them; pre-existing behaviour, independent of the reserved-word scan) */
    /* ... but reserved words still are not, even in 120 */
    EXPECT_TRUE(compile_rejected(GL_VERTEX_SHADER,
        "#version 120\nuniform float common;\nvoid main(){ gl_Position = vec4(common); }\n"));
    mglMakeContextCurrent(mglCoreContext());
}

/* integer mix(genIType, genIType, genBType) exists only from GLSL 4.50 on
 * (or with GL_EXT_shader_integer_mix, which MGL does not advertise); glslang's
 * builtin mix() accepts the integer forms at every version, so a #version 330
 * shader using them must be rejected before glslang (cts_glsl_compat.c:
 * reject_integer_mix, tess r24 — its headless test was not on the branch) */
TEST(integer_mix_version_rule)
{
    /* 330: rejected */
    EXPECT_TRUE(compile_rejected(GL_FRAGMENT_SHADER,
        "#version 330\nout vec4 o;\nvoid main(){ ivec2 a = mix(ivec2(1), ivec2(2), bvec2(true)); o = vec4(a, 0, 1); }\n"));
    /* 4.50+: legal */
    EXPECT_TRUE(!compile_rejected(GL_FRAGMENT_SHADER,
        "#version 460\nout vec4 o;\nvoid main(){ ivec2 a = mix(ivec2(1), ivec2(2), bvec2(true)); o = vec4(a, 0, 1); }\n"));
    /* float mix at every version stays legal */
    EXPECT_TRUE(!compile_rejected(GL_FRAGMENT_SHADER,
        "#version 330\nout vec4 o;\nvoid main(){ float a = mix(1.0, 2.0, true); o = vec4(a, 0, 0, 1); }\n"));
}

int main(void)
{
    RUN(reserved_words_rejected);
    RUN(keywords_rejected);
    RUN(lenient_slots_rejected);
    RUN(legal_keyword_uses_accepted);
    RUN(legacy_gls_unaffected);
    RUN(integer_mix_version_rule);
    return mgl_test_summary();
}
