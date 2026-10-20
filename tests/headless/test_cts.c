/* test_cts.c — regressions for Khronos GL CTS (KHR-GL46) failure clusters fixed in MGL.
 * Each test mirrors one CTS case (named in a comment) that used to fail.
 */
#include "mgl_test.h"

static GLuint make_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";
        glGetShaderInfoLog(s, sizeof log - 1, NULL, log);
        fprintf(stderr, "  [shader compile log] %s", log);
    }
    return s;
}

static GLuint link(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    if (vs) glAttachShader(p, vs);
    if (fs) glAttachShader(p, fs);
    glLinkProgram(p);
    return p;
}

static GLint link_ok(GLuint p)
{
    GLint ok = 0, len = 0;
    char log[2048] = "";
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
    if (len > 0) glGetProgramInfoLog(p, sizeof log - 1, NULL, log);
    if (!ok) fprintf(stderr, "  [link log] %s", log);
    return ok;
}

/* ------------------------------------------------------------------ cluster #3: bool block members */

/* KHR-GL46.shaders.uniform_block.*.bool: glslang stores bool/bvecN uniform-block members as
 * uint/uvecN ("a 32-bit int where non-0 means true" — GLSL-to-SPIR-V convention).  GL
 * introspection must report the GLSL type (GL_BOOL / GL_BOOL_VEC*), storage stays uint. */
static const char *boolblk_vs =
    "#version 460 core\n"
    "in vec2 p;\n"
    "layout(std140) uniform Block { bool var; bvec2 v2; bvec3 v3; bvec4 v4; bool arr[3]; float f; } blk[2];\n"
    "struct Inner { bool x; bvec2 y; float z; };\n"
    "layout(std140) uniform Blk2 { Inner s; Inner sa[2]; } blk2;\n"
    "void main() {\n"
    "    gl_Position = vec4(p, blk[0].v2.x ? 1.0 : 0.0, blk2.s.y.x ? 1.0 : 0.0);\n"
    "}\n";

TEST(cts_bool_block_members_report_glsl_type)
{
    GLuint vs = make_shader(GL_VERTEX_SHADER, boolblk_vs);
    GLuint p = link(vs, 0);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);

    struct { const char *name; GLenum type; } want[] = {
        { "Block.var",  GL_BOOL },
        { "Block.v2",   GL_BOOL_VEC2 },
        { "Block.v3",   GL_BOOL_VEC3 },
        { "Block.v4",   GL_BOOL_VEC4 },
        { "Block.arr[0]", GL_BOOL },
        { "Block.f",    GL_FLOAT },
        { "Blk2.s.x",   GL_BOOL },
        { "Blk2.s.y",   GL_BOOL_VEC2 },
        { "Blk2.sa[0].x", GL_BOOL },
    };
    for (unsigned i = 0; i < sizeof want / sizeof want[0]; i++)
    {
        GLuint idx = 0;
        glGetUniformIndices(p, 1, &want[i].name, &idx);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        if (idx == 0xFFFFFFFFu) { EXPECT_TRUE(0 && "uniform index not found"); continue; }
        GLenum type = 0;
        glGetActiveUniformsiv(p, 1, &idx, GL_UNIFORM_TYPE, (GLint *)&type);
        if (type != want[i].type)
            fprintf(stderr, "  %s: type 0x%04x want 0x%04x\n", want[i].name, type, want[i].type);
        EXPECT_EQ_INT(type, want[i].type);
    }

    /* named-block members have no location, and the UBO layout is unchanged (uint storage) */
    EXPECT_EQ_INT(glGetUniformLocation(p, "Block.var"), -1);
    GLuint bi = glGetUniformBlockIndex(p, "Block[0]");
    EXPECT_TRUE(bi != 0xFFFFFFFFu);
    GLint size = 0;
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_DATA_SIZE, &size);
    EXPECT_TRUE(size >= 64);   /* var+v2+v3+v4+arr[3]+f laid out in std140 */

    glDeleteShader(vs); glDeleteProgram(p);
}

/* same for std430 buffer blocks (GL_BUFFER_VARIABLE queries) */
static const char *boolssbo_fs =
    "#version 460 core\n"
    "layout(location=0) out vec4 colour;\n"
    "layout(std430) buffer Data { bool b; bvec3 v; float f; };\n"
    "void main() { colour = vec4(b ? 1.0 : 0.0, v.x ? 1.0 : 0.0, f, 1.0); }\n";

TEST(cts_bool_ssbo_members_report_glsl_type)
{
    static const char *vs_src = "#version 330 core\nin vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    GLuint vs = make_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = make_shader(GL_FRAGMENT_SHADER, boolssbo_fs);
    GLuint p = link(vs, fs);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);

    GLint nb = 0;
    glGetProgramInterfaceiv(p, GL_BUFFER_VARIABLE, GL_ACTIVE_RESOURCES, &nb);
    EXPECT_TRUE(nb >= 3);
    struct { const char *name; GLenum type; } want[] = {
        { "b", GL_BOOL }, { "v", GL_BOOL_VEC3 }, { "f", GL_FLOAT },
    };
    int seen[3] = {0};
    for (int i = 0; i < nb; i++)
    {
        char name[256] = "";
        GLenum props[] = { GL_TYPE, GL_NAME_LENGTH };
        GLint vals[2] = {0};
        glGetProgramResourceiv(p, GL_BUFFER_VARIABLE, i, 2, props, 2, NULL, vals);
        glGetProgramResourceName(p, GL_BUFFER_VARIABLE, i, sizeof name - 1, NULL, name);
        for (unsigned k = 0; k < 3; k++)
            if (!strcmp(name, want[k].name)) { seen[k] = 1; EXPECT_EQ_INT((GLenum)vals[0], want[k].type); }
    }
    for (int k = 0; k < 3; k++) EXPECT_EQ_INT(seen[k], 1);

    glDeleteShader(vs); glDeleteShader(fs); glDeleteProgram(p);
}

/* KHR-GL46.shaders.uniform_block.single_basic_type.shared/packed: the GL-only block packings
 * shared/packed are legal GLSL but rejected when generating SPIR-V ("'shared' : not allowed
 * when generating SPIR-V"); MGL strips them (default packing is introspectable). */
static const char *packed_vs =
    "#version 460 core\n"
    "in vec2 p;\n"
    "layout(shared) uniform A { float a; vec3 b; };\n"
    "layout(packed, row_major) uniform B { mat2 m; float f; };\n"
    "void main() { gl_Position = vec4(p, a, m[0].x); }\n";

TEST(cts_shared_packed_block_layouts_compile)
{
    GLuint vs = make_shader(GL_VERTEX_SHADER, packed_vs);
    GLuint p = link(vs, 0);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);
    EXPECT_TRUE(glGetUniformBlockIndex(p, "A") != 0xFFFFFFFFu);
    EXPECT_TRUE(glGetUniformBlockIndex(p, "B") != 0xFFFFFFFFu);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glDeleteShader(vs); glDeleteProgram(p);
}


/* KHR-GL46.pixelstoragemodes.teximage2d.r11g11b10f.0_0 — the CTS fragment shader declares
 * `uniform highp sampler2D sampler;`: `sampler` is an MSL type name, so the SPIRV-Cross
 * MSL output ("sampler samplerSmplr") did not compile in Metal ("must use 'struct' tag"). */
/* ------------------------------------------------------------------ cluster #1: MSL reserved words */

static const char *psm_vs =
    "#version 330 core\n"
    "in vec2 position;\n"
    "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";

static const char *psm_fs_sampler =
    "#version 330 core\n"
    "layout(location=0) out vec4 colour;\n"
    "uniform highp sampler2D sampler;\n"
    "uniform vec2 texcoords;\n"
    "void main() { colour = texelFetch(sampler, ivec2(texcoords), 0); }\n";

TEST(cts_msl_reserved_sampler_uniform)
{
    GLuint vs = make_shader(GL_VERTEX_SHADER, psm_vs);
    GLuint fs = make_shader(GL_FRAGMENT_SHADER, psm_fs_sampler);
    GLuint p = link(vs, fs);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);

    /* introspection keeps the GLSL name */
    GLint loc = glGetUniformLocation(p, "sampler");
    EXPECT_TRUE(loc >= 0);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteShader(vs); glDeleteShader(fs); glDeleteProgram(p);
}

/* MSL keywords / built-in type names used as struct members, block members and functions */
static const char *reserved_fs =
    "#version 450 core\n"
    "layout(location=0) out vec4 colour;\n"
    "uniform sampler2D tex;\n"
    "uniform float kernel;\n"                              /* MSL kernel qualifier    */
    "layout(std140) uniform Blk { float device; float thread; float texture; float sampler; } blk;\n"
    "float saturate(float x) { return x < 0.0 ? 0.0 : x; }\n" /* metal std lib function */
    "struct float4 { float x; };\n"                          /* MSL matrix type name   */
    "void main() {\n"
    "    float4 f; f.x = 0.25;\n"
    "    colour = texelFetch(tex, ivec2(f.x, blk.texture), 0)\n"
    "            + vec4(kernel, blk.device, blk.thread, saturate(blk.sampler));\n"
    "}\n";

TEST(cts_msl_reserved_words_everywhere)
{
    GLuint vs = make_shader(GL_VERTEX_SHADER, psm_vs);
    GLuint fs = make_shader(GL_FRAGMENT_SHADER, reserved_fs);
    GLuint p = link(vs, fs);
    EXPECT_EQ_INT(link_ok(p), GL_TRUE);

    EXPECT_TRUE(glGetUniformLocation(p, "kernel") >= 0);
    /* members of a *named* uniform block are introspectable via glGetActiveUniform but
     * have no location (GL 4.6 §7.3.1.1: names associated with a named block get -1) */
    EXPECT_EQ_INT(glGetUniformLocation(p, "Blk.device"), -1);

    GLint count = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &count);
    int seen_blk[4] = {0};
    static const char *const want[] = { "Blk.device", "Blk.thread", "Blk.texture", "Blk.sampler" };
    for (int i = 0; i < count; i++)
    {
        char name[256] = "";
        GLsizei len = 0; GLint size = 0; GLenum type = 0;
        glGetActiveUniform(p, i, sizeof name - 1, &len, &size, &type, name);
        for (int k = 0; k < 4; k++)
            if (!strcmp(name, want[k])) { seen_blk[k] = 1; EXPECT_EQ_INT(type, GL_FLOAT); }
    }
    for (int k = 0; k < 4; k++) EXPECT_EQ_INT(seen_blk[k], 1);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteShader(vs); glDeleteShader(fs); glDeleteProgram(p);
}


/* ------------------------------------------------------------------ cluster #11: UBO instance arrays */

static const char *uboarr_vs =
    "#version 460 core\n"
    "in vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";

static const char *fs =
    "#version 460 core\n"
    "layout(location=0) out vec4 colour;\n"
    "layout(std140) uniform Block { float f; vec4 v; } blk[2];\n"
    "void main() { colour = blk[0].v * 0.25 + vec4(blk[1].f) * 0.5; }\n";

static const char *fs_cm =
    "#version 460 core\n"
    "layout(location=0) out vec4 colour;\n"
    "layout(row_major, std140) uniform Block { mat2 m; } blk[2];\n"
    "void main() { colour = vec4(blk[0].m * vec2(1.0, 0.0), blk[1].m * vec2(0.0, 1.0)); }\n";

TEST(cts_ubo_instance_array_basic)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &uboarr_vs, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) { glDeleteProgram(p); return; }

    /* introspection: one block per instance */
    GLint nb = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCKS, &nb);
    printf("active blocks: %d\n", nb);
    EXPECT_EQ_INT(nb, 2);

    /* data: blk[0] = { f=0.0, v=(1,1,1,1) }, blk[1] = { f=1.0, v=0 }; instances are bound at
     * GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT boundaries like the CTS does */
    float data[128] = {0};
    data[4] = data[5] = data[6] = data[7] = 1.0f;
    data[64] = 1.0f;
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_UNIFORM_BUFFER, buf);
    glBufferData(GL_UNIFORM_BUFFER, sizeof data, data, GL_STATIC_DRAW);

    for (int i = 0; i < nb; i++)
    {
        GLint binding = 0;
        glGetActiveUniformBlockiv(p, i, GL_UNIFORM_BLOCK_BINDING, &binding);
        printf("block %d binding %d\n", i, binding);
        glBindBufferRange(GL_UNIFORM_BUFFER, i, buf, (GLintptr)(i * 256), 256);
    }

    /* draw full-screen quad */
    static const float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint va = 0, vbo = 0;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    unsigned char px[4] = {0};
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("pixel: %d %d %d %d\n", px[0], px[1], px[2], px[3]);
    /* colour = v*0.25 + f*0.5 = (0.25,0.25,0.25,0.25) + 0.5 = 0.75 -> 191 */
    EXPECT_TRUE(px[0] > 180 && px[0] < 202);

    glDeleteBuffers(1, &buf); glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &va); glDeleteProgram(p);
}

TEST(cts_ubo_instance_array_row_major_mat2)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &uboarr_vs, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs_cm, NULL); glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) { glDeleteProgram(p); return; }

    /* row-major mat2 (std140: every row padded to vec4, stride 16) */
    float data[256] = {0};
    data[0] = 0.25f; data[1] = 0.50f;         /* row 0 */
    data[4] = 0.75f; data[5] = 1.00f;         /* row 1 @ +16 */
    data[64 + 0] = 0.10f; data[64 + 1] = 0.20f;
    data[64 + 4] = 0.30f; data[64 + 5] = 0.40f;
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_UNIFORM_BUFFER, buf);
    glBufferData(GL_UNIFORM_BUFFER, sizeof data, data, GL_STATIC_DRAW);

    GLint nb = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCKS, &nb);
    EXPECT_EQ_INT(nb, 2);
    for (int i = 0; i < nb; i++)
        glBindBufferRange(GL_UNIFORM_BUFFER, i, buf, (GLintptr)(i * 256), 256);

    static const float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint va = 0, vbo = 0;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    unsigned char px[4] = {0};
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("pixel: %d %d %d %d\n", px[0], px[1], px[2], px[3]);
    /* blk[0].m*(1,0) = (row0.x, row1.x) = (0.25, 0.75); blk[1].m*(0,1) = (row0.y, row1.y) = (0.20, 0.40) */
    EXPECT_TRUE(abs(px[0] - 64) <= 2 && abs(px[1] - 191) <= 2 && abs(px[2] - 51) <= 2 && abs(px[3] - 102) <= 2);

    glDeleteBuffers(1, &buf); glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &va); glDeleteProgram(p);
}


static const char *uboarr_vs2 =
    "#version 460 core\n"
    "in vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";

static int uboarr_draw_and_read(const char *fs_src, const float *data, size_t data_bytes,
                         int nblocks, unsigned char px[4])
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &uboarr_vs, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    const char *src = fs_src;
    glShaderSource(f, 1, &src, NULL); glCompileShader(f);
    GLint cs = 0; glGetShaderiv(f, GL_COMPILE_STATUS, &cs);
    if (!cs) { char log[1024]=""; glGetShaderInfoLog(f,sizeof log-1,NULL,log); printf("  compile: %s", log); }
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]=""; glGetProgramInfoLog(p,sizeof log-1,NULL,log); printf("  link: %s", log); }
    glDeleteShader(v); glDeleteShader(f);
    if (!ok) { glDeleteProgram(p); return -1; }

    GLint nb = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCKS, &nb);
    printf("  blocks=%d", nb);
    if (nb != nblocks) printf(" (want %d)", nblocks);
    printf("\n");

    GLuint buf = 0;
    GLint align = 0;
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &align);
    if (align < 1) align = 256;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_UNIFORM_BUFFER, buf);
    glBufferData(GL_UNIFORM_BUFFER, data_bytes, data, GL_STATIC_DRAW);
    /* the CTS packs the instances of one buffer at UNIFORM_BUFFER_OFFSET_ALIGNMENT
     * boundaries and binds each instance's block index separately */
    for (int i = 0; i < nb; i++)
    {
        GLint dsz = 0;
        glGetActiveUniformBlockiv(p, i, GL_UNIFORM_BLOCK_DATA_SIZE, &dsz);
        GLintptr off = (GLintptr)i * align;
        glBindBufferRange(GL_UNIFORM_BUFFER, i, buf, off, dsz);
        printf("  bind block %d size %d off %lld\n", i, dsz, (long long)off);
    }

    static const float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint va = 0, vbo = 0;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glDeleteBuffers(1, &buf); glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &va); glDeleteProgram(p);
    return 0;
}

static int uboarr_near(unsigned char px[4], int r, int g, int b, int a)
{
    return abs(px[0]-r)<=2 && abs(px[1]-g)<=2 && abs(px[2]-b)<=2 && abs(px[3]-a)<=2;
}

TEST(cts_ubo_single_row_major_mat2)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 colour;\n"
        "layout(row_major, std140) uniform Block { mat2 m; };\n"
        "void main() { vec2 r = m * vec2(1.0, 0.0); colour = vec4(r, 0.0, 1.0); }\n";
    /* std140 matrices are arrays of vectors padded to vec4 (stride 16 for each row) */
    float data[8] = {0};
    data[0]=0.25f; data[1]=0.50f;   /* row 0 */
    data[4]=0.75f; data[5]=1.00f;   /* row 1 @ +16 */
    unsigned char px[4] = {0};
    uboarr_draw_and_read(fs, data, sizeof data, 1, px);
    printf("  row-major single: %d %d %d %d\n", px[0],px[1],px[2],px[3]);
    EXPECT_TRUE(uboarr_near(px, 64, 191, 0, 255));   /* (row0.x, row1.x) = (0.25, 0.75) */
}

TEST(cts_ubo_single_column_major_mat2)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 colour;\n"
        "layout(column_major, std140) uniform Block { mat2 m; };\n"
        "void main() { vec2 r = m * vec2(1.0, 0.0); colour = vec4(r, 0.0, 1.0); }\n";
    float data[8] = {0};
    data[0]=0.25f; data[1]=0.50f;   /* column 0 */
    data[2]=0.75f; data[3]=1.00f;   /* column 1 */
    unsigned char px[4] = {0};
    uboarr_draw_and_read(fs, data, sizeof data, 1, px);
    printf("  column-major single: %d %d %d %d\n", px[0],px[1],px[2],px[3]);
    EXPECT_TRUE(uboarr_near(px, 64, 128, 0, 255));   /* m*(1,0) = column0 = (0.25, 0.50) */
}

TEST(cts_ubo_instance_array_float)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 colour;\n"
        "layout(std140) uniform Block { float f; } blk[2];\n"
        "void main() { colour = vec4(blk[0].f, blk[1].f, 0.0, 1.0); }\n";
    float data[128] = {0};
    data[0] = 0.25f;                 /* blk[0].f */
    data[64] = 0.75f;                /* blk[1].f @256B */
    unsigned char px[4] = {0};
    uboarr_draw_and_read(fs, data, sizeof data, 2, px);
    printf("  instance array float: %d %d %d %d\n", px[0],px[1],px[2],px[3]);
    EXPECT_TRUE(uboarr_near(px, 64, 191, 0, 255));
}

TEST(cts_ubo_instance_array_vec4)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 colour;\n"
        "layout(std140) uniform Block { vec4 v; } blk[2];\n"
        "void main() { colour = blk[0].v * 0.25 + blk[1].v * 0.5; }\n";
    float data[128] = {0};
    data[0]=data[1]=data[2]=data[3]=1.0f;   /* blk[0].v */
    data[64]=data[65]=data[66]=data[67]=1.0f; /* blk[1].v @256B */
    unsigned char px[4] = {0};
    uboarr_draw_and_read(fs, data, sizeof data, 2, px);
    printf("  instance array vec4: %d %d %d %d\n", px[0],px[1],px[2],px[3]);
    EXPECT_TRUE(uboarr_near(px, 191, 191, 191, 191));   /* 0.25+0.5 */
}

int main(void)
{
    RUN(cts_msl_reserved_sampler_uniform);
    RUN(cts_msl_reserved_words_everywhere);
    RUN(cts_bool_block_members_report_glsl_type);
    RUN(cts_bool_ssbo_members_report_glsl_type);
    RUN(cts_shared_packed_block_layouts_compile);
    RUN(cts_ubo_instance_array_basic);
    RUN(cts_ubo_instance_array_row_major_mat2);
    RUN(cts_ubo_single_row_major_mat2);
    RUN(cts_ubo_single_column_major_mat2);
    RUN(cts_ubo_instance_array_float);
    RUN(cts_ubo_instance_array_vec4);
    return mgl_test_summary();
}
