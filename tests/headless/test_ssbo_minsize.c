/* test_ssbo_minsize — GL_BUFFER_DATA_SIZE of a block whose LAST member is a
 * runtime ("unsized") array must report the minimum buffer size INCLUDING one
 * element of that array (GL 4.6 §9.2.3 / ARB_program_interface_query).
 *
 * The exact shapes of dEQP-GL45-ES31
 *   functional.program_interface_query.shader_storage_block.buffer_data_size.
 *   {named_block,unnamed_block,block_array}
 * — MGL used to report the size WITHOUT the mandatory trailing element
 * (16 for { vec4 m; float arr[]; } where the minimum is 20), so the CTS lower
 * bound check ("buffer size was less than minimum buffer data size") failed.
 */
#include "mgl_test.h"

struct BlockQuery { char name[96]; GLint data_size; };

/* enumerate GL_SHADER_STORAGE_BLOCK resources with their GL_BUFFER_DATA_SIZE */
static int query_blocks(GLuint prog, struct BlockQuery out[8])
{
    GLint n = 0;
    glGetProgramInterfaceiv(prog, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &n);
    if (n > 8) n = 8;
    for (int i = 0; i < n; i++) {
        GLsizei len = 0;
        glGetProgramResourceName(prog, GL_SHADER_STORAGE_BLOCK, i, sizeof out[i].name, &len, out[i].name);
        GLenum prop = GL_BUFFER_DATA_SIZE;
        GLsizei pl = 0;
        out[i].data_size = -1;
        glGetProgramResourceiv(prog, GL_SHADER_STORAGE_BLOCK, i, 1, &prop, 1, &len, &out[i].data_size);
        (void)pl;
    }
    return n;
}

static GLint find_size(int n, const struct BlockQuery *q, const char *name)
{
    for (int i = 0; i < n; i++)
        if (!strcmp(q[i].name, name)) return q[i].data_size;
    return -1;
}

/* the CTS shader: std430, three blocks, TargetInterface ends in an unsized array */
static const char *kNamedCs =
    "#version 450\n"
    "layout(local_size_x=1) in;\n"
    "struct StructType { highp vec4 structMemberBasic; highp float structMemberArray[2]; };\n"
    "layout(binding=0, std430) buffer PrecedingInterface {\n"
    "    highp vec4 precedingMember;\n"
    "    highp float precedingMemberUnsizedArray[];\n"
    "} precedingInstance;\n"
    "layout(binding=1, std430) buffer TargetInterface {\n"
    "    highp vec4 blockMemberBasic;\n"
    "    highp float blockMemberArray[3];\n"
    "    StructType blockMemberStruct[2];\n"
    "    highp float blockMemberUnsizedArray[];\n"
    "} targetInstance;\n"
    "layout(binding=3, std430) buffer TrailingInterface {\n"
    "    highp vec4 trailingMember;\n"
    "} trailingInstance;\n"
    "void main() {\n"
    "    targetInstance.blockMemberUnsizedArray[0] = precedingInstance.precedingMember.x\n"
    "        + trailingInstance.trailingMember.x;\n"
    "}\n";

TEST(named_block)
{
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &kNamedCs, NULL);
    glCompileShader(cs);
    GLint ok = 0; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glDeleteShader(cs); glDeleteProgram(prog); return; }

    struct BlockQuery q[8];
    int n = query_blocks(prog, q);
    EXPECT_EQ_INT(n, 3);

    /* CTS "tight packing" minimums: preceding 16+4=20, target 16+12+2*24+4=80,
     * trailing 16.  MGL reports the std430 minimum incl. one trailing element:
     * preceding 20, target 100 (offset 96 + stride 4), trailing 16. */
    GLint s = find_size(n, q, "PrecedingInterface");
    EXPECT_TRUE(s >= 20);          /* was 16 — the regression */
    EXPECT_EQ_INT(s, 20);
    s = find_size(n, q, "TargetInterface");
    EXPECT_TRUE(s >= 80);
    EXPECT_EQ_INT(s, 100);
    s = find_size(n, q, "TrailingInterface");
    EXPECT_TRUE(s >= 16);
    EXPECT_EQ_INT(s, 16);
    EXPECT_NO_GL_ERROR();

    glDeleteShader(cs);
    glDeleteProgram(prog);
}

/* the unnamed_block CTS variant: an anonymous block (no instance name after the
 * member list is NOT what the CTS uses — it uses `buffer Target { ... } target;`
 * with the INSTANCE name only… actually the CTS "unnamed_block" declares
 * `buffer Target { ... };` with members as globals; glsl_block_compat renames
 * the block to a synthetic mgl_anon_block_N and GL exposes the BLOCK name —
 * query by index instead of name). */
static const char *kUnnamedCs =
    "#version 450\n"
    "layout(local_size_x=1) in;\n"
    "layout(binding=2, std430) buffer {\n"
    "    highp vec4 anonMember;\n"
    "    highp float anonUnsized[];\n"
    "};\n"
    "void main() { anonUnsized[0] = anonMember.y; }\n";

TEST(unnamed_block)
{
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &kUnnamedCs, NULL);
    glCompileShader(cs);
    GLint ok = 0; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glDeleteShader(cs); glDeleteProgram(prog); return; }

    struct BlockQuery q[8];
    int n = query_blocks(prog, q);
    EXPECT_EQ_INT(n, 1);
    /* minimum 16 + 4; MGL reports the std430 minimum incl. one element */
    EXPECT_TRUE(q[0].data_size >= 20);
    EXPECT_EQ_INT(q[0].data_size, 20);
    EXPECT_NO_GL_ERROR();

    glDeleteShader(cs);
    glDeleteProgram(prog);
}

/* the block_array CTS variant: an array of the block — every ELEMENT is a
 * resource with its own (identical) minimum. */
static const char *kArrayCs =
    "#version 450\n"
    "layout(local_size_x=1) in;\n"
    "layout(binding=0, std430) buffer ArrayInterface {\n"
    "    highp vec4 memberBasic;\n"
    "    highp float memberUnsized[];\n"
    "} inst[2];\n"
    "void main() { inst[0].memberUnsized[0] = inst[1].memberBasic.z; }\n";

TEST(block_array)
{
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &kArrayCs, NULL);
    glCompileShader(cs);
    GLint ok = 0; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glDeleteShader(cs); glDeleteProgram(prog); return; }

    struct BlockQuery q[8];
    int n = query_blocks(prog, q);
    EXPECT_EQ_INT(n, 2);
    for (int i = 0; i < n; i++) {
        EXPECT_TRUE(q[i].data_size >= 20);   /* was 16 — the regression */
        EXPECT_EQ_INT(q[i].data_size, 20);
    }
    EXPECT_NO_GL_ERROR();

    glDeleteShader(cs);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(named_block);
    RUN(unnamed_block);
    RUN(block_array);
    return mgl_test_summary();
}
