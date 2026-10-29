/* test_img3d — non-layered image2D bindings of GL_TEXTURE_3D textures
 * (KHR-GL46.shader_image_load_store.non-layered_binding, 3d leg): each draw
 * copies one z-slice of a 3D texture to another through imageLoad/imageStore
 * with the images bound NON-layered (layered = GL_FALSE).  Metal cannot alias
 * a 3D texture as 2D (no view, no blit, silent no-op bindings — see
 * MGLRenderer+Geometry.m), so MGL runs the slice through a 2D shadow plane;
 * this checks the shadow copy-in/out, layer by layer, and that reads of the 3D
 * texture (glGetTexImage) see the stores.
 */
#include "mgl_test.h"
#include <string.h>

static const char *kVs =
    "#version 460 core\n"
    "layout(location=0) in vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";

static const char *kFs =
    "#version 460 core\n"
    "layout(binding=0, rgba8) readonly uniform image2D src;\n"
    "layout(binding=1, rgba8) writeonly uniform image2D dst;\n"
    "void main() {\n"
    "    ivec2 c = ivec2(gl_FragCoord.xy);\n"
    "    imageStore(dst, c, imageLoad(src, c) * 2.0 + 0.25);\n"
    "}\n";

static GLubyte src_data[4][4][4][4];   /* [z][y][x][rgba] */
static GLubyte dst_back[4][4][4][4];

TEST(img3d_nonlayered_copy)
{
    GLuint prog = 0, vao = 0, src = 0, dst = 0, vbo = 0;
    static const float quad[] = {
        -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    };

    prog = glCreateProgram();
    for (int st = 0; st < 2; st++)
    {
        GLuint sh = glCreateShader(st ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);
        const char *src_c = st ? kFs : kVs;
        glShaderSource(sh, 1, &src_c, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(prog);
    EXPECT_NO_GL_ERROR();
    {
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 1);
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);

    for (int z = 0; z < 4; z++)
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                for (int c = 0; c < 4; c++)
                    src_data[z][y][x][c] = (GLubyte)(z * 64 + y * 16 + x * 4 + c);

    glGenTextures(1, &src);
    glBindTexture(GL_TEXTURE_3D, src);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, 4, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 4, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, src_data);
    EXPECT_NO_GL_ERROR();

    glGenTextures(1, &dst);
    glBindTexture(GL_TEXTURE_3D, dst);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, 4, 4, 4);
    EXPECT_NO_GL_ERROR();

    glUseProgram(prog);
    glViewport(0, 0, 4, 4);

    for (int z = 0; z < 4; z++)
    {
        glBindImageTexture(0, src, 0, GL_FALSE, z, GL_READ_ONLY, GL_RGBA8);
        glBindImageTexture(1, dst, 0, GL_FALSE, z, GL_WRITE_ONLY, GL_RGBA8);
        EXPECT_NO_GL_ERROR();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        EXPECT_NO_GL_ERROR();
    }

    memset(dst_back, 0xee, sizeof(dst_back));
    glBindTexture(GL_TEXTURE_3D, dst);
    glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_UNSIGNED_BYTE, dst_back);
    EXPECT_NO_GL_ERROR();

    for (int z = 0; z < 4; z++)
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                for (int c = 0; c < 4; c++)
                {
                    int v = src_data[z][y][x][c];
                    int want = v * 2 + 64 > 255 ? 255 : v * 2 + 64;   /* v/255*2+0.25 */
                    int got = dst_back[z][y][x][c];
                    if (got < want - 2 || got > want + 2)
                        MGL_FAIL("z=%d (%d,%d) ch%d: %d, expected %d (src %d)", z, x, y, c, got, want, v);
                    mgl_test_checks++;
                }

    glDeleteTextures(1, &src);
    glDeleteTextures(1, &dst);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(img3d_nonlayered_copy);
    return mgl_test_summary();
}
