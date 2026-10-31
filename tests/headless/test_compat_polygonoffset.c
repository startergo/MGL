/* ARB_polygon_offset_clamp on Metal:
 *  - Metal scales the depth-bias by 2^-24 regardless of the depth attachment; GL's r is
 *    the attachment's resolvable difference (1/(2^n-1) for n-bit fixed point), so MGL
 *    pre-scales the bias for a DEPTH_COMPONENT16 attachment (256x).
 *  - Metal's clamp is GL's one-sided clamp: clamp>0 -> min(o,clamp), clamp<0 -> max(o,clamp),
 *    clamp==0 -> unclamped.
 * CTS: KHR-GL46.polygon_offset_clamp.PolygonOffsetClampMinMax/ZeroInfinity. */
#include "mgl_test.h"

static const char* poc_vs = "#version 450 core\nin vec3 v; void main(){gl_Position=vec4(v,1);}";
static const char* poc_fs = "#version 450 core\nout vec4 fc; void main(){fc=vec4(0,1,0,1);}";
static GLuint poc_prog;
static GLuint poc_vao, poc_vb;

static void poc_setup(void)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);   glShaderSource(v,1,&poc_vs,0); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(f,1,&poc_fs,0); glCompileShader(f);
    poc_prog = glCreateProgram(); glAttachShader(poc_prog,v); glAttachShader(poc_prog,f); glLinkProgram(poc_prog);
    const float q[12] = {-1,-1,0.5f, 1,-1,0.5f, 1,1,0.5f, -1,1,0.5f};
    glGenVertexArrays(1,&poc_vao); glBindVertexArray(poc_vao);
    glGenBuffers(1,&poc_vb); glBindBuffer(GL_ARRAY_BUFFER,poc_vb);
    glBufferData(GL_ARRAY_BUFFER,sizeof q,q,GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,12,0); glEnableVertexAttribArray(0);
}

/* draws a z=0.5 quad with polygon offset (units,clamp) into a 64x64 dfmt depth texture,
 * returns the read-back depth at the centre */
static float poc_depth(GLenum dfmt, float units, float clampv)
{
    GLuint tex, fb;
    glGenTextures(1,&tex); glBindTexture(GL_TEXTURE_2D,tex);
    glTexStorage2D(GL_TEXTURE_2D,1,dfmt,64,64);
    glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,tex,0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_ALWAYS); glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffsetClamp(0.0f, units, clampv);
    glUseProgram(poc_prog); glBindVertexArray(poc_vao); glViewport(0,0,64,64);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_FAN,0,4);
    float d = 1.0f;
    glReadPixels(32,32,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&d);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glDisable(GL_DEPTH_TEST); glDisable(GL_POLYGON_OFFSET_FILL); glPolygonOffsetClamp(0,0,0);
    glDeleteTextures(1,&tex); glDeleteFramebuffers(1,&fb);
    return d;
}

TEST(polygon_offset_clamp_d16_r_is_attachment_resolvable_difference)
{
    poc_setup();
    /* units=-1000, clamp=0: offset must be 1000 * 1/(2^16-1), NOT Metal's 1000*2^-24 */
    float d = poc_depth(GL_DEPTH_COMPONENT16, -1000.0f, 0.0f);
    EXPECT_NEAR(d, 0.75f - 1000.0f/65535.0f, 2e-6f);
    d = poc_depth(GL_DEPTH_COMPONENT16, 1000.0f, 0.0f);
    EXPECT_NEAR(d, 0.75f + 1000.0f/65535.0f, 2e-6f);
}

TEST(polygon_offset_clamp_one_sided)
{
    /* clamp<0 clamps only negative offsets (max(o,clamp)) */
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT16, -1000.0f, -1e-4f), 0.75f - 1e-4f, 2e-6f);
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT16,  1000.0f, -1e-4f), 0.75f + 1000.0f/65535.0f, 2e-6f);
    /* clamp>0 clamps only positive offsets (min(o,clamp)) */
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT16,  1000.0f,  1e-4f), 0.75f + 1e-4f, 2e-6f);
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT16, -1000.0f,  1e-4f), 0.75f - 1000.0f/65535.0f, 2e-6f);
    /* clamp=0 never clamps */
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT16, -4000.0f, 0.0f), 0.75f - 4000.0f/65535.0f, 2e-6f);
}

TEST(polygon_offset_clamp_d24_d32f_unchanged)
{
    /* 24-bit and float32 depths: r stays 2^-24 (Metal native) */
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT24, -1000.0f, 0.0f), 0.75f - 1000.0f/16777216.0f, 2e-6f);
    EXPECT_NEAR(poc_depth(GL_DEPTH_COMPONENT32F, -1000.0f, 0.0f), 0.75f - 1000.0f/16777216.0f, 2e-6f);
    /* |o| = 1000*2^-24 < |clamp|, so a one-sided clamp does not bind here */
    EXPECT_NEAR(poc_depth(GL_DEPTH24_STENCIL8,  -1000.0f, -1e-4f), 0.75f - 1000.0f/16777216.0f, 2e-6f);
}

TEST(polygon_offset_clamp_d16_renderbuffer)
{
    /* the r scale must also be picked up when the depth attachment is a renderbuffer */
    GLuint rb, fb;
    glGenRenderbuffers(1,&rb); glBindRenderbuffer(GL_RENDERBUFFER,rb);
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT16,64,64);
    glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_ALWAYS); glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffsetClamp(0.0f, -1000.0f, 0.0f);
    glUseProgram(poc_prog); glBindVertexArray(poc_vao); glViewport(0,0,64,64);
    glClear(GL_DEPTH_BUFFER_BIT); glDrawArrays(GL_TRIANGLE_FAN,0,4);
    float d = 1.0f; glReadPixels(32,32,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&d);
    /* stored on the true Depth16Unorm lattice: half-ulp slack */
    EXPECT_NEAR(d, 0.75f - 1000.0f/65535.0f, 3.0e-5f);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glDisable(GL_DEPTH_TEST); glDisable(GL_POLYGON_OFFSET_FILL);
    glDeleteRenderbuffers(1,&rb); glDeleteFramebuffers(1,&fb);
}

int main(void)
{
    RUN(polygon_offset_clamp_d16_r_is_attachment_resolvable_difference);
    RUN(polygon_offset_clamp_one_sided);
    RUN(polygon_offset_clamp_d24_d32f_unchanged);
    RUN(polygon_offset_clamp_d16_renderbuffer);
    return mgl_test_summary();
}
