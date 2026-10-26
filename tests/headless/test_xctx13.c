/* test_xctx13.c — glmark2 [jellyfish] caustic repro on a CORE-profile context
 * (what glmark2-macos actually gets through the NSOpenGL interposer).
 * Real jellyfish.vert/frag, real jellyfish256.rgba on unit 0, caustics RGB on
 * unit 1 with 33 texture names (the real scene's count), LINEAR+REPEAT set
 * after upload, SRC_ALPHA/ONE_MINUS_SRC_ALPHA blending.  The oracle: swapping
 * ONLY the unit-1 texture between the real caustic web and solid black must
 * change the bell pixels a lot (the caustic term), and sampling at uv>1 must
 * wrap (REPEAT), not clamp to the edge texel. */
#include "mgl_test.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static float k_vert[4 * 11] = {
    /* pos(3)         normal(3)     color(3)      uv(2) */
    -1.f, -1.f, 0.f,  0.f, 0.f, 1.f,  1.f, 0.f, 0.f,  0.f, 0.f,
     1.f, -1.f, 0.f,  0.f, 1.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f,
     1.f,  1.f, 0.f,  1.f, 0.f, 0.f,  0.f, 0.f, 1.f,  1.f, 1.f,
    -1.f,  1.f, 0.f,  0.f, 1.f, 1.f,  1.f, 1.f, 0.f,  0.f, 1.f,
};
static const GLushort k_idx[6] = { 0, 1, 2, 0, 2, 3 };

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n; char *s;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    s = (char *)calloc(1, n + 1);
    if (fread(s, 1, n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    fclose(f);
    return s;
}

static GLubyte *slurp_bin(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n; GLubyte *b;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = (GLubyte *)malloc(n);
    if (fread(b, 1, n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f);
    return b;
}

static GLuint compile1(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok; char log[4096];
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "compile %x: %s\n", type, log); }
    EXPECT_TRUE(ok);
    return sh;
}

TEST(jellyfish_core_caustics)
{
    /* glmark2-macos gets a COMPATIBILITY profile through the interposer (its
     * GLSL-100 shaders only compile on the compat translator path) */
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      0x00000002 /* GL_CONTEXT_COMPATIBILITY_PROFILE_BIT */);
    char *vsrc = slurp("/opt/homebrew/share/glmark2/shaders/jellyfish.vert");
    char *fsrc = slurp("/opt/homebrew/share/glmark2/shaders/jellyfish.frag");
    GLubyte *caust = slurp_bin("/tmp/xctx/caust01.rgb");     /* 256*256*3 */
    GLubyte *skin = slurp_bin("/tmp/xctx/jelly256.rgba");    /* 256*256*4 */
    GLuint tex[34], black, prog, vs, fs, vao, vbo, ibo;
    GLint lp, ln, lc, lt, loc;
    GLfloat uWorld[16], uVP[16], uWIT[16];
    GLubyte px_web[64][4], px_black[64][4];
    int diff = 0;

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);
    if (!vsrc || !fsrc || !caust || !skin)
    {
        fprintf(stderr, "  note: shaders/textures missing, skip\n");
        return;
    }

    /* 34 textures so names run past the initial hash-table size (33 real +
     * black), like the scene's 1 skin + 32 caustics */
    for (int i = 0; i < 34; i++) tex[i] = 0;
    glGenTextures(34, tex);

    /* unit 0: the skin (RGBA), CLAMP_TO_EDGE + LINEAR like Texture::load */
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, skin);

    /* unit 1: the caustic web (RGB), LINEAR then WRAP=REPEAT set AFTER the
     * upload, exactly like the scene */
    for (int i = 1; i <= 32; i++)
    {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, caust);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    /* a solid black 2x2 texture to swap in as "no caustics" */
    {
        GLubyte z[2 * 2 * 3];
        memset(z, 0, sizeof z);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex[33]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, z);
    }
    while (glGetError() != GL_NO_ERROR) {}

    vs = compile1(GL_VERTEX_SHADER, vsrc);
    fs = compile1(GL_FRAGMENT_SHADER, fsrc);
    prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog);
    { GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok); EXPECT_TRUE(ok); }
    lp = glGetAttribLocation(prog, "aVertexPosition");
    ln = glGetAttribLocation(prog, "aVertexNormal");
    lc = glGetAttribLocation(prog, "aVertexColor");
    lt = glGetAttribLocation(prog, "aTextureCoord");
    EXPECT_TRUE(lp >= 0 && ln >= 0 && lc >= 0 && lt >= 0);
    glUseProgram(prog);

    /* xctx11's working matrices: uWorld = scale(3,5,2), VP and WIT identity
     * (full-screen quad; vWorld = (3x, 5y, 0) drives the caustic uv across
     * the whole texture, including outside [0,1] for the REPEAT check) */
    {
        memset(uWorld, 0, sizeof uWorld);
        uWorld[0] = 3.0f; uWorld[5] = 5.0f; uWorld[10] = 2.0f; uWorld[15] = 1.0f;
        for (int i = 0; i < 16; i++) { uVP[i] = (i % 5 == 0) ? 1.0f : 0.0f; uWIT[i] = (i % 5 == 0) ? 1.0f : 0.0f; }
    }
    loc = glGetUniformLocation(prog, "uWorld");          if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, uWorld);
    loc = glGetUniformLocation(prog, "uWorldViewProj");  if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, uVP);
    loc = glGetUniformLocation(prog, "uWorldInvTranspose"); if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, uWIT);
    loc = glGetUniformLocation(prog, "uLightPos");       if (loc >= 0) glUniform3f(loc, 3.f, 15.f, 10.f);
    loc = glGetUniformLocation(prog, "uLightRadius");    if (loc >= 0) glUniform1f(loc, 100.f);
    loc = glGetUniformLocation(prog, "uLightCol");       if (loc >= 0) glUniform4f(loc, 1.f, 1.f, 1.f, 1.f);
    loc = glGetUniformLocation(prog, "uAmbientCol");     if (loc >= 0) glUniform4f(loc, 0.1f, 0.1f, 0.1f, 1.f);
    loc = glGetUniformLocation(prog, "uFresnelCol");     if (loc >= 0) glUniform4f(loc, 0.2f, 0.2f, 0.2f, 1.f);
    loc = glGetUniformLocation(prog, "uFresnelPower");   if (loc >= 0) glUniform1f(loc, 1.f);
    loc = glGetUniformLocation(prog, "uCurrentTime");    if (loc >= 0) glUniform1f(loc, 0.0f);
    loc = glGetUniformLocation(prog, "uSampler");        if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(prog, "uSampler1");       if (loc >= 0) glUniform1i(loc, 1);
    while (glGetError() != GL_NO_ERROR) {}

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_vert, k_vert, GL_STATIC_DRAW);
    /* all four vertices sample ONE opaque, bright skin texel (the real mesh's
     * uvs hit the skin; a random uv lands in the fully-transparent white
     * region and the quad vanishes under blending) */
    {
        int best = -1;
        for (int i = 0; i < 256 * 256; i++)
        {
            int a = skin[i * 4 + 3];
            int lum = skin[i * 4] + skin[i * 4 + 1] + skin[i * 4 + 2];
            if (a > 240 && lum > 60) { best = i; break; }
        }
        if (best < 0) best = 0;
        float u = ((best % 256) + 0.5f) / 256.0f;
        float v = ((best / 256) + 0.5f) / 256.0f;
        fprintf(stderr, "  note: skin texel %d uv=(%.3f,%.3f) rgba=%d,%d,%d,%d\n",
                best, u, v, skin[best*4], skin[best*4+1], skin[best*4+2], skin[best*4+3]);
        for (int i = 0; i < 4; i++)
        {
            k_vert[i * 11 + 9] = u;
            k_vert[i * 11 + 10] = v;
        }
        glBufferData(GL_ARRAY_BUFFER, sizeof k_vert, k_vert, GL_DYNAMIC_DRAW);
    }
    {
        GLsizei st = 11 * (GLsizei)sizeof(float);
        glEnableVertexAttribArray((GLuint)lp);
        glVertexAttribPointer((GLuint)lp, 3, GL_FLOAT, GL_FALSE, st, (void *)(0 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)ln);
        glVertexAttribPointer((GLuint)ln, 3, GL_FLOAT, GL_FALSE, st, (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)lc);
        glVertexAttribPointer((GLuint)lc, 3, GL_FLOAT, GL_FALSE, st, (void *)(6 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)lt);
        glVertexAttribPointer((GLuint)lt, 3, GL_FLOAT, GL_FALSE, st, (void *)(9 * sizeof(float)));
    }
    glGenBuffers(1, &ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof k_idx, k_idx, GL_STATIC_DRAW);

    glViewport(0, 0, 8, 8);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    /* sanity: the same vertex stage with a trivial colour FS covers the centre
     * (isolates vertex/attribute problems from the real fragment shader).  The
     * real program's own uniforms are already set below, so the pass only
     * swaps the fragment shader. */
    {
        static const char *dbg_fs =
            "precision mediump float;\n"
            "varying vec4 vWorld;\n"
            "void main() { gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0); }\n";
        GLuint dprog, dvs, dfs;
        GLubyte s64[64][4];
        dvs = compile1(GL_VERTEX_SHADER, vsrc);
        dfs = compile1(GL_FRAGMENT_SHADER, dbg_fs);
        dprog = glCreateProgram();
        glAttachShader(dprog, dvs); glAttachShader(dprog, dfs);
        glLinkProgram(dprog);
        { GLint ok = 0; glGetProgramiv(dprog, GL_LINK_STATUS, &ok); EXPECT_TRUE(ok); }
        glUseProgram(dprog);
        /* re-point the attribute pointers at THIS program's locations (the
         * compiler may not reuse the real program's slots) */
        {
            GLint dp = glGetAttribLocation(dprog, "aVertexPosition");
            GLsizei st = 11 * (GLsizei)sizeof(float);
            if (dp >= 0)
            {
                glEnableVertexAttribArray((GLuint)dp);
                glVertexAttribPointer((GLuint)dp, 3, GL_FLOAT, GL_FALSE, st, (void *)0);
            }
        }
        {
            GLint dv = glGetUniformLocation(dprog, "uWorldViewProj");
            GLfloat ident[16];
            for (int i = 0; i < 16; i++) ident[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            if (dv >= 0) glUniformMatrix4fv(dv, 1, GL_FALSE, ident);
            dv = glGetUniformLocation(dprog, "uWorld");
            if (dv >= 0) glUniformMatrix4fv(dv, 1, GL_FALSE, uWorld);
        }
        glDisable(GL_BLEND);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
        { GLenum e = glGetError(); if (e) MGL_FAIL("sanity draw error 0x%x", e); }
        glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, s64);
        while (glGetError() != GL_NO_ERROR) {}
        {
            int cov = 0;
            for (int i = 0; i < 64; i++)
                if (s64[i][0] > 100 || s64[i][2] > 100) cov++;
            fprintf(stderr, "  note: sanity coverage=%d/64\n", cov);
            EXPECT_TRUE(cov > 0);
        }
        glDeleteProgram(dprog); glDeleteShader(dvs); glDeleteShader(dfs);
        /* restore the real program's vertex attribute bindings */
        glUseProgram(prog);
        {
            GLsizei st = 11 * (GLsizei)sizeof(float);
            glEnableVertexAttribArray((GLuint)lp);
            glVertexAttribPointer((GLuint)lp, 3, GL_FLOAT, GL_FALSE, st, (void *)0);
        }
        glEnable(GL_BLEND);
    }

    /* pass A: real caustic web on unit 1 */
    glClearColor(0.02f, 0.03f, 0.09f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, tex[0]);
    glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, tex[17]);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
    { GLenum e = glGetError(); if (e) MGL_FAIL("draw A error 0x%x", e); }
    glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, px_web);
    while (glGetError() != GL_NO_ERROR) {}

    /* pass B: solid black "caustics" — the caustic term must vanish */
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, tex[0]);
    glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, tex[33]);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
    { GLenum e = glGetError(); if (e) MGL_FAIL("draw B error 0x%x", e); }
    glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, px_black);
    while (glGetError() != GL_NO_ERROR) {}

    /* the caustic web is mid-grey noise (mean ~0.4): every bell pixel must be
     * measurably brighter with it than with black */
    for (int i = 0; i < 4; i++)
        fprintf(stderr, "  note: px_web[%d]=%d,%d,%d,%d px_black[%d]=%d,%d,%d,%d\n", i,
                px_web[i][0], px_web[i][1], px_web[i][2], px_web[i][3], i,
                px_black[i][0], px_black[i][1], px_black[i][2], px_black[i][3]);
    for (int i = 0; i < 64; i++)
    {
        int dw = (px_web[i][0] + px_web[i][1] + px_web[i][2]) -
                 (px_black[i][0] + px_black[i][1] + px_black[i][2]);
        if (dw < 24) diff++;   /* >8 per channel on average */
    }
    if (diff) fprintf(stderr, "  note: %d/64 pixels show no caustic contribution\n", diff);
    EXPECT_EQ_INT(diff, 0);

    /* the caustic web must vary across the bell: strong channel spread */
    {
        int mn = 255, mx = 0;
        for (int i = 0; i < 64; i++)
        {
            int v = (px_web[i][0] + px_web[i][1] + px_web[i][2]) / 3;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        fprintf(stderr, "  note: web min=%d max=%d spread=%d\n", mn, mx, mx - mn);
        EXPECT_TRUE(mx - mn > 25);
    }

    /* REPEAT-after-upload sampling check (the scene sets WRAP after the
     * glTexImage2D): a 4x4 grayscale texture, LINEAR, sample at u=10.0 wraps
     * to 2.0 -> the 2x2 average of texels (1,1),(2,1),(1,2),(2,2) = 72;
     * CLAMP_TO_EDGE would clamp to the u=3 column -> 136 */
    {
        static const char *wrap_fs =
            "precision mediump float;\n"
            "varying vec2 vP;\n"
            "uniform sampler2D s;\n"
            "void main() { gl_FragColor = texture2D(s, vec2(10.0, 2.0)); }\n";
        static const char *wrap_vs =
            "attribute vec2 aP;\n"
            "varying vec2 vP;\n"
            "void main() { vP = aP * 0.5 + 0.5; gl_Position = vec4(aP, 0.0, 1.0); }\n";
        GLubyte wr_rgba[4 * 4 * 4];
        GLubyte c4[4];
        GLuint wt, wprog, wvs, wfs, wvao, wvbo;
        float quad[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
        GLushort widx[6] = { 0, 1, 2, 0, 2, 3 };
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                for (int k = 0; k < 4; k++)
                    wr_rgba[(y * 4 + x) * 4 + k] = (GLubyte)(x * 40 + y * 8);
        glGenTextures(1, &wt);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, wt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, wr_rgba);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        while (glGetError() != GL_NO_ERROR) {}
        wvs = compile1(GL_VERTEX_SHADER, wrap_vs);
        wfs = compile1(GL_FRAGMENT_SHADER, wrap_fs);
        wprog = glCreateProgram();
        glAttachShader(wprog, wvs); glAttachShader(wprog, wfs);
        glLinkProgram(wprog);
        { GLint ok = 0; glGetProgramiv(wprog, GL_LINK_STATUS, &ok); EXPECT_TRUE(ok); }
        glUseProgram(wprog);
        { GLint l = glGetUniformLocation(wprog, "s"); glUniform1i(l, 1); }
        { GLint l = glGetAttribLocation(wprog, "aP");
          glEnableVertexAttribArray((GLuint)l);
          glVertexAttribPointer((GLuint)l, 2, GL_FLOAT, GL_FALSE, 0, 0); }
        glGenVertexArrays(1, &wvao); glBindVertexArray(wvao);
        glGenBuffers(1, &wvbo); glBindBuffer(GL_ARRAY_BUFFER, wvbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
        { GLint l = glGetAttribLocation(wprog, "aP");
          glEnableVertexAttribArray((GLuint)l);
          glVertexAttribPointer((GLuint)l, 2, GL_FLOAT, GL_FALSE, 0, 0); }
        {
            GLuint wibo;
            glGenBuffers(1, &wibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof widx, widx, GL_STATIC_DRAW);
        }
        glDisable(GL_BLEND);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
        { GLenum e = glGetError(); if (e) MGL_FAIL("wrap draw error 0x%x", e); }
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c4);
        while (glGetError() != GL_NO_ERROR) {}
        fprintf(stderr, "  note: wrapped sample = %d (expect 72, clamp would be 136)\n", c4[0]);
        EXPECT_TRUE(c4[0] > 68 && c4[0] < 76);
        glUseProgram(prog);
        glEnable(GL_BLEND);
    }

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(jellyfish_core_caustics);
    return mgl_test_summary();
}
