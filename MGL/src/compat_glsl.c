/*
 * compat_glsl.c — compatibility-profile GLSL (compat domain).
 *
 * glslang cannot compile the compatibility profile to SPIR-V, so legacy
 * shaders (#version 110..150 and "#version 4xx compatibility") are translated
 * at the source level into GLSL 460 core before compilation:
 *
 *   attribute/varying            -> in/out
 *   gl_Vertex/gl_Normal/...      -> layout(location=N) in (fixed locations,
 *                                   shared with the generated fixed function)
 *   gl_ModelViewMatrix & friends -> mgl_* uniforms fed by the compat domain
 *   gl_FrontColor/gl_TexCoord[]  -> mgl_* varyings (same names both stages)
 *   gl_FragColor/gl_FragData[n]  -> mgl_* outputs at the draw-buffer locations
 *   texture2D/texture3D/...      -> texture() shims in the preamble
 *   ftransform(), gl_ClipVertex  -> handled with a main() wrapper
 *
 * mgl_compat_sync_builtin_uniforms() feeds the mgl_* uniforms of a linked
 * program from the matrix/lighting state at every draw.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

#define CS() (ctx->state.ext_compat)

#pragma mark translator state

typedef struct {
    const char *src;
    size_t len, pos;
    /* output */
    char *out;
    size_t olen, ocap;

    /* which legacy names were used (drives the preamble) */
    unsigned used;
    int subst;                     /* >0 when legacy syntax was found */
    int user_loc;                  /* next location for user attributes */
    unsigned fragdata_used;         /* bit n -> gl_FragData[n] */
    int stage;                     /* GL_VERTEX_SHADER / GL_FRAGMENT_SHADER */
    int version;                   /* declared #version number */
    int profile_compat;
} Tr;

enum {
    U_VERTEX = 1 << 0, U_NORMAL = 1 << 1, U_COLOR = 1 << 2, U_SECONDARY = 1 << 3,
    U_FOGCOORD = 1 << 4, U_TEXCOORD0 = 1 << 5,      /* texcoord mask: bits 5..12 -> units 0..7 */
    U_FRONT_COLOR = 1 << 13, U_BACK_COLOR = 1 << 14, U_FRONT_SEC = 1 << 15, U_BACK_SEC = 1 << 16,
    U_TEXCOORD_OUT0 = 1 << 17,                      /* varyings from the VS: bits 17..24 */
    U_FRAG_COLOR = 1 << 25, U_FRAG_DATA = 1 << 26, U_FRAG_DEPTH = 1 << 27,
    U_MVM = 1 << 28, U_MATRICES = 1 << 29, U_LIGHTS = 1 << 30, U_MISC = 1u << 31,
};

#define TEXCOORD_BIT(u)  (U_TEXCOORD0 << (u))
#define TEXOUT_BIT(u)    (U_TEXCOORD_OUT0 << (u))

static void tr_add(Tr *t, const char *s, size_t n)
{
    if (t->olen + n + 1 > t->ocap)
    {
        while (t->olen + n + 1 > t->ocap)
            t->ocap = t->ocap ? t->ocap * 2 : 4096;
        t->out = (char *)realloc(t->out, t->ocap);
    }
    memcpy(t->out + t->olen, s, n);
    t->olen += n;
    t->out[t->olen] = 0;
}

static void tr_adds(Tr *t, const char *s)
{
    tr_add(t, s, strlen(s));
}

#pragma mark token scanning

static int tr_peek(Tr *t)
{
    return t->pos < t->len ? (unsigned char)t->src[t->pos] : -1;
}

static int tr_at(Tr *t, size_t off)
{
    return t->pos + off < t->len ? (unsigned char)t->src[t->pos + off] : -1;
}

static bool ident_start(int c)
{
    return isalpha(c) || c == '_';
}

static bool ident_char(int c)
{
    return isalnum(c) || c == '_';
}

/* reads an identifier at t->pos; returns its length (0 = none) */
static size_t tr_ident(Tr *t, const char **start)
{
    size_t n = 0;

    *start = t->src + t->pos;

    if (!ident_start(tr_peek(t)))
        return 0;

    while (t->pos + n < t->len && ident_char((unsigned char)t->src[t->pos + n]))
        n++;

    return n;
}

static bool match(const char *id, size_t n, const char *kw)
{
    return strlen(kw) == n && memcmp(id, kw, n) == 0;
}

#pragma mark substitution

/* rewrite one identifier; returns true when it consumed something */
static bool tr_subst_ident(Tr *t, const char *id, size_t n)
{
    /* counted so shaders without legacy syntax pass through untouched */
    /* vertex attributes (compat) */
    if (t->stage == GL_VERTEX_SHADER)
    {
        if (match(id, n, "gl_Vertex")) { tr_adds(t, "mglVertex"); t->used |= U_VERTEX; return (t->subst++, true); }
        if (match(id, n, "gl_Normal")) { tr_adds(t, "mglNormal"); t->used |= U_NORMAL; return (t->subst++, true); }
        if (match(id, n, "gl_Color")) { tr_adds(t, "mglColor"); t->used |= U_COLOR; return (t->subst++, true); }
        if (match(id, n, "gl_SecondaryColor")) { tr_adds(t, "mglSecondaryColor"); t->used |= U_SECONDARY; return (t->subst++, true); }
        if (match(id, n, "gl_FogCoord")) { tr_adds(t, "mglFogCoord"); t->used |= U_FOGCOORD; return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord0")) { tr_adds(t, "mglTexCoord0"); t->used |= TEXCOORD_BIT(0); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord1")) { tr_adds(t, "mglTexCoord1"); t->used |= TEXCOORD_BIT(1); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord2")) { tr_adds(t, "mglTexCoord2"); t->used |= TEXCOORD_BIT(2); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord3")) { tr_adds(t, "mglTexCoord3"); t->used |= TEXCOORD_BIT(3); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord4")) { tr_adds(t, "mglTexCoord4"); t->used |= TEXCOORD_BIT(4); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord5")) { tr_adds(t, "mglTexCoord5"); t->used |= TEXCOORD_BIT(5); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord6")) { tr_adds(t, "mglTexCoord6"); t->used |= TEXCOORD_BIT(6); return (t->subst++, true); }
        if (match(id, n, "gl_MultiTexCoord7")) { tr_adds(t, "mglTexCoord7"); t->used |= TEXCOORD_BIT(7); return (t->subst++, true); }

        /* varyings written by the vertex shader */
        if (match(id, n, "gl_FrontColor")) { tr_adds(t, "mglFrontColor"); t->used |= U_FRONT_COLOR; return (t->subst++, true); }
        if (match(id, n, "gl_BackColor")) { tr_adds(t, "mglBackColor"); t->used |= U_BACK_COLOR; return (t->subst++, true); }
        if (match(id, n, "gl_FrontSecondaryColor")) { tr_adds(t, "mglFrontSecondaryColor"); t->used |= U_FRONT_SEC; return (t->subst++, true); }
        if (match(id, n, "gl_BackSecondaryColor")) { tr_adds(t, "mglBackSecondaryColor"); t->used |= U_BACK_SEC; return (t->subst++, true); }
        if (match(id, n, "gl_TexCoord")) { tr_adds(t, "mglTexCoordOut"); t->used |= 0; return (t->subst++, true); }
        if (match(id, n, "gl_ClipVertex")) { tr_adds(t, "mglClipVertex"); t->used |= U_MISC; return (t->subst++, true); }
        if (match(id, n, "ftransform")) { tr_adds(t, "mgl_ftransform"); t->used |= U_MVM | U_MATRICES; return (t->subst++, true); }
    }
    else
    {
        if (match(id, n, "gl_Color")) { tr_adds(t, "mglFrontColor"); t->used |= U_FRONT_COLOR; return (t->subst++, true); }
        if (match(id, n, "gl_SecondaryColor")) { tr_adds(t, "mglFrontSecondaryColor"); t->used |= U_FRONT_SEC; return (t->subst++, true); }
        if (match(id, n, "gl_TexCoord")) { tr_adds(t, "mglTexCoordIn"); return (t->subst++, true); }
        if (match(id, n, "gl_FragColor")) { tr_adds(t, "mglFragColor"); t->used |= U_FRAG_COLOR; return (t->subst++, true); }
        if (match(id, n, "gl_FragData")) { t->used |= U_FRAG_DATA; return false; }   /* handled by the caller */
        if (match(id, n, "gl_FogFragCoord")) { tr_adds(t, "mglFogFragCoord"); t->used |= U_MISC; return (t->subst++, true); }
    }

    /* matrices and other built-in uniforms (both stages) */
    if (match(id, n, "gl_ModelViewMatrix")) { tr_adds(t, "mglModelViewMatrix"); t->used |= U_MVM; return (t->subst++, true); }
    if (match(id, n, "gl_ProjectionMatrix")) { tr_adds(t, "mglProjectionMatrix"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewProjectionMatrix")) { tr_adds(t, "mglModelViewProjectionMatrix"); t->used |= U_MVM | U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewMatrixInverse")) { tr_adds(t, "mglModelViewMatrixInverse"); t->used |= U_MVM; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewMatrixTranspose")) { tr_adds(t, "mglModelViewMatrixTranspose"); t->used |= U_MVM; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewMatrixInverseTranspose")) { tr_adds(t, "mglModelViewMatrixInverseTranspose"); t->used |= U_MVM; return (t->subst++, true); }
    if (match(id, n, "gl_ProjectionMatrixInverse")) { tr_adds(t, "mglProjectionMatrixInverse"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ProjectionMatrixTranspose")) { tr_adds(t, "mglProjectionMatrixTranspose"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ProjectionMatrixInverseTranspose")) { tr_adds(t, "mglProjectionMatrixInverseTranspose"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewProjectionMatrixInverse")) { tr_adds(t, "mglModelViewProjectionMatrixInverse"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewProjectionMatrixTranspose")) { tr_adds(t, "mglModelViewProjectionMatrixTranspose"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_ModelViewProjectionMatrixInverseTranspose")) { tr_adds(t, "mglModelViewProjectionMatrixInverseTranspose"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_TextureMatrix")) { tr_adds(t, "mglTextureMatrix"); t->used |= U_MATRICES; return (t->subst++, true); }
    if (match(id, n, "gl_NormalMatrix")) { tr_adds(t, "mglNormalMatrix"); t->used |= U_MVM; return (t->subst++, true); }
    if (match(id, n, "gl_NormalScale")) { tr_adds(t, "mglNormalScale"); t->used |= U_MVM; return (t->subst++, true); }
    if (match(id, n, "gl_LightModel")) { tr_adds(t, "mglLightModel"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_FrontLightModelProduct")) { tr_adds(t, "mglFrontLightModelProduct"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_BackLightModelProduct")) { tr_adds(t, "mglBackLightModelProduct"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_LightSource")) { tr_adds(t, "mglLightSource"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_FrontLightProduct")) { tr_adds(t, "mglFrontLightProduct"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_BackLightProduct")) { tr_adds(t, "mglBackLightProduct"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_FrontMaterial")) { tr_adds(t, "mglFrontMaterial"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_BackMaterial")) { tr_adds(t, "mglBackMaterial"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_Fog")) { tr_adds(t, "mglFog"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_ClipPlane")) { tr_adds(t, "mglClipPlane"); t->used |= U_LIGHTS; return (t->subst++, true); }
    if (match(id, n, "gl_Point")) { tr_adds(t, "mglPoint"); t->used |= U_LIGHTS; return (t->subst++, true); }

    /* deprecated texture functions */
    if (match(id, n, "texture2D")) { tr_adds(t, "texture"); return (t->subst++, true); }
    if (match(id, n, "texture3D")) { tr_adds(t, "texture"); return (t->subst++, true); }
    if (match(id, n, "textureCube")) { tr_adds(t, "texture"); return (t->subst++, true); }
    if (match(id, n, "texture2DProj")) { tr_adds(t, "textureProj"); return (t->subst++, true); }
    if (match(id, n, "texture1D")) { tr_adds(t, "texture"); return (t->subst++, true); }
    if (match(id, n, "shadow2D")) { tr_adds(t, "mgl_shadow2D"); return (t->subst++, true); }
    if (match(id, n, "shadow2DProj")) { tr_adds(t, "mgl_shadow2DProj"); return (t->subst++, true); }

    return false;
}

/* true when the identifier is preceded only by whitespace since a '.' */
static bool after_dot(Tr *t)
{
    size_t p = t->pos;

    while (p > 0 && isspace((unsigned char)t->src[p - 1]))
        p--;

    return p > 0 && t->src[p - 1] == '.';
}

#pragma mark the translation

char *mgl_compat_translate_shader(GLMContext ctx, GLuint type, GLsizei count,
                                  const GLchar *const *string, const GLint *length)
{
    Tr t;
    size_t total = 0;
    char *joined;

    (void)ctx;

    if (count <= 0 || !string)
        return NULL;

    /* join the strings */
    for (GLsizei i = 0; i < count; i++)
        total += (length && length[i] >= 0) ? (size_t)length[i] : strlen(string[i]);

    joined = (char *)malloc(total + 1);
    if (!joined)
        return NULL;

    {
        char *p = joined;
        for (GLsizei i = 0; i < count; i++)
        {
            size_t n = (length && length[i] >= 0) ? (size_t)length[i] : strlen(string[i]);
            memcpy(p, string[i], n);
            p += n;
        }
        *p = 0;
    }

    memset(&t, 0, sizeof t);
    t.src = joined;
    t.len = total;
    t.stage = (int)type;
    t.user_loc = MAX_ATTRIBS - 1;
    t.out = NULL;
    t.ocap = 0;
    t.olen = 0;

    /* pass 1: scan (rewrites identifiers, notes usage, strips the version) */
    bool in_comment = false, in_line_comment = false, in_preprocessor = false;

    while (t.pos < t.len)
    {
        char c = t.src[t.pos];

        /* comment handling */
        if (!in_line_comment && !in_comment && c == '/' && tr_at(&t, 1) == '*')
        {
            in_comment = true;
            tr_add(&t, &c, 1);
            t.pos++;
            continue;
        }
        if (in_comment)
        {
            if (c == '*' && tr_at(&t, 1) == '/')
            {
                in_comment = false;
                tr_add(&t, "*/", 2);
                t.pos += 2;
            }
            else
            {
                tr_add(&t, &c, 1);
                t.pos++;
            }
            continue;
        }
        if (!in_line_comment && c == '/' && tr_at(&t, 1) == '/')
        {
            in_line_comment = true;
            tr_add(&t, "//", 2);
            t.pos += 2;
            continue;
        }
        if (in_line_comment)
        {
            if (c == '\n')
            {
                in_line_comment = false;
                in_preprocessor = false;
            }
            tr_add(&t, &c, 1);
            t.pos++;
            continue;
        }

        /* #version: capture and drop (rewritten in the preamble) */
        if (c == '#' && (t.pos + 8 <= t.len) && strncmp(t.src + t.pos, "#version", 8) == 0)
        {
            size_t e = t.pos;
            while (e < t.len && t.src[e] != '\n')
                e++;
            {
                char line[128];
                size_t n = e - t.pos;
                if (n >= sizeof line) n = sizeof line - 1;
                memcpy(line, t.src + t.pos, n);
                line[n] = 0;
                int v = 110;
                char prof[32] = { 0 };
                if (sscanf(line, "#version %d %31s", &v, prof) >= 1)
                {
                    t.version = v;
                    if (strcmp(prof, "compatibility") == 0)
                        t.profile_compat = 1;
                }
            }
            t.pos = e;      /* keep the newline for the next iteration */
            continue;
        }

        if (c == '#')
        {
            in_preprocessor = true;
            tr_add(&t, &c, 1);
            t.pos++;
            continue;
        }

        /* attribute / varyings keywords */
        if (ident_start((unsigned char)c) && !after_dot(&t))
        {
            const char *id;
            size_t n = tr_ident(&t, &id);

            if (n)
            {
                if (match(id, n, "attribute") && !in_preprocessor)
                {
                    /* explicit location: the legacy fixed locations are taken
                     * by gl_Vertex/gl_Color/... when the shader uses them */
                    char loc[48];
                    snprintf(loc, sizeof loc, "layout(location = %d) in", t.user_loc--);
                    tr_adds(&t, loc);
                    t.subst++;
                    t.pos += n;
                    continue;
                }
                if (match(id, n, "varying") && !in_preprocessor)
                {
                    tr_adds(&t, t.stage == GL_VERTEX_SHADER ? "out" : "in");
                    t.subst++;
                    t.pos += n;
                    continue;
                }

                if (tr_subst_ident(&t, id, n))
                {
                    t.pos += n;
                    continue;
                }

                /* gl_FragData[n] -> mglFragData_n at location n */
                if (t.stage != GL_VERTEX_SHADER && match(id, n, "gl_FragData"))
                {
                    const char *q = t.src + t.pos + n;
                    size_t rem = t.len - (t.pos + n);
                    int idx = -1;

                    while (rem > 0 && (*q == ' ' || *q == '\t')) { q++; rem--; }

                    if (rem > 0 && *q == '[')
                    {
                        const char *r = q + 1;
                        int v = 0;
                        while ((size_t)(r - t.src) < t.len && isdigit((unsigned char)*r))
                        {
                            v = v * 10 + (*r - '0');
                            r++;
                        }
                        while ((size_t)(r - t.src) < t.len && (*r == ' ' || *r == '\t')) r++;
                        if ((size_t)(r - t.src) < t.len && *r == ']' && v >= 0 && v < 8)
                            idx = v;
                    }

                    if (idx >= 0)
                    {
                        char name[32];
                        snprintf(name, sizeof name, "mglFragData_%d", idx);
                        tr_adds(&t, name);
                        t.fragdata_used |= 1u << idx;
                        t.pos += n;
                        /* skip "[idx]" */
                        t.pos = (size_t)(q - t.src) + 1;   /* after '[' */
                        while (t.pos < t.len && isdigit((unsigned char)t.src[t.pos])) t.pos++;
                        while (t.pos < t.len && (t.src[t.pos] == ' ' || t.src[t.pos] == '\t')) t.pos++;
                        if (t.pos < t.len && t.src[t.pos] == ']') t.pos++;
                        continue;
                    }

                    tr_adds(&t, "mglFragColor");
                    t.used |= U_FRAG_COLOR;
                    t.pos += n;
                    continue;
                }
            }
        }

        if (c == '\n')
            in_preprocessor = false;

        tr_add(&t, &c, 1);
        t.pos++;
    }

    /* no legacy syntax and a modern #version: leave the source untouched */
    if (t.subst == 0 && t.version >= 330 && !t.profile_compat)
    {
        free(joined);
        free(t.out);
        return NULL;
    }

    /* build the final source: preamble + rewritten body */
    {
        char *body = t.out;
        char *src = (char *)malloc((body ? strlen(body) : 0) + 8192);

        if (!src)
        {
            free(body);
            free(joined);
            return NULL;
        }

        char *p = src;
        p += sprintf(p, "#version 460 core\n");

        if (t.stage == GL_VERTEX_SHADER)
        {
            if (t.used & U_VERTEX) p += sprintf(p, "layout(location=%d) in vec4 mglVertex;\n", MGLC_ATTR_VERTEX);
            if (t.used & U_NORMAL) p += sprintf(p, "layout(location=%d) in vec3 mglNormal;\n", MGLC_ATTR_NORMAL);
            if (t.used & U_COLOR) p += sprintf(p, "layout(location=%d) in vec4 mglColor;\n", MGLC_ATTR_COLOR);
            if (t.used & U_SECONDARY) p += sprintf(p, "layout(location=%d) in vec4 mglSecondaryColor;\n", MGLC_ATTR_SECONDARY);
            if (t.used & U_FOGCOORD) p += sprintf(p, "layout(location=%d) in float mglFogCoord;\n", MGLC_ATTR_FOG);
            for (int u = 0; u < MGLC_MAX_COORDS; u++)
                if (t.used & TEXCOORD_BIT(u))
                    p += sprintf(p, "layout(location=%d) in vec4 mglTexCoord%d;\n", MGLC_ATTR_TEXCOORD(u), u);

            /* varyings: always declared so VS/FS pairs always link */
            p += sprintf(p, "out vec4 mglFrontColor;\n");
            p += sprintf(p, "out vec4 mglBackColor;\n");
            p += sprintf(p, "out vec4 mglFrontSecondaryColor;\n");
            p += sprintf(p, "out vec4 mglBackSecondaryColor;\n");
            p += sprintf(p, "out vec4 mglTexCoordOut[%d];\n", MGLC_MAX_COORDS);
            p += sprintf(p, "out float mglFogFragCoord;\n");
            p += sprintf(p, "out vec4 mglClipVertex;\n");
        }
        else
        {
            p += sprintf(p, "in vec4 mglFrontColor;\n");
            p += sprintf(p, "in vec4 mglBackColor;\n");
            p += sprintf(p, "in vec4 mglFrontSecondaryColor;\n");
            p += sprintf(p, "in vec4 mglBackSecondaryColor;\n");
            p += sprintf(p, "in vec4 mglTexCoordIn[%d];\n", MGLC_MAX_COORDS);
            p += sprintf(p, "in float mglFogFragCoord;\n");
            p += sprintf(p, "in vec4 mglClipVertex;\n");

            if (t.fragdata_used)
            {
                for (int i = 0; i < 8; i++)
                    if (t.fragdata_used & (1u << i))
                        p += sprintf(p, "layout(location=%d) out vec4 mglFragData_%d;\n", i, i);
            }
            else
            {
                p += sprintf(p, "layout(location=0) out vec4 mglFragColor;\n");
            }
        }

        /* matrices / state uniforms */
        if (t.used & U_MVM)
        {
            p += sprintf(p, "uniform mat4 mglModelViewMatrix;\n");
            p += sprintf(p, "uniform mat4 mglModelViewMatrixInverse;\n");
            p += sprintf(p, "uniform mat4 mglModelViewMatrixTranspose;\n");
            p += sprintf(p, "uniform mat4 mglModelViewMatrixInverseTranspose;\n");
            p += sprintf(p, "uniform mat3 mglNormalMatrix;\n");
            p += sprintf(p, "uniform float mglNormalScale;\n");
        }
        if (t.used & U_MATRICES)
        {
            p += sprintf(p, "uniform mat4 mglProjectionMatrix;\n");
            p += sprintf(p, "uniform mat4 mglProjectionMatrixInverse;\n");
            p += sprintf(p, "uniform mat4 mglProjectionMatrixTranspose;\n");
            p += sprintf(p, "uniform mat4 mglProjectionMatrixInverseTranspose;\n");
            p += sprintf(p, "uniform mat4 mglModelViewProjectionMatrix;\n");
            p += sprintf(p, "uniform mat4 mglModelViewProjectionMatrixInverse;\n");
            p += sprintf(p, "uniform mat4 mglModelViewProjectionMatrixTranspose;\n");
            p += sprintf(p, "uniform mat4 mglModelViewProjectionMatrixInverseTranspose;\n");
            p += sprintf(p, "uniform mat4 mglTextureMatrix[%d];\n", MGLC_MAX_COORDS);
        }
        if (t.used & (U_MVM | U_MATRICES))
        {
            /* ftransform() reads gl_Vertex even when the shader does not */
            if (!(t.used & U_VERTEX))
                p += sprintf(p, "layout(location=%d) in vec4 mglVertex;\n", MGLC_ATTR_VERTEX);
            p += sprintf(p, "vec4 mgl_ftransform() { return mglModelViewProjectionMatrix * mglVertex; }\n");
        }
        if (t.used & U_LIGHTS)
        {
            p += sprintf(p,
                "struct mgl_LightSourceParameters { vec4 ambient; vec4 diffuse; vec4 specular; vec4 position; "
                "vec3 spotDirection; float spotExponent; float spotCutoff; float spotCosCutoff; "
                "float constantAttenuation; float linearAttenuation; float quadraticAttenuation; };\n");
            p += sprintf(p, "uniform mgl_LightSourceParameters mglLightSource[%d];\n", MGLC_MAX_LIGHTS);
            p += sprintf(p,
                "struct mgl_MaterialParameters { vec4 emission; vec4 ambient; vec4 diffuse; vec4 specular; float shininess; };\n");
            p += sprintf(p, "uniform mgl_MaterialParameters mglFrontMaterial;\n");
            p += sprintf(p, "uniform mgl_MaterialParameters mglBackMaterial;\n");
            p += sprintf(p, "uniform vec4 mglLightModelAmbient;\n");
            p += sprintf(p, "uniform vec4 mglFrontLightModelProduct;\n");
            p += sprintf(p, "uniform vec4 mglBackLightModelProduct;\n");
            p += sprintf(p,
                "struct mgl_LightProducts { vec4 ambient; vec4 diffuse; vec4 specular; };\n");
            p += sprintf(p, "uniform mgl_LightProducts mglFrontLightProduct[%d];\n", MGLC_MAX_LIGHTS);
            p += sprintf(p, "uniform mgl_LightProducts mglBackLightProduct[%d];\n", MGLC_MAX_LIGHTS);
            p += sprintf(p,
                "struct mgl_FogParameters { vec4 color; float density; float start; float end; float scale; };\n");
            p += sprintf(p, "uniform mgl_FogParameters mglFog;\n");
            p += sprintf(p, "uniform vec4 mglClipPlane[%d];\n", MGLC_MAX_CLIP_PLANES);
            p += sprintf(p,
                "struct mgl_PointParameters { float size; float sizeMin; float sizeMax; vec3 attenuation; float fadeThresholdSize; };\n");
            p += sprintf(p, "uniform mgl_PointParameters mglPoint;\n");
        }

        /* shadow helper functions */
        p += sprintf(p,
            "vec4 mgl_shadow2D(sampler2DShadow s, vec3 c) { return vec4(texture(s, c)); }\n"
            "vec4 mgl_shadow2DProj(sampler2DShadow s, vec4 c) { return vec4(textureProj(s, c)); }\n");

        p += sprintf(p, "%s", body ? body : "");

        free(body);
        free(joined);
        return src;
    }
}

#pragma mark built-in uniform sync for translated user shaders

/* look up the location of "name" or "name[0]"; -1 when absent */
static GLint uloc(GLMContext ctx, GLuint prog, const char *name)
{
    return mglGetUniformLocation(ctx, prog, name);
}

static GLint uloc_arr(GLMContext ctx, GLuint prog, const char *name, int idx)
{
    char buf[96];
    snprintf(buf, sizeof buf, "%s[%d]", name, idx);
    return mglGetUniformLocation(ctx, prog, buf);
}

static GLint uloc_mem(GLMContext ctx, GLuint prog, const char *name, int idx, const char *member)
{
    char buf[96];
    snprintf(buf, sizeof buf, "%s[%d].%s", name, idx, member);
    GLint l = mglGetUniformLocation(ctx, prog, buf);
    return l;
}

void mgl_compat_sync_builtin_uniforms(GLMContext ctx, Program *program)
{
    struct MGLExtState_compat *s = CS();
    GLuint prog = program->name;
    GLint l;

    if (!program || !program->linked_glsl_program)
        return;

    /* cached per program and state serial */
    struct MGLCBuiltinProg_t *bp = NULL;

    for (int i = 0; i < s->builtin_prog_count; i++)
        if (s->builtin_prog[i].prog == program)
            bp = &s->builtin_prog[i];

    if (!bp)
    {
        if (s->builtin_prog_count < 8)
            bp = &s->builtin_prog[s->builtin_prog_count++];
        else
            bp = &s->builtin_prog[0];
        bp->prog = program;
        bp->link_id = NULL;
        bp->sync_serial = 0;
    }

    if (bp->link_id == program->linked_glsl_program && bp->sync_serial == s->state_serial)
        return;

    bp->link_id = program->linked_glsl_program;
    bp->sync_serial = s->state_serial;

    /* matrices */
    l = uloc(ctx, prog, "mglModelViewMatrix");
    if (l >= 0) mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, s->modelview_f);
    l = uloc(ctx, prog, "mglProjectionMatrix");
    if (l >= 0) mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, s->projection_f);

    {
        GLfloat mvp[16];
        GLdouble mvpr[16], pr[16];
        const GLdouble *mv = s->modelview.stack[s->modelview.depth].m;

        memcpy(pr, s->projection.stack[s->projection.depth].m, sizeof pr);
        mglc_matrix_mul(mvpr, pr, mv);
        for (int i = 0; i < 16; i++)
            mvp[i] = (GLfloat)mvpr[i];

        l = uloc(ctx, prog, "mglModelViewProjectionMatrix");
        if (l >= 0) mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, mvp);
    }

    {
        GLdouble inv[16], tr[16];
        GLfloat f[16];
        const GLdouble *mv = s->modelview.stack[s->modelview.depth].m;

        if (mglc_matrix_invert(inv, mv))
        {
            l = uloc(ctx, prog, "mglModelViewMatrixInverse");
            if (l >= 0)
            {
                for (int i = 0; i < 16; i++) f[i] = (GLfloat)inv[i];
                mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, f);
            }
            mglc_matrix_transpose(tr, inv);
            l = uloc(ctx, prog, "mglModelViewMatrixInverseTranspose");
            if (l >= 0)
            {
                for (int i = 0; i < 16; i++) f[i] = (GLfloat)tr[i];
                mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, f);
            }
            mglc_matrix_transpose(tr, mv);
            l = uloc(ctx, prog, "mglModelViewMatrixTranspose");
            if (l >= 0)
            {
                for (int i = 0; i < 16; i++) f[i] = (GLfloat)tr[i];
                mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, f);
            }
        }

        /* normal matrix + scale */
        {
            GLfloat nm[9];
            if (mglc_matrix_invert(inv, mv))
            {
                mglc_matrix_transpose(tr, inv);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        nm[i * 3 + j] = (GLfloat)tr[i * 4 + j];
            }
            else
            {
                nm[0] = nm[4] = nm[8] = 1; nm[1] = nm[2] = nm[3] = nm[5] = nm[6] = nm[7] = 0;
            }

            l = uloc(ctx, prog, "mglNormalMatrix");
            if (l >= 0) mglUniformMatrix3fv(ctx, l, 1, GL_FALSE, nm);

            double len = sqrt(mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]);
            l = uloc(ctx, prog, "mglNormalScale");
            if (l >= 0 && len > 0) mglUniform1f(ctx, l, (GLfloat)(1.0 / len));
        }
    }

    {
        const GLdouble *pr = s->projection.stack[s->projection.depth].m;
        GLdouble inv[16], tr[16];
        GLfloat f[16];

        if (mglc_matrix_invert(inv, pr))
        {
            l = uloc(ctx, prog, "mglProjectionMatrixInverse");
            if (l >= 0)
            {
                for (int i = 0; i < 16; i++) f[i] = (GLfloat)inv[i];
                mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, f);
            }
            mglc_matrix_transpose(tr, inv);
            l = uloc(ctx, prog, "mglProjectionMatrixInverseTranspose");
            if (l >= 0)
            {
                for (int i = 0; i < 16; i++) f[i] = (GLfloat)tr[i];
                mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, f);
            }
            mglc_matrix_transpose(tr, pr);
            l = uloc(ctx, prog, "mglProjectionMatrixTranspose");
            if (l >= 0)
            {
                for (int i = 0; i < 16; i++) f[i] = (GLfloat)tr[i];
                mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, f);
            }
        }
    }

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        l = uloc_arr(ctx, prog, "mglTextureMatrix", u);
        if (l >= 0) mglUniformMatrix4fv(ctx, l, 1, GL_FALSE, s->texture_f[u]);
    }

    /* lights / materials / fog */
    for (int i = 0; i < MGLC_MAX_LIGHTS; i++)
    {
        MGLCLight *lt = &s->lights[i];
        struct { const char *name; const GLfloat *v; GLint n; } tab[] = {
            { "ambient", lt->ambient, 4 },
            { "diffuse", lt->diffuse, 4 },
            { "specular", lt->specular, 4 },
            { "position", lt->position, 4 },
            { "spotDirection", lt->spot_direction, 3 },
            { NULL, NULL, 0 }
        };

        for (int k = 0; tab[k].name; k++)
        {
            l = uloc_mem(ctx, prog, "mglLightSource", i, tab[k].name);
            if (l >= 0)
                mglUniform4fv(ctx, l, 1, tab[k].v);
        }

        l = uloc_mem(ctx, prog, "mglLightSource", i, "spotExponent");
        if (l >= 0) mglUniform1f(ctx, l, lt->spot_exponent);
        l = uloc_mem(ctx, prog, "mglLightSource", i, "spotCutoff");
        if (l >= 0) mglUniform1f(ctx, l, lt->spot_cutoff);
        l = uloc_mem(ctx, prog, "mglLightSource", i, "spotCosCutoff");
        if (l >= 0) mglUniform1f(ctx, l, (lt->spot_cutoff >= 180.0f) ? -2.0f : cosf(lt->spot_cutoff * (float)M_PI / 180.0f));
        l = uloc_mem(ctx, prog, "mglLightSource", i, "constantAttenuation");
        if (l >= 0) mglUniform1f(ctx, l, lt->constant_attenuation);
        l = uloc_mem(ctx, prog, "mglLightSource", i, "linearAttenuation");
        if (l >= 0) mglUniform1f(ctx, l, lt->linear_attenuation);
        l = uloc_mem(ctx, prog, "mglLightSource", i, "quadraticAttenuation");
        if (l >= 0) mglUniform1f(ctx, l, lt->quadratic_attenuation);
    }

    {
        struct { const char *base; MGLCMaterial *m; } mats[2] = {
            { "mglFrontMaterial", &s->material_front },
            { "mglBackMaterial", &s->material_back },
        };

        for (int k = 0; k < 2; k++)
        {
            l = uloc(ctx, prog, mats[k].base);
            (void)l;
            char buf[96];

            snprintf(buf, sizeof buf, "%s.emission", mats[k].base);
            l = uloc(ctx, prog, buf);
            if (l >= 0) mglUniform4fv(ctx, l, 1, mats[k].m->emission);
            snprintf(buf, sizeof buf, "%s.ambient", mats[k].base);
            l = uloc(ctx, prog, buf);
            if (l >= 0) mglUniform4fv(ctx, l, 1, mats[k].m->ambient);
            snprintf(buf, sizeof buf, "%s.diffuse", mats[k].base);
            l = uloc(ctx, prog, buf);
            if (l >= 0) mglUniform4fv(ctx, l, 1, mats[k].m->diffuse);
            snprintf(buf, sizeof buf, "%s.specular", mats[k].base);
            l = uloc(ctx, prog, buf);
            if (l >= 0) mglUniform4fv(ctx, l, 1, mats[k].m->specular);
            snprintf(buf, sizeof buf, "%s.shininess", mats[k].base);
            l = uloc(ctx, prog, buf);
            if (l >= 0) mglUniform1f(ctx, l, mats[k].m->shininess);
        }
    }

    l = uloc(ctx, prog, "mglLightModelAmbient");
    if (l >= 0) mglUniform4fv(ctx, l, 1, s->light_model_ambient);
    l = uloc(ctx, prog, "mglFrontLightModelProduct");
    if (l >= 0)
    {
        GLfloat v[4];
        for (int i = 0; i < 4; i++)
            v[i] = s->material_front.ambient[i] * s->light_model_ambient[i];
        mglUniform4fv(ctx, l, 1, v);
    }
    l = uloc(ctx, prog, "mglBackLightModelProduct");
    if (l >= 0)
    {
        GLfloat v[4];
        for (int i = 0; i < 4; i++)
            v[i] = s->material_back.ambient[i] * s->light_model_ambient[i];
        mglUniform4fv(ctx, l, 1, v);
    }

    /* per-light products (material * light) */
    for (int i = 0; i < MGLC_MAX_LIGHTS; i++)
    {
        MGLCLight *lt = &s->lights[i];
        GLfloat v[4];

        for (int side = 0; side < 2; side++)
        {
            MGLCMaterial *m = side ? &s->material_back : &s->material_front;
            const char *base = side ? "mglBackLightProduct" : "mglFrontLightProduct";

            for (int c = 0; c < 4; c++)
                v[c] = m->ambient[c] * lt->ambient[c];
            l = uloc_mem(ctx, prog, base, i, "ambient");
            if (l >= 0) mglUniform4fv(ctx, l, 1, v);

            for (int c = 0; c < 4; c++)
                v[c] = m->diffuse[c] * lt->diffuse[c];
            l = uloc_mem(ctx, prog, base, i, "diffuse");
            if (l >= 0) mglUniform4fv(ctx, l, 1, v);

            for (int c = 0; c < 4; c++)
                v[c] = m->specular[c] * lt->specular[c];
            l = uloc_mem(ctx, prog, base, i, "specular");
            if (l >= 0) mglUniform4fv(ctx, l, 1, v);
        }
    }

    /* fog */
    l = uloc(ctx, prog, "mglFog.color");
    if (l >= 0) mglUniform4fv(ctx, l, 1, s->fog_color);
    l = uloc(ctx, prog, "mglFog.density");
    if (l >= 0) mglUniform1f(ctx, l, s->fog_density);
    l = uloc(ctx, prog, "mglFog.start");
    if (l >= 0) mglUniform1f(ctx, l, s->fog_start);
    l = uloc(ctx, prog, "mglFog.end");
    if (l >= 0) mglUniform1f(ctx, l, s->fog_end);
    l = uloc(ctx, prog, "mglFog.scale");
    if (l >= 0) mglUniform1f(ctx, l, s->fog_scale);

    /* clip planes + point params */
    for (int i = 0; i < MGLC_MAX_CLIP_PLANES; i++)
    {
        l = uloc_arr(ctx, prog, "mglClipPlane", i);
        if (l >= 0)
        {
            GLfloat v[4] = { (GLfloat)s->clip_plane[i][0], (GLfloat)s->clip_plane[i][1],
                             (GLfloat)s->clip_plane[i][2], (GLfloat)s->clip_plane[i][3] };
            mglUniform4fv(ctx, l, 1, v);
        }
    }

    l = uloc(ctx, prog, "mglPoint.size");
    if (l >= 0) mglUniform1f(ctx, l, s->point_size);
    l = uloc(ctx, prog, "mglPoint.sizeMin");
    if (l >= 0) mglUniform1f(ctx, l, s->point_params[0]);
    l = uloc(ctx, prog, "mglPoint.sizeMax");
    if (l >= 0) mglUniform1f(ctx, l, s->point_params[1]);
    l = uloc(ctx, prog, "mglPoint.attenuation");
    if (l >= 0)
    {
        GLfloat v[3] = { s->point_params[3], s->point_params[4], s->point_params[5] };
        mglUniform3fv(ctx, l, 1, v);
    }
}
