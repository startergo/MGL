/* Writable multisample images — KHR-GL46.gl_spirv.spirv_validation_capabilities
 * (the image/sampler type zoo).  Metal's texture2d_ms / texture2d_ms_array are
 * READ-ONLY: `texture2d_ms<float, access::write>' fails the Metal compile
 * ("invalid access value for 'texture2d_ms'"), and there is no .write() member
 * at all.  MGL reports GL_MAX_IMAGE_SAMPLES = 0 (MS image writes do not exist
 * on Metal), but a shader that still declares image2DMS / image2DMSArray with
 * stores must compile and link.  Emulation: the declaration's access is
 * rewritten to read, and every `<name>.write(...)' call on such a texture
 * becomes `(void)0' — the store is dropped (documented gap; MS image READS
 * work through read(coord, sample) and stay untouched).
 * Returns a malloc'd fixed copy, or NULL when nothing needed changing. */

/*
 * (implementation note: kept out of msl_reserved.c — this is the gl_spirv
 * capabilities-zoo pass; the other MSL fixups live in msl_reserved.c)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include "msl_reserved.h"

char *mgl_msl_fix_ms_image_writes(const char *msl)
{
    static const char kMs[]       = "texture2d_ms<";
    static const char kMsArr[]    = "texture2d_ms_array<";
    static const char kWrite[]    = "access::write";
    static const char kReadWrite[] = "access::read_write";
    static const char kRead[]     = "access::read";

    if (!msl) return NULL;
    if (!strstr(msl, kMs) && !strstr(msl, kMsArr)) return NULL;

    /* -------- pass A: collect writable _ms texture declarations ---------------- */
    char *work = strdup(msl);
    if (!work) return NULL;

    const char *names[64];
    int nn = 0;
    bool any = false;

    for (const char *p = work; *p; )
    {
        const char *decl = NULL;
        if (!strncmp(p, kMsArr, strlen(kMsArr))) decl = p + strlen(kMsArr) - 1;
        else if (!strncmp(p, kMs, strlen(kMs)))  decl = p + strlen(kMs) - 1;
        if (!decl) { p++; continue; }

        /* matching '>' of the template argument list */
        int depth = 1;
        const char *q = decl + 1;
        while (*q && depth > 0)
        {
            if (*q == '<') depth++;
            else if (*q == '>') depth--;
            q++;
        }
        if (depth != 0) { p = q; continue; }
        const char *close = q - 1;           /* the '>' */

        /* writable? ("access::write" also matches inside "access::read_write" —
         * both are rewritten, so a prefix test for "access::read_write" first) */
        const char *acc = NULL;
        size_t acclen = 0;
        for (const char *r = decl + 1; r < close; r++)
        {
            if (!strncmp(r, kReadWrite, strlen(kReadWrite))) { acc = r; acclen = strlen(kReadWrite); break; }
            if (!strncmp(r, kWrite, strlen(kWrite))) { acc = r; acclen = strlen(kWrite); break; }
        }
        if (acc)
        {
            /* variable name after '>' */
            const char *n = close + 1;
            while (*n == ' ' || *n == '\t') n++;
            const char *n0 = n;
            while (*n && (isalnum((unsigned char)*n) || *n == '_')) n++;
            if (n > n0 && nn < 64)
            {
                size_t nb = (size_t)(n - n0);
                char *nm = malloc(nb + 1);
                if (nm)
                {
                    memcpy(nm, n0, nb);
                    nm[nb] = 0;
                    names[nn++] = nm;
                }
                /* same-length in-place rewrite keeps every later offset valid */
                char *w = (char *)acc;
                memcpy(w, kRead, strlen(kRead));
                for (size_t k2 = strlen(kRead); k2 < acclen; k2++) w[k2] = ' ';
                any = true;
            }
        }
        p = q;
    }
    if (!any || nn == 0)
    {
        free(work);
        return NULL;
    }

    /* -------- pass B: drop `<name>.write(...)' statements --------------------- */
    size_t outcap = strlen(msl) + 64;
    char *out = malloc(outcap);
    if (!out) { for (int i = 0; i < nn; i++) free((void *)names[i]); free(work); return NULL; }
    char *d = out;
    const char *s = work;
    while (*s)
    {
        bool hit = false;
        for (int i = 0; i < nn && !hit; i++)
        {
            size_t nb = strlen(names[i]);
            if (strncmp(s, names[i], nb) != 0) continue;
            if (strncmp(s + nb, ".write(", 7) != 0) continue;
            /* word boundary: not a member of another object / longer identifier */
            if (s > work && (isalnum((unsigned char)s[-1]) || s[-1] == '_' || s[-1] == '.')) continue;
            /* balanced-paren scan over the call */
            const char *open = s + nb + 6;   /* the '(' of ".write(" */
            int depth = 1;
            const char *q = open + 1;
            while (*q && depth > 0)
            {
                if (*q == '(') depth++;
                else if (*q == ')') depth--;
                q++;
            }
            if (depth != 0) continue;        /* unbalanced: leave untouched */
            memcpy(d, "(void)0", 7);
            d += 7;
            s = q;
            hit = true;
        }
        if (!hit) *d++ = *s++;
    }
    *d = 0;
    for (int i = 0; i < nn; i++) free((void *)names[i]);
    free(work);
    return out;
}
