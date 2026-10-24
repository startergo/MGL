/* tex_param_internal.h — shared core of texture and sampler parameters (texparam domain). */
#ifndef TEX_PARAM_INTERNAL_H
#define TEX_PARAM_INTERNAL_H

#include <stdbool.h>
#include "glm_context.h"

/* source / destination value types */
enum {
    TP_FLOAT = 0,      /* *f / *fv */
    TP_INT,            /* *i / *iv  (border color: normalized conversion) */
    TP_PURE_INT,       /* *Iiv */
    TP_UINT            /* *Iuiv */
};

bool   mglTexParamIsSamplerState(GLenum pname);
GLenum mglTexParamSet(TextureParameter *p, GLenum target, bool is_sampler, GLenum pname,
                      const void *v, int vt, bool vector, bool *changed, bool dsa);
GLenum mglTexParamGet(GLMContext ctx, Texture *tex, TextureParameter *p, bool is_sampler,
                      GLenum pname, void *out, int vt);

#endif
