#!/usr/bin/env python3
"""Pre-place per-domain extension slots so parallel agents never need to edit
shared headers for new state / Metal hooks / glGet pnames / init code."""
import re
from pathlib import Path

DOMAINS = {  # domain -> primary .c file (owns init/get)
    "fb": "framebuffer_ext.c",
    "queries": "queries.c",
    "debug": "debug.c",
    "introspect": "program_introspection.c",
    "program": "program_ext.c",
    "xfb": "transform_feedback.c",
    "vattrib": "vertex_attrib.c",
    "rstate": "state_ext.c",
    "texparam": "texture_query.c",
    "texdata": "textures.c",
    "texext": "textures_ext.c",
    "buffers": "draw_ext.c",
}
SRC = Path("MGL/src")
INC = Path("MGL/include")

# ---- mgl_ext.h -------------------------------------------------------------
fwd = "\n".join(f"struct MGLExtState_{d};\nstruct MGLExtMetal_{d};" for d in DOMAINS)
protos = "\n".join(
    f"void mgl_ext_init_{d}(GLMContext ctx);\n"
    f"bool mgl_ext_get_{d}(GLMContext ctx, GLenum pname, GLuint type, void *data);" for d in DOMAINS)
(INC / "mgl_ext.h").write_text(f"""/*
 * mgl_ext.h — per-domain extension slots (see agents/README.md, "Ownership").
 *
 * Each domain <d> owns:
 *   MGL/src/mgl_ext_<d>.h            struct MGLExtState_<d> / struct MGLExtMetal_<d>
 *   ctx->state.ext_<d>               its private GL state (allocated in mgl_ext_init_<d>)
 *   ctx->mtl_funcs.ext_<d>           its Metal hook table (filled by the
 *                                    -bindExt<D>FuncsToGLMContext: category method)
 *   mgl_ext_get_<d>()                extra glGet* pnames; return true if handled
 */
#ifndef MGL_EXT_H
#define MGL_EXT_H

#include <stdbool.h>

#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

{fwd}

/* glGet value types (match get.c) */
enum {{ MGL_GET_BOOL = 0, MGL_GET_INT = 1, MGL_GET_FLOAT = 2, MGL_GET_DOUBLE = 3, MGL_GET_INT64 = 4 }};

/* store element [index] of a glGet result in the caller's requested type */
static inline void mgl_get_store(GLuint type, void *data, int index, double v)
{{
    switch (type) {{
        case MGL_GET_BOOL:   ((GLboolean *)data)[index] = v != 0.0 ? GL_TRUE : GL_FALSE; break;
        case MGL_GET_INT:    ((GLint *)data)[index] = (GLint)v; break;
        case MGL_GET_FLOAT:  ((GLfloat *)data)[index] = (GLfloat)v; break;
        case MGL_GET_DOUBLE: ((GLdouble *)data)[index] = v; break;
        case MGL_GET_INT64:  ((GLint64 *)data)[index] = (GLint64)v; break;
    }}
}}

#ifdef __cplusplus
extern "C" {{
#endif
{protos}

void mgl_ext_init_all(GLMContext ctx);
bool mgl_ext_get(GLMContext ctx, GLenum pname, GLuint type, void *data);
#ifdef __cplusplus
}}
#endif

#endif
""")

# ---- mgl_ext.c -------------------------------------------------------------
(SRC / "mgl_ext.c").write_text(f"""/* mgl_ext.c — fan-out of per-domain init and glGet hooks. Do not add logic here. */
#include "glm_context.h"
#include "mgl_ext.h"

void mgl_ext_init_all(GLMContext ctx)
{{
{chr(10).join(f"    mgl_ext_init_{d}(ctx);" for d in DOMAINS)}
}}

bool mgl_ext_get(GLMContext ctx, GLenum pname, GLuint type, void *data)
{{
    return {(chr(10) + "        || ").join(f"mgl_ext_get_{d}(ctx, pname, type, data)" for d in DOMAINS)};
}}
""")

# ---- per-domain headers + stubs in primary files ---------------------------
for d, primary in DOMAINS.items():
    h = SRC / f"mgl_ext_{d}.h"
    if not h.exists():
        h.write_text(f"""/* mgl_ext_{d}.h — private state and Metal hooks of the '{d}' domain. Owned by agent '{d}'. */
#ifndef MGL_EXT_{d.upper()}_H
#define MGL_EXT_{d.upper()}_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_{d}; allocate in mgl_ext_init_{d}(). */
struct MGLExtState_{d} {{
    int reserved;
}};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_{d}; filled by the renderer category. */
struct MGLExtMetal_{d} {{
    void *reserved;
}};

#endif
""")
    p = SRC / primary
    t = p.read_text()
    if f"mgl_ext_init_{d}" not in t:
        t = t.replace('#include "glm_context.h"\n', f'#include "glm_context.h"\n#include "mgl_ext_{d}.h"\n', 1)
        if f'mgl_ext_{d}.h' not in t:
            t = f'#include "mgl_ext_{d}.h"\n' + t
        t = t.rstrip() + f"""

#pragma mark - domain '{d}' hooks

void mgl_ext_init_{d}(GLMContext ctx)
{{
    (void)ctx;
}}

bool mgl_ext_get_{d}(GLMContext ctx, GLenum pname, GLuint type, void *data)
{{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}}
"""
        p.write_text(t)

# ---- glm_context.h slots ---------------------------------------------------
gh = INC / "glm_context.h"
t = gh.read_text()
if "ext_fb" not in t:
    t = t.replace('#include "hash_table.h"', '#include "hash_table.h"\n#include "mgl_ext.h"', 1) \
        if '#include "hash_table.h"' in t else t
    state_slots = "    // per-domain extension state (mgl_ext.h) — do not reorder\n" + \
        "".join(f"    struct MGLExtState_{d} *ext_{d};\n" for d in DOMAINS) + "\n"
    t = t.replace("    // put at end, big chunk of yuck\n", state_slots + "    // put at end, big chunk of yuck\n", 1)
    metal_slots = "\n    // per-domain Metal hook tables (mgl_ext.h) — do not reorder\n" + \
        "".join(f"    struct MGLExtMetal_{d} *ext_{d};\n" for d in DOMAINS)
    anchor = "    void (*mtlDispatchComputeIndirect)(GLMContext ctx, GLintptr indirect);\n"
    assert anchor in t
    t = t.replace(anchor, anchor + metal_slots, 1)
    if '#include "mgl_ext.h"' not in t:
        t = t.replace("typedef struct GLMContextRec_t *GLMContext;\n\nstruct GLMMetalFuncs",
                      "typedef struct GLMContextRec_t *GLMContext;\n#include \"mgl_ext.h\"\n\nstruct GLMMetalFuncs", 1)
    gh.write_text(t)

# ---- glm_context.c init ----------------------------------------------------
gc = SRC / "glm_context.c"
t = gc.read_text()
if "mgl_ext_init_all" not in t:
    t = t.replace("    init_dispatch(ctx);\n", "    init_dispatch(ctx);\n\n    mgl_ext_init_all(ctx);\n", 1)
    t = t.replace('#include "error.h"\n', '#include "error.h"\n#include "mgl_ext.h"\n', 1)
    gc.write_text(t)

# ---- get.c fallback --------------------------------------------------------
g = SRC / "get.c"
t = g.read_text()
if "mgl_ext_get(" not in t:
    anchor = "        case 0x82DA: RET_TYPE_VAR(type, max_vertex_attrib_bindings); break; // GL_MAX_VERTEX_ATTRIB_BINDINGS\n    }\n}"
    assert anchor in t
    t = t.replace(anchor, anchor.replace("    }\n}", """
        default:
            // domain extensions (mgl_ext.h); unknown pnames are an error
            if (!mgl_ext_get(ctx, pname, type, data))
                ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}"""), 1)
    t = t.replace('#include "glm_context.h"\n', '#include "glm_context.h"\n#include "mgl_ext.h"\n', 1)
    g.write_text(t)

# ---- MGLRenderer.m hook points ----------------------------------------------
m = SRC / "MGLRenderer.m"
t = m.read_text()
if "extHookRenderPassDescriptor" not in t:
    helper = '''
#pragma mark domain extension hook fan-out (see MGLRenderer_Private.h)
// Calls every category method named ext<Domain><suffix> (single object argument).
static void mglCallExtHooks(id self, const char *suffix, id arg)
{
    unsigned int n = 0;
    Method *methods = class_copyMethodList([self class], &n);
    size_t sl = strlen(suffix);
    for (unsigned int i = 0; i < n; i++) {
        const char *sel = sel_getName(method_getName(methods[i]));
        size_t l = strlen(sel);
        if (strncmp(sel, "ext", 3) == 0 && l > sl && strcmp(sel + l - sl, suffix) == 0) {
            void (*imp)(id, SEL, id) = (void (*)(id, SEL, id))method_getImplementation(methods[i]);
            imp(self, method_getName(methods[i]), arg);
        }
    }
    free(methods);
}

'''
    t = t.replace("// Main class performing the rendering\n@implementation MGLRenderer\n",
                  helper + "// Main class performing the rendering\n@implementation MGLRenderer\n", 1)
    a1 = "        _currentRenderEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor: _renderPassDescriptor];\n"
    assert a1 in t
    t = t.replace(a1, "        mglCallExtHooks(self, \"RenderPassDescriptor:\", _renderPassDescriptor); // extHookRenderPassDescriptor\n" + a1, 1)
    a2 = "    _currentRenderEncoder.label = @\"GL Render Encoder\";\n"
    assert a2 in t
    t = t.replace(a2, a2 + "    mglCallExtHooks(self, \"RenderEncoderCreated:\", _currentRenderEncoder);\n", 1)
    a3 = "    // Create a render command encoder.\n    [_currentRenderEncoder setRenderPipelineState: _pipelineState];\n\n    return true;\n}"
    assert a3 in t
    t = t.replace(a3, a3.replace("\n    return true;\n}",
                  "\n    if (draw_command)\n        mglCallExtHooks(self, \"PreDraw:\", _currentRenderEncoder);\n\n    return true;\n}"), 1)
    a4 = "        [commandBuffer commit];\n"
    i = t.index("- (void)commitCommandBufferWithAGXRecovery:")
    j = t.index(a4, i)
    t = t[:j] + "        mglCallExtHooks(self, \"WillCommit:\", commandBuffer);\n" + t[j:]
    m.write_text(t)

ph = SRC / "MGLRenderer_Private.h"
t = ph.read_text()
if "ext hook methods" not in t:
    t = t.replace(" */\n#ifndef MGLRenderer_Private_h", """ *
 * ext hook methods — a category may also implement any of (called for every domain):
 *     - (void) ext<Domain>RenderPassDescriptor: (MTLRenderPassDescriptor *) desc;  // before encoder creation
 *     - (void) ext<Domain>RenderEncoderCreated: (id<MTLRenderCommandEncoder>) enc;
 *     - (void) ext<Domain>PreDraw: (id<MTLRenderCommandEncoder>) enc;            // end of processGLState(true)
 *     - (void) ext<Domain>WillCommit: (id<MTLCommandBuffer>) cmdbuf;              // in commitCommandBufferWithAGXRecovery
 */
#ifndef MGLRenderer_Private_h""", 1)
    ph.write_text(t)
print("scaffold done")
