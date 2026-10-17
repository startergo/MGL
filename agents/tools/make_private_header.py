#!/usr/bin/env python3
"""Split MGLRenderer.m's private types + ivars into MGLRenderer_Private.h so that
category files (MGLRenderer+<Domain>.m) can access renderer state. Also makes
-bindObjFuncsToGLMContext: call every -bindExt*FuncsToGLMContext: category method."""
import re
from pathlib import Path

M = Path("MGL/src/MGLRenderer.m")
H = Path("MGL/src/MGLRenderer_Private.h")
text = M.read_text()

start = text.index("typedef struct SyncList_t")
impl = text.index("// Main class performing the rendering\n@implementation MGLRenderer\n{")
types_block = text[start:impl]
ivar_open = text.index("{", impl)
ivar_close = text.index("\n}\n", ivar_open)
ivars = text[ivar_open + 1:ivar_close]

sigs = []
for m in re.finditer(r"^(-|\+)\s*\([^)]*\)[^{;]*", text, re.M):
    s = re.sub(r"//[^\n]*", "", m.group(0))
    s = re.sub(r"\s+", " ", s).strip().rstrip(";").rstrip()
    if s not in sigs and "initMGLRendererFromContext" not in s and "createMGLRendererFromContext" not in s:
        sigs.append(s)

header = f"""/*
 * MGLRenderer_Private.h — internal interface of MGLRenderer.
 *
 * Category files (MGLRenderer+<Domain>.m) import this to reach renderer state.
 * A category may implement
 *     - (void) bindExt<Domain>FuncsToGLMContext: (GLMContext) glm_ctx;
 * which -bindObjFuncsToGLMContext: calls automatically (objc runtime scan), to
 * install its mtl_funcs hooks without editing MGLRenderer.m.
 */
#ifndef MGLRenderer_Private_h
#define MGLRenderer_Private_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <AppKit/AppKit.h>

#import "MGLRenderer.h"
#import "glm_context.h"
#import "spirv_cross_c.h"

{types_block.rstrip()}

@interface MGLRenderer ()
{{{ivars}
}}

{chr(10).join(s + ';' for s in sigs)}

@end

MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);

#endif /* MGLRenderer_Private_h */
"""
H.write_text(header)

new_text = (text[:start] + '#import "MGLRenderer_Private.h"\n\n'
            + "// Main class performing the rendering\n@implementation MGLRenderer\n"
            + text[ivar_close + 3:])

binder_anchor = "- (void) bindObjFuncsToGLMContext: (GLMContext) glm_ctx\n{"
assert binder_anchor in new_text, "binder anchor missing"
ext_call = """
    // Category hooks: call every -bindExt*FuncsToGLMContext: (see MGLRenderer_Private.h)
    {
        unsigned int n = 0;
        Method *methods = class_copyMethodList([self class], &n);
        for (unsigned int i = 0; i < n; i++) {
            const char *sel = sel_getName(method_getName(methods[i]));
            if (strncmp(sel, "bindExt", 7) == 0 && strstr(sel, "FuncsToGLMContext:")) {
                void (*imp)(id, SEL, GLMContext) = (void (*)(id, SEL, GLMContext))method_getImplementation(methods[i]);
                imp(self, method_getName(methods[i]), glm_ctx);
            }
        }
        free(methods);
    }
"""
# insert the scan right before the closing brace of bindObjFuncsToGLMContext
bi = new_text.index(binder_anchor)
close = new_text.index("\n}\n", bi)
new_text = new_text[:close] + "\n" + ext_call + new_text[close:]
M.write_text(new_text)
print(f"wrote {H} ({len(sigs)} method decls)")
