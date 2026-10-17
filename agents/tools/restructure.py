#!/usr/bin/env python3
"""One-shot restructure: move mgl* functions into per-domain source files so
parallel agents own disjoint files. Idempotent-ish: functions already in their
target are left alone. Run from repo root:  python3 agents/tools/restructure.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stub_scanner as ss  # noqa: E402

SRC = Path("MGL/src")
CATCHALL = "mgl_funcs_to_be_implemented.c"

RULES = [  # applied to functions in the catch-all file, first match wins
    ("queries.c", r"Quer|ConditionalRender"),
    ("debug.c", r"Debug|ObjectLabel|ObjectPtrLabel|GraphicsResetStatus"),
    ("legacy_compat.c", r"^mgl(GetnMap|GetnPixelMap|ColorP|NormalP|SecondaryColorP|VertexP\d)"),
    ("transform_feedback.c", r"TransformFeedback"),
    ("program_introspection.c", r"ProgramUniform|Subroutine|GetProgramStageiv|GetUniform|GetnUniform|FragData|"
                                r"ProgramResource|ProgramInterface|ShaderStorageBlockBinding|AtomicCounter"),
    ("program_ext.c", r"ActiveShaderProgram|ProgramPipeline|CreateShaderProgramv|ProgramBinary|ProgramParameteri|"
                      r"ShaderBinary|SpecializeShader|ReleaseShaderCompiler|ValidateProgram|GetShaderPrecisionFormat"),
    ("vertex_attrib.c", r"VertexAttrib"),
    ("get.c", r"GetBooleani_v|GetDoublei_v|GetFloati_v"),
    ("framebuffer_ext.c", r"ClearBuffer|ReadnPixels|GetMultisamplefv"),
    ("state_ext.c", r"Viewport|Scissor|DepthRange|ClampColor|ClipControl|ColorMaski|ProvokingVertex|PatchParameter|"
                    r"PolygonOffsetClamp|PrimitiveRestartIndex|SampleMaski|MinSampleShading"),
    ("textures_ext.c", r"TexBuffer|Multisample"),
    ("textures.c", r"GetnTexImage|CopyImageSubData"),
    ("draw_ext.c", r"IndirectCount"),
    ("buffers.c", r"GetBufferParameteri64v"),
]

EXPLICIT = {  # (source file) -> {function: target}
    "state.c": {"mglGetPointerv": "debug.c"},
    "fence.c": {"mglTextureBarrier": "textures_ext.c"},
    "textures.c": {
        **{n: "textures_ext.c" for n in (
            "mglInvalidateTexImage", "mglInvalidateTexSubImage", "mglTexImage2DMultisample",
            "mglTexImage3DMultisample", "mglTextureStorage2DMultisample", "mglTextureStorage3DMultisample",
            "mglTextureView", "mglTextureBuffer", "mglTextureBufferRange")},
        "mglClientActiveTexture": "legacy_compat.c",
        **{n: "texture_query.c" for n in (
            "mglGetTextureParameterfv", "mglGetTextureParameterIiv", "mglGetTextureParameterIuiv",
            "mglGetTextureParameteriv", "mglGetTexParameterIiv", "mglGetTexParameterIuiv")},
        "mglSampleCoverage": "state_ext.c",
    },
    "get.c": {"mglGetInternalformati64v": "texture_query.c", "mglGetInternalformativ": "texture_query.c"},
    "program.c": {n: "program_introspection.c" for n in (
        "mglBindAttribLocation", "mglGetActiveAttrib", "mglGetActiveUniform", "mglGetAttachedShaders")},
}

HEADER = """/*
 * {name}
 * MGL — {desc}
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
{extra}
"""

DESCS = {
    "queries.c": "query objects, timers, conditional rendering",
    "debug.c": "KHR_debug output, object labels, robustness status",
    "legacy_compat.c": "compatibility-profile entry points (core profile: GL_INVALID_OPERATION)",
    "transform_feedback.c": "transform feedback objects and draws",
    "program_introspection.c": "program interface queries, ProgramUniform*, subroutines",
    "program_ext.c": "separable programs, pipelines, binaries, SPIR-V shaders, validation",
    "vertex_attrib.c": "generic vertex attribute values and queries",
    "framebuffer_ext.c": "ClearBuffer*, robust reads, multisample queries",
    "state_ext.c": "indexed viewport/scissor, clip control, sample and raster state",
    "textures_ext.c": "texture buffers, views, multisample storage, invalidation, barriers",
    "draw_ext.c": "GL 4.6 indirect-count draws",
    "texture_query.c": "texture parameter / level / internalformat queries",
}

EXTRA = {  # forward declarations some moved bodies rely on
    "transform_feedback.c": "\nTransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name);\n"
                            "TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name);\n"
                            "TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name);\n",
    "textures.c": "",
}
TEX_DECLS = "\nextern Texture *findTexture(GLMContext ctx, GLuint texture);\nextern Texture *currentTexture(GLMContext ctx, GLuint index);\n"


def extract(text, name):
    for m in ss.FUNC_RE.finditer(text):
        if m.group("name") == name:
            _, end = ss._body(text, m.end() - 1)
            start = m.start()
            # include an immediately preceding comment block
            pre = text.rfind("\n\n", 0, start)
            chunk_start = pre + 2 if pre >= 0 and text[pre + 2:start].lstrip().startswith(("//", "/*")) else start
            return chunk_start, end + 1
    return None


def main():
    moves = {}  # src -> [(name, target)]
    cat_text = (SRC / CATCHALL).read_text()
    for m in ss.FUNC_RE.finditer(cat_text):
        name = m.group("name")
        for target, pat in RULES:
            if re.search(pat, name):
                moves.setdefault(CATCHALL, []).append((name, target))
                break
        else:
            print(f"UNMATCHED {name}")
    for src, mp in EXPLICIT.items():
        for name, target in mp.items():
            moves.setdefault(src, []).append((name, target))

    appended = {}
    for src, items in moves.items():
        path = SRC / src
        text = path.read_text()
        for name, target in items:
            rng = extract(text, name)
            if not rng:
                print(f"  not found: {name} in {src}")
                continue
            chunk = text[rng[0]:rng[1]]
            text = text[:rng[0]] + text[rng[1]:]
            appended.setdefault(target, []).append(chunk.strip() + "\n")
        path.write_text(re.sub(r"\n{3,}", "\n\n", text))

    for target, chunks in appended.items():
        tpath = SRC / target
        if tpath.exists():
            body = tpath.read_text().rstrip() + "\n\n"
        else:
            extra = EXTRA.get(target, "") + (TEX_DECLS if target.startswith("text") else "")
            body = HEADER.format(name=target, desc=DESCS.get(target, ""), extra=extra)
        tpath.write_text(body + "\n".join(chunks))
        print(f"{target}: +{len(chunks)}")

    left = [m.group("name") for m in ss.FUNC_RE.finditer((SRC / CATCHALL).read_text())]
    if not left:
        (SRC / CATCHALL).unlink()
        print(f"removed {CATCHALL}")


if __name__ == "__main__":
    main()
