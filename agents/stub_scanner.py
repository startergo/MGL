#!/usr/bin/env python3
"""
stub_scanner.py — inventory of unimplemented / stubbed MGL entry points.

Parses every `mgl*(GLMContext ctx, ...)` definition in MGL/src/*.c|*.m and
classifies its body, then cross-checks the GL 4.6 core command list from
MGL/include/GL/glcorearb.h against the dispatch table (MGL/src/glm_dispatch.c).

Usage:
    python3 stub_scanner.py                 # human summary
    python3 stub_scanner.py --json          # machine-readable (used by agents)
    python3 stub_scanner.py --list STUB     # list functions of one class
    python3 stub_scanner.py --file mgl_funcs_to_be_implemented.c

Classes:
    ASSERT      body contains assert(0) / abort()  — crashes the client
    EMPTY       body is empty or only (void) casts / comments / return
    NOOP        body mentions "no-op" / "stub" / "not implemented" / "TODO"
    IMPLEMENTED anything else
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("MGL_PROJECT_DIR") or Path(__file__).resolve().parent.parent)
SRC_DIR = PROJECT_DIR / "MGL" / "src"
GLCOREARB = PROJECT_DIR / "MGL" / "include" / "GL" / "glcorearb.h"
DISPATCH = SRC_DIR / "glm_dispatch.c"

# Source files that are legacy (compatibility-profile) entry points.
LEGACY_FILES = {"non_core_unimplemented.c", "legacy_compat.c"}

FUNC_RE = re.compile(
    r"^(?P<ret>[A-Za-z_][\w \*]*?)\s*\b(?P<name>mgl[A-Z]\w*)\s*\(\s*GLMContext\s+ctx(?P<args>[^)]*)\)\s*\{",
    re.M,
)

NOOP_WORDS = re.compile(r"no-?op|\bstub\b|not implemented|unimplemented|TODO|FIXME|placeholder|basic stub",
                        re.I)


def _body(text: str, open_brace: int) -> tuple[str, int]:
    depth, i = 0, open_brace
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i], i
        elif c == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
        elif text.startswith("/*", i):
            i = text.find("*/", i) + 1
        i += 1
    return text[open_brace + 1:], len(text)


def _strip(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    return code


def classify(body: str) -> str:
    code = _strip(body)
    if re.search(r"\bassert\s*\(\s*(0|false|!\"[^\"]*\")\s*\)|\babort\s*\(\s*\)", code):
        return "ASSERT"
    trimmed = re.sub(r"\(\s*void\s*\)\s*\w+\s*;", "", code)
    trimmed = re.sub(r"\breturn\s*(0|NULL|GL_FALSE|GL_TRUE|0\.0f?|\w+)?\s*;", "", trimmed)
    if not trimmed.strip():
        return "EMPTY"
    # "Body sets an error and returns" with nothing else is a stub too
    only_err = re.sub(r"(STATE\(error\)\s*=\s*GL_\w+\s*;|ERROR_RETURN\w*\s*\([^)]*\)\s*;|"
                      r"ERROR_CHECK_RETURN\w*\s*\([^;]*\)\s*;|if\s*\([^)]*\)\s*\{?\s*\}?)", "", trimmed)
    if not only_err.strip():
        return "EMPTY"
    if NOOP_WORDS.search(body) and len(code.strip().splitlines()) < 12:
        return "NOOP"
    return "IMPLEMENTED"


def scan_sources():
    funcs = {}
    for path in sorted(list(SRC_DIR.glob("*.c")) + list(SRC_DIR.glob("*.m"))):
        text = path.read_text(errors="replace")
        for m in FUNC_RE.finditer(text):
            body, end = _body(text, m.end() - 1)
            name = m.group("name")
            line = text.count("\n", 0, m.start()) + 1
            cls = classify(body)
            funcs[name] = {
                "name": name, "file": path.name, "line": line,
                "class": cls, "legacy": path.name in LEGACY_FILES,
                "lines": body.count("\n"),
            }
    return funcs


def core_commands():
    """GL commands declared in glcorearb.h, grouped by GL_VERSION_x_y."""
    text = GLCOREARB.read_text(errors="replace")
    cmds, version = {}, None
    for line in text.splitlines():
        mv = re.match(r"#ifndef (GL_VERSION_\d_\d)", line)
        if mv:
            version = mv.group(1)
            continue
        if re.match(r"#ifndef GL_(ARB|KHR|EXT|AMD|NV|INTEL|MESA|OVR|APPLE)_", line):
            version = None
            continue
        mc = re.match(r"GLAPI .*?APIENTRY (gl\w+)\s*\(", line)
        if mc and version:
            cmds[mc.group(1)] = version
    return cmds


def gl_to_mgl(name: str) -> str:
    return "mgl" + name[2:]


def dispatched():
    text = DISPATCH.read_text(errors="replace")
    return set(re.findall(r"=\s*(mgl\w+)\s*;", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", metavar="CLASS")
    ap.add_argument("--file")
    ap.add_argument("--include-legacy", action="store_true")
    args = ap.parse_args()

    funcs = scan_sources()
    core = core_commands()
    disp = dispatched()

    missing_impl = sorted(c for c in core if gl_to_mgl(c) not in funcs)
    missing_dispatch = sorted(c for c in core if gl_to_mgl(c) in funcs and gl_to_mgl(c) not in disp)

    selected = [f for f in funcs.values()
                if (args.include_legacy or not f["legacy"])
                and (not args.file or f["file"] == args.file)]
    for f in selected:
        f["gl_version"] = core.get("gl" + f["name"][3:])

    if args.json:
        json.dump({
            "functions": sorted(selected, key=lambda f: (f["file"], f["line"])),
            "core_missing_implementation": missing_impl,
            "core_missing_dispatch": missing_dispatch,
        }, sys.stdout, indent=1)
        return

    if args.list:
        for f in sorted(selected, key=lambda f: (f["file"], f["line"])):
            if f["class"] == args.list.upper():
                print(f"{f['file']}:{f['line']}\t{f['name']}\t{f.get('gl_version') or '-'}")
        return

    counts = {}
    for f in selected:
        counts.setdefault(f["class"], 0)
        counts[f["class"]] += 1
    legacy = [f for f in funcs.values() if f["legacy"]]
    print(f"MGL stub inventory — {SRC_DIR}")
    print(f"  core entry points scanned : {len(selected)}")
    for k in ("IMPLEMENTED", "NOOP", "EMPTY", "ASSERT"):
        print(f"  {k:<26}: {counts.get(k, 0)}")
    print(f"  legacy (compat) functions : {len(legacy)}  "
          f"(assert: {sum(1 for f in legacy if f['class'] == 'ASSERT')})")
    print(f"  GL 4.6 core commands      : {len(core)}")
    print(f"  ...without mgl* impl      : {len(missing_impl)}")
    print(f"  ...not in dispatch table  : {len(missing_dispatch)}")
    for c in missing_impl[:40]:
        print(f"      missing: {c} ({core[c]})")


if __name__ == "__main__":
    main()
