"""
Agent 1 — Builder
Runs `make lib`, extracts compiler/linker errors, asks the model for one safe,
literal source fix, applies it with source_patcher, and retries.
"""

import re
import subprocess
import time

import config
import llm_client
import source_patcher

SYSTEM_PROMPT = """You are the Builder Agent for MGL, an OpenGL 4.6 implementation on Apple Metal
(C + Objective-C/ARC, glslang -> SPIR-V -> SPIRV-Cross -> MSL, built with `make lib` into libmgl.dylib).

Given compiler/linker output and source excerpts, you:
1. Identify the first real failure (ignore warnings unless -Werror).
2. Classify: COMPILE_ERROR | LINK_ERROR | MISSING_FILE | DEPENDENCY | UNKNOWN
3. Propose ONE minimal fix as an exact literal replacement in a single file.

Rules:
- Writable: MGL/src/*, MGL/include/mgl_ext*.h, tests/headless/*. Never edit external/, generated
  gl_core.c / glm_dispatch.c / glm_dispatch.h, GL/KHR/glm headers, or the Makefile.
- Fixes must be correct code, not stubs: never "fix" an error by emptying a function body,
  adding assert(0), or deleting a GL entry point.
- Undefined symbol at link time -> implement or declare it properly (extern in the right header).
- DEPENDENCY (missing external libs) -> escalate; the human runs external/build_external.sh.

RESPOND WITH RAW JSON ONLY:
{
  "status": "FAILED_FIXABLE" | "FAILED_ESCALATE",
  "failure_type": "...",
  "failure_file": "relative path or null",
  "failure_line": "exact error line or null",
  "diagnosis": "one sentence",
  "patch": {"file": "relative/path", "old": "exact existing text", "new": "replacement"} or null,
  "escalation_reason": "reason or null"
}"""

ERROR_RE = re.compile(r"^(?P<file>[\w./+-]+\.(?:c|m|h|cpp)):(?P<line>\d+):\d+: (?:fatal )?error: (?P<msg>.*)$", re.M)


def run_build(clean: bool = False) -> dict:
    if clean:
        subprocess.run(["make", "clean"], cwd=config.PROJECT_DIR, capture_output=True)
    cmd = ["make", f"-j{config.MAKE_JOBS}", "lib", f"build_dir={config.rel(config.BUILD_DIR)}"]
    print(f"[Builder] $ {' '.join(cmd)}")
    start = time.time()
    try:
        r = subprocess.run(cmd, cwd=config.PROJECT_DIR, capture_output=True, text=True,
                           timeout=config.BUILD_TIMEOUT)
        rc, out = r.returncode, r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = -1, (e.stdout or "") + "\n[build timed out]"
    return {"returncode": rc, "combined": config.scrub(out), "elapsed_seconds": round(time.time() - start),
            "success": rc == 0 and (config.BUILD_DIR / "libmgl.dylib").exists()}


def extract_error_lines(log: str) -> str:
    lines = log.splitlines()
    keep = []
    keys = ("error:", "Undefined symbols", "ld: ", "clang: error", "note:", "make: ***", "referenced from")
    for i, line in enumerate(lines):
        if any(k in line for k in keys):
            keep.extend(lines[max(0, i - 1):i + 3])
    seen, uniq = set(), []
    for line in keep:
        if line not in seen:
            seen.add(line)
            uniq.append(line)
    return "=== ERRORS ===\n" + "\n".join(uniq[:150]) + "\n\n=== TAIL ===\n" + "\n".join(lines[-40:])


def source_context(log: str) -> str:
    out = []
    for m in list(ERROR_RE.finditer(log))[:3]:
        f, ln = m.group("file"), int(m.group("line"))
        p = config.PROJECT_DIR / f
        if p.exists():
            src = p.read_text(errors="replace").splitlines()
            lo, hi = max(0, ln - 15), min(len(src), ln + 10)
            out.append(f"--- {f}:{ln}\n" + "\n".join(f"{n + 1:5d}  {src[n]}" for n in range(lo, hi)))
    return "\n\n".join(out)


def analyze(build: dict, attempt: int) -> dict:
    msg = (f"Build attempt {attempt}, exit code {build['returncode']}, {build['elapsed_seconds']}s\n\n"
           f"{extract_error_lines(build['combined'])}\n\n=== SOURCE ===\n{source_context(build['combined'])}\n\n"
           "Return the JSON now.")
    result = llm_client.ask_json(SYSTEM_PROMPT, msg, 2048)
    return result or {"status": "FAILED_ESCALATE", "diagnosis": "model returned no JSON",
                      "patch": None, "escalation_reason": "unparseable response"}


def run(max_attempts: int = 3, clean: bool = False) -> dict:
    print(f"\n{'=' * 60}\n[Builder] Building libmgl ({config.rel(config.BUILD_DIR)})\n{'=' * 60}")
    history = []
    for attempt in range(1, max_attempts + 1):
        build = run_build(clean=clean and attempt == 1)
        if build["success"]:
            print(f"[Builder] ✓ build succeeded (attempt {attempt}, {build['elapsed_seconds']}s)")
            warnings = len(re.findall(r"warning:", build["combined"]))
            return {"agent": "builder", "status": "SUCCESS", "attempts": attempt,
                    "warnings": warnings, "history": history}
        analysis = analyze(build, attempt)
        history.append({"attempt": attempt, "returncode": build["returncode"], "analysis": analysis})
        print(f"[Builder] Diagnosis: {analysis.get('diagnosis')}")
        patch = analysis.get("patch")
        if analysis.get("status") != "FAILED_FIXABLE" or not patch or attempt == max_attempts:
            return {"agent": "builder", "status": "ESCALATED", "attempts": attempt, "history": history,
                    "escalation_reason": analysis.get("escalation_reason") or "no applicable fix",
                    "last_log_tail": extract_error_lines(build["combined"])}
        res = source_patcher.patch(patch.get("file", ""), patch.get("old", ""), patch.get("new", ""),
                                   f"builder fix: {analysis.get('diagnosis', '')[:60]}")
        print(f"[Builder] patch {patch.get('file')}: {'applied' if res['ok'] else res['reason']}")
        if not res["ok"]:
            return {"agent": "builder", "status": "ESCALATED", "attempts": attempt, "history": history,
                    "escalation_reason": f"patch rejected: {res['reason']}",
                    "last_log_tail": extract_error_lines(build["combined"])}
    return {"agent": "builder", "status": "FAILED", "attempts": max_attempts, "history": history}
