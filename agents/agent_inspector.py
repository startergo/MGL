"""
Agent 2 — Inspector
Replaces the wined3d PE export/import diff with MGL's equivalents:
  1. stub_scanner.py   — every GL 4.6 core entry point classified (EMPTY/NOOP/ASSERT/IMPLEMENTED)
  2. export check      — every core gl* symbol exported from libmgl.dylib (nm)
  3. headless tests    — tests/headless/run.sh, PASS/FAIL per test case
Then asks the model to rank findings and root-cause test failures.
"""

import re
import subprocess

import config
import llm_client
import stub_scanner

SYSTEM_PROMPT = """You are the Inspector Agent for MGL (OpenGL 4.6 on Apple Metal).
You receive: a stub inventory (functions still empty/no-op/asserting), missing exported GL symbols,
and headless test results (PASS/FAIL lines with diagnostics).

Classify each finding:
- CRITICAL: crash/assert in a core entry point, missing export, test crash, wrong rendering result
- WARNING: stubbed core function, failing error-code test, wrong query value
- INFO: legacy compatibility-profile functions (must only raise GL_INVALID_OPERATION), documented
  Metal limitations (no fp64, no geometry/tessellation stages, no logic op, line width 1.0)

For failures give the likely root cause (file + function) and a concrete fix direction.

Respond ONLY with JSON:
{
  "summary": "2-3 sentences",
  "stub_counts": {"EMPTY": n, "NOOP": n, "ASSERT": n},
  "findings": [
    {"severity": "CRITICAL|WARNING|INFO", "area": "domain or file", "description": "...",
     "likely_cause": "...", "suggested_fix": "..."}
  ],
  "priority_fixes": ["ordered list"]
}"""


def stub_inventory() -> dict:
    funcs = stub_scanner.scan_sources()
    core = stub_scanner.core_commands()
    stubs = [f for f in funcs.values() if not f["legacy"] and f["class"] != "IMPLEMENTED"]
    legacy_assert = [f["name"] for f in funcs.values() if f["legacy"] and f["class"] == "ASSERT"]
    return {
        "core_commands": len(core),
        "missing_implementation": [c for c in core if stub_scanner.gl_to_mgl(c) not in funcs],
        "stubs": [{"name": f["name"], "file": f["file"], "line": f["line"], "class": f["class"]}
                  for f in sorted(stubs, key=lambda f: (f["file"], f["line"]))],
        "legacy_asserting": legacy_assert,
    }


def missing_exports() -> list:
    lib = config.BUILD_DIR / "libmgl.dylib"
    if not lib.exists():
        return ["<libmgl.dylib not built>"]
    r = subprocess.run(["nm", "-gU", str(lib)], capture_output=True, text=True)
    exported = {s.lstrip("_") for s in re.findall(r"\s_?(gl\w+)$", r.stdout, re.M)}
    return sorted(c for c in stub_scanner.core_commands() if c not in exported)


def run_tests(selection=None) -> dict:
    cmd = ["tests/headless/run.sh"] + list(selection or [])
    try:
        r = subprocess.run(cmd, cwd=config.PROJECT_DIR, capture_output=True, text=True,
                           timeout=config.TEST_TIMEOUT)
        out, rc = r.stdout + "\n" + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") + "\n[tests timed out]", -1
    out = config.scrub(out)
    passed = re.findall(r"^PASS (\S+)", out, re.M)
    failed = re.findall(r"^FAIL (\S+)", out, re.M)
    crashed = re.findall(r"^(?:FAILED|BUILD-FAIL) (\S+.*)$", out, re.M)
    diags = [line for line in out.splitlines() if line.startswith("  [") or "Assertion" in line][:200]
    return {"returncode": rc, "passed": len(passed), "failed": failed, "crashed": crashed,
            "diagnostics": diags}


def run(tests=None, analyze: bool = True) -> dict:
    print(f"\n{'=' * 60}\n[Inspector] Inventory, exports, headless tests\n{'=' * 60}")
    inv = stub_inventory()
    exp = missing_exports()
    res = run_tests(tests)
    counts = {}
    for s in inv["stubs"]:
        counts[s["class"]] = counts.get(s["class"], 0) + 1
    print(f"[Inspector] stubs: {counts or 'none'}; legacy asserting: {len(inv['legacy_asserting'])}; "
          f"missing exports: {len(exp)}; tests: {res['passed']} pass / {len(res['failed'])} fail / "
          f"{len(res['crashed'])} crashed")
    report = {"agent": "inspector", "inventory": inv, "missing_exports": exp, "tests": res,
              "stub_counts": counts}
    clean = not inv["stubs"] and not exp and not res["failed"] and not res["crashed"] \
        and not inv["legacy_asserting"]
    if clean:
        report["analysis"] = {"summary": "No stubs, all exports present, all tests pass.",
                              "findings": [], "priority_fixes": [], "stub_counts": counts}
    elif analyze:
        payload = {"stub_counts": counts, "stubs": inv["stubs"][:150],
                   "legacy_asserting_count": len(inv["legacy_asserting"]),
                   "missing_exports": exp[:100], "tests": res}
        import json
        report["analysis"] = llm_client.ask_json(SYSTEM_PROMPT, json.dumps(payload, indent=1), 4096) \
            or {"error": "unparseable inspector response"}
    report["clean"] = clean
    return report
