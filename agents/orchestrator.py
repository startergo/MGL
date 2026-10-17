#!/usr/bin/env python3
"""
Lead Engineer — MGL multi-agent orchestrator.
Builder (make lib + auto-fix) → Inspector (stub inventory, exports, headless tests)
→ Lead synthesis (prioritized literal source patches) → optionally apply and loop.

Usage:
    python3 orchestrator.py                    # one interactive cycle
    python3 orchestrator.py --inspect-only     # skip build, use existing build/libmgl.dylib
    python3 orchestrator.py --build-only
    python3 orchestrator.py --tests test_queries test_fb
    python3 orchestrator.py --auto --max-cycles 5
    python3 orchestrator.py --no-llm           # build + inventory + tests only, no API calls
"""

import argparse
import datetime
import json
import sys
import textwrap

import config
import llm_client
import source_patcher

LEAD_SYSTEM = """You are the Lead Engineer for MGL, an OpenGL 4.6 core-profile implementation on Apple Metal.
Goal: every GL 4.6 core entry point does real, spec-correct work on Metal; no stubs.

Architecture: gl_core.c -> ctx->dispatch -> mgl* functions in MGL/src/*.c; Metal in MGL/src/MGLRenderer.m
and category files MGLRenderer+<Domain>.m (import MGLRenderer_Private.h). Per-domain state/hook slots
are declared in MGL/include/mgl_ext.h and MGL/src/mgl_ext_<domain>.h. Shaders: glslang -> SPIR-V ->
SPIRV-Cross -> MSL. Tests: tests/headless/test_*.c with mgl_test.h.

You get the Builder and Inspector reports. Produce prioritized fixes. Each auto-applicable fix is an
exact literal replacement in ONE file: {"file": "MGL/src/x.c", "old": "exact text", "new": "text"}
(old = "" creates a new file). Writable: MGL/src/, MGL/include/mgl_ext*, tests/headless/.
Never edit external/, generated dispatch files, GL headers, or the Makefile. Never replace code with
stubs. Larger work (new Metal emulation paths) -> change_type MANUAL with a precise description.

Respond with JSON only:
{
  "cycle_summary": "2-3 sentences",
  "overall_status": "GOOD|NEEDS_WORK|BROKEN",
  "fixes": [{"priority": 1, "severity": "CRITICAL|WARNING|INFO", "title": "...",
             "change_type": "SOURCE_PATCH|MANUAL", "description": "...",
             "patch": {"file": "...", "old": "...", "new": "..."} or null}],
  "next_action": "REBUILD|INSPECT_ONLY|DONE|MANUAL_REVIEW"
}"""


def save_report(cycle: int, data: dict, label: str):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = config.REPORTS_DIR / f"{stamp}_cycle{cycle:02d}_{label}.json"
    path.write_text(config.scrub(json.dumps(data, indent=2)))
    print(f"[Lead] report: {config.rel(path)}")


def print_lead(lead: dict):
    print(f"\n{'=' * 60}\n  LEAD ENGINEER — {lead.get('overall_status')}\n{'=' * 60}")
    print(textwrap.fill(lead.get("cycle_summary", ""), 76, initial_indent="  ", subsequent_indent="  "))
    for fix in lead.get("fixes", []):
        print(f"\n  [{fix.get('priority')}] {fix.get('severity')} {fix.get('title')} ({fix.get('change_type')})")
        print(textwrap.fill(fix.get("description", ""), 72, initial_indent="      ", subsequent_indent="      "))
        if fix.get("patch"):
            print(f"      file: {fix['patch'].get('file')}")
    print(f"\n  Next action: {lead.get('next_action')}")


def synthesize(builder: dict, inspector: dict | None) -> dict:
    if builder.get("status") == "SUCCESS" and inspector and inspector.get("clean"):
        return {"cycle_summary": "Build clean, no stubs, all exports present, all headless tests pass.",
                "overall_status": "GOOD", "fixes": [], "next_action": "DONE"}
    payload = {"builder": {k: builder.get(k) for k in ("status", "attempts", "warnings", "escalation_reason",
                                                        "last_log_tail", "history")},
               "inspector": inspector.get("analysis") if inspector else None,
               "test_diagnostics": (inspector or {}).get("tests", {}).get("diagnostics", [])[:80]}
    # Give the model exact source around the first stubbed functions so patches can quote text.
    excerpts = []
    for stub in ((inspector or {}).get("inventory", {}).get("stubs") or [])[:6]:
        ctx = source_patcher.get_context(f"MGL/src/{stub['file']}", stub["name"] + "(", 20)
        if ctx:
            excerpts.append(f"--- MGL/src/{stub['file']} ({stub['name']})\n{ctx}")
    msg = (json.dumps(payload, indent=1) + "\n\n=== SOURCE EXCERPTS ===\n" + "\n\n".join(excerpts)
           + "\n\nReturn the JSON now.")
    return llm_client.ask_json(LEAD_SYSTEM, msg, 6000) or {
        "cycle_summary": "Lead response could not be parsed.", "overall_status": "NEEDS_WORK",
        "fixes": [], "next_action": "MANUAL_REVIEW"}


def apply_fixes(lead: dict) -> list:
    results = []
    for fix in lead.get("fixes", []):
        title = fix.get("title", "?")
        p = fix.get("patch")
        if fix.get("change_type") != "SOURCE_PATCH" or not p:
            results.append({"title": title, "skipped": True, "reason": fix.get("change_type")})
            continue
        r = source_patcher.patch(p.get("file", ""), p.get("old", ""), p.get("new", ""), title)
        print(f"[Lead] {'✓' if r['ok'] else '✗'} {title}{'' if r['ok'] else ' — ' + r['reason']}")
        results.append({"title": title, "applied": r["ok"], "reason": r.get("reason")})
    return results


def main():
    ap = argparse.ArgumentParser(description="MGL multi-agent build/inspect orchestrator")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--clean", action="store_true", help="make clean before the first build")
    ap.add_argument("--tests", nargs="*", help="test file stems (default: all)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=5)
    ap.add_argument("--auto", action="store_true", help="apply lead fixes and keep cycling")
    ap.add_argument("--no-llm", action="store_true", help="no model calls: build, inventory, tests only")
    args = ap.parse_args()

    import agent_builder
    import agent_inspector

    print(f"\n{'#' * 60}\n  MGL Multi-Agent Orchestrator   project: {config.PROJECT_DIR.name}\n"
          f"  model: {'(none)' if args.no_llm else llm_client.get_model()}\n{'#' * 60}")

    for cycle in range(1, args.max_cycles + 1):
        print(f"\n>>> CYCLE {cycle}/{args.max_cycles}")
        if args.inspect_only:
            builder = {"agent": "builder", "status": "SUCCESS", "attempts": 0, "history": []}
        elif args.no_llm:
            b = agent_builder.run_build(clean=args.clean)
            builder = {"agent": "builder", "status": "SUCCESS" if b["success"] else "FAILED", "attempts": 1,
                       "last_log_tail": None if b["success"] else agent_builder.extract_error_lines(b["combined"])}
        else:
            builder = agent_builder.run(args.max_attempts, clean=args.clean)
        save_report(cycle, builder, "builder")
        if builder["status"] != "SUCCESS":
            print(f"[Lead] build status {builder['status']}: {builder.get('escalation_reason', '')}")
            if builder.get("last_log_tail"):
                print(builder["last_log_tail"][-2500:])

        inspector = None
        if not args.build_only and builder["status"] == "SUCCESS":
            inspector = agent_inspector.run(args.tests, analyze=not args.no_llm)
            save_report(cycle, inspector, "inspector")

        if args.no_llm:
            ok = builder["status"] == "SUCCESS" and (inspector is None or inspector.get("clean"))
            sys.exit(0 if ok else 1)

        lead = synthesize(builder, inspector)
        save_report(cycle, lead, "lead")
        print_lead(lead)
        nxt = lead.get("next_action", "MANUAL_REVIEW")
        if nxt == "DONE":
            print("\n[Lead] ✓ done")
            break
        if args.auto:
            res = apply_fixes(lead)
            if not any(r.get("applied") for r in res):
                print("[Lead] --auto: nothing auto-applicable — stopping for manual review")
                break
        elif sys.stdin.isatty():
            if input("\n  Apply SOURCE_PATCH fixes and run another cycle? [y/N]: ").strip().lower() != "y":
                break
            apply_fixes(lead)
        else:
            break
        args.inspect_only = False
    print(f"\n[Orchestrator] reports in {config.rel(config.REPORTS_DIR)}")


if __name__ == "__main__":
    main()
