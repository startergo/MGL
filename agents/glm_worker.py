#!/usr/bin/env python3
"""
glm_worker.py — autonomous coding agent for one MGL domain, driven by GLM (Z.AI)
through the Anthropic-compatible tool-use API (see llm_client.py; AGENT_PROVIDER
=anthropic also works).

The agent gets a task brief (agents/tasks/<domain>.md) plus agents/BRIEF.md and works
inside ONE git worktree with four tools:

    run_shell(command, timeout)     bash in the worktree (per-command timeout, output trimmed)
    read_file(path, offset, limit)  numbered lines
    write_file(path, content)       create/overwrite (inside the worktree only)
    str_replace(path, old, new)     exact single-occurrence replacement

It stops when the model ends its turn without a tool call (that final text is the
report, written to <worktree>/agent_report.md) or after --max-turns.

    python3 agents/glm_worker.py --domain texdata            # worktree wt/texdata
    python3 agents/glm_worker.py --domain tess --max-turns 400
    python3 agents/glm_worker.py --domain geom --resume       # continue from saved transcript

Logs: <worktree>/agent.log (human), <worktree>/.agent/transcript.json (resume state).
"""

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config        # noqa: E402
import llm_client    # noqa: E402

WORKTREE_ROOT = Path(os.environ.get("MGL_WORKTREE_ROOT") or config.PROJECT_DIR / "wt")
TASKS_DIR = config.AGENTS_DIR / "tasks"
MAX_TOOL_OUTPUT = int(os.environ.get("GLM_MAX_TOOL_OUTPUT", "20000"))
CONTEXT_BUDGET_CHARS = int(os.environ.get("GLM_CONTEXT_BUDGET_CHARS", "450000"))
DEFAULT_CMD_TIMEOUT = 300
MAX_CMD_TIMEOUT = 1800

# Commands the agent may never run (it works in a git worktree on the user's machine).
DENY = [
    r"\bgit\s+push\b", r"\bgit\s+remote\b", r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)",
    r"\bgit\s+checkout\s+(gl46-port|main|master)\b", r"\bgit\s+worktree\b", r"\bgit\s+branch\s+-[dD]\b",
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+(/|~|\.\.|\$HOME)", r"\bsudo\b", r"\bcurl\b", r"\bwget\b",
    r"\bssh\b", r"\bscp\b", r"\bnc\b", r"\blaunchctl\b", r"\bdefaults\s+write\b",
    r"(^|\s)(cd|pushd)\s+(/|~|\.\.)", r"\bactivate_project\b",
]

TOOLS = [
    {"name": "run_shell",
     "description": "Run a bash command in the worktree root (cwd is reset every call). Use it to build "
                    "(make -j8 lib), run tests (tests/headless/run.sh test_x), grep, sed -n, git status/diff/"
                    "add/commit/merge. Output is trimmed. Commands are killed after `timeout` seconds.",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": "seconds, default 300, max 1800"}},
         "required": ["command"]}},
    {"name": "read_file",
     "description": "Read a text file inside the worktree, with line numbers. Use offset/limit for large files.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "relative to the worktree root"},
         "offset": {"type": "integer", "description": "1-based first line"},
         "limit": {"type": "integer", "description": "max lines, default 400"}},
         "required": ["path"]}},
    {"name": "write_file",
     "description": "Create or overwrite a file inside the worktree with the full content given.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]}},
    {"name": "str_replace",
     "description": "Replace an exact string that occurs exactly once in a file inside the worktree. "
                    "Include enough surrounding context to make old_str unique.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
         "required": ["path", "old_str", "new_str"]}},
]

SYSTEM_TEMPLATE = """You are an autonomous senior graphics engineer working on MGL, an OpenGL 4.6 core-profile
implementation on Apple Metal (C + Objective-C/ARC; GLSL -> glslang -> SPIR-V -> SPIRV-Cross -> MSL).
You are the '{domain}' domain agent. Other domains are already implemented and merged; a lead engineer
merges your branch when you finish.

You work ONLY inside your git worktree (the cwd of every tool call), branch gl46/{domain}, on macOS
with Xcode command-line tools. You have four tools: run_shell, read_file, write_file, str_replace.
Paths are relative to the worktree root.

Working rules:
- Explore before editing: grep/sed -n the relevant code, understand existing helpers and hooks.
- Make small, verifiable steps: edit -> `make -j8 lib 2>&1 | grep -E 'error|warning: implicit' | head -40`
  -> run your tests -> commit (`git add -A MGL tests && git commit -m "{domain}: ..."`).
- Never run long-lived or interactive processes; tests already have a per-test timeout.
- Never push, never touch other worktrees or the main checkout, never edit external/.
- Real implementations only: no empty bodies, no assert(0), no "TODO"/no-op stubs. If Metal truly
  cannot do something, implement state + queries + spec errors and document the gap.
- Before finishing: full `tests/headless/run.sh` must show no new failures, `python3 agents/stub_scanner.py
  --list EMPTY|NOOP|ASSERT` must be clean for your files, all work committed.
- When finished, reply WITHOUT any tool call with your final report: what you implemented and how
  (Metal mechanism), shared-file edits, test results (paste the == summary lines), remaining gaps.

The brief below was written for agents that reached the Mac through a remote shell tool: ignore its
instructions about ToolSearch, serena, activate_project and "your own tools act on another machine" —
your run_shell / read_file / write_file / str_replace tools run directly on the Mac in your worktree.

=== PROJECT BRIEF (agents/BRIEF.md) ===
{brief}

=== YOUR TASK (agents/tasks/{domain}.md) ===
{task}
"""


def log(fp, text):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    fp.write(f"[{stamp}] {text}\n")
    fp.flush()


def inside(worktree: Path, rel: str) -> Path:
    p = (worktree / rel).resolve()
    if p != worktree and worktree not in p.parents:
        raise ValueError(f"path escapes the worktree: {rel}")
    if (worktree / "external") in p.parents or p == worktree / "external":
        raise ValueError("external/ is read-only (shared prebuilt dependencies)")
    return p


def trim(text: str) -> str:
    text = config.scrub(text)
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    half = MAX_TOOL_OUTPUT // 2
    return text[:half] + f"\n... [{len(text) - MAX_TOOL_OUTPUT} chars trimmed] ...\n" + text[-half:]


def tool_run_shell(worktree: Path, args: dict) -> str:
    cmd = args.get("command", "")
    for pat in DENY:
        if re.search(pat, cmd):
            return f"REFUSED: command matches a blocked pattern ({pat}). Work inside the worktree only."
    timeout = max(1, min(int(args.get("timeout") or DEFAULT_CMD_TIMEOUT), MAX_CMD_TIMEOUT))
    try:
        proc = subprocess.Popen(["/bin/bash", "-c", cmd], cwd=worktree, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                env={**os.environ, "MGL_PROJECT_DIR": str(worktree)})
        try:
            out, _ = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            rc = f"TIMEOUT after {timeout}s (process group killed)"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    return trim(f"exit={rc}\n{out}")


def tool_read_file(worktree: Path, args: dict) -> str:
    p = inside(worktree, args["path"])
    if not p.is_file():
        return f"ERROR: not a file: {args['path']}"
    lines = p.read_text(errors="replace").splitlines()
    off = max(1, int(args.get("offset") or 1))
    lim = max(1, int(args.get("limit") or 400))
    chunk = lines[off - 1:off - 1 + lim]
    body = "\n".join(f"{off + i:6d}\t{line}" for i, line in enumerate(chunk))
    more = f"\n[{len(lines) - (off - 1 + len(chunk))} more lines]" if off - 1 + len(chunk) < len(lines) else ""
    return trim(body + more)


def tool_write_file(worktree: Path, args: dict) -> str:
    p = inside(worktree, args["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return f"wrote {args['path']} ({len(args['content'])} bytes)"


def tool_str_replace(worktree: Path, args: dict) -> str:
    p = inside(worktree, args["path"])
    if not p.is_file():
        return f"ERROR: not a file: {args['path']}"
    text = p.read_text()
    n = text.count(args["old_str"])
    if n != 1:
        return f"ERROR: old_str occurs {n} times in {args['path']} (must be exactly 1)"
    p.write_text(text.replace(args["old_str"], args["new_str"], 1))
    return f"replaced 1 occurrence in {args['path']}"


HANDLERS = {"run_shell": tool_run_shell, "read_file": tool_read_file,
            "write_file": tool_write_file, "str_replace": tool_str_replace}


def block_to_dict(b):
    d = b.model_dump() if hasattr(b, "model_dump") else dict(b)
    return {k: v for k, v in d.items() if v is not None}


def compact(messages: list) -> None:
    """Keep the conversation under budget: elide old tool results (newest kept intact)."""
    def size():
        return len(json.dumps(messages, default=str))
    if size() <= CONTEXT_BUDGET_CHARS:
        return
    for msg in messages[1:-8]:
        if msg["role"] != "user" or not isinstance(msg["content"], list):
            continue
        for part in msg["content"]:
            if part.get("type") == "tool_result" and isinstance(part.get("content"), str) \
                    and len(part["content"]) > 400:
                part["content"] = part["content"][:300] + "\n[older output elided to save context]"
        if size() <= CONTEXT_BUDGET_CHARS:
            return


def create_with_retry(client, **kwargs):
    delay = 5
    for attempt in range(8):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001  (rate limits, overload, network)
            msg = str(e)
            # Z.AI 1113 = account has no balance / resource package: retrying cannot help
            if "1113" in msg or "Insufficient balance" in msg or "invalid api key" in msg.lower() \
                    or getattr(e, "status_code", None) in (401, 403):
                raise SystemExit(f"FATAL provider error, not retrying: {msg[:200]}\n"
                                 "Fix the account/key, then re-run agents/run_glm_domains.sh <domains> "
                                 "(workers resume from their saved transcripts).")
            if attempt == 7:
                raise
            print(f"  API error ({type(e).__name__}: {str(e)[:160]}); retry in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)


def take_inbox(state_dir):
    """Supervisor messages: text dropped in <worktree>/.agent/inbox.md is delivered once."""
    f = state_dir / "inbox.md"
    if not f.exists():
        return ""
    text = f.read_text().strip()
    f.rename(state_dir / f"inbox.delivered.{int(time.time())}.md")
    return text


def main():
    ap = argparse.ArgumentParser(description="GLM coding agent for one MGL domain")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--worktree", help="default: $MGL_WORKTREE_ROOT/<domain>")
    ap.add_argument("--task", help="default: agents/tasks/<domain>.md")
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("GLM_MAX_TURNS", "300")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("GLM_MAX_TOKENS", "16384")))
    ap.add_argument("--resume", action="store_true", help="continue from .agent/transcript.json")
    args = ap.parse_args()

    worktree = Path(args.worktree).resolve() if args.worktree else (WORKTREE_ROOT / args.domain).resolve()
    if not (worktree / ".git").exists():
        raise SystemExit(f"not a git worktree: {config.rel(worktree)} (run agents/tools/setup_worktrees.sh)")
    task_path = Path(args.task) if args.task else TASKS_DIR / f"{args.domain}.md"
    brief = (config.AGENTS_DIR / "BRIEF.md").read_text()
    task = task_path.read_text()
    state_dir = worktree / ".agent"
    state_dir.mkdir(exist_ok=True)
    transcript = state_dir / "transcript.json"
    logf = open(worktree / "agent.log", "a")

    client, model = llm_client.get_client(), llm_client.get_model()
    system = SYSTEM_TEMPLATE.format(domain=args.domain, brief=brief, task=task)
    if args.resume and transcript.exists():
        messages = json.loads(transcript.read_text())
        note = take_inbox(state_dir)
        messages.append({"role": "user", "content": "Continue the task from where you left off. "
                         "Check `git status` and `git log --oneline -5` first."
                         + ("\n\nMESSAGE FROM THE SUPERVISOR (takes priority over earlier plans):\n" + note if note else "")})
        log(logf, f"resume ({len(messages)} messages)")
    else:
        messages = [{"role": "user", "content": "Start the task. Begin by reading the relevant code."}]
        log(logf, f"start domain={args.domain} model={model} provider={llm_client.get_provider()}")

    gen = llm_client.generation_kwargs(args.max_tokens)
    for turn in range(1, args.max_turns + 1):
        compact(messages)
        resp = create_with_retry(client, model=model, system=system, messages=messages, tools=TOOLS, **gen)
        content = [block_to_dict(b) for b in resp.content]
        messages.append({"role": "assistant", "content": content})
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
        if text:
            log(logf, f"turn {turn} says: {text[:600]}")
        calls = [b for b in content if b.get("type") == "tool_use"]
        if not calls:
            (worktree / "agent_report.md").write_text(text + "\n")
            log(logf, f"finished after {turn} turns; report in agent_report.md")
            transcript.write_text(json.dumps(messages, default=str))
            print(f"[{args.domain}] done in {turn} turns")
            return 0
        results = []
        for c in calls:
            name, inp = c["name"], c.get("input") or {}
            brief_inp = json.dumps(inp)[:300]
            try:
                out = HANDLERS[name](worktree, inp) if name in HANDLERS else f"ERROR: unknown tool {name}"
            except Exception as e:  # noqa: BLE001
                out = f"ERROR: {type(e).__name__}: {e}"
            log(logf, f"turn {turn} {name} {brief_inp} -> {out[:200].replace(chr(10), ' | ')}")
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": out})
        note = take_inbox(state_dir)
        if note:
            results.append({"type": "text", "text": "MESSAGE FROM THE SUPERVISOR (takes priority over earlier plans):\n" + note})
            log(logf, f"turn {turn} inbox delivered: {note[:200]}")
        messages.append({"role": "user", "content": results})
        transcript.write_text(json.dumps(messages, default=str))
    log(logf, f"stopped: reached --max-turns {args.max_turns} (resume with --resume)")
    print(f"[{args.domain}] max turns reached; resume with --resume")
    return 2


if __name__ == "__main__":
    sys.exit(main())
