"""
Safe source patcher (counterpart of wined3d-windows' dockerfile_patcher).

Applies exact literal {"file", "old", "new"} replacements — no sed, no regex,
no shell — only inside config.WRITABLE_PREFIXES, rejects anything protected,
requires `old` to occur exactly once, writes atomically, and commits each fix
on its own so it can be reverted with `git revert`.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import config


def parse_patch_json(text: str) -> dict | None:
    try:
        obj = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and {"file", "old", "new"} <= obj.keys():
        return obj
    return None


def _allowed(relpath: str) -> str | None:
    if relpath.startswith("/") or ".." in Path(relpath).parts:
        return "path must be relative to the project"
    if any(relpath.startswith(p) for p in config.PROTECTED_PREFIXES):
        return "protected path"
    if not any(relpath.startswith(p) for p in config.WRITABLE_PREFIXES):
        return "outside agent-writable paths"
    return None


def patch(relpath: str, old: str, new: str, title: str, commit: bool = True) -> dict:
    reason = _allowed(relpath)
    if reason:
        return {"ok": False, "reason": reason}
    path = config.PROJECT_DIR / relpath
    if old == "":  # create a new file
        if path.exists():
            return {"ok": False, "reason": "file exists; give non-empty 'old' text"}
        text = new
    else:
        if not path.exists():
            return {"ok": False, "reason": "file not found"}
        cur = path.read_text()
        n = cur.count(old)
        if n != 1:
            return {"ok": False, "reason": f"'old' text occurs {n} times (must be exactly 1)"}
        text = cur.replace(old, new, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".patch-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    if commit:
        subprocess.run(["git", "add", relpath], cwd=config.PROJECT_DIR, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", f"agents: {title}"], cwd=config.PROJECT_DIR,
                       capture_output=True)
    return {"ok": True}


def get_context(relpath: str, needle: str, context_lines: int = 12) -> str:
    """Excerpt of a source file around `needle` so the model can quote exact text."""
    path = config.PROJECT_DIR / relpath
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            lo, hi = max(0, i - context_lines), min(len(lines), i + context_lines + 1)
            return "\n".join(f"{n + 1:5d}  {lines[n]}" for n in range(lo, hi))
    return ""
