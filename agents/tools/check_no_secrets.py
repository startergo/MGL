#!/usr/bin/env python3
"""
Fail if tracked agent/test files contain machine-specific or sensitive data:
absolute home paths, e-mail addresses, API keys/tokens, private hostnames/IPs.

    python3 agents/tools/check_no_secrets.py            # scan agents/ and tests/
    python3 agents/tools/check_no_secrets.py PATH...    # scan specific paths

Install as a pre-commit hook:
    ln -s ../../agents/tools/check_no_secrets.py .git/hooks/pre-commit
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = [ROOT / "agents", ROOT / "tests"]
SKIP_DIRS = {"__pycache__", "output", "_reports", ".git"}
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".dylib", ".o", ".a"}

PATTERNS = [
    ("absolute home path", re.compile(r"(?<![\w$])/(?:Users|home)/[A-Za-z0-9._-]+")),
    ("Windows user path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._-]+")),
    ("e-mail address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("Anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")),
    ("generic API key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer/secret assignment", re.compile(r"(?i)\b(?:api[_-]?key|auth[_-]?token|secret|password)\s*[:=]\s*['\"][^'\"$<{]{8,}['\"]")),
    ("private IP", re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b")),
    ("local hostname", re.compile(r"\b[\w-]+\.local\b")),
]
# Placeholders that are fine to appear in docs.
ALLOW = re.compile(r"noreply@anthropic\.com|user@example\.com|<your[^>]*>")


def scan(path: Path):
    hits = []
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for f in files:
        if any(part in SKIP_DIRS for part in f.parts) or f.suffix in SKIP_SUFFIXES or f.name == Path(__file__).name:
            continue
        try:
            text = f.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            clean = ALLOW.sub("", line)
            for label, rx in PATTERNS:
                if rx.search(clean):
                    hits.append((f.relative_to(ROOT) if f.is_relative_to(ROOT) else f.name, n, label, line.strip()[:100]))
    return hits


def main():
    paths = [Path(p).resolve() for p in sys.argv[1:]] or DEFAULT_PATHS
    hits = [h for p in paths if p.exists() for h in scan(p)]
    for f, n, label, line in hits:
        print(f"{f}:{n}: {label}: {line}")
    if hits:
        print(f"\n{len(hits)} finding(s). Replace with environment variables (see agents/.env.example).")
        sys.exit(1)
    print("check_no_secrets: clean")


if __name__ == "__main__":
    main()
