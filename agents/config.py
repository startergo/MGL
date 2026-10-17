"""
Central configuration for the MGL agents. Everything comes from environment
variables; defaults are relative to the repository so nothing machine- or
user-specific is ever stored in this folder.

  MGL_PROJECT_DIR   repository root            (default: parent of agents/)
  MGL_BUILD_DIR     build output dir            (default: $MGL_PROJECT_DIR/build)
  MGL_REPORTS_DIR   JSON reports                (default: $MGL_BUILD_DIR/_reports)
  MGL_MAKE_JOBS     parallel make jobs          (default: CPU count)
  MGL_BUILD_TIMEOUT seconds for one build       (default: 1800)
  MGL_TEST_TIMEOUT  seconds for the test suite  (default: 900)

LLM settings live in llm_client.py (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN,
ANTHROPIC_BASE_URL, AGENT_MODEL, AGENT_REASONING_EFFORT). Put them in your
shell profile or an untracked .env (see .env.example) — never in a tracked file.
"""

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


AGENTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _env_path("MGL_PROJECT_DIR", AGENTS_DIR.parent)
BUILD_DIR = _env_path("MGL_BUILD_DIR", PROJECT_DIR / "build")
REPORTS_DIR = _env_path("MGL_REPORTS_DIR", BUILD_DIR / "_reports")
MAKE_JOBS = int(os.environ.get("MGL_MAKE_JOBS") or os.cpu_count() or 4)
BUILD_TIMEOUT = int(os.environ.get("MGL_BUILD_TIMEOUT", "1800"))
TEST_TIMEOUT = int(os.environ.get("MGL_TEST_TIMEOUT", "900"))

SRC_DIR = PROJECT_DIR / "MGL" / "src"
INCLUDE_DIR = PROJECT_DIR / "MGL" / "include"
TESTS_DIR = PROJECT_DIR / "tests" / "headless"

# Agents may edit only these (relative to PROJECT_DIR).
WRITABLE_PREFIXES = ("MGL/src/", "MGL/include/mgl_ext", "tests/headless/")
# Never modified by agents.
PROTECTED_PREFIXES = ("external/", "MGL/include/GL/", "MGL/include/KHR/", "MGL/include/glm/",
                      "MGL/src/gl_core.c", "MGL/src/glm_dispatch.c", "MGL/include/glm_dispatch.h",
                      "Makefile", ".git")


def rel(path) -> str:
    """Path relative to the project, for logs and reports (never absolute)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return Path(path).name


def scrub(text: str) -> str:
    """Remove the project/home prefixes from tool output before it is logged,
    saved to a report, or sent to the model."""
    if not text:
        return text
    for prefix, repl in ((str(PROJECT_DIR) + "/", ""), (str(PROJECT_DIR), "."),
                         (str(Path.home()), "~")):
        text = text.replace(prefix, repl)
    return text
