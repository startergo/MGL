# MGL multi-agent build & port system

Adapted from the wined3d-windows agents: the same Builder → Inspector → Lead Engineer
loop, retargeted from Docker/PE DLLs to MGL's `make lib` build, its GL 4.6 stub
inventory, and the headless Metal test suite. Plus the tooling used to split the
GL 4.6 port across parallel domain agents.

Nothing in this folder contains machine-specific paths, user names, or keys. All
configuration comes from environment variables (see `.env.example`), with defaults
relative to the repository. `tools/check_no_secrets.py` enforces this.

## Architecture

```
orchestrator.py          Lead Engineer — synthesizes reports into prioritized literal patches
 ├── agent_builder.py    Agent 1: `make lib`, extracts compiler/linker errors, one safe fix per attempt
 └── agent_inspector.py  Agent 2: stub inventory + exported-symbol check + headless tests
source_patcher.py        exact {"file","old","new"} replacement, writable-path allowlist, git commit per fix
stub_scanner.py          classifies every mgl* entry point: IMPLEMENTED / EMPTY / NOOP / ASSERT
config.py                env-driven paths and limits; scrubs absolute paths from logs/reports
llm_client.py            GLM (Z.AI, default) or Anthropic client from env
```

## Setup

```bash
cd agents
pip3 install -r requirements.txt --break-system-packages
cp .env.example .env          # git-ignored; fill in your Z.AI key (GLM is the default)
set -a; . ./.env; set +a      # or export the variables in your shell profile
```

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_PROVIDER` | `glm` | `glm` (Z.AI) or `anthropic` |
| `ANTHROPIC_AUTH_TOKEN` | — | Z.AI API key (required for GLM unless `--no-llm`) |
| `ANTHROPIC_API_KEY` | — | Anthropic key when `AGENT_PROVIDER=anthropic` |
| `ANTHROPIC_BASE_URL` | `https://api.z.ai/api/anthropic` (GLM) | endpoint override |
| `AGENT_MODEL` | `glm-5.3` (`claude-opus-5-5` for anthropic) | model id |
| `AGENT_REASONING_EFFORT` | `high` | GLM reasoning effort (`high` / `max`) |
| `MGL_PROJECT_DIR` | parent of `agents/` | repository root |
| `MGL_BUILD_DIR` | `$MGL_PROJECT_DIR/build` | build output |
| `MGL_REPORTS_DIR` | `$MGL_BUILD_DIR/_reports` | JSON reports |
| `MGL_WORKTREE_ROOT` | `$MGL_PROJECT_DIR/wt` | domain worktrees |
| `MGL_MAKE_JOBS`, `MGL_BUILD_TIMEOUT`, `MGL_TEST_TIMEOUT` | CPU count, 1800, 900 | limits |

## Usage

```bash
python3 orchestrator.py                       # one interactive cycle
python3 orchestrator.py --no-llm              # build + inventory + tests, no API calls (CI-friendly, exit code)
python3 orchestrator.py --inspect-only        # reuse existing build/libmgl.dylib
python3 orchestrator.py --tests test_fb test_queries
python3 orchestrator.py --auto --max-cycles 5 # apply SOURCE_PATCH fixes and loop until DONE

python3 stub_scanner.py                       # inventory summary
python3 stub_scanner.py --list EMPTY --file queries.c
python3 stub_scanner.py --json                # machine-readable
```

## Agent roles

**Builder** runs `make -j$MGL_MAKE_JOBS lib`. On failure it sends the error lines plus source
excerpts around each error to the model, which returns one literal patch. Patches outside
`MGL/src/`, `MGL/include/mgl_ext*`, `tests/headless/`, or touching protected files
(`external/`, generated `gl_core.c`/`glm_dispatch.*`, GL headers, `Makefile`) are rejected.
Stub-style "fixes" (emptying bodies, `assert(0)`) are forbidden by the prompt.

**Inspector** replaces the PE export/import diff with:
1. `stub_scanner.py` — GL 4.6 core commands (from `MGL/include/GL/glcorearb.h`) without a real body,
2. `nm -gU build/libmgl.dylib` — core `gl*` symbols not exported,
3. `tests/headless/run.sh` — PASS/FAIL per test with diagnostics.

**Lead Engineer** merges both reports into prioritized fixes: `SOURCE_PATCH` (auto-applicable,
committed one per fix so `git revert` undoes it) or `MANUAL` (described for a human/agent).

## Fix categories

| Type | Description | Auto-applied? |
|---|---|---|
| `SOURCE_PATCH` | exact literal replacement in one writable file | yes (`--auto` or on confirm) |
| `MANUAL` | new Metal emulation path, cross-domain refactor, external deps | no — reported |

## Parallel domain port (how GL 4.6 was split)

The port was done by 12 agents working at once, each on its own branch/worktree:

| Domain | Owns |
|---|---|
| `fb` | framebuffers, renderbuffers, readback, blits, ClearBuffer* |
| `queries` | query objects, timers, pipeline statistics, conditional render |
| `debug` | KHR_debug, labels, glGet*/GetString(i), compat-profile entry points |
| `introspect` | uniforms, ProgramUniform*, program interface queries, subroutines |
| `program` | programs/shaders, pipelines, binaries, SPIR-V (ARB_gl_spirv) |
| `xfb` | transform feedback |
| `vattrib` | generic vertex attributes, VAO queries, vertex formats |
| `rstate` | blend/raster/depth/stencil state, viewport & scissor arrays, clip control |
| `texparam` | texture/sampler parameters, level and internalformat queries |
| `texdata` | texture images, compressed formats, copies, GetTexImage |
| `texext` | buffer textures, texture views, multisample textures, barriers |
| `buffers` | buffer objects, mapping, draws, indirect/count draws, compute, sync |

Conflict avoidance (set up by `tools/restructure.py`, `tools/make_private_header.py`,
`tools/make_ext_scaffold.py`):

- each domain's functions live in their own source files;
- Metal code goes in `MGL/src/MGLRenderer+<Domain>.m` categories (`MGLRenderer_Private.h` exposes the
  renderer ivars); `-bindExt*FuncsToGLMContext:` and `-ext<Domain>RenderPassDescriptor:/RenderEncoderCreated:/PreDraw:/WillCommit:`
  methods are discovered at runtime, so no shared registration code changes;
- per-domain state/hook slots (`ctx->state.ext_<d>`, `ctx->mtl_funcs.ext_<d>`), a glGet fan-out,
  and registries for indexed glGet, link, and pre-draw hooks are in `MGL/include/mgl_ext.h`.

```bash
agents/tools/setup_worktrees.sh gl46-port         # wt/<domain> on branch gl46/<domain>
# ... agents work, following agents/BRIEF.md ...
agents/tools/merge_domains.sh                     # merge all, build, scan, test
```

## Before committing or uploading

```bash
python3 agents/tools/check_no_secrets.py          # fails on absolute home paths, e-mails, keys, private hosts
ln -s ../../agents/tools/check_no_secrets.py .git/hooks/pre-commit   # optional hook
```

Reports and build logs pass through `config.scrub()`, which strips the project and home
directory prefixes, so saved JSON contains repository-relative paths only.
