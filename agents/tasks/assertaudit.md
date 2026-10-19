# Domain: assertaudit — make sure no removed assert/stub hid missing code

Background: since the branch point 792cbab, 626 `assert(0)` lines were removed from MGL/src. Most were stub
bodies replaced by real implementations. But some may have been replaced by an error return, a silent
no-op, `return 0`, `break`, or `MTLPixelFormatInvalid` on input that the OpenGL 4.6 core spec says is
LEGAL. That is missing functionality, and it must be implemented, not rejected.

## Step 1: inventory (commit agents/assertaudit_report.md)
Get every removed site: `git diff 792cbab gl46-port -- MGL/src` (hunks with `-.*assert\s*\(\s*(0|false)`).
For each, look at what replaced it in the CURRENT code and classify:
- IMPL: real implementation now. Done.
- ILLEGAL: now raises the GL error the 4.6 spec mandates for an input the spec forbids. Cite the spec rule.
- INTERNAL: an internal invariant (caller already validated); an error/early return is fine.
- MISSING: legal input that now errors, no-ops, returns a dummy, or is silently ignored. This is the work.
- DEAD: code with no callers / inside #if 0. Delete it.
Also scan the CURRENT tree for the same smell without an assert history: switch `default:` branches in
format/type/enum mappers, "unsupported"/"not supported" paths, and functions that only set an error.
The report is a table: file:line, function, input that reaches it, class, reason.

## Step 2: fix every MISSING item, one commit per item, failing test first
Implement the real behaviour. Where Metal lacks a feature, emulate it (format conversion, wider format plus
repack, compute pass) instead of rejecting. Only if emulation is genuinely impossible, record it in the
report with the reason; that becomes a decision for the user, not a silent error.

## Step 3: delete DEAD code (one commit).

## Rules
- Do NOT edit files owned by other active workers: geometry.c, MGLRenderer+Geometry.m, mgl_ext_geom*.h,
  transform_feedback.c (geom), MGLRenderer+Tessellation.m, tessellation.c, mgl_ext_tess.h (tess), and
  pixel_utils.c / MGLTextures.m mtlPixelFormatForGLFormatType (tess is fixing those right now). If a
  MISSING item is in one of those files, list it in the report with "owner: geom" or "owner: tess" and move on.
- Run every test under `perl -e 'alarm 120; exec @ARGV' ...` or tests/headless/run.sh. Never rm -rf build.
- Full suite green before each commit.
