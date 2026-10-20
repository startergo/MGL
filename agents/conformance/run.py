#!/usr/bin/env python3
"""Khronos GL CTS (KHR-GL46) runner for MGL, with per-case timeouts and crash recovery.

    python3 agents/conformance/run.py [--pattern 'KHR-GL46.*'] [--limit N]
                                      [--batch N] [--per-case-timeout S]
                                      [--max-seconds S] [--resume]
                                      [--summarize-only]

How it works
------------
* The full case list is exported once (glcts --deqp-runmode=stdout-caselist).
* Cases are run in batches: glcts gets a trie-format caselist file
  (--deqp-caselist-file) and writes a .qpa log per batch.
* The .qpa is parsed for `#beginTestCaseResult <name>` + `<Result StatusCode=..>`
  pairs; every case that started but never got a result is the "victim" of a
  crash or hang: the process is killed at the batch deadline, the victim is
  recorded as Timeout/Crash, and the remaining cases of the batch are re-run
  in the next batch (so a single bad case cannot wedge the run).
* results.json (in the results dir) is the checkpoint: it is rewritten after
  every batch, and --resume skips cases that already have a result.
* --summarize-only prints the per-group table and failure clusters from
  results.json.

Everything (case list, logs, results) lives outside the repo, in
$MGL_CTS_RESULTS (default <cts>/mgl-results).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# --------------------------------------------------------------------------- paths


def find_cts_dir(explicit):
    cands = [explicit, os.environ.get("MGL_CTS_DIR"),
             os.path.join(os.path.expanduser("~"), "MGL-cts")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "external", "openglcts")):
            return c
    sys.exit("cannot find the VK-GL-CTS tree (set MGL_CTS_DIR); run setup.sh first")


def default_results_dir(cts_dir):
    return os.environ.get("MGL_CTS_RESULTS", os.path.join(cts_dir, "mgl-results"))


# --------------------------------------------------------------------------- trie case lists


def to_trie(paths):
    """Flat case paths -> CTS trie format {pkg{group{leaf,leaf}}}}"""
    root = {}
    for p in paths:
        node = root
        for part in p.split("."):
            node = node.setdefault(part, {})
    def ser(node):
        out = []
        for k in sorted(node):
            if node[k]:
                out.append(k + "{" + ser(node[k]) + "}")
            else:
                out.append(k)
        return ",".join(out)
    return "{" + ser(root) + "}"


# --------------------------------------------------------------------------- qpa parsing

BEGIN_RE = re.compile(r"^#beginTestCaseResult (\S+)")
RESULT_RE = re.compile(r'<Result StatusCode="(\w+)"')
STATUS_LINE_RE = re.compile(r"^\s*#?(Pass|Fail|QualityWarning|NotSupported|InternalError|Crash|Timeout|ResourceError|Waived)\b")


def parse_qpa(path):
    """-> (ordered list of (case, status), set of cases that began)"""
    results = []
    began = []
    cur = None
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = BEGIN_RE.match(line)
                if m:
                    cur = m.group(1)
                    began.append(cur)
                    continue
                m = RESULT_RE.search(line)
                if m and cur:
                    results.append((cur, m.group(1)))
                    cur = None
    except FileNotFoundError:
        pass
    return results, began


def extract_reason(path, case):
    """First interesting line of the case body, for failure clustering."""
    begin_re = re.compile(r"^#beginTestCaseResult (\S+)$")
    try:
        with open(path, "r", errors="replace") as fh:
            in_case = False
            body = []
            for line in fh:
                m = begin_re.match(line.rstrip())
                if m:
                    in_case = (m.group(1) == case)
                    body = []
                    continue
                if in_case:
                    if RESULT_RE.search(line):
                        break
                    body.append(line.rstrip())
            for ln in reversed(body):
                s = ln.strip()
                if not s:
                    continue
                if re.search(r"(comparison failed|expected|got|error|invalid|mismatch|failed|not supported|assert)",
                             s, re.I):
                    return s[:160]
            for ln in reversed(body):
                s = ln.strip()
                if s and not s.startswith("<") and not s.startswith("#"):
                    return s[:160]
    except FileNotFoundError:
        pass
    return ""


# --------------------------------------------------------------------------- runner


class Runner:
    def __init__(self, args):
        self.cts = find_cts_dir(args.cts_dir)
        self.bin_dir = os.path.join(self.cts, "build", "external", "openglcts", "modules")
        self.glcts = os.path.join(self.bin_dir, "glcts")
        self.results_dir = args.results_dir or default_results_dir(self.cts)
        self.logs_dir = os.path.join(self.results_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.mgl_lib = args.mgl_lib or os.path.join(os.getcwd(), "build", "libmgl.dylib")
        if not os.path.exists(self.mgl_lib):
            sys.exit("missing %s -- build MGL first (make lib)" % self.mgl_lib)
        self.pattern = args.pattern
        self.batch = args.batch
        self.per_case = args.per_case_timeout
        self.startup = 45.0
        self.args = args

        self.caselist_path = os.path.join(self.results_dir, "caselist.txt")
        self.results_path = os.path.join(self.results_dir, "results.json")

    # -- case list -------------------------------------------------------
    def caselist(self):
        if os.path.exists(self.caselist_path) and not self.args.refresh_caselist:
            with open(self.caselist_path) as fh:
                return [l.strip() for l in fh if l.strip()]
        cmd = [self.glcts, "--deqp-runmode=stdout-caselist", "--deqp-case=" + self.pattern]
        env = dict(os.environ, MGL_LIBRARY=self.mgl_lib)
        out = subprocess.run(cmd, cwd=self.bin_dir, env=env, capture_output=True,
                             text=True, timeout=300).stdout
        cases = [l[len("TEST: "):].strip() for l in out.splitlines() if l.startswith("TEST: ")]
        with open(self.caselist_path, "w") as fh:
            fh.write("\n".join(cases) + "\n")
        return cases

    # -- one batch --------------------------------------------------------
    def run_batch(self, idx, cases):
        trie_path = os.path.join(self.results_dir, "batch.trie")
        with open(trie_path, "w") as fh:
            fh.write(to_trie(cases))
        qpa = os.path.join(self.logs_dir, "batch_%05d.qpa" % idx)
        stdout_p = os.path.join(self.logs_dir, "batch_%05d.out" % idx)
        stderr_p = os.path.join(self.logs_dir, "batch_%05d.err" % idx)
        cmd = [self.glcts,
               "--deqp-caselist-file=" + trie_path,
               "--deqp-log-filename=" + qpa,
               "--deqp-surface-type=fbo",
               "--deqp-gl-context-type=mgl",
               "--deqp-visibility=hidden",
               "--deqp-surface-width=256", "--deqp-surface-height=256",
               "--deqp-log-images=disable", "--deqp-log-shader-sources=disable",
               "--deqp-log-decompiled-spirv=disable", "--deqp-log-empty-loginfo=disable",
               "--deqp-log-flush=enable",
               "--deqp-base-seed=0"]
        env = dict(os.environ, MGL_LIBRARY=self.mgl_lib)
        deadline = self.startup + self.per_case * len(cases)
        t0 = time.time()
        timed_out = False
        with open(stdout_p, "wb") as so, open(stderr_p, "wb") as se:
            try:
                proc = subprocess.Popen(cmd, cwd=self.bin_dir, env=env, stdout=so, stderr=se,
                                        start_new_session=True)
                try:
                    rc = proc.wait(timeout=deadline)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    rc = -9
            except OSError as e:
                print("failed to start glcts: %s" % e, file=sys.stderr)
                return {}, cases, 0.0
        dt = time.time() - t0

        results, began = parse_qpa(qpa)
        got = dict(results)
        order = {c: i for i, c in enumerate(cases)}
        began_sorted = sorted(began, key=lambda c: order.get(c, 1 << 30))

        # victim = first begun case without a result (or first case if the
        # process died before starting anything)
        unfinished = [c for c in cases if c not in got]
        if unfinished:
            victim = None
            for c in began_sorted:
                if c in unfinished:
                    victim = c
                    break
            if victim is None and not began:
                victim = unfinished[0]
            if victim is not None:
                got[victim] = "Timeout" if timed_out else "Crash"
                reasons = {victim: ("batch killed at %ds deadline" % deadline) if timed_out
                           else ("glcts exited rc=%s" % rc)}
                return got, [c for c in unfinished if c != victim], reasons
        return got, [], {}

    # -- main loop ---------------------------------------------------------
    def run(self):
        results = {}
        if self.args.resume and os.path.exists(self.results_path):
            with open(self.results_path) as fh:
                results = json.load(fh).get("results", {})
        cases = [c for c in self.caselist() if c not in results]
        if self.args.limit:
            cases = cases[:self.args.limit]
        total = len(cases)
        print("running %d cases (batch=%d, per-case timeout=%ds)"
              % (total, self.batch, self.per_case))
        t_start = time.time()
        consecutive_bad = 0
        idx = 0
        while cases:
            idx += 1
            if self.args.max_seconds and time.time() - t_start > self.args.max_seconds:
                print("time budget reached; %d cases left" % len(cases))
                break
            batch, cases = cases[:self.batch], cases[self.batch:]
            got, requeue, reasons = self.run_batch(idx, batch)
            for c, st in got.items():
                results[c] = {"status": st, "reason": reasons.get(c) or extract_reason(
                    os.path.join(self.logs_dir, "batch_%05d.qpa" % idx), c)}
            cases = requeue + cases
            with open(self.results_path + ".tmp", "w") as fh:
                json.dump({"results": results}, fh, indent=0, sort_keys=True)
            os.replace(self.results_path + ".tmp", self.results_path)
            n_ok = len(got) - sum(1 for s in got.values() if s in ("Crash", "Timeout"))
            consecutive_bad = consecutive_bad + 1 if n_ok == 0 else 0
            print("[%3d] %4d/%4d done | batch: %d results (%d bad) | %s"
                  % (idx, total - len(cases), total, len(got), len(got) - n_ok,
                     time.strftime("%H:%M:%S")), flush=True)
            if consecutive_bad >= 10:
                print("aborting: 10 consecutive batches produced no results", file=sys.stderr)
                break
        self.summarize(results)

    # -- summary -----------------------------------------------------------
    def summarize(self, results):
        counts = {}
        for case, r in results.items():
            st = r["status"]
            counts[st] = counts.get(st, 0) + 1
        total = len(results)
        print("\n== totals (%d cases):" % total)
        for st in sorted(counts, key=lambda s: -counts[s]):
            print("  %-14s %6d  (%.1f%%)" % (st, counts[st], 100.0 * counts[st] / max(total, 1)))

        groups = {}
        for case, r in results.items():
            parts = case.split(".")
            g = ".".join(parts[:2]) if len(parts) > 2 else parts[0]
            groups.setdefault(g, {}).setdefault(r["status"], 0)
            groups[g][r["status"]] += 1
        print("\n== per-group (sorted by fails+crashes+timeouts):")
        def bad(g):
            return sum(g.get(s, 0) for s in ("Fail", "Crash", "Timeout", "InternalError", "ResourceError"))
        rows = sorted(groups.items(), key=lambda kv: (-bad(kv[1]), -sum(kv[1].values())))
        with open(os.path.join(self.results_dir, "summary.txt"), "w") as fh:
            for g, cc in rows:
                line = "%-46s %5d cases  " % (g, sum(cc.values())) + \
                       " ".join("%s=%d" % (s, n) for s, n in sorted(cc.items(), key=lambda x: -x[1]))
                print("  " + line)
                fh.write(line + "\n")

        # failure clusters: group Fail cases by normalized reason
        clusters = {}
        for case, r in results.items():
            if r["status"] not in ("Fail", "InternalError", "Crash", "Timeout"):
                continue
            reason = r.get("reason") or ""
            sig = re.sub(r"\d+(\.\d+)?", "N", reason)[:80]
            clusters.setdefault((r["status"], sig), []).append(case)
        rows = sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:25]
        print("\n== top failure clusters:")
        with open(os.path.join(self.results_dir, "clusters.txt"), "w") as fh:
            for (st, sig), cs in rows:
                line = "[%s x%4d] %s\n    e.g. %s" % (st, len(cs), sig or "(no detail)", cs[0])
                print("  " + line.replace("\n", "\n  "))
                fh.write(line + "\n")
                for c in cs[:5]:
                    fh.write("      %s\n" % c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="KHR-GL46.*")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--per-case-timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--refresh-caselist", action="store_true")
    ap.add_argument("--cts-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--mgl-lib")
    args = ap.parse_args()

    r = Runner(args)
    if args.summarize_only:
        if os.path.exists(r.results_path):
            with open(r.results_path) as fh:
                results = json.load(fh)["results"]
            r.summarize(results)
        else:
            sys.exit("no results yet at %s" % r.results_path)
        return
    r.run()


if __name__ == "__main__":
    main()
