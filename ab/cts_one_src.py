#!/usr/bin/env python3
"""Run ONE dEQP case with --deqp-log-shader-sources=enable (and decompiled SPIR-V)
so the failing case's GLSL + toolchain view lands in the .qpa.  Scratch helper."""
import os, subprocess, sys

BIN = "/tmp/.-cts-new/build/external/openglcts/modules/glcts"
case = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "one_src"
lib = os.environ.get("MGL_LIBRARY", os.path.join(os.getcwd(), "build", "libmgl.dylib"))
lst = "/tmp/%s.txt" % name
open(lst, "w").write(case + "\n")
qpa = "/tmp/%s.qpa" % name
env = dict(os.environ, MGL_LIBRARY=lib, MGL_LOG_DIR="")
extra = []
for a in sys.argv[3:]:
    extra.append(a)
rc = subprocess.call(["perl", "-e", "alarm shift; exec @ARGV", "300", BIN,
                      "--deqp-caselist-file=" + lst, "--deqp-log-filename=" + qpa,
                      "--deqp-surface-type=fbo", "--deqp-gl-context-type=mgl",
                      "--deqp-visibility=hidden", "--deqp-surface-width=64",
                      "--deqp-surface-height=64", "--deqp-log-images=disable",
                      "--deqp-log-shader-sources=enable",
                      "--deqp-log-decompiled-spirv=enable",
                      "--deqp-log-empty-loginfo=disable",
                      "--deqp-log-flush=enable", "--deqp-base-seed=-1"] + extra,
                     cwd="/tmp", env=env, stdout=sys.stdout, stderr=subprocess.STDOUT)
print("rc=%d qpa=%s" % (rc, qpa))
