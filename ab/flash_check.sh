#!/bin/bash
# flash_check.sh — [jellyfish] present-path flash regression check (compat).
#
# The user-visible bug: intermittent whole-frame bright flashes ("like a
# camera flash") in glmark2 [jellyfish].  Root cause: the presented IOSurface
# (layer.contents) is sampled by the compositor asynchronously — the swap
# command buffer is not synchronized with it.  With the old present order
# (blit the frame, then force its alpha opaque) one display refresh could
# land between the two encoders and composite the translucent frame over the
# white window backdrop.  Fixed by forcing the copy SOURCE opaque before the
# blit (compatSurfacePresent), so the surface transitions old -> opaque-new
# in a single step.
#
# This check captures the glmark2 window ~150 times over ~15 s and verifies
# the mean luminance stays within a band: no frame more than +15 above the
# median (a flash is +17 or more; scene motion is a few units).
#
# usage: bash ab/flash_check.sh [libmgl_nsgl] (default: build/nsgl/...)
# env: MGL_PRESENT_PROBE_US=<us> to widen any post-blit window (debug aid —
# with the old order 50000 us makes the flash near-constant and the check
# fails immediately; with the fixed order the same probe must still pass).
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="${1:-$WT/build/nsgl/libmgl_nsgl.dylib}"
TAG="${2:-FLCHK}"
D=/tmp/mgl_flash_check
mkdir -p "$D" "$D/frames"
rm -f "$D/$TAG".b*.png "$D/frames"/*.raw

env DYLD_INSERT_LIBRARIES="$LIB" \
    /opt/homebrew/bin/glmark2-macos --data-path /opt/homebrew/share/glmark2 \
    -s 640x480 -b jellyfish:duration=40 > "$D/$TAG.log" 2>&1 &
pid=$!
sleep 5
osascript -e 'tell application "System Events" to set frontmost of process "glmark2-macos" to true' 2>/dev/null
sleep 1
# wait until the scene is visible (window-region mean < 170): the info page
# is ~240, the jellyfish scene ~110-130
for t in 1 2 3 4 5 6; do
    screencapture -x -m "$D/probe.png" 2>/dev/null
    ffmpeg -v error -i "$D/probe.png" -vf "crop=1280:1016:200:748,scale=80:63" \
        -pix_fmt gray -f rawvideo -y "$D/probe.raw" 2>/dev/null
    m=$(python3 -c "d=open('$D/probe.raw','rb').read(); print(round(sum(d)/len(d)))" 2>/dev/null || echo 999)
    [ "$m" -lt 170 ] && break
    osascript -e 'tell application "System Events" to set frontmost of process "glmark2-macos" to true' 2>/dev/null
    sleep 2
done
for i in $(seq -w 1 150); do screencapture -x -m "$D/$TAG.b$i.png" 2>/dev/null; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null

n=0
for f in "$D/$TAG".b*.png; do
    ffmpeg -v error -i "$f" -vf "crop=1280:1016:200:748,scale=80:63" \
        -pix_fmt gray -f rawvideo -y "$D/frames/$(basename "$f" .png).raw" && n=$((n+1))
done

python3 - "$n" "$D/frames" <<'EOF'
import sys, glob
n, d = int(sys.argv[1]), sys.argv[2]
W, H = 80, 63
vals = []
for f in sorted(glob.glob(d + '/*.raw')):
    data = open(f, 'rb').read()
    vals.append(sum(data) / (W * H))
if len(vals) < 50:
    print(f"FAIL: only {len(vals)} usable captures (window not frontmost?)"); sys.exit(1)
head = vals[:120]                     # skip any end-of-run score page
vs = sorted(head); med = vs[len(vs) // 2]
# the jellyfish scene must be VISIBLE for the check to mean anything: its
# window-region mean is ~110-130; a flat ~240 is the info/score page or a
# covered window (vacuous pass guard)
if med > 170 or med < 40:
    print(f"INVALID: scene not visible (median {med:.1f}; window covered or "
          "info/score page) — bring glmark2 to front and retry"); sys.exit(2)
flash = [round(v, 1) for v in head if v > med + 15]
print(f"flash_check: n={len(head)} median={med:.1f} min={vs[0]:.1f} max={vs[-1]:.1f} "
      f"frames>med+15: {len(flash)}")
if flash:
    print(f"FAIL: {len(flash)} flash frames: {flash[:10]}")
    sys.exit(1)
print("PASS: no flash frames (band median+/-, +15 margin)")
EOF
rc=$?
[ $rc -eq 0 ] && echo "flash_check: PASS" || echo "flash_check: FAIL"
exit $rc
