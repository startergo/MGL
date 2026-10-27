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
# one glmark2 on the machine at a time: the whole check (launch -> capture ->
# kill) is one glmark2 session, so re-exec under the shared lock (supervisor
# note 01:18 EDT: overlapping glmark2 runs skew every luminance/FPS number).
if [ -z "${MGL_FLASH_LOCK:-}" ]; then
    MGL_FLASH_LOCK=1 exec /usr/bin/lockf -k /tmp/mgl_glmark2.lock /bin/bash "$0" "$@"
fi
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
for i in $(seq -w 1 150); do
    screencapture -x -m "$D/$TAG.b$i.png" 2>/dev/null
    # parallel-agent interference guard: keep the window frontmost — a covered
    # window is a flat 240 (score/info page or another agent's window on top)
    case $i in *0) osascript -e 'tell application "System Events" to set frontmost of process "glmark2-macos" to true' 2>/dev/null;; esac
done
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
frames = []                            # (mean, data) — keep the pixels: a flash must
for f in sorted(glob.glob(d + '/*.raw')):   # correlate with the scene it flashes OVER
    data = open(f, 'rb').read()
    frames.append((sum(data) / (W * H), data))
if len(frames) < 50:
    print(f"FAIL: only {len(frames)} usable captures (window not frontmost?)"); sys.exit(1)
head = frames[:120]                     # skip any end-of-run score page
vals = [v[0] for v in head]; vs = sorted(vals); med = vs[len(vs) // 2]
covered = sum(1 for v in vals if v > 230)
if covered > len(head) * 0.25:
    print(f"INVALID: {covered}/{len(head)} covered frames (>230) — the window was "
          "occluded during the run (another agent's window / score page); retry"); sys.exit(2)
# the jellyfish scene must be VISIBLE for the check to mean anything: its
# window-region mean is ~110-130; a flat ~240 is the info/score page or a
# covered window (vacuous pass guard)
if med > 170 or med < 40:
    print(f"INVALID: scene not visible (median {med:.1f}; window covered or "
          "info/score page) — bring glmark2 to front and retry"); sys.exit(2)

# A bright frame is only the present-path flash when it still shows the scene:
# the translucent frame composited over the white backdrop correlates with the
# neighbouring scene frame.  A flat bright frame that does NOT correlate is a
# foreign window that covered glmark2 for a moment (parallel agents; below the
# 25% INVALID threshold) — not a flash of ours.  Compare 255-x so the white
# backdrop subtracts out.
def corr(a, b):
    ma = sum(a) / len(a); mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a); db = sum((y - mb) ** 2 for y in b)
    return num / ((da * db) ** 0.5) if da > 0 and db > 0 else 0.0

flash, foreign = [], []
last_scene = None
for i, (m, data) in enumerate(head):
    # i<12: the scene's fade-in still brightens the first captures (the mean
    # crosses 131..115 on the way down) — not a present-path artifact
    if m > med + 15 and i >= 12:
        inv = [255 - x for x in data]
        if last_scene is not None and corr(inv, last_scene) > 0.35:
            flash.append(round(m, 1))
        elif m > 230:
            foreign.append(round(m, 1))   # covered by a foreign window: not our flash
        else:
            flash.append(round(m, 1))     # bright, structured, no reference: count it
    if m < 170:
        last_scene = [255 - x for x in data]
print(f"flash_check: n={len(head)} median={med:.1f} min={vs[0]:.1f} max={vs[-1]:.1f} "
      f"frames>med+15: {len(flash) + len(foreign)}"
      + (f" ({len(foreign)} foreign-window)" if foreign else ""))
if flash:
    print(f"FAIL: {len(flash)} flash frames: {flash[:10]}")
    sys.exit(1)
print("PASS: no flash frames (band median+/-, +15 margin)")
EOF
rc=$?
[ $rc -eq 0 ] && echo "flash_check: PASS" || echo "flash_check: FAIL"
exit $rc
