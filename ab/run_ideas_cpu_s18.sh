#!/bin/bash
# run_ideas_cpu_s18.sh — per-draw CPU on [ideas] (supervisor task 2): C-new tip
# (s17 per-draw commits + s18 present) vs gl46-port 5178835 (no s17 commits).
# FPS is present/GPU bound; CPU per frame is the decision metric.
WT="$(cd "$(dirname "$0")/.." && pwd)"
NEW=$WT/build/nsgl/libmgl_nsgl.dylib
OLD=$WT/ab/gl46port/build/nsgl/libmgl_nsgl.dylib
DATA=/opt/homebrew/share/glmark2
for r in 1 2; do
  for who in new old; do
    L=$([ $who = new ] && echo $NEW || echo $OLD)
    MGL_PROFILE=1 DYLD_INSERT_LIBRARIES="$L" /opt/homebrew/bin/glmark2-macos \
        --data-path $DATA -b ideas:duration=10 2>&1 \
        | grep -o "draw=[0-9.]*|[0-9.]*|[0-9.]*" | tail -5 > /tmp/mgl_ideas_$who.$r
    /usr/bin/time -p env DYLD_INSERT_LIBRARIES="$L" /opt/homebrew/bin/glmark2-macos \
        --data-path $DATA -b ideas:duration=10 > /dev/null 2> /tmp/mgl_ideas_time_$who.$r
    u=$(grep user /tmp/mgl_ideas_time_$who.$r)
    s=$(grep sys /tmp/mgl_ideas_time_$who.$r)
    echo "round$r $who $u $s load=$(sysctl -n vm.loadavg | cut -c9-13)"
  done
done
python3 - <<'EOF'
for who in ("new", "old"):
    tot = n = 0
    for r in (1, 2):
        for line in open(f"/tmp/mgl_ideas_{who}.{r}"):
            p = line.strip().split("=")[1].split("|")
            tot += float(p[1]); n += 1
    print(f"{who}: draw-bucket {tot/n:.4f} ms/frame (n={n})")
EOF
