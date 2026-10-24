#!/bin/bash
# Interleaved native vs MGL on one scene (same machine state, alternating order).
WT=$1; SCENE=$2; R=${3:-4}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
LIB=$WT/build/nsgl/libmgl_nsgl.dylib
n() { /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$SCENE" 2>&1 | grep "glmark2 Score" | tr -dc '0-9'; }
m() { DYLD_INSERT_LIBRARIES=$LIB /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$SCENE" 2>&1 | grep "glmark2 Score" | tr -dc '0-9'; }
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then a=$(n); b=$(m); else b=$(m); a=$(n); fi
  echo "round$r $SCENE native=$a mgl=$b ratio_mgl_vs_native=$(python3 -c "print(f'{float('$b')/max(float('$a'),1):.3f}')")"
done
