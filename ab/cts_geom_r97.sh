#!/bin/bash
# cts_geom_r97.sh — re-run the r96 known-fail set (14 PIQ + 9 geometry_shading)
# against one lib, with the per-family config used by the r96 comparison.
# usage: bash ab/cts_geom_r97.sh [libmgl.dylib]   (default build/libmgl.dylib)
set -u
LIB=${1:-$(pwd)/build/libmgl.dylib}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
ES31=$G/gl_cts/data/mustpass/gl/aosp_mustpass/main/gl45-es31-main.txt
mkdir -p /tmp/cts_r97
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null

cases="
dEQP-GL45-ES31.functional.program_interface_query.uniform.atomic_counter_buffer_index.default_block.array.var_array_array
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.sampler_2d
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.sampler_2d_array
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.sampler_3d
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.isampler_2d
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.image_2d
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.iimage_2d_array
dEQP-GL45-ES31.functional.program_interface_query.uniform.location.default_block.array.array.atomic_uint
dEQP-GL45-ES31.functional.program_interface_query.uniform.random.74
dEQP-GL45-ES31.functional.program_interface_query.program_input.type.interface_blocks.patch_in.block_array.array.mat3x2
dEQP-GL45-ES31.functional.program_interface_query.program_output.type.interface_blocks.patch_out.block_array.array.mat3x2
dEQP-GL45-ES31.functional.program_interface_query.shader_storage_block.buffer_data_size.named_block
dEQP-GL45-ES31.functional.program_interface_query.shader_storage_block.buffer_data_size.unnamed_block
dEQP-GL45-ES31.functional.program_interface_query.shader_storage_block.buffer_data_size.block_array
dEQP-GL45-ES31.functional.geometry_shading.basic.output_256
dEQP-GL45-ES31.functional.geometry_shading.basic.output_max
dEQP-GL45-ES31.functional.geometry_shading.basic.output_vary_by_attribute
dEQP-GL45-ES31.functional.geometry_shading.basic.output_vary_by_uniform
dEQP-GL45-ES31.functional.geometry_shading.basic.output_vary_by_texture
dEQP-GL45-ES31.functional.geometry_shading.basic.primitive_id_in_restarted
dEQP-GL45-ES31.functional.geometry_shading.instanced.invocation_output_vary_by_attribute
dEQP-GL45-ES31.functional.geometry_shading.instanced.invocation_output_vary_by_uniform
dEQP-GL45-ES31.functional.geometry_shading.instanced.invocation_output_vary_by_texture
"

for C in $cases; do
  while pgrep -x glcts >/dev/null 2>&1; do sleep 20; done
  Q=/tmp/cts_r97/$(echo "$C" | sed 's/.*functional\.//; s/\./_/g').qpa
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 600 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-case="$C" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 --deqp-base-seed=-1 \
    --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$Q" > /tmp/cts_r97/out 2>/tmp/cts_r97/err
  rc=$?
  r=$(grep -o 'StatusCode="[A-Za-z]*"' "$Q" | tail -1 | sed 's/StatusCode="//; s/"//')
  printf '%-28s rc=%-3d %s\n' "$(echo "$C" | sed 's/.*functional\.//')" "$rc" "${r:-none}"
done
