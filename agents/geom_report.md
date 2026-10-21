READY d638595 (geometry_shader 117/135 under the 480 s cap; rendering 33/33; limits 9/9; headless green)

# geom domain — working handoff (round 29)

Branch gl46/geom at d638595.  All work committed; build clean; tests/headless/run.sh
fully green (56 test binaries).  CTS note: run everything under the same
--max-seconds cap when comparing numbers - the round-28 "119/136" was measured with a
bigger budget; under the 480 s cap the pristine ec487f8 build scores 110/135 and this
branch 117/135 (the delta is exactly the seven rendering cases, no other case moved;
see the base29full vs geom29a/geom29b results dirs under the MGL CTS checkout).

## Session results (commits ec487f8..d638595)

* rendering 33/33 (9c45dc0).  Three bugs behind the 7 strip/loop failures:
  - The geometry kernel appended each thread's emitted strips to the raster index list
    through an atomicAdd reservation, so the strips landed in whatever order the racing
    atomics returned (GL fixes the rasterization order as the emission order).  Each
    thread now writes into a per-thread scratch segment (mgl_lscr, 3*max_vertices words
    per thread, Metal slot 30 - the last free one, Apple GPUs expose 31 buffer slots);
    new kernels mgl_geom_lzero/lscan/lcopy (kGeomCtlMSL) zero the per-thread counts,
    exclusive-scan them into bases and copy the segments into the raster list in
    primitive order.  Layout of the grown gIdx: [ raster list | scratch | counts+bases ].
    The lzero pass matters: an indirect draw's exact group count is smaller than the
    worst-case bound, so threads beyond the grid never ran to publish a zero - that
    broke gs_output_block_after_tessellation until the zero-fill ran first.
  - The capture index list and the primitive slots overlapped for non-indexed
    GPU-assembly draws (the slots were bound at HDR + index_words = HDR + 0, which is
    where the capture list lives), and the capture list capacity was sized by the worst
    primitive count although the prep kernel writes one entry per vertex.  The geometry
    kernel now binds the slots at HDR + cap_idx like the prep kernel / tess import do,
    and cap_idx is the drawn vertex-count bound.
  - geomCanAssembleGPU returned false with no element-array buffer bound, so every
    non-indexed draw took the CPU assembly path - where geom_assemble dropped the
    closing segment of a GL_LINE_LOOP (loop stopped at i = n - 2).  Fixed both.
  - test_gsrender.c gained the CTS triangles+triangle_strip replica (arrays and the
    reversed-array/reversed-index elements variant) and a line-loop corner check that
    covers the arrays, elements and odd-offset (CPU assembly) paths.
* primitive counting (d638595): ctl[13+stream] (PRIMITIVES_GENERATED /
  GEOMETRY_SHADER_PRIMITIVES_EMITTED) counted EndPrimitive calls; a points output
  emits every vertex as its own primitive, so it now increments per vertex for points
  and per closed non-empty strip otherwise.  TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN for
  the geometry path was ctl[13] verbatim; the kernel flags the last record of each
  emitted primitive (bit 8 of the record's stream word; the low 24 bits hold the
  per-invocation vertex index which stays < 256) and the xfb scatter / CPU capture count
  a primitive when its final vertex's writes all landed (the per-variable fill test is
  monotonic in the write offset).  primitive_queries.points now reads the exact written
  count (8); its remaining failure is PRIMITIVES_GENERATED.

## Remaining failures under the 480 s cap (18) and where to look

* primitive_queries 3 + primitive_counter: MGL/src/queries.c queries_pre_draw feeds
  GEOMETRY_SHADER_PRIMITIVES_EMITTED (and, via the same path, PRIMITIVES_GENERATED)
  with the INPUT primitive count when a GS is active; the exact per-stream output count
  is only known on the GPU (ctl[13+s], per-draw buffer).  Design sketch: keep a
  persistent shared accumulator gQacc [reset bitfield | epoch | gen0..3 | emit0..3];
  dispatch a one-thread kernel after the GS kernel that adds ctl[13+s] into it (or
  stores, when the reset bit for that word is set, then clears the bit); queries.c
  begin_query sets the reset bit, end_query waits for the last command buffer that ran
  the kernel and reads the word (bit still set => no draw ran => 0).  Needs ~3 surgical
  edits in queries.c (begin / pre_draw skip / end) + hooks in mgl_ext_geom.h and
  MGLRenderer+Geometry.m.  CLIPPING_INPUT_PRIMITIVES is fed the input count too.
* api.getProgramiv3 (linker-visible GEOMETRY_VERTICES_OUT etc. on a non-linked
  program), api.max_shader_storage_blocks, api.max_atomic_counters/buffers (limits;
  max_image_uniforms NotSupported), linking more_ACs/ACBs (2): see the round-28 notes.
* layered_fbo 2, layered_framebuffer 1, layered_rendering_fbo_no_attachment 1: layered
  framebuffer validation / no-attachment rendering with gl_Layer.
* nonarray_input: glslang rejects the CTS's non-arrayed GS inputs (needs a source
  rewrite before glslang or an accept-list), see round-22 notes in this file.
* program_resource, api.program_pipeline_vs_gs_capture: program introspection of the
  capture variant; "vertex capture variant Metal compile failed: cannot reserve
  'buffer' resource location at index 3" for program_resource - the capture shader's
  own SSBOs plus the pinned MGL_GEOM_MSL_VSOUT(25) collide; check geom_build_pins'
  span rule for the vertex stage.

# geom domain — working handoff (round 28)

Branch gl46/geom at 769ef33 (merged gl46-port 0e1baff at 31370c4; .ctslink dropped).
All work committed; build clean; tests/headless/run.sh fully green.

## Session results (commits 99943f1..769ef33)

* GS default-uniform block fixes (99943f1, 119a045, c782461): spv_parse word counts
  (4 call sites divided by 4 => modules truncated to a quarter, DUB member offsets
  never seen), struct-vs-variable id in geom_dub_layout, array declarators in
  geom_strip_plain_uniforms (suffix kept, no keep-reset), preprocessor lines after
  the whitespace skip + leading-directive hoist + comment blanking + qualifier strip
  + opaque types rejected => geometry_shader.limits 9/9.
* init-kernel slot-0 shadowing (99943f1): gCtl init bound buffer(0) after the GS
  kernel's stage buffers; compute bindings persist across setComputePipelineState.
* ARB entry-point aliases (3e7ee0d, shared file gl_core.c): glwInitExtGL.inl
  overwrites core pointers with loader->get("gl...ARB") for advertised extensions;
  MGL advertised ARB_draw_instanced / ARB_instanced_arrays / ARB_gl_spirv without
  exporting the aliases => every CTS instanced draw called NULL. Added thin aliases;
  this alone took rendering.rendering from 0/33 to 23/33.
* GPU assembly GL_LINES (769ef33): prep kernel grouped GL_LINES with the stride-1
  line-strip modes; now p*k+v like the other disjoint modes.

## geometry_shader status: 119/136 pass

rendering.rendering 26/33; limits 9/9; linking 9/9; primitive_queries 9/9; the rest
of the groups as before (api 6 fails incl. 2 NotSupported atomic counters, layered_*
4, nonarray_input 1, primitive_counter 1, program_resource 1).

## Remaining rendering failures (7) and where to look

lines_input_*_line_loop_drawcall (3), triangles_input_*_triangle_strip_drawcall (2),
triangles_with_adjacency_*_triangle_strip_adjacency_drawcall (2).  All are draw modes
that need index/prim rewriting; a scratch (tests deleted at commit time, recreate from
this description) showed: triangles-GS + GL_TRIANGLE_STRIP lit=99 but expected ~190
(the 2nd strip primitive's vertices wrong), GL_TRIANGLE_FAN lit=73, lines-GS +
GL_LINE_LOOP lit=58 for a 4-line loop.  Suspect the capture pass / prep assembly for
strip/fan/loop draw modes feeding gl_in[] wrong vertices for primitives after the
first (the CPU assembly geom_assemble in geometry.c has per-mode cases - check mode
GL_TRIANGLE_STRIP=5, GL_TRIANGLE_FAN=6, GL_LINE_LOOP=2 map to the same table as the
prep MSL switch, whose case numbers are the GL enums).  NOTE: geom_mode_matches
already accepts strips/fans for triangles input, so the INVALID_OPERATION seen in an
earlier scratch was a mangled test shader, not a driver bug.

## Next steps for a fresh session

1. Fix the strip/fan/loop capture+assembly (above) -> rendering should reach 33/33.
2. Re-run the other groups: api.getProgramiv3, layered_fbo 2, layered_framebuffer 1,
   layered_rendering_fbo_no_attachment 1, nonarray_input, primitive_counter,
   program_resource (see ~/MGL-cts/gsr-results/results.json).
3. Watch for the stale-dylib race: `make -j4 lib` sometimes says "Nothing to be done"
   while an object was rebuilt in the same second - re-run make or check
   `strings build/libmgl.dylib | grep <new symbol>` before testing.
4. CTS runs: python3 agents/conformance/run.py --pattern ... --batch 12
   --per-case-timeout 90 --max-seconds 480 --results-dir ~/MGL-cts/gsr-results
   [--refresh-caselist when changing pattern] - foreground only, one batch at a time.
   one.sh truncates stderr to 800 chars; run glcts directly for traces.

---
(older rounds below; the DUB/array work from round 27 is all merged and committed)

# geom domain — working handoff (round 27, checkpoint)

Session context: branch gl46/geom; everything through commit a376bfb (adjacency table
10.4, SEPARATE_ATTRIBS xfb, GS/VS stage texture arrays + pre-realize, source-transform
#define/block fixes, MAX_GEOMETRY_UNIFORM_BLOCKS capped at 14) is committed and green
(full headless suite 5344 checks / 0 failures, CTS geometry_shader 90+/136, adjacency
8/8, limits 7/9).  The uncommitted delta below is the uniform-block-ARRAY work.

## What I am fixing right now (uncommitted checkpoint)

Uniform-block instance arrays (`layout(binding=0) uniform UB { int entry; }
ub[N];`) in the GEOMETRY kernel: `array<UB,N>` in the generated MSL spans N consecutive
buffer argument slots, but the kernel pinned ONE slot per resource and the bind walk
bound one buffer per resource.  Implemented in the working tree:

* geometry.c `geom_build_pins`: per-type counters advance by the array size
  (spv_value_type), UBO span (`ub_span`) becomes the base for UC/DUB/SB flats, returns
  the total slot span; both callers (capture kernel + GS kernel) fail the LINK with a
  clear info-log message when the span exceeds MGL_GEOM_KERNEL_USER_SLOTS (22, new in
  mgl_ext_geom.h — the kernel's own SSBOs are pinned at fixed slots 22..29).
* MGLRenderer.m `mapGLBuffersToMTLBufferMap` (shared): a UB/SB resource with
  array_size>1 now appends one list entry PER ELEMENT (binding base..base+N-1, offset
  per element), so every stage's enumeration matches the expanded argument slots
  (FS/VS UBO arrays were silently broken before, same as textures were).

Results: `layout(binding=0) uniform UB { int entry; } ub[8]` all-1.0 works; CTS
limits.max_input_components/combined/max_texture_units now PASS (7/9 limits pass).

## The open bug (next step)

`tests/headless/test_gsubo.c` (NOT RUN, main() disabled): NB=14 blocks with DISTINCT
values (i+1, sum must be 105 — same shape as KHR-GL46.geometry_shader.limits.
max_uniform_blocks, which fails "Expected: 105 Extracted: 104").  The kernel's sum
reads 104: exactly ONE element contributes 0 — almost certainly element 0 (value 1) or
the last.  CTS cluster: limits.max_uniform_blocks + max_uniform_components (the latter
still Expected 524800 / Extracted 0: a default-uniform array of ~131200 vec4s — likely
DUB-size or glslang scale; separate issue).

Debug plan:
1. Re-enable RUN(GsUBOArray) in tests/headless/test_gsubo.c; make the GS emit one
   vertex per element with ub[i].c.x as g_s and read all 14 back (which element is 0?).
2. Suspects: (a) the DUB entry — mapGLBuffers puts the DUB (UNIFORM_CONSTANT) AFTER the
   UBO group, but geom_build_pins pins DUB at ub_span (=14) while the bind list has it
   at index 14 ✓ should match; check with MGL_GEOM_DUMP=/tmp/x + MGL_DUMP_MSL=1 that
   the kernel MSL really has array<UB,14> [[buffer(0)]] and DUB [[buffer(14)]];
   (b) std140 stride: MGL array<UB,14> element stride must be 16 (int padded) — check
   the MSL struct; (c) the FIRST slot colliding with something bound earlier on the
   same encoder (the prep kernel? gCtl at MSL 29...).
3. After the fix: re-enable the test, run CTS geometry_shader.limits (want 9/9),
   then the whole group + full suite, commit.

## CTS snapshot (current tree)

geometry_shader 90+ Pass / 136 total; biggest remaining clusters:
rendering.* (~30 pixel-diff failures), linking.more_ACs_in_GS_than_supported (must FAIL
link when atomic counters/blocks exceed limits), primitive_queries (3), api singles
(program_pipeline_vs_gs_capture, fb_texture_invalid_level_number, getProgramiv3).

## Rules reminder (supervisor)

