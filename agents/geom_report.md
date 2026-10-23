READY 56bf738 (texture_gather 38 -> 71/75 Pass; suite 438 PASS lines, test_geom 593/593; limits cluster triaged — root cause + plan below)

# geom domain — round 38 handoff (KHR-GL46.texture_gather, KHR-GL46.limits triage)

Round 38 (this session). Round 37 was merged as 7f78e8c; `git merge gl46-port` done
(3a1ad9c). Commits this round (all on gl46/geom): bde0539, 283daae, ff85ae7, 56bf738.

## texture_gather: 38/75 -> 71/75 Pass  (chunks: geom38k1 = 71, geom38h1 = 71)

### What was implemented (Metal mechanism)

1. **Vertex-stage shadow (depth) gather via texel reads + compare-op function constant**
   (program.c `mgl_fix_gathers_vs`, msl pass).  AGX vertex functions return RAW values
   for `sample_compare` (the sampler's compare function is ignored), and sampling a
   depth texture through a plain `texture2d` with a compare sampler is UNRELIABLE there
   (flaky 0s/1s).  So `.gather_compare(s,xy,ref[,off])` in VERTEX MSL becomes
   `mgl_gcmp_vs()`: four `depth2d[_array].read()` taps of the 2x2 block + manual
   `ref OP d`.  OP (GL_TEXTURE_COMPARE_FUNC of the unit feeding the texture slot)
   rides in as **function constant 32010+slot** (`mgl_shadow_op_N`, declared with
   `is_function_constant_defined()` so unspecialized compiles default to LEQUAL);
   function constants work at any scope — the CTS gathers from a NESTED function.
   `rstateSpecialize` fills them (`geomShadowFCValues:stage:hash:`,
   MGLRenderer+Geometry.m) and mixes a checksum into its cache key; a per-draw
   staleness check (`geomShadowCheckStale:`, called from bindTextures) dirties
   DIRTY_PROGRAM when the compare func changes so the PSO re-specializes.
   `spirv[VS].shadow_slots` (new field in glm_context.h) records which slots emulated.
2. **Plain (non-shadow) gathers in vertex AND compute** — the hardware `.gather` also
   returns zeros in non-fragment functions: rewritten to `mgl_gath_vs()` = four texel
   reads, channel-selectable (the MSL `component::x/y/z/w` argument is passed through
   `mgl_gsel`).  float/int/uint channels, 2D/2D-array/RECT.  Compute shadow gathers
   keep hardware `sample_compare` taps (it works there), now with the rect flag.
3. **Rectangle textures**: SPIRV-Cross's MSL backend has NO DimRect mapping (it emits
   `unknown_texture_type`, which never compiles — sampler2DRect was broken everywhere).
   `mgl_spirv_rect_to_2d` (program.c) rewrites the MSL-facing module only: OpTypeImage
   Dim Rect->2D, and every sampled-image variable on such a type gets its OpName
   extended with `_mglrect` (module rebuilt word-by-word) so the gather rewrites can
   pick the unnormalized tap base `floor(uv-0.5)` vs `floor(uv*size-0.5)`.  The GL
   view (`spirv[stage].ir`) keeps DimRect, so introspection/unit lookup still see
   GL_TEXTURE_RECTANGLE; the unnormalized MTLSamplerState comes from the texture
   target as before.
4. **textureGatherOffsets (ConstOffsets)**: SPIRV-Cross DROPS the ConstOffsets image
   operand (Metal needs compile-time-constant gather offsets).  GLSL-level rewrite in
   cts_glsl_compat.c (`rewrite_gather_offsets`): expands `textureGatherOffsets` into
   four `textureGatherOffset(..., OFFi, ...)` calls on channel `.w` (= the texel at
   base+offset[i] per ARB_texture_gather), INLINING the const ivec2[4] initializer
   elements so the offsets survive to MSL.  Handles both argument orders
   ((S,P,offs[,comp]) and shadow (S,P,refZ,offs)) and picks vec4/ivec4/uvec4 from the
   sampler's declared channel type.  Also fixed the chain in mgl_cts_glsl_compat
   (a NULL from an earlier step used to skip every later rewrite).
5. **Tess stages**: the same gather rewrite now runs for _TESS_CONTROL_SHADER and
   _TESS_EVALUATION_SHADER (post-tess vertex functions — gather broken there too).
6. **Crash fixes found while chasing a flaky glcts crash**:
   - program.c gather-call emission: the accumulating `snprintf` offset was the
     UNTRUNCATED length — a long argument list could push the next snprintf past the
     stack buffer (clamped now).
   - texextRefreshStage (MGLRenderer+TextureExt.m) cast a TextureParameter's
     `mtl_data` straight to MTLSamplerState, but `texparamCacheFor` replaces it with
     an MGLTexParamCache object -> type confusion (EXC_BAD_ACCESS in
     setVertexSamplerState, or heap corruption detected later in the link path).
     New `texparamCachedSampler:` (MGLRenderer+TextureParams.m) unwraps it.
   - texextRefreshStage bound at the GL *unit* index instead of the program's Metal
     texture/sampler slot (diverge once units != slots; sampler index >= 16 is OOR).

### Remaining 4 texture_gather failures
- `plain-gather-depth-cube`, `-cube-array`: cube shadow gathers need the cube-face
  projection emulated with `depthcube.read(uint2, face, level)` + the tap-quad order
  per face — not attempted (documented gap; feasible next extension of k_shadow_helpers).
- `base-level`: FS/hardware gather honors the base-level view, but VERTEX-stage
  sampling of a level-range MTLTexture view returns level-0 data in this VM (VM quirk;
  verified with textureLod in VS vs FS).  Would need the view's base level as another
  function constant + absolute-level reads — texture-domain territory mostly.
- `gather-tesselation-shader`: TCS/TES gather MSL is rewritten correctly (verified in
  the dump), but the tess execution still returns a wrong component; needs a tess
  pipeline debug scratch (TCS v2/v3 in, TES ecol out).
- `incomplete-texture`, `incomplete-texture-last-comp`: GL texture-completeness
  enforcement at gather/draw time (incomplete => black) — texture-domain policy,
  not implemented.
- **FLAKY batch crash**: full-cluster runs (~1 in 3) abort mid-batch with heap
  corruption detected later in a link (stacks: `_xzm_xzone_malloc_freelist` in
  SPIRV-Cross's parse of `prepared`, or in strdup in compile_stage).  NOT reproduced
  in isolation: the single case, `--limit 12/15/16` (9 runs, MallocErrorAbort) and
  300-cycle link/draw scratches (2D and RECT) are all clean.  The texparam-cache
  cast above was one real cause and is fixed; if it persists, next step is
  DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib over a --limit window (note:
  guard malloc is ~20x slower, batch needs splitting) or ASan on libmgl.

## KHR-GL46.limits: 123/140 Pass, 17 Fail (geom38lim)

Root cause (max_vertex_attribs shown in the log; the others are the same class):
"Shader builtin has value: 64 which is different then the value of corresponding
limit: 16" — the limits state is seeded from **macOS CGL via getMacOSDefaults()**
(glm_params.c ~line 150-175: glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, ...) etc. = legacy
GL 16/32/8...), while glslang's builtin constants say 64/32/8/1024/....  The CTS
compares the query against the shader builtin, so both must agree.

Plan for next round:
1. List glslang's TBuiltInResource values actually used (glslang/StandAlone/
   ResourceLimits.cpp defaults: gl_MaxVertexAttribs 64, gl_MaxTextureImageUnits 32,
   gl_MaxVertexTextureImageUnits 32, gl_MaxFragmentTextureImageUnits 32,
   gl_MaxDrawBuffers 8, gl_MaxFragmentUniformVectors 1024, gl_MaxVertexUniformVectors
   4096, gl_MaxVaryingVectors 60, gl_MaxComputeWorkGroupSize 65535/65535/65535+1(?)...).
2. Override the MGL state after getMacOSDefaults (glm_params.c or glm_context.c) with
   those values IF MGL really supports them (Metal: 31 vertex attributes, 31 textures,
   31 buffers, 8 color attachments; work-group sizes from the device), and make get.c
   report those (many already LIMIT()ed).
3. Verify each failing limit against the actual list: max_vertex_attribs,
   max_texture_image_units, max_vertex_texture_image_units, max_fragment_uniform_*,
   max_varying_*, max_draw_buffers, max_uniform_buffer_bindings, max_combined_*,
   max_*_atomic_counter_buffers/counters, max_compute_work_group_size,
   max_geometry_image_uniforms, max_transform_feedback_interleaved_components,
   max_shader_storage_buffer_bindings (geom38lim results.json has the exact 17).

## Shared-file edits this round (surgical; all listed)
- MGL/src/program.c (mine anyway): gather/rect MSL passes + call-site.
- MGL/src/cts_glsl_compat.c (GLSL compat layer, shared with ssbo): textureGatherOffsets
  rewrite + chain fix.
- MGL/include/glm_context.h: Spirv.shadow_slots field.
- MGL/src/MGLRenderer.m: vertex-stage staleness hook in bindTexturesToCurrentRenderEncoder.
- MGL/src/MGLRenderer+RenderState.m: rstateSpecialize fills shadow FCs + key, and
  specializes shadow MSL without rstate patch markers.
- MGL/src/MGLRenderer_Private.h: 3 method decls + inline slot check.
- MGL/src/MGLRenderer+TextureParams.m: texparamCachedSampler: (cache unwrap).
- MGL/src/MGLRenderer+TextureExt.m: texextRefreshStage crash fixes (cast + Metal slot).

## Test status
- tests/headless/run.sh: 58/58 tests, 438 PASS lines, 0 failures.
- tests/headless/run.sh test_geom: 593 checks, 0 failures.
- CTS: texture_gather 71/75 (geom38k1; single-case reruns of all 71 pass; the 4 fails
  documented above), limits 123/140 (geom38lim).
- Scratch tests used (NOT committed): scratch_vdg*.c, scratch_vda.c, scratch_vdr*.c,
  scratch_vgo.c, scratch_vbl*.c, scratch_mip*.c, scratch_mipup.c, scratch_lnk*.c,
  scratch_vdc.c — keep scratch_* out of commits (USER RULE).

## Rules honored
make -j4 only; one CTS batch at a time in the foreground (--max-seconds <= 480);
no background test processes left (checked `ps`); no git stash; no `git add -A` at
the repo root (explicit paths); scratch/_scratch files and .agent/ not committed;
no absolute paths or secrets in commits (check_no_secrets hook clean).
