# tess domain — round 18 final report

**READY** (sha at the bottom of this file — the last commit on `gl46/tess`).

Branch `gl46/tess`; gl46-port merges included: **4b71da1** (tess r17), **beab36b** (compat,
vsync-pin fix + NULL-program guard), **a2f16f4** (geom r46, image-atomic sidecars), clean
rebuild after each.

## 1. Geometry-shader atomic counters (the 2 NotSupported cases) — done

`KHR-GL46.geometry_shader.api.max_atomic_counters` and `.max_atomic_counter_buffers` were
the last two NotSupported cases of the group. Both **Pass** now;
`KHR-GL46.geometry_shader` is **136/136 (100%)**.

Mechanism (all on the existing geometry emulation; no new Metal features):

* `MGL/src/get.c` (shared, surgical): `GL_MAX_GEOMETRY_ATOMIC_COUNTERS` reports
  `LIMIT(state.var, 1024)`, `GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS` reports
  `MGL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS` (8) — like the vertex/fragment/compute stages
  (glslang's per-stage limits are seeded from the same queries, so `gl_MaxGeometryAtomic…`,
  the CTS `limits` group and the linking limit checks stay consistent).
* `MGL/src/geometry.c`: `mgl_geom_compile_stage()` no longer fails the link when the GS
  declares atomic counters. It enumerates the declarations (one SPIR-V AtomicCounter
  variable per GLSL declaration; `atomic_uint acs[N]` is ONE variable of OpTypeArray =
  N counters in one buffer) and enforces the GL link errors when the counters / buffers
  exceed the reported limits (`linking.more_ACs_in_GS[_buffers]_than_supported` still
  fails the link correctly). `geom_build()` injects the
  `volatile device atomic_uint*` arguments into the kernel MSL with the same pass the
  standard stages use (`mgl_msl_inject_atomic_counters`), placed **above the kernel's own
  `[[buffer(n)]]` user arguments** (scanned from the entry signature), and records the
  Metal indices (`gp->ac_msl[]`).
* `MGL/src/MGLRenderer+Geometry.m`: `geomBindStageBuffers` skips the buffer map's own
  atomic-counter entries (they expand an array declaration into one entry per ELEMENT —
  the kernel has ONE pointer argument per declaration — and the expansion can run past
  the user slot range into the emulation's pinned pool buffers) and binds the GL
  atomic-counter buffers itself at the recorded indices.
* **Vertex-stage ACs with a GS**: the capture variant is the only vertex function of a
  geometry program, so its MSL gets the injected arguments too (above its own args;
  `geomCapturePass` binds them) — fixes
  `KHR-GL46.geometry_shader.program_resource.program_resource`.
* **XFB capture variant** (`MGL/src/transform_feedback.c`): the capture MSL (a recompile
  of the VS) now gets the injected atomic_uint arguments at the same Metal indices
  program.c refined into the stage resource list (`mgl_geom_stage_ac_arg_base()`, exported
  from geometry.c) — fixes the pre-existing
  `KHR-GL46.shader_atomic_counters.basic-usage-vs` fail.

Regression guard: `tests/headless/test_geom.c`'s `gs_atomic_counter_link_fails` asserted
the old "link must fail" behavior; it now verifies the link SUCCEEDS, the counter counts
every invocation (4 points -> 4), and 9 atomic counter buffers still fail the link.

Known gap in this area: `KHR-GL46.shader_atomic_counters.advanced-usage-multi-stage`
(pre-existing fail, unchanged): with transform feedback active AND rasterization enabled
the VS runs twice (once in the XFB capture pass, once for the raster draw), so VS-side
atomic side effects double and the final-counter check (`init + vertices + fragments`)
fails. Needs the capture+replay raster path for non-geometry programs.

## 2. Full KHR-GL46 mustpass sweep (gl46-master, 9035 cases)

One glcts at a time, sequential batches of 64, per-case timeout 20 s, on the merged tree
(all three merges above + the geometry-AC work). Results:
`~/MGL-cts/mgl-r18-sweep/results.json` (sweep table below).

**Pass 8364/9035 = 92.6 %** (97.1 % excluding the 418 NotSupported).

| group | cases | pass | non-pass | rate |
|---|---|---|---|---|
| KHR-GL46.CommonBugs | 8 | 6 | 2F | 75% |
| KHR-GL46.aggressive_optimizations | 12 | 12 | – | 100% |
| KHR-GL46.blend_equation_advanced | 88 | 0 | 88NS | 0% |
| KHR-GL46.buffer_storage | 9 | 7 | 2F | 78% |
| KHR-GL46.clip_control | 8 | 8 | – | 100% |
| KHR-GL46.clip_control_ARB | 8 | 8 | – | 100% |
| KHR-GL46.clip_distance | 3 | 3 | – | 100% |
| KHR-GL46.compute_shader | 42 | 42 | – | 100% |
| KHR-GL46.conditional_render_inverted | 2 | 2 | – | 100% |
| KHR-GL46.constant_expressions | 1392 | 1392 | – | 100% |
| KHR-GL46.copy_image | 13 | 7 | 6F | 54% |
| KHR-GL46.cull_distance | 3 | 1 | 2F | 33% |
| KHR-GL46.direct_state_access | 371 | 370 | 1IE | 100% |
| KHR-GL46.draw_elements_base_vertex_tests | 12 | 10 | 2NS | 83% |
| KHR-GL46.draw_indirect | 70 | 68 | 2F | 97% |
| KHR-GL46.draw_indirect_43 | 2 | 2 | – | 100% |
| KHR-GL46.explicit_uniform_location | 48 | 32 | 16F | 67% |
| KHR-GL46.exposed_extensions | 1 | 1 | – | 100% |
| KHR-GL46.ext_texture_shadow_lod | 16 | 0 | 16NS | 0% |
| KHR-GL46.geometry_shader | 136 | 136 | – | 100% |
| KHR-GL46.get_texture_sub_image | 2 | 2 | – | 100% |
| KHR-GL46.gl_spirv | 9 | 4 | 5F | 44% |
| KHR-GL46.glsl_noperspective | 1 | 1 | – | 100% |
| KHR-GL46.gpu_shader5 | 22 | 11 | 11F | 50% |
| KHR-GL46.gpu_shader5_gl | 3 | 0 | 3NS | 0% |
| KHR-GL46.gpu_shader_fp64 | 659 | 658 | 1F | 100% |
| KHR-GL46.incomplete_texture_access | 1 | 0 | 1F | 0% |
| KHR-GL46.indirect_parameters_tests | 3 | 2 | 1X | 67% |
| KHR-GL46.info | 6 | 6 | – | 100% |
| KHR-GL46.internalformat | 52 | 25 | 27NS | 48% |
| KHR-GL46.layout_binding | 21 | 9 | 12F | 43% |
| KHR-GL46.layout_location | 35 | 35 | – | 100% |
| KHR-GL46.limits | 140 | 140 | – | 100% |
| KHR-GL46.map_buffer_alignment | 2 | 2 | – | 100% |
| KHR-GL46.multi_bind | 17 | 10 | 7F | 59% |
| KHR-GL46.nearest_edge | 2 | 2 | – | 100% |
| KHR-GL46.packed_depth_stencil | 25 | 21 | 4F | 84% |
| KHR-GL46.packed_pixels | 250 | 226 | 24F | 90% |
| KHR-GL46.parallel_shader_compile | 3 | 0 | 3NS | 0% |
| KHR-GL46.pipeline_statistics_query_tests_ARB | 10 | 6 | 4F | 60% |
| KHR-GL46.pixelstoragemodes | 896 | 736 | 160NS | 82% |
| KHR-GL46.polygon_offset_clamp | 3 | 3 | – | 100% |
| KHR-GL46.post_depth_coverage_tests | 2 | 0 | 2NS | 0% |
| KHR-GL46.program_interface_query | 43 | 30 | 13F | 70% |
| KHR-GL46.sample_shading | 13 | 5 | 8F | 38% |
| KHR-GL46.sample_variables | 190 | 152 | 38NS | 80% |
| KHR-GL46.separable_programs_tf | 2 | 2 | – | 100% |
| KHR-GL46.sepshaderobjs | 8 | 5 | 3F | 62% |
| KHR-GL46.shader_atomic_counter_ops_tests | 4 | 3 | 1F | 75% |
| KHR-GL46.shader_atomic_counters | 30 | 28 | 2F | 93% |
| KHR-GL46.shader_ballot_tests | 4 | 0 | 4NS | 0% |
| KHR-GL46.shader_bitfield_operation | 324 | 324 | – | 100% |
| KHR-GL46.shader_draw_parameters_tests | 13 | 0 | 12F 1NS | 0% |
| KHR-GL46.shader_group_vote | 5 | 0 | 5F | 0% |
| KHR-GL46.shader_image_load_store | 51 | 38 | 13F | 75% |
| KHR-GL46.shader_image_size | 74 | 74 | – | 100% |
| KHR-GL46.shader_multisample_interpolation | 109 | 97 | 12F | 89% |
| KHR-GL46.shader_storage_buffer_object | 124 | 119 | 5F | 96% |
| KHR-GL46.shader_subroutine | 28 | 1 | 27NS | 4% |
| KHR-GL46.shader_texture_image_samples_tests | 4 | 0 | 4NS | 0% |
| KHR-GL46.shader_viewport_layer_array | 3 | 0 | 3NS | 0% |
| KHR-GL46.shaders | 2382 | 2345 | 29F 2T 6NS | 98% |
| KHR-GL46.shaders30 | 650 | 650 | – | 100% |
| KHR-GL46.shaders42 | 2 | 2 | – | 100% |
| KHR-GL46.shaders44 | 12 | 12 | – | 100% |
| KHR-GL46.shaders45 | 7 | 1 | 1F 5NS | 14% |
| KHR-GL46.shading_language_420pack | 43 | 29 | 13F 1X | 67% |
| KHR-GL46.sparse_buffer_tests | 3 | 0 | 3NS | 0% |
| KHR-GL46.sparse_texture2_tests | 6 | 0 | 6NS | 0% |
| KHR-GL46.sparse_texture_clamp_tests | 3 | 0 | 3NS | 0% |
| KHR-GL46.sparse_texture_tests | 6 | 0 | 6NS | 0% |
| KHR-GL46.spirv_extensions | 1 | 1 | – | 100% |
| KHR-GL46.stencil_texturing | 1 | 0 | 1F | 0% |
| KHR-GL46.sync | 1 | 1 | – | 100% |
| KHR-GL46.tessellation_shader | 58 | 57 | 1T | 98% |
| KHR-GL46.texture_barrier | 4 | 4 | – | 100% |
| KHR-GL46.texture_barrier_ARB | 4 | 0 | 4NS | 0% |
| KHR-GL46.texture_border_clamp | 32 | 26 | 6F | 81% |
| KHR-GL46.texture_buffer | 16 | 14 | 2F | 88% |
| KHR-GL46.texture_cube_map_array | 27 | 20 | 5F 2NS | 74% |
| KHR-GL46.texture_filter_anisotropic | 2 | 2 | – | 100% |
| KHR-GL46.texture_filter_minmax_tests | 5 | 0 | 5NS | 0% |
| KHR-GL46.texture_gather | 75 | 71 | 4F | 95% |
| KHR-GL46.texture_repeat_mode | 162 | 162 | – | 100% |
| KHR-GL46.texture_size_promotion | 1 | 0 | 1IE | 0% |
| KHR-GL46.texture_swizzle | 4 | 2 | 1F 1X | 50% |
| KHR-GL46.texture_view | 7 | 1 | 6F | 14% |
| KHR-GL46.transform_feedback | 20 | 20 | – | 100% |
| KHR-GL46.transform_feedback_overflow_query_ARB | 13 | 13 | – | 100% |
| KHR-GL46.vertex_attrib_64bit | 4 | 2 | 2F | 50% |
| KHR-GL46.vertex_attrib_binding | 30 | 30 | – | 100% |
| KHR-GL46.viewport_array | 17 | 15 | 2F | 88% |
| **TOTAL** | **9035** | **8364** | 245F 418NS 3X 3T 2IE | **92.6%** |

Status codes: F=fail, X=crash, T=timeout (batch killed at deadline), IE=internal error,
NS=not supported. The 3 T cases were re-run individually: the tessellation one PASSES
(batch-kill artifact); `shaders.preprocessor.function_redefinition_3_fragment` /
`_vertex` are REAL long-running cases (compiler domain).

### 5 largest failing clusters (F+X+T+IE, NS listed separately)

| # | cluster | fails | owner guess |
|---|---|---|---|
| 1 | `shaders` | 29F + 2T (slow) + 6NS | shaders/compiler (preprocessor redefinition timeouts; 29 assorted fails) |
| 2 | `packed_pixels` | 24F | textures/format conversion (packed pixel upload/download) |
| 3 | `explicit_uniform_location` | 16F | program introspection / link |
| 4 | `shading_language_420pack` | 13F + 1X | compiler (GLSL 420 features) |
| 5 | `program_interface_query` | 13F | program introspection |
| – | `layout_binding` | 12F | compiler/link (3 of them `atomic_uint_layout_binding_*`) |
| – | `shader_draw_parameters` | 12F | draw path (gl_BaseVertex/gl_DrawID) |
| – | `shader_image_load_store` | 13F | geom has partial ownership (was 35 -> 38 in r46) |
| – | `gpu_shader5` | 11F | compiler |
| – | `sample_shading` | 8F | raster |
| – | `texture_cube_map_array` | 5F + 2NS | tess (see open item below) |
| – | crashes: `shading_language_420pack.array_comma.vertex`, `texture_swizzle.texture_swizzle`, `indirect_parameters_tests.ParameterBufferOperations` | 3X | compiler / textures / indirect-draw |

NotSupported-heavy groups (extension not exposed, by design): blend_equation_advanced 88,
pixelstoragemodes 160, sample_variables 38, shader_subroutine 27, internalformat 27,
texture_shadow_lod 16, sparse_texture2 21, etc.

### tess-area sweep findings

* `KHR-GL46.tessellation_shader` **57 Pass + 1 T (artifact; passes standalone)** — no
  regression.
* `KHR-GL46.geometry_shader` **136/136**, `KHR-GL46.transform_feedback` 20/20,
  `transform_feedback_overflow_query` 13/13, `shader_atomic_counters` 28/30
  (basic-usage-vs fixed this round; advanced-usage-multi-stage pre-existing, see gap).
* `KHR-GL46.shader_storage_buffer_object.advanced-write-tessellation` — FAIL (open).
* `KHR-GL46.texture_cube_map_array.image_op_tessellation_control_sh` — FAIL (open;
  investigation below).

## 3. Open tess items (started this round, not finished)

**texture_cube_map_array.image_op_tessellation_control_sh** (also `_evaluation_sh`
passes; the `_control_sh` variant fails "Wrong value … layer[0] … IMMUTABLE storage",
4 resolutions x f/i/u): a TCS writes 6 imageCubeArrays (read f/i/u + write f/i/u) inside
`layout(vertices=1) out` + an isolines/point_mode TES (the CPU-reference route). A
faithful headless replication (tests/headless scratch, since removed) of the whole CTS
flow — 4 resolutions x mutable/immutable, zero-filled write textures, CTS-style
verification (copy-compute over a non-layered `image2D` view for the float texture,
`framebufferTextureLayer` + `readPixels` for i/u) — PASSES 3/3 runs, and MGL_TESS_DEBUG
tracing shows the CTS run itself dispatches the TCS kernel with all six images bound to
fresh textures every round (patches=1, pv=1, storage 6). So the mechanism works; the
remaining difference between the CTS process and the replication is unidentified (next
round: instrument `mglReadPixels`/the readback in the glcts process and compare, or
bisect the CTS's own draw sequence further).

**shader_storage_buffer_object.advanced-write-tessellation** — not investigated yet this
round.

## 4. Test results

* Full headless suite: **481 PASS, 0 FAIL** (test_geom updated for the new GS-AC
  behavior; 596 checks, 0 failures).
* `python3 agents/stub_scanner.py --list 'EMPTY|NOOP|ASSERT'` clean for
  tessellation.c, transform_feedback.c, geometry.c, get.c.
* CTS spot checks on the final build: geometry_shader.api.max_atomic_counters Pass,
  .max_atomic_counter_buffers Pass, geometry_shader.program_resource.program_resource
  Pass, geometry_shader.limits.max_texture_units Pass,
  shader_atomic_counters.basic-usage-vs Pass,
  tessellation_shader.gl_InvocationID_PatchVerticesIn_PrimitiveID Pass.

## 5. Shared-file edits this round

* `MGL/src/get.c` — geometry atomic-counter limits (2 cases).
* `MGL/src/MGLRenderer_Private.h` — declared `nullShaderBuffer` (used by the geometry
  kernel's atomic-counter binding; implementation unchanged).
* `MGL/src/program.c` — none.
* `tests/headless/test_geom.c` — `gs_atomic_counter_link_fails` updated (see above).
