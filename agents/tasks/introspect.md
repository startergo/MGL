# Domain: introspect — uniforms, program interface queries, ProgramUniform*, subroutines

Worktree `wt/introspect`, branch `gl46/introspect`. Your files: `MGL/src/uniforms.c`,
`MGL/src/program_introspection.c`, `MGL/src/mgl_ext_introspect.h`, `MGL/src/MGLRenderer+Introspect.m`
(if present), `tests/headless/test_introspect.c`.

## State when you start
A previous agent implemented most of this domain and was stopped mid-edit:
- Last commit `70dc331` (merged gl46-port). `stub_scanner` shows no stubs in your files.
- `MGL/src/shaders.c` has an **uncommitted edit** — inspect `git diff MGL/src/shaders.c`, decide whether
  it is a complete, correct change (finish it) or a half-done experiment (revert it with
  `git checkout -- MGL/src/shaders.c`). Explain which in your report.
- First run `git merge gl46-port` (it now also contains `gl46/texdata`), rebuild, and run the full
  `tests/headless/run.sh`.

## Finish and verify
Make sure every one of these is real and covered by `tests/headless/test_introspect.c`:
- GetUniformfv/iv/uiv/dv, GetnUniform*v (bufSize → INVALID_OPERATION), GetUniformLocation.
- GetUniformIndices, GetActiveUniformsiv (TYPE, SIZE, NAME_LENGTH, BLOCK_INDEX, OFFSET, ARRAY_STRIDE,
  MATRIX_STRIDE, IS_ROW_MAJOR, ATOMIC_COUNTER_BUFFER_INDEX), GetActiveUniformName, GetActiveUniform,
  GetActiveAttrib, GetAttribLocation, GetAttachedShaders (program domain's attachment list:
  `MGL_PROGRAM_OBJ(p)->attached` / `num_attached`; `Program.shader_slots[]` holds LINKED stages).
- BindAttribLocation / BindFragDataLocation(Indexed) taking effect at the next link (pre/post link hooks,
  `mgl_ext_register_pre_link` / `post_link`), GetFragDataLocation / GetFragDataIndex, dual-source index 1.
- GetUniformBlockIndex, GetActiveUniformBlockiv/Name, UniformBlockBinding and ShaderStorageBlockBinding
  changing the Metal buffer feeding the block at draw time.
- GetActiveAtomicCounterBufferiv; full GL 4.3 program interface query (GetProgramInterfaceiv,
  GetProgramResourceIndex/Name/iv/Location/LocationIndex) for UNIFORM, UNIFORM_BLOCK,
  ATOMIC_COUNTER_BUFFER, PROGRAM_INPUT, PROGRAM_OUTPUT, BUFFER_VARIABLE, SHADER_STORAGE_BLOCK,
  TRANSFORM_FEEDBACK_VARYING/BUFFER (via `mgl_xfb_program_info()`), *_SUBROUTINE(_UNIFORM).
- ProgramUniform* (all 60) sharing the Uniform* implementation; location -1 ignored; type errors.
- Subroutines: GetSubroutineIndex, GetSubroutineUniformLocation, GetActiveSubroutine*,
  UniformSubroutinesuiv, GetUniformSubroutineuiv, GetProgramStageiv (glslang cannot emit SPIR-V for
  `subroutine`; a source rewrite to int uniform + switch is the expected emulation).
- **Known bug to fix:** default-block uniforms in the VERTEX stage (`layout(location=N) uniform float k;`)
  failed on normal draws (`mapGLBuffersToMTLBufferMap` raised INVALID_OPERATION). Verify with a draw
  test that reads the uniform in the vertex shader and checks pixels.
- GetProgramiv ACTIVE_* counts: the program domain computes them from SPIR-V; if your reflection is the
  source of truth, serve them through `mgl_program_register_programiv_hook` (mgl_ext_program.h) so the
  two can never disagree.
- Doubles (dvec/dmat): Metal has no fp64 — store doubles, convert to float for the GPU, return doubles.

Errors: use `ERROR_RETURN(GL_X); return;` (the macro does not return by itself).
