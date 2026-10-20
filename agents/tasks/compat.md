# Domain: compat - OpenGL 4.6 COMPATIBILITY profile (user decision 2026-10-19)

MGL today is core-only: 391 compatibility entry points return GL_INVALID_OPERATION (legacy_compat.c,
non_core_unimplemented.c), GL_CONTEXT_PROFILE_MASK is hard-wired to CORE (get.c:487, glm_params.c:220),
client arrays / VAO 0 are rejected (vertex_arrays.c:432, draw_buffers.c:146). Goal: a real 4.6
compatibility-profile context, selectable at context creation, while a core context keeps exact core
behaviour (core must still reject compat-only calls and VAO 0). Advertise GL_ARB_compatibility and
GL_CONTEXT_COMPATIBILITY_PROFILE_BIT only for compat contexts.

Reference design: Mesa (MIT licence, compatible with Apache-2.0; ~/mesa on this Mac). Read, adapt, cite:
- src/mesa/main/ffvertex_prog.c (fixed-function vertex program generator), src/mesa/main/ff_fragment_shader.cpp
  (texenv/fog/alpha fixed-function fragment), src/mesa/vbo/ (immediate mode + display-list vertex capture),
  src/mesa/main/dlist.c (display lists), attrib.c (push/pop attrib), drawpix.c / st_cb_drawpixels.c / st_cb_bitmap.c,
  st_cb_rasterpos.c, feedback.c / st_cb_feedback.c, eval.c, src/compiler/glsl builtin compat uniforms/variables.
Emulate on Metal the way Mesa emulates on modern hardware: everything becomes generated shaders + buffers.

## Milestones (one or more commits each, suite green before every commit, failing test first)
1. Profile plumbing: context creation option for compat (find how MGL contexts are created - glm_params.c, CGL
   attrs, GLFW hints - and add the compat profile), GL_CONTEXT_PROFILE_MASK, version string, extension list,
   per-context dispatch so compat entry points are live only in compat contexts. Compat GLSL: #version 110-150
   and "#version 4xx compatibility" shaders (gl_Vertex/gl_Normal/gl_Color/gl_MultiTexCoordN/gl_FogCoord attribute
   aliasing, gl_FragColor/gl_FragData, gl_ModelViewMatrix & all built-in state uniforms, ftransform, gl_ClipVertex,
   gl_FrontColor/gl_BackColor/gl_TexCoord[] varyings) through glslang (it supports compatibility profile).
2. Vertex specification: immediate mode glBegin/glEnd + glVertex/Color/Normal/TexCoord/MultiTexCoord/... batching
   into streaming buffers; client-side vertex arrays and VAO 0 (upload ranges per draw); legacy array pointers
   (glVertexPointer, glColorPointer, glEnableClientState, glInterleavedArrays, glArrayElement); GL_QUADS,
   GL_QUAD_STRIP, GL_POLYGON (index conversion; flat-shading provoking vertex rules); glRect*.
3. Fixed-function vertex: matrix stacks (glMatrixMode, glLoadMatrix, glMultMatrix, glPush/PopMatrix, glOrtho,
   glFrustum, glRotate/Translate/Scale, texture matrices), lighting (all glLight/glMaterial/glLightModel params,
   two-sided, local viewer, separate specular, glColorMaterial), normalize/rescale normal, texgen (all modes),
   user clip planes (glClipPlane -> clip distances), fog coordinate, point size/attenuation, glPolygonMode +
   front/back colors. Generated shaders cached by a state key (like Mesa's ffvertex key).
4. Fixed-function fragment: glTexEnv (all modes incl. GL_COMBINE / combine operands/scales, texture units),
   fog (linear/exp/exp2, per-vertex/per-fragment), alpha test, glShadeModel, separate specular, point sprites,
   line/polygon stipple (fragment-shader pattern), glPixelZoom-independent pieces.
5. Display lists: glNewList/glEndList/glCallList(s)/glGenLists/glDeleteLists/glListBase, recording every
   compat+core command allowed in lists (and executing immediately the ones that are not), with vertex data
   captured into buffers (Mesa vbo_save).
6. Attribute stacks and raster ops: glPush/PopAttrib (all groups), glPush/PopClientAttrib; glRasterPos/glWindowPos,
   glDrawPixels, glBitmap, glCopyPixels, glPixelZoom, glPixelTransfer/glPixelMap (maps), glAccum + accumulation
   buffer (extra float attachment), glIndex*/color-index only as far as the spec requires in RGBA mode.
7. Legacy texture state/formats: GL_ALPHA/LUMINANCE/LUMINANCE_ALPHA/INTENSITY (swizzles), GL_CLAMP and
   texture borders, GL_GENERATE_MIPMAP, glTexEnv LOD bias, glPrioritizeTextures/glAreTexturesResident, proxy
   behaviour; glGet of all compat state (glGetLight/Material/TexEnv/TexGen/ClipPlane/Map/PixelMap/PolygonStipple).
8. glRenderMode GL_SELECT/GL_FEEDBACK (select/name stack, feedback buffer; GPU xfb or CPU path), evaluators
   (glMap1/2, glEvalCoord/Mesh/Point, glMapGrid), glEdgeFlag.
Tests: new tests/headless/test_compat_*.c with pixel-exact checks per milestone, plus a test that a CORE context
still rejects compat calls. Keep a running progress log in agents/compat_report.md (commit it) so a fresh session
can continue; agent_report.md stays untracked. Keep shell output short so the context lasts.
