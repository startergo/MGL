# MGL
# OpenGL 4.6 on Metal

MGL implements OpenGL 4.6 (core and compatibility profiles) on top of Apple's Metal. GLSL is compiled to SPIR-V with glslang and translated to Metal Shading Language with SPIRV-Cross; the GL state machine is mapped onto Metal render and compute encoders.

This branch, `gl46-port`, is the GL 4.6 port of Michael Larson's original MGL. It is developed against the Khronos OpenGL conformance tests (CTS). The current pass/fail picture, what changed in each merge and what is still open are kept in [docs/gl46-status.md](docs/gl46-status.md).

The context reports `GL_VERSION` "4.6 Metal" and `GL_SHADING_LANGUAGE_VERSION` "4.60". MGL runs on Apple Silicon and on Intel Macs; the build targets the host architecture.

## What works

- **Shader stages:** vertex, fragment and compute run natively in Metal. Geometry and tessellation shaders, which Metal does not have, are emulated with compute kernels feeding a pass-through vertex stage.
- **Buffers:** vertex, index, uniform (512 binding points), shader storage, atomic counter, texture, indirect-draw and query buffers; persistent and coherent mapping.
- **Transform feedback,** including from geometry and tessellation stages.
- **Textures and samplers:** 1D/2D/3D, cube, arrays, multisample, rectangle, buffer textures, texture views, image load/store, compressed formats, sRGB decode, swizzles and border colours.
- **Framebuffers:** FBOs, multiple render targets, blits, invalidation, and framebuffer fetch (`GL_EXT_shader_framebuffer_fetch`).
- **Render state:** blending (including advanced blend equations), depth and stencil, indexed viewports and scissors, clip and cull distances, polygon offset, primitive restart.
- **Queries and sync:** occlusion, timer, primitive and pipeline-statistics queries; fences.
- **Double precision (fp64)** shaders, lowered to the 32-bit types Metal supports.
- **Program introspection,** separate shader objects / program pipelines, and SPIR-V shaders (`GL_ARB_gl_spirv`).
- **Compatibility profile:** fixed-function vertex and fragment state, client arrays, immediate mode and display lists.

Some functionality is still missing: the unimplemented entry points are stubs that `assert(0)`. See [docs/gl46-status.md](docs/gl46-status.md) for the conformance numbers and the known failures.

## Building

Requirements: macOS with Metal, the Xcode command-line tools, CMake and Python 3.

```sh
cd external
bash clone_external.sh      # OpenGL-Registry, SPIRV-Cross, SPIRV-Headers, SPIRV-Tools, glslang, ezxml
bash build_external.sh
cd ..
make lib                    # build/libmgl.dylib (plus the modified GLFW: build/libglfw.dylib)
```

Other targets:

| target | builds |
|---|---|
| `make glx` | a GLX 1.4 front end, so X11 apps (XQuartz) that link libGL get GL 4.6 on MGL |
| `make nsgl` | an NSOpenGL interposer: `DYLD_INSERT_LIBRARIES=<abs>/build/nsgl/libmgl_nsgl.dylib` runs NSOpenGL apps (for example glmark2-macos) on MGL |

## Testing

`libmgl.dylib` creates a headless Metal-backed context when it loads, so a test is a plain C program that calls GL entry points (see `tests/headless/mgl_test.h`).

```sh
tests/headless/run.sh                 # build and run the whole headless suite
tests/headless/run.sh test_queries    # one test (file stem)
```

The suite has about 220 test programs. `run.sh` refuses to run against a `build/libmgl.dylib` that is older than the sources.

Conformance runs use the Khronos CTS (`glcts`), built separately; `tests/cts_one.sh` runs one case and `tests/cts_group.sh 'KHR-GL46.geometry_shader'` runs a group.

## GLFW support
I modified a version of GLFW to work with MGL, it replaces the default MacOS OpenGL contexts. The changes are included in the repository (`external/glfw`) and are built by `make lib`.

(additional notes from conversy/MGL repo: the modified GLFW version is not mandatory; the GLX front end and the NSOpenGL interposer above are other ways to run existing applications.)

## Mapping OpenGL onto Metal
I mapped as much of the functionality of Metal I could into OpenGL 4.6, it's surprising how much I was able to map directly from OpenGL state to Metal. But I suppose Metal evolved to meet market requirements and since OpenGL was in place the same features existed just in another form.

Metal Hiearchy

    Command Queue
        Command Buffer
            Render Pipeline State <- created with RenderPipelineDescriptor
                Render Encoder  <-- created with RenderPassDescriptor
                    RenderPassDescriptor
                        colorAttachments
                            pixelFormat
                            sourceRGBBlendFactor
                            sourceAlphaBlendFactor
                            destinationRGBBlendFactor
                            destinationAlphaBlendFactor
                            alphaBlendOperation
                            writeMask
                        depthAttachment
                            pixelFormat
                        stencilAttachment
                            pixelFormat
                        visibilityResultBuffer
                        renderTargetArrayLength
                        defaultRasterSampleCount
                        samplePositions
                        renderTargetWidth
                        renderTargetHeight


                DirectState for Render Encoder
                    Vertex Function
                    Fragment Function
                    Vertex Descriptor
                    Vertex Buffers
                    Vertex Textures
                    Viewport
                    FrontFacingWinding
                    CullMode
                    DepthClipMode
                    DepthBias
                    ScissorRect
                    TriangleFillMode
                    Fragment Buffers
                    Fragment Textures
                    Fragment Samplers
                    BlendColor
                    StencilReferenceValue
                    VisibilityResultMode
                    ColorStoreMode
                    DepthStoreMode
                    DepthStoreAction
                    StencilStoreAction
                    DepthStoreActionOptions
                    StencilStoreActionOptions

                Descriptor State used to change render ecoder
                    Depth Stencil State <- created with DepthStencil Descriptor
                        DepthStencilPipeline Descriptor
                            DepthStencil Descriptor
                                readMask
                                writeMask
                                depthCompareFunction
                                depthWriteEnabled
                                stencilCompareFunction
                                stencilFailureOperation
                                depthFailureOperation
                                depthStencilPassOperation
                                frontFaceStencil <- created wtih StencilDescriptor
                                backFaceStencil <- created wtih StencilDescriptor

## Parsing the OpenGL 4.6 XML spec
In the beginning I used ezxml to parse the gl.xml file for all the enums and functions, then printed out one giant file with all the functions. I then used the same parser to create the dispatch tables and data structures. As each functional part was built I separated blocks of functions into these functions like buffers / textures / shaders / programs.

## SPIRV to Metal
I really couldn't have done this project without all the SPIRV support from Khronos, once I found out I could translate GLSL into Metal using some of the SPIRV tools this project became a reality.

Some parts of GLSL have no Metal equivalent. Geometry and tessellation shaders are the largest: MGL runs them as compute kernels that write their output vertices to buffers, which a pass-through vertex function then rasterizes. Double-precision arithmetic is lowered to 32-bit operations, and several other gaps (rectangle images, subgroup votes, arrays of arrays in stage I/O) are rewritten at the SPIR-V level before SPIRV-Cross runs.

## OpenGL functions and how they work
Each OpenGL function starts in gl_core.c

```C
void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)
{
    GLMContext ctx = GET_CONTEXT();

    ctx->dispatch.tex_image2D(ctx, target, level, internalformat, width, height, border, format, type, pixels);
}
```

glTexImage2D calls into a dispatch table which lands on a mgl equivalent (mglTexImage2D in the textures code). The mgl function checks input parameters, then modifies the OpenGL state and marks it dirty. I use macros for error checking parameters... because I got sick of writing.

```C
if (expr == false) {
  set error
  return;
}
```

Most functions work this way: many entry points check their parameters and then share one function that does the actual work, like the one all the TexImage calls use to create a texture level.

## How state reaches Metal
State changes are evaluated on direct commands, the ones that issue work to the GPU: drawing primitives, copying data, executing compute kernels.

A draw call such as glDrawArrays checks its parameters and calls into the Objective-C renderer (MGLRenderer and its categories) through a jump table in the context. Before encoding the draw, the renderer processes any dirty GL state in `processGLState`: it binds buffers and textures to their Metal slots, updates the render encoder's dynamic state, and selects the render pipeline state. A new render encoder is only started when the render targets change.

Two caches keep this cheap. Pipeline state objects are cached by the content of everything they are built from (shader functions, vertex layout, pipeline-baked render state, attachment formats). A per-draw fast path skips the whole state walk when nothing a draw depends on has changed since the previous draw. GL calls that re-send a value that is already current do not mark anything dirty.

Most of this lives in MGLRenderer.m and its category files (`MGLRenderer+*.m`, one per domain: draw, framebuffer, render state, texture data, geometry, tessellation, compatibility). At first it can look like a giant piece of code, but it's simple once you figure out the mapping process.

## Binding buffers and textures from shader layouts
You have to bind all the buffers and textures to Metal, there are mapping operations you need to do to transfer OpenGL buffers / textures to Metal. Since all of this is driven by the GLSL shader in 4.5 which requires you to map your bindings and locations

An example vertex shader defines a vertex buffer object at location 0 and a uniform buffer at binding 0.  These are not the same location, but relative to the type.

    const char* vertex_shader =
    GLSL(450 core,
         layout(location = 0) in vec3 position;

         layout(binding = 0) uniform matrices
         {
             mat4 rotMatrix;
         };

         void main() {
            gl_Position = rotMatrix * vec4(position, 1.0);
         }
    );

Each binding point in OpenGL is a 2D array one for each type, like GL_ARRAY_BUFFER

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

Indexes an array like

    buffers[GL_ARRAY_BUFFER].buffer = vbo

So internal to processGLState the vertex buffer array is taken apart, buffers and attributes mapped, then we walk through uniforms / textures through their binding points in the GL state and map these to Metal state.

Once you walk a glDrawArrays call through processGLState, you will get a jist of how this all works.

## Where to start
Build the library, run `tests/headless/run.sh`, then pick a headless test close to what you want to change and read it next to the code it exercises. Adding functionality works best the same way: read the GL spec for the function you need, write a headless test for it, and make it pass.

## Why the focus on 4.6 functionality?
OpenGL is huge, and the effort was intended capture 4.6 functionality rather than try to implement it all (and that includes a lot of 3.x functionality). This is a good path to embed functionality into the base then add in all the older functionality later using the paths you want rather than hacking apart older paths to make modern OpenGL functionality.

## Contributing
If you want to contribute that would be great, it's all written in C.. in the same style all of the OpenGL framework from Apple was written in. If you don't like the coding style, don't change it. Just follow the same coding style and put your efforts into testing and functionality.

We really need people to contribute to the functionality and testing rather than trying to just build MGL and see if it works with their application. If you can write C code you can probably figure out how MGL works and contribute.

Its a great way to learn the internals of OpenGL and Metal at the same time, and if you are a college student.. or even an expirienced coder its going to look great on a resume that you actually did work on OpenGL internals and its not just a black box to you. A typical driver for a graphics driver will have hundreds of thousands of lines of code and its all state driven regardless of what they say about new interfaces like Metal or Vulkan, because hardware doesn't match these interfaces directly you have to evaluate state like OpenGL does but at a more abstract level.

## Questions?
For the original MGL, you can reach Mike at sandstormsoftware@gmail.com, it would be great to see this used by others and developed into a full fledged project anyone can use.

Cheers

Mike
