/*-------------------------------------------------------------------------
 * MGL platform port for the Khronos OpenGL Conformance Test Suite
 * -----------------------------------------------------------------------
 * Copyright (c) 2025, MGL project
 *
 * Minimal headless platform: libmgl.dylib creates a process-global
 * Metal-backed GL 4.6 core context in a load-time constructor, so the
 * platform only has to load the library, resolve the entry points and
 * expose a glu::RenderContext. Rendering goes through an FBO-backed
 * "default framebuffer" (glu::FboRenderContext, --deqp-surface-type=fbo).
 *--------------------------------------------------------------------*/
#ifndef _TCUMGLPLATFORM_HPP
#define _TCUMGLPLATFORM_HPP

#include "tcuPlatform.hpp"
#include "gluPlatform.hpp"
#include "vkPlatform.hpp"

namespace tcu
{
namespace mgl
{

class Platform : public tcu::Platform, public glu::Platform
{
public:
    Platform(void);
    virtual ~Platform(void);

    virtual bool processEvents(void);
    virtual const glu::Platform &getGLPlatform(void) const;
    /* The CTS's GL tests only touch the "Vulkan platform" for memory limits
     * (ShaderImageLoadStoreBase::scaleDimensionsToMemory asks how big a test
     * surface may be).  MGL is GL-only, so createLibrary() throws, but the
     * limits are reported from the host (Apple GPUs use unified memory, so
     * totalDeviceLocalMemory stays 0 = "counts against system memory"). */
    virtual const vk::Platform &getVulkanPlatform(void) const;
};

} // namespace mgl
} // namespace tcu

tcu::Platform *createPlatform(void);

#endif // _TCUMGLPLATFORM_HPP
