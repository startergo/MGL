/*-------------------------------------------------------------------------
 * MGL platform port for the Khronos OpenGL Conformance Test Suite
 * -----------------------------------------------------------------------
 * Copyright (c) 2025, MGL project
 *
 * The platform dlopens libmgl.dylib (path taken from the MGL_LIBRARY
 * environment variable, default "libmgl.dylib"). Loading the library
 * creates MGL's process-global headless Metal GL 4.6 core context and
 * leaves it current, so a RenderContext only needs to resolve the GL
 * entry points through the library and report a render target.
 *
 * The intended way to run the suite is with an FBO-backed default
 * framebuffer:  --deqp-surface-type=fbo --deqp-gl-context-type=mgl
 * (glu::createRenderContext() then wraps this context in a
 * glu::FboRenderContext that honours --deqp-surface-width/height and the
 * requested bit depths).
 *
 * Known limitations (single-context library):
 *   - shared contexts alias the one global MGL context;
 *   - ES context requests are NotSupported.
 *--------------------------------------------------------------------*/

#include "tcuMGLPlatform.hpp"

#include "deDynamicLibrary.hpp"
#include "tcuDefs.hpp"
#include "tcuRenderTarget.hpp"
#include "gluRenderConfig.hpp"
#include "gluRenderContext.hpp"
#include "glwEnums.hpp"
#include "glwFunctions.hpp"
#include "glwInitFunctions.hpp"

#include <cstdlib>
#include <sys/sysctl.h>
#include <sys/types.h>

namespace tcu
{
namespace mgl
{
namespace
{

const char *getLibraryPath(void)
{
    const char *env = std::getenv("MGL_LIBRARY");
    return (env && env[0] != '\0') ? env : "libmgl.dylib";
}

// C entry points exported by libmgl.dylib (glm_context.c)
typedef void *(*MGLGetCurrentContextFunc)(void);
typedef void (*MGLSetCurrentContextFunc)(void *);

// glw::FunctionLoader backed by de::DynamicLibrary (dlsym).
class MglFunctionLoader : public glw::FunctionLoader
{
public:
    MglFunctionLoader(const de::DynamicLibrary *library) : m_library(library)
    {
    }

    glw::GenericFuncType get(const char *name) const
    {
        // getFunction() returns DE_NULL for unknown symbols, which is what we
        // want: missing entry points show up as null pointers exactly like on
        // any other driver and the affected tests fail on their own merits.
        return reinterpret_cast<glw::GenericFuncType>(m_library->getFunction(name));
    }

private:
    const de::DynamicLibrary *m_library;
};

// A GL context wrapper around the single process-global MGL context.
class MglRenderContext : public glu::RenderContext
{
public:
    MglRenderContext(const glu::RenderConfig &config);
    virtual ~MglRenderContext(void)
    {
    }

    virtual glu::ContextType getType(void) const
    {
        return m_type;
    }
    virtual const glw::Functions &getFunctions(void) const
    {
        return m_functions;
    }
    virtual const tcu::RenderTarget &getRenderTarget(void) const
    {
        return m_renderTarget;
    }
    virtual void postIterate(void);
    virtual void makeCurrent(void);
    virtual glw::GenericFuncType getProcAddress(const char *name) const;

private:
    MglRenderContext(const MglRenderContext &);
    MglRenderContext &operator=(const MglRenderContext &);

    const glu::ContextType m_type;
    de::DynamicLibrary m_library;
    glw::Functions m_functions;
    tcu::RenderTarget m_renderTarget;
    void *m_ownContext;
    MGLGetCurrentContextFunc m_getCurrentContext;
    MGLSetCurrentContextFunc m_setCurrentContext;
};

MglRenderContext::MglRenderContext(const glu::RenderConfig &config)
    : m_type(config.type)
    , m_library(getLibraryPath())
    , m_functions()
    , m_renderTarget()
    , m_ownContext(DE_NULL)
    , m_getCurrentContext(DE_NULL)
    , m_setCurrentContext(DE_NULL)
{
    // Loading the library has already created (and made current) MGL's
    // headless Metal context.  Compatibility-profile requests build the
    // fixed-function context instead: mglCompatCreateContext() makes it the
    // current (global) MGL context, so every gl* entry point now runs in it.
    MglFunctionLoader loader(&m_library);

    if (config.type.getAPI().getProfile() == glu::PROFILE_COMPATIBILITY)
    {
        typedef void *(*MGLCompatCreateContextFunc)(void);
        MGLCompatCreateContextFunc createCompat =
            reinterpret_cast<MGLCompatCreateContextFunc>(m_library.getFunction("mglCompatCreateContext"));
        if (createCompat == DE_NULL)
            throw tcu::NotSupportedError("libmgl does not export mglCompatCreateContext", DE_NULL, __FILE__,
                                         __LINE__);
        if (createCompat() == DE_NULL)
            throw tcu::InternalError("mglCompatCreateContext() failed", DE_NULL, __FILE__, __LINE__);
    }

    m_getCurrentContext =
        reinterpret_cast<MGLGetCurrentContextFunc>(m_library.getFunction("MGLgetCurrentContext"));
    m_setCurrentContext =
        reinterpret_cast<MGLSetCurrentContextFunc>(m_library.getFunction("MGLsetCurrentContext"));
    m_ownContext = (m_getCurrentContext != DE_NULL) ? m_getCurrentContext() : DE_NULL;

    glu::initFunctions(&m_functions, &loader, config.type.getAPI());
    glu::initExtensionFunctions(&m_functions, &loader, config.type.getAPI());

    // Drain errors left over from context bring-up so the first test case
    // starts from a clean error state.
    while (m_functions.getError != DE_NULL && m_functions.getError() != GL_NO_ERROR)
        ;

    // Size the (rarely used) window-path default framebuffer: MGL's default
    // framebuffer follows the viewport when there is no drawable layer.
    const int width = config.width == glu::RenderConfig::DONT_CARE ? 256 : config.width;
    const int height = config.height == glu::RenderConfig::DONT_CARE ? 256 : config.height;
    if (config.width != glu::RenderConfig::DONT_CARE && config.height != glu::RenderConfig::DONT_CARE &&
        m_functions.viewport != DE_NULL)
    {
        m_functions.viewport(0, 0, width, height);
        while (m_functions.getError() != GL_NO_ERROR)
            ;
    }

    m_renderTarget = tcu::RenderTarget(width, height, tcu::PixelFormat(8, 8, 8, 8), 24, 8, 0);
}

void MglRenderContext::postIterate(void)
{
    // Headless: nothing to swap.
    if (m_functions.finish != DE_NULL)
        m_functions.finish();
}

void MglRenderContext::makeCurrent(void)
{
    // MGL keeps one global context; re-assert ours (the library auto-created a
    // core context at load; a compatibility run swapped in a compat context).
    if (m_setCurrentContext != DE_NULL && m_ownContext != DE_NULL)
        m_setCurrentContext(m_ownContext);
}

glw::GenericFuncType MglRenderContext::getProcAddress(const char *name) const
{
    return reinterpret_cast<glw::GenericFuncType>(m_library.getFunction(name));
}

// Context factory registered as "mgl" (--deqp-gl-context-type=mgl).
class MglContextFactory : public glu::ContextFactory
{
public:
    MglContextFactory(void) : glu::ContextFactory("mgl", "MGL headless Metal GL 4.6 core context")
    {
    }

    virtual glu::RenderContext *createContext(const glu::RenderConfig &config, const tcu::CommandLine &cmdLine,
                                              const glu::RenderContext *sharedContext) const;

private:
    MglContextFactory(const MglContextFactory &);
    MglContextFactory &operator=(const MglContextFactory &);
};

glu::RenderContext *MglContextFactory::createContext(const glu::RenderConfig &config,
                                                     const tcu::CommandLine &cmdLine,
                                                     const glu::RenderContext *sharedContext) const
{
    DE_UNREF(cmdLine);

    const glu::ApiType api = config.type.getAPI();

    if (api.getProfile() == glu::PROFILE_ES)
        throw tcu::NotSupportedError("MGL is a desktop GL implementation (no ES contexts)", DE_NULL, __FILE__,
                                     __LINE__);

    // Compatibility-profile requests are honoured: MGL builds a compat context
    // (fixed-function pipeline) through the same dylib; see MglRenderContext.
    if (api.getMajorVersion() > 4 || (api.getMajorVersion() == 4 && api.getMinorVersion() > 6))
        throw tcu::NotSupportedError("MGL implements OpenGL 4.6", DE_NULL, __FILE__, __LINE__);

    // Context flags (debug/robust/forward) are accepted but not honoured by
    // MGL; tests that require them will report their own failures.

    if (sharedContext != DE_NULL)
        tcu::print("WARNING: mgl platform: shared GL contexts alias the single MGL context\n");

    return new MglRenderContext(config);
}

} // namespace

/*--------------------------------------------------------------------
 *  vk::Platform stub: the GL CTS asks the platform for the Vulkan
 *  memory limits (scaleDimensionsToMemory) even in pure-GL tests; the
 *  base-class default throws NotSupportedError, which those cases then
 *  report as failures.  Only the limits are answered; a real Vulkan
 *  library is not available (createLibrary throws, like the null
 *  platform's other paths).
 *------------------------------------------------------------------*/
namespace
{

class MglVulkanPlatform : public vk::Platform
{
public:
    virtual vk::Library *createLibrary(void) const
    {
        throw tcu::NotSupportedError("MGL is a GL-only platform (no Vulkan driver)", DE_NULL, __FILE__, __LINE__);
    }

    virtual void getMemoryLimits(vk::PlatformMemoryLimits &limits) const
    {
        int64_t memSize = 8ll * 1024 * 1024 * 1024; // fallback: 8 GB
        size_t sz       = sizeof(memSize);
        if (sysctlbyname("hw.memsize", &memSize, &sz, DE_NULL, 0) != 0 || memSize <= 0)
            memSize = 8ll * 1024 * 1024 * 1024;

        // Apple Silicon has unified memory: DEVICE_LOCAL counts against system
        // memory, which the CTS expresses as totalDeviceLocalMemory = 0.
        limits.totalSystemMemory                 = (size_t)(memSize / 2); // leave room for everything else
        limits.totalDeviceLocalMemory            = 0;
        limits.deviceMemoryAllocationGranularity = 256;
        limits.devicePageSize                    = 16384;
        limits.devicePageTableEntrySize          = 8;
        limits.devicePageTableHierarchyLevels    = 3;
    }
};

} // namespace

const vk::Platform &Platform::getVulkanPlatform(void) const
{
    static MglVulkanPlatform platform;
    return platform;
}

Platform::Platform(void)
{
    m_contextFactoryRegistry.registerFactory(new MglContextFactory());
}

Platform::~Platform(void)
{
}

bool Platform::processEvents(void)
{
    // No window system events to process.
    return true;
}

const glu::Platform &Platform::getGLPlatform(void) const
{
    return *this;
}

} // namespace mgl
} // namespace tcu

tcu::Platform *createPlatform(void)
{
    return new tcu::mgl::Platform();
}
