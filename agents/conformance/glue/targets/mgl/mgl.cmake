message("*** Using MGL target (headless Metal GL 4.6 core via libmgl.dylib)")

set(DEQP_TARGET_NAME		"MGL")

# The platform port dlopens libmgl.dylib at run time (path overridable with
# the MGL_LIBRARY environment variable) -- no link-time GL dependency.
set(DEQP_SUPPORT_GLES1		OFF)
set(DEQP_SUPPORT_VG			OFF)
set(DEQP_SUPPORT_WGL		OFF)
set(DEQP_SUPPORT_GLX		OFF)
set(DEQP_SUPPORT_EGL		OFF)
set(DEQP_PLATFORM_LIBRARIES	)

set(TCUTIL_PLATFORM_SRCS
	mgl/tcuMGLPlatform.hpp
	mgl/tcuMGLPlatform.cpp
	)
