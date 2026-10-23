/*
 * glx_types.h — GLX shim types and GLX token values (public GLX numbers).
 *
 * Part of the MGL GLX front end (MGL/glx/): a GLX 1.4 implementation on top of
 * MGL contexts + Xlib, for X11 apps (XQuartz) that link libGL.
 *
 * The token values MUST match the system <GL/glx.h>; they are defined here so
 * this header is self-contained (apps bring their own glx.h; we never include
 * it because we provide the definitions ourselves).
 */
#ifndef MGL_GLX_TYPES_H
#define MGL_GLX_TYPES_H

/* XID-compatible resource types (X11 XID/VisualID are unsigned long; the
 * shim includes the real X11 headers, this header stays X-free so the pure
 * config logic can be unit-tested without an X11 installation) */
typedef unsigned long mglx_xid;
typedef unsigned long mglx_visualid;

/* opaque handles.  The app's glx.h declares GLXContext and GLXFBConfig as
 * void or XID compatible types; the shim uses its own struct types under
 * the public names so its own prototypes compile without the system glx.h. */
typedef struct MGLXContextRec *GLXContext;
typedef struct MGLXFBConfigRec *GLXFBConfig;
typedef GLXContext MGLXContext;
typedef GLXFBConfig MGLXFBConfig;
typedef mglx_xid GLXContextID;
typedef mglx_xid GLXDrawable;
typedef mglx_xid GLXPixmap;
typedef mglx_xid GLXWindow;
typedef mglx_xid GLXPbuffer;

typedef struct MGLXFBConfigRec MGLXFBConfigRec;

/* GLX 1.0 visual attributes (glXChooseVisual / glXGetConfig) */
#define GLX_NONE                           0x8000
#define GLX_USE_GL                          1
#define GLX_BUFFER_SIZE                     2
#define GLX_LEVEL                           3
#define GLX_RGBA                            4
#define GLX_DOUBLEBUFFER                    5
#define GLX_STEREO                          6
#define GLX_AUX_BUFFERS                     7
#define GLX_RED_SIZE                        8
#define GLX_GREEN_SIZE                      9
#define GLX_BLUE_SIZE                      10
#define GLX_ALPHA_SIZE                     11
#define GLX_DEPTH_SIZE                     12
#define GLX_STENCIL_SIZE                   13
#define GLX_ACCUM_RED_SIZE                 14
#define GLX_ACCUM_GREEN_SIZE               15
#define GLX_ACCUM_BLUE_SIZE                16
#define GLX_ACCUM_ALPHA_SIZE               17

/* GLX errors (offsets from the extension base) */
#define GLX_BAD_SCREEN                      1
#define GLX_BAD_ATTRIBUTE                   2
#define GLX_NO_EXTENSION                    3
#define GLX_BAD_VISUAL                      4
#define GLX_BAD_CONTEXT                     5
#define GLX_BAD_VALUE                       6
#define GLX_BAD_ENUM                        7
#define GLX_BAD_DRAWABLE                    8
#define GLX_BAD_CURRENT_WINDOW              9
#define GLX_BAD_RENDER_AREA                10
#define GLX_BAD_CONTEXT_STATE             11
#define GLX_BAD_DRAWABLE_STATE            12

/* glXGetClientString / glXQueryServerString names */
#define GLX_VENDOR                          1
#define GLX_VERSION                         2
#define GLX_EXTENSIONS                      3

/* GLX 1.1+ fbconfig attributes (glXChooseFBConfig / glXGetFBConfigAttrib) */
#define GLX_X_VISUAL_TYPE                 0x22
#define GLX_TRANSPARENT_TYPE              0x23
#define GLX_TRANSPARENT_INDEX_VALUE       0x24
#define GLX_TRANSPARENT_RED_VALUE         0x25
#define GLX_TRANSPARENT_GREEN_VALUE       0x26
#define GLX_TRANSPARENT_BLUE_VALUE        0x27
#define GLX_TRANSPARENT_ALPHA_VALUE       0x28
#define GLX_VISUAL_ID                     0x800B
#define GLX_SCREEN                        0x800C
#define GLX_NON_CONFORMANT_CONFIG         0x800D
#define GLX_DRAWABLE_TYPE                 0x8010
#define GLX_RENDER_TYPE                   0x8011
#define GLX_X_RENDERABLE                  0x8012
#define GLX_FBCONFIG_ID                   0x8013
#define GLX_RGBA_TYPE                     0x8014
#define GLX_COLOR_INDEX_TYPE              0x8015
#define GLX_MAX_PBUFFER_WIDTH             0x8016
#define GLX_MAX_PBUFFER_HEIGHT            0x8017
#define GLX_MAX_PBUFFER_PIXELS            0x8018
#define GLX_PRESERVED_CONTENTS            0x801B
#define GLX_LARGEST_PBUFFER               0x801C
#define GLX_WIDTH                         0x801D
#define GLX_HEIGHT                        0x801E
#define GLX_EVENT_MASK                    0x801F
#define GLX_DAMAGED                       0x8020
#define GLX_SAVED                         0x8021
#define GLX_WINDOW                        0x8022
#define GLX_PBUFFER                       0x8023
#define GLX_PBUFFER_HEIGHT                0x8040
#define GLX_PBUFFER_WIDTH                 0x8041
#define GLX_SWAP_INTERVAL_EXT             0x20F0
#define GLX_MAX_SWAP_INTERVAL_EXT         0x20F1
#define GLX_LATE_SWAPS_TEAR_EXT           0x20F3

#define GLX_WINDOW_BIT                     0x00000001
#define GLX_PIXMAP_BIT                     0x00000002
#define GLX_PBUFFER_BIT                    0x00000004
#define GLX_AUX_BUFFERS_BIT                0x00000010
#define GLX_FRONT_LEFT_BUFFER_BIT          0x00000020
#define GLX_FRONT_RIGHT_BUFFER_BIT         0x00000040
#define GLX_BACK_LEFT_BUFFER_BIT           0x00000080
#define GLX_BACK_RIGHT_BUFFER_BIT          0x00000100
#define GLX_DEPTH_BUFFER_BIT               0x00000200
#define GLX_STENCIL_BUFFER_BIT             0x00000400
#define GLX_ACCUM_BUFFER_BIT               0x00000800
#define GLX_MULTISAMPLE_BIT                0x20000000
#define GLX_RGBA_BIT                       0x00000001
#define GLX_COLOR_INDEX_BIT                0x00000002

#define GLX_SLOW_CONFIG                    0x8001
#define GLX_TRUE_COLOR                     0x8002
#define GLX_DIRECT_COLOR                   0x8003
#define GLX_PSEUDO_COLOR                   0x8004
#define GLX_STATIC_COLOR                   0x8005
#define GLX_GRAY_SCALE                     0x8006
#define GLX_STATIC_GRAY                    0x8007
#define GLX_TRANSPARENT_RGB                0x8008
#define GLX_TRANSPARENT_INDEX              0x8009

#define GLX_SAMPLES                       100001
#define GLX_SAMPLE_BUFFERS                100000

#define GLX_PBUFFER_CLOBBER_MASK       0x08000000

/* GLX_ARB_create_context */
#define GLX_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB     0x2092
#define GLX_CONTEXT_FLAGS_ARB             0x2094
#define GLX_CONTEXT_PROFILE_MASK_ARB      0x9126
#define GLX_CONTEXT_DEBUG_BIT_ARB         0x00000001
#define GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001
#define GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
#define GLX_CONTEXT_ROBUST_ACCESS_BIT_ARB 0x00000004
#define GLX_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB 0x8256
#define GLX_NO_RESET_NOTIFICATION_ARB     0x8261
#define GLX_LOSE_CONTEXT_ON_RESET_ARB     0x8252

/*
 * One framebuffer configuration. All configs map to X's TrueColor 24-bit
 * visuals (the window's pixel format on screen is always 8:8:8 RGB; the GL
 * buffers live in MGL's offscreen framebuffer), so the differences between
 * configs are only the GL-level sizes MGL is asked to provide.
 */
struct MGLXFBConfigRec {
    int fbconfig_id;        /* GLX_FBCONFIG_ID, stable per display */
    int screen;
    mglx_visualid visualid; /* X visual a window for this config should use */
    int depth;              /* X depth of that visual */
    int x_visual_type;      /* GLX_TRUE_COLOR */
    int drawable_type;      /* GLX_WINDOW_BIT|GLX_PIXMAP_BIT|GLX_PBUFFER_BIT */
    int render_type;        /* GLX_RGBA_BIT */
    int x_renderable;       /* GL_TRUE */
    int doublebuffer;       /* GL_TRUE/GL_FALSE */
    int stereo;             /* GL_FALSE (never) */
    int red_size, green_size, blue_size, alpha_size;
    int buffer_size;        /* GLX_BUFFER_SIZE: depth of the color buffer */
    int level;              /* main plane 0 */
    int aux_buffers;        /* 0 */
    int depth_size, stencil_size;
    int accum_red, accum_green, accum_blue, accum_alpha;
    int sample_buffers, samples;
    int transparent_type;   /* GLX_NONE */
    int srgb;               /* GLX_FRAMEBUFFER_SRGB_CAPABLE */
    int caveat;             /* GLX_NONE (not GLX_SLOW_CONFIG) */
    int max_pbuffer_width, max_pbuffer_height, max_pbuffer_pixels;
};

/* number of configs we synthesize per screen */
#define MGLX_NCONFIGS 8

/* pure (no Display*) helpers, unit-testable headlessly (glx_config.c) */
int  mglx_config_matches(const MGLXFBConfigRec *cfg, const int *attribs,
                         int is_fbconfig_attribs, const char **bad_attrib);
/* fill a config table for a screen given the chosen X visual (visualid/depth) */
void mglx_build_configs(MGLXFBConfigRec *out, int n, int screen,
                        mglx_visualid visualid, int depth, int first_id);
/* one attribute of a config, as glXGetFBConfigAttrib/glXGetConfig report it;
 * returns GLX_NONE when 'attrib' is not a GLX attribute (caller raises) */
int  mglx_config_attrib(const MGLXFBConfigRec *cfg, int attrib);
/* sort order for glXChooseFBConfig (GLX "better" configs first); <0: a better */
int  mglx_config_compare(const MGLXFBConfigRec *a, const MGLXFBConfigRec *b);
/* the strings reported by glXGetClientString/glXQueryServerString */
const char *mglx_client_string(int name);
const char *mglx_extensions_string(void);

#endif /* MGL_GLX_TYPES_H */
