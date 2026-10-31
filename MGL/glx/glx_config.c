/*
 * glx_config.c — pure fbconfig/visual logic of the GLX shim (no Display*
 * calls): the synthesized config table for a screen, attribute filtering for
 * glXChooseVisual / glXChooseFBConfig, the attribute readout used by
 * glXGetConfig / glXGetFBConfigAttrib, the config sort order and the client
 * strings.  Unit-tested headlessly by tests/headless/test_glx_config.c.
 *
 * Licensed under the Apache License, Version 2.0 (see the repository LICENSE).
 */
#include <stdlib.h>
#include <string.h>
#include "glx_types.h"

/* GL_TRUE/GL_FALSE without pulling in GL headers */
#define MGLX_TRUE  1
#define MGLX_FALSE 0

/*
 * Config table for one screen. All configs use the screen's TrueColor 24-bit
 * visual; the variants cover the combinations X11 apps typically ask for
 * (single/double buffer, with/without depth+stencil, with/without alpha and
 * accumulation, and an MSAA variant; all are backed by the same MGL
 * offscreen framebuffer, whose depth/stencil is always present — the sizes
 * below are what glXGetConfig reports and what the GL context provides).
 */
void mglx_build_configs(MGLXFBConfigRec *out, int n, int screen,
                        mglx_visualid visualid, int depth, int first_id)
{
    if (n > MGLX_NCONFIGS)
        n = MGLX_NCONFIGS;
    for (int i = 0; i < n; i++)
    {
        MGLXFBConfigRec *c = &out[i];
        memset(c, 0, sizeof *c);
        c->fbconfig_id = first_id + i;
        c->screen = screen;
        c->visualid = visualid;
        c->depth = depth;
        c->x_visual_type = GLX_TRUE_COLOR;
        c->drawable_type = GLX_WINDOW_BIT | GLX_PIXMAP_BIT | GLX_PBUFFER_BIT;
        c->render_type = GLX_RGBA_BIT;
        c->x_renderable = MGLX_TRUE;
        c->stereo = MGLX_FALSE;
        c->red_size = c->green_size = c->blue_size = 8;
        c->buffer_size = 24;                    /* RGB without alpha */
        c->level = 0;
        c->aux_buffers = 0;
        c->accum_red = c->accum_green = c->accum_blue = c->accum_alpha = 0;
        c->sample_buffers = 0;
        c->samples = 0;
        c->transparent_type = GLX_NONE;
        c->caveat = GLX_NONE;
        c->max_pbuffer_width = 8192;
        c->max_pbuffer_height = 8192;
        c->max_pbuffer_pixels = 8192 * 8192;

        /* the combinations, ordered best-first (see mglx_config_compare):
         * double-buffered with depth+stencil is what ~every app wants */
        static const struct { int dbl, alpha, ds, accum, msaa; } table[MGLX_NCONFIGS] = {
            /* keep in mglx_config_compare order: double buffered, depth+stencil,
             * alpha, then the special variants, then the single-buffered ones */
            { 1, 1, 1, 0, 0 },
            { 1, 1, 1, 0, 1 },
            { 1, 1, 1, 1, 0 },
            { 1, 0, 1, 0, 0 },
            { 1, 1, 0, 0, 0 },
            { 0, 1, 1, 0, 0 },
            { 0, 1, 0, 0, 0 },
            { 0, 0, 0, 1, 0 },
        };
        c->doublebuffer = table[i].dbl ? MGLX_TRUE : MGLX_FALSE;
        if (table[i].alpha) { c->alpha_size = 8; c->buffer_size = 32; }
        if (table[i].ds)    { c->depth_size = 24; c->stencil_size = 8; }
        if (table[i].accum) { c->accum_red = c->accum_green = c->accum_blue = c->accum_alpha = 16; }
        if (table[i].msaa)  { c->sample_buffers = 1; c->samples = 4; }
    }
}

int mglx_config_attrib(const MGLXFBConfigRec *cfg, int attrib)
{
    switch (attrib)
    {
        case GLX_X_VISUAL_TYPE:         return cfg->x_visual_type;
        case GLX_TRANSPARENT_TYPE:      return cfg->transparent_type;
        case GLX_TRANSPARENT_INDEX_VALUE:
        case GLX_TRANSPARENT_RED_VALUE:
        case GLX_TRANSPARENT_GREEN_VALUE:
        case GLX_TRANSPARENT_BLUE_VALUE:
        case GLX_TRANSPARENT_ALPHA_VALUE: return -1;    /* not transparent */
        case GLX_VISUAL_ID:             return (int)cfg->visualid;
        case GLX_SCREEN:                return cfg->screen;
        case GLX_FBCONFIG_ID:           return cfg->fbconfig_id;
        case GLX_DRAWABLE_TYPE:         return cfg->drawable_type;
        case GLX_RENDER_TYPE:           return cfg->render_type;
        case GLX_X_RENDERABLE:          return cfg->x_renderable;
        case GLX_DOUBLEBUFFER:          return cfg->doublebuffer;
        case GLX_STEREO:                return cfg->stereo;
        case GLX_RED_SIZE:              return cfg->red_size;
        case GLX_GREEN_SIZE:            return cfg->green_size;
        case GLX_BLUE_SIZE:             return cfg->blue_size;
        case GLX_ALPHA_SIZE:            return cfg->alpha_size;
        case GLX_BUFFER_SIZE:           return cfg->buffer_size;
        case GLX_LEVEL:                 return cfg->level;
        case GLX_AUX_BUFFERS:           return cfg->aux_buffers;
        case GLX_DEPTH_SIZE:            return cfg->depth_size;
        case GLX_STENCIL_SIZE:          return cfg->stencil_size;
        case GLX_ACCUM_RED_SIZE:        return cfg->accum_red;
        case GLX_ACCUM_GREEN_SIZE:      return cfg->accum_green;
        case GLX_ACCUM_BLUE_SIZE:       return cfg->accum_blue;
        case GLX_ACCUM_ALPHA_SIZE:      return cfg->accum_alpha;
        case GLX_SAMPLE_BUFFERS:        return cfg->sample_buffers;
        case GLX_SAMPLES:               return cfg->samples;
        case GLX_MAX_PBUFFER_WIDTH:     return cfg->max_pbuffer_width;
        case GLX_MAX_PBUFFER_HEIGHT:    return cfg->max_pbuffer_height;
        case GLX_MAX_PBUFFER_PIXELS:    return cfg->max_pbuffer_pixels;
        case GLX_SWAP_INTERVAL_EXT:     return 1;   /* fixed 1 in the shim */
        case GLX_MAX_SWAP_INTERVAL_EXT: return 1;
        case GLX_LATE_SWAPS_TEAR_EXT:    return 0;
        /* the legacy boolean RGBA attribute: we are always RGBA */
        case GLX_RGBA:                   return MGLX_TRUE;
        default:                         return GLX_NONE; /* not a GLX attribute */
    }
}

/* min-size attribute test (GLX semantics): a config passes when its value is
 * >= the requested value; GLX_DONT_CARE (0xFFFFFFFF / negative) means ignore.
 * Boolean attributes (GLX_USE_GL, GLX_RGBA, GLX_DOUBLEBUFFER, GLX_STEREO) must
 * match exactly.  GLX_NONE (0) as a size means "smallest available" i.e. any. */
#define GLX_DONT_CARE_INT (-1)   /* apps pass 0xFFFFFFFF which truncates to -1 */

int mglx_config_matches(const MGLXFBConfigRec *cfg, const int *attribs,
                        int is_fbconfig_attribs, const char **bad_attrib)
{
    if (!attribs)
        return 1;

    /* GLX attribute lists mix name/value pairs with value-less boolean
     * tokens ({GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24, None}):
     * walk one token at a time and only consume a value for attributes that
     * take one; a boolean followed by True(1)/False(0) consumes it too, so
     * strict pair-style lists work as well. */
    for (int i = 0; attribs[i] != 0; i++)
    {
        int attrib = attribs[i];
        int value = 1;          /* boolean default: request it */
        int have;

        /* attributes only legal in glXChooseFBConfig */
        if (!is_fbconfig_attribs &&
            (attrib == GLX_DRAWABLE_TYPE || attrib == GLX_RENDER_TYPE ||
             attrib == GLX_X_RENDERABLE || attrib == GLX_FBCONFIG_ID ||
             attrib == GLX_MAX_PBUFFER_WIDTH || attrib == GLX_MAX_PBUFFER_HEIGHT ||
             attrib == GLX_MAX_PBUFFER_PIXELS || attrib == GLX_SAMPLE_BUFFERS ||
             attrib == GLX_SAMPLES || attrib == GLX_TRANSPARENT_TYPE))
        {
            if (bad_attrib) *bad_attrib = "fbconfig-only attribute in glXChooseVisual";
            return -1;
        }

        switch (attrib)
        {
            /* value-carrying attributes: the next token is the value */
            case GLX_BUFFER_SIZE: case GLX_LEVEL: case GLX_RED_SIZE:
            case GLX_GREEN_SIZE: case GLX_BLUE_SIZE: case GLX_ALPHA_SIZE:
            case GLX_DEPTH_SIZE: case GLX_STENCIL_SIZE:
            case GLX_ACCUM_RED_SIZE: case GLX_ACCUM_GREEN_SIZE:
            case GLX_ACCUM_BLUE_SIZE: case GLX_ACCUM_ALPHA_SIZE:
            case GLX_AUX_BUFFERS: case GLX_SAMPLES: case GLX_SAMPLE_BUFFERS:
            case GLX_VISUAL_ID: case GLX_FBCONFIG_ID: case GLX_SCREEN:
            case GLX_DRAWABLE_TYPE: case GLX_RENDER_TYPE: case GLX_X_VISUAL_TYPE:
            case GLX_TRANSPARENT_TYPE: case GLX_TRANSPARENT_INDEX_VALUE:
            case GLX_TRANSPARENT_RED_VALUE: case GLX_TRANSPARENT_GREEN_VALUE:
            case GLX_TRANSPARENT_BLUE_VALUE: case GLX_TRANSPARENT_ALPHA_VALUE:
            case GLX_MAX_PBUFFER_WIDTH: case GLX_MAX_PBUFFER_HEIGHT:
            case GLX_MAX_PBUFFER_PIXELS:
                value = attribs[++i];
                break;

            /* boolean attributes: value-less by convention (the terminator
             * None that follows is NOT a False); an explicit True(1) or
             * GLX_DONT_CARE right behind one is consumed so strict pair-style
             * lists keep parsing */
            case GLX_USE_GL: case GLX_RGBA: case GLX_DOUBLEBUFFER:
            case GLX_STEREO: case GLX_X_RENDERABLE:
                if (attribs[i + 1] == 1 || attribs[i + 1] == GLX_DONT_CARE_INT)
                    value = attribs[++i];
                break;

            default:
                if (bad_attrib) *bad_attrib = "unknown attribute";
                return -1;
        }

        have = mglx_config_attrib(cfg, attrib);

        if (value == GLX_DONT_CARE_INT)
            continue;

        switch (attrib)
        {
            case GLX_USE_GL:
            case GLX_RGBA:
            case GLX_DOUBLEBUFFER:
            case GLX_STEREO:
            case GLX_X_RENDERABLE:
                if (have != value)
                    return 0;
                break;
            case GLX_LEVEL:
                if (have != value)
                    return 0;   /* no overlay/underlay planes */
                break;
            case GLX_X_VISUAL_TYPE:
                if (have != value)
                    return 0;
                break;
            case GLX_TRANSPARENT_TYPE:
                if (value != GLX_NONE)
                    return 0;   /* no transparency */
                break;
            case GLX_VISUAL_ID:
            case GLX_FBCONFIG_ID:
            case GLX_SCREEN:
                if (have != value)
                    return 0;
                break;
            case GLX_DRAWABLE_TYPE:
                if (value && (have & value) != value)
                    return 0;
                break;
            case GLX_RENDER_TYPE:
                if ((have & value) == 0)
                    return 0;   /* we only have RGBA configs */
                break;
            default:
                /* min-size semantics; 0 (GLX_NONE) = don't care */
                if (value > 0 && have < value)
                    return 0;
                break;
        }
    }
    return 1;
}

/* sort: prefer double buffered, then depth, then alpha, then no-accum, then
 * no-msaa (this mirrors the order the config table is generated in, so the
 * table is already sorted; the comparator is used after attribute filtering
 * to keep the "best match first" guarantee) */
int mglx_config_compare(const MGLXFBConfigRec *a, const MGLXFBConfigRec *b)
{
    if (a->doublebuffer != b->doublebuffer) return b->doublebuffer - a->doublebuffer;
    if (a->depth_size   != b->depth_size)   return b->depth_size   - a->depth_size;
    if (a->stencil_size != b->stencil_size) return b->stencil_size - a->stencil_size;
    if (a->alpha_size   != b->alpha_size)   return b->alpha_size   - a->alpha_size;
    if (a->accum_red    != b->accum_red)    return a->accum_red    - b->accum_red;
    if (a->samples      != b->samples)      return a->samples      - b->samples;
    return a->fbconfig_id - b->fbconfig_id;
}

static const char *mglx_ext_string =
    "GLX_ARB_create_context "
    "GLX_ARB_create_context_profile "
    "GLX_ARB_get_proc_address "
    "GLX_EXT_swap_control "
    "GLX_EXT_swap_control_tear "
    "GLX_SGI_swap_control "
    "GLX_MESA_swap_control "
    "GLX_SGIX_fbconfig "
    "GLX_SGIX_pbuffer "
    "GLX_SGI_make_current_read "
    "GLX_EXT_import_context "
    "GLX_MESA_query_renderer "
    "GLX_ARB_multisample "
    "GLX_EXT_visual_info "
    "GLX_EXT_visual_rating";

const char *mglx_extensions_string(void)
{
    return mglx_ext_string;
}

const char *mglx_client_string(int name)
{
    switch (name)
    {
        case GLX_VENDOR:     return "MGL (Metal)";
        case GLX_VERSION:    return "1.4 MGL";
        case GLX_EXTENSIONS: return mglx_ext_string;
        default:             return NULL;
    }
}
