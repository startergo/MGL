/*
 * tessellation.c
 * MGL — tessellation shaders: Metal tessellation pipeline (TCS/TES), patches
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_tess.h"

#pragma mark - domain 'tess' hooks

void mgl_ext_init_tess(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_tess(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
