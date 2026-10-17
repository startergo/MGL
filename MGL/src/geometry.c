/*
 * geometry.c
 * MGL — geometry shaders: compute-based emulation, streams, layered output
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_geom.h"

#pragma mark - domain 'geom' hooks

void mgl_ext_init_geom(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_geom(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
