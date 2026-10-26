/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * error.h
 * MGL
 *
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "error.h"
#include "mgl_ext_debug.h"

/* Default builds must be silent: apps (and the CTS) make plenty of spec-legal
 * error probes, so the per-error stderr lines are gated behind MGL_DEBUG. */
static int mgl_dbg(void)
{
    static int v = -1;
    if (v < 0)
    {
        const char *e = getenv("MGL_DEBUG");
        v = (e && *e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return v;
}

GLenum  mglGetError(GLMContext ctx)
{
    GLenum err;

    err = ctx->state.error;

    if (err != GL_NO_ERROR && mgl_dbg())
        fprintf(stderr, "MGL DEBUG: mglGetError returning 0x%x (%d)\n", err, err);

    ctx->state.error = GL_NO_ERROR;

    return err;
}


void error_func(GLMContext ctx, const char *func, GLenum error)
{
    if (mgl_dbg())
        fprintf(stderr, "MGL GL Error in %s: 0x%x (%d)\n", func, error, error);

    /* KHR_debug: every error generates a GL_DEBUG_TYPE_ERROR message, even when
       an earlier error is still pending in the sticky error flag */
    mgl_debug_report_error(ctx, func, error);

    if (ctx->state.error)
        return;

    ctx->state.error = error;

    /* Temporarily disabled to allow QEMU to continue despite errors */
    // if (ctx->assert_on_error)
    //     assert(0);
}
