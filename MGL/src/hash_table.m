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
 * hash_table.c
 * MGL
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <assert.h>
#include <stdint.h>  // For SIZE_MAX and UINT_MAX
#include <limits.h>  // For UINT_MAX fallback

#ifdef __APPLE__
#include <Metal/Metal.h>
#endif

#include "hash_table.h"
#include "glm_context.h"

void initHashTable(HashTable *ptr, GLuint size)
{
    size_t len;

    len = sizeof(HashObj) * size;

    ptr->current_name = 1;
    ptr->size = size;
    ptr->keys = (HashObj *)malloc(len);
    assert(ptr->keys);

    bzero(ptr->keys, len);
}

GLuint getNewName(HashTable *table)
{
    return table->current_name++;
}

void *searchHashTable(HashTable *table, GLuint name)
{
    assert(table);
    
    if (name >= table->size)
    {
        /* a name past the grown table simply has no object yet (glGen* reserves
         * names; they enter the table on first bind) — NULL, no log line */
        return NULL;
    }

    return table->keys[name].data;
}

void insertHashElement(HashTable *table, GLuint name, void *data)
{
    assert(table);

    if (name < table->size)
    {
        assert(table->keys[name].data == NULL);

        table->keys[name].data = data;

        return;
    }

    // CRITICAL SECURITY FIX: Prevent integer overflow in hash table resizing
    // some calls allow the user to specify a name...
    while(table->size <= name)
    {
        GLuint old_size = table->size;

        // CRITICAL: Check for integer overflow before multiplication
        if (old_size > UINT_MAX / 2) {
            // SECURITY: Hash table size would overflow, use maximum safe size
            fprintf(stderr, "MGL SECURITY ERROR: Hash table size would overflow, capping at maximum safe size\n");
            // We could return an error here, but for now we'll cap at UINT_MAX to prevent overflow
            // This might limit functionality but prevents critical security vulnerability
            if (old_size == UINT_MAX) {
                break; // Can't grow any further
            }
            table->size = UINT_MAX;
        } else {
            table->size = old_size * 2; // Safe multiplication
        }

        /* table->size is 32-bit and size_t 64-bit, so size * sizeof(HashObj)
         * cannot overflow the allocation size (the old SIZE_MAX check could never
         * fire); keep that true at compile time */
        _Static_assert(sizeof(size_t) >= 8, "hash table sizing assumes a 64-bit size_t");

        table->keys = (HashObj *)realloc(table->keys, table->size * sizeof(HashObj));
        if (!table->keys) {
            // CRITICAL: Handle allocation failure to prevent crash
            fprintf(stderr, "MGL SECURITY ERROR: Hash table allocation failed\n");
            table->size = old_size; // Restore old size
            return;
        }

        // Initialize new entries
        for (GLuint i = old_size; i < table->size; i++) {
            table->keys[i].data = NULL;
        }
    }

    /* the growth loop stops at UINT_MAX entries (valid slots 0..UINT_MAX-1), so a
     * name of UINT_MAX never fits — refuse it instead of writing one slot past the
     * end of the table (from startergo/MGL main 97c8189e..f5b74da2) */
    if (name >= table->size)
    {
        fprintf(stderr, "MGL: insertHashElement - name %u exceeds table size %u, unable to insert\n",
                name, table->size);
        return;
    }

    assert(table->keys[name].data == NULL);
    table->keys[name].data = data;
}

void deleteHashElement(HashTable *table, GLuint name)
{
    assert(table);

    if (name >= table->size)
        return;

    void *obj_data = table->keys[name].data;

    /* Release the Metal object the GL object owned.  These fields were filled
     * with CFBridgingRetain, so NULLing them without a release leaked every
     * deleted texture/buffer/program/shader (glmark2 deletes its objects at the
     * end of every scene).  The table-to-type mapping needs a context; the
     * tables of the current one cover the common case. */
    if (obj_data) {
        extern GLMContext _ctx;

        if (_ctx)
        {
            if (table == &_ctx->state.shader_table) {
                Shader *shader = (Shader *)obj_data;
                if (shader->mtl_data.function) { CFRelease(shader->mtl_data.function); shader->mtl_data.function = NULL; }
                if (shader->mtl_data.library) { CFRelease(shader->mtl_data.library); shader->mtl_data.library = NULL; }
            }
            else if (table == &_ctx->state.program_table) {
                Program *program = (Program *)obj_data;
                if (program->mtl_data) { CFRelease(program->mtl_data); program->mtl_data = NULL; }
            }
            else if (table == &_ctx->state.texture_table) {
                Texture *texture = (Texture *)obj_data;
                if (texture->mtl_data) { CFRelease(texture->mtl_data); texture->mtl_data = NULL; }
                if (texture->params.mtl_data) { CFRelease(texture->params.mtl_data); texture->params.mtl_data = NULL; }
            }
            else if (table == &_ctx->state.buffer_table) {
                Buffer *buffer = (Buffer *)obj_data;
                if (buffer->data.mtl_data) { CFRelease(buffer->data.mtl_data); buffer->data.mtl_data = NULL; }
                mgl_buffer_pool_free(&buffer->data);   /* r153 */
            }
        }
    }

    table->keys[name].data = NULL;
}
