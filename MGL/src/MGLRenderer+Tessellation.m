/*
 * MGLRenderer+Tessellation.m — Metal side of the 'tess' domain.
 *
 * Executes a GL_PATCHES draw. Two paths (see mgl_ext_tess.h for the design):
 *
 *   hardware:  triangles/quads without point_mode use the GPU tessellation pipe:
 *              compute:  VS kernel (SPIRV-Cross VERTEX_FOR_TESSELLATION) over
 *                        (vertices, instances) -> optional gather kernel (indexed
 *                        draws) -> TCS kernel (multi-patch) writing control points,
 *                        patch data and MTLTriangle/QuadTessellationFactorsHalf
 *              render:   the post-tessellation TES ([[patch(...)]]) with
 *                        drawPatches / drawIndexedPatches + setTessellationFactorBuffer
 *
 *   emulated:  isolines and point_mode (Metal's tessellator has no isoline domain
 *              and cannot emit points): the same VS/TCS compute stages, then the
 *              D3D11-style tessellator of MGLTessKernels.h runs as compute kernels
 *              (count -> emit -> indirect draw args) and the TES — rewritten to a
 *              plain vertex function reading (tess coord, patch id) / control points
 *              / patch data from buffers — draws the result through an indirect
 *              draw. No CPU readback on either path.
 *
 * The default-levels case (no TCS) needs no TCS kernel: the factors are the
 * glPatchParameterfv defaults and control points pass through from the VS kernel.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Compat.h"   /* compat: bind shadow (perf session 3) */
#import "MGLTessKernels.h"
#import "glm_context.h"
#import "mgl_ext_gpuaudit.h"   /* per-buffer content serial for the scan cache */
#import "mgl_ext_rstate.h"     /* the clip-space fixup's function-constant indices */
#import "mgl.h"
#import "mgl_ext.h"
#import "mgl_ext_tess.h"
#include <mach/mach_time.h>   /* the opt-in CPU phase accounting (below) */
#include <stdatomic.h>        /* the ring's claim/publish protocol (below) */
#import "mgl_ext_queries.h"  /* tess-count queries force the exact CPU path */
size_t mgl_tess_factor_stride(GLMContext ctx, MGLTessProgram *tp);
#import "mgl_ext_rstate.h"
#import "mgl_ext_vattrib.h"
#import "mgl_ext_introspect.h"   /* mgl_introspect_slot_owner: internal slot -> GL location */
#import "mgl_ext_rstate.h"
#import "mgl_ext_buffers.h"
#import "mgl_ext_program.h"
#import "glcorearb.h"

#define MAX_ATTRIB_SLOTS 16

/* Trace / configuration flags, evaluated once. getenv() per draw would sit directly
 * on the GL_PATCHES draw path; these are read at most once per process.
 *   MGL_TESS_DEBUG     — per-draw tracing
 *   MGL_TESS_CPU       — force the CPU tessellator (reference path)
 *   MGL_TESS_NO_FIXUP  — do not inject the clip-space fixup into the plain TES */
static int tTraceFlag = -1, tCpuFlag = -1, tNoFixupFlag = -1;
static bool tTrace(void)
{
    if (tTraceFlag < 0)
        tTraceFlag = (getenv("MGL_TESS_DEBUG") != NULL) ? 1 : 0;
    return tTraceFlag != 0;
}
static bool tCpuReference(void)
{
    if (tCpuFlag < 0)
        tCpuFlag = (getenv("MGL_TESS_CPU") != NULL) ? 1 : 0;
    return tCpuFlag != 0;
}

/* MGLRenderer implements these in its class extension (MGLRenderer.m); declared here so
 * the tessellation compute-kernel path can feed the shared 1x1 fallbacks. */
@interface MGLRenderer (TessFallbacks)
- (id<MTLTexture>) computeFallbackTexture;
- (id<MTLSamplerState>) computeFallbackSampler;
@end

static bool tNoFixup(void)
{
    if (tNoFixupFlag < 0)
        tNoFixupFlag = (getenv("MGL_TESS_NO_FIXUP") != NULL) ? 1 : 0;
    return tNoFixupFlag != 0;
}

/* A kernel PSO cache entry retains the MTLFunction it was built from: keys use object
 * addresses, and a freed program/function whose address is reallocated (programs are
 * created and deleted every frame in real applications) must never hit a stale PSO —
 * that returned e.g. a 3-control-point TCS kernel for a 4-control-point program. */
@interface MGLTessPsoEntry : NSObject
@property (nonatomic, strong) id<MTLComputePipelineState> pso;
@property (nonatomic, strong) id<MTLFunction> fn;
@property (nonatomic, assign) void *program;   /* for the VS cache's extra dimension */
@end

@implementation MGLTessPsoEntry
@end

@implementation MGLRenderer (Tessellation)

/* gl_DrawID / gl_BaseVertex (GL_ARB_shader_draw_parameters): the raster vertex
 * function reads them from MGL_DRAW_PARAMS (draw_buffers.c stamps
 * ctx->state.ext_tess per exec_draw — GL's base vertex is 0 for arrays draws,
 * Metal's [[base_vertex]] would be `first').  Inline bytes: no CPU stall.
 * O(1) gate (glmark2 perf r29): the link-time uses_draw_params flag. */
- (void) extTessBindDrawParams: (id<MTLRenderCommandEncoder>) enc
{
    if (!mgl_tess_draw_params_bound(ctx))
        return;
    uint32_t dp[2] = { 0, 0 };
    if (ctx->state.ext_tess)
    {
        dp[0] = ctx->state.ext_tess->draw_id;
        dp[1] = (uint32_t)ctx->state.ext_tess->draw_base_vertex;
    }
    [enc setVertexBytes: dp length: sizeof dp atIndex: MGL_DRAW_PARAMS];
}

/* ------------------------------------------------------------------ scratch state
 * Categories cannot add ivars; the scratch buffers are file-static and grow on demand.
 * They hold no GL objects, so lifetime is "until program exit".
 */
/* per-draw scratch ring: each draw gets an exclusive block, released when the command
 * buffer(s) that encode it complete. Never resized, never shared between draws.
 *
 * Depth (r43): a slot retires only when its command buffer COMPLETES, and Metal fires
 * the completion handlers with ~0.5-2 ms of jitter (worse under load).  With draws
 * encoding in ~100 us, that jitter is worth 5-20 in-flight draws: the old 8-slot ring
 * went dry in every tight GL_PATCHES loop (measured: 327 spinning acquires in a
 * 1000-draw bench, ~330 us/draw sleeping in tessRingAcquire).  24 slots absorb the
 * jitter; the METAL buffers are still allocated lazily and only ever grow to the
 * largest block handed out, so an idle ring costs eight pointers. */
#define MGL_TESS_RING 24
static id<MTLBuffer> tRing[MGL_TESS_RING];
static size_t tRingPos[MGL_TESS_RING];
static int tRingScan = 0;
static size_t tRingCap[MGL_TESS_RING];

/* blocks above this size are freed on retire rather than kept (two stay warm): a
 * capture draw's scratch is tens of MB and would otherwise be pinned per slot —
 * 24 slots x 10-60 MB is not acceptable, and a workload that repeats such draws is
 * GPU-bound anyway (the ~100 us mmap is noise there). */
#define TESS_RING_BIG (4u << 20)

static size_t tessRingAlign(size_t v) { return (v + 255) & ~(size_t)255; }

/* r44: upper bound on patches per draw (instances included).  The hardware
 * path's minimum per-patch ring footprint is ~50 B (factor record + float
 * levels + patch data), so 4M patches is already >200 MB of scratch; anything
 * bigger belongs to a workload that is far past what a single Metal buffer can
 * back, and the guard keeps every derived count (uint32 kernel words, size_t
 * offsets) provably in range. */
#define MGL_TESS_MAX_TOTAL_PATCHES (1LL << 22)

/* exhaustion diagnostics (mgl_tess_ring_spins): acquires that had to poll for a
 * slot, the worst poll count and the total time slept — the health signal for the
 * keep-alive/depth tuning below (a healthy hot loop stays at or near zero) */
static unsigned tRingSpinCount;
static unsigned tRingSpinMax;
static uint64_t tRingSleptNs;
unsigned mgl_tess_ring_spins(uint64_t *max_ns, uint64_t *total_ns)
{
    if (max_ns) *max_ns = (uint64_t)tRingSpinMax * 100u;   /* usleep(100) per poll */
    if (total_ns) *total_ns = tRingSleptNs;
    return tRingSpinCount;
}

/* MGL_TESS_RING_TRACE=1: log every acquire/release/retire so a wedged ring (a draw
 * that never retires its slot) can be found in a CTS log. */
static int tRingTrace(void)
{
    static int v = -1;
    if (v < 0) v = getenv("MGL_TESS_RING_TRACE") ? 1 : 0;
    return v;
}

/* acquire a block of `bytes` from the ring; returns the buffer and sets *offset.
 * Grows a free slot if needed (creating a NEW buffer; busy ones are never touched).
 *
 * Concurrency protocol (r43/r44 hardening): only Metal completion-handler threads
 * and the GL thread touch the ring.  A slot is CLAIMED through the state word
 * (FREE -> BUSY); a retire CASes BUSY -> BUSY|RETIRING at the generation it was
 * armed for, mutates exclusively in that state, then publishes FREE with the
 * generation bumped — an acquire that observes FREE sees the slot's final
 * tRing/tRingCap/tRingPos by construction, and the twin releases (completion
 * handler + GL-thread poll) can never both act. */
static void tessRingPendPoll(void);
static void tessRingPendPollFull(bool need_slot);
static id<MTLBuffer> tessRingAcquire(id<MTLDevice> device, size_t bytes, size_t *offset,
                                     int *slot, MGLRenderer *self);
/* r44: a slot's retire can now arrive TWICE — the completion handler (exact but
 * jumpy: Metal fires it 0.5-2 ms after the GPU finished, worse under load) and
 * the GL thread's own poll of the in-flight list (every acquire that needs a
 * slot).  A plain busy flag cannot tell "the poll already released this" from
 * "the next draw claimed it again", so the late handler would free a slot that
 * has a new owner — and two releasers racing would both run the release's
 * mutations (big-block free) on a slot one of them no longer owns.  The state
 * word packs all of it: bits2.. = generation of the current claim, bit1 =
 * RETIRING (a release won and is mutating; claims must not take it), bit0 =
 * BUSY.  A retire CASes BUSY->BUSY|RETIRING at the generation it was armed
 * for, mutates under that exclusive state, then publishes FREE with the
 * generation bumped: whichever of the duplicate retires runs second finds the
 * word already moved and is an exact no-op, and a claim can never observe a
 * slot mid-mutation. */
static atomic_uint tRingWord[MGL_TESS_RING];

static inline bool tessSlotBusy(int i)      /* busy or retiring: not claimable */
{
    return (atomic_load_explicit(&tRingWord[i], memory_order_acquire) & 3u) != 0;
}
static inline uint32_t tessSlotGen(int i)
{
    return atomic_load_explicit(&tRingWord[i], memory_order_acquire) >> 2;
}
/* claim (GL thread): FREE -> BUSY, keeping the generation the last release
 * left; the caller then owns every non-atomic field of the slot.  false = not
 * free. */
static inline bool tessSlotClaim(int i)
{
    unsigned w = atomic_load_explicit(&tRingWord[i], memory_order_acquire);
    do {
        if (w & 3u)
            return false;
    } while (!atomic_compare_exchange_weak_explicit(&tRingWord[i], &w, (w & ~3u) | 1u,
                 memory_order_acq_rel, memory_order_acquire));
    return true;
}
/* the one full release: mutations + publish, exclusive through the RETIRING
 * state.  `gen` must be the generation the retire was armed for; a stale gen
 * (someone else owns the slot now, or the twin retire won) is a no-op. */
/* big-block recycle counter (mgl_tess_ring_freed_big): blocks above
 * TESS_RING_BIG that were freed on retire instead of being pinned in a slot;
 * mgl_tess_ring_big_slots counts slots currently holding such a block.  A burst
 * of big draws must keep BOTH small (free draws reuse a warm big slot, at most a
 * couple grow for in-flight work) — test_tess.c asserts it, the regression
 * guard for the ring's memory bound. */
static unsigned long tRingFreedBig;
static void tessRingRetireGen(int slot, uint32_t gen)
{
    unsigned exp = (gen << 2) | 1u;         /* BUSY, not retiring */
    unsigned win = (gen << 2) | 3u;         /* BUSY + RETIRING */
    if (slot < 0 || slot >= MGL_TESS_RING)
        return;
    if (!atomic_compare_exchange_strong_explicit(&tRingWord[slot], &exp, win,
            memory_order_acq_rel, memory_order_acquire))
    {
        /* the word moved under us: the twin retire (handler or poll) already
         * acted, or a stale retire found a new owner — exactly what the
         * generation is for; count it so tests can see the guard work */
        atomic_fetch_add_explicit(&tRingDupRetires, 1u, memory_order_relaxed);
        return;
    }
    if (tRingTrace())
        fprintf(stderr, "MGL ring: release slot=%d gen=%u\n", slot, gen);
    /* big transient blocks are not worth pinning: keep two warm (a repeating
     * big-draw workload then reallocates nothing) and free the rest, so the
     * deeper ring cannot multiply a capture draw's tens of MB into gigabytes.
     * Every mutation below happens while the slot is RETIRING — a state no
     * claim can take — so an acquire that later observes BUSY==0 sees the
     * FINAL tRing/tRingCap/tRingPos (the r43 hardening: the recycle used to
     * publish first, so a claim racing it could take a nil'd buffer). */
    if (tRing[slot] && tRingCap[slot] > TESS_RING_BIG)
    {
        int warm = 0;
        for (int i = 0; i < MGL_TESS_RING; i++)
            if (i != slot && !tessSlotBusy(i) && tRing[i] && tRingCap[i] > TESS_RING_BIG)
                    warm++;
        if (warm >= 2)
        {
            size_t freed = tRingCap[slot];
            tRing[slot] = nil;      /* ARC drops the MTLBuffer */
            tRingCap[slot] = 0;
            tRingFreedBig++;
            if (tRingTrace())
                fprintf(stderr, "MGL ring: freed big slot=%d (%zu bytes, %d warm)\n",
                        slot, freed, warm);
        }
    }
    tRingPos[slot] = 0;            /* reuse from the start once retired */
    atomic_store_explicit(&tRingWord[slot], (gen + 1u) << 2, memory_order_release);
}
/* owner-side release (the draw's own early exit) and the stale-busy recovery:
 * release whatever generation currently holds the slot */
static inline void tessRingRetireCurrent(int slot)
{
    if (tessSlotBusy(slot))
        tessRingRetireGen(slot, tessSlotGen(slot));
}
/* tess draws encoded on the current shared command buffer (AGX keep-alive,
 * see the flush at the end of the hardware path) */
static unsigned tHwDrawsOnCb = 0;
/* >0: the ring went dry recently - flush every draw for the next 64 draws
 * (see the keep-alive comment at the end of the hardware path) */
static unsigned tRingExhaust = 0;
static id<MTLBuffer> tessRingAcquire(id<MTLDevice> device, size_t bytes, size_t *offset,
                                     int *slot, MGLRenderer *self)
{
    tessRingPendPoll();
    for (int i = 0; i < MGL_TESS_RING; i++)
    {
        int idx = (tRingScan + i) % MGL_TESS_RING;
        /* claim FIRST, read pos/cap under the claim: a free slot's fields are
         * only stable until another claim, and between an earlier read and this
         * one a competing GL thread (a second context: the ring is
         * process-global) can run a whole claim->retire cycle — including the
         * big-block recycle that nils tRing[idx].  Fields read while the slot
         * is owned are untouchable, so a stale-cap handout is impossible. */
        if (tessSlotClaim(idx))
        {
            size_t cap = tRingCap[idx], pos = tRingPos[idx];

            if (cap >= pos + bytes)
            {
                *offset = pos;
                tRingPos[idx] = pos + tessRingAlign(bytes);
                *slot = idx;
                tRingScan = (idx + 1) % MGL_TESS_RING;
                if (tRingTrace())
                    fprintf(stderr, "MGL ring: acquire slot=%d bytes=%zu cap=%zu (fast)\n",
                            idx, bytes, tRingCap[idx]);
                return tRing[idx];
            }
            /* no room: hand the slot straight back.  Nothing was mutated (pos,
             * cap, buffer untouched), so publishing FREE needs no RETIRING
             * dance — bump the generation so no stale retire can match. */
            atomic_store_explicit(&tRingWord[idx], (tessSlotGen(idx) + 1u) << 2,
                                  memory_order_release);
        }
    }
    /* no free slot with room: find a free one and grow it (or take the largest) */
    int best = -1;
    for (int i = 0; i < MGL_TESS_RING; i++)
        if (!tessSlotBusy(i) && (best < 0 || tRingCap[i] > tRingCap[best]))
            best = i;
    if (best < 0)
    {
        /* All slots are busy: earlier draws hold them and their command buffers are
         * already committed (each tess draw commits its own), so their completion
         * handlers are about to release.  A burst of tiny draws with no readback in
         * between (e.g. discarded patches, 0 records) can outrun the GPU — wait for a
         * slot instead of failing the draw.  If a slot does not free up quickly, the
         * last open command buffer may still be sitting uncommitted (nothing forced a
         * flush): commit it so its completion handlers can retire the slots.  Bounded
         * (100 us steps then 250 us, with an exact wait on a parked buffer at ~1 ms),
         * then give up. */
        tRingSpinCount++;
        for (int spin = 0; spin < 20000; spin++)
        {
            bool any = false;
            /* the pressure path: poll the in-flight handler retires too, so a
             * slot whose command buffer already completed frees NOW instead of
             * when Metal's handler thread gets scheduled */
            tessRingPendPollFull(true);
            if ((unsigned)spin > tRingSpinMax)
                tRingSpinMax = (unsigned)spin;
            tRingSleptNs += (uint64_t)(spin < 10 ? 100 : 250) * 1000u;  /* the usleep below */
            if (tRingTrace() && (spin % 2000) == 0)
            {
                unsigned bm = 0, nfree = 0;
                size_t used = 0;
                for (int i = 0; i < MGL_TESS_RING; i++)
                {
                    bool b = tessSlotBusy(i);
                    bm |= (unsigned)b << i;
                    if (!b) nfree++;
                }
                fprintf(stderr, "MGL ring: SPIN %d free=%u/%d busy-bm-lo=%x used=%zu"
                        " cb=%d enc=%d\n", spin, nfree, MGL_TESS_RING, bm & 0xff, used,
                        self && self->_currentCommandBuffer ?
                            (int)self->_currentCommandBuffer.status : -1,
                        self && self->_currentRenderEncoder ? 1 : 0);
            }

            for (int i = 0; i < MGL_TESS_RING && !any; i++)
                any = !tessSlotBusy(i);
            if (any)
            {
                    return tessRingAcquire(device, bytes, offset, slot, self);
            }
            if (spin == 50 && self && self->_currentCommandBuffer &&
                self->_currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
            {
                /* shared-buffer draws (r42): ring blocks retire with the command
                 * buffer the draw encoded into, which stays open across a burst of
                 * draws — the eight slots can all be busy on it.  End the render
                 * encoding and commit it so the completion handlers can retire the
                 * slots; the next draw's extBufEnsureCommandBuffer (or this draw's
                 * compute phase, which picks a fresh buffer when the current one is
                 * committed) reopens one. */
                [self endRenderEncoding];
                [self->_currentCommandBuffer commit];
                self->_currentCommandBuffer = nil;
                tHwDrawsOnCb = 0;
                tRingExhaust = 64;   /* go per-draw for a while (keep-alive) */
            }
            if (spin == 10)
            {
                /* r43 wait efficiency: after ~1 ms of polling, stop burning CPU.
                 * Parking holds the not-yet-committed buffers, but committed ones
                 * pass through here too (a retired draw's shared buffer that got
                 * committed between park and now): block on the first COMMITTED
                 * parked buffer's completion — an exact wake instead of more
                 * syscalls.  The open shared buffer is skipped (it is committed
                 * at spin 50 below or by the next flush); buffers retired via
                 * completion handlers are not parked and keep the poll path.
                 * MGL_TESS_NOWAIT=1 keeps the pure usleep backoff (A/B knob). */
                static int noWait = -1;
                if (noWait < 0)
                    noWait = getenv("MGL_TESS_NOWAIT") != NULL;
                if (!noWait)
                for (int i = 0; i < MGL_TESS_RING; i++)
                {
                    id<MTLCommandBuffer> pcb = tRingPendCb[i];
                    if (pcb && pcb != self->_currentCommandBuffer &&
                        pcb.status >= MTLCommandBufferStatusCommitted &&
                        pcb.status < MTLCommandBufferStatusCompleted)
                    {
                        [pcb waitUntilCompleted];
                        break;
                    }
                }
            }
            if (spin == 19999 && !self->_currentCommandBuffer && !self->_currentRenderEncoder)
            {
                /* 2 s with no progress, no open encoder and no uncommitted command
                 * buffer: there is no in-flight GPU work that could still fire the
                 * pending completion handlers, so the busy flags are stale (a draw
                 * that exited without registering its release).  Retire them rather
                 * than failing every later draw. */
                for (int i = 0; i < MGL_TESS_RING; i++)
                    tessRingRetireCurrent(i);
                return tessRingAcquire(device, bytes, offset, slot, self);
            }
            usleep(spin < 10 ? 100 : 250);
        }
        return nil;                    /* genuinely stuck: caller must give up */
    }
    /* claim first, grow after: the mutations then happen only on a slot this
     * thread owns (the claim is uncontended in practice — GL contexts serialize
     * — but exact under C11, and the re-entry covers the impossible loss) */
    if (!tessSlotClaim(best))
        return tessRingAcquire(device, bytes, offset, slot, self);
    {
        /* r44 bound: `bytes` comes from per-draw count arithmetic that has already
         * been capped (see the draw entry), but the grow must also respect Metal's
         * own per-buffer limit — newBufferWithLength above it fails (and for absurd
         * values can abort), so refuse the block here and let the caller drop the
         * draw instead of the process. */
        NSUInteger maxbuf = device.maxBufferLength ? device.maxBufferLength : (1u << 30);
        size_t cap = tRingCap[best] + tessRingAlign(bytes) + (1u << 20);
        if (cap > (size_t)maxbuf)
        {
            cap = tessRingAlign(bytes);
            if (cap > (size_t)maxbuf)
            {
                static bool warned = false;
                if (!warned)
                {
                    NSLog(@"MGL tess: draw dropped: scratch block %zu bytes exceeds the Metal buffer limit %llu",
                          bytes, (unsigned long long)maxbuf);
                    warned = true;
                }
                tessRingRetireGen(best, tessSlotGen(best));
                return nil;
            }
        }
        id<MTLBuffer> nb = [device newBufferWithLength: cap
                              options: MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
        if (!nb)
        {
            tessRingRetireGen(best, tessSlotGen(best));
            return nil;
        }
        tRing[best] = nb;
        tRingCap[best] = cap;
        tRingPos[best] = 0;
    }
    *offset = tRingPos[best];
    tRingPos[best] += tessRingAlign(bytes);
    *slot = best;
    tRingScan = (best + 1) % MGL_TESS_RING;
    if (tRingTrace())
        fprintf(stderr, "MGL ring: acquire slot=%d bytes=%zu cap=%zu (grow)\n",
                best, bytes, tRingCap[best]);
    return tRing[best];
}

/* big-block recycle counter (mgl_tess_ring_freed_big): blocks above
 * TESS_RING_BIG that were freed on retire instead of being pinned in a slot;
 * mgl_tess_ring_big_slots counts slots currently holding such a block.  A burst
 * of big draws must keep BOTH small (free draws reuse a warm big slot, at most a
 * couple grow for in-flight work) — test_tess.c asserts it, the regression
 * guard for the ring's memory bound. */
static unsigned long tRingFreedBig;   /* definition: see the r44 state-word block */
unsigned long mgl_tess_ring_freed_big(void) { return tRingFreedBig; }
unsigned mgl_tess_ring_big_slots(void)
{
    unsigned n = 0;
    for (int i = 0; i < MGL_TESS_RING; i++)
        if (tRing[i] && tRingCap[i] > TESS_RING_BIG)
            n++;
    return n;
}

/* deferred ring retires: a completion handler must be added BEFORE the command buffer
 * is committed (Metal asserts otherwise — tessellation_shader_tc_barriers aborts the
 * process through its link-failure path, where the draw bails AFTER a flush), so a
 * buffer that is already committed/scheduled parks here and the next ring poll (every
 * acquire spin) releases its slot once the GPU finished it */
static id<MTLCommandBuffer> tRingPendCb[MGL_TESS_RING];
static int tRingPendSlot[MGL_TESS_RING];
static uint32_t tRingPendGen[MGL_TESS_RING];

/* r44 in-flight retires: retires armed through a completion handler are ALSO
 * polled here from the GL thread, so a slot frees as soon as the GPU has
 * actually finished the command buffer — not when Metal gets around to firing
 * the handler (0.5-2 ms of jitter after completion, several ms under load;
 * measured as the dominant per-draw cost of every GPU-contended GL_PATCHES
 * loop: the ring went dry while 20+ slots sat on finished buffers).  The
 * handler keeps running (it is still the exact, unpolluted wake for workloads
 * that stop drawing); whichever of the two arrives second is a no-op through
 * the generation the retire recorded.  The list is written only by the GL
 * thread (retire-on / this poll); the handler never touches it. */
#define MGL_TESS_FLY 32
static id<MTLCommandBuffer> tRingFlyCb[MGL_TESS_FLY];
static int tRingFlySlot[MGL_TESS_FLY];
static uint32_t tRingFlyGen[MGL_TESS_FLY];
/* retires whose twin had already acted (handler after poll or poll after
 * handler): the generation guard turning what used to be a double free of a
 * possibly re-claimed slot into a no-op — mgl_tess_ring_dup_retires() */
static atomic_uint tRingDupRetires;
unsigned mgl_tess_ring_dup_retires(void)
{
    return atomic_load_explicit(&tRingDupRetires, memory_order_relaxed);
}

/* poll the parked (committed-before-retire) and in-flight (handler-armed)
 * retires: release every slot whose command buffer the GPU has finished.
 * `need_slot` false = the cheap maintenance call at the top of a healthy
 * acquire (parked list only, a dozen loads); true = an acquire that found no
 * free slot also polls the in-flight handlers — that is the pressure path the
 * fly list exists for. */
static void tessRingPendPollFull(bool need_slot)
{
    for (int i = 0; i < MGL_TESS_RING; i++)
        if (tRingPendCb[i])
        {
            id<MTLCommandBuffer> cb = tRingPendCb[i];
            if (cb.status >= MTLCommandBufferStatusCompleted || cb.status == MTLCommandBufferStatusError)
            {
                tRingPendCb[i] = nil;
                tessRingRetireGen(tRingPendSlot[i], tRingPendGen[i]);
            }
        }
    if (!need_slot)
        return;
    for (int i = 0; i < MGL_TESS_FLY; i++)
        if (tRingFlyCb[i])
        {
            id<MTLCommandBuffer> cb = tRingFlyCb[i];
            if (cb.status >= MTLCommandBufferStatusCompleted || cb.status == MTLCommandBufferStatusError)
            {
                tRingFlyCb[i] = nil;
                tessRingRetireGen(tRingFlySlot[i], tRingFlyGen[i]);
            }
        }
}
static void tessRingPendPoll(void) { tessRingPendPollFull(false); }

/* Retire a ring block on an early exit (draw bailed after the block was acquired).
 * `cb` is the command buffer that used the block: the CPU reference path has already
 * committed AND waited on it, so the block is free immediately; the error paths have
 * at most an in-flight / uncommitted buffer, so the slot retires when it completes.
 * Without this, a culled patch (all relevant outer levels <= 0 -> 0 indices, a normal
 * hot path: KHR-GL46 ...single.isolines_tessellation draws level -1 configurations
 * under a PRIMITIVES_GENERATED query) leaked one slot per draw until the ring wedged
 * and every later draw burned the 2 s acquire timeout. */
static void tessRingRetireOn(id<MTLCommandBuffer> cb, int slot)
{
    uint32_t gen;

    if (slot < 0)
        return;
    tessRingPendPoll();
    if (!cb || cb.status >= MTLCommandBufferStatusCompleted ||
        cb.status == MTLCommandBufferStatusError)
    {
        tessRingRetireCurrent(slot);
        return;
    }
    /* the slot is still claimed by the draw that is retiring it: its generation
     * is what every deferred release of this retire must present */
    gen = tessSlotGen(slot);
    if (cb.status >= MTLCommandBufferStatusCommitted)
    {
        /* committed or scheduled: a completion handler would assert (it must be added
         * before commit) — park the retire for the next ring poll instead */
        for (int i = 0; i < MGL_TESS_RING; i++)
        {
            if (!tRingPendCb[i])
            {
                if (tRingTrace())
                    fprintf(stderr, "MGL ring: retire deferred slot=%d (cb committed)\n", slot);
                tRingPendCb[i] = cb;
                tRingPendSlot[i] = slot;
                tRingPendGen[i] = gen;
                return;
            }
        }
        /* the park list is full (eight in-flight draws, none completed): the slot stays
         * busy until one frees up — the acquire spin polls this list */
        return;
    }
    {
        int s = slot;
        uint32_t g = gen;
        /* handler (exact wake, also the only releaser if no further tess draw
         * ever polls) AND the in-flight list (GL-thread poll, frees the slot as
         * soon as the GPU finished instead of when the handler thread runs);
         * the generation makes the second one to arrive a no-op */
        [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
            (void)b;
            tessRingRetireGen(s, g);
        }];
        for (int i = 0; i < MGL_TESS_FLY; i++)
            if (!tRingFlyCb[i])
            {
                tRingFlyCb[i] = cb;
                tRingFlySlot[i] = slot;
                tRingFlyGen[i] = g;
                return;
            }
    }
}

static id<MTLLibrary> tKernelLib;
static id<MTLComputePipelineState> tCountPso, tScanPso, tEmitPso, tArgsPso, tBlobPso, tPermutePso;

static NSMutableDictionary *tVsPsoCache;    /* VS kernel PSO, per program+VAO */
static NSMutableDictionary *tTcsPsoCache;   /* TCS kernel PSO, per program    */


static MGLTessPsoEntry *tessPsoLookup(NSMutableDictionary *cache, NSString *key,
                                      Program *program, id<MTLFunction> fn)
{
    MGLTessPsoEntry *e = cache[key];
    if (e && (e.program != (void *)program || e.fn != fn))
        e = nil;       /* same key, recycled addresses */
    return e;
}

static void tessPsoStore(NSMutableDictionary *cache, NSString *key, Program *program,
                         id<MTLFunction> fn, id<MTLComputePipelineState> pso)
{
    MGLTessPsoEntry *e = cache[key];
    if (!e)
    {
        e = [[MGLTessPsoEntry alloc] init];
        cache[key] = e;
    }
    e.pso = pso;
    e.fn = fn;
    e.program = (void *)program;
}
static id<MTLComputePipelineState> tGatherPso;

/* grown-out scratch buffers stay alive for the lifetime of the process: a command
 * buffer still in flight may reference them, and Metal requires a resource to outlive
 * every encoder that used it */
static id<MTLBuffer> tSmallAttr;     /* small vertex-attribute uploads */
static id<MTLBuffer> tShadowBuf;     /* CPU-converted attribute stream */
static id<MTLBuffer> tRetired[16];
static int tNumRetired;

static bool tessBufferEnsure(id<MTLDevice> device, id<MTLBuffer> __strong *buf, size_t bytes)
{
    if (*buf && (*buf).length >= bytes)
        return true;
    /* metal requires length != 0 */
    if (bytes == 0)
        bytes = 16;
    id<MTLBuffer> nb = [device newBufferWithLength: bytes options: MTLResourceStorageModeShared];
    if (!nb)
        return false;
    if (*buf && tNumRetired < 16)
        tRetired[tNumRetired++] = *buf;
    *buf = nb;
    return true;
}

static id<MTLFunction> tessStageFunction(Program *program, int stage)
{
    Shader *sh = program ? program->shader_slots[stage] : NULL;
    if (!sh || !sh->mtl_data.function)
        return nil;
    return (__bridge id<MTLFunction>)(sh->mtl_data.function);
}

/* The hardware TES function must declare the same control point count N as
 * drawPatches gets (Metal validates the pair; gl_PatchVerticesIn also reads N).
 * With a TCS the count is fixed by GL_TESS_CONTROL_OUTPUT_VERTICES; without one
 * it is GL_PATCH_VERTICES, which can change between draws — recompile the
 * post-tessellation function with the new N when that happens (the rewrite is
 * a textual substitution on the stored MSL, cheap and cached per N). */
void *mgl_tess_hw_tes_function(GLMContext ctx, MGLTessProgram *tp)
{
    int n = tp->has_tcs ? tp->tcs_out_vertices : mgl_tess_patch_vertices_in(ctx, tp);
    if (n < 1) n = 1;
    if (n > 32) n = 32;
    if (tp->tes_hw_function && tp->tes_hw_n == n)
        return tp->tes_hw_function;

    const char *msl = tp->program && tp->program->spirv[_TESS_EVALUATION_SHADER].msl_str ?
                      tp->program->spirv[_TESS_EVALUATION_SHADER].msl_str : NULL;
    if (!msl)
        return NULL;

    char *rewritten = mgl_tess_rewrite_tes_hw(msl, n);
    if (!rewritten)
        return NULL;

    if (getenv("MGL_TESS_DEBUG")) {
        const char *p = strstr(rewritten, "patch(");
        NSLog(@"MGL tess: TES rebuild N=%d -> %.*s", n, p ? 40 : 4, p ? p : "none");
        FILE *df = fopen("/tmp/tes_rebuild.metal", "w");
        if (df) { fputs(rewritten, df); fclose(df); }
    }

    MGLRenderer *self = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    (void)self;
    /* compile through the program domain's path: it specializes the rstate
     * clip-space function constants (y flip / z01 / point size) exactly like
     * the link-time compile did */
    const char *ep = tp->program->shader_slots[_TESS_EVALUATION_SHADER] ?
                     tp->program->shader_slots[_TESS_EVALUATION_SHADER]->entry_point : "main0";
    void *library = NULL, *function = NULL;
    char *errlog = NULL;
    if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, rewritten, (ep && *ep) ? ep : "main0",
                                                 &library, &function, &errlog)) {
        NSLog(@"MGL tess: TES recompile for patch N=%d failed: %s", n, errlog ? errlog : "?");
        free(errlog);
        return NULL;
    }
    free(errlog);
    if (tp->tes_hw_function) CFBridgingRelease(tp->tes_hw_function);
    if (tp->tes_hw_library) CFBridgingRelease(tp->tes_hw_library);
    tp->tes_hw_function = function;
    tp->tes_hw_library = library;
    tp->tes_hw_n = n;
    return tp->tes_hw_function;
}

/* ------------------------------------------------------------------ VS attribute plan
 * The VS of a tessellation pipeline is a compute kernel: its attributes are fed
 * through a compute-pipeline stage input descriptor (MTLStepFunctionThreadPositionInGridX/Y),
 * so the "can Metal fetch the array directly" decisions of the vattrib domain are
 * re-made here from the same VAO state (see vattribGenerateVertexDescriptor).
 */
enum { TESS_FETCH_NONE = 0, TESS_FETCH_DIRECT, TESS_FETCH_SHADOW, TESS_FETCH_CONST };

typedef struct TessAttribSlot {
    int     loc;
    int     fetch;              /* TESS_FETCH_* */
    int     kind;               /* MGL_VA_KIND_* */
    int     mtl_format;         /* MTLVertexFormat for DIRECT */
    size_t  offset;             /* attribute offset inside its binding */
    int     binding;            /* GL vertex binding 0..15 */
    int     buffer_index;       /* Metal buffer index in the compute pipeline */
    int     shadow_buffer;      /* Metal buffer index of the converted stream */
    int     shadow_words;       /* words per element of the converted stream */
    int     binding_step;       /* divisor of the slot's GL binding */
    uint8_t dbl;                /* fp64 word slot: 1 low (comp 0), 2 high (comp 2) */
    int     dbl_words;          /* u32 words this slot fetches */
    int     glloc;              /* GL attribute location feeding this slot (== loc except
                                   for dvec3/dvec4-based inputs, which span two internal
                                   slots per element but one GL location) */
} TessAttribSlot;

typedef struct TessVSPlan {
    TessAttribSlot slot[MAX_ATTRIB_SLOTS];
    int            count;
    int            binding_buffer_index[16];    /* GL binding -> Metal index (-1 none) */
    size_t         binding_stride[16];
    int            binding_step[16];            /* 0 per-vertex, >0 per-instance divisor, -1 constant */
    size_t         binding_offset[16];          /* extra byte offset (first / baseinstance) */
    int            num_buffers;                /* highest Metal index used + 1 */
    int            const_buffer;               /* Metal index of the generic-value buffer */
} TessVSPlan;

/* VS input locations and their base type kinds, from the VS stage reflection list */
static void tessVsInputKinds(GLMContext ctx, Program *program, bool has[MAX_ATTRIBS], int kind[MAX_ATTRIBS])
{
    /* Walk the GL-view vertex module directly.  The resource list carries type
     * ids of the *compiled* module, which for fp64 programs is the software-
     * double-lowered one (u32 word pairs) while spirv[_VERTEX_SHADER].ir keeps
     * the GL view (Float64 attributes); mixing the id spaces misresolves the
     * base type and the fetch format ends up Float4 for a `double` attribute. */
    uint32_t *ir = program->spirv[_VERTEX_SHADER].ir;
    size_t n = program->spirv[_VERTEX_SHADER].size;
    size_t k = 5;

    memset(has, 0, MAX_ATTRIBS * sizeof(bool));
    memset(kind, 0, MAX_ATTRIBS * sizeof(int));
    if (!ir) return;

    while (k + 4 < n)
    {
        uint32_t wc = ir[k] >> 16, op = ir[k] & 0xffff;

        if (wc == 0) break;
        if (op == 59 /* OpVariable */ && ir[k + 3] == SpvStorageClassInput)
        {
            /* OpVariable layout: (result TYPE, result ID, storage class) */
            uint32_t var_id = ir[k + 2], type_id = ir[k + 1];
            const char *name = NULL;
            int loc = -1;
            size_t j = 5;

            while (j + 3 < n)
            {
                uint32_t wc2 = ir[j] >> 16, op2 = ir[j] & 0xffff;
                if (wc2 == 0) break;
                if (op2 == 71 /* OpDecorate */ && ir[j + 1] == var_id && ir[j + 2] == 30 /* Location */)
                    loc = (int)ir[j + 3];
                else if (op2 == 5 /* OpName */ && ir[j + 1] == var_id)
                    name = (const char *)&ir[j + 2];
                if (loc >= 0 && name) break;
                j += wc2;
            }
            /* skip builtins (gl_*): they carry no location anyway */
            if (loc >= 0 && loc < MAX_ATTRIBS &&
                !(name && name[0] == 'g' && name[1] == 'l' && (name[2] == 0 || name[2] == '_')))
            {
                int pass = 0;
                bool done = false;
                int elems = 1, cols = 1, rows = 1, dbl = 0;
                while (!done && pass++ < 6 && type_id)
                {
                    size_t m = 5;
                    while (m + 3 < n)
                    {
                        uint32_t wc3 = ir[m] >> 16, op3 = ir[m] & 0xffff;
                        if (wc3 == 0) break;
                        /* match only type declarations (19..33): instructions
                         * without a result id (OpCapability et al) carry their
                         * operands in the same word slot and false-match small
                         * type ids */
                        if (op3 >= 19 && op3 <= 33 && ir[m + 1] == type_id)
                        {
                            if (op3 == 23 /* OpTypeVector */) { rows = ir[m + 3]; type_id = ir[m + 2]; break; }
                            if (op3 == 32 /* OpTypePointer */) { type_id = ir[m + 3]; break; }
                            if (op3 == 28 /* OpTypeArray */)
                            {
                                /* OpConstant: (result TYPE, result ID, value):
                                 * length id is at ir[m+3], the constant's own
                                 * result id at ir[m+2]; fall back to 1 */
                                size_t c = 5;
                                uint32_t alen = 1;

                                while (c + 3 < n)
                                {
                                    uint32_t wc4 = ir[c] >> 16, op4 = ir[c] & 0xffff;

                                    if (wc4 == 0) break;
                                    if (op4 == 43 /* OpConstant */ && ir[c + 2] == ir[m + 3])
                                    {
                                        alen = ir[c + 3];
                                        break;
                                    }
                                    c += wc4;
                                }
                                elems *= (int)alen;
                                type_id = ir[m + 2];
                                break;
                            }
                            if (op3 == 24 /* OpTypeMatrix: its n is the column count */) { cols = ir[m + 3]; type_id = ir[m + 2]; break; }
                            if (op3 == 22 /* OpTypeFloat */)
                            {
                                dbl = (ir[m + 2] == 64);
                                kind[loc] = dbl ? MGL_VA_KIND_DOUBLE : MGL_VA_KIND_FLOAT;   /* (result, width) */
                                done = true; break;
                            }
                            if (op3 == 21 /* OpTypeInt */)
                            {
                                kind[loc] = ir[m + 3] ? MGL_VA_KIND_INT : MGL_VA_KIND_UINT;
                                done = true; break;
                            }
                            done = true; break;
                        }
                        m += wc3;
                    }
                }
                /* GLSL 4.60 §4.4.1: the attribute occupies elems * cols
                 * locations, a dvecR one ceil(R/2) — and the fp64-lowered MSL
                 * declares one Metal attribute at each of them (SPIRV-Cross
                 * flattens wide doubles, matrices and arrays), so every one
                 * must be covered by the VS stage-input descriptor. */
                if (name || loc >= 0)
                {
                    int locs = elems * cols * (dbl ? (rows + 1) / 2 : 1);

                    if (locs < 1) locs = 1;
                    if (loc + locs > MAX_ATTRIBS) locs = MAX_ATTRIBS - loc;
                    for (int q = 0; q < locs; q++)
                    {
                        has[loc + q] = true;
                        kind[loc + q] = kind[loc];
                    }
                if (getenv("MGL_TESS_DEBUG"))
                    NSLog(@"MGL tess: vs input %s loc=%d elems=%d cols=%d rows=%d dbl=%d -> %d slots",
                          name ? name : "?", loc, elems, cols, rows, dbl,
                          elems * cols * (dbl ? (rows + 1) / 2 : 1));
                }
            }
        }
        k += wc;
    }
    (void)ctx;
}

/* forward declarations (helpers used by the draw path) */
static bool tessBindVSAttributes(GLMContext ctx, id<MTLComputeCommandEncoder> enc, TessVSPlan *plan, GLuint vertex_count);
static bool tessConvertShadowStream(GLMContext ctx, TessAttribSlot *s, GLuint vertex_count);
static uint16_t tessFloatToHalf(float f);
static float tessHalfToFloat(uint16_t h);
static bool tessKernels(GLMContext ctx);
GLsizei tess_idx_used = 0;
static bool tessReadIndicesInto(GLMContext ctx, GLenum type, size_t offset, GLsizei count,
                                GLint basevertex, uint32_t **dst, uint32_t *minv, uint32_t *maxv);

/* format decision of the vattrib domain (MGLRenderer+VertexAttrib.m) */
extern MTLVertexFormat mglVertexFormatForAttrib(GLenum type, GLint size, bool normalized, bool integer);

static bool tessSignedInt(GLenum t)   { return t == GL_BYTE || t == GL_SHORT || t == GL_INT; }
static bool tessUnsignedInt(GLenum t) { return t == GL_UNSIGNED_BYTE || t == GL_UNSIGNED_SHORT || t == GL_UNSIGNED_INT; }

static bool tessBuildVSPlan(GLMContext ctx, Program *program, TessVSPlan *plan,
                            GLint first, GLint baseinstance)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    bool has[MAX_ATTRIBS];
    int kind[MAX_ATTRIBS];
    int next_index;

    memset(plan, 0, sizeof *plan);
    for (int i = 0; i < 16; i++)
        plan->binding_buffer_index[i] = -1;
    plan->const_buffer = -1;
    if (!va)
        return false;

    tessVsInputKinds(ctx, program, has, kind);

    /* uniform/storage buffers occupy 0..k-1 of the compute pipeline; attribute
       bindings continue after them; the VS kernel specials live at 30/31 */
    {
        int k = program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER].count +
                program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STORAGE_BUFFER].count +
                program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER].count;
        next_index = k > 8 ? k : 8;
    }

    for (int loc = 0; loc < MAX_ATTRIB_SLOTS; loc++)
    {
        TessAttribSlot *s = &plan->slot[plan->count];
        MGLVertexAttribFormat *f;
        MGLVertexBinding *b;
        MTLVertexFormat format = MTLVertexFormatInvalid;
        bool aligned;
        GLsizei elem;

        if (!has[loc])
            continue;
        /* `loc` is the internal (Metal) slot; resolve the GL attribute location
           that feeds it (identical except for dvec3/dvec4-based inputs) */
        GLint glloc = loc, cbase = 0;
        mgl_introspect_slot_owner(ctx, program->name, loc, &glloc, &cbase);
        if (glloc < 0) glloc = loc;
        f = &va->fmt[glloc];
        b = &va->binding[f->binding];

        s->loc = loc;
        s->glloc = glloc;
        s->kind = kind[loc];
        s->binding_step = b->divisor;

        if (!(ctx->state.vao->enabled_attribs & (1u << glloc)) || b->buffer == NULL)
        {
            /* disabled array: the current generic value is a constant */
            if (plan->const_buffer < 0)
            {
                plan->const_buffer = next_index++;
                if (next_index >= MGL_TESS_VS_PARAMS)
                {
                    NSLog(@"MGL ERROR: tess: vertex compute pipeline ran out of buffer indices");
                    return false;
                }
            }
            s->fetch = TESS_FETCH_CONST;
            if (s->kind == MGL_VA_KIND_DOUBLE)
            {
                /* word slots of the current double value (dvec3/dvec4 span two) */
                int n = (f->size == GL_BGRA) ? 4 : f->size;
                s->dbl = 1;
                s->dbl_words = (n > 2) ? 2 : n;
                if (n >= 3 && plan->count + 1 < MAX_ATTRIB_SLOTS && loc + 1 < MAX_ATTRIBS)
                {
                    TessAttribSlot *h = &plan->slot[plan->count + 1];
                    *h = *s;
                    h->loc = loc + 1;
                    h->dbl = 2;
                    h->dbl_words = n - 2;
                    plan->count++;
                }
            }
            plan->count++;
            continue;
        }

        elem = mgl_va_element_size(f->type, f->size);
        aligned = (b->stride % 4) == 0 && (b->offset % 4) == 0 && (f->relativeoffset % 4) == 0 &&
                  (b->stride == 0 || f->relativeoffset + elem <= (GLuint)b->stride);

        if (s->kind == MGL_VA_KIND_DOUBLE)
        {
            /* fp64 attribute: fetched as raw u32 words; dvec3/dvec4 occupy a
             * second word slot at loc+1 (mirrors vattribGenerateVertexDescriptor) */
            int n = (f->size == GL_BGRA) ? 4 : f->size;
            GLsizei delem = n * 8;
            s->dbl = 1;
            s->dbl_words = (n > 2) ? 2 : n;
            s->binding = f->binding;
            s->offset = (size_t)b->offset + (size_t)f->relativeoffset;

            if (f->is_long && aligned)
            {
                s->fetch = TESS_FETCH_DIRECT;
                s->mtl_format = (s->dbl_words >= 2) ? MTLVertexFormatUInt4 : MTLVertexFormatUInt2;
            }
            else
            {
                s->fetch = TESS_FETCH_SHADOW;
                s->shadow_words = n * 2;        /* u32 words per element */
            }

            if (plan->binding_buffer_index[s->binding] < 0)
            {
                plan->binding_buffer_index[s->binding] = next_index++;
                plan->binding_stride[s->binding] = b->stride ? (size_t)b->stride : (size_t)delem;
                plan->binding_step[s->binding] = b->divisor;
            }
            if (s->fetch == TESS_FETCH_SHADOW)
                s->shadow_buffer = next_index++;

            if (n >= 3 && plan->count + 1 < MAX_ATTRIB_SLOTS && loc + 1 < MAX_ATTRIBS)
            {
                TessAttribSlot *h = &plan->slot[plan->count + 1];   /* after the low slot */
                *h = *s;
                h->loc = loc + 1;
                h->dbl = 2;
                h->dbl_words = n - 2;
                if (h->fetch == TESS_FETCH_DIRECT)
                    h->offset += 16;
                plan->count++;                                    /* the hi slot counts too */
            }

            if (next_index >= MGL_TESS_VS_PARAMS)
            {
                NSLog(@"MGL ERROR: tess: vertex compute pipeline ran out of buffer indices");
                return false;
            }
            plan->count++;
            continue;
        }

        if (aligned && !f->is_long)
        {
            if (s->kind == MGL_VA_KIND_FLOAT)
            {
                bool unnormalized_int = !f->normalized && (tessSignedInt(f->type) || tessUnsignedInt(f->type));
                if (!f->integer && !unnormalized_int)
                    format = mglVertexFormatForAttrib(f->type, f->size, f->normalized, false);
            }
            else if (s->kind != MGL_VA_KIND_DOUBLE)
            {
                bool sign_ok = (s->kind == MGL_VA_KIND_INT) ? tessSignedInt(f->type) : tessUnsignedInt(f->type);
                if (sign_ok && (f->integer || !f->normalized) && f->size != GL_BGRA)
                    format = mglVertexFormatForAttrib(f->type, f->size, false, true);
            }
        }

        if (format != MTLVertexFormatInvalid)
        {
            s->fetch = TESS_FETCH_DIRECT;
            s->mtl_format = (int)format;
        }
        else
        {
            s->fetch = TESS_FETCH_SHADOW;
            s->shadow_words = 4;     /* float4 / int4 / uint4 converted stream */
        }

        s->binding = f->binding;
        s->offset = (size_t)b->offset + (size_t)f->relativeoffset;

        if (plan->binding_buffer_index[s->binding] < 0)
        {
            plan->binding_buffer_index[s->binding] = next_index++;
            plan->binding_stride[s->binding] = b->stride ? (size_t)b->stride : (size_t)elem;
            plan->binding_step[s->binding] = b->divisor;
        }
        if (s->fetch == TESS_FETCH_SHADOW)
            s->shadow_buffer = next_index++;

        if (next_index >= MGL_TESS_VS_PARAMS)
        {
            NSLog(@"MGL ERROR: tess: vertex compute pipeline ran out of buffer indices");
            return false;
        }
        plan->count++;
    }

    /* fetch offsets: per-vertex bindings start at 'first', per-instance at 'baseinstance'
       (per GL, per-instance attribute indexing includes the base instance) */
    tessVSPlanSetOffsets(plan, first, baseinstance);
    plan->num_buffers = next_index;
    return true;
}

/* re-apply when 'first' becomes known late: indexed draws only learn their real
 * base vertex (the minimum referenced index) from the element-buffer scan */
static void tessVSPlanSetOffsets(TessVSPlan *plan, GLuint first, GLuint baseinstance)
{
    for (int b = 0; b < 16; b++)
    {
        if (plan->binding_buffer_index[b] < 0)
            continue;
        if (plan->binding_step[b] > 0)
            plan->binding_offset[b] = (size_t)baseinstance * plan->binding_stride[b];
        else
            plan->binding_offset[b] = (size_t)first * plan->binding_stride[b];
    }
}

/* absolute vertex capacity the draw can reference: the smallest vertex-rate
 * binding (buffer bytes available after its offset) / stride. The GPU-driven
 * indexed path sizes its vertex-output region with this instead of the scan's
 * min/max — the vertex kernel then indexes its output absolutely ([0..maxv]),
 * so no scan result is ever needed on the CPU. */
static GLuint tessVSCapacity(GLMContext ctx, TessVSPlan *plan, GLsizei count)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    GLuint cap = 0xffffffffu;      /* unbounded until the first vertex-rate binding */
    bool any = false;

    for (int b = 0; b < 16; b++)
    {
        Buffer *buf;
        size_t avail;

        if (plan->binding_buffer_index[b] < 0 || plan->binding_step[b] > 0)
            continue;                     /* unbound or per-instance */
        buf = va->binding[b].buffer;
        if (!buf || !buf->data.buffer_data || plan->binding_stride[b] == 0)
            continue;
        avail = (size_t)buf->size > plan->binding_offset[b]
                    ? (size_t)buf->size - plan->binding_offset[b] : 0;
        {
            GLuint c = (GLuint)(avail / plan->binding_stride[b]);
            if (c < 1) c = 1;
            if (c < cap) cap = c;
            any = true;
        }
    }
    return any ? cap : (GLuint)(count > 0 ? count : 1);
}

/* does the element range contain the primitive-restart value? Reads the CPU-visible
 * storage of the element buffer directly — no GPU sync (restart re-forms patches, so
 * this decision must be made when encoding the draw; everything else stays on the GPU) */
static bool tessRangeHasRestart(GLMContext ctx, GLenum type, size_t offset, GLsizei count, uint32_t restart)
{
    Buffer *ebuf = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
    const unsigned char *p;
    GLsizei i;

    if (!ebuf || !ebuf->data.buffer_data || restart == 0xffffffffu)
        return false;
    p = (const unsigned char *)ebuf->data.buffer_data + offset;
    if (type == GL_UNSIGNED_BYTE)
    {
        for (i = 0; i < count; i++)
            if ((uint32_t)p[i] == restart) return true;
    }
    else if (type == GL_UNSIGNED_SHORT)
    {
        const uint16_t *q = (const uint16_t *)(const void *)p;
        for (i = 0; i < count; i++)
            if ((uint32_t)q[i] == restart) return true;
    }
    else
    {
        const uint32_t *q = (const uint32_t *)(const void *)p;
        for (i = 0; i < count; i++)
            if (q[i] == restart) return true;
    }
    return false;
}

/* ------------------------------------------------------------------ PSO caches */

/* the compute PSO of the VS kernel depends on the program (kernel function) and on the
   current vertex-array layout; the key mirrors xfbPipelineKey */
static NSString *tessVSKey(GLMContext ctx, Program *program, TessVSPlan *plan)
{
    NSMutableString *key = [NSMutableString stringWithFormat: @"%p-", (void *)program];
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        MGLVertexAttribFormat *f = &va->fmt[s->loc];
        MGLVertexBinding *b = &va->binding[f->binding];
        [key appendFormat: @"%d:%d:%x:%d:%u:%d:%u;", s->loc, s->fetch, f->type, f->size,
                         f->normalized, b->stride, b->divisor];
    }
    return key;
}

/* tess perf (r43): the stage-input descriptor a VS-kernel PSO wraps is fully
 * determined by the freshly built plan (plus the function object); the NSString
 * key above costs ~5 us of string churn per draw in the dictionary lookup path.
 * A 64-bit FNV-1a over the plan bytes — binding_offset excluded, it only shifts
 * the bind, never the PSO — plus the program/function identity hit a direct-
 * mapped entry and skip the string build AND the dictionary hash entirely.
 * Never cleared: like tVsPsoCache below, entries live as long as the process
 * (the PSO is referenced by the PSO cache anyway). */
static uint64_t tessVsPlanSig(const TessVSPlan *plan)
{
    const uint8_t *p = (const uint8_t *)plan;
    size_t off = offsetof(TessVSPlan, binding_offset);
    size_t end = off + 16 * sizeof(size_t);
    uint64_t h = 1469598103934665603ull;

    for (size_t i = 0; i < sizeof *plan; i++)
    {
        if (i >= off && i < end)
            continue;
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}
#define TESS_VS_FAST 4
static struct {
    GLMContext ctx;
    Program *program;
    void *fn;
    uint64_t sig;
    id<MTLComputePipelineState> pso;
} tVsPsoFast[TESS_VS_FAST];
static unsigned tVsPsoFastNext;

/* Vertex-kernel variant of a SEPARABLE vs-only pipeline program: compiled at LINK time
 * by compile_stage (the SPIRV-Cross vertex-for-tessellation options cannot be applied to
 * the already-generated plain vertex MSL) and cached in the tess ext state, keyed by the
 * program's name.  The plain vertex function (with the rstate fixup) stays the program's
 * own for normal draws; the kernel carries NO fixup — the pipeline's TES is the last
 * pre-rasterization stage and applies it there. */
static id<MTLFunction> tessPipelineVsKernel(GLMContext ctx, id<MTLDevice> device, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    (void)device;
    if (!S)
        return nil;
    for (unsigned slot = 0; slot < S->vs_kernel_n; slot++)
        if (S->vs_kernel[slot].name == program->name && S->vs_kernel[slot].function)
            return (__bridge id<MTLFunction>)S->vs_kernel[slot].function;
    return nil;
}

static id<MTLComputePipelineState> tessVsPSO(GLMContext ctx, id<MTLDevice> device, Program *program,
                                             TessVSPlan *plan, MGLTessProgram *draw_tp)
{
    id<MTLFunction> fn = tessStageFunction(program, _VERTEX_SHADER);
    MGLTessProgram *vs_tp = mgl_tess_program(ctx, program);
    NSString *key;
    id<MTLComputePipelineState> pso;
    MTLComputePipelineDescriptor *desc;
    MTLStageInputOutputDescriptor *sd;

    /* canonical-record variant of the kernel (link-time recompile) when the
     * SPIRV-Cross output struct was rewritten to the canonical control point record.
     * A pipeline PAIR record (draw_tp) carries its own variant compiled against the
     * pair's layouts — it wins for the program it was built from */
    if (draw_tp && draw_tp->vs_function && draw_tp->pair_vs == program)
        fn = (__bridge id<MTLFunction>)draw_tp->vs_function;
    else if (vs_tp && vs_tp->vs_function && vs_tp->program == program)
        fn = (__bridge id<MTLFunction>)vs_tp->vs_function;
    if ((!fn || fn.functionType != MTLFunctionTypeKernel))
    {
        /* a vs-only program (pipeline draw, or a program bound with glUseProgram
         * next to a pipeline's TCS/TES — GL 4.6 §7.3 stage resolution) keeps the
         * plain vertex function: the tessellation kernel variant comes from the
         * tess domain's cache.  SEPARABLE programs compiled it at link time; a
         * non-separable program's module was cached prepared and the kernel MSL
         * is generated on demand (mgl_tess_vs_kernel_msl inside the pair build —
         * the same lookup mgl_tess_vs_kernel_fn resolves there) */
        id<MTLFunction> kfn = tessPipelineVsKernel(ctx, device, program);

        if (!kfn && MGL_PROGRAM_OBJ(program)->separable != GL_TRUE)
        {
            /* lazy build: force the kernel MSL generation for this program */
            mgl_tess_vs_kernel_msl(ctx, program->name, NULL);
            kfn = tessPipelineVsKernel(ctx, device, program);
        }
        if (kfn && kfn.functionType == MTLFunctionTypeKernel)
            fn = kfn;
    }
    if (!fn || fn.functionType != MTLFunctionTypeKernel)
    {
        NSLog(@"MGL ERROR: tess VS stage function is not a kernel (prog %u fn=%@ type=%ld)",
              program->name, fn, (long)(fn ? fn.functionType : -1));
        return nil;
    }

    /* direct-mapped fast path (see tessVsPlanSig): same program, same kernel
     * function, same plan signature -> the PSO of the previous draw */
    {
        uint64_t sig = tessVsPlanSig(plan);
        void *fnp = (__bridge void *)fn;
        for (unsigned i = 0; i < TESS_VS_FAST; i++)
            if (tVsPsoFast[i].ctx == ctx && tVsPsoFast[i].program == program &&
                tVsPsoFast[i].fn == fnp && tVsPsoFast[i].sig == sig && tVsPsoFast[i].pso)
                return tVsPsoFast[i].pso;
    }

    key = tessVSKey(ctx, program, plan);
    if (tVsPsoCache)
    {
        MGLTessPsoEntry *e = tessPsoLookup(tVsPsoCache, key, program, fn);
        if (e)
        {
            uint64_t sig2 = tessVsPlanSig(plan);
            tVsPsoFast[tVsPsoFastNext].ctx = ctx;
            tVsPsoFast[tVsPsoFastNext].program = program;
            tVsPsoFast[tVsPsoFastNext].fn = (__bridge void *)fn;
            tVsPsoFast[tVsPsoFastNext].sig = sig2;
            tVsPsoFast[tVsPsoFastNext].pso = e.pso;
            tVsPsoFastNext = (tVsPsoFastNext + 1) % TESS_VS_FAST;
            return e.pso;
        }
    }

    desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;

    sd = [MTLStageInputOutputDescriptor new];
    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        int bidx = s->binding;

        if (s->fetch == TESS_FETCH_CONST)
        {
            /* current generic value: 16-byte slot per location, constant step */
            sd.attributes[s->loc].format = (s->dbl) ? ((s->dbl_words >= 2) ? MTLAttributeFormatUInt4 : MTLAttributeFormatUInt2) :
                                           (s->kind == MGL_VA_KIND_INT) ? MTLAttributeFormatInt4 :
                                           (s->kind == MGL_VA_KIND_UINT) ? MTLAttributeFormatUInt4 :
                                           MTLAttributeFormatFloat4;
            sd.attributes[s->loc].offset = (NSUInteger)(s->loc * 16);
            sd.attributes[s->loc].bufferIndex = plan->const_buffer;
            continue;
        }
        if (s->fetch == TESS_FETCH_SHADOW)
        {
            sd.attributes[s->loc].format = (s->dbl) ? ((s->dbl_words >= 2) ? MTLAttributeFormatUInt4 : MTLAttributeFormatUInt2) :
                                           (s->kind == MGL_VA_KIND_INT) ? MTLAttributeFormatInt4 :
                                           (s->kind == MGL_VA_KIND_UINT) ? MTLAttributeFormatUInt4 :
                                           MTLAttributeFormatFloat4;
            /* fp64: both word slots of one attribute share the converted stream,
             * the high half living at byte 16 of each element */
            sd.attributes[s->loc].offset = (s->dbl == 2) ? 16 : 0;
            sd.attributes[s->loc].bufferIndex = s->shadow_buffer;
            continue;
        }

        sd.attributes[s->loc].format = (MTLAttributeFormat)s->mtl_format;
        sd.attributes[s->loc].offset = s->offset;
        sd.attributes[s->loc].bufferIndex = plan->binding_buffer_index[bidx];
    }

    for (int b = 0; b < 16; b++)
    {
        int idx = plan->binding_buffer_index[b];
        if (idx < 0)
            continue;
        sd.layouts[idx].stride = plan->binding_stride[b];
        if (plan->binding_step[b] > 0)
        {
            sd.layouts[idx].stepFunction = MTLStepFunctionThreadPositionInGridY;
            sd.layouts[idx].stepRate = (NSUInteger)plan->binding_step[b];
        }
        else
        {
            sd.layouts[idx].stepFunction = MTLStepFunctionThreadPositionInGridX;
            sd.layouts[idx].stepRate = 1;
        }
    }

    /* constant + converted streams */
    if (plan->const_buffer >= 0)
    {
        sd.layouts[plan->const_buffer].stride = 16;
        sd.layouts[plan->const_buffer].stepFunction = MTLStepFunctionConstant;
        sd.layouts[plan->const_buffer].stepRate = 0;
    }
    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        if (s->fetch != TESS_FETCH_SHADOW)
            continue;
        if (s->dbl)
        {
            /* one double-word element per vertex: 16 bytes for dvec1/dvec2,
             * 32 for dvec3/dvec4 (high half read at byte offset 16) */
            sd.layouts[s->shadow_buffer].stride = (s->shadow_words > 4) ? 32 : 16;
        }
        else
        {
            sd.layouts[s->shadow_buffer].stride = 16;
        }
        if (s->binding_step > 0)
        {
            sd.layouts[s->shadow_buffer].stepFunction = MTLStepFunctionThreadPositionInGridY;
            sd.layouts[s->shadow_buffer].stepRate = (NSUInteger)s->binding_step;
        }
        else
        {
            sd.layouts[s->shadow_buffer].stepFunction = MTLStepFunctionThreadPositionInGridX;
            sd.layouts[s->shadow_buffer].stepRate = 1;
        }
    }
    desc.stageInputDescriptor = sd;

    if (tTrace())
    {
        NSMutableString *dbg = [NSMutableString stringWithString: @"MGL tess: VS stage-input descriptor:"];
        for (int i = 0; i < MAX_ATTRIB_SLOTS; i++)
        {
            MTLAttributeDescriptor *a = sd.attributes[i];
            if (a.format != MTLAttributeFormatInvalid)
                [dbg appendFormat: @" attr[%d]{fmt=%ld,off=%lu,buf=%lu}", i, (long)a.format,
                                    (unsigned long)a.offset, (unsigned long)a.bufferIndex];
        }
        for (int i = 0; i < 31; i++)
        {
            MTLBufferLayoutDescriptor *l = sd.layouts[i];
            if (l.stride)
                [dbg appendFormat: @" layout[%d]{stride=%lu,fn=%ld,rate=%lu}", i, (unsigned long)l.stride,
                                    (long)l.stepFunction, (unsigned long)l.stepRate];
        }
        NSLog(@"%@", dbg);
    }

    NSError *err = nil;
    pso = [device newComputePipelineStateWithDescriptor: desc
                                                      options: MTLPipelineOptionNone
                                                   reflection: nil
                                                        error: &err];
    if (!pso)
    {
        NSLog(@"MGL ERROR: tess vertex kernel PSO failed: %@", err.localizedDescription);
        return nil;
    }
    if (!tVsPsoCache)
        tVsPsoCache = [[NSMutableDictionary alloc] init];
    tessPsoStore(tVsPsoCache, key, program, fn, pso);
    tVsPsoFast[tVsPsoFastNext].ctx = ctx;
    tVsPsoFast[tVsPsoFastNext].program = program;
    tVsPsoFast[tVsPsoFastNext].fn = (__bridge void *)fn;
    tVsPsoFast[tVsPsoFastNext].sig = tessVsPlanSig(plan);
    tVsPsoFast[tVsPsoFastNext].pso = pso;
    tVsPsoFastNext = (tVsPsoFastNext + 1) % TESS_VS_FAST;
    return pso;
}

static id<MTLComputePipelineState> tessTcsPSO(GLMContext ctx, id<MTLDevice> device,
                                              MGLTessProgram *tp, Program *program)
{
    id<MTLFunction> fn = (tp && tp->tcs_function) ? (__bridge id<MTLFunction>)tp->tcs_function
                                                  : tessStageFunction(program, _TESS_CONTROL_SHADER);
    id<MTLComputePipelineState> pso;

    if (!fn || fn.functionType != MTLFunctionTypeKernel)
        return nil;
    if (tTcsPsoCache)
    {
        MGLTessPsoEntry *e = tessPsoLookup(tTcsPsoCache,
                                           [NSString stringWithFormat: @"%p", (void *)program],
                                           program, fn);
        if (e)
            return e.pso;
    }

    NSError *err = nil;
    pso = [device newComputePipelineStateWithFunction: fn error: &err];
    if (!pso)
    {
        NSLog(@"MGL ERROR: tess control kernel PSO failed: %@", err.localizedDescription);
        return nil;
    }
    if (!tTcsPsoCache)
        tTcsPsoCache = [[NSMutableDictionary alloc] init];
    tessPsoStore(tTcsPsoCache, [NSString stringWithFormat: @"%p", (void *)program],
                 program, fn, pso);
    return pso;
}

/* kernels used to expand an indexed draw's vertex list into the linear
 * control-point stream the TCS consumes — reading the GL element buffer's raw bytes
 * directly on the GPU (ubyte/ushort/uint indices, base vertex, primitive restart):
 *   mgl_tess_scan  : atomically min/max the vertex indices (sizes the VS grid) and
 *                    count restarts, into a small record in the per-draw scratch
 *   mgl_tess_gather: expand the indices into the VS output record list
 * Both run in the compute command buffer, before the VS kernel — the VS grid must be
 * sized from the scan, so the scan runs in its own (event-ordered) earlier buffer. */
static id<MTLComputePipelineState> tessScanPSO, tessScanSetupPSO, tessGatherRawPso;

static id<MTLComputePipelineState> tessScanPSOCreate(id<MTLDevice> device)
{
    static NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void mgl_tess_scan(device const uchar* ib [[buffer(0)]],\n"
        "                          constant uint& count [[buffer(1)]],\n"
        "                          constant uint& itype [[buffer(2)]],\n"
        "                          constant uint& base  [[buffer(3)]],\n"
        "                          constant uint& restart [[buffer(4)]],\n"
        "                          device atomic_uint* out [[buffer(5)]],\n"
        "                          uint3 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x >= count) return;\n"
        "    uint v;\n"
        "    if (itype == 0u) v = ib[gid.x];\n"
        "    else if (itype == 1u) v = ((device const ushort*)ib)[gid.x];\n"
        "    else v = ((device const uint*)ib)[gid.x];\n"
        "    if (restart != 0xffffffffu && v == restart) { atomic_fetch_add_explicit(out + 2, 1u, memory_order_relaxed); return; }\n"
        "    v += base;\n"
        "    atomic_fetch_min_explicit(out, v, memory_order_relaxed);\n"
        "    atomic_fetch_max_explicit(out + 1, v, memory_order_relaxed);\n"
        "}\n"
        /* consumes the scan atomics and writes the vertex kernel's parameter block on
         * the GPU. The vertex stage indexes its output ABSOLUTELY (vertex id), so only
         * max+1 matters; vfirst is always 0 on this path and min is not consumed. The
         * kernel is dispatched at the CPU-known capacity and guards on this vcount. */
        "kernel void mgl_tess_scan_setup(device const atomic_uint* out [[buffer(0)]],\n"
        "                                constant uint& instances [[buffer(1)]],\n"
        "                                constant uint& baseinstance [[buffer(2)]],\n"
        "                                constant uint& capacity [[buffer(3)]],\n"
        "                                device uint* vsparams [[buffer(5)]]) {\n"
        "    uint maxv = atomic_load_explicit(out + 1, memory_order_relaxed);\n"
        "    uint vcount = maxv + 1u;                 /* absolute range [0..maxv] */\n"
        "    if (vcount > capacity) vcount = capacity;\n"
        "    vsparams[0] = 0u;                        /* vfirst: absolute indexing */\n"
        "    vsparams[1] = vcount;\n"
        "    vsparams[2] = instances;\n"
        "    vsparams[3] = baseinstance;\n"
        "}\n";

    if (tessScanPSO)
        return tessScanPSO;
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource: src options: nil error: &err];
    if (!lib)
    {
        NSLog(@"MGL ERROR: tess scan library failed: %@", err.localizedDescription);
        return nil;
    }
    tessScanPSO = [device newComputePipelineStateWithFunction: [lib newFunctionWithName: @"mgl_tess_scan"] error:&err];
    if (!tessScanPSO)
        NSLog(@"MGL ERROR: tess scan PSO failed: %@", err.localizedDescription);
    tessScanSetupPSO = [device newComputePipelineStateWithFunction: [lib newFunctionWithName: @"mgl_tess_scan_setup"] error:&err];
    if (!tessScanSetupPSO)
        NSLog(@"MGL ERROR: tess scan-setup PSO failed: %@", err.localizedDescription);
    return tessScanPSO;
}

static id<MTLComputePipelineState> tessGatherPSO(id<MTLDevice> device)
{
    static NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void mgl_tess_gather(device const uchar* srcbuf [[buffer(0)]],\n"
        "                            device const uchar* ib [[buffer(1)]],\n"
        "                            device uint* dst [[buffer(2)]],\n"
        "                            constant uint& words [[buffer(3)]],\n"
        "                            constant uint& base  [[buffer(4)]],\n"
        "                            constant uint& total [[buffer(5)]],\n"
        "                            constant uint& itype [[buffer(6)]],\n"
        "                            constant uint& minv [[buffer(7)]],\n"
        "                            constant uint& restart [[buffer(8)]],\n"
        "                            constant uint& modulo [[buffer(9)]],\n"
        "                            uint3 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x >= total) return;\n"
        "    uint g = (modulo != 0u) ? (gid.x % modulo) : gid.x;\n"
        "    uint v;\n"
        "    if (itype == 0u) v = ib[g];\n"
        "    else if (itype == 1u) v = ((device const ushort*)ib)[g];\n"
        "    else v = ((device const uint*)ib)[g];\n"
        "    if (restart != 0xffffffffu && v == restart) return;\n"
        "    v += base - minv;\n"
        "    for (uint w = 0; w < words; w++)\n"
        "        dst[gid.x * words + w] = ((device const uint*)srcbuf)[v * words + w];\n"
        "}\n";

    if (tGatherPso)
        return tGatherPso;
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource: src options: nil error: &err];
    if (!lib)
    {
        NSLog(@"MGL ERROR: tess gather library failed: %@", err.localizedDescription);
        return nil;
    }
    tGatherPso = [device newComputePipelineStateWithFunction: [lib newFunctionWithName: @"mgl_tess_gather"] error: &err];
    if (!tGatherPso)
        NSLog(@"MGL ERROR: tess gather PSO failed: %@", err.localizedDescription);
    return tGatherPso;
}

/* ------------------------------------------------------------------ per-stage resource binding
 * The kernels read the uniform/storage buffers and textures of their own stage: bind
 * them at the indices SPIRV-Cross assigned (mirrors mapGLBuffersToMTLBufferMap /
 * bindTexturesToComputeEncoder but for an arbitrary stage).
 */
/* the buffers domain's ext hook (lazy MTLBuffer creation), from buffers.c's mtlext() */
static inline struct MGLExtMetal_buffers *mtlextB(GLMContext ctx)
{
    return ctx->mtl_funcs.ext_buffers;
}

static void tessBindStageUniforms(GLMContext ctx, id<MTLComputeCommandEncoder> enc, Program *program, int stage)
{
    static const struct { int spvc; int glt; } types[4] = {
        { SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,    GL_UNIFORM_BUFFER },
        { SPVC_RESOURCE_TYPE_STORAGE_BUFFER,    GL_SHADER_STORAGE_BUFFER },
        { SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,    GL_ATOMIC_COUNTER_BUFFER },
    };

    if (!program)
        return;

    /* default-block uniforms: MGL's introspection exposes them as UNIFORM_CONSTANT
     * resources backed by a per-program per-stage buffer (the "dub") — the render path
     * binds it in mapGLBuffersToMTLBufferMap.  Without it the TCS/TES kernels read
     * zeroed uniforms (gl_TessLevel* written from a uniform tessellated at level 0). */
    {
        SpirvResourceList *list = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];

        for (int r = 0; r < list->count && r < 24; r++)
        {
            Buffer *dub = mgl_introspect_default_uniform_buffer(ctx, program, stage, r);

            if (!dub || !dub->data.buffer_data)
                continue;
            /* 0xffffffff is the established "no Metal binding" sentinel (the stage
             * MSL emission left it there when SPIRV-Cross stripped the block - every
             * default uniform of the stage is dead, e.g. a VERTEX stage that only
             * declares one).  Passing it to setBuffer:atIndex: faults inside the
             * driver (KHR repro: a GL_PATCHES draw whose vertex stage declares an
             * unused uniform); there is no MSL argument to feed, so skip it. */
            if (list->list[r].msl_binding == 0xffffffffu)
                continue;
            if (mtlextB(ctx)->ensure_buffer)
                mtlextB(ctx)->ensure_buffer(ctx, dub);
            if (!dub->data.mtl_data)
                continue;
            [enc setBuffer: (__bridge id<MTLBuffer>)dub->data.mtl_data
                   offset: 0
                  atIndex: (NSUInteger)list->list[r].msl_binding];
        }
    }

    for (int t = 0; t < 3; t++)
    {
        SpirvResourceList *list = &program->spirv_resources_list[stage][types[t].spvc];
        /* buffer_base is indexed by the internal buffer-type index, not by the GLenum */
        int glt = (types[t].glt == GL_UNIFORM_BUFFER) ? _UNIFORM_BUFFER :
                  (types[t].glt == GL_SHADER_STORAGE_BUFFER) ? _SHADER_STORAGE_BUFFER :
                  _ATOMIC_COUNTER_BUFFER;

        for (int r = 0; r < list->count && r < 24; r++)
        {
            /* bind at the buffer index the stage's MSL actually uses (msl_binding from
             * SPIRV-Cross, see compile_stage); a running counter made UBOs and SSBOs
             * overwrite each other's slots */
            /* 0xffffffff = stripped from the MSL (see the dub loop above): there
             * is no argument to feed */
            if (list->list[r].msl_binding == 0xffffffffu)
                continue;
            int index = (int)list->list[r].msl_binding;
            GLuint b = list->list[r].binding;
            if (b >= MAX_BINDABLE_BUFFERS)
                continue;   /* binding point out of range for this target */
            Buffer *buf = ctx->state.buffer_base[glt].buffers[b].buf;
            if (!buf || !buf->data.buffer_data)
                continue;
            /* a GL_UNIFORM_BUFFER's MTLBuffer is created lazily by the render
             * path; the tess kernels read the buffer directly, so force it here
             * (same hook glBindBuffer uses for shader-written targets) */
            if (!buf->data.mtl_data && mtlextB(ctx) && mtlextB(ctx)->ensure_buffer)
                mtlextB(ctx)->ensure_buffer(ctx, buf);
            if (!buf->data.mtl_data)
                continue;
                        [enc setBuffer: (__bridge id<MTLBuffer>)buf->data.mtl_data
                     offset: (NSUInteger)ctx->state.buffer_base[glt].buffers[b].offset
                    atIndex: (NSUInteger)index];
            if (tTrace())
                NSLog(@"MGL tess: bind stage %d type %d r=%d gl_binding=%u msl=%d buf=%p", stage, t, r,
                      b, index, (void *)buf->data.mtl_data);
            /* storage / atomic buffers can be written by the kernel: the CPU has to see it
             * (glMapBufferRange / glBufferSubData sync off this tracking) */
            if (t > 0)
                mgl_buffers_mark_gpu_use(ctx, buf, true);
        }
    }
}

static void tessBindStageTextures(GLMContext ctx, id<MTLComputeCommandEncoder> enc, Program *program, int stage)
{
    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    SpirvResourceList *list = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];

    if (!program)
        return;

    for (int r = 0; r < list->count && r < 16; r++)
    {
        SpirvResource *sres = &list->list[r];
        GLuint unit = sres->binding;
        if (sres->msl_binding == 0xffffffffu)
            continue;   /* SPIRV-Cross stripped the resource: no MSL argument to feed */
        NSUInteger msl = (NSUInteger)sres->msl_binding;
        GLuint elems = sres->array_size ? sres->array_size : 1;

        for (GLuint e = 0; e < elems; e++)
        {
            /* GL sampler semantics: the dimension picks which binding of the unit
             * feeds the resource (mgl_texture_for_sampler, like the render path) */
            Texture *tex = mgl_texture_for_sampler(ctx, unit + e, sres->target);
            if (!tex || !tex->mtl_data)
            {
                /* unbound unit: legal GL (fetches undefined) — feed the shared 1x1
                 * fallback so the dispatch still runs */
                id<MTLTexture> fb = [self2 computeFallbackTexture];
                id<MTLSamplerState> fs = [self2 computeFallbackSampler];
                if (fb) [enc setTexture: fb atIndex: msl + e];
                if (fs) [enc setSamplerState: fs atIndex: msl + e];
                continue;
            }
            [self2 bindMTLTexture: tex];
            id<MTLTexture> texture = [self2 texparamTextureForTex: tex base: (__bridge id<MTLTexture>)tex->mtl_data];
            id<MTLSamplerState> sampler = [self2 texparamSamplerForUnit: unit + e texture: tex];
            if (texture) [enc setTexture: texture atIndex: msl + e];
            /* the MSL sampled-image arguments pair texture(N) with sampler(N) */
            if (sampler) [enc setSamplerState: sampler atIndex: msl + e];
            if (tTrace())
                NSLog(@"MGL tess: bindtex stage %d '%s' unit %u tex %p(name %u)->%p samp %p at %lu dirty=%llx", stage,
                      sres->name ? sres->name : "?", unit + e, (void *)tex, tex->name,
                      (void *)(__bridge void *)texture, (void *)(__bridge void *)sampler,
                      (unsigned long)(msl + e), (unsigned long long)tex->dirty_bits);
        }
    }

    /* storage images (glBindImageTexture): the image unit's level/layer view bound at the
     * MSL slot SPIRV-Cross assigned (msl_binding — texture slots continue after the
     * sampled images).  Mirrors the render path's storage-image binding (MGLRenderer.m /
     * geomImageUnitView, shared with the compute and geometry paths). */
    SpirvResourceList *slist = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];

    for (int r = 0; r < slist->count && r < 16; r++)
    {
        SpirvResource *sres = &slist->list[r];
        int unit = sres->binding;
        GLuint elems = sres->array_size ? sres->array_size : 1;

        for (GLuint e = 0; e < elems; e++)
        {
            /* geom/GL semantics: image-ARRAY elements are independent uniforms
             * (glProgramUniform1i per element; see the render binder in MGLRenderer.m
             * for the same rule) — element e's unit need not be base+e */
            GLuint ue = mgl_res_elem_unit(sres, e);
            if (ue >= TEXTURE_UNITS) continue;
            ImageUnit *iu = &STATE(image_units[ue]);
            Texture *tex = iu->tex;

            if (!tex) continue;             /* unbound unit: legal GL, undefined results */
            [self2 bindMTLTexture: tex];
            if (!tex->mtl_data) continue;
            id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
            texture = [self2 geomImageUnitViewRes: iu base: texture elemClass: sres->elem_class];
            if (sres->msl_binding < 31)
                [enc setTexture: texture atIndex: (NSUInteger)(sres->msl_binding + e)];
        }
    }

    /* geom: those kernels' image ATOMICS run through the sidecar buffers SPIRV-Cross's
     * emulation declared (`<name>_atomic [[buffer(N)]]`, SpirvResource.msl_atomic) —
     * bind + seed them like the compute stage does, or the atomics write an unbound
     * slot and are lost (shader_image_load_store.advanced-allStages-oneImage) */
    [self2 geomBindStageImageAtomicsCompute: program encoder: enc stage: stage];
}

/* ---- index-scan cache ------------------------------------------------------------
 * The scan of an indexed GL_PATCHES draw (min/max referenced vertex, restart count)
 * is a GPU kernel whose result the ring layout needs on the CPU — one wait per draw.
 * The same element range is redrawn every frame in practice, so the results are
 * cached here, keyed on everything they depend on: the element buffer object, its
 * CPU-visible contents generation (gpuaudit serial — bumped by every glBufferSubData /
 * map write / glBufferData), the Metal storage pointer and size, the range (offset,
 * count), the index type and the primitive-restart value/enabled flag.
 * A hit skips the scan kernel, the wait and the 16-byte scratch allocation; for
 * restart ranges the gapless widened index list is cached as an MTLBuffer, so the
 * CPU index read-back and the per-instance widening loop are skipped as well (the
 * gather kernel indexes it with a modulo instead of a per-instance copy). */
#define TESS_SCAN_CACHE 8
#define TESS_SCAN_CACHE_MAX_WIDENED (8u * 1024u * 1024u)

struct TessScanEnt {
    bool valid;
    GLMContext ctx;                 /* buffers are per-context */
    const Buffer *ebuf;
    const void *mtl_data;
    size_t bufsize;
    uint64_t serial;                /* mgl_gpuaudit_content() at cache time */
    size_t offset;
    uint32_t count;
    uint32_t itype;                 /* 0 ubyte / 1 ushort / 2 uint */
    uint32_t restart;               /* 0xffffffffu when restart is disabled */
    /* results */
    uint32_t minv, maxv, restart_count, idx_used;
    void *widened;                  /* retained id<MTLBuffer>, gapless uint32 list */
    bool restart_known;             /* has_restart decided for this content */
    bool has_restart;
    uint64_t used;                  /* LRU clock */
};
static struct TessScanEnt tScanEnt[TESS_SCAN_CACHE];
static uint64_t tScanClock;

/* ---- test-only GPU-time accounting --------------------------------------------
 * Opt-in (mgl_tess_diag_enable, tests/headless/mgl_tess_diag.h): every command
 * buffer the tess path commits or encodes into gets a completed handler that sums
 * its GPU execution time and counts it. Disabled, there is no per-draw cost. */
static bool tDiagEnabled = false;
static double tGpuTimeUs;
static unsigned tCbsCompleted;

/* ---- test-only CPU phase accounting -------------------------------------------
 * Per-phase CPU time of the hardware draw path (setup / index analysis / ring
 * acquire / VS bind / gather / TCS / default factors / blob / ordering / render
 * pass / post-draw commit), read through mgl_tess_phase_us().  Independent of the
 * GPU-time diag above (whose completed handlers change Metal's completion path
 * and would contaminate the numbers): mgl_tess_phase_reset() arms it, the next
 * mgl_tess_diag-based run or process exit leaves it alone. Zero cost when not
 * armed (the disabled begin returns 0 and the end skips). */
enum {
    TESS_PH_SETUP, TESS_PH_INDEX, TESS_PH_RING, TESS_PH_VSBIND, TESS_PH_GATHER,
    TESS_PH_TCS, TESS_PH_NOTCS, TESS_PH_BLOB, TESS_PH_ORDER, TESS_PH_RENDER,
    TESS_PH_POST, TESS_PH_TOTAL, TESS_PHASES
};
static uint64_t tPhaseNs[TESS_PHASES];
static uint64_t tPhaseCnt[TESS_PHASES];
static double tPhaseScale = 0.0;
static bool tPhaseOn = false;

static inline uint64_t tessPhaseBegin(void)
{
    return tPhaseOn ? mach_absolute_time() : 0;
}
static inline void tessPhaseEnd(int p, uint64_t t0)
{
    if (t0)
    {
        if (!tPhaseScale)
        {
            mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            tPhaseScale = (double)tb.numer / (double)tb.denom;
        }
        tPhaseNs[p] += mach_absolute_time() - t0;
        tPhaseCnt[p]++;
    }
}
void mgl_tess_phase_reset(void)      /* also ARMS the accounting */
{
    memset(tPhaseNs, 0, sizeof tPhaseNs);
    memset(tPhaseCnt, 0, sizeof tPhaseCnt);
    tPhaseOn = true;
}
double mgl_tess_phase_us(int p)      /* total us across all draws */
{
    return (p >= 0 && p < TESS_PHASES) ? (double)tPhaseNs[p] * tPhaseScale / 1e3 : 0.0;
}
uint64_t mgl_tess_phase_cnt(int p)
{
    return (p >= 0 && p < TESS_PHASES) ? tPhaseCnt[p] : 0;
}

void mgl_tess_diag_enable(int enable) { tDiagEnabled = enable != 0; }
void mgl_tess_gpu_time_reset(void) { tGpuTimeUs = 0.0; tCbsCompleted = 0; }
double mgl_tess_gpu_time_us(void) { return tGpuTimeUs; }
unsigned mgl_tess_cbs_completed(void) { return tCbsCompleted; }

static void tessTimeCommandBuffer(id<MTLCommandBuffer> cb)
{
    if (!cb || !tDiagEnabled)
        return;
    if (cb.status >= MTLCommandBufferStatusCommitted)
        return;   /* a handler must be added before commit — skip the sample */
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        if (b.GPUEndTime > b.GPUStartTime)
            tGpuTimeUs += (b.GPUEndTime - b.GPUStartTime) * 1e6;
        tCbsCompleted++;
    }];
}

static struct TessScanEnt *tessScanLookup(GLMContext ctx, const Buffer *ebuf, size_t offset,
                                          uint32_t count, uint32_t itype, uint32_t restart)
{
    uint64_t serial = mgl_gpuaudit_content(ctx, (Buffer *)ebuf);

    for (int i = 0; i < TESS_SCAN_CACHE; i++)
    {
        struct TessScanEnt *e = &tScanEnt[i];
        if (!e->valid)
            continue;
        if (e->ctx == ctx && e->ebuf == ebuf && e->mtl_data == ebuf->data.mtl_data &&
            e->bufsize == ebuf->data.buffer_size && e->serial == serial && e->offset == offset &&
            e->count == count && e->itype == itype && e->restart == restart)
        {
            e->used = ++tScanClock;
            if (tTrace())
                NSLog(@"MGL tess: scan cache hit: min=%u max=%u restarts=%u idx_used=%u",
                      e->minv, e->maxv, e->restart_count, e->idx_used);
            return e;
        }
    }
    return NULL;
}

static struct TessScanEnt *tessScanEntry(GLMContext ctx, const Buffer *ebuf, size_t offset,
                                         uint32_t count, uint32_t itype, uint32_t restart)
{
    /* pick the LRU slot (or an invalid one) */
    struct TessScanEnt *best = &tScanEnt[0];
    for (int i = 0; i < TESS_SCAN_CACHE; i++)
    {
        if (!tScanEnt[i].valid)
        {
            best = &tScanEnt[i];
            break;
        }
        if (tScanEnt[i].used < best->used)
            best = &tScanEnt[i];
    }
    if (best->widened)
    {
        CFBridgingRelease(best->widened);
        best->widened = NULL;
    }
    memset(best, 0, sizeof *best);
    best->valid = true;
    best->ctx = ctx;
    best->ebuf = ebuf;
    best->mtl_data = ebuf->data.mtl_data;
    best->bufsize = ebuf->data.buffer_size;
    best->serial = mgl_gpuaudit_content(ctx, (Buffer *)ebuf);
    best->offset = offset;
    best->count = count;
    best->itype = itype;
    best->restart = restart;
    best->used = ++tScanClock;
    return best;
}

/* ------------------------------------------------------------------ the draw */

/* Compile + specialize the TES-output capture kernel (idempotent; shared by the GPU
 * emulated path and the CPU reference path).  The plain TES carries the rasterizer's
 * clip-space fixup (rstate patches the LAST pre-raster stage); the capture must record
 * gl_Position exactly as the shader wrote it, so the kernel is specialized with the
 * fixup's function constants set to their no-op values. */
static bool tessEnsureCapturePso(GLMContext ctx, MGLTessProgram *tp, Program *prog_tes)
{
    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    bool cap_msl_had_fixup = false;

    if (tp->capture_pso)
        return true;
    if (!tp->capture_function)
    {
        char *msl = mgl_tess_capture_msl(ctx, tp);   /* fills cap_members[]/cap_stride */
        char entry[160];

        snprintf(entry, sizeof entry, "%s_cap", prog_tes->spirv[_TESS_EVALUATION_SHADER].entry_point);
        if (msl && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
        {
            void *lib = NULL, *fn = NULL;
            char *err = NULL;

            cap_msl_had_fixup = (strstr(msl, "mgl_rs_vertex_fixup") != NULL);
            if (getenv("MGL_DUMP_MSL") && msl)
                fprintf(stderr, "==== tess capture kernel MSL:\n%s\n==== end\n", msl);
            if (ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, entry, &lib, &fn, &err))
            {
                tp->capture_function = fn;
                tp->capture_library = lib;
                tp->capture_fc_fixup = cap_msl_had_fixup;
            }
            else
                NSLog(@"MGL ERROR: tess: capture kernel failed to compile: %s", err ? err : "?");
            free(err);
        }
        free(msl);
    }
    if (tp->capture_function && !tp->capture_pso)
    {
        NSError *perr = nil;
        id<MTLFunction> cfn = (__bridge id<MTLFunction>)tp->capture_function;
        id<MTLFunction> cspec = cfn;

        if (tp->capture_fc_fixup)
        {
            MTLFunctionConstantValues *cv = [[MTLFunctionConstantValues alloc] init];
            bool t = true, f = false;
            float psz = 1.0f;

            [cv setConstantValue:&t type:MTLDataTypeBool atIndex:MGL_RS_FC_ORIGIN_UPPER_LEFT];
            [cv setConstantValue:&t type:MTLDataTypeBool atIndex:MGL_RS_FC_DEPTH_ZERO_TO_ONE];
            [cv setConstantValue:&t type:MTLDataTypeBool atIndex:MGL_RS_FC_PROGRAM_POINT_SIZE];
            [cv setConstantValue:&psz type:MTLDataTypeFloat atIndex:MGL_RS_FC_POINT_SIZE];
            [cv setConstantValue:&f type:MTLDataTypeBool atIndex:MGL_RS_FC_RASTERIZER_DISCARD];
            cspec = [((__bridge id<MTLLibrary>)tp->capture_library)
                        newFunctionWithName: cfn.name constantValues: cv error: &perr];
        }
        if (cspec)
        {
            id<MTLComputePipelineState> pso = [self2->_device newComputePipelineStateWithFunction: cspec error: &perr];

            if (pso)
                tp->capture_pso = (__bridge_retained void *)pso;
            else
                NSLog(@"MGL ERROR: tess: capture kernel pipeline failed: %@",
                      perr ? perr.localizedDescription : @"?");
        }
        else
            NSLog(@"MGL ERROR: tess: capture kernel specialization failed: %@",
                  perr ? perr.localizedDescription : @"?");
    }
    return tp->capture_pso != NULL;
}

/* Publish a TES-output capture for consumers (geometry stage / transform feedback):
 * valid until the next GL_PATCHES draw; the ring slot is released by the draw's
 * completion, and consumers copy or consume before then. */
static void tessPublishCapture(GLMContext ctx, id<MTLBuffer> ring, MGLTessProgram *tp,
                               size_t recs_off, uint32_t recs_cap,
                               size_t idx_off, uint32_t idx_cap, size_t cnt_off)
{
    struct MGLExtState_tess *S2 = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessCapture *c;

    if (!S2 || !tp->cap_stride)
        return;
    if (tTrace())
        NSLog(@"MGL tess: capture published: recs@%zu (cap %u) idx@%zu cnt@%zu stride %u members %u",
              recs_off, recs_cap, idx_off, cnt_off, tp->cap_stride, (unsigned)tp->cap_members_n);
    c = &S2->capture;
    /* the previous capture's ring object goes away with this publish */
    if (c->buffer_object && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
        ctx->mtl_funcs.ext_program->release_obj(ctx, c->buffer_object);
    c->buffer_object = (void *)CFBridgingRetain(ring);
    c->buffer = ring.contents;      /* CPU address, not the object! */
    c->records_offset = recs_off;
    c->index_offset = idx_off;
    c->count_offset = cnt_off;
    c->stride = tp->cap_stride;
    c->records = recs_cap;
    c->indices = idx_cap;
    c->prim = tp->point_mode ? GL_POINTS :
              (tp->domain == MGL_TESS_DOMAIN_ISOLINES ? GL_LINES : GL_TRIANGLES);
    c->emission_order = false;   /* set by the caller: GPU capture = emission order,
                                  * CPU reference capture = coordinate order */
    c->members = (uint32_t)tp->cap_members_n;
    for (int i = 0; i < tp->cap_members_n && i < MGL_TESS_CAPTURE_MAX; i++)
        c->member[i] = tp->cap_members[i];
    c->serial++;
    c->valid = true;
}

/* Retire the published TES-output capture: the contract is "valid until the next
 * GL_PATCHES draw", and a draw that tessellates no patch (count < GL_PATCH_VERTICES)
 * records NOTHING — leaving the previous capture published made the xfb route repack
 * an earlier draw's records into the current transform feedback object. */
static void tessRetireCapture(GLMContext ctx)
{
    struct MGLExtState_tess *S2 = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessCapture *c;

    if (!S2 || !S2->capture.valid)
        return;
    c = &S2->capture;
    if (c->buffer_object && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
        ctx->mtl_funcs.ext_program->release_obj(ctx, c->buffer_object);
    memset(c, 0, sizeof *c);      /* valid = false, no dangling pointers */
}

- (bool) extTessActive
{
    /* O(1) (glmark2 perf r29): the stage gate — a tess record exists exactly for
     * programs whose successful link carried a TES (mgl_ext_post_link_tess; a
     * relink that drops the TES deletes it).  Called from every state walk
     * (mapBuffersToMTL, bindTexturesToCurrentRenderEncoder, the pipeline
     * descriptor build, compat's fast path), so the per-program record hash
     * walk of mgl_tess_active_program is not paid by non-tessellation programs. */
    return mgl_tess_stage_bound(ctx);
}

/* The per-draw ring-block layout of extTessDrawPatches accumulates block-RELATIVE
 * sub-offsets (need grows from 0), but tessRingAcquire hands out ABSOLUTE slot
 * positions — 0 only for a brand-new slot.  Rebase every sub-offset right after the
 * acquire, before the first bind (setBuffer offsets, ring.contents pointers, the
 * tessellation-factor buffer): without this a draw landing mid-slot — several tess
 * draws per frame, or a reused retired slot, both routine for the CTS — bound its
 * TCS control points / factors / patch data at the slot's START instead, and a
 * multi-patch GL_PATCHES draw tessellated stale ring bytes (KHR-GL46.shader_ballot_
 * tests' 2-patch quad collapsed to one patch).  off_vs_out is the block base and the
 * VS output lives at its start, so it is passed through unchanged. */
#define TESS_REBASE_RING(off_vs_out_) do { \
    off_vsparams += (off_vs_out_); off_scanout += (off_vs_out_); \
    off_tcs_cp += (off_vs_out_); off_tcs_out += (off_vs_out_); \
    off_tcs_patch += (off_vs_out_); off_factors += (off_vs_out_); \
    off_levels_f += (off_vs_out_); off_patch_data += (off_vs_out_); \
    off_gpu_emu += (off_vs_out_); off_capture += (off_vs_out_); \
    off_capture_perm += (off_vs_out_); off_params += (off_vs_out_); \
} while (0)

- (bool) extTessDrawPatches: (bool) indexed type: (GLenum) type offset: (size_t) offset
                      first: (GLint) first count: (GLsizei) count instances: (GLsizei) instances
                basevertex: (GLint) basevertex baseinstance: (GLint) baseinstance
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);

    /* compat CS: the tess pass drives the command buffer directly (its own
     * render/compute encoders, occasional commits); the worker must be idle */
    [self compatCSDrain];

    Program *prog_vs = ctx->state.program, *prog_tcs = NULL, *prog_tes = NULL;
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

    /* this draw's capture supersedes the published one — including when it captures
     * nothing (an early return below: 0 patches, bad vertex setup), which must leave
     * NO capture for consumers to pack */
    tessRetireCapture(ctx);
    /* INPUT patch size: GL_PATCH_VERTICES decides how many vertex-shader outputs form
     * one patch.  With a TCS present gl_PatchVerticesOut (what the TES sees as
     * gl_PatchVerticesIn) is a DIFFERENT number — using it here merged patches when
     * the two disagreed (CTS data_pass_through: GL_PATCH_VERTICES=1, TCS out=2). */
    GLsizei patch_verts_in = ctx->state.ext_rstate ?
        ((struct MGLExtState_rstate *)ctx->state.ext_rstate)->patch_vertices : 3;
    if (patch_verts_in < 1)
        patch_verts_in = 1;
    GLsizei patches, total_patches;
    size_t cp_record = tp->cp_record_size;
    size_t patch_record = tp->tcs_patch_out_size ? tp->tcs_patch_out_size : 16;
    /* the control point record stride: the FULL TCS/VS output record (the TES may
     * only read a subset of it; its offsets were remapped at link time) */
    size_t cp_stride = tp->cp_record_full_size ? tp->cp_record_full_size : cp_record;
    if (cp_stride == 0)
        cp_stride = 16;    /* legal GL: a TCS may write only tess levels (no per-vertex
                              outputs) — the record is then unused, but the gather /
                              tessellator stages still need a stride */
    /* the VS->TCS record stride (what the vertex kernel writes and the gather copies):
     * equal to cp_record_full_size when there is no TCS, but canon.size when there is
     * (one canonical record per interface — tessellation.c).  Sizing the vertex-output
     * region and the gather with cp_record_full_size made the gather region OVERLAP
     * the vertex records' tail whenever the two layouts differ, and multi-patch
     * indexed draws fed the TCS garbage control points (KHR-GL46.shader_ballot_tests'
     * 2-patch quad rendered one collapsed patch). */
    size_t vs_rec = tp->vs_record_size ? tp->vs_record_size : cp_stride;
    uint32_t minv = 0, maxv = 0;
    TessVSPlan plan;
    id<MTLCommandBuffer> compute_cb;
    id<MTLComputeCommandEncoder> cenc;
    id<MTLComputePipelineState> vs_pso, tcs_pso = nil, gather_pso = nil;
    bool need_readback = tp->has_tcs != 0;
    bool use_cpu_reference = tCpuReference();
    /* Counting queries on tessellated output need EXACT vertex counts; the
     * hardware path computes tess factors on the GPU, so its counts are not
     * knowable on the CPU.  Take the exact-count CPU reference path while
     * such a query is active (rare: measurement runs only). */
    if (mgl_queries_tess_counts_active(ctx))
        use_cpu_reference = true;
    bool use_hardware = (tp->domain != MGL_TESS_DOMAIN_ISOLINES) && !tp->point_mode &&
                        !use_cpu_reference && !tp->capture;   /* capture needs the GPU-visible stream */
    if (tTrace())
        NSLog(@"MGL tess: draw: tp=%p capture=%d point_mode=%d domain=%d", (void *)tp,
              tp->capture ? 1 : 0, tp->point_mode ? 1 : 0, (int)tp->domain);
    bool gpu_emulated = false;
    if (tTrace())
        NSLog(@"MGL tess: draw path: cpu_ref=%d hw=%d kernels=%d capture=%d",
              (int)use_cpu_reference, (int)use_hardware, (int)tessKernels(ctx), (int)tp->capture);
    bool idx_mode_widened = false;
    id<MTLBuffer> widened_mtl = nil;   /* cached gapless index list (restart ranges) */
    bool gpu_indexed = false;          /* indexed draw driven entirely by GPU scan results */
    bool have_restart = false;
    size_t off_vsparams = 0, off_scanout = 0;   /* GPU-written VS parameter block + scan atomics */
    size_t off_capture = 0;         /* TES-output capture records (tp->capture) */
    size_t off_capture_perm = 0;    /* capture-only: coordinates in emission order */
    size_t off_levels_f = 0;        /* float copy of the tess levels per patch */
    size_t cap_pub_records = 0, cap_pub_idx = 0, cap_pub_count = 0;   /* published offsets */
    uint32_t cap_pub_recs_cap = 0, cap_pub_idx_cap = 0;               /* and their capacities */
    id<MTLBuffer> tess_gather_ib = nil;
    size_t tess_gather_offset = 0;
    uint32_t tess_gather_itype = 0, tess_gather_minv = 0, tess_gather_restart = 0xffffffffu, tess_gather_count = 0, tess_gather_modulo = 0, tess_gather_base = 0;
    bool tess_gather_raw = false;
    GLuint vcount;
    GLsizei vfirst = first;
    id<MTLBuffer> cp_source;
    size_t cp_source_off = 0;

    /* per-draw scratch: one exclusive ring block, never shared with an in-flight draw
     * and never resized in place (the old singletons were overwritten by the next draw
     * while the GPU still read them). All ranges are offsets into the block. */
    id<MTLBuffer> ring = nil;
    id<MTLCommandBuffer> emu_cb = nil;
    id<MTLComputeCommandEncoder> kenc = nil;
    int ring_slot = -1;
    size_t off_vs_out = 0, off_tcs_cp = 0, off_tcs_out = 0;
    size_t off_tcs_patch = 0, off_factors = 0, off_patch_data = 0, off_gpu_emu = 0;
    size_t off_params = 0;       /* plain-TES parameter words: per-draw, in the ring */
    size_t dbg_coords_off = 0;   /* debug dump only: coordinate stream of the tessellator */
    size_t dbg_emu_off = 0;      /* debug dump only: the emulated scratch block */
    size_t coords_cap_dbg = 0;   /* debug dump only: coordinate capacity (index base) */
    size_t dbg_ctl_off = 0;      /* debug dump only: control words (allocs/totals/args) */
    size_t off_coords = 0, off_indices = 0;
    size_t fstride = mgl_tess_factor_stride(ctx, tp);      /* 8 tri / 12 quad */
    uint64_t ph_total = tessPhaseBegin();
    uint64_t ph_setup = 0, ph_idx = 0, ph_ring = 0, ph_vs = 0, ph_gather = 0;
    uint64_t ph_tcs = 0, ph_notcs = 0, ph_blob = 0, ph_order = 0, ph_render = 0, ph_post = 0;
    ph_setup = ph_total;

    if (!tp || !va || cp_stride == 0)
    {
        if (tTrace())
            NSLog(@"MGL tess: draw bail: tp=%p va=%p cp_stride=%zu",
                  (void *)tp, (const void *)va, cp_stride);
        return false;
    }

    if (patch_verts_in < 1)
        patch_verts_in = 1;
    patches = count / patch_verts_in;             /* trailing incomplete patch is dropped */
    if (tTrace())
        NSLog(@"MGL tess: draw patches=%d (count=%d pv=%d)", (int)patches, (int)count, (int)patch_verts_in);
    if (patches <= 0)
        return true;
    /* r44 size guards: `patches * instances` and `count * instances` are
     * GLsizei x GLsizei — signed overflow above 2^31 is UB, and even a valid
     * product then truncates through every uint32 GPU word and the ring sizing
     * below, sizing the scratch block WRONG (the kernels would write past it:
     * a silent GPU memory corruption, not a failed draw).  A draw this large
     * cannot be executed anyway (the block would exceed Metal's per-buffer
     * limit by orders of magnitude), so drop it; GL generates no error for
     * running out of drawable resources. */
    {
        long long total64 = (long long)patches * (long long)instances;
        long long gather64 = (long long)count * (long long)(instances > 0 ? instances : 1);

        if (total64 > MGL_TESS_MAX_TOTAL_PATCHES || gather64 > 0x7fffffffLL)
        {
            static bool warned = false;
            if (!warned)
            {
                NSLog(@"MGL tess: draw dropped: %lld patches / %lld gathered control points "
                      "(count=%d inst=%d) exceed the GPU scratch budget", total64, gather64,
                      (int)count, (int)instances);
                warned = true;
            }
            return true;
        }
    }
    total_patches = patches * instances;

    if (ctx->state.program_pipeline)
    {
        /* GL 4.6 §7.3: a current program established by glUseProgram() serves the
         * stages it DEFINES; the bound pipeline only fills the rest.  (The CTS
         * tessellation utilities draw with useProgram() while a pipeline from an
         * earlier run is still bound — taking the pipeline's TCS here handed the
         * kernels a DELETED program whose default-uniform list was empty, so the
         * tess-level uniforms read zero and the patch tessellated to nothing:
         * gl_MaxPatchVertices_Position_PointSize.)  The composite a bound pipeline
         * installs during processGLState has name 0 and defines no stages. */
        Program *cur = (ctx->state.program && ctx->state.program->name != 0) ? ctx->state.program : NULL;
        prog_tes = (cur && cur->spirv[_TESS_EVALUATION_SHADER].ir) ? cur
                 : (ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER] ?: ctx->state.program);
        prog_tcs = (cur && cur->spirv[_TESS_CONTROL_SHADER].ir) ? cur
                 : ctx->state.program_pipeline->stage_programs[_TESS_CONTROL_SHADER];
        prog_vs = (cur && cur->spirv[_VERTEX_SHADER].ir) ? cur
                : (ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER] ?: prog_tes);
    }
    else
    {
        prog_tcs = prog_tes = ctx->state.program;
    }

    /* ---- 1. vertex stage: compute kernel over (vertices, instances) ---- */
    if (!tessBuildVSPlan(ctx, prog_vs, &plan, vfirst, baseinstance))
    {
        if (tTrace())
            NSLog(@"MGL tess: draw bail: VS plan");
        return false;
    }
    vs_pso = tessVsPSO(ctx, _device, prog_vs, &plan, tp);
    if (!vs_pso)
    {
        if (tTrace())
            NSLog(@"MGL tess: draw bail: VS PSO");
        return false;
    }

    /* ---- indexed draws: everything on the GPU --------------------------------
     * A scan kernel min/max's the vertex indices (sizes the VS grid) into a scratch
     * record; a wait on that tiny buffer is the only CPU sync in the whole draw
     * (the VS grid dimension must be known at encode time, so the scan needs its own
     * command buffer). The gather kernel then expands the raw element bytes (ubyte/
     * ushort/uint, +base vertex, primitive restart) into the control-point stream. */
    tessPhaseEnd(TESS_PH_SETUP, ph_setup);
    ph_idx = tessPhaseBegin();
    if (indexed)
    {
        id<MTLBuffer> ebuf_mtl = nil;
        uint32_t itype = (type == GL_UNSIGNED_SHORT) ? 1u : (type == GL_UNSIGNED_INT) ? 2u : 0u;
        uint32_t restart = ctx->state.caps.primitive_restart ?
            (type == GL_UNSIGNED_BYTE ? 0xFFu : type == GL_UNSIGNED_SHORT ? 0xFFFFu : 0xFFFFFFFFu) : 0xffffffffu;
        uint32_t esize = itype == 0 ? 1 : (itype == 1 ? 2 : 4);
        id<MTLComputePipelineState> scan_pso = tessScanPSOCreate(_device);
        gather_pso = tessGatherPSO(_device);
        if (!scan_pso || !tessScanSetupPSO || !gather_pso)
        {
            if (tTrace())
                NSLog(@"MGL tess: draw bail: scan/gather PSO");
            return false;
        }
        if (offset > (size_t)ctx->state.vao->element_array.buffer->size ||
            (size_t)count * esize > (size_t)ctx->state.vao->element_array.buffer->size - offset)
             return false;
        {
            Buffer *ebuf = ctx->state.vao->element_array.buffer;
            [self bindMTLBuffer: ebuf];
            ebuf_mtl = (__bridge id<MTLBuffer>)(ebuf->data.mtl_data);
            if (!ebuf_mtl)
                 return false;
        }

        /* ---- index analysis -------------------------------------------------------
         * Restart present in the range? (patches re-form around a restart, so the range
         * must be re-written into a gapless list — the only decision that needs the
         * CPU). The answer is cached per element-buffer content like the round-13 scan
         * results (the gpuaudit content serial invalidates it), so the CPU scan runs at
         * most once per content; otherwise the draw is fully GPU-driven and the scan's
         * min/max never reach the CPU. */
        if (restart != 0xffffffffu)
        {
            Buffer *ebuf = ctx->state.vao->element_array.buffer;
            struct TessScanEnt *se = tessScanLookup(ctx, ebuf, offset, (uint32_t)count, itype, restart);

            if (se && se->restart_known)
                have_restart = se->has_restart;
            else
            {
                mgl_tess_note_restart_check();
                have_restart = tessRangeHasRestart(ctx, type, offset, count, restart);
                if (!se)
                    se = tessScanEntry(ctx, ebuf, offset, (uint32_t)count, itype, restart);
                se->restart_known = true;
                se->has_restart = have_restart;
            }
        }

        if (have_restart)
        {
            Buffer *ebuf2 = ctx->state.vao->element_array.buffer;
            struct TessScanEnt *se = tessScanLookup(ctx, ebuf2, offset, (uint32_t)count, itype, restart);

            if (se && se->widened)
            {
                /* gapless list already cached for this range */
                minv = (GLsizei)se->minv;
                maxv = (GLsizei)se->maxv;
                tess_idx_used = (GLsizei)se->idx_used;
                widened_mtl = (__bridge id<MTLBuffer>)se->widened;
                idx_mode_widened = true;
            }
            else
            {
                /* read the indices once on the CPU (CPU-visible storage — no GPU wait),
                 * drop the restarts — an incomplete patch between restarts is discarded,
                 * like the GL primitive-restart rules for GL_PATCHES — and cache the
                 * gapless list as a Metal buffer the gather kernel indexes with a modulo */
                uint32_t *idx;
                uint32_t kept = 0, lo = 0xffffffffu, hi = 0;
                uint32_t rv = (uint32_t)((int64_t)restart + basevertex);   /* idx[] includes basevertex */
                struct TessScanEnt *ne;

                if (!tessReadIndicesInto(ctx, type, offset, count, basevertex, &idx, &minv, &maxv))
                     return false;
                mgl_tess_note_cpu_index_read();

                for (GLsizei i = 0; i < count; i++)
                {
                    if ((uint32_t)idx[i] == rv)
                    {
                        kept -= kept % (uint32_t)patch_verts_in;  /* drop the partial patch */
                        continue;
                    }
                    idx[kept++] = idx[i];
                }
                kept -= kept % (uint32_t)patch_verts_in;          /* trailing partial patch */
                tess_idx_used = (GLsizei)kept;
                for (uint32_t i = 0; i < kept; i++)
                {
                    if (idx[i] < lo) lo = idx[i];
                    if (idx[i] > hi) hi = idx[i];
                }
                if (lo == 0xffffffffu) { lo = 0; hi = 0; }
                minv = (GLsizei)lo;
                maxv = (GLsizei)hi;

                ne = se ? se : tessScanEntry(ctx, ebuf2, offset, (uint32_t)count, itype, restart);
                ne->restart_known = true;
                ne->has_restart = true;
                ne->minv = lo;                   /* min/max over the kept indices */
                ne->maxv = hi;
                ne->idx_used = kept;
                ne->restart_count = 1;

                if ((size_t)kept * 4 <= TESS_SCAN_CACHE_MAX_WIDENED)
                {
                    id<MTLBuffer> wb = [_device newBufferWithLength: (NSUInteger)kept * 4
                                                                options: 0];
                    if (wb)
                    {
                        for (uint32_t i = 0; i < kept; i++)
                            ((uint32_t *)wb.contents)[i] = idx[i] - lo;
                        ne->widened = (void *)CFBridgingRetain(wb);
                        widened_mtl = wb;
                    }
                }
                free(idx);
                idx_mode_widened = (widened_mtl != nil);
                if (!idx_mode_widened)
                    NSLog(@"MGL tess: gapless index list too large to cache; drawing with restart-split patches");
            }
        }

        vfirst = (GLsizei)minv;
        vcount = maxv - minv + 1;
        if (idx_mode_widened)
        {
            /* patches are formed by the gapless list; the gather reads the cached
             * widened buffer directly (already minv-normalised, uint32) */
            GLsizei inst = instances > 0 ? instances : 1;
            patches = tess_idx_used / patch_verts_in;
            total_patches = patches * inst;

            {
                size_t need = (size_t)vcount * (size_t)inst * vs_rec;   /* vertex kernel output */
                off_tcs_cp = need; need += tessRingAlign((size_t)tess_idx_used * (size_t)inst * vs_rec);
                if (tp->has_tcs)
                {
                    off_tcs_out = need;
                    need += tessRingAlign((size_t)total_patches * tp->tcs_out_vertices * cp_stride);
                    off_tcs_patch = need;
                    need += tessRingAlign((size_t)total_patches * patch_record);
                }
                off_factors = need;
                need += tessRingAlign((size_t)total_patches * fstride);
                /* float copy of the levels (GL: the TES reads back the TCS's exact
                 * floats; the half factor record quantises them) */
                off_levels_f = need;
                need += tessRingAlign((size_t)total_patches * 24);
                /* patch-data blob for every path (see the layout note in the
                 * non-restart branch) */
                off_patch_data = need;
                need += tessRingAlign((size_t)total_patches * tp->tes_patch_in_size);
                if (!use_hardware)
                {
                    off_gpu_emu = need;
                    need += tessRingAlign((size_t)total_patches * (4225 * 16 + 24576 * 4) + 4096);
                }
                /* the TES-output capture records live in the SAME ring block, appended
                 * after the emulated scratch — the block must be sized for them, or the
                 * capture kernel writes past the pool buffer (records for big draws are
                 * megabytes).  The record layout is built on first use; build it now. */
                if (!use_hardware && tp->capture)
                {
                    char *cm = mgl_tess_capture_msl(ctx, tp);
                    free(cm);
                    if (tp->cap_stride && total_patches)
                    {
                        /* one record per EMITTED vertex: the D3D11 layout's worst case
                         * (see the emulated block below for the derivations) */
                        uint32_t vcaps = tp->domain == MGL_TESS_DOMAIN_QUADS ? 4225 :
                                         tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 4160 : 3200;
                        uint32_t icaps = tp->point_mode ? vcaps :
                                         tp->domain == MGL_TESS_DOMAIN_QUADS ? 24576 :
                                         tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 8192 : 18432;
                        off_capture = need;
                        need += tessRingAlign((size_t)total_patches *
                                  (size_t)icaps * (size_t)tp->cap_stride * 4);
                        /* coordinates re-gathered into emission order for the capture */
                        off_capture_perm = need;
                        need += tessRingAlign((size_t)total_patches *
                                  (size_t)icaps * 16);
                    }
                }
                /* the plain TES's parameter words (16 B) live in this draw's ring block:
                 * a shared singleton would be overwritten by the next draw while the GPU
                 * still read it */
                off_params = need;
                need += 16;

                ph_ring = tessPhaseBegin();
                ring = tessRingAcquire(_device, need + 1024, &off_vs_out, &ring_slot, self);
                tessPhaseEnd(TESS_PH_RING, ph_ring);
                if (!ring)
                {
                    NSLog(@"MGL ERROR: tess: scratch ring exhausted");
                     return false;
                }
                TESS_REBASE_RING(off_vs_out);
            }
            /* stash the gather parameters where the compute phase dispatch picks them
             * up: the widened list is minv-normalised already, so base/minv/restart
             * neutral and the modulo indexes it per instance */
            tess_gather_ib = widened_mtl;
            tess_gather_offset = 0;
            tess_gather_itype = 2;
            tess_gather_minv = 0;
            tess_gather_base = 0;
            tess_gather_restart = 0xffffffffu;
            tess_gather_count = (uint32_t)tess_idx_used * (uint32_t)inst;
            tess_gather_raw = false;
            tess_gather_modulo = (uint32_t)tess_idx_used;
            goto indexed_ring_done;
        }

        {
            /* ---- GPU-driven path: the scan's results never reach the CPU ----
             * The vertex kernel indexes its output ABSOLUTELY (vertex id, vfirst = 0),
             * so the ring is sized by the attribute capacity (tessVSCapacity) instead
             * of the scan's min/max. The scan + setup kernels — dispatched below into
             * their own command buffer, before the compute buffer, FIFO-ordered —
             * write the vertex kernel's indirect dispatch grid and its parameter
             * block; nothing is read back and nothing waits. */
            size_t inst = instances > 0 ? (size_t)instances : 1;
            GLuint cap = tessVSCapacity(ctx, &plan, count);
            size_t need = (size_t)cap * inst * vs_rec;       /* vs out (absolute) */

            gpu_indexed = true;
            vfirst = 0;                          /* absolute vertex indexing */
            vcount = (GLsizei)cap;               /* dispatch/grid bound (the GPU clamps vcount) */
            /* the GPU-scan path's vertex kernel output (vcount = cap): sized with
             * the VS->TCS record — NOT cp_stride (see vs_rec above) */
            off_vsparams = need;                 /* the vertex kernel's GPU-written parameter block */
            off_scanout = need + 32;             /* scan atomics (min/max/restarts) */
            need += 64;

            off_tcs_cp = need; need += tessRingAlign((size_t)count * inst * vs_rec);
            if (tp->has_tcs)
            {
                off_tcs_out = need;
                need += tessRingAlign((size_t)total_patches * tp->tcs_out_vertices * cp_stride);
                off_tcs_patch = need;
                need += tessRingAlign((size_t)total_patches * patch_record);
            }
            off_factors = need;
            need += tessRingAlign((size_t)total_patches * fstride);
            /* float copy of the levels (see the widened-index branch) */
            off_levels_f = need;
            need += tessRingAlign((size_t)total_patches * 24);
            /* patch-data blob for every path (see the layout note in the widened branch) */
            off_patch_data = need;
            need += tessRingAlign((size_t)total_patches * tp->tes_patch_in_size);
            if (!use_hardware)
            {
                off_gpu_emu = need;
                need += tessRingAlign((size_t)total_patches * (4225 * 16 + 24576 * 4) + 4096);
            }
            /* capture records: see the widened-index branch above (same in-block sizing
             * rule — the records follow the emulated scratch) */
            if (!use_hardware && tp->capture)
            {
                char *cm = mgl_tess_capture_msl(ctx, tp);
                free(cm);
                if (tp->cap_stride && total_patches)
                {
                    /* one record per EMITTED vertex (see the widened-index branch) */
                    uint32_t vcaps = tp->domain == MGL_TESS_DOMAIN_QUADS ? 4225 :
                                     tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 4160 : 3200;
                    uint32_t icaps = tp->point_mode ? vcaps :
                                     tp->domain == MGL_TESS_DOMAIN_QUADS ? 24576 :
                                     tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 8192 : 18432;
                    off_capture = need;
                    need += tessRingAlign((size_t)total_patches *
                              (size_t)icaps * (size_t)tp->cap_stride * 4);
                    off_capture_perm = need;
                    need += tessRingAlign((size_t)total_patches *
                              (size_t)icaps * 16);
                }
            }
            /* the plain TES's parameter words (16 B) live in this draw's ring block:
             * a shared singleton would be overwritten by the next draw while the GPU
             * still read it */
            off_params = need;
            need += 16;

            ph_ring = tessPhaseBegin();
            ring = tessRingAcquire(_device, need + 1024, &off_vs_out, &ring_slot, self);
            tessPhaseEnd(TESS_PH_RING, ph_ring);
            if (!ring)
            {
                NSLog(@"MGL ERROR: tess: scratch ring exhausted");
                 return false;
            }
            TESS_REBASE_RING(off_vs_out);

            /* scan -> setup: min/max/restart on the GPU, then the vertex kernel's
             * parameter block; committed before the compute buffer (FIFO), never waited on */
            {
                uint32_t init[3] = { 0xffffffffu, 0, 0 };
                uint32_t instu = (uint32_t)inst, basei = (uint32_t)baseinstance, capu = cap;
                id<MTLCommandBuffer> scb = [_commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> senc = [scb computeCommandEncoder];

                memcpy((uint8_t *)ring.contents + off_scanout, init, sizeof init);
                [senc setComputePipelineState: scan_pso];
                [senc setBuffer: ebuf_mtl offset: offset atIndex: 0];
                [senc setBytes: &count length: 4 atIndex: 1];
                [senc setBytes: &itype length: 4 atIndex: 2];
                [senc setBytes: &basevertex length: 4 atIndex: 3];
                [senc setBytes: &restart length: 4 atIndex: 4];
                [senc setBuffer: ring offset: off_scanout atIndex: 5];
                [senc dispatchThreadgroups: MTLSizeMake(((NSUInteger)count + 63) / 64, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
                [senc setComputePipelineState: tessScanSetupPSO];
                [senc setBuffer: ring offset: off_scanout atIndex: 0];
                [senc setBytes: &instu length: 4 atIndex: 1];
                [senc setBytes: &basei length: 4 atIndex: 2];
                [senc setBytes: &capu length: 4 atIndex: 3];
                [senc setBuffer: ring offset: off_vsparams atIndex: 5];
                [senc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
                [senc endEncoding];
                tessTimeCommandBuffer(scb);
                [scb commit];
            }
        }
        /* stash the gather parameters where the compute phase dispatch picks them up:
         * absolute indexing — the gather shifts by base only (minv is unused) */
        tess_gather_ib = ebuf_mtl;
        tess_gather_offset = offset;
        tess_gather_itype = itype;
        tess_gather_minv = 0;
        tess_gather_base = (uint32_t)basevertex;
        tess_gather_restart = restart;
        tess_gather_count = (uint32_t)count * (uint32_t)(instances > 0 ? instances : 1);
        tess_gather_raw = true;
        tess_gather_modulo = 0;
    }
    else
    {
        vcount = (GLuint)count;
        {
            size_t inst = instances > 0 ? (size_t)instances : 1;
            size_t need = (size_t)vcount * inst * vs_rec;                    /* vs out */
            if (tp->has_tcs)
            {
                off_tcs_out = need;
                need += tessRingAlign((size_t)total_patches * tp->tcs_out_vertices * cp_stride);
                off_tcs_patch = need;
                need += tessRingAlign((size_t)total_patches * patch_record);
            }
            off_factors = need;
            need += tessRingAlign((size_t)total_patches * fstride);
            /* float copy of the levels (see the widened-index branch) */
            off_levels_f = need;
            need += tessRingAlign((size_t)total_patches * 24);
            /* patch-data blob for every path (see the indexed branch above) */
            off_patch_data = need;
            need += tessRingAlign((size_t)total_patches * tp->tes_patch_in_size);
            if (!use_hardware)
            {
                off_gpu_emu = need;
                need += tessRingAlign((size_t)total_patches * (4225 * 16 + 24576 * 4) + 4096);
            }
            /* capture records: see the indexed branch above (same in-block sizing rule) */
            if (!use_hardware && tp->capture)
            {
                char *cm = mgl_tess_capture_msl(ctx, tp);
                free(cm);
                if (tp->cap_stride && total_patches)
                {
                    /* one record per EMITTED vertex (see the widened-index branch) */
                    uint32_t vcaps = tp->domain == MGL_TESS_DOMAIN_QUADS ? 4225 :
                                     tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 4160 : 3200;
                    uint32_t icaps = tp->point_mode ? vcaps :
                                     tp->domain == MGL_TESS_DOMAIN_QUADS ? 24576 :
                                     tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 8192 : 18432;
                    off_capture = need;
                    need += tessRingAlign((size_t)total_patches *
                              (size_t)icaps * (size_t)tp->cap_stride * 4);
                    off_capture_perm = need;
                    need += tessRingAlign((size_t)total_patches *
                              (size_t)icaps * 16);
                }
            }

            /* the plain TES's parameter words (16 B) live in this draw's ring block:
             * a shared singleton would be overwritten by the next draw while the GPU
             * still read it */
            off_params = need;
            need += 16;

            ph_ring = tessPhaseBegin();
            ring = tessRingAcquire(_device, need + 1024, &off_vs_out, &ring_slot, self);
            tessPhaseEnd(TESS_PH_RING, ph_ring);
            if (!ring)
            {
                NSLog(@"MGL ERROR: tess: scratch ring exhausted");
                 return false;
            }
            TESS_REBASE_RING(off_vs_out);
        }
    }
    indexed_ring_done: ;   /* the widened-index CPU fallback skips the raw gather */

    /* ---- 2. compute phase (VS kernel, optional gather, optional TCS kernel) ---- */
    tessPhaseEnd(TESS_PH_INDEX, ph_idx);   /* the non-indexed branch is folded into it */
    [self endRenderEncoding];

    /* indexed draws learned their real base vertex (min referenced index) from the
     * CPU-side restart analysis above — the vertex-rate attribute fetch offsets must
     * follow it. (The GPU-driven path keeps vfirst = 0: absolute indexing.) */
    if (indexed && idx_mode_widened)
        tessVSPlanSetOffsets(&plan, (GLuint)vfirst, (GLuint)baseinstance);

    bool debug_wait = tTrace() && use_hardware;
    /* geom: realize the tessellation stages' textures while no encoder is open —
     * tessBindStageTextures binds them inside the kernels' compute encoder, and a
     * deferred upload there opens a blit encoder Metal forbids (crash fixed by this
     * pre-pass; see geomRealizeStageTextures).  This must ALSO run before compute_cb is
     * picked: geomRealizeStageTextures commits+waits when it uploaded pending CPU data,
     * and a compute_cb captured before that would be a committed buffer (Metal forbids
     * encoding into it) — with the uploads ordered before the kernels, the kernels'
     * image stores are no longer clobbered by the deferred upload. */
    [self geomRealizeStageTextures: prog_vs stage: _VERTEX_SHADER];
    if (prog_tcs) [self geomRealizeStageTextures: prog_tcs stage: _TESS_CONTROL_SHADER];
    if (prog_tes) [self geomRealizeStageTextures: prog_tes stage: _TESS_EVALUATION_SHADER];
    /* geom: create + seed the stages' image-atomic sidecars while no encoder is open
     * (the seed is its own compute pass, so it must precede the kernels' encoder;
     * geomBindStageImageAtomicsCompute binds them inside tessBindStageTextures) */
    [self geomPrepareStageImageAtomicsCompute: prog_vs stage: _VERTEX_SHADER];
    if (prog_tcs) [self geomPrepareStageImageAtomicsCompute: prog_tcs stage: _TESS_CONTROL_SHADER];
    if (prog_tes) [self geomPrepareStageImageAtomicsCompute: prog_tes stage: _TESS_EVALUATION_SHADER];

    if ((need_readback && !use_hardware) || debug_wait)
    {
        /* this draw gets its own command buffer (committed+waited below for the factor
         * readback).  Pending work on the shared _currentCommandBuffer — deferred
         * texture uploads, generateMipmap blits — must not end up AFTER it: the queue
         * runs buffers in commit order, so commit the shared buffer first
         * (KHR-GL46.texture_gather.gather-tesselation-shader: the TCS kernel read an
         * uncommitted level-0 upload as zeros).  No wait: pure ordering. */
        if (_currentCommandBuffer && _currentCommandBuffer.status <= MTLCommandBufferStatusEnqueued)
        {
            [self endRenderEncoding];
            [self commitCommandBufferWithAGXRecovery:_currentCommandBuffer];
        }
        compute_cb = [_commandQueue commandBuffer];
        tessTimeCommandBuffer(compute_cb);
    }
    else
    {
        /* kernels and the render pass share one command buffer: Metal executes encoders
           in order, so the TCS output is complete before the render pass reads it */
        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
            _currentCommandBuffer = [_commandQueue commandBuffer];
        compute_cb = _currentCommandBuffer;
        tessTimeCommandBuffer(compute_cb);   /* shared with the render pass: fires at frame commit */
    }

    cenc = [compute_cb computeCommandEncoder];

    /* VS kernel */
    {
        uint32_t params[4] = { (uint32_t)vfirst, (uint32_t)vcount, (uint32_t)instances,
                               (uint32_t)baseinstance };

        if (tTrace())
            NSLog(@"MGL tess: VS kernel dispatch %u x %d threads%s", vcount, instances,
                  gpu_indexed ? " (indirect, GPU-decided)" : "");
ph_vs = tessPhaseBegin();
[cenc setComputePipelineState: vs_pso];
        tessBindStageUniforms(ctx, cenc, prog_vs, _VERTEX_SHADER);
        tessBindStageTextures(ctx, cenc, prog_vs, _VERTEX_SHADER);
        if (prog_vs->uses_draw_params)
        {
            /* gl_DrawID / gl_BaseVertex in a tessellation pipeline: the vertex
             * kernel's own draw-parameters buffer (the raster index is taken
             * by the VS output buffer here) */
            uint32_t dp[2] = { 0, 0 };
            if (ctx->state.ext_tess)
            {
                dp[0] = ctx->state.ext_tess->draw_id;
                dp[1] = (uint32_t)ctx->state.ext_tess->draw_base_vertex;
            }
            [cenc setBytes: dp length: sizeof dp atIndex: MGL_DRAW_PARAMS_KERNEL];
        }
        if (!tessBindVSAttributes(ctx, cenc, &plan, (GLuint)vcount))
        {
            [cenc endEncoding];
            tessRingRetireOn(compute_cb, ring_slot);
             return false;
        }
        [cenc setBuffer: ring offset: off_vs_out atIndex: MGL_TESS_VS_OUT_BUF];
        /* the vertex kernel fetches attributes through the compute stage-in
         * descriptor: Metal requires the fetched index region (X vertices,
         * Y instances) or the dispatch is dropped */
        [cenc setStageInRegion: MTLRegionMake2D(0, 0, (NSUInteger)vcount, (NSUInteger)instances)];
        if (gpu_indexed)
        {
            /* GPU-driven indexed draw: the scan-setup kernel wrote the parameter
             * block (real vcount) — the grid is dispatched at the CPU-known upper
             * bound (the attribute capacity) and every thread beyond the GPU-decided
             * vcount returns before writing, so no scan result reaches the CPU.
             * (Indirect dispatch is not used: the stage-in attribute fetch of the
             * vertex kernel does not follow an indirect grid.) */
            [cenc setBuffer: ring offset: off_vsparams atIndex: MGL_TESS_VS_PARAMS];
            [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)vcount, (NSUInteger)instances, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
        }
        else
        {
            /* inline: a shared CPU-rewritten buffer races draws still in flight */
            [cenc setBytes: params length: sizeof params atIndex: MGL_TESS_VS_PARAMS];
            [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)vcount, (NSUInteger)instances, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
        }
    }

    tessPhaseEnd(TESS_PH_VSBIND, ph_vs);
    cp_source = ring;
    cp_source_off = off_vs_out;

    /* gather kernel (indexed): linear control-point stream per index occurrence —
     * reads the GL element buffer's raw bytes directly (base vertex, no restart) */
    ph_gather = tessPhaseBegin();
    if (indexed && tess_gather_raw && tess_gather_ib)
    {
        /* gather words = the VS->TCS record: the source records are the vertex
         * kernel's output and the destination stream is what the TCS reads — one
         * canonical record per interface (vs_rec; NOT cp_stride = the TCS-out
         * record the TES reads) */
        uint32_t gparams[6] = { (uint32_t)(vs_rec / 4), (uint32_t)basevertex, tess_gather_count,
                                tess_gather_itype, tess_gather_minv, tess_gather_restart };
        NSUInteger groups = ((NSUInteger)tess_gather_count + 63) / 64;

        [cenc setComputePipelineState: gather_pso];
        [cenc setBuffer: ring offset: off_vs_out atIndex: 0];
        [cenc setBuffer: tess_gather_ib offset: tess_gather_offset atIndex: 1];
        [cenc setBuffer: ring offset: off_tcs_cp atIndex: 2];
        for (int gi = 0; gi < 6; gi++)
            [cenc setBytes: &gparams[gi] length: sizeof(uint32_t) atIndex: 3 + gi];
        [cenc setBytes: &tess_gather_modulo length: sizeof(uint32_t) atIndex: 9];
        [cenc dispatchThreadgroups: MTLSizeMake(groups, 1, 1) threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];

        cp_source = ring;
        cp_source_off = off_tcs_cp;
    }
    else if (indexed)
    {
        /* one dispatch for both index modes; every kernel parameter is set explicitly
         * (the raw path used to leave itype/minv/restart unbound, relying on whatever
         * the encoder carried over):
         *   raw:     reads the element buffer, shifts by base - minv (both include
         *            basevertex, so v = raw_index - min_raw_index)
         *   widened: reads the cached gapless uint32 list (already minv-normalised)
         *            with a per-instance modulo */
        uint32_t gparams[7] = { (uint32_t)(vs_rec / 4), tess_gather_base, tess_gather_count,
                                tess_gather_itype, tess_gather_minv, tess_gather_restart,
                                tess_gather_modulo };
        NSUInteger groups = ((NSUInteger)tess_gather_count + 63) / 64;

        [cenc setComputePipelineState: gather_pso];
        [cenc setBuffer: ring offset: off_vs_out atIndex: 0];
        [cenc setBuffer: tess_gather_ib offset: tess_gather_offset atIndex: 1];
        [cenc setBuffer: ring offset: off_tcs_cp atIndex: 2];
        for (int gi = 0; gi < 7; gi++)
            [cenc setBytes: &gparams[gi] length: sizeof(uint32_t) atIndex: 3 + gi];
        [cenc dispatchThreadgroups: MTLSizeMake(groups, 1, 1) threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
        if (tTrace())
            NSLog(@"MGL tess: gather: gc=%u base=%u itype=%u minv=%u restart=%x modulo=%u widened=%d vfirst=%d vcount=%u",
                  tess_gather_count, tess_gather_base, tess_gather_itype, tess_gather_minv,
                  tess_gather_restart, tess_gather_modulo, idx_mode_widened, vfirst, vcount);

        cp_source = ring;
        cp_source_off = off_tcs_cp;
    }

    tessPhaseEnd(TESS_PH_GATHER, ph_gather);

    /* TCS kernel */
    ph_tcs = tessPhaseBegin();
    if (tp->has_tcs)
    {
        /* [0] = gl_PatchVerticesIn, [1] = total patches (instances included; the
         * record writes index by the global patch), [2] = patches per instance —
         * gl_PrimitiveID restarts at 0 for each instance (see tcs_patch_primid) */
        uint32_t iparams[3] = { (uint32_t)patch_verts_in, (uint32_t)total_patches,
                                (uint32_t)patches };
        NSUInteger threads = (NSUInteger)total_patches * (NSUInteger)tp->tcs_out_vertices;

        /* the triangle-domain kernel writes 8-byte factor records (recompiled at link:
         * SPIRV-Cross types the buffer as the quad struct) */
        tcs_pso = tessTcsPSO(ctx, _device, tp, prog_tcs);
        if (!tcs_pso)
        {
            [cenc endEncoding];
            tessRingRetireOn(compute_cb, ring_slot);
             return false;
        }

        [cenc setComputePipelineState: tcs_pso];
        if (tTrace())
            fprintf(stderr, "MGL tess: tcs dispatch enc=%p pso=%p\n",
                    (void *)(__bridge void *)cenc, (void *)(__bridge void *)tcs_pso);
        tessBindStageUniforms(ctx, cenc, prog_tcs, _TESS_CONTROL_SHADER);
        tessBindStageTextures(ctx, cenc, prog_tcs, _TESS_CONTROL_SHADER);
        [cenc setBuffer: ring offset: cp_source_off atIndex: MGL_TESS_TCS_IN];
        [cenc setBuffer: ring offset: off_tcs_out atIndex: MGL_TESS_TCS_OUT];
        [cenc setBuffer: ring offset: off_tcs_patch atIndex: MGL_TESS_TCS_PATCH_OUT];
        [cenc setBuffer: ring offset: off_factors atIndex: MGL_TESS_TCS_FACTORS];
        if (tp->tcs_levels_float)
            [cenc setBuffer: ring offset: off_levels_f atIndex: 30];   /* GL's float level copy */
        [cenc setBytes: iparams length: sizeof iparams atIndex: MGL_TESS_TCS_PARAMS];
        /* one threadgroup per PATCH: barrier() in a TCS synchronises the patch's
         * invocations, which requires them to share a threadgroup — with one thread
         * per group (the old shape) glslang's threadgroup_barrier(mem_device|thread)
         * synchronised nothing and post-barrier cross-invocation reads saw unwritten
         * records (tessellation_shader_tc_barriers read [0,0,0,0]).  A group of
         * exactly tcs_out_vertices threads never pads, so the kernel's bounds guard
         * cannot strand non-invocation threads before the barrier (a deadlock). */
        [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)total_patches, 1, 1)
              threadsPerThreadgroup: MTLSizeMake((NSUInteger)tp->tcs_out_vertices, 1, 1)];
        if (getenv("MGL_TESS_DUMP_CAP"))
        {
            /* one-off diagnostics: printed later, once emu_cb completed below */
        }

        if (tTrace() && gpu_indexed)
        {
            /* debug: dump the GPU-written records once the compute buffer completes */
            __block const uint8_t *base = (const uint8_t *)ring.contents;
            __block size_t dv = off_vsparams, ds = off_scanout, dvo = off_vs_out;
            if (compute_cb.status < MTLCommandBufferStatusCommitted)
            [compute_cb addCompletedHandler:^(id<MTLCommandBuffer> b) {   /* handler must precede commit */
                const uint32_t *v = (const uint32_t *)(base + dv);
                const uint32_t *so = (const uint32_t *)(base + ds);
                const float *vo = (const float *)(base + dvo);
                (void)b;
                NSLog(@"MGL tess: gpu-indexed: scanout min=%u max=%u restarts=%u | "
                      @"vsparams vfirst=%u vcount=%u inst=%u base=%u | pos: %@",
                      so[0], so[1], so[2], v[0], v[1], v[2], v[3],
                      [NSString stringWithFormat: @"(%.1f,%.1f) (%.1f,%.1f) (%.1f,%.1f) (%.1f,%.1f) (%.1f,%.1f) (%.1f,%.1f)",
                        vo[4], vo[5], vo[12], vo[13], vo[20], vo[21], vo[28], vo[29], vo[36], vo[37], vo[44], vo[45]]);
            }];
        }

        cp_source = ring;
        cp_source_off = off_tcs_out;
    }
    else
    {
        ph_notcs = tessPhaseBegin();
        /* no TCS: the default tessellation levels are CPU-known constants; the control
           points pass through from the vertex kernel output (patch p control point i
           lives at [instance * count + p * patch_verts_in + i] == [p * patch_verts_in + i]) */
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        size_t fhalfs = fstride / 2;                 /* 4 triangles / 6 quads+isolines */
        int ne = (fhalfs == 4) ? 3 : 4;              /* edge levels of the factor record */
        uint16_t *hf = (uint16_t *)((uint8_t *)ring.contents + off_factors);
        float *lvf = (float *)((uint8_t *)ring.contents + off_levels_f);
        float f6[6];

        f6[0] = rs->patch_default_outer[0];
        f6[1] = rs->patch_default_outer[1];
        f6[2] = rs->patch_default_outer[2];
        f6[3] = rs->patch_default_outer[3];
        f6[4] = rs->patch_default_inner[0];
        f6[5] = rs->patch_default_inner[1];

        /* the ring block is written directly (no per-draw heap array): the half
         * factor records the GPU tessellator / drawPatches and the blob kernel
         * read, and the float copy of the same levels (GL's exact-precision
         * contract) — constants written before dispatch, no GPU dependency */
        for (int p = 0; p < total_patches; p++)
        {
            for (int i = 0; i < ne; i++)
                hf[(size_t)p * fhalfs + i] = tessFloatToHalf(f6[i]);
            for (int i = ne; i < (int)fhalfs; i++)
                hf[(size_t)p * fhalfs + i] = tessFloatToHalf(f6[4 + (i - ne)]);
            memcpy(lvf + (size_t)p * 6, f6, 6 * sizeof(float));
        }

        if (use_cpu_reference)
        {
            /* reference path: the CPU blob build and tessellator stay here; the GPU
             * paths take both from the blob kernel / the GPU tessellator below */
            float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));
            for (int p = 0; p < total_patches; p++)
                memcpy(factors + (size_t)p * 6, f6, 6 * sizeof(float));
            mgl_tess_build_patch_blob(ctx, tp, NULL, 0, factors, (unsigned)total_patches);
            mgl_tess_generate(ctx, tp, factors, (unsigned)total_patches);
            free(factors);
        }
        /* the GPU paths skip mgl_tess_note_counts (a full CPU tessellator count
         * pass per patch): its only consumer is a count query, and an active
         * tess count query forces use_cpu_reference above, which just ran it */
    }

    /* patch-data blob on the GPU, replacing the CPU blob build of the emulated and the
     * TCS-less hardware path: one thread per patch, after the TCS kernel in the same
     * encoder. FIFO command-buffer order (and the render pass's event wait) hand the
     * result to the tessellator kernels and the render pass without any CPU round-trip.
     * The hardware path with a TCS does not need it: the TES reads the TCS patch output
     * struct directly. */
    /* the blob kernel must run from the FIRST capture draw: tessKernels() compiles the
     * static kernel set lazily (it is otherwise first needed by the emulated path below,
     * which runs AFTER this dispatch — on a first draw the blob PSO did not exist yet
     * and the TES patch-in blob stayed zero, zeroing every per-patch capture member) */
    tessPhaseEnd(TESS_PH_TCS, ph_tcs);
    tessPhaseEnd(TESS_PH_NOTCS, ph_notcs);

    ph_blob = tessPhaseBegin();
    if (!use_cpu_reference && !(use_hardware && tp->has_tcs) && tp->tes_patch_in_size &&
        (tBlobPso || tessKernels(ctx)))
    {
        struct MGLTessBlobCfg cfg;

        memset(&cfg, 0, sizeof cfg);
        cfg.patches = (uint32_t)total_patches;
        cfg.tes_stride = (uint32_t)tp->tes_patch_in_size;
        cfg.tcs_stride = (uint32_t)patch_record;
        cfg.fhalfs = (uint32_t)(fstride / 2);
        cfg.triangles = (tp->domain == MGL_TESS_DOMAIN_TRIANGLES) ? 1u : 0u;
        cfg.n = (uint32_t)(tp->blob_map_n < MGL_TESS_BLOB_MAX ? tp->blob_map_n : MGL_TESS_BLOB_MAX);
        for (int e = 0; e < (int)cfg.n; e++)
        {
            cfg.kind[e] = tp->blob_kind[e];
            cfg.dst[e] = tp->blob_dst[e];
            cfg.src[e] = tp->blob_src[e];
            cfg.size[e] = tp->blob_size[e];
        }
        [cenc setComputePipelineState: tBlobPso];
        [cenc setBytes: &cfg length: sizeof cfg atIndex: 0];
        [cenc setBuffer: ring offset: (tp->has_tcs ? off_tcs_patch : 0) atIndex: 1];   /* only read with a TCS */
        [cenc setBuffer: ring offset: off_factors atIndex: 2];
        [cenc setBuffer: ring offset: off_patch_data atIndex: 3];
        [cenc setBuffer: ring offset: off_levels_f atIndex: 4];   /* GL's exact float levels */
        [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)total_patches, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
        if (tTrace())
            NSLog(@"MGL tess: blob kernel: %u patches, map n=%u, tes stride %u, tcs stride %u",
                  cfg.patches, cfg.n, cfg.tes_stride, cfg.tcs_stride);
    }

    [cenc endEncoding];
    tessPhaseEnd(TESS_PH_BLOB, ph_blob);

    /* GPU-side ordering for EVERY non-reference path (no CPU wait): the compute buffer
     * signals an event the render buffer waits for; the factors and patch data stay on
     * the GPU. Only MGL_TESS_CPU commits and waits on the CPU. */
    ph_order = tessPhaseBegin();
    bool cpu_reference = use_cpu_reference;
    if (cpu_reference)
    {
        [compute_cb commit];
        [compute_cb waitUntilCompleted];
        if (compute_cb.error)
            NSLog(@"MGL tess: compute cb error %@", compute_cb.error);

        /* a fresh render command buffer: the reference path's wait may race the
         * renderer's flush, and Metal forbids encoding on a committed buffer — the
         * stale encoder must go and the buffer it belongs to must be committed, or
         * everything already encoded on it (glClear, earlier draws) is lost */
        if (_currentCommandBuffer && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
        {
            [self endRenderEncoding];
            [_currentCommandBuffer commit];
        }
        else if (_currentCommandBuffer)
            [self endRenderEncoding];
        _currentCommandBuffer = [_commandQueue commandBuffer];
        ctx->state.dirty_bits = DIRTY_ALL;
    }
    else if (use_hardware && compute_cb == _currentCommandBuffer &&
             _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
    {
        /* tess perf (r42): the compute encoders above were appended to the SAME
         * command buffer the render pass will use — Metal executes the encoders of
         * one command buffer strictly in encoding order, so the VS/TCS/blob kernels
         * are complete before the render pass reads their output.  No event, no
         * commit, no fresh command buffer: the old per-draw signal/wait pair also
         * committed every earlier draw on the shared buffer and forced a full
         * pipeline rebuild (measured ~230 us of CPU per GL_PATCHES draw with
         * .agent/ab_patches.c; a plain drawArrays is ~5 us).  The render encoder was
         * ended before the compute phase; DIRTY_RENDER_STATE makes the next
         * processGLState open a new render pass on this same buffer, after the
         * kernels.  Ring blocks now retire only when the shared buffer completes
         * (glFinish / readback / the ring's own forced commit in tessRingAcquire). */
        ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
    }
    else
    {
        id<MTLEvent> ev = [_device newEvent];
        uint64_t evv = 1;
        [compute_cb encodeSignalEvent: ev value: evv];
        [compute_cb commit];

        if (_currentCommandBuffer && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
        {
            [self endRenderEncoding];
            [_currentCommandBuffer commit];
            ctx->state.dirty_bits = DIRTY_ALL;
        }
        _currentCommandBuffer = [_commandQueue commandBuffer];
        [_currentCommandBuffer encodeWaitForEvent: ev value: evv];
        ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
    }

    /* ---- 3a. hardware tessellation path (triangles / quads): the TCS kernel wrote the
     * factors on the GPU; the render pass feeds them to drawPatches directly. No
     * readback, no CPU tessellation. ---- */
    if (use_hardware)
    {
        /* no TCS: the half factor records and the patch-data blob were already written
           above (constants / blob kernel); control points pass through from the vertex
           kernel output */

        /* ---- 4a. render: drawPatches with the post-tessellation TES ---- */
        tessPhaseEnd(TESS_PH_ORDER, ph_order);
        ph_render = tessPhaseBegin();
        {
            id<MTLRenderCommandEncoder> enc;
            GLsizei cp_count = tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in;
            size_t patch_off = tp->has_tcs ? off_tcs_patch : off_patch_data;

            if (![self processGLState: true])
            {
                tessRingRetireOn(_currentCommandBuffer, ring_slot);
                 return false;
            }
            enc = _currentRenderEncoder;
            if (!enc)
            {
                tessRingRetireOn(_currentCommandBuffer, ring_slot);
                 return false;
            }

            /* compat perf: shadow-filtered binds (the tess rings change offset per
             * draw; the shadow re-issues only what actually changed) */
            [self compatBindVertexBuffer:ring offset:cp_source_off atIndex:MGL_TESS_VD_CP];
            [self compatBindVertexBuffer:ring offset:patch_off atIndex:MGL_TESS_VD_PATCH];
            [self compatBindVertexBuffer:ring offset:off_levels_f atIndex:MGL_TESS_VD_FACTORS];
            [enc setTessellationFactorBuffer: ring offset: off_factors instanceStride: 0];

            if (tTrace())
            {
                NSLog(@"MGL tess: drawPatches cp=%d patches=%d (hw, fstride=%zu, tcs=%d outv=%d hwn=%d pvin=%d)", cp_count, total_patches, fstride, tp->has_tcs, tp->tcs_out_vertices, tp->tes_hw_n, patch_verts_in);
            }

            /* ONE drawPatches for every patch of the draw: the factor records are at the
             * domain stride (8 bytes for triangles), which is what the tessellator reads. */
            [enc drawPatches: (NSUInteger)cp_count
                  patchStart: 0
                   patchCount: (NSUInteger)total_patches
             patchIndexBuffer: nil
       patchIndexBufferOffset: 0
                instanceCount: 1
                 baseInstance: 0];
            if (tTrace())
            {
                [compute_cb waitUntilCompleted];
                float *cp = (float *)((uint8_t *)ring.contents + cp_source_off);
                for (int k = 0; k < total_patches * cp_count && k < 8; k++)
                {
                    size_t wds = cp_stride / 4;
                    fprintf(stderr, "MGL tess: idxcp[%d] wds=%zu words:", k, wds);
                    for (size_t q = 0; q < wds && q < 10; q++) fprintf(stderr, " %g", cp[k * wds + q]);
                    fprintf(stderr, "\n");
                }
                uint16_t *hf3 = (uint16_t *)((uint8_t *)ring.contents + off_factors);
                fprintf(stderr, "MGL tess: idxfactors:");
                for (int k = 0; k < total_patches * 4; k++) fprintf(stderr, " %04x", hf3[k]);
                fprintf(stderr, "\n");
            }


            
            /* tess perf (r42): a full DIRTY_ALL here forced the NEXT draw through
             * the whole descriptor rebuild (~126 us of NSString-keyed pipeline
             * descriptor per draw, measured with .agent/ab_patches.c) even though
             * the encoder binds of this draw are the only thing that changed.
             * DIRTY_VAO re-runs the vertex-buffer map list (this draw bound the
             * tess rings at MGL_TESS_VD_CP/PATCH/FACTORS, so GL attribute binds
             * must be re-issued) and keeps the compat pipeline fast path
             * admissible (it declines on DIRTY_FBO); DIRTY_RENDER_STATE rebuilds
             * the render pass, which every following tess draw needs anyway (its
             * compute encoders must be ordered before its own drawPatches).  The
             * bind-shadow generation bump in extCompatRenderEncoderCreated keeps
             * per-index encoder binds from leaking into the new encoder. */
            ctx->state.dirty_bits = DIRTY_VAO | DIRTY_RENDER_STATE;
            tessPhaseEnd(TESS_PH_RENDER, ph_render);
            ph_post = tessPhaseBegin();
            {
                int slot = ring_slot;
                id<MTLCommandBuffer> cb = _currentCommandBuffer;
                if (tRingTrace())
                    fprintf(stderr, "MGL ring: register(hw) slot=%d cb=%d\n", slot,
                            cb ? (int)cb.status : -1);
                tessRingRetireOn(cb, slot);   /* handles committed/completed buffers */
            }

            /* AGX keep-alive (r42, pressure-driven in r43): a command buffer that
             * carries several tess draws and is then left uncommitted lets the
             * driver's submission and completion path go idle — Metal batches
             * command-buffer submissions on a timer, and a burst that out-runs the
             * ring slots ends up sleeping a whole batch window inside
             * tessRingAcquire (~6.7 ms here, ~250 ms under heavy load; measured
             * with .agent/ab_patches.c).  Committing the shared buffer keeps
             * completions flowing, and completions are exactly what retires ring
             * slots, so the cadence is now driven by the ring's own occupancy:
             * plenty free -> every 4th draw (the batcher amortises the commit),
             * moderately free -> every 2nd, running low -> every draw, and an
             * actual exhaustion forces per-draw commits for the next 64 draws
             * (the tight-loop shape where submission latency dominates). */
            if (tRingExhaust > 0)
                tRingExhaust--;
            {
                unsigned free_slots = 0;
                for (int i = 0; i < MGL_TESS_RING; i++)
                    free_slots += !tessSlotBusy(i);
                /* hysteresis (min-of-7 ab: a plain low/high threshold oscillated
                 * around the pressure point and cost ~6 us/draw on the noTCS
                 * shape): cadence 2 is the steady state, 4 only with most of the
                 * ring free, 1 when it runs low.  MGL_TESS_KAMAX caps the cadence
                 * (A/B knob for the r44 poll-based retirement, where slots free
                 * at GPU completion instead of handler firing). */
                static int kaMax = -1;
                unsigned ka_every;

                if (kaMax < 0)
                {
                    const char *e = getenv("MGL_TESS_KAMAX");
                    kaMax = (e && atoi(e) >= 1 && atoi(e) <= 4) ? atoi(e) : 4;
                }
                ka_every = tRingExhaust ? 1u :
                                    (free_slots <= (unsigned)MGL_TESS_RING / 6) ? 1u :
                                    (free_slots <= (unsigned)(MGL_TESS_RING * 2) / 3) ? 2u : 4u;
                if (ka_every > (unsigned)kaMax)
                    ka_every = (unsigned)kaMax;
                if (++tHwDrawsOnCb >= ka_every)
                {
                    tHwDrawsOnCb = 0;
                    if (_currentCommandBuffer &&
                        _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
                    {
                        [self endRenderEncoding];
                        [_currentCommandBuffer commit];
                        _currentCommandBuffer = nil;
                    }
                }
            }
        }
        tessPhaseEnd(TESS_PH_POST, ph_post);
        tessPhaseEnd(TESS_PH_TOTAL, ph_total);
        return true;
    }

    /* ---- 3b. emulated path (isolines / point_mode): GPU tessellator ----
     * count -> alloc -> emit -> indirect-args kernels run right after the TCS in the
     * same compute encoder; the plain-vertex TES then draws from the coordinate and
     * index buffers through an indirect draw. No CPU readback. (MGL_TESS_CPU=1 keeps
     * the CPU tessellator of tessellation.c as a reference path.) */
    if (!use_cpu_reference && tessKernels(ctx))
    {
        struct TessKParams { uint32_t patches, domain, spacing, point_mode, ccw; } kp;
        if (tRingTrace())
            fprintf(stderr, "MGL ring: stage emu-entry slot=%d hw=%d cpu_ref=%d readback=%d cap=%d\n",
                    ring_slot, use_hardware, use_cpu_reference, need_readback, tp->capture ? 1 : 0);

        /* MGL_TESS_PATCHDBG: dump the TCS patch-out blob of patch 0 once the VS/TCS
         * command buffer completed (debug stall only) */
        if (getenv("MGL_TESS_PATCHDBG") && ring && off_tcs_patch < ring.length)
        {
            id<MTLCommandBuffer> wcb = compute_cb ? compute_cb : _currentCommandBuffer;
            if (wcb) [wcb waitUntilCompleted];
            const int *pb = (const int *)((const uint8_t *)ring.contents + off_tcs_patch);
            fprintf(stderr, "MGL tess: patch blob @%zu:", off_tcs_patch);
            for (int i = 0; i < 16; i++) fprintf(stderr, " %d", pb[i]);
            fprintf(stderr, "\n");
        }

        /* the VS/TCS command buffer is already committed: the tessellator kernels run
         * in their own, event-ordered command buffer (no CPU stall) */
        emu_cb = [_commandQueue commandBuffer];
        tessTimeCommandBuffer(emu_cb);
        kenc = [emu_cb computeCommandEncoder];
        cenc = kenc;
        size_t emu_off = off_gpu_emu;                /* range reserved in the ring block */
        size_t coords_off = emu_off + (((size_t)total_patches * 2 + 2 + 5 + 3) & ~(size_t)3) * 4;
        dbg_coords_off = coords_off;
        dbg_emu_off = emu_off;
        dbg_ctl_off = emu_off;
        size_t coords_cap = (size_t)total_patches * (tp->domain == MGL_TESS_DOMAIN_QUADS ? 4225 :
                            tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 4160 : 3200);
        /* D3D11-layout worst case per patch: quads 4x65-4 + 63^2 = 4225 vertices /
           24576 indices; triangles 192 + 3*31*32 + 1 = 3169 vertices / 18432 indices;
           isolines 64 x 65 vertices / 8192 indices (64 lines of 64 segments) */
        size_t indices_cap = (size_t)total_patches * (tp->point_mode ? (tp->domain == MGL_TESS_DOMAIN_QUADS ? 4225 :
                             tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 4160 : 3200) :
                             tp->domain == MGL_TESS_DOMAIN_QUADS ? 24576 :
                             tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 8192 : 18432);
        coords_cap_dbg = coords_cap;
        size_t need = (coords_off - emu_off) + coords_cap * 16 + indices_cap * 4;
        if (tp->capture)
        {
            /* the layout (record stride) comes from the capture-kernel rewrite */
            char *cap_msl = mgl_tess_capture_msl(ctx, tp);

            if (tp->cap_stride)
            {
                /* the records were reserved inside the draw's ring block when it was
                 * sized (see the layout blocks); appending them after the emulated
                 * scratch is only the fallback for a capture that appeared later */
                if (!off_capture)
                    off_capture = emu_off + need;
                need += coords_cap * (size_t)tp->cap_stride * 4;
            }
            free(cap_msl);      /* the compile step below rebuilds it if needed */
        }

        if (need <= (size_t)(64u << 20) && ring)
        {
            uint8_t *base8 = (uint8_t *)ring.contents;
            uint32_t *base = (uint32_t *)(base8 + emu_off);
            if (tTrace())
                NSLog(@"MGL tess: emu dispatch: patches=%u total=%u emu_off=%zu coords_off=%zu "
                      "ring=%p contents=%p len=%zu cap=%u need=%zu [vs_out=%zu tcs_out=%zu tcs_patch=%zu factors=%zu patch_data=%zu cp_off=%zu stride=%zu]",
                      (unsigned)patches, (unsigned)total_patches, emu_off, coords_off,
                      (void *)(__bridge void *)ring, ring.contents, (unsigned long)ring.length,
                      (unsigned)coords_cap, need, off_vs_out, off_tcs_out, off_tcs_patch,
                      off_factors, off_patch_data, cp_source_off, cp_stride);
            kp.patches = (uint32_t)total_patches;
            kp.domain = (uint32_t)tp->domain;
            kp.spacing = (uint32_t)tp->spacing;
            kp.point_mode = (uint32_t)(tp->point_mode ? 1 : 0);
            kp.ccw = (uint32_t)(tp->vertex_order_ccw ? 1 : 0);
            base[total_patches * 2 + 0] = 0;    /* coord total */
            base[total_patches * 2 + 1] = 0;    /* index total */

            [cenc setComputePipelineState: tCountPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: ring offset: off_levels_f atIndex: 1];   /* GL's exact float levels */
            [cenc setBuffer: ring offset: emu_off atIndex: 2];
            [cenc setBuffer: ring offset: emu_off + (size_t)(total_patches * 2) * 4 atIndex: 3];
            [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)total_patches, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            /* deterministic patch order: prefix-scan the per-patch counts into
             * bases before the emit (see mgl_tess_scan) */
            [cenc setComputePipelineState: tScanPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: ring offset: emu_off atIndex: 2];
            [cenc setBuffer: ring offset: emu_off + (size_t)(total_patches * 2) * 4 atIndex: 3];
            [cenc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            [cenc setComputePipelineState: tEmitPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: ring offset: off_levels_f atIndex: 1];   /* GL's exact float levels */
            [cenc setBuffer: ring offset: emu_off atIndex: 2];
            [cenc setBuffer: ring offset: coords_off atIndex: 4];
            [cenc setBuffer: ring offset: coords_off + coords_cap * 16 atIndex: 5];
            [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)total_patches, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            [cenc setComputePipelineState: tArgsPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: ring offset: emu_off + (size_t)(total_patches * 2) * 4 atIndex: 3];
            [cenc setBuffer: ring offset: emu_off + (size_t)(total_patches * 2 + 2) * 4 atIndex: 6];
            [cenc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            [cenc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            /* capture permute: transform feedback records the vertices in GENERATION
             * order (GL 4.6 13.4) — the coordinates re-gathered through the index list.
             * The bound count is the GPU-written totals array (guard on totals[1]).
             * point_mode emits no index list (totals[1] == 0): every tessellated vertex
             * is a point and the capture walks the coordinate grid itself, so the
             * permute would only copy an unwritten buffer. */
            if (off_capture_perm && !tp->point_mode && tPermutePso)
            {
                [cenc setComputePipelineState: tPermutePso];
                [cenc setBuffer: ring offset: coords_off atIndex: 0];
                [cenc setBuffer: ring offset: coords_off + coords_cap * 16 atIndex: 1];
                [cenc setBuffer: ring offset: emu_off + (size_t)(total_patches * 2) * 4 atIndex: 2];
                [cenc setBuffer: ring offset: off_capture_perm atIndex: 3];
                [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)(indices_cap + 63) / 64, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
            }

            /* TES-output capture (tess -> geometry / transform feedback producer):
             * the plain TES rewritten as a kernel, one record per tessellated vertex,
             * dispatched after the tessellator in the same encoder. The vertex count
             * it guards on is the GPU-written draw argument. */
            if (off_capture)
            {
                if (tessEnsureCapturePso(ctx, tp, prog_tes) && tp->capture_pso)
                {                    uint32_t cc = (uint32_t)coords_cap;

                    [cenc setComputePipelineState: (__bridge id<MTLComputePipelineState>)tp->capture_pso];
                    tessBindStageUniforms(ctx, cenc, prog_tes, _TESS_EVALUATION_SHADER);
                    tessBindStageTextures(ctx, cenc, prog_tes, _TESS_EVALUATION_SHADER);
                    [cenc setBuffer: ring offset: (off_capture_perm && !tp->point_mode ?
                                                   off_capture_perm : coords_off)
                             atIndex: MGL_TESS_TES_COORDS];
                    [cenc setBuffer: ring offset: cp_source_off atIndex: MGL_TESS_TES_CP];
                    [cenc setBuffer: ring offset: off_patch_data atIndex: MGL_TESS_TES_PATCH];
                    {
                        /* [1] = patches per instance: gl_PrimitiveID restarts per instance */
                        uint32_t cparams[4] = { (uint32_t)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in), (uint32_t)patches, 0, 0 };
                        [cenc setBytes: cparams length: sizeof cparams atIndex: MGL_TESS_TES_PARAMS];
                    }
                    [cenc setBuffer: ring offset: emu_off + (size_t)(total_patches * 2) * 4 +
                                       (tp->point_mode ? 0 : 4)
                             atIndex: MGL_TESS_TES_CAP_COUNT];
                    [cenc setBuffer: ring offset: off_capture atIndex: MGL_TESS_TES_CAP_RECS];
                    /* the guard is the EMITTED vertex count (totals[1]); triangles and
                     * quads emit more vertices than the coordinate grid holds */
                    {
                        NSUInteger nthr = (NSUInteger)coords_cap;
                        if (nthr < indices_cap) nthr = indices_cap;
                        [cenc dispatchThreadgroups: MTLSizeMake((nthr + 63) / 64, 1, 1)
                              threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
                    }
                    cap_pub_records = off_capture;
                    /* the consumer's primitive slots come from the tessellator's uint32 index
                     * list (global coordinate indices, i.e. exactly the capture's record
                     * indices) — NOT off_indices, which is the CPU path's list and stays 0 */
                    cap_pub_idx = coords_off + coords_cap * 16;
                    /* the capacity is in EMITTED vertices: the index list can be longer
                     * than the coordinate grid (triangles: 12288 indices vs 2145 coords) */
                    cap_pub_recs_cap = (uint32_t)(coords_cap > indices_cap ? coords_cap : indices_cap);
                    cap_pub_idx_cap = (uint32_t)indices_cap;
                    /* totals[0] is the GPU-written VERTEX count and totals[1] the index count
                     * (args[0] would be the index count: 3x the records for triangle domains) */
                    /* the count consumers read: totals[1] (emitted vertices) except
                     * point_mode, which captures the tessellated POINTS (totals[0]) —
                     * the same rule the args kernel uses for the indirect draw */
                    cap_pub_count = emu_off + (size_t)(total_patches * 2) * 4 +
                                    (tp->point_mode ? 0 : 4);
                }
                else if (tTrace())
                    NSLog(@"MGL tess: draw bail: capture PSO failed to build (off_capture=%zu)",
                          off_capture);
            }

            /* expose the GPU tessellated output (geometry / xfb consumers) */
            S->gpu_scratch = (__bridge void *)ring;
            S->gpu_coords_offset = coords_off;
            S->gpu_indices_offset = coords_off + coords_cap * 16;
            S->gpu_args_offset = emu_off + (size_t)(total_patches * 2 + 2) * 4;

            gpu_emulated = true;
        }
        else
        {
            /* GPU tessellator scratch too large (%zu bytes): CPU path.  The factors live
             * in ring memory written by compute_cb — which this path committed (event
             * form), never waited on — so wait for it before the CPU tessellator reads
             * them; reading right after commit saw zeros and produced empty captures
             * (tc_barriers draws 1024 patches -> 221 MB scratch -> this fallback). */
            if (tTrace())
                NSLog(@"MGL tess: GPU tessellator scratch too large (%zu bytes), using CPU path", need);
            if (compute_cb && compute_cb.status < MTLCommandBufferStatusCompleted)
                [compute_cb waitUntilCompleted];
        }
    }
    if (gpu_emulated && kenc)
    {
        [kenc endEncoding];
        /* publish the capture for consumers (geometry / transform feedback):
         * valid until the next GL_PATCHES draw; the ring slot is released by the
         * draw's completion, and consumers copy or consume before then */
        if (cap_pub_records && tp->capture_pso)
        {
            tessPublishCapture(ctx, ring, tp, cap_pub_records, cap_pub_recs_cap,
                               cap_pub_idx, cap_pub_idx_cap, cap_pub_count);
            /* GPU path: the capture kernel read the emission-order coordinate permute
             * (non-point_mode), so records[] are per emitted vertex in emission order
             * and the published index list is the tessellator's coordinate-index list,
             * NOT a record map — consumers must not re-index through it
             * (MGLTessCapture.emission_order, geometry.c's tess->geometry import) */
            ((struct MGLExtState_tess *)ctx->state.ext_tess)->capture.emission_order = true;
        }
        {
            id<MTLEvent> ev2 = [_device newEvent];
            [emu_cb encodeSignalEvent: ev2 value: 1];
            [emu_cb commit];
            if (getenv("MGL_TESS_EMUDBG"))
            {
                /* one-off diagnostics: the tessellator's per-patch counts, the TCS's
                 * control-point records and the half factors, after completion */
                NSUInteger wcp = (NSUInteger)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in);
                [emu_cb waitUntilCompleted];
                const uint32_t *tot = (const uint32_t *)((const uint8_t *)ring.contents + dbg_emu_off);
                const uint32_t *scanp = (const uint32_t *)((const uint8_t *)ring.contents +
                                      dbg_emu_off + (size_t)total_patches * 2 * 4);
                const float *coord = (const float *)((const uint8_t *)ring.contents + dbg_coords_off);
                const float *tcout = (const float *)((const uint8_t *)ring.contents + off_tcs_out);
                const uint16_t *hf = (const uint16_t *)((const uint8_t *)ring.contents + off_factors);
                NSLog(@"emudbg: patches=%u totals coords=%u idx=%u", (unsigned)total_patches, tot[0], tot[1]);
                for (unsigned p = 0; p < total_patches && p < 4; p++)
                    NSLog(@"emudbg:  patch %u: cnt c=%u i=%u base c=%u i=%u factors %04x %04x %04x %04x %04x %04x",
                          p, scanp[p * 4 + 0], scanp[p * 4 + 1], scanp[p * 4 + 2], scanp[p * 4 + 3],
                          hf[(size_t)p * (fstride / 2) + 0], hf[(size_t)p * (fstride / 2) + 1],
                          hf[(size_t)p * (fstride / 2) + 2], hf[(size_t)p * (fstride / 2) + 3],
                          hf[(size_t)p * (fstride / 2) + 4], hf[(size_t)p * (fstride / 2) + 5]);
                for (NSUInteger k = 0; k < wcp * total_patches && k < 6; k++)
                {
                    size_t wds = cp_stride / 4;
                    NSString *s = @"";
                    for (size_t q = 0; q < wds && q < 8; q++)
                        s = [s stringByAppendingFormat: @" %g", tcout[k * wds + q]];
                    NSLog(@"emudbg:  tcsout[%llu]:%@", (unsigned long long)k, s);
                }
                for (unsigned k = 0; k < 8; k++)
                    NSLog(@"emudbg:  coord[%u] = (%.3f,%.3f,%.3f,%.3f)", k,
                          coord[k * 4], coord[k * 4 + 1], coord[k * 4 + 2], coord[k * 4 + 3]);
                if (off_capture)
                {
                    const float *crec = (const float *)((const uint8_t *)ring.contents + off_capture);
                    const uint32_t *cidx = (const uint32_t *)((const uint8_t *)ring.contents +
                                            dbg_coords_off + coords_cap_dbg * 16);
                    unsigned n = tot[1];
                    if (n > 12) n = 12;
                    if (n == 0) n = 12;   /* the totals word is only written by tArgsPso elsewhere */
                    for (unsigned k = 0; k < n; k++)
                        NSLog(@"emudbg:  caprec[%u] = (%.2f,%.2f,%.2f,%.2f) col(%.1f,%.1f,%.1f,%.1f)", k,
                              crec[(size_t)k * tp->cap_stride],
                              crec[(size_t)k * tp->cap_stride + 1],
                              crec[(size_t)k * tp->cap_stride + 2],
                              crec[(size_t)k * tp->cap_stride + 3],
                              crec[(size_t)k * tp->cap_stride + 4],
                              crec[(size_t)k * tp->cap_stride + 5],
                              crec[(size_t)k * tp->cap_stride + 6],
                              crec[(size_t)k * tp->cap_stride + 7]);
                    for (unsigned k = 0; k < n; k++)
                        NSLog(@"emudbg:  capidx[%u] = %u", k, cidx[k]);
                }
            }
            if (getenv("MGL_TESS_DUMP_CAP") && cap_pub_records && tp->capture_pso)
            {
                /* one-off diagnostics: read the published capture records back and
                 * print the te_pointsize column (chasing a CTS point-mode mismatch) */
                uint32_t psw = 0xffffffff;
                unsigned nrec = 30;

                for (int i = 0; i < tp->cap_members_n; i++)
                    if (!strcmp(tp->cap_members[i].name, "te_pointsize"))
                        psw = tp->cap_members[i].offset;
                [emu_cb waitUntilCompleted];
                if (psw != 0xffffffff)
                {
                    const float *base = (const float *)((uint8_t *)ring.contents + cap_pub_records);
                    const volatile uint32_t *cntp =
                        (const volatile uint32_t *)((uint8_t *)ring.contents + cap_pub_count);
                    const unsigned short *hf =
                        (const unsigned short *)((uint8_t *)ring.contents + off_factors);
                    const float *ff = (const float *)((uint8_t *)ring.contents + off_levels_f);

                    NSLog(@"capdump: stride=%uw count=%u ps@%u members=%d pair=%d dom=%d"
                          @" | factors half %u %u %u %u|%u %u float %.2f %.2f %.2f %.2f|%.2f %.2f",
                          tp->cap_stride, *cntp, psw, tp->cap_members_n, tp->pair_program != NULL,
                          tp->domain, hf[0], hf[1], hf[2], hf[3], hf[4], hf[5],
                          ff[0], ff[1], ff[2], ff[3], ff[4], ff[5]);
                    if (nrec > *cntp) nrec = *cntp;
                    for (unsigned r = 0; r < nrec; r++)
                    {
                        const float *rec = (const float *)((const uint8_t *)base +
                                                           (size_t)r * tp->cap_stride * 4u);

                        if (r < 2 || rec[psw] != base[psw])
                            NSLog(@"  rec %3u/%u: ps=%.4f pos=(%.1f,%.1f,%.1f,%.1f) v1=%.4f v2=%d",
                                  r, *cntp, rec[psw], rec[0], rec[1], rec[2], rec[3], rec[4],
                                  *(const int *)(const void *)&rec[6]);
                    }
                }
            }
            if (_currentCommandBuffer && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
            {
                [self endRenderEncoding];
                [_currentCommandBuffer commit];
                ctx->state.dirty_bits = DIRTY_ALL;
            }
            _currentCommandBuffer = [_commandQueue commandBuffer];
            [_currentCommandBuffer encodeWaitForEvent: ev2 value: 1];
            ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
        }
    }

    /* plain-vertex variant of the TES: needed by every non-hardware path (isolines,
     * point_mode, capture and the CPU reference draw — which renders with a plain
     * indexed draw, not drawPatches) */
    if ((gpu_emulated || use_cpu_reference) && !tp->plain_function)
    {
        char *msl = mgl_tess_plain_msl(ctx, tp);
        if (!tNoFixup())
        {
            char *patched = mgl_rstate_patch_vertex_msl(msl, prog_tes->spirv[_TESS_EVALUATION_SHADER].entry_point);
            if (patched) { free(msl); msl = patched; }
        }
        if (msl && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
        {
            void *lib = NULL, *fn = NULL;
            char *err = NULL;
            if (ctx->mtl_funcs.ext_program->compile_msl(ctx, msl,
                    prog_tes->spirv[_TESS_EVALUATION_SHADER].entry_point, &lib, &fn, &err))
            {
                tp->plain_function = fn;
                tp->plain_library = lib;
                ctx->state.dirty_bits |= DIRTY_PROGRAM;
            }
            else
                NSLog(@"MGL ERROR: tess: plain TES variant failed to compile: %s", err ? err : "?");
            free(err);
        }
        free(msl);
    }

    /* geom: when a geometry stage consumes the TES capture, geom_route_tess forced
     * GL_RASTERIZER_DISCARD — the plain-TES draw below would re-run the TES body a
     * SECOND time (its image atomics / SSBO stores would double:
     * shader_image_load_store.advanced-allStages-oneImage) for output nobody reads;
     * the capture kernel already executed the TES once per tessellated vertex.  Skip
     * it and just retire the ring block. */
    if (gpu_emulated && tp->capture && ctx->state.caps.rasterizer_discard == GL_TRUE)
    {
        int rslot = ring_slot;
        id<MTLCommandBuffer> rcb = _currentCommandBuffer;

        if (tRingTrace())
            fprintf(stderr, "MGL ring: retire(capture-only; plain-TES draw skipped) slot=%d cb=%d\n",
                    rslot, rcb ? (int)rcb.status : -1);
        tessRingRetireOn(rcb, rslot);
        ctx->state.dirty_bits = DIRTY_ALL;
        return true;
    }

    /* GPU emulated path: render with an indirect draw; the patch-data blob was
     * assembled by the blob kernel in the compute stage above (no read-back, no
     * CPU wait) */
    if (gpu_emulated)
    {
        {
            id<MTLRenderCommandEncoder> enc;

            if (tRingTrace())
                fprintf(stderr, "MGL ring: stage emu-render slot=%d\n", ring_slot);
            if (![self processGLState: true])
            {
                tessRingRetireOn(_currentCommandBuffer, ring_slot);
                 return false;
            }
            enc = _currentRenderEncoder;
            if (!enc)
            {
                tessRingRetireOn(_currentCommandBuffer, ring_slot);
                 return false;
            }

            /* compat perf: shadow-filtered bind */
            [self compatBindVertexBuffer:(__bridge id<MTLBuffer>)S->gpu_scratch offset:S->gpu_coords_offset atIndex:MGL_TESS_TES_COORDS];
            /* the TES reads the control points where the VS kernel (no TCS) or the TCS
             * kernel wrote them — cp_source_off, like the hardware path's binding */
            [self compatBindVertexBuffer:cp_source offset:cp_source_off atIndex:MGL_TESS_TES_CP];
            if (tTrace())
            {
                void *cpPtr = (__bridge void *)cp_source;
                void *ringPtr = (__bridge void *)ring;

                NSLog(@"MGL tess: emu render bind: coords=%p+%zu cp=%p+%zu ring=%p(%zu)",
                      S->gpu_scratch, (unsigned long)S->gpu_coords_offset,
                      cpPtr, (unsigned long)cp_source_off, ringPtr, (unsigned long)ring.length);
            }
            [self compatBindVertexBuffer:ring offset:off_patch_data atIndex:MGL_TESS_TES_PATCH];
            {
                /* the plain TES takes mgl_tess_params as a buffer POINTER: the words must
                 * come from a real buffer (setVertexBytes would pass them by value) —
                 * this draw's exclusive ring slot, never overwritten by a later draw */
                uint32_t tes_params[4] = { (uint32_t)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in), (uint32_t)patches, 0, 0 };

                memcpy((uint8_t *)ring.contents + off_params, tes_params, sizeof tes_params);
                [self compatBindVertexBuffer:ring offset:off_params atIndex:MGL_TESS_TES_PARAMS];

                if (tTrace())
                {
                    /* trace facility (MGL_TESS_DEBUG): block once and dump the per-patch
                     * records the GPU produced — TCS patch output, blob, factors and the
                     * first coordinates of the tessellator's stream */
                    [emu_cb waitUntilCompleted];
                    const float *cps = (const float *)((uint8_t *)ring.contents + off_tcs_out);
                    const float *blob = (const float *)((uint8_t *)ring.contents + off_patch_data);
                    const float *ptch = (const float *)((uint8_t *)ring.contents + off_tcs_patch);
                    const float *coords = (const float *)((uint8_t *)ring.contents + dbg_coords_off);
                    const uint16_t *hf = (const uint16_t *)((uint8_t *)ring.contents + off_factors);
                    if (dbg_ctl_off)
                    {
                        const uint32_t *ctl = (const uint32_t *)((uint8_t *)ring.contents + dbg_ctl_off);
                        fprintf(stderr, "MGL tess: trace ctl(allocs0,1 totals0,1 args0..3):");
                        for (int q = 0; q < 8; q++)
                            fprintf(stderr, " %u", ctl[q]);
                        fprintf(stderr, "\n");
                    }
                    fprintf(stderr, "MGL tess: trace tcs_out(x,y):");
                    for (unsigned i = 0; i < (unsigned)(total_patches * tp->tcs_out_vertices); i++)
                        fprintf(stderr, " [%.3f %.3f]", cps[i * cp_stride / 4 + 4], cps[i * cp_stride / 4 + 5]);
                    fprintf(stderr, "\nMGL tess: trace tcs_patch:");
                    for (int p = 0; p < total_patches; p++)
                        fprintf(stderr, " [%.2f %.2f %.2f %.2f]", ptch[p * 4 + 0], ptch[p * 4 + 1],
                                ptch[p * 4 + 2], ptch[p * 4 + 3]);
                    fprintf(stderr, "\nMGL tess: trace blob:");
                    for (int p = 0; p < total_patches; p++)
                        fprintf(stderr, " [%.2f %.2f %.2f %.2f]", blob[p * 4 + 0], blob[p * 4 + 1],
                                blob[p * 4 + 2], blob[p * 4 + 3]);
                    if (cap_pub_records && tp->cap_stride)
                    {
                        const unsigned *cr = (const unsigned *)((uint8_t *)ring.contents + cap_pub_records);
                        fprintf(stderr, "\nMGL tess: trace caprec(stride %u):", tp->cap_stride);
                        for (int q = 0; q < 3 * (int)tp->cap_stride && q < 96; q++)
                            fprintf(stderr, " %08x", cr[q]);
                        fprintf(stderr, "\nMGL tess: trace caprec25:");
                        for (int q = 25 * (int)tp->cap_stride; q < 27 * (int)tp->cap_stride; q++)
                            fprintf(stderr, " %08x", cr[q]);
                        fprintf(stderr, "\n");
                    }
                    fprintf(stderr, "\nMGL tess: trace factors(half):");
                    for (int p = 0; p < total_patches; p++)
                        fprintf(stderr, " [%x %x %x %x %x %x]", hf[p * 6 + 0], hf[p * 6 + 1], hf[p * 6 + 2],
                                hf[p * 6 + 3], hf[p * 6 + 4], hf[p * 6 + 5]);
                    fprintf(stderr, "\nMGL tess: trace coords(u v . pid):");
                    for (unsigned i = 0; i < 16; i++)
                        fprintf(stderr, " [%.3f %.3f %.0f]", coords[i * 4 + 0], coords[i * 4 + 1], coords[i * 4 + 3]);
                    fprintf(stderr, "\n");
                }
            }

            if (tp->point_mode)
                [enc drawPrimitives: MTLPrimitiveTypePoint
                       indirectBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch
                 indirectBufferOffset: S->gpu_args_offset];
            else
                [enc drawIndexedPrimitives: (tp->domain == MGL_TESS_DOMAIN_ISOLINES ?
                                            MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle)
                                  indexType: MTLIndexTypeUInt32
                                indexBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch
                          indexBufferOffset: S->gpu_indices_offset
                             indirectBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch
                       indirectBufferOffset: S->gpu_args_offset];

            /* the ring block retires with this command buffer (the capture, if any,
             * holds its own reference to the buffer object) */
            {
                int rslot = ring_slot;
                id<MTLCommandBuffer> rcb = _currentCommandBuffer;

                if (tRingTrace())
                    fprintf(stderr, "MGL ring: register(emu) slot=%d cb=%d\n", rslot,
                            rcb ? (int)rcb.status : -1);
                tessRingRetireOn(rcb, rslot);   /* the wait above may already have
                                                 * completed (or committed) the buffer */
            }

            if (tTrace() && dbg_ctl_off)
            {
                const uint8_t *rc = (const uint8_t *)ring.contents;
                size_t o = dbg_ctl_off;

                if (_currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
                    goto skip_ctl_trace;   /* handler must precede commit */
                [ _currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> b) {
                    const uint32_t *c = (const uint32_t *)(rc + o);
                    (void)b;
                    fprintf(stderr, "MGL tess: trace ctl@renderDone: %u %u %u %u %u %u %u %u\n",
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
                    
                } ];
            skip_ctl_trace: ;
            }

            ctx->state.dirty_bits = DIRTY_ALL;
            return true;
        }
    }

    /* CPU reference path (MGL_TESS_CPU=1): the factors are already on the CPU side —
     * compute_cb was committed and waited on above — so read them from shared storage
     * and tessellate. (Do not commit compute_cb again: it is already completed.) */
    {
    if (need_readback)
    {
        {
            float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));

            if (tp->tcs_levels_float)
            {
                /* exact: the TCS also wrote a float copy (GL's level contract); the half
                 * factor record would quantise e.g. 2.7 to 2.69922 */
                memcpy(factors, (uint8_t *)ring.contents + off_levels_f,
                       (size_t)total_patches * 6 * sizeof(float));
            }
            else
            {
                const uint16_t *hf = (const uint16_t *)((uint8_t *)ring.contents + off_factors);

                for (int p = 0; p < total_patches; p++)
                {
                    const uint16_t *f = hf + p * (fstride / 2);  /* 4 halfs tri / 6 halfs quad */
                    if (tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
                    {
                        /* MTLTriangleTessellationFactorsHalf: { edge[3], inside } — the
                         * outer[3] slot a triangle patch never uses stays 0 */
                        factors[p * 6 + 0] = tessHalfToFloat(f[0]);
                        factors[p * 6 + 1] = tessHalfToFloat(f[1]);
                        factors[p * 6 + 2] = tessHalfToFloat(f[2]);
                        factors[p * 6 + 3] = 0.0f;
                        factors[p * 6 + 4] = tessHalfToFloat(f[3]);
                        factors[p * 6 + 5] = 0.0f;
                    }
                    else
                    {
                        factors[p * 6 + 0] = tessHalfToFloat(f[0]);
                        factors[p * 6 + 1] = tessHalfToFloat(f[1]);
                        factors[p * 6 + 2] = tessHalfToFloat(f[2]);
                        factors[p * 6 + 3] = tessHalfToFloat(f[3]);
                        factors[p * 6 + 4] = tessHalfToFloat(f[4]);
                        factors[p * 6 + 5] = tessHalfToFloat(f[5]);
                    }
                }
            }
            mgl_tess_build_patch_blob(ctx, tp, (const unsigned char *)((uint8_t *)ring.contents + off_tcs_patch),
                                      patch_record, factors, (unsigned)total_patches);
            mgl_tess_generate(ctx, tp, factors, (unsigned)total_patches);
            if (tTrace())
                fprintf(stderr, "MGL tess: cpu path patch 0 factors %g %g %g %g / %g %g -> %d verts %d idx\n",
                        factors[0], factors[1], factors[2], factors[3], factors[4], factors[5],
                        S->tess_vertex_count, S->tess_index_count);
            free(factors);
        }
    }
    else if (!use_cpu_reference)
    {
        /* no TCS and we still fell back to the CPU path (GPU tessellator unavailable or
         * its scratch was too large): the default-level blob and the CPU tessellator run
         * here — with MGL_TESS_CPU=1 they already ran in the compute stage above */
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));

        for (int p = 0; p < total_patches; p++)
        {
            factors[p * 6 + 0] = rs->patch_default_outer[0];
            factors[p * 6 + 1] = rs->patch_default_outer[1];
            factors[p * 6 + 2] = rs->patch_default_outer[2];
            factors[p * 6 + 3] = rs->patch_default_outer[3];
            factors[p * 6 + 4] = rs->patch_default_inner[0];
            factors[p * 6 + 5] = rs->patch_default_inner[1];
        }
        mgl_tess_build_patch_blob(ctx, tp, NULL, 0, factors, (unsigned)total_patches);
        mgl_tess_generate(ctx, tp, factors, (unsigned)total_patches);
        free(factors);
    }

    if (!S->tess_index_count)
    {
        /* degenerate: nothing to rasterize (culled patch / all levels zero) — the
         * block was only used by the (already completed) compute buffer */
        tessRingRetireOn(compute_cb, ring_slot);
        if (tRingTrace())
            fprintf(stderr, "MGL ring: retire(degenerate) slot=%d\n", ring_slot);
        return true;
    }

    /* ---- 4. render phase (CPU reference path): TES is the pipeline's vertex function ---- */
    {
        size_t coord_bytes = (size_t)S->tess_vertex_count * 16;
        size_t index_bytes = (size_t)S->tess_index_count * sizeof(uint32_t);
        uint32_t tes_params[4] = { (uint32_t)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in), (uint32_t)patches, 0, 0 };
        id<MTLRenderCommandEncoder> enc;
        MTLPrimitiveType prim = tp->point_mode ? MTLPrimitiveTypePoint :
                                (tp->domain == MGL_TESS_DOMAIN_ISOLINES && !tp->point_mode) ?
                                    MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;

        {
            size_t base = off_gpu_emu ? off_gpu_emu : (off_factors + tessRingAlign((size_t)total_patches * fstride));
            size_t need = tessRingAlign(coord_bytes) + tessRingAlign(index_bytes) +
                          tessRingAlign((size_t)S->patch_count * tp->tes_patch_in_size);
            if (tRingCap[ring_slot] < base + need)
            {
                NSLog(@"MGL ERROR: tess: CPU reference scratch overflow");
                tessRingRetireOn(compute_cb, ring_slot);
                 return false;
            }
            off_coords = base;
            off_indices = base + tessRingAlign(coord_bytes);
            off_patch_data = off_indices + tessRingAlign(index_bytes);
        }
        memcpy((uint8_t *)ring.contents + off_coords, S->coord_buf, coord_bytes);
        if (tTrace())
        {
            const float *c = (const float *)S->coord_buf;
            const float *r = (const float *)((uint8_t *)ring.contents + off_coords);
            fprintf(stderr, "MGL tess: cpu draw: coords@%zu (%u x16B) first:", off_coords,
                    S->tess_vertex_count);
            for (unsigned q = 0; q + 3 < S->tess_vertex_count * 4 && q / 4 < 6; q += 4)
                fprintf(stderr, " [%g %g %g %g]", c[q], c[q + 1], c[q + 2], c[q + 3]);
            fprintf(stderr, " ring: [%g %g %g %g]", r[0], r[1], r[2], r[3]);
            fprintf(stderr, "\n");
        }
        memcpy((uint8_t *)ring.contents + off_indices, S->index_buf, index_bytes);
        memcpy((uint8_t *)ring.contents + off_patch_data, S->patch_data,
               (size_t)S->patch_count * tp->tes_patch_in_size);

        /* TES-output capture on the CPU reference path: the coordinate stream, index
         * list and patch blob were just written into this draw's ring block on the CPU,
         * so the capture kernel can run over them exactly like over the GPU tessellator's
         * stream — one record per tessellated vertex, the count word CPU-written.  This
         * is what makes transform feedback (and the geometry stage) work when the GPU
         * tessellator's worst-case scratch does not fit its budget and the draw fell back
         * here (e.g. many patches at once). */
        if (tp->capture && tessEnsureCapturePso(ctx, tp, prog_tes) && tp->cap_stride)
        {
            size_t cnt_off = off_patch_data + tessRingAlign((size_t)S->patch_count * tp->tes_patch_in_size);
            size_t recs_off = cnt_off + 16;
            size_t need = 16 + tessRingAlign((size_t)S->tess_vertex_count * tp->cap_stride * 4);

            if (tRingCap[ring_slot] < recs_off + need)
                NSLog(@"MGL ERROR: tess: CPU reference capture scratch overflow");
            else
            {
                if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
                {
                    _currentCommandBuffer = [_commandQueue commandBuffer];
                    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
                }
                {
                    id<MTLComputeCommandEncoder> ccenc = [_currentCommandBuffer computeCommandEncoder];

                    if (ccenc)
                    {
                        uint32_t n = S->tess_vertex_count;

                        memcpy((uint8_t *)ring.contents + cnt_off, &n, 4);
                        [ccenc setComputePipelineState: (__bridge id<MTLComputePipelineState>)tp->capture_pso];
                        tessBindStageUniforms(ctx, ccenc, prog_tes, _TESS_EVALUATION_SHADER);
                        tessBindStageTextures(ctx, ccenc, prog_tes, _TESS_EVALUATION_SHADER);
                        [ccenc setBuffer: ring offset: off_coords atIndex: MGL_TESS_TES_COORDS];
                        [ccenc setBuffer: cp_source offset: cp_source_off atIndex: MGL_TESS_TES_CP];
                        [ccenc setBuffer: ring offset: off_patch_data atIndex: MGL_TESS_TES_PATCH];
                        {
                            /* [1] = patches per instance: gl_PrimitiveID restarts per instance */
                        uint32_t cparams[4] = { (uint32_t)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in), (uint32_t)patches, 0, 0 };

                            [ccenc setBytes: cparams length: sizeof cparams atIndex: MGL_TESS_TES_PARAMS];
                        }
                        [ccenc setBuffer: ring offset: cnt_off atIndex: MGL_TESS_TES_CAP_COUNT];
                        [ccenc setBuffer: ring offset: recs_off atIndex: MGL_TESS_TES_CAP_RECS];
                        [ccenc dispatchThreadgroups: MTLSizeMake(((NSUInteger)n + 63) / 64, 1, 1)
                              threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
                        [ccenc endEncoding];
                        tessPublishCapture(ctx, ring, tp, recs_off, n, off_indices,
                                           S->tess_index_count, cnt_off);
                        /* CPU reference path: one record per unique tessellated
                         * coordinate, so the published index list IS the record map
                         * (opposite of the GPU path's emission-order permute) */
                        ((struct MGLExtState_tess *)ctx->state.ext_tess)->capture.emission_order = false;
                    }
                }
            }
        }

        /* processGLState builds the render encoder and the (TES-based) pipeline */
        if (![self processGLState: true])
        {
            tessRingRetireOn(_currentCommandBuffer, ring_slot);
             return false;
        }
        enc = _currentRenderEncoder;
        if (!enc)
        {
            tessRingRetireOn(_currentCommandBuffer, ring_slot);
             return false;
        }

        [enc setVertexBuffer: ring offset: off_coords atIndex: MGL_TESS_TES_COORDS];
        [enc setVertexBuffer: cp_source offset: cp_source_off atIndex: MGL_TESS_TES_CP];
        [enc setVertexBuffer: ring offset: off_patch_data atIndex: MGL_TESS_TES_PATCH];
        {
            /* the plain TES takes mgl_tess_params as a buffer POINTER (see above);
             * this draw's exclusive ring slot, never overwritten by a later draw */
            memcpy((uint8_t *)ring.contents + off_params, tes_params, sizeof tes_params);
            [enc setVertexBuffer: ring offset: off_params atIndex: MGL_TESS_TES_PARAMS];
        }

        if (tTrace())
            NSLog(@"MGL tess: CPU reference path drawing %d idx", S->tess_index_count);
        if (tp->point_mode)
            [enc drawPrimitives: MTLPrimitiveTypePoint vertexStart: 0
                     vertexCount: (NSUInteger)S->tess_vertex_count];
        else
            [enc drawIndexedPrimitives: prim indexCount: (NSUInteger)S->tess_index_count
                             indexType: MTLIndexTypeUInt32 indexBuffer: ring indexBufferOffset: off_indices];

        /* the CPU reference path's block retires with the render buffer as well */
        {
            int rslot = ring_slot;
            id<MTLCommandBuffer> rcb = _currentCommandBuffer;

            if (tRingTrace())
                fprintf(stderr, "MGL ring: register(cpu) slot=%d cb=%d\n", rslot,
                        rcb ? (int)rcb.status : -1);
            tessRingRetireOn(rcb, rslot);   /* the wait above may already have
                                            * completed (or committed) the buffer */
        }
        ctx->state.dirty_bits = DIRTY_ALL;
    }   /* CPU reference path */

    if (tRingTrace())
        fprintf(stderr, "MGL ring: stage fallthrough-return slot=%d gpu_emu=%d\n",
                ring_slot, gpu_emulated);
    return true;
}


}
/* read the draw's index list into a freshly malloc'd widened (+ base vertex) array;
 * the caller copies it into the per-draw scratch block */
static bool tessReadIndicesInto(GLMContext ctx, GLenum type, size_t offset, GLsizei count,
                                GLint basevertex, uint32_t **dst, uint32_t *minv, uint32_t *maxv)
{
    Buffer *ebuf = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
    const uint8_t *ptr;
    size_t esize = (type == GL_UNSIGNED_SHORT) ? 2 : (type == GL_UNSIGNED_INT) ? 4 : 1;
    uint32_t lo = 0xffffffff, hi = 0;
    uint32_t *out;

    *dst = NULL;
    if (!ebuf || !ebuf->data.buffer_data)
         return false;
    if (offset > (size_t)ebuf->size || (size_t)count * esize > (size_t)ebuf->size - offset)
         return false;

    mgl_buffers_prepare_cpu_read(ctx, ebuf);
    ptr = (const uint8_t *)ebuf->data.buffer_data + offset;

    out = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
    if (!out)
         return false;

    for (GLsizei i = 0; i < count; i++)
    {
        uint32_t v;
        switch (type)
        {
            case GL_UNSIGNED_SHORT: v = ((const uint16_t *)ptr)[i]; break;
            case GL_UNSIGNED_INT:   v = ((const uint32_t *)ptr)[i]; break;
            default:                v = ptr[i]; break;
        }
        out[i] = (uint32_t)basevertex + v;
        if (out[i] < lo) lo = out[i];
        if (out[i] > hi) hi = out[i];
    }
    if (lo == 0xffffffff) { lo = 0; hi = 0; }
    *dst = out;
    *minv = lo;
    *maxv = hi;
    return true;
}

/* forward declarations (used by the draw above) */
static bool tessBindVSAttributes(GLMContext ctx, id<MTLComputeCommandEncoder> enc, TessVSPlan *plan, GLuint vertex_count);
static bool tessConvertShadowStream(GLMContext ctx, TessAttribSlot *s, GLuint vertex_count);

static bool tessDrawPatchesHook(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                                GLint first, GLsizei count, GLsizei instances,
                                GLint basevertex, GLint baseinstance);

/* bind the attribute/const/shadow buffers of the VS kernel compute pipeline */
static bool tessBindVSAttributes(GLMContext ctx, id<MTLComputeCommandEncoder> enc, TessVSPlan *plan, GLuint vertex_count)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

    /* per-binding source buffers */
    for (int b = 0; b < 16; b++)
    {
        int idx = plan->binding_buffer_index[b];
        Buffer *buf = idx < 0 ? NULL : va->binding[b].buffer;

        if (idx < 0)
            continue;
        if (tTrace())
            NSLog(@"MGL tess: VS binding %d -> mtl index %d (buf=%p mtl=%p fetch...)", b, idx,
                  (void *)buf, buf ? buf->data.mtl_data : NULL);
        if (!buf || !buf->data.buffer_data)
        {
            [enc setBuffer: nil offset: 0 atIndex: (NSUInteger)idx];
            continue;
        }
        {
            MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
            [self2 bindMTLBuffer: buf];
            if (tTrace())
                NSLog(@"MGL tess: VS binding %d idx %d: mtl_data=%p", b, idx, buf->data.mtl_data);
        }
        if (buf->data.mtl_data)
        {
            id<MTLBuffer> mtl = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
            [enc setBuffer: mtl offset: plan->binding_offset[b] atIndex: (NSUInteger)idx];
        }
        else
        {
            /* small buffer without a Metal object: compute stage-in needs a real buffer */
            if (!tessBufferEnsure(((__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj)->_device, &tSmallAttr, 8192))
                 return false;
            memcpy(tSmallAttr.contents, (const void *)(uintptr_t)buf->data.buffer_data,
                   (size_t)buf->size < 8192 ? (size_t)buf->size : 8192);
            [enc setBuffer: tSmallAttr offset: plan->binding_offset[b] atIndex: (NSUInteger)idx];
        }
    }

    /* current generic values (disabled arrays): one 16-byte slot per location */
    if (plan->const_buffer >= 0)
    {
        struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
        uint8_t data[MAX_ATTRIB_SLOTS * 16];

        memset(data, 0, sizeof data);
        if (vs)
        {
            for (int i = 0; i < plan->count; i++)
            {
                TessAttribSlot *s = &plan->slot[i];
                if (s->fetch == TESS_FETCH_CONST && !s->dbl)
                    memcpy(data + s->loc * 16, vs->current[s->loc].v.f, 16);
                else if (s->fetch == TESS_FETCH_CONST)
                {
                    /* word slots of the current double value (the GL location holds
                       all four components: one glVertexAttribL*d call feeds both) */
                    MGLCurrentAttrib *cur = &vs->current[s->glloc];
                    int c0 = (s->dbl == 2) ? 2 : 0;
                    GLuint *w = (GLuint *)(data + s->loc * 16);
                    for (int c = c0; c < c0 + 2; c++)
                    {
                        GLdouble v;
                        switch (cur->kind)
                        {
                            case MGL_VA_KIND_INT:    v = (GLdouble)cur->v.i[c]; break;
                            case MGL_VA_KIND_UINT:   v = (GLdouble)cur->v.u[c]; break;
                            case MGL_VA_KIND_DOUBLE: v = cur->d[c]; break;
                            default:                 v = (GLdouble)cur->v.f[c]; break;
                        }
                        memcpy(&w[(c - c0) * 2], &v, 8);
                    }
                }
            }
        }
        [enc setBytes: data length: sizeof(data) atIndex: (NSUInteger)plan->const_buffer];
    }

    /* CPU-converted streams */
    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        if (s->fetch != TESS_FETCH_SHADOW)
            continue;
        if (s->dbl == 2)
            continue;   /* both word slots share one stream, converted by the low slot */
        if (!tessConvertShadowStream(ctx, s, vertex_count))
             return false;
        [enc setBuffer: tShadowBuf offset: 0 atIndex: (NSUInteger)s->shadow_buffer];
    }
    return true;
}

/* CPU conversion of a non-Metal-fetchable attribute stream (same decisions as the
   vattrib domain's shadow streams: GL_FIXED / doubles / BGRA / misaligned) */
static bool tessConvertShadowStream(GLMContext ctx, TessAttribSlot *s, GLuint vertex_count)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    int own = (s->dbl == 2) ? s->loc - 1 : s->loc;
    MGLVertexAttribFormat *f = &va->fmt[own];
    MGLVertexBinding *b = &va->binding[f->binding];
    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    const uint8_t *src = b->buffer && b->buffer->data.buffer_data ?
                         (const uint8_t *)b->buffer->data.buffer_data : NULL;
    size_t stride = b->stride ? (size_t)b->stride : (size_t)mgl_va_element_size(f->type, f->size);
    size_t stride_out = s->dbl ? ((size_t)(s->shadow_words) * 4 + 15) & ~(size_t)15 : 16;
    size_t need = (size_t)vertex_count * stride_out + 16;

    if (!tessBufferEnsure(self2->_device, &tShadowBuf, need))
         return false;

    if (src)
        mgl_buffers_prepare_cpu_read(ctx, b->buffer);

    {
        uint8_t *dst = (uint8_t *)tShadowBuf.contents;
        for (GLuint i = 0; i < vertex_count; i++)
        {
            const uint8_t *el = src + (size_t)b->offset + (size_t)f->relativeoffset + (size_t)i * stride;
            float *out = (float *)(dst + (size_t)i * stride_out);

            if (!src)
            {
                memset(out, 0, stride_out);
                continue;
            }

            if (s->dbl)
            {
                /* raw u32 words of the double value (see mgl_va_fetch_dbl_words) */
                mgl_va_fetch_dbl_words(f, el, (GLuint *)out);
            }
            else if (s->kind == MGL_VA_KIND_INT || s->kind == MGL_VA_KIND_UINT)
            {
                GLint iv[4];
                mgl_va_fetch_int(f, el, iv);
                memcpy(out, iv, sizeof iv);
            }
            else
            {
                GLfloat fv[4];
                mgl_va_fetch_float(f, el, fv);
                memcpy(out, fv, sizeof fv);
            }
        }
    }
    return true;
}

static bool tessDrawPatchesHook(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                                GLint first, GLsizei count, GLsizei instances,
                                GLint basevertex, GLint baseinstance)
{
    return [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj
            extTessDrawPatches: indexed type: type offset: offset
            first: first count: count instances: instances
            basevertex: basevertex baseinstance: baseinstance];
}

#pragma mark ext hooks

- (void) bindExtTessFuncsToGLMContext: (GLMContext) glm_ctx
{
    if (!glm_ctx->mtl_funcs.ext_tess)
        glm_ctx->mtl_funcs.ext_tess = (struct MGLExtMetal_tess *)calloc(1, sizeof(struct MGLExtMetal_tess));
    if (!glm_ctx->mtl_funcs.ext_tess)
        return;
    glm_ctx->mtl_funcs.ext_tess->draw_patches = tessDrawPatchesHook;
}


/* compile the GPU tessellator kernels (once) */
static bool tessKernels(GLMContext ctx)
{
    if (tCountPso && tScanPso && tEmitPso && tArgsPso && tBlobPso && tPermutePso)
        return true;

    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    NSError *err = nil;

    if (!tKernelLib)
    {
        tKernelLib = [self2->_device newLibraryWithSource: kTessKernelSource options: nil error: &err];
        if (!tKernelLib)
        {
            NSLog(@"MGL ERROR: tess: tessellator kernels failed to compile: %@", err.localizedDescription);
            return false;
        }
    }
    if (!tCountPso) tCountPso = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_count"] error: &err];
    if (!tScanPso)  tScanPso  = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_scan"]  error: &err];
    if (!tEmitPso)  tEmitPso  = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_emit"]  error: &err];
    if (!tArgsPso)  tArgsPso  = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_args"]  error: &err];
    if (!tBlobPso)  tBlobPso  = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_blob"]  error: &err];
    if (!tPermutePso) tPermutePso = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_permute"] error: &err];
    if (!tCountPso || !tScanPso || !tEmitPso || !tArgsPso || !tBlobPso || !tPermutePso)
    {
        NSLog(@"MGL ERROR: tess: tessellator kernel PSO failed: %@", err.localizedDescription);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ hardware pipeline */

/* half -> float (factor readback on the emulated path) */
static float tessHalfToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;

    if (exp == 0)
    {
        if (man == 0)
            bits = sign;
        else
        {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    }
    else if (exp == 0x1f)
        bits = sign | 0x7f800000 | (man << 13);
    else
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* float -> half (default factor levels on the hardware path) */
static uint16_t tessFloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t man = bits & 0x7fffff;

    if (((bits >> 23) & 0xff) == 0xff)
        return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));   /* inf/nan */
    if (((bits >> 23) & 0xff) == 0)
    {
        if (man == 0)
            return (uint16_t)sign;
        /* subnormal float -> half subnormal (rare for levels) */
        int e = -1;
        while (!(man & 0x400000)) { man <<= 1; e--; }
        man >>= 13;
        return (uint16_t)(sign | ((uint32_t)(e + 1) << 10) | man);
    }
    if (exp >= 0x1f)
        return (uint16_t)(sign | 0x7c00);                        /* overflow -> inf */
    if (exp <= 0)
        return (uint16_t)(sign | 0x3ff);                         /* underflow -> max subnormal */

    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

/* MSL member type -> Metal vertex format of the TES interface attributes */
static MTLVertexFormat tessMemberFormat(const char *type)
{
    if (!strcmp(type, "float"))  return MTLVertexFormatFloat;
    if (!strcmp(type, "float2")) return MTLVertexFormatFloat2;
    if (!strcmp(type, "float3")) return MTLVertexFormatFloat3;
    if (!strcmp(type, "float4")) return MTLVertexFormatFloat4;
    if (!strcmp(type, "int"))    return MTLVertexFormatInt;
    if (!strcmp(type, "int2"))   return MTLVertexFormatInt2;
    if (!strcmp(type, "int3"))   return MTLVertexFormatInt3;
    if (!strcmp(type, "int4"))   return MTLVertexFormatInt4;
    if (!strcmp(type, "uint"))   return MTLVertexFormatUInt;
    if (!strcmp(type, "uint2"))  return MTLVertexFormatUInt2;
    if (!strcmp(type, "uint3"))  return MTLVertexFormatUInt3;
    if (!strcmp(type, "uint4"))  return MTLVertexFormatUInt4;
    if (!strcmp(type, "half"))   return MTLVertexFormatHalf;
    if (!strcmp(type, "half2"))  return MTLVertexFormatHalf2;
    if (!strcmp(type, "half3"))  return MTLVertexFormatHalf3;
    if (!strcmp(type, "half4"))  return MTLVertexFormatHalf4;
    return MTLVertexFormatInvalid;
}

/* vertex descriptor of the post-tessellation TES:
 *   buffer MGL_TESS_VD_CP      control point records  (PerPatchControlPoint)
 *   buffer MGL_TESS_VD_PATCH   per-patch data          (PerPatch)
 *   buffer MGL_TESS_VD_FACTORS half tess levels        (PerPatch, stride 12)
 */
- (MTLVertexDescriptor *) extTessVertexDescriptor
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);
    MTLVertexDescriptor *vd;

    if (!tp)
        return nil;
    vd = [MTLVertexDescriptor vertexDescriptor];

    bool any_cp = false;
    for (int i = 0; i < tp->num_tes_cp; i++)
    {
        MGLTessPatchMember *m = &tp->tes_cp[i];
        MTLVertexFormat fmt = tessMemberFormat(m->type);
        size_t offset = m->offset;

        if (fmt == MTLVertexFormatInvalid || m->attr < 0)
            continue;

        /* offsets were remapped to the full record layout (by GL location) at link time */
        vd.attributes[m->attr].format = fmt;
        vd.attributes[m->attr].offset = (NSUInteger)offset;
        vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_CP;
        any_cp = true;
    }
    /* stride: the FULL control point record (TCS/VS output), even if the TES reads a subset.
     * Metal validates that every laid-out buffer index carries at least one attribute —
     * a TES that reads only patch data (no per-vertex ins, like tc_barriers' patch-array
     * test) leaves buffer 8 empty, and an orphan layout aborts the pipeline build */
    if (any_cp)
    {
        vd.layouts[MGL_TESS_VD_CP].stride = (NSUInteger)(tp->cp_record_full_size ?
            tp->cp_record_full_size : (tp->cp_record_size ? tp->cp_record_size : 16));
        vd.layouts[MGL_TESS_VD_CP].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
        vd.layouts[MGL_TESS_VD_CP].stepRate = 1;
    }
    if (getenv("MGL_TESS_DEBUG"))
    {
        NSMutableString *ds = [NSMutableString string];
        for (int i = 0; i < tp->num_tes_cp; i++)
        {
            MGLTessPatchMember *m = &tp->tes_cp[i];
            if (m->attr < 0) continue;
            [ds appendFormat:@" a%d@%zu", m->attr, m->offset];
        }
        [ds appendFormat:@" stride=%zu full=%zu cp_record_size=%zu ncp=%d",
            (size_t)vd.layouts[MGL_TESS_VD_CP].stride, tp->cp_record_full_size, tp->cp_record_size, tp->num_tes_cp];
        NSLog(@"MGL tess: tes vd CP:%@", ds);
    }

    bool any_patch = false, any_factors = false;
    for (int i = 0; i < tp->num_tes_patch_in; i++)
    {
        MGLTessPatchMember *m = &tp->tes_patch_in[i];
        MTLVertexFormat fmt;

        if (m->attr < 0)
            continue;

        if (m->is_tess_level == 1)
        {
            /* gl_TessLevelOuter / packed gl_TessLevel: float4 at the head of the float
             * level copy (GL needs the exact values; the half factor record quantises) */
            vd.attributes[m->attr].format = MTLVertexFormatFloat4;
            vd.attributes[m->attr].offset = 0;
            vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_FACTORS;
            any_factors = true;
            continue;
        }
        if (m->is_tess_level == 2)
        {
            /* gl_TessLevelInner: float2 at byte 16 (after the four outer floats) */
            vd.attributes[m->attr].format = MTLVertexFormatFloat2;
            vd.attributes[m->attr].offset = 16;
            vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_FACTORS;
            any_factors = true;
            continue;
        }

        fmt = tessMemberFormat(m->type);
        if (fmt == MTLVertexFormatInvalid)
            continue;
        /* m->offset is the canonical per-patch record offset (link time): identical
         * for the TCS kernel's rewritten patchOut struct, the blob and this
         * descriptor, matched by varying location rather than by name */
        vd.attributes[m->attr].format = fmt;
        vd.attributes[m->attr].offset = (NSUInteger)m->offset;
        vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_PATCH;
        any_patch = true;
    }
    if (any_patch)
    {
        vd.layouts[MGL_TESS_VD_PATCH].stride = (NSUInteger)(tp->has_tcs ?
            (tp->tcs_patch_out_size ? tp->tcs_patch_out_size : 16) :
            (tp->tes_patch_in_size ? tp->tes_patch_in_size : 16));
        vd.layouts[MGL_TESS_VD_PATCH].stepFunction = MTLVertexStepFunctionPerPatch;
        vd.layouts[MGL_TESS_VD_PATCH].stepRate = 1;
    }
    if (any_factors)
    {
        vd.layouts[MGL_TESS_VD_FACTORS].stride = 24;  /* mgl_tess_levels_f: float o[4], in[2] */
        vd.layouts[MGL_TESS_VD_FACTORS].stepFunction = MTLVertexStepFunctionPerPatch;
        vd.layouts[MGL_TESS_VD_FACTORS].stepRate = 1;
    }

    return vd;
}

/* tessellation stage of the render pipeline descriptor */
- (void) extTessConfigurePipeline: (MTLRenderPipelineDescriptor *) desc
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);
    if (!tp || !desc)
        return;

    switch (tp->spacing)
    {
        case MGL_TESS_SPACING_FRACTIONAL_EVEN: desc.tessellationPartitionMode = MTLTessellationPartitionModeFractionalEven; break;
        case MGL_TESS_SPACING_FRACTIONAL_ODD:  desc.tessellationPartitionMode = MTLTessellationPartitionModeFractionalOdd; break;
        default:                               desc.tessellationPartitionMode = MTLTessellationPartitionModeInteger; break;
    }
    desc.maxTessellationFactor = MGL_TESS_MAX_LEVEL;
    desc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
    desc.tessellationOutputWindingOrder = tp->vertex_order_ccw ? MTLWindingCounterClockwise : MTLWindingClockwise;
}

/* Metal requires MTLRenderPipelineDescriptor.inputPrimitiveTopology to be set when
 * the vertex function writes [[render_target_array_index]] / [[viewport_array_index]]
 * — i.e. gl_Layer / gl_ViewportIndex written directly from the vertex or evaluation
 * stage (GL_ARB_shader_viewport_layer_array); without it the PSO build is rejected
 * ("inputPrimitiveTopology is not specified").  The geometry emulation sets its own
 * (geomConfigureTopology); this covers the GS-less pipelines, whose final
 * pre-rasterization stage is the plain vertex function or the post-tessellation
 * vertex function (TES). */
- (void) tessConfigureLayeredTopology: (MTLRenderPipelineDescriptor *) pd
{
    if (!pd || pd.inputPrimitiveTopology != MTLPrimitiveTopologyClassUnspecified)
        return;

    Program *program = STATE(program);
    if (!program || program->shader_slots[_GEOMETRY_SHADER])
        return;                       /* geometry domain owns that path */

    int stage = program->shader_slots[_TESS_EVALUATION_SHADER] ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    const char *msl = program->spirv[stage].msl_str;
    if (!msl)
        return;

    /* does this stage write the layer at all?  (no caching: MSL strings are
     * freed/reallocated at the same addresses across programs, so a
     * pointer-keyed cache gets poisoned; the scan only runs when the pipeline
     * descriptor is rebuilt, not per draw) */
    bool layered_out = strstr(msl, "[[render_target_array_index") != NULL;
    if (!layered_out)
        return;


    GLenum mode = ctx->state.ext_tess ? ctx->state.ext_tess->draw_mode : GL_NONE;
    switch (mode)
    {
        case GL_POINTS:
            pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
            break;
        case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP:
            pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
            break;
        case GL_PATCHES:
        {
            /* post-tessellation topology: the TES layout (point_mode wins,
             * isolines rasterize as lines, triangles/quads as triangles) */
            MGLTessProgram *tp = mgl_tess_active_program(ctx);
            if (tp && tp->point_mode)
                pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
            else if (tp && tp->domain == MGL_TESS_DOMAIN_ISOLINES)
                pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
            else
                pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            break;
        }
        default:
            pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            break;
    }
}

/* Compile options for MSL emitted by MGL's shader toolchain.
 *
 * The fp64 double-single library (MGL_FP64_FAST=1, MGL/src/fp64_df64.glsl) is
 * built from Knuth-style error-free transformations: quick-two-sum,
 * two-sum, fma-recovered product error.  Metal's default fast-math
 * REASSOCIATES those float expressions (e.g. e - (hi - s) -> (e - hi) + s),
 * which rounds away the low float of every pair and halves the precision;
 * observed live on M4 Pro.  When the MSL carries the library's df64_*
 * identifiers (they survive glslang -> SPIR-V -> SPIRV-Cross), compile with
 * precise math instead.  All other sources keep the historic default
 * (options:nil), so the default fp64 path is unaffected.
 */
- (MTLCompileOptions *) mslCompileOptionsForSource: (NSString *) src
{
    if (!src || [src rangeOfString: @"df64_"].location == NSNotFound)
        return nil;

    MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *))
    {
        opt.mathMode = MTLMathModeSafe;
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    }
    return opt;
}
@end
