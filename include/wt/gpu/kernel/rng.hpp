/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "math.hpp"

/*
 * Random numbers for GPU path tracing.
 *
 * Counter based: the n-th number drawn by a path is a function of (seed, path
 * index, n), so paths need no stored generator state and any path can be
 * replayed exactly. The function is Philox4x32-10 (Salmon et al., "Parallel
 * Random Numbers: As Easy as 1, 2, 3", SC 2011), the generator of cuRAND and
 * NumPy, which needs only 32-bit integer arithmetic.
 */

namespace wt::gpu::kernel {

struct u32x4 {
    u32 x, y, z, w;
};

WT_GPU_FN u32 mulhi(u32 a, u32 b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#elif defined(__METAL_VERSION__)
    return metal::mulhi(a, b);
#else
    return u32((static_cast<unsigned long long>(a)*b) >> 32);
#endif
}

/** @brief Philox4x32-10 of `counter` with `key` (k0,k1). */
WT_GPU_FN u32x4 philox4x32_10(u32x4 c, u32 k0, u32 k1) {
    for (int round=0; round<10; ++round) {
        const u32 hi0 = mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u*c.x;
        const u32 hi1 = mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u*c.z;
        c = { hi1^c.y^k0, lo1, hi0^c.w^k1, lo0 };
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

/** @brief A uniform number in [0,1) from 32 random bits (or from 64 with 64-bit `real`). */
WT_GPU_FN real to_unit(u32 a, u32 b) {
#if defined(WT_GPU_DBL_PRECISION)
    // 53 bits
    return (double(a>>5)*67108864.0 + double(b>>6)) * (1.0/9007199254740992.0);
#else
    (void)b;
    // 24 bits
    return float(a>>8) * (1.0f/16777216.0f);
#endif
}

struct rng {
    u32 seed;
    u32 path_lo, path_hi;
    u32 counter;

    WT_GPU_FN static rng for_path(u32 seed, u32 path_lo, u32 path_hi) {
        return { seed, path_lo, path_hi, 0 };
    }

    WT_GPU_FN u32x4 next() {
        return philox4x32_10(u32x4{ counter++, 0, path_lo, path_hi }, seed, 0x5bd1e995u);
    }

    /** @brief A uniform number in [0,1). */
    WT_GPU_FN real r() {
        const u32x4 h = next();
        return to_unit(h.x, h.y);
    }
    /** @brief Two uniform numbers in [0,1). */
    WT_GPU_FN vec2 r2() {
        const u32x4 h = next();
        return { to_unit(h.x, h.y), to_unit(h.z, h.w) };
    }
};

}
