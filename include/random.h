#pragma once

#include <cstdint>

#include "types.h"

#if defined(USECUDA)
    #include <curand_kernel.h>
#elif defined(USEHIP)
    #include <hiprand/hiprand_kernel.h>
#endif


// Random numbers for the Monte Carlo ray tracer.
//
// This is the second and last file that knows which backend is in use, and it exists
// for the same reason as types.h: the vendors ship the generators, and their device
// APIs are not interchangeable, so the #ifdef lives here rather than in the kernels.
// Everything above this header sees two types, Rng and Qrng_2d, and nothing else.
//
//   backend | uniforms          | launch positions
//   --------|-------------------|--------------------------------
//   CUDA    | curand XORWOW     | curand scrambled Sobol32, Joe-Kuo
//   HIP     | hiprand XORWOW    | hiprand scrambled Sobol32, Joe-Kuo
//   host    | xorshift64*       | xorshift64*
//
// The tracer draws a photon's launch pixel from a two-dimensional low-discrepancy
// sequence rather than a pseudo-random one, which is what keeps the photons evenly
// spread over the domain and is worth roughly a factor of two in photon count for the
// same noise. cuRAND and rocRAND both provide that; the host build does not get one.
// It falls back to the pseudo-random stream, so a CPU run is noisier than a GPU run at
// the same photon count. That build is there to run the test suite, where the
// tolerances are set from the photon count anyway, so it is not worth carrying a
// Sobol implementation of our own.


// The vendors' generators are __device__ functions, where KOKKOS_INLINE_FUNCTION is
// __host__ __device__, and nvcc refuses the one calling the other. So everything that
// touches a generator carries this instead, all the way up to the photon walk in
// raytracer_kernels.h. It costs nothing: none of it is ever called on the host.
#if defined(USECUDA) || defined(USEHIP)
    #define RTE3D_DEVICE_FUNCTION __device__ inline
#else
    #define RTE3D_DEVICE_FUNCTION inline
#endif


namespace Rand
{
    #if defined(USECUDA)
    using Rng_state = curandState;
    using Qrng_state = curandStateScrambledSobol32_t;
    #elif defined(USEHIP)
    using Rng_state = hiprandState;
    using Qrng_state = hiprandStateScrambledSobol32_t;
    #endif


    // The direction numbers and scramble constants of the quasi-random sequence, on
    // the device. Trivially copyable, so a kernel captures it by value; empty on the
    // host build, where Qrng_2d ignores it.
    struct Qrng_vectors
    {
        unsigned int* vectors = nullptr;    // (2, 32) direction numbers
        unsigned int* constants = nullptr;  // (2) scramble constants
    };


    // Owns what the above points at, for as long as the tracer runs. The vendor
    // returns the tables from a host call, so they are copied to the device once.
    struct Qrng_table
    {
        Array_1d<unsigned int> vectors;
        Array_1d<unsigned int> constants;

        static Qrng_table make();

        Qrng_vectors view() const
        { return Qrng_vectors{vectors.data(), constants.data()}; }
    };


    // A uniform deviate in [0, 1). Seeded per photon, so a run is reproducible for a
    // given photon count and backend.
    struct Rng
    {
        RTE3D_DEVICE_FUNCTION
        explicit Rng(const unsigned int seed)
        {
            #if defined(USECUDA)
            curand_init(seed, seed, 0, &state);
            #elif defined(USEHIP)
            hiprand_init(seed, seed, 0, &state);
            #else
            // SplitMix64 on the seed, so that neighbouring seeds do not start
            // correlated streams; xorshift64* takes it from there.
            state = seed + 0x9e3779b97f4a7c15ULL;
            state = (state ^ (state >> 30)) * 0xbf58476d1ce4e5b9ULL;
            state = (state ^ (state >> 27)) * 0x94d049bb133111ebULL;
            state = state ^ (state >> 31);
            state |= 1ULL;
            #endif
        }

        RTE3D_DEVICE_FUNCTION
        TF operator()()
        {
            // One minus the deviate, as the reference does: the vendors return
            // (0, 1] and the sampling wants [0, 1).
            #if defined(USECUDA)
            if constexpr (sizeof(TF) == sizeof(float))
                return TF(1.) - curand_uniform(&state);
            else
                return TF(1.) - curand_uniform_double(&state);
            #elif defined(USEHIP)
            if constexpr (sizeof(TF) == sizeof(float))
                return TF(1.) - hiprand_uniform(&state);
            else
                return TF(1.) - hiprand_uniform_double(&state);
            #else
            return TF(next() >> 11) * TF(1./9007199254740992.);   // 2^-53
            #endif
        }

        #if defined(USECUDA) || defined(USEHIP)
        Rng_state state;
        #else
        RTE3D_DEVICE_FUNCTION
        std::uint64_t next()
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545f4914f6cdd1dULL;
        }

        std::uint64_t state;
        #endif
    };


    // The two-dimensional sequence the launch pixel is drawn from. next() gives a pair
    // of 32-bit integers, which the caller maps onto the horizontal grid.
    //
    // offset is the photon's index in the global sequence, so that the threads of one
    // launch together walk one contiguous block of it.
    struct Qrng_2d
    {
        RTE3D_DEVICE_FUNCTION
        Qrng_2d(const Qrng_vectors& v, const unsigned int offset)
            #if !defined(USECUDA) && !defined(USEHIP)
            : rng_x(offset*2 + 1), rng_y(offset*2 + 2)
            #endif
        {
            #if defined(USECUDA)
            curand_init(v.vectors, v.constants[0], offset, &state_x);
            curand_init(v.vectors + 32, v.constants[1], offset, &state_y);
            #elif defined(USEHIP)
            hiprand_init(v.vectors, v.constants[0], offset, &state_x);
            hiprand_init(v.vectors + 32, v.constants[1], offset, &state_y);
            #else
            (void) v;
            #endif
        }

        RTE3D_DEVICE_FUNCTION
        void next(unsigned int& x, unsigned int& y)
        {
            #if defined(USECUDA)
            x = curand(&state_x);
            y = curand(&state_y);
            #elif defined(USEHIP)
            x = hiprand(&state_x);
            y = hiprand(&state_y);
            #else
            x = static_cast<unsigned int>(rng_x.next() >> 32);
            y = static_cast<unsigned int>(rng_y.next() >> 32);
            #endif
        }

        #if defined(USECUDA) || defined(USEHIP)
        Qrng_state state_x;
        Qrng_state state_y;
        #else
        Rng rng_x;
        Rng rng_y;
        #endif
    };
}
