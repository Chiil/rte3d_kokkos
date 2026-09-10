#include "random.h"


// The direction numbers of the quasi-random sequence come from a host call into the
// vendor library, so they are fetched once and copied to the device. The host build
// has no such sequence and returns empty Views; Rand::Qrng_2d ignores them there.
Rand::Qrng_table Rand::Qrng_table::make()
{
    #if defined(USECUDA) || defined(USEHIP)
    constexpr int ndim = 2;         // the launch pixel is two-dimensional
    constexpr int nbit = 32;        // direction numbers per dimension

    unsigned int* vectors_h = nullptr;
    unsigned int* constants_h = nullptr;

    #if defined(USECUDA)
    curandDirectionVectors32_t* v = nullptr;
    curandGetDirectionVectors32(&v, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    curandGetScrambleConstants32(&constants_h);
    #else
    hiprandDirectionVectors32_t* v = nullptr;
    hiprandGetDirectionVectors32(&v, HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    hiprandGetScrambleConstants32(&constants_h);
    #endif

    vectors_h = reinterpret_cast<unsigned int*>(v);

    Qrng_table table;
    table.vectors = Array_1d<unsigned int>(
            Kokkos::view_alloc("qrng_vectors", Kokkos::WithoutInitializing), ndim*nbit);
    table.constants = Array_1d<unsigned int>(
            Kokkos::view_alloc("qrng_constants", Kokkos::WithoutInitializing), ndim);

    Kokkos::deep_copy(table.vectors,
            Array_map_1d_host<const unsigned int>(vectors_h, ndim*nbit));
    Kokkos::deep_copy(table.constants,
            Array_map_1d_host<const unsigned int>(constants_h, ndim));

    return table;
    #else
    return Qrng_table();
    #endif
}
