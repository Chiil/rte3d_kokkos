# Settings for an Ubuntu workstation with an NVIDIA card, and the reference for any
# CUDA build. The GPU here is an RTX A4500 (Ampere, GA102, sm_86); adapt the two
# architecture lines for another card.
#
# nvcc comes from the NVIDIA HPC SDK:
#
# export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/bin:$PATH
#
# cmake .. -DSYST=ubuntu_cuda -DUSEGPU=1 -DUSESP=1

if (USEGPU)
    # Everything goes through nvcc, so that the __device__ functions in include/ and
    # include_kernels/ are compiled as device code. LUMI's HIP build has no such line
    # because the Cray compiler takes HIP source as it comes.
    set(ENV{CXX} ${CMAKE_CURRENT_SOURCE_DIR}/extern/kokkos/bin/nvcc_wrapper)
    set(ENV{NVCC_WRAPPER_DEFAULT_COMPILER} g++)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES 86) # Adapt for other cards.
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE) # Selects the CUDA backend.
    set(Kokkos_ARCH_AMPERE86 ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(USER_CXX_FLAGS "-Wall -fPIC")
else()
    set(ENV{CXX} g++)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    set(USER_CXX_FLAGS "-Wall -fPIC -fopenmp")
endif()

set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3")
set(USER_CXX_FLAGS_DEBUG "-g -O0")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

# cuRAND, which the ray tracer draws its photons from, is added by the top-level
# CMakeLists when the CUDA backend is on.
set(LIBS "")
