# Settings for LUMI, whose GPU partition is AMD MI250X, and the reference for any HIP
# build. Taken from microhh_simple's config of the same name, minus the file-format
# libraries rte3d does not link.
#
# module --force purge
# module load LUMI/25.09
# module load partition/G
# module load craype-accel-amd-gfx90a
# module load PrgEnv-cray
# module load rocm
# module load cray-python

set(ENV{CXX} CC)

if (USEGPU)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    set(CMAKE_HIP_ARCHITECTURES gfx90a) # Adapt for other cards.
    set(GPU_TARGETS gfx90a)
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE) # Selects the HIP backend.
    set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
    set(Kokkos_ARCH_AMD_GFX90A ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(USER_CXX_FLAGS "-Wall -fPIC")
else()
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    set(USER_CXX_FLAGS "-Wall -fPIC -fopenmp")
endif()

set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG "-g -O0")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

# hipRAND, which the ray tracer draws its photons from, is added by the top-level
# CMakeLists when the HIP backend is on.
set(LIBS "")
