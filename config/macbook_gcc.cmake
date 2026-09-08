# This is a settings file for a Macbook with Homebrew, using gcc.
#
# Homebrew's gcc is versioned (g++-15, g++-16, ...) and the plain `g++` on a Mac is
# clang, so search for a versioned one rather than hard-coding a release.
if (NOT DEFINED ENV{CXX})
    find_program(GNU_CXX NAMES g++-16 g++-15 g++-14 g++-13)
    find_program(GNU_CC  NAMES gcc-16 gcc-15 gcc-14 gcc-13)
    if (NOT GNU_CXX)
        message(FATAL_ERROR "No versioned Homebrew g++ found. Install one with `brew install gcc`, "
                            "or set CXX yourself.")
    endif()
    set(ENV{CXX} ${GNU_CXX})
    set(ENV{CC} ${GNU_CC})
    message(STATUS "Compiler: " $ENV{CXX})
endif()

set(USER_CXX_FLAGS "-Wall -fPIC -fopenmp")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG "-g -O0")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

include_directories("/opt/homebrew/include")
link_directories("/opt/homebrew/lib")

# rte3d has no external library dependencies.
set(LIBS "")
