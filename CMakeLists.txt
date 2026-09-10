cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/config)


# Load system specific settings prior to initializing the project.
include(${SYST} OPTIONAL RESULT_VARIABLE SYSTINC)
if (${SYSTINC} STREQUAL "NOTFOUND")
    message(FATAL_ERROR "Config file config/" ${SYST} ".cmake does not exist.")
endif()


project(rte3d CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Switch for GPU usage. The backend (CUDA or HIP) is auto-detected from the Kokkos
# flags set by the system config file.
if (USEGPU)
    add_definitions("-DUSEGPU")
    if (Kokkos_ENABLE_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_definitions("-DUSECUDA")
        # The ray tracer draws its photons from cuRAND; nothing else needs it.
        set(LIBS ${LIBS} CUDA::curand)
        message(STATUS "GPU: Enabled (CUDA)")
    elseif (Kokkos_ENABLE_HIP)
        enable_language(HIP)
        find_package(HIP REQUIRED)
        add_definitions("-DUSEHIP")
        # The counterpart of cuRAND above.
        find_package(hiprand REQUIRED)
        set(LIBS ${LIBS} hip::hiprand)
        message(STATUS "GPU: Enabled (HIP)")
    else()
        message(FATAL_ERROR "USEGPU set but config enables neither Kokkos_ENABLE_CUDA nor Kokkos_ENABLE_HIP")
    endif()
else()
    message(STATUS "GPU: Disabled")
endif()


# Switch for float precision.
if (USESP)
    add_definitions("-DUSE_SINGLE_PRECISION")
    message(STATUS "Precision: Single (32-bits floats)")
else()
    message(STATUS "Precision: Double (64-bits floats)")
endif()


# Set the default build type to RELEASE.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release." FORCE)
else()
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${BUILD_TYPE} CACHE STRING
    "Choose the type of build, options are: None Debug Release." FORCE)
endif()


# Transfer the compiler options. This is not the advised CMake way, but
# it helps most users to be able to set the full flags in the system .conf.
if (NOT HAS_CACHE)
    set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS} CACHE STRING
        "Flags used by the CXX-compiler during all build types." FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE ${USER_CXX_FLAGS_RELEASE} CACHE STRING
        "Flags used by the CXX-compiler during release build types." FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG ${USER_CXX_FLAGS_DEBUG} CACHE STRING
        "Flags used by the CXX-compiler during debug build types." FORCE)

    message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})
    set(HAS_CACHE TRUE CACHE BOOL "CMakeCache.txt created." FORCE)

    # Make sure that ccmake only contains build type.
    mark_as_advanced(HAS_CACHE)
endif()


# Set pybind11 Python discovery mode to NEW
set(PYBIND11_FINDPYTHON NEW)

# Load the subdirectories with compilation targets.
add_subdirectory(extern/kokkos)
add_subdirectory(extern/pybind11)
add_subdirectory(src)
add_subdirectory(main_python)
