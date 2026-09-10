file(GLOB sourcefiles
    "runtime.cpp"
    "runtime_python.cpp"
    "rte_sw.cpp"
    "rte_sw_python.cpp"
    "rte_lw.cpp"
    "rte_lw_python.cpp"
    "optical_props.cpp"
    "optical_props_python.cpp"
    "fluxes.cpp"
    "fluxes_python.cpp"
    "gas_optics.cpp"
    "gas_optics_python.cpp"
    "gas_concs.cpp"
    "gas_concs_python.cpp"
    "gas_optics_load.cpp"
    "gas_optics_load_python.cpp"
    "cloud_optics.cpp"
    "cloud_optics_python.cpp"
    "random.cpp"
    "raytracer.cpp"
    "raytracer_python.cpp")

# The ray tracer is the one place in rte3d where correctly-rounded transcendentals are
# not worth their price. It samples a Monte Carlo distribution, where two ulp on a sine
# is far below the noise the photon count leaves behind. Without this, Kokkos::sin, cos
# and log lower to the software implementations -- argument-reduction slow paths and
# all -- where rte-rrtmgp-cpp's kernel gets single hardware instructions. Measured on
# RCEMIP at 65536 columns, single precision: the kernel halves in size, 3816 SASS
# instructions to 1992, and the solve is 1 to 3 percent faster, with the fluxes moving
# by 0.01 percent per column, a fraction of the Monte Carlo noise.
#
# Scoped to this one file on purpose. Gas optics and the transport solvers are compared
# against the Fortran reference at 1e-13 and have to keep IEEE semantics.
if (USEGPU)
    if (Kokkos_ENABLE_CUDA)
        set_source_files_properties(raytracer.cpp PROPERTIES COMPILE_OPTIONS "-use_fast_math")
    elseif (Kokkos_ENABLE_HIP)
        set_source_files_properties(raytracer.cpp PROPERTIES COMPILE_OPTIONS "-ffast-math")
    endif()
endif()

add_library(rte3d_core STATIC ${sourcefiles})

target_include_directories(rte3d_core PRIVATE "../include" "../include_kernels" "../extern/pybind11/include" SYSTEM ${INCLUDE_DIRS})

target_link_libraries(rte3d_core PRIVATE pybind11::embed Kokkos::kokkos ${LIBS})
