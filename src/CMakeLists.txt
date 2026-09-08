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
    "gas_concs_python.cpp")

add_library(rte3d_core STATIC ${sourcefiles})

target_include_directories(rte3d_core PRIVATE "../include" "../include_kernels" "../extern/pybind11/include" SYSTEM ${INCLUDE_DIRS})

target_link_libraries(rte3d_core PRIVATE pybind11::embed Kokkos::kokkos ${LIBS})
