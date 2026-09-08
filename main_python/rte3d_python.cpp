#include <pybind11/pybind11.h>

#include "types.h"
#include "runtime.h"
#include "rte_sw.h"
#include "rte_lw.h"
#include "optical_props.h"
#include "fluxes.h"
#include "gas_optics.h"


PYBIND11_MODULE(rte3d_python, m)
{
    m.doc() = "RTE-RRTMGP in Kokkos: row-major arrays with the g-point as outermost dimension.";

    Runtime::init_python_bindings(m);
    Rte_sw::init_python_bindings(m);
    Rte_lw::init_python_bindings(m);
    Optical_props::init_python_bindings(m);
    Fluxes::init_python_bindings(m);
    Gas_optics::init_python_bindings(m);
}
