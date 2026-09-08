#pragma once

#include "source_functions.h"
#include "types.h"


// Longwave RTE solvers. Free functions on plain Views, as in rte_sw.h.
//
// Every solver works on a single g-point: nothing here carries a g-point dimension,
// so the working set is (nlay, ncol) whatever the spectral resolution. The caller
// loops g-points and accumulates.
namespace Rte_lw
{
    // No-scattering solver with multi-angle quadrature. Reference: lw_solver_noscat,
    // which sums single-angle solutions over a user-supplied set of secants and
    // weights.
    //
    // flux_up and flux_dn are overwritten. flux_up_jac, the surface-temperature
    // Jacobian, is *accumulated into*: it is spectrally integrated, hence (nlev, ncol)
    // and not one per g-point, so the caller zeroes it before the g-point loop. Pass
    // an empty View to skip it.
    //
    // The approximate-scattering rescaling of Tang et al. 2018 is not implemented; see
    // the note in rte_lw.cpp.
    void solver_noscat(
            const bool top_at_1,
            const Array_map_2d<const TF>& secants,   // (nmus, ncol) quadrature secants
            const Array_1d<const TF>& weights,       // (nmus)       quadrature weights
            const Array_map_2d<const TF>& tau,       // (nlay, ncol)
            const Source_func_lw& sources,
            const Array_map_1d<const TF>& sfc_emis,  // (ncol)
            const Array_map_1d<const TF>& inc_flux,  // (ncol) incident diffuse flux, may be empty
            const Array_map_2d<TF>& flux_up,         // (nlev, ncol)
            const Array_map_2d<TF>& flux_dn,         // (nlev, ncol)
            const Array_2d<TF>& flux_up_jac);        // (nlev, ncol), may be empty

    // Two-stream solver with scattering. Reference: lw_solver_2stream.
    void solver_2stream(
            const bool top_at_1,
            const Array_map_2d<const TF>& tau,       // (nlay, ncol)
            const Array_map_2d<const TF>& ssa,       // (nlay, ncol)
            const Array_map_2d<const TF>& g,         // (nlay, ncol)
            const Source_func_lw& sources,
            const Array_map_1d<const TF>& sfc_emis,  // (ncol)
            const Array_map_1d<const TF>& inc_flux,  // (ncol)
            const Array_map_2d<TF>& flux_up,         // (nlev, ncol)
            const Array_map_2d<TF>& flux_dn);        // (nlev, ncol)

    void init_python_bindings(py::module_& m);
}
