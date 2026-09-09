#pragma once

#include "types.h"


// Shortwave RTE solvers. Free functions on plain Views: no solver object, no state.
//
// Every solver works on a single g-point, so arrays are (nlay, ncol) or (nlev, ncol)
// with nlev = nlay+1, row-major, column fastest-varying, and the boundary conditions
// are (ncol). mu0 is (nlay, ncol) and shared by every g-point. The caller loops
// g-points and accumulates; nothing here carries a g-point dimension.
namespace Rte_sw
{
    // Scratch for solver_2stream, sized for one g-point. Owned by the caller and
    // built once, before the g-point loop: on a GPU, allocating it per call is a
    // cudaMalloc/cudaFree pair per g-point, and cudaFree synchronizes the device.
    struct Two_stream_scratch
    {
        Array_2d<TF> Rdif, Tdif;            // (nlay, ncol)
        Array_2d<TF> source_up, source_dn;  // (nlay, ncol)
        Array_2d<TF> albedo, src;           // (nlev, ncol)
        Array_2d<TF> denom;                 // (nlay, ncol)
        Array_1d<TF> src_sfc;               // (ncol)

        static Two_stream_scratch make(const int nlay, const int ncol);
    };

    // Direct beam only, no scattering. Reference: sw_solver_noscat.
    void solver_noscat(
            const bool top_at_1,
            const Array_map_2d<const TF>& tau,          // (nlay, ncol)
            const Array_map_2d<const TF>& mu0,          // (nlay, ncol)
            const Array_map_1d<const TF>& inc_flux_dir, // (ncol)
            const Array_map_2d<TF>& flux_dir);          // (nlev, ncol)

    // Two-stream with scattering. Reference: sw_solver_2stream.
    //
    // flux_dn is returned as the total downward flux, diffuse plus direct, matching
    // the reference. inc_flux_dif may be an empty View, which means a zero diffuse
    // boundary condition.
    void solver_2stream(
            const bool top_at_1,
            const Array_map_2d<const TF>& tau,          // (nlay, ncol)
            const Array_map_2d<const TF>& ssa,          // (nlay, ncol)
            const Array_map_2d<const TF>& g,            // (nlay, ncol)
            const Array_map_2d<const TF>& mu0,          // (nlay, ncol)
            const Array_map_1d<const TF>& sfc_alb_dir,  // (ncol)
            const Array_map_1d<const TF>& sfc_alb_dif,  // (ncol)
            const Array_map_1d<const TF>& inc_flux_dir, // (ncol)
            const Array_map_1d<const TF>& inc_flux_dif, // (ncol), may be empty
            const Array_map_2d<TF>& flux_up,            // (nlev, ncol)
            const Array_map_2d<TF>& flux_dn,            // (nlev, ncol)
            const Array_map_2d<TF>& flux_dir,           // (nlev, ncol)
            const Two_stream_scratch& scratch);

    void init_python_bindings(py::module_& m);
}
