#pragma once

#include "source_functions.h"
#include "types.h"


// Longwave RTE solvers. Free functions on plain Views, as in rte_sw.h.
//
// Every solver works on a single g-point: nothing here carries a g-point dimension,
// so the working set is (nlay, ncol) whatever the spectral resolution. The caller
// loops g-points and accumulates.
//
// The scratch a solver needs is owned by the caller and built once, before the
// g-point loop. Allocating it per call is what a Kokkos View constructor would do,
// and on a GPU that is a cudaMalloc/cudaFree pair per g-point -- with cudaFree
// synchronizing the device, so it serializes the launches as well.
namespace Rte_lw
{
    // Scratch for solver_noscat, sized for one g-point.
    struct Noscat_scratch
    {
        Array_2d<TF> trans;      // (nlay, ncol)
        Array_2d<TF> source_up;  // (nlay, ncol)
        Array_2d<TF> source_dn;  // (nlay, ncol)
        Array_2d<TF> rad_up;     // (nlev, ncol)
        Array_2d<TF> rad_dn;     // (nlev, ncol)
        Array_2d<TF> rad_up_jac; // (nlev, ncol), empty unless Jacobians are wanted

        // The quadrature weights on the host: they scale each angle's contribution
        // from host code, so keeping a mirror here saves a blocking device-to-host
        // copy per g-point.
        Kokkos::View<TF*, Kokkos::LayoutRight, Kokkos::HostSpace> weights_h;

        static Noscat_scratch make(
                const int nlay, const int ncol,
                const Array_1d<const TF>& weights,
                const bool do_jacobians);
    };

    // Scratch for solver_2stream, sized for one g-point.
    struct Two_stream_scratch
    {
        Array_2d<TF> Rdif, Tdif;            // (nlay, ncol)
        Array_2d<TF> source_up, source_dn;  // (nlay, ncol)
        Array_2d<TF> albedo, src;           // (nlev, ncol)
        Array_2d<TF> denom;                 // (nlay, ncol)
        Array_1d<TF> albedo_sfc, src_sfc;   // (ncol)

        static Two_stream_scratch make(const int nlay, const int ncol);
    };

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
            const Array_2d<TF>& flux_up_jac,         // (nlev, ncol), may be empty
            const Noscat_scratch& scratch);

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
            const Array_map_2d<TF>& flux_dn,         // (nlev, ncol)
            const Two_stream_scratch& scratch);

    void init_python_bindings(py::module_& m);
}
