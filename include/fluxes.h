#pragma once

#include "types.h"


// Spectral reduction of fluxes. Arrays are (ngpt, nlev, ncol) going in and
// (nlev, ncol) or (nbnd, nlev, ncol) coming out.
namespace Fluxes
{
    // Sum a spectrally-resolved flux over all g-points.
    void sum_broadband(
            const Array_3d<const TF>& spectral_flux,  // (ngpt, nlev, ncol)
            const Array_2d<TF>& broadband_flux);      // (nlev, ncol)

    // Net (down minus up) summed over all g-points, from spectrally-resolved fluxes.
    void net_broadband(
            const Array_3d<const TF>& spectral_flux_dn,
            const Array_3d<const TF>& spectral_flux_up,
            const Array_2d<TF>& broadband_flux_net);

    // Net from fluxes that have already been spectrally integrated.
    void net_broadband(
            const Array_2d<const TF>& flux_dn,        // (nlev, ncol)
            const Array_2d<const TF>& flux_up,
            const Array_2d<TF>& broadband_flux_net);

    // Sum within each band. band_lims is (nbnd, 2), holding the first and last
    // g-point of each band; unlike the reference's (2, nbnd) gpt_lims these are
    // 0-based and the end is inclusive, matching the reference's convention otherwise.
    void sum_byband(
            const Array_2d<const int>& band_lims,     // (nbnd, 2)
            const Array_3d<const TF>& spectral_flux,  // (ngpt, nlev, ncol)
            const Array_3d<TF>& byband_flux);         // (nbnd, nlev, ncol)

    void net_byband(
            const Array_2d<const int>& band_lims,
            const Array_3d<const TF>& spectral_flux_dn,
            const Array_3d<const TF>& spectral_flux_up,
            const Array_3d<TF>& byband_flux_net);

    // Add one g-point's flux to a running total. This is what the per-g-point solve
    // uses, where the reductions above would need the whole spectrum in memory at
    // once. The caller zeroes the totals before the g-point loop.
    void accumulate_broadband(
            const Array_map_2d<const TF>& gpt_flux,   // (nlev, ncol)
            const Array_2d<TF>& broadband_flux);      // (nlev, ncol)

    // ibnd is the band this g-point belongs to, from Kdist_gas::gpt_band_h.
    void accumulate_byband(
            const int ibnd,
            const Array_map_2d<const TF>& gpt_flux,   // (nlev, ncol)
            const Array_3d<TF>& byband_flux);         // (nbnd, nlev, ncol)

    void init_python_bindings(py::module_& m);
}
