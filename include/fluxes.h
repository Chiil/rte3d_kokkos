#pragma once

#include "types.h"


// Where one g-point's flux, as a solver produces it, is to go.
//
// A single-g-point solve keeps the flux: gpt is the (nlev, ncol) array the caller
// reads afterwards. A whole-spectrum solve never looks at it -- it only wants the
// totals -- so it hands the solver those instead and the solver's last kernel adds
// into them, which saves writing a per-g-point flux and reading it straight back once
// per g-point. Both at once is legal, and is what the shortwave needs: its adding
// sweep reads the downward flux it wrote at the level before.
struct Flux_sink
{
    Array_map_2d<TF> gpt;       // (nlev, ncol), written when not empty
    Array_map_2d<TF> broadband; // (nlev, ncol), added into when not empty
    Array_map_2d<TF> byband;    // (nlev, ncol), this g-point's band, added into when not empty

    // One g-point's flux at (ilev, icol). accumulate adds to the g-point array rather
    // than overwriting it, which the longwave's quadrature angles need; the totals are
    // always added to, since the caller zeroed them before the g-point loop.
    KOKKOS_INLINE_FUNCTION
    void put(const int ilev, const int icol, const TF value, const bool accumulate = false) const
    {
        if (gpt.size() > 0)
            gpt(ilev, icol) = accumulate ? gpt(ilev, icol) + value : value;

        add(ilev, icol, value);
    }

    // The totals alone, for a value the g-point array already holds.
    KOKKOS_INLINE_FUNCTION
    void add(const int ilev, const int icol, const TF value) const
    {
        if (broadband.size() > 0)
            broadband(ilev, icol) += value;

        if (byband.size() > 0)
            byband(ilev, icol) += value;
    }
};


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
