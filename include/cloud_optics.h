#pragma once

#include "optical_props.h"
#include "types.h"


// Cloud optical properties from lookup tables, following ty_cloud_optics_rrtmgp.
//
// The tables are indexed by particle size, linearly interpolated between nsteps evenly
// spaced entries spanning [size_lwr, size_upr]. Liquid and ice are computed separately
// and combined, which is what the reference does to avoid a second division.
//
// Only the lookup-table path is implemented. The reference also offers Pade
// approximants, but no shipped coefficient file contains them.
//
// The spectral dimension is whatever the coefficient file provides: the -bnd files are
// resolved by band, the -g### files by g-point. A band-resolved result is combined with
// gas optics through the by-band increments in Optical_props.
struct Cloud_optics
{
    // Liquid tables, (nspec, nsize_liq).
    Array_2d<TF> lut_extliq, lut_ssaliq, lut_asyliq;

    // Ice tables for the selected roughness, (nspec, nsize_ice).
    Array_2d<TF> lut_extice, lut_ssaice, lut_asyice;

    TF radliq_lwr = TF(0.);
    TF radliq_upr = TF(0.);
    TF radice_lwr = TF(0.);
    TF radice_upr = TF(0.);
    TF liq_step_size = TF(0.);
    TF ice_step_size = TF(0.);

    // Build from the tables as read from a coefficient file. ext/ssa/asy for ice carry
    // a leading roughness dimension; icergh selects one, 0-based.
    //
    // Note the size bounds are the table's own interpolation range. Recent coefficient
    // files name the ice bounds diamice_lwr/upr where older ones said radice_lwr/upr;
    // the numbers are unchanged, but RRTMGP now documents the quantity as an effective
    // diameter, so a host model passing an effective radius would be wrong by a factor
    // of two.
    static Cloud_optics load_lut(
            const TF radliq_lwr, const TF radliq_upr,
            const TF radice_lwr, const TF radice_upr,
            const Array_2d_h<TF>& extliq,   // (nspec, nsize_liq)
            const Array_2d_h<TF>& ssaliq,
            const Array_2d_h<TF>& asyliq,
            const Array_3d_h<TF>& extice,   // (nrghice, nspec, nsize_ice)
            const Array_3d_h<TF>& ssaice,
            const Array_3d_h<TF>& asyice,
            const int icergh);

    int nroughness_types = 0;
};


namespace Clouds
{
    // Cloud optical properties for a two-stream calculation. Water paths in g/m2,
    // particle sizes in microns. A layer is cloudy where its water path is positive.
    void compute(
            const Cloud_optics& c,
            const Array_2d<const TF>& clwp,   // (nlay, ncol) liquid water path
            const Array_2d<const TF>& ciwp,   // (nlay, ncol) ice water path
            const Array_2d<const TF>& reliq,  // (nlay, ncol) liquid particle size
            const Array_2d<const TF>& reice,  // (nlay, ncol) ice particle size
            const Array_3d<TF>& tau,          // (nspec, nlay, ncol)
            const Array_3d<TF>& ssa,
            const Array_3d<TF>& g);

    // Absorption optical depth only, for a longwave calculation without scattering:
    // (1 - ssa) * tau, summed over liquid and ice.
    void compute(
            const Cloud_optics& c,
            const Array_2d<const TF>& clwp,
            const Array_2d<const TF>& ciwp,
            const Array_2d<const TF>& reliq,
            const Array_2d<const TF>& reice,
            const Array_3d<TF>& tau);

    void init_python_bindings(py::module_& m);
}
