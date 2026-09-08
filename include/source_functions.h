#pragma once

#include "types.h"


// One g-point's Planck sources: what the longwave solvers consume, the useful half of
// the reference's ty_source_func_lw. A plain aggregate: construct it with designated
// initialisers or fill the members directly.
//
// The views are unmanaged, so a slice of a spectrally resolved set can be passed
// straight in; see Source_func_lw_spectral::gpt below.
struct Source_func_lw
{
    Array_map_2d<const TF> lay_source;      // (nlay, ncol) Planck source at layer average temperature
    Array_map_2d<const TF> lev_source;      // (nlev, ncol) Planck source at layer edges
    Array_map_1d<const TF> sfc_source;      // (ncol)       surface source function
    Array_map_1d<const TF> sfc_source_jac;  // (ncol)       d(surface source)/d(surface temperature)
};


// The same for every g-point, as the Planck kernel still produces them. Transitional:
// once compute_planck_source runs per g-point this goes away and nothing carries a
// g-point dimension.
struct Source_func_lw_spectral
{
    Array_3d<TF> lay_source;      // (ngpt, nlay, ncol)
    Array_3d<TF> lev_source;      // (ngpt, nlev, ncol)
    Array_2d<TF> sfc_source;      // (ngpt, ncol)
    Array_2d<TF> sfc_source_jac;  // (ngpt, ncol), may be empty

    Source_func_lw gpt(const int igpt) const
    {
        return Source_func_lw{
                slice_2d(lay_source, igpt),
                slice_2d(lev_source, igpt),
                slice_1d(sfc_source, igpt),
                sfc_source_jac.size() > 0 ? slice_1d(sfc_source_jac, igpt)
                                          : Array_map_1d<TF>()};
    }
};
