#pragma once

#include "types.h"


// Optical property containers and the operations on them. Plain structs holding Views,
// free functions to combine them.
//
// Arrays are (ngpt, nlay, ncol); phase function moments add a trailing moment
// dimension, (ngpt, nlay, ncol, nmom), the reverse of the reference's
// (nmom, ncol, nlay, ngpt).
struct Optical_props_1scl
{
    Array_3d<TF> tau;
};

struct Optical_props_2str
{
    Array_3d<TF> tau;
    Array_3d<TF> ssa;
    Array_3d<TF> g;
};

struct Optical_props_nstr
{
    Array_3d<TF> tau;
    Array_3d<TF> ssa;
    Array_4d<TF> p;
};


namespace Optical_props
{
    // Maps each g-point to an index in the set being added. The reference has two
    // copies of every increment routine -- one where both sets are on g-points, one
    // where the second is on bands -- differing only in that index. Passing the map
    // explicitly collapses the eighteen routines to nine.
    //
    // identity_map gives the same-resolution case; band_map expands the reference's
    // (2, nbnd) gpt_lims into a per-g-point band index.
    Array_1d<int> identity_map(const int ngpt);
    Array_1d<int> band_map(const Array_2d<const int>& gpt_lims, const int ngpt);

    // Delta-scaling. The f variant takes a user-supplied forward-scattering fraction;
    // the other assumes f = g*g.
    void delta_scale_2str(const Optical_props_2str& props);
    void delta_scale_2str_f(const Optical_props_2str& props, const Array_3d<const TF>& f);

    // Increment the first set of optical properties by the second. gpt2set maps a
    // g-point of the first set onto an index of the second.
    void increment_1scalar_by_1scalar(
            const Array_3d<TF>& tau1,
            const Array_3d<const TF>& tau2,
            const Array_1d<const int>& gpt2set);

    // Also covers 1scalar_by_nstream: the reference's two bodies are identical, since
    // only the scattering fraction of the second set matters.
    void increment_1scalar_by_2stream(
            const Array_3d<TF>& tau1,
            const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2,
            const Array_1d<const int>& gpt2set);

    // Also covers nstream_by_1scalar: adding a purely absorbing set leaves the phase
    // function untouched, so the reference's two bodies are again identical.
    void increment_2stream_by_1scalar(
            const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1,
            const Array_3d<const TF>& tau2,
            const Array_1d<const int>& gpt2set);

    void increment_2stream_by_2stream(
            const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_3d<TF>& g1,
            const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_3d<const TF>& g2,
            const Array_1d<const int>& gpt2set);

    void increment_2stream_by_nstream(
            const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_3d<TF>& g1,
            const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_4d<const TF>& p2,
            const Array_1d<const int>& gpt2set);

    void increment_nstream_by_2stream(
            const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_4d<TF>& p1,
            const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_3d<const TF>& g2,
            const Array_1d<const int>& gpt2set);

    void increment_nstream_by_nstream(
            const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_4d<TF>& p1,
            const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_4d<const TF>& p2,
            const Array_1d<const int>& gpt2set);

    // Column subsetting. col_start is 0-based and col_end exclusive, unlike the
    // reference's inclusive 1-based colS/colE.
    void extract_subset(
            const Array_3d<const TF>& array_in, const int col_start, const int col_end,
            const Array_3d<TF>& array_out);

    void extract_subset(
            const Array_4d<const TF>& array_in, const int col_start, const int col_end,
            const Array_4d<TF>& array_out);

    // Absorption optical thickness, tau*(1 - ssa), for a subset of columns.
    void extract_subset_absorption_tau(
            const Array_3d<const TF>& tau_in, const Array_3d<const TF>& ssa_in,
            const int col_start, const int col_end,
            const Array_3d<TF>& tau_out);

    void init_python_bindings(py::module_& m);
}
