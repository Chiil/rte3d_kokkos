#pragma once

#include "types.h"


// Results of the k-distribution interpolation, consumed by the optical depth and
// Planck source kernels.
//
// The reference materialises the interpolation weights as fmajor(2,2,2,ncol,nlay,nflav)
// and fminor(2,2,ncol,nlay,nflav). We do not: both are pure functions of ftemp, fpress
// and feta, which this struct stores instead, and Gas_optics_kernels::interp_weights
// reconstructs them where they are used. That is twelve stored values per
// (flavour, layer, column) traded for four multiplies -- at ncol = 1e5, nlay = 60 and
// nflav = 10 it is the difference between roughly 7 GB and 2.4 GB.
//
// Column is the fastest-varying dimension everywhere, unlike the reference, where ncol
// sits in the middle of fmajor and friends.
struct Interp_state
{
    Array_2d<int> jtemp;    // (nlay, ncol)          temperature interpolation index
    Array_2d<TF> ftemp;     // (nlay, ncol)          temperature interpolation fraction
    Array_2d<int> jpress;   // (nlay, ncol)          pressure interpolation index
    Array_2d<TF> fpress;    // (nlay, ncol)          pressure interpolation fraction
    Array_2d<Bool> tropo;   // (nlay, ncol)          lower (1) or upper (0) atmosphere

    Array_4d<int> jeta;     // (nflav, 2, nlay, ncol) binary species interpolation index
    Array_4d<TF> feta;      // (nflav, 2, nlay, ncol) binary species interpolation fraction
    Array_4d<TF> col_mix;   // (nflav, 2, nlay, ncol) combined major species column amount

    // Allocate for the given problem size.
    static Interp_state create(const int nflav, const int nlay, const int ncol);
};


namespace Gas_optics
{
    // Locate each (layer, column) in the k-distribution's temperature, pressure and
    // binary-species grids. Reference: interpolation in
    // rrtmgp-kernels/mo_gas_optics_rrtmgp_kernels.F90.
    //
    // Indices in flavor and the gas dimension of col_gas and vmr_ref are 0-based, with
    // 0 meaning dry air, matching the reference's 0:ngas bounds. The returned jtemp,
    // jpress and jeta are 0-based, unlike the reference's.
    void interpolation(
            const Array_2d<const int>& flavor,      // (nflav, 2)
            const Array_1d<const TF>& press_ref_log,// (npres)
            const Array_1d<const TF>& temp_ref,     // (ntemp)
            const TF press_ref_log_delta,
            const TF temp_ref_min,
            const TF temp_ref_delta,
            const TF press_ref_trop_log,
            const int neta,                         // size of the binary species grid
            const Array_3d<const TF>& vmr_ref,      // (ntemp, ngas+1, 2)
            const Array_2d<const TF>& play,         // (nlay, ncol)
            const Array_2d<const TF>& tlay,         // (nlay, ncol)
            const Array_3d<const TF>& col_gas,      // (ngas+1, nlay, ncol)
            const Interp_state& state);

    // Materialise the weights the reference stores, from the compact form above.
    // Test support only: the solvers reconstruct them in place via
    // Gas_optics_kernels::interp_weights rather than reading them from memory.
    void expand_weights(
            const Interp_state& state,
            const Array_5d<TF>& fminor,   // (nflav, 2, 2, nlay, ncol)
            const Array_6d<TF>& fmajor);  // (nflav, 2, 2, 2, nlay, ncol)

    void init_python_bindings(py::module_& m);
}
