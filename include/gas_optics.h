#pragma once

#include "gas_concs.h"
#include "source_functions.h"
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


// One of the two sets of minor absorbers, lower or upper atmosphere.
//
// The reference loops minor absorbers serially and, for each, walks a per-column layer
// range. We parallelise over (g-point, layer, column) instead, which needs the inverse
// mapping: for each g-point, which minor absorbers contribute. gpt_offset and gpt_minor
// hold that as a CSR list, built once by Minor_absorbers::build_map.
struct Minor_absorbers
{
    Array_3d<TF> kminor;                 // (nminork, neta, ntemp)
    Array_2d<int> minor_limits_gpt;      // (nminor, 2) first and last g-point, 0-based inclusive
    Array_1d<Bool> scales_with_density;  // (nminor)
    Array_1d<Bool> scale_by_complement;  // (nminor)
    Array_1d<int> idx_minor;             // (nminor) index into the gas dimension of col_gas
    Array_1d<int> idx_minor_scaling;     // (nminor) second gas affecting absorption, 0 for none
    Array_1d<int> kminor_start;          // (nminor) 0-based start in kminor

    Array_1d<int> gpt_offset;            // (ngpt+1) CSR row offsets
    Array_1d<int> gpt_minor;             // (nnz)    absorber index per entry
    Array_1d<int> flavor;                // (nminor) 0-based flavour, from the range's first g-point

    // Build gpt_offset, gpt_minor and flavor. gpoint_flavor is (ngpt, 2), 0-based;
    // itropo selects its column (0 lower, 1 upper).
    void build_map(const Array_2d<const int>& gpoint_flavor, const int ngpt, const int itropo);
};


// The gas-optics k-distribution: everything read from the coefficient file that the
// optical depth kernels need.
struct Kdist_gas
{
    Array_2d<int> gpoint_flavor;   // (ngpt, 2) 0-based flavour per g-point, lower and upper
    Array_2d<int> band_lims_gpt;   // (nbnd, 2) first and last g-point, 0-based inclusive
    Array_4d<TF> kmajor;           // (ngpt, npres+1, neta, ntemp)
    Array_1d<int> gpt_band;        // (ngpt) band each g-point belongs to
    Array_1d<int> band_gpt_start;  // (nbnd) first g-point of each band

    Minor_absorbers lower;
    Minor_absorbers upper;

    int idx_h2o = -1;              // index of water vapour in the gas dimension of col_gas

    // Shortwave only: Rayleigh scattering coefficients, lower and upper atmosphere.
    Array_4d<TF> krayl;            // (2, ngpt, neta, ntemp)

    // Longwave only: the Planck tables.
    Array_4d<TF> pfracin;          // (ngpt, npres+1, neta, ntemp)
    Array_2d<TF> totplnk;          // (nbnd, nplancktemp)
    TF totplnk_delta = TF(0.);
    TF temp_ref_min = TF(0.);
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

    // Absorption optical depth from major and minor gases. Accumulates into tau, as
    // the reference does. Reference: compute_tau_absorption.
    void compute_tau_absorption(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,     // (nlay, ncol)
            const Array_2d<const TF>& tlay,     // (nlay, ncol)
            const Array_3d<const TF>& col_gas,  // (ngas+1, nlay, ncol)
            const Array_3d<TF>& tau);           // (ngpt, nlay, ncol), accumulated into

    // Dry air column amount [molecules/cm2] from the water vapour mixing ratio and
    // the level pressures. Reference: get_col_dry.
    //
    // latitude may be empty, in which case a constant gravity is used, as the
    // reference does when the argument is absent.
    void compute_col_dry(
            const Array_2d<const TF>& vmr_h2o,   // (nlay, ncol)
            const Array_2d<const TF>& plev,      // (nlev, ncol)
            const Array_1d<const TF>& latitude,  // (ncol), may be empty
            const Array_2d<TF>& col_dry);        // (nlay, ncol)

    // Column gas amounts for every gas the k-distribution knows about, in the order
    // given by gas_names. Index 0 of the gas dimension is dry air, as in the
    // reference's 0:ngas bounds.
    //
    // col_dry may be empty, in which case it is computed from plev and the water
    // vapour concentration.
    void compute_col_gas(
            const Gas_concs& gas_concs,
            const std::vector<std::string>& gas_names,
            const Array_2d<const TF>& plev,      // (nlev, ncol)
            const Array_2d<const TF>& col_dry,   // (nlay, ncol), may be empty
            const Array_1d<const TF>& latitude,  // (ncol), may be empty
            const Array_3d<TF>& col_gas);        // (ngas+1, nlay, ncol)

    // Rayleigh scattering optical depth. Assigned, not accumulated, as in the
    // reference. Reference: compute_tau_rayleigh.
    void compute_tau_rayleigh(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& col_dry,   // (nlay, ncol)
            const Array_3d<const TF>& col_gas,   // (ngas+1, nlay, ncol)
            const Array_3d<TF>& tau_rayleigh);   // (ngpt, nlay, ncol)

    // Planck sources at layer centres, layer edges and the surface, plus the
    // surface-temperature Jacobian. Fills the same Source_func_lw the longwave
    // solvers consume. Reference: compute_Planck_source.
    void compute_planck_source(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& tlay,      // (nlay, ncol)
            const Array_2d<const TF>& tlev,      // (nlev, ncol)
            const Array_1d<const TF>& tsfc,      // (ncol)
            const int sfc_lay,                   // 0-based layer adjacent to the surface
            const Source_func_lw& sources);

    // Materialise the weights the reference stores, from the compact form above.
    // Test support only: the solvers reconstruct them in place via
    // Gas_optics_kernels::interp_weights rather than reading them from memory.
    void expand_weights(
            const Interp_state& state,
            const Array_5d<TF>& fminor,   // (nflav, 2, 2, nlay, ncol)
            const Array_6d<TF>& fmajor);  // (nflav, 2, 2, 2, nlay, ncol)

    void init_python_bindings(py::module_& m);
}
