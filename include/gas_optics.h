#pragma once

#include <string>
#include <vector>

#include "gas_concs.h"
#include "raytracer.h"
#include "raytracer_lw.h"
#include "rte_lw.h"
#include "rte_sw.h"
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

    // The binary-species interpolation, (nflav, 2, nlay, ncol) each. Empty unless the
    // caller asks for them: each g-point reads one flavour, so storing all of them
    // costs 24 bytes of traffic per cell per g-point against 240 bytes held, and at
    // ncol = 65536, nlay = 256 the three together are 4 GB. The kernels rebuild the
    // one flavour they need with Gas_optics_kernels::eta_interp instead, which is the
    // fmajor/fminor argument above one step further. Only the tests, which compare
    // them against the reference, ask for them.
    Array_4d<int> jeta;     // binary species interpolation index
    Array_4d<TF> feta;      // binary species interpolation fraction
    Array_4d<TF> col_mix;   // combined major species column amount

    // First and last layer, 1-based inclusive, on each side of the tropopause, per
    // column; a zero start means the column has no layers on that side. Derived from
    // tropo and the pressure profile, and read by the minor-absorber loop. It lives
    // here because it has no g-point dimension and would otherwise be rebuilt for
    // every g-point.
    Array_2d<int> lower_limits;  // (ncol, 2)
    Array_2d<int> upper_limits;  // (ncol, 2)

    // Allocate for the given problem size. store_eta allocates jeta, feta and col_mix,
    // which only the kernel-by-kernel tests need; see above.
    static Interp_state create(
            const int nflav, const int nlay, const int ncol, const bool store_eta = false);
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
    Array_1d<int> idx_minor_scaling;     // (nminor) second gas affecting absorption, -1 for none
    Array_1d<int> kminor_start;          // (nminor) 0-based start in kminor

    Array_1d<int> gpt_offset;            // (ngpt+1) CSR row offsets
    Array_1d<int> gpt_minor;             // (nnz)    absorber index per entry
    Array_1d<int> flavor;                // (nminor) 0-based flavour, from the range's first g-point

    // Build gpt_offset, gpt_minor and flavor. gpoint_flavor is (ngpt, 2), 0-based;
    // itropo selects its column (0 lower, 1 upper).
    void build_map(const Array_2d<const int>& gpoint_flavor, const int ngpt, const int itropo);
};


// The raw contents of a coefficient file, before reduction. Python fills this from
// the NetCDF; nothing here has been matched against the host's gas list yet.
//
// Index conventions follow the file, which means Fortran's: key_species, band2gpt,
// minor_limits_gpt and kminor_start are all 1-based. Gas_optics::load converts.
struct Kdist_file
{
    std::vector<std::string> gas_names;                                // (ngas_file)
    std::vector<std::string> gas_minor, identifier_minor;              // (n_minor_absorbers)
    std::vector<std::string> minor_gases_lower, minor_gases_upper;     // (nminor_*)
    std::vector<std::string> scaling_gas_lower, scaling_gas_upper;     // (nminor_*)

    Array_3d_h<int> key_species;      // (nbnd, 2, 2)
    Array_2d_h<int> band2gpt;         // (nbnd, 2)
    Array_1d_h<TF> press_ref;         // (npres)
    Array_1d_h<TF> temp_ref;          // (ntemp)
    TF press_ref_trop = TF(0.);
    TF temp_ref_p = TF(0.);
    TF temp_ref_t = TF(0.);
    Array_3d_h<TF> vmr_ref;           // (ntemp, ngas_file+1, 2)
    Array_4d_h<TF> kmajor;            // (ngpt, npres+1, neta, ntemp)

    Array_3d_h<TF> kminor_lower, kminor_upper;                 // (ncontrib, neta, ntemp)
    Array_2d_h<int> minor_limits_gpt_lower, minor_limits_gpt_upper;   // (nminor, 2)
    Array_1d_h<Bool> minor_scales_with_density_lower, minor_scales_with_density_upper;
    Array_1d_h<Bool> scale_by_complement_lower, scale_by_complement_upper;
    Array_1d_h<int> kminor_start_lower, kminor_start_upper;    // (nminor)

    // Longwave only.
    Array_2d_h<TF> totplnk;           // (nbnd, nplancktemp)
    Array_4d_h<TF> planck_frac;       // (ngpt, npres+1, neta, ntemp)

    // Shortwave only.
    Array_4d_h<TF> rayl;              // (2, ngpt, neta, ntemp), empty if absent
    Array_1d_h<TF> solar_source_quiet, solar_source_facular, solar_source_sunspot;  // (ngpt)
    TF mg_default = TF(0.);
    TF sb_default = TF(0.);
};


// The gas-optics k-distribution: everything read from the coefficient file that the
// optical depth kernels need.
//
// Never capture this whole struct in a KOKKOS_LAMBDA: gas_names is a host container.
// Capture the individual Views, or the Minor_absorbers, as the kernels do.
struct Kdist_gas
{
    // The gases this k-distribution was reduced to, in the order the gas dimension of
    // col_gas uses. Index 0 of that dimension is dry air, so gas_names[i] sits at
    // col_gas index i+1.
    std::vector<std::string> gas_names;

    // Interpolation grid and the scalars derived from it, so a caller never has to
    // recompute them. The reference keeps these as components of ty_gas_optics_rrtmgp.
    Array_1d<TF> press_ref_log;    // (npres) log of the reference pressures
    Array_1d<TF> temp_ref;         // (ntemp)
    TF press_ref_log_delta = TF(0.);
    TF temp_ref_min = TF(0.);
    TF temp_ref_max = TF(0.);
    TF temp_ref_delta = TF(0.);
    TF press_ref_trop_log = TF(0.);
    int neta = 0;

    Array_2d<int> flavor;          // (nflav, 2) the pair of major species, as col_gas indices
    Array_2d<int> gpoint_flavor;   // (ngpt, 2) 0-based flavour per g-point, lower and upper
    Array_3d<TF> vmr_ref;          // (ntemp, ngas+1, 2)
    Array_2d<int> band_lims_gpt;   // (nbnd, 2) first and last g-point, 0-based inclusive
    Array_4d<TF> kmajor;           // (ngpt, npres+1, neta, ntemp)
    Array_1d<int> gpt_band;        // (ngpt) band each g-point belongs to
    Array_1d<int> band_gpt_start;  // (nbnd) first g-point of each band

    // gpt_band on the host. The g-point loop needs the band index as a scalar, to pick
    // a band's cloud properties and to place the flux in a by-band total; reading it
    // back from the device every g-point would be a synchronisation per iteration.
    Array_1d_h<int> gpt_band_h;    // (ngpt)

    Minor_absorbers lower;
    Minor_absorbers upper;

    int idx_h2o = -1;              // index of water vapour in the gas dimension of col_gas

    // Shortwave only: Rayleigh scattering coefficients, lower and upper atmosphere,
    // and the spectral solar source at the top of the atmosphere.
    Array_4d<TF> krayl;            // (2, ngpt, neta, ntemp)
    Array_1d<TF> solar_source;     // (ngpt)

    // Longwave only: the Planck tables.
    Array_4d<TF> pfracin;          // (ngpt, npres+1, neta, ntemp)
    Array_2d<TF> totplnk;          // (nbnd, nplancktemp)
    TF totplnk_delta = TF(0.);
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

    // Absorption optical depth from major and minor gases, for one g-point.
    // Overwrites tau. The reference accumulates into it, but every caller here gives
    // the kernel a g-point slice of its own, so the accumulation only ever bought a
    // full-size zeroing per g-point. Reference: compute_tau_absorption.
    void compute_tau_absorption(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,     // (nlay, ncol)
            const Array_2d<const TF>& tlay,     // (nlay, ncol)
            const Array_3d<const TF>& col_gas,  // (ngas+1, nlay, ncol)
            const int igpt,
            const Array_map_2d<TF>& tau);       // (nlay, ncol), accumulated into

    // Absorption and Rayleigh scattering together, for one g-point: tau comes out as
    // the total extinction, ssa as the Rayleigh fraction of it and g as zero, which is
    // the reference's combine_abs_and_rayleigh. One kernel rather than three, because
    // Rayleigh takes the same flavour as the major species and so reuses the
    // interpolation weights already in registers.
    // Absorption optical depth and the Planck fraction for one g-point. The two are
    // the same major-species interpolation of two tables, so the longwave takes them
    // from one kernel; see compute_tau_sw for the shortwave's counterpart.
    void compute_tau_lw(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,     // (nlay, ncol)
            const Array_2d<const TF>& tlay,     // (nlay, ncol)
            const Array_3d<const TF>& col_gas,  // (ngas+1, nlay, ncol)
            const int igpt,
            const Array_map_2d<TF>& tau,        // (nlay, ncol)
            const Array_map_2d<TF>& pfrac);     // (nlay, ncol)

    void compute_tau_sw(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,     // (nlay, ncol)
            const Array_2d<const TF>& tlay,     // (nlay, ncol)
            const Array_3d<const TF>& col_gas,  // (ngas+1, nlay, ncol)
            const int igpt,
            const Array_map_2d<TF>& tau,        // (nlay, ncol)
            const Array_map_2d<TF>& ssa,        // (nlay, ncol)
            const Array_map_2d<TF>& g);         // (nlay, ncol)

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

    // Reduce a coefficient file to the gases the host actually supplies, and build
    // every derived index array. Reference: load / init_abs_coeffs, plus the helpers
    // create_flavor, create_gpoint_flavor, create_idx_minor, create_idx_minor_scaling,
    // create_key_species_reduce and reduce_minor_arrays.
    //
    // Everything in the result is 0-based, unlike the reference.
    Kdist_gas load(const Kdist_file& file, const Gas_concs& available_gases);

    // ---- the fused per-g-point solve --------------------------------------------
    //
    // solve_lw and solve_sw run gas optics, the cloud increment and transport for one
    // g-point at a time and accumulate the fluxes, so no array in the whole pipeline
    // carries a g-point dimension. The loop body is public as solve_lw_gpt /
    // solve_sw_gpt, for callers that want a single g-point -- the Monte Carlo ray
    // tracer, and the tests.

    // The atmosphere a solve sees. col_dry may be empty, in which case it is derived
    // from plev and the water vapour, as the reference does. tlev and tsfc are
    // longwave only.
    struct Atmosphere
    {
        Array_2d<const TF> play;     // (nlay, ncol)
        Array_2d<const TF> plev;     // (nlev, ncol)
        Array_2d<const TF> tlay;     // (nlay, ncol)
        Array_2d<const TF> tlev;     // (nlev, ncol)
        Array_1d<const TF> tsfc;     // (ncol)
        Array_2d<const TF> col_dry;  // (nlay, ncol), may be empty
    };

    // Cloud (or aerosol) optical properties, resolved by band. Band-resolved rather
    // than by g-point because that is how the lookup tables give them; the g-point
    // loop takes the slice for its own band. Empty tau means no clouds. ssa and g are
    // empty for an absorption-only set, as the longwave all-sky case uses.
    struct Band_props
    {
        Array_3d<const TF> tau;   // (nbnd, nlay, ncol)
        Array_3d<const TF> ssa;   // (nbnd, nlay, ncol), may be empty
        Array_3d<const TF> g;     // (nbnd, nlay, ncol), may be empty
    };

    // Everything a per-g-point solve needs that does not depend on the g-point, plus
    // the one g-point's working set that every iteration reuses. Built by prepare().
    struct Solve_state
    {
        Array_3d<TF> col_gas;   // (ngas+1, nlay, ncol)
        Interp_state interp;
        int sfc_lay = 0;        // 0-based layer adjacent to the surface

        // One g-point's optical properties and Planck sources, overwritten each
        // iteration. This is the whole point: (nlay, ncol), never (ngpt, nlay, ncol).
        Array_2d<TF> tau, ssa, g;
        Array_2d<TF> lay_source, lev_source, pfrac;
        Array_1d<TF> sfc_source, sfc_source_jac;

        // This g-point's fluxes, before they are accumulated.
        Array_2d<TF> flux_up, flux_dn, flux_dir;

        // The transport solvers' own working set, likewise reused every iteration.
        // Only the one the band in question uses is allocated.
        Rte_lw::Noscat_scratch lw_noscat;
        Rte_sw::Two_stream_scratch sw_2stream;

        Source_func_lw sources() const
        { return Source_func_lw{lay_source, lev_source, sfc_source, sfc_source_jac}; }
    };

    // Column gas amounts and the table interpolation, once for the whole spectrum,
    // plus every array the per-g-point loop reuses. do_lw allocates the Planck
    // sources and finds the surface layer; weights is the longwave quadrature, whose
    // host mirror the no-scattering solver needs, and is ignored otherwise.
    Solve_state prepare(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool do_lw,
            const bool do_jacobian = false,
            const Array_1d<const TF>& weights = Array_1d<const TF>());

    // Where the accumulated fluxes go. The broadband views are required; the by-band
    // ones may be empty, in which case no by-band reduction is done. dir is shortwave
    // only, and up_jac longwave only.
    struct Fluxes_out
    {
        Array_2d<TF> up;          // (nlev, ncol)
        Array_2d<TF> dn;          // (nlev, ncol)
        Array_2d<TF> dir;         // (nlev, ncol), shortwave only
        Array_2d<TF> up_jac;      // (nlev, ncol), longwave only, may be empty

        Array_3d<TF> up_byband;   // (nbnd, nlev, ncol), may be empty
        Array_3d<TF> dn_byband;
        Array_3d<TF> dir_byband;
    };

    // One g-point, end to end: absorption optical depth, Planck sources, the cloud
    // increment, and transport. The fluxes go wherever the sinks say: a caller after a
    // single g-point points them at state.flux_up / flux_dn, while solve_lw below
    // points them straight at the spectral totals, so that no per-g-point flux is ever
    // written and read back. flux_up_jac is accumulated into if it is not empty.
    //
    // secants is (nmus, ncol) and shared by every g-point, as the reference's
    // rte_lw fills it. sfc_emis and inc_flux are (ngpt, ncol); this takes their
    // g-point slices.
    void solve_lw_gpt(
            const Kdist_gas& k,
            const Solve_state& state,
            const Atmosphere& atm,
            const bool top_at_1,
            const int igpt,
            const Array_map_2d<const TF>& secants,   // (nmus, ncol)
            const Array_1d<const TF>& weights,       // (nmus)
            const Array_map_1d<const TF>& sfc_emis,  // (ncol)
            const Array_map_1d<const TF>& inc_flux,  // (ncol), may be empty
            const Band_props& clouds,
            const Flux_sink& flux_up,
            const Flux_sink& flux_dn,
            const Array_2d<TF>& flux_up_jac);        // (nlev, ncol), may be empty

    // As above for the shortwave. flux_dn and flux_dir need their g-point arrays: the
    // adding sweep and the direct beam both read back what they wrote a level before.
    void solve_sw_gpt(
            const Kdist_gas& k,
            const Solve_state& state,
            const Atmosphere& atm,
            const bool top_at_1,
            const int igpt,
            const Array_2d<const TF>& mu0,               // (nlay, ncol)
            const Array_map_1d<const TF>& sfc_alb_dir,   // (ncol)
            const Array_map_1d<const TF>& sfc_alb_dif,   // (ncol)
            const Array_map_1d<const TF>& inc_flux_dir,  // (ncol)
            const Array_map_1d<const TF>& inc_flux_dif,  // (ncol), may be empty
            const Band_props& clouds,
            const Flux_sink& flux_up,
            const Flux_sink& flux_dn,
            const Flux_sink& flux_dir);

    // The whole spectrum: prepare, then loop the above and accumulate.
    void solve_lw(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool top_at_1,
            const Array_2d<const TF>& secants,     // (nmus, ncol)
            const Array_1d<const TF>& weights,     // (nmus)
            const Array_2d<const TF>& sfc_emis,    // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux,    // (ngpt, ncol), may be empty
            const Band_props& clouds,
            const Fluxes_out& fluxes);

    void solve_sw(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool top_at_1,
            const Array_2d<const TF>& mu0,             // (nlay, ncol)
            const Array_2d<const TF>& sfc_alb_dir,     // (ngpt, ncol)
            const Array_2d<const TF>& sfc_alb_dif,     // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux_dir,    // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux_dif,    // (ngpt, ncol), may be empty
            const Band_props& clouds,
            const Fluxes_out& fluxes);

    // The whole shortwave spectrum through the Monte Carlo ray tracer instead of the
    // two-stream solver. Same gas optics and the same cloud properties; only the
    // transport differs, and with it the shape of what comes out -- three-dimensional
    // absorption and boundary fluxes rather than a profile per column.
    //
    // The columns are the ray tracer's horizontal grid, ncol = grid.nx*grid.ny with
    // the column index i + j*nx. The sun is one direction for the whole domain, so
    // mu0 and azi are scalars, and toa_src is the solar irradiance per g-point on the
    // host, since the tracer needs it as a number rather than an array.
    //
    // Cloud properties are handed to the tracer separately rather than incremented
    // into the gas ones: a scattering event has to know whether a cloud droplet or a
    // molecule did it, and the two have different phase functions.
    void solve_sw_rt(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool top_at_1,
            const Raytracer::Grid& grid,
            const int photons_per_pixel,
            const bool independent_column,
            const TF mu0,
            const TF azi,
            const Array_1d_h<const TF>& toa_src,      // (ngpt), on the host
            const Array_2d<const TF>& sfc_alb_dir,    // (ngpt, ncol)
            const Band_props& clouds,
            const Raytracer::Fluxes_rt& fluxes);

    // The whole longwave spectrum through the Monte Carlo ray tracer instead of the
    // no-scattering solver, as solve_sw_rt is for the shortwave.
    //
    // The columns are the ray tracer's horizontal grid, ncol = grid.nx*grid.ny with
    // the column index i + j*nx. Clouds are handed to the tracer separately rather
    // than incremented into the gas optical depth, so that a scattering event can tell
    // which of the two deflected the photon; clouds.ssa and clouds.g may be empty,
    // which is a cloud that only absorbs, and is what a longwave case without
    // scattering wants.
    //
    // Layers above the box are lumped into its top cell, emission included. The
    // reference instead runs the plane-parallel solver over the whole column and feeds
    // its downward flux at the top of the box in as a scalar; see the note in
    // raytracer_lw.h.
    void solve_lw_rt(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool top_at_1,
            const Raytracer_lw::Grid& grid,
            const int photons_per_pixel,
            const bool independent_column,
            const Array_2d<const TF>& sfc_emis,   // (ngpt, ncol)
            const Band_props& clouds,
            const Raytracer_lw::Fluxes_lw& fluxes);

    // Full longwave gas optics: interpolation, absorption optical depth and the
    // Planck sources. Reference: ty_gas_optics_rrtmgp%gas_optics for the longwave.
    // Kept for the kernel-by-kernel tests; the solve above is the way to run a case.
    void gas_optics_lw(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Array_2d<const TF>& play,     // (nlay, ncol)
            const Array_2d<const TF>& plev,     // (nlev, ncol)
            const Array_2d<const TF>& tlay,     // (nlay, ncol)
            const Array_2d<const TF>& tlev,     // (nlev, ncol)
            const Array_1d<const TF>& tsfc,     // (ncol)
            const Array_2d<const TF>& col_dry,  // (nlay, ncol), may be empty
            const Array_3d<TF>& tau,            // (ngpt, nlay, ncol)
            const Source_func_lw_spectral& sources);

    // Full shortwave gas optics. tau is the total extinction and ssa the fraction of
    // it that is Rayleigh scattering; g is left to the caller to zero, as the
    // reference's combine_abs_and_rayleigh does.
    void gas_optics_sw(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Array_2d<const TF>& play,
            const Array_2d<const TF>& plev,
            const Array_2d<const TF>& tlay,
            const Array_2d<const TF>& col_dry,
            const Array_3d<TF>& tau,
            const Array_3d<TF>& ssa);

    // Rayleigh scattering optical depth for one g-point. Assigned, not accumulated, as
    // in the reference. Reference: compute_tau_rayleigh.
    void compute_tau_rayleigh(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& col_dry,       // (nlay, ncol)
            const Array_3d<const TF>& col_gas,       // (ngas+1, nlay, ncol)
            const int igpt,
            const Array_map_2d<TF>& tau_rayleigh);   // (nlay, ncol)

    // Planck sources at layer centres, layer edges and the surface, plus the
    // surface-temperature Jacobian. Reference: compute_Planck_source. The longwave
    // solvers take one g-point at a time, so they consume Source_func_lw_spectral::gpt.
    // pfrac is this g-point's share of its band's Planck irradiance, which
    // compute_tau_lw produces along the way.
    void compute_planck_source(
            const Kdist_gas& k,
            const Array_2d<const TF>& tlay,      // (nlay, ncol)
            const Array_2d<const TF>& tlev,      // (nlev, ncol)
            const Array_1d<const TF>& tsfc,      // (ncol)
            const int sfc_lay,                   // 0-based layer adjacent to the surface
            const int igpt,
            const Source_func_lw& sources,
            const Array_map_2d<const TF>& pfrac);   // (nlay, ncol)

    // The Planck fraction on its own. Test support: the solve path takes it from
    // compute_tau_lw, which computes it from the interpolation it is already doing.
    void compute_pfrac(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_3d<const TF>& col_gas,   // (ngas+1, nlay, ncol)
            const int igpt,
            const Array_map_2d<TF>& pfrac);      // (nlay, ncol)

    // Materialise the weights the reference stores, from the compact form above.
    // Test support only: the solvers reconstruct them in place via
    // Gas_optics_kernels::interp_weights rather than reading them from memory.
    void expand_weights(
            const Interp_state& state,
            const Array_5d<TF>& fminor,   // (nflav, 2, 2, nlay, ncol)
            const Array_6d<TF>& fmajor);  // (nflav, 2, 2, 2, nlay, ncol)

    void init_python_bindings(py::module_& m);
    void init_load_python_bindings(py::module_& m);
    void init_frontend_python_bindings(py::module_& m);
}
