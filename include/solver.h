#pragma once

#include "cloud_optics.h"
#include "gas_concs.h"
#include "gas_optics.h"
#include "raytracer.h"
#include "raytracer_lw.h"
#include "rte_lw.h"
#include "rte_sw.h"
#include "source_functions.h"
#include "types.h"


// Running a case. solve_lw and solve_sw run gas optics, the cloud increment and
// transport one g-point at a time, accumulating the fluxes. So no array in the
// pipeline carries more of the spectrum than one g-point. The loop body is public as
// gas_optics_lw_gpt / gas_optics_sw_gpt and solve_lw_gpt / solve_sw_gpt, for callers
// that want a single g-point -- the Monte Carlo ray tracer, and the tests.
//
// This is the layer above both Gas_optics, which gives it the optical properties of a
// g-point, and Rte_lw / Rte_sw, whose transport kernels it drives: Solver::solve_lw
// calls Rte_lw::solver_noscat, not the other way round.
namespace Solver
{
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

    // Cloud optical properties of one band, which every g-point of that band shares:
    // the lookup tables are resolved by band, not by g-point. Empty tau means no
    // clouds. ssa and g are empty for an absorption-only set, as the longwave all-sky
    // case uses.
    struct Cloud_band
    {
        Array_map_2d<const TF> tau;   // (nlay, ncol)
        Array_map_2d<const TF> ssa;   // (nlay, ncol), may be empty
        Array_map_2d<const TF> g;     // (nlay, ncol), may be empty
    };

    // The clouds as a caller has them: water paths and particle sizes, with room for
    // the optical properties of one band. The solves compute each band's as they come
    // to it, with cloud_band, so the clouds never cost more than three (nlay, ncol)
    // arrays whatever the number of bands. A null optics means no clouds.
    struct Cloud_input
    {
        const Cloud_optics* optics = nullptr;
        Array_2d<const TF> clwp, ciwp, reliq, reice;   // (nlay, ncol)
        Array_3d<TF> tau, ssa, g;   // (1, nlay, ncol); ssa and g only for two_stream
        bool delta_scale = false;
    };

    // Without two_stream the clouds only absorb, which is what a no-scattering
    // longwave solve takes, and delta_scale is ignored.
    Cloud_input make_clouds(
            const Cloud_optics* optics,
            const Array_2d<const TF>& clwp,
            const Array_2d<const TF>& ciwp,
            const Array_2d<const TF>& reliq,
            const Array_2d<const TF>& reice,
            const bool two_stream,
            const bool delta_scale);

    // The cloud optical properties of band ibnd, delta-scaled when asked, computed
    // into c's one-band arrays: they hold until the next call. Empty when there are
    // no clouds.
    Cloud_band cloud_band(const Cloud_input& c, const int ibnd);

    // Everything a per-g-point solve needs that does not depend on the g-point, plus
    // the one g-point's working set that every iteration reuses. Built by prepare().
    struct Solve_state
    {
        Array_3d<TF> col_gas;   // (ngas+1, nlay, ncol)
        Interp_state interp;
        int sfc_lay = 0;        // 0-based layer adjacent to the surface

        // One g-point's optical properties, (nlay, ncol), overwritten each iteration.
        // ssa and g are empty for the longwave without scattering, pfrac for the
        // shortwave.
        Array_2d<TF> tau, ssa, g, pfrac;

        // One g-point's Planck sources, overwritten each iteration.
        Array_2d<TF> lay_source, lev_source;
        Array_1d<TF> sfc_source, sfc_source_jac;

        // This g-point's fluxes, before they are accumulated.
        Array_2d<TF> flux_up, flux_dn, flux_dir;

        // The transport solvers' own working set, likewise reused every iteration.
        // Only the one the band in question uses is allocated.
        Rte_lw::Noscat_scratch lw_noscat;
        Rte_lw::Two_stream_scratch lw_2stream;
        Rte_sw::Two_stream_scratch sw_2stream;

        Source_func_lw sources() const
        { return Source_func_lw{lay_source, lev_source, sfc_source, sfc_source_jac}; }
    };

    // Column gas amounts and the table interpolation, once for the whole spectrum,
    // plus every array the per-g-point loop reuses. do_lw allocates the Planck
    // sources and finds the surface layer; weights is the longwave quadrature, whose
    // host mirror the no-scattering solver needs, and is ignored otherwise.
    // lw_scattering allocates the two-stream working set instead of the quadrature's.
    // Without with_transport the per-g-point fluxes and the transport solvers' working
    // set are left out, for a caller that traces rays instead.
    Solve_state prepare(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool do_lw,
            const bool do_jacobian = false,
            const Array_1d<const TF>& weights = Array_1d<const TF>(),
            const bool lw_scattering = false,
            const bool with_transport = true);

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

    // Gas optics for g-point igpt into the state's arrays: absorption optical depth
    // and Planck fraction in the longwave, total extinction and Rayleigh fraction in
    // the shortwave. with_g also zeroes the shortwave's g, which only a cloud
    // increment needs.
    void gas_optics_lw_gpt(
            const Kdist_gas& k,
            const Solve_state& state,
            const Atmosphere& atm,
            const int igpt);

    void gas_optics_sw_gpt(
            const Kdist_gas& k,
            const Solve_state& state,
            const Atmosphere& atm,
            const int igpt,
            const bool with_g);

    // One g-point, from its gas optics to its fluxes: Planck sources, the cloud
    // increment, and transport. The g-point's gas optics must already be in the
    // state; the cloud increment goes into them in place, and clouds are those of
    // igpt's band, as cloud_band gives them. The fluxes go wherever the sinks say: a
    // caller after a single g-point points them at state.flux_up / flux_dn, while
    // solve_lw below points them straight at the spectral totals, so that no
    // per-g-point flux is ever written and read back. flux_up_jac is accumulated into
    // if it is not empty.
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
            const Cloud_band& clouds,
            const bool scattering,
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
            const Cloud_band& clouds,
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
            const Cloud_input& clouds,
            const bool scattering,                 // solve with scattering, not by quadrature
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
            const Cloud_input& clouds,
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
            const Cloud_input& clouds,
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
    // min_mfp_grid_ratio makes the tracer skip the g-points it cannot learn anything
    // from. Where the gas is opaque on the scale of a grid cell -- the shortest gas
    // mean free path in the box below min_mfp_grid_ratio times the horizontal grid
    // spacing -- a photon is absorbed before it can cross a cell, so there is no
    // horizontal transport to resolve and the plane-parallel solve is the same answer,
    // reached with no photons and no Monte Carlo noise. Returns how many g-points were
    // actually traced. Zero traces everything; the reference's default is 1.
    //
    // The fallback solve follows the caller's own scattering switch, so it is the same
    // solver a plane-parallel run of the case would have used.
    //
    // lump_above chooses what to do with an atmosphere deeper than the box. Lumping
    // it into the box's top cell, which is the default and what rte-rrtmgp-cpp's input
    // files are shaped for, keeps every watt in the domain but gives the cell one
    // temperature and the box's own depth, so what it sends down is not what the air
    // it stands in for would send. Without it the box is the whole of what the tracer
    // sees -- the grid must then hold only the resolved cells -- and the air above
    // enters as the downward flux a plane-parallel solve of the full column leaves at
    // the box's top, which is how the reference does it.
    int solve_lw_rt(
            const Kdist_gas& k,
            const Gas_concs& gas_concs,
            const Atmosphere& atm,
            const bool top_at_1,
            const Raytracer_lw::Grid& grid,
            const int photons_per_pixel,
            const bool independent_column,
            const Array_2d<const TF>& sfc_emis,   // (ngpt, ncol)
            const Array_2d<const TF>& secants,    // (nmus, ncol) for the fallback
            const Array_1d<const TF>& weights,    // (nmus)
            const TF min_mfp_grid_ratio,
            const Cloud_input& clouds,
            const bool scattering,
            const bool lump_above,
            const Raytracer_lw::Fluxes_lw& fluxes);

    void init_python_bindings(py::module_& m);
}
