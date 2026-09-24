#pragma once

#include "raytracer.h"
#include "types.h"


// The longwave Monte Carlo ray tracer: forward, three-dimensional, null-collision.
//
// The Kokkos counterpart of raytracer_lw.cu in rte-rrtmgp-cpp. It walks photons
// through the same box as the shortwave tracer of raytracer.h, over the same
// null-collision grid, with the same phase functions -- everything in
// raytracer_common.h is shared. Two things are its own.
//
// The first is where photons come from. Shortwave photons all enter at the top of the
// domain; longwave photons are emitted by every cell of the atmosphere, by the surface
// and by the top boundary, in proportion to the local Planck source and the local
// absorption. So a photon launch has to draw an emitter out of (nz+2)*ncol of them,
// which is what the cumulative distribution in Scratch is for.
//
// The second is what they score. A longwave photon deposits absorbed weight as it
// walks, and when it terminates its *total* deposit is taken back off the cell that
// emitted it. flux_net is therefore absorption minus emission -- the flux divergence,
// which is what a longwave heating rate is built from -- and a photon reabsorbed in
// its own cell contributes exactly zero rather than a large positive and a large
// negative that only cancel in the mean.
//
// Aerosols and the Mie phase function are left out, as in the shortwave tracer.
namespace Raytracer_lw
{
    using Raytracer::Count;
    using Raytracer::Grid;
    using Raytracer::Optics_cell;
    using Raytracer::Vector;


    // Where the fluxes go, accumulated over g-points by the caller. flux_net is per
    // unit height [W/m3] and signed: positive where the cell absorbs more than it
    // emits.
    // Two levels are reported above the surface, and with layers lumped they are not
    // the same level. The box's top cell stands in for the whole atmosphere above, so
    // the top of the box is the top of the *atmosphere* -- toa_dn is zero there, as it
    // should be, and toa_up is the outgoing longwave. What a host model coupled to the
    // resolved field wants is the pair at the bottom of that cell, where the resolved
    // domain actually ends: tod_dn is the flux the air above sends down into it. With
    // nothing lumped the two levels coincide and both pairs carry the same numbers.
    struct Fluxes_lw
    {
        Array_1d<TF> toa_dn;    // (ncol) downward at the top of the atmosphere
        Array_1d<TF> toa_up;    // (ncol) upward at the top of the atmosphere
        Array_1d<TF> tod_dn;    // (ncol) downward at the top of the resolved domain
        Array_1d<TF> tod_up;    // (ncol) upward at the top of the resolved domain
        Array_1d<TF> sfc_dn;    // (ncol) downward at the surface
        Array_1d<TF> sfc_up;    // (ncol) upward at the surface
        Array_2d<TF> flux_net;  // (nz, ncol) absorbed minus emitted

        static Fluxes_lw make(const Grid& grid);

        void zero() const;
    };


    // Everything a trace reuses from one g-point to the next. Built once before the
    // g-point loop, as Raytracer::Scratch is, because freeing a device allocation per
    // g-point synchronizes the device.
    struct Scratch
    {
        Array_2d<Optics_cell> optics;   // (nz, ncol)
        Array_3d<TF> k_null;            // (kn_z, kn_y, kn_x) maximum extinction per block

        // The emitted power of every source, and its running total. Row 0 of the
        // emission is the surface, rows 1 to nz the cells of the box from the bottom
        // up, and row nz+1 the top boundary -- the same slot order as the reference's
        // alias table. The cumulative distribution is flat over those slots, and a
        // photon launch finds its emitter by searching it.
        //
        // In double whatever TF is: a float running total over millions of cells stops
        // resolving the small emitters once it passes 2^24 times their weight, and
        // they would become unsamplable. Its last element is the total emitted power,
        // so no separate reduction is needed and nothing has to reach the host.
        Array_2d<TF> emission;          // (nz+2, ncol)
        Array_1d<double> cdf;           // ((nz+2)*ncol)

        // Photon counts, zeroed at the start of every trace.
        Array_1d<Count> toa_dn, toa_up, tod_dn, tod_up, sfc_dn, sfc_up;   // (ncol)
        Array_2d<Count> atmos;                                            // (nz, ncol)

        static Scratch make(const Grid& grid);
    };


    // Trace one g-point.
    //
    // tau_gas and ssa_gas are the gas optical depth and its single-scattering albedo,
    // the second empty, or zero, unless the caller has folded something scattering
    // into the gas. The cloud triple may be empty, which means a clear sky.
    // lay_source is the Planck source at the layer average temperature, and
    // sfc_source its surface counterpart, both as Source_func_lw holds them:
    // radiances, so the emission carries the pi that Rte_lw::solver_2stream also
    // applies.
    //
    // All of the (nlay, ncol) arrays are read in the caller's own vertical
    // orientation, given by top_at_1, and nlay may exceed nz: the layers from nz-1
    // upward are lumped into the top cell of the box, their emission included.
    //
    // inc_dif is a diffuse irradiance entering the top of the box. It is meant for a
    // grid that resolves the whole atmosphere, where the reference feeds in what a
    // plane-parallel solve above the box gives; with layers lumped into the top cell
    // that cell already emits downward, and passing both would count them twice.
    //
    // photons_per_pixel is per g-point: the trace shoots photons_per_pixel*nx*ny of
    // them, spread over the emitters rather than over the pixels. independent_column
    // switches off horizontal transport.
    void trace_rays(
            const Grid& grid,
            const bool top_at_1,
            const bool independent_column,
            const int photons_per_pixel,
            const int igpt,
            const Array_map_2d<const TF>& tau_gas,     // (nlay, ncol)
            const Array_map_2d<const TF>& ssa_gas,     // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& tau_cld,     // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& ssa_cld,     // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& asy_cld,     // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& lay_source,  // (nlay, ncol)
            const Array_map_1d<const TF>& sfc_source,  // (ncol)
            const Array_map_1d<const TF>& sfc_emis,    // (ncol)
            const TF inc_dif,
            const Fluxes_lw& fluxes,
            const Scratch& scratch);

    // Add a plane-parallel solve of one g-point into the ray tracer's own fluxes.
    //
    // Where a g-point's gas is opaque on the scale of a grid cell, a photon cannot
    // cross one before it is absorbed, so horizontal transport has nothing to move and
    // the plane-parallel answer is the same answer -- reached without tracing anything
    // and without Monte Carlo noise. Reference: convert_1d_to_rt_output, which the
    // reference reaches through min_mfp_grid_ratio.
    //
    // flux_up and flux_dn are one g-point's profiles, (nlev, ncol), in the caller's own
    // vertical orientation. Only the levels bounding the box are read, plus the one
    // between its top cell and the cell below, which is where the resolved domain ends
    // when the caller has lumped the atmosphere above into that cell.
    void add_plane_parallel(
            const Grid& grid,
            const bool top_at_1,
            const int nlay,
            const bool lumped,
            const Array_map_2d<const TF>& flux_up,   // (nlev, ncol)
            const Array_map_2d<const TF>& flux_dn,   // (nlev, ncol)
            const Fluxes_lw& fluxes);

    void init_python_bindings(py::module_& m);
}
