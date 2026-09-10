#pragma once

#include "random.h"
#include "types.h"


// The shortwave Monte Carlo ray tracer: forward, three-dimensional, null-collision.
//
// Where the two-stream solver of rte_sw.h treats every column on its own, this one
// follows photons through a Cartesian box, so a cloud casts a shadow on its neighbour
// and lights the column beside it from the side. It is the Kokkos counterpart of
// raytracer_sw.cu in rte-rrtmgp-cpp, with aerosols and the Mie phase function left
// out: clouds scatter as Henyey-Greenstein with the asymmetry parameter the RRTMGP
// tables give, gases as Rayleigh.
//
// The domain is nx by ny by nz cells of dx by dy by dz, periodic in x and y. The
// column index is i + j*nx, which is how the case reader already flattens (y, x), so
// no array is ever reshaped. Optical properties come in as (nlay, ncol) for one
// g-point, as everywhere else in rte3d, with nlay >= nz: the layers from nz-1 upward
// are lumped into the top cell of the box, so the tracer sees the whole atmosphere
// while resolving only the part that is worth resolving.
//
// One call traces one g-point and adds its share into the fluxes. Nothing here carries
// a g-point dimension.
namespace Raytracer
{
    template<typename T>
    struct Vector
    {
        T x, y, z;
    };


    // The scattering part of a cell's optical properties, kept apart from the total
    // extinction because a scattering event has to pick which of the two did it.
    struct Optics_scat
    {
        TF k_sca_gas;   // Rayleigh scattering coefficient [1/m]
        TF k_sca_cld;   // cloud scattering coefficient [1/m]
        TF asy_cld;     // cloud asymmetry parameter
    };


    // The ray-tracing box, and the coarser grid the null-collision extinction is taken
    // over. kn_* of 1 in every direction is legal and makes the tracer take a single
    // maximum over the whole domain; the finer that grid, the fewer null collisions,
    // at the price of the memory it takes and the cell crossings it adds.
    struct Grid
    {
        int nx = 0, ny = 0, nz = 0;
        TF dx = TF(0.), dy = TF(0.), dz = TF(0.);
        int kn_x = 1, kn_y = 1, kn_z = 1;

        // Build a grid, filling in any null-collision block count left at zero with
        // blocks a quarter of the domain across. That is a compromise: coarser and the
        // tracer takes null collisions it did not have to, finer and it crosses block
        // faces it did not have to.
        static Grid make(const int nx, const int ny, const int nz,
                         const TF dx, const TF dy, const TF dz,
                         const int kn_x = 0, const int kn_y = 0, const int kn_z = 0);

        int ncol() const { return nx*ny; }
        Vector<int> cells() const { return Vector<int>{nx, ny, nz}; }
        Vector<TF> d() const { return Vector<TF>{dx, dy, dz}; }
        Vector<int> kn_cells() const { return Vector<int>{kn_x, kn_y, kn_z}; }
        Vector<TF> size() const { return Vector<TF>{nx*dx, ny*dy, nz*dz}; }
    };


    // Where the fluxes go, accumulated over g-points by the caller. The two-dimensional
    // ones are at the surface and at the top of the box; the three-dimensional ones are
    // the absorbed flux per unit height [W/m3], which is what a heating rate is built
    // from and what the tracer produces naturally.
    struct Fluxes_rt
    {
        Array_1d<TF> tod_dn;    // (ncol) downward at the top of the domain
        Array_1d<TF> tod_up;    // (ncol) upward at the top of the domain
        Array_1d<TF> sfc_dir;   // (ncol) downward direct at the surface
        Array_1d<TF> sfc_dif;   // (ncol) downward diffuse at the surface
        Array_1d<TF> sfc_up;    // (ncol) upward at the surface
        Array_2d<TF> abs_dir;   // (nz, ncol) absorbed direct
        Array_2d<TF> abs_dif;   // (nz, ncol) absorbed diffuse

        static Fluxes_rt make(const Grid& grid);

        void zero() const;
    };


    // Everything a trace reuses from one g-point to the next: the cell properties it
    // derives from the optical depths, the null-collision maxima, and the photon
    // counts. Built once before the g-point loop, as Rte_sw::Two_stream_scratch is,
    // because freeing a device allocation per g-point synchronizes the device.
    struct Scratch
    {
        Array_2d<TF> k_ext;             // (nz, ncol) extinction coefficient [1/m]
        Array_2d<Optics_scat> scat;     // (nz, ncol)
        Array_3d<TF> k_null;            // (kn_z, kn_y, kn_x) maximum extinction per block

        // Photon counts, zeroed at the start of every trace. Separate from the fluxes
        // because a count becomes a flux only once the trace is over.
        Array_1d<TF> tod_dn, tod_up, sfc_dir, sfc_dif, sfc_up;   // (ncol)
        Array_2d<TF> atmos_dir, atmos_dif;                       // (nz, ncol)

        Rand::Qrng_table qrng;

        static Scratch make(const Grid& grid);
    };


    // Trace one g-point.
    //
    // tau_gas and ssa_gas are the gas optical depth and its Rayleigh single-scattering
    // albedo, as Gas_optics::compute_tau_sw produces them. The cloud triple may be
    // empty, which means a clear sky. All five are (nlay, ncol) and are read in the
    // caller's own vertical orientation, given by top_at_1.
    //
    // mu0 and azi are the sun's cosine zenith and its azimuth in radians, one pair for
    // the whole domain; azi is measured from north, increasing clockwise, as the case
    // files give it. inc_dir and inc_diffuse are the irradiances entering the top of
    // the domain, the first already multiplied by mu0.
    //
    // photons_per_pixel is per g-point. independent_column switches off horizontal
    // transport, which turns the tracer into an expensive but exact column solver and
    // is what a comparison against the two-stream wants.
    void trace_rays(
            const Grid& grid,
            const bool top_at_1,
            const bool independent_column,
            const int photons_per_pixel,
            const int igpt,
            const Array_map_2d<const TF>& tau_gas,   // (nlay, ncol)
            const Array_map_2d<const TF>& ssa_gas,   // (nlay, ncol)
            const Array_map_2d<const TF>& tau_cld,   // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& ssa_cld,   // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& asy_cld,   // (nlay, ncol), may be empty
            const Array_map_1d<const TF>& sfc_alb,   // (ncol)
            const TF mu0,
            const TF azi,
            const TF inc_dir,
            const TF inc_dif,
            const Fluxes_rt& fluxes,
            const Scratch& scratch);

    void init_python_bindings(py::module_& m);
}
