#pragma once

#include <cstdint>
#include <vector>

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
// The domain is nx by ny by nz cells of dx by dy by dz, periodic in x and y, where dz
// may vary with height. The column index is i + j*nx, which is how the case reader
// already flattens (y, x), so no array is ever reshaped. Optical properties come in as
// (nlay, ncol) for one g-point, as everywhere else in rte3d, with nlay >= nz: the
// layers from nz-1 upward are lumped into the top cell of the box, so the tracer sees
// the whole atmosphere while resolving only the part that is worth resolving.
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


    // A cell's optical properties, all four of them together and aligned so that the
    // walk fetches them in one instruction.
    //
    // The extinction used to live in an array of its own, beside a three-wide
    // scattering struct. Every collision then issued four separate 32-bit loads for
    // four contiguous numbers of one cell, and the photon kernel is bound by the L1
    // queue those loads sit in -- 74 percent of its stall cycles are spent waiting
    // for it, with DRAM at 3.5 percent. One aligned struct is one wide load instead.
    struct alignas(16) Optics_cell
    {
        TF k_ext;       // extinction coefficient [1/m]
        TF k_sca_gas;   // Rayleigh scattering coefficient [1/m]
        TF k_sca_cld;   // cloud scattering coefficient [1/m]
        TF asy_cld;     // cloud asymmetry parameter
    };


    // A photon count: weight scored into a cell, held in fixed point with 32 fractional
    // bits. The walk scores with atomics, and the GPU does not order them twice the
    // same way; floating-point addition would then round differently from one run to
    // the next, where integer addition is associative, so any order gives the same
    // bits. Unsigned so that the atomic is a native one: a negative score, which the
    // longwave's emission is, wraps around and comes back out of from_count signed.
    // And uint64_t in particular, not unsigned long long: that is the type Kokkos has
    // a native 64-bit add for, and the other one is the same width but falls through
    // to a compare-and-swap loop, which cost the shortwave walk a quarter of its time.
    //
    // The 31 integer bits hold two billion photons' worth of weight in one cell, far
    // beyond any launch. Below them, a float weight times a power of two is exact, so
    // at single precision the conversion loses nothing a weight of 2^-9 or more has.
    using Count = std::uint64_t;

    KOKKOS_INLINE_FUNCTION constexpr TF count_unit() { return TF(4294967296.); }   // 2^32

    KOKKOS_INLINE_FUNCTION
    Count to_count(const TF w)
    {
        return static_cast<Count>(static_cast<long long>(Kokkos::round(w*count_unit())));
    }

    KOKKOS_INLINE_FUNCTION
    TF from_count(const Count c)
    {
        return static_cast<TF>(static_cast<long long>(c))/count_unit();
    }


    // One point of a vertical lookup table: the cell the bottom of its interval is in,
    // and the face, if any, that lies inside the interval. The table's spacing is
    // finer than the thinnest cell, so there is never more than one. Aligned so that
    // the walk reads both in one load.
    struct alignas(2*sizeof(TF)) Lut_entry
    {
        TF face;    // the face inside the interval, or the largest TF if there is none
        int k;      // the cell below that face
    };


    // The vertical axis of a grid: n cells between the n+1 face heights lev, the first
    // of which is the surface at zero. The walk asks it which cell a height is in and
    // where a cell's faces are.
    //
    // dz_constant picks how. With equal spacing that is a multiplication by d_inv, as
    // it is in x and y. Otherwise a lookup table on a spacing finer than the thinnest
    // cell gives, in one load, the cell at the bottom of the table interval and the
    // face inside it, and one comparison against that face finishes the job. The walk
    // is compiled for both, and the grid says which applies.
    //
    // rte-rrtmgp-cpp's height_to_int keeps the cell alone in its table and corrects it
    // by stepping against lev, which is two loads where the second has to wait for
    // the first; in a walk bound by its loads, that chain was most of what a stretched
    // grid cost.
    struct Z_axis
    {
        Array_map_1d<const TF> lev;         // (n+1) face heights [m]
        Array_map_1d<const Lut_entry> lut;  // (lut_size)
        TF d = TF(0.), d_inv = TF(0.);
        TF lut_d_inv = TF(0.);
        int n = 0, lut_size = 0;

        template<bool dz_constant> KOKKOS_INLINE_FUNCTION
        int index(const TF z) const
        {
            if constexpr (dz_constant)
            {
                // The cast rounds toward zero, which is the rounding this wants.
                const int k = static_cast<int>(z*d_inv);
                return k < n ? k : n - 1;
            }
            else
            {
                const int i = static_cast<int>(z*lut_d_inv);
                const Lut_entry e = lut(i < 0 ? 0 : (i < lut_size ? i : lut_size - 1));
                return z >= e.face ? e.k + 1 : e.k;
            }
        }

        // The bottom face of cell k, and so the top face of cell k-1.
        template<bool dz_constant> KOKKOS_INLINE_FUNCTION
        TF face(const int k) const
        {
            if constexpr (dz_constant)
                return k*d;
            else
                return lev(k);
        }
    };


    // The ray-tracing box, and the coarser grid the null-collision extinction is taken
    // over. kn_* of 1 in every direction is legal and makes the tracer take a single
    // maximum over the whole domain; the finer that grid, the fewer null collisions,
    // at the price of the memory it takes and the cell crossings it adds.
    //
    // The vertical is given by its nz+1 face heights, which need not be equally
    // spaced. The null-collision blocks in z then have their faces on cell faces, block
    // kn_k starting at cell kn_k*nz/kn_z, so that each holds whole cells; there are
    // never more of them than there are cells.
    struct Grid
    {
        int nx = 0, ny = 0, nz = 0;
        TF dx = TF(0.), dy = TF(0.);
        int kn_x = 1, kn_y = 1, kn_z = 1;

        // The face heights, on the host and on the device, counted from the surface:
        // make subtracts the first one.
        std::vector<TF> z_lev;          // (nz+1)
        Array_1d<TF> z_lev_d;           // (nz+1)
        Array_1d<TF> kn_z_lev_d;        // (kn_z+1)
        Array_1d<Lut_entry> z_lut, kn_z_lut;  // (lut_size)
        TF lut_dz = TF(0.);

        // Whether every cell is equally deep, to the round-off of the face heights.
        // The walk then takes the equally spaced arithmetic, which is faster.
        bool dz_constant = true;

        // Build a grid, filling in any null-collision block count left at zero with
        // blocks a quarter of the domain across. That is a compromise: coarser and the
        // tracer takes null collisions it did not have to, finer and it crosses block
        // faces it did not have to.
        static Grid make(const int nx, const int ny, const int nz,
                         const TF dx, const TF dy, const std::vector<TF>& z_lev,
                         const int kn_x = 0, const int kn_y = 0, const int kn_z = 0);

        int ncol() const { return nx*ny; }
        TF dz(const int k) const { return z_lev[k+1] - z_lev[k]; }
        TF dz_min() const;
        Vector<int> cells() const { return Vector<int>{nx, ny, nz}; }
        Vector<int> kn_cells() const { return Vector<int>{kn_x, kn_y, kn_z}; }
        Vector<TF> size() const { return Vector<TF>{nx*dx, ny*dy, z_lev[nz]}; }

        // The vertical axes of the box and of the null-collision grid, as the walk
        // reads them.
        Z_axis z_axis() const;
        Z_axis kn_z_axis() const;
    };


    // Where the fluxes go, accumulated over g-points by the caller. The two-dimensional
    // ones are at the surface, at the top of the atmosphere (toa, the top of the lumped
    // cell) and at the top of the resolved domain (tod, the bottom of the lumped cell;
    // the same as toa when nothing was lumped); the three-dimensional ones are
    // the absorbed flux per unit height [W/m3], which is what a heating rate is built
    // from and what the tracer produces naturally.
    struct Fluxes_rt
    {
        Array_1d<TF> toa_dn;    // (ncol) downward at the top of the atmosphere
        Array_1d<TF> toa_up;    // (ncol) upward at the top of the atmosphere
        Array_1d<TF> tod_dn;    // (ncol) downward at the top of the resolved domain
        Array_1d<TF> tod_up;    // (ncol) upward at the top of the resolved domain
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
        Array_2d<Optics_cell> optics;   // (nz, ncol)
        Array_3d<TF> k_null;            // (kn_z, kn_y, kn_x) maximum extinction per block

        // Photon counts, zeroed at the start of every trace. Separate from the fluxes
        // because a count becomes a flux only once the trace is over.
        Array_1d<Count> toa_dn, toa_up, tod_dn, tod_up;          // (ncol)
        Array_1d<Count> sfc_dir, sfc_dif, sfc_up;                // (ncol)
        Array_2d<Count> atmos_dir, atmos_dif;                    // (nz, ncol)

        Rand::Qrng_table qrng;

        static Scratch make(const Grid& grid);
    };


    // Smallest extinction the null-collision grid is allowed to hold. Without it a
    // transparent block would give an infinite free path, and the reciprocal of it
    // appears in the collision test. Reference: k_ext_null_min.
    //
    // A function rather than a constant, as eps() and min_k() in the solver kernels
    // are: a device lambda that passes it by reference to Kokkos::max would otherwise
    // odr-use a host-side object.
    KOKKOS_INLINE_FUNCTION constexpr TF k_null_min() { return TF(1.e-3); }

    // Floor on a cell's extinction, so that the ratios the walk forms out of it stay
    // finite in a cell that happens to be empty.
    KOKKOS_INLINE_FUNCTION constexpr TF k_ext_min() { return TF(1.e-25); }


    // The layer a ray-tracing cell reads, in the caller's own vertical orientation.
    KOKKOS_INLINE_FUNCTION
    int layer_of(const int k, const int nlay, const bool top_at_1)
    {
        return top_at_1 ? nlay - 1 - k : k;
    }


    // Extinction and the two scattering coefficients per cell, from the optical depths
    // of one g-point, and the null-collision maxima taken over them. Shared with the
    // longwave tracer, which builds exactly the same scene out of its own optical
    // depths; only the sources and the scoring differ between the two.
    //
    // Gas and cloud stay apart: a scattering event has to pick which of the two did
    // it, and each has its own phase function. The cloud triple may be empty, and so
    // may ssa_gas, for a gas that does not scatter.
    void bundle_optics(
            const Grid& grid, const bool top_at_1, const int nlay,
            const Array_map_2d<const TF>& tau_gas,   // (nlay, ncol)
            const Array_map_2d<const TF>& ssa_gas,   // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& tau_cld,   // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& ssa_cld,   // (nlay, ncol), may be empty
            const Array_map_2d<const TF>& asy_cld,   // (nlay, ncol), may be empty
            const Array_2d<Optics_cell>& optics);    // (nz, ncol)

    // The top cell of the box holds everything above it: the atmosphere reaches far
    // higher than the part worth resolving in three dimensions, and lumping the rest
    // into one homogeneous cell keeps the optical depth right without the cells.
    // Call it after bundle_optics, which it overwrites the top row of.
    void bundle_optics_tod(
            const Grid& grid, const bool top_at_1, const int nlay,
            const Array_map_2d<const TF>& tau_gas,
            const Array_map_2d<const TF>& ssa_gas,
            const Array_map_2d<const TF>& tau_cld,
            const Array_map_2d<const TF>& ssa_cld,
            const Array_map_2d<const TF>& asy_cld,
            const Array_2d<Optics_cell>& optics);

    // Largest extinction in each block of the coarse grid, which is what the
    // null-collision transport marches on.
    void create_knull_grid(
            const Grid& grid, const Array_map_2d<const Optics_cell>& optics,
            const Array_3d<TF>& k_null);


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
    //
    // seed picks the photon paths: a run is reproducible for a given seed, and a
    // caller that traces the same scene repeatedly, as a model does every radiation
    // step, passes a new seed each time so that the Monte Carlo noise is not the same
    // pattern every call. Seed 0 is the default.
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
            const Scratch& scratch,
            const unsigned int seed = 0);

    void init_python_bindings(py::module_& m);
}
