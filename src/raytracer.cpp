#include <algorithm>
#include <cmath>

#include "raytracer.h"
#include "raytracer_kernels.h"
#include "types.h"


namespace
{
    using Raytracer::Grid;
    using Raytracer::Optics_scat;
    using Raytracer::Vector;

    // Smallest extinction the null-collision grid is allowed to hold. Without it a
    // transparent block would give an infinite free path, and the reciprocal of it
    // appears in the collision test. Reference: k_ext_null_min.
    constexpr TF k_null_min = TF(1.e-3);

    // Floor on a cell's extinction, so that the ratios the walk forms out of it stay
    // finite in a cell that happens to be empty.
    constexpr TF k_ext_min = TF(1.e-25);


    // The layer a ray-tracing cell reads, in the caller's own vertical orientation.
    KOKKOS_INLINE_FUNCTION
    int layer_of(const int k, const int nlay, const bool top_at_1)
    {
        return top_at_1 ? nlay - 1 - k : k;
    }


    // The next power of two at or above n, and never below two: a Sobol pair maps onto
    // a power-of-two lattice, so the sequence covers that rather than the grid itself.
    unsigned int next_pow2(const int n)
    {
        unsigned int p = 2;
        while (p < static_cast<unsigned int>(n))
            p *= 2;

        return p;
    }


    // Extinction and the two scattering coefficients per cell, from the optical depths
    // of one g-point. Gas and cloud stay apart: a scattering event has to pick which
    // of the two did it, and each has its own phase function.
    void bundle_optics(
            const Grid& grid, const bool top_at_1, const int nlay,
            const Array_map_2d<const TF>& tau_gas,
            const Array_map_2d<const TF>& ssa_gas,
            const Array_map_2d<const TF>& tau_cld,
            const Array_map_2d<const TF>& ssa_cld,
            const Array_map_2d<const TF>& asy_cld,
            const Array_2d<TF>& k_ext,
            const Array_2d<Optics_scat>& scat)
    {
        const int ncol = grid.ncol();
        const TF dz_inv = TF(1.)/grid.dz;
        const bool clouds = tau_cld.size() > 0;

        parallel_for_2d("rt_bundle_optics", {0, 0}, {grid.nz - 1, ncol},
            KOKKOS_LAMBDA(const int k, const int icol)
            {
                const int ilay = layer_of(k, nlay, top_at_1);

                const TF k_gas = tau_gas(ilay, icol)*dz_inv;
                const TF k_cld = clouds ? tau_cld(ilay, icol)*dz_inv : TF(0.);

                k_ext(k, icol) = Kokkos::max(k_ext_min, k_gas + k_cld);

                scat(k, icol).k_sca_gas = k_gas*ssa_gas(ilay, icol);
                scat(k, icol).k_sca_cld = clouds ? k_cld*ssa_cld(ilay, icol) : TF(0.);
                scat(k, icol).asy_cld = clouds ? asy_cld(ilay, icol) : TF(0.);
            });
    }


    // The top cell of the box holds everything above it: the atmosphere reaches far
    // higher than the part worth resolving in three dimensions, and lumping the rest
    // into one homogeneous cell keeps the optical depth right without the cells.
    // Reference: bundles_optical_props_tod.
    void bundle_optics_tod(
            const Grid& grid, const bool top_at_1, const int nlay,
            const Array_map_2d<const TF>& tau_gas,
            const Array_map_2d<const TF>& ssa_gas,
            const Array_map_2d<const TF>& tau_cld,
            const Array_map_2d<const TF>& ssa_cld,
            const Array_map_2d<const TF>& asy_cld,
            const Array_2d<TF>& k_ext,
            const Array_2d<Optics_scat>& scat)
    {
        const int ncol = grid.ncol();
        const int ktod = grid.nz - 1;
        const TF dz_inv = TF(1.)/grid.dz;
        const bool clouds = tau_cld.size() > 0;

        parallel_for_1d("rt_bundle_optics_tod", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                TF tau_gas_sum = TF(0.);
                TF sca_gas_sum = TF(0.);
                TF tau_cld_sum = TF(0.);
                TF sca_cld_sum = TF(0.);
                TF scag_cld_sum = TF(0.);

                for (int k=ktod; k<nlay; ++k)
                {
                    const int ilay = layer_of(k, nlay, top_at_1);

                    tau_gas_sum += tau_gas(ilay, icol);
                    sca_gas_sum += tau_gas(ilay, icol)*ssa_gas(ilay, icol);

                    if (clouds)
                    {
                        const TF sca = tau_cld(ilay, icol)*ssa_cld(ilay, icol);
                        tau_cld_sum += tau_cld(ilay, icol);
                        sca_cld_sum += sca;
                        scag_cld_sum += sca*asy_cld(ilay, icol);
                    }
                }

                k_ext(ktod, icol) = Kokkos::max(k_ext_min, (tau_gas_sum + tau_cld_sum)*dz_inv);

                scat(ktod, icol).k_sca_gas = sca_gas_sum*dz_inv;
                scat(ktod, icol).k_sca_cld = sca_cld_sum*dz_inv;
                scat(ktod, icol).asy_cld = sca_cld_sum > TF(0.) ? scag_cld_sum/sca_cld_sum : TF(0.);
            });
    }


    // Largest extinction in each block of the coarse grid, which is what the
    // null-collision transport marches on. Reference: create_knull_grid.
    void create_knull_grid(
            const Grid& grid, const Array_2d<const TF>& k_ext, const Array_3d<TF>& k_null)
    {
        const Vector<int> cells = grid.cells();
        const Vector<int> kn = grid.kn_cells();

        const TF fx = TF(cells.x)/TF(kn.x);
        const TF fy = TF(cells.y)/TF(kn.y);
        const TF fz = TF(cells.z)/TF(kn.z);

        parallel_for_3d("rt_knull_grid", {0, 0, 0}, {kn.z, kn.y, kn.x},
            KOKKOS_LAMBDA(const int kn_k, const int kn_j, const int kn_i)
            {
                const int i0 = static_cast<int>(kn_i*fx);
                const int i1 = Kokkos::min(cells.x - 1, static_cast<int>((kn_i + 1)*fx));
                const int j0 = static_cast<int>(kn_j*fy);
                const int j1 = Kokkos::min(cells.y - 1, static_cast<int>((kn_j + 1)*fy));
                const int k0 = static_cast<int>(kn_k*fz);
                const int k1 = Kokkos::min(cells.z - 1, static_cast<int>((kn_k + 1)*fz));

                TF k_max = k_null_min;
                for (int k=k0; k<=k1; ++k)
                    for (int j=j0; j<=j1; ++j)
                        for (int i=i0; i<=i1; ++i)
                            k_max = Kokkos::max(k_max, k_ext(k, i + j*cells.x));

                k_null(kn_k, kn_j, kn_i) = k_max;
            });
    }


    // One launch of the photon walk. Templated on the transport mode so that the
    // independent-column test is compiled away rather than taken per step; in an
    // anonymous namespace, so the two closure types cannot collide with another
    // translation unit's.
    template<bool independent_column>
    void launch_photons(
            const Rt_kernels::Scene& scene, const int nthread,
            const int photons_per_thread, const int photons_extra,
            const unsigned int gpt_offset)
    {
        Kokkos::parallel_for("rt_trace_photons",
            Kokkos::RangePolicy<Default_exec>(0, nthread),
            KOKKOS_LAMBDA(const int n)
            {
                // The photons left over after an even split go to the first threads,
                // one each, so that the total is exactly what the caller asked for.
                const int nshoot = photons_per_thread + (n < photons_extra ? 1 : 0);
                const unsigned int start = static_cast<unsigned int>(n)*photons_per_thread
                        + static_cast<unsigned int>(n < photons_extra ? n : photons_extra);

                Rt_kernels::trace_photons<independent_column>(
                        scene, nshoot, gpt_offset + start, gpt_offset + start);
            });
    }


    // Photon counts to fluxes. Every photon carries toa_src/photons_per_pixel watts
    // per square metre; the absorbed flux is per unit height, so it is divided by the
    // cell depth as well. Reference: count_to_flux_2d and count_to_flux_3d.
    void count_to_flux(
            const Grid& grid, const TF flux_per_photon,
            const Raytracer::Scratch& s, const Raytracer::Fluxes_rt& f)
    {
        const int ncol = grid.ncol();
        const TF per_volume = flux_per_photon/grid.dz;

        const auto tod_dn = s.tod_dn, tod_up = s.tod_up;
        const auto sfc_dir = s.sfc_dir, sfc_dif = s.sfc_dif, sfc_up = s.sfc_up;
        const auto atmos_dir = s.atmos_dir, atmos_dif = s.atmos_dif;

        const auto f_tod_dn = f.tod_dn, f_tod_up = f.tod_up;
        const auto f_sfc_dir = f.sfc_dir, f_sfc_dif = f.sfc_dif, f_sfc_up = f.sfc_up;
        const auto f_abs_dir = f.abs_dir, f_abs_dif = f.abs_dif;

        parallel_for_1d("rt_count_to_flux_2d", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                f_tod_dn(icol) += tod_dn(icol)*flux_per_photon;
                f_tod_up(icol) += tod_up(icol)*flux_per_photon;
                f_sfc_dir(icol) += sfc_dir(icol)*flux_per_photon;
                f_sfc_dif(icol) += sfc_dif(icol)*flux_per_photon;
                f_sfc_up(icol) += sfc_up(icol)*flux_per_photon;
            });

        parallel_for_2d("rt_count_to_flux_3d", {0, 0}, {grid.nz, ncol},
            KOKKOS_LAMBDA(const int k, const int icol)
            {
                f_abs_dir(k, icol) += atmos_dir(k, icol)*per_volume;
                f_abs_dif(k, icol) += atmos_dif(k, icol)*per_volume;
            });
    }
}


Raytracer::Grid Raytracer::Grid::make(
        const int nx, const int ny, const int nz,
        const TF dx, const TF dy, const TF dz,
        const int kn_x, const int kn_y, const int kn_z)
{
    const auto blocks = [](const int n, const int given)
    { return given > 0 ? given : std::max(1, n/4); };

    Grid grid;
    grid.nx = nx; grid.ny = ny; grid.nz = nz;
    grid.dx = dx; grid.dy = dy; grid.dz = dz;
    grid.kn_x = blocks(nx, kn_x);
    grid.kn_y = blocks(ny, kn_y);
    grid.kn_z = blocks(nz, kn_z);

    return grid;
}


Raytracer::Fluxes_rt Raytracer::Fluxes_rt::make(const Grid& grid)
{
    const int ncol = grid.ncol();

    return Fluxes_rt{
            Array_1d<TF>("rt_tod_dn", ncol),
            Array_1d<TF>("rt_tod_up", ncol),
            Array_1d<TF>("rt_sfc_dir", ncol),
            Array_1d<TF>("rt_sfc_dif", ncol),
            Array_1d<TF>("rt_sfc_up", ncol),
            Array_2d<TF>("rt_abs_dir", grid.nz, ncol),
            Array_2d<TF>("rt_abs_dif", grid.nz, ncol)};
}


void Raytracer::Fluxes_rt::zero() const
{
    Kokkos::deep_copy(tod_dn, TF(0.));
    Kokkos::deep_copy(tod_up, TF(0.));
    Kokkos::deep_copy(sfc_dir, TF(0.));
    Kokkos::deep_copy(sfc_dif, TF(0.));
    Kokkos::deep_copy(sfc_up, TF(0.));
    Kokkos::deep_copy(abs_dir, TF(0.));
    Kokkos::deep_copy(abs_dif, TF(0.));
}


Raytracer::Scratch Raytracer::Scratch::make(const Grid& grid)
{
    const int ncol = grid.ncol();

    Scratch s;
    s.k_ext = Array_2d<TF>(Kokkos::view_alloc("rt_k_ext", Kokkos::WithoutInitializing),
                           grid.nz, ncol);
    s.scat = Array_2d<Optics_scat>(
            Kokkos::view_alloc("rt_scat", Kokkos::WithoutInitializing), grid.nz, ncol);
    s.k_null = Array_3d<TF>(Kokkos::view_alloc("rt_k_null", Kokkos::WithoutInitializing),
                            grid.kn_z, grid.kn_y, grid.kn_x);

    s.tod_dn = Array_1d<TF>("rt_tod_dn_count", ncol);
    s.tod_up = Array_1d<TF>("rt_tod_up_count", ncol);
    s.sfc_dir = Array_1d<TF>("rt_sfc_dir_count", ncol);
    s.sfc_dif = Array_1d<TF>("rt_sfc_dif_count", ncol);
    s.sfc_up = Array_1d<TF>("rt_sfc_up_count", ncol);
    s.atmos_dir = Array_2d<TF>("rt_atmos_dir_count", grid.nz, ncol);
    s.atmos_dif = Array_2d<TF>("rt_atmos_dif_count", grid.nz, ncol);

    s.qrng = Rand::Qrng_table::make();

    return s;
}


void Raytracer::trace_rays(
        const Grid& grid,
        const bool top_at_1,
        const bool independent_column,
        const int photons_per_pixel,
        const int igpt,
        const Array_map_2d<const TF>& tau_gas,
        const Array_map_2d<const TF>& ssa_gas,
        const Array_map_2d<const TF>& tau_cld,
        const Array_map_2d<const TF>& ssa_cld,
        const Array_map_2d<const TF>& asy_cld,
        const Array_map_1d<const TF>& sfc_alb,
        const TF mu0,
        const TF azi,
        const TF inc_dir,
        const TF inc_dif,
        const Fluxes_rt& fluxes,
        const Scratch& scratch)
{
    const int nlay = static_cast<int>(tau_gas.extent(0));
    const int ncol = grid.ncol();

    bundle_optics(grid, top_at_1, nlay, tau_gas, ssa_gas, tau_cld, ssa_cld, asy_cld,
                  scratch.k_ext, scratch.scat);
    bundle_optics_tod(grid, top_at_1, nlay, tau_gas, ssa_gas, tau_cld, ssa_cld, asy_cld,
                      scratch.k_ext, scratch.scat);

    create_knull_grid(
            grid,
            Array_map_2d<const TF>(scratch.k_ext.data(), grid.nz, ncol),
            scratch.k_null);

    Kokkos::deep_copy(scratch.tod_dn, TF(0.));
    Kokkos::deep_copy(scratch.tod_up, TF(0.));
    Kokkos::deep_copy(scratch.sfc_dir, TF(0.));
    Kokkos::deep_copy(scratch.sfc_dif, TF(0.));
    Kokkos::deep_copy(scratch.sfc_up, TF(0.));
    Kokkos::deep_copy(scratch.atmos_dir, TF(0.));
    Kokkos::deep_copy(scratch.atmos_dif, TF(0.));

    Rt_kernels::Scene scene;
    scene.k_ext = Array_map_2d<const TF>(scratch.k_ext.data(), grid.nz, ncol);
    scene.scat = Array_map_2d<const Optics_scat>(scratch.scat.data(), grid.nz, ncol);
    scene.k_null = Array_map_3d<const TF>(
            scratch.k_null.data(), grid.kn_z, grid.kn_y, grid.kn_x);
    scene.sfc_alb = sfc_alb;

    scene.tod_dn = Array_map_1d<TF>(scratch.tod_dn.data(), ncol);
    scene.tod_up = Array_map_1d<TF>(scratch.tod_up.data(), ncol);
    scene.sfc_dir = Array_map_1d<TF>(scratch.sfc_dir.data(), ncol);
    scene.sfc_dif = Array_map_1d<TF>(scratch.sfc_dif.data(), ncol);
    scene.sfc_up = Array_map_1d<TF>(scratch.sfc_up.data(), ncol);
    scene.atmos_dir = Array_map_2d<TF>(scratch.atmos_dir.data(), grid.nz, ncol);
    scene.atmos_dif = Array_map_2d<TF>(scratch.atmos_dif.data(), grid.nz, ncol);

    scene.grid_cells = grid.cells();
    scene.grid_d = grid.d();
    scene.grid_size = grid.size();
    scene.kn_grid = grid.kn_cells();
    scene.kn_grid_d = Vector<TF>{scene.grid_size.x/grid.kn_x,
                                 scene.grid_size.y/grid.kn_y,
                                 scene.grid_size.z/grid.kn_z};

    // The sun's direction of travel. The azimuth is measured from north and increases
    // clockwise, as the case files give it, so it turns into the mathematical
    // convention by the quarter turn below. Reference: raytracer_sw.cu.
    const TF zenith = std::acos(mu0);
    scene.sun_direction = Vector<TF>{
            TF(-std::sin(zenith)*std::cos(TF(0.5*M_PI) - azi)),
            TF(-std::sin(zenith)*std::sin(TF(0.5*M_PI) - azi)),
            TF(-mu0)};

    scene.inc_dir = inc_dir;
    scene.inc_dif = inc_dif;
    scene.qrng_nx = next_pow2(grid.nx);
    scene.qrng_ny = next_pow2(grid.ny);
    scene.qrng = scratch.qrng.view();

    // Photons are drawn over the power-of-two lattice the quasi-random sequence
    // covers, and the ones that land outside the grid are thrown away, which is what
    // leaves photons_per_pixel of them in every pixel that exists.
    const long long photons_total =
            static_cast<long long>(photons_per_pixel)*scene.qrng_nx*scene.qrng_ny;

    // Enough threads to fill the machine, but never more than there are photons.
    #ifdef USEGPU
    constexpr long long target_threads = 1LL << 19;
    #else
    const long long target_threads = 16LL*Default_exec().concurrency();
    #endif

    const int nthread = static_cast<int>(std::min(photons_total, target_threads));
    const int photons_per_thread = static_cast<int>(photons_total/nthread);
    const int photons_extra = static_cast<int>(photons_total % nthread);

    const unsigned int gpt_offset = static_cast<unsigned int>(igpt*photons_total);

    if (independent_column)
        launch_photons<true>(scene, nthread, photons_per_thread, photons_extra, gpt_offset);
    else
        launch_photons<false>(scene, nthread, photons_per_thread, photons_extra, gpt_offset);

    count_to_flux(grid, (inc_dir + inc_dif)/photons_per_pixel, scratch, fluxes);
}
