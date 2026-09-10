#include <algorithm>
#include <cmath>

#include "raytracer.h"
#include "raytracer_lw.h"
#include "raytracer_lw_kernels.h"
#include "types.h"


namespace
{
    using Raytracer_lw::Grid;
    using Raytracer_lw::Optics_cell;
    using Raytracer_lw::Vector;
    using Raytracer::layer_of;


    // The absorption optical depth of one layer, gas and cloud together. What a layer
    // emits is proportional to it: an isotropic emitter of absorption optical depth
    // tau_abs at Planck source B radiates 4*pi*tau_abs*B per unit area over the whole
    // sphere.
    KOKKOS_INLINE_FUNCTION
    TF tau_abs(
            const Array_map_2d<const TF>& tau_gas, const Array_map_2d<const TF>& ssa_gas,
            const Array_map_2d<const TF>& tau_cld, const Array_map_2d<const TF>& ssa_cld,
            const bool clouds, const bool cld_scatters, const int ilay, const int icol)
    {
        const TF gas = tau_gas(ilay, icol)*(TF(1.) - ssa_gas(ilay, icol));
        const TF cld = !clouds ? TF(0.)
                : (cld_scatters ? tau_cld(ilay, icol)*(TF(1.) - ssa_cld(ilay, icol))
                                : tau_cld(ilay, icol));

        return gas + cld;
    }


    // The emitted power of every source, in the slot order the walk expects: row 0 the
    // surface, rows 1 to nz the cells of the box from the bottom up, row nz+1 the top
    // boundary. Reference: get_emitted_power.
    //
    // A cell radiates 4*pi*tau_abs*B; the surface radiates pi*emis*B into the
    // hemisphere above it, the same pi that Rte_lw::solver_2stream applies to the same
    // sfc_source. Everything here is per unit area, so no cell thickness appears: the
    // optical depth already carries it.
    //
    // Read off tau and ssa rather than off the bundled Optics_cell, because the top
    // cell needs the layers it lumped one at a time -- see below -- and the two would
    // otherwise be computed two different ways.
    void bundle_emission(
            const Grid& grid, const bool top_at_1, const int nlay,
            const Array_map_2d<const TF>& tau_gas,
            const Array_map_2d<const TF>& ssa_gas,
            const Array_map_2d<const TF>& tau_cld,
            const Array_map_2d<const TF>& ssa_cld,
            const Array_map_2d<const TF>& lay_source,
            const Array_map_1d<const TF>& sfc_source,
            const Array_map_1d<const TF>& sfc_emis,
            const TF inc_dif,
            const Array_2d<TF>& emission)
    {
        const int ncol = grid.ncol();
        const int nz = grid.nz;
        const bool clouds = tau_cld.size() > 0;
        const bool cld_scatters = clouds && ssa_cld.size() > 0;

        parallel_for_1d("rt_lw_emission_boundaries", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                emission(0, icol) = TF(M_PI)*sfc_emis(icol)*sfc_source(icol);
                emission(nz + 1, icol) = inc_dif;
            });

        // Every cell but the top one, whose lumped layers need a loop of their own.
        parallel_for_2d("rt_lw_emission", {0, 0}, {nz - 1, ncol},
            KOKKOS_LAMBDA(const int k, const int icol)
            {
                const int ilay = layer_of(k, nlay, top_at_1);
                emission(k + 1, icol) = TF(4.*M_PI)
                        *tau_abs(tau_gas, ssa_gas, tau_cld, ssa_cld, clouds, cld_scatters, ilay, icol)
                        *lay_source(ilay, icol);
            });

        // The top cell stands in for every layer above the box, so its emission is the
        // sum of theirs. Not the lumped absorption times one Planck source: the
        // temperature varies over the layers that were lumped, and it is the product
        // that is summed, not the factors.
        const int ktod = nz - 1;

        parallel_for_1d("rt_lw_emission_tod", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                TF power = TF(0.);
                for (int k=ktod; k<nlay; ++k)
                {
                    const int ilay = layer_of(k, nlay, top_at_1);
                    power += tau_abs(tau_gas, ssa_gas, tau_cld, ssa_cld, clouds, cld_scatters, ilay, icol)
                            *lay_source(ilay, icol);
                }

                emission(ktod + 1, icol) = TF(4.*M_PI)*power;
            });
    }


    // One launch of the photon walk. Templated on the transport mode so that the
    // independent-column test is compiled away rather than taken per step; in an
    // anonymous namespace, so the two closure types cannot collide with another
    // translation unit's.
    template<bool independent_column>
    void launch_photons(
            const Rt_lw_kernels::Scene& scene, const int nthread,
            const int photons_per_thread, const int photons_extra,
            const unsigned int gpt_offset)
    {
        Kokkos::parallel_for("rt_lw_trace_photons",
            Kokkos::RangePolicy<Default_exec>(0, nthread),
            KOKKOS_LAMBDA(const int n)
            {
                // The photons left over after an even split go to the first threads,
                // one each, so that the total is exactly what the caller asked for.
                const int nshoot = photons_per_thread + (n < photons_extra ? 1 : 0);
                const unsigned int start = static_cast<unsigned int>(n)*photons_per_thread
                        + static_cast<unsigned int>(n < photons_extra ? n : photons_extra);

                Rt_lw_kernels::trace_photons<independent_column>(
                        scene, nshoot, gpt_offset + start);
            });
    }


    // Photon counts to fluxes. Every photon carries the same power, the total emitted
    // power over the photon count, and that total is the last element of the
    // cumulative distribution -- so it is read here on the device rather than copied
    // to the host to scale a launch parameter. The net flux is per unit height, so it
    // is divided by the cell depth as well.
    void count_to_flux(
            const Grid& grid, const double photons_total,
            const Raytracer_lw::Scratch& s, const Raytracer_lw::Fluxes_lw& f)
    {
        const int ncol = grid.ncol();
        const TF dz_inv = TF(1.)/grid.dz;

        const auto cdf = s.cdf;
        const int nslot = static_cast<int>(cdf.extent(0));

        const auto tod_dn = s.tod_dn, tod_up = s.tod_up;
        const auto sfc_dn = s.sfc_dn, sfc_up = s.sfc_up;
        const auto atmos = s.atmos;

        const auto f_tod_dn = f.tod_dn, f_tod_up = f.tod_up;
        const auto f_sfc_dn = f.sfc_dn, f_sfc_up = f.sfc_up;
        const auto f_net = f.flux_net;

        parallel_for_1d("rt_lw_count_to_flux_2d", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                const TF per_photon = TF(cdf(nslot - 1)/photons_total);

                f_tod_dn(icol) += tod_dn(icol)*per_photon;
                f_tod_up(icol) += tod_up(icol)*per_photon;
                f_sfc_dn(icol) += sfc_dn(icol)*per_photon;
                f_sfc_up(icol) += sfc_up(icol)*per_photon;
            });

        parallel_for_2d("rt_lw_count_to_flux_3d", {0, 0}, {grid.nz, ncol},
            KOKKOS_LAMBDA(const int k, const int icol)
            {
                const TF per_volume = TF(cdf(nslot - 1)/photons_total)*dz_inv;

                f_net(k, icol) += atmos(k, icol)*per_volume;
            });
    }
}


Raytracer_lw::Fluxes_lw Raytracer_lw::Fluxes_lw::make(const Grid& grid)
{
    const int ncol = grid.ncol();

    return Fluxes_lw{
            Array_1d<TF>("rt_lw_tod_dn", ncol),
            Array_1d<TF>("rt_lw_tod_up", ncol),
            Array_1d<TF>("rt_lw_sfc_dn", ncol),
            Array_1d<TF>("rt_lw_sfc_up", ncol),
            Array_2d<TF>("rt_lw_flux_net", grid.nz, ncol)};
}


void Raytracer_lw::Fluxes_lw::zero() const
{
    Kokkos::deep_copy(tod_dn, TF(0.));
    Kokkos::deep_copy(tod_up, TF(0.));
    Kokkos::deep_copy(sfc_dn, TF(0.));
    Kokkos::deep_copy(sfc_up, TF(0.));
    Kokkos::deep_copy(flux_net, TF(0.));
}


Raytracer_lw::Scratch Raytracer_lw::Scratch::make(const Grid& grid)
{
    const int ncol = grid.ncol();
    const auto no_init = Kokkos::WithoutInitializing;

    Scratch s;
    s.optics = Array_2d<Optics_cell>(
            Kokkos::view_alloc("rt_lw_optics", no_init), grid.nz, ncol);
    s.k_null = Array_3d<TF>(Kokkos::view_alloc("rt_lw_k_null", no_init),
                            grid.kn_z, grid.kn_y, grid.kn_x);

    s.emission = Array_2d<TF>(Kokkos::view_alloc("rt_lw_emission", no_init),
                              grid.nz + 2, ncol);
    s.cdf = Array_1d<double>(Kokkos::view_alloc("rt_lw_cdf", no_init),
                             std::size_t(grid.nz + 2)*ncol);

    s.tod_dn = Array_1d<TF>("rt_lw_tod_dn_count", ncol);
    s.tod_up = Array_1d<TF>("rt_lw_tod_up_count", ncol);
    s.sfc_dn = Array_1d<TF>("rt_lw_sfc_dn_count", ncol);
    s.sfc_up = Array_1d<TF>("rt_lw_sfc_up_count", ncol);
    s.atmos = Array_2d<TF>("rt_lw_atmos_count", grid.nz, ncol);

    return s;
}


void Raytracer_lw::trace_rays(
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
        const Array_map_2d<const TF>& lay_source,
        const Array_map_1d<const TF>& sfc_source,
        const Array_map_1d<const TF>& sfc_emis,
        const TF inc_dif,
        const Fluxes_lw& fluxes,
        const Scratch& scratch)
{
    const int nlay = static_cast<int>(tau_gas.extent(0));
    const int ncol = grid.ncol();
    const int nslot = (grid.nz + 2)*ncol;

    Raytracer::bundle_optics(grid, top_at_1, nlay, tau_gas, ssa_gas,
                             tau_cld, ssa_cld, asy_cld, scratch.optics);
    Raytracer::bundle_optics_tod(grid, top_at_1, nlay, tau_gas, ssa_gas,
                                 tau_cld, ssa_cld, asy_cld, scratch.optics);

    const auto optics = Array_map_2d<const Optics_cell>(scratch.optics.data(), grid.nz, ncol);

    Raytracer::create_knull_grid(grid, optics, scratch.k_null);

    bundle_emission(grid, top_at_1, nlay, tau_gas, ssa_gas, tau_cld, ssa_cld,
                    lay_source, sfc_source, sfc_emis, inc_dif, scratch.emission);

    // The running total of the emitted power, which is what a launch samples. In
    // double whatever TF is, and its last element is the total, so nothing about the
    // source distribution ever reaches the host.
    const auto emission = Array_map_1d<const TF>(scratch.emission.data(), nslot);
    const auto cdf = scratch.cdf;

    Kokkos::parallel_scan("rt_lw_emission_cdf",
        Kokkos::RangePolicy<Default_exec>(0, nslot),
        KOKKOS_LAMBDA(const int i, double& running, const bool final)
        {
            running += emission(i);
            if (final)
                cdf(i) = running;
        });

    Kokkos::deep_copy(scratch.tod_dn, TF(0.));
    Kokkos::deep_copy(scratch.tod_up, TF(0.));
    Kokkos::deep_copy(scratch.sfc_dn, TF(0.));
    Kokkos::deep_copy(scratch.sfc_up, TF(0.));
    Kokkos::deep_copy(scratch.atmos, TF(0.));

    Rt_lw_kernels::Scene scene;
    scene.optics = optics;
    scene.k_null = Array_map_3d<const TF>(
            scratch.k_null.data(), grid.kn_z, grid.kn_y, grid.kn_x);
    scene.sfc_emis = sfc_emis;

    scene.cdf = Array_map_1d<const double>(scratch.cdf.data(), nslot);
    scene.nslot = nslot;

    scene.tod_dn = Array_map_1d<TF>(scratch.tod_dn.data(), ncol);
    scene.tod_up = Array_map_1d<TF>(scratch.tod_up.data(), ncol);
    scene.sfc_dn = Array_map_1d<TF>(scratch.sfc_dn.data(), ncol);
    scene.sfc_up = Array_map_1d<TF>(scratch.sfc_up.data(), ncol);
    scene.atmos = Array_map_2d<TF>(scratch.atmos.data(), grid.nz, ncol);

    scene.grid_cells = grid.cells();
    scene.grid_d = grid.d();
    scene.grid_size = grid.size();
    scene.kn_grid = grid.kn_cells();
    scene.kn_grid_d = Vector<TF>{scene.grid_size.x/grid.kn_x,
                                 scene.grid_size.y/grid.kn_y,
                                 scene.grid_size.z/grid.kn_z};

    // The walk indexes cells by multiplying with these rather than dividing.
    scene.grid_d_inv = Vector<TF>{TF(1.)/scene.grid_d.x,
                                  TF(1.)/scene.grid_d.y,
                                  TF(1.)/scene.grid_d.z};
    scene.kn_grid_d_inv = Vector<TF>{TF(1.)/scene.kn_grid_d.x,
                                     TF(1.)/scene.kn_grid_d.y,
                                     TF(1.)/scene.kn_grid_d.z};

    // Photons are spread over the emitters rather than over the pixels, so the count
    // is a domain total; per pixel is how the caller thinks about it, and how the
    // shortwave tracer takes it.
    const long long photons_total =
            static_cast<long long>(photons_per_pixel)*grid.nx*grid.ny;

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

    count_to_flux(grid, static_cast<double>(photons_total), scratch, fluxes);
}
