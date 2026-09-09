#include "rte_sw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::adding;


Rte_sw::Two_stream_scratch Rte_sw::Two_stream_scratch::make(const int nlay, const int ncol)
{
    const auto no_init = Kokkos::WithoutInitializing;
    const int nlev = nlay + 1;

    Two_stream_scratch s;

    s.Rdif = Array_2d<TF>(Kokkos::view_alloc("Rdif", no_init), nlay, ncol);
    s.Tdif = Array_2d<TF>(Kokkos::view_alloc("Tdif", no_init), nlay, ncol);
    s.source_up = Array_2d<TF>(Kokkos::view_alloc("source_up", no_init), nlay, ncol);
    s.source_dn = Array_2d<TF>(Kokkos::view_alloc("source_dn", no_init), nlay, ncol);
    s.carry = Array_2d<TF>(Kokkos::view_alloc("carry", no_init), 2, ncol);
    s.albedo = Array_2d<TF>(Kokkos::view_alloc("albedo", no_init), nlev, ncol);
    s.src = Array_2d<TF>(Kokkos::view_alloc("src", no_init), nlev, ncol);
    s.src_sfc = Array_1d<TF>(Kokkos::view_alloc("src_sfc", no_init), ncol);

    return s;
}


namespace
{
    template<bool top_at_1>
    void solver_noscat_impl(
            const Array_map_2d<const TF>& tau,
            const Array_map_2d<const TF>& mu0,
            const Array_map_1d<const TF>& inc_flux_dir,
            const Array_map_2d<TF>& flux_dir)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));

        const int lev_toa = V::lev_toa(nlay);
        const int lay_toa = V::lay_from_toa(0, nlay);

        parallel_for_column_sweep("sw_noscat_transport", nlay, ncol,
            KOKKOS_LAMBDA(const int j, const int icol)
            {
                if (j == 0)
                    flux_dir(lev_toa, icol) = inc_flux_dir(icol) * mu0(lay_toa, icol);

                const int ilay = V::lay_from_toa(j, nlay);
                flux_dir(ilay + V::lev_dn(), icol) =
                        flux_dir(ilay + V::lev_up(), icol)
                        * Kokkos::exp(-tau(ilay, icol) / mu0(ilay, icol));
            });
    }
}


namespace
{
    template<bool top_at_1>
    void solver_2stream_impl(
            const Array_map_2d<const TF>& tau,
            const Array_map_2d<const TF>& ssa,
            const Array_map_2d<const TF>& g,
            const Array_map_2d<const TF>& mu0,
            const Array_map_1d<const TF>& sfc_alb_dir,
            const Array_map_1d<const TF>& sfc_alb_dif,
            const Array_map_1d<const TF>& inc_flux_dir,
            const Array_map_1d<const TF>& inc_flux_dif,
            const Flux_sink& flux_up,
            const Flux_sink& flux_dn,
            const Flux_sink& flux_dir,
            const Rte_sw::Two_stream_scratch& scratch)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));
        const int nlev = nlay + 1;

        // One g-point's worth of scratch: (nlay, ncol) whatever the spectral
        // resolution, which is what makes the solver fit on a GPU.
        const Array_2d<TF> Rdif = scratch.Rdif;
        const Array_2d<TF> Tdif = scratch.Tdif;
        const Array_2d<TF> source_up = scratch.source_up;
        const Array_2d<TF> source_dn = scratch.source_dn;
        const Array_2d<TF> albedo = scratch.albedo;
        const Array_2d<TF> src = scratch.src;
        const Array_1d<TF> src_sfc = scratch.src_sfc;

        const int lev_toa = V::lev_toa(nlay);
        const int lay_toa = V::lay_from_toa(0, nlay);
        const int lev_sfc = V::lev_sfc(nlay);
        const int lay_sfc = V::lay_from_sfc(0, nlay);

        // An empty g is an isotropic phase function -- a clear-sky solve, where the
        // asymmetry parameter is zero everywhere. Writing that array and reading it
        // back is two full passes over (nlay, ncol) per g-point, and the sweeps are
        // bound by exactly that.
        const bool has_g = g.size() > 0;

        // The direct beam attenuates level by level; the sweep keeps that in its
        // carry, and the array is what the spectral totals are read from at the end.
        const Array_map_2d<TF> dir = flux_dir.gpt;

        // The layer's two-stream coefficients and the direct beam attenuating downward
        // through them, in one pass. The cell properties have no vertical dependence,
        // so they could be a parallel region of their own -- but only Rdif and Tdif
        // outlive the layer, and computing them here keeps Rdir, Tdir and Tnoscat in
        // registers instead of writing and reading back three (nlay, ncol) arrays.
        // The sweep is bound by bandwidth, not arithmetic, so the extra work is free.
        // Taken from rte-rrtmgp-cpp's sw_source_2stream_kernel.
        //
        // The beam's boundary condition at the top and the surface source it leaves
        // behind ride along at the ends of the sweep, saving two parallel regions.
        parallel_for_column_sweep_carry("sw_2stream_direct", nlay, ncol, scratch.carry,
            KOKKOS_LAMBDA(const int j, const int icol, TF carried[2])
            {
                if (j == 0)
                {
                    const TF dir_toa = inc_flux_dir(icol) * mu0(lay_toa, icol);

                    dir(lev_toa, icol) = dir_toa;
                    carried[0] = dir_toa;
                }

                const int ilay = V::lay_from_toa(j, nlay);
                const TF dir_inc = carried[0];

                TF Rdif_l, Tdif_l, Rdir_l, Tdir_l, Tnoscat_l;
                Rte_kernels::sw_two_stream(
                        tau(ilay, icol), ssa(ilay, icol),
                        has_g ? g(ilay, icol) : TF(0.), mu0(ilay, icol),
                        Rdif_l, Tdif_l, Rdir_l, Tdir_l, Tnoscat_l);

                Rdif(ilay, icol) = Rdif_l;
                Tdif(ilay, icol) = Tdif_l;

                // T and R for the direct beam were computed with a nominal mu0 even
                // where the sun is below the horizon; zero those out again.
                const bool sunlit = mu0(ilay, icol) > TF(0.);
                source_up(ilay, icol) = sunlit ? Rdir_l * dir_inc : TF(0.);
                source_dn(ilay, icol) = sunlit ? Tdir_l * dir_inc : TF(0.);

                const TF dir_out = Tnoscat_l * dir_inc;

                dir(ilay + V::lev_dn(), icol) = dir_out;
                carried[0] = dir_out;

                // Source for upward radiation at the surface, now that the beam has
                // reached it.
                if (j == nlay-1)
                    src_sfc(icol) = mu0(lay_sfc, icol) > TF(0.)
                            ? dir_out * sfc_alb_dir(icol)
                            : TF(0.);
            });

        adding<top_at_1>(
                nlay, ncol,
                sfc_alb_dif, src_sfc, inc_flux_dif,
                Rdif, Tdif, source_dn, source_up,
                flux_up, flux_dn,
                albedo, src, scratch.carry);

        // adding() computes only the diffuse flux; flux_dn is the total. This is also
        // where the direct beam reaches the spectral totals, its g-point array having
        // held it since the sweep above.
        parallel_for_2d("sw_2stream_total", {0, 0}, {nlev, ncol},
            KOKKOS_LAMBDA(const int ilev, const int icol)
            {
                const TF dir_l = dir(ilev, icol);

                flux_dn.put(ilev, icol, dir_l, true);
                flux_dir.add(ilev, icol, dir_l);
            });
    }
}


void Rte_sw::solver_noscat(
        const bool top_at_1,
        const Array_map_2d<const TF>& tau,
        const Array_map_2d<const TF>& mu0,
        const Array_map_1d<const TF>& inc_flux_dir,
        const Array_map_2d<TF>& flux_dir)
{
    if (top_at_1)
        solver_noscat_impl<true>(tau, mu0, inc_flux_dir, flux_dir);
    else
        solver_noscat_impl<false>(tau, mu0, inc_flux_dir, flux_dir);
}


void Rte_sw::solver_2stream(
        const bool top_at_1,
        const Array_map_2d<const TF>& tau,
        const Array_map_2d<const TF>& ssa,
        const Array_map_2d<const TF>& g,
        const Array_map_2d<const TF>& mu0,
        const Array_map_1d<const TF>& sfc_alb_dir,
        const Array_map_1d<const TF>& sfc_alb_dif,
        const Array_map_1d<const TF>& inc_flux_dir,
        const Array_map_1d<const TF>& inc_flux_dif,
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Flux_sink& flux_dir,
        const Two_stream_scratch& scratch)
{
    if (top_at_1)
        solver_2stream_impl<true>(
                tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
                flux_up, flux_dn, flux_dir, scratch);
    else
        solver_2stream_impl<false>(
                tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
                flux_up, flux_dn, flux_dir, scratch);
}
