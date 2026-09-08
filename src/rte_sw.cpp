#include "rte_sw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::adding;


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
            const Array_map_2d<TF>& flux_up,
            const Array_map_2d<TF>& flux_dn,
            const Array_map_2d<TF>& flux_dir)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));
        const int nlev = nlay + 1;

        // One g-point's worth of scratch: (nlay, ncol) whatever the spectral
        // resolution, which is what makes the solver fit on a GPU.
        Array_2d<TF> Rdif(Kokkos::view_alloc("Rdif", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> Tdif(Kokkos::view_alloc("Tdif", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> Rdir(Kokkos::view_alloc("Rdir", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> Tdir(Kokkos::view_alloc("Tdir", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> Tnoscat(Kokkos::view_alloc("Tnoscat", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> source_up(Kokkos::view_alloc("source_up", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> source_dn(Kokkos::view_alloc("source_dn", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> albedo(Kokkos::view_alloc("albedo", Kokkos::WithoutInitializing), nlev, ncol);
        Array_2d<TF> src(Kokkos::view_alloc("src", Kokkos::WithoutInitializing), nlev, ncol);
        Array_2d<TF> denom(Kokkos::view_alloc("denom", Kokkos::WithoutInitializing), nlay, ncol);
        Array_1d<TF> src_sfc(Kokkos::view_alloc("src_sfc", Kokkos::WithoutInitializing), ncol);

        // Cell properties. Layers are independent, so this is parallel over
        // (layer, column); only the direct beam below needs a sweep.
        parallel_for_2d("sw_2stream_cell", {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol)
            {
                TF Rdif_l, Tdif_l, Rdir_l, Tdir_l, Tnoscat_l;
                Rte_kernels::sw_two_stream(
                        tau(ilay, icol), ssa(ilay, icol), g(ilay, icol), mu0(ilay, icol),
                        Rdif_l, Tdif_l, Rdir_l, Tdir_l, Tnoscat_l);

                Rdif(ilay, icol) = Rdif_l;
                Tdif(ilay, icol) = Tdif_l;
                Rdir(ilay, icol) = Rdir_l;
                Tdir(ilay, icol) = Tdir_l;
                Tnoscat(ilay, icol) = Tnoscat_l;
            });

        const int lev_toa = V::lev_toa(nlay);
        const int lay_toa = V::lay_from_toa(0, nlay);
        const int lev_sfc = V::lev_sfc(nlay);
        const int lay_sfc = V::lay_from_sfc(0, nlay);

        // The direct beam attenuating downward, and the diffuse sources it feeds. The
        // beam's boundary condition at the top and the surface source it leaves behind
        // ride along at the ends of the sweep, saving two parallel regions.
        parallel_for_column_sweep("sw_2stream_direct", nlay, ncol,
            KOKKOS_LAMBDA(const int j, const int icol)
            {
                if (j == 0)
                    flux_dir(lev_toa, icol) = inc_flux_dir(icol) * mu0(lay_toa, icol);

                const int ilay = V::lay_from_toa(j, nlay);
                const TF dir_inc = flux_dir(ilay + V::lev_up(), icol);

                // T and R for the direct beam were computed with a nominal mu0 even
                // where the sun is below the horizon; zero those out again.
                const bool sunlit = mu0(ilay, icol) > TF(0.);
                source_up(ilay, icol) = sunlit ? Rdir(ilay, icol) * dir_inc : TF(0.);
                source_dn(ilay, icol) = sunlit ? Tdir(ilay, icol) * dir_inc : TF(0.);

                flux_dir(ilay + V::lev_dn(), icol) = Tnoscat(ilay, icol) * dir_inc;

                // Source for upward radiation at the surface, now that the beam has
                // reached it.
                if (j == nlay-1)
                    src_sfc(icol) = mu0(lay_sfc, icol) > TF(0.)
                            ? flux_dir(lev_sfc, icol) * sfc_alb_dir(icol)
                            : TF(0.);
            });

        adding<top_at_1>(
                nlay, ncol,
                sfc_alb_dif, src_sfc, inc_flux_dif,
                Rdif, Tdif, source_dn, source_up,
                flux_up, flux_dn,
                albedo, src, denom);

        // adding() computes only the diffuse flux; flux_dn is the total.
        parallel_for_2d("sw_2stream_total", {0, 0}, {nlev, ncol},
            KOKKOS_LAMBDA(const int ilev, const int icol)
            {
                flux_dn(ilev, icol) += flux_dir(ilev, icol);
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
        const Array_map_2d<TF>& flux_up,
        const Array_map_2d<TF>& flux_dn,
        const Array_map_2d<TF>& flux_dir)
{
    if (top_at_1)
        solver_2stream_impl<true>(
                tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
                flux_up, flux_dn, flux_dir);
    else
        solver_2stream_impl<false>(
                tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
                flux_up, flux_dn, flux_dir);
}
