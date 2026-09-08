#include "rte_lw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::pi;
using Rte_kernels::adding;


// The reference's lw_solver_noscat also offers do_rescaling, the approximate treatment
// of scattering of Tang et al. 2018 (10.1175/JAS-D-18-0014.1), through
// lw_transport_1rescl. That is not implemented here. Its two orientation branches read
// the upward radiance at different levels relative to the layer during the second
// downward sweep -- the top-at-1 branch at the upstream level, the other at the
// destination level -- and that asymmetry should be resolved against the reference
// authors before it is reproduced.
namespace
{
    template<bool top_at_1>
    void solver_noscat_impl(
            const Array_map_2d<const TF>& secants,
            const Array_1d<const TF>& weights,
            const Array_map_2d<const TF>& tau,
            const Source_func_lw& sources,
            const Array_map_1d<const TF>& sfc_emis,
            const Array_map_1d<const TF>& inc_flux,
            const Array_map_2d<TF>& flux_up,
            const Array_map_2d<TF>& flux_dn,
            const Array_2d<TF>& flux_up_jac)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));
        const int nlev = nlay + 1;
        const int nmus = static_cast<int>(weights.extent(0));

        const bool do_jacobians = flux_up_jac.size() > 0;

        const Array_map_2d<const TF> lay_source = sources.lay_source;
        const Array_map_2d<const TF> lev_source = sources.lev_source;
        const Array_map_1d<const TF> sfc_source = sources.sfc_source;
        const Array_map_1d<const TF> sfc_source_jac = sources.sfc_source_jac;

        // Per-column scratch, the same shapes the reference keeps.
        Array_2d<TF> trans(Kokkos::view_alloc("trans", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> source_up(Kokkos::view_alloc("source_up", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> source_dn(Kokkos::view_alloc("source_dn", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> rad_up(Kokkos::view_alloc("rad_up", Kokkos::WithoutInitializing), nlev, ncol);
        Array_2d<TF> rad_dn(Kokkos::view_alloc("rad_dn", Kokkos::WithoutInitializing), nlev, ncol);
        Array_2d<TF> rad_up_jac("rad_up_jac", do_jacobians ? nlev : 0, do_jacobians ? ncol : 0);

        Kokkos::deep_copy(flux_up, TF(0.));
        Kokkos::deep_copy(flux_dn, TF(0.));

        // The weights are needed on the host to scale each angle's contribution.
        const auto weights_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights);

        const int lev_toa = V::lev_toa(nlay);
        const int lev_sfc = V::lev_sfc(nlay);

        // Each quadrature angle is an independent single-angle solve; the reference
        // sums them, and so do we.
        for (int imu=0; imu<nmus; ++imu)
        {
            const TF scaling = pi * weights_h(imu);

            // Optical path, transmission and the layer source functions. No vertical
            // dependence, so this is the fully parallel part of the solver.
            parallel_for_2d("lw_noscat_source", {0, 0}, {nlay, ncol},
                KOKKOS_LAMBDA(const int ilay, const int icol)
                {
                    const TF tau_loc = tau(ilay, icol) * secants(imu, icol);
                    const TF trans_l = Kokkos::exp(-tau_loc);
                    trans(ilay, icol) = trans_l;

                    TF source_up_l, source_dn_l;
                    Rte_kernels::lw_source_noscat(
                            lay_source(ilay, icol),
                            lev_source(ilay + V::lev_up(), icol),
                            lev_source(ilay + V::lev_dn(), icol),
                            tau_loc, trans_l, source_up_l, source_dn_l);

                    source_up(ilay, icol) = source_up_l;
                    source_dn(ilay, icol) = source_dn_l;
                });

            // Transport down, from the top of the atmosphere. Transport is for
            // intensity, so the incident flux is converted assuming azimuthal
            // isotropy; that boundary condition rides along at j == 0 rather than
            // costing a parallel region of its own.
            parallel_for_column_sweep("lw_noscat_transport_dn", nlay, ncol,
                KOKKOS_LAMBDA(const int j, const int icol)
                {
                    if (j == 0)
                        rad_dn(lev_toa, icol) = inc_flux(icol) / scaling;

                    const int ilay = V::lay_from_toa(j, nlay);
                    rad_dn(ilay + V::lev_dn(), icol) =
                            trans(ilay, icol) * rad_dn(ilay + V::lev_up(), icol)
                            + source_dn(ilay, icol);
                });

            // Transport up, starting from surface reflection and emission.
            parallel_for_column_sweep("lw_noscat_transport_up", nlay, ncol,
                KOKKOS_LAMBDA(const int j, const int icol)
                {
                    if (j == 0)
                    {
                        const TF emis = sfc_emis(icol);

                        rad_up(lev_sfc, icol) = rad_dn(lev_sfc, icol) * (TF(1.) - emis)
                                              + emis * sfc_source(icol);
                        if (do_jacobians)
                            rad_up_jac(lev_sfc, icol) = emis * sfc_source_jac(icol);
                    }

                    const int ilay = V::lay_from_sfc(j, nlay);
                    const TF trans_l = trans(ilay, icol);

                    rad_up(ilay + V::lev_up(), icol) =
                            trans_l * rad_up(ilay + V::lev_dn(), icol)
                            + source_up(ilay, icol);

                    if (do_jacobians)
                        rad_up_jac(ilay + V::lev_up(), icol) =
                                trans_l * rad_up_jac(ilay + V::lev_dn(), icol);
                });

            // Convert intensity back to flux, assuming azimuthal isotropy, and
            // accumulate this angle's contribution. The Jacobian is spectrally
            // integrated, so it accumulates over g-points as well and the caller
            // zeroes it.
            parallel_for_2d("lw_noscat_accumulate", {0, 0}, {nlev, ncol},
                KOKKOS_LAMBDA(const int ilev, const int icol)
                {
                    flux_up(ilev, icol) += scaling * rad_up(ilev, icol);
                    flux_dn(ilev, icol) += scaling * rad_dn(ilev, icol);

                    if (do_jacobians)
                        flux_up_jac(ilev, icol) += scaling * rad_up_jac(ilev, icol);
                });
        }
    }
}


void Rte_lw::solver_noscat(
        const bool top_at_1,
        const Array_map_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_map_2d<const TF>& tau,
        const Source_func_lw& sources,
        const Array_map_1d<const TF>& sfc_emis,
        const Array_map_1d<const TF>& inc_flux,
        const Array_map_2d<TF>& flux_up,
        const Array_map_2d<TF>& flux_dn,
        const Array_2d<TF>& flux_up_jac)
{
    if (top_at_1)
        solver_noscat_impl<true>(
                secants, weights, tau, sources, sfc_emis, inc_flux, flux_up, flux_dn, flux_up_jac);
    else
        solver_noscat_impl<false>(
                secants, weights, tau, sources, sfc_emis, inc_flux, flux_up, flux_dn, flux_up_jac);
}


namespace
{
    template<bool top_at_1>
    void solver_2stream_impl(
            const Array_map_2d<const TF>& tau,
            const Array_map_2d<const TF>& ssa,
            const Array_map_2d<const TF>& g,
            const Source_func_lw& sources,
            const Array_map_1d<const TF>& sfc_emis,
            const Array_map_1d<const TF>& inc_flux,
            const Array_map_2d<TF>& flux_up,
            const Array_map_2d<TF>& flux_dn)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));
        const int nlev = nlay + 1;

        const Array_map_2d<const TF> lev_source = sources.lev_source;
        const Array_map_1d<const TF> sfc_source = sources.sfc_source;

        Array_2d<TF> Rdif(Kokkos::view_alloc("Rdif", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> Tdif(Kokkos::view_alloc("Tdif", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> source_up(Kokkos::view_alloc("source_up", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> source_dn(Kokkos::view_alloc("source_dn", Kokkos::WithoutInitializing), nlay, ncol);
        Array_2d<TF> albedo(Kokkos::view_alloc("albedo", Kokkos::WithoutInitializing), nlev, ncol);
        Array_2d<TF> src(Kokkos::view_alloc("src", Kokkos::WithoutInitializing), nlev, ncol);
        Array_2d<TF> denom(Kokkos::view_alloc("denom", Kokkos::WithoutInitializing), nlay, ncol);
        Array_1d<TF> albedo_sfc(Kokkos::view_alloc("albedo_sfc", Kokkos::WithoutInitializing), ncol);
        Array_1d<TF> src_sfc(Kokkos::view_alloc("src_sfc", Kokkos::WithoutInitializing), ncol);

        // Cell properties, and the source function for diffuse radiation. Layers are
        // independent here, so this whole part is parallel over (layer, column).
        parallel_for_2d("lw_2stream_cell", {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol)
            {
                const TF tau_l = tau(ilay, icol);

                TF gamma1, gamma2, Rdif_l, Tdif_l;
                Rte_kernels::lw_two_stream(
                        tau_l, ssa(ilay, icol), g(ilay, icol),
                        gamma1, gamma2, Rdif_l, Tdif_l);

                Rdif(ilay, icol) = Rdif_l;
                Tdif(ilay, icol) = Tdif_l;

                TF source_up_l, source_dn_l;
                Rte_kernels::lw_source_2str(
                        lev_source(ilay + V::lev_up(), icol),
                        lev_source(ilay + V::lev_dn(), icol),
                        gamma1, gamma2, Rdif_l, Tdif_l, tau_l,
                        source_up_l, source_dn_l);

                source_up(ilay, icol) = source_up_l;
                source_dn(ilay, icol) = source_dn_l;
            });

        parallel_for_1d("lw_2stream_boundary", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                const TF emis = sfc_emis(icol);

                albedo_sfc(icol) = TF(1.) - emis;
                src_sfc(icol) = pi * emis * sfc_source(icol);
            });

        adding<top_at_1>(
                nlay, ncol,
                albedo_sfc, src_sfc, inc_flux,
                Rdif, Tdif, source_dn, source_up,
                flux_up, flux_dn,
                albedo, src, denom);
    }
}


void Rte_lw::solver_2stream(
        const bool top_at_1,
        const Array_map_2d<const TF>& tau,
        const Array_map_2d<const TF>& ssa,
        const Array_map_2d<const TF>& g,
        const Source_func_lw& sources,
        const Array_map_1d<const TF>& sfc_emis,
        const Array_map_1d<const TF>& inc_flux,
        const Array_map_2d<TF>& flux_up,
        const Array_map_2d<TF>& flux_dn)
{
    if (top_at_1)
        solver_2stream_impl<true>(tau, ssa, g, sources, sfc_emis, inc_flux, flux_up, flux_dn);
    else
        solver_2stream_impl<false>(tau, ssa, g, sources, sfc_emis, inc_flux, flux_up, flux_dn);
}
