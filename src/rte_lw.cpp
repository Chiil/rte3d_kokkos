#include "rte_lw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::pi;
using Rte_kernels::adding;


Rte_lw::Noscat_scratch Rte_lw::Noscat_scratch::make(
        const int nlay, const int ncol,
        const Array_1d<const TF>& weights,
        const bool do_jacobians)
{
    const auto no_init = Kokkos::WithoutInitializing;
    const int nlev = nlay + 1;

    Noscat_scratch s;

    s.trans = Array_2d<TF>(Kokkos::view_alloc("trans", no_init), nlay, ncol);
    s.source_up = Array_2d<TF>(Kokkos::view_alloc("source_up", no_init), nlay, ncol);
    s.source_dn = Array_2d<TF>(Kokkos::view_alloc("source_dn", no_init), nlay, ncol);
    s.rad_up = Array_2d<TF>(Kokkos::view_alloc("rad_up", no_init), nlev, ncol);
    s.rad_dn = Array_2d<TF>(Kokkos::view_alloc("rad_dn", no_init), nlev, ncol);
    s.rad_up_jac = Array_2d<TF>(
            Kokkos::view_alloc("rad_up_jac", no_init), do_jacobians ? nlev : 0, do_jacobians ? ncol : 0);

    s.weights_h = decltype(s.weights_h)(
            Kokkos::view_alloc("weights_h", no_init), weights.extent(0));
    Kokkos::deep_copy(s.weights_h, weights);

    return s;
}


Rte_lw::Two_stream_scratch Rte_lw::Two_stream_scratch::make(const int nlay, const int ncol)
{
    const auto no_init = Kokkos::WithoutInitializing;
    const int nlev = nlay + 1;

    Two_stream_scratch s;

    s.Rdif = Array_2d<TF>(Kokkos::view_alloc("Rdif", no_init), nlay, ncol);
    s.Tdif = Array_2d<TF>(Kokkos::view_alloc("Tdif", no_init), nlay, ncol);
    s.source_up = Array_2d<TF>(Kokkos::view_alloc("source_up", no_init), nlay, ncol);
    s.source_dn = Array_2d<TF>(Kokkos::view_alloc("source_dn", no_init), nlay, ncol);
    s.albedo = Array_2d<TF>(Kokkos::view_alloc("albedo", no_init), nlev, ncol);
    s.src = Array_2d<TF>(Kokkos::view_alloc("src", no_init), nlev, ncol);
    s.denom = Array_2d<TF>(Kokkos::view_alloc("denom", no_init), nlay, ncol);
    s.albedo_sfc = Array_1d<TF>(Kokkos::view_alloc("albedo_sfc", no_init), ncol);
    s.src_sfc = Array_1d<TF>(Kokkos::view_alloc("src_sfc", no_init), ncol);

    return s;
}


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
            const Flux_sink& flux_up,
            const Flux_sink& flux_dn,
            const Array_2d<TF>& flux_up_jac,
            const Rte_lw::Noscat_scratch& scratch)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));
        const int nlev = nlay + 1;
        const int nmus = static_cast<int>(weights.extent(0));

        const bool do_jacobians = flux_up_jac.size() > 0;
        const bool has_inc_flux = inc_flux.size() > 0;

        const Array_map_2d<const TF> lay_source = sources.lay_source;
        const Array_map_2d<const TF> lev_source = sources.lev_source;
        const Array_map_1d<const TF> sfc_source = sources.sfc_source;
        const Array_map_1d<const TF> sfc_source_jac = sources.sfc_source_jac;

        // Per-column scratch, the same shapes the reference keeps.
        const Array_2d<TF> trans = scratch.trans;
        const Array_2d<TF> source_up = scratch.source_up;
        const Array_2d<TF> source_dn = scratch.source_dn;
        const Array_2d<TF> rad_up = scratch.rad_up;
        const Array_2d<TF> rad_dn = scratch.rad_dn;
        const Array_2d<TF> rad_up_jac = scratch.rad_up_jac;

        // The weights are needed on the host to scale each angle's contribution; the
        // caller copied them there once when it built the scratch.
        const auto weights_h = scratch.weights_h;

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
                        rad_dn(lev_toa, icol) = has_inc_flux ? inc_flux(icol) / scaling : TF(0.);

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

            // Convert intensity back to flux, assuming azimuthal isotropy, and hand
            // this angle's contribution to the sink -- a g-point flux for a caller
            // that wants one, the running spectral totals for a whole-spectrum solve,
            // which is a full write and read back saved per g-point.
            //
            // The first angle assigns to a g-point flux rather than accumulating,
            // which is what zeroing it up front would have bought -- two full-size
            // memsets per g-point. The Jacobian is spectrally integrated and so keeps
            // accumulating; the caller zeroes that one.
            const bool later_mu = imu > 0;

            parallel_for_2d("lw_noscat_accumulate", {0, 0}, {nlev, ncol},
                KOKKOS_LAMBDA(const int ilev, const int icol)
                {
                    flux_up.put(ilev, icol, scaling * rad_up(ilev, icol), later_mu);
                    flux_dn.put(ilev, icol, scaling * rad_dn(ilev, icol), later_mu);

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
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Array_2d<TF>& flux_up_jac,
        const Noscat_scratch& scratch)
{
    if (top_at_1)
        solver_noscat_impl<true>(
                secants, weights, tau, sources, sfc_emis, inc_flux,
                flux_up, flux_dn, flux_up_jac, scratch);
    else
        solver_noscat_impl<false>(
                secants, weights, tau, sources, sfc_emis, inc_flux,
                flux_up, flux_dn, flux_up_jac, scratch);
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
            const Flux_sink& flux_up,
            const Flux_sink& flux_dn,
            const Rte_lw::Two_stream_scratch& scratch)
    {
        using V = Vert<top_at_1>;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));

        const Array_map_2d<const TF> lev_source = sources.lev_source;
        const Array_map_1d<const TF> sfc_source = sources.sfc_source;

        const Array_2d<TF> Rdif = scratch.Rdif;
        const Array_2d<TF> Tdif = scratch.Tdif;
        const Array_2d<TF> source_up = scratch.source_up;
        const Array_2d<TF> source_dn = scratch.source_dn;
        const Array_2d<TF> albedo = scratch.albedo;
        const Array_2d<TF> src = scratch.src;
        const Array_2d<TF> denom = scratch.denom;
        const Array_1d<TF> albedo_sfc = scratch.albedo_sfc;
        const Array_1d<TF> src_sfc = scratch.src_sfc;

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
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Two_stream_scratch& scratch)
{
    if (top_at_1)
        solver_2stream_impl<true>(
                tau, ssa, g, sources, sfc_emis, inc_flux, flux_up, flux_dn, scratch);
    else
        solver_2stream_impl<false>(
                tau, ssa, g, sources, sfc_emis, inc_flux, flux_up, flux_dn, scratch);
}
