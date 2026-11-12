#include "rte_lw.h"
#include "rte_solver_kernels.h"

using Rte_kernels::Vert;
using Rte_kernels::pi;
using Rte_kernels::adding;


Rte_lw::Noscat_scratch Rte_lw::Noscat_scratch::make(
        const int ncol, const Array_1d<const TF>& weights)
{
    const auto no_init = Kokkos::WithoutInitializing;

    Noscat_scratch s;

    s.rad_dn_sfc = Array_1d<TF>(Kokkos::view_alloc("rad_dn_sfc", no_init), ncol);
    s.carry = Array_2d<TF>(Kokkos::view_alloc("carry", no_init), 2, ncol);

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
    s.carry = Array_2d<TF>(Kokkos::view_alloc("carry", no_init), 2, ncol);
    s.albedo = Array_2d<TF>(Kokkos::view_alloc("albedo", no_init), nlev, ncol);
    s.src = Array_2d<TF>(Kokkos::view_alloc("src", no_init), nlev, ncol);
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
        const int nmus = static_cast<int>(weights.extent(0));

        const bool do_jacobians = flux_up_jac.size() > 0;
        const bool has_inc_flux = inc_flux.size() > 0;

        const Array_map_2d<const TF> lay_source = sources.lay_source;
        const Array_map_2d<const TF> lev_source = sources.lev_source;
        const Array_map_1d<const TF> sfc_source = sources.sfc_source;
        const Array_map_1d<const TF> sfc_source_jac = sources.sfc_source_jac;

        const Array_1d<TF> rad_dn_sfc = scratch.rad_dn_sfc;
        const Array_2d<TF> carry = scratch.carry;

        // The weights are needed on the host to scale each angle's contribution; the
        // caller copied them there once when it built the scratch.
        const auto weights_h = scratch.weights_h;

        const int lev_toa = V::lev_toa(nlay);
        const int lev_sfc = V::lev_sfc(nlay);

        // Each quadrature angle is an independent single-angle solve; the reference
        // sums them, and so do we.
        //
        // The whole solve is the two sweeps. Each computes the transmission and the
        // layer source it needs as it goes, and hands its radiance, converted back to
        // flux assuming azimuthal isotropy, straight to the sink -- a g-point flux for
        // a caller that wants one, the running spectral totals for a whole-spectrum
        // solve. A separate source kernel and a separate accumulation kernel would
        // write five (nlay, ncol) arrays for the sweeps to read back, and at the
        // column counts this runs at the solver is bound by exactly that traffic: SP
        // all-sky on an A4500 at 16384 columns, the source and accumulation kernels
        // took 27 of the solver's 44 ms. The up sweep repeats the down sweep's
        // exponential instead of reading it, which is cheaper than storing it.
        //
        // It is a trade towards large column counts in single precision, which is
        // what production runs at: RCEMIP longwave at 65536 columns 1278 -> 1022 ms.
        // The sweeps run only ncol threads, so the arithmetic moved into them costs
        // where there are too few columns to hide it or where it is expensive: SP
        // all-sky at 1024 columns 17 -> 26 ms, DP twice as slow there, and the CPU
        // builds 8-19% slower at 1024 columns. Keeping the source kernel and moving
        // only the accumulation into the sweeps gains half as much at 65536 columns
        // and still costs 38% at 1024.
        //
        // The first angle assigns to a g-point flux rather than accumulating, which
        // is what zeroing it up front would have bought -- two full-size memsets per
        // g-point. The Jacobian is spectrally integrated and so keeps accumulating;
        // the caller zeroes that one.
        for (int imu=0; imu<nmus; ++imu)
        {
            const TF scaling = pi * weights_h(imu);
            const bool later_mu = imu > 0;

            // Transport down, from the top of the atmosphere. Transport is for
            // intensity, so the incident flux is converted assuming azimuthal
            // isotropy; that boundary condition rides along at j == 0 rather than
            // costing a parallel region of its own.
            parallel_for_column_sweep_carry("lw_noscat_transport_dn", nlay, ncol, carry,
                KOKKOS_LAMBDA(const int j, const int icol, TF carried[2])
                {
                    if (j == 0)
                    {
                        const TF rad_toa = has_inc_flux ? inc_flux(icol) / scaling : TF(0.);

                        flux_dn.put(lev_toa, icol, scaling * rad_toa, later_mu);
                        carried[0] = rad_toa;
                    }

                    const int ilay = V::lay_from_toa(j, nlay);
                    TF trans, source_up, source_dn;
                    Rte_kernels::lw_layer_noscat(
                            tau(ilay, icol), secants(imu, icol), lay_source(ilay, icol),
                            lev_source(ilay + V::lev_up(), icol),
                            lev_source(ilay + V::lev_dn(), icol),
                            trans, source_up, source_dn);

                    const TF rad = trans * carried[0] + source_dn;

                    flux_dn.put(ilay + V::lev_dn(), icol, scaling * rad, later_mu);
                    carried[0] = rad;

                    // The surface boundary condition of the up sweep.
                    if (j == nlay-1)
                        rad_dn_sfc(icol) = rad;
                });

            // Transport up, starting from surface reflection and emission.
            parallel_for_column_sweep_carry("lw_noscat_transport_up", nlay, ncol, carry,
                KOKKOS_LAMBDA(const int j, const int icol, TF carried[2])
                {
                    if (j == 0)
                    {
                        const TF emis = sfc_emis(icol);

                        const TF rad_sfc = rad_dn_sfc(icol) * (TF(1.) - emis)
                                         + emis * sfc_source(icol);

                        flux_up.put(lev_sfc, icol, scaling * rad_sfc, later_mu);
                        carried[0] = rad_sfc;

                        if (do_jacobians)
                        {
                            const TF jac_sfc = emis * sfc_source_jac(icol);

                            flux_up_jac(lev_sfc, icol) += scaling * jac_sfc;
                            carried[1] = jac_sfc;
                        }
                    }

                    const int ilay = V::lay_from_sfc(j, nlay);
                    const int lev_up = ilay + V::lev_up();
                    TF trans, source_up, source_dn;
                    Rte_kernels::lw_layer_noscat(
                            tau(ilay, icol), secants(imu, icol), lay_source(ilay, icol),
                            lev_source(ilay + V::lev_up(), icol),
                            lev_source(ilay + V::lev_dn(), icol),
                            trans, source_up, source_dn);

                    const TF rad = trans * carried[0] + source_up;

                    flux_up.put(lev_up, icol, scaling * rad, later_mu);
                    carried[0] = rad;

                    if (do_jacobians)
                    {
                        const TF jac = trans * carried[1];

                        flux_up_jac(lev_up, icol) += scaling * jac;
                        carried[1] = jac;
                    }
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
        const Array_1d<TF> albedo_sfc = scratch.albedo_sfc;
        const Array_1d<TF> src_sfc = scratch.src_sfc;

        // Cell properties, and the source function for diffuse radiation. Layers are
        // independent here, so this whole part is parallel over (layer, column).
        parallel_for_2d("lw_2stream_cell", {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol)
            {
                const TF tau_l = tau(ilay, icol);

                TF Rdif_l, Tdif_l, absorptance, source_gradient;
                Rte_kernels::lw_two_stream(
                        tau_l, ssa(ilay, icol), g(ilay, icol),
                        Rdif_l, Tdif_l, absorptance, source_gradient);

                Rdif(ilay, icol) = Rdif_l;
                Tdif(ilay, icol) = Tdif_l;

                TF source_up_l, source_dn_l;
                Rte_kernels::lw_source_2str(
                        lev_source(ilay + V::lev_up(), icol),
                        lev_source(ilay + V::lev_dn(), icol),
                        absorptance, source_gradient,
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
                albedo, src, scratch.carry);
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
