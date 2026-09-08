#include "fluxes.h"


void Fluxes::sum_broadband(
        const Array_3d<const TF>& spectral_flux,
        const Array_2d<TF>& broadband_flux)
{
    const int ngpt = static_cast<int>(spectral_flux.extent(0));

    parallel_for_2d("sum_broadband", {0, 0},
        {static_cast<int64_t>(broadband_flux.extent(0)), static_cast<int64_t>(broadband_flux.extent(1))},
        KOKKOS_LAMBDA(const int ilev, const int icol)
        {
            TF sum = TF(0.);
            for (int igpt=0; igpt<ngpt; ++igpt)
                sum += spectral_flux(igpt, ilev, icol);

            broadband_flux(ilev, icol) = sum;
        });
}


void Fluxes::net_broadband(
        const Array_3d<const TF>& spectral_flux_dn,
        const Array_3d<const TF>& spectral_flux_up,
        const Array_2d<TF>& broadband_flux_net)
{
    const int ngpt = static_cast<int>(spectral_flux_dn.extent(0));

    parallel_for_2d("net_broadband_full", {0, 0},
        {static_cast<int64_t>(broadband_flux_net.extent(0)), static_cast<int64_t>(broadband_flux_net.extent(1))},
        KOKKOS_LAMBDA(const int ilev, const int icol)
        {
            TF sum = TF(0.);
            for (int igpt=0; igpt<ngpt; ++igpt)
                sum += spectral_flux_dn(igpt, ilev, icol) - spectral_flux_up(igpt, ilev, icol);

            broadband_flux_net(ilev, icol) = sum;
        });
}


void Fluxes::net_broadband(
        const Array_2d<const TF>& flux_dn,
        const Array_2d<const TF>& flux_up,
        const Array_2d<TF>& broadband_flux_net)
{
    parallel_for_2d("net_broadband_precalc", {0, 0},
        {static_cast<int64_t>(broadband_flux_net.extent(0)), static_cast<int64_t>(broadband_flux_net.extent(1))},
        KOKKOS_LAMBDA(const int ilev, const int icol)
        {
            broadband_flux_net(ilev, icol) = flux_dn(ilev, icol) - flux_up(ilev, icol);
        });
}


void Fluxes::sum_byband(
        const Array_2d<const int>& band_lims,
        const Array_3d<const TF>& spectral_flux,
        const Array_3d<TF>& byband_flux)
{
    parallel_for_3d("sum_byband", {0, 0, 0},
        {static_cast<int64_t>(byband_flux.extent(0)), static_cast<int64_t>(byband_flux.extent(1)), static_cast<int64_t>(byband_flux.extent(2))},
        KOKKOS_LAMBDA(const int ibnd, const int ilev, const int icol)
        {
            TF sum = TF(0.);
            for (int igpt=band_lims(ibnd, 0); igpt<=band_lims(ibnd, 1); ++igpt)
                sum += spectral_flux(igpt, ilev, icol);

            byband_flux(ibnd, ilev, icol) = sum;
        });
}


void Fluxes::net_byband(
        const Array_2d<const int>& band_lims,
        const Array_3d<const TF>& spectral_flux_dn,
        const Array_3d<const TF>& spectral_flux_up,
        const Array_3d<TF>& byband_flux_net)
{
    parallel_for_3d("net_byband_full", {0, 0, 0},
        {static_cast<int64_t>(byband_flux_net.extent(0)), static_cast<int64_t>(byband_flux_net.extent(1)), static_cast<int64_t>(byband_flux_net.extent(2))},
        KOKKOS_LAMBDA(const int ibnd, const int ilev, const int icol)
        {
            TF sum = TF(0.);
            for (int igpt=band_lims(ibnd, 0); igpt<=band_lims(ibnd, 1); ++igpt)
                sum += spectral_flux_dn(igpt, ilev, icol) - spectral_flux_up(igpt, ilev, icol);

            byband_flux_net(ibnd, ilev, icol) = sum;
        });
}
