#include <limits>
#include <stdexcept>
#include <string>

#include "cloud_optics.h"
#include "cloud_optics_kernels.h"


namespace
{
    // Copy one roughness slice of an ice table into a device View.
    Array_2d<TF> ice_slice(const Array_3d_h<TF>& table, const int icergh, const std::string& label)
    {
        const int nspec = static_cast<int>(table.extent(1));
        const int nsize = static_cast<int>(table.extent(2));

        Array_2d_h<TF> host(label, nspec, nsize);
        for (int i=0; i<nspec; ++i)
            for (int j=0; j<nsize; ++j)
                host(i, j) = table(icergh, i, j);

        Array_2d<TF> out(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), nspec, nsize);
        Kokkos::deep_copy(out, host);

        return out;
    }


    Array_2d<TF> to_device_2d(const Array_2d_h<TF>& host)
    {
        Array_2d<TF> out(Kokkos::view_alloc(host.label(), Kokkos::WithoutInitializing),
                         host.extent(0), host.extent(1));
        Kokkos::deep_copy(out, host);

        return out;
    }
}


Cloud_optics Cloud_optics::load_lut(
        const TF radliq_lwr, const TF radliq_upr,
        const TF radice_lwr, const TF radice_upr,
        const Array_2d_h<TF>& extliq,
        const Array_2d_h<TF>& ssaliq,
        const Array_2d_h<TF>& asyliq,
        const Array_3d_h<TF>& extice,
        const Array_3d_h<TF>& ssaice,
        const Array_3d_h<TF>& asyice,
        const int icergh)
{
    const int nrghice = static_cast<int>(extice.extent(0));
    if (icergh < 0 || icergh >= nrghice)
        throw std::invalid_argument("Ice roughness index is out of range.");

    if (extice.extent(1) != extliq.extent(0))
        throw std::invalid_argument("Liquid and ice tables disagree on the spectral dimension.");

    Cloud_optics c;
    c.nroughness_types = nrghice;

    c.lut_extliq = to_device_2d(extliq);
    c.lut_ssaliq = to_device_2d(ssaliq);
    c.lut_asyliq = to_device_2d(asyliq);

    c.lut_extice = ice_slice(extice, icergh, "lut_extice");
    c.lut_ssaice = ice_slice(ssaice, icergh, "lut_ssaice");
    c.lut_asyice = ice_slice(asyice, icergh, "lut_asyice");

    c.radliq_lwr = radliq_lwr;
    c.radliq_upr = radliq_upr;
    c.radice_lwr = radice_lwr;
    c.radice_upr = radice_upr;

    const int nsize_liq = static_cast<int>(extliq.extent(1));
    const int nsize_ice = static_cast<int>(extice.extent(2));

    c.liq_step_size = (radliq_upr - radliq_lwr) / static_cast<TF>(nsize_liq - 1);
    c.ice_step_size = (radice_upr - radice_lwr) / static_cast<TF>(nsize_ice - 1);

    return c;
}


namespace
{
    // Liquid and ice contributions for one (spectral point, layer, column), each as
    // tau, tau*ssa and tau*ssa*g. A layer with no condensate contributes nothing.
    KOKKOS_INLINE_FUNCTION
    void cloud_totals(
            const Cloud_optics& c,
            const TF clwp, const TF ciwp, const TF reliq, const TF reice,
            const int ispec, const int nsize_liq, const int nsize_ice,
            TF& tau, TF& taussa, TF& taussag)
    {
        tau = TF(0.);
        taussa = TF(0.);
        taussag = TF(0.);

        if (clwp > TF(0.))
        {
            TF t, ts, tsg;
            Cloud_optics_kernels::lookup(
                    clwp, reliq, nsize_liq, c.liq_step_size, c.radliq_lwr,
                    c.lut_extliq, c.lut_ssaliq, c.lut_asyliq, ispec, t, ts, tsg);

            tau += t;
            taussa += ts;
            taussag += tsg;
        }

        if (ciwp > TF(0.))
        {
            TF t, ts, tsg;
            Cloud_optics_kernels::lookup(
                    ciwp, reice, nsize_ice, c.ice_step_size, c.radice_lwr,
                    c.lut_extice, c.lut_ssaice, c.lut_asyice, ispec, t, ts, tsg);

            tau += t;
            taussa += ts;
            taussag += tsg;
        }
    }
}


void Clouds::compute(
        const Cloud_optics& c,
        const Array_2d<const TF>& clwp,
        const Array_2d<const TF>& ciwp,
        const Array_2d<const TF>& reliq,
        const Array_2d<const TF>& reice,
        const Array_3d<TF>& tau,
        const Array_3d<TF>& ssa,
        const Array_3d<TF>& g)
{
    const int nspec = static_cast<int>(tau.extent(0));
    const int nlay = static_cast<int>(tau.extent(1));
    const int ncol = static_cast<int>(tau.extent(2));

    const int nsize_liq = static_cast<int>(c.lut_extliq.extent(1));
    const int nsize_ice = static_cast<int>(c.lut_extice.extent(1));

    constexpr TF eps = std::numeric_limits<TF>::epsilon();

    parallel_for_3d("cloud_optics_2str", {0, 0, 0}, {nspec, nlay, ncol},
        KOKKOS_LAMBDA(const int ispec, const int ilay, const int icol)
        {
            TF t, ts, tsg;
            cloud_totals(c, clwp(ilay, icol), ciwp(ilay, icol),
                         reliq(ilay, icol), reice(ilay, icol),
                         ispec, nsize_liq, nsize_ice, t, ts, tsg);

            g(ispec, ilay, icol) = tsg / Kokkos::max(eps, ts);
            ssa(ispec, ilay, icol) = ts / Kokkos::max(eps, t);
            tau(ispec, ilay, icol) = t;
        });
}


void Clouds::compute(
        const Cloud_optics& c,
        const Array_2d<const TF>& clwp,
        const Array_2d<const TF>& ciwp,
        const Array_2d<const TF>& reliq,
        const Array_2d<const TF>& reice,
        const Array_3d<TF>& tau)
{
    const int nspec = static_cast<int>(tau.extent(0));
    const int nlay = static_cast<int>(tau.extent(1));
    const int ncol = static_cast<int>(tau.extent(2));

    const int nsize_liq = static_cast<int>(c.lut_extliq.extent(1));
    const int nsize_ice = static_cast<int>(c.lut_extice.extent(1));

    parallel_for_3d("cloud_optics_1scl", {0, 0, 0}, {nspec, nlay, ncol},
        KOKKOS_LAMBDA(const int ispec, const int ilay, const int icol)
        {
            // Absorption optical depth is (1 - ssa)*tau = tau - tau*ssa, taken per
            // phase and then summed, as the reference does. Summing the phases first
            // and subtracting once loses most of the precision in the shortwave, where
            // cloud ssa is close to one and the two terms nearly cancel.
            TF absorption = TF(0.);

            if (clwp(ilay, icol) > TF(0.))
            {
                TF t, ts, tsg;
                Cloud_optics_kernels::lookup(
                        clwp(ilay, icol), reliq(ilay, icol), nsize_liq,
                        c.liq_step_size, c.radliq_lwr,
                        c.lut_extliq, c.lut_ssaliq, c.lut_asyliq, ispec, t, ts, tsg);

                absorption += t - ts;
            }

            if (ciwp(ilay, icol) > TF(0.))
            {
                TF t, ts, tsg;
                Cloud_optics_kernels::lookup(
                        ciwp(ilay, icol), reice(ilay, icol), nsize_ice,
                        c.ice_step_size, c.radice_lwr,
                        c.lut_extice, c.lut_ssaice, c.lut_asyice, ispec, t, ts, tsg);

                absorption += t - ts;
            }

            tau(ispec, ilay, icol) = absorption;
        });
}
