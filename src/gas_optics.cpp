#include <limits>

#include "gas_optics.h"
#include "gas_optics_kernels.h"


namespace
{
    // The reference's guard against dividing by a vanishing column amount:
    // 2*tiny(col_mix), twice the smallest normal number.
    constexpr TF tiny = std::numeric_limits<TF>::min();
}


Interp_state Interp_state::create(const int nflav, const int nlay, const int ncol)
{
    const auto no_init = Kokkos::WithoutInitializing;

    Interp_state s;
    s.jtemp = Array_2d<int>(Kokkos::view_alloc("jtemp", no_init), nlay, ncol);
    s.ftemp = Array_2d<TF>(Kokkos::view_alloc("ftemp", no_init), nlay, ncol);
    s.jpress = Array_2d<int>(Kokkos::view_alloc("jpress", no_init), nlay, ncol);
    s.fpress = Array_2d<TF>(Kokkos::view_alloc("fpress", no_init), nlay, ncol);
    s.tropo = Array_2d<Bool>(Kokkos::view_alloc("tropo", no_init), nlay, ncol);

    s.jeta = Array_4d<int>(Kokkos::view_alloc("jeta", no_init), nflav, 2, nlay, ncol);
    s.feta = Array_4d<TF>(Kokkos::view_alloc("feta", no_init), nflav, 2, nlay, ncol);
    s.col_mix = Array_4d<TF>(Kokkos::view_alloc("col_mix", no_init), nflav, 2, nlay, ncol);

    return s;
}


void Gas_optics::interpolation(
        const Array_2d<const int>& flavor,
        const Array_1d<const TF>& press_ref_log,
        const Array_1d<const TF>& temp_ref,
        const TF press_ref_log_delta,
        const TF temp_ref_min,
        const TF temp_ref_delta,
        const TF press_ref_trop_log,
        const int neta,
        const Array_3d<const TF>& vmr_ref,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const Interp_state& state)
{
    const int nlay = static_cast<int>(play.extent(0));
    const int ncol = static_cast<int>(play.extent(1));
    const int nflav = static_cast<int>(flavor.extent(0));
    const int ntemp = static_cast<int>(temp_ref.extent(0));
    const int npres = static_cast<int>(press_ref_log.extent(0));

    const auto jtemp = state.jtemp;
    const auto ftemp = state.ftemp;
    const auto jpress = state.jpress;
    const auto fpress = state.fpress;
    const auto tropo = state.tropo;
    const auto jeta = state.jeta;
    const auto feta = state.feta;
    const auto col_mix = state.col_mix;

    // Temperature and pressure location, and which half of the atmosphere we are in.
    parallel_for_2d("interpolation_grid", {0, 0}, {nlay, ncol},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            const TF t = tlay(ilay, icol);
            const TF log_p = Kokkos::log(play(ilay, icol));

            // The reference clamps a 1-based index into [1, ntemp-1]. We store it
            // 0-based, so the stored range is [0, ntemp-2].
            const int jt = Kokkos::min(ntemp - 1, Kokkos::max(1,
                    static_cast<int>((t - (temp_ref_min - temp_ref_delta)) / temp_ref_delta)));
            jtemp(ilay, icol) = jt - 1;
            ftemp(ilay, icol) = (t - temp_ref(jt - 1)) / temp_ref_delta;

            // locpress stays 1-based, since fpress is its fractional part relative to
            // the 1-based index.
            const TF locpress = TF(1.) + (log_p - press_ref_log(0)) / press_ref_log_delta;
            const int jp = Kokkos::min(npres - 1, Kokkos::max(1, static_cast<int>(locpress)));
            jpress(ilay, icol) = jp - 1;
            fpress(ilay, icol) = locpress - static_cast<TF>(jp);

            tropo(ilay, icol) = log_p > press_ref_trop_log ? Bool(1) : Bool(0);
        });

    // Binary species parameter, per flavour.
    parallel_for_3d("interpolation_eta", {0, 0, 0}, {nflav, nlay, ncol},
        KOKKOS_LAMBDA(const int iflav, const int ilay, const int icol)
        {
            const int igas0 = flavor(iflav, 0);
            const int igas1 = flavor(iflav, 1);

            // itropo = 0 lower atmosphere, 1 upper.
            const int itropo = tropo(ilay, icol) ? 0 : 1;
            const int jt = jtemp(ilay, icol);

            for (int itemp=0; itemp<2; ++itemp)
            {
                // Ratio of reference volume mixing ratios that puts eta at 0.5, for
                // this flavour and reference temperature level.
                const TF ratio_eta_half = vmr_ref(jt + itemp, igas0, itropo)
                                        / vmr_ref(jt + itemp, igas1, itropo);

                const TF mix = col_gas(igas0, ilay, icol)
                             + ratio_eta_half * col_gas(igas1, ilay, icol);
                col_mix(iflav, itemp, ilay, icol) = mix;

                // A branch, not a select: the reference warns at length that with
                // merge() both arms are evaluated and this division can trap.
                TF eta;
                if (mix > TF(2.) * tiny)
                    eta = col_gas(igas0, ilay, icol) / mix;
                else
                    eta = TF(0.5);

                const TF loceta = eta * static_cast<TF>(neta - 1);
                jeta(iflav, itemp, ilay, icol) = Kokkos::min(static_cast<int>(loceta), neta - 2);
                feta(iflav, itemp, ilay, icol) = loceta - Kokkos::floor(loceta);
            }
        });
}


void Gas_optics::expand_weights(
        const Interp_state& state,
        const Array_5d<TF>& fminor,
        const Array_6d<TF>& fmajor)
{
    const int nflav = static_cast<int>(fminor.extent(0));
    const int nlay = static_cast<int>(fminor.extent(3));
    const int ncol = static_cast<int>(fminor.extent(4));

    const auto ftemp = state.ftemp;
    const auto fpress = state.fpress;
    const auto feta = state.feta;

    parallel_for_3d("expand_weights", {0, 0, 0}, {nflav, nlay, ncol},
        KOKKOS_LAMBDA(const int iflav, const int ilay, const int icol)
        {
            TF fmin[2][2], fmaj[2][2][2];
            Gas_optics_kernels::interp_weights(
                    ftemp(ilay, icol), fpress(ilay, icol),
                    feta(iflav, 0, ilay, icol), feta(iflav, 1, ilay, icol),
                    fmin, fmaj);

            for (int itemp=0; itemp<2; ++itemp)
                for (int ieta=0; ieta<2; ++ieta)
                {
                    fminor(iflav, itemp, ieta, ilay, icol) = fmin[itemp][ieta];

                    for (int ipress=0; ipress<2; ++ipress)
                        fmajor(iflav, itemp, ipress, ieta, ilay, icol) = fmaj[itemp][ipress][ieta];
                }
        });
}
