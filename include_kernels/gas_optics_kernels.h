#pragma once

#include <limits>

#include "types.h"


namespace Gas_optics_kernels
{
    // The reference's guard against dividing by a vanishing column amount:
    // 2*tiny(col_mix), twice the smallest normal number. A function rather than a
    // namespace-scope constant so it is usable inside device lambdas; see the note in
    // rte_solver_kernels.h.
    KOKKOS_INLINE_FUNCTION constexpr TF tiny() { return std::numeric_limits<TF>::min(); }


    // The binary-species interpolation for one flavour: the reference's col_mix, jeta
    // and feta, for the two bracketing reference temperatures jtemp and jtemp+1.
    //
    // The reference stores these as (nflav, 2, nlay, ncol) arrays, and so did we until
    // the g-point loop made it clear what that costs: each g-point reads one flavour,
    // 24 of the 240 bytes per cell, once per g-point. Recomputing needs two col_gas
    // values and a vmr_ref lookup from a table small enough to stay in cache, for two
    // divides -- the same trade the header note on Interp_state describes for fmajor
    // and fminor.
    template<typename Flavor, typename Vmr_ref, typename Col_gas>
    KOKKOS_INLINE_FUNCTION
    void eta_interp(
            const Flavor& flavor,      // (nflav, 2)
            const Vmr_ref& vmr_ref,    // (ntemp, ngas+1, 2)
            const Col_gas& col_gas,    // (ngas+1, nlay, ncol)
            const int neta,
            const int iflav,
            const int itropo,          // 0 lower atmosphere, 1 upper
            const int jtemp,           // 0-based temperature index
            const int ilay, const int icol,
            TF col_mix[2], int jeta[2], TF feta[2])
    {
        const int igas0 = flavor(iflav, 0);
        const int igas1 = flavor(iflav, 1);

        const TF col0 = col_gas(igas0, ilay, icol);
        const TF col1 = col_gas(igas1, ilay, icol);

        for (int itemp=0; itemp<2; ++itemp)
        {
            // Ratio of reference volume mixing ratios that puts eta at 0.5, for this
            // flavour and reference temperature level.
            const TF ratio_eta_half = vmr_ref(jtemp + itemp, igas0, itropo)
                                    / vmr_ref(jtemp + itemp, igas1, itropo);

            const TF mix = col0 + ratio_eta_half * col1;
            col_mix[itemp] = mix;

            // A branch, not a select: the reference warns at length that with merge()
            // both arms are evaluated and this division can trap.
            TF eta;
            if (mix > TF(2.) * tiny())
                eta = col0 / mix;
            else
                eta = TF(0.5);

            const TF loceta = eta * static_cast<TF>(neta - 1);
            // The fraction is relative to the clamped index, not the reference's
            // loceta - floor(loceta): at eta = 1, which a flavour's absent second gas
            // gives exactly, that would be 0 and put all weight on node neta-2.
            jeta[itemp] = Kokkos::min(static_cast<int>(loceta), neta - 2);
            feta[itemp] = loceta - static_cast<TF>(jeta[itemp]);
        }
    }

    // Reconstruct the major and minor interpolation weights the reference stores as
    // fmajor and fminor. See the note on Interp_state for why they are recomputed
    // rather than kept.
    //
    // feta is the binary-species fraction for the two bracketing reference
    // temperatures. Output index order follows the reference, 0-based:
    //   fminor[itemp][ieta]
    //   fmajor[itemp][ipress][ieta]
    KOKKOS_INLINE_FUNCTION
    void interp_weights(
            const TF ftemp, const TF fpress, const TF feta_0, const TF feta_1,
            TF fminor[2][2], TF fmajor[2][2][2])
    {
        const TF feta[2] = {feta_0, feta_1};

        for (int itemp=0; itemp<2; ++itemp)
        {
            // 1 - ftemp for the lower reference temperature, ftemp for the upper.
            const TF ftemp_term = itemp == 0 ? TF(1.) - ftemp : ftemp;

            fminor[itemp][0] = (TF(1.) - feta[itemp]) * ftemp_term;
            fminor[itemp][1] =           feta[itemp]  * ftemp_term;

            for (int ieta=0; ieta<2; ++ieta)
            {
                fmajor[itemp][0][ieta] = (TF(1.) - fpress) * fminor[itemp][ieta];
                fmajor[itemp][1][ieta] =           fpress  * fminor[itemp][ieta];
            }
        }
    }


    // The 2x2x2 major-species interpolation of a (gpt, press, eta, temp) table at one
    // g-point: the innermost loop of both the absorption optical depth and the Planck
    // fraction, which differ only in the table and in what scales each of the two
    // bracketing temperatures -- the column mixing for kmajor, unity for pfracin.
    template<typename Table>
    KOKKOS_INLINE_FUNCTION
    TF interp_major(
            const Table& table,
            const int igpt, const int jpress, const int jtemp,
            const int jeta[2], const TF fmajor[2][2][2], const TF weight[2])
    {
        TF sum = TF(0.);

        for (int itemp=0; itemp<2; ++itemp)
        {
            const int je = jeta[itemp];
            TF acc = TF(0.);

            for (int ipress=0; ipress<2; ++ipress)
                for (int ieta=0; ieta<2; ++ieta)
                    acc += fmajor[itemp][ipress][ieta]
                         * table(igpt, jpress + ipress, je + ieta, jtemp + itemp);

            sum += weight[itemp] * acc;
        }

        return sum;
    }


    // Linear interpolation along the last axis of a two-dimensional table, for one
    // entry of the first. Reference: interpolate1D.
    //
    // The fraction is taken from the unclamped position while the index is clamped, so
    // values outside the table extrapolate rather than saturate. That is the
    // reference's behaviour, reproduced deliberately.
    template<typename Table>
    KOKKOS_INLINE_FUNCTION
    TF interpolate_1d(
            const TF val, const TF offset, const TF delta,
            const Table& table, const int irow, const int nval)
    {
        const TF val0 = (val - offset) / delta;
        const TF frac = val0 - static_cast<int>(val0);

        // The reference clamps a 1-based index to [1, nval-1]; 0-based that is
        // [0, nval-2].
        const int index = Kokkos::min(nval - 2, Kokkos::max(0, static_cast<int>(val0)));

        return table(irow, index) + frac * (table(irow, index + 1) - table(irow, index));
    }
}
