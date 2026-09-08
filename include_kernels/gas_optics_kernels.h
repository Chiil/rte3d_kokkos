#pragma once

#include "types.h"


namespace Gas_optics_kernels
{
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
