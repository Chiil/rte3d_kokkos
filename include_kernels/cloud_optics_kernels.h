#pragma once

#include "types.h"


namespace Cloud_optics_kernels
{
    // One entry of the size lookup, linearly interpolated. Returns the optical depth
    // and the products tau*ssa and tau*ssa*g, which is the form the reference
    // accumulates in: it avoids dividing twice when liquid and ice are combined.
    //
    // Reference: compute_all_from_table.
    template<typename Table>
    KOKKOS_INLINE_FUNCTION
    void lookup(
            const TF water_path, const TF re,
            const int nsteps, const TF step_size, const TF offset,
            const Table& tau_table, const Table& ssa_table, const Table& asy_table,
            const int ispec,
            TF& tau, TF& taussa, TF& taussag)
    {
        // The reference clamps a 1-based index to nsteps-1; 0-based that is nsteps-2.
        const TF position = (re - offset) / step_size;
        const int index = Kokkos::min(static_cast<int>(Kokkos::floor(position)), nsteps - 2);

        // The fraction is relative to the clamped index, so a size above the table's
        // range extrapolates from the last interval.
        const TF fint = position - static_cast<TF>(index);

        const TF ext = tau_table(ispec, index)
                     + fint * (tau_table(ispec, index + 1) - tau_table(ispec, index));
        const TF ssa = ssa_table(ispec, index)
                     + fint * (ssa_table(ispec, index + 1) - ssa_table(ispec, index));
        const TF asy = asy_table(ispec, index)
                     + fint * (asy_table(ispec, index + 1) - asy_table(ispec, index));

        tau = water_path * ext;
        taussa = tau * ssa;
        taussag = taussa * asy;
    }
}
