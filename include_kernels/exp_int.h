#pragma once

#include "types.h"


// The exponential integrals the longwave tracer needs to turn a flux leaving an
// isothermal slab back into the source that slab must have.
namespace Exp_int
{
    // E1(x) for x > 0. Abramowitz and Stegun 5.1.53 below one, absolute error under
    // 2e-7, and 5.1.56 above, relative error under 5e-5.
    KOKKOS_INLINE_FUNCTION
    TF exp_int_1(const TF x)
    {
        if (x <= TF(1.))
        {
            const TF p = TF(-0.57721566) + x*(TF(0.99999193) + x*(TF(-0.24991055)
                    + x*(TF(0.05519968) + x*(TF(-0.00976004) + x*TF(0.00107857)))));
            return p - Kokkos::log(x);
        }

        const TF num = x*x + TF(2.334733)*x + TF(0.250621);
        const TF den = x*x + TF(3.330657)*x + TF(1.681534);
        return Kokkos::exp(-x)/x*num/den;
    }


    // 1 - 2*E3(x): the fraction of what an isothermal slab of absorption optical depth
    // x emits into a hemisphere that leaves through the face on that side. It is what
    // takes a slab at source B to the flux pi*B*(1 - 2*E3(x)) out of either face.
    //
    // With E3(x) = (exp(-x)*(1 - x) + x^2*E1(x))/2 this is
    // 1 - exp(-x) + x*exp(-x) - x^2*E1(x), and the first two terms are written with
    // expm1 so that a thin slab, where the whole thing tends to 2x, does not lose it to
    // cancellation against the one.
    KOKKOS_INLINE_FUNCTION
    TF one_minus_2_exp_int_3(const TF x)
    {
        if (x <= TF(0.))
            return TF(0.);

        return -Kokkos::expm1(-x) + x*Kokkos::exp(-x) - x*x*exp_int_1(x);
    }
}
