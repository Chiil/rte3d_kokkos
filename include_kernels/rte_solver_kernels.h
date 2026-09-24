#pragma once

#include <limits>

#include "fluxes.h"
#include "types.h"


namespace Rte_kernels
{
    // Functions rather than namespace-scope constants so they are usable inside
    // device lambdas: a host constexpr variable is ODR-used (its address taken) when
    // passed by reference to Kokkos::max etc., which nvcc rejects in device code.
    KOKKOS_INLINE_FUNCTION constexpr TF eps() { return std::numeric_limits<TF>::epsilon(); }

    // Lower limit on k, suggested by Chiel van Heerwaarden: k = 0 for isotropic,
    // conservative scattering, and this limit keeps the relative error in Rdif below
    // 0.1% down to tau = 1e-9 while avoiding the division by zero.
    KOKKOS_INLINE_FUNCTION constexpr TF min_k() { return TF(1.e4) * eps(); }


    // Vertical orientation of the arrays. The reference writes every loop out twice,
    // once per orientation; instead we carry the two level offsets of a layer and
    // template the solvers on the orientation, so the compiler still sees explicit
    // loops but the physics is written once.
    //
    // Layer ilay is bounded by levels ilay+lev_up() and ilay+lev_dn(), where "up" and
    // "dn" are directions in space, not array order.
    template<bool top_at_1>
    struct Vert
    {
        KOKKOS_INLINE_FUNCTION static constexpr int lev_up() { return top_at_1 ? 0 : 1; }
        KOKKOS_INLINE_FUNCTION static constexpr int lev_dn() { return top_at_1 ? 1 : 0; }

        KOKKOS_INLINE_FUNCTION static constexpr int lev_toa(const int nlay) { return top_at_1 ? 0 : nlay; }
        KOKKOS_INLINE_FUNCTION static constexpr int lev_sfc(const int nlay) { return top_at_1 ? nlay : 0; }

        // Layer visited as the j-th step of a sweep that starts at the top of the
        // atmosphere and works downward.
        KOKKOS_INLINE_FUNCTION static constexpr int lay_from_toa(const int j, const int nlay)
        { return top_at_1 ? j : nlay-1-j; }

        // Layer visited as the j-th step of a sweep that starts at the surface and
        // works upward.
        KOKKOS_INLINE_FUNCTION static constexpr int lay_from_sfc(const int j, const int nlay)
        { return top_at_1 ? nlay-1-j : j; }
    };


    // exp(-x) and 1 - exp(-x) for x >= 0, both to full relative precision with one
    // transcendental: through expm1 for thin layers, where 1 - exp(-x) would cancel,
    // and through exp for thick ones, where 1 + expm1(-x) would lose exp(-x).
    KOKKOS_INLINE_FUNCTION
    void exp_and_one_minus_exp(const TF x, TF& exp_minusx, TF& one_minus_exp_minusx)
    {
        if (x < TF(0.5))
        {
            one_minus_exp_minusx = -Kokkos::expm1(-x);
            exp_minusx = TF(1.) - one_minus_exp_minusx;
        }
        else
        {
            exp_minusx = Kokkos::exp(-x);
            one_minus_exp_minusx = TF(1.) - exp_minusx;
        }
    }


    // Meador and Weaver direct-beam reflectance and transmittance into the diffuse
    // field, Eqs 14-15, at one mu0. RT_term and the exponentials are those of the
    // diffuse solution; Tnoscat is exp(-tau/mu0). Singular at k*mu0 = 1.
    //
    // The numerators are regrouped from the reference's so that nothing cancels as
    // k -> 0: with e1 = exp(-k*tau), e2 = e1^2 and T = Tnoscat, the combinations
    // 1 + e2 - 2*e1*T and T*(1 + e2) - 2*e1 appear, which are written through 1 - e1
    // and 1 - T. Their O(1) parts would otherwise cancel to O(k), leaving a relative
    // error of eps/k -- the reason min_k had to be large.
    KOKKOS_INLINE_FUNCTION
    void sw_direct(
            const TF w0, const TF g, const TF gamma1, const TF gamma2, const TF k,
            const TF exp_minusktau, const TF one_minus_exp_minusktau, const TF RT_term,
            const TF mu0, const TF Tnoscat, const TF one_minus_Tnoscat,
            TF& Rdir, TF& Tdir)
    {
        const TF k_mu = k * mu0;

        // Eq 14, top and bottom multiplied by exp(-k*tau).
        const TF RT_term_dir = w0 * RT_term / ((TF(1.) - k_mu) * (TF(1.) + k_mu));

        const TF gamma3 = (TF(2.) - TF(3.) * mu0 * g) * TF(0.25);
        const TF gamma4 = TF(1.) - gamma3;
        const TF alpha1 = gamma1 * gamma4 + gamma2 * gamma3;  // Eq 16
        const TF alpha2 = gamma1 * gamma3 + gamma2 * gamma4;  // Eq 17

        const TF one_minus_e1_sq = one_minus_exp_minusktau * one_minus_exp_minusktau;
        const TF one_minus_e2 = one_minus_exp_minusktau * (TF(1.) + exp_minusktau);
        const TF two_e1_one_minus_T = TF(2.) * exp_minusktau * one_minus_Tnoscat;

        // (1 - k*mu0)*(alpha2 + k*gamma3) - (1 + k*mu0)*(alpha2 - k*gamma3)*e2
        //     - 2*(k*gamma3 - alpha2*k*mu0)*e1*T
        Rdir = RT_term_dir *
            (k * (gamma3 - mu0 * alpha2) * (one_minus_e1_sq + two_e1_one_minus_T) +
             one_minus_e2 * (alpha2 - k_mu * k * gamma3));

        // Eq 15, top and bottom multiplied by exp(-k*tau) and the whole multiplied
        // through by exp(-tau/mu0) to prefer underflow to overflow. The direct
        // transmittance is omitted.
        // (1 + k*mu0)*(alpha1 + k*gamma4)*T - (1 - k*mu0)*(alpha1 - k*gamma4)*e2*T
        //     - 2*(k*gamma4 + alpha1*k*mu0)*e1
        Tdir = -RT_term_dir *
            (Tnoscat * one_minus_e2 * (alpha1 + k_mu * k * gamma4) +
             k * (gamma4 + mu0 * alpha1) * (Tnoscat * one_minus_e1_sq - two_e1_one_minus_T));
    }


    // Zdunkowski Practical Improved Flux Method "PIFM" (Zdunkowski et al., 1980,
    // Contributions to Atmospheric Physics 53, 147-66), with the Meador and Weaver
    // direct-beam terms. Reference: sw_dif_and_source in mo_rte_solver_kernels.F90.
    //
    // Returns the layer's diffuse reflectance and transmittance, the direct-beam
    // reflectance and transmittance into the diffuse field, and the unscattered
    // direct transmittance.
    KOKKOS_INLINE_FUNCTION
    void sw_two_stream(
            const TF tau, const TF w0, const TF g, const TF mu0,
            TF& Rdif, TF& Tdif, TF& Rdir, TF& Tdir, TF& Tnoscat)
    {
        const TF gamma1 = (TF(8.) - w0 * (TF(5.) + TF(3.)*g)) * TF(0.25);
        const TF gamma2 = (TF(3.) * (w0 * (TF(1.) - g))) * TF(0.25);

        // Eq 18; k = sqrt(gamma1^2 - gamma2^2), limited below to avoid dividing by 0.
        // gamma1 - gamma2 is exactly 2*(1 - w0); written so, it does not cancel as
        // w0 -> 1.
        const TF k = Kokkos::sqrt(Kokkos::max(
                TF(2.) * (TF(1.) - w0) * (gamma1 + gamma2), min_k()));

        TF exp_minusktau, one_minus_exp_minusktau;
        exp_and_one_minus_exp(k*tau, exp_minusktau, one_minus_exp_minusktau);
        const TF exp_minus2ktau = exp_minusktau * exp_minusktau;
        const TF one_minus_exp_minus2ktau = one_minus_exp_minusktau * (TF(1.) + exp_minusktau);

        // Refactored to avoid rounding errors when k and gamma1 differ greatly in magnitude.
        const TF RT_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
                                     gamma1 * one_minus_exp_minus2ktau);

        Rdif = RT_term * gamma2 * one_minus_exp_minus2ktau;  // Eq 25
        Tdif = RT_term * TF(2.) * k * exp_minusktau;         // Eq 26

        // On a round earth mu0 can increase with depth, so levels with mu0 <= 0 have no
        // direct beam. Compute with a nominal value here and mask the result at the
        // call site.
        const TF mu0_s = Kokkos::max(Kokkos::sqrt(eps()), mu0);
        TF one_minus_Tnoscat;
        exp_and_one_minus_exp(tau/mu0_s, Tnoscat, one_minus_Tnoscat);

        // Eqs 14-15 have a removable singularity at k*mu0 = 1: numerator and
        // denominator both vanish, and their rounding error grows like eps/|1 - k*mu0|.
        // Within a window of eps^(1/3) of it, interpolate linearly between the window
        // edges, which costs O(window^2); the two errors balance at that width. The
        // reference only replaces a denominator below eps with +eps, which in single
        // precision loses Rdir and Tdir entirely near resonance.
        const TF window = Kokkos::cbrt(eps());
        const TF x = k * mu0_s - TF(1.);
        if (Kokkos::abs(x) < window)
        {
            const TF mu0_lo = (TF(1.) - window) / k;
            const TF mu0_hi = (TF(1.) + window) / k;
            TF T_lo, one_minus_T_lo, T_hi, one_minus_T_hi;
            exp_and_one_minus_exp(tau/mu0_lo, T_lo, one_minus_T_lo);
            exp_and_one_minus_exp(tau/mu0_hi, T_hi, one_minus_T_hi);

            TF Rdir_lo, Tdir_lo, Rdir_hi, Tdir_hi;
            sw_direct(w0, g, gamma1, gamma2, k, exp_minusktau, one_minus_exp_minusktau, RT_term,
                      mu0_lo, T_lo, one_minus_T_lo, Rdir_lo, Tdir_lo);
            sw_direct(w0, g, gamma1, gamma2, k, exp_minusktau, one_minus_exp_minusktau, RT_term,
                      mu0_hi, T_hi, one_minus_T_hi, Rdir_hi, Tdir_hi);

            const TF frac = (x + window) / (TF(2.) * window);
            Rdir = Rdir_lo + frac * (Rdir_hi - Rdir_lo);
            Tdir = Tdir_lo + frac * (Tdir_hi - Tdir_lo);
        }
        else
            sw_direct(w0, g, gamma1, gamma2, k, exp_minusktau, one_minus_exp_minusktau, RT_term,
                      mu0_s, Tnoscat, one_minus_Tnoscat, Rdir, Tdir);

        // The beam is reflected, penetrates unscattered, or penetrates and is
        // scattered on the way; the rest is absorbed. Clamping to that budget keeps
        // the equations safe in single precision. Credit: Robin Hogan, Peter Ukkonen.
        Rdir = Kokkos::max(TF(0.), Kokkos::min(Rdir, TF(1.) - Tnoscat));
        Tdir = Kokkos::max(TF(0.), Kokkos::min(Tdir, TF(1.) - Tnoscat - Rdir));
    }

    inline constexpr TF pi = TF(3.14159265358979323846);

    // Longwave source function for diffuse radiation, using the linear-in-tau
    // assumption of Clough et al., 1992, doi:10.1029/92JD01419, Eq 13.
    //
    // lev_source_up and lev_source_dn are the Planck sources at the level on the
    // upward and downward side of the layer, which is what the reference selects with
    // its source_inc / source_dec pointer swap.
    KOKKOS_INLINE_FUNCTION
    void lw_source_noscat(
            const TF lay_source, const TF lev_source_up, const TF lev_source_dn,
            const TF tau_loc, const TF trans,
            TF& source_up, TF& source_dn)
    {
        // Weighting factor. Below the threshold the rounding error in the direct form
        // (~tau^2) is of order epsilon, so use a 3rd order series expansion instead.
        // Thanks to Peter Blossey (UW) for the idea and Dmitry Alexeev (Nvidia) for
        // suggesting 3rd order.
        const TF tau_thresh = Kokkos::sqrt(Kokkos::sqrt(eps()));

        const TF fact = tau_loc > tau_thresh
                ? (TF(1.) - trans)/tau_loc - trans
                : tau_loc * (TF(0.5) + tau_loc * (TF(-1.)/TF(3.) + tau_loc * TF(1.)/TF(8.)));

        source_dn = (TF(1.) - trans) * lev_source_dn + TF(2.) * fact * (lay_source - lev_source_dn);
        source_up = (TF(1.) - trans) * lev_source_up + TF(2.) * fact * (lay_source - lev_source_up);
    }

    // Transport of diffuse radiation through a vertically layered atmosphere, after
    // Shonk and Hogan 2008, doi:10.1175/2007JCLI1940.1 (SH08). Shared by longwave and
    // shortwave. Reference: adding in mo_rte_solver_kernels.F90.
    //
    // Two sequential sweeps over the column, issued separately so
    // parallel_for_column_sweep can pick the loop nesting the backend wants. Each
    // sweep applies its own starting condition at j == 0 rather than in a launch of
    // its own: the branch is invariant in the vectorized column loop, and at modest
    // column counts the saved parallel regions are worth more than the branch costs.
    //
    // albedo and src carry state between the sweeps and are (nlev, ncol) scratch.
    // flux_dn_toa is the incident diffuse flux, and may be empty for a zero boundary
    // condition.
    //
    // The reference also carries denom, SH08's 1/(1 - rdif*albedo_below), from the
    // first sweep to the second. It is not kept here: the second sweep reads rdif and
    // that same albedo anyway, so rebuilding it costs one multiply and one divide and
    // saves writing and reading a whole (nlay, ncol) array -- which at these sizes the
    // sweeps are entirely bound by.
    //
    // Neither sweep reads an array it has written: the three values that cross from
    // one layer to the next ride the carry instead. So neither flux needs a g-point
    // array, and a caller after nothing but the spectral totals can leave both out.
    // static, and so a copy per translation unit, deliberately: nvcc identifies the
    // closure type of an extended device lambda by its enclosing function, so the two
    // sweeps below get the same mangled type in rte_lw.cpp and rte_sw.cpp, which both
    // instantiate this template. The linker then merges the two, and the host-side
    // launch calls into a closure that was never registered -- a segmentation fault in
    // the shortwave two-stream solver, with no CUDA error to show for it. Internal
    // linkage keeps the two apart. Do not make this inline again.
    template<bool top_at_1>
    static void adding(
            const int nlay, const int ncol,
            const Array_map_1d<const TF>& albedo_sfc,   // (ncol)
            const Array_map_1d<const TF>& src_sfc,      // (ncol)
            const Array_map_1d<const TF>& flux_dn_toa,  // (ncol), may be empty
            const Array_map_2d<const TF>& rdif, const Array_map_2d<const TF>& tdif,
            const Array_map_2d<const TF>& src_dn, const Array_map_2d<const TF>& src_up,
            const Flux_sink& flux_up, const Flux_sink& flux_dn,
            const Array_2d<TF>& albedo, const Array_2d<TF>& src,
            const Array_2d<TF>& carry)   // (2, ncol) scratch for the sweeps
    {

        using V = Vert<top_at_1>;

        const int lev_sfc = V::lev_sfc(nlay);
        const int lev_toa = V::lev_toa(nlay);
        const bool has_dif_bc = flux_dn_toa.size() > 0;

        // From the surface upward, accumulate the reflectivity to diffuse radiation
        // below each level (alpha in SH08) and the source of diffuse upwelling
        // radiation (G in SH08), both of which start at the surface.
        // Two values cross from one layer to the next: the albedo and the source below
        // the level in hand.
        parallel_for_column_sweep_carry("adding_up", nlay, ncol, carry,
            KOKKOS_LAMBDA(const int j, const int icol, TF carried[2])
            {
                if (j == 0)
                {
                    albedo(lev_sfc, icol) = albedo_sfc(icol);
                    src   (lev_sfc, icol) = src_sfc(icol);

                    carried[0] = albedo_sfc(icol);
                    carried[1] = src_sfc(icol);
                }

                const int ilay = V::lay_from_sfc(j, nlay);
                const int lev_above = ilay + V::lev_up();

                const TF rdif_l = rdif(ilay, icol);
                const TF tdif_l = tdif(ilay, icol);
                const TF albedo_below = carried[0];

                const TF denom_l = TF(1.) / (TF(1.) - rdif_l * albedo_below);  // Eq 10

                const TF albedo_l =                                           // Eq 9
                        rdif_l + tdif_l*tdif_l * albedo_below * denom_l;

                // Eq 11: upward emission at the top of the layer, plus radiation emitted
                // at the bottom, transmitted through and reflected from the layers below.
                const TF src_l =
                        src_up(ilay, icol)
                        + tdif_l * denom_l * (carried[1]
                                              + albedo_below * src_dn(ilay, icol));

                albedo(lev_above, icol) = albedo_l;
                src(lev_above, icol) = src_l;

                carried[0] = albedo_l;
                carried[1] = src_l;
            });

        // From the top of the atmosphere downward, compute the fluxes.
        parallel_for_column_sweep_carry("adding_dn", nlay, ncol, carry,
            KOKKOS_LAMBDA(const int j, const int icol, TF carried[2])
            {
                if (j == 0)
                {
                    // Eq 12 at the top of the domain: reflection of the incident
                    // diffuse flux plus emission from below.
                    const TF flux_dn_l = has_dif_bc ? flux_dn_toa(icol) : TF(0.);

                    flux_dn.put(lev_toa, icol, flux_dn_l);
                    flux_up.put(lev_toa, icol,
                                flux_dn_l * albedo(lev_toa, icol) + src(lev_toa, icol));

                    carried[0] = flux_dn_l;
                }

                const int ilay = V::lay_from_toa(j, nlay);
                const int lev_dst = ilay + V::lev_dn();

                const TF rdif_l = rdif(ilay, icol);
                const TF albedo_l = albedo(lev_dst, icol);
                const TF src_l = src(lev_dst, icol);

                // Eq 10 again, from the two values this sweep was reading anyway.
                const TF denom_l = TF(1.) / (TF(1.) - rdif_l * albedo_l);

                const TF flux_dn_l =                                          // Eq 13
                        (tdif(ilay, icol) * carried[0]
                         + rdif_l * src_l
                         + src_dn(ilay, icol)) * denom_l;

                flux_dn.put(lev_dst, icol, flux_dn_l);
                flux_up.put(lev_dst, icol, flux_dn_l * albedo_l + src_l);     // Eq 12

                carried[0] = flux_dn_l;
            });
    }


    // Longwave two-stream diffuse reflectance and transmittance for a layer, plus the
    // coupling coefficients the source function needs. Reference: lw_two_stream.
    //
    // The coefficients differ from the shortwave because the phase function is more
    // isotropic; we follow Fu et al. 1997,
    // doi:10.1175/1520-0469(1997)054<2799:MSPITI>2.0.CO;2, with a diffusivity secant
    // of 1.66.
    KOKKOS_INLINE_FUNCTION
    void lw_two_stream(
            const TF tau, const TF w0, const TF g,
            TF& gamma1, TF& gamma2, TF& Rdif, TF& Tdif)
    {
        // The reference writes this as `real(wp), parameter :: LW_diff_sec = 1.66`.
        // The literal has default (single) real kind and is only then promoted, so its
        // value is 1.65999996662139893, not 1.66 -- a 2e-8 relative difference. Match
        // it exactly rather than "fix" it, or every comparison drifts by that much.
        constexpr TF lw_diff_sec = static_cast<TF>(1.66f);

        gamma1 = lw_diff_sec * (TF(1.) - TF(0.5) * w0 * (TF(1.) + g));  // Fu et al. Eq 2.9
        gamma2 = lw_diff_sec *           TF(0.5) * w0 * (TF(1.) - g);   // Fu et al. Eq 2.10

        // Eq 18; k = sqrt(gamma1^2 - gamma2^2). gamma1 - gamma2 is exactly
        // lw_diff_sec*(1 - w0); written so, it does not cancel as w0 -> 1. Note the
        // floor is a plain 1e-12 here, not the epsilon-derived min_k the shortwave uses.
        const TF k = Kokkos::sqrt(Kokkos::max(
                lw_diff_sec * (TF(1.) - w0) * (gamma1 + gamma2), TF(1.e-12)));

        TF exp_minusktau, one_minus_exp_minusktau;
        exp_and_one_minus_exp(k*tau, exp_minusktau, one_minus_exp_minusktau);
        const TF exp_minus2ktau = exp_minusktau * exp_minusktau;
        const TF one_minus_exp_minus2ktau = one_minus_exp_minusktau * (TF(1.) + exp_minusktau);

        // Refactored to avoid rounding errors when k and gamma1 differ greatly in magnitude.
        const TF RT_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
                                     gamma1 * one_minus_exp_minus2ktau);

        Rdif = RT_term * gamma2 * one_minus_exp_minus2ktau;  // Eq 25
        Tdif = RT_term * TF(2.) * k * exp_minusktau;         // Eq 26
    }


    // Longwave source function for upward and downward emission at levels, using the
    // linear-in-tau assumption. Straight from ECRAD, via Toon et al. (JGR 1989)
    // Eqs 26-27. Sources come out in flux units, hence the factor of pi.
    //
    // lev_source_top and lev_source_bot are the Planck sources at the levels above and
    // below the layer: ilay + lev_up() and ilay + lev_dn().
    KOKKOS_INLINE_FUNCTION
    void lw_source_2str(
            const TF lev_source_top, const TF lev_source_bot,
            const TF gamma1, const TF gamma2, const TF rdif, const TF tdif, const TF tau,
            TF& source_up, TF& source_dn)
    {
        if (tau > TF(1.0e-8))
        {
            const TF Z = (lev_source_bot - lev_source_top) / (tau * (gamma1 + gamma2));

            const TF Zup_top    =  Z + lev_source_top;
            const TF Zup_bottom =  Z + lev_source_bot;
            const TF Zdn_top    = -Z + lev_source_top;
            const TF Zdn_bottom = -Z + lev_source_bot;

            source_up = pi * (Zup_top    - rdif * Zdn_top    - tdif * Zup_bottom);
            source_dn = pi * (Zdn_bottom - rdif * Zup_bottom - tdif * Zdn_top);
        }
        else
        {
            source_up = TF(0.);
            source_dn = TF(0.);
        }
    }
}
