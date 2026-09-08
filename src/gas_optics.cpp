#include <limits>
#include <vector>

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


void Minor_absorbers::build_map(
        const Array_2d<const int>& gpoint_flavor, const int ngpt, const int itropo)
{
    const int nminor = static_cast<int>(minor_limits_gpt.extent(0));

    auto limits_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, minor_limits_gpt);
    auto gpt_flavor_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gpoint_flavor);

    // Count contributors per g-point, then fill. A minor absorber covers a contiguous
    // g-point range, and ranges from different absorbers may overlap.
    std::vector<int> count(ngpt + 1, 0);
    for (int imnr=0; imnr<nminor; ++imnr)
        for (int igpt=limits_h(imnr, 0); igpt<=limits_h(imnr, 1); ++igpt)
            ++count[igpt + 1];

    for (int igpt=0; igpt<ngpt; ++igpt)
        count[igpt + 1] += count[igpt];

    const int nnz = count[ngpt];

    gpt_offset = Array_1d<int>(Kokkos::view_alloc("gpt_offset", Kokkos::WithoutInitializing), ngpt + 1);
    gpt_minor = Array_1d<int>(Kokkos::view_alloc("gpt_minor", Kokkos::WithoutInitializing), nnz);
    flavor = Array_1d<int>(Kokkos::view_alloc("minor_flavor", Kokkos::WithoutInitializing), nminor);

    auto offset_h = Kokkos::create_mirror_view(gpt_offset);
    auto minor_h = Kokkos::create_mirror_view(gpt_minor);
    auto flavor_h = Kokkos::create_mirror_view(flavor);

    for (int igpt=0; igpt<=ngpt; ++igpt)
        offset_h(igpt) = count[igpt];

    std::vector<int> fill(ngpt, 0);
    for (int imnr=0; imnr<nminor; ++imnr)
    {
        // The reference takes the flavour from the first g-point of the absorber's
        // range, not from each g-point separately.
        flavor_h(imnr) = gpt_flavor_h(limits_h(imnr, 0), itropo);

        for (int igpt=limits_h(imnr, 0); igpt<=limits_h(imnr, 1); ++igpt)
            minor_h(offset_h(igpt) + fill[igpt]++) = imnr;
    }

    Kokkos::deep_copy(gpt_offset, offset_h);
    Kokkos::deep_copy(gpt_minor, minor_h);
    Kokkos::deep_copy(flavor, flavor_h);
}


void Gas_optics::compute_tau_absorption(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const Array_3d<TF>& tau)
{
    const int ngpt = static_cast<int>(tau.extent(0));
    const int nlay = static_cast<int>(tau.extent(1));
    const int ncol = static_cast<int>(tau.extent(2));

    const auto jtemp = state.jtemp;
    const auto ftemp = state.ftemp;
    const auto jpress = state.jpress;
    const auto fpress = state.fpress;
    const auto tropo = state.tropo;
    const auto jeta = state.jeta;
    const auto feta = state.feta;
    const auto col_mix = state.col_mix;

    // Layer limits of the lower and upper atmosphere, per column. The reference finds
    // the extreme pressure among the layers on each side of the tropopause and treats
    // everything between there and the domain edge as belonging to that side. For a
    // monotonic pressure profile that is exactly the set of layers where tropo holds,
    // but we reproduce the range form so non-monotonic input matches too.
    Array_2d<int> lower_limits(Kokkos::view_alloc("lower_limits", Kokkos::WithoutInitializing), ncol, 2);
    Array_2d<int> upper_limits(Kokkos::view_alloc("upper_limits", Kokkos::WithoutInitializing), ncol, 2);

    // top_at_1 is decided from the first column, as in the reference.
    auto play_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, play);
    const bool top_at_1 = play_h(0, 0) < play_h(nlay - 1, 0);

    parallel_for_1d("tau_absorption_limits", 0, ncol,
        KOKKOS_LAMBDA(const int icol)
        {
            // minloc over layers where tropo holds, maxloc where it does not. Zero
            // means no such layer, matching the reference's guard value.
            int arg_min = 0;
            int arg_max = 0;
            TF p_min = TF(0.);
            TF p_max = TF(0.);

            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const TF p = play(ilay, icol);

                if (tropo(ilay, icol))
                {
                    if (arg_min == 0 || p < p_min) { p_min = p; arg_min = ilay + 1; }
                }
                else
                {
                    if (arg_max == 0 || p > p_max) { p_max = p; arg_max = ilay + 1; }
                }
            }

            if (top_at_1)
            {
                lower_limits(icol, 0) = arg_min;  lower_limits(icol, 1) = nlay;
                upper_limits(icol, 0) = 1;        upper_limits(icol, 1) = arg_max;
            }
            else
            {
                lower_limits(icol, 0) = 1;        lower_limits(icol, 1) = arg_min;
                upper_limits(icol, 0) = arg_max;  upper_limits(icol, 1) = nlay;
            }
        });

    const auto gpoint_flavor = k.gpoint_flavor;
    const auto band_gpt_start = k.band_gpt_start;
    const auto gpt_band = k.gpt_band;
    const auto kmajor = k.kmajor;
    const int idx_h2o = k.idx_h2o;

    const Minor_absorbers lower = k.lower;
    const Minor_absorbers upper = k.upper;

    parallel_for_3d("compute_tau_absorption", {0, 0, 0}, {ngpt, nlay, ncol},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int itropo = tropo(ilay, icol) ? 0 : 1;
            const int jt = jtemp(ilay, icol);
            const TF ft = ftemp(ilay, icol);
            const TF fp = fpress(ilay, icol);

            TF tau_l = TF(0.);

            // ---- major species -------------------------------------------------
            // The flavour comes from the first g-point of this g-point's band.
            {
                const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt)), itropo);

                TF fmin[2][2], fmaj[2][2][2];
                Gas_optics_kernels::interp_weights(
                        ft, fp, feta(iflav, 0, ilay, icol), feta(iflav, 1, ilay, icol), fmin, fmaj);

                // The reference indexes kmajor at jpress-1 and jpress with a 1-based
                // jpress+itropo; 0-based that is jpress+itropo and one beyond.
                const int jp0 = jpress(ilay, icol) + itropo;

                for (int itemp=0; itemp<2; ++itemp)
                {
                    const int je = jeta(iflav, itemp, ilay, icol);
                    TF acc = TF(0.);

                    for (int ipress=0; ipress<2; ++ipress)
                        for (int ieta=0; ieta<2; ++ieta)
                            acc += fmaj[itemp][ipress][ieta]
                                 * kmajor(igpt, jp0 + ipress, je + ieta, jt + itemp);

                    tau_l += col_mix(iflav, itemp, ilay, icol) * acc;
                }
            }

            // ---- minor species -------------------------------------------------
            for (int side=0; side<2; ++side)
            {
                const Minor_absorbers& m = side == 0 ? lower : upper;
                const auto& limits = side == 0 ? lower_limits : upper_limits;

                // The reference walks a per-column layer range; a zero start means
                // this column has no layers on this side of the tropopause.
                if (limits(icol, 0) == 0)
                    continue;
                if (ilay + 1 < limits(icol, 0) || ilay + 1 > limits(icol, 1))
                    continue;

                for (int i=m.gpt_offset(igpt); i<m.gpt_offset(igpt + 1); ++i)
                {
                    const int imnr = m.gpt_minor(i);

                    TF scaling = col_gas(m.idx_minor(imnr), ilay, icol);

                    if (m.scales_with_density(imnr))
                    {
                        // Pressure in hPa, as the density scaling expects.
                        scaling *= TF(0.01) * play(ilay, icol) / tlay(ilay, icol);

                        const int idx_scaling = m.idx_minor_scaling(imnr);
                        if (idx_scaling > 0)
                        {
                            const TF vmr_fact = TF(1.) / col_gas(0, ilay, icol);
                            const TF dry_fact =
                                    TF(1.) / (TF(1.) + col_gas(idx_h2o, ilay, icol) * vmr_fact);

                            const TF f = col_gas(idx_scaling, ilay, icol) * vmr_fact * dry_fact;
                            scaling *= m.scale_by_complement(imnr) ? TF(1.) - f : f;
                        }
                    }

                    const int iflav = m.flavor(imnr);
                    const int ik = m.kminor_start(imnr) + (igpt - m.minor_limits_gpt(imnr, 0));

                    TF fmin[2][2], fmaj[2][2][2];
                    Gas_optics_kernels::interp_weights(
                            ft, fp, feta(iflav, 0, ilay, icol), feta(iflav, 1, ilay, icol), fmin, fmaj);

                    TF acc = TF(0.);
                    for (int itemp=0; itemp<2; ++itemp)
                    {
                        const int je = jeta(iflav, itemp, ilay, icol);
                        for (int ieta=0; ieta<2; ++ieta)
                            acc += fmin[itemp][ieta] * m.kminor(ik, je + ieta, jt + itemp);
                    }

                    tau_l += scaling * acc;
                }
            }

            tau(igpt, ilay, icol) += tau_l;
        });
}


void Gas_optics::compute_tau_rayleigh(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& col_dry,
        const Array_3d<const TF>& col_gas,
        const Array_3d<TF>& tau_rayleigh)
{
    const int ngpt = static_cast<int>(tau_rayleigh.extent(0));
    const int nlay = static_cast<int>(tau_rayleigh.extent(1));
    const int ncol = static_cast<int>(tau_rayleigh.extent(2));

    const auto jtemp = state.jtemp;
    const auto ftemp = state.ftemp;
    const auto fpress = state.fpress;
    const auto tropo = state.tropo;
    const auto jeta = state.jeta;
    const auto feta = state.feta;

    const auto gpoint_flavor = k.gpoint_flavor;
    const auto band_gpt_start = k.band_gpt_start;
    const auto gpt_band = k.gpt_band;
    const auto krayl = k.krayl;
    const int idx_h2o = k.idx_h2o;

    parallel_for_3d("compute_tau_rayleigh", {0, 0, 0}, {ngpt, nlay, ncol},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int itropo = tropo(ilay, icol) ? 0 : 1;
            const int jt = jtemp(ilay, icol);

            // As for the major species, the flavour comes from the first g-point of
            // this g-point's band.
            const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt)), itropo);

            TF fmin[2][2], fmaj[2][2][2];
            Gas_optics_kernels::interp_weights(
                    ftemp(ilay, icol), fpress(ilay, icol),
                    feta(iflav, 0, ilay, icol), feta(iflav, 1, ilay, icol), fmin, fmaj);

            TF kr = TF(0.);
            for (int itemp=0; itemp<2; ++itemp)
            {
                const int je = jeta(iflav, itemp, ilay, icol);
                for (int ieta=0; ieta<2; ++ieta)
                    kr += fmin[itemp][ieta] * krayl(itropo, igpt, je + ieta, jt + itemp);
            }

            tau_rayleigh(igpt, ilay, icol) =
                    kr * (col_gas(idx_h2o, ilay, icol) + col_dry(ilay, icol));
        });
}


void Gas_optics::compute_planck_source(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& tlay,
        const Array_2d<const TF>& tlev,
        const Array_1d<const TF>& tsfc,
        const int sfc_lay,
        const Source_func_lw& sources)
{
    const int ngpt = static_cast<int>(sources.lay_source.extent(0));
    const int nlay = static_cast<int>(sources.lay_source.extent(1));
    const int ncol = static_cast<int>(sources.lay_source.extent(2));
    const int nlev = nlay + 1;
    const int nplancktemp = static_cast<int>(k.totplnk.extent(1));

    const auto jtemp = state.jtemp;
    const auto ftemp = state.ftemp;
    const auto jpress = state.jpress;
    const auto fpress = state.fpress;
    const auto tropo = state.tropo;
    const auto jeta = state.jeta;
    const auto feta = state.feta;

    const auto gpoint_flavor = k.gpoint_flavor;
    const auto band_gpt_start = k.band_gpt_start;
    const auto gpt_band = k.gpt_band;
    const auto pfracin = k.pfracin;
    const auto totplnk = k.totplnk;
    const TF temp_ref_min = k.temp_ref_min;
    const TF totplnk_delta = k.totplnk_delta;

    // Fraction of each band's Planck irradiance belonging to each g-point. This is the
    // major-species interpolation with unit column mixing.
    Array_3d<TF> pfrac(Kokkos::view_alloc("pfrac", Kokkos::WithoutInitializing), ngpt, nlay, ncol);

    parallel_for_3d("planck_pfrac", {0, 0, 0}, {ngpt, nlay, ncol},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int itropo = tropo(ilay, icol) ? 0 : 1;
            const int jt = jtemp(ilay, icol);
            const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt)), itropo);

            TF fmin[2][2], fmaj[2][2][2];
            Gas_optics_kernels::interp_weights(
                    ftemp(ilay, icol), fpress(ilay, icol),
                    feta(iflav, 0, ilay, icol), feta(iflav, 1, ilay, icol), fmin, fmaj);

            const int jp0 = jpress(ilay, icol) + itropo;

            TF frac = TF(0.);
            for (int itemp=0; itemp<2; ++itemp)
            {
                const int je = jeta(iflav, itemp, ilay, icol);
                for (int ipress=0; ipress<2; ++ipress)
                    for (int ieta=0; ieta<2; ++ieta)
                        frac += fmaj[itemp][ipress][ieta]
                              * pfracin(igpt, jp0 + ipress, je + ieta, jt + itemp);
            }

            pfrac(igpt, ilay, icol) = frac;
        });

    const auto lay_source = sources.lay_source;
    const auto lev_source = sources.lev_source;
    const auto sfc_source = sources.sfc_source;
    const auto sfc_source_jac = sources.sfc_source_jac;

    // The band Planck function is cheap to interpolate, so it is recomputed per
    // g-point rather than stored as (nbnd, nlev, ncol).
    parallel_for_3d("planck_lay_source", {0, 0, 0}, {ngpt, nlay, ncol},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const TF planck = Gas_optics_kernels::interpolate_1d(
                    tlay(ilay, icol), temp_ref_min, totplnk_delta,
                    totplnk, gpt_band(igpt), nplancktemp);

            lay_source(igpt, ilay, icol) = pfrac(igpt, ilay, icol) * planck;
        });

    parallel_for_3d("planck_lev_source", {0, 0, 0}, {ngpt, nlev, ncol},
        KOKKOS_LAMBDA(const int igpt, const int ilev, const int icol)
        {
            const TF planck = Gas_optics_kernels::interpolate_1d(
                    tlev(ilev, icol), temp_ref_min, totplnk_delta,
                    totplnk, gpt_band(igpt), nplancktemp);

            // The two array ends take the adjacent layer's fraction; interior levels
            // take the geometric mean of the layers either side. These are array
            // positions, not physical top and surface, so this is orientation-agnostic.
            TF frac;
            if (ilev == 0)
                frac = pfrac(igpt, 0, icol);
            else if (ilev == nlay)
                frac = pfrac(igpt, nlay - 1, icol);
            else
                frac = Kokkos::sqrt(pfrac(igpt, ilev - 1, icol) * pfrac(igpt, ilev, icol));

            lev_source(igpt, ilev, icol) = frac * planck;
        });

    parallel_for_2d("planck_sfc_source", {0, 0}, {ngpt, ncol},
        KOKKOS_LAMBDA(const int igpt, const int icol)
        {
            const int ibnd = gpt_band(igpt);

            const TF planck = Gas_optics_kernels::interpolate_1d(
                    tsfc(icol), temp_ref_min, totplnk_delta, totplnk, ibnd, nplancktemp);

            // The Jacobian is a one-Kelvin finite difference, as in the reference.
            const TF planck_up = Gas_optics_kernels::interpolate_1d(
                    tsfc(icol) + TF(1.), temp_ref_min, totplnk_delta, totplnk, ibnd, nplancktemp);

            const TF frac = pfrac(igpt, sfc_lay, icol);

            sfc_source(igpt, icol) = frac * planck;
            sfc_source_jac(igpt, icol) = frac * (planck_up - planck);
        });
}
