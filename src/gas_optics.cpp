#include <limits>
#include <vector>

#include "fluxes.h"
#include "gas_optics.h"
#include "gas_optics_kernels.h"
#include "optical_props.h"
#include "rte_lw.h"
#include "rte_sw.h"


Interp_state Interp_state::create(
        const int nflav, const int nlay, const int ncol, const bool store_eta)
{
    const auto no_init = Kokkos::WithoutInitializing;

    Interp_state s;
    s.jtemp = Array_2d<int>(Kokkos::view_alloc("jtemp", no_init), nlay, ncol);
    s.ftemp = Array_2d<TF>(Kokkos::view_alloc("ftemp", no_init), nlay, ncol);
    s.jpress = Array_2d<int>(Kokkos::view_alloc("jpress", no_init), nlay, ncol);
    s.fpress = Array_2d<TF>(Kokkos::view_alloc("fpress", no_init), nlay, ncol);
    s.tropo = Array_2d<Bool>(Kokkos::view_alloc("tropo", no_init), nlay, ncol);

    // Empty unless asked for; the kernels rebuild the flavour they need. See the note
    // on Interp_state.
    const int nflav_s = store_eta ? nflav : 0;
    const int nlay_s = store_eta ? nlay : 0;
    const int ncol_s = store_eta ? ncol : 0;

    s.jeta = Array_4d<int>(Kokkos::view_alloc("jeta", no_init), nflav_s, 2, nlay_s, ncol_s);
    s.feta = Array_4d<TF>(Kokkos::view_alloc("feta", no_init), nflav_s, 2, nlay_s, ncol_s);
    s.col_mix = Array_4d<TF>(Kokkos::view_alloc("col_mix", no_init), nflav_s, 2, nlay_s, ncol_s);

    s.lower_limits = Array_2d<int>(Kokkos::view_alloc("lower_limits", no_init), ncol, 2);
    s.upper_limits = Array_2d<int>(Kokkos::view_alloc("upper_limits", no_init), ncol, 2);

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

    // Binary species parameter, per flavour. Only materialised when the caller asked
    // for it: the g-point kernels rebuild the one flavour they need with the same
    // helper, and pay no memory for the other nine.
    if (jeta.size() > 0)
        parallel_for_3d("interpolation_eta", {0, 0, 0}, {nflav, nlay, ncol},
            KOKKOS_LAMBDA(const int iflav, const int ilay, const int icol)
            {
                // itropo = 0 lower atmosphere, 1 upper.
                const int itropo = tropo(ilay, icol) ? 0 : 1;

                TF col_mix_l[2], feta_l[2];
                int jeta_l[2];
                Gas_optics_kernels::eta_interp(
                        flavor, vmr_ref, col_gas, neta, iflav, itropo,
                        jtemp(ilay, icol), ilay, icol, col_mix_l, jeta_l, feta_l);

                for (int itemp=0; itemp<2; ++itemp)
                {
                    col_mix(iflav, itemp, ilay, icol) = col_mix_l[itemp];
                    jeta(iflav, itemp, ilay, icol) = jeta_l[itemp];
                    feta(iflav, itemp, ilay, icol) = feta_l[itemp];
                }
            });

    // Layer limits of the lower and upper atmosphere, per column, which the
    // minor-absorber loop of compute_tau_absorption walks. The reference finds the
    // extreme pressure among the layers on each side of the tropopause and treats
    // everything between there and the domain edge as belonging to that side. For a
    // monotonic pressure profile that is exactly the set of layers where tropo holds,
    // but we reproduce the range form so non-monotonic input matches too.
    //
    // This has no g-point dimension, so it is built here, once, rather than inside the
    // g-point loop that follows.
    const auto lower_limits = state.lower_limits;
    const auto upper_limits = state.upper_limits;

    // top_at_1 is decided from the first column, as in the reference.
    auto play_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, play);
    const bool top_at_1 = play_h(0, 0) < play_h(nlay - 1, 0);

    parallel_for_1d("interpolation_limits", 0, ncol,
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


namespace
{
    // Absorption optical depth, and for the shortwave the Rayleigh scattering that
    // goes with it. One kernel for both because the two read the same interpolation
    // state and, since Rayleigh takes the same flavour as the major species, the same
    // reconstructed weights: the scattering costs one more table interpolation and no
    // extra memory traffic at all.
    //
    // With do_rayleigh, tau comes out as the total extinction and ssa as the Rayleigh
    // fraction of it, and g is zeroed -- what combine_abs_and_rayleigh does in the
    // reference. Without it, ssa and g are unused and tau is absorption alone.
    template<bool do_rayleigh, bool do_pfrac>
    void compute_tau_impl(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,
            const Array_2d<const TF>& tlay,
            const Array_3d<const TF>& col_gas,
            const int igpt,
            const Array_map_2d<TF>& tau,
            const Array_map_2d<TF>& ssa,
            const Array_map_2d<TF>& g,
            const Array_map_2d<TF>& pfrac)
    {
        const bool write_g = g.size() > 0;

        const int nlay = static_cast<int>(tau.extent(0));
        const int ncol = static_cast<int>(tau.extent(1));

        const auto jtemp = state.jtemp;
        const auto ftemp = state.ftemp;
        const auto jpress = state.jpress;
        const auto fpress = state.fpress;
        const auto tropo = state.tropo;
        const auto lower_limits = state.lower_limits;
        const auto upper_limits = state.upper_limits;

        // The binary-species interpolation is rebuilt per flavour rather than read from
        // memory; see the note on Interp_state.
        const auto flavor = k.flavor;
        const auto vmr_ref = k.vmr_ref;
        const int neta = k.neta;

        const auto gpoint_flavor = k.gpoint_flavor;
        const auto band_gpt_start = k.band_gpt_start;
        const auto gpt_band = k.gpt_band;
        const auto kmajor = k.kmajor;
        const int idx_h2o = k.idx_h2o;

        // One of these is empty: krayl in the longwave, pfracin in the shortwave.
        const auto krayl = k.krayl;
        const auto pfracin = k.pfracin;

        const Minor_absorbers lower = k.lower;
        const Minor_absorbers upper = k.upper;

        const char* name = do_rayleigh ? "compute_tau_sw"
                         : do_pfrac ? "compute_tau_lw" : "compute_tau_absorption";

        parallel_for_2d(name, {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol)
            {
                const int itropo = tropo(ilay, icol) ? 0 : 1;
                const int jt = jtemp(ilay, icol);
                const TF ft = ftemp(ilay, icol);
                const TF fp = fpress(ilay, icol);

                TF tau_l;
                TF tau_rayleigh_l = TF(0.);

                // The binary-species interpolation for the flavour in hand. Held across
                // the minor-absorber loop below, which mostly asks for the same flavour
                // again, and rebuilt when it does not.
                TF col_mix[2], feta[2];
                int jeta[2];
                int iflav_have = -1;

                // ---- major species -------------------------------------------------
                // The flavour comes from the first g-point of this g-point's band.
                {
                    const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt)), itropo);

                    Gas_optics_kernels::eta_interp(
                            flavor, vmr_ref, col_gas, neta, iflav, itropo, jt, ilay, icol,
                            col_mix, jeta, feta);
                    iflav_have = iflav;

                    TF fmin[2][2], fmaj[2][2][2];
                    Gas_optics_kernels::interp_weights(ft, fp, feta[0], feta[1], fmin, fmaj);

                    // The reference indexes kmajor at jpress-1 and jpress with a 1-based
                    // jpress+itropo; 0-based that is jpress+itropo and one beyond.
                    const int jp0 = jpress(ilay, icol) + itropo;

                    tau_l = Gas_optics_kernels::interp_major(
                            kmajor, igpt, jp0, jt, jeta, fmaj, col_mix);

                    // The Planck fraction is that same interpolation of another table,
                    // with unit column mixing: this g-point's share of its band's
                    // Planck irradiance.
                    if (do_pfrac)
                    {
                        const TF unit_weight[2] = {TF(1.), TF(1.)};

                        pfrac(ilay, icol) = Gas_optics_kernels::interp_major(
                                pfracin, igpt, jp0, jt, jeta, fmaj, unit_weight);
                    }

                    // A plain if, not if constexpr: nvcc will not let an extended
                    // device lambda first-capture a variable inside a constexpr-if,
                    // and do_rayleigh is a compile-time constant either way, so the
                    // dead branch still costs nothing.
                    if (do_rayleigh)
                    {
                        TF kr = TF(0.);
                        for (int itemp=0; itemp<2; ++itemp)
                        {
                            const int je = jeta[itemp];
                            for (int ieta=0; ieta<2; ++ieta)
                                kr += fmin[itemp][ieta] * krayl(itropo, igpt, je + ieta, jt + itemp);
                        }

                        tau_rayleigh_l = kr * (col_gas(idx_h2o, ilay, icol)
                                               + col_gas(0, ilay, icol));
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

                        if (iflav != iflav_have)
                        {
                            Gas_optics_kernels::eta_interp(
                                    flavor, vmr_ref, col_gas, neta, iflav, itropo, jt,
                                    ilay, icol, col_mix, jeta, feta);
                            iflav_have = iflav;
                        }

                        TF fmin[2][2], fmaj[2][2][2];
                        Gas_optics_kernels::interp_weights(ft, fp, feta[0], feta[1], fmin, fmaj);

                        TF acc = TF(0.);
                        for (int itemp=0; itemp<2; ++itemp)
                        {
                            const int je = jeta[itemp];
                            for (int ieta=0; ieta<2; ++ieta)
                                acc += fmin[itemp][ieta] * m.kminor(ik, je + ieta, jt + itemp);
                        }

                        tau_l += scaling * acc;
                    }
                }

                if (do_rayleigh)
                {
                    // Combine, as combine_abs_and_rayleigh does: tau becomes the total
                    // extinction and ssa the scattering fraction of it. g is zero for a
                    // pure gas atmosphere.
                    const TF t = tau_l + tau_rayleigh_l;

                    tau(ilay, icol) = t;
                    ssa(ilay, icol) = t > TF(2.) * Gas_optics_kernels::tiny()
                            ? tau_rayleigh_l / t : TF(0.);

                    // Zero for a pure gas atmosphere, and left empty when nothing
                    // downstream is going to add to it.
                    if (write_g)
                        g(ilay, icol) = TF(0.);
                }
                else
                    tau(ilay, icol) = tau_l;
            });
    }
}


void Gas_optics::compute_tau_absorption(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const int igpt,
        const Array_map_2d<TF>& tau)
{
    compute_tau_impl<false, false>(
            k, state, play, tlay, col_gas, igpt, tau,
            Array_map_2d<TF>(), Array_map_2d<TF>(), Array_map_2d<TF>());
}


void Gas_optics::compute_tau_lw(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const int igpt,
        const Array_map_2d<TF>& tau,
        const Array_map_2d<TF>& pfrac)
{
    compute_tau_impl<false, true>(
            k, state, play, tlay, col_gas, igpt, tau,
            Array_map_2d<TF>(), Array_map_2d<TF>(), pfrac);
}


void Gas_optics::compute_tau_sw(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const int igpt,
        const Array_map_2d<TF>& tau,
        const Array_map_2d<TF>& ssa,
        const Array_map_2d<TF>& g)
{
    compute_tau_impl<true, false>(
            k, state, play, tlay, col_gas, igpt, tau, ssa, g, Array_map_2d<TF>());
}


void Gas_optics::compute_tau_rayleigh(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& col_dry,
        const Array_3d<const TF>& col_gas,
        const int igpt,
        const Array_map_2d<TF>& tau_rayleigh)
{
    const int nlay = static_cast<int>(tau_rayleigh.extent(0));
    const int ncol = static_cast<int>(tau_rayleigh.extent(1));

    const auto jtemp = state.jtemp;
    const auto ftemp = state.ftemp;
    const auto fpress = state.fpress;
    const auto tropo = state.tropo;

    const auto flavor = k.flavor;
    const auto vmr_ref = k.vmr_ref;
    const int neta = k.neta;

    const auto gpoint_flavor = k.gpoint_flavor;
    const auto band_gpt_start = k.band_gpt_start;
    const auto gpt_band = k.gpt_band;
    const auto krayl = k.krayl;
    const int idx_h2o = k.idx_h2o;

    parallel_for_2d("compute_tau_rayleigh", {0, 0}, {nlay, ncol},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            const int itropo = tropo(ilay, icol) ? 0 : 1;
            const int jt = jtemp(ilay, icol);

            // As for the major species, the flavour comes from the first g-point of
            // this g-point's band.
            const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt)), itropo);

            TF col_mix[2], feta[2];
            int jeta[2];
            Gas_optics_kernels::eta_interp(
                    flavor, vmr_ref, col_gas, neta, iflav, itropo, jt, ilay, icol,
                    col_mix, jeta, feta);

            TF fmin[2][2], fmaj[2][2][2];
            Gas_optics_kernels::interp_weights(
                    ftemp(ilay, icol), fpress(ilay, icol), feta[0], feta[1], fmin, fmaj);

            TF kr = TF(0.);
            for (int itemp=0; itemp<2; ++itemp)
            {
                const int je = jeta[itemp];
                for (int ieta=0; ieta<2; ++ieta)
                    kr += fmin[itemp][ieta] * krayl(itropo, igpt, je + ieta, jt + itemp);
            }

            tau_rayleigh(ilay, icol) =
                    kr * (col_gas(idx_h2o, ilay, icol) + col_dry(ilay, icol));
        });
}


void Gas_optics::compute_pfrac(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_3d<const TF>& col_gas,
        const int igpt,
        const Array_map_2d<TF>& pfrac)
{
    const int nlay = static_cast<int>(pfrac.extent(0));
    const int ncol = static_cast<int>(pfrac.extent(1));

    const auto jtemp = state.jtemp;
    const auto ftemp = state.ftemp;
    const auto jpress = state.jpress;
    const auto fpress = state.fpress;
    const auto tropo = state.tropo;

    const auto flavor = k.flavor;
    const auto vmr_ref = k.vmr_ref;
    const int neta = k.neta;

    const auto gpoint_flavor = k.gpoint_flavor;
    const auto band_gpt_start = k.band_gpt_start;
    const auto gpt_band = k.gpt_band;
    const auto pfracin = k.pfracin;

    parallel_for_2d("planck_pfrac", {0, 0}, {nlay, ncol},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            const int itropo = tropo(ilay, icol) ? 0 : 1;
            const int jt = jtemp(ilay, icol);
            const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt)), itropo);

            TF col_mix[2], feta[2];
            int jeta[2];
            Gas_optics_kernels::eta_interp(
                    flavor, vmr_ref, col_gas, neta, iflav, itropo, jt, ilay, icol,
                    col_mix, jeta, feta);

            TF fmin[2][2], fmaj[2][2][2];
            Gas_optics_kernels::interp_weights(
                    ftemp(ilay, icol), fpress(ilay, icol), feta[0], feta[1], fmin, fmaj);

            const TF unit_weight[2] = {TF(1.), TF(1.)};

            pfrac(ilay, icol) = Gas_optics_kernels::interp_major(
                    pfracin, igpt, jpress(ilay, icol) + itropo, jt, jeta, fmaj, unit_weight);
        });
}


void Gas_optics::compute_planck_source(
        const Kdist_gas& k,
        const Array_2d<const TF>& tlay,
        const Array_2d<const TF>& tlev,
        const Array_1d<const TF>& tsfc,
        const int sfc_lay,
        const int igpt,
        const Source_func_lw& sources,
        const Array_map_2d<const TF>& pfrac)
{
    const int nlay = static_cast<int>(sources.lay_source.extent(0));
    const int ncol = static_cast<int>(sources.lay_source.extent(1));
    const int nlev = nlay + 1;
    const int nplancktemp = static_cast<int>(k.totplnk.extent(1));

    const auto gpt_band = k.gpt_band;
    const auto totplnk = k.totplnk;
    const TF temp_ref_min = k.temp_ref_min;
    const TF totplnk_delta = k.totplnk_delta;

    const auto lay_source = sources.lay_source;
    const auto lev_source = sources.lev_source;
    const auto sfc_source = sources.sfc_source;
    const auto sfc_source_jac = sources.sfc_source_jac;
    const bool do_jacobian = sfc_source_jac.size() > 0;

    // The band Planck function is cheap to interpolate, so it is recomputed per
    // g-point rather than stored as (nbnd, nlev, ncol). One kernel over levels writes
    // both sources: the layer below each level and the level itself read the same two
    // Planck fractions.
    parallel_for_2d("planck_source", {0, 0}, {nlev, ncol},
        KOKKOS_LAMBDA(const int ilev, const int icol)
        {
            const TF planck_lev = Gas_optics_kernels::interpolate_1d(
                    tlev(ilev, icol), temp_ref_min, totplnk_delta,
                    totplnk, gpt_band(igpt), nplancktemp);

            // The two array ends take the adjacent layer's fraction; interior levels
            // take the geometric mean of the layers either side. These are array
            // positions, not physical top and surface, so this is orientation-agnostic.
            TF frac;
            if (ilev == 0)
                frac = pfrac(0, icol);
            else if (ilev == nlay)
                frac = pfrac(nlay - 1, icol);
            else
                frac = Kokkos::sqrt(pfrac(ilev - 1, icol) * pfrac(ilev, icol));

            lev_source(ilev, icol) = frac * planck_lev;

            // The layer at this array position, where there is one; the level loop is
            // one longer.
            if (ilev < nlay)
            {
                const TF planck_lay = Gas_optics_kernels::interpolate_1d(
                        tlay(ilev, icol), temp_ref_min, totplnk_delta,
                        totplnk, gpt_band(igpt), nplancktemp);

                lay_source(ilev, icol) = pfrac(ilev, icol) * planck_lay;
            }
        });

    parallel_for_1d("planck_sfc_source", 0, ncol,
        KOKKOS_LAMBDA(const int icol)
        {
            const TF planck = Gas_optics_kernels::interpolate_1d(
                    tsfc(icol), temp_ref_min, totplnk_delta, totplnk,
                    gpt_band(igpt), nplancktemp);

            const TF frac = pfrac(sfc_lay, icol);
            sfc_source(icol) = frac * planck;

            if (do_jacobian)
            {
                // The Jacobian is a one-Kelvin finite difference, as in the reference.
                const TF planck_up = Gas_optics_kernels::interpolate_1d(
                        tsfc(icol) + TF(1.), temp_ref_min, totplnk_delta,
                        totplnk, gpt_band(igpt), nplancktemp);

                sfc_source_jac(icol) = frac * (planck_up - planck);
            }
        });
}


namespace
{
    // Physical constants, from mo_gas_optics_constants. m_dry and grav are runtime
    // settable in the reference; here they are the documented defaults.
    constexpr TF m_h2o = TF(0.018016);
    constexpr TF m_dry = TF(0.028964);
    constexpr TF avogad = TF(6.02214076e23);
    constexpr TF grav = TF(9.80665);

    // Helmert's formula for the latitude dependence of gravity.
    constexpr TF helmert1 = TF(9.80665);
    constexpr TF helmert2 = TF(0.02586);
    constexpr TF pi_tf = TF(3.14159265358979323846);
}


void Gas_optics::compute_col_dry(
        const Array_2d<const TF>& vmr_h2o,
        const Array_2d<const TF>& plev,
        const Array_1d<const TF>& latitude,
        const Array_2d<TF>& col_dry)
{
    const int nlay = static_cast<int>(col_dry.extent(0));
    const int ncol = static_cast<int>(col_dry.extent(1));

    const bool has_latitude = latitude.size() > 0;

    parallel_for_2d("compute_col_dry", {0, 0}, {nlay, ncol},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            const TF g0 = has_latitude
                    ? helmert1 - helmert2 * Kokkos::cos(TF(2.) * pi_tf * latitude(icol) / TF(180.))
                    : grav;

            const TF delta_plev = Kokkos::abs(plev(ilay, icol) - plev(ilay + 1, icol));

            const TF vmr = vmr_h2o(ilay, icol);
            const TF fact = TF(1.) / (TF(1.) + vmr);
            const TF m_air = (m_dry + m_h2o * vmr) * fact;

            col_dry(ilay, icol) = TF(10.) * delta_plev * avogad * fact
                                / (TF(1000.) * m_air * TF(100.) * g0);
        });
}


void Gas_optics::compute_col_gas(
        const Gas_concs& gas_concs,
        const std::vector<std::string>& gas_names,
        const Array_2d<const TF>& plev,
        const Array_2d<const TF>& col_dry,
        const Array_1d<const TF>& latitude,
        const Array_3d<TF>& col_gas)
{
    const int ngas = static_cast<int>(gas_names.size());
    const int nlay = static_cast<int>(col_gas.extent(1));
    const int ncol = static_cast<int>(col_gas.extent(2));

    if (static_cast<int>(col_gas.extent(0)) != ngas + 1)
        throw std::invalid_argument("col_gas must have ngas+1 entries in its gas dimension.");

    // Gather the mixing ratios in the k-distribution's own gas order.
    Array_3d<TF> vmr("vmr", ngas + 1, nlay, ncol);

    for (int igas=0; igas<ngas; ++igas)
    {
        auto slice = Kokkos::subview(vmr, igas + 1, Kokkos::ALL, Kokkos::ALL);
        Array_2d<TF> gas(Kokkos::view_alloc("vmr_gas", Kokkos::WithoutInitializing), nlay, ncol);

        gas_concs.get_vmr(gas_names[igas], gas);
        Kokkos::deep_copy(slice, gas);
    }

    // Dry air column, supplied or derived from the water vapour profile.
    Array_2d<TF> col_dry_work;
    if (col_dry.size() > 0)
    {
        col_dry_work = Array_2d<TF>(Kokkos::view_alloc("col_dry", Kokkos::WithoutInitializing), nlay, ncol);
        Kokkos::deep_copy(col_dry_work, col_dry);
    }
    else
    {
        col_dry_work = Array_2d<TF>(Kokkos::view_alloc("col_dry", Kokkos::WithoutInitializing), nlay, ncol);

        Array_2d<TF> vmr_h2o(Kokkos::view_alloc("vmr_h2o", Kokkos::WithoutInitializing), nlay, ncol);
        gas_concs.get_vmr("h2o", vmr_h2o);

        compute_col_dry(vmr_h2o, plev, latitude, col_dry_work);
    }

    const Array_3d<const TF> vmr_c = vmr;
    const Array_2d<const TF> col_dry_c = col_dry_work;

    parallel_for_3d("compute_col_gas", {0, 0, 0}, {ngas + 1, nlay, ncol},
        KOKKOS_LAMBDA(const int igas, const int ilay, const int icol)
        {
            // Index 0 is dry air itself.
            col_gas(igas, ilay, icol) = igas == 0
                    ? col_dry_c(ilay, icol)
                    : vmr_c(igas, ilay, icol) * col_dry_c(ilay, icol);
        });
}


namespace
{
    // Run the interpolation for a k-distribution and atmosphere. Shared by the
    // longwave and shortwave entry points.
    Interp_state interpolate_for(
            const Kdist_gas& k,
            const Array_2d<const TF>& play,
            const Array_2d<const TF>& tlay,
            const Array_3d<const TF>& col_gas)
    {
        const int nflav = static_cast<int>(k.flavor.extent(0));
        const int nlay = static_cast<int>(play.extent(0));
        const int ncol = static_cast<int>(play.extent(1));

        Interp_state state = Interp_state::create(nflav, nlay, ncol);

        Gas_optics::interpolation(
                k.flavor, k.press_ref_log, k.temp_ref,
                k.press_ref_log_delta, k.temp_ref_min, k.temp_ref_delta,
                k.press_ref_trop_log, k.neta, k.vmr_ref,
                play, tlay, col_gas, state);

        return state;
    }
}


Gas_optics::Solve_state Gas_optics::prepare(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool do_lw,
        const bool do_jacobian,
        const Array_1d<const TF>& weights)
{
    const auto no_init = Kokkos::WithoutInitializing;

    const int nlay = static_cast<int>(atm.play.extent(0));
    const int ncol = static_cast<int>(atm.play.extent(1));
    const int nlev = nlay + 1;
    const int ngas = static_cast<int>(k.gas_names.size());

    Solve_state s;

    s.col_gas = Array_3d<TF>("col_gas", ngas + 1, nlay, ncol);
    compute_col_gas(gas_concs, k.gas_names, atm.plev, atm.col_dry, Array_1d<TF>(), s.col_gas);

    s.interp = interpolate_for(k, atm.play, atm.tlay, s.col_gas);

    // The surface is the layer at whichever end of the array is at higher pressure.
    auto play_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, atm.play);
    s.sfc_lay = play_h(0, 0) > play_h(nlay - 1, 0) ? 0 : nlay - 1;

    s.tau = Array_2d<TF>(Kokkos::view_alloc("tau", no_init), nlay, ncol);
    s.ssa = Array_2d<TF>(Kokkos::view_alloc("ssa", no_init), nlay, ncol);
    s.g = Array_2d<TF>(Kokkos::view_alloc("g", no_init), nlay, ncol);

    if (do_lw)
    {
        s.lay_source = Array_2d<TF>(Kokkos::view_alloc("lay_source", no_init), nlay, ncol);
        s.pfrac = Array_2d<TF>(Kokkos::view_alloc("pfrac", no_init), nlay, ncol);
        s.lev_source = Array_2d<TF>(Kokkos::view_alloc("lev_source", no_init), nlev, ncol);
        s.sfc_source = Array_1d<TF>(Kokkos::view_alloc("sfc_source", no_init), ncol);
        s.sfc_source_jac = Array_1d<TF>(
                Kokkos::view_alloc("sfc_source_jac", no_init), do_jacobian ? ncol : 0);
    }

    s.flux_up = Array_2d<TF>(Kokkos::view_alloc("flux_up_gpt", no_init), nlev, ncol);
    s.flux_dn = Array_2d<TF>(Kokkos::view_alloc("flux_dn_gpt", no_init), nlev, ncol);
    s.flux_dir = Array_2d<TF>(
            Kokkos::view_alloc("flux_dir_gpt", no_init), do_lw ? 0 : nlev, do_lw ? 0 : ncol);

    // The solver's scratch, allocated here so the g-point loop never allocates.
    if (do_lw)
        s.lw_noscat = Rte_lw::Noscat_scratch::make(nlay, ncol, weights, do_jacobian);
    else
        s.sw_2stream = Rte_sw::Two_stream_scratch::make(nlay, ncol);

    return s;
}


namespace
{
    // This g-point's band slice of a band-resolved property set, or an empty view when
    // the set is absent.
    Array_map_2d<const TF> band_slice(const Array_3d<const TF>& a, const int ibnd)
    {
        if (a.size() == 0)
            return Array_map_2d<const TF>();

        return slice_2d(a, ibnd);
    }
}


void Gas_optics::solve_lw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const bool top_at_1,
        const int igpt,
        const Array_map_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_map_1d<const TF>& sfc_emis,
        const Array_map_1d<const TF>& inc_flux,
        const Band_props& clouds,
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Array_2d<TF>& flux_up_jac)
{
    // The Planck fraction comes out of the same interpolation as the optical depth.
    compute_tau_lw(
            k, state.interp, atm.play, atm.tlay, state.col_gas, igpt,
            state.tau, state.pfrac);

    compute_planck_source(
            k, atm.tlay, atm.tlev, atm.tsfc, state.sfc_lay, igpt,
            state.sources(), state.pfrac);

    // Clouds are absorption-only in the longwave here, matching the all-sky driver:
    // the solver is the no-scattering one, so they enter as an optical depth.
    const int ibnd = k.gpt_band_h(igpt);
    const auto cloud_tau = band_slice(clouds.tau, ibnd);

    if (cloud_tau.size() > 0)
        Optical_props::increment_1scalar_by_1scalar(state.tau, cloud_tau);

    Rte_lw::solver_noscat(
            top_at_1, secants, weights, state.tau, state.sources(), sfc_emis, inc_flux,
            flux_up, flux_dn, flux_up_jac, state.lw_noscat);
}


void Gas_optics::solve_sw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const bool top_at_1,
        const int igpt,
        const Array_2d<const TF>& mu0,
        const Array_map_1d<const TF>& sfc_alb_dir,
        const Array_map_1d<const TF>& sfc_alb_dif,
        const Array_map_1d<const TF>& inc_flux_dir,
        const Array_map_1d<const TF>& inc_flux_dif,
        const Band_props& clouds,
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Flux_sink& flux_dir)
{
    const int ibnd = k.gpt_band_h(igpt);
    const auto cloud_tau = band_slice(clouds.tau, ibnd);

    const auto tau = state.tau;
    const auto ssa = state.ssa;

    // The asymmetry parameter is zero for a pure gas atmosphere, so it is only worth
    // an array when clouds are going to make it something else; the solver reads an
    // empty g as isotropic. That saves writing and reading a whole (nlay, ncol) array
    // per g-point in the clear-sky case.
    const Array_map_2d<TF> g = cloud_tau.size() > 0
            ? Array_map_2d<TF>(state.g) : Array_map_2d<TF>();

    // Absorption, Rayleigh and the combine in one pass; the dry air column Rayleigh
    // needs is index 0 of col_gas, which the kernel reads for itself.
    compute_tau_sw(k, state.interp, atm.play, atm.tlay, state.col_gas, igpt, tau, ssa, g);

    if (cloud_tau.size() > 0)
        Optical_props::increment_2stream_by_2stream(
                tau, ssa, g,
                cloud_tau, band_slice(clouds.ssa, ibnd), band_slice(clouds.g, ibnd));

    Rte_sw::solver_2stream(
            top_at_1, tau, ssa, g, mu0,
            sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif,
            flux_up, flux_dn, flux_dir, state.sw_2stream);
}


namespace
{
    // Where this g-point's flux is to land: the running broadband total, its band's
    // total when one was asked for, and -- only where a solver reads back what it
    // wrote -- a g-point array of its own.
    Flux_sink sink(
            const int ibnd,
            const Array_2d<TF>& broadband,
            const Array_3d<TF>& byband,
            const Array_map_2d<TF>& gpt = Array_map_2d<TF>())
    {
        return Flux_sink{
                gpt, broadband,
                byband.size() > 0 ? slice_2d(byband, ibnd) : Array_map_2d<TF>()};
    }

    void zero(const Array_2d<TF>& a)
    {
        if (a.size() > 0)
            Kokkos::deep_copy(a, TF(0.));
    }

    void zero(const Array_3d<TF>& a)
    {
        if (a.size() > 0)
            Kokkos::deep_copy(a, TF(0.));
    }
}


void Gas_optics::solve_lw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Array_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_2d<const TF>& sfc_emis,
        const Array_2d<const TF>& inc_flux,
        const Band_props& clouds,
        const Fluxes_out& fluxes)
{
    const int ngpt = static_cast<int>(k.kmajor.extent(0));

    const Solve_state state = prepare(
            k, gas_concs, atm, true, fluxes.up_jac.size() > 0, weights);

    zero(fluxes.up);       zero(fluxes.dn);       zero(fluxes.up_jac);
    zero(fluxes.up_byband); zero(fluxes.dn_byband);

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        // The solver adds into the totals itself; no g-point flux is written, since
        // the no-scattering solver never reads one back.
        const int ibnd = k.gpt_band_h(igpt);

        solve_lw_gpt(
                k, state, atm, top_at_1, igpt, secants, weights,
                slice_1d(sfc_emis, igpt),
                inc_flux.size() > 0 ? slice_1d(inc_flux, igpt) : Array_map_1d<const TF>(),
                clouds,
                sink(ibnd, fluxes.up, fluxes.up_byband),
                sink(ibnd, fluxes.dn, fluxes.dn_byband),
                fluxes.up_jac);
    }
}


void Gas_optics::solve_sw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Array_2d<const TF>& mu0,
        const Array_2d<const TF>& sfc_alb_dir,
        const Array_2d<const TF>& sfc_alb_dif,
        const Array_2d<const TF>& inc_flux_dir,
        const Array_2d<const TF>& inc_flux_dif,
        const Band_props& clouds,
        const Fluxes_out& fluxes)
{
    const int ngpt = static_cast<int>(k.kmajor.extent(0));

    const Solve_state state = prepare(k, gas_concs, atm, false);

    zero(fluxes.up); zero(fluxes.dn); zero(fluxes.dir);
    zero(fluxes.up_byband); zero(fluxes.dn_byband); zero(fluxes.dir_byband);

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        // Only the direct beam keeps a g-point array, which the last kernel reads to
        // put it in the totals; the two diffuse fluxes go straight there.
        const int ibnd = k.gpt_band_h(igpt);

        solve_sw_gpt(
                k, state, atm, top_at_1, igpt, mu0,
                slice_1d(sfc_alb_dir, igpt), slice_1d(sfc_alb_dif, igpt),
                slice_1d(inc_flux_dir, igpt),
                inc_flux_dif.size() > 0 ? slice_1d(inc_flux_dif, igpt)
                                        : Array_map_1d<const TF>(),
                clouds,
                sink(ibnd, fluxes.up, fluxes.up_byband),
                sink(ibnd, fluxes.dn, fluxes.dn_byband),
                sink(ibnd, fluxes.dir, fluxes.dir_byband, state.flux_dir));
    }
}


void Gas_optics::gas_optics_lw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& plev,
        const Array_2d<const TF>& tlay,
        const Array_2d<const TF>& tlev,
        const Array_1d<const TF>& tsfc,
        const Array_2d<const TF>& col_dry,
        const Array_3d<TF>& tau,
        const Source_func_lw_spectral& sources)
{
    const int nlay = static_cast<int>(play.extent(0));
    const int ncol = static_cast<int>(play.extent(1));

    Array_3d<TF> col_gas("col_gas", static_cast<int>(k.gas_names.size()) + 1, nlay, ncol);
    compute_col_gas(gas_concs, k.gas_names, plev, col_dry, Array_1d<TF>(), col_gas);

    const Interp_state state = interpolate_for(k, play, tlay, col_gas);

    // The surface is the layer at whichever end of the array is at higher pressure.
    auto play_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, play);
    const int sfc_lay = play_h(0, 0) > play_h(nlay - 1, 0) ? 0 : nlay - 1;

    const int ngpt = static_cast<int>(tau.extent(0));

    // Allocated once, not per g-point.
    Array_2d<TF> pfrac(Kokkos::view_alloc("pfrac", Kokkos::WithoutInitializing), nlay, ncol);

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        compute_tau_lw(k, state, play, tlay, col_gas, igpt, slice_2d(tau, igpt), pfrac);
        compute_planck_source(
                k, tlay, tlev, tsfc, sfc_lay, igpt, sources.gpt(igpt), pfrac);
    }
}


void Gas_optics::gas_optics_sw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& plev,
        const Array_2d<const TF>& tlay,
        const Array_2d<const TF>& col_dry,
        const Array_3d<TF>& tau,
        const Array_3d<TF>& ssa)
{
    const int ngpt = static_cast<int>(tau.extent(0));
    const int nlay = static_cast<int>(play.extent(0));
    const int ncol = static_cast<int>(play.extent(1));

    Array_3d<TF> col_gas("col_gas", static_cast<int>(k.gas_names.size()) + 1, nlay, ncol);
    compute_col_gas(gas_concs, k.gas_names, plev, col_dry, Array_1d<TF>(), col_gas);

    const Interp_state state = interpolate_for(k, play, tlay, col_gas);

    // Rayleigh scattering needs the dry air column, which compute_col_gas put at
    // index 0.
    const auto col_dry_view = Kokkos::subview(col_gas, 0, Kokkos::ALL, Kokkos::ALL);
    Array_2d<TF> col_dry_work(Kokkos::view_alloc("col_dry", Kokkos::WithoutInitializing), nlay, ncol);
    Kokkos::deep_copy(col_dry_work, col_dry_view);

    // One g-point's Rayleigh optical depth, reused every iteration.
    Array_2d<TF> tau_rayleigh(Kokkos::view_alloc("tau_rayleigh", Kokkos::WithoutInitializing),
                              nlay, ncol);
    const Array_2d<const TF> tau_rayleigh_c = tau_rayleigh;

    constexpr TF tiny_tf = std::numeric_limits<TF>::min();

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        const auto tau_g = slice_2d(tau, igpt);
        const auto ssa_g = slice_2d(ssa, igpt);

        compute_tau_absorption(k, state, play, tlay, col_gas, igpt, tau_g);
        compute_tau_rayleigh(k, state, col_dry_work, col_gas, igpt, tau_rayleigh);

        // Combine, as combine_abs_and_rayleigh does: tau is the total extinction and
        // ssa the scattering fraction of it.
        parallel_for_2d("combine_abs_and_rayleigh", {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol)
            {
                const TF t = tau_g(ilay, icol) + tau_rayleigh_c(ilay, icol);

                ssa_g(ilay, icol) = t > TF(2.)*tiny_tf
                        ? tau_rayleigh_c(ilay, icol) / t
                        : TF(0.);
                tau_g(ilay, icol) = t;
            });
    }
}
