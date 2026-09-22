#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gas_optics.h"
#include "gas_optics_kernels.h"


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

    minor_limits_gpt_h = Array_2d_h<int>(
            Kokkos::view_alloc("minor_limits_gpt_h", Kokkos::WithoutInitializing), nminor, 2);
    Kokkos::deep_copy(minor_limits_gpt_h, limits_h);
}


namespace
{
    // The g-point limits of the minor absorbers that can touch the block [igpt0,
    // igpt1): the first and one past the last whose range overlaps it, from the host
    // copy of the limits. The absorbers come in band order, so the span is tight;
    // the kernel still checks each absorber in it.
    void minor_span(
            const Minor_absorbers& m, const int igpt0, const int igpt1, int& lo, int& hi)
    {
        const auto& lim = m.minor_limits_gpt_h;
        const int nminor = static_cast<int>(lim.extent(0));

        lo = nminor;
        hi = 0;

        for (int imnr=0; imnr<nminor; ++imnr)
            if (lim(imnr, 0) < igpt1 && lim(imnr, 1) >= igpt0)
            {
                lo = std::min(lo, imnr);
                hi = imnr + 1;
            }
    }


    // A single g-point's (nlay, ncol) output as a block of one.
    Array_map_3d<TF> as_block(const Array_map_2d<TF>& a)
    {
        return Array_map_3d<TF>(a.data(), a.size() > 0 ? 1 : 0, a.extent(0), a.extent(1));
    }


    // Absorption optical depth, and for the shortwave the Rayleigh scattering that
    // goes with it. One kernel for both because the two read the same interpolation
    // state and, since Rayleigh takes the same flavour as the major species, the same
    // reconstructed weights: the scattering costs one more table interpolation and no
    // extra memory traffic at all.
    //
    // One launch covers a block of g-points [igpt0, igpt1) from a single band. All
    // that does not depend on the g-point -- the interpolation state, the band's
    // binary-species interpolation and weights, and each minor absorber's scaling --
    // is then done once per block instead of once per g-point, and the tables are all
    // that is left to read per g-point. Each g-point still sums its terms in the order
    // the per-g-point kernel did.
    //
    // With do_rayleigh, tau comes out as the total extinction and ssa as the Rayleigh
    // fraction of it, and g is zeroed -- what combine_abs_and_rayleigh does in the
    // reference. Without it, ssa and g are unused and tau is absorption alone.
    //
    // max_blk is the block width the g-point loops are unrolled to; see
    // compute_tau_block, which picks it.
    template<bool do_rayleigh, bool do_pfrac, int max_blk>
    void compute_tau_impl(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,
            const Array_2d<const TF>& tlay,
            const Array_3d<const TF>& col_gas,
            const int igpt0,
            const int igpt1,
            const Array_map_3d<TF>& tau,
            const Array_map_3d<TF>& ssa,
            const Array_map_3d<TF>& g,
            const Array_map_3d<TF>& pfrac)
    {
        const int nblk = igpt1 - igpt0;

        const bool write_g = g.size() > 0;

        const int nlay = static_cast<int>(tau.extent(1));
        const int ncol = static_cast<int>(tau.extent(2));

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

        int lower_lo, lower_hi, upper_lo, upper_hi;
        minor_span(lower, igpt0, igpt1, lower_lo, lower_hi);
        minor_span(upper, igpt0, igpt1, upper_lo, upper_hi);

        const char* name = do_rayleigh ? "compute_tau_sw"
                         : do_pfrac ? "compute_tau_lw" : "compute_tau_absorption";

        // The g-point loops below run to the compile-time max_blk and test against
        // nblk inside, rather than stopping at nblk, so that they unroll and tau_l
        // stays in registers.
        parallel_for_2d(name, {0, 0}, {nlay, ncol},
            KOKKOS_LAMBDA(const int ilay, const int icol)
            {
                const int itropo = tropo(ilay, icol) ? 0 : 1;
                const int jt = jtemp(ilay, icol);
                const TF ft = ftemp(ilay, icol);
                const TF fp = fpress(ilay, icol);

                TF tau_l[max_blk];
                TF tau_rayleigh_l[max_blk];

                // The binary-species interpolation for the flavour in hand. Held across
                // the minor-absorber loop below, which mostly asks for the same flavour
                // again, and rebuilt when it does not.
                TF col_mix[2], feta[2];
                int jeta[2];
                int iflav_have = -1;

                // ---- major species -------------------------------------------------
                // The flavour comes from the first g-point of the block's band.
                {
                    const int iflav = gpoint_flavor(band_gpt_start(gpt_band(igpt0)), itropo);

                    Gas_optics_kernels::eta_interp(
                            flavor, vmr_ref, col_gas, neta, iflav, itropo, jt, ilay, icol,
                            col_mix, jeta, feta);
                    iflav_have = iflav;

                    TF fmin[2][2], fmaj[2][2][2];
                    Gas_optics_kernels::interp_weights(ft, fp, feta[0], feta[1], fmin, fmaj);

                    // The reference indexes kmajor at jpress-1 and jpress with a 1-based
                    // jpress+itropo; 0-based that is jpress+itropo and one beyond.
                    const int jp0 = jpress(ilay, icol) + itropo;

                    // Rayleigh scales with the water vapour and dry air columns.
                    const TF col_rayleigh = do_rayleigh
                            ? col_gas(idx_h2o, ilay, icol) + col_gas(0, ilay, icol) : TF(0.);

                    for (int ig=0; ig<max_blk; ++ig)
                    {
                        if (ig >= nblk)
                            continue;

                        const int igpt = igpt0 + ig;

                        tau_l[ig] = Gas_optics_kernels::interp_major(
                                kmajor, igpt, jp0, jt, jeta, fmaj, col_mix);

                        // The Planck fraction is that same interpolation of another
                        // table, with unit column mixing: this g-point's share of its
                        // band's Planck irradiance.
                        if (do_pfrac)
                        {
                            const TF unit_weight[2] = {TF(1.), TF(1.)};

                            pfrac(ig, ilay, icol) = Gas_optics_kernels::interp_major(
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

                            tau_rayleigh_l[ig] = kr * col_rayleigh;
                        }
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

                    const int imnr_lo = side == 0 ? lower_lo : upper_lo;
                    const int imnr_hi = side == 0 ? lower_hi : upper_hi;

                    // Ascending absorber index, which is the order the per-g-point sum
                    // takes them in.
                    for (int imnr=imnr_lo; imnr<imnr_hi; ++imnr)
                    {
                        const int gpt_lo = m.minor_limits_gpt(imnr, 0);
                        const int gpt_hi = m.minor_limits_gpt(imnr, 1);

                        if (gpt_lo >= igpt1 || gpt_hi < igpt0)
                            continue;

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

                        if (iflav != iflav_have)
                        {
                            Gas_optics_kernels::eta_interp(
                                    flavor, vmr_ref, col_gas, neta, iflav, itropo, jt,
                                    ilay, icol, col_mix, jeta, feta);
                            iflav_have = iflav;
                        }

                        TF fmin[2][2], fmaj[2][2][2];
                        Gas_optics_kernels::interp_weights(ft, fp, feta[0], feta[1], fmin, fmaj);

                        const int ik0 = m.kminor_start(imnr) - gpt_lo;

                        for (int ig=0; ig<max_blk; ++ig)
                        {
                            const int igpt = igpt0 + ig;

                            if (ig >= nblk || igpt < gpt_lo || igpt > gpt_hi)
                                continue;

                            const int ik = ik0 + igpt;

                            TF acc = TF(0.);
                            for (int itemp=0; itemp<2; ++itemp)
                            {
                                const int je = jeta[itemp];
                                for (int ieta=0; ieta<2; ++ieta)
                                    acc += fmin[itemp][ieta] * m.kminor(ik, je + ieta, jt + itemp);
                            }

                            tau_l[ig] += scaling * acc;
                        }
                    }
                }

                for (int ig=0; ig<max_blk; ++ig)
                {
                    if (ig >= nblk)
                        continue;

                    if (do_rayleigh)
                    {
                        // Combine, as combine_abs_and_rayleigh does: tau becomes the
                        // total extinction and ssa the scattering fraction of it. g is
                        // zero for a pure gas atmosphere.
                        const TF t = tau_l[ig] + tau_rayleigh_l[ig];

                        tau(ig, ilay, icol) = t;
                        ssa(ig, ilay, icol) = t > TF(2.) * Gas_optics_kernels::tiny()
                                ? tau_rayleigh_l[ig] / t : TF(0.);

                        // Zero for a pure gas atmosphere, and left empty when nothing
                        // downstream is going to add to it.
                        if (write_g)
                            g(ig, ilay, icol) = TF(0.);
                    }
                    else
                        tau(ig, ilay, icol) = tau_l[ig];
                }
            });
    }


    // Run compute_tau_impl unrolled to the narrowest width that holds the block. The
    // width costs registers whether or not the g-points are there, so a lone g-point
    // must not pay for sixteen.
    template<bool do_rayleigh, bool do_pfrac>
    void compute_tau_block(
            const Kdist_gas& k,
            const Interp_state& state,
            const Array_2d<const TF>& play,
            const Array_2d<const TF>& tlay,
            const Array_3d<const TF>& col_gas,
            const int igpt0,
            const int igpt1,
            const Array_map_3d<TF>& tau,
            const Array_map_3d<TF>& ssa,
            const Array_map_3d<TF>& g,
            const Array_map_3d<TF>& pfrac)
    {
        static_assert(Gas_optics::max_gpt_block == 16);

        const int nblk = igpt1 - igpt0;
        if (nblk < 1 || nblk > Gas_optics::max_gpt_block)
            throw std::invalid_argument("A g-point block must hold 1 to max_gpt_block g-points.");

        // The major flavour is taken per band, so a block must not straddle two.
        if (k.gpt_band_h.size() > 0 && k.gpt_band_h(igpt0) != k.gpt_band_h(igpt1 - 1))
            throw std::invalid_argument("A g-point block must lie within one band.");

        if (nblk == 1)
            compute_tau_impl<do_rayleigh, do_pfrac, 1>(
                    k, state, play, tlay, col_gas, igpt0, igpt1, tau, ssa, g, pfrac);
        else if (nblk <= 4)
            compute_tau_impl<do_rayleigh, do_pfrac, 4>(
                    k, state, play, tlay, col_gas, igpt0, igpt1, tau, ssa, g, pfrac);
        else if (nblk <= 8)
            compute_tau_impl<do_rayleigh, do_pfrac, 8>(
                    k, state, play, tlay, col_gas, igpt0, igpt1, tau, ssa, g, pfrac);
        else
            compute_tau_impl<do_rayleigh, do_pfrac, 16>(
                    k, state, play, tlay, col_gas, igpt0, igpt1, tau, ssa, g, pfrac);
    }
}


std::vector<std::pair<int, int>> Gas_optics::gpt_blocks(const Kdist_gas& k)
{
    const int ngpt = static_cast<int>(k.gpt_band_h.extent(0));

    std::vector<std::pair<int, int>> blocks;

    int igpt0 = 0;
    for (int igpt=1; igpt<=ngpt; ++igpt)
        if (igpt == ngpt || k.gpt_band_h(igpt) != k.gpt_band_h(igpt0)
                || igpt - igpt0 == max_gpt_block)
        {
            blocks.emplace_back(igpt0, igpt);
            igpt0 = igpt;
        }

    return blocks;
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
    compute_tau_block<false, false>(
            k, state, play, tlay, col_gas, igpt, igpt + 1, as_block(tau),
            Array_map_3d<TF>(), Array_map_3d<TF>(), Array_map_3d<TF>());
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
    compute_tau_block<false, true>(
            k, state, play, tlay, col_gas, igpt, igpt + 1, as_block(tau),
            Array_map_3d<TF>(), Array_map_3d<TF>(), as_block(pfrac));
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
    compute_tau_block<true, false>(
            k, state, play, tlay, col_gas, igpt, igpt + 1,
            as_block(tau), as_block(ssa), as_block(g), Array_map_3d<TF>());
}


void Gas_optics::compute_tau_lw_block(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const int igpt0,
        const int igpt1,
        const Array_map_3d<TF>& tau,
        const Array_map_3d<TF>& pfrac)
{
    compute_tau_block<false, true>(
            k, state, play, tlay, col_gas, igpt0, igpt1, tau,
            Array_map_3d<TF>(), Array_map_3d<TF>(), pfrac);
}


void Gas_optics::compute_tau_sw_block(
        const Kdist_gas& k,
        const Interp_state& state,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas,
        const int igpt0,
        const int igpt1,
        const Array_map_3d<TF>& tau,
        const Array_map_3d<TF>& ssa,
        const Array_map_3d<TF>& g)
{
    compute_tau_block<true, false>(
            k, state, play, tlay, col_gas, igpt0, igpt1, tau, ssa, g, Array_map_3d<TF>());
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

    // Helmert's gravity is a function of the column alone, so it is built once per
    // column here rather than once per cell inside the kernel. On the host, and
    // deliberately: it is the only transcendental gas optics evaluates on the device,
    // and cosf's argument-reduction fallback is double precision, which a
    // single-precision build has no business emitting. This file cannot be compiled
    // with fast math the way the ray tracer is, because its kernels are judged against
    // the Fortran reference at 1e-12.
    //
    // Empty unless the caller gave a latitude, so the common path allocates nothing.
    Array_1d<TF> g0;
    if (has_latitude)
    {
        g0 = Array_1d<TF>(Kokkos::view_alloc("g0", Kokkos::WithoutInitializing), ncol);

        auto g0_h = Kokkos::create_mirror_view(g0);
        auto lat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, latitude);

        for (int icol=0; icol<ncol; ++icol)
            g0_h(icol) = helmert1 - helmert2*std::cos(TF(2.)*pi_tf*lat_h(icol)/TF(180.));

        Kokkos::deep_copy(g0, g0_h);
    }

    const Array_1d<const TF> g0_c = g0;

    parallel_for_2d("compute_col_dry", {0, 0}, {nlay, ncol},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            const TF g0 = has_latitude ? g0_c(icol) : grav;

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


Interp_state Gas_optics::interpolate(
        const Kdist_gas& k,
        const Array_2d<const TF>& play,
        const Array_2d<const TF>& tlay,
        const Array_3d<const TF>& col_gas)
{
    const int nflav = static_cast<int>(k.flavor.extent(0));
    const int nlay = static_cast<int>(play.extent(0));
    const int ncol = static_cast<int>(play.extent(1));

    Interp_state state = Interp_state::create(nflav, nlay, ncol);

    interpolation(
            k.flavor, k.press_ref_log, k.temp_ref,
            k.press_ref_log_delta, k.temp_ref_min, k.temp_ref_delta,
            k.press_ref_trop_log, k.neta, k.vmr_ref,
            play, tlay, col_gas, state);

    return state;
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

    const Interp_state state = interpolate(k, play, tlay, col_gas);

    // The surface is the layer at whichever end of the array is at higher pressure.
    auto play_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, play);
    const int sfc_lay = play_h(0, 0) > play_h(nlay - 1, 0) ? 0 : nlay - 1;

    // One block's Planck fractions, allocated once.
    Array_3d<TF> pfrac(Kokkos::view_alloc("pfrac", Kokkos::WithoutInitializing),
                       max_gpt_block, nlay, ncol);

    for (const auto& [igpt0, igpt1] : gpt_blocks(k))
    {
        compute_tau_lw_block(
                k, state, play, tlay, col_gas, igpt0, igpt1,
                slice_block(tau, igpt0, igpt1), slice_block(pfrac, 0, igpt1 - igpt0));

        for (int igpt=igpt0; igpt<igpt1; ++igpt)
            compute_planck_source(
                    k, tlay, tlev, tsfc, sfc_lay, igpt, sources.gpt(igpt),
                    slice_2d(pfrac, igpt - igpt0));
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
    const int nlay = static_cast<int>(play.extent(0));
    const int ncol = static_cast<int>(play.extent(1));

    Array_3d<TF> col_gas("col_gas", static_cast<int>(k.gas_names.size()) + 1, nlay, ncol);
    compute_col_gas(gas_concs, k.gas_names, plev, col_dry, Array_1d<TF>(), col_gas);

    const Interp_state state = interpolate(k, play, tlay, col_gas);

    // Absorption, Rayleigh and their combination in one pass, as the solver does it.
    for (const auto& [igpt0, igpt1] : gpt_blocks(k))
        compute_tau_sw_block(
                k, state, play, tlay, col_gas, igpt0, igpt1,
                slice_block(tau, igpt0, igpt1), slice_block(ssa, igpt0, igpt1),
                Array_map_3d<TF>());
}
