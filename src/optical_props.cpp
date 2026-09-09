#include <limits>

#include "optical_props.h"


namespace
{
    // The reference guards its divisions with eps = 3*tiny(1.0), three times the
    // smallest normal number. Note this is tiny, not epsilon: it only prevents
    // division by zero, it is not a rounding threshold. A function rather than a
    // namespace-scope constant so it is usable inside device lambdas: a host
    // constexpr variable is ODR-used (its address taken) when passed by reference
    // to Kokkos::max, which nvcc rejects in device code.
    KOKKOS_INLINE_FUNCTION constexpr TF eps() { return TF(3.) * std::numeric_limits<TF>::min(); }
}


Array_1d<int> Optical_props::identity_map(const int ngpt)
{
    Array_1d<int> map(Kokkos::view_alloc("gpt2set", Kokkos::WithoutInitializing), ngpt);

    parallel_for_1d("identity_map", 0, ngpt, KOKKOS_LAMBDA(const int igpt) { map(igpt) = igpt; });

    return map;
}


Array_1d<int> Optical_props::band_map(const Array_2d<const int>& gpt_lims, const int ngpt)
{
    const int nbnd = static_cast<int>(gpt_lims.extent(0));

    auto lims_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gpt_lims);

    Array_1d<int> map(Kokkos::view_alloc("gpt2set", Kokkos::WithoutInitializing), ngpt);
    auto map_h = Kokkos::create_mirror_view(map);

    for (int ibnd=0; ibnd<nbnd; ++ibnd)
        for (int igpt=lims_h(ibnd, 0); igpt<=lims_h(ibnd, 1); ++igpt)
            map_h(igpt) = ibnd;

    Kokkos::deep_copy(map, map_h);

    return map;
}


void Optical_props::delta_scale_2str(const Optical_props_2str& props)
{
    const auto& tau = props.tau;
    const auto& ssa = props.ssa;
    const auto& g = props.g;

    parallel_for_3d("delta_scale_2str", {0, 0, 0}, {static_cast<int64_t>(tau.extent(0)), static_cast<int64_t>(tau.extent(1)), static_cast<int64_t>(tau.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            // Forward-scattering fraction is the square of the asymmetry parameter.
            const TF f = g(igpt, ilay, icol) * g(igpt, ilay, icol);
            const TF wf = ssa(igpt, ilay, icol) * f;

            tau(igpt, ilay, icol) = (TF(1.) - wf) * tau(igpt, ilay, icol);
            ssa(igpt, ilay, icol) = (ssa(igpt, ilay, icol) - wf) / Kokkos::max(eps(), TF(1.) - wf);
            g  (igpt, ilay, icol) = (g  (igpt, ilay, icol) - f ) / Kokkos::max(eps(), TF(1.) - f );
        });
}


void Optical_props::delta_scale_2str_f(const Optical_props_2str& props, const Array_3d<const TF>& f)
{
    const auto& tau = props.tau;
    const auto& ssa = props.ssa;
    const auto& g = props.g;

    parallel_for_3d("delta_scale_2str_f", {0, 0, 0}, {static_cast<int64_t>(tau.extent(0)), static_cast<int64_t>(tau.extent(1)), static_cast<int64_t>(tau.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const TF f_l = f(igpt, ilay, icol);
            const TF wf = ssa(igpt, ilay, icol) * f_l;

            tau(igpt, ilay, icol) = (TF(1.) - wf) * tau(igpt, ilay, icol);
            ssa(igpt, ilay, icol) = (ssa(igpt, ilay, icol) - wf ) / Kokkos::max(eps(), TF(1.) - wf );
            g  (igpt, ilay, icol) = (g  (igpt, ilay, icol) - f_l) / Kokkos::max(eps(), TF(1.) - f_l);
        });
}


void Optical_props::increment_1scalar_by_1scalar(
        const Array_3d<TF>& tau1,
        const Array_3d<const TF>& tau2,
        const Array_1d<const int>& gpt2set)
{
    parallel_for_3d("increment_1scalar_by_1scalar", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            tau1(igpt, ilay, icol) += tau2(gpt2set(igpt), ilay, icol);
        });
}


void Optical_props::increment_1scalar_by_2stream(
        const Array_3d<TF>& tau1,
        const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2,
        const Array_1d<const int>& gpt2set)
{
    parallel_for_3d("increment_1scalar_by_2stream", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int i2 = gpt2set(igpt);
            tau1(igpt, ilay, icol) += tau2(i2, ilay, icol) * (TF(1.) - ssa2(i2, ilay, icol));
        });
}


void Optical_props::increment_2stream_by_1scalar(
        const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1,
        const Array_3d<const TF>& tau2,
        const Array_1d<const int>& gpt2set)
{
    parallel_for_3d("increment_2stream_by_1scalar", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const TF tau12 = tau1(igpt, ilay, icol) + tau2(gpt2set(igpt), ilay, icol);

            ssa1(igpt, ilay, icol) = tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol)
                                   / Kokkos::max(eps(), tau12);
            tau1(igpt, ilay, icol) = tau12;
        });
}


void Optical_props::increment_2stream_by_2stream(
        const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_3d<TF>& g1,
        const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_3d<const TF>& g2,
        const Array_1d<const int>& gpt2set)
{
    parallel_for_3d("increment_2stream_by_2stream", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int i2 = gpt2set(igpt);

            const TF tau12 = tau1(igpt, ilay, icol) + tau2(i2, ilay, icol);
            const TF tauscat12 = tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol)
                               + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol);

            g1(igpt, ilay, icol) =
                    (tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol) * g1(igpt, ilay, icol)
                     + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol) * g2(i2, ilay, icol))
                    / Kokkos::max(eps(), tauscat12);

            ssa1(igpt, ilay, icol) = tauscat12 / Kokkos::max(eps(), tau12);
            tau1(igpt, ilay, icol) = tau12;
        });
}


void Optical_props::increment_1scalar_by_1scalar(
        const Array_map_2d<TF>& tau1,
        const Array_map_2d<const TF>& tau2)
{
    parallel_for_2d("increment_1scalar_by_1scalar_gpt", {0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1))},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            tau1(ilay, icol) += tau2(ilay, icol);
        });
}


void Optical_props::increment_2stream_by_2stream(
        const Array_map_2d<TF>& tau1, const Array_map_2d<TF>& ssa1,
        const Array_map_2d<TF>& g1,
        const Array_map_2d<const TF>& tau2, const Array_map_2d<const TF>& ssa2,
        const Array_map_2d<const TF>& g2)
{
    parallel_for_2d("increment_2stream_by_2stream_gpt", {0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1))},
        KOKKOS_LAMBDA(const int ilay, const int icol)
        {
            const TF tau12 = tau1(ilay, icol) + tau2(ilay, icol);
            const TF tauscat12 = tau1(ilay, icol) * ssa1(ilay, icol)
                               + tau2(ilay, icol) * ssa2(ilay, icol);

            g1(ilay, icol) =
                    (tau1(ilay, icol) * ssa1(ilay, icol) * g1(ilay, icol)
                     + tau2(ilay, icol) * ssa2(ilay, icol) * g2(ilay, icol))
                    / Kokkos::max(eps(), tauscat12);

            ssa1(ilay, icol) = tauscat12 / Kokkos::max(eps(), tau12);
            tau1(ilay, icol) = tau12;
        });
}


void Optical_props::increment_2stream_by_nstream(
        const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_3d<TF>& g1,
        const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_4d<const TF>& p2,
        const Array_1d<const int>& gpt2set)
{
    parallel_for_3d("increment_2stream_by_nstream", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int i2 = gpt2set(igpt);

            const TF tau12 = tau1(igpt, ilay, icol) + tau2(i2, ilay, icol);
            const TF tauscat12 = tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol)
                               + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol);

            // The first phase function moment is the asymmetry parameter.
            g1(igpt, ilay, icol) =
                    (tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol) * g1(igpt, ilay, icol)
                     + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol) * p2(i2, ilay, icol, 0))
                    / Kokkos::max(eps(), tauscat12);

            ssa1(igpt, ilay, icol) = tauscat12 / Kokkos::max(eps(), tau12);
            tau1(igpt, ilay, icol) = tau12;
        });
}


void Optical_props::increment_nstream_by_2stream(
        const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_4d<TF>& p1,
        const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_3d<const TF>& g2,
        const Array_1d<const int>& gpt2set)
{
    const int nmom1 = static_cast<int>(p1.extent(3));

    parallel_for_3d("increment_nstream_by_2stream", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int i2 = gpt2set(igpt);

            const TF tau12 = tau1(igpt, ilay, icol) + tau2(i2, ilay, icol);
            const TF tauscat12 = tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol)
                               + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol);

            // A two-stream phase function expands to moments g, g^2, g^3, ...
            const TF g2_l = g2(i2, ilay, icol);
            TF moment = TF(1.);

            for (int imom=0; imom<nmom1; ++imom)
            {
                moment *= g2_l;
                p1(igpt, ilay, icol, imom) =
                        (tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol) * p1(igpt, ilay, icol, imom)
                         + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol) * moment)
                        / Kokkos::max(eps(), tauscat12);
            }

            ssa1(igpt, ilay, icol) = tauscat12 / Kokkos::max(eps(), tau12);
            tau1(igpt, ilay, icol) = tau12;
        });
}


void Optical_props::increment_nstream_by_nstream(
        const Array_3d<TF>& tau1, const Array_3d<TF>& ssa1, const Array_4d<TF>& p1,
        const Array_3d<const TF>& tau2, const Array_3d<const TF>& ssa2, const Array_4d<const TF>& p2,
        const Array_1d<const int>& gpt2set)
{
    // Moments beyond the shorter of the two expansions are left untouched, as in the
    // reference.
    const int mom_lim = Kokkos::min(static_cast<int>(p1.extent(3)), static_cast<int>(p2.extent(3)));

    parallel_for_3d("increment_nstream_by_nstream", {0, 0, 0},
        {static_cast<int64_t>(tau1.extent(0)), static_cast<int64_t>(tau1.extent(1)), static_cast<int64_t>(tau1.extent(2))},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            const int i2 = gpt2set(igpt);

            const TF tau12 = tau1(igpt, ilay, icol) + tau2(i2, ilay, icol);
            const TF tauscat12 = tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol)
                               + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol);

            for (int imom=0; imom<mom_lim; ++imom)
                p1(igpt, ilay, icol, imom) =
                        (tau1(igpt, ilay, icol) * ssa1(igpt, ilay, icol) * p1(igpt, ilay, icol, imom)
                         + tau2(i2, ilay, icol) * ssa2(i2, ilay, icol) * p2(i2, ilay, icol, imom))
                        / Kokkos::max(eps(), tauscat12);

            ssa1(igpt, ilay, icol) = tauscat12 / Kokkos::max(eps(), tau12);
            tau1(igpt, ilay, icol) = tau12;
        });
}


void Optical_props::extract_subset(
        const Array_3d<const TF>& array_in, const int col_start, const int col_end,
        const Array_3d<TF>& array_out)
{
    parallel_for_3d("extract_subset_3d", {0, 0, col_start},
        {static_cast<int64_t>(array_in.extent(0)), static_cast<int64_t>(array_in.extent(1)), col_end},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            array_out(igpt, ilay, icol - col_start) = array_in(igpt, ilay, icol);
        });
}


void Optical_props::extract_subset(
        const Array_4d<const TF>& array_in, const int col_start, const int col_end,
        const Array_4d<TF>& array_out)
{
    const int nmom = static_cast<int>(array_in.extent(3));

    parallel_for_3d("extract_subset_4d", {0, 0, col_start},
        {static_cast<int64_t>(array_in.extent(0)), static_cast<int64_t>(array_in.extent(1)), col_end},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            for (int imom=0; imom<nmom; ++imom)
                array_out(igpt, ilay, icol - col_start, imom) = array_in(igpt, ilay, icol, imom);
        });
}


void Optical_props::extract_subset_absorption_tau(
        const Array_3d<const TF>& tau_in, const Array_3d<const TF>& ssa_in,
        const int col_start, const int col_end,
        const Array_3d<TF>& tau_out)
{
    parallel_for_3d("extract_subset_absorption_tau", {0, 0, col_start},
        {static_cast<int64_t>(tau_in.extent(0)), static_cast<int64_t>(tau_in.extent(1)), col_end},
        KOKKOS_LAMBDA(const int igpt, const int ilay, const int icol)
        {
            tau_out(igpt, ilay, icol - col_start) =
                    tau_in(igpt, ilay, icol) * (TF(1.) - ssa_in(igpt, ilay, icol));
        });
}
