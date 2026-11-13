#include <stdexcept>
#include <type_traits>

#include "fluxes.h"
#include "gas_optics.h"
#include "optical_props.h"
#include "raytracer.h"
#include "rte_lw.h"
#include "rte_sw.h"
#include "solver.h"


Solver::Cloud_input Solver::make_clouds(
        const Cloud_optics* optics,
        const Array_2d<const TF>& clwp,
        const Array_2d<const TF>& ciwp,
        const Array_2d<const TF>& reliq,
        const Array_2d<const TF>& dgice,
        const bool two_stream,
        const bool delta_scale)
{
    Cloud_input c;
    if (optics == nullptr)
        return c;

    c.optics = optics;
    c.clwp = clwp;
    c.ciwp = ciwp;
    c.reliq = reliq;
    c.dgice = dgice;

    const int nlay = static_cast<int>(clwp.extent(0));
    const int ncol = static_cast<int>(clwp.extent(1));

    // One band's worth; cloud_band overwrites it band by band.
    const auto no_init = Kokkos::WithoutInitializing;
    c.tau = Array_3d<TF>(Kokkos::view_alloc("cloud_tau", no_init), 1, nlay, ncol);

    if (two_stream)
    {
        c.ssa = Array_3d<TF>(Kokkos::view_alloc("cloud_ssa", no_init), 1, nlay, ncol);
        c.g = Array_3d<TF>(Kokkos::view_alloc("cloud_g", no_init), 1, nlay, ncol);
        c.delta_scale = delta_scale;
    }

    return c;
}


Solver::Cloud_band Solver::cloud_band(const Cloud_input& c, const int ibnd)
{
    Cloud_band band;
    if (c.optics == nullptr)
        return band;

    const auto slot = [](const Array_3d<const TF>& a) { return slice_2d(a, 0); };

    if (c.ssa.size() == 0)
    {
        Clouds::compute(*c.optics, c.clwp, c.ciwp, c.reliq, c.dgice, c.tau, ibnd);
        band.tau = slot(c.tau);
        return band;
    }

    Clouds::compute(*c.optics, c.clwp, c.ciwp, c.reliq, c.dgice, c.tau, c.ssa, c.g, ibnd);
    if (c.delta_scale)
        Optical_props::delta_scale_2str(Optical_props_2str{c.tau, c.ssa, c.g});

    band.tau = slot(c.tau);
    band.ssa = slot(c.ssa);
    band.g = slot(c.g);
    return band;
}


Solver::Solve_state Solver::prepare(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool do_lw,
        const bool do_jacobian,
        const Array_1d<const TF>& weights,
        const bool lw_scattering,
        const bool with_transport)
{
    const auto no_init = Kokkos::WithoutInitializing;

    const int nlay = static_cast<int>(atm.play.extent(0));
    const int ncol = static_cast<int>(atm.play.extent(1));
    const int nlev = nlay + 1;
    const int ngas = static_cast<int>(k.gas_names.size());

    Solve_state s;

    s.col_gas = Array_3d<TF>("col_gas", ngas + 1, nlay, ncol);
    Gas_optics::compute_col_gas(
            gas_concs, k.gas_names, atm.plev, atm.col_dry, s.col_gas);

    s.interp = Gas_optics::interpolate(k, atm.play, atm.tlay, s.col_gas);

    // The surface is the layer at whichever end of the array is at higher pressure.
    s.sfc_lay = Gas_optics::is_top_at_1(atm.play) ? nlay - 1 : 0;

    // One g-point's optical properties. The longwave without scattering has no use
    // for ssa and g, and only the longwave has a Planck fraction.
    const bool with_ssa = !do_lw || lw_scattering;

    s.tau = Array_2d<TF>(Kokkos::view_alloc("tau", no_init), nlay, ncol);
    s.ssa = Array_2d<TF>(Kokkos::view_alloc("ssa", no_init),
                         with_ssa ? nlay : 0, with_ssa ? ncol : 0);
    s.g = Array_2d<TF>(Kokkos::view_alloc("g", no_init),
                       with_ssa ? nlay : 0, with_ssa ? ncol : 0);

    if (do_lw)
    {
        s.lay_source = Array_2d<TF>(Kokkos::view_alloc("lay_source", no_init), nlay, ncol);
        s.pfrac = Array_2d<TF>(Kokkos::view_alloc("pfrac", no_init), nlay, ncol);
        s.lev_source = Array_2d<TF>(Kokkos::view_alloc("lev_source", no_init), nlev, ncol);
        s.sfc_source = Array_1d<TF>(Kokkos::view_alloc("sfc_source", no_init), ncol);
        s.sfc_source_jac = Array_1d<TF>(
                Kokkos::view_alloc("sfc_source_jac", no_init), do_jacobian ? ncol : 0);
    }

    // What only the plane-parallel transport needs: a caller that traces every
    // g-point has no use for it.
    if (!with_transport)
        return s;

    s.flux_up = Array_2d<TF>(Kokkos::view_alloc("flux_up_gpt", no_init), nlev, ncol);
    s.flux_dn = Array_2d<TF>(Kokkos::view_alloc("flux_dn_gpt", no_init), nlev, ncol);
    s.flux_dir = Array_2d<TF>(
            Kokkos::view_alloc("flux_dir_gpt", no_init), do_lw ? 0 : nlev, do_lw ? 0 : ncol);

    // The solver's scratch, allocated here so the g-point loop never allocates.
    if (do_lw)
        if (lw_scattering)
            s.lw_2stream = Rte_lw::Two_stream_scratch::make(nlay, ncol);
        else
            s.lw_noscat = Rte_lw::Noscat_scratch::make(ncol, weights);
    else
        s.sw_2stream = Rte_sw::Two_stream_scratch::make(nlay, ncol);

    return s;
}


void Solver::gas_optics_lw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const int igpt)
{
    // The Planck fraction comes out of the same interpolation as the optical depth.
    Gas_optics::compute_tau_lw(
            k, state.interp, atm.play, atm.tlay, state.col_gas, igpt,
            state.tau, state.pfrac);
}


void Solver::gas_optics_sw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const int igpt,
        const bool with_g)
{
    // Absorption, Rayleigh and the combine in one pass; the dry air column Rayleigh
    // needs is index 0 of col_gas, which the kernel reads for itself.
    Gas_optics::compute_tau_sw(
            k, state.interp, atm.play, atm.tlay, state.col_gas, igpt,
            state.tau, state.ssa, with_g ? Array_map_2d<TF>(state.g) : Array_map_2d<TF>());
}


namespace
{
    // The clouds of one band into one g-point's longwave gas optics, in place.
    // Without scattering the clouds only absorb: the solver cannot deflect anything,
    // so they enter as an optical depth and nothing else. With it the gas still only
    // absorbs, so its single-scattering albedo and asymmetry are zero and the cloud's
    // survive the combination unchanged. Zeroed rather than special-cased so that the
    // same tested increment the shortwave uses does the work here too.
    void add_clouds_lw(
            const Solver::Solve_state& state, const Solver::Cloud_band& clouds,
            const bool scattering)
    {
        const bool cloudy = clouds.tau.size() > 0;

        if (!scattering)
        {
            if (cloudy)
                Optical_props::increment_1scalar_by_1scalar(state.tau, clouds.tau);

            return;
        }

        Kokkos::deep_copy(state.ssa, TF(0.));
        Kokkos::deep_copy(state.g, TF(0.));

        if (cloudy)
            Optical_props::increment_2stream_by_2stream(
                    state.tau, state.ssa, state.g, clouds.tau, clouds.ssa, clouds.g);
    }
}


void Solver::solve_lw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const bool top_at_1,
        const int igpt,
        const Array_map_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_map_1d<const TF>& sfc_emis,
        const Array_map_1d<const TF>& inc_flux,
        const Cloud_band& clouds,
        const bool scattering,
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Array_2d<TF>& flux_up_jac)
{
    // The sources come from the Planck fraction the gas optics left.
    Gas_optics::compute_planck_source(
            k, atm.tlay, atm.tlev, atm.tsfc, state.sfc_lay, igpt,
            state.sources(), state.pfrac);

    add_clouds_lw(state, clouds, scattering);

    if (!scattering)
    {
        Rte_lw::solver_noscat(
                top_at_1, secants, weights, state.tau, state.sources(), sfc_emis,
                inc_flux, flux_up, flux_dn, flux_up_jac, state.lw_noscat);

        return;
    }

    Rte_lw::solver_2stream(
            top_at_1, state.tau, state.ssa, state.g, state.sources(), sfc_emis,
            inc_flux, flux_up, flux_dn, state.lw_2stream);
}


void Solver::solve_sw_gpt(
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
        const Cloud_band& clouds,
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Flux_sink& flux_dir)
{
    const auto cloud_tau = clouds.tau;

    const Array_map_2d<TF> tau = state.tau;
    const Array_map_2d<TF> ssa = state.ssa;

    // The asymmetry parameter is zero for a pure gas atmosphere, so it is only worth
    // an array when clouds are going to make it something else; the solver reads an
    // empty g as isotropic. That saves writing and reading a whole (nlay, ncol) array
    // per g-point in the clear-sky case. The gas optics zeroed it if so.
    const Array_map_2d<TF> g = cloud_tau.size() > 0 ? Array_map_2d<TF>(state.g) : Array_map_2d<TF>();

    if (cloud_tau.size() > 0)
        Optical_props::increment_2stream_by_2stream(
                tau, ssa, g,
                cloud_tau, clouds.ssa, clouds.g);

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

    // Band ibnd's clouds into band, unless they are there already: every g-point of a
    // band shares them, and the loops come to a band's g-points one after another.
    void enter_band(
            const Solver::Cloud_input& clouds, const int ibnd,
            int& ibnd_now, Solver::Cloud_band& band)
    {
        if (ibnd == ibnd_now)
            return;

        band = Solver::cloud_band(clouds, ibnd);
        ibnd_now = ibnd;
    }

    template<typename View>
    void zero(const View& a)
    {
        if (a.size() > 0)
            Kokkos::deep_copy(a, TF(0.));
    }
}


void Solver::solve_lw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Array_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_2d<const TF>& sfc_emis,
        const Array_2d<const TF>& inc_flux,
        const Cloud_input& clouds,
        const bool scattering,
        const Fluxes_out& fluxes)
{
    if (scattering && fluxes.up_jac.size() > 0)
        throw std::invalid_argument(
                "The longwave two-stream solver does not produce a surface Jacobian.");

    const Solve_state state = prepare(
            k, gas_concs, atm, true, fluxes.up_jac.size() > 0, weights, scattering);

    zero(fluxes.up);       zero(fluxes.dn);       zero(fluxes.up_jac);
    zero(fluxes.up_byband); zero(fluxes.dn_byband);

    Cloud_band cloud;
    int cloud_ibnd = -1;

    const int ngpt = static_cast<int>(k.kmajor.extent(0));

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        const int ibnd = k.gpt_band_h(igpt);

        gas_optics_lw_gpt(k, state, atm, igpt);
        enter_band(clouds, ibnd, cloud_ibnd, cloud);

        // The solver adds into the totals itself; no g-point flux is written, since
        // the no-scattering solver never reads one back.
        solve_lw_gpt(
                k, state, atm, top_at_1, igpt, secants, weights,
                slice_1d(sfc_emis, igpt),
                inc_flux.size() > 0 ? slice_1d(inc_flux, igpt) : Array_map_1d<const TF>(),
                cloud, scattering,
                sink(ibnd, fluxes.up, fluxes.up_byband),
                sink(ibnd, fluxes.dn, fluxes.dn_byband),
                fluxes.up_jac);
    }
}


void Solver::solve_sw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Array_2d<const TF>& mu0,
        const Array_2d<const TF>& sfc_alb_dir,
        const Array_2d<const TF>& sfc_alb_dif,
        const Array_2d<const TF>& inc_flux_dir,
        const Array_2d<const TF>& inc_flux_dif,
        const Cloud_input& clouds,
        const Fluxes_out& fluxes)
{
    const Solve_state state = prepare(
            k, gas_concs, atm, false, false, Array_1d<const TF>(), false);

    zero(fluxes.up); zero(fluxes.dn); zero(fluxes.dir);
    zero(fluxes.up_byband); zero(fluxes.dn_byband); zero(fluxes.dir_byband);

    // g is only worth zeroing when clouds are going to add to it.
    const bool with_g = clouds.tau.size() > 0;

    Cloud_band cloud;
    int cloud_ibnd = -1;

    const int ngpt = static_cast<int>(k.kmajor.extent(0));

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        const int ibnd = k.gpt_band_h(igpt);

        gas_optics_sw_gpt(k, state, atm, igpt, with_g);
        enter_band(clouds, ibnd, cloud_ibnd, cloud);

        // Only the direct beam keeps a g-point array, which the last kernel reads to
        // put it in the totals; the two diffuse fluxes go straight there.
        solve_sw_gpt(
                k, state, atm, top_at_1, igpt, mu0,
                slice_1d(sfc_alb_dir, igpt), slice_1d(sfc_alb_dif, igpt),
                slice_1d(inc_flux_dir, igpt),
                inc_flux_dif.size() > 0 ? slice_1d(inc_flux_dif, igpt)
                                        : Array_map_1d<const TF>(),
                cloud,
                sink(ibnd, fluxes.up, fluxes.up_byband),
                sink(ibnd, fluxes.dn, fluxes.dn_byband),
                sink(ibnd, fluxes.dir, fluxes.dir_byband, state.flux_dir));
    }
}


void Solver::solve_sw_rt(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Raytracer::Grid& grid,
        const int photons_per_pixel,
        const bool independent_column,
        const TF mu0,
        const TF azi,
        const Array_1d_h<const TF>& toa_src,
        const Array_2d<const TF>& sfc_alb_dir,
        const Cloud_input& clouds,
        const Raytracer::Fluxes_rt& fluxes)
{
    // Every g-point is traced, so there is no plane-parallel solve to prepare for.
    const Solve_state state = prepare(
            k, gas_concs, atm, false, false, Array_1d<const TF>(), false, false);
    const auto scratch = Raytracer::Scratch::make(grid);

    fluxes.zero();

    Cloud_band cloud;
    int cloud_ibnd = -1;

    const int ngpt = static_cast<int>(k.kmajor.extent(0));

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        // Absorption and Rayleigh scattering, as the two-stream path computes them.
        // The asymmetry parameter is not asked for: the gas scatters by the Rayleigh
        // phase function, which the tracer samples directly.
        gas_optics_sw_gpt(k, state, atm, igpt, false);
        enter_band(clouds, k.gpt_band_h(igpt), cloud_ibnd, cloud);

        Raytracer::trace_rays(
                grid, top_at_1, independent_column, photons_per_pixel, igpt,
                state.tau, state.ssa, cloud.tau, cloud.ssa, cloud.g,
                slice_1d(sfc_alb_dir, igpt),
                mu0, azi, toa_src(igpt)*mu0, TF(0.),
                fluxes, scratch);
    }
}


int Solver::solve_lw_rt(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Raytracer_lw::Grid& grid,
        const int photons_per_pixel,
        const bool independent_column,
        const Array_2d<const TF>& sfc_emis,
        const Array_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const TF min_mfp_grid_ratio,
        const Cloud_input& clouds,
        const bool scattering,
        const Raytracer_lw::Fluxes_lw& fluxes)
{
    const int ngpt = static_cast<int>(k.kmajor.extent(0));
    const int nlay = static_cast<int>(atm.play.extent(0));
    const int ncol = static_cast<int>(atm.play.extent(1));

    const Solve_state state = prepare(
            k, gas_concs, atm, true, false, weights, scattering);
    const auto scratch = Raytracer_lw::Scratch::make(grid);

    // The shortest gas mean free path the box may hold before the tracer gives up on
    // it, as an optical depth per cell: a mean free path of ratio*min(dx, dy) is an
    // optical depth of dz/(ratio*min(dx, dy)) across a cell of depth dz.
    const TF d_xy_min = std::min(grid.dx, grid.dy);
    const TF tau_max_traced = min_mfp_grid_ratio > TF(0.)
            ? grid.dz/(min_mfp_grid_ratio*d_xy_min)
            : std::numeric_limits<TF>::infinity();

    // The cells the threshold may ask about: those that stand for one layer each. When
    // the atmosphere runs deeper than the box the top cell is not one of them -- it
    // holds every layer above the box at once, so neither the single layer that shares
    // its index nor its own summed optical depth says anything about whether a photon
    // could cross a *resolved* cell sideways. Leaving it in made the choice of
    // g-points depend on how much atmosphere was handed in above the same box.
    const int nz_scan = nlay > grid.nz ? grid.nz - 1 : grid.nz;

    const int nz = grid.nz;

    // The plane-parallel solves write their g-point's fluxes and nothing else.
    const Flux_sink up{state.flux_up};
    const Flux_sink dn{state.flux_dn};

    // The air lumped into the box's top cell is warm at its bottom and cold at its
    // top, and the tracer's homogeneous cell can only reproduce what it emits out of
    // each end if it is told: so every traced g-point first solves that air on its
    // own, over a black surface that emits nothing, and hands the tracer the flux
    // leaving its bottom and its top. Without scattering that is exact, since what
    // the air emits does not depend on what passes through it; with scattering clouds
    // above the box it leaves out what they would send back of the box's own
    // radiation, which the tracer's lumped cell then scatters itself.
    //
    // The lumped layers are nz-1 upward from the surface: the top nlump of the arrays
    // stored bottom-up, the first nlump stored top-down.
    const bool lumped = nlay > nz;
    const int nlump = nlay - nz + 1;
    const int lump_off = top_at_1 ? 0 : nz - 1;

    const auto lump = [=](const auto& v, const int rows)
    {
        using T = typename std::decay_t<decltype(v)>::value_type;
        return v.size() > 0 ? Array_map_2d<T>(v.data() + lump_off*ncol, rows, ncol)
                            : Array_map_2d<T>();
    };

    // The black, cold floor under the lumped air.
    Array_1d<TF> lump_sfc_source;
    Array_1d<TF> lump_sfc_emis;
    if (lumped)
    {
        lump_sfc_source = Array_1d<TF>("lump_sfc_source", ncol);
        lump_sfc_emis = Array_1d<TF>("lump_sfc_emis", ncol);
        Kokkos::deep_copy(lump_sfc_emis, TF(1.));
    }

    // That solve writes the first nlump+1 rows of the g-point flux arrays, and the two
    // the tracer wants are rows of them: the bottom of the lumped air and its top.
    const Array_map_2d<TF> lump_flux_up(state.flux_up.data(), nlump + 1, ncol);
    const Array_map_2d<TF> lump_flux_dn(state.flux_dn.data(), nlump + 1, ncol);
    const int lump_lev_bot = top_at_1 ? nlump : 0;
    const int lump_lev_top = top_at_1 ? 0 : nlump;
    const auto lump_dn = Array_map_1d<const TF>(
            lumped ? state.flux_dn.data() + lump_lev_bot*ncol : nullptr, lumped ? ncol : 0);
    const auto lump_up = Array_map_1d<const TF>(
            lumped ? state.flux_up.data() + lump_lev_top*ncol : nullptr, lumped ? ncol : 0);

    fluxes.zero();

    int traced = 0;

    Cloud_band cloud;
    int cloud_ibnd = -1;

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        // The sources come out of the gas optics' Planck fraction, exactly as
        // solve_lw_gpt does it.
        gas_optics_lw_gpt(k, state, atm, igpt);
        enter_band(clouds, k.gpt_band_h(igpt), cloud_ibnd, cloud);

        const auto tau = state.tau;

        // The largest gas optical depth across a cell of the resolved box, which is
        // where the shortest mean free path is. Clouds are left out of it, as the
        // reference leaves them out: the threshold asks whether the *gas* alone
        // already closes the cell. Reference: max_tau_gas.
        TF tau_max = TF(0.);
        if (min_mfp_grid_ratio > TF(0.))
        {
            Kokkos::parallel_reduce("lw_rt_max_tau",
                Kokkos::MDRangePolicy<Default_exec, Kokkos::Rank<2>>({0, 0}, {nz_scan, ncol}),
                KOKKOS_LAMBDA(const int kc, const int icol, TF& acc)
                {
                    const TF t = tau(Raytracer::layer_of(kc, nlay, top_at_1), icol);
                    if (t > acc)
                        acc = t;
                },
                Kokkos::Max<TF>(tau_max));
        }

        if (tau_max < tau_max_traced)
        {
            ++traced;

            // The sources come out of the gas optics' Planck fraction, exactly as
            // solve_lw_gpt does it.
            Gas_optics::compute_planck_source(
                    k, atm.tlay, atm.tlev, atm.tsfc, state.sfc_lay, igpt,
                    state.sources(), state.pfrac);

            if (lumped)
            {
                // The clouds go into the optical depth for the solve, and come out
                // again for the tracer, which keeps gas and cloud apart.
                const bool cloudy = cloud.tau.size() > 0;
                add_clouds_lw(state, cloud, scattering);

                const Source_func_lw lump_sources{
                        lump(state.lay_source, nlump), lump(state.lev_source, nlump + 1),
                        Array_map_1d<TF>(lump_sfc_source.data(), ncol), Array_map_1d<TF>()};
                const auto floor_emis = Array_map_1d<const TF>(lump_sfc_emis.data(), ncol);

                if (scattering)
                    Rte_lw::solver_2stream(
                            top_at_1, lump(state.tau, nlump), lump(state.ssa, nlump),
                            lump(state.g, nlump), lump_sources,
                            floor_emis, Array_map_1d<const TF>(),
                            Flux_sink{lump_flux_up}, Flux_sink{lump_flux_dn},
                            state.lw_2stream);
                else
                    Rte_lw::solver_noscat(
                            top_at_1, secants, weights, lump(state.tau, nlump), lump_sources,
                            floor_emis, Array_map_1d<const TF>(),
                            Flux_sink{lump_flux_up}, Flux_sink{lump_flux_dn},
                            Array_2d<TF>(), state.lw_noscat);

                if (cloudy)
                    Gas_optics::compute_tau_lw(
                            k, state.interp, atm.play, atm.tlay, state.col_gas, igpt,
                            state.tau, state.pfrac);
            }

            Raytracer_lw::trace_rays(
                    grid, top_at_1, independent_column, photons_per_pixel, igpt,
                    state.tau, Array_map_2d<const TF>(),
                    cloud.tau, cloud.ssa, cloud.g,
                    state.lay_source, state.sfc_source,
                    slice_1d(sfc_emis, igpt),
                    TF(0.), lump_dn, lump_up,
                    fluxes, scratch);
        }
        else
        {
            // Opaque within a cell: solve the whole column plane-parallel instead,
            // with whichever solver a plane-parallel run of this case would have used,
            // and give the box what that solve has in it. The air above needs no lump
            // here: its cell's net flux is simply what the column loses between the
            // box's top cell and the top of the atmosphere.
            solve_lw_gpt(k, state, atm, top_at_1, igpt, secants, weights,
                         slice_1d(sfc_emis, igpt), Array_map_1d<const TF>(),
                         cloud, scattering, up, dn, Array_2d<TF>());

            Raytracer_lw::add_plane_parallel(
                    grid, top_at_1, nlay, state.flux_up, state.flux_dn, fluxes);
        }
    }

    return traced;
}
