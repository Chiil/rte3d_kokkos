#include <stdexcept>

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
        const Array_2d<const TF>& reice,
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
    c.reice = reice;

    const int nspec = static_cast<int>(optics->lut_extliq.extent(0));
    const int nlay = static_cast<int>(clwp.extent(0));
    const int ncol = static_cast<int>(clwp.extent(1));

    const auto no_init = Kokkos::WithoutInitializing;
    c.tau = Array_3d<TF>(Kokkos::view_alloc("cloud_tau", no_init), nspec, nlay, ncol);

    if (two_stream)
    {
        c.ssa = Array_3d<TF>(Kokkos::view_alloc("cloud_ssa", no_init), nspec, nlay, ncol);
        c.g = Array_3d<TF>(Kokkos::view_alloc("cloud_g", no_init), nspec, nlay, ncol);
        c.delta_scale = delta_scale;
    }

    return c;
}


Solver::Band_props Solver::cloud_props(const Cloud_input& c)
{
    Band_props props;
    if (c.optics == nullptr)
        return props;

    if (c.ssa.size() == 0)
    {
        Clouds::compute(*c.optics, c.clwp, c.ciwp, c.reliq, c.reice, c.tau);
        props.tau = c.tau;
        return props;
    }

    Clouds::compute(*c.optics, c.clwp, c.ciwp, c.reliq, c.reice, c.tau, c.ssa, c.g);
    if (c.delta_scale)
        Optical_props::delta_scale_2str(Optical_props_2str{c.tau, c.ssa, c.g});

    props.tau = c.tau;
    props.ssa = c.ssa;
    props.g = c.g;
    return props;
}


Solver::Solve_state Solver::prepare(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool do_lw,
        const bool do_jacobian,
        const Array_1d<const TF>& weights,
        const bool lw_scattering)
{
    const auto no_init = Kokkos::WithoutInitializing;

    const int nlay = static_cast<int>(atm.play.extent(0));
    const int ncol = static_cast<int>(atm.play.extent(1));
    const int nlev = nlay + 1;
    const int ngas = static_cast<int>(k.gas_names.size());

    Solve_state s;

    s.col_gas = Array_3d<TF>("col_gas", ngas + 1, nlay, ncol);
    Gas_optics::compute_col_gas(
            gas_concs, k.gas_names, atm.plev, atm.col_dry, Array_1d<TF>(), s.col_gas);

    s.interp = Gas_optics::interpolate(k, atm.play, atm.tlay, s.col_gas);

    // The surface is the layer at whichever end of the array is at higher pressure.
    auto play_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, atm.play);
    s.sfc_lay = play_h(0, 0) > play_h(nlay - 1, 0) ? 0 : nlay - 1;

    // One block of g-points. The longwave without scattering has no use for ssa and
    // g, and only the longwave has a Planck fraction.
    const int nblk = Gas_optics::max_gpt_block;
    const bool with_ssa = !do_lw || lw_scattering;

    s.tau = Array_3d<TF>(Kokkos::view_alloc("tau", no_init), nblk, nlay, ncol);
    s.ssa = Array_3d<TF>(Kokkos::view_alloc("ssa", no_init),
                         with_ssa ? nblk : 0, with_ssa ? nlay : 0, with_ssa ? ncol : 0);
    s.g = Array_3d<TF>(Kokkos::view_alloc("g", no_init),
                       with_ssa ? nblk : 0, with_ssa ? nlay : 0, with_ssa ? ncol : 0);

    if (do_lw)
    {
        s.lay_source = Array_2d<TF>(Kokkos::view_alloc("lay_source", no_init), nlay, ncol);
        s.pfrac = Array_3d<TF>(Kokkos::view_alloc("pfrac", no_init), nblk, nlay, ncol);
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
        if (lw_scattering)
            s.lw_2stream = Rte_lw::Two_stream_scratch::make(nlay, ncol);
        else
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


void Solver::gas_optics_lw_block(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const int igpt0,
        const int igpt1)
{
    // The Planck fraction comes out of the same interpolation as the optical depth.
    Gas_optics::compute_tau_lw_block(
            k, state.interp, atm.play, atm.tlay, state.col_gas, igpt0, igpt1,
            slice_block(state.tau, 0, igpt1 - igpt0),
            slice_block(state.pfrac, 0, igpt1 - igpt0));
}


void Solver::gas_optics_sw_block(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const int igpt0,
        const int igpt1,
        const bool with_g)
{
    // Absorption, Rayleigh and the combine in one pass; the dry air column Rayleigh
    // needs is index 0 of col_gas, which the kernel reads for itself.
    Gas_optics::compute_tau_sw_block(
            k, state.interp, atm.play, atm.tlay, state.col_gas, igpt0, igpt1,
            slice_block(state.tau, 0, igpt1 - igpt0),
            slice_block(state.ssa, 0, igpt1 - igpt0),
            with_g ? slice_block(state.g, 0, igpt1 - igpt0) : Array_map_3d<TF>());
}


void Solver::solve_lw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const bool top_at_1,
        const int igpt,
        const int islot,
        const Array_map_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_map_1d<const TF>& sfc_emis,
        const Array_map_1d<const TF>& inc_flux,
        const Band_props& clouds,
        const bool scattering,
        const Flux_sink& flux_up,
        const Flux_sink& flux_dn,
        const Array_2d<TF>& flux_up_jac)
{
    const auto opt = state.optics(islot);

    // The sources come from the Planck fraction the block's gas optics left.
    Gas_optics::compute_planck_source(
            k, atm.tlay, atm.tlev, atm.tsfc, state.sfc_lay, igpt,
            state.sources(), opt.pfrac);

    const int ibnd = k.gpt_band_h(igpt);
    const auto cloud_tau = band_slice(clouds.tau, ibnd);

    if (!scattering)
    {
        // Clouds are absorption-only on this path: the solver cannot deflect anything,
        // so they enter as an optical depth and nothing else.
        if (cloud_tau.size() > 0)
            Optical_props::increment_1scalar_by_1scalar(opt.tau, cloud_tau);

        Rte_lw::solver_noscat(
                top_at_1, secants, weights, opt.tau, state.sources(), sfc_emis,
                inc_flux, flux_up, flux_dn, flux_up_jac, state.lw_noscat);

        return;
    }

    // With scattering the gas still only absorbs, so its single-scattering albedo and
    // asymmetry are zero and the cloud's survive the combination unchanged. Zeroed
    // rather than special-cased so that the same tested increment the shortwave uses
    // does the work here too.
    Kokkos::deep_copy(opt.ssa, TF(0.));
    Kokkos::deep_copy(opt.g, TF(0.));

    if (cloud_tau.size() > 0)
        Optical_props::increment_2stream_by_2stream(
                opt.tau, opt.ssa, opt.g,
                cloud_tau, band_slice(clouds.ssa, ibnd), band_slice(clouds.g, ibnd));

    Rte_lw::solver_2stream(
            top_at_1, opt.tau, opt.ssa, opt.g, state.sources(), sfc_emis,
            inc_flux, flux_up, flux_dn, state.lw_2stream);
}


void Solver::solve_sw_gpt(
        const Kdist_gas& k,
        const Solve_state& state,
        const Atmosphere& atm,
        const bool top_at_1,
        const int igpt,
        const int islot,
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

    const auto opt = state.optics(islot);
    const auto tau = opt.tau;
    const auto ssa = opt.ssa;

    // The asymmetry parameter is zero for a pure gas atmosphere, so it is only worth
    // an array when clouds are going to make it something else; the solver reads an
    // empty g as isotropic. That saves writing and reading a whole (nlay, ncol) array
    // per g-point in the clear-sky case. The block's gas optics zeroed it if so.
    const Array_map_2d<TF> g = cloud_tau.size() > 0 ? opt.g : Array_map_2d<TF>();

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


void Solver::solve_lw(
        const Kdist_gas& k,
        const Gas_concs& gas_concs,
        const Atmosphere& atm,
        const bool top_at_1,
        const Array_2d<const TF>& secants,
        const Array_1d<const TF>& weights,
        const Array_2d<const TF>& sfc_emis,
        const Array_2d<const TF>& inc_flux,
        const Band_props& clouds,
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

    for (const auto& [igpt0, igpt1] : Gas_optics::gpt_blocks(k))
    {
        gas_optics_lw_block(k, state, atm, igpt0, igpt1);

        for (int igpt=igpt0; igpt<igpt1; ++igpt)
        {
            // The solver adds into the totals itself; no g-point flux is written,
            // since the no-scattering solver never reads one back.
            const int ibnd = k.gpt_band_h(igpt);

            solve_lw_gpt(
                    k, state, atm, top_at_1, igpt, igpt - igpt0, secants, weights,
                    slice_1d(sfc_emis, igpt),
                    inc_flux.size() > 0 ? slice_1d(inc_flux, igpt) : Array_map_1d<const TF>(),
                    clouds, scattering,
                    sink(ibnd, fluxes.up, fluxes.up_byband),
                    sink(ibnd, fluxes.dn, fluxes.dn_byband),
                    fluxes.up_jac);
        }
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
        const Band_props& clouds,
        const Fluxes_out& fluxes)
{
    const Solve_state state = prepare(k, gas_concs, atm, false);

    zero(fluxes.up); zero(fluxes.dn); zero(fluxes.dir);
    zero(fluxes.up_byband); zero(fluxes.dn_byband); zero(fluxes.dir_byband);

    // g is only worth zeroing when clouds are going to add to it.
    const bool with_g = clouds.tau.size() > 0;

    for (const auto& [igpt0, igpt1] : Gas_optics::gpt_blocks(k))
    {
        gas_optics_sw_block(k, state, atm, igpt0, igpt1, with_g);

        for (int igpt=igpt0; igpt<igpt1; ++igpt)
        {
            // Only the direct beam keeps a g-point array, which the last kernel reads
            // to put it in the totals; the two diffuse fluxes go straight there.
            const int ibnd = k.gpt_band_h(igpt);

            solve_sw_gpt(
                    k, state, atm, top_at_1, igpt, igpt - igpt0, mu0,
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
        const Band_props& clouds,
        const Raytracer::Fluxes_rt& fluxes)
{
    const Solve_state state = prepare(k, gas_concs, atm, false);
    const auto scratch = Raytracer::Scratch::make(grid);

    fluxes.zero();

    for (const auto& [igpt0, igpt1] : Gas_optics::gpt_blocks(k))
    {
        // Absorption and Rayleigh scattering, as the two-stream path computes them.
        // The asymmetry parameter is not asked for: the gas scatters by the Rayleigh
        // phase function, which the tracer samples directly.
        gas_optics_sw_block(k, state, atm, igpt0, igpt1, false);

        for (int igpt=igpt0; igpt<igpt1; ++igpt)
        {
            const int ibnd = k.gpt_band_h(igpt);
            const auto opt = state.optics(igpt - igpt0);

            Raytracer::trace_rays(
                    grid, top_at_1, independent_column, photons_per_pixel, igpt,
                    opt.tau, opt.ssa,
                    band_slice(clouds.tau, ibnd), band_slice(clouds.ssa, ibnd),
                    band_slice(clouds.g, ibnd),
                    slice_1d(sfc_alb_dir, igpt),
                    mu0, azi, toa_src(igpt)*mu0, TF(0.),
                    fluxes, scratch);
        }
    }
}


namespace
{
    // Collapse the layers above the ray tracer's box into the box's top layer, in
    // place, so that a plane-parallel solve sees the very column the tracer walks.
    //
    // The box is already a contiguous slab of the layer arrays -- the bottom nz of them
    // when the caller stores bottom-up, the top nz when it stores top-down -- so all
    // this has to write is the one row at the box's top, plus the level above it. Both
    // arrays are rebuilt from the tables for every g-point, so there is nothing in them
    // to preserve.
    //
    // The Planck source of the collapsed layer is weighted by absorption optical depth,
    // which is what Raytracer_lw's own bundle_emission_tod sums: it is the product that
    // is summed over the lumped layers, not the factors. That is what makes the two
    // paths the same physical problem rather than two approximations of it.
    void lump_column_into_box(
            const int nz, const int nlay, const int ncol, const bool top_at_1,
            const bool scattering,
            const Array_map_2d<TF>& tau,
            const Array_map_2d<TF>& ssa,
            const Array_map_2d<TF>& g,
            const Array_map_2d<TF>& lay_source,
            const Array_map_2d<TF>& lev_source)
    {
        const int off = top_at_1 ? nlay - nz : 0;
        const int ilump = Raytracer::layer_of(nz - 1, nlay, top_at_1);

        parallel_for_1d("lw_rt_lump_column", 0, ncol,
            KOKKOS_LAMBDA(const int icol)
            {
                TF tau_sum = TF(0.);
                TF src_sum = TF(0.);
                TF sca_sum = TF(0.);
                TF scag_sum = TF(0.);

                for (int k=nz-1; k<nlay; ++k)
                {
                    const int ilay = Raytracer::layer_of(k, nlay, top_at_1);
                    const TF t = tau(ilay, icol);

                    tau_sum += t;
                    src_sum += t*lay_source(ilay, icol);

                    if (scattering)
                    {
                        // Optical depth adds, but the other two are weighted means:
                        // the albedo by optical depth and the asymmetry by the part of
                        // it that scatters. The same weighting bundle_optics_tod gives
                        // the tracer's own top cell.
                        const TF sca = t*ssa(ilay, icol);

                        sca_sum += sca;
                        scag_sum += sca*g(ilay, icol);
                    }
                }

                tau(ilump, icol) = tau_sum;
                lay_source(ilump, icol) = tau_sum > TF(0.) ? src_sum/tau_sum : TF(0.);

                if (scattering)
                {
                    ssa(ilump, icol) = tau_sum > TF(0.) ? sca_sum/tau_sum : TF(0.);
                    g(ilump, icol) = sca_sum > TF(0.) ? scag_sum/sca_sum : TF(0.);
                }

                // The collapsed layer is one homogeneous slab, exactly as the tracer
                // treats it, so both its edges carry the same source. Its lower edge is
                // shared with the resolved layer below, whose top source this therefore
                // also sets -- see the note in solve_lw_rt.
                // The solvers take the source as linear in optical depth between a
                // layer's two edges. Give both edges the weighted source and the
                // collapsed layer radiates as one isothermal slab, which is what the
                // tracer's homogeneous top cell is; leave the profile alone and it
                // runs from the air at the top of the box to the top of the atmosphere,
                // radiating to space at stratospheric temperature and 52 W/m2 short.
                //
                // The inner edge is shared with the resolved layer below, so setting it
                // costs that layer its own top source. The trade is worth taking and
                // was measured either way: what leaves through the top does not depend
                // on the inner edge at all -- a slab this opaque never shows it to
                // space -- but what the slab sends *down* does, and leaving it true put
                // 15 W/m2 of the cooling in the wrong cells just under the box's top.
                // Setting it cuts that to 5.
                lev_source(off + nz, icol) = lay_source(ilump, icol);
                lev_source(off + nz - 1, icol) = lay_source(ilump, icol);
            });
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
        const Band_props& clouds,
        const bool scattering,
        const bool lump_above,
        const Raytracer_lw::Fluxes_lw& fluxes)
{
    const int ngpt = static_cast<int>(k.kmajor.extent(0));
    const int nlay = static_cast<int>(atm.play.extent(0));
    const int ncol = static_cast<int>(atm.play.extent(1));

    const Solve_state state = prepare(
            k, gas_concs, atm, true, false, weights, scattering);
    const auto scratch = Raytracer_lw::Scratch::make(grid);

    // The gas does not scatter in the longwave, and the tracer wants that as an array
    // rather than as a special case. Allocated once and left at zero.
    const Array_2d<TF> ssa_gas("lw_rt_ssa_gas", nlay, ncol);

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

    // Where the box's slab of the layer arrays begins. Stored bottom-up the box is the
    // bottom nz layers; stored top-down it is the top nz, at the far end of the array.
    const int nz = grid.nz;
    const int lay_off = top_at_1 ? nlay - nz : 0;

    // An atmosphere deeper than the box, with the caller asking for it to stay outside
    // rather than be lumped into the box's top cell. Every g-point is then solved
    // plane-parallel over the full column first, and what that solve leaves coming
    // down at the box's top is what enters the box -- the reference's arrangement.
    const bool truncate = !lump_above && nlay > nz;

    // The level where the box ends, in the full column's own numbering.
    const int lev_box_top = top_at_1 ? nlay - nz : nz;

    fluxes.zero();

    int traced = 0;

    const auto blocks = Gas_optics::gpt_blocks(k);
    int iblk = -1;
    int igpt0 = 0;

    for (int igpt=0; igpt<ngpt; ++igpt)
    {
        const int ibnd = k.gpt_band_h(igpt);

        // The gas optics of the block igpt falls in, on entering it. The sources come
        // out of its Planck fraction, exactly as solve_lw_gpt does it.
        if (iblk < 0 || igpt == blocks[iblk].second)
        {
            ++iblk;
            igpt0 = blocks[iblk].first;
            gas_optics_lw_block(k, state, atm, igpt0, blocks[iblk].second);
        }

        const auto opt = state.optics(igpt - igpt0);
        const auto tau = opt.tau;

        Gas_optics::compute_planck_source(
                k, atm.tlay, atm.tlev, atm.tsfc, state.sfc_lay, igpt,
                state.sources(), opt.pfrac);

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

        // With the air above the box left outside it, the column is solved
        // plane-parallel first, whatever becomes of the box afterwards: the tracer
        // needs what that solve leaves coming down at the box's top, and a g-point
        // that is not traced needs the solve anyway.
        TF inc_dif = TF(0.);

        if (truncate)
        {
            solve_lw_gpt(k, state, atm, top_at_1, igpt, igpt - igpt0, secants, weights,
                         slice_1d(sfc_emis, igpt), Array_map_1d<const TF>(),
                         clouds, scattering,
                         Flux_sink{state.flux_up, Array_map_2d<TF>(), Array_map_2d<TF>()},
                         Flux_sink{state.flux_dn, Array_map_2d<TF>(), Array_map_2d<TF>()},
                         Array_2d<TF>());

            // That solve incremented the clouds into the optical depth, and the tracer
            // takes the two apart: it has to know which of gas and cloud deflected a
            // photon. So put the gas back, which costs one interpolation.
            Gas_optics::compute_tau_lw(
                    k, state.interp, atm.play, atm.tlay, state.col_gas, igpt,
                    opt.tau, opt.pfrac);

            // One number for the whole box, as the tracer takes it and as the
            // reference feeds it: the mean over the columns of what arrives at the
            // box's top.
            const auto flux_dn = state.flux_dn;
            TF sum = TF(0.);

            Kokkos::parallel_reduce("lw_rt_inc_dif",
                Kokkos::RangePolicy<Default_exec>(0, ncol),
                KOKKOS_LAMBDA(const int icol, TF& acc)
                {
                    acc += flux_dn(lev_box_top, icol);
                },
                sum);

            inc_dif = sum/TF(ncol);
        }

        if (tau_max < tau_max_traced)
        {
            ++traced;

            // Only the box's own layers are handed over when the air above stays
            // outside, so that the tracer has nothing left to lump.
            const auto box_rows = [=](const Array_map_2d<const TF>& v)
            {
                if (!truncate || v.size() == 0)
                    return v;

                return Array_map_2d<const TF>(v.data() + lay_off*ncol, nz, ncol);
            };

            Raytracer_lw::trace_rays(
                    grid, top_at_1, independent_column, photons_per_pixel, igpt,
                    box_rows(opt.tau), box_rows(Array_map_2d<const TF>(
                            ssa_gas.data(), ssa_gas.extent(0), ssa_gas.extent(1))),
                    box_rows(band_slice(clouds.tau, ibnd)),
                    box_rows(band_slice(clouds.ssa, ibnd)),
                    box_rows(band_slice(clouds.g, ibnd)),
                    box_rows(state.lay_source), state.sfc_source,
                    slice_1d(sfc_emis, igpt),
                    inc_dif,
                    fluxes, scratch);
        }
        else if (truncate)
        {
            // The full column is already solved; the box is the part of it the tracer
            // would have covered.
            Raytracer_lw::add_plane_parallel(
                    grid, top_at_1, nz, false,
                    Array_map_2d<const TF>(state.flux_up.data() + lay_off*ncol, nz + 1, ncol),
                    Array_map_2d<const TF>(state.flux_dn.data() + lay_off*ncol, nz + 1, ncol),
                    fluxes);
        }
        else
        {
            // Opaque within a cell: solve the column instead, with whichever solver a
            // plane-parallel run of this case would have used.
            const auto cloud_tau = band_slice(clouds.tau, ibnd);
            const Flux_sink up{state.flux_up, Array_map_2d<TF>(), Array_map_2d<TF>()};
            const Flux_sink dn{state.flux_dn, Array_map_2d<TF>(), Array_map_2d<TF>()};

            if (scattering)
            {
                Kokkos::deep_copy(opt.ssa, TF(0.));
                Kokkos::deep_copy(opt.g, TF(0.));

                if (cloud_tau.size() > 0)
                    Optical_props::increment_2stream_by_2stream(
                            opt.tau, opt.ssa, opt.g, cloud_tau,
                            band_slice(clouds.ssa, ibnd), band_slice(clouds.g, ibnd));
            }
            else if (cloud_tau.size() > 0)
                Optical_props::increment_1scalar_by_1scalar(opt.tau, cloud_tau);

            // Clouds first, then the collapse, so that what weights the lumped Planck
            // source is the total absorption -- the same quantity bundle_emission
            // weights the tracer's own emission by -- and so that the albedo and the
            // asymmetry being combined are the ones the solver will read.
            if (nlay > nz)
                lump_column_into_box(nz, nlay, ncol, top_at_1, scattering,
                                     opt.tau, opt.ssa, opt.g,
                                     state.lay_source, state.lev_source);

            const auto box_2d = [=](const Array_map_2d<TF>& v, const int rows)
            { return Array_map_2d<TF>(v.data() + lay_off*ncol, rows, ncol); };

            const Source_func_lw box_sources{
                    box_2d(state.lay_source, nz), box_2d(state.lev_source, nz + 1),
                    state.sfc_source, Array_map_1d<TF>()};

            if (scattering)
                Rte_lw::solver_2stream(
                        top_at_1, box_2d(opt.tau, nz), box_2d(opt.ssa, nz),
                        box_2d(opt.g, nz), box_sources,
                        slice_1d(sfc_emis, igpt), Array_map_1d<const TF>(),
                        up, dn, state.lw_2stream);
            else
                Rte_lw::solver_noscat(
                        top_at_1, secants, weights, box_2d(opt.tau, nz), box_sources,
                        slice_1d(sfc_emis, igpt), Array_map_1d<const TF>(),
                        up, dn, Array_2d<TF>(), state.lw_noscat);

            // The solve covered the box and nothing else, so as far as the conversion
            // is concerned the atmosphere is exactly nz layers deep.
            Raytracer_lw::add_plane_parallel(
                    grid, top_at_1, nz, nlay > nz,
                    Array_map_2d<const TF>(state.flux_up.data() + lay_off*ncol, nz + 1, ncol),
                    Array_map_2d<const TF>(state.flux_dn.data() + lay_off*ncol, nz + 1, ncol),
                    fluxes);
        }
    }

    return traced;
}
