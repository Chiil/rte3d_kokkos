"""Every solver rte3d has, run over one atmosphere and compared against each other.

Seven solvers, four in the longwave and three in the shortwave:

| band | solver | what it is |
|---|---|---|
| sw | two-stream | the plane-parallel adding solver, a flux profile per column |
| sw | ray tracing 1D | the Monte Carlo tracer with horizontal transport switched off |
| sw | ray tracing 3D | the same tracer, photons free to move sideways |
| lw | no-scattering | the multi-angle plane-parallel solver, clouds absorb and emit |
| lw | two-stream | the plane-parallel solver with cloud scattering |
| lw | ray tracing 1D | the tracer, column by column |
| lw | ray tracing 3D | the tracer over the whole domain |

They are run on two atmospheres, which is what makes the comparison say something. The
RCEMIP sounding holds the same profile in every column, so the third dimension has
nothing to act on and the two tracers must agree to round-off; the LES cloud field is
genuinely three-dimensional, so they must not, and the direction and size of the
difference is the check. Both cases are the ones in `cases/`, at a size that keeps the
suite quick: RCEMIP over 8 x 8 x 64 instead of 64 x 64 x 256, and the cloud field
cropped to a 32 x 32 window of its 128 x 128.

Everything with a photon in it is Monte Carlo, so the tolerances here are set by the
photon budget rather than by round-off, except where two runs are the same arithmetic
in a different order and the tolerance says so.
"""
import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')
CASES = os.path.join(os.path.dirname(__file__), '..', 'cases')

LES_TOML = os.path.join(CASES, 'les_cloudfield', 'les_cloudfield.toml')
LES_INPUT = os.path.join(CASES, 'les_cloudfield', 'les_cloudfield_input.nc')

# The RCEMIP sounding, at a size that says everything the committed 64 x 64 x 256 case
# would: every column holds the same profile, so the horizontal is there for the tracer
# to move photons through and for nothing else.
NX = NY = 8
NLAY = 64

# The window cropped out of the cloud field. The tracer's cost is the domain, so a
# quarter of it in each direction is a sixteenth of the work.
WINDOW = 32

# Photons per pixel per g-point. Enough that a domain-mean flux is good to a few
# tenths of a percent, which is what the tolerances below are written against.
PHOTONS = 256
LES_PHOTONS = 64

# Skip the g-points whose gas is opaque within a grid cell and solve those
# plane-parallel, which is what the cloud field's own settings ask for and what
# rte-rrtmgp-cpp defaults to. It is not an economy here so much as a condition for the
# comparison to mean anything: with 1.1 km cells most of the longwave spectrum cannot
# cross one, so those g-points carry no horizontal transport to resolve and tracing
# them is all variance. Traced in full, the domain-mean downward flux at the surface
# of the RCEMIP case wanders over 393 to 509 W/m2 between photon budgets from 64 to
# 4096; with the threshold on, the same budgets give 431 to 436.
MIN_MFP_GRID_RATIO = 1.0

requires_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, 'rrtmgp-gas-lw-g256.nc')),
    reason='rrtmgp-data submodule not checked out')

requires_les = pytest.mark.skipif(
    not os.path.exists(LES_INPUT),
    reason='cases/les_cloudfield/les_cloudfield_input.nc not fetched; '
           'run python cases/fetch_data.py les_cloudfield')


def permutation_rtol(rte3d):
    """Tolerance for two sums of the same numbers in a different order.

    A horizontally uniform atmosphere makes the three-dimensional trace a permutation
    of the independent-column one: a photon that leaves its pixel arrives in a column
    identical to the one it left, so the set of scores is the same and only which
    column holds which changes. The counters the tracer scores into are fixed point,
    so their order does not matter; what is left is the order the columns are added
    in the mean, and the absorbed weight, which a photon holds until it leaves a cell:
    leaving sideways cuts that into different pieces than leaving up or down, and
    each piece is rounded to the counters' 2^-32. Measured on this case, the two runs
    agree to 2e-4 in single precision; in double the boundary fluxes agree exactly
    and the absorption to 1e-10, which is that rounding.
    """
    return 1e-2 if rte3d.runtime().precision == 'single' else 1e-9


def case_scripts():
    """cases/ on the path: the scripts a user runs, imported as modules."""
    import sys

    sys.path.insert(0, CASES)

    import make_input
    import run_case

    return make_input, run_case


def solve_everything(rte3d, atm, files, photons, delta_cloud=True,
                     min_mfp_grid_ratio=0.0, gpt_block=1):
    """Run all seven solvers over one atmosphere and return their output by name.

    The two bands each load their own k-distribution and cloud coefficients once, and
    every solver of that band works from them, so what differs between the entries
    below is the transport and nothing else. gpt_block is the gas optics' g-point
    block width, passed to every solver.
    """
    from rte3d.case import (gpoint_bands, make_gas_concs, solve_lw, solve_lw_rt,
                            solve_sw, solve_sw_rt)
    from rte3d.kdist import read_cloud_optics, read_kdist

    out = {}

    for band in ('lw', 'sw'):
        f = read_kdist(files[f'gas_{band}'])
        gas_concs = make_gas_concs(rte3d, atm, f['gas_names'])
        kdist = rte3d.load_kdist(f, gas_concs)
        gpt_band = gpoint_bands(f)
        cloud_optics = rte3d.load_cloud_optics(
            read_cloud_optics(files[f'cloud_{band}']), files['icergh'])

        args = (rte3d, kdist, gas_concs, atm, gpt_band, cloud_optics)

        if band == 'lw':
            out['lw_noscat'] = solve_lw(*args, delta_cloud=delta_cloud,
                                        gpt_block=gpt_block)
            out['lw_2str'] = solve_lw(*args, scattering=True, delta_cloud=delta_cloud,
                                      gpt_block=gpt_block)

            for name, icol in (('lw_rt_1d', True), ('lw_rt_3d', False)):
                out[name] = solve_lw_rt(*args, photons_per_pixel=photons,
                                        independent_column=icol,
                                        min_mfp_grid_ratio=min_mfp_grid_ratio,
                                        delta_cloud=delta_cloud, gpt_block=gpt_block)
        else:
            out['sw_2str'] = solve_sw(*args, delta_cloud=delta_cloud, gpt_block=gpt_block)

            for name, icol in (('sw_rt_1d', True), ('sw_rt_3d', False)):
                out[name] = solve_sw_rt(*args, delta_cloud=delta_cloud,
                                        photons_per_pixel=photons,
                                        independent_column=icol, gpt_block=gpt_block)

        out[f'ngpt_{band}'] = kdist.ngpt

    return out


def default_files():
    """The coefficient files a case gets when its settings name none."""
    _, run_case = case_scripts()

    files = dict(run_case.FILES)
    for name in ('gas_lw', 'gas_sw', 'cloud_lw', 'cloud_sw'):
        files[name] = run_case.coefficients(name, files)

    return files


@pytest.fixture(scope='session')
def rcemip(rte3d, tmp_path_factory):
    """The RCEMIP sounding with its two cloud layers, solved seven ways.

    Session-scoped: the seven solves are one atmosphere's worth of work and every test
    below reads a different part of the same answer.
    """
    from rte3d.case import read_case

    make_input, _ = case_scripts()

    path = make_input.make_input(
        str(tmp_path_factory.mktemp('rcemip')/'rcemip_input.nc'),
        nx=NX, ny=NY, nlay=NLAY, clouds=True)

    atm = read_case(path)
    return atm, solve_everything(rte3d, atm, default_files(), PHOTONS,
                                 min_mfp_grid_ratio=MIN_MFP_GRID_RATIO)


@pytest.fixture(scope='session')
def les(rte3d, tmp_path_factory):
    """The LES cumulus field, cropped to its cloudiest window and solved seven ways.

    The settings come from the case's own .toml, so this runs the case as it is
    committed -- its k-distributions, its cloud coefficients, its delta scaling and its
    mean-free-path threshold -- and differs from a run of it only in the solvers it
    asks for and the size of the domain.
    """
    import xarray as xr

    from rte3d.case import read_case

    _, run_case = case_scripts()

    switches, files, _, longwave = run_case.read_settings(LES_TOML)
    for name in ('gas_lw', 'gas_sw', 'cloud_lw', 'cloud_sw'):
        files[name] = run_case.coefficients(name, files)

    d = xr.open_dataset(LES_INPUT)

    # Cloud covers one cell in eighty of this field, so an arbitrary window of it may
    # hold almost none. The cloudiest of the sixteen non-overlapping windows is where
    # the solvers have the most to disagree about, and picking it by the field's own
    # water path keeps the choice reproducible.
    lwp = d['lwp'].sum('lay').values
    blocks = lwp.reshape(lwp.shape[0]//WINDOW, WINDOW,
                         lwp.shape[1]//WINDOW, WINDOW).sum(axis=(1, 3))
    j, i = (n*WINDOW for n in np.unravel_index(blocks.argmax(), blocks.shape))

    sub = d.isel(x=slice(i, i + WINDOW), xh=slice(i, i + WINDOW + 1),
                 y=slice(j, j + WINDOW), yh=slice(j, j + WINDOW + 1))

    # The null-collision grid cannot be coarser than the domain it blocks up.
    for name in ('ngrid_x', 'ngrid_y'):
        sub[name] = np.int32(min(int(d[name]), WINDOW))

    path = str(tmp_path_factory.mktemp('les')/'les_input.nc')
    sub.to_netcdf(path)

    atm = read_case(path)
    return atm, solve_everything(rte3d, atm, files, LES_PHOTONS,
                                 delta_cloud=switches['delta_cloud'],
                                 min_mfp_grid_ratio=longwave['min_mfp_grid_ratio'])


def mean(field):
    """The domain mean of a scored flux, which is what a photon budget resolves."""
    return float(np.mean(np.asarray(field)))


def sw_budget(out, dz):
    """What enters the shortwave domain, and what leaves it or stays [W/m2]."""
    entering = mean(out['rt_flux_tod_dn'])
    leaving = (mean(out['rt_flux_tod_up']) + mean(out['rt_flux_sfc_dir'])
               + mean(out['rt_flux_sfc_dif']) - mean(out['rt_flux_sfc_up']))
    absorbed = dz*mean((out['rt_flux_abs_dir'] + out['rt_flux_abs_dif']).sum(axis=0))

    return entering, leaving + absorbed


# ---------------------------------------------------------------------------
# The RCEMIP sounding: every column the same, so the third dimension is a no-op.
# ---------------------------------------------------------------------------

@requires_data
def test_rcemip_runs_every_solver(rcemip):
    """All seven produce fluxes of the right shape, and none of them is nonsense."""
    atm, out = rcemip

    for name in ('sw_2str', 'lw_noscat', 'lw_2str'):
        assert out[name]['solve_time'] > 0.0
        for key, flux in out[name].items():
            if key == 'solve_time':
                continue

            assert flux.shape == (atm['nlev'], atm['ncol'])
            assert np.all(np.isfinite(flux))
            assert np.all(flux >= 0.0)

    for name in ('sw_rt_1d', 'sw_rt_3d'):
        assert out[name]['rt_flux_sfc_dir'].shape == (atm['ncol'],)
        assert out[name]['rt_flux_abs_dir'].shape == (atm['grid']['nz'], atm['ncol'])

    for name in ('lw_rt_1d', 'lw_rt_3d'):
        assert out[name]['rt_lw_flux_sfc_dn'].shape == (atm['ncol'],)
        assert out[name]['rt_lw_flux_abs'].shape == (atm['grid']['nz'], atm['ncol'])

        # Part of the spectrum went plane-parallel, and part of it was traced.
        assert 0 < int(out[name]['n_gpt_traced']) < out['ngpt_lw']


@requires_data
@pytest.mark.parametrize('gpt_block', [4, 16])
def test_rcemip_gpt_block_width_changes_nothing(rcemip, rte3d, gpt_block):
    """A wider g-point block only shares the gas optics' g-point-independent work.

    Each g-point's optical depth is the same arithmetic whatever block it sits in, so
    every solver, the tracers' photon walks included, gives the same bits.
    """
    atm, out = rcemip
    wide = solve_everything(rte3d, atm, default_files(), PHOTONS,
                            min_mfp_grid_ratio=MIN_MFP_GRID_RATIO, gpt_block=gpt_block)

    for name, fluxes in out.items():
        if not isinstance(fluxes, dict):
            continue

        for key, flux in fluxes.items():
            if key != 'solve_time':
                np.testing.assert_array_equal(wide[name][key], flux, err_msg=f'{name} {key}')


@requires_data
@pytest.mark.parametrize('solver', ['sw_rt_1d', 'sw_rt_3d'])
def test_rcemip_shortwave_tracer_closes_its_energy_budget(rcemip, solver):
    """In at the top, out at the boundaries or absorbed on the way."""
    atm, out = rcemip

    entering, accounted = sw_budget(out[solver], atm['grid']['dz'])
    assert accounted == pytest.approx(entering, rel=0.01)


@requires_data
@pytest.mark.parametrize('band', ['sw', 'lw'])
def test_rcemip_uniform_field_makes_the_tracers_agree(rcemip, rte3d, band):
    """With the same profile everywhere, horizontal transport changes nothing.

    A photon that wanders sideways arrives in a column identical to the one it left,
    so the three-dimensional trace scores the same values as the independent-column
    one and merely holds them in different pixels. The domain means are then the same
    sum in a different order -- see permutation_rtol -- which is a far tighter
    statement than any photon budget could make, and one that fails the moment the
    two transports stop solving the same problem.
    """
    _, out = rcemip

    icol, three_d = out[f'{band}_rt_1d'], out[f'{band}_rt_3d']
    rtol = permutation_rtol(rte3d)

    for name, flux in icol.items():
        if name in ('n_gpt_traced', 'solve_time'):
            continue

        scale = np.abs(np.asarray(flux)).max()
        assert mean(three_d[name]) == pytest.approx(mean(flux), rel=rtol,
                                                    abs=rtol*scale), name

    # And they are not trivially the same array: the columns really were shuffled.
    assert not np.allclose(icol['rt_flux_sfc_dir' if band == 'sw'
                                else 'rt_lw_flux_sfc_dn'],
                           three_d['rt_flux_sfc_dir' if band == 'sw'
                                   else 'rt_lw_flux_sfc_dn'])


@requires_data
def test_rcemip_shortwave_tracer_matches_the_two_stream(rcemip):
    """The two solvers differ only in transport, and on this profile only a little.

    The direct beam is Beer-Lambert in both and neither approximates it, so it agrees
    to the photon noise. The rest is where the two-stream's own approximation lives --
    two hemispheres against a full angular integration -- and the difference is a
    percent or two, which is the size of that approximation rather than of an error.
    """
    _, out = rcemip

    two_stream, traced = out['sw_2str'], out['sw_rt_3d']

    assert mean(traced['rt_flux_sfc_dir']) == pytest.approx(
        mean(two_stream['flux_dir'][0]), rel=0.01)

    assert mean(traced['rt_flux_tod_dn']) == pytest.approx(
        mean(two_stream['flux_dn'][-1]), rel=1e-6)

    total_dn = mean(traced['rt_flux_sfc_dir']) + mean(traced['rt_flux_sfc_dif'])
    assert total_dn == pytest.approx(mean(two_stream['flux_dn'][0]), rel=0.03)

    assert mean(traced['rt_flux_sfc_up']) == pytest.approx(
        mean(two_stream['flux_up'][0]), rel=0.03)

    assert mean(traced['rt_flux_tod_up']) == pytest.approx(
        mean(two_stream['flux_up'][-1]), rel=0.03)


@requires_data
def test_rcemip_longwave_tracer_matches_the_plane_parallel_solver(rcemip):
    """The tracer integrates over angle where the plane-parallel solver picks one.

    The plane-parallel longwave here runs at a single secant, the 1.66 diffusivity
    angle, which is an approximation to the hemispheric integral the tracer performs
    outright. That is worth a percent or two here, on a sounding where nothing varies
    horizontally and the two are otherwise solving the same problem.
    """
    _, out = rcemip

    plane, traced = out['lw_noscat'], out['lw_rt_3d']

    assert mean(traced['rt_lw_flux_sfc_dn']) == pytest.approx(
        mean(plane['flux_dn'][0]), rel=0.05)
    assert mean(traced['rt_lw_flux_sfc_up']) == pytest.approx(
        mean(plane['flux_up'][0]), rel=0.05)
    assert mean(traced['rt_lw_flux_toa_up']) == pytest.approx(
        mean(plane['flux_up'][-1]), rel=0.05)

    # Nothing shines down on the atmosphere from above.
    assert mean(traced['rt_lw_flux_toa_dn']) == 0.0


@requires_data
def test_rcemip_longwave_scattering_lets_less_out(rcemip):
    """Cloud scattering in the longwave keeps radiation in.

    The no-scattering solver treats a cloud as absorption and emission alone; the
    two-stream one lets it deflect radiation as well, which sends part of what the
    warm lower atmosphere emits back down. The ice cloud is where this acts, so the
    top of the atmosphere sees less and the surface sees a little more.
    """
    _, out = rcemip

    noscat, scattering = out['lw_noscat'], out['lw_2str']

    assert mean(scattering['flux_up'][-1]) < mean(noscat['flux_up'][-1])
    assert mean(scattering['flux_dn'][0]) > mean(noscat['flux_dn'][0])

    # A few percent of it: the clouds are there, and they are not the whole story.
    assert mean(scattering['flux_up'][-1]) == pytest.approx(
        mean(noscat['flux_up'][-1]), rel=0.10)

    # The surface emits the same either way, which is what says the two runs saw the
    # same surface and differ only above it.
    assert mean(scattering['flux_up'][0]) == pytest.approx(
        mean(noscat['flux_up'][0]), rel=1e-9)


# ---------------------------------------------------------------------------
# The LES cloud field: genuinely three-dimensional, so the tracers part company.
# ---------------------------------------------------------------------------

@requires_data
@requires_les
def test_les_window_carries_cloud(les):
    """The cropped field is cloudy, which is the whole point of choosing it."""
    atm, _ = les

    assert atm['ncol'] == WINDOW*WINDOW
    assert atm['grid']['nz'] == 201        # 200 resolved cells and the lumped one

    cloudy = np.mean(atm['lwp'] > 0.0)
    assert cloudy > 0.02

    # Liquid only, as the field is described.
    assert np.all(atm['iwp'] == 0.0)


@requires_data
@requires_les
@pytest.mark.parametrize('solver', ['sw_rt_1d', 'sw_rt_3d'])
def test_les_shortwave_tracer_closes_its_energy_budget(les, solver):
    """Photons are conserved over a broken cloud field as over a uniform one."""
    atm, out = les

    entering, accounted = sw_budget(out[solver], atm['grid']['dz'])
    assert accounted == pytest.approx(entering, rel=0.01)


@requires_data
@requires_les
def test_les_independent_column_reproduces_the_two_stream_direct_beam(les):
    """Without horizontal transport the direct beam is the two-stream's, exactly.

    Both attenuate the beam down each column on its own, with no approximation in
    either, so this pins the tracer's geometry against a solver that has none: the
    slant path, the cell crossings and the null collisions all have to be right for a
    field this broken to come out at the same number.
    """
    _, out = les

    assert mean(out['sw_rt_1d']['rt_flux_sfc_dir']) == pytest.approx(
        mean(out['sw_2str']['flux_dir'][0]), rel=0.01)


@requires_data
@requires_les
def test_les_third_dimension_moves_the_direct_beam_into_the_diffuse(les):
    """What the third dimension is worth on this field, and which way it goes.

    An independent column sees only the cloud above itself, so the clear ones pass the
    beam untouched and the domain mean of exp(-tau/mu0) is carried by them. A slanted
    path through a periodic domain samples the whole field instead -- at this sun and
    over this depth it crosses the window several times -- so far less of the beam
    survives to the surface, and what it loses comes back as diffuse light. Jensen's
    inequality, which is the mechanism the plane-parallel solvers cannot represent.
    """
    _, out = les

    icol, three_d = out['sw_rt_1d'], out['sw_rt_3d']

    assert mean(three_d['rt_flux_sfc_dir']) < 0.5*mean(icol['rt_flux_sfc_dir'])
    assert mean(three_d['rt_flux_sfc_dif']) > mean(icol['rt_flux_sfc_dif'])

    # The sun is the same, and so is the total the surface receives, to within the
    # tenths of a percent this is: the beam is redistributed, not created or destroyed.
    for out_ in (icol, three_d):
        assert mean(out_['rt_flux_tod_dn']) == pytest.approx(
            mean(out['sw_2str']['flux_dn'][-1]), rel=1e-6)

    def sfc_dn(o):
        return mean(o['rt_flux_sfc_dir']) + mean(o['rt_flux_sfc_dif'])

    assert sfc_dn(three_d) == pytest.approx(sfc_dn(icol), rel=0.10)


@requires_data
@requires_les
def test_les_longwave_solvers_agree_at_the_surface(les):
    """All four longwave solvers see the same surface, to within a percent.

    The field is liquid-only and thin, so cloud scattering has little to change, and
    the case's own mean-free-path threshold sends the opaque half of the spectrum
    through the plane-parallel solver in the traced runs as well -- which is what makes
    the four agree here as closely as they do.
    """
    _, out = les

    reference = mean(out['lw_noscat']['flux_dn'][0])

    assert mean(out['lw_2str']['flux_dn'][0]) == pytest.approx(reference, rel=0.02)
    for solver in ('lw_rt_1d', 'lw_rt_3d'):
        assert mean(out[solver]['rt_lw_flux_sfc_dn']) == pytest.approx(
            reference, rel=0.02)
        assert mean(out[solver]['rt_lw_flux_sfc_up']) == pytest.approx(
            mean(out['lw_noscat']['flux_up'][0]), rel=0.01)


@requires_data
@requires_les
def test_les_longwave_tracer_only_traces_the_transparent_g_points(les):
    """The case's threshold of 1 splits the spectrum, as it does in the reference."""
    _, out = les

    for solver in ('lw_rt_1d', 'lw_rt_3d'):
        traced = int(out[solver]['n_gpt_traced'])
        assert 0 < traced < out['ngpt_lw']


@requires_data
@requires_les
def test_les_longwave_lumped_top_cell_lets_more_out(les):
    """Above the box the tracer holds one cell where the plane-parallel solver holds 132.

    The ray-tracing grid is the LES box, four kilometres deep; the sounding above it is
    lumped into a single cell on top so that the tracer still sees a whole atmosphere.
    That cell absorbs less of the upwelling flux than the layers it replaces, so more
    escapes at the top than the plane-parallel solver, which resolves them, reports.
    The two agree at the surface -- see the test above -- because the lumping is far
    from it. This is the approximation documented in include/raytracer_lw.h, not an
    error, and the test is here to say how large it is on this case.
    """
    _, out = les

    plane = mean(out['lw_noscat']['flux_up'][-1])

    for solver in ('lw_rt_1d', 'lw_rt_3d'):
        escaping = mean(out[solver]['rt_lw_flux_toa_up'])
        assert escaping > plane
        assert escaping < 1.5*plane
