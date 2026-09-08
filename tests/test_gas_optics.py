"""Step 2a: the RRTMGP gas-optics kernels.

Reference layouts are the reverse of rte3d's, which for a Fortran column-major array
means the same memory -- with one exception. rte3d keeps the column as the fastest
varying dimension in the per-flavour intermediates, where the reference puts ncol in
the middle of fmajor, fminor, col_mix and jeta. The transposes below are the only
place that difference shows up; the library never transposes anything.
"""
import numpy as np
import pytest


def tolerance(rte3d):
    return 1e-5 if rte3d.runtime().precision == 'single' else 1e-12


def kdist(rng, ngas=8, nflav=5, ntemp=14, npres=59, neta=9):
    """A synthetic k-distribution grid with RRTMGP's real shape and ranges."""
    temp_ref = np.linspace(160.0, 355.0, ntemp)
    press_ref_log = np.linspace(np.log(1.1e5), np.log(1.0), npres)

    return dict(
        # Major species pairs. Index 0 of the gas dimension is dry air, so flavours
        # reference gases 1..ngas.
        flavor=np.ascontiguousarray(rng.integers(1, ngas + 1, (nflav, 2)).astype(np.int32)),
        press_ref_log=press_ref_log,
        temp_ref=temp_ref,
        press_ref_log_delta=float(press_ref_log[1] - press_ref_log[0]),
        temp_ref_min=float(temp_ref[0]),
        temp_ref_delta=float(temp_ref[1] - temp_ref[0]),
        press_ref_trop_log=float(np.log(9948.431564193395)),
        neta=neta,
        vmr_ref=rng.uniform(1e-8, 1e-2, (ntemp, ngas + 1, 2)),
    )


def atmosphere(rng, ngas, nlay, ncol, zero_columns=False):
    play = 10.0**rng.uniform(np.log10(1.0), np.log10(1.1e5), (nlay, ncol))
    col_gas = rng.uniform(1e18, 1e24, (ngas + 1, nlay, ncol))

    if zero_columns:
        # Drives col_mix below 2*tiny, exercising the eta = 0.5 fallback the reference
        # warns must stay a branch rather than a merge().
        col_gas[:, :, ::3] = 0.0

    return dict(
        play=play,
        tlay=rng.uniform(160.0, 355.0, (nlay, ncol)),
        col_gas=col_gas,
    )


@pytest.mark.parametrize('zero_columns', [False, True])
@pytest.mark.parametrize('nlay,ncol', [(42, 7), (1, 5), (12, 1)])
def test_interpolation_matches_reference(rte3d, fortran_ref, nlay, ncol, zero_columns):
    rng = np.random.default_rng(30)
    k = kdist(rng)
    a = atmosphere(rng, k['vmr_ref'].shape[1] - 1, nlay, ncol, zero_columns)

    expected = fortran_ref.interpolation(**k, **a)
    actual = rte3d.interpolation(**k, **a)

    tol = tolerance(rte3d)

    # The reference's indices are 1-based; rte3d's are 0-based.
    np.testing.assert_array_equal(actual['jtemp'], expected['jtemp'] - 1)
    np.testing.assert_array_equal(actual['jpress'], expected['jpress'] - 1)
    np.testing.assert_array_equal(actual['tropo'].astype(np.int32), expected['tropo'])

    # (nflav, nlay, ncol, 2) -> (nflav, 2, nlay, ncol)
    np.testing.assert_array_equal(
        actual['jeta'], np.transpose(expected['jeta'], (0, 3, 1, 2)) - 1)
    np.testing.assert_allclose(
        actual['col_mix'], np.transpose(expected['col_mix'], (0, 3, 1, 2)),
        rtol=tol, atol=0.0)

    # (nflav, nlay, ncol, itemp, ieta) -> (nflav, itemp, ieta, nlay, ncol)
    np.testing.assert_allclose(
        actual['fminor'], np.transpose(expected['fminor'], (0, 3, 4, 1, 2)),
        rtol=tol, atol=0.0)

    # (nflav, nlay, ncol, itemp, ipress, ieta) -> (nflav, itemp, ipress, ieta, nlay, ncol)
    np.testing.assert_allclose(
        actual['fmajor'], np.transpose(expected['fmajor'], (0, 3, 4, 5, 1, 2)),
        rtol=tol, atol=0.0)


def test_interpolation_clamps_out_of_range_inputs(rte3d, fortran_ref):
    """Temperatures and pressures outside the table must clamp to its edges, the same
    way in both implementations."""
    rng = np.random.default_rng(31)
    k = kdist(rng)
    ngas = k['vmr_ref'].shape[1] - 1
    nlay, ncol = 6, 4

    a = atmosphere(rng, ngas, nlay, ncol)
    a['tlay'][0, :] = 100.0     # below temp_ref_min
    a['tlay'][1, :] = 500.0     # above the table
    a['play'][2, :] = 2.0e5     # below the lowest reference level
    a['play'][3, :] = 1.0e-3    # above the top

    expected = fortran_ref.interpolation(**k, **a)
    actual = rte3d.interpolation(**k, **a)

    np.testing.assert_array_equal(actual['jtemp'], expected['jtemp'] - 1)
    np.testing.assert_array_equal(actual['jpress'], expected['jpress'] - 1)
    np.testing.assert_allclose(
        actual['fmajor'], np.transpose(expected['fmajor'], (0, 3, 4, 5, 1, 2)),
        rtol=tolerance(rte3d), atol=0.0)


def test_weights_are_a_partition_of_unity(rte3d):
    """fmajor spans pressure x eta x temperature, so it must sum to one per flavour,
    layer and column. Needs no reference, and catches a dropped or duplicated term."""
    rng = np.random.default_rng(32)
    k = kdist(rng)
    a = atmosphere(rng, k['vmr_ref'].shape[1] - 1, 20, 6)

    out = rte3d.interpolation(**k, **a)

    np.testing.assert_allclose(
        out['fmajor'].sum(axis=(1, 2, 3)), 1.0, rtol=tolerance(rte3d), atol=0.0)
    np.testing.assert_allclose(
        out['fminor'].sum(axis=(1, 2)), 1.0, rtol=tolerance(rte3d), atol=0.0)


def minor_set(rng, band_lims, ngas, nminor, seed_shift=0):
    """A set of minor absorbers, each covering one band's g-point range.

    kminor_start is the cumulative offset of each absorber's block in kminor, so the
    blocks tile the table exactly, as in the real coefficient files.
    """
    nbnd = band_lims.shape[0]
    which_band = rng.integers(0, nbnd, nminor)
    limits = np.ascontiguousarray(band_lims[which_band].astype(np.int32))

    widths = limits[:, 1] - limits[:, 0] + 1
    starts = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int32)
    nminork = int(widths.sum())

    return dict(
        kminor=rng.uniform(0.0, 1e-3, (nminork, 9, 14)),
        minor_limits_gpt=limits,
        scales_with_density=rng.integers(0, 2, nminor).astype(np.int8),
        scale_by_complement=rng.integers(0, 2, nminor).astype(np.int8),
        idx_minor=rng.integers(1, ngas + 1, nminor).astype(np.int32),
        # 0 means no second gas scales this absorber, the reference's sentinel.
        idx_minor_scaling=(rng.integers(0, ngas + 1, nminor)).astype(np.int32),
        kminor_start=np.ascontiguousarray(starts),
    )


def kdist_full(rng, nbnd=4, gpts_per_band=4, ngas=8, nflav=5, ntemp=14, npres=59, neta=9):
    """A synthetic k-distribution, 0-based throughout, in rte3d's layouts."""
    ngpt = nbnd * gpts_per_band

    band_lims = np.stack([
        np.arange(nbnd) * gpts_per_band,
        np.arange(nbnd) * gpts_per_band + gpts_per_band - 1], axis=1).astype(np.int32)
    gpt_band = np.repeat(np.arange(nbnd), gpts_per_band).astype(np.int32)

    # Flavour is constant within a band, as it is in the real files: both the reference
    # and rte3d take it from the first g-point of a band or minor range.
    band_flavor = rng.integers(0, nflav, (nbnd, 2))

    k = kdist(rng, ngas=ngas, nflav=nflav, ntemp=ntemp, npres=npres, neta=neta)
    k.update(
        kdist=dict(
            flavor=k['flavor'],
            gpoint_flavor=np.ascontiguousarray(band_flavor[gpt_band].astype(np.int32)),
            band_lims_gpt=band_lims,
            gpt_band=gpt_band,
            band_gpt_start=np.ascontiguousarray(band_lims[:, 0].copy()),
            kmajor=rng.uniform(0.0, 1e-2, (ngpt, npres + 1, neta, ntemp)),
            idx_h2o=1,
        ))
    for prefix, nminor in (('lower_', 6), ('upper_', 4)):
        for key, val in minor_set(rng, band_lims, ngas, nminor).items():
            k['kdist'][prefix + key] = val

    return k


def flatten_kdist(k):
    """Split the bundle from kdist_full into the k-distribution dict and the grid
    arguments the interpolation takes.

    flavor belongs to both: the interpolation takes it directly, and rte3d's tau kernel
    reads it out of the k-distribution.
    """
    kd = k.pop('kdist')
    kd['flavor'] = k.pop('flavor')
    return kd, k


@pytest.mark.parametrize('monotonic', [True, False])
@pytest.mark.parametrize('nlay,ncol', [(20, 6), (1, 4), (8, 1)])
def test_tau_absorption_matches_reference(rte3d, fortran_ref, nlay, ncol, monotonic):
    """The reference's own interpolation output is fed straight to its tau kernel, so
    only the tau kernel is compared here; interpolation is checked separately above.

    The non-monotonic case matters because the reference bounds the minor-gas layer
    loop by a contiguous range found with minloc/maxloc rather than by the tropo mask.
    For a real atmosphere the two agree; for scrambled pressures they do not, and
    rte3d reproduces the range form.
    """
    rng = np.random.default_rng(40)
    bundle = kdist_full(rng)
    kd, grid = flatten_kdist(bundle)

    ngas = grid['vmr_ref'].shape[1] - 1
    a = atmosphere(rng, ngas, nlay, ncol)
    if monotonic:
        a['play'] = np.sort(a['play'], axis=0)

    interp = fortran_ref.interpolation(flavor=kd['flavor'], **grid, **a)
    expected = fortran_ref.compute_tau_absorption(
        kd, interp, a['play'], a['tlay'], a['col_gas'], grid['neta'])

    actual = rte3d.compute_tau_absorption(kdist=kd, **grid, **a)

    np.testing.assert_allclose(actual, expected, rtol=tolerance(rte3d), atol=0.0)
    assert np.all(actual >= 0.0)
    assert actual.max() > 0.0


@pytest.mark.parametrize('nlay,ncol', [(20, 6), (1, 4), (8, 1)])
def test_tau_rayleigh_matches_reference(rte3d, fortran_ref, nlay, ncol):
    rng = np.random.default_rng(50)
    bundle = kdist_full(rng)
    kd, grid = flatten_kdist(bundle)

    ngpt = kd['kmajor'].shape[0]
    neta, ntemp = grid['neta'], grid['temp_ref'].shape[0]
    kd['krayl'] = rng.uniform(0.0, 1e-26, (2, ngpt, neta, ntemp))

    ngas = grid['vmr_ref'].shape[1] - 1
    a = atmosphere(rng, ngas, nlay, ncol)
    col_dry = rng.uniform(1e22, 1e25, (nlay, ncol))

    interp = fortran_ref.interpolation(flavor=kd['flavor'], **grid, **a)
    expected = fortran_ref.compute_tau_rayleigh(
        kd, interp, col_dry, a['col_gas'], grid['neta'])

    actual = rte3d.compute_tau_rayleigh(kdist=kd, col_dry=col_dry, **grid, **a)

    np.testing.assert_allclose(actual, expected, rtol=tolerance(rte3d), atol=0.0)
    assert actual.max() > 0.0


@pytest.mark.parametrize('nlay,ncol', [(20, 6), (1, 4), (8, 1)])
def test_planck_source_matches_reference(rte3d, fortran_ref, nlay, ncol):
    rng = np.random.default_rng(51)
    bundle = kdist_full(rng)
    kd, grid = flatten_kdist(bundle)

    ngpt = kd['kmajor'].shape[0]
    nbnd = kd['band_lims_gpt'].shape[0]
    npres = grid['press_ref_log'].shape[0]
    neta, ntemp = grid['neta'], grid['temp_ref'].shape[0]
    nplancktemp = 196

    kd['pfracin'] = rng.uniform(0.0, 1.0, (ngpt, npres + 1, neta, ntemp))
    kd['totplnk'] = rng.uniform(0.0, 100.0, (nbnd, nplancktemp))
    kd['temp_ref_min'] = grid['temp_ref_min']
    kd['totplnk_delta'] = 1.0

    ngas = grid['vmr_ref'].shape[1] - 1
    a = atmosphere(rng, ngas, nlay, ncol)
    tlev = rng.uniform(160.0, 355.0, (nlay + 1, ncol))
    tsfc = rng.uniform(200.0, 320.0, ncol)
    sfc_lay = nlay - 1

    interp = fortran_ref.interpolation(flavor=kd['flavor'], **grid, **a)
    expected = fortran_ref.compute_planck_source(
        kd, interp, a['tlay'], tlev, tsfc, sfc_lay, grid['neta'])

    actual = rte3d.compute_planck_source(
        kdist=kd, tlev=tlev, tsfc=tsfc, sfc_lay=sfc_lay, **grid, **a)

    tol = tolerance(rte3d)
    for key in ('lay_source', 'lev_source', 'sfc_source', 'sfc_source_jac'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=tol, atol=0.0,
                                   err_msg=f'{key} differs from the reference')


def test_planck_sources_feed_the_longwave_solver(rte3d):
    """The Planck kernel's output shapes are exactly what Source_func_lw and the
    longwave solvers already take, so the two halves compose without adaptation."""
    rng = np.random.default_rng(52)
    bundle = kdist_full(rng)
    kd, grid = flatten_kdist(bundle)

    ngpt = kd['kmajor'].shape[0]
    nbnd = kd['band_lims_gpt'].shape[0]
    npres = grid['press_ref_log'].shape[0]
    nlay, ncol = 12, 5

    kd['pfracin'] = rng.uniform(0.0, 1.0, (ngpt, npres + 1, grid['neta'], 14))
    kd['totplnk'] = rng.uniform(1.0, 100.0, (nbnd, 196))
    kd['temp_ref_min'] = grid['temp_ref_min']
    kd['totplnk_delta'] = 1.0

    a = atmosphere(rng, grid['vmr_ref'].shape[1] - 1, nlay, ncol)
    src = rte3d.compute_planck_source(
        kdist=kd, tlev=rng.uniform(160.0, 355.0, (nlay + 1, ncol)),
        tsfc=rng.uniform(200.0, 320.0, ncol), sfc_lay=nlay - 1, **grid, **a)

    flux_up, flux_dn, _ = rte3d.lw_solver_noscat(
        True,
        secants=np.full((1, ngpt, ncol), 1.66),
        weights=np.array([0.5]),
        tau=10.0**rng.uniform(-4.0, 1.0, (ngpt, nlay, ncol)),
        lay_source=src['lay_source'], lev_source=src['lev_source'],
        sfc_emis=np.full((ngpt, ncol), 0.98),
        sfc_source=src['sfc_source'],
        inc_flux=np.zeros((ngpt, ncol)))

    assert np.all(np.isfinite(flux_up)) and np.all(np.isfinite(flux_dn))
    assert np.all(flux_up >= 0.0) and np.all(flux_dn >= 0.0)
