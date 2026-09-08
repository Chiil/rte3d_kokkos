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
