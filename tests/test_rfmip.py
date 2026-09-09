"""RFMIP clear-sky: the end-to-end oracle.

The reference fluxes in extern/rrtmgp-data were produced by RRTMGP itself, so this
exercises the whole chain at once -- reading the coefficient file, the k-distribution
reduction, all four gas-optics kernels, and the RTE solvers -- against numbers no part
of rte3d had a hand in producing.

The tolerance is RFMIP's own acceptance threshold, 5.8e-2 W/m2, which is what
compare-to-reference.py in the reference repository uses. rte3d comes in well inside
it; the residual is the difference between this k-distribution and the one the
reference fluxes were generated with in 2018.
"""
import os

import numpy as np
import pytest

from compare import assert_flux_close, flux_threshold

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')
RFMIP = os.path.join(DATA, 'examples', 'rfmip-clear-sky')
INPUT = os.path.join(
    RFMIP, 'inputs', 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc')


def _reference(name):
    return os.path.join(RFMIP, 'reference',
                        f'{name}_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc')


requires_data = pytest.mark.skipif(
    not os.path.exists(INPUT), reason='rrtmgp-data submodule not checked out')


def _setup(rte3d, kdist_file):
    from rte3d.kdist import read_kdist
    from rte3d.rfmip import read_rfmip, make_gas_concs

    f = read_kdist(os.path.join(DATA, kdist_file))
    atm = read_rfmip(INPUT, f['gas_names'])
    gas_concs = make_gas_concs(rte3d, atm['gases'])

    return rte3d.load_kdist(f, gas_concs), gas_concs, atm


@requires_data
def test_rfmip_longwave(rte3d):
    from rte3d.rfmip import read_reference, solve_lw

    kdist, gas_concs, atm = _setup(rte3d, 'rrtmgp-gas-lw-g256.nc')
    up, dn = solve_lw(rte3d, kdist, gas_concs, atm)

    ref_up = read_reference(_reference('rlu'), 'rlu')
    ref_dn = read_reference(_reference('rld'), 'rld')

    assert up.shape == ref_up.shape

    err_up = np.abs(up - ref_up).max()
    err_dn = np.abs(dn - ref_dn).max()

    assert err_up < flux_threshold(rte3d), f'upward flux differs by {err_up} W/m2'
    assert err_dn < flux_threshold(rte3d), f'downward flux differs by {err_dn} W/m2'

    # Outgoing longwave for the 100 RFMIP sites, as a guard against a result that is
    # numerically close but physically absurd.
    assert 150.0 < up[0].mean() < 350.0


@requires_data
def test_rfmip_shortwave(rte3d):
    from rte3d.rfmip import read_reference, solve_sw

    kdist, gas_concs, atm = _setup(rte3d, 'rrtmgp-gas-sw-g224.nc')
    up, dn, daytime = solve_sw(rte3d, kdist, gas_concs, atm)

    ref_up = read_reference(_reference('rsu'), 'rsu')
    ref_dn = read_reference(_reference('rsd'), 'rsd')

    # Night-time columns carry no meaningful flux in either implementation.
    err_up = np.abs(up[:, daytime] - ref_up[:, daytime]).max()
    err_dn = np.abs(dn[:, daytime] - ref_dn[:, daytime]).max()

    assert err_up < flux_threshold(rte3d), f'upward flux differs by {err_up} W/m2'
    assert err_dn < flux_threshold(rte3d), f'downward flux differs by {err_dn} W/m2'

    assert daytime.sum() == 51
    # Incoming solar at the top of the atmosphere must equal the prescribed TSI
    # times the cosine of the zenith angle.
    expected_toa = (atm['total_solar_irradiance'][daytime]
                    * np.cos(np.radians(atm['solar_zenith_angle'][daytime])))
    assert_flux_close(dn[0, daytime], expected_toa, rte3d, rtol=1e-12)


@requires_data
def test_rfmip_longwave_across_experiments(rte3d):
    """RFMIP's eighteen experiments span pre-industrial to quadrupled CO2, so this
    checks the gas concentrations actually reach the optics rather than a fixed
    profile being reused."""
    from rte3d.kdist import read_kdist
    from rte3d.rfmip import read_rfmip, make_gas_concs, read_reference, solve_lw

    f = read_kdist(os.path.join(DATA, 'rrtmgp-gas-lw-g256.nc'))
    ref_up = read_reference(_reference('rlu'), 'rlu', expt=0)

    olr = {}
    for expt in (0, 1, 4):
        atm = read_rfmip(INPUT, f['gas_names'], expt=expt)
        gas_concs = make_gas_concs(rte3d, atm['gases'])
        kdist = rte3d.load_kdist(f, gas_concs)

        up, _ = solve_lw(rte3d, kdist, gas_concs, atm)
        olr[expt] = up[0].mean()

        expected = read_reference(_reference('rlu'), 'rlu', expt=expt)
        assert np.abs(up - expected).max() < flux_threshold(rte3d)

    # Experiment 1 is pre-industrial, so it must emit more than present-day.
    assert olr[1] > olr[0]


@requires_data
def test_reference_fluxes_have_no_missing_values():
    """xarray warns that the reference files declare _FillValue in two dtypes and
    will decode both to NaN. Nothing is actually masked, and this keeps it that way:
    a NaN slipping into the reference would silently turn a comparison into a pass."""
    import xarray as xr

    for name in ('rlu', 'rld', 'rsu', 'rsd'):
        values = xr.open_dataset(_reference(name))[name].values
        assert not np.isnan(values).any(), f'{name} contains missing values'


@requires_data
def test_netcdf_reads_match_ncdump():
    """netCDF4's Cython extension is built against older numpy headers and warns about
    it. The warning is ignored in pytest.ini on the strength of this check: values read
    through it match the netCDF C library exactly.

    The literals are ncdump's output, which prints 15 significant digits and so does
    not always round-trip to the exact stored double; hence a tolerance of a few
    ulp rather than exact equality. That is still many orders of magnitude tighter
    than any corruption an ABI mismatch would cause.
    """
    import xarray as xr

    d = xr.open_dataset(os.path.join(DATA, 'rrtmgp-gas-lw-g256.nc'))

    np.testing.assert_allclose(
        d['press_ref'].values[:3],
        [109663.315842846, 89784.7291650418, 73509.5189241974], rtol=1e-14)
    np.testing.assert_allclose(
        d['kmajor'].values[0, 0, 0, :3],
        [2.53463350242747e-22, 7.16234088408182e-22, 1.48047377615368e-21], rtol=1e-14)

    # Integers are exact in both representations.
    np.testing.assert_array_equal(d['temp_ref'].values, np.arange(160.0, 356.0, 15.0))
