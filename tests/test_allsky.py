"""The RRTMGP all-sky case, without aerosols.

A cloudy counterpart to RFMIP, and the first case to exercise cloud optics and the
by-band increments end to end. It is also stored surface-first -- top_at_1 is False --
where RFMIP is top-first, so between them the two cases cover both branches of every
solver and of the gas optics.

The reference files carry both the inputs and the fluxes RRTMGP produced from them.
"""
import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')
REFERENCE = os.path.join(DATA, 'examples', 'all-sky', 'reference')

LW_REF = os.path.join(REFERENCE, 'rrtmgp-allsky-lw-no-aerosols.nc')
SW_REF = os.path.join(REFERENCE, 'rrtmgp-allsky-sw-no-aerosols.nc')

requires_data = pytest.mark.skipif(
    not os.path.exists(LW_REF), reason='rrtmgp-data submodule not checked out')

# rte3d reproduces this case to round-off, so the tolerance is far tighter than
# RFMIP's 5.8e-2 W/m2. Unlike the RFMIP fluxes, these reference files were generated
# with the same coefficient data that ships alongside them.
TOLERANCE = 1e-9


def _setup(rte3d, gas_file, cloud_file, ref_file):
    from rte3d.kdist import read_kdist, read_cloud_optics
    from rte3d.allsky import read_allsky, make_gas_concs, ICERGH

    atm = read_allsky(ref_file)
    gas_concs = make_gas_concs(rte3d, atm)

    kdist = rte3d.load_kdist(read_kdist(os.path.join(DATA, gas_file)), gas_concs)
    cloud_optics = rte3d.load_cloud_optics(
        read_cloud_optics(os.path.join(DATA, cloud_file)), ICERGH)

    return kdist, cloud_optics, gas_concs, atm


@requires_data
def test_allsky_longwave(rte3d):
    from rte3d.allsky import solve_lw

    kdist, cloud_optics, gas_concs, atm = _setup(
        rte3d, 'rrtmgp-gas-lw-g256.nc', 'rrtmgp-clouds-lw-bnd.nc', LW_REF)

    up, dn = solve_lw(rte3d, kdist, cloud_optics, gas_concs, atm)

    np.testing.assert_allclose(up, atm['lw_flux_up'], rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(dn, atm['lw_flux_dn'], rtol=0.0, atol=TOLERANCE)

    # Layer 0 is the surface here, so the outgoing flux is the last level.
    assert 100.0 < up[-1].mean() < 300.0


@requires_data
def test_allsky_shortwave(rte3d):
    from rte3d.allsky import solve_sw, MU0

    kdist, cloud_optics, gas_concs, atm = _setup(
        rte3d, 'rrtmgp-gas-sw-g224.nc', 'rrtmgp-clouds-sw-bnd.nc', SW_REF)

    up, dn, direct = solve_sw(rte3d, kdist, cloud_optics, gas_concs, atm)

    np.testing.assert_allclose(up, atm['sw_flux_up'], rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(dn, atm['sw_flux_dn'], rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(direct, atm['sw_flux_dir'], rtol=0.0, atol=TOLERANCE)

    # Incoming solar at the top is the total solar irradiance times the zenith cosine.
    np.testing.assert_allclose(dn[-1], kdist.solar_source.sum()*MU0, rtol=1e-12)


@requires_data
def test_clouds_change_the_fluxes(rte3d):
    """Every third column in this case is cloud-free, which makes the cloudy and clear
    columns directly comparable. Clouds must reflect more and transmit less."""
    from rte3d.allsky import solve_sw

    kdist, cloud_optics, gas_concs, atm = _setup(
        rte3d, 'rrtmgp-gas-sw-g224.nc', 'rrtmgp-clouds-sw-bnd.nc', SW_REF)

    up, dn, _ = solve_sw(rte3d, kdist, cloud_optics, gas_concs, atm)

    cloudy = atm['lwp'].sum(axis=0) > 0.0
    assert cloudy.any() and not cloudy.all(), 'expected a mix of cloudy and clear columns'

    assert up[-1][cloudy].mean() > 2.0*up[-1][~cloudy].mean(), 'clouds must reflect'
    assert dn[0][cloudy].mean() < 0.5*dn[0][~cloudy].mean(), 'clouds must shade the surface'


@requires_data
def test_ice_roughness_matters(rte3d):
    """The all-sky driver selects the middle ice roughness type rather than the first.

    That is easy to overlook and moves the shortwave fluxes by several W/m2, which is
    a hundred times RFMIP's acceptance threshold, so it is worth pinning.
    """
    from rte3d.kdist import read_kdist, read_cloud_optics
    from rte3d.allsky import read_allsky, make_gas_concs, solve_sw, ICERGH

    atm = read_allsky(SW_REF)
    gas_concs = make_gas_concs(rte3d, atm)
    kdist = rte3d.load_kdist(read_kdist(os.path.join(DATA, 'rrtmgp-gas-sw-g224.nc')), gas_concs)
    cloud_file = read_cloud_optics(os.path.join(DATA, 'rrtmgp-clouds-sw-bnd.nc'))

    errors = []
    for icergh in range(3):
        cloud_optics = rte3d.load_cloud_optics(cloud_file, icergh)
        _, dn, _ = solve_sw(rte3d, kdist, cloud_optics, gas_concs, atm)
        errors.append(np.abs(dn - atm['sw_flux_dn']).max())

    assert errors[ICERGH] < TOLERANCE
    for icergh in (0, 2):
        assert errors[icergh] > 1.0, f'roughness {icergh} should differ visibly'
