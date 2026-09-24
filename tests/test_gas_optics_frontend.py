"""Step 2b: the gas-optics frontend, on the real coefficient files.

Everything below runs against extern/rrtmgp-data, not synthetic tables, so it also
exercises rte3d.kdist.read_kdist and in particular its transposes.

The four kernels and the reduction each have their own oracle elsewhere. What is under
test here is the composition: that the frontend feeds them the right arrays in the
right order, and that the loader's dimension handling is right.
"""
import os

import numpy as np
import pytest

from compare import assert_close, rtol_for

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')
LW_FILE = os.path.join(DATA, 'rrtmgp-gas-lw-g256.nc')
SW_FILE = os.path.join(DATA, 'rrtmgp-gas-sw-g224.nc')

# A plausible present-day composition; the values matter only in that they are
# physical, since the comparison is against the reference given the same input.
CONCS = dict(h2o=None, co2=348e-6, o3=None, n2o=306e-9, co=0.0,
             ch4=1650e-9, o2=0.2095, n2=0.7808)


def requires_data():
    return pytest.mark.skipif(
        not os.path.exists(LW_FILE),
        reason='rrtmgp-data submodule not checked out')


def profile(nlay=42, ncol=4, seed=0):
    """A monotonic, physically ordered atmosphere: index 0 at the top."""
    rng = np.random.default_rng(seed)

    plev = np.exp(np.linspace(np.log(1.0e2), np.log(1.005e5), nlay + 1))[:, None] \
        * np.ones((1, ncol))
    play = 0.5*(plev[:-1] + plev[1:])

    # Roughly a standard atmosphere, with enough spread across columns to matter.
    tlev = 200.0 + 100.0*(plev/plev.max())**0.3 + rng.uniform(-3.0, 3.0, plev.shape)
    tlay = 0.5*(tlev[:-1] + tlev[1:])

    h2o = 1e-6 + 2e-2*(play/play.max())**3
    o3 = 1e-7*np.exp(-((np.log(play) - np.log(3e3))**2)/2.0)

    return dict(play=play, plev=plev, tlay=tlay, tlev=tlev,
                tsfc=tlev[-1].copy(), h2o=h2o, o3=o3)


def make_concs(rte3d, atm):
    g = rte3d.Gas_concs()
    for name, value in CONCS.items():
        g.set_vmr(name, atm[name] if value is None else value)
    return g


@requires_data()
def test_load_real_lw_file(rte3d):
    from rte3d.kdist import read_kdist
    atm = profile()
    kdist = rte3d.load_kdist(read_kdist(LW_FILE), make_concs(rte3d, atm))

    # The file has 19 absorbers; only the eight supplied survive, in file order.
    assert kdist.gas_names == list(CONCS)
    assert kdist.idx_h2o == list(CONCS).index('h2o') + 1
    assert kdist.neta == 9


@requires_data()
@pytest.mark.parametrize('supply_col_dry', [False, True])
def test_lw_gas_optics_matches_reference(rte3d, fortran_ref, supply_col_dry):
    """Compare tau and the Planck sources against the reference kernels, driven with
    rte3d's own reduced k-distribution converted back to 1-based indices."""
    from rte3d.kdist import read_kdist

    atm = profile()
    gas_concs = make_concs(rte3d, atm)
    kdist = rte3d.load_kdist(read_kdist(LW_FILE), gas_concs)
    a = kdist.arrays()

    col_dry = (rte3d.compute_col_dry(atm['h2o'], atm['plev'])
               if supply_col_dry else None)
    col_gas = rte3d.compute_col_gas(gas_concs, kdist.gas_names, atm['plev'], col_dry)

    grid = dict(
        press_ref_log=a['press_ref_log'], temp_ref=a['temp_ref'],
        press_ref_log_delta=a['press_ref_log_delta'],
        temp_ref_min=a['temp_ref_min'], temp_ref_delta=a['temp_ref_delta'],
        press_ref_trop_log=a['press_ref_trop_log'], neta=kdist.neta,
        vmr_ref=a['vmr_ref'])

    interp = fortran_ref.interpolation(
        flavor=a['flavor'], **grid, play=atm['play'], tlay=atm['tlay'], col_gas=col_gas)

    expected_tau = fortran_ref.compute_tau_absorption(
        a, interp, atm['play'], atm['tlay'], col_gas, kdist.neta)

    actual = rte3d.gas_optics_lw(
        kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'], atm['tlev'],
        atm['tsfc'], col_dry)

    assert_close(actual['tau'], expected_tau, rtol=rtol_for(rte3d, 1e-11))

    # Optical depths of a real atmosphere span many orders of magnitude, and none
    # should be negative.
    assert actual['tau'].min() >= 0.0
    assert actual['tau'].max() > 1.0

    expected_src = fortran_ref.compute_planck_source(
        {**a, 'pfracin': a['pfracin'], 'totplnk': a['totplnk'],
         'temp_ref_min': a['temp_ref_min'], 'totplnk_delta': a['totplnk_delta']},
        interp, atm['tlay'], atm['tlev'], atm['tsfc'],
        atm['play'].shape[0] - 1, kdist.neta)

    for key in ('lay_source', 'lev_source', 'sfc_source', 'sfc_source_jac'):
        assert_close(actual[key], expected_src[key], rtol=rtol_for(rte3d, 1e-11),
                                   err_msg=f'{key} differs from the reference')


@requires_data()
def test_sw_gas_optics_matches_reference(rte3d, fortran_ref):
    """tau and ssa against the reference's absorption and Rayleigh kernels, combined
    as combine_abs_and_rayleigh does. This runs the g-point block kernel over whole
    bands, which the kernel-by-kernel tests, one g-point at a time, do not."""
    from rte3d.kdist import read_kdist

    atm = profile()
    gas_concs = make_concs(rte3d, atm)
    kdist = rte3d.load_kdist(read_kdist(SW_FILE), gas_concs)
    a = kdist.arrays()

    col_gas = rte3d.compute_col_gas(gas_concs, kdist.gas_names, atm['plev'])

    grid = dict(
        press_ref_log=a['press_ref_log'], temp_ref=a['temp_ref'],
        press_ref_log_delta=a['press_ref_log_delta'],
        temp_ref_min=a['temp_ref_min'], temp_ref_delta=a['temp_ref_delta'],
        press_ref_trop_log=a['press_ref_trop_log'], neta=kdist.neta,
        vmr_ref=a['vmr_ref'])

    interp = fortran_ref.interpolation(
        flavor=a['flavor'], **grid, play=atm['play'], tlay=atm['tlay'], col_gas=col_gas)

    tau_abs = fortran_ref.compute_tau_absorption(
        a, interp, atm['play'], atm['tlay'], col_gas, kdist.neta)
    tau_ray = fortran_ref.compute_tau_rayleigh(
        a, interp, np.ascontiguousarray(col_gas[0]), col_gas, kdist.neta)

    expected_tau = tau_abs + tau_ray
    expected_ssa = tau_ray/expected_tau

    tau, ssa = rte3d.gas_optics_sw(
        kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'])

    assert_close(tau, expected_tau, rtol=rtol_for(rte3d, 1e-11))
    assert_close(ssa, expected_ssa, rtol=rtol_for(rte3d, 1e-11))


@requires_data()
def test_sw_gas_optics_is_physical(rte3d):
    """The shortwave path has no separate oracle here; check the invariants that
    combine_abs_and_rayleigh must satisfy."""
    from rte3d.kdist import read_kdist

    atm = profile()
    gas_concs = make_concs(rte3d, atm)
    kdist = rte3d.load_kdist(read_kdist(SW_FILE), gas_concs)

    tau, ssa = rte3d.gas_optics_sw(
        kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'])

    assert np.all(np.isfinite(tau)) and np.all(np.isfinite(ssa))
    assert tau.min() >= 0.0
    assert 0.0 <= ssa.min() and ssa.max() <= 1.0

    # Rayleigh scattering dominates the shortest wavelengths and is negligible in the
    # near infrared, so the single-scattering albedo must span a wide range.
    assert ssa.max() > 0.5


@requires_data()
def test_lw_gas_optics_feeds_the_solver(rte3d):
    """Gas optics straight into the longwave solver: the whole chain in one call
    sequence, with no adapter between the halves."""
    from rte3d.kdist import read_kdist

    atm = profile(nlay=42, ncol=3)
    gas_concs = make_concs(rte3d, atm)
    kdist = rte3d.load_kdist(read_kdist(LW_FILE), gas_concs)

    out = rte3d.gas_optics_lw(
        kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'], atm['tlev'], atm['tsfc'])

    ngpt, nlay, ncol = out['tau'].shape

    flux_up, flux_dn, _ = rte3d.lw_solver_noscat(
        True,                                    # index 0 is the top of the atmosphere
        secants=np.full((1, ngpt, ncol), 1.66),
        weights=np.array([0.5]),
        tau=out['tau'],
        lay_source=out['lay_source'], lev_source=out['lev_source'],
        sfc_emis=np.full((ngpt, ncol), 0.98),
        sfc_source=out['sfc_source'],
        inc_flux=np.zeros((ngpt, ncol)))

    olr = flux_up[:, 0, :].sum(axis=0)
    surface_down = flux_dn[:, -1, :].sum(axis=0)

    # Broadband outgoing longwave radiation for an Earth-like column.
    assert np.all((olr > 120.0) & (olr < 350.0)), olr
    assert np.all((surface_down > 100.0) & (surface_down < 500.0)), surface_down
