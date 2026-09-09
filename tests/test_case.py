"""Running an arbitrary case supplied as a NetCDF file.

The file layout is rte-rrtmgp-cpp's, documented in rte3d.case. The all-sky case is
re-emitted in that layout here and solved through the generic path, which has to
reproduce what rte3d.allsky computes from the same numbers: same gases, same clouds,
same boundary conditions, only a different way in.
"""
import os

import numpy as np
import pytest

from compare import assert_close

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')
REFERENCE = os.path.join(DATA, 'examples', 'all-sky', 'reference')

LW_REF = os.path.join(REFERENCE, 'rrtmgp-allsky-lw-no-aerosols.nc')
SW_REF = os.path.join(REFERENCE, 'rrtmgp-allsky-sw-no-aerosols.nc')

requires_data = pytest.mark.skipif(
    not os.path.exists(LW_REF), reason='rrtmgp-data submodule not checked out')

# The two paths run the same kernels on the same inputs, so they agree bit for bit.
TOLERANCE = 0.0


def write_input(path, atm, nbnd, band_dim, gridded=True, **extra):
    """Write an all-sky atmosphere out as a user case file.

    The columns become a 1 x ncol grid, so the (lay, y, x) reshape is exercised
    without inventing a second horizontal dimension the data does not have.
    """
    import xarray as xr

    from rte3d.allsky import FIXED_GASES, MU0, SFC_ALBEDO, SFC_EMIS

    nlay, ncol = atm['play'].shape

    if gridded:
        dims, shape = ('y', 'x'), (1, ncol)
    else:
        dims, shape = ('col',), (ncol,)

    def field(values, vertical):
        return ((vertical,) + dims, values.reshape((-1,) + shape))

    variables = {
        'p_lay': field(atm['play'], 'lay'), 'p_lev': field(atm['plev'], 'lev'),
        't_lay': field(atm['tlay'], 'lay'), 't_lev': field(atm['tlev'], 'lev'),
        'vmr_h2o': field(atm['h2o'], 'lay'), 'vmr_o3': field(atm['o3'], 'lay'),
        'lwp': field(atm['lwp'], 'lay'), 'iwp': field(atm['iwp'], 'lay'),
        'rel': field(atm['rel'], 'lay'), 'dei': field(atm['rei'], 'lay'),
        't_sfc': (dims, atm['tsfc'].reshape(shape)),
        'mu0': (dims, np.full(shape, MU0)),
        'emis_sfc': (dims + (band_dim,), np.full(shape + (nbnd,), SFC_EMIS)),
        'sfc_alb_dir': (dims + (band_dim,), np.full(shape + (nbnd,), SFC_ALBEDO)),
        'sfc_alb_dif': (dims + (band_dim,), np.full(shape + (nbnd,), SFC_ALBEDO)),
    }

    for name, value in FIXED_GASES.items():
        variables[f'vmr_{name}'] = ((), np.float64(value))

    variables.update(extra)
    xr.Dataset(variables).to_netcdf(path)

    return path


def setup(rte3d, tmp_path, band, gas_file, cloud_file, ref_file, **kwargs):
    """Both paths, from the same all-sky reference file."""
    from rte3d.allsky import ICERGH, read_allsky
    from rte3d.case import gpoint_bands, make_gas_concs, read_case
    from rte3d.kdist import read_cloud_optics, read_kdist

    reference = read_allsky(ref_file)

    f = read_kdist(os.path.join(DATA, gas_file))
    nbnd = f['band2gpt'].shape[0]

    path = write_input(str(tmp_path/f'{band}_input.nc'), reference,
                       nbnd, f'band_{band}', **kwargs)
    atm = read_case(path)

    gas_concs = make_gas_concs(rte3d, atm, f['gas_names'])
    kdist = rte3d.load_kdist(f, gas_concs)
    cloud_optics = rte3d.load_cloud_optics(
        read_cloud_optics(os.path.join(DATA, cloud_file)), ICERGH)

    return kdist, cloud_optics, gas_concs, atm, gpoint_bands(f), reference


@requires_data
def test_case_longwave(rte3d, tmp_path):
    from rte3d.allsky import make_gas_concs as allsky_gas_concs
    from rte3d.allsky import solve_lw as allsky_lw
    from rte3d.case import solve_lw

    kdist, cloud_optics, gas_concs, atm, gpt_band, reference = setup(
        rte3d, tmp_path, 'lw', 'rrtmgp-gas-lw-g256.nc', 'rrtmgp-clouds-lw-bnd.nc', LW_REF)

    assert atm['top_at_1'] is False

    out = solve_lw(rte3d, kdist, gas_concs, atm, gpt_band, cloud_optics)

    up, dn = allsky_lw(rte3d, kdist, cloud_optics,
                       allsky_gas_concs(rte3d, reference), reference)

    assert_close(out['flux_up'], up, TOLERANCE, 'flux_up')
    assert_close(out['flux_dn'], dn, TOLERANCE, 'flux_dn')


@requires_data
def test_case_shortwave(rte3d, tmp_path):
    from rte3d.allsky import make_gas_concs as allsky_gas_concs
    from rte3d.allsky import solve_sw as allsky_sw
    from rte3d.case import solve_sw

    kdist, cloud_optics, gas_concs, atm, gpt_band, reference = setup(
        rte3d, tmp_path, 'sw', 'rrtmgp-gas-sw-g224.nc', 'rrtmgp-clouds-sw-bnd.nc', SW_REF)

    out = solve_sw(rte3d, kdist, gas_concs, atm, gpt_band, cloud_optics)

    up, dn, dir_ = allsky_sw(rte3d, kdist, cloud_optics,
                             allsky_gas_concs(rte3d, reference), reference)

    assert_close(out['flux_up'], up, TOLERANCE, 'flux_up')
    assert_close(out['flux_dn'], dn, TOLERANCE, 'flux_dn')
    assert_close(out['flux_dir'], dir_, TOLERANCE, 'flux_dir')


@requires_data
def test_case_flat_columns(rte3d, tmp_path):
    """A file with a bare col dimension instead of x and y reads the same."""
    from rte3d.allsky import read_allsky
    from rte3d.case import read_case
    from rte3d.kdist import read_kdist

    reference = read_allsky(LW_REF)
    nbnd = read_kdist(os.path.join(DATA, 'rrtmgp-gas-lw-g256.nc'))['band2gpt'].shape[0]

    gridded = read_case(write_input(str(tmp_path/'g.nc'), reference, nbnd, 'band_lw'))
    flat = read_case(write_input(str(tmp_path/'f.nc'), reference, nbnd, 'band_lw',
                                 gridded=False))

    assert gridded['dims'] == ('y', 'x') and flat['dims'] == ('col',)
    for name in ('play', 'tlay', 'tsfc', 'sfc_emis', 'mu0'):
        assert np.array_equal(gridded[name], flat[name])


@requires_data
def test_case_gas_ranks(rte3d, tmp_path):
    """A gas may be a scalar, a profile or a full field, as in the reference driver."""
    from rte3d.allsky import read_allsky
    from rte3d.case import make_gas_concs, read_case
    from rte3d.kdist import read_kdist

    reference = read_allsky(LW_REF)
    nlay, ncol = reference['play'].shape
    nbnd = read_kdist(os.path.join(DATA, 'rrtmgp-gas-lw-g256.nc'))['band2gpt'].shape[0]

    profile = np.linspace(1e-6, 2e-6, nlay)
    atm = read_case(write_input(str(tmp_path/'ranks.nc'), reference, nbnd, 'band_lw',
                                vmr_n2o=(('lay',), profile)))

    assert isinstance(atm['gases']['co2'], float)
    assert atm['gases']['n2o'].shape == (nlay,)
    assert atm['gases']['h2o'].shape == (nlay, ncol)

    # Every rank has to survive the trip into Gas_concs, where it is expanded.
    gas_concs = make_gas_concs(rte3d, atm)
    expanded = gas_concs.get_vmr('n2o', nlay, ncol)
    assert np.allclose(expanded, np.broadcast_to(profile[:, None], (nlay, ncol)))
