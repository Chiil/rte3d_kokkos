"""Running an arbitrary case supplied as a NetCDF file.

The file layout is rte-rrtmgp-cpp's, documented in rte3d.case. The all-sky case is
re-emitted in that layout here and solved through the generic path, which has to
reproduce what rte3d.allsky computes from the same numbers: same gases, same clouds,
same boundary conditions, only a different way in.

The generated case goes the other way round: cases/user/make_input.py writes the file
and run_case.py reads it back and solves it, which is the path a user of this repository
actually walks. That case has what the all-sky one does not -- well-mixed gases stored
as scalars, surface conditions resolved by band, an absolute solar irradiance per
column -- and its fluxes are checked against what the boundary conditions dictate.
"""
import os

import numpy as np
import pytest

from compare import assert_close, assert_flux_close

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')
REFERENCE = os.path.join(DATA, 'examples', 'all-sky', 'reference')

LW_REF = os.path.join(REFERENCE, 'rrtmgp-allsky-lw-no-aerosols.nc')
SW_REF = os.path.join(REFERENCE, 'rrtmgp-allsky-sw-no-aerosols.nc')

# The generated case is the same profile in every column, so a few of them say
# everything a full domain would, on a coarse grid.
NX, NY, NLAY = 3, 2, 48

requires_data = pytest.mark.skipif(
    not os.path.exists(LW_REF), reason='rrtmgp-data submodule not checked out')

# The Stefan-Boltzmann constant, for what the surface has to emit.
SIGMA = 5.670374419e-8


def case_scripts():
    """cases/user on the path: the scripts a user runs, imported as modules."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cases', 'user'))

    import make_input
    import run_case

    return make_input, run_case

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

def generated_case(tmp_path, name='mycase', clouds=False):
    """Write a case with cases/user/make_input.py and read it back with rte3d.case."""
    from rte3d.case import read_case

    make_input, _ = case_scripts()
    path = make_input.make_input(str(tmp_path/f'{name}_input.nc'),
                                 nx=NX, ny=NY, nlay=NLAY, clouds=clouds)

    return make_input, read_case(path)


@requires_data
def test_generated_case_reads_back(rte3d, tmp_path):
    """What the reader makes of a file the generator wrote.

    Between them these are the parts of the layout the all-sky case above does not
    have: scalar gases, band-resolved surface conditions, and an absolute irradiance.
    """
    make_input, atm = generated_case(tmp_path)

    assert (atm['nx'], atm['ny'], atm['nlay'], atm['nlev']) == (NX, NY, NLAY, NLAY + 1)
    assert atm['ncol'] == NX*NY

    # The profile is written surface-first, and the orientation is read from it.
    assert atm['top_at_1'] is False

    assert isinstance(atm['gases']['co2'], float)
    assert atm['gases']['h2o'].shape == (NLAY, NX*NY)
    assert atm['sfc_emis'].shape == (make_input.NBND_LW, NX*NY)
    assert atm['sfc_alb_dir'].shape == (make_input.NBND_SW, NX*NY)
    assert np.all(atm['tsi'] == make_input.TSI)

    # Clear sky by default: the file has no cloud fields and none are invented.
    assert 'lwp' not in atm


@requires_data
def test_generated_case_solves(rte3d, tmp_path):
    """The fluxes the boundary conditions dictate, at the two ends of the column."""
    from rte3d.case import gpoint_bands, make_gas_concs, solve_lw, solve_sw
    from rte3d.kdist import read_kdist

    make_input, atm = generated_case(tmp_path)
    toa, sfc = -1, 0      # the profile is surface-first

    for band, gas_file in (('lw', 'rrtmgp-gas-lw-g256.nc'),
                           ('sw', 'rrtmgp-gas-sw-g224.nc')):
        f = read_kdist(os.path.join(DATA, gas_file))
        gas_concs = make_gas_concs(rte3d, atm, f['gas_names'])
        kdist = rte3d.load_kdist(f, gas_concs)

        solve = solve_lw if band == 'lw' else solve_sw
        out = solve(rte3d, kdist, gas_concs, atm, gpoint_bands(f))

        # Every column holds the same profile, so every column gets the same flux.
        for flux in out.values():
            assert_flux_close(flux, np.broadcast_to(flux[:, :1], flux.shape), rte3d,
                              err_msg=f'{band} columns differ')

        if band == 'lw':
            # A black surface at the sea surface temperature, to within what the
            # 16 bands cover of the Planck integral.
            emitted = SIGMA*make_input.SST**4
            assert out['flux_up'][sfc] == pytest.approx(emitted, rel=1e-3)

            # Nothing comes down from space, and the outgoing longwave is less than
            # what the surface sent up.
            assert np.all(out['flux_dn'][toa] == 0.0)
            assert np.all(out['flux_up'][toa] < out['flux_up'][sfc])
        else:
            # The direct beam at the top of the atmosphere is the irradiance the case
            # asks for, projected onto the horizontal.
            incoming = make_input.TSI*np.cos(np.deg2rad(make_input.SOLAR_ZENITH_ANGLE))
            assert_flux_close(out['flux_dir'][toa], np.full(atm['ncol'], incoming), rte3d,
                              err_msg='incoming solar')
            assert_flux_close(out['flux_dn'][toa], out['flux_dir'][toa], rte3d,
                              err_msg='no diffuse light at the top')

            # The surface reflects the albedo it was given.
            assert_flux_close(out['flux_up'][sfc],
                              make_input.SFC_ALBEDO*out['flux_dn'][sfc], rte3d,
                              err_msg='surface reflection')


@requires_data
def test_run_case_end_to_end(rte3d, tmp_path, monkeypatch):
    """make_input.py, then run_case.py, as a user runs them.

    The output file is the deliverable, so this checks what lands in it: the input's
    own layout, the net flux, and band fluxes that add up to the broadband ones.
    """
    import sys

    import xarray as xr

    make_input, run_case = case_scripts()

    monkeypatch.chdir(tmp_path)
    make_input.make_input('mycase_input.nc', nx=NX, ny=NY, nlay=NLAY, clouds=True)
    make_input.make_settings('mycase.toml', clouds=True)

    monkeypatch.setattr(sys, 'argv', ['run_case.py', 'mycase', '--output-bnd-fluxes'])
    assert run_case.main() == 0

    out = xr.open_dataset(tmp_path/'mycase_output.nc')

    for band in ('lw', 'sw'):
        up, dn = out[f'{band}_flux_up'], out[f'{band}_flux_dn']
        assert up.dims == ('lev', 'y', 'x')
        assert up.shape == (NLAY + 1, NY, NX)

        assert_flux_close(out[f'{band}_flux_net'].values, (dn - up).values, rte3d,
                          err_msg=f'{band} net flux')

        # The bands partition the spectrum, so they add up to the broadband flux.
        byband = out[f'{band}_bnd_flux_up']
        assert byband.dims == (f'band_{band}', 'lev', 'y', 'x')
        assert_flux_close(byband.sum(f'band_{band}').values, up.values, rte3d,
                          err_msg=f'{band} bands do not add up')

    # The clouds were actually used: an overcast column reflects far more sunlight
    # than the clear-sky reflection of a 0.07 albedo.
    assert np.all(out['sw_flux_up'][-1] > 0.2*out['sw_flux_dn'][-1])
