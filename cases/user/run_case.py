#!/usr/bin/env python
"""Run a user-supplied case from a NetCDF file.

    python cases/user/run_case.py CASE [--no-shortwave] [--cloud-optics] [-o out.nc]

CASE names the case: settings are read from CASE.toml, the atmosphere from
CASE_input.nc, and the fluxes are written to CASE_output.nc. This is the flow of
rte-rrtmgp-cpp's test executable, which is also where the input file layout comes
from, so cases written for it run here unchanged. rte3d.case documents that layout.

The settings file is optional; without one every switch takes its default. Command
line flags override the file, so a case stays reproducible from CASE.toml alone while
a single run can still be varied. See example.toml for the full set.
"""
import argparse
import os
import sys
import tomllib

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA, Timer  # noqa: E402

import rte3d  # noqa: E402
from rte3d.case import (gpoint_bands, make_gas_concs, read_case,  # noqa: E402
                        solve_lw, solve_sw)
from rte3d.kdist import read_cloud_optics, read_kdist  # noqa: E402

SWITCHES = dict(longwave=True, shortwave=True, cloud_optics=False,
                delta_cloud=True, output_bnd_fluxes=False)

FILES = dict(gas_lw='rrtmgp-gas-lw-g256.nc', gas_sw='rrtmgp-gas-sw-g224.nc',
             cloud_lw='rrtmgp-clouds-lw-bnd.nc', cloud_sw='rrtmgp-clouds-sw-bnd.nc',
             icergh=0)


def read_settings(path):
    """Settings from CASE.toml, if there is one, over the defaults.

    Keys are written with dashes in the file, as in the reference's ini files, and
    with underscores here.
    """
    switches, files = dict(SWITCHES), dict(FILES)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            settings = tomllib.load(f)

        for section, target in (('switches', switches), ('files', files)):
            for key, value in settings.get(section, {}).items():
                key = key.replace('-', '_')
                if key not in target:
                    raise SystemExit(f'{path}: unknown [{section}] key "{key}"')
                target[key] = value

    return switches, files


def coefficients(name, files):
    """A coefficient file, taken as given if it is a path and from the data otherwise."""
    path = files[name]
    return path if os.path.sep in path else os.path.join(DATA, path)


def band_fluxes(out, prefix, results):
    """Move one solver's output into the result dict, adding the net flux."""
    for key, flux in out.items():
        results[prefix + key[len('flux'):]] = flux

    results[prefix + '_net'] = out['flux_dn'] - out['flux_up']
    if 'flux_dn_byband' in out:
        results[prefix + '_net_byband'] = out['flux_dn_byband'] - out['flux_up_byband']


def run_band(band, atm, switches, files, results):
    """Load the coefficients for one band, solve, and collect the fluxes."""
    f = read_kdist(coefficients(f'gas_{band}', files))
    gas_concs = make_gas_concs(rte3d, atm, f['gas_names'])
    kdist = rte3d.load_kdist(f, gas_concs)
    gpt_band = gpoint_bands(f)

    cloud_optics = None
    if switches['cloud_optics']:
        cloud_optics = rte3d.load_cloud_optics(
            read_cloud_optics(coefficients(f'cloud_{band}', files)), files['icergh'])

    byband = switches['output_bnd_fluxes']
    timer = Timer(f'{band} solve')

    if band == 'lw':
        out = timer.run(lambda: solve_lw(rte3d, kdist, gas_concs, atm, gpt_band,
                                         cloud_optics, byband))
    else:
        out = timer.run(lambda: solve_sw(rte3d, kdist, gas_concs, atm, gpt_band,
                                         cloud_optics, byband,
                                         switches['delta_cloud']))

    print(timer.report(ncol=atm['ncol']))
    band_fluxes(out, f'{band}_flux', results)

    return kdist.nbnd


def write_output(path, atm, results, nbnd):
    """Write the fluxes back in the input's own layout.

    A field comes out of the solvers as (nlev, ncol), or (nbnd, nlev, ncol) by band;
    the columns fold back to (y, x) exactly as they were flattened.
    """
    dims, shape = atm['dims'], atm['shape']

    variables = {
        'p_lay': (('lay',) + dims, atm['play'].reshape((-1,) + shape)),
        'p_lev': (('lev',) + dims, atm['plev'].reshape((-1,) + shape)),
    }

    for name, flux in results.items():
        band = ('band_lw' if name.startswith('lw') else 'band_sw',)
        if name.endswith('_byband'):
            variables[name[:-len('_byband')].replace('flux', 'bnd_flux')] = (
                band + ('lev',) + dims, flux.reshape((nbnd[name[:2]], -1) + shape))
        else:
            variables[name] = (('lev',) + dims, flux.reshape((-1,) + shape))

    xr.Dataset(variables).to_netcdf(path)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('case', help='case name: CASE.toml, CASE_input.nc, CASE_output.nc')
    p.add_argument('--settings', help='settings file (default: CASE.toml)')
    p.add_argument('-i', '--input', help='input file (default: CASE_input.nc)')
    p.add_argument('-o', '--output', help='output file (default: CASE_output.nc)')

    flag = argparse.BooleanOptionalAction
    p.add_argument('--longwave', action=flag, default=None, help='solve the longwave')
    p.add_argument('--shortwave', action=flag, default=None, help='solve the shortwave')
    p.add_argument('--cloud-optics', action=flag, default=None,
                   help='read lwp, iwp, rel and dei and include clouds')
    p.add_argument('--delta-cloud', action=flag, default=None,
                   help='delta-scale the shortwave cloud properties (default: on)')
    p.add_argument('--output-bnd-fluxes', action=flag, default=None,
                   help='also write the fluxes resolved by band')
    args = p.parse_args()

    switches, files = read_settings(args.settings or f'{args.case}.toml')
    for key in switches:
        if getattr(args, key) is not None:
            switches[key] = getattr(args, key)

    atm = read_case(args.input or f'{args.case}_input.nc')
    print(f'{atm["ncol"]} columns, {atm["nlay"]} layers, '
          f'{"top" if atm["top_at_1"] else "surface"} at index 0')

    results, nbnd = {}, {}
    for band in ('lw', 'sw'):
        if switches['longwave' if band == 'lw' else 'shortwave']:
            nbnd[band] = run_band(band, atm, switches, files, results)

    if not results:
        raise SystemExit('Nothing to do: both the longwave and the shortwave are off.')

    write_output(args.output or f'{args.case}_output.nc', atm, results, nbnd)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
