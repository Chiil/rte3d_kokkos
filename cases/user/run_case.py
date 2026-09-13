#!/usr/bin/env python
"""Run a user-supplied case from a NetCDF file.

    python cases/user/run_case.py CASE

CASE names the case: settings are read from CASE.toml, the atmosphere from
CASE_input.nc, and the fluxes are written to CASE_output.nc. This is the flow of
rte-rrtmgp-cpp's test executable, which is also where the input file layout comes
from, so cases written for it run here unchanged. rte3d.case documents that layout.

CASE.toml is the only place a setting lives, so a case is reproducible from the two
files that carry its name and nothing else. The file is optional; without one every
switch takes its default. See example.toml for the full set.
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
                        solve_lw, solve_lw_rt, solve_sw, solve_sw_rt)
from rte3d.kdist import read_cloud_optics, read_kdist  # noqa: E402

SWITCHES = dict(longwave=True, shortwave=True, cloud_optics=False,
                delta_cloud=True, output_bnd_fluxes=False)

# Which shortwave solver, or both. Named as in rte-rrtmgp-cpp's ini files, whose
# [shortwave] section this is.
SHORTWAVE = dict(plane_parallel=True, raytracing=False,
                 photons_per_pixel=256, independent_column=False)

# The same for the longwave. scattering follows lw-scattering in those ini files:
# without it clouds absorb and emit but do not deflect a photon.
LONGWAVE = dict(plane_parallel=True, raytracing=False,
                photons_per_pixel=256, independent_column=False, scattering=False,
                min_mfp_grid_ratio=0.0)

FILES = dict(gas_lw='rrtmgp-gas-lw-g256.nc', gas_sw='rrtmgp-gas-sw-g224.nc',
             cloud_lw='rrtmgp-clouds-lw-bnd.nc', cloud_sw='rrtmgp-clouds-sw-bnd.nc',
             icergh=0)


def read_settings(path):
    """Settings from CASE.toml, if there is one, over the defaults.

    Keys are written with dashes in the file, as in the reference's ini files, and
    with underscores here.
    """
    switches, files = dict(SWITCHES), dict(FILES)
    shortwave, longwave = dict(SHORTWAVE), dict(LONGWAVE)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            settings = tomllib.load(f)

        for section, target in (('switches', switches), ('files', files),
                                ('shortwave', shortwave), ('longwave', longwave)):
            for key, value in settings.get(section, {}).items():
                key = key.replace('-', '_')
                if key not in target:
                    raise SystemExit(f'{path}: unknown [{section}] key "{key}"')
                target[key] = value

    return switches, files, shortwave, longwave


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


def run_band(band, atm, switches, files, shortwave, longwave,
             results, rt_results):
    """Load the coefficients for one band, solve, and collect the fluxes.

    Either band may run either solver or both: the plane-parallel one, whose fluxes are
    a profile per column, and the ray tracer, whose are three-dimensional. They share
    the gas optics and the cloud properties and differ only in transport, so running
    both is the way to see what the third dimension is worth.
    """
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
        if longwave['plane_parallel']:
            out = timer.run(lambda: solve_lw(rte3d, kdist, gas_concs, atm, gpt_band,
                                             cloud_optics, byband,
                                             longwave['scattering'],
                                             switches['delta_cloud']))
            print(timer.report(ncol=atm['ncol']))
            band_fluxes(out, f'{band}_flux', results)

        if longwave['raytracing']:
            grid = atm['grid']
            print(f'lw ray tracer                tracing '
                  f'{longwave["photons_per_pixel"]} photons per pixel through '
                  f'{grid["nx"]}x{grid["ny"]}x{grid["nz"]} cells, '
                  f'{kdist.ngpt} g-points', flush=True)

            # Once, not the timer's usual four: a trace is minutes rather than
            # milliseconds, and there is nothing to warm up.
            rt_timer = Timer('lw ray tracer')
            rt_results.update(rt_timer.run(
                lambda: solve_lw_rt(rte3d, kdist, gas_concs, atm, gpt_band,
                                    cloud_optics,
                                    longwave['photons_per_pixel'],
                                    longwave['independent_column'],
                                    longwave['scattering'],
                                    longwave['min_mfp_grid_ratio']),
                repeats=1, warmup=0))
            print(rt_timer.report(ncol=atm['ncol']))

            traced = rt_results.pop('n_gpt_traced', kdist.ngpt)
            if traced < kdist.ngpt:
                print(f'lw ray tracer                traced {traced} of {kdist.ngpt} '
                      f'g-points; the rest were opaque within a cell and were solved '
                      f'plane-parallel')
    else:
        if shortwave['plane_parallel']:
            out = timer.run(lambda: solve_sw(rte3d, kdist, gas_concs, atm, gpt_band,
                                             cloud_optics, byband,
                                             switches['delta_cloud']))
            print(timer.report(ncol=atm['ncol']))
            band_fluxes(out, f'{band}_flux', results)

        if shortwave['raytracing']:
            grid = atm['grid']
            print(f'sw ray tracer                tracing '
                  f'{shortwave["photons_per_pixel"]} photons per pixel through '
                  f'{grid["nx"]}x{grid["ny"]}x{grid["nz"]} cells, '
                  f'{kdist.ngpt} g-points', flush=True)

            # Once, not the timer's usual four: a trace is minutes rather than
            # milliseconds, and there is nothing to warm up.
            rt_timer = Timer('sw ray tracer')
            rt_results.update(rt_timer.run(
                lambda: solve_sw_rt(rte3d, kdist, gas_concs, atm, gpt_band,
                                    cloud_optics, switches['delta_cloud'],
                                    shortwave['photons_per_pixel'],
                                    shortwave['independent_column']),
                repeats=1, warmup=0))
            print(rt_timer.report(ncol=atm['ncol']))

    return kdist.nbnd


def write_output(path, atm, results, rt_results, nbnd):
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

    # The ray tracer's fluxes are not profiles: two-dimensional at the surface and the
    # top of the domain, three-dimensional for the absorption, which is per unit height.
    for name, flux in rt_results.items():
        variables[name] = ((dims, flux.reshape(shape)) if flux.ndim == 1
                           else (('z',) + dims, flux.reshape((-1,) + shape)))

    xr.Dataset(variables).to_netcdf(path)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('case', help='case name: CASE.toml, CASE_input.nc, CASE_output.nc')
    args = p.parse_args()

    switches, files, shortwave, longwave = read_settings(f'{args.case}.toml')

    if switches['shortwave'] and not (shortwave['plane_parallel']
                                      or shortwave['raytracing']):
        raise SystemExit('The shortwave is on but neither solver is.')
    if switches['longwave'] and not (longwave['plane_parallel']
                                     or longwave['raytracing']):
        raise SystemExit('The longwave is on but neither solver is.')

    atm = read_case(f'{args.case}_input.nc')
    print(f'{atm["ncol"]} columns, {atm["nlay"]} layers, '
          f'{"top" if atm["top_at_1"] else "surface"} at index 0')

    results, rt_results, nbnd = {}, {}, {}
    for band in ('lw', 'sw'):
        if switches['longwave' if band == 'lw' else 'shortwave']:
            nbnd[band] = run_band(band, atm, switches, files, shortwave, longwave,
                                  results, rt_results)

    if not results and not rt_results:
        raise SystemExit('Nothing to do: both the longwave and the shortwave are off.')

    write_output(f'{args.case}_output.nc', atm, results, rt_results, nbnd)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
